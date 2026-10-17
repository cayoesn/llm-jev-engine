"""Hybrid O(1) Deterministic Fast-Path and System-1 Neural Confidence Gate Fabric."""

from __future__ import annotations

import math
import re
import time
from typing import Any

from app.api.schemas import (
    AuditVerificationResponse,
    EngineStatsResponse,
    GateAction,
    PrimitiveResult,
    PrimitiveSpec,
    SharedStateDecisionRequest,
    SharedStateDecisionResponse,
)
from app.config import Settings, get_settings
from app.core.observability import (
    CryptographicAuditChain,
    FinOpsTelemetryTracker,
    LangfuseTraceRecorder,
    bind_request_context,
    record_decision_metrics,
)
from app.core.parallel_sampler import SharedStateParallelSampler
from app.core.primitives_engine import SharedStateContext
from app.core.resilience import (
    CircuitBreaker,
    NoOpCircuitBreaker,
    ResilientExecutor,
    TokenBucketRateLimiter,
)

FAST_PATH_ADVERSARIAL_REGEX = re.compile(
    r"(?i)(ignore\s+(all\s+)?(previous|prior)\s+instructions|"
    r"system\s+prompt\s+override|jailbreak\s+mode|"
    r"drop\s+table\s+users|union\s+select\s+password|"
    r"reveal\s+confidential\s+system\s+instructions|"
    r"bypass\s+all\s+safety\s+guardrails)"
)


class ConfidenceGate:
    """Deterministic multi-tier confidence gate routing decisions by epistemic certainty.

    Applies strict monotonic thresholds over the joint minimum calibrated confidence:
    - C_min >= auto_threshold (0.80) and zero abstentions -> AUTO_EXECUTE
    - fallback_threshold (0.55) <= C_min < auto_threshold and zero abstentions -> FALLBACK_SYSTEM2
    - C_min < fallback_threshold (0.55) or any abstention -> ESCALATE_HITL
    """

    def __init__(
        self,
        auto_threshold: float = 0.80,
        fallback_threshold: float = 0.55,
    ) -> None:
        """Initialize the ConfidenceGate with auto-execution and fallback thresholds.

        Args:
            auto_threshold: Minimum calibrated confidence required for AUTO_EXECUTE.
            fallback_threshold: Minimum calibrated confidence required for FALLBACK_SYSTEM2.
        """
        if not (0.0 <= fallback_threshold <= auto_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy 0.0 <= fallback_threshold <= auto_threshold <= 1.0."
            )
        self._auto_threshold = float(auto_threshold)
        self._fallback_threshold = float(fallback_threshold)

    @property
    def auto_threshold(self) -> float:
        """Return the AUTO_EXECUTE threshold."""
        return self._auto_threshold

    @property
    def fallback_threshold(self) -> float:
        """Return the FALLBACK_SYSTEM2 threshold."""
        return self._fallback_threshold

    def resolve(
        self,
        results: dict[str, PrimitiveResult],
        fast_path_triggered: bool = False,
    ) -> tuple[float, GateAction]:
        """Compute joint minimum calibrated confidence and deterministic GateAction.

        Args:
            results: Mapping of primitive names to evaluated PrimitiveResult instances.
            fast_path_triggered: Whether Stage-1 adversarial fast-path guard triggered.

        Returns:
            Tuple of (overall_confidence, GateAction).
        """
        if not results:
            return 0.0, "ESCALATE_HITL"

        min_conf = min(res.calibrated_confidence for res in results.values())
        any_abstained = any(res.abstained for res in results.values())

        if fast_path_triggered or any_abstained or min_conf < self._fallback_threshold:
            return round(min_conf, 6), "ESCALATE_HITL"
        if min_conf >= self._auto_threshold:
            return round(min_conf, 6), "AUTO_EXECUTE"
        return round(min_conf, 6), "FALLBACK_SYSTEM2"


class DecisionFabric:
    """Three-stage System-1 Neural Decision Fabric with O(1) Fast-Path, Parallel Sampling, and Audit Chain."""

    def __init__(
        self,
        sampler: SharedStateParallelSampler | None = None,
        confidence_gate: ConfidenceGate | None = None,
        executor: ResilientExecutor | None = None,
        audit_chain: CryptographicAuditChain | None = None,
        finops_tracker: FinOpsTelemetryTracker | None = None,
        trace_recorder: LangfuseTraceRecorder | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the DecisionFabric and all resilience/observability subcomponents."""
        self._settings = settings or get_settings()
        self._sampler = sampler or SharedStateParallelSampler(settings=self._settings)
        self._gate = confidence_gate or ConfidenceGate(
            auto_threshold=self._settings.JEV_MIN_CALIBRATED_CONFIDENCE_AUTO,
            fallback_threshold=self._settings.JEV_MIN_CALIBRATED_CONFIDENCE_FALLBACK,
        )
        if executor is not None:
            self._executor = executor
        else:
            breaker: CircuitBreaker | NoOpCircuitBreaker = CircuitBreaker(
                failure_threshold=self._settings.JEV_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                recovery_timeout_seconds=self._settings.JEV_CIRCUIT_BREAKER_RECOVERY_SECONDS,
            )
            limiter = TokenBucketRateLimiter(
                rate_per_second=self._settings.JEV_RATE_LIMIT_RPS,
                burst_capacity=self._settings.JEV_RATE_LIMIT_BURST,
            )
            self._executor = ResilientExecutor(circuit_breaker=breaker, rate_limiter=limiter)

        self._audit_chain = audit_chain or CryptographicAuditChain(
            secret_key=self._settings.JEV_AUDIT_HMAC_SECRET
        )
        self._finops = finops_tracker or FinOpsTelemetryTracker(settings=self._settings)
        self._tracer = trace_recorder or LangfuseTraceRecorder(settings=self._settings)

    @property
    def sampler(self) -> SharedStateParallelSampler:
        """Return the SharedStateParallelSampler."""
        return self._sampler

    @property
    def confidence_gate(self) -> ConfidenceGate:
        """Return the ConfidenceGate."""
        return self._gate

    @property
    def executor(self) -> ResilientExecutor:
        """Return the ResilientExecutor."""
        return self._executor

    @property
    def audit_chain(self) -> CryptographicAuditChain:
        """Return the CryptographicAuditChain."""
        return self._audit_chain

    @property
    def finops_tracker(self) -> FinOpsTelemetryTracker:
        """Return the FinOpsTelemetryTracker."""
        return self._finops

    @property
    def trace_recorder(self) -> LangfuseTraceRecorder:
        """Return the LangfuseTraceRecorder."""
        return self._tracer

    @staticmethod
    def _matches_adversarial_fast_path(canonical_json: str) -> bool:
        """Check in O(1) regex pass whether the serialized state triggers adversarial guardrails."""
        return bool(FAST_PATH_ADVERSARIAL_REGEX.search(canonical_json))

    def _synthesize_fast_path_results(
        self,
        primitives: list[PrimitiveSpec],
    ) -> dict[str, PrimitiveResult]:
        """Construct deterministic fail-safe PrimitiveResult objects when fast-path blocks execution."""
        results: dict[str, PrimitiveResult] = {}
        for spec in primitives:
            if spec.kind == "noul":
                results[spec.name] = PrimitiveResult(
                    name=spec.name,
                    kind="noul",
                    selected_value="FALSE",
                    probability_distribution={"TRUE": 0.01, "FALSE": 0.99},
                    calibrated_confidence=0.92,
                    shannon_entropy_bits=0.0808,
                    normalized_entropy=0.0808,
                    credible_interval_95=(0.94, 1.0),
                    abstained=False,
                    latency_ms=0.05,
                    gate_action="ESCALATE_HITL",
                )
            elif spec.kind == "choice" and spec.options is not None:
                keys = list(spec.options.keys())
                target_key = keys[0]
                for candidate in keys:
                    if any(
                        tag in candidate.lower()
                        for tag in ("human", "escalate", "block", "review", "security")
                    ):
                        target_key = candidate
                        break
                num_k = len(keys)
                rem_prob = round(0.04 / max(1, num_k - 1), 6)
                dist = {k: rem_prob for k in keys}
                dist[target_key] = round(1.0 - rem_prob * (num_k - 1), 6)
                results[spec.name] = PrimitiveResult(
                    name=spec.name,
                    kind="choice",
                    selected_value=target_key,
                    probability_distribution=dist,
                    calibrated_confidence=0.90,
                    shannon_entropy_bits=0.25,
                    normalized_entropy=round(0.25 / max(1.0, math.log2(num_k)), 6),
                    credible_interval_95=(0.86, 0.99),
                    abstained=False,
                    latency_ms=0.05,
                    gate_action="ESCALATE_HITL",
                )
            elif spec.kind == "score" and spec.levels is not None:
                sorted_lvls = sorted(spec.levels.keys())
                min_lvl = sorted_lvls[0]
                num_l = len(sorted_lvls)
                rem_prob = round(0.04 / max(1, num_l - 1), 6)
                dist = {str(lvl): rem_prob for lvl in sorted_lvls}
                dist[str(min_lvl)] = round(1.0 - rem_prob * (num_l - 1), 6)
                results[spec.name] = PrimitiveResult(
                    name=spec.name,
                    kind="score",
                    selected_value=float(min_lvl),
                    probability_distribution=dist,
                    calibrated_confidence=0.90,
                    shannon_entropy_bits=0.25,
                    normalized_entropy=round(0.25 / max(1.0, math.log2(num_l)), 6),
                    credible_interval_95=(float(min_lvl), float(min_lvl) + 0.25),
                    expected_score=float(min_lvl),
                    score_variance=0.04,
                    abstained=False,
                    latency_ms=0.05,
                    gate_action="ESCALATE_HITL",
                )
        return results

    def _finalize_response(
        self,
        req_id: str,
        context: SharedStateContext,
        results: dict[str, PrimitiveResult],
        fast_path_triggered: bool,
        cache_hits: int,
        start_ns: int,
    ) -> SharedStateDecisionResponse:
        """Compute gate action, FinOps telemetry, and cryptographic audit record."""
        overall_conf, gate_action = self._gate.resolve(
            results=results,
            fast_path_triggered=fast_path_triggered,
        )
        total_latency_ms = max(0.02, (time.perf_counter_ns() - start_ns) / 1_000_000.0)
        gain_metrics = self._finops.compute_gain_metrics(
            state_input_tokens=context.token_count,
            num_primitives=len(results),
            jev_latency_ms=total_latency_ms,
            fast_path_triggered=fast_path_triggered,
        )
        audit_record = self._audit_chain.append(
            request_id=req_id,
            state_hash=context.state_hash,
            gate_action=gate_action,
            overall_confidence=overall_conf,
            decisions={k: v.selected_value for k, v in results.items()},
        )
        record_decision_metrics(
            workflow="system_one",
            gate_action=gate_action,
            primitives_map={k: v.kind for k, v in results.items()},
            latency_seconds=total_latency_ms / 1000.0,
        )
        self._tracer.record_trace(
            request_id=req_id,
            workflow="system_one",
            state_hash=context.state_hash,
            gate_action=gate_action,
            overall_confidence=overall_conf,
            latency_ms=total_latency_ms,
            gain_metrics=gain_metrics,
        )
        return SharedStateDecisionResponse(
            request_id=req_id,
            state_hash=context.state_hash,
            results=results,
            overall_confidence=overall_conf,
            gate_action=gate_action,
            fast_path_triggered=fast_path_triggered,
            cache_hit_count=cache_hits,
            total_latency_ms=round(total_latency_ms, 4),
            audit_signature=audit_record.signature,
            gain_metrics=gain_metrics,
        )

    async def evaluate(
        self,
        request: SharedStateDecisionRequest,
    ) -> SharedStateDecisionResponse:
        """Execute the 3-stage System-1 Decision Fabric asynchronously.

        Args:
            request: Validated SharedStateDecisionRequest containing state and primitives.

        Returns:
            Immutable SharedStateDecisionResponse with decisions, confidence gate, and audit digest.
        """
        start_ns = time.perf_counter_ns()
        req_id = bind_request_context(request.request_id)

        context = self._sampler.engine.encode_state(request.state)
        if request.fast_path_enabled and self._matches_adversarial_fast_path(
            context.canonical_json
        ):
            results = self._synthesize_fast_path_results(request.primitives)
            return self._finalize_response(
                req_id=req_id,
                context=context,
                results=results,
                fast_path_triggered=True,
                cache_hits=0,
                start_ns=start_ns,
            )

        sampled_ctx, results, cache_hits = await self._executor.execute_async(
            self._sampler.sample_parallel,
            request.state,
            request.primitives,
            request.bypass_cache,
            request.backend,
        )
        return self._finalize_response(
            req_id=req_id,
            context=sampled_ctx,
            results=results,
            fast_path_triggered=False,
            cache_hits=cache_hits,
            start_ns=start_ns,
        )

    async def decide(
        self,
        request: SharedStateDecisionRequest,
    ) -> SharedStateDecisionResponse:
        """Alias for evaluate() executing a SharedStateDecisionRequest."""
        return await self.evaluate(request)

    def evaluate_sync(
        self,
        request: SharedStateDecisionRequest,
    ) -> SharedStateDecisionResponse:
        """Execute the 3-stage System-1 Decision Fabric synchronously."""
        start_ns = time.perf_counter_ns()
        req_id = bind_request_context(request.request_id)

        context = self._sampler.engine.encode_state(request.state)
        if request.fast_path_enabled and self._matches_adversarial_fast_path(
            context.canonical_json
        ):
            results = self._synthesize_fast_path_results(request.primitives)
            return self._finalize_response(
                req_id=req_id,
                context=context,
                results=results,
                fast_path_triggered=True,
                cache_hits=0,
                start_ns=start_ns,
            )

        sampled_ctx, results, cache_hits = self._executor.execute_sync(
            self._sampler.sample_sync,
            request.state,
            request.primitives,
            request.bypass_cache,
        )
        return self._finalize_response(
            req_id=req_id,
            context=sampled_ctx,
            results=results,
            fast_path_triggered=False,
            cache_hits=cache_hits,
            start_ns=start_ns,
        )

    def get_stats(self) -> EngineStatsResponse:
        """Return a unified telemetry snapshot of cache, FinOps, and circuit breaker state."""
        cache_stats = self._sampler.cache.stats()
        finops_snap = self._finops.snapshot()
        return EngineStatsResponse(
            total_requests=int(finops_snap["total_requests"]),
            total_primitives_evaluated=int(finops_snap["total_primitives_evaluated"]),
            cache_hits=int(cache_stats["hits"]),
            cache_misses=int(cache_stats["misses"]),
            cache_hit_ratio=float(cache_stats["hit_rate"]),
            fast_path_hits=int(finops_snap["fast_path_hits"]),
            total_jev_input_tokens=int(finops_snap["total_jev_input_tokens"]),
            total_jev_output_tokens_billed=0,
            cumulative_jev_cost_usd=float(finops_snap["cumulative_jev_cost_usd"]),
            cumulative_system2_cost_avoided_usd=float(
                finops_snap["cumulative_system2_cost_avoided_usd"]
            ),
            circuit_breaker_state=self._executor.circuit_breaker.state,
        )

    def clear_cache(self) -> int:
        """Clear all entries from the BoundedDecisionCache and return evicted count."""
        return self._sampler.cache.clear()

    def verify_audit_chain(self) -> AuditVerificationResponse:
        """Verify cryptographic integrity of the SHA-256 HMAC decision audit chain."""
        return self._audit_chain.verify_chain()


NeuralDecisionFabric = DecisionFabric
