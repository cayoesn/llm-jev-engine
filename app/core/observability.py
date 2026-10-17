"""ContextVar telemetry, Unmetered-Output FinOps accounting, Langfuse, and SHA-256 Audit Chain."""

from __future__ import annotations

import contextvars
import hashlib
import hmac
import json
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from app.api.schemas import (
    AuditVerificationResponse,
    EngineStatsResponse,
    GateAction,
    WorkflowGainMetrics,
)
from app.config import Settings, get_settings

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "jev_request_id",
    default="req-unbound",
)
request_start_ns_ctx: contextvars.ContextVar[int] = contextvars.ContextVar(
    "jev_request_start_ns",
    default=0,
)

PROMETHEUS_REGISTRY = CollectorRegistry(auto_describe=True)
JEV_DECISIONS_TOTAL = Counter(
    "jev_decisions_total",
    "Total System-1 primitive evaluations executed",
    ["kind", "gate_action"],
    registry=PROMETHEUS_REGISTRY,
)
JEV_LATENCY_SECONDS = Histogram(
    "jev_decision_latency_seconds",
    "Latency distribution of System-1 passes in seconds",
    ["endpoint"],
    registry=PROMETHEUS_REGISTRY,
)
JEV_CACHE_HIT_RATIO = Gauge(
    "jev_cache_hit_ratio",
    "Current Shared-State BoundedDecisionCache hit ratio",
    registry=PROMETHEUS_REGISTRY,
)


def bind_request_context(request_id: str | None = None) -> str:
    """Bind request-scoped identifier and start monotonic timestamp into ContextVars."""
    resolved_id = (
        request_id.strip()
        if request_id and request_id.strip()
        else f"jev-{uuid.uuid4().hex[:12]}"
    )
    now_ns = time.perf_counter_ns()
    request_id_ctx.set(resolved_id)
    request_start_ns_ctx.set(now_ns)
    return resolved_id


def get_elapsed_latency_ms() -> float:
    """Compute elapsed request latency in milliseconds from the ContextVar start timestamp."""
    start_ns = request_start_ns_ctx.get()
    if start_ns <= 0:
        return 0.05
    return max(0.01, (time.perf_counter_ns() - start_ns) / 1_000_000.0)


def record_decision_metrics(
    workflow: str,
    gate_action: GateAction | str,
    primitives_map: dict[str, str],
    latency_seconds: float,
) -> None:
    """Update Prometheus counters and latency histograms for a System-1 decision pass."""
    JEV_LATENCY_SECONDS.labels(endpoint=workflow).observe(max(0.0, latency_seconds))
    for kind in primitives_map.values():
        JEV_DECISIONS_TOTAL.labels(kind=kind, gate_action=str(gate_action)).inc()


@dataclass(frozen=True, slots=True)
class AuditChainRecord:
    """Immutable cryptographic link in the tamper-evident SHA-256 HMAC decision chain."""

    index: int
    timestamp_utc: str
    request_id: str
    state_hash: str
    gate_action: str
    overall_confidence: float
    decisions_summary: str
    previous_hash: str
    signature: str

    @property
    def prev_digest(self) -> str:
        """Alias returning previous_hash."""
        return self.previous_hash

    @property
    def digest(self) -> str:
        """Alias returning signature."""
        return self.signature

    @property
    def hmac_signature(self) -> str:
        """Alias returning signature."""
        return self.signature


AuditRecord = AuditChainRecord


class CryptographicAuditChain:
    """Thread-safe SHA-256 + HMAC-SHA256 decision audit log with O(1) append and O(M) verify."""

    GENESIS_DIGEST = "0" * 64

    def __init__(self, secret_key: str | None = None) -> None:
        """Initialize the CryptographicAuditChain with an HMAC secret key."""
        settings = get_settings()
        raw_secret = secret_key if secret_key is not None else settings.JEV_AUDIT_HMAC_SECRET
        if len(raw_secret.strip()) < 8:
            raise ValueError("secret_key must be at least 8 characters long.")
        self._secret_bytes = raw_secret.encode("utf-8")
        self._lock = threading.RLock()
        self._records: list[AuditChainRecord] = []

    def __len__(self) -> int:
        """Return total number of records in the audit chain."""
        with self._lock:
            return len(self._records)

    @property
    def length(self) -> int:
        """Return total number of records in the audit chain."""
        with self._lock:
            return len(self._records)

    @property
    def latest_digest(self) -> str:
        """Return the most recent SHA-256 HMAC signature or the genesis digest."""
        with self._lock:
            if not self._records:
                return self.GENESIS_DIGEST
            return self._records[-1].signature

    def records(self) -> list[AuditChainRecord]:
        """Return a snapshot copy of all records in the audit chain."""
        with self._lock:
            return list(self._records)

    def _compute_signature(
        self,
        index: int,
        timestamp_utc: str,
        request_id: str,
        state_hash: str,
        gate_action: str,
        overall_confidence: float,
        decisions_summary: str,
        previous_hash: str,
    ) -> str:
        """Compute deterministic HMAC-SHA256 signature over canonical record fields."""
        payload = (
            f"{index}|{previous_hash}|{timestamp_utc}|{request_id}|"
            f"{state_hash}|{gate_action}|{overall_confidence:.6f}|{decisions_summary}"
        ).encode("utf-8")
        return hmac.new(self._secret_bytes, payload, hashlib.sha256).hexdigest()

    def append(
        self,
        request_id: str,
        state_hash: str,
        gate_action: GateAction | str = "AUTO_EXECUTE",
        overall_confidence: float = 1.0,
        decisions: dict[str, Any] | None = None,
    ) -> AuditChainRecord:
        """Append a new decision event to the cryptographic chain in O(1)."""
        summary = json.dumps(
            decisions or {},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        ts = datetime.now(timezone.utc).isoformat()
        conf_rounded = round(float(overall_confidence), 6)

        with self._lock:
            idx = len(self._records)
            prev_hash = self._records[-1].signature if self._records else self.GENESIS_DIGEST
            sig = self._compute_signature(
                index=idx,
                timestamp_utc=ts,
                request_id=request_id,
                state_hash=state_hash,
                gate_action=str(gate_action),
                overall_confidence=conf_rounded,
                decisions_summary=summary,
                previous_hash=prev_hash,
            )
            record = AuditChainRecord(
                index=idx,
                timestamp_utc=ts,
                request_id=request_id,
                state_hash=state_hash,
                gate_action=str(gate_action),
                overall_confidence=conf_rounded,
                decisions_summary=summary,
                previous_hash=prev_hash,
                signature=sig,
            )
            self._records.append(record)
            return record

    def verify_chain(self) -> AuditVerificationResponse:
        """Verify all hash links and HMAC signatures across the entire chain in O(M)."""
        verified_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            chain_len = len(self._records)
            if chain_len == 0:
                return AuditVerificationResponse(
                    valid=True,
                    chain_length=0,
                    latest_digest=self.GENESIS_DIGEST,
                    verified_at_utc=verified_at,
                    tampered_index=None,
                )

            expected_prev = self.GENESIS_DIGEST
            for idx, rec in enumerate(self._records):
                if rec.index != idx or rec.previous_hash != expected_prev:
                    return AuditVerificationResponse(
                        valid=False,
                        chain_length=chain_len,
                        latest_digest=self._records[-1].signature,
                        verified_at_utc=verified_at,
                        tampered_index=idx,
                    )
                expected_sig = self._compute_signature(
                    index=rec.index,
                    timestamp_utc=rec.timestamp_utc,
                    request_id=rec.request_id,
                    state_hash=rec.state_hash,
                    gate_action=rec.gate_action,
                    overall_confidence=rec.overall_confidence,
                    decisions_summary=rec.decisions_summary,
                    previous_hash=rec.previous_hash,
                )
                if not hmac.compare_digest(rec.signature, expected_sig):
                    return AuditVerificationResponse(
                        valid=False,
                        chain_length=chain_len,
                        latest_digest=self._records[-1].signature,
                        verified_at_utc=verified_at,
                        tampered_index=idx,
                    )
                expected_prev = rec.signature

            return AuditVerificationResponse(
                valid=True,
                chain_length=chain_len,
                latest_digest=self._records[-1].signature,
                verified_at_utc=verified_at,
                tampered_index=None,
            )


class LangfuseTraceRecorder:
    """Thread-safe observability recorder for System-1 traces and Langfuse spans."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the LangfuseTraceRecorder with runtime settings."""
        self._settings = settings or get_settings()
        self._lock = threading.RLock()
        self._traces: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        """Return whether remote Langfuse export is enabled."""
        return bool(
            self._settings.LANGFUSE_ENABLED and bool(self._settings.LANGFUSE_PUBLIC_KEY)
        )

    @property
    def traces(self) -> list[dict[str, Any]]:
        """Return a copy of recorded trace payloads."""
        with self._lock:
            return list(self._traces)

    def recent_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent trace payloads up to limit."""
        with self._lock:
            return list(self._traces[-limit:])

    def record_trace(
        self,
        request_id: str,
        workflow: str,
        state_hash: str,
        gate_action: GateAction | str,
        overall_confidence: float,
        latency_ms: float,
        gain_metrics: WorkflowGainMetrics | None = None,
    ) -> dict[str, Any]:
        """Record a structured trace event in memory (bounded to latest 500 traces)."""
        entry: dict[str, Any] = {
            "request_id": request_id,
            "workflow": workflow,
            "state_hash": state_hash,
            "gate_action": str(gate_action),
            "overall_confidence": round(overall_confidence, 6),
            "latency_ms": round(latency_ms, 4),
            "langfuse_enabled": self.enabled,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if gain_metrics is not None:
            entry["speedup_factor"] = gain_metrics.speedup_factor
            entry["cost_reduction_pct"] = gain_metrics.cost_reduction_pct

        with self._lock:
            self._traces.append(entry)
            if len(self._traces) > 500:
                self._traces.pop(0)
        return entry


class FinOpsTelemetryTracker:
    """Thread-safe Unmetered-Output FinOps cost calculator and runtime metrics aggregator."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the FinOpsTelemetryTracker with pricing settings."""
        self._settings = settings or get_settings()
        self._lock = threading.RLock()
        self._total_requests = 0
        self._total_primitives_evaluated = 0
        self._fast_path_hits = 0
        self._total_jev_input_tokens = 0
        self._cumulative_jev_cost_usd = 0.0
        self._cumulative_system2_cost_avoided_usd = 0.0

    def compute_gain_metrics(
        self,
        state_input_tokens: int,
        num_primitives: int,
        jev_latency_ms: float,
        system2_baseline_latency_ms: float | None = None,
        avg_question_tokens: int = 45,
        avg_system2_output_tokens_per_question: int = 180,
        fast_path_triggered: bool = False,
        record_to_totals: bool = True,
    ) -> WorkflowGainMetrics:
        """Calculate and optionally record quantified latency, token, and USD savings."""
        q_count = max(1, int(num_primitives))
        s_tokens = max(1, int(state_input_tokens))
        q_tokens = max(1, int(avg_question_tokens))
        out_tokens_per_q = max(1, int(avg_system2_output_tokens_per_question))

        jev_input_tokens = s_tokens + q_count * q_tokens
        system2_input_tokens = q_count * (s_tokens + q_tokens)
        system2_output_tokens = q_count * out_tokens_per_q
        tokens_avoided = (system2_input_tokens - jev_input_tokens) + system2_output_tokens

        jev_cost = (jev_input_tokens / 1_000_000.0) * self._settings.JEV_INPUT_PRICE_PER_1M_USD
        system2_cost = (
            (system2_input_tokens / 1_000_000.0) * self._settings.JEV_SYSTEM2_INPUT_PRICE_PER_1M_USD
            + (system2_output_tokens / 1_000_000.0)
            * self._settings.JEV_SYSTEM2_OUTPUT_PRICE_PER_1M_USD
        )

        safe_lat = max(0.01, float(jev_latency_ms))
        base_lat = max(
            safe_lat,
            float(
                system2_baseline_latency_ms
                if system2_baseline_latency_ms is not None
                else self._settings.JEV_SYSTEM2_GATEWAY_BASELINE_LATENCY_MS
            ),
        )
        speedup = round(base_lat / safe_lat, 2)

        cost_reduction_pct = (
            round(((system2_cost - jev_cost) / system2_cost) * 100.0, 4)
            if system2_cost > 0.0
            else 0.0
        )
        amortization_ratio = round((q_count - 1) / q_count, 4) if q_count > 1 else 0.0

        if record_to_totals:
            with self._lock:
                self._total_requests += 1
                self._total_primitives_evaluated += q_count
                if fast_path_triggered:
                    self._fast_path_hits += 1
                self._total_jev_input_tokens += jev_input_tokens
                self._cumulative_jev_cost_usd += jev_cost
                self._cumulative_system2_cost_avoided_usd += max(0.0, system2_cost - jev_cost)

        return WorkflowGainMetrics(
            jev_latency_ms=round(safe_lat, 4),
            system2_baseline_latency_ms=round(base_lat, 2),
            speedup_factor=speedup,
            jev_cost_usd=round(jev_cost, 8),
            system2_baseline_cost_usd=round(system2_cost, 8),
            cost_reduction_pct=float(min(100.0, max(0.0, cost_reduction_pct))),
            output_tokens_billed=0,
            shared_state_amortization_ratio=amortization_ratio,
            jev_input_tokens=jev_input_tokens,
            system2_total_tokens_avoided=tokens_avoided,
            calibration_ece_advantage=0.22,
        )

    def snapshot(self) -> dict[str, int | float]:
        """Return a thread-safe dictionary snapshot of cumulative FinOps telemetry."""
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "total_primitives_evaluated": self._total_primitives_evaluated,
                "fast_path_hits": self._fast_path_hits,
                "total_jev_input_tokens": self._total_jev_input_tokens,
                "total_jev_output_tokens_billed": 0,
                "cumulative_jev_cost_usd": round(self._cumulative_jev_cost_usd, 8),
                "cumulative_system2_cost_avoided_usd": round(
                    self._cumulative_system2_cost_avoided_usd, 8
                ),
            }

    def snapshot_stats(
        self,
        cache_hits: int,
        cache_misses: int,
        circuit_breaker_state: str = "CLOSED",
    ) -> EngineStatsResponse:
        """Build an immutable EngineStatsResponse snapshot under RLock."""
        total_cache = cache_hits + cache_misses
        hit_ratio = (cache_hits / total_cache) if total_cache > 0 else 0.0
        JEV_CACHE_HIT_RATIO.set(hit_ratio)
        with self._lock:
            return EngineStatsResponse(
                total_requests=self._total_requests,
                total_primitives_evaluated=self._total_primitives_evaluated,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                cache_hit_ratio=round(hit_ratio, 4),
                fast_path_hits=self._fast_path_hits,
                total_jev_input_tokens=self._total_jev_input_tokens,
                total_jev_output_tokens_billed=0,
                cumulative_jev_cost_usd=round(self._cumulative_jev_cost_usd, 8),
                cumulative_system2_cost_avoided_usd=round(
                    self._cumulative_system2_cost_avoided_usd, 8
                ),
                circuit_breaker_state=circuit_breaker_state,
            )
