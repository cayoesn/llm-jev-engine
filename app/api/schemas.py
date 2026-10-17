"""Immutable Pydantic v2 domain and API contracts for the Jev System-1 Decision Engine."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PrimitiveKind = Literal["choice", "noul", "score"]
GateAction = Literal["AUTO_EXECUTE", "FALLBACK_SYSTEM2", "ESCALATE_HITL"]
GatewayPathway = Literal[
    "semantic_cache",
    "local_slm_fast",
    "frontier_reasoning_llm",
    "human_review_queue",
]
DepartmentQueue = Literal[
    "billing",
    "technical_sre",
    "sales_account",
    "legal_compliance",
]
RagPipelineAction = Literal[
    "publish_response",
    "rewrite_with_citations",
    "trigger_corrective_rag",
    "block_and_escalate",
]


class PrimitiveSpec(BaseModel):
    """Specification for a single non-autoregressive System-1 decision primitive.

    Supports three native Jev primitives:
    - 'choice': Multi-class selection over 2..255 semantic options.
    - 'noul': Binary proposition verification returning calibrated probability and Wilson CI.
    - 'score': Ordinal rubric evaluation over 2..10 discrete levels producing a continuous mean.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(default="primitive", min_length=1, max_length=128)
    kind: PrimitiveKind
    question: str = Field(..., min_length=3, max_length=4096)
    options: dict[str, str] | None = None
    levels: dict[int, str] | None = None
    abstention_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    calibration_temperature: float | None = Field(default=None, gt=0.0, le=10.0)

    @field_validator("options", mode="before")
    @classmethod
    def normalize_options(cls, value: Any) -> dict[str, str] | None:
        """Normalize list/tuple option shorthand into a semantic rubric dictionary."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            normalized: dict[str, str] = {}
            for item in value:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("All choice options in a list must be non-empty strings.")
                cleaned = item.strip()
                normalized[cleaned] = cleaned
            return normalized
        if isinstance(value, dict):
            cleaned_dict: dict[str, str] = {}
            for key, desc in value.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("Choice option keys must be non-empty strings.")
                if not isinstance(desc, str) or not desc.strip():
                    raise ValueError("Choice option semantic rubrics must be non-empty strings.")
                cleaned_dict[key.strip()] = desc.strip()
            return cleaned_dict
        raise ValueError("options must be a dictionary of option->rubric or list of strings.")

    @field_validator("levels", mode="before")
    @classmethod
    def normalize_levels(cls, value: Any) -> dict[int, str] | None:
        """Validate and coerce ordinal level keys to integers with non-empty descriptions."""
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("levels must be a dictionary mapping integer levels to rubrics.")
        cleaned_levels: dict[int, str] = {}
        for raw_level, rubric in value.items():
            level_int = int(raw_level)
            if not isinstance(rubric, str) or not rubric.strip():
                raise ValueError("Score level semantic rubrics must be non-empty strings.")
            cleaned_levels[level_int] = rubric.strip()
        return cleaned_levels

    @model_validator(mode="after")
    def validate_primitive_constraints(self) -> PrimitiveSpec:
        """Verify structural invariants between primitive kind, options, and levels."""
        if self.kind == "choice":
            if self.options is None or len(self.options) < 2 or len(self.options) > 255:
                raise ValueError(
                    "choice primitive requires between 2 and 255 options in 'options'."
                )
            if self.levels is not None:
                raise ValueError("choice primitive cannot define 'levels'.")
        elif self.kind == "score":
            if self.levels is None or len(self.levels) < 2 or len(self.levels) > 10:
                raise ValueError(
                    "score primitive requires between 2 and 10 ordinal levels in 'levels'."
                )
            if self.options is not None:
                raise ValueError("score primitive cannot define 'options'.")
        elif self.kind == "noul":
            if self.options is not None or self.levels is not None:
                raise ValueError("noul primitive does not accept 'options' or 'levels'.")
        return self


class PrimitiveResult(BaseModel):
    """Calibrated single-pass evaluation output for a System-1 decision primitive."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1)
    kind: PrimitiveKind
    selected_value: str | int | float
    probability_distribution: dict[str, float] = Field(..., min_length=2)
    calibrated_confidence: float = Field(..., ge=0.0, le=1.0)
    shannon_entropy_bits: float = Field(..., ge=0.0)
    normalized_entropy: float = Field(..., ge=0.0, le=1.0)
    credible_interval_95: tuple[float, float]
    expected_score: float | None = None
    score_variance: float | None = Field(default=None, ge=0.0)
    abstained: bool = False
    latency_ms: float = Field(..., ge=0.0)
    gate_action: GateAction = Field(default="AUTO_EXECUTE")

    @field_validator("probability_distribution")
    @classmethod
    def validate_probability_simplex(cls, dist: dict[str, float]) -> dict[str, float]:
        """Ensure all probabilities lie in [0, 1] and sum to unity within numerical tolerance."""
        total = 0.0
        for label, prob in dist.items():
            if prob < -1e-6 or prob > 1.0 + 1e-6:
                raise ValueError(f"Probability for '{label}' must lie within [0.0, 1.0].")
            total += prob
        if abs(total - 1.0) > 1e-2:
            raise ValueError(f"Probability distribution must sum to 1.0 (got {total:.5f}).")
        return dist

    @field_validator("credible_interval_95")
    @classmethod
    def validate_credible_interval(cls, interval: tuple[float, float]) -> tuple[float, float]:
        """Verify that the lower bound does not exceed the upper bound."""
        low, high = interval
        if low > high + 1e-7:
            raise ValueError(
                f"Credible interval lower bound ({low}) cannot exceed upper bound ({high})."
            )
        return interval


class WorkflowGainMetrics(BaseModel):
    """Quantified latency, token, calibration, and financial telemetry comparing Jev vs System-2."""

    model_config = ConfigDict(frozen=True)

    jev_latency_ms: float = Field(..., ge=0.0)
    system2_baseline_latency_ms: float = Field(..., ge=0.0)
    speedup_factor: float = Field(..., gt=0.0)
    jev_cost_usd: float = Field(..., ge=0.0)
    system2_baseline_cost_usd: float = Field(..., ge=0.0)
    cost_reduction_pct: float = Field(..., ge=0.0, le=100.0)
    output_tokens_billed: int = Field(default=0, ge=0, le=0)
    shared_state_amortization_ratio: float = Field(..., ge=0.0, le=1.0)
    jev_input_tokens: int = Field(default=0, ge=0)
    system2_total_tokens_avoided: int = Field(default=0, ge=0)
    calibration_ece_advantage: float = Field(default=0.22, ge=0.0)

    @property
    def system2_estimated_latency_ms(self) -> float:
        """Alias for system2_baseline_latency_ms."""
        return self.system2_baseline_latency_ms

    @property
    def jev_output_tokens(self) -> int:
        """Alias for output_tokens_billed (always 0 for System-1)."""
        return self.output_tokens_billed

    @property
    def system2_cost_usd(self) -> float:
        """Alias for system2_baseline_cost_usd."""
        return self.system2_baseline_cost_usd

    @property
    def cost_savings_pct(self) -> float:
        """Alias for cost_reduction_pct."""
        return self.cost_reduction_pct


class SharedStateDecisionRequest(BaseModel):
    """Batch request for evaluating multiple System-1 primitives over a shared state context."""

    model_config = ConfigDict(frozen=True)

    request_id: str | None = None
    state: dict[str, Any] = Field(..., min_length=1)
    primitives: list[PrimitiveSpec] = Field(..., min_length=1, max_length=64)
    bypass_cache: bool = False
    fast_path_enabled: bool = True
    backend: Literal["spectral_simd", "open_weights"] = "spectral_simd"

    @model_validator(mode="before")
    @classmethod
    def coerce_questions_or_primitives(cls, data: Any) -> Any:
        """Accept either 'primitives' (list) or 'questions' (dict or list) as input."""
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if "primitives" not in payload and "questions" in payload:
            raw_questions = payload.pop("questions")
            if isinstance(raw_questions, dict):
                converted: list[Any] = []
                for key, spec in raw_questions.items():
                    if isinstance(spec, PrimitiveSpec):
                        if spec.name != key:
                            converted.append(spec.model_copy(update={"name": key}))
                        else:
                            converted.append(spec)
                    elif isinstance(spec, dict):
                        spec_copy = dict(spec)
                        spec_copy["name"] = key
                        converted.append(spec_copy)
                    else:
                        converted.append(spec)
                payload["primitives"] = converted
            else:
                payload["primitives"] = raw_questions
        return payload

    @model_validator(mode="after")
    def verify_unique_primitive_names(self) -> SharedStateDecisionRequest:
        """Ensure every primitive in a shared-state request has a distinct name."""
        seen: set[str] = set()
        for primitive in self.primitives:
            if primitive.name in seen:
                raise ValueError(f"Duplicate primitive name '{primitive.name}' in request.")
            seen.add(primitive.name)
        return self

    @property
    def questions(self) -> dict[str, PrimitiveSpec]:
        """Return the primitives indexed by their unique name."""
        return {primitive.name: primitive for primitive in self.primitives}


class SharedStateDecisionResponse(BaseModel):
    """Unified response returned after evaluating all primitives in a single parallel pass."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(..., min_length=1)
    state_hash: str = Field(..., min_length=8)
    results: dict[str, PrimitiveResult] = Field(..., min_length=1)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    gate_action: GateAction
    fast_path_triggered: bool = False
    cache_hit_count: int = Field(default=0, ge=0)
    total_latency_ms: float = Field(..., ge=0.0)
    audit_signature: str = Field(default="")
    gain_metrics: WorkflowGainMetrics | None = None

    @property
    def decisions(self) -> dict[str, PrimitiveResult]:
        """Alias returning the keyed primitive results dictionary."""
        return self.results


class CalibrationPoint(BaseModel):
    """Single labeled calibration observation consisting of raw logits and ground-truth index."""

    model_config = ConfigDict(frozen=True)

    logits: list[float] = Field(..., min_length=2, max_length=255)
    target_index: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_target_in_bounds(self) -> CalibrationPoint:
        """Verify that target_index references a valid class index in logits."""
        if self.target_index >= len(self.logits):
            raise ValueError(
                f"target_index ({self.target_index}) must be < len(logits) ({len(self.logits)})."
            )
        return self


class CalibrationMetricsReport(BaseModel):
    """Empirical reliability metrics quantifying probability calibration quality."""

    model_config = ConfigDict(frozen=True)

    ece: float = Field(..., ge=0.0, le=1.0)
    brier_score: float = Field(..., ge=0.0, le=2.0)
    nll: float = Field(..., ge=0.0)
    mean_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    empirical_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    num_samples: int = Field(default=0, ge=0)
    num_bins: int = Field(default=10, ge=1, le=100)


class CalibrationFitRequest(BaseModel):
    """Request payload for post-hoc RLCD temperature or Platt scaling optimization."""

    model_config = ConfigDict(frozen=True)

    samples: list[CalibrationPoint] = Field(..., min_length=2)
    num_bins: int = Field(default=10, ge=2, le=50)
    method: Literal["temperature", "platt"] = "temperature"

    @model_validator(mode="before")
    @classmethod
    def accept_points_alias(cls, data: Any) -> Any:
        """Allow 'points' as an alias for 'samples' in calibration fit requests."""
        if isinstance(data, dict) and "samples" not in data and "points" in data:
            copied = dict(data)
            copied["samples"] = copied.pop("points")
            return copied
        return data

    @property
    def points(self) -> list[CalibrationPoint]:
        """Alias returning the list of calibration samples."""
        return self.samples


class CalibrationFitResponse(BaseModel):
    """Result of fitting RLCD post-hoc calibration parameters on a validation dataset."""

    model_config = ConfigDict(frozen=True)

    optimal_temperature: float = Field(..., gt=0.0)
    platt_scale_a: float = Field(default=1.0)
    platt_bias_b: float = Field(default=0.0)
    pre_calibration: CalibrationMetricsReport
    post_calibration: CalibrationMetricsReport
    ece_reduction_pct: float
    nll_reduction_pct: float = Field(default=0.0)


class GatewayRoutingRequest(BaseModel):
    """Workflow 1 request: AI Gateway & Hot-Path Cascade Routing."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(..., min_length=1, max_length=32768)
    system_context: str = Field(default="", max_length=16384)
    tenant_tier: Literal["free", "pro", "enterprise"] = "enterprise"
    latency_budget_ms: float = Field(default=50.0, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GatewayRoutingResponse(BaseModel):
    """Workflow 1 response: single-pass AI Gateway routing decision and gain telemetry."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(..., min_length=1)
    selected_pathway: GatewayPathway
    safety_compliant: bool
    safety_probability: float = Field(..., ge=0.0, le=1.0)
    reasoning_complexity_score: float = Field(..., ge=1.0, le=5.0)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    gate_action: GateAction
    frontier_llm_deflected: bool
    primitive_breakdown: dict[str, PrimitiveResult]
    gain_metrics: WorkflowGainMetrics
    audit_signature: str = Field(default="")


class SupportTriageRequest(BaseModel):
    """Workflow 2 request: CRM & Support Ticket Triage with Confidence-Gated Automation."""

    model_config = ConfigDict(frozen=True)

    ticket_id: str = Field(..., min_length=1, max_length=128)
    customer_message: str = Field(..., min_length=3, max_length=32768)
    customer_tier: Literal["standard", "premium", "enterprise"] = "enterprise"
    account_arr_usd: float = Field(default=0.0, ge=0.0)
    previous_interactions_count: int = Field(default=0, ge=0)


class SupportTriageResponse(BaseModel):
    """Workflow 2 response: multi-primitive CRM ticket triage and deterministic confidence gate."""

    model_config = ConfigDict(frozen=True)

    ticket_id: str = Field(..., min_length=1)
    department_queue: DepartmentQueue
    urgent_sla: bool
    urgent_sla_probability: float = Field(..., ge=0.0, le=1.0)
    explicit_defect: bool
    explicit_defect_probability: float = Field(..., ge=0.0, le=1.0)
    customer_frustration_score: float = Field(..., ge=1.0, le=5.0)
    minimum_primitive_confidence: float = Field(..., ge=0.0, le=1.0)
    automation_gate_action: GateAction
    primitive_breakdown: dict[str, PrimitiveResult]
    gain_metrics: WorkflowGainMetrics
    audit_signature: str = Field(default="")

    @property
    def gate_action(self) -> GateAction:
        """Alias returning automation_gate_action."""
        return self.automation_gate_action


class RagInlineEvalRequest(BaseModel):
    """Workflow 3 request: JevEval Inline Production Anti-LLM-as-a-Judge for RAG pipelines."""

    model_config = ConfigDict(frozen=True)

    user_query: str = Field(..., min_length=3, max_length=16384)
    retrieved_context: str = Field(..., min_length=3, max_length=65536)
    candidate_answer: str = Field(..., min_length=1, max_length=32768)
    policy_guidelines: str = Field(
        default="Strictly factual, grounded in context, no PII or unauthorized commitments.",
        min_length=1,
    )


class RagInlineEvalResponse(BaseModel):
    """Workflow 3 response: sub-5ms inline RAG groundedness, policy, and quality judgment."""

    model_config = ConfigDict(frozen=True)

    eval_id: str = Field(..., min_length=1)
    context_faithful: bool
    faithfulness_probability: float = Field(..., ge=0.0, le=1.0)
    policy_compliant: bool
    policy_compliance_probability: float = Field(..., ge=0.0, le=1.0)
    answer_quality_score: float = Field(..., ge=0.0, le=5.0)
    pipeline_action: RagPipelineAction
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    gate_action: GateAction
    inline_verification_passed: bool
    primitive_breakdown: dict[str, PrimitiveResult]
    gain_metrics: WorkflowGainMetrics
    audit_signature: str = Field(default="")


class FinOpsBenchmarkRequest(BaseModel):
    """Workflow 4 request: Unmetered-Output & Shared-State FinOps ROI Simulator."""

    model_config = ConfigDict(frozen=True)

    monthly_requests: int = Field(default=1_000_000, ge=1, le=10_000_000_000)
    questions_per_request: int = Field(default=4, ge=1, le=255)
    avg_state_input_tokens: int = Field(default=600, ge=10, le=128_000)
    avg_question_tokens: int = Field(default=45, ge=5, le=4096)
    avg_system2_output_tokens_per_question: int = Field(default=180, ge=1, le=16_384)
    jev_input_price_per_1m_usd: float = Field(default=0.042, gt=0.0)
    system2_input_price_per_1m_usd: float = Field(default=2.50, gt=0.0)
    system2_output_price_per_1m_usd: float = Field(default=10.00, gt=0.0)


class FinOpsBenchmarkResponse(BaseModel):
    """Workflow 4 response: quantified FinOps ROI, token elimination, and compute savings."""

    model_config = ConfigDict(frozen=True)

    monthly_requests: int = Field(..., ge=1)
    questions_per_request: int = Field(..., ge=1)
    jev_total_input_tokens: int = Field(..., ge=0)
    jev_total_output_tokens_billed: int = Field(default=0, ge=0, le=0)
    system2_total_input_tokens: int = Field(..., ge=0)
    system2_total_output_tokens: int = Field(..., ge=0)
    shared_state_input_token_reduction_pct: float = Field(..., ge=0.0, le=100.0)
    output_token_elimination_pct: float = Field(default=100.0, ge=100.0, le=100.0)
    jev_monthly_cost_usd: float = Field(..., ge=0.0)
    system2_monthly_cost_usd: float = Field(..., ge=0.0)
    monthly_savings_usd: float = Field(..., ge=0.0)
    annual_savings_usd: float = Field(..., ge=0.0)
    net_cost_reduction_pct: float = Field(..., ge=0.0, le=100.0)
    compute_hours_saved_monthly: float = Field(..., ge=0.0)
    gain_metrics: WorkflowGainMetrics


class AuditVerificationResponse(BaseModel):
    """Cryptographic integrity report for the SHA-256 HMAC decision audit chain."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    chain_length: int = Field(..., ge=0)
    latest_digest: str
    verified_at_utc: str
    tampered_index: int | None = None


class HealthResponse(BaseModel):
    """Liveness and readiness status payload for the Jev System-1 engine."""

    model_config = ConfigDict(frozen=True)

    status: Literal["healthy", "degraded"] = "healthy"
    engine: str = "llm-jev-engine"
    version: str = "1.0.0"
    circuit_breaker_state: str = "CLOSED"
    cache_entries: int = Field(default=0, ge=0)
    audit_chain_length: int = Field(default=0, ge=0)


class EngineStatsResponse(BaseModel):
    """Runtime telemetry snapshot of cache efficiency, fast-path hits, and FinOps savings."""

    model_config = ConfigDict(frozen=True)

    total_requests: int = Field(default=0, ge=0)
    total_primitives_evaluated: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    cache_hit_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    fast_path_hits: int = Field(default=0, ge=0)
    total_jev_input_tokens: int = Field(default=0, ge=0)
    total_jev_output_tokens_billed: int = Field(default=0, ge=0)
    cumulative_jev_cost_usd: float = Field(default=0.0, ge=0.0)
    cumulative_system2_cost_avoided_usd: float = Field(default=0.0, ge=0.0)
    circuit_breaker_state: str = "CLOSED"


SystemOneRequest = SharedStateDecisionRequest
SystemOneResponse = SharedStateDecisionResponse
CalibrationReportResponse = CalibrationFitResponse
TicketTriageRequest = SupportTriageRequest
TicketTriageResponse = SupportTriageResponse
JevEvalRequest = RagInlineEvalRequest
JevEvalResponse = RagInlineEvalResponse
