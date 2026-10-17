"""FastAPI v1 REST API router for the Jev System-1 Non-Autoregressive Decision Engine."""

from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.schemas import (
    AuditVerificationResponse,
    CacheEvictionResponse,
    CalibrationFitRequest,
    CalibrationFitResponse,
    EngineStatsResponse,
    FinOpsBenchmarkRequest,
    FinOpsBenchmarkResponse,
    GatewayRoutingRequest,
    GatewayRoutingResponse,
    HealthResponse,
    PrimitiveResult,
    RagInlineEvalRequest,
    RagInlineEvalResponse,
    SharedStateDecisionRequest,
    SharedStateDecisionResponse,
    SingleChoiceRequest,
    SingleNoulRequest,
    SingleScoreRequest,
    SupportTriageRequest,
    SupportTriageResponse,
)
from app.config import get_settings
from app.core.calibration_rlcd import RLCDCalibrator
from app.core.decision_fabric import DecisionFabric
from app.core.enterprise_workflows import EnterpriseWorkflowOrchestrator
from app.core.jev_eval import JevEval
from app.core.observability import PROMETHEUS_REGISTRY
from app.core.parallel_sampler import BoundedDecisionCache, SharedStateParallelSampler
from app.core.primitives_engine import PrimitivesEngine

router = APIRouter()


def ensure_engine_state(request: Request) -> tuple[
    DecisionFabric,
    EnterpriseWorkflowOrchestrator,
    RLCDCalibrator,
]:
    """Ensure singleton engine components are initialized on request.app.state."""
    state = request.app.state
    fabric = getattr(state, "fabric", None)
    workflows = getattr(state, "workflows", None)
    calibrator = getattr(state, "calibrator", None)
    if (
        isinstance(fabric, DecisionFabric)
        and isinstance(workflows, EnterpriseWorkflowOrchestrator)
        and isinstance(calibrator, RLCDCalibrator)
    ):
        return fabric, workflows, calibrator

    settings = get_settings()
    calibrator = RLCDCalibrator(default_temperature=settings.JEV_DEFAULT_TEMPERATURE)
    primitives_engine = PrimitivesEngine(calibrator=calibrator)
    cache = BoundedDecisionCache(
        max_size=settings.JEV_SHARED_CACHE_MAX_SIZE,
        ttl_seconds=settings.JEV_SHARED_CACHE_TTL_SECONDS,
    )
    sampler = SharedStateParallelSampler(
        primitives_engine=primitives_engine,
        cache=cache,
        settings=settings,
    )
    fabric = DecisionFabric(sampler=sampler, settings=settings)
    jev_eval = JevEval(decision_fabric=fabric, settings=settings)
    workflows = EnterpriseWorkflowOrchestrator(
        decision_fabric=fabric,
        jev_eval=jev_eval,
        settings=settings,
    )
    state.settings = settings
    state.calibrator = calibrator
    state.primitives_engine = primitives_engine
    state.cache = cache
    state.sampler = sampler
    state.fabric = fabric
    state.jev_eval = jev_eval
    state.workflows = workflows
    return fabric, workflows, calibrator


@router.get("/health", response_model=HealthResponse, tags=["Observability"])
async def get_health(request: Request) -> HealthResponse:
    """Return engine liveness, circuit breaker state, cache occupancy, and audit chain length."""
    fabric, _, _ = ensure_engine_state(request)
    cb_state = fabric.executor.circuit_breaker.state
    status_val: Literal["healthy", "degraded"] = (
        "healthy" if cb_state == "CLOSED" else "degraded"
    )
    return HealthResponse(
        status=status_val,
        engine="llm-jev-engine",
        version="1.0.0",
        circuit_breaker_state=cb_state,
        cache_entries=fabric.sampler.cache.size,
        audit_chain_length=fabric.audit_chain.length,
    )


@router.get("/metrics", tags=["Observability"])
async def get_prometheus_metrics(request: Request) -> PlainTextResponse:
    """Expose Prometheus metrics in standard text exposition format."""
    fabric, _, _ = ensure_engine_state(request)
    _ = fabric.get_stats()
    payload = generate_latest(PROMETHEUS_REGISTRY).decode("utf-8")
    return PlainTextResponse(content=payload, media_type=CONTENT_TYPE_LATEST)


@router.get("/api/v1/stats", response_model=EngineStatsResponse, tags=["Observability"])
async def get_engine_stats(request: Request) -> EngineStatsResponse:
    """Return runtime telemetry snapshot of cache hit ratio, fast-path hits, and FinOps savings."""
    fabric, _, _ = ensure_engine_state(request)
    return fabric.get_stats()


@router.post(
    "/api/v1/system-one",
    response_model=SharedStateDecisionResponse,
    tags=["System-1 Primitives"],
)
async def evaluate_system_one(
    payload: SharedStateDecisionRequest,
    request: Request,
) -> SharedStateDecisionResponse:
    """Evaluate a batch of non-autoregressive System-1 primitives over a shared state context."""
    fabric, _, _ = ensure_engine_state(request)
    return await fabric.evaluate(payload)


@router.post(
    "/api/v1/decide",
    response_model=SharedStateDecisionResponse,
    tags=["System-1 Primitives"],
)
async def decide_shared_state(
    payload: SharedStateDecisionRequest,
    request: Request,
) -> SharedStateDecisionResponse:
    """Alias endpoint evaluating a SharedStateDecisionRequest in a single parallel pass."""
    fabric, _, _ = ensure_engine_state(request)
    return await fabric.evaluate(payload)


@router.post(
    "/api/v1/primitives/choice",
    response_model=PrimitiveResult,
    tags=["System-1 Primitives"],
)
async def evaluate_choice_primitive(
    payload: SingleChoiceRequest,
    request: Request,
) -> PrimitiveResult:
    """Evaluate a single 'choice' multi-class categorical decision primitive."""
    fabric, _, _ = ensure_engine_state(request)
    shared_resp = await fabric.evaluate(payload.to_shared_request())
    return cast(PrimitiveResult, shared_resp.results[payload.name])


@router.post(
    "/api/v1/primitives/noul",
    response_model=PrimitiveResult,
    tags=["System-1 Primitives"],
)
async def evaluate_noul_primitive(
    payload: SingleNoulRequest,
    request: Request,
) -> PrimitiveResult:
    """Evaluate a single 'noul' binary proposition verification primitive."""
    fabric, _, _ = ensure_engine_state(request)
    shared_resp = await fabric.evaluate(payload.to_shared_request())
    return cast(PrimitiveResult, shared_resp.results[payload.name])


@router.post(
    "/api/v1/primitives/score",
    response_model=PrimitiveResult,
    tags=["System-1 Primitives"],
)
async def evaluate_score_primitive(
    payload: SingleScoreRequest,
    request: Request,
) -> PrimitiveResult:
    """Evaluate a single 'score' ordinal rubric expected-value primitive."""
    fabric, _, _ = ensure_engine_state(request)
    shared_resp = await fabric.evaluate(payload.to_shared_request())
    return cast(PrimitiveResult, shared_resp.results[payload.name])


@router.post(
    "/api/v1/calibration/fit",
    response_model=CalibrationFitResponse,
    tags=["RLCD Calibration"],
)
async def fit_rlcd_calibration(
    payload: CalibrationFitRequest,
    request: Request,
) -> CalibrationFitResponse:
    """Fit post-hoc RLCD temperature and Platt scaling parameters via Golden-Section search."""
    _, _, calibrator = ensure_engine_state(request)
    return calibrator.fit_from_request(payload)


@router.post(
    "/api/v1/calibrate",
    response_model=CalibrationFitResponse,
    tags=["RLCD Calibration"],
)
async def calibrate_alias(
    payload: CalibrationFitRequest,
    request: Request,
) -> CalibrationFitResponse:
    """Alias endpoint for fitting RLCD calibration parameters."""
    _, _, calibrator = ensure_engine_state(request)
    return calibrator.fit_from_request(payload)


@router.post(
    "/api/v1/workflows/gateway",
    response_model=GatewayRoutingResponse,
    tags=["Enterprise Workflows"],
)
async def run_gateway_workflow(
    payload: GatewayRoutingRequest,
    request: Request,
) -> GatewayRoutingResponse:
    """Execute Workflow 1: AI Gateway & Hot-Path Cascade Routing in a single pass."""
    _, workflows, _ = ensure_engine_state(request)
    return await workflows.run_gateway_routing(payload)


@router.post(
    "/api/v1/workflows/triage",
    response_model=SupportTriageResponse,
    tags=["Enterprise Workflows"],
)
async def run_triage_workflow(
    payload: SupportTriageRequest,
    request: Request,
) -> SupportTriageResponse:
    """Execute Workflow 2: CRM & Support Ticket Triage with Confidence-Gated Automation."""
    _, workflows, _ = ensure_engine_state(request)
    return await workflows.run_support_triage(payload)


@router.post(
    "/api/v1/eval/rag",
    response_model=RagInlineEvalResponse,
    tags=["Enterprise Workflows"],
)
async def run_rag_inline_eval(
    payload: RagInlineEvalRequest,
    request: Request,
) -> RagInlineEvalResponse:
    """Execute Workflow 3: JevEval Inline Production Anti-LLM-as-a-Judge for RAG pipelines."""
    _, workflows, _ = ensure_engine_state(request)
    return await workflows.run_rag_inline_eval(payload)


@router.post(
    "/api/v1/workflows/rag-eval",
    response_model=RagInlineEvalResponse,
    tags=["Enterprise Workflows"],
)
async def run_rag_inline_eval_alias(
    payload: RagInlineEvalRequest,
    request: Request,
) -> RagInlineEvalResponse:
    """Alias endpoint executing Workflow 3: JevEval Inline Production Anti-LLM-as-a-Judge."""
    _, workflows, _ = ensure_engine_state(request)
    return await workflows.run_rag_inline_eval(payload)


@router.post(
    "/api/v1/workflows/finops-benchmark",
    response_model=FinOpsBenchmarkResponse,
    tags=["Enterprise Workflows"],
)
async def run_finops_benchmark(
    payload: FinOpsBenchmarkRequest,
    request: Request,
) -> FinOpsBenchmarkResponse:
    """Execute Workflow 4: Unmetered-Output & Shared-State FinOps ROI Benchmark."""
    _, workflows, _ = ensure_engine_state(request)
    return await workflows.run_finops_benchmark(payload)


@router.get(
    "/api/v1/audit/verify",
    response_model=AuditVerificationResponse,
    tags=["Audit & Cache"],
)
async def verify_audit_chain(request: Request) -> AuditVerificationResponse:
    """Verify cryptographic integrity of the SHA-256 HMAC decision audit chain."""
    fabric, _, _ = ensure_engine_state(request)
    return fabric.verify_audit_chain()


@router.delete(
    "/api/v1/cache",
    response_model=CacheEvictionResponse,
    tags=["Audit & Cache"],
)
async def clear_decision_cache(request: Request) -> CacheEvictionResponse:
    """Evict all entries from the Shared-State BoundedDecisionCache."""
    fabric, _, _ = ensure_engine_state(request)
    cleared = fabric.clear_cache()
    return CacheEvictionResponse(cleared_entries=cleared, status="cleared")
