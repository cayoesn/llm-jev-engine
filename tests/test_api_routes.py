"""End-to-end integration tests for FastAPI routes, lifespan, middleware, and resilience handlers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.core.resilience import CircuitBreakerOpenError, RateLimitExceededError
from app.main import app, create_app


@pytest.mark.asyncio
async def test_health_metrics_and_stats_endpoints() -> None:
    """Verify /health, /metrics, and /api/v1/stats return valid telemetry and headers."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health_resp = await client.get(
            "/health",
            headers={"X-Request-ID": "req-health-check-01"},
        )
        assert health_resp.status_code == 200
        assert health_resp.headers["X-Request-ID"] == "req-health-check-01"
        assert float(health_resp.headers["X-Process-Time-Ms"]) >= 0.0
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        assert health_data["engine"] == "llm-jev-engine"

        metrics_resp = await client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert "jev_decisions_total" in metrics_resp.text
        assert "jev_decision_latency_seconds" in metrics_resp.text
        assert "jev_cache_hit_ratio" in metrics_resp.text

        stats_resp = await client.get("/api/v1/stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.json()
        assert stats_data["total_jev_output_tokens_billed"] == 0
        assert stats_data["circuit_breaker_state"] == "CLOSED"


@pytest.mark.asyncio
async def test_system_one_and_single_primitive_endpoints() -> None:
    """Verify /api/v1/system-one, /api/v1/decide, and single-primitive endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        shared_payload = {
            "state": {
                "ticket": "Production database outage and API 500 crash errors requires urgent SRE fix."
            },
            "questions": {
                "queue": {
                    "kind": "choice",
                    "question": "Which queue owns this ticket?",
                    "options": {
                        "sre": "outage database api crash 500 error",
                        "billing": "invoice refund charge",
                    },
                },
                "is_urgent": {
                    "kind": "noul",
                    "question": "Is this an urgent production outage?",
                },
                "severity": {
                    "kind": "score",
                    "question": "What is the incident severity?",
                    "levels": {
                        1: "minor cosmetic question",
                        3: "moderate issue",
                        5: "critical production database outage crash",
                    },
                },
            },
        }
        sys1_resp = await client.post("/api/v1/system-one", json=shared_payload)
        assert sys1_resp.status_code == 200
        sys1_data = sys1_resp.json()
        assert sys1_data["results"]["queue"]["selected_value"] == "sre"
        assert sys1_data["results"]["is_urgent"]["selected_value"] in ("TRUE", "FALSE")
        assert sys1_data["gain_metrics"]["output_tokens_billed"] == 0

        decide_resp = await client.post("/api/v1/decide", json=shared_payload)
        assert decide_resp.status_code == 200
        assert decide_resp.json()["cache_hit_count"] >= 1

        choice_resp = await client.post(
            "/api/v1/primitives/choice",
            json={
                "state": {"message": "Double charged on my invoice need refund"},
                "name": "dept",
                "question": "Which department handles this?",
                "options": {
                    "billing": "invoice charged double refund payment",
                    "sre": "outage crash latency",
                },
            },
        )
        assert choice_resp.status_code == 200
        assert choice_resp.json()["selected_value"] == "billing"

        noul_resp = await client.post(
            "/api/v1/primitives/noul",
            json={
                "state": {"fact": "The service is healthy, verified, and compliant."},
                "name": "verified",
                "question": "Is the service verified and compliant?",
            },
        )
        assert noul_resp.status_code == 200
        assert noul_resp.json()["kind"] == "noul"

        score_resp = await client.post(
            "/api/v1/primitives/score",
            json={
                "state": {"review": "Furious customer cancelling contract immediately."},
                "name": "churn_risk",
                "question": "What is the churn risk level?",
                "levels": {
                    1: "calm happy",
                    3: "neutral",
                    5: "furious cancelling contract immediately",
                },
            },
        )
        assert score_resp.status_code == 200
        assert score_resp.json()["expected_score"] is not None


@pytest.mark.asyncio
async def test_calibration_and_all_four_enterprise_workflow_routes() -> None:
    """Verify RLCD calibration fit and all 4 Enterprise Workflow REST endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        cal_payload = {
            "samples": [
                {"logits": [4.5, 0.5, -1.0], "target_index": 0},
                {"logits": [0.2, 3.8, -0.5], "target_index": 1},
                {"logits": [-1.2, 0.1, 4.1], "target_index": 2},
                {"logits": [3.2, 2.8, 0.0], "target_index": 1},
                {"logits": [2.9, 3.1, -0.4], "target_index": 0},
            ],
            "num_bins": 5,
        }
        cal_resp = await client.post("/api/v1/calibration/fit", json=cal_payload)
        assert cal_resp.status_code == 200
        assert cal_resp.json()["optimal_temperature"] > 0.0

        cal_alias_resp = await client.post("/api/v1/calibrate", json=cal_payload)
        assert cal_alias_resp.status_code == 200

        gw_resp = await client.post(
            "/api/v1/workflows/gateway",
            json={
                "prompt": "Hello ping status check what are support hours?",
                "system_context": "Enterprise AI Gateway",
                "tenant_tier": "enterprise",
            },
        )
        assert gw_resp.status_code == 200
        gw_data = gw_resp.json()
        assert gw_data["selected_pathway"] in (
            "semantic_cache",
            "local_slm_fast",
            "frontier_reasoning_llm",
            "human_review_queue",
        )
        assert gw_data["gain_metrics"]["output_tokens_billed"] == 0

        triage_resp = await client.post(
            "/api/v1/workflows/triage",
            json={
                "ticket_id": "TCK-9901",
                "customer_message": (
                    "We were charged twice on invoice INV-402 and demand an immediate refund!"
                ),
                "customer_tier": "enterprise",
                "account_arr_usd": 180000.0,
            },
        )
        assert triage_resp.status_code == 200
        triage_data = triage_resp.json()
        assert triage_data["department_queue"] == "billing"
        assert triage_data["gain_metrics"]["output_tokens_billed"] == 0

        rag_payload = {
            "user_query": "What is the RPO and encryption standard?",
            "retrieved_context": "Enterprise tier provides 15-minute RPO and AES-256 encryption.",
            "candidate_answer": "Enterprise tier guarantees 15-minute RPO with AES-256 encryption.",
        }
        rag_resp = await client.post("/api/v1/eval/rag", json=rag_payload)
        assert rag_resp.status_code == 200
        assert rag_resp.json()["gain_metrics"]["output_tokens_billed"] == 0

        rag_alias_resp = await client.post("/api/v1/workflows/rag-eval", json=rag_payload)
        assert rag_alias_resp.status_code == 200

        finops_resp = await client.post(
            "/api/v1/workflows/finops-benchmark",
            json={
                "monthly_requests": 2000000,
                "questions_per_request": 4,
                "avg_state_input_tokens": 600,
                "avg_question_tokens": 45,
                "avg_system2_output_tokens_per_question": 180,
            },
        )
        assert finops_resp.status_code == 200
        finops_data = finops_resp.json()
        assert finops_data["jev_total_output_tokens_billed"] == 0
        assert finops_data["net_cost_reduction_pct"] > 99.0

        audit_resp = await client.get("/api/v1/audit/verify")
        assert audit_resp.status_code == 200
        audit_data = audit_resp.json()
        assert audit_data["valid"] is True
        assert audit_data["chain_length"] >= 1

        cache_resp = await client.delete("/api/v1/cache")
        assert cache_resp.status_code == 200
        assert cache_resp.json()["status"] == "cleared"


def test_lifespan_and_resilience_exception_handlers() -> None:
    """Verify lifespan initialization and HTTP 503 / 429 resilience exception handlers."""
    custom_app = create_app()

    @custom_app.get("/trigger-cb-open")
    async def trigger_cb() -> dict[str, str]:
        raise CircuitBreakerOpenError("Simulated circuit open")

    @custom_app.get("/trigger-rate-limit")
    async def trigger_rl() -> dict[str, str]:
        raise RateLimitExceededError("Simulated rate limit exceeded")

    with TestClient(custom_app) as client:
        health = client.get("/health")
        assert health.status_code == 200

        cb_resp = client.get("/trigger-cb-open")
        assert cb_resp.status_code == 503
        assert cb_resp.json()["error"] == "circuit_breaker_open"

        rl_resp = client.get("/trigger-rate-limit")
        assert rl_resp.status_code == 429
        assert rl_resp.json()["error"] == "rate_limit_exceeded"
