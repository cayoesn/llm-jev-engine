"""Comprehensive unit tests for configuration and immutable Pydantic v2 schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    AuditVerificationResponse,
    CalibrationFitRequest,
    CalibrationFitResponse,
    CalibrationMetricsReport,
    CalibrationPoint,
    EngineStatsResponse,
    FinOpsBenchmarkRequest,
    FinOpsBenchmarkResponse,
    GatewayRoutingRequest,
    GatewayRoutingResponse,
    HealthResponse,
    PrimitiveResult,
    PrimitiveSpec,
    RagInlineEvalRequest,
    RagInlineEvalResponse,
    SharedStateDecisionRequest,
    SharedStateDecisionResponse,
    SupportTriageRequest,
    SupportTriageResponse,
    WorkflowGainMetrics,
)
from app.config import Settings, get_settings


def test_settings_defaults_and_aliases(test_settings: Settings) -> None:
    """Verify Settings defaults, FinOps unmetered pricing, and property aliases."""
    cached = get_settings()
    assert isinstance(cached, Settings)
    assert test_settings.JEV_INPUT_PRICE_PER_1M_USD == pytest.approx(0.042)
    assert test_settings.JEV_OUTPUT_PRICE_PER_1M_USD == pytest.approx(0.0)
    assert test_settings.JEV_SYSTEM2_INPUT_PRICE_PER_1M_USD == pytest.approx(2.50)
    assert test_settings.JEV_SYSTEM2_OUTPUT_PRICE_PER_1M_USD == pytest.approx(10.00)
    assert test_settings.CONFIDENCE_AUTO_THRESHOLD == pytest.approx(0.80)
    assert test_settings.CONFIDENCE_FALLBACK_THRESHOLD == pytest.approx(0.55)
    assert test_settings.JEV_COST_PER_1M_INPUT_TOKENS_USD == pytest.approx(0.042)
    assert test_settings.JEV_COST_PER_1M_OUTPUT_TOKENS_USD == pytest.approx(0.0)
    assert test_settings.SYSTEM2_LLM_COST_PER_1M_INPUT_USD == pytest.approx(2.50)
    assert test_settings.SYSTEM2_LLM_COST_PER_1M_OUTPUT_USD == pytest.approx(10.00)


def test_primitive_spec_valid_choice_noul_score(
    sample_choice_spec: PrimitiveSpec,
    sample_noul_spec: PrimitiveSpec,
    sample_score_spec: PrimitiveSpec,
) -> None:
    """Verify valid creation and immutability of choice, noul, and score PrimitiveSpecs."""
    assert sample_choice_spec.kind == "choice"
    assert sample_choice_spec.options is not None
    assert len(sample_choice_spec.options) == 3
    assert sample_noul_spec.kind == "noul"
    assert sample_noul_spec.options is None
    assert sample_score_spec.kind == "score"
    assert sample_score_spec.levels is not None
    assert len(sample_score_spec.levels) == 5

    with pytest.raises(ValidationError):
        setattr(sample_choice_spec, "name", "mutated_name")


def test_primitive_spec_list_options_and_string_levels_normalization() -> None:
    """Verify that list options and string integer level keys are normalized properly."""
    choice_from_list = PrimitiveSpec.model_validate(
        {
            "name": "route",
            "kind": "choice",
            "question": "Select the best pathway",
            "options": ["cache", "slm", "frontier"],
        }
    )
    assert choice_from_list.options == {
        "cache": "cache",
        "slm": "slm",
        "frontier": "frontier",
    }

    score_from_str_keys = PrimitiveSpec.model_validate(
        {
            "name": "complexity",
            "kind": "score",
            "question": "Rate complexity from 1 to 3",
            "levels": {"1": "low", "2": "medium", "3": "high"},
        }
    )
    assert score_from_str_keys.levels == {1: "low", 2: "medium", 3: "high"}


def test_primitive_spec_invalid_constraints() -> None:
    """Verify validation errors when primitive invariants are violated."""
    with pytest.raises(ValidationError):
        PrimitiveSpec(
            name="bad_choice",
            kind="choice",
            question="Only one option provided?",
            options={"only_one": "single rubric"},
        )

    with pytest.raises(ValidationError):
        PrimitiveSpec(
            name="choice_with_levels",
            kind="choice",
            question="Which option?",
            options={"a": "opt a", "b": "opt b"},
            levels={1: "lvl 1", 2: "lvl 2"},
        )

    with pytest.raises(ValidationError):
        PrimitiveSpec(
            name="bad_score",
            kind="score",
            question="Only one level?",
            levels={1: "only one"},
        )

    with pytest.raises(ValidationError):
        PrimitiveSpec(
            name="score_with_options",
            kind="score",
            question="Rate quality",
            levels={1: "low", 2: "high"},
            options={"a": "opt a", "b": "opt b"},
        )

    with pytest.raises(ValidationError):
        PrimitiveSpec(
            name="noul_with_options",
            kind="noul",
            question="Is this grounded?",
            options={"yes": "true", "no": "false"},
        )

    with pytest.raises(ValidationError):
        PrimitiveSpec.model_validate(
            {
                "name": "bad_list_option",
                "kind": "choice",
                "question": "Empty option in list",
                "options": ["valid", "   "],
            }
        )

    with pytest.raises(ValidationError):
        PrimitiveSpec(
            name="bad_dict_key",
            kind="choice",
            question="Empty key in dict",
            options={"": "desc", "valid": "desc"},
        )

    with pytest.raises(ValidationError):
        PrimitiveSpec(
            name="bad_dict_val",
            kind="choice",
            question="Empty value in dict",
            options={"k1": "  ", "k2": "desc"},
        )

    with pytest.raises(ValidationError):
        PrimitiveSpec.model_validate(
            {
                "name": "bad_options_type",
                "kind": "choice",
                "question": "Invalid options type",
                "options": 12345,
            }
        )

    with pytest.raises(ValidationError):
        PrimitiveSpec.model_validate(
            {
                "name": "bad_levels_type",
                "kind": "score",
                "question": "Invalid levels type",
                "levels": ["low", "high"],
            }
        )

    with pytest.raises(ValidationError):
        PrimitiveSpec(
            name="bad_level_rubric",
            kind="score",
            question="Empty level rubric",
            levels={1: "valid", 2: "   "},
        )


def test_primitive_result_validations_and_immutability(
    sample_primitive_result: PrimitiveResult,
) -> None:
    """Verify probability simplex and credible interval validations on PrimitiveResult."""
    assert sample_primitive_result.selected_value == "billing"
    assert sample_primitive_result.gate_action == "AUTO_EXECUTE"

    with pytest.raises(ValidationError):
        setattr(sample_primitive_result, "abstained", True)

    with pytest.raises(ValidationError):
        PrimitiveResult(
            name="invalid_sum",
            kind="noul",
            selected_value="yes",
            probability_distribution={"yes": 0.40, "no": 0.40},
            calibrated_confidence=0.5,
            shannon_entropy_bits=0.9,
            normalized_entropy=0.9,
            credible_interval_95=(0.2, 0.6),
            latency_ms=0.5,
        )

    with pytest.raises(ValidationError):
        PrimitiveResult(
            name="invalid_prob_range",
            kind="noul",
            selected_value="yes",
            probability_distribution={"yes": 1.2, "no": -0.2},
            calibrated_confidence=0.5,
            shannon_entropy_bits=0.9,
            normalized_entropy=0.9,
            credible_interval_95=(0.2, 0.6),
            latency_ms=0.5,
        )

    with pytest.raises(ValidationError):
        PrimitiveResult(
            name="inverted_ci",
            kind="noul",
            selected_value="yes",
            probability_distribution={"yes": 0.7, "no": 0.3},
            calibrated_confidence=0.6,
            shannon_entropy_bits=0.8,
            normalized_entropy=0.8,
            credible_interval_95=(0.85, 0.45),
            latency_ms=0.5,
        )


def test_shared_state_decision_request_and_response(
    sample_choice_spec: PrimitiveSpec,
    sample_noul_spec: PrimitiveSpec,
    sample_primitive_result: PrimitiveResult,
    sample_gain_metrics: WorkflowGainMetrics,
) -> None:
    """Verify SharedStateDecisionRequest coercion of primitives/questions and response aliases."""
    req_from_list = SharedStateDecisionRequest(
        state={"ticket": "Double billed on invoice"},
        primitives=[sample_choice_spec, sample_noul_spec],
    )
    assert set(req_from_list.questions.keys()) == {"queue", "urgent_sla"}

    same_name_spec = PrimitiveSpec(
        name="same_key",
        kind="noul",
        question="Does the key match the spec name?",
    )
    req_from_questions_dict = SharedStateDecisionRequest.model_validate(
        {
            "state": {"ticket": "Double billed"},
            "questions": {
                "custom_queue": sample_choice_spec,
                "same_key": same_name_spec,
                "raw_noul": {
                    "kind": "noul",
                    "question": "Is this a duplicate billing dispute?",
                },
            },
        }
    )
    assert set(req_from_questions_dict.questions.keys()) == {
        "custom_queue",
        "same_key",
        "raw_noul",
    }

    req_from_questions_list = SharedStateDecisionRequest.model_validate(
        {
            "state": {"ticket": "Double billed"},
            "questions": [sample_choice_spec],
        }
    )
    assert len(req_from_questions_list.primitives) == 1

    with pytest.raises(ValidationError):
        SharedStateDecisionRequest(
            state={"ticket": "Double billed"},
            primitives=[sample_choice_spec, sample_choice_spec],
        )

    resp = SharedStateDecisionResponse(
        request_id="req-001",
        state_hash="a1b2c3d4e5f6",
        results={"queue": sample_primitive_result},
        overall_confidence=0.89,
        gate_action="AUTO_EXECUTE",
        total_latency_ms=1.1,
        gain_metrics=sample_gain_metrics,
    )
    assert resp.decisions["queue"].selected_value == "billing"
    assert resp.gain_metrics is not None
    assert resp.gain_metrics.system2_estimated_latency_ms == pytest.approx(850.0)
    assert resp.gain_metrics.jev_output_tokens == 0
    assert resp.gain_metrics.system2_cost_usd == pytest.approx(0.0092)
    assert resp.gain_metrics.cost_savings_pct == pytest.approx(99.84)


def test_calibration_models_validation() -> None:
    """Verify CalibrationPoint bounds, CalibrationFitRequest aliases, and FitResponse."""
    p1 = CalibrationPoint(logits=[2.4, -0.8, 0.1], target_index=0)
    p2 = CalibrationPoint(logits=[-1.1, 3.2, 0.4], target_index=1)
    assert p1.target_index == 0

    with pytest.raises(ValidationError):
        CalibrationPoint(logits=[1.0, 2.0], target_index=2)

    fit_req = CalibrationFitRequest.model_validate({"points": [p1, p2], "num_bins": 10})
    assert len(fit_req.points) == 2

    pre_report = CalibrationMetricsReport(
        ece=0.18,
        brier_score=0.24,
        nll=0.62,
        mean_confidence=0.88,
        empirical_accuracy=0.70,
        num_samples=2,
        num_bins=10,
    )
    post_report = CalibrationMetricsReport(
        ece=0.02,
        brier_score=0.14,
        nll=0.38,
        mean_confidence=0.72,
        empirical_accuracy=0.70,
        num_samples=2,
        num_bins=10,
    )
    fit_resp = CalibrationFitResponse(
        optimal_temperature=1.42,
        pre_calibration=pre_report,
        post_calibration=post_report,
        ece_reduction_pct=88.89,
        nll_reduction_pct=38.71,
    )
    assert fit_resp.optimal_temperature == pytest.approx(1.42)


def test_four_enterprise_workflow_models_and_system_contracts(
    sample_primitive_result: PrimitiveResult,
    sample_gain_metrics: WorkflowGainMetrics,
) -> None:
    """Verify all 4 Enterprise Workflow request/response models and audit/health contracts."""
    gw_req = GatewayRoutingRequest(prompt="Reset my password")
    gw_resp = GatewayRoutingResponse(
        request_id="gw-1",
        selected_pathway="local_slm_fast",
        safety_compliant=True,
        safety_probability=0.98,
        reasoning_complexity_score=1.45,
        overall_confidence=0.91,
        gate_action="AUTO_EXECUTE",
        frontier_llm_deflected=True,
        primitive_breakdown={"queue": sample_primitive_result},
        gain_metrics=sample_gain_metrics,
    )
    assert gw_req.tenant_tier == "enterprise"
    assert gw_resp.frontier_llm_deflected is True

    triage_req = SupportTriageRequest(
        ticket_id="TCK-900",
        customer_message="We were charged twice on invoice INV-42!",
    )
    triage_resp = SupportTriageResponse(
        ticket_id=triage_req.ticket_id,
        department_queue="billing",
        urgent_sla=True,
        urgent_sla_probability=0.92,
        explicit_defect=False,
        explicit_defect_probability=0.08,
        customer_frustration_score=4.15,
        minimum_primitive_confidence=0.86,
        automation_gate_action="AUTO_EXECUTE",
        primitive_breakdown={"queue": sample_primitive_result},
        gain_metrics=sample_gain_metrics,
    )
    assert triage_resp.gate_action == "AUTO_EXECUTE"

    rag_req = RagInlineEvalRequest(
        user_query="What is the SLA uptime guarantee?",
        retrieved_context="Enterprise plans include a 99.99% uptime SLA guarantee.",
        candidate_answer="The uptime guarantee for enterprise plans is 99.99%.",
    )
    rag_resp = RagInlineEvalResponse(
        eval_id="eval-101",
        context_faithful=True,
        faithfulness_probability=0.97,
        policy_compliant=True,
        policy_compliance_probability=0.99,
        answer_quality_score=0.94,
        pipeline_action="publish_response",
        overall_confidence=0.93,
        gate_action="AUTO_EXECUTE",
        inline_verification_passed=True,
        primitive_breakdown={"queue": sample_primitive_result},
        gain_metrics=sample_gain_metrics,
    )
    assert rag_req.user_query.startswith("What")
    assert rag_resp.inline_verification_passed is True

    finops_req = FinOpsBenchmarkRequest(monthly_requests=2_000_000, questions_per_request=4)
    finops_resp = FinOpsBenchmarkResponse(
        monthly_requests=finops_req.monthly_requests,
        questions_per_request=finops_req.questions_per_request,
        jev_total_input_tokens=1_560_000_000,
        jev_total_output_tokens_billed=0,
        system2_total_input_tokens=5_160_000_000,
        system2_total_output_tokens=1_440_000_000,
        shared_state_input_token_reduction_pct=69.77,
        output_token_elimination_pct=100.0,
        jev_monthly_cost_usd=65.52,
        system2_monthly_cost_usd=27300.0,
        monthly_savings_usd=27234.48,
        annual_savings_usd=326813.76,
        net_cost_reduction_pct=99.76,
        compute_hours_saved_monthly=1886.67,
        gain_metrics=sample_gain_metrics,
    )
    assert finops_resp.jev_total_output_tokens_billed == 0

    audit_resp = AuditVerificationResponse(
        valid=True,
        chain_length=12,
        latest_digest="abc123def456",
        verified_at_utc="2026-10-17T16:55:00Z",
    )
    health_resp = HealthResponse()
    stats_resp = EngineStatsResponse()
    assert audit_resp.valid is True
    assert health_resp.status == "healthy"
    assert stats_resp.total_jev_output_tokens_billed == 0
