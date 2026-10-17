"""Shared pytest fixtures for the llm-jev-engine hermetic test suite."""

from __future__ import annotations

import pytest

from app.api.schemas import (
    PrimitiveResult,
    PrimitiveSpec,
    WorkflowGainMetrics,
)
from app.config import Settings, get_settings


@pytest.fixture()
def test_settings() -> Settings:
    """Return a deterministic Settings instance for unit and integration testing."""
    get_settings.cache_clear()
    return Settings(
        JEV_ENV="test",
        JEV_LOG_LEVEL="DEBUG",
        JEV_SHARED_CACHE_MAX_SIZE=256,
        JEV_SHARED_CACHE_TTL_SECONDS=60.0,
        JEV_DEFAULT_TEMPERATURE=0.85,
        JEV_MIN_CALIBRATED_CONFIDENCE_AUTO=0.80,
        JEV_MIN_CALIBRATED_CONFIDENCE_FALLBACK=0.55,
    )


@pytest.fixture()
def sample_choice_spec() -> PrimitiveSpec:
    """Return a valid choice PrimitiveSpec fixture with semantic rubrics."""
    return PrimitiveSpec(
        name="queue",
        kind="choice",
        question="Which support queue should handle this customer ticket?",
        options={
            "billing": "Payments, duplicate charges, invoices, refund requests",
            "technical_sre": "API errors, production outages, crash logs, latency spikes",
            "sales_account": "Enterprise licensing, seat expansion, contract renewals",
        },
    )


@pytest.fixture()
def sample_noul_spec() -> PrimitiveSpec:
    """Return a valid noul PrimitiveSpec fixture."""
    return PrimitiveSpec(
        name="urgent_sla",
        kind="noul",
        question="Does this customer ticket require an urgent response within 1 hour?",
    )


@pytest.fixture()
def sample_score_spec() -> PrimitiveSpec:
    """Return a valid score PrimitiveSpec fixture with 5 ordinal rubric levels."""
    return PrimitiveSpec(
        name="frustration",
        kind="score",
        question="How frustrated is the customer based on message tone and urgency?",
        levels={
            1: "Calm informational inquiry",
            2: "Slightly annoyed about a minor inconvenience",
            3: "Moderately frustrated by repeated issues",
            4: "Very angry, demanding immediate refund or escalation",
            5: "Critical churn threat or legal compliance escalation",
        },
    )


@pytest.fixture()
def sample_gain_metrics() -> WorkflowGainMetrics:
    """Return a representative WorkflowGainMetrics instance for schema and route tests."""
    return WorkflowGainMetrics(
        jev_latency_ms=1.42,
        system2_baseline_latency_ms=850.0,
        speedup_factor=598.59,
        jev_cost_usd=0.000015,
        system2_baseline_cost_usd=0.0092,
        cost_reduction_pct=99.84,
        output_tokens_billed=0,
        shared_state_amortization_ratio=0.75,
        jev_input_tokens=360,
        system2_total_tokens_avoided=1420,
        calibration_ece_advantage=0.23,
    )


@pytest.fixture()
def sample_primitive_result() -> PrimitiveResult:
    """Return a representative calibrated PrimitiveResult fixture."""
    return PrimitiveResult(
        name="queue",
        kind="choice",
        selected_value="billing",
        probability_distribution={
            "billing": 0.88,
            "technical_sre": 0.08,
            "sales_account": 0.04,
        },
        calibrated_confidence=0.89,
        shannon_entropy_bits=0.64,
        normalized_entropy=0.11,
        credible_interval_95=(0.81, 0.94),
        abstained=False,
        latency_ms=0.65,
    )
