"""Unit and integration tests for the declarative TypeSafeClient SDK."""

from __future__ import annotations

from typing import Any, cast

import pytest

from app.api.schemas import SharedStateDecisionResponse
from app.sdk import AwaitableSystemOneResponse, TypeSafeClient, choice, noul, score


@pytest.mark.asyncio
async def test_typesafe_client_declarative_system_one_async_and_sync() -> None:
    """Verify official-style TypeSafeClient.system_one() in both awaited and sync modes."""
    client = TypeSafeClient()
    state = {"ticket": "I was charged twice this month on my invoice and nobody replies"}
    questions = {
        "queue": choice(
            "Which team handles this?",
            {
                "billing": "payments, refunds, invoices, duplicate charges",
                "technical": "bugs, outages, API integrations, crashes",
                "sales": "pricing, enterprise upgrades, contracts",
            },
        ),
        "urgent": noul("Does the customer need a reply within the hour?"),
        "sentiment": score(
            "How frustrated is the customer?",
            levels={
                1: "calm inquiry",
                2: "mildly annoyed",
                3: "moderately frustrated",
                4: "very angry or demanding refund",
                5: "critical churn or legal escalation risk",
            },
        ),
    }

    awaited_res = await client.system_one(state=state, questions=questions)
    assert isinstance(awaited_res, SharedStateDecisionResponse)
    assert isinstance(awaited_res, AwaitableSystemOneResponse)
    assert "queue" in awaited_res
    assert awaited_res["queue"].selected_value == "billing"
    assert awaited_res.results["urgent"].selected_value == "TRUE"
    assert awaited_res.decisions["sentiment"].expected_score is not None
    assert awaited_res.decisions["sentiment"].expected_score > 3.2
    assert awaited_res.gain_metrics is not None
    assert awaited_res.gain_metrics.output_tokens_billed == 0

    sync_res = client.system_one(state="Duplicate invoice charge refund", questions=questions)
    assert sync_res["queue"].selected_value == "billing"

    async_res = await client.asystem_one(
        state=state,
        questions=[
            choice("Select option", ["billing", "technical"], name="q_choice"),
            noul("Is this billing?", name="q_noul"),
            score("Severity", {1: "low", 5: "high"}, name="q_score"),
        ],
        request_id="sdk-req-001",
        bypass_cache=True,
    )
    assert async_res.request_id == "sdk-req-001"
    assert set(async_res.results.keys()) == {"q_choice", "q_noul", "q_score"}


def test_typesafe_client_invalid_questions_validation() -> None:
    """Verify TypeSafeClient raises descriptive TypeError on malformed question specs."""
    client = TypeSafeClient()
    with pytest.raises(TypeError, match="PrimitiveSpec"):
        client.system_one(state={"a": 1}, questions=cast(Any, {"bad": "not-a-spec"}))
    with pytest.raises(TypeError, match="PrimitiveSpec"):
        client.system_one(state={"a": 1}, questions=cast(Any, ["not-a-spec"]))
    with pytest.raises(TypeError, match="questions must be"):
        client.system_one(state={"a": 1}, questions=cast(Any, 123))
