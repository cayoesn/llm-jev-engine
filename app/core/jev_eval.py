"""Workflow 3: JevEval Deterministic Inline Production Anti-LLM-as-a-Judge Engine."""

from __future__ import annotations

import time

from app.api.schemas import (
    GateAction,
    PrimitiveSpec,
    RagInlineEvalRequest,
    RagInlineEvalResponse,
    RagPipelineAction,
    SharedStateDecisionRequest,
)
from app.config import Settings, get_settings
from app.core.decision_fabric import DecisionFabric
from app.core.observability import record_decision_metrics


class JevEval:
    """Single-pass non-autoregressive RAG and Agent evaluator replacing conversational LLM judges.

    Evaluates groundedness (`context_faithful: noul`), safety/policy adherence
    (`policy_compliant: noul`), continuous rubric quality (`answer_quality: score`),
    and remediation routing (`pipeline_action: choice`) in a single shared-state pass.
    """

    def __init__(
        self,
        decision_fabric: DecisionFabric | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize JevEval with a DecisionFabric and runtime Settings."""
        self._settings = settings or get_settings()
        self._fabric = decision_fabric or DecisionFabric(settings=self._settings)

    @property
    def decision_fabric(self) -> DecisionFabric:
        """Return the underlying DecisionFabric."""
        return self._fabric

    @staticmethod
    def _build_eval_primitives() -> list[PrimitiveSpec]:
        """Construct the 4 canonical System-1 primitives for inline RAG verification."""
        return [
            PrimitiveSpec(
                name="context_faithful",
                kind="noul",
                question=(
                    "Is the candidate answer strictly grounded in and supported by the "
                    "retrieved context without hallucination?"
                ),
                abstention_threshold=0.25,
            ),
            PrimitiveSpec(
                name="policy_compliant",
                kind="noul",
                question=(
                    "Does the candidate answer comply with safety, privacy, and corporate "
                    "policy guidelines?"
                ),
                abstention_threshold=0.25,
            ),
            PrimitiveSpec(
                name="answer_quality",
                kind="score",
                question="What is the overall factual quality and helpfulness of the candidate answer?",
                levels={
                    1: "poor ungrounded fabricated hallucinated answer",
                    2: "weak incomplete or tangential answer",
                    3: "fair partially helpful answer with minor gaps",
                    4: "good accurate grounded answer matching context",
                    5: "excellent comprehensive faithful and clear answer",
                },
                abstention_threshold=0.20,
            ),
            PrimitiveSpec(
                name="pipeline_action",
                kind="choice",
                question="Which inline RAG remediation action should be taken for this response?",
                options={
                    "publish_response": (
                        "accurate grounded faithful answer safe to publish directly"
                    ),
                    "rewrite_with_citations": (
                        "partially grounded answer needing citation refinement"
                    ),
                    "trigger_corrective_rag": (
                        "unverified claims or missing context requiring corrective retrieval"
                    ),
                    "block_and_escalate": (
                        "hallucinated fabricated unsafe or policy violating answer"
                    ),
                },
                abstention_threshold=0.20,
            ),
        ]

    def _assemble_eval_response(
        self,
        request: RagInlineEvalRequest,
        fabric_resp_id: str,
        state_hash: str,
        results: dict,
        fast_path_triggered: bool,
        start_ns: int,
    ) -> RagInlineEvalResponse:
        """Synthesize the calibrated RagInlineEvalResponse from the 4 primitive results."""
        faithful_res = results["context_faithful"]
        policy_res = results["policy_compliant"]
        quality_res = results["answer_quality"]

        faithfulness_prob = float(faithful_res.probability_distribution.get("TRUE", 0.0))
        policy_prob = float(policy_res.probability_distribution.get("TRUE", 0.0))
        context_faithful = bool(faithful_res.selected_value == "TRUE" and not fast_path_triggered)
        policy_compliant = bool(policy_res.selected_value == "TRUE" and not fast_path_triggered)
        quality_score = float(
            quality_res.expected_score
            if quality_res.expected_score is not None
            else quality_res.selected_value
        )

        if fast_path_triggered or not policy_compliant:
            pipeline_action: RagPipelineAction = "block_and_escalate"
            hallucination_severity = "critical"
            gate_action: GateAction = "ESCALATE_HITL"
            inline_passed = False
        elif not context_faithful:
            if faithfulness_prob < 0.30:
                pipeline_action = "block_and_escalate"
                hallucination_severity = "critical"
                gate_action = "ESCALATE_HITL"
            else:
                pipeline_action = "trigger_corrective_rag"
                hallucination_severity = "major"
                gate_action = "FALLBACK_SYSTEM2"
            inline_passed = False
        else:
            if quality_score >= 3.2:
                pipeline_action = "publish_response"
                hallucination_severity = "none"
                gate_action = "AUTO_EXECUTE"
                inline_passed = True
            else:
                pipeline_action = "rewrite_with_citations"
                hallucination_severity = "minor"
                gate_action = "FALLBACK_SYSTEM2"
                inline_passed = False

        overall_conf = round(
            min(r.calibrated_confidence for r in results.values()),
            6,
        )
        elapsed_ms = max(0.05, (time.perf_counter_ns() - start_ns) / 1_000_000.0)
        combined_text = (
            f"{request.user_query} {request.retrieved_context} "
            f"{request.candidate_answer} {request.policy_guidelines}"
        )
        token_estimate = max(16, len(combined_text) // 4)
        gain_metrics = self._fabric.finops_tracker.compute_gain_metrics(
            state_input_tokens=token_estimate,
            num_primitives=4,
            jev_latency_ms=elapsed_ms,
            system2_baseline_latency_ms=self._settings.JEV_SYSTEM2_JUDGE_BASELINE_LATENCY_MS,
            fast_path_triggered=fast_path_triggered,
        )
        audit_rec = self._fabric.audit_chain.append(
            request_id=fabric_resp_id,
            state_hash=state_hash,
            gate_action=gate_action,
            overall_confidence=overall_conf,
            decisions={
                "context_faithful": context_faithful,
                "policy_compliant": policy_compliant,
                "answer_quality_score": round(quality_score, 4),
                "pipeline_action": pipeline_action,
            },
        )
        record_decision_metrics(
            workflow="jev_eval_rag",
            gate_action=gate_action,
            primitives_map={k: v.kind for k, v in results.items()},
            latency_seconds=elapsed_ms / 1000.0,
        )
        return RagInlineEvalResponse(
            eval_id=fabric_resp_id,
            context_faithful=context_faithful,
            faithfulness_probability=round(faithfulness_prob, 6),
            policy_compliant=policy_compliant,
            policy_compliance_probability=round(policy_prob, 6),
            answer_quality_score=round(quality_score, 4),
            pipeline_action=pipeline_action,
            overall_confidence=overall_conf,
            gate_action=gate_action,
            inline_verification_passed=inline_passed,
            primitive_breakdown=results,
            gain_metrics=gain_metrics,
            audit_signature=audit_rec.signature,
            hallucination_severity=hallucination_severity,
        )

    async def evaluate_rag(self, request: RagInlineEvalRequest) -> RagInlineEvalResponse:
        """Evaluate a RAG candidate response asynchronously in a single shared-state pass.

        Args:
            request: Validated RagInlineEvalRequest with query, context, and candidate answer.

        Returns:
            RagInlineEvalResponse with groundedness, policy compliance, quality score, and gain metrics.
        """
        start_ns = time.perf_counter_ns()
        state_payload = {
            "user_query": request.user_query,
            "retrieved_context": request.retrieved_context,
            "candidate_answer": request.candidate_answer,
            "policy_guidelines": request.policy_guidelines,
        }
        decision_req = SharedStateDecisionRequest(
            state=state_payload,
            primitives=self._build_eval_primitives(),
        )
        fabric_resp = await self._fabric.evaluate(decision_req)
        return self._assemble_eval_response(
            request=request,
            fabric_resp_id=fabric_resp.request_id,
            state_hash=fabric_resp.state_hash,
            results=fabric_resp.results,
            fast_path_triggered=fabric_resp.fast_path_triggered,
            start_ns=start_ns,
        )

    async def evaluate(self, request: RagInlineEvalRequest) -> RagInlineEvalResponse:
        """Alias for evaluate_rag()."""
        return await self.evaluate_rag(request)

    def evaluate_rag_sync(self, request: RagInlineEvalRequest) -> RagInlineEvalResponse:
        """Evaluate a RAG candidate response synchronously in a single shared-state pass."""
        start_ns = time.perf_counter_ns()
        state_payload = {
            "user_query": request.user_query,
            "retrieved_context": request.retrieved_context,
            "candidate_answer": request.candidate_answer,
            "policy_guidelines": request.policy_guidelines,
        }
        decision_req = SharedStateDecisionRequest(
            state=state_payload,
            primitives=self._build_eval_primitives(),
        )
        fabric_resp = self._fabric.evaluate_sync(decision_req)
        return self._assemble_eval_response(
            request=request,
            fabric_resp_id=fabric_resp.request_id,
            state_hash=fabric_resp.state_hash,
            results=fabric_resp.results,
            fast_path_triggered=fabric_resp.fast_path_triggered,
            start_ns=start_ns,
        )


JevEvalEngine = JevEval
