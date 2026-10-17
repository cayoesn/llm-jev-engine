"""Workflows 1, 2, 3, and 4: Real-World Enterprise System-1 Orchestrator and FinOps Engine."""

from __future__ import annotations

import time
from typing import cast

from app.api.schemas import (
    DepartmentQueue,
    FinOpsBenchmarkRequest,
    FinOpsBenchmarkResponse,
    GateAction,
    GatewayPathway,
    GatewayRoutingRequest,
    GatewayRoutingResponse,
    PrimitiveSpec,
    RagInlineEvalRequest,
    RagInlineEvalResponse,
    SharedStateDecisionRequest,
    SupportTriageRequest,
    SupportTriageResponse,
    WorkflowGainMetrics,
)
from app.config import Settings, get_settings
from app.core.decision_fabric import DecisionFabric
from app.core.jev_eval import JevEval
from app.core.observability import record_decision_metrics


class EnterpriseWorkflowOrchestrator:
    """Executes all 4 production Jev Enterprise Workflows with quantified gain telemetry.

    Workflows:
        1. AI Gateway & Hot-Path Cascade Routing (`run_gateway_routing` / `route_ai_gateway`)
        2. CRM & Support Ticket Triage (`run_support_triage` / `triage_crm_ticket`)
        3. Inline Production Anti-LLM-as-a-Judge (`run_rag_inline_eval` / `evaluate_rag_inline`)
        4. Unmetered-Output FinOps ROI Benchmark (`run_finops_benchmark`)
    """

    def __init__(
        self,
        decision_fabric: DecisionFabric | None = None,
        jev_eval: JevEval | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the EnterpriseWorkflowOrchestrator."""
        self._settings = settings or get_settings()
        self._fabric = decision_fabric or DecisionFabric(settings=self._settings)
        self._jev_eval = jev_eval or JevEval(
            decision_fabric=self._fabric,
            settings=self._settings,
        )

    @property
    def decision_fabric(self) -> DecisionFabric:
        """Return the underlying DecisionFabric."""
        return self._fabric

    @property
    def jev_eval(self) -> JevEval:
        """Return the underlying JevEval engine."""
        return self._jev_eval

    @staticmethod
    def _build_gateway_primitives() -> list[PrimitiveSpec]:
        """Construct the 3 concurrent primitives for Workflow 1 (AI Gateway Routing)."""
        return [
            PrimitiveSpec(
                name="compute_pathway",
                kind="choice",
                question="Which compute pathway should handle this incoming gateway prompt?",
                options={
                    "semantic_cache": (
                        "hello hi ping status faq reset password greeting simple hours thanks"
                    ),
                    "local_slm_fast": (
                        "routine summary formatting classification standard extraction"
                    ),
                    "frontier_reasoning_llm": (
                        "distributed consensus theorem proof multi architecture optimization "
                        "eigenvalue stochastic cryptographic raft paxos byzantine"
                    ),
                    "human_review_queue": (
                        "ignore bypass jailbreak override injection reveal system_prompt "
                        "hack exploit malware ssn"
                    ),
                },
                abstention_threshold=0.20,
            ),
            PrimitiveSpec(
                name="safety_compliant",
                kind="noul",
                question=(
                    "Is the prompt safe, policy-compliant, and free of prompt injection "
                    "or jailbreak attempts?"
                ),
                abstention_threshold=0.25,
            ),
            PrimitiveSpec(
                name="reasoning_complexity",
                kind="score",
                question="What is the reasoning depth and analytical complexity required?",
                levels={
                    1: "trivial greeting ping status check or simple FAQ lookup",
                    2: "basic single-step factual question or formatting",
                    3: "moderate analytical summarization or coding task",
                    4: "advanced multi-step architectural or mathematical optimization",
                    5: "frontier theorem proof distributed consensus or formal verification",
                },
                abstention_threshold=0.20,
            ),
        ]

    @staticmethod
    def _build_triage_primitives() -> list[PrimitiveSpec]:
        """Construct the 4 concurrent primitives for Workflow 2 (CRM & Support Triage)."""
        return [
            PrimitiveSpec(
                name="department_queue",
                kind="choice",
                question="Which enterprise support department queue owns this customer ticket?",
                options={
                    "billing": (
                        "payment payments refund refunds invoice invoices charged charge "
                        "charges double twice credit subscription fee overcharged"
                    ),
                    "technical_sre": (
                        "bug bugs outage outages api crash crashes error errors latency "
                        "sre timeout production defect stacktrace 500 502 503 database"
                    ),
                    "sales_account": (
                        "pricing upgrade upgrades contract contracts enterprise license "
                        "licensing seat seats quote renewal discount expansion"
                    ),
                    "legal_compliance": (
                        "compliance gdpr hipaa subpoena lawsuit attorney lawyer "
                        "regulation regulatory breach nda arbitration"
                    ),
                },
                abstention_threshold=0.35,
            ),
            PrimitiveSpec(
                name="urgent_sla",
                kind="noul",
                question="Does this customer ticket require an urgent SLA response within one hour?",
                abstention_threshold=0.25,
            ),
            PrimitiveSpec(
                name="explicit_defect",
                kind="noul",
                question="Does this ticket report a technical software bug, crash, or outage defect?",
                abstention_threshold=0.25,
            ),
            PrimitiveSpec(
                name="customer_frustration",
                kind="score",
                question="How frustrated or dissatisfied is the customer in this ticket?",
                levels={
                    1: "calm polite inquiry thank thanks informational",
                    2: "mildly annoyed routine question",
                    3: "moderately frustrated waiting for reply",
                    4: "very angry charged twice demanding refund unacceptable",
                    5: "furious critical churn cancelling lawsuit immediate escalation",
                },
                abstention_threshold=0.20,
            ),
        ]

    async def run_gateway_routing(
        self,
        request: GatewayRoutingRequest,
    ) -> GatewayRoutingResponse:
        """Execute Workflow 1: AI Gateway & Hot-Path Cascade Routing in a single pass.

        Args:
            request: Validated GatewayRoutingRequest.

        Returns:
            GatewayRoutingResponse with selected compute pathway, safety status, and gain metrics.
        """
        start_ns = time.perf_counter_ns()
        state_payload = {
            "prompt": request.prompt,
            "system_context": request.system_context,
            "tenant_tier": request.tenant_tier,
            "metadata": request.metadata,
        }
        decision_req = SharedStateDecisionRequest(
            state=state_payload,
            primitives=self._build_gateway_primitives(),
        )
        fabric_resp = await self._fabric.evaluate(decision_req)
        results = fabric_resp.results

        pathway_res = results["compute_pathway"]
        safety_res = results["safety_compliant"]
        complexity_res = results["reasoning_complexity"]

        safety_prob = float(safety_res.probability_distribution.get("TRUE", 0.0))
        safety_ok = bool(
            safety_res.selected_value == "TRUE" and not fabric_resp.fast_path_triggered
        )
        complexity_score = float(
            complexity_res.expected_score
            if complexity_res.expected_score is not None
            else complexity_res.selected_value
        )

        if fabric_resp.fast_path_triggered or not safety_ok:
            selected_pathway: GatewayPathway = "human_review_queue"
            gate_action: GateAction = "ESCALATE_HITL"
        elif (
            complexity_score >= 3.4
            or pathway_res.selected_value == "frontier_reasoning_llm"
        ):
            selected_pathway = "frontier_reasoning_llm"
            gate_action = fabric_resp.gate_action
        elif complexity_score <= 1.85 or pathway_res.selected_value == "semantic_cache":
            selected_pathway = "semantic_cache"
            gate_action = "AUTO_EXECUTE"
        else:
            raw_choice = str(pathway_res.selected_value)
            if raw_choice in (
                "semantic_cache",
                "local_slm_fast",
                "frontier_reasoning_llm",
                "human_review_queue",
            ):
                selected_pathway = cast(GatewayPathway, raw_choice)
            else:
                selected_pathway = "local_slm_fast"
            gate_action = fabric_resp.gate_action

        frontier_deflected = bool(selected_pathway != "frontier_reasoning_llm")
        elapsed_ms = max(0.05, (time.perf_counter_ns() - start_ns) / 1_000_000.0)
        token_est = max(8, len(f"{request.prompt} {request.system_context}") // 4)
        gain_metrics = self._fabric.finops_tracker.compute_gain_metrics(
            state_input_tokens=token_est,
            num_primitives=3,
            jev_latency_ms=elapsed_ms,
            system2_baseline_latency_ms=self._settings.JEV_SYSTEM2_GATEWAY_BASELINE_LATENCY_MS,
            fast_path_triggered=fabric_resp.fast_path_triggered,
        )
        record_decision_metrics(
            workflow="gateway_routing",
            gate_action=gate_action,
            primitives_map={k: v.kind for k, v in results.items()},
            latency_seconds=elapsed_ms / 1000.0,
        )
        return GatewayRoutingResponse(
            request_id=fabric_resp.request_id,
            selected_pathway=selected_pathway,
            safety_compliant=safety_ok,
            safety_probability=round(safety_prob, 6),
            reasoning_complexity_score=round(complexity_score, 4),
            overall_confidence=fabric_resp.overall_confidence,
            gate_action=gate_action,
            frontier_llm_deflected=frontier_deflected,
            primitive_breakdown=results,
            gain_metrics=gain_metrics,
            audit_signature=fabric_resp.audit_signature,
        )

    async def route_ai_gateway(
        self,
        request: GatewayRoutingRequest,
    ) -> GatewayRoutingResponse:
        """Alias for run_gateway_routing()."""
        return await self.run_gateway_routing(request)

    async def run_support_triage(
        self,
        request: SupportTriageRequest,
    ) -> SupportTriageResponse:
        """Execute Workflow 2: CRM & Support Ticket Triage with Confidence-Gated Automation.

        Args:
            request: Validated SupportTriageRequest.

        Returns:
            SupportTriageResponse with queue assignment, SLA urgency, frustration score, and gate.
        """
        start_ns = time.perf_counter_ns()
        state_payload = {
            "ticket_id": request.ticket_id,
            "customer_message": request.customer_message,
            "customer_tier": request.customer_tier,
            "account_arr_usd": request.account_arr_usd,
            "previous_interactions_count": request.previous_interactions_count,
        }
        decision_req = SharedStateDecisionRequest(
            state=state_payload,
            primitives=self._build_triage_primitives(),
        )
        fabric_resp = await self._fabric.evaluate(decision_req)
        results = fabric_resp.results

        queue_res = results["department_queue"]
        urgent_res = results["urgent_sla"]
        defect_res = results["explicit_defect"]
        frust_res = results["customer_frustration"]

        raw_queue = str(queue_res.selected_value)
        if raw_queue in ("billing", "technical_sre", "sales_account", "legal_compliance"):
            dept_queue = cast(DepartmentQueue, raw_queue)
        else:
            dept_queue = "technical_sre"

        urgent_prob = float(urgent_res.probability_distribution.get("TRUE", 0.0))
        urgent_bool = bool(urgent_res.selected_value == "TRUE" or urgent_prob >= 0.55)
        defect_prob = float(defect_res.probability_distribution.get("TRUE", 0.0))
        defect_bool = bool(defect_res.selected_value == "TRUE" or defect_prob >= 0.55)
        frustration_score = float(
            frust_res.expected_score
            if frust_res.expected_score is not None
            else frust_res.selected_value
        )

        min_conf = float(queue_res.calibrated_confidence)
        if queue_res.abstained or fabric_resp.fast_path_triggered:
            gate_action: GateAction = "ESCALATE_HITL"
        elif min_conf >= self._settings.JEV_MIN_CALIBRATED_CONFIDENCE_AUTO:
            gate_action = "AUTO_EXECUTE"
        elif min_conf >= self._settings.JEV_MIN_CALIBRATED_CONFIDENCE_FALLBACK:
            gate_action = "FALLBACK_SYSTEM2"
        else:
            gate_action = "ESCALATE_HITL"

        elapsed_ms = max(0.05, (time.perf_counter_ns() - start_ns) / 1_000_000.0)
        token_est = max(12, len(request.customer_message) // 4 + 16)
        gain_metrics = self._fabric.finops_tracker.compute_gain_metrics(
            state_input_tokens=token_est,
            num_primitives=4,
            jev_latency_ms=elapsed_ms,
            system2_baseline_latency_ms=self._settings.JEV_SYSTEM2_GATEWAY_BASELINE_LATENCY_MS,
            fast_path_triggered=fabric_resp.fast_path_triggered,
        )
        record_decision_metrics(
            workflow="support_triage",
            gate_action=gate_action,
            primitives_map={k: v.kind for k, v in results.items()},
            latency_seconds=elapsed_ms / 1000.0,
        )
        return SupportTriageResponse(
            ticket_id=request.ticket_id,
            department_queue=dept_queue,
            urgent_sla=urgent_bool,
            urgent_sla_probability=round(urgent_prob, 6),
            explicit_defect=defect_bool,
            explicit_defect_probability=round(defect_prob, 6),
            customer_frustration_score=round(frustration_score, 4),
            minimum_primitive_confidence=round(min_conf, 6),
            automation_gate_action=gate_action,
            primitive_breakdown=results,
            gain_metrics=gain_metrics,
            audit_signature=fabric_resp.audit_signature,
        )

    async def triage_crm_ticket(
        self,
        request: SupportTriageRequest,
    ) -> SupportTriageResponse:
        """Alias for run_support_triage()."""
        return await self.run_support_triage(request)

    async def run_rag_inline_eval(
        self,
        request: RagInlineEvalRequest,
    ) -> RagInlineEvalResponse:
        """Execute Workflow 3: JevEval Inline Production Anti-LLM-as-a-Judge."""
        return await self._jev_eval.evaluate_rag(request)

    async def evaluate_rag_inline(
        self,
        request: RagInlineEvalRequest,
    ) -> RagInlineEvalResponse:
        """Alias for run_rag_inline_eval()."""
        return await self._jev_eval.evaluate_rag(request)

    def calculate_finops_benchmark(
        self,
        request: FinOpsBenchmarkRequest,
    ) -> FinOpsBenchmarkResponse:
        """Compute Workflow 4: Unmetered-Output & Shared-State FinOps ROI Benchmark."""
        n_reqs = request.monthly_requests
        q_count = request.questions_per_request
        s_tokens = request.avg_state_input_tokens
        q_tokens = request.avg_question_tokens
        out_tokens_per_q = request.avg_system2_output_tokens_per_question

        jev_in_per_req = s_tokens + q_count * q_tokens
        jev_total_in = n_reqs * jev_in_per_req
        jev_total_out = 0

        sys2_in_per_req = q_count * (s_tokens + q_tokens)
        sys2_total_in = n_reqs * sys2_in_per_req
        sys2_total_out = n_reqs * q_count * out_tokens_per_q

        if sys2_total_in > 0:
            input_reduction_pct = round(
                ((sys2_total_in - jev_total_in) / sys2_total_in) * 100.0,
                4,
            )
        else:
            input_reduction_pct = 0.0

        jev_monthly_usd = (jev_total_in / 1_000_000.0) * request.jev_input_price_per_1m_usd
        sys2_monthly_usd = (
            (sys2_total_in / 1_000_000.0) * request.system2_input_price_per_1m_usd
            + (sys2_total_out / 1_000_000.0) * request.system2_output_price_per_1m_usd
        )
        monthly_savings = max(0.0, sys2_monthly_usd - jev_monthly_usd)
        annual_savings = monthly_savings * 12.0
        net_reduction_pct = (
            round((monthly_savings / sys2_monthly_usd) * 100.0, 4)
            if sys2_monthly_usd > 0.0
            else 0.0
        )

        jev_lat_ms = 2.2
        sys2_lat_ms = 850.0
        compute_hours_saved = round(
            (n_reqs * ((sys2_lat_ms - jev_lat_ms) / 1000.0)) / 3600.0,
            2,
        )
        amortization_ratio = round((q_count - 1) / q_count, 4) if q_count > 1 else 0.0

        per_req_jev_usd = jev_monthly_usd / n_reqs
        per_req_sys2_usd = sys2_monthly_usd / n_reqs
        gain_metrics = WorkflowGainMetrics(
            jev_latency_ms=jev_lat_ms,
            system2_baseline_latency_ms=sys2_lat_ms,
            speedup_factor=round(sys2_lat_ms / jev_lat_ms, 2),
            jev_cost_usd=round(per_req_jev_usd, 8),
            system2_baseline_cost_usd=round(per_req_sys2_usd, 8),
            cost_reduction_pct=net_reduction_pct,
            output_tokens_billed=0,
            shared_state_amortization_ratio=amortization_ratio,
            jev_input_tokens=jev_in_per_req,
            system2_total_tokens_avoided=(sys2_in_per_req + q_count * out_tokens_per_q)
            - jev_in_per_req,
            calibration_ece_advantage=0.22,
        )

        return FinOpsBenchmarkResponse(
            monthly_requests=n_reqs,
            questions_per_request=q_count,
            jev_total_input_tokens=jev_total_in,
            jev_total_output_tokens_billed=jev_total_out,
            system2_total_input_tokens=sys2_total_in,
            system2_total_output_tokens=sys2_total_out,
            shared_state_input_token_reduction_pct=input_reduction_pct,
            output_token_elimination_pct=100.0,
            jev_monthly_cost_usd=round(jev_monthly_usd, 4),
            system2_monthly_cost_usd=round(sys2_monthly_usd, 4),
            monthly_savings_usd=round(monthly_savings, 4),
            annual_savings_usd=round(annual_savings, 4),
            net_cost_reduction_pct=net_reduction_pct,
            compute_hours_saved_monthly=compute_hours_saved,
            gain_metrics=gain_metrics,
        )

    async def run_finops_benchmark(
        self,
        request: FinOpsBenchmarkRequest,
    ) -> FinOpsBenchmarkResponse:
        """Execute Workflow 4 asynchronously: Unmetered-Output & Shared-State FinOps Benchmark."""
        return self.calculate_finops_benchmark(request)


EnterpriseWorkflowsEngine = EnterpriseWorkflowOrchestrator
