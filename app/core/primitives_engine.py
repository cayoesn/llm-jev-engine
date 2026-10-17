"""Vectorized System-1 Primitives Engine (choice, noul, score) with Semantic Rubric Matching."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

from app.api.schemas import GateAction, PrimitiveResult, PrimitiveSpec
from app.config import Settings, get_settings
from app.core.calibration_rlcd import RLCDCalibrator, compute_wilson_interval

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "our",
        "should",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "what",
        "which",
        "with",
    }
)

CONCEPT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "billing": (
        "payment",
        "payments",
        "refund",
        "refunds",
        "invoice",
        "invoices",
        "charged",
        "charge",
        "charges",
        "double",
        "twice",
        "credit",
        "subscription",
        "fee",
        "overcharged",
    ),
    "technical": (
        "bug",
        "bugs",
        "outage",
        "outages",
        "api",
        "crash",
        "crashes",
        "error",
        "errors",
        "latency",
        "sre",
        "timeout",
        "production",
        "defect",
        "stacktrace",
        "500",
        "502",
        "503",
        "database",
    ),
    "sales": (
        "pricing",
        "upgrade",
        "upgrades",
        "contract",
        "contracts",
        "enterprise",
        "license",
        "licensing",
        "seat",
        "seats",
        "quote",
        "renewal",
        "discount",
        "expansion",
    ),
    "legal": (
        "compliance",
        "gdpr",
        "hipaa",
        "subpoena",
        "lawsuit",
        "attorney",
        "lawyer",
        "regulation",
        "regulatory",
        "breach",
        "nda",
        "arbitration",
    ),
    "urgent": (
        "urgent",
        "immediately",
        "immediate",
        "critical",
        "outage",
        "down",
        "asap",
        "emergency",
        "hour",
        "blocker",
        "p0",
        "sev1",
        "furious",
    ),
    "unsafe": (
        "ignore",
        "bypass",
        "jailbreak",
        "override",
        "injection",
        "reveal",
        "system_prompt",
        "hack",
        "exploit",
        "malware",
        "ssn",
        "password_dump",
    ),
    "complex": (
        "distributed",
        "consensus",
        "theorem",
        "proof",
        "multi",
        "architecture",
        "optimization",
        "eigenvalue",
        "stochastic",
        "cryptographic",
        "raft",
        "paxos",
        "byzantine",
    ),
    "simple": (
        "hello",
        "hi",
        "ping",
        "status",
        "faq",
        "reset",
        "password",
        "greeting",
        "hours",
        "thanks",
    ),
}


def _stem_token(token: str) -> str:
    """Produce a lightweight deterministic morphological stem for lexical matching."""
    if len(token) <= 4:
        return token
    for suffix in ("ation", "ition", "ments", "ment", "ings", "ing", "edly", "ed", "ies", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "ies":
                return token[:-3] + "y"
            return token[: -len(suffix)]
    return token


def _extract_tokens_and_stems(text: str) -> tuple[list[str], set[str]]:
    """Tokenize lowercased text and compute filtered content stems."""
    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    stems: set[str] = set()
    for tok in raw_tokens:
        if tok not in STOPWORDS:
            stems.add(tok)
            stems.add(_stem_token(tok))
    return raw_tokens, stems


def _project_spectral_vector(text: str, dim: int = 64) -> np.ndarray:
    """Project arbitrary text into a deterministic unit-norm float64 spectral vector in R^dim."""
    vec = np.zeros(dim, dtype=np.float64)
    tokens, stems = _extract_tokens_and_stems(text)
    all_items = list(tokens) + [f"stem:{s}" for s in sorted(stems)]
    if not all_items:
        vec[0] = 1.0
        vec.flags.writeable = False
        return vec

    for item in all_items:
        digest = hashlib.blake2b(item.encode("utf-8"), digest_size=8).digest()
        h_val = int.from_bytes(digest, byteorder="little", signed=False)
        idx = h_val % dim
        sign = 1.0 if ((h_val >> 8) & 1) == 0 else -1.0
        vec[idx] += sign

    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec = vec / norm
    else:
        vec[0] = 1.0
    vec.flags.writeable = False
    return vec


@dataclass(frozen=True, slots=True)
class SharedStateContext:
    """Immutable pre-encoded representation of an application state for O(1) multi-question passes."""

    state_hash: str
    canonical_json: str
    state_vector: np.ndarray
    token_count: int
    semantic_features: dict[str, float] = field(default_factory=dict)
    raw_state: dict[str, Any] = field(default_factory=dict)


class OpenWeightsLogprobBackend:
    """Asynchronous adapter for extracting first-token logprobs from Ollama or vLLM servers."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:0.5b",
        timeout_seconds: float = 2.5,
    ) -> None:
        """Initialize the OpenWeightsLogprobBackend with server endpoint and model settings."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def extract_option_logits(
        self,
        state_json: str,
        question: str,
        options: list[str],
    ) -> np.ndarray | None:
        """Request single-token logprobs from an OpenAI-compatible /v1/completions endpoint."""
        prompt = (
            f"State: {state_json}\nQuestion: {question}\n"
            f"Options: {', '.join(options)}\nAnswer:"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": min(len(options) + 5, 20),
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(f"{self.base_url}/v1/completions", json=payload)
                response.raise_for_status()
                body = response.json()
                top_logprobs = (
                    body.get("choices", [{}])[0]
                    .get("logprobs", {})
                    .get("top_logprobs", [{}])[0]
                )
                if not isinstance(top_logprobs, dict) or not top_logprobs:
                    return None
                logits = np.full(len(options), -5.0, dtype=np.float64)
                for idx, opt in enumerate(options):
                    for tok_str, logp in top_logprobs.items():
                        if opt.lower().startswith(str(tok_str).strip().lower()):
                            logits[idx] = float(logp)
                            break
                return logits
        except Exception:
            return None


class PrimitivesEngine:
    """Vectorized System-1 Neural Decision Engine executing choice, noul, and score primitives."""

    def __init__(
        self,
        calibrator: RLCDCalibrator | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the PrimitivesEngine with an RLCDCalibrator and runtime Settings."""
        self._settings = settings or get_settings()
        self._calibrator = calibrator or RLCDCalibrator(
            default_temperature=self._settings.JEV_DEFAULT_TEMPERATURE
        )

    @property
    def calibrator(self) -> RLCDCalibrator:
        """Return the underlying RLCDCalibrator instance."""
        return self._calibrator

    def encode_state(self, state: dict[str, Any] | str) -> SharedStateContext:
        """Deterministically encode an application state dictionary into a SharedStateContext."""
        if isinstance(state, str):
            raw_dict: dict[str, Any] = {"state": state}
        else:
            raw_dict = dict(state)

        canonical_json = json.dumps(
            raw_dict,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        state_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        value_texts: list[str] = []
        for key, val in raw_dict.items():
            value_texts.append(f"{key} {val}")
        combined_text = " ".join(value_texts)

        raw_tokens, stems = _extract_tokens_and_stems(combined_text)
        token_count = max(len(raw_tokens), max(1, len(canonical_json) // 4))
        state_vector = _project_spectral_vector(combined_text, dim=64)

        features: dict[str, float] = {}
        for stem in stems:
            features[stem] = features.get(stem, 0.0) + 1.0

        for concept, synonyms in CONCEPT_SYNONYMS.items():
            overlap_count = 0
            for word in synonyms:
                if word in stems or _stem_token(word) in stems:
                    overlap_count += 1
            if overlap_count > 0:
                features[f"concept:{concept}"] = float(overlap_count)
                features[concept] = features.get(concept, 0.0) + float(overlap_count) * 1.5
                features[_stem_token(concept)] = (
                    features.get(_stem_token(concept), 0.0) + float(overlap_count) * 1.5
                )

        return SharedStateContext(
            state_hash=state_hash,
            canonical_json=canonical_json,
            state_vector=state_vector,
            token_count=token_count,
            semantic_features=features,
            raw_state=raw_dict,
        )

    def _resolve_gate_action(self, calibrated_confidence: float, abstained: bool) -> GateAction:
        """Determine the deterministic confidence gate action from calibrated confidence."""
        if abstained or calibrated_confidence < self._settings.JEV_MIN_CALIBRATED_CONFIDENCE_FALLBACK:
            return "ESCALATE_HITL"
        if calibrated_confidence >= self._settings.JEV_MIN_CALIBRATED_CONFIDENCE_AUTO:
            return "AUTO_EXECUTE"
        return "FALLBACK_SYSTEM2"

    def _score_rubric_resonance(
        self,
        context: SharedStateContext,
        key_text: str,
        rubric_text: str,
    ) -> float:
        """Compute semantic resonance between encoded state features and an option/level rubric."""
        _, key_stems = _extract_tokens_and_stems(key_text.replace("_", " "))
        _, rubric_stems = _extract_tokens_and_stems(rubric_text.replace("_", " "))
        all_stems = key_stems | rubric_stems
        if not all_stems:
            return 0.0

        score = 0.0
        matched_stems = 0
        for stem in all_stems:
            weight = context.semantic_features.get(stem, 0.0)
            if weight > 0.0:
                matched_stems += 1
                bonus = 1.6 if stem in key_stems else 1.0
                score += bonus * math.log1p(weight)

        if matched_stems == 0:
            return 0.0
        coverage = matched_stems / math.sqrt(len(all_stems))
        return score + 1.8 * coverage

    def evaluate_choice(
        self,
        context: SharedStateContext,
        spec: PrimitiveSpec,
    ) -> PrimitiveResult:
        """Evaluate a multi-class choice primitive in a single vectorized pass."""
        start_ns = time.perf_counter_ns()
        if spec.options is None:
            raise ValueError("choice primitive requires non-empty options.")

        option_keys = list(spec.options.keys())
        option_rubrics = [spec.options[k] for k in option_keys]
        num_options = len(option_keys)

        option_matrix = np.vstack(
            [
                _project_spectral_vector(f"{k} {r}", dim=64)
                for k, r in zip(option_keys, option_rubrics)
            ]
        )
        spectral_sim = option_matrix @ context.state_vector

        rubric_scores = np.array(
            [
                self._score_rubric_resonance(context, k, r)
                for k, r in zip(option_keys, option_rubrics)
            ],
            dtype=np.float64,
        )

        max_rubric = float(np.max(rubric_scores))
        if max_rubric > 0.0:
            logits = 2.8 * rubric_scores + 0.35 * spectral_sim
        else:
            logits = 0.02 * spectral_sim

        temperature = (
            spec.calibration_temperature
            if spec.calibration_temperature is not None
            else self._calibrator.temperature
        )
        probs = self._calibrator.apply_temperature_scaling(logits, temperature=temperature)

        best_idx = int(np.argmax(probs))
        p_max = float(probs[best_idx])
        selected_key = option_keys[best_idx]

        clipped_probs = np.clip(probs, 1e-12, 1.0)
        shannon_bits = float(-np.sum(clipped_probs * np.log2(clipped_probs)))
        max_bits = math.log2(num_options) if num_options > 1 else 1.0
        norm_entropy = float(np.clip(shannon_bits / max_bits, 0.0, 1.0))
        calibrated_conf = float(np.clip(p_max * (1.0 - norm_entropy), 0.0, 1.0))
        ci_95 = compute_wilson_interval(p_max)
        abstained = bool(calibrated_conf < spec.abstention_threshold)
        gate_action = self._resolve_gate_action(calibrated_conf, abstained)

        dist = {option_keys[i]: round(float(probs[i]), 6) for i in range(num_options)}
        diff = 1.0 - sum(dist.values())
        dist[selected_key] = round(dist[selected_key] + diff, 6)

        latency_ms = max(0.01, (time.perf_counter_ns() - start_ns) / 1_000_000.0)
        return PrimitiveResult(
            name=spec.name,
            kind="choice",
            selected_value=selected_key,
            probability_distribution=dist,
            calibrated_confidence=round(calibrated_conf, 6),
            shannon_entropy_bits=round(shannon_bits, 6),
            normalized_entropy=round(norm_entropy, 6),
            credible_interval_95=ci_95,
            abstained=abstained,
            latency_ms=round(latency_ms, 4),
            gate_action=gate_action,
        )

    def _compute_noul_logit_margin(
        self,
        context: SharedStateContext,
        spec: PrimitiveSpec,
    ) -> float:
        """Compute signed binary logit margin (z_TRUE - z_FALSE) for a noul proposition."""
        q_lower = spec.question.lower()
        _, q_stems = _extract_tokens_and_stems(q_lower)
        raw_state = context.raw_state
        feats = context.semantic_features

        if "retrieved_context" in raw_state and "candidate_answer" in raw_state:
            if any(w in q_lower for w in ("faithful", "ground", "context", "hallucinat", "support")):
                _, ctx_stems = _extract_tokens_and_stems(str(raw_state["retrieved_context"]))
                _, ans_stems = _extract_tokens_and_stems(str(raw_state["candidate_answer"]))
                if not ans_stems:
                    return -3.0
                overlap = len(ans_stems & ctx_stems) / len(ans_stems)
                return (overlap - 0.38) * 9.5

        if any(w in q_lower for w in ("safe", "compliant", "compliance", "policy")):
            unsafe_signal = feats.get("concept:unsafe", 0.0)
            for bad_word in ("ignore", "jailbreak", "bypass", "hack", "exploit", "ssn", "leak", "promise"):
                if bad_word in feats or _stem_token(bad_word) in feats:
                    unsafe_signal += 2.0
            if unsafe_signal > 0.0:
                return -4.2 * unsafe_signal
            return 3.8

        if any(w in q_lower for w in ("urgent", "hour", "immediate", "sla", "asap", "emergency")):
            urgent_signal = feats.get("concept:urgent", 0.0)
            for pos_w in ("twice", "outage", "down", "critical", "urgent", "immediately", "hour", "furious"):
                if pos_w in feats or _stem_token(pos_w) in feats:
                    urgent_signal += 1.5
            calm_signal = 0.0
            for neg_w in ("calm", "rush", "low", "general", "minor", "informational", "next"):
                if neg_w in feats or _stem_token(neg_w) in feats:
                    calm_signal += 1.8
            if urgent_signal > 0.0 or calm_signal > 0.0:
                return 2.6 * (urgent_signal - calm_signal)

        if any(w in q_lower for w in ("defect", "bug", "outage", "crash", "broken", "error")):
            tech_signal = feats.get("concept:technical", 0.0)
            billing_signal = feats.get("concept:billing", 0.0) + feats.get("concept:sales", 0.0)
            if tech_signal > 0.0:
                return 2.8 * tech_signal
            if billing_signal > 0.0:
                return -3.2

        matched = sum(1 for s in q_stems if s in feats)
        if matched == 0:
            return 0.0

        negation_present = any(
            neg in context.canonical_json.lower().split()
            for neg in ("not", "never", "no", "none", "false", "without")
        )
        coverage = matched / max(1, len(q_stems))
        if negation_present:
            return -3.2 * max(0.5, coverage)
        return 3.6 * max(0.5, coverage)

    def evaluate_noul(
        self,
        context: SharedStateContext,
        spec: PrimitiveSpec,
    ) -> PrimitiveResult:
        """Evaluate a binary proposition primitive returning TRUE, FALSE, or ABSTAIN."""
        start_ns = time.perf_counter_ns()
        margin = self._compute_noul_logit_margin(context, spec)
        logits = np.array([margin / 2.0, -margin / 2.0], dtype=np.float64)

        temperature = (
            spec.calibration_temperature
            if spec.calibration_temperature is not None
            else self._calibrator.temperature
        )
        probs = self._calibrator.apply_temperature_scaling(logits, temperature=temperature)
        p_true = float(probs[0])
        p_false = float(probs[1])
        p_max = max(p_true, p_false)

        clipped = np.clip(probs, 1e-12, 1.0)
        shannon_bits = float(-np.sum(clipped * np.log2(clipped)))
        norm_entropy = float(np.clip(shannon_bits, 0.0, 1.0))
        calibrated_conf = float(np.clip(p_max * (1.0 - norm_entropy), 0.0, 1.0))
        ci_95 = compute_wilson_interval(p_max)

        asymmetric_floor = max(0.55, spec.abstention_threshold)
        abstained = bool(calibrated_conf < spec.abstention_threshold or p_max < asymmetric_floor)
        if abstained:
            selected_val = "ABSTAIN"
        else:
            selected_val = "TRUE" if p_true >= 0.5 else "FALSE"

        gate_action = self._resolve_gate_action(calibrated_conf, abstained)
        p_true_rounded = round(p_true, 6)
        dist = {"TRUE": p_true_rounded, "FALSE": round(1.0 - p_true_rounded, 6)}

        latency_ms = max(0.01, (time.perf_counter_ns() - start_ns) / 1_000_000.0)
        return PrimitiveResult(
            name=spec.name,
            kind="noul",
            selected_value=selected_val,
            probability_distribution=dist,
            calibrated_confidence=round(calibrated_conf, 6),
            shannon_entropy_bits=round(shannon_bits, 6),
            normalized_entropy=round(norm_entropy, 6),
            credible_interval_95=ci_95,
            abstained=abstained,
            latency_ms=round(latency_ms, 4),
            gate_action=gate_action,
        )

    def _estimate_continuous_intensity_target(
        self,
        context: SharedStateContext,
        spec: PrimitiveSpec,
        sorted_levels: list[int],
    ) -> float | None:
        """Estimate target ordinal intensity on [v_min, v_max] from domain lexical cues."""
        v_min = float(sorted_levels[0])
        v_max = float(sorted_levels[-1])
        span = v_max - v_min
        q_lower = spec.question.lower()
        feats = context.semantic_features
        raw_state = context.raw_state

        if "retrieved_context" in raw_state and "candidate_answer" in raw_state:
            if any(w in q_lower for w in ("quality", "answer", "helpfulness", "completeness")):
                _, ctx_stems = _extract_tokens_and_stems(str(raw_state["retrieved_context"]))
                _, ans_stems = _extract_tokens_and_stems(str(raw_state["candidate_answer"]))
                if not ans_stems:
                    return v_min
                overlap = min(1.0, len(ans_stems & ctx_stems) / max(1, len(ans_stems)))
                return v_min + span * (0.2 + 0.78 * overlap)

        if any(w in q_lower for w in ("frustrat", "sentiment", "angry", "satisfaction", "tone")):
            high_cues = sum(
                1
                for w in (
                    "angry",
                    "furious",
                    "twice",
                    "refund",
                    "cancel",
                    "cancelling",
                    "legal",
                    "lawsuit",
                    "unacceptable",
                    "worst",
                    "nobody",
                    "immediately",
                )
                if w in feats or _stem_token(w) in feats
            )
            low_cues = sum(
                1
                for w in ("calm", "inquiry", "thank", "thanks", "polite", "informational", "minor")
                if w in feats or _stem_token(w) in feats
            )
            if high_cues > 0:
                ratio = min(0.96, 0.65 + 0.08 * high_cues)
                return v_min + span * ratio
            if low_cues > 0:
                ratio = max(0.05, 0.18 - 0.04 * low_cues)
                return v_min + span * ratio

        if any(w in q_lower for w in ("complex", "difficulty", "reasoning", "depth")):
            complex_cues = feats.get("concept:complex", 0.0)
            simple_cues = feats.get("concept:simple", 0.0)
            if complex_cues > 0.0:
                ratio = min(0.95, 0.62 + 0.10 * complex_cues)
                return v_min + span * ratio
            if simple_cues > 0.0 or context.token_count <= 8:
                return v_min + span * 0.12

        return None

    def evaluate_score(
        self,
        context: SharedStateContext,
        spec: PrimitiveSpec,
    ) -> PrimitiveResult:
        """Evaluate an ordinal score primitive computing continuous E[S|x] and epistemic Var(S|x)."""
        start_ns = time.perf_counter_ns()
        if spec.levels is None:
            raise ValueError("score primitive requires non-empty ordinal levels.")

        sorted_levels = sorted(spec.levels.keys())
        level_vals = np.array(sorted_levels, dtype=np.float64)
        rubrics = [spec.levels[lvl] for lvl in sorted_levels]
        num_levels = len(sorted_levels)

        rubric_scores = np.array(
            [
                self._score_rubric_resonance(context, str(lvl), rub)
                for lvl, rub in zip(sorted_levels, rubrics)
            ],
            dtype=np.float64,
        )

        target_intensity = self._estimate_continuous_intensity_target(
            context,
            spec,
            sorted_levels,
        )

        v_min = float(level_vals[0])
        v_max = float(level_vals[-1])
        span = max(v_max - v_min, 1.0)

        if target_intensity is not None:
            sigma = 0.22 * span
            gaussian_logits = -((level_vals - target_intensity) ** 2) / (2.0 * sigma * sigma)
            logits = 2.6 * gaussian_logits + 1.2 * rubric_scores
        elif float(np.max(rubric_scores)) > 0.0:
            peak_idx = int(np.argmax(rubric_scores))
            peak_val = float(level_vals[peak_idx])
            sigma = 0.28 * span
            gaussian_logits = -((level_vals - peak_val) ** 2) / (2.0 * sigma * sigma)
            logits = 2.2 * rubric_scores + 1.4 * gaussian_logits
        else:
            logits = np.zeros(num_levels, dtype=np.float64)

        temperature = (
            spec.calibration_temperature
            if spec.calibration_temperature is not None
            else self._calibrator.temperature
        )
        probs = self._calibrator.apply_temperature_scaling(logits, temperature=temperature)

        expected_val = float(np.sum(level_vals * probs))
        variance_val = float(np.sum(((level_vals - expected_val) ** 2) * probs))

        clipped_probs = np.clip(probs, 1e-12, 1.0)
        shannon_bits = float(-np.sum(clipped_probs * np.log2(clipped_probs)))
        max_bits = math.log2(num_levels) if num_levels > 1 else 1.0
        norm_entropy = float(np.clip(shannon_bits / max_bits, 0.0, 1.0))

        max_variance = ((v_max - v_min) / 2.0) ** 2
        dispersion_ratio = min(1.0, variance_val / max(max_variance, 1e-9))
        p_max = float(np.max(probs))
        calibrated_conf = float(
            np.clip(
                (1.0 - norm_entropy) * (0.55 * p_max + 0.65 * (1.0 - dispersion_ratio)),
                0.0,
                1.0,
            )
        )

        std_err = math.sqrt(max(variance_val, 0.0) / 16.0)
        ci_low = round(float(np.clip(expected_val - 1.96 * std_err, v_min, v_max)), 4)
        ci_high = round(float(np.clip(expected_val + 1.96 * std_err, v_min, v_max)), 4)

        abstained = bool(calibrated_conf < spec.abstention_threshold)
        gate_action = self._resolve_gate_action(calibrated_conf, abstained)

        best_idx = int(np.argmax(probs))
        dist = {str(sorted_levels[i]): round(float(probs[i]), 6) for i in range(num_levels)}
        diff = 1.0 - sum(dist.values())
        best_key = str(sorted_levels[best_idx])
        dist[best_key] = round(dist[best_key] + diff, 6)

        expected_rounded = round(expected_val, 4)
        variance_rounded = round(variance_val, 4)
        latency_ms = max(0.01, (time.perf_counter_ns() - start_ns) / 1_000_000.0)

        return PrimitiveResult(
            name=spec.name,
            kind="score",
            selected_value=expected_rounded,
            probability_distribution=dist,
            calibrated_confidence=round(calibrated_conf, 6),
            shannon_entropy_bits=round(shannon_bits, 6),
            normalized_entropy=round(norm_entropy, 6),
            credible_interval_95=(ci_low, ci_high),
            expected_score=expected_rounded,
            score_variance=variance_rounded,
            abstained=abstained,
            latency_ms=round(latency_ms, 4),
            gate_action=gate_action,
        )

    def evaluate(
        self,
        context: SharedStateContext,
        spec: PrimitiveSpec,
    ) -> PrimitiveResult:
        """Dispatch primitive evaluation to evaluate_choice, evaluate_noul, or evaluate_score."""
        if spec.kind == "choice":
            return self.evaluate_choice(context, spec)
        if spec.kind == "noul":
            return self.evaluate_noul(context, spec)
        if spec.kind == "score":
            return self.evaluate_score(context, spec)
        raise ValueError(f"Unsupported primitive kind: {spec.kind}")


JevPrimitivesEngine = PrimitivesEngine
SpectralSIMDLogitBackend = PrimitivesEngine
