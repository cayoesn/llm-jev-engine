"""Declarative TypeSafe AI Python SDK (TypeSafeClient.system_one, choice, noul, score)."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, Literal

from app.api.schemas import (
    PrimitiveResult,
    PrimitiveSpec,
    SharedStateDecisionRequest,
    SharedStateDecisionResponse,
)
from app.config import Settings, get_settings
from app.core.decision_fabric import DecisionFabric


def choice(
    question: str,
    options: dict[str, str] | list[str] | tuple[str, ...],
    *,
    name: str = "choice_primitive",
    abstention_threshold: float = 0.45,
    calibration_temperature: float | None = None,
) -> PrimitiveSpec:
    """Declare a multi-class System-1 `choice` primitive over 2..255 semantic options.

    Args:
        question: Natural-language question evaluated against the shared state.
        options: Either a dict mapping option_key -> semantic rubric, or a list of option keys.
        name: Identifier key for this primitive (overridden when passed in a questions dict).
        abstention_threshold: Minimum calibrated confidence required before abstaining.
        calibration_temperature: Optional per-primitive RLCD temperature scaling override.

    Returns:
        Immutable PrimitiveSpec configured with kind='choice'.
    """
    return PrimitiveSpec(
        name=name,
        kind="choice",
        question=question,
        options=options,
        abstention_threshold=abstention_threshold,
        calibration_temperature=calibration_temperature,
    )


def noul(
    question: str,
    *,
    name: str = "noul_primitive",
    abstention_threshold: float = 0.45,
    calibration_temperature: float | None = None,
) -> PrimitiveSpec:
    """Declare a binary proposition System-1 `noul` primitive returning TRUE, FALSE, or ABSTAIN.

    Args:
        question: Binary proposition evaluated against the shared state.
        name: Identifier key for this primitive (overridden when passed in a questions dict).
        abstention_threshold: Minimum calibrated confidence required before abstaining.
        calibration_temperature: Optional per-primitive RLCD temperature scaling override.

    Returns:
        Immutable PrimitiveSpec configured with kind='noul'.
    """
    return PrimitiveSpec(
        name=name,
        kind="noul",
        question=question,
        abstention_threshold=abstention_threshold,
        calibration_temperature=calibration_temperature,
    )


def score(
    question: str,
    levels: dict[int, str],
    *,
    name: str = "score_primitive",
    abstention_threshold: float = 0.45,
    calibration_temperature: float | None = None,
) -> PrimitiveSpec:
    """Declare an ordinal rubric System-1 `score` primitive over 2..10 discrete levels.

    Args:
        question: Assessment prompt evaluated against the shared state.
        levels: Dictionary mapping integer ordinal levels to semantic rubric descriptions.
        name: Identifier key for this primitive (overridden when passed in a questions dict).
        abstention_threshold: Minimum calibrated confidence required before abstaining.
        calibration_temperature: Optional per-primitive RLCD temperature scaling override.

    Returns:
        Immutable PrimitiveSpec configured with kind='score'.
    """
    return PrimitiveSpec(
        name=name,
        kind="score",
        question=question,
        levels=levels,
        abstention_threshold=abstention_threshold,
        calibration_temperature=calibration_temperature,
    )


class AwaitableSystemOneResponse(SharedStateDecisionResponse):
    """Immutable System-1 response supporting both synchronous access and `await` expressions."""

    def __await__(self) -> Generator[Any, None, SharedStateDecisionResponse]:
        """Allow `await client.system_one(...)` in async workflows without coroutine overhead in sync code."""

        async def _resolve() -> SharedStateDecisionResponse:
            return self

        return _resolve().__await__()

    def __getitem__(self, key: str) -> PrimitiveResult:
        """Provide dictionary-style lookup (`response['queue']`) into the evaluated primitive results."""
        return self.results[key]

    def __contains__(self, key: object) -> bool:
        """Check whether a primitive key is present in the evaluated results."""
        return key in self.results


class TypeSafeClient:
    """Official-style declarative Python SDK client for the Jev System-1 Decision Engine.

    Evaluates a shared `state` against a dictionary or list of typed primitives
    (`choice`, `noul`, `score`) in a single non-autoregressive pass.
    """

    def __init__(
        self,
        decision_fabric: DecisionFabric | None = None,
        settings: Settings | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the TypeSafeClient with an embedded DecisionFabric."""
        self._settings = settings or get_settings()
        self._fabric = decision_fabric or DecisionFabric(settings=self._settings)
        self._api_key = api_key or "jev-open-source-local-key"

    @property
    def decision_fabric(self) -> DecisionFabric:
        """Return the underlying DecisionFabric instance."""
        return self._fabric

    @staticmethod
    def _normalize_question_specs(
        questions: dict[str, PrimitiveSpec] | list[PrimitiveSpec],
    ) -> list[PrimitiveSpec]:
        """Bind dictionary keys as primitive names or validate a list of PrimitiveSpecs."""
        if isinstance(questions, dict):
            bound: list[PrimitiveSpec] = []
            for key, spec in questions.items():
                if not isinstance(spec, PrimitiveSpec):
                    raise TypeError(
                        f"Question '{key}' must be a PrimitiveSpec built via choice(), noul(), or score()."
                    )
                if spec.name != key:
                    bound.append(spec.model_copy(update={"name": key}))
                else:
                    bound.append(spec)
            return bound
        if isinstance(questions, list):
            for item in questions:
                if not isinstance(item, PrimitiveSpec):
                    raise TypeError(
                        "All items in questions list must be PrimitiveSpec instances."
                    )
            return list(questions)
        raise TypeError("questions must be a dict[str, PrimitiveSpec] or list[PrimitiveSpec].")

    def _build_request(
        self,
        state: dict[str, Any] | str,
        questions: dict[str, PrimitiveSpec] | list[PrimitiveSpec],
        request_id: str | None = None,
        bypass_cache: bool = False,
        fast_path_enabled: bool = True,
        backend: Literal["spectral_simd", "open_weights"] = "spectral_simd",
    ) -> SharedStateDecisionRequest:
        """Construct a validated SharedStateDecisionRequest from SDK arguments."""
        state_dict: dict[str, Any] = (
            {"state": state} if isinstance(state, str) else dict(state)
        )
        primitives = self._normalize_question_specs(questions)
        return SharedStateDecisionRequest(
            request_id=request_id,
            state=state_dict,
            primitives=primitives,
            bypass_cache=bypass_cache,
            fast_path_enabled=fast_path_enabled,
            backend=backend,
        )

    def system_one(
        self,
        state: dict[str, Any] | str,
        questions: dict[str, PrimitiveSpec] | list[PrimitiveSpec],
        *,
        request_id: str | None = None,
        bypass_cache: bool = False,
        fast_path_enabled: bool = True,
        backend: Literal["spectral_simd", "open_weights"] = "spectral_simd",
    ) -> AwaitableSystemOneResponse:
        """Evaluate multiple System-1 primitives over a shared state in a single pass.

        Supports both synchronous (`res = client.system_one(...)`) and asynchronous
        (`res = await client.system_one(...)`) invocation idioms.

        Args:
            state: Application state dictionary or string context.
            questions: Keyed dictionary or list of `choice()`, `noul()`, and `score()` specs.
            request_id: Optional correlation identifier.
            bypass_cache: Force fresh evaluation bypassing the BoundedDecisionCache.
            fast_path_enabled: Enable Stage-1 deterministic adversarial regex guard.
            backend: Execution backend ('spectral_simd' or 'open_weights').

        Returns:
            AwaitableSystemOneResponse (subclass of SharedStateDecisionResponse).
        """
        req = self._build_request(
            state=state,
            questions=questions,
            request_id=request_id,
            bypass_cache=bypass_cache,
            fast_path_enabled=fast_path_enabled,
            backend=backend,
        )
        resp = self._fabric.evaluate_sync(req)
        return AwaitableSystemOneResponse.model_validate(resp.model_dump())

    async def asystem_one(
        self,
        state: dict[str, Any] | str,
        questions: dict[str, PrimitiveSpec] | list[PrimitiveSpec],
        *,
        request_id: str | None = None,
        bypass_cache: bool = False,
        fast_path_enabled: bool = True,
        backend: Literal["spectral_simd", "open_weights"] = "spectral_simd",
    ) -> AwaitableSystemOneResponse:
        """Asynchronously evaluate multiple System-1 primitives via `asyncio.gather`.

        Args:
            state: Application state dictionary or string context.
            questions: Keyed dictionary or list of `choice()`, `noul()`, and `score()` specs.
            request_id: Optional correlation identifier.
            bypass_cache: Force fresh evaluation bypassing the BoundedDecisionCache.
            fast_path_enabled: Enable Stage-1 deterministic adversarial regex guard.
            backend: Execution backend ('spectral_simd' or 'open_weights').

        Returns:
            AwaitableSystemOneResponse containing all calibrated primitive decisions.
        """
        req = self._build_request(
            state=state,
            questions=questions,
            request_id=request_id,
            bypass_cache=bypass_cache,
            fast_path_enabled=fast_path_enabled,
            backend=backend,
        )
        resp = await self._fabric.evaluate(req)
        return AwaitableSystemOneResponse.model_validate(resp.model_dump())
