"""Shared-State Parallel Multi-Question Sampler and O(1) Bounded LRU Decision Cache."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any

from app.api.schemas import PrimitiveResult, PrimitiveSpec
from app.config import Settings, get_settings
from app.core.primitives_engine import (
    OpenWeightsLogprobBackend,
    PrimitivesEngine,
    SharedStateContext,
)


def compute_primitive_hash(spec: PrimitiveSpec) -> str:
    """Compute a canonical SHA-256 digest uniquely identifying a PrimitiveSpec configuration.

    Complexity:
        Time: O(K) where K is the number of options or levels in the specification.
        Space: O(K) for serialized canonical JSON bytes.

    Args:
        spec: Immutable PrimitiveSpec instance to hash.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    payload = {
        "name": spec.name,
        "kind": spec.kind,
        "question": spec.question,
        "options": spec.options,
        "levels": spec.levels,
        "abstention_threshold": spec.abstention_threshold,
        "calibration_temperature": spec.calibration_temperature,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class BoundedDecisionCache:
    """Thread-safe O(1) LRU cache with TTL expiration for calibrated System-1 decisions.

    Uses a single re-entrant lock (`threading.RLock`) and `collections.OrderedDict`
    to guarantee O(1) amortized lookup, insertion, and LRU eviction.
    """

    def __init__(
        self,
        max_size: int = 4096,
        ttl_seconds: float = 300.0,
        max_entries: int | None = None,
    ) -> None:
        """Initialize the bounded LRU cache.

        Args:
            max_size: Maximum number of cached (state_hash, primitive_hash) entries.
            ttl_seconds: Time-to-live in seconds before a cached entry expires.
            max_entries: Optional alias overriding max_size when provided.
        """
        resolved_max = max_entries if max_entries is not None else max_size
        if resolved_max < 1:
            raise ValueError("max_size must be at least 1.")
        if ttl_seconds <= 0.0:
            raise ValueError("ttl_seconds must be positive.")
        self._max_size = resolved_max
        self._ttl_seconds = float(ttl_seconds)
        self._lock = threading.RLock()
        self._store: OrderedDict[tuple[str, str], tuple[float, PrimitiveResult]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _normalize_key(
        state_hash_or_key: str | tuple[str, str],
        primitive_hash: str | None = None,
    ) -> tuple[str, str]:
        """Normalize composite key arguments into a 2-tuple of (state_hash, primitive_hash)."""
        if isinstance(state_hash_or_key, tuple):
            return (str(state_hash_or_key[0]), str(state_hash_or_key[1]))
        if primitive_hash is None:
            return (str(state_hash_or_key), "")
        return (str(state_hash_or_key), str(primitive_hash))

    def get(
        self,
        state_hash_or_key: str | tuple[str, str],
        primitive_hash: str | None = None,
    ) -> PrimitiveResult | None:
        """Retrieve a cached PrimitiveResult in O(1) time, refreshing its LRU position.

        Args:
            state_hash_or_key: SHA-256 state hash string or (state_hash, primitive_hash) tuple.
            primitive_hash: SHA-256 primitive hash string when first argument is state_hash.

        Returns:
            Cached PrimitiveResult if present and unexpired, otherwise None.
        """
        key = self._normalize_key(state_hash_or_key, primitive_hash)
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            expires_at, result = entry
            if now >= expires_at:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key, last=True)
            self._hits += 1
            return result

    def put(
        self,
        state_hash_or_key: str | tuple[str, str],
        primitive_hash_or_result: str | PrimitiveResult,
        result: PrimitiveResult | None = None,
    ) -> None:
        """Insert or update a PrimitiveResult in O(1) time with automatic LRU eviction.

        Args:
            state_hash_or_key: SHA-256 state hash string or (state_hash, primitive_hash) tuple.
            primitive_hash_or_result: Primitive hash string, or PrimitiveResult when key is a tuple.
            result: PrimitiveResult to cache when key components are passed separately.
        """
        if isinstance(primitive_hash_or_result, PrimitiveResult):
            key = self._normalize_key(state_hash_or_key, None)
            target_result = primitive_hash_or_result
        else:
            if result is None:
                raise ValueError("result must be provided when primitive_hash is a string.")
            key = self._normalize_key(state_hash_or_key, primitive_hash_or_result)
            target_result = result

        expires_at = time.monotonic() + self._ttl_seconds
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key, last=True)
            self._store[key] = (expires_at, target_result)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self) -> int:
        """Purge all entries from the cache and return the number of evicted items."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def __len__(self) -> int:
        """Return the current number of entries stored in the cache."""
        with self._lock:
            return len(self._store)

    @property
    def size(self) -> int:
        """Return the current number of cached entries."""
        with self._lock:
            return len(self._store)

    @property
    def max_size(self) -> int:
        """Return the maximum capacity of the cache."""
        return self._max_size

    @property
    def hits(self) -> int:
        """Return cumulative cache hits."""
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        """Return cumulative cache misses."""
        with self._lock:
            return self._misses

    @property
    def hit_rate(self) -> float:
        """Return the empirical cache hit ratio in [0.0, 1.0]."""
        with self._lock:
            total = self._hits + self._misses
            return round(self._hits / total, 6) if total > 0 else 0.0

    def stats(self) -> dict[str, int | float]:
        """Return a thread-safe snapshot of cache capacity and hit/miss telemetry."""
        with self._lock:
            total = self._hits + self._misses
            ratio = round(self._hits / total, 6) if total > 0 else 0.0
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": ratio,
            }


class SharedStateParallelSampler:
    """Evaluates multiple System-1 primitives concurrently over a single encoded state pass.

    Encodes the input `state` dictionary once in O(1) amortized time and dispatches
    all primitive evaluations concurrently via `asyncio.gather`, eliminating redundant
    prompt serialization and input token billing across N questions.
    """

    def __init__(
        self,
        primitives_engine: PrimitivesEngine | None = None,
        cache: BoundedDecisionCache | None = None,
        settings: Settings | None = None,
        open_weights_backend: OpenWeightsLogprobBackend | None = None,
    ) -> None:
        """Initialize the SharedStateParallelSampler.

        Args:
            primitives_engine: Vectorized PrimitivesEngine instance.
            cache: BoundedDecisionCache for O(1) memoization.
            settings: Application Settings instance.
            open_weights_backend: Optional Ollama/vLLM first-token logprob adapter.
        """
        self._settings = settings or get_settings()
        self._engine = primitives_engine or PrimitivesEngine(settings=self._settings)
        self._cache = cache or BoundedDecisionCache(
            max_size=self._settings.JEV_CACHE_MAX_SIZE,
            ttl_seconds=self._settings.JEV_CACHE_TTL_SECONDS,
        )
        self._open_weights = open_weights_backend or OpenWeightsLogprobBackend(
            base_url=self._settings.OLLAMA_BASE_URL,
            model=self._settings.OLLAMA_MODEL,
            timeout_seconds=self._settings.JEV_OPEN_WEIGHTS_TIMEOUT_SECONDS,
        )

    @property
    def engine(self) -> PrimitivesEngine:
        """Return the underlying PrimitivesEngine."""
        return self._engine

    @property
    def cache(self) -> BoundedDecisionCache:
        """Return the underlying BoundedDecisionCache."""
        return self._cache

    @staticmethod
    def _normalize_primitives(
        primitives: list[PrimitiveSpec] | dict[str, PrimitiveSpec],
    ) -> list[PrimitiveSpec]:
        """Convert a list or keyed dictionary of PrimitiveSpec objects into a canonical list."""
        if isinstance(primitives, dict):
            normalized: list[PrimitiveSpec] = []
            for key, spec in primitives.items():
                if spec.name != key:
                    normalized.append(spec.model_copy(update={"name": key}))
                else:
                    normalized.append(spec)
            return normalized
        return list(primitives)

    async def _evaluate_single(
        self,
        context: SharedStateContext,
        spec: PrimitiveSpec,
        bypass_cache: bool,
        backend: str,
    ) -> tuple[str, PrimitiveResult, bool]:
        """Evaluate a single primitive against the shared state context with O(1) cache lookup."""
        prim_hash = compute_primitive_hash(spec)
        if not bypass_cache:
            cached = self._cache.get(context.state_hash, prim_hash)
            if cached is not None:
                return spec.name, cached, True

        if backend == "open_weights" and spec.kind == "choice" and spec.options is not None:
            option_keys = list(spec.options.keys())
            ext_logits = await self._open_weights.extract_option_logits(
                state_json=context.canonical_json,
                question=spec.question,
                options=option_keys,
            )
            if ext_logits is not None and len(ext_logits) == len(option_keys):
                result = self._engine.evaluate_choice(context, spec)
                if not bypass_cache:
                    self._cache.put(context.state_hash, prim_hash, result)
                return spec.name, result, False

        await asyncio.sleep(0)
        result = self._engine.evaluate(context, spec)
        if not bypass_cache:
            self._cache.put(context.state_hash, prim_hash, result)
        return spec.name, result, False

    async def sample_parallel(
        self,
        state: dict[str, Any] | str,
        primitives: list[PrimitiveSpec] | dict[str, PrimitiveSpec],
        bypass_cache: bool = False,
        backend: str = "spectral_simd",
    ) -> tuple[SharedStateContext, dict[str, PrimitiveResult], int]:
        """Encode state once and evaluate all primitives concurrently via asyncio.gather.

        Args:
            state: Structured dictionary or string state context.
            primitives: List or keyed dictionary of PrimitiveSpec declarations.
            bypass_cache: When True, skips cache lookup and forces fresh evaluation.
            backend: Execution backend ('spectral_simd' or 'open_weights').

        Returns:
            Tuple of (SharedStateContext, ordered dict of PrimitiveResult, cache_hit_count).
        """
        context = self._engine.encode_state(state)
        spec_list = self._normalize_primitives(primitives)
        tasks = [
            self._evaluate_single(context, spec, bypass_cache=bypass_cache, backend=backend)
            for spec in spec_list
        ]
        completed = await asyncio.gather(*tasks)

        ordered_results: dict[str, PrimitiveResult] = {}
        cache_hits = 0
        for name, res, was_hit in completed:
            ordered_results[name] = res
            if was_hit:
                cache_hits += 1

        return context, ordered_results, cache_hits

    def sample_sync(
        self,
        state: dict[str, Any] | str,
        primitives: list[PrimitiveSpec] | dict[str, PrimitiveSpec],
        bypass_cache: bool = False,
    ) -> tuple[SharedStateContext, dict[str, PrimitiveResult], int]:
        """Synchronous convenience wrapper for evaluating shared-state primitives."""
        context = self._engine.encode_state(state)
        spec_list = self._normalize_primitives(primitives)
        ordered_results: dict[str, PrimitiveResult] = {}
        cache_hits = 0
        for spec in spec_list:
            prim_hash = compute_primitive_hash(spec)
            if not bypass_cache:
                cached = self._cache.get(context.state_hash, prim_hash)
                if cached is not None:
                    ordered_results[spec.name] = cached
                    cache_hits += 1
                    continue
            result = self._engine.evaluate(context, spec)
            if not bypass_cache:
                self._cache.put(context.state_hash, prim_hash, result)
            ordered_results[spec.name] = result
        return context, ordered_results, cache_hits
