"""Thread-safe CircuitBreaker, NoOpCircuitBreaker, TokenBucketRateLimiter, and ResilientExecutor."""

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, TypeVar

T = TypeVar("T")
CircuitState = Literal["CLOSED", "OPEN", "HALF_OPEN"]


class ResilienceException(RuntimeError):
    """Base exception raised when a resilience policy rejects or aborts execution."""


class CircuitBreakerOpenError(ResilienceException):
    """Raised when an operation is rejected because the circuit breaker is OPEN."""


class RateLimitExceededError(ResilienceException):
    """Raised when the token bucket rate limiter has insufficient tokens."""


class CircuitBreakerProtocol(Protocol):
    """Structural protocol for circuit breaker implementations."""

    @property
    def state(self) -> CircuitState:
        """Return current circuit state."""

    def allow_request(self) -> bool:
        """Return True if a request is permitted to execute."""

    def before_call(self) -> None:
        """Verify circuit allows execution or raise CircuitBreakerOpenError."""

    def record_success(self) -> None:
        """Record a successful execution."""

    def record_failure(self) -> None:
        """Record a failed execution."""

    def reset(self) -> None:
        """Reset circuit breaker state to CLOSED."""


class NoOpCircuitBreaker:
    """Null-object circuit breaker that unconditionally permits execution in O(1)."""

    @property
    def state(self) -> CircuitState:
        """Always return CLOSED."""
        return "CLOSED"

    @property
    def failure_count(self) -> int:
        """Always return 0 failures."""
        return 0

    def allow_request(self) -> bool:
        """Unconditionally permit request execution."""
        return True

    def before_call(self) -> None:
        """Unconditionally permit execution."""
        return None

    def record_success(self) -> None:
        """No-op success callback."""
        return None

    def record_failure(self) -> None:
        """No-op failure callback."""
        return None

    def reset(self) -> None:
        """No-op state reset."""
        return None


class CircuitBreaker:
    """Thread-safe three-state (CLOSED, OPEN, HALF_OPEN) circuit breaker protected by RLock."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
        recovery_timeout_seconds: float | None = None,
    ) -> None:
        """Initialize the CircuitBreaker with failure threshold and cooldown window."""
        resolved_recovery = (
            recovery_timeout_seconds if recovery_timeout_seconds is not None else recovery_seconds
        )
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1.")
        if resolved_recovery <= 0.0:
            raise ValueError("recovery_timeout_seconds must be > 0.0.")
        self._lock = threading.RLock()
        self._failure_threshold = int(failure_threshold)
        self._recovery_seconds = float(resolved_recovery)
        self._state: CircuitState = "CLOSED"
        self._failure_count = 0
        self._opened_at_monotonic = 0.0

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state, transitioning OPEN -> HALF_OPEN after cooldown."""
        with self._lock:
            if self._state == "OPEN":
                elapsed = time.monotonic() - self._opened_at_monotonic
                if elapsed >= self._recovery_seconds:
                    self._state = "HALF_OPEN"
            return self._state

    @property
    def failure_count(self) -> int:
        """Return the consecutive failure counter."""
        with self._lock:
            return self._failure_count

    def allow_request(self) -> bool:
        """Return True if the circuit is CLOSED or probing in HALF_OPEN state."""
        with self._lock:
            current = self.state
            return current in ("CLOSED", "HALF_OPEN")

    def before_call(self) -> None:
        """Verify the circuit allows execution or raise CircuitBreakerOpenError."""
        if not self.allow_request():
            raise CircuitBreakerOpenError("Circuit breaker is OPEN.")

    def record_success(self) -> None:
        """Reset failure count and close the circuit after a successful call."""
        with self._lock:
            self._failure_count = 0
            self._state = "CLOSED"

    def record_failure(self) -> None:
        """Increment failure count and trip circuit to OPEN when threshold is reached."""
        with self._lock:
            self._failure_count += 1
            if self._state == "HALF_OPEN" or self._failure_count >= self._failure_threshold:
                self._state = "OPEN"
                self._opened_at_monotonic = time.monotonic()

    def reset(self) -> None:
        """Explicitly reset the circuit breaker to CLOSED with zero failures."""
        with self._lock:
            self._state = "CLOSED"
            self._failure_count = 0
            self._opened_at_monotonic = 0.0


class TokenBucketRateLimiter:
    """Thread-safe O(1) token bucket rate limiter protected by a single RLock."""

    def __init__(
        self,
        rate_per_second: float = 500.0,
        burst_capacity: int = 100,
    ) -> None:
        """Initialize the TokenBucketRateLimiter with refill rate and burst capacity."""
        if rate_per_second <= 0.0:
            raise ValueError("rate_per_second must be > 0.0.")
        if burst_capacity < 1:
            raise ValueError("burst_capacity must be >= 1.")
        self._lock = threading.RLock()
        self._rate = float(rate_per_second)
        self._capacity = float(burst_capacity)
        self._tokens = float(burst_capacity)
        self._last_refill = time.monotonic()

    def _refill_unlocked(self) -> None:
        """Refill tokens proportionally to elapsed monotonic time."""
        now = time.monotonic()
        elapsed = max(0.0, now - self._last_refill)
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    @property
    def available_tokens(self) -> float:
        """Return the currently available token count after refilling."""
        with self._lock:
            self._refill_unlocked()
            return self._tokens

    def allow_request(self, cost: float = 1.0) -> bool:
        """Attempt to consume 'cost' tokens in O(1); return True if permitted."""
        with self._lock:
            self._refill_unlocked()
            if self._tokens >= cost:
                self._tokens -= cost
                return True
            return False

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Alias for allow_request() attempting to consume tokens without raising."""
        return self.allow_request(cost=tokens)

    def acquire(self, tokens: float = 1.0) -> None:
        """Consume tokens from the bucket or raise RateLimitExceededError."""
        if not self.allow_request(cost=tokens):
            raise RateLimitExceededError("Token bucket rate limit exceeded.")

    def reset(self) -> None:
        """Restore the bucket to full burst capacity."""
        with self._lock:
            self._tokens = self._capacity
            self._last_refill = time.monotonic()


class ResilientExecutor:
    """Composes rate limiting, circuit breaking, and deterministic fallback execution."""

    def __init__(
        self,
        circuit_breaker: CircuitBreakerProtocol | None = None,
        rate_limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        """Initialize the ResilientExecutor with optional circuit breaker and rate limiter."""
        self._circuit_breaker: CircuitBreakerProtocol = circuit_breaker or NoOpCircuitBreaker()
        self._rate_limiter = rate_limiter

    @property
    def circuit_breaker(self) -> CircuitBreakerProtocol:
        """Return the active CircuitBreakerProtocol instance."""
        return self._circuit_breaker

    @property
    def rate_limiter(self) -> TokenBucketRateLimiter | None:
        """Return the configured TokenBucketRateLimiter if present."""
        return self._rate_limiter

    def execute(
        self,
        operation: Callable[..., T],
        *args: Any,
        fallback: Callable[..., T] | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute a synchronous callable with rate-limit, circuit-breaker, and fallback protection."""
        if self._rate_limiter is not None and not self._rate_limiter.allow_request():
            if fallback is not None:
                return fallback(*args, **kwargs)
            raise RateLimitExceededError("Rate limit exceeded in ResilientExecutor.")

        if not self._circuit_breaker.allow_request():
            if fallback is not None:
                return fallback(*args, **kwargs)
            raise CircuitBreakerOpenError("Circuit breaker is OPEN in ResilientExecutor.")

        try:
            result = operation(*args, **kwargs)
            self._circuit_breaker.record_success()
            return result
        except Exception:
            self._circuit_breaker.record_failure()
            if fallback is not None:
                return fallback(*args, **kwargs)
            raise

    def execute_sync(
        self,
        operation: Callable[..., T],
        *args: Any,
        fallback: Callable[..., T] | None = None,
        **kwargs: Any,
    ) -> T:
        """Alias for execute() running a synchronous callable."""
        return self.execute(operation, *args, fallback=fallback, **kwargs)

    async def aexecute(
        self,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        fallback: Callable[..., Awaitable[T] | T] | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute an async callable with rate-limit, circuit-breaker, and fallback protection."""
        if self._rate_limiter is not None and not self._rate_limiter.allow_request():
            if fallback is not None:
                fb_res = fallback(*args, **kwargs)
                return await fb_res if inspect.isawaitable(fb_res) else fb_res
            raise RateLimitExceededError("Rate limit exceeded in ResilientExecutor.")

        if not self._circuit_breaker.allow_request():
            if fallback is not None:
                fb_res = fallback(*args, **kwargs)
                return await fb_res if inspect.isawaitable(fb_res) else fb_res
            raise CircuitBreakerOpenError("Circuit breaker is OPEN in ResilientExecutor.")

        try:
            result = await operation(*args, **kwargs)
            self._circuit_breaker.record_success()
            return result
        except Exception:
            self._circuit_breaker.record_failure()
            if fallback is not None:
                fb_res = fallback(*args, **kwargs)
                return await fb_res if inspect.isawaitable(fb_res) else fb_res
            raise

    async def execute_async(
        self,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        fallback: Callable[..., Awaitable[T] | T] | None = None,
        **kwargs: Any,
    ) -> T:
        """Alias for aexecute() running an asynchronous callable."""
        return await self.aexecute(operation, *args, fallback=fallback, **kwargs)
