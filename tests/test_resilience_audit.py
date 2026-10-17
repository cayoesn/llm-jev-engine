"""Unit and concurrency tests for resilience patterns and cryptographic audit chain."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.core.observability import (
    AuditChainRecord,
    CryptographicAuditChain,
    FinOpsTelemetryTracker,
    LangfuseTraceRecorder,
    bind_request_context,
    get_elapsed_latency_ms,
    record_decision_metrics,
    request_start_ns_ctx,
)
from app.core.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    NoOpCircuitBreaker,
    RateLimitExceededError,
    ResilientExecutor,
    TokenBucketRateLimiter,
)


def test_circuit_breaker_state_transitions_and_recovery() -> None:
    """Verify CLOSED -> OPEN -> HALF_OPEN -> CLOSED state transitions."""
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.05)
    assert breaker.state == "CLOSED"
    assert breaker.failure_count == 0
    breaker.before_call()
    breaker.record_success()
    assert breaker.state == "CLOSED"

    breaker.record_failure()
    assert breaker.state == "CLOSED"
    assert breaker.failure_count == 1

    breaker.record_failure()
    assert breaker.state == "OPEN"

    with pytest.raises(CircuitBreakerOpenError):
        breaker.before_call()

    time.sleep(0.06)
    assert breaker.state == "HALF_OPEN"
    breaker.before_call()
    breaker.record_success()
    assert breaker.state == "CLOSED"
    assert breaker.failure_count == 0


def test_circuit_breaker_invalid_parameters_and_reset() -> None:
    """Verify validation errors on invalid parameters and manual reset()."""
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError, match="recovery_timeout_seconds"):
        CircuitBreaker(recovery_timeout_seconds=0.0)

    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=10.0)
    breaker.record_failure()
    assert breaker.state == "OPEN"
    breaker.reset()
    assert breaker.state == "CLOSED"
    assert breaker.failure_count == 0


def test_noop_circuit_breaker_behavior() -> None:
    """Verify NoOpCircuitBreaker always remains CLOSED."""
    noop = NoOpCircuitBreaker()
    assert noop.state == "CLOSED"
    assert noop.failure_count == 0
    assert noop.allow_request() is True
    noop.before_call()
    noop.record_failure()
    noop.record_success()
    noop.reset()
    assert noop.state == "CLOSED"


def test_token_bucket_rate_limiter_refill_and_exhaustion() -> None:
    """Verify token consumption, RateLimitExceededError on exhaustion, and time refill."""
    with pytest.raises(ValueError, match="rate_per_second"):
        TokenBucketRateLimiter(rate_per_second=0.0)
    with pytest.raises(ValueError, match="burst_capacity"):
        TokenBucketRateLimiter(rate_per_second=10.0, burst_capacity=0)

    limiter = TokenBucketRateLimiter(rate_per_second=50.0, burst_capacity=2)
    assert limiter.available_tokens >= 1.9
    assert limiter.try_acquire(1.0) is True
    limiter.acquire(1.0)
    assert limiter.try_acquire(1.0) is False

    with pytest.raises(RateLimitExceededError):
        limiter.acquire(1.0)

    time.sleep(0.03)
    assert limiter.try_acquire(1.0) is True
    limiter.reset()
    assert limiter.available_tokens == pytest.approx(2.0, rel=1e-2)


@pytest.mark.asyncio
async def test_resilient_executor_sync_and_async_paths_with_fallbacks() -> None:
    """Verify ResilientExecutor composes rate limiting, circuit breaking, and fallbacks."""
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=1.0)
    limiter = TokenBucketRateLimiter(rate_per_second=100.0, burst_capacity=10)
    executor = ResilientExecutor(circuit_breaker=breaker, rate_limiter=limiter)
    assert executor.circuit_breaker is breaker
    assert executor.rate_limiter is limiter

    val = executor.execute_sync(lambda x, y: x + y, 3, 4)
    assert val == 7

    async def async_mul(a: int, b: int) -> int:
        await asyncio.sleep(0)
        return a * b

    res = await executor.execute_async(async_mul, 5, 6)
    assert res == 30

    def failing_sync() -> int:
        raise RuntimeError("backend failure")

    async def failing_async() -> int:
        raise ValueError("async backend failure")

    async def async_fb() -> int:
        return 999

    assert executor.execute(failing_sync, fallback=lambda: 42) == 42
    assert await executor.aexecute(failing_async, fallback=async_fb) == 999

    assert breaker.state == "OPEN"
    assert executor.execute(lambda: 1, fallback=lambda: 77) == 77
    assert await executor.aexecute(async_mul, 1, 2, fallback=lambda a, b: 88) == 88

    with pytest.raises(CircuitBreakerOpenError):
        executor.execute(lambda: 1)
    with pytest.raises(CircuitBreakerOpenError):
        await executor.aexecute(async_mul, 1, 2)

    breaker.reset()
    tiny_limiter = TokenBucketRateLimiter(rate_per_second=0.1, burst_capacity=1)
    lim_exec = ResilientExecutor(circuit_breaker=breaker, rate_limiter=tiny_limiter)
    assert lim_exec.execute(lambda: 10) == 10
    assert lim_exec.execute(lambda: 10, fallback=lambda: 55) == 55
    with pytest.raises(RateLimitExceededError):
        lim_exec.execute(lambda: 10)

    assert await lim_exec.aexecute(async_mul, 2, 3, fallback=lambda a, b: 66) == 66
    with pytest.raises(RateLimitExceededError):
        await lim_exec.aexecute(async_mul, 2, 3)


def test_cryptographic_audit_chain_append_verify_and_tamper_detection() -> None:
    """Verify SHA-256 HMAC hash chaining, thread safety, and tamper detection."""
    with pytest.raises(ValueError, match="secret_key"):
        CryptographicAuditChain(secret_key="short")

    chain = CryptographicAuditChain(secret_key="test-super-secret-hmac-key")
    assert len(chain) == 0
    assert chain.length == 0
    assert chain.latest_digest == "0" * 64

    empty_report = chain.verify_chain()
    assert empty_report.valid is True
    assert empty_report.chain_length == 0

    def append_worker(idx: int) -> AuditChainRecord:
        return chain.append(
            request_id=f"req-{idx}",
            state_hash=f"hash-{idx:04d}",
            gate_action="AUTO_EXECUTE",
            overall_confidence=0.91,
            decisions={"q1": "billing", "q2": "TRUE"},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(append_worker, range(25)))

    assert len(records) == 25
    assert len(chain) == 25
    assert chain.length == 25
    assert chain.latest_digest != "0" * 64
    assert records[0].prev_digest == records[0].previous_hash
    assert records[0].digest == records[0].signature
    assert records[0].hmac_signature == records[0].signature

    report = chain.verify_chain()
    assert report.valid is True
    assert report.chain_length == 25
    assert report.tampered_index is None
    assert len(chain.records()) == 25

    internal_list = getattr(chain, "_records")
    corrupted = AuditChainRecord(
        index=internal_list[10].index,
        timestamp_utc=internal_list[10].timestamp_utc,
        request_id=internal_list[10].request_id,
        state_hash="tampered-state-hash",
        gate_action=internal_list[10].gate_action,
        overall_confidence=internal_list[10].overall_confidence,
        decisions_summary=internal_list[10].decisions_summary,
        previous_hash=internal_list[10].previous_hash,
        signature=internal_list[10].signature,
    )
    internal_list[10] = corrupted

    tampered_report = chain.verify_chain()
    assert tampered_report.valid is False
    assert tampered_report.tampered_index == 10

    broken_link = AuditChainRecord(
        index=999,
        timestamp_utc=internal_list[0].timestamp_utc,
        request_id=internal_list[0].request_id,
        state_hash=internal_list[0].state_hash,
        gate_action=internal_list[0].gate_action,
        overall_confidence=internal_list[0].overall_confidence,
        decisions_summary=internal_list[0].decisions_summary,
        previous_hash="bad_prev",
        signature=internal_list[0].signature,
    )
    internal_list[0] = broken_link
    link_report = chain.verify_chain()
    assert link_report.valid is False
    assert link_report.tampered_index == 0


def test_contextvars_finops_tracker_and_langfuse_recorder() -> None:
    """Verify ContextVar request isolation, FinOps gain computation, and Langfuse recording."""
    request_start_ns_ctx.set(0)
    assert get_elapsed_latency_ms() == 0.05

    req_id = bind_request_context("custom-req-99")
    assert req_id == "custom-req-99"
    auto_id = bind_request_context()
    assert auto_id.startswith("jev-")
    assert get_elapsed_latency_ms() >= 0.0

    tracker = FinOpsTelemetryTracker()
    gains = tracker.compute_gain_metrics(
        state_input_tokens=500,
        num_primitives=4,
        jev_latency_ms=1.8,
        fast_path_triggered=True,
    )
    assert gains.output_tokens_billed == 0
    assert gains.jev_output_tokens == 0
    assert gains.speedup_factor > 100.0
    assert gains.cost_reduction_pct > 95.0
    assert gains.shared_state_amortization_ratio == 0.75

    snap = tracker.snapshot()
    assert snap["total_requests"] == 1
    assert snap["total_primitives_evaluated"] == 4
    assert snap["fast_path_hits"] == 1

    engine_stats = tracker.snapshot_stats(cache_hits=8, cache_misses=2, circuit_breaker_state="CLOSED")
    assert engine_stats.cache_hit_ratio == 0.8
    assert engine_stats.total_requests == 1

    recorder = LangfuseTraceRecorder()
    assert recorder.enabled is False
    trace_entry = recorder.record_trace(
        request_id=req_id,
        workflow="unit_test",
        state_hash="abcdef123456",
        gate_action="AUTO_EXECUTE",
        overall_confidence=0.93,
        latency_ms=1.8,
        gain_metrics=gains,
    )
    assert trace_entry["request_id"] == "custom-req-99"
    assert len(recorder.recent_traces()) == 1
    assert len(recorder.traces) == 1

    record_decision_metrics(
        workflow="unit_test",
        gate_action="AUTO_EXECUTE",
        primitives_map={"q1": "choice", "q2": "noul"},
        latency_seconds=0.0018,
    )
