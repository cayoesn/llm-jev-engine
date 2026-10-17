"""FastAPI application entrypoint, lifespan DI, middleware, and exception handlers for Jev."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app.api.routes import ensure_engine_state, router
from app.core.observability import bind_request_context, get_elapsed_latency_ms
from app.core.resilience import CircuitBreakerOpenError, RateLimitExceededError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and tear down singleton System-1 engine components during application lifespan."""
    dummy_scope = {"type": "http", "app": app}
    dummy_req = Request(dummy_scope)
    ensure_engine_state(dummy_req)
    yield


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application instance."""
    application = FastAPI(
        title="Jev System-1 Non-Autoregressive Decision Engine",
        description=(
            "Single-pass System-1 Neural Decision & Calibration Engine implementing "
            "choice/noul/score primitives, RLCD post-hoc calibration, Shared-State "
            "Parallel Sampling, JevEval inline RAG judgment, and Unmetered-Output FinOps."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Bind request-scoped ContextVars and inject latency and correlation headers."""
        incoming_id = request.headers.get("X-Request-ID")
        bound_id = bind_request_context(incoming_id)
        response = await call_next(request)
        elapsed_ms = get_elapsed_latency_ms()
        response.headers["X-Request-ID"] = bound_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.4f}"
        return response

    @application.exception_handler(CircuitBreakerOpenError)
    async def handle_circuit_breaker_open(
        request: Request,
        exc: CircuitBreakerOpenError,
    ) -> JSONResponse:
        """Return HTTP 503 Service Unavailable when the circuit breaker is OPEN."""
        return JSONResponse(
            status_code=503,
            content={
                "error": "circuit_breaker_open",
                "detail": str(exc),
                "path": request.url.path,
            },
        )

    @application.exception_handler(RateLimitExceededError)
    async def handle_rate_limit_exceeded(
        request: Request,
        exc: RateLimitExceededError,
    ) -> JSONResponse:
        """Return HTTP 429 Too Many Requests when the token bucket is exhausted."""
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "detail": str(exc),
                "path": request.url.path,
            },
        )

    application.include_router(router)
    return application


app = create_app()
