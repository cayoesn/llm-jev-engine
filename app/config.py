"""Centralized environment configuration for the Jev System-1 Decision Engine."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime settings governing System-1 calibration, caching, resilience, and FinOps.

    Attributes:
        JEV_ENV: Deployment environment identifier.
        JEV_LOG_LEVEL: Structured logger verbosity level.
        JEV_SHARED_CACHE_MAX_SIZE: Maximum entries in the O(1) OrderedDict LRU decision cache.
        JEV_SHARED_CACHE_TTL_SECONDS: Time-to-live in seconds for cached primitive evaluations.
        JEV_DEFAULT_TEMPERATURE: Default post-hoc RLCD softmax temperature T > 0.
        JEV_MIN_CALIBRATED_CONFIDENCE_AUTO: Minimum confidence threshold for AUTO_EXECUTE gating.
        JEV_MIN_CALIBRATED_CONFIDENCE_FALLBACK: Minimum confidence threshold for FALLBACK_SYSTEM2.
        JEV_CIRCUIT_BREAKER_FAILURE_THRESHOLD: Consecutive backend failures before opening circuit.
        JEV_CIRCUIT_BREAKER_RECOVERY_SECONDS: Cooldown window before transitioning to HALF_OPEN.
        JEV_RATE_LIMIT_RPS: Token bucket sustained refill rate in requests per second.
        JEV_RATE_LIMIT_BURST: Maximum burst capacity for the token bucket rate limiter.
        JEV_AUDIT_HMAC_SECRET: Secret key used for SHA-256 HMAC cryptographic decision chaining.
        JEV_INPUT_PRICE_PER_1M_USD: Unmetered-output System-1 price per 1M input tokens in USD.
        JEV_OUTPUT_PRICE_PER_1M_USD: System-1 output token price in USD (strictly 0.0).
        JEV_SYSTEM2_INPUT_PRICE_PER_1M_USD: Baseline System-2 frontier LLM price per 1M input tokens.
        JEV_SYSTEM2_OUTPUT_PRICE_PER_1M_USD: Baseline System-2 frontier LLM price per 1M output tokens.
        JEV_OPEN_WEIGHTS_BASE_URL: Base URL for optional Ollama or vLLM logprob inference server.
        JEV_OPEN_WEIGHTS_MODEL: Model identifier for open-weights first-token logprob extraction.
        JEV_OPEN_WEIGHTS_TIMEOUT_SECONDS: HTTP timeout in seconds for open-weights backend calls.
        LANGFUSE_ENABLED: Flag enabling asynchronous Langfuse observability trace emission.
        LANGFUSE_PUBLIC_KEY: Langfuse project public key.
        LANGFUSE_SECRET_KEY: Langfuse project secret key.
        LANGFUSE_HOST: Self-hosted Langfuse server endpoint URL.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    JEV_ENV: Literal["development", "staging", "production", "test"] = "production"
    JEV_LOG_LEVEL: str = "INFO"
    JEV_SHARED_CACHE_MAX_SIZE: int = Field(default=2048, ge=16, le=1_000_000)
    JEV_SHARED_CACHE_TTL_SECONDS: float = Field(default=300.0, gt=0.0)
    JEV_DEFAULT_TEMPERATURE: float = Field(default=0.85, gt=0.0, le=10.0)
    JEV_MIN_CALIBRATED_CONFIDENCE_AUTO: float = Field(default=0.80, ge=0.0, le=1.0)
    JEV_MIN_CALIBRATED_CONFIDENCE_FALLBACK: float = Field(default=0.55, ge=0.0, le=1.0)
    JEV_CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=5, ge=1)
    JEV_CIRCUIT_BREAKER_RECOVERY_SECONDS: float = Field(default=30.0, gt=0.0)
    JEV_RATE_LIMIT_RPS: float = Field(default=500.0, gt=0.0)
    JEV_RATE_LIMIT_BURST: int = Field(default=100, ge=1)
    JEV_AUDIT_HMAC_SECRET: str = Field(
        default="jev-system-one-default-hmac-secret-2026",
        min_length=8,
    )
    JEV_INPUT_PRICE_PER_1M_USD: float = Field(default=0.042, ge=0.0)
    JEV_OUTPUT_PRICE_PER_1M_USD: float = Field(default=0.0, ge=0.0, le=0.0)
    JEV_SYSTEM2_INPUT_PRICE_PER_1M_USD: float = Field(default=2.50, gt=0.0)
    JEV_SYSTEM2_OUTPUT_PRICE_PER_1M_USD: float = Field(default=10.00, gt=0.0)
    JEV_OPEN_WEIGHTS_BASE_URL: str = "http://localhost:11434"
    JEV_OPEN_WEIGHTS_MODEL: str = "qwen2.5:0.5b"
    JEV_OPEN_WEIGHTS_TIMEOUT_SECONDS: float = Field(default=2.5, gt=0.0)
    JEV_SYSTEM2_GATEWAY_BASELINE_LATENCY_MS: float = Field(default=850.0, gt=0.0)
    JEV_SYSTEM2_JUDGE_BASELINE_LATENCY_MS: float = Field(default=1850.0, gt=0.0)
    LANGFUSE_ENABLED: bool = False
    LANGFUSE_PUBLIC_KEY: str = "pk-lf-jev"
    LANGFUSE_SECRET_KEY: str = "sk-lf-jev"
    LANGFUSE_HOST: str = "http://localhost:3016"

    @property
    def JEV_CACHE_MAX_SIZE(self) -> int:
        """Alias returning JEV_SHARED_CACHE_MAX_SIZE."""
        return self.JEV_SHARED_CACHE_MAX_SIZE

    @property
    def JEV_CACHE_TTL_SECONDS(self) -> float:
        """Alias returning JEV_SHARED_CACHE_TTL_SECONDS."""
        return self.JEV_SHARED_CACHE_TTL_SECONDS

    @property
    def OLLAMA_BASE_URL(self) -> str:
        """Alias returning JEV_OPEN_WEIGHTS_BASE_URL."""
        return self.JEV_OPEN_WEIGHTS_BASE_URL

    @property
    def OLLAMA_MODEL(self) -> str:
        """Alias returning JEV_OPEN_WEIGHTS_MODEL."""
        return self.JEV_OPEN_WEIGHTS_MODEL

    @property
    def CONFIDENCE_AUTO_THRESHOLD(self) -> float:
        """Return the minimum calibrated confidence required for autonomous execution."""
        return self.JEV_MIN_CALIBRATED_CONFIDENCE_AUTO

    @property
    def CONFIDENCE_FALLBACK_THRESHOLD(self) -> float:
        """Return the minimum calibrated confidence required before human escalation."""
        return self.JEV_MIN_CALIBRATED_CONFIDENCE_FALLBACK

    @property
    def JEV_COST_PER_1M_INPUT_TOKENS_USD(self) -> float:
        """Return the Jev System-1 cost per 1M input tokens in USD."""
        return self.JEV_INPUT_PRICE_PER_1M_USD

    @property
    def JEV_COST_PER_1M_OUTPUT_TOKENS_USD(self) -> float:
        """Return the Jev System-1 cost per 1M output tokens in USD (unmetered 0.0)."""
        return self.JEV_OUTPUT_PRICE_PER_1M_USD

    @property
    def SYSTEM2_LLM_COST_PER_1M_INPUT_USD(self) -> float:
        """Return the System-2 baseline LLM cost per 1M input tokens in USD."""
        return self.JEV_SYSTEM2_INPUT_PRICE_PER_1M_USD

    @property
    def SYSTEM2_LLM_COST_PER_1M_OUTPUT_USD(self) -> float:
        """Return the System-2 baseline LLM cost per 1M output tokens in USD."""
        return self.JEV_SYSTEM2_OUTPUT_PRICE_PER_1M_USD


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of the application Settings."""
    return Settings()
