"""
Centralised application settings loaded from environment variables.
Uses pydantic-settings so every value can be overridden via env vars or a .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration."""

    # ── Nebius LLM ─────────────────────────────────────────
    nebius_api_key: str = ""
    nebius_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    llm_max_tokens: int = 800
    llm_temperature: float = 0.2

    # ── GitHub ──────────────────────────────────────────────
    github_token: str | None = None
    github_api_base: str = "https://api.github.com"

    # ── HTTP client ─────────────────────────────────────────
    http_connect_timeout: float = 5.0
    http_read_timeout: float = 10.0
    http_max_retries: int = 3
    http_backoff_base: float = 0.5

    # ── Payload limits ──────────────────────────────────────
    readme_max_chars: int = 15_000
    tree_max_entries: int = 300
    file_max_chars: int = 10_000
    max_context_chars: int = 40_000

    # ── Cache ───────────────────────────────────────────────
    cache_ttl_seconds: int = 600

    # ── Server ──────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    model_config = {"env_prefix": "", "env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton used across the app
settings = Settings()
