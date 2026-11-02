"""Application configuration loaded from environment variables.

Settings are read from the environment, and optionally from a local ``.env``
file during development. ``.env`` is git-ignored; see ``.env.example``.
"""

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SUPPORTED_AI_PROVIDERS = ("openrouter", "gemini")


class Settings(BaseSettings):
    """Configuration values used by the backend.

    Values are validated when the application starts, so a misconfigured
    ``.env`` fails loudly instead of causing strange behaviour later.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Location of the SQLite database file. Relative paths are resolved from
    # the directory the backend is started in (the project root).
    database_url: str = "sqlite:///./footygpt.db"

    # Which AI provider to use. Configurable via AI_PROVIDER; defaults to
    # OpenRouter for backward compatibility.
    ai_provider: str = "openrouter"

    # Optional secondary AI provider used when the primary provider encounters
    # a temporary failure. Empty means fallback is disabled. When set, it must
    # differ from ``ai_provider`` and must be one of the supported providers.
    # Fallback may incur additional API charges; enable only when explicitly
    # configured and verified.
    ai_fallback_provider: str | None = None

    # OpenRouter. The API key has no default: if it is missing, the backend
    # still starts, but POST /chat returns a configuration error.
    # SecretStr keeps the key out of repr(), logs, and error messages. Read the
    # real value only via ``openrouter_api_key_value``.
    openrouter_api_key: SecretStr | None = Field(default=None, min_length=1)
    openrouter_model: str | None = Field(default=None, min_length=1)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Google Gemini. The API key has no default; if it is missing while Gemini
    # is the selected provider, POST /chat returns a configuration error.
    gemini_api_key: SecretStr | None = Field(default=None, min_length=1)
    gemini_model: str | None = Field(default=None, min_length=1)
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # AI behaviour. The bounds keep a mistake in .env from causing timeouts
    # of an hour or runaway token costs.
    ai_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    ai_max_history_messages: int = Field(default=10, ge=0, le=100)
    ai_max_tokens: int = Field(default=800, ge=1, le=4000)
    ai_temperature: float = Field(default=0.3, ge=0, le=2)

    # LangSmith observability. Tracing is disabled by default; the backend
    # starts and works normally even without LangSmith credentials.
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = Field(default=None, min_length=1)
    langsmith_project: str = "FootyGPT"

    @property
    def langsmith_api_key_value(self) -> str | None:
        """The raw LangSmith API key, for building requests only."""
        return self.langsmith_api_key.get_secret_value() if self.langsmith_api_key else None

    @property
    def langsmith_enabled(self) -> bool:
        """Whether LangSmith tracing is enabled and has an API key."""
        return bool(self.langsmith_tracing and self.langsmith_api_key)

    @field_validator(
        "openrouter_api_key",
        "openrouter_model",
        "gemini_api_key",
        "gemini_model",
        "ai_fallback_provider",
        mode="before",
    )
    @classmethod
    def blank_means_unset(cls, value: object) -> object:
        """Treat an empty environment variable as "not configured"."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("ai_provider", mode="before")
    @classmethod
    def ai_provider_must_be_supported(cls, value: str) -> str:
        """Only allow providers we have adapters for."""
        if value not in SUPPORTED_AI_PROVIDERS:
            raise ValueError(f"AI_PROVIDER must be one of {SUPPORTED_AI_PROVIDERS}")
        return value

    @field_validator("ai_fallback_provider", mode="after")
    @classmethod
    def ai_fallback_must_be_supported_and_distinct(cls, value: str | None) -> str | None:
        """Fallback must be a supported provider and differ from primary."""
        if value is None:
            return None
        if value not in SUPPORTED_AI_PROVIDERS:
            raise ValueError(f"AI_FALLBACK_PROVIDER must be one of {SUPPORTED_AI_PROVIDERS}")
        return value

    @field_validator("ai_fallback_provider")
    @classmethod
    def ai_fallback_must_not_equal_primary(cls, value: str | None, info) -> str | None:
        """Fallback provider must not be the same as the primary provider."""
        if value is None:
            return None
        primary = info.data.get("ai_provider")
        if primary == value:
            raise ValueError("AI_FALLBACK_PROVIDER must not be the same as AI_PROVIDER")
        return value

    @field_validator("openrouter_base_url", "gemini_base_url")
    @classmethod
    def base_url_must_be_http(cls, value: str) -> str:
        """Only allow http(s) URLs; the key must never be sent elsewhere."""
        if not value.startswith(("https://", "http://")):
            raise ValueError("BASE_URL must start with https:// or http://")
        return value.rstrip("/")

    @property
    def openrouter_api_key_value(self) -> str | None:
        """The raw OpenRouter API key, for building the Authorization header only."""
        return self.openrouter_api_key.get_secret_value() if self.openrouter_api_key else None

    @property
    def gemini_api_key_value(self) -> str | None:
        """The raw Gemini API key, for building the Authorization header only."""
        return self.gemini_api_key.get_secret_value() if self.gemini_api_key else None

    @property
    def ai_configured(self) -> bool:
        """Whether an API key and model are both set for the active provider."""
        if self.ai_provider == "gemini":
            return bool(self.gemini_api_key and self.gemini_model)
        return bool(self.openrouter_api_key and self.openrouter_model)

    def provider_configured(self, provider: str) -> bool:
        """Whether an API key and model are both set for the given provider."""
        if provider == "gemini":
            return bool(self.gemini_api_key and self.gemini_model)
        return bool(self.openrouter_api_key and self.openrouter_model)

    @property
    def fallback_configured(self) -> bool:
        """Whether fallback is configured and ready for the fallback provider."""
        fallback = self.ai_fallback_provider
        if fallback is None:
            return False
        return self.provider_configured(fallback)


settings = Settings()
