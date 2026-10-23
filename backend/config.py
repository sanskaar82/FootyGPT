"""Application configuration loaded from environment variables.

Settings are read from the environment, and optionally from a local ``.env``
file during development. ``.env`` is git-ignored; see ``.env.example``.
"""

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration values used by the backend.

    Values are validated when the application starts, so a misconfigured
    ``.env`` fails loudly instead of causing strange behaviour later.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Location of the SQLite database file. Relative paths are resolved from
    # the directory the backend is started in (the project root).
    database_url: str = "sqlite:///./footygpt.db"

    # OpenRouter. The API key has no default: if it is missing, the backend
    # still starts, but POST /chat returns a configuration error.
    # SecretStr keeps the key out of repr(), logs, and error messages. Read the
    # real value only via ``openrouter_api_key_value``.
    openrouter_api_key: SecretStr | None = Field(default=None, min_length=1)
    openrouter_model: str | None = Field(default=None, min_length=1)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # AI behaviour. The bounds keep a mistake in .env from causing timeouts
    # of an hour or runaway token costs.
    ai_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    ai_max_history_messages: int = Field(default=10, ge=0, le=100)
    ai_max_tokens: int = Field(default=800, ge=1, le=4000)
    ai_temperature: float = Field(default=0.3, ge=0, le=2)

    @field_validator("openrouter_api_key", "openrouter_model", mode="before")
    @classmethod
    def blank_means_unset(cls, value: object) -> object:
        """Treat an empty environment variable as "not configured"."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("openrouter_base_url")
    @classmethod
    def base_url_must_be_http(cls, value: str) -> str:
        """Only allow http(s) URLs; the key must never be sent elsewhere."""
        if not value.startswith(("https://", "http://")):
            raise ValueError("OPENROUTER_BASE_URL must start with https:// or http://")
        return value.rstrip("/")

    @property
    def openrouter_api_key_value(self) -> str | None:
        """The raw API key, for building the Authorization header only."""
        return self.openrouter_api_key.get_secret_value() if self.openrouter_api_key else None

    @property
    def ai_configured(self) -> bool:
        """Whether an API key and model are both set."""
        return bool(self.openrouter_api_key and self.openrouter_model)


settings = Settings()
