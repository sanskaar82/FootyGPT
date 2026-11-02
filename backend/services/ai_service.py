"""AI service entry point, delegating to a provider adapter.

Provider selection is driven by ``settings.ai_provider`` (env var
``AI_PROVIDER``). The default is OpenRouter for backward compatibility.
Gemini is supported via ``GeminiProvider``.

Automatic provider fallback is driven by ``settings.ai_fallback_provider``
(env var ``AI_FALLBACK_PROVIDER``). When configured and distinct from the
primary provider, fallback is attempted once if the primary provider raises
a retryable error (rate limit, timeout, or retryable server error).
Fallback is disabled by default.

LangSmith observability is integrated at this shared layer so both providers
are traced without duplicating logic. Tracing is disabled by default and
never affects the AI request.

The public API (constructor signature, ``complete``, ``model``,
``get_ai_service``) is unchanged so existing routes and tests keep working.
"""

import logging

import httpx

from backend.config import Settings, settings, SUPPORTED_AI_PROVIDERS
from backend.exceptions import (
    AIConfigError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
)
from backend.services.providers.base import AIProvider
from backend.services.providers.gemini import GeminiProvider
from backend.services.providers.openrouter import MAX_ANSWER_CHARS, OpenRouterProvider
from backend.services.tracing import LangSmithTracer

logger = logging.getLogger(__name__)

# Exception types that trigger fallback when raised by the primary provider.
_FALLBACK_TRIGGER_EXCEPTIONS = (AIRateLimitError, AITimeoutError)


def _build_provider(
    config: Settings,
    provider_name: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AIProvider:
    """Instantiate the provider adapter for the given provider name.

    Unknown providers are rejected loudly rather than falling back silently.
    """
    if provider_name not in SUPPORTED_AI_PROVIDERS:
        raise AIConfigError(f"Unsupported AI provider: {provider_name!r}")

    if provider_name == "openrouter":
        return OpenRouterProvider(config, transport=transport)

    if provider_name == "gemini":
        return GeminiProvider(config, transport=transport)

    # Defensive: should be unreachable due to the allowed-provider check above.
    raise AIConfigError(f"Unsupported AI provider: {provider_name!r}")


def _provider_name(provider: AIProvider) -> str:
    """Return the short provider name (e.g. 'openrouter', 'gemini')."""
    return provider.__class__.__name__.replace("Provider", "").lower()


class AIService:
    """Backward-compatible wrapper around AI providers with optional fallback.

    Delegates to the primary provider selected by ``config.ai_provider``.
    When ``config.ai_fallback_provider`` is set and differs from the primary,
    a secondary provider is instantiated. If the primary raises a retryable
    error, the fallback provider is attempted once.

    The constructor signature, ``complete`` method, ``model`` attribute, and
    ``get_ai_service`` function are preserved so routes and tests need no
    changes.

    After a successful ``complete`` call, ``last_provider`` and
    ``last_model`` reflect the provider and model that actually produced the
    response (which may be the fallback provider).
    """

    def __init__(
        self,
        config: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.primary_provider = _build_provider(config, config.ai_provider, transport=transport)
        self._fallback_provider: AIProvider | None = None
        self._fallback_name: str | None = None

        fallback_name = config.ai_fallback_provider
        if (
            fallback_name is not None
            and fallback_name != config.ai_provider
            and config.provider_configured(fallback_name)
        ):
            self._fallback_provider = _build_provider(
                config, fallback_name, transport=transport
            )
            self._fallback_name = fallback_name

        self._tracer = LangSmithTracer(config)

        # Track which provider/model actually produced the last response.
        self._last_provider: str | None = None
        self._last_model: str | None = None

    @property
    def model(self) -> str | None:
        """The model of the primary provider (for backward compatibility).

        Use ``last_model`` for the model that actually produced the last
        response (which may be the fallback provider).
        """
        return self.primary_provider.model

    @property
    def provider(self) -> AIProvider:
        """Backward-compatible alias for the primary provider."""
        return self.primary_provider

    @property
    def last_provider(self) -> str | None:
        """The provider that produced the last successful response."""
        return self._last_provider

    @property
    def last_model(self) -> str | None:
        """The model that produced the last successful response."""
        return self._last_model

    async def _attempt(
        self,
        provider: AIProvider,
        provider_name: str,
        messages: list[dict[str, str]],
        sink,
    ) -> str:
        """Run one provider attempt, recording result or error on the sink."""
        result = await provider.complete(messages)
        sink.record_result(response=result)
        self._last_provider = provider_name
        self._last_model = provider.model
        return result

    async def complete(self, messages: list[dict[str, str]]) -> str:
        primary_name = _provider_name(self.primary_provider)

        async def _run_primary() -> str:
            return await self.primary_provider.complete(messages)

        async def _run_fallback() -> str:
            assert self._fallback_provider is not None
            return await self._fallback_provider.complete(messages)

        if not self._tracer.enabled:
            # Tracing disabled: try primary, then fallback once on retryable error.
            try:
                result = await _run_primary()
                self._last_provider = primary_name
                self._last_model = self.primary_provider.model
                return result
            except _FALLBACK_TRIGGER_EXCEPTIONS as primary_exc:
                if self._fallback_provider is not None:
                    try:
                        result = await _run_fallback()
                        self._last_provider = self._fallback_name
                        self._last_model = self._fallback_provider.model
                        return result
                    except Exception:
                        # Fallback also failed; re-raise the primary exception.
                        raise primary_exc
                raise primary_exc
            except AIProviderError as exc:
                if exc.retryable and self._fallback_provider is not None:
                    try:
                        result = await _run_fallback()
                        self._last_provider = self._fallback_name
                        self._last_model = self._fallback_provider.model
                        return result
                    except Exception:
                        raise exc
                raise

        # Tracing enabled: record primary and fallback as separate LangSmith runs.
        async with self._tracer.trace_run(
            provider=primary_name,
            model=self.primary_provider.model or "",
            messages=messages,
            attempt=1,
            attempt_type="primary",
        ) as sink:
            try:
                return await self._attempt(
                    self.primary_provider, primary_name, messages, sink
                )
            except _FALLBACK_TRIGGER_EXCEPTIONS as exc:
                sink.record_error(exc=exc)
                if self._fallback_provider is not None:
                    fallback_name = self._fallback_name or "unknown"
                    async with self._tracer.trace_run(
                        provider=fallback_name,
                        model=self._fallback_provider.model or "",
                        messages=messages,
                        attempt=2,
                        attempt_type="fallback",
                    ) as fallback_sink:
                        try:
                            return await self._attempt(
                                self._fallback_provider,
                                fallback_name,
                                messages,
                                fallback_sink,
                            )
                        except Exception as fallback_exc:
                            fallback_sink.record_error(exc=fallback_exc)
                            raise exc
                raise
            except AIProviderError as exc:
                sink.record_error(exc=exc)
                if exc.retryable and self._fallback_provider is not None:
                    fallback_name = self._fallback_name or "unknown"
                    async with self._tracer.trace_run(
                        provider=fallback_name,
                        model=self._fallback_provider.model or "",
                        messages=messages,
                        attempt=2,
                        attempt_type="fallback",
                    ) as fallback_sink:
                        try:
                            return await self._attempt(
                                self._fallback_provider,
                                fallback_name,
                                messages,
                                fallback_sink,
                            )
                        except Exception as fallback_exc:
                            fallback_sink.record_error(exc=fallback_exc)
                            raise exc
                raise
            except AIConfigError as exc:
                sink.record_error(exc=exc)
                raise


def get_ai_service() -> AIService:
    return AIService(settings)
