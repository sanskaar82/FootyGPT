"""Optional LangSmith observability for AI requests.

Tracing is disabled by default and never affects the AI request. All
LangSmith errors are swallowed so the chatbot keeps working even when
LangSmith is unreachable or misconfigured.
"""

import datetime
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)


class LangSmithTracer:
    """Optional LangSmith tracing wrapper.

    Creates a ``langsmith.Client`` only when tracing is enabled and an API
    key is present. Every LangSmith call is wrapped in ``try/except`` so
    tracing failures never affect the AI request.
    """

    def __init__(self, config: Any):
        self.config = config
        self._client: Any = None  # None = not yet created, False = creation failed

    @property
    def enabled(self) -> bool:
        if not self.config.langsmith_enabled:
            return False
        return self._ensure_client() is not None

    def _ensure_client(self) -> Any:
        """Lazily create the LangSmith client. Returns None if unavailable."""
        if self._client is False:
            return None
        if self._client is not None:
            return self._client
        if not self.config.langsmith_enabled:
            return None
        try:
            from langsmith import Client  # type: ignore

            self._client = Client(
                api_key=self.config.langsmith_api_key_value,
            )
            return self._client
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "LangSmith client could not be created; tracing disabled: %s",
                type(exc).__name__,
            )
            self._client = False
            return None

    @asynccontextmanager
    async def trace_run(
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        attempt: int = 1,
        attempt_type: str = "primary",
    ):
        """Async context manager that traces an AI request.

        Yields a sink with ``record_result`` and ``record_error`` methods.
        When tracing is disabled, yields a no-op sink.
        """
        if not self.enabled:
            yield _NoopTrace()
            return

        start = time.perf_counter()
        client = self._ensure_client()
        run_id = str(uuid.uuid4())
        creation_succeeded = False

        if client is not None:
            try:
                client.create_run(
                    name="footygpt-chat",
                    run_type="llm",
                    inputs={"messages": messages},
                    id=run_id,
                    extra={
                        "metadata": {
                            "provider": provider,
                            "model": model,
                            "attempt": attempt,
                            "attempt_type": attempt_type,
                        }
                    },
                )
                creation_succeeded = True
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "LangSmith create_run failed: %s", type(exc).__name__
                )

        yield _TraceSink(
            client,
            run_id,
            start,
            provider,
            model,
            messages,
            attempt,
            attempt_type,
            creation_succeeded,
        )


class _NoopTrace:
    """No-op sink used when tracing is disabled."""

    def record_result(
        self, response: str = "", token_usage: dict[str, Any] | None = None
    ) -> None:
        pass

    def record_error(self, exc: Exception = None) -> None:
        pass


class _TraceSink:
    """Records the result or error of a traced AI request to LangSmith."""

    def __init__(
        self,
        client: Any,
        run_id: str | None,
        start: float,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        attempt: int = 1,
        attempt_type: str = "primary",
        creation_succeeded: bool = False,
    ):
        self._client = client
        self._run_id = run_id
        self._start = start
        self._provider = provider
        self._model = model
        self._messages = messages
        self._attempt = attempt
        self._attempt_type = attempt_type
        self._creation_succeeded = creation_succeeded
        self._finalized: bool = False

    def _latency_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def _finalize(
        self,
        outputs: dict[str, Any],
        *,
        error: str | None = None,
        status: str,
        error_category: str | None = None,
    ) -> None:
        """Finalize the LangSmith run exactly once.

        Calls ``update_run`` with ``end_time`` to transition the run out of
        ``pending`` status into a terminal state. Safe to call multiple times;
        only the first call has any effect.
        """
        if self._finalized:
            return
        if not self._creation_succeeded:
            return
        if self._run_id is None or self._client is None:
            return
        self._finalized = True
        try:
            metadata: dict[str, Any] = {
                "provider": self._provider,
                "model": self._model,
                "attempt": self._attempt,
                "attempt_type": self._attempt_type,
                "latency_ms": round(self._latency_ms(), 2),
                "status": status,
            }
            if error_category is not None:
                metadata["error_category"] = error_category
            self._client.update_run(
                run_id=self._run_id,
                outputs=outputs,
                error=error,
                end_time=datetime.datetime.now(),
                extra={"metadata": metadata},
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("LangSmith end_run failed: %s", type(exc).__name__)

    def record_result(
        self, response: str = "", token_usage: dict[str, Any] | None = None
    ) -> None:
        outputs: dict[str, Any] = {"response": response}
        if token_usage:
            outputs["token_usage"] = token_usage
        self._finalize(outputs=outputs, status="success")

    def record_error(self, exc: Exception) -> None:
        self._finalize(
            outputs={"error": type(exc).__name__},
            error=type(exc).__name__,
            status="error",
            error_category=type(exc).__name__,
        )
