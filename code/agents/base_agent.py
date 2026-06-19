from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class AgentError(RuntimeError):
    """Base exception for agent execution failures."""


class AgentRetryExhaustedError(AgentError):
    """Raised when an agent fails after all retry attempts."""


@dataclass(frozen=True, slots=True)
class RetryConfig:
    max_attempts: int = 3
    initial_delay_seconds: float = 0.5
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 8.0
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds cannot be negative")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if self.max_delay_seconds < 0:
            raise ValueError("max_delay_seconds cannot be negative")


@dataclass(frozen=True, slots=True)
class AgentRunContext:
    claim_id: str | None = None
    user_id: str | None = None
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_log_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if self.claim_id:
            fields["claim_id"] = self.claim_id
        if self.user_id:
            fields["user_id"] = self.user_id
        if self.request_id:
            fields["request_id"] = self.request_id
        if self.metadata:
            fields["metadata"] = dict(self.metadata)
        return fields


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _STANDARD_LOG_RECORD_KEYS:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=True)


class BaseAgent(ABC, Generic[InputT, OutputT]):
    def __init__(
        self,
        *,
        name: str | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.name = name or self.__class__.__name__
        self.retry_config = retry_config or RetryConfig()
        self.logger = logger or logging.getLogger(f"agents.{self.name}")

    def run(self, input_data: InputT, context: AgentRunContext | None = None) -> OutputT:
        context = context or AgentRunContext()
        self.validate_input(input_data)

        start_time = time.perf_counter()
        attempt = 1
        delay_seconds = self.retry_config.initial_delay_seconds

        self._log(
            logging.INFO,
            "agent_run_started",
            context=context,
            attempt=attempt,
        )

        while True:
            try:
                output = self._execute(input_data, context)
                self.validate_output(output)

                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                self._log(
                    logging.INFO,
                    "agent_run_succeeded",
                    context=context,
                    attempt=attempt,
                    elapsed_ms=elapsed_ms,
                )
                return output

            except self.retry_config.retryable_exceptions as exc:
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                can_retry = attempt < self.retry_config.max_attempts

                self._log(
                    logging.WARNING if can_retry else logging.ERROR,
                    "agent_run_failed",
                    context=context,
                    attempt=attempt,
                    elapsed_ms=elapsed_ms,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    will_retry=can_retry,
                    exc_info=not can_retry,
                )

                if not can_retry:
                    raise AgentRetryExhaustedError(
                        f"{self.name} failed after {attempt} attempt(s)"
                    ) from exc

                self.on_retry(exc, attempt, context)
                time.sleep(delay_seconds)
                delay_seconds = min(
                    delay_seconds * self.retry_config.backoff_multiplier,
                    self.retry_config.max_delay_seconds,
                )
                attempt += 1

    def validate_input(self, input_data: InputT) -> None:
        if input_data is None:
            raise ValueError(f"{self.name} received empty input")

    def validate_output(self, output: OutputT) -> None:
        if output is None:
            raise ValueError(f"{self.name} produced empty output")

    def on_retry(
        self,
        exception: BaseException,
        attempt: int,
        context: AgentRunContext,
    ) -> None:
        return None

    @abstractmethod
    def _execute(self, input_data: InputT, context: AgentRunContext) -> OutputT:
        raise NotImplementedError

    def _log(
        self,
        level: int,
        event: str,
        *,
        context: AgentRunContext,
        exc_info: bool = False,
        **fields: Any,
    ) -> None:
        extra = {
            "event": event,
            "agent": self.name,
            **context.as_log_fields(),
            **fields,
        }
        self.logger.log(level, event, extra=extra, exc_info=exc_info)


def configure_structured_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


_STANDARD_LOG_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}
