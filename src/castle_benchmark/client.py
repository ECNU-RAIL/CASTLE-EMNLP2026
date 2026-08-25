"""OpenAI-compatible client helpers with credential-safe error reporting."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv


class ClientConfigurationError(RuntimeError):
    """Raised for incomplete public API configuration."""


def redact_secrets(message: object) -> str:
    """Redact common API-key forms before an exception reaches a result file or log."""
    text = str(message)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", text)
    text = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(authorization:\s*bearer\s+)[^\s,;]+", r"\1[REDACTED]", text)
    return text


@dataclass(frozen=True)
class ClientSettings:
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 3

    def resolved_base_url(self) -> str:
        return self.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


class ChatClient:
    """Thin retrying wrapper around an OpenAI-compatible chat-completions API."""

    def __init__(self, settings: ClientSettings) -> None:
        load_dotenv()
        api_key = os.getenv(settings.api_key_env)
        if not api_key:
            raise ClientConfigurationError(
                f"Set the `{settings.api_key_env}` environment variable before making API calls."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised in installation failures
            raise ClientConfigurationError("Install dependencies with `pip install -r requirements.txt`.") from exc

        self._client = OpenAI(
            api_key=api_key,
            base_url=settings.resolved_base_url(),
            timeout=settings.timeout_seconds,
        )
        self._settings = settings

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Return normalized text, finish reason, and token usage."""
        last_error: Exception | None = None
        for attempt in range(1, self._settings.max_retries + 1):
            try:
                result = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if not result.choices or result.choices[0].message.content is None:
                    raise RuntimeError("Provider returned an empty completion.")
                usage = result.usage
                return {
                    "text": result.choices[0].message.content.strip(),
                    "finish_reason": result.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    },
                }
            except Exception as exc:  # Provider errors are persisted as sanitized records by callers.
                last_error = exc
                if attempt < self._settings.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise RuntimeError(redact_secrets(last_error)) from last_error
