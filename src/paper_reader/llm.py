"""Small OpenAI-compatible client for the team's LiteLLM gateway."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import PaperReaderConfig, load_config

LOGGER = logging.getLogger(__name__)
MAX_RETRIES = 4


class LLMError(RuntimeError):
    """Base error returned by the configured LLM gateway."""


class LLMHTTPError(LLMError):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(f"LiteLLM request failed with HTTP {status_code}: {detail}")


class RequestRateLimiter:
    """Process-local pacing for request starts across all LLM workloads."""

    def __init__(self, qps: float) -> None:
        self.qps = qps
        self._interval = 1.0 / qps
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def wait(self) -> float:
        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_allowed_at - now)
            self._next_allowed_at = max(now, self._next_allowed_at) + self._interval
        if wait_seconds:
            time.sleep(wait_seconds)
        return wait_seconds


@dataclass(frozen=True, slots=True)
class LLMClient:
    base_url: str
    api_key: str
    timeout: float
    rate_limiter: RequestRateLimiter

    @classmethod
    def from_config(cls, config: PaperReaderConfig) -> "LLMClient":
        return cls(
            base_url=config.llm_base_url.rstrip("/"),
            api_key=config.llm_api_key,
            timeout=config.llm_timeout,
            rate_limiter=_limiter_for_qps(config.llm_qps),
        )

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        session_id: str | None = None,
    ) -> str:
        response = self._request(messages, model=model, stream=False, session_id=session_id)
        try:
            payload = json.loads(response.read().decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            raise LLMError("LiteLLM returned invalid JSON.") from exc
        finally:
            response.close()
        return _completion_content(payload)

    def stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        session_id: str | None = None,
    ) -> Iterator[str]:
        response = self._request(messages, model=model, stream=True, session_id=session_id)
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise LLMError("LiteLLM stream contained invalid JSON.") from exc
                content = _delta_content(payload)
                if content:
                    yield content
        finally:
            response.close()

    def _request(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        stream: bool,
        session_id: str | None,
    ) -> Any:
        if not self.api_key:
            raise LLMError("PAPER_READER_LLM_API_KEY is required to call the LiteLLM gateway.")
        if not self.base_url:
            raise LLMError("PAPER_READER_LLM_BASE_URL is required to call the LiteLLM gateway.")

        encoded = json.dumps(
            {"model": model, "messages": list(messages), "stream": stream}, ensure_ascii=False
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        if session_id:
            headers["x-litellm-session-id"] = session_id

        for attempt in range(MAX_RETRIES):
            self.rate_limiter.wait()
            request = Request(self.endpoint, data=encoded, headers=headers, method="POST")
            try:
                return urlopen(request, timeout=self.timeout)
            except HTTPError as exc:
                detail = _error_detail(exc)
                if not _is_retryable_status(exc.code) or attempt == MAX_RETRIES - 1:
                    raise LLMHTTPError(exc.code, detail) from exc
                _backoff(attempt, exc)
            except URLError as exc:
                raise LLMError(f"Could not reach LiteLLM gateway: {exc.reason}") from exc
        raise AssertionError("unreachable")


_limiter_lock = threading.Lock()
_limiters: dict[float, RequestRateLimiter] = {}


def _limiter_for_qps(qps: float) -> RequestRateLimiter:
    with _limiter_lock:
        return _limiters.setdefault(qps, RequestRateLimiter(qps))


def get_llm_client(config: PaperReaderConfig | None = None) -> LLMClient:
    return LLMClient.from_config(config or load_config())


def resolve_model(model: str | None, config: PaperReaderConfig | None = None) -> str:
    """Use saved per-prompt models only when they are in the deployment allowlist."""
    config = config or load_config()
    requested = (model or "").strip() or config.llm_model
    allowed = config.llm_allowed_models
    if requested.casefold() in allowed:
        return requested
    LOGGER.warning(
        "Ignoring unapproved LLM model %r; using configured default %r.", requested, config.llm_model
    )
    return config.llm_model


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def _backoff(attempt: int, error: HTTPError) -> None:
    retry_after = error.headers.get("Retry-After") if error.headers else None
    try:
        seconds = float(retry_after) if retry_after else float(2**attempt)
    except ValueError:
        seconds = float(2**attempt)
    time.sleep(min(seconds, 30.0))


def _error_detail(error: HTTPError) -> str:
    try:
        raw = error.read().decode("utf-8", "replace")
    except OSError:
        return error.reason or "request rejected"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:500] or str(error.reason or "request rejected")
    error_payload = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error_payload, dict):
        return str(error_payload.get("message") or error_payload.get("detail") or raw[:500])
    return str(error_payload or payload.get("message") or raw[:500]) if isinstance(payload, dict) else raw[:500]


def _completion_content(payload: object) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("LiteLLM response did not contain a chat completion.") from exc
    text = _content_to_text(content).strip()
    if not text:
        raise LLMError("LiteLLM returned an empty chat completion.")
    return text


def _delta_content(payload: object) -> str:
    try:
        content = payload["choices"][0]["delta"].get("content")  # type: ignore[index,union-attr]
    except (KeyError, IndexError, TypeError):
        return ""
    return _content_to_text(content)


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "") for item in content if isinstance(item, dict)
        )
    return ""
