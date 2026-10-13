from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from src.paper_reader.config import load_config
from src.paper_reader.llm import LLMClient, LLMHTTPError, RequestRateLimiter, resolve_model


class FakeResponse:
    def __init__(self, *, body: bytes = b"", lines: list[bytes] | None = None) -> None:
        self.body = body
        self.lines = lines or []
        self.closed = False

    def read(self) -> bytes:
        return self.body

    def __iter__(self):
        return iter(self.lines)

    def close(self) -> None:
        self.closed = True


class LiteLLMClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = LLMClient(
            base_url="https://gateway.example/v1",
            api_key="test-key",
            timeout=3,
            rate_limiter=RequestRateLimiter(1_000_000),
        )

    def test_complete_posts_openai_payload_and_bearer_auth(self) -> None:
        response = FakeResponse(body=json.dumps({"choices": [{"message": {"content": "完成"}}]}).encode())
        with patch("src.paper_reader.llm.urlopen", return_value=response) as mocked_open:
            answer = self.client.complete([{"role": "user", "content": "你好"}], model="glm-5.3")

        self.assertEqual(answer, "完成")
        request = mocked_open.call_args.args[0]
        self.assertEqual(request.full_url, "https://gateway.example/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(request.data), {"model": "glm-5.3", "messages": [{"role": "user", "content": "你好"}], "stream": False})
        self.assertTrue(response.closed)

    def test_stream_yields_sse_deltas_and_sends_session_header(self) -> None:
        response = FakeResponse(
            lines=[
                'data: {"choices":[{"delta":{"content":"第"}}]}\n'.encode(),
                'data: {"choices":[{"delta":{"content":"一段"}}]}\n'.encode(),
                b"data: [DONE]\n",
            ]
        )
        with patch("src.paper_reader.llm.urlopen", return_value=response) as mocked_open:
            chunks = list(
                self.client.stream(
                    [{"role": "user", "content": "stream"}], model="deepseek-flash", session_id="chat-42"
                )
            )

        self.assertEqual(chunks, ["第", "一段"])
        request = mocked_open.call_args.args[0]
        self.assertEqual(request.get_header("X-litellm-session-id"), "chat-42")
        self.assertTrue(response.closed)

    def test_retries_rate_limit_then_returns_completion(self) -> None:
        rate_limited = HTTPError(
            "https://gateway.example/v1/chat/completions",
            429,
            "Too Many Requests",
            {"Retry-After": "0"},
            io.BytesIO(b'{"error":{"message":"slow down"}}'),
        )
        response = FakeResponse(body=b'{"choices":[{"message":{"content":"recovered"}}]}')
        with patch("src.paper_reader.llm.urlopen", side_effect=[rate_limited, response]) as mocked_open:
            with patch("src.paper_reader.llm.time.sleep"):
                answer = self.client.complete([{"role": "user", "content": "retry"}], model="glm-5.3")

        self.assertEqual(answer, "recovered")
        self.assertEqual(mocked_open.call_count, 2)

    def test_unknown_stored_model_falls_back_to_configured_default(self) -> None:
        config = load_config(
            environ={
                "PAPER_READER_ENV_FILE": "/definitely/missing/paper-reader.env",
                "PAPER_READER_LLM_MODEL": "glm-5.3",
                "PAPER_READER_LLM_ALLOWED_MODELS": "glm-5.3,deepseek-flash",
            }
        )
        with self.assertLogs("src.paper_reader.llm", level="WARNING"):
            self.assertEqual(resolve_model("untrusted-model", config), "glm-5.3")
        self.assertEqual(resolve_model("deepseek-flash", config), "deepseek-flash")

    def test_does_not_retry_client_errors(self) -> None:
        rejected = HTTPError(
            "https://gateway.example/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":{"message":"bad key"}}'),
        )
        with patch("src.paper_reader.llm.urlopen", side_effect=rejected) as mocked_open:
            with self.assertRaisesRegex(LLMHTTPError, "bad key"):
                self.client.complete([{"role": "user", "content": "nope"}], model="glm-5.3")

        self.assertEqual(mocked_open.call_count, 1)


if __name__ == "__main__":
    unittest.main()
