import io
import unittest
from unittest.mock import patch

from protoagi.openai_compat import OpenAICompatibleClient


def _sse_lines(*chunks: str) -> bytes:
    body_parts: list[str] = []
    for chunk in chunks:
        body_parts.append(f"data: {chunk}\n\n")
    body_parts.append("data: [DONE]\n\n")
    return "".join(body_parts).encode("utf-8")


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._buffer = io.BytesIO(body)

    def __iter__(self):
        return iter(self._buffer.readlines())

    def close(self) -> None:
        return None


class StreamingTests(unittest.TestCase):
    def test_chat_completion_stream_yields_content(self) -> None:
        client = OpenAICompatibleClient("http://127.0.0.1:8080/v1", "test-model")
        body = _sse_lines(
            '{"choices":[{"delta":{"content":"Hello "}}]}',
            '{"choices":[{"delta":{"content":"world"}}]}',
        )
        with patch("protoagi.openai_compat.urlopen", return_value=FakeResponse(body)):
            chunks = list(
                client.chat_completion_stream(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=32,
                )
            )
        self.assertEqual(chunks, ["Hello ", "world"])

    def test_stream_ignores_non_data_lines(self) -> None:
        client = OpenAICompatibleClient("http://127.0.0.1:8080/v1", "test-model")
        body = (
            b": keepalive\n\n"
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        with patch("protoagi.openai_compat.urlopen", return_value=FakeResponse(body)):
            chunks = list(
                client.chat_completion_stream(
                    [{"role": "user", "content": "x"}],
                    max_tokens=8,
                )
            )
        self.assertEqual(chunks, ["ok"])


if __name__ == "__main__":
    unittest.main()
