import os
import pytest
from daisy.llm.groq_client import GroqClient


class FakeConfig:
    class GroqConfig:
        api_key_env = "GROQ_API_KEY"
        model = "llama-3.3-70b-versatile"
        temperature = 0.7
        max_tokens = 1024
        base_url = "https://api.groq.com/openai/v1"

    llm = type("LLMConfig", (), {
        "groq": GroqConfig(),
    })()


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqClient(FakeConfig())


def test_creates_client_with_key(monkeypatch, mocker):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    mock_async_openai = mocker.patch("daisy.llm.groq_client.AsyncOpenAI")

    client = GroqClient(FakeConfig())
    assert client.model == "llama-3.3-70b-versatile"
    mock_async_openai.assert_called_once_with(
        api_key="test-key-123",
        base_url="https://api.groq.com/openai/v1",
    )


@pytest.mark.asyncio
async def test_stream_tokens_yields_deltas(monkeypatch, mocker):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class FakeChoice:
        class Delta:
            content = "Hello"
        delta = Delta()

    class FakeChunk:
        choices = [FakeChoice()]

    async def fake_stream():
        yield FakeChunk()
        yield FakeChunk()

    mock_create = mocker.AsyncMock(return_value=fake_stream())
    mocker.patch("daisy.llm.groq_client.AsyncOpenAI")

    client = GroqClient(FakeConfig())
    client.client.chat.completions.create = mock_create

    messages = [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "hi"},
    ]
    tokens = [t async for t in client.stream_tokens(messages)]
    assert tokens == ["Hello", "Hello"]
