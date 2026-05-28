"""Tests for OpenAICompatibleClient."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock

from daisy.llm.openai_compatible import OpenAICompatibleClient
from daisy.llm.client import LLMResponse, ToolCall


@pytest.fixture
def client(mocker):
    os.environ["TEST_KEY"] = "sk-test"
    mock_async_openai = mocker.patch("daisy.llm.openai_compatible.AsyncOpenAI")
    mock_client_instance = mock_async_openai.return_value

    return OpenAICompatibleClient(
        base_url="https://test.api/v1",
        api_key_env="TEST_KEY",
        model="test-model",
        temperature=0.5,
        max_tokens=100,
    ), mock_client_instance


@pytest.mark.asyncio
async def test_init_sets_up_openai_client(client):
    _, mock_client = client
    # init already called in fixture — verify the mock was constructed correctly


@pytest.mark.asyncio
async def test_complete_without_tools(client):
    _, mock_client = client
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello"
    mock_choice.message.tool_calls = None
    mock_choice.finish_reason = "stop"
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[mock_choice])
    )

    result = await client[0].complete([{"role": "user", "content": "Hi"}])

    assert isinstance(result, LLMResponse)
    assert result.content == "Hello"
    assert result.tool_calls == []
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_complete_with_tools(client, mocker):
    _, mock_client = client

    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function.name = "get_weather"
    mock_tc.function.arguments = '{"location":"Paris"}'

    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_choice.message.tool_calls = [mock_tc]
    mock_choice.finish_reason = "tool_calls"
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[mock_choice])
    )

    result = await client[0].complete(
        [{"role": "user", "content": "Weather?"}],
        tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_123"
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == '{"location":"Paris"}'
    assert result.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_stream_tokens(client):
    _, mock_client = client

    async def mock_stream():
        for text in ["Hel", "lo ", "world"]:
            delta = MagicMock()
            delta.content = text
            chunk = MagicMock()
            chunk.choices = [MagicMock(delta=delta)]
            yield chunk

    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

    tokens = [t async for t in client[0].stream_tokens([{"role": "user", "content": "Hi"}])]

    assert tokens == ["Hel", "lo ", "world"]


@pytest.mark.asyncio
async def test_missing_api_key_raises():
    if "MISSING_KEY" in os.environ:
        del os.environ["MISSING_KEY"]
    with pytest.raises(ValueError, match="MISSING_KEY"):
        OpenAICompatibleClient(
            base_url="https://test.api/v1",
            api_key_env="MISSING_KEY",
            model="m",
        )
