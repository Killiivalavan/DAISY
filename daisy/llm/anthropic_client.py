"""Anthropic Claude client — translates between OpenAI-canonical format
and the Anthropic Messages API.
"""

import json
import os
from typing import AsyncIterator

from daisy.llm.client import LLMClient, LLMResponse, ToolCall


def _to_anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns (system_prompt, anthropic_messages) — system is a top-level
    parameter in Anthropic, not a message role.
    """
    system_parts: list[str] = []
    converted: list[dict] = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content")

        if role == "system":
            if content:
                system_parts.append(content)
            continue

        if role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg["tool_call_id"],
                            "content": content or "",
                        }
                    ],
                }
            )
            continue

        if role == "assistant" and msg.get("tool_calls"):
            content_blocks = []
            for tc in msg["tool_calls"]:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]),
                    }
                )
            if msg.get("content"):
                content_blocks.insert(0, {"type": "text", "text": msg["content"]})
            converted.append({"role": "assistant", "content": content_blocks})
            continue

        # Plain user / assistant message
        out = {"role": role, "content": _wrap_text(content) if content else []}
        converted.append(out)

    system = "\n".join(system_parts) if system_parts else None
    return system, converted


def _wrap_text(text: str) -> list[dict]:
    """Wrap a plain string as Anthropic text content blocks."""
    return [{"type": "text", "text": text}]


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tools to Anthropic format."""
    out = []
    for t in tools:
        func = t["function"]
        out.append(
            {
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return out


class AnthropicClient:
    """LLM client for the Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key_env: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required to use Claude. "
                "Install it with: pip install anthropic"
            )
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} environment variable not set")
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def complete(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> LLMResponse:
        system, an_messages = _to_anthropic_messages(messages)
        an_tools = _to_anthropic_tools(tools) if tools else None

        kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": an_messages,
            "temperature": self._temperature,
        }
        if system:
            kwargs["system"] = system
        if an_tools:
            kwargs["tools"] = an_tools

        response = await self._client.messages.create(**kwargs)

        content = None
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input),
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

    async def stream_tokens(self, messages: list[dict]) -> AsyncIterator[str]:
        system, an_messages = _to_anthropic_messages(messages)

        kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": an_messages,
            "temperature": self._temperature,
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield delta.text
