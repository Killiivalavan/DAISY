"""OpenAI-compatible client — works with any provider speaking the
OpenAI Chat Completions protocol (Groq, OpenRouter, Together, local vLLM, etc.).
"""

import os
from typing import AsyncIterator

from openai import AsyncOpenAI

from daisy.llm.client import LLMClient, LLMResponse, ToolCall


class OpenAICompatibleClient:
    """LLM client for any OpenAI-compatible API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key_env: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} environment variable not set")

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def complete(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            tools=tools,
            stream=False,
        )
        msg = response.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason or "stop",
        )

    async def stream_tokens(self, messages: list[dict]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
