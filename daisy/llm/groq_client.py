import json
import os
from openai import AsyncOpenAI


class GroqClient:
    def __init__(self, config):
        api_key = os.environ.get(config.llm.groq.api_key_env)
        if not api_key:
            raise ValueError(f"{config.llm.groq.api_key_env} environment variable not set")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=config.llm.groq.base_url,
        )
        self.model = config.llm.groq.model
        self.temperature = config.llm.groq.temperature
        self.max_tokens = config.llm.groq.max_tokens

    async def complete(self, messages: list[dict], tools: list[dict] = None):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=tools,
            stream=False,
        )
        message = response.choices[0].message
        return message

    async def stream_tokens(self, messages: list[dict]):
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
