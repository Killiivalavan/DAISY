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
        self.system_prompt = self._load_system_prompt(config.llm.system_prompt_path)

    def _load_system_prompt(self, path: str) -> str:
        try:
            with open(path, encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return "You are D.A.I.S.Y., a personal AI assistant running on a server called Andromeda. Address the user as 'Boss'. Be sharp, efficient, and precise. Lead with short, direct sentences."

    async def stream_tokens(self, user_message: str):
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
