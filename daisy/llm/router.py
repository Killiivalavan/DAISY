"""LLM Router — lazily instantiates provider clients, dispatches calls
by logical role, and handles fallback chains on failure.
"""

import logging
from typing import AsyncIterator

from daisy.llm.client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class LLMRouter:
    """Routes LLM calls to the right provider+model per logical role.

    Each role (main_agent, summarizer, sub_agent, announcement) can use
    a different provider and model.  On failure, the router walks the
    fallback chain defined in config for that role.
    """

    def __init__(self, config):
        self._config = config
        self._clients: dict[tuple, LLMClient] = {}

    # ------------------------------------------------------------------
    # Public API — called by the state machine, memory manager, tools
    # ------------------------------------------------------------------

    async def complete(
        self, role: str, messages: list[dict], tools: list[dict] | None = None
    ) -> LLMResponse:
        """Non-streaming completion with optional tools (main agent tool loop)."""
        chain = self._build_chain(role)
        last_error = None
        for entry in chain:
            try:
                client = self._get_client(**entry)
                return await client.complete(messages, tools=tools)
            except Exception as exc:
                logger.warning(
                    f"[Router] {role}: provider={entry['provider']} "
                    f"model={entry['model']} failed: {exc}"
                )
                last_error = exc
        raise RuntimeError(
            f"[Router] All providers exhausted for role '{role}'. "
            f"Last error: {last_error}"
        )

    async def stream_tokens(
        self, role: str, messages: list[dict]
    ) -> AsyncIterator[str]:
        """Streaming text tokens (final response, announcement, summarizer, sub-agent)."""
        chain = self._build_chain(role)
        last_error = None
        for entry in chain:
            try:
                client = self._get_client(**entry)
                async for token in client.stream_tokens(messages):
                    yield token
                return
            except Exception as exc:
                logger.warning(
                    f"[Router] {role}: provider={entry['provider']} "
                    f"model={entry['model']} failed: {exc}"
                )
                last_error = exc
        raise RuntimeError(
            f"[Router] All providers exhausted for role '{role}'. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_chain(self, role: str) -> list[dict]:
        """Build ordered list of {provider, model, temperature, max_tokens} dicts.

        Primary entry comes first, followed by fallback entries in order.
        """
        routing = getattr(self._config.routing, role, None)
        if routing is None:
            raise ValueError(
                f"No routing entry for role '{role}'. "
                f"Available: main_agent, summarizer, sub_agent, announcement"
            )
        chain = [
            {
                "provider": routing.provider,
                "model": routing.model,
                "temperature": routing.temperature,
                "max_tokens": routing.max_tokens,
            }
        ]
        for fb in routing.fallback:
            chain.append(
                {
                    "provider": fb["provider"],
                    "model": fb["model"],
                    "temperature": fb.get("temperature", routing.temperature),
                    "max_tokens": fb.get("max_tokens", routing.max_tokens),
                }
            )
        return chain

    def _get_client(
        self, *, provider: str, model: str, temperature: float, max_tokens: int
    ) -> LLMClient:
        """Get or create a cached client for the given (provider, model, temp, mtok)."""
        prov_cfg = self._config.providers[provider]
        model_id = prov_cfg.models.get(model, model)

        cache_key = (provider, model_id, temperature, max_tokens)
        if cache_key not in self._clients:
            self._clients[cache_key] = self._build_client(
                prov_type=prov_cfg.type,
                base_url=prov_cfg.base_url or "",
                api_key_env=prov_cfg.api_key_env,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._clients[cache_key]

    @staticmethod
    def _build_client(
        *, prov_type: str, base_url: str, api_key_env: str, model: str,
        temperature: float, max_tokens: int,
    ) -> LLMClient:
        """Dispatch to the right client class based on provider type."""
        if prov_type == "openai_compatible":
            from daisy.llm.openai_compatible import OpenAICompatibleClient

            return OpenAICompatibleClient(
                base_url=base_url,
                api_key_env=api_key_env,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif prov_type == "anthropic":
            from daisy.llm.anthropic_client import AnthropicClient

            return AnthropicClient(
                api_key_env=api_key_env,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif prov_type == "google":
            from daisy.llm.gemini_client import GeminiClient

            return GeminiClient(
                api_key_env=api_key_env,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Unknown provider type: {prov_type}")
