"""Tests for LLMRouter — role routing, provider chain, fallback."""

import os
import pytest
from unittest.mock import AsyncMock

from daisy.llm.router import LLMRouter
from daisy.llm.client import LLMResponse


@pytest.fixture
def router_cfg(monkeypatch):
    """Build config dict that load_config can parse, then patch provider clients."""
    from daisy.utils.config_loader import Config

    raw = {
        "llm": {
            "providers": {
                "fake": {
                    "type": "openai_compatible",
                    "api_key_env": "FAKE_KEY",
                    "base_url": "http://fake",
                    "models": {"default": "fake-model"},
                },
                "fake2": {
                    "type": "openai_compatible",
                    "api_key_env": "FAKE_KEY2",
                    "base_url": "http://fake2",
                    "models": {"default": "fake2-model"},
                },
            },
            "routing": {
                "main_agent": {
                    "provider": "fake",
                    "model": "default",
                    "temperature": 0.5,
                    "max_tokens": 100,
                    "fallback": [],
                },
                "announcement": {
                    "provider": "fake",
                    "model": "default",
                    "temperature": 0.5,
                    "max_tokens": 100,
                    "fallback": [],
                },
            },
        },
    }
    os.environ["FAKE_KEY"] = "sk-test"
    os.environ["FAKE_KEY2"] = "sk-test2"
    cfg = Config(**raw)
    router = LLMRouter(config=cfg.llm)
    return router


class FakeStreamClient:
    def __init__(self, tokens):
        self._tokens = tokens

    async def stream_tokens(self, messages):
        for t in self._tokens:
            yield t


class FakeCompleteClient:
    def __init__(self, content="hello", tool_calls=None):
        self._content = content
        self._tool_calls = tool_calls or []

    async def complete(self, messages, tools=None):
        return LLMResponse(
            content=self._content,
            tool_calls=self._tool_calls,
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_complete_delegates_to_client(router_cfg):
    client = FakeCompleteClient(content="hi there")
    router_cfg._clients[("fake", "fake-model", 0.5, 100)] = client

    result = await router_cfg.complete("main_agent", [{"role": "user", "content": "hi"}])
    assert isinstance(result, LLMResponse)
    assert result.content == "hi there"


@pytest.mark.asyncio
async def test_stream_tokens_yields_client_tokens(router_cfg):
    client = FakeStreamClient(["Hel", "lo"])
    router_cfg._clients[("fake", "fake-model", 0.5, 100)] = client

    tokens = [t async for t in router_cfg.stream_tokens(
        "main_agent", [{"role": "user", "content": "hi"}])]
    assert tokens == ["Hel", "lo"]


@pytest.mark.asyncio
async def test_stream_tokens_falls_back_on_exception(router_cfg):
    """When primary fails, fallback client is tried."""
    good = FakeStreamClient(["recovery"])

    router_cfg._clients[("fake", "fake-model", 0.5, 100)] = BadStreamClient()
    router_cfg._clients[("fake2", "fake2-model", 0.5, 100)] = good

    # Add fallback to the routing config
    router_cfg._config.routing.main_agent.fallback = [
        {"provider": "fake2", "model": "default", "temperature": 0.5, "max_tokens": 100}
    ]

    tokens = [t async for t in router_cfg.stream_tokens(
        "main_agent", [{"role": "user", "content": "hi"}])]
    assert tokens == ["recovery"]


class BadStreamClient(FakeStreamClient):
    def __init__(self):
        super().__init__(["first"])

    async def stream_tokens(self, messages):
        raise RuntimeError("connection lost")
        yield  # pragma: no cover


@pytest.mark.asyncio
async def test_complete_raises_when_all_exhausted(router_cfg):
    router_cfg._clients[("fake", "fake-model", 0.5, 100)] = BadCompleteClient()
    router_cfg._config.routing.main_agent.fallback = []

    with pytest.raises(RuntimeError, match="All providers exhausted"):
        await router_cfg.complete("main_agent", [{"role": "user", "content": "hi"}])


class BadCompleteClient(FakeCompleteClient):
    def __init__(self):
        super().__init__(content="bad")

    async def complete(self, messages, tools=None):
        raise RuntimeError("all down")


def test_build_chain_raises_for_unknown_role(router_cfg):
    with pytest.raises(ValueError, match="No routing entry"):
        router_cfg._build_chain("unknown_role")


def test_get_client_caches_by_key(router_cfg):
    c1 = router_cfg._get_client(provider="fake", model="default", temperature=0.5, max_tokens=100)
    c2 = router_cfg._get_client(provider="fake", model="default", temperature=0.5, max_tokens=100)
    assert c1 is c2


def test_get_client_different_temp_new_cache_entry(router_cfg):
    c1 = router_cfg._get_client(provider="fake", model="default", temperature=0.5, max_tokens=100)
    c2 = router_cfg._get_client(provider="fake", model="default", temperature=0.9, max_tokens=100)
    assert c1 is not c2
