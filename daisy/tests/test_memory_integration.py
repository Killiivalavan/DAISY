import asyncio
import os
import tempfile
import pytest
from pathlib import Path

from daisy.memory.memory_manager import MemoryManager


class FakeMemoryConfig:
    max_turns = 5
    db_path = ":memory:"
    inject_facts = True
    max_facts_to_inject = 10


class FakeConfig:
    def __init__(self):
        self.memory = FakeMemoryConfig()


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = responses or ["Here is a summary of the conversation."]

    async def stream_tokens(self, role, messages):
        for r in self.responses:
            yield r


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as tmp:
        yield str(Path(tmp) / "test.db")


def _make_mgr(db_path):
    cfg = FakeConfig()
    cfg.memory.db_path = db_path
    return MemoryManager(cfg)


@pytest.mark.asyncio
async def test_full_conversation_turn(db_path):
    mgr = _make_mgr(db_path)

    await mgr.record_turn("user", "hello")
    await mgr.record_turn("assistant", "hi boss")
    ctx = await mgr.build_context("whats the weather")

    assert len(ctx) >= 3
    assert ctx[0]["role"] == "system"
    assert ctx[-1] == {"role": "user", "content": "whats the weather"}
    assert mgr.buffer.message_count == 2


@pytest.mark.asyncio
async def test_remember_command_across_turns(db_path):
    mgr = _make_mgr(db_path)

    await mgr.record_turn("user", "remember my name is John")
    await mgr.record_turn("assistant", "got it boss")
    await mgr.record_turn("user", "what is my name")

    ctx = await mgr.build_context("what is my name")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    fact_block = [b for b in system_blocks if "Known facts" in b]

    assert len(fact_block) == 1
    assert "name: John" in fact_block[0]


@pytest.mark.asyncio
async def test_multiple_remember_commands(db_path):
    mgr = _make_mgr(db_path)

    await mgr.record_turn("user", "remember my name is John")
    await mgr.record_turn("user", "remember my project is DAISY")
    await mgr.record_turn("user", "remember my favorite color is blue")

    facts = await mgr.store.get_all_facts()
    assert len(facts) == 3


@pytest.mark.asyncio
async def test_remember_this_stores_previous_exchange(db_path):
    mgr = _make_mgr(db_path)

    await mgr.record_turn("user", "I love the new voice feature")
    await mgr.record_turn("assistant", "im glad you like it boss")
    await mgr.record_turn("user", "remember this")

    facts = await mgr.store.get_all_facts()
    assert len(facts) == 1
    assert facts[0]["category"] == "saved_conversation"
    assert facts[0]["key"] == "i love the new voice feature"


@pytest.mark.asyncio
async def test_session_persistence_across_restart(db_path):
    mgr1 = _make_mgr(db_path)
    await mgr1.record_turn("user", "remember my name is John")
    await mgr1.store.end_session(mgr1._session_id, "Discussed names")
    mgr1.buffer.clear()
    mgr1.store.close()

    mgr2 = _make_mgr(db_path)
    fact = await mgr2.store.get_fact("name")
    assert fact is not None
    assert fact["value"] == "John"

    summary = await mgr2.store.get_last_session_summary()
    assert summary == "Discussed names"
    mgr2.store.close()


@pytest.mark.asyncio
async def test_conversation_buffer_stays_within_limit(db_path):
    cfg = FakeConfig()
    cfg.memory.max_turns = 2
    cfg.memory.db_path = db_path
    mgr = MemoryManager(cfg)

    for i in range(6):
        await mgr.record_turn("user", f"msg{i}")
        await mgr.record_turn("assistant", f"resp{i}")

    assert mgr.buffer.message_count <= 4


@pytest.mark.asyncio
async def test_facts_injected_across_sessions(db_path):
    mgr1 = _make_mgr(db_path)
    await mgr1.record_turn("user", "remember my pet is a dog")
    await mgr1.store.end_session(mgr1._session_id, "Talked about pets")
    mgr1.store.close()

    mgr2 = _make_mgr(db_path)
    await mgr2.record_turn("user", "what pet do i have")
    ctx = await mgr2.build_context("what pet do i have")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]

    fact_block = [b for b in system_blocks if "Known facts" in b]
    assert len(fact_block) == 1
    assert "pet: a dog" in fact_block[0]

    summary_block = [b for b in system_blocks if "Previous session" in b]
    assert len(summary_block) == 1
    assert "pets" in summary_block[0]
    mgr2.store.close()


@pytest.mark.asyncio
async def test_summarize_session_stores_summary(db_path):
    mgr = _make_mgr(db_path)
    await mgr.record_turn("user", "hello")
    await mgr.record_turn("assistant", "hi boss")
    await mgr.record_turn("user", "whats the weather")
    await mgr.record_turn("assistant", "its sunny")

    await mgr.summarize_session(FakeLLM())
    summary = await mgr.store.get_last_session_summary()
    assert summary is not None
    assert len(summary) > 0


@pytest.mark.asyncio
async def test_summarize_session_cancelled_gracefully(db_path):
    mgr = _make_mgr(db_path)
    await mgr.record_turn("user", "hello")
    await mgr.record_turn("assistant", "hi")

    class CancellingLLM:
        async def stream_tokens(self, role, messages):
            raise asyncio.CancelledError()
            yield  # pragma: no cover

    with pytest.raises(asyncio.CancelledError):
        await mgr.summarize_session(CancellingLLM())


@pytest.mark.asyncio
async def test_build_context_structure(db_path):
    mgr = _make_mgr(db_path)
    await mgr.store.store_fact("name", "John")
    await mgr.record_turn("user", "first")
    await mgr.record_turn("assistant", "first response")
    await mgr.store.end_session(mgr._session_id, "Previous chat")
    mgr._session_id = await mgr.store.start_session()

    ctx = await mgr.build_context("current")
    roles = [m["role"] for m in ctx]

    assert roles[0] == "system"
    assert "system" in roles[1:3]
    assert {"role": "user", "content": "first"} in ctx
    assert {"role": "assistant", "content": "first response"} in ctx
    assert ctx[-1] == {"role": "user", "content": "current"}


@pytest.mark.asyncio
async def test_memory_manager_end_session_cleans_up(db_path):
    mgr = _make_mgr(db_path)
    await mgr.record_turn("user", "hello")
    await mgr.record_turn("assistant", "hi")
    await mgr.end_session()
    assert mgr.buffer.message_count == 0
