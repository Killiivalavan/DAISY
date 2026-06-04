import pytest
from daisy.memory.memory_manager import MemoryManager


class FakeMemoryConfig:
    max_turns = 20
    db_path = ":memory:"
    inject_facts = True
    max_facts_to_inject = 15


class FakeConfig:
    def __init__(self):
        self.memory = FakeMemoryConfig()


@pytest.fixture
def mm(tmp_path):
    cfg = FakeConfig()
    cfg.memory.db_path = str(tmp_path / "memory.db")
    return MemoryManager(cfg)


@pytest.mark.asyncio
async def test_record_turn_adds_to_buffer(mm):
    await mm.record_turn("user", "hello")
    assert mm.buffer.message_count == 1


@pytest.mark.asyncio
async def test_record_turn_remember_command_stores_fact(mm):
    await mm.record_turn("user", "remember my name is John")
    fact = await mm.store.get_fact("name")
    assert fact is not None
    assert fact["value"] == "John"


@pytest.mark.asyncio
async def test_record_turn_remember_with_that(mm):
    await mm.record_turn("user", "remember that my project is DAISY")
    fact = await mm.store.get_fact("project")
    assert fact is not None
    assert fact["value"] == "DAISY"


@pytest.mark.asyncio
async def test_record_turn_remember_without_my(mm):
    await mm.record_turn("user", "remember the API key is sk-123")
    fact = await mm.store.get_fact("the API key")
    assert fact is not None
    assert fact["value"] == "sk-123"


@pytest.mark.asyncio
async def test_record_turn_remember_with_was(mm):
    await mm.record_turn("user", "remember that the weather was sunny")
    fact = await mm.store.get_fact("the weather")
    assert fact is not None
    assert fact["value"] == "sunny"


@pytest.mark.asyncio
async def test_record_turn_remember_with_were(mm):
    await mm.record_turn("user", "remember that my keys were on the table")
    fact = await mm.store.get_fact("keys")
    assert fact is not None
    assert fact["value"] == "on the table"


@pytest.mark.asyncio
async def test_record_turn_remember_with_are(mm):
    await mm.record_turn("user", "remember that my hobbies are coding")
    fact = await mm.store.get_fact("hobbies")
    assert fact is not None
    assert fact["value"] == "coding"


@pytest.mark.asyncio
async def test_record_turn_remember_uppercase(mm):
    await mm.record_turn("user", "REMEMBER MY NAME IS JOHN")
    fact = await mm.store.get_fact("name")
    assert fact is not None
    assert fact["value"] == "JOHN"


@pytest.mark.asyncio
async def test_record_turn_remember_case_insensitive_verb(mm):
    await mm.record_turn("user", "Remember That My Favorite Color IS Blue")
    fact = await mm.store.get_fact("favorite color")
    assert fact is not None
    assert fact["value"] == "Blue"


@pytest.mark.asyncio
async def test_record_turn_remember_this_stores_conversation(mm):
    await mm.record_turn("user", "hello")
    await mm.record_turn("assistant", "hi there boss")
    await mm.record_turn("user", "remember this")
    facts = await mm.store.get_all_facts()
    assert len(facts) == 1
    assert facts[0]["category"] == "saved_conversation"
    assert "hi there boss" in facts[0]["value"]


@pytest.mark.asyncio
async def test_record_turn_remember_this_no_assistant_yet(mm):
    await mm.record_turn("user", "remember this")
    assert await mm.store.get_all_facts() == []


@pytest.mark.asyncio
async def test_record_turn_remember_this_buffer_too_small(mm):
    await mm.record_turn("user", "only one message")
    await mm.record_turn("user", "remember this")
    assert await mm.store.get_all_facts() == []


@pytest.mark.asyncio
async def test_record_turn_content_none_does_not_crash(mm):
    await mm.record_turn("user", None)
    assert mm.buffer.message_count == 1


@pytest.mark.asyncio
async def test_record_turn_empty_content_does_not_remember(mm):
    await mm.record_turn("user", "")
    assert await mm.store.get_all_facts() == []


@pytest.mark.asyncio
async def test_record_turn_assistant_does_not_trigger_remember(mm):
    await mm.record_turn("assistant", "remember my name is John")
    assert await mm.store.get_all_facts() == []


@pytest.mark.asyncio
async def test_remember_does_not_match_normal_speech(mm):
    await mm.record_turn("user", "I remember when we built this")
    assert await mm.store.get_all_facts() == []


@pytest.mark.asyncio
async def test_remember_empty_key_after_strip_not_stored(mm):
    await mm.record_turn("user", "remember   is   ")
    assert await mm.store.get_all_facts() == []


@pytest.mark.asyncio
async def test_remember_empty_value_not_stored(mm):
    await mm.record_turn("user", "remember key is   ")
    assert await mm.store.get_all_facts() == []


@pytest.mark.asyncio
async def test_remember_this_key_truncated_at_80(mm):
    long_key = "x" * 100
    await mm.record_turn("user", long_key)
    await mm.record_turn("assistant", "response")
    await mm.record_turn("user", "remember this")
    facts = await mm.store.get_all_facts()
    assert len(facts[0]["key"]) == 80


@pytest.mark.asyncio
async def test_remember_this_value_truncated_at_200(mm):
    await mm.record_turn("user", "hi")
    long_val = "x" * 300
    await mm.record_turn("assistant", long_val)
    await mm.record_turn("user", "remember this")
    facts = await mm.store.get_all_facts()
    assert len(facts[0]["value"]) == 200


@pytest.mark.asyncio
async def test_build_context_starts_with_system(mm):
    await mm.record_turn("user", "hello")
    ctx = await mm.build_context("hello")
    assert ctx[0]["role"] == "system"
    assert "Boss" in ctx[0]["content"] or "D.A.I.S.Y." in ctx[0]["content"]


@pytest.mark.asyncio
async def test_build_context_includes_user_message(mm):
    ctx = await mm.build_context("test message")
    assert ctx[-1] == {"role": "user", "content": "test message"}


@pytest.mark.asyncio
async def test_build_context_includes_history(mm):
    await mm.record_turn("user", "first")
    await mm.record_turn("assistant", "first response")
    ctx = await mm.build_context("second")
    contents = [m["content"] for m in ctx]
    assert "first" in contents
    assert "first response" in contents


@pytest.mark.asyncio
async def test_build_context_injects_facts(mm):
    await mm.store.store_fact("name", "John")
    ctx = await mm.build_context("what is my name")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    fact_block = [b for b in system_blocks if "Known facts" in b]
    assert len(fact_block) == 1
    assert "name: John" in fact_block[0]


@pytest.mark.asyncio
async def test_build_context_injects_session_summary(mm):
    await mm.store.end_session(1, "Previous chat about weather")
    ctx = await mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    summary_block = [b for b in system_blocks if "Previous session" in b]
    assert len(summary_block) == 1
    assert "weather" in summary_block[0]


@pytest.mark.asyncio
async def test_build_context_facts_and_summary_both_injected(mm):
    await mm.store.store_fact("name", "John")
    await mm.store.end_session(1, "Previous chat")
    ctx = await mm.build_context("what is my name")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    assert any("Known facts" in b for b in system_blocks)
    assert any("Previous session" in b for b in system_blocks)


@pytest.mark.asyncio
async def test_build_context_inject_facts_disabled(mm):
    mm._config.inject_facts = False
    await mm.store.store_fact("name", "John")
    ctx = await mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    fact_block = [b for b in system_blocks if "Known facts" in b]
    assert len(fact_block) == 0


@pytest.mark.asyncio
async def test_build_context_max_facts_zero(mm):
    mm._config.max_facts_to_inject = 0
    await mm.store.store_fact("name", "John")
    ctx = await mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    fact_block = [b for b in system_blocks if "Known facts" in b]
    assert len(fact_block) == 0


@pytest.mark.asyncio
async def test_build_context_max_facts_negative(mm):
    mm._config.max_facts_to_inject = -1
    await mm.store.store_fact("name", "John")
    ctx = await mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    fact_block = [b for b in system_blocks if "Known facts" in b]
    assert len(fact_block) == 0


@pytest.mark.asyncio
async def test_build_context_no_facts_stored(mm):
    ctx = await mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    fact_block = [b for b in system_blocks if "Known facts" in b]
    assert len(fact_block) == 0


@pytest.mark.asyncio
async def test_build_context_no_session_summary(mm):
    ctx = await mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    summary_block = [b for b in system_blocks if "Previous session" in b]
    assert len(summary_block) == 0


@pytest.mark.asyncio
async def test_build_context_empty_user_message(mm):
    ctx = await mm.build_context("")
    assert ctx[-1]["content"] == ""


@pytest.mark.asyncio
async def test_build_context_custom_category_prefix(mm):
    await mm.store.store_fact("project", "DAISY", category="work")
    ctx = await mm.build_context("tell me about my project")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    fact_block = [b for b in system_blocks if "Known facts" in b]
    assert "[work] project: DAISY" in fact_block[0]


@pytest.mark.asyncio
async def test_end_session_clears_buffer(mm):
    await mm.record_turn("user", "hello")
    await mm.record_turn("assistant", "hi")
    await mm.end_session()
    assert mm.buffer.message_count == 0


@pytest.mark.asyncio
async def test_end_session_clear_when_empty(mm):
    await mm.end_session()
    assert mm.buffer.message_count == 0


@pytest.mark.asyncio
async def test_end_session_called_twice(mm):
    await mm.record_turn("user", "hi")
    await mm.record_turn("assistant", "hello")
    await mm.end_session()
    await mm.end_session()
    assert mm.buffer.message_count == 0


@pytest.mark.asyncio
async def test_system_prompt_fallback_when_file_missing(mm):
    import asyncio
    prompt = await mm._load_system_prompt()
    assert "Andromeda" in prompt
    assert "Boss" in prompt


@pytest.mark.asyncio
async def test_summarize_session_early_return_on_empty_buffer(mm):
    await mm.summarize_session(None)
    assert True


@pytest.mark.asyncio
async def test_summarize_session_single_message_returns_early(mm):
    await mm.record_turn("user", "hello")
    await mm.summarize_session(None)
    assert True


@pytest.mark.asyncio
async def test_summarize_session_with_llm_error_handled(mm):
    class FailingLLM:
        async def stream_tokens(self, role, messages):
            raise ConnectionError("API down")
            yield  # pragma: no cover

    await mm.record_turn("user", "hello")
    await mm.record_turn("assistant", "hi")
    await mm.summarize_session(FailingLLM())
    assert True


@pytest.mark.asyncio
async def test_build_context_respects_order():
    from pathlib import Path
    import tempfile
    cfg = FakeConfig()
    with tempfile.TemporaryDirectory() as tmp:
        cfg.memory.db_path = str(Path(tmp) / "m.db")
        mgr = MemoryManager(cfg)
        await mgr.store.store_fact("name", "John")
        await mgr.record_turn("user", "first msg")
        await mgr.record_turn("assistant", "first resp")
        await mgr.store.end_session(1, "Summary text")
        ctx = await mgr.build_context("what is my name")
        roles = [m["role"] for m in ctx]
        contents = [m["content"] for m in ctx]
        system_idx = [i for i, r in enumerate(roles) if r == "system"]
        assert len(system_idx) == 3
        assert "Known facts" in contents[system_idx[1]]
        assert "Previous session" in contents[system_idx[2]]
        # The user message should be present as the last content
        assert ctx[-1] == {"role": "user", "content": "what is my name"}
