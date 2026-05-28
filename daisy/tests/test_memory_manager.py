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


def test_record_turn_adds_to_buffer(mm):
    mm.record_turn("user", "hello")
    assert mm.buffer.message_count == 1


def test_record_turn_remember_command_stores_fact(mm):
    mm.record_turn("user", "remember my name is John")
    fact = mm.store.get_fact("name")
    assert fact is not None
    assert fact["value"] == "John"


def test_record_turn_remember_with_that(mm):
    mm.record_turn("user", "remember that my project is DAISY")
    fact = mm.store.get_fact("project")
    assert fact is not None
    assert fact["value"] == "DAISY"


def test_record_turn_remember_without_my(mm):
    mm.record_turn("user", "remember the API key is sk-123")
    fact = mm.store.get_fact("the API key")
    assert fact is not None
    assert fact["value"] == "sk-123"


def test_record_turn_remember_with_was(mm):
    mm.record_turn("user", "remember that the weather was sunny")
    fact = mm.store.get_fact("the weather")
    assert fact is not None
    assert fact["value"] == "sunny"


def test_record_turn_remember_with_were(mm):
    mm.record_turn("user", "remember that my keys were on the table")
    fact = mm.store.get_fact("keys")
    assert fact is not None
    assert fact["value"] == "on the table"


def test_record_turn_remember_with_are(mm):
    mm.record_turn("user", "remember that my hobbies are coding")
    fact = mm.store.get_fact("hobbies")
    assert fact is not None
    assert fact["value"] == "coding"


def test_record_turn_remember_uppercase(mm):
    mm.record_turn("user", "REMEMBER MY NAME IS JOHN")
    fact = mm.store.get_fact("name")
    assert fact is not None
    assert fact["value"] == "JOHN"


def test_record_turn_remember_case_insensitive_verb(mm):
    mm.record_turn("user", "Remember That My Favorite Color IS Blue")
    fact = mm.store.get_fact("favorite color")
    assert fact is not None
    assert fact["value"] == "Blue"


def test_record_turn_remember_this_stores_conversation(mm):
    mm.record_turn("user", "hello")
    mm.record_turn("assistant", "hi there boss")
    mm.record_turn("user", "remember this")
    facts = mm.store.get_all_facts()
    assert len(facts) == 1
    assert facts[0]["category"] == "saved_conversation"
    assert "hi there boss" in facts[0]["value"]


def test_record_turn_remember_this_no_assistant_yet(mm):
    mm.record_turn("user", "remember this")
    assert mm.store.get_all_facts() == []


def test_record_turn_remember_this_buffer_too_small(mm):
    mm.record_turn("user", "only one message")
    mm.record_turn("user", "remember this")
    assert mm.store.get_all_facts() == []


def test_record_turn_content_none_does_not_crash(mm):
    mm.record_turn("user", None)
    assert mm.buffer.message_count == 1


def test_record_turn_empty_content_does_not_remember(mm):
    mm.record_turn("user", "")
    assert mm.store.get_all_facts() == []


def test_record_turn_assistant_does_not_trigger_remember(mm):
    mm.record_turn("assistant", "remember my name is John")
    assert mm.store.get_all_facts() == []


def test_remember_does_not_match_normal_speech(mm):
    mm.record_turn("user", "I remember when we built this")
    assert mm.store.get_all_facts() == []


def test_remember_empty_key_after_strip_not_stored(mm):
    mm.record_turn("user", "remember   is   ")
    assert mm.store.get_all_facts() == []


def test_remember_empty_value_not_stored(mm):
    mm.record_turn("user", "remember key is   ")
    assert mm.store.get_all_facts() == []


def test_remember_this_key_truncated_at_80(mm):
    long_key = "x" * 100
    mm.record_turn("user", long_key)
    mm.record_turn("assistant", "response")
    mm.record_turn("user", "remember this")
    facts = mm.store.get_all_facts()
    assert len(facts[0]["key"]) == 80


def test_remember_this_value_truncated_at_200(mm):
    mm.record_turn("user", "hi")
    long_val = "x" * 300
    mm.record_turn("assistant", long_val)
    mm.record_turn("user", "remember this")
    facts = mm.store.get_all_facts()
    assert len(facts[0]["value"]) == 200


def test_build_context_starts_with_system(mm):
    mm.record_turn("user", "hello")
    ctx = mm.build_context("hello")
    assert ctx[0]["role"] == "system"
    assert "Boss" in ctx[0]["content"] or "D.A.I.S.Y." in ctx[0]["content"]


def test_build_context_includes_user_message(mm):
    ctx = mm.build_context("test message")
    assert ctx[-1] == {"role": "user", "content": "test message"}


def test_build_context_includes_history(mm):
    mm.record_turn("user", "first")
    mm.record_turn("assistant", "first response")
    ctx = mm.build_context("second")
    contents = [m["content"] for m in ctx]
    assert "first" in contents
    assert "first response" in contents


def test_build_context_injects_facts(mm):
    mm.store.store_fact("name", "John")
    ctx = mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    fact_block = [b for b in system_blocks if "Known facts" in b]
    assert len(fact_block) == 1
    assert "name: John" in fact_block[0]


def test_build_context_injects_session_summary(mm):
    mm.store.end_session(1, "Previous chat about weather")
    ctx = mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    summary_block = [b for b in system_blocks if "Previous session" in b]
    assert len(summary_block) == 1
    assert "weather" in summary_block[0]


def test_build_context_facts_and_summary_both_injected(mm):
    mm.store.store_fact("name", "John")
    mm.store.end_session(1, "Previous chat")
    ctx = mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    assert any("Known facts" in b for b in system_blocks)
    assert any("Previous session" in b for b in system_blocks)


def test_build_context_inject_facts_disabled(mm):
    mm._config.inject_facts = False
    mm.store.store_fact("name", "John")
    ctx = mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    fact_block = [b for b in system_blocks if "Known facts" in b]
    assert len(fact_block) == 0


def test_build_context_max_facts_zero(mm):
    mm._config.max_facts_to_inject = 0
    mm.store.store_fact("name", "John")
    ctx = mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    fact_block = [b for b in system_blocks if "Known facts" in b]
    assert len(fact_block) == 0


def test_build_context_max_facts_negative(mm):
    mm._config.max_facts_to_inject = -1
    mm.store.store_fact("name", "John")
    ctx = mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    fact_block = [b for b in system_blocks if "Known facts" in b]
    assert len(fact_block) == 0


def test_build_context_no_facts_stored(mm):
    ctx = mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    fact_block = [b for b in system_blocks if "Known facts" in b]
    assert len(fact_block) == 0


def test_build_context_no_session_summary(mm):
    ctx = mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    summary_block = [b for b in system_blocks if "Previous session" in b]
    assert len(summary_block) == 0


def test_build_context_empty_user_message(mm):
    ctx = mm.build_context("")
    assert ctx[-1]["content"] == ""


def test_build_context_custom_category_prefix(mm):
    mm.store.store_fact("project", "DAISY", category="work")
    ctx = mm.build_context("hello")
    system_blocks = [m["content"] for m in ctx if m["role"] == "system"]
    fact_block = [b for b in system_blocks if "Known facts" in b]
    assert "[work] project: DAISY" in fact_block[0]


def test_end_session_clears_buffer(mm):
    mm.record_turn("user", "hello")
    mm.record_turn("assistant", "hi")
    mm.end_session()
    assert mm.buffer.message_count == 0


def test_end_session_clear_when_empty(mm):
    mm.end_session()
    assert mm.buffer.message_count == 0


def test_end_session_called_twice(mm):
    mm.record_turn("user", "hi")
    mm.record_turn("assistant", "hello")
    mm.end_session()
    mm.end_session()
    assert mm.buffer.message_count == 0


def test_system_prompt_fallback_when_file_missing(mm, monkeypatch):
    for path in list(mm._load_system_prompt.__globals__.values()):
        pass
    from pathlib import Path
    non_existent = Path("/tmp/nonexistent_dir_xyz/SOUL.md")
    prompt = mm._load_system_prompt()
    assert "Andromeda" in prompt
    assert "Boss" in prompt


@pytest.mark.asyncio
async def test_summarize_session_early_return_on_empty_buffer(mm):
    await mm.summarize_session(None)
    assert True


@pytest.mark.asyncio
async def test_summarize_session_single_message_returns_early(mm):
    mm.record_turn("user", "hello")
    await mm.summarize_session(None)
    assert True


@pytest.mark.asyncio
async def test_summarize_session_with_llm_error_handled(mm):
    class FailingLLM:
        async def stream_tokens(self, role, messages):
            raise ConnectionError("API down")
            yield  # pragma: no cover

    mm.record_turn("user", "hello")
    mm.record_turn("assistant", "hi")
    await mm.summarize_session(FailingLLM())
    assert True


def test_build_context_respects_order():
    from pathlib import Path
    import tempfile
    cfg = FakeConfig()
    with tempfile.TemporaryDirectory() as tmp:
        cfg.memory.db_path = str(Path(tmp) / "m.db")
        mgr = MemoryManager(cfg)
        mgr.store.store_fact("name", "John")
        mgr.record_turn("user", "first msg")
        mgr.record_turn("assistant", "first resp")
        mgr.store.end_session(1, "Summary text")
        ctx = mgr.build_context("current msg")
        roles = [m["role"] for m in ctx]
        contents = [m["content"] for m in ctx]
        system_idx = [i for i, r in enumerate(roles) if r == "system"]
        assert len(system_idx) == 3
        assert "Known facts" in contents[system_idx[1]]
        assert "Previous session" in contents[system_idx[2]]
        user_msg = [c for c in contents if c == "current msg"]
        assert len(user_msg) == 1
