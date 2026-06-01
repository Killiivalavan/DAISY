import asyncio
import sqlite3

import pytest

from daisy.memory.sqlite_store import SQLiteStore


@pytest.mark.asyncio
async def test_store_and_get_fact(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("name", "John")
    fact = await db.get_fact("name")
    assert fact is not None
    assert fact["key"] == "name"
    assert fact["value"] == "John"
    assert fact["category"] == "general"


@pytest.mark.asyncio
async def test_store_updates_existing_fact(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("name", "John")
    await db.store_fact("name", "Jane")
    fact = await db.get_fact("name")
    assert fact["value"] == "Jane"


@pytest.mark.asyncio
async def test_get_fact_returns_none_if_missing(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    assert await db.get_fact("nonexistent") is None


@pytest.mark.asyncio
async def test_search_facts_fts(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("project", "DAISY voice assistant")
    await db.store_fact("color", "blue")
    results = await db.search_facts("DAISY")
    assert len(results) == 1
    assert results[0]["key"] == "project"


@pytest.mark.asyncio
async def test_search_facts_multiple_results(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("weather_api", "OpenWeatherMap")
    await db.store_fact("project", "Weather app")
    results = await db.search_facts("weather")
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_facts_no_match(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("name", "John")
    results = await db.search_facts("ZYXW_impossible")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_facts_empty_string(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("name", "John")
    results = await db.search_facts("")
    assert results == []


@pytest.mark.asyncio
async def test_search_facts_whitespace_string(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("name", "John")
    results = await db.search_facts("   ")
    assert results == []


@pytest.mark.asyncio
async def test_search_facts_special_fts5_chars(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("email", "user@example.com")
    results = await db.search_facts("user@example.com")
    assert len(results) == 1
    assert results[0]["key"] == "email"


@pytest.mark.asyncio
async def test_search_facts_quoted_term(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("project", "DAISY v2 voice assistant")
    results = await db.search_facts("voice assistant")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_facts_obeys_limit(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("a", "xyz value")
    await db.store_fact("b", "xyz value")
    await db.store_fact("c", "xyz value")
    results = await db.search_facts("xyz", limit=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_facts_limit_zero(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("name", "John")
    results = await db.search_facts("John", limit=0)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_get_all_facts_empty(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    assert await db.get_all_facts() == []


@pytest.mark.asyncio
async def test_get_all_facts_one(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("name", "John")
    assert len(await db.get_all_facts()) == 1


@pytest.mark.asyncio
async def test_get_all_facts_many(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    for i in range(100):
        await db.store_fact(f"key{i}", f"val{i}")
    facts = await db.get_all_facts()
    assert len(facts) == 100


@pytest.mark.asyncio
async def test_delete_fact(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("name", "John")
    await db.delete_fact("name")
    assert await db.get_fact("name") is None


@pytest.mark.asyncio
async def test_delete_fact_mixed_case_key(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("Name", "John")
    assert await db.get_fact("NAME") is not None  # get_fact lowercases
    await db.delete_fact("NAME")  # delete_fact should also lowercase
    assert await db.get_fact("name") is None


@pytest.mark.asyncio
async def test_delete_fact_nonexistent(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.delete_fact("nonexistent")
    assert await db.get_all_facts() == []


@pytest.mark.asyncio
async def test_delete_fact_then_re_add(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("name", "John")
    await db.delete_fact("name")
    await db.store_fact("name", "Jane")
    fact = await db.get_fact("name")
    assert fact["value"] == "Jane"


@pytest.mark.asyncio
async def test_store_fact_empty_key(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("", "empty key")
    fact = await db.get_fact("")
    assert fact is not None
    assert fact["value"] == "empty key"


@pytest.mark.asyncio
async def test_store_fact_very_long_value(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    long_val = "x" * 10000
    await db.store_fact("long", long_val)
    fact = await db.get_fact("long")
    assert len(fact["value"]) == 10000


@pytest.mark.asyncio
async def test_store_fact_special_characters(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("api_key", "sk-abc123!@#$%^&*()_+-=[]{}|;':\",./<>?`~")
    fact = await db.get_fact("api_key")
    assert fact is not None


@pytest.mark.asyncio
async def test_store_fact_null_bytes(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.store_fact("binary", "val\x00with\x00nulls")
    fact = await db.get_fact("binary")
    assert fact is not None


@pytest.mark.asyncio
async def test_session_lifecycle(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    sid = await db.start_session()
    assert sid is not None
    await db.end_session(sid, "Discussed the weather")
    summary = await db.get_last_session_summary()
    assert summary == "Discussed the weather"


@pytest.mark.asyncio
async def test_multiple_sessions(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.end_session(await db.start_session(), "Session 1")
    await db.end_session(await db.start_session(), "Session 2")
    assert await db.get_last_session_summary() == "Session 2"


@pytest.mark.asyncio
async def test_get_last_session_summary_none_when_empty(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.start_session()
    assert await db.get_last_session_summary() is None


@pytest.mark.asyncio
async def test_end_session_none_summary(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    sid = await db.start_session()
    await db.end_session(sid, None)
    assert await db.get_last_session_summary() is None


@pytest.mark.asyncio
async def test_end_session_none_does_not_overwrite_summary(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    sid = await db.start_session()
    await db.end_session(sid, "original summary")
    await db.end_session(sid, None)
    assert await db.get_last_session_summary() == "original summary"


@pytest.mark.asyncio
async def test_end_session_non_existent_id(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    await db.end_session(99999, "orphan summary")
    assert await db.get_last_session_summary() is None


@pytest.mark.asyncio
async def test_multiple_start_sessions_unique_ids(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    ids = {await db.start_session() for _ in range(10)}
    assert len(ids) == 10


@pytest.mark.asyncio
async def test_all_null_summaries_returns_none(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    for _ in range(3):
        sid = await db.start_session()
        await db.end_session(sid, None)
    assert await db.get_last_session_summary() is None


@pytest.mark.asyncio
async def test_close_then_operation_raises(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.close()
    try:
        await db.get_all_facts()
        assert False, "should have raised"
    except sqlite3.ProgrammingError:
        pass


def test_db_created_with_wal(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    row = db._conn.execute("PRAGMA journal_mode").fetchone()
    assert row[0] == "wal"
