import sqlite3
from daisy.memory.sqlite_store import SQLiteStore


def test_store_and_get_fact(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("name", "John")
    fact = db.get_fact("name")
    assert fact is not None
    assert fact["key"] == "name"
    assert fact["value"] == "John"
    assert fact["category"] == "general"


def test_store_updates_existing_fact(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("name", "John")
    db.store_fact("name", "Jane")
    fact = db.get_fact("name")
    assert fact["value"] == "Jane"


def test_get_fact_returns_none_if_missing(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    assert db.get_fact("nonexistent") is None


def test_search_facts_fts(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("project", "DAISY voice assistant")
    db.store_fact("color", "blue")
    results = db.search_facts("DAISY")
    assert len(results) == 1
    assert results[0]["key"] == "project"


def test_search_facts_multiple_results(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("weather_api", "OpenWeatherMap")
    db.store_fact("project", "Weather app")
    results = db.search_facts("weather")
    assert len(results) == 2


def test_search_facts_no_match(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("name", "John")
    results = db.search_facts("ZYXW_impossible")
    assert len(results) == 0


def test_search_facts_empty_string(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("name", "John")
    results = db.search_facts("")
    assert results == []


def test_search_facts_whitespace_string(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("name", "John")
    results = db.search_facts("   ")
    assert results == []


def test_search_facts_special_fts5_chars(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("email", "user@example.com")
    results = db.search_facts("user@example.com")
    assert len(results) == 1
    assert results[0]["key"] == "email"


def test_search_facts_quoted_term(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("project", "DAISY v2 voice assistant")
    results = db.search_facts("voice assistant")
    assert len(results) == 1


def test_search_facts_obeys_limit(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("a", "xyz value")
    db.store_fact("b", "xyz value")
    db.store_fact("c", "xyz value")
    results = db.search_facts("xyz", limit=2)
    assert len(results) == 2


def test_search_facts_limit_zero(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("name", "John")
    results = db.search_facts("John", limit=0)
    assert len(results) == 0


def test_get_all_facts_empty(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    assert db.get_all_facts() == []


def test_get_all_facts_one(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("name", "John")
    assert len(db.get_all_facts()) == 1


def test_get_all_facts_many(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    for i in range(100):
        db.store_fact(f"key{i}", f"val{i}")
    facts = db.get_all_facts()
    assert len(facts) == 100


def test_delete_fact(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("name", "John")
    db.delete_fact("name")
    assert db.get_fact("name") is None


def test_delete_fact_nonexistent(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.delete_fact("nonexistent")
    assert db.get_all_facts() == []


def test_delete_fact_then_re_add(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("name", "John")
    db.delete_fact("name")
    db.store_fact("name", "Jane")
    fact = db.get_fact("name")
    assert fact["value"] == "Jane"


def test_store_fact_empty_key(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("", "empty key")
    fact = db.get_fact("")
    assert fact is not None
    assert fact["value"] == "empty key"


def test_store_fact_very_long_value(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    long_val = "x" * 10000
    db.store_fact("long", long_val)
    fact = db.get_fact("long")
    assert len(fact["value"]) == 10000


def test_store_fact_special_characters(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("api_key", "sk-abc123!@#$%^&*()_+-=[]{}|;':\",./<>?`~")
    fact = db.get_fact("api_key")
    assert fact is not None


def test_store_fact_null_bytes(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.store_fact("binary", "val\x00with\x00nulls")
    fact = db.get_fact("binary")
    assert fact is not None


def test_session_lifecycle(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    sid = db.start_session()
    assert sid is not None
    db.end_session(sid, "Discussed the weather")
    summary = db.get_last_session_summary()
    assert summary == "Discussed the weather"


def test_multiple_sessions(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.end_session(db.start_session(), "Session 1")
    db.end_session(db.start_session(), "Session 2")
    assert db.get_last_session_summary() == "Session 2"


def test_get_last_session_summary_none_when_empty(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.start_session()
    assert db.get_last_session_summary() is None


def test_end_session_none_summary(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    sid = db.start_session()
    db.end_session(sid, None)
    assert db.get_last_session_summary() is None


def test_end_session_none_does_not_overwrite_summary(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    sid = db.start_session()
    db.end_session(sid, "original summary")
    db.end_session(sid, None)
    assert db.get_last_session_summary() == "original summary"


def test_end_session_non_existent_id(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.end_session(99999, "orphan summary")
    assert db.get_last_session_summary() is None


def test_multiple_start_sessions_unique_ids(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    ids = {db.start_session() for _ in range(10)}
    assert len(ids) == 10


def test_all_null_summaries_returns_none(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    for _ in range(3):
        sid = db.start_session()
        db.end_session(sid, None)
    assert db.get_last_session_summary() is None


def test_close_then_operation_raises(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    db.close()
    try:
        db.get_all_facts()
        assert False, "should have raised"
    except sqlite3.ProgrammingError:
        pass


def test_db_created_with_wal(tmp_path):
    db = SQLiteStore(str(tmp_path / "test.db"))
    row = db._conn.execute("PRAGMA journal_mode").fetchone()
    assert row[0] == "wal"
