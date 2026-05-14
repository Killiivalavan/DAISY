import pytest
from daisy.llm.sentence_splitter import SentenceSplitter


def test_empty_token():
    s = SentenceSplitter()
    assert s.process_token("") is None


def test_no_boundary_returns_none():
    s = SentenceSplitter()
    assert s.process_token("hello") is None
    assert s.process_token(" world") is None


def test_single_sentence_period():
    s = SentenceSplitter()
    result = s.process_token("Hello world.")
    assert result == "Hello world."


def test_single_sentence_exclamation():
    s = SentenceSplitter()
    result = s.process_token("Wow!")
    assert result == "Wow!"


def test_single_sentence_question():
    s = SentenceSplitter()
    result = s.process_token("Is it working?")
    assert result == "Is it working?"


def test_multiple_tokens_one_sentence():
    s = SentenceSplitter()
    assert s.process_token("Hello") is None
    assert s.process_token(" world") is None
    result = s.process_token(".")
    assert result == "Hello world."


def test_multiple_sentences_streaming():
    s = SentenceSplitter()
    assert s.process_token("First.") == "First."
    assert s.process_token(" Second.") == "Second."


def test_semicolon_boundary():
    s = SentenceSplitter()
    result = s.process_token("Do this;")
    assert result == "Do this;"


def test_flush_returns_remaining():
    s = SentenceSplitter()
    assert s.process_token("Partial") is None
    assert s.flush() == "Partial"


def test_flush_empty():
    s = SentenceSplitter()
    assert s.flush() == ""


def test_flush_after_complete_sentence():
    s = SentenceSplitter()
    s.process_token("Done.")
    assert s.flush() == ""


def test_multiple_sentences_same_token():
    s = SentenceSplitter()
    assert s.process_token("Hi! How are you?") == "Hi!"
    assert s.process_token(" Good.")
    assert s.process_token("") == "Good."


def test_boundary_with_trailing_space():
    s = SentenceSplitter()
    result = s.process_token("Yes. ")
    assert result == "Yes."


def test_boundary_at_end_of_stream():
    s = SentenceSplitter()
    result = s.process_token("End.")
    assert result == "End."
