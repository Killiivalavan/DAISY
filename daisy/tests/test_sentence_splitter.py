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
    assert s.process_token("Hello world.") is None  # no whitespace after period
    assert s.flush() == "Hello world."


def test_single_sentence_exclamation():
    s = SentenceSplitter()
    assert s.process_token("Wow!") is None  # no whitespace after !
    assert s.flush() == "Wow!"


def test_single_sentence_question():
    s = SentenceSplitter()
    assert s.process_token("Is it working?") is None  # no whitespace after ?
    assert s.flush() == "Is it working?"


def test_multiple_tokens_one_sentence():
    s = SentenceSplitter()
    assert s.process_token("Hello") is None
    assert s.process_token(" world") is None
    assert s.process_token(".") is None  # period at end of buffer, no whitespace
    assert s.flush() == "Hello world."


def test_multiple_sentences_streaming():
    s = SentenceSplitter()
    assert s.process_token("First.") is None  # period at end, no whitespace yet
    assert s.process_token(" Second.") == "First."  # ". " triggers split
    assert s.flush() == "Second."


def test_semicolons_are_not_boundaries():
    s = SentenceSplitter()
    assert s.process_token("Do this; do that.") is None
    assert s.flush() == "Do this; do that."


def test_decimal_not_split():
    s = SentenceSplitter()
    assert s.process_token("Disk is 67.") is None  # decimal point, no whitespace
    assert s.process_token("9% full.") is None
    assert s.flush() == "Disk is 67.9% full."


def test_decimal_not_split_streaming():
    s = SentenceSplitter()
    assert s.process_token("Disk is 67.") is None
    assert s.process_token("9% full boss.") is None  # period at end, no trailing ws
    assert s.flush() == "Disk is 67.9% full boss."


def test_flush_returns_remaining():
    s = SentenceSplitter()
    assert s.process_token("Partial") is None
    assert s.flush() == "Partial"


def test_flush_empty():
    s = SentenceSplitter()
    assert s.flush() == ""


def test_flush_after_complete_sentence():
    s = SentenceSplitter()
    assert s.process_token("Done. ") == "Done."
    assert s.flush() == ""


def test_multiple_sentences_same_token():
    s = SentenceSplitter()
    assert s.process_token("Hi! How are you?") == "Hi!"
    # Buffer is now " How are you?"
    assert s.process_token(" Good. ") == "How are you?"
    # Buffer is now " Good. " — the ". " triggers on the next process_token call
    assert s.process_token("") == "Good."
    assert s.flush() == ""


def test_boundary_with_trailing_space():
    s = SentenceSplitter()
    assert s.process_token("Yes. ") == "Yes."


def test_boundary_at_end_of_stream():
    s = SentenceSplitter()
    assert s.process_token("End.") is None  # no whitespace after period
    assert s.flush() == "End."
