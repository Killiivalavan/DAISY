from daisy.memory.conversation_buffer import ConversationBuffer


def test_empty_on_init():
    buf = ConversationBuffer(max_turns=20)
    assert buf.message_count == 0
    assert buf.turn_count == 0
    assert buf.get_context() == []


def test_add_user_message():
    buf = ConversationBuffer(max_turns=20)
    buf.add("user", "hello")
    assert buf.message_count == 1
    assert buf.get_context() == [{"role": "user", "content": "hello"}]


def test_add_user_and_assistant():
    buf = ConversationBuffer(max_turns=20)
    buf.add("user", "hello")
    buf.add("assistant", "hi there")
    assert buf.message_count == 2
    assert buf.turn_count == 1


def test_max_turns_trims_oldest():
    buf = ConversationBuffer(max_turns=2)
    buf.add("user", "msg1")
    buf.add("assistant", "resp1")
    buf.add("user", "msg2")
    buf.add("assistant", "resp2")
    buf.add("user", "msg3")
    assert buf.message_count == 4
    assert buf.get_context()[0]["content"] == "resp1"
    assert buf.get_context()[-1]["content"] == "msg3"


def test_get_context_returns_copy():
    buf = ConversationBuffer(max_turns=20)
    buf.add("user", "hello")
    ctx = buf.get_context()
    ctx.append({"role": "assistant", "content": "injected"})
    assert buf.message_count == 1


def test_clear():
    buf = ConversationBuffer(max_turns=20)
    buf.add("user", "hello")
    buf.add("assistant", "hi")
    buf.clear()
    assert buf.message_count == 0
    assert buf.get_context() == []


def test_turn_count_floor():
    buf = ConversationBuffer(max_turns=20)
    assert buf.turn_count == 0
    buf.add("user", "hello")
    assert buf.turn_count == 0
    buf.add("assistant", "hi")
    assert buf.turn_count == 1
    buf.add("user", "another")
    assert buf.turn_count == 1


def test_message_count():
    buf = ConversationBuffer(max_turns=20)
    assert buf.message_count == 0
    buf.add("user", "a")
    assert buf.message_count == 1
    buf.add("assistant", "b")
    assert buf.message_count == 2


def test_max_turns_zero_clamped_to_one():
    buf = ConversationBuffer(max_turns=0)
    assert buf._max_messages == 2
    buf.add("user", "m1")
    buf.add("assistant", "r1")
    buf.add("user", "m2")
    assert buf.message_count == 2
    assert buf.get_context()[0]["content"] == "r1"


def test_max_turns_negative_clamped():
    buf = ConversationBuffer(max_turns=-5)
    assert buf._max_messages == 2


def test_add_after_clear():
    buf = ConversationBuffer(max_turns=20)
    buf.add("user", "before")
    buf.clear()
    buf.add("user", "after")
    assert buf.message_count == 1
    assert buf.get_context()[0]["content"] == "after"


def test_clear_idempotent():
    buf = ConversationBuffer(max_turns=20)
    buf.add("user", "m1")
    buf.clear()
    buf.clear()
    assert buf.message_count == 0


def test_context_dict_isolation():
    buf = ConversationBuffer(max_turns=20)
    buf.add("user", "hello")
    ctx = buf.get_context()
    ctx[0]["content"] = "mutated"
    assert buf.get_context()[0]["content"] == "hello"


def test_add_none_role():
    buf = ConversationBuffer(max_turns=20)
    buf.add(None, "hello")
    assert buf.message_count == 1
    assert buf.get_context()[0]["role"] is None


def test_add_none_content():
    buf = ConversationBuffer(max_turns=20)
    buf.add("user", None)
    assert buf.message_count == 1
    assert buf.get_context()[0]["content"] is None


def test_add_empty_content():
    buf = ConversationBuffer(max_turns=20)
    buf.add("user", "")
    assert buf.message_count == 1
    assert buf.get_context()[0]["content"] == ""


def test_turn_count_boundaries():
    buf = ConversationBuffer(max_turns=20)
    assert buf.turn_count == 0
    buf.add("user", "a")
    assert buf.turn_count == 0
    buf.add("assistant", "b")
    assert buf.turn_count == 1
    buf.add("user", "c")
    assert buf.turn_count == 1
    buf.add("assistant", "d")
    assert buf.turn_count == 2


def test_very_long_content():
    buf = ConversationBuffer(max_turns=2)
    long = "x" * 100000
    buf.add("user", long)
    buf.add("assistant", "ok")
    assert buf.message_count == 2
    assert len(buf.get_context()[0]["content"]) == 100000
