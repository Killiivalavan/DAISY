class ConversationBuffer:
    def __init__(self, max_turns: int = 20):
        self._messages: list[dict] = []
        self._max_messages = max(max_turns, 1) * 2

    def add(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self._max_messages:
            self._messages.pop(0)

    def get_context(self) -> list[dict]:
        return [dict(m) for m in self._messages]

    def clear(self):
        self._messages.clear()

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def turn_count(self) -> int:
        return len(self._messages) // 2
