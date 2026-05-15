import re


class SentenceSplitter:
    def __init__(self, max_length: int = 150):
        self._buffer = ""
        self._max_length = max_length

    def process_token(self, token: str):
        self._buffer += token

        if len(self._buffer) >= self._max_length:
            sentence = self._buffer.strip()
            self._buffer = ""
            return sentence

        match = re.search(r"[.!?;](?:\s|$)", self._buffer)
        if match:
            split_point = match.end()
            sentence = self._buffer[:split_point].strip()
            self._buffer = self._buffer[split_point:]
            if sentence:
                return sentence
        return None

    def flush(self) -> str:
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else ""
