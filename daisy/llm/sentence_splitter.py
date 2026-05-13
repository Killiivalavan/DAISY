import re


class SentenceSplitter:
    def __init__(self):
        self._buffer = ""

    def process_token(self, token: str):
        self._buffer += token
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
