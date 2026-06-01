import asyncio
import logging
import re
from pathlib import Path

from daisy.memory.conversation_buffer import ConversationBuffer
from daisy.memory.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

_REMEMBER_PATTERN = re.compile(
    r"remember\s+"
    r"(?:that\s+)?"
    r"(?:my\s+)?"
    r"(.+?)"
    r"\s+(?:is|are|was|were)\s+"
    r"(.+)",
    re.IGNORECASE,
)

# Broader pattern: catches "save/note/keep in mind/don't forget", leading
# words ("please", "also"), commas after the trigger word, and more copula
# verbs ("equals", "means").
_REMEMBER_BROAD = re.compile(
    r"(?:please\s+|also\s+|and\s+|can\s+you\s+|could\s+you\s+)?"
    r"(?:remember|save|note|keep\s+in\s+mind|don'?t\s+forget)"
    r"(?:\s+(?:that|this|for\s+later|down))?[,\s:]+"
    r"(?:that\s+)?(?:my\s+)?"
    r"(.+?)"
    r"\s+(?:is|are|was|were|equals|means)\s+"
    r"(.+)",
    re.IGNORECASE,
)

_REMEMBER_THIS = re.compile(r"remember\s+this", re.IGNORECASE)


class MemoryManager:
    def __init__(self, config):
        self.buffer = ConversationBuffer(config.memory.max_turns)
        self.store = SQLiteStore(config.memory.db_path)
        self._config = config.memory
        # Use sync internal method during init (asyncio event loop not running yet)
        self._session_id = self.store._start_session_sync()

    async def record_turn(self, role: str, content: str):
        self.buffer.add(role, content)
        if role == "user" and content:
            await self._parse_remember_command(content)

    async def _parse_remember_command(self, text: str):
        if not text or not text.strip():
            return

        # Try patterns in order: narrow, broad, then "remember this"
        m = _REMEMBER_PATTERN.search(text)
        if not m:
            m = _REMEMBER_BROAD.search(text)
        if m:
            key = m.group(1).strip()
            value = m.group(2).strip()
            if key and value:
                await self.store.store_fact(key.lower(), value)
                logger.info(f"[Memory] Stored fact: {key} = {value}")
            return

        m = _REMEMBER_THIS.search(text)
        if m and self.buffer.message_count >= 2:
            # The "remember this" message is the last one in the buffer.
            # Look at the preceding messages for what to remember.
            hist = self.buffer.get_context()
            preceding = hist[:-1]
            last_user = next(
                (m["content"] for m in reversed(preceding) if m["role"] == "user"), ""
            )
            last_assistant = next(
                (m["content"] for m in reversed(preceding) if m["role"] == "assistant"), ""
            )
            if last_assistant and last_user:
                key = (last_user[:80] if len(last_user) > 80 else last_user)
                value = (last_assistant[:200] if len(last_assistant) > 200 else last_assistant)
                await self.store.store_fact(key, value, category="saved_conversation")
                logger.info(f"[Memory] Stored conversation snippet: {key}...")

    async def build_context(self, user_message: str) -> list[dict]:
        messages = [{"role": "system", "content": await self._load_system_prompt()}]

        if self._config.inject_facts:
            limit = max(self._config.max_facts_to_inject, 0)
            facts = await self.store.get_all_facts(limit=limit)
            if facts:
                fact_lines = []
                for f in facts[:limit]:
                    label = f"{f['key']}: {f['value']}"
                    if f["category"] != "general":
                        label = f"[{f['category']}] {label}"
                    fact_lines.append(label)
                messages.append(
                    {
                        "role": "system",
                        "content": "Known facts:\n" + "\n".join(fact_lines),
                    }
                )

        last_summary = await self.store.get_last_session_summary()
        if last_summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Previous session: {last_summary}",
                }
            )

        messages.extend(self.buffer.get_context())
        messages.append({"role": "user", "content": user_message})
        return messages

    async def _load_system_prompt(self) -> str:
        path = Path("SOUL.md")
        try:
            return (await asyncio.to_thread(path.read_text, encoding="utf-8")).strip()
        except FileNotFoundError:
            return (
                "You are D.A.I.S.Y., a personal AI assistant running on a server "
                "called Andromeda. Address the user as 'Boss'. Be sharp, efficient, "
                "and precise. Lead with short, direct sentences."
            )

    async def summarize_session(self, llm_router):
        context = self.buffer.get_context()
        if len(context) < 2:
            return

        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in context
        )

        summary_messages = [
            {
                "role": "system",
                "content": (
                    "Summarize this conversation in 1-2 sentences, focusing on "
                    "key facts, decisions, and topics discussed."
                ),
            },
            {"role": "user", "content": conversation_text},
        ]

        try:
            parts = []
            async for token in llm_router.stream_tokens("summarizer", summary_messages):
                parts.append(token)

            summary = "".join(parts).strip()
            if summary:
                await self.store.end_session(self._session_id, summary)
                logger.info(f"[Memory] Session summary stored: {summary[:80]}...")
        except asyncio.CancelledError:
            logger.debug("[Memory] Summarization cancelled (user spoke again)")
            raise
        except Exception as e:
            logger.error(f"[Memory] Summarization failed: {e}")

    async def end_session(self, summary: str | None = None):
        await self.store.end_session(self._session_id, summary)
        self.buffer.clear()
