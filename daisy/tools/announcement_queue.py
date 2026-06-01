import asyncio
import logging

logger = logging.getLogger(__name__)


class AnnouncementQueue:
    def __init__(self):
        self._queue = asyncio.Queue()

    async def push(self, announcement: dict):
        await self._queue.put(announcement)
        logger.info(f"[Announce] Queued: {announcement.get('summary', '')[:60]}")

    async def pop(self) -> dict:
        return await self._queue.get()

    @property
    def has_pending(self) -> bool:
        return not self._queue.empty()
