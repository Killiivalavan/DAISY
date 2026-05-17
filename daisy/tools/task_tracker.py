import asyncio
import uuid
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TaskRecord:
    task_id: str
    description: str
    status: str  # running | done | failed | cancelled
    created_at: float
    elapsed: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    coro: Optional[asyncio.Task] = None
    notify_on_complete: bool = False


class TaskTracker:
    def __init__(self):
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()
        self._announce_callback = None

    def set_announce_callback(self, callback):
        self._announce_callback = callback

    async def create_task(
        self,
        description: str,
        coro,
        notify_on_complete: bool = False,
    ) -> str:
        task_id = uuid.uuid4().hex[:12]
        record = TaskRecord(
            task_id=task_id,
            description=description,
            status="running",
            created_at=time.monotonic(),
            notify_on_complete=notify_on_complete,
        )
        async def _wrapped():
            try:
                result = await coro
                record.status = "done"
                record.elapsed = time.monotonic() - record.created_at
                record.result = result
                logger.info(f"[Task] {description} completed in {record.elapsed:.1f}s")
                if record.notify_on_complete and self._announce_callback:
                    await self._announce_callback(task_id, description, result)
            except asyncio.CancelledError:
                record.status = "cancelled"
                record.elapsed = time.monotonic() - record.created_at
                logger.info(f"[Task] {description} cancelled")
            except Exception as e:
                record.status = "failed"
                record.elapsed = time.monotonic() - record.created_at
                record.error = str(e)
                logger.error(f"[Task] {description} failed: {e}")

        async with self._lock:
            record.coro = asyncio.create_task(_wrapped())
            self._tasks[task_id] = record

        return task_id

    async def get_task(self, task_id: str) -> Optional[TaskRecord]:
        async with self._lock:
            return self._tasks.get(task_id)

    async def list_tasks(self, limit: int = 10) -> list[TaskRecord]:
        async with self._lock:
            sorted_tasks = sorted(
                self._tasks.values(),
                key=lambda t: t.created_at,
                reverse=True,
            )
            return sorted_tasks[:limit]

    async def cancel_task(self, task_id: str) -> bool:
        async with self._lock:
            record = self._tasks.get(task_id)
            if not record or record.status != "running":
                return False
            record.coro.cancel()
            return True
