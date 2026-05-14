import asyncio
import logging
from typing import Callable, Any, Dict, List

logger = logging.getLogger(__name__)

class EventBus:
    """
    A simple pub/sub event bus for asyncio coroutines.
    Allows decoupling components by publishing and subscribing to events.
    """
    def __init__(self):
        # Maps event names to a list of subscriber callbacks
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable):
        """
        Register an async callback function to an event type.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.debug(f"[EventBus] Subscribed {callback.__name__} to '{event_type}'")

    async def publish(self, event_type: str, data: Any = None):
        """
        Emit an event and trigger all registered subscribers concurrently.
        """
        if event_type not in self._subscribers:
            logger.debug(f"[EventBus] Event '{event_type}' fired with no subscribers.")
            return

        logger.debug(f"[EventBus] Publishing '{event_type}' to {len(self._subscribers[event_type])} subscribers")

        # Create tasks for all subscribers and run them in parallel
        tasks = []
        for callback in self._subscribers[event_type]:
            # Provide data if the callback expects it, otherwise call without
            import inspect
            sig = inspect.signature(callback)
            if len(sig.parameters) > 0:
                tasks.append(asyncio.create_task(callback(data)))
            else:
                tasks.append(asyncio.create_task(callback()))

        if tasks:
            # We use return_exceptions=True so one failing subscriber doesn't crash others
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"[EventBus] Error in subscriber for '{event_type}': {result}")
