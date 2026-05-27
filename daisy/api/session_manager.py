"""Session manager — tracks connected WebSocket clients and broadcasts events."""

import asyncio
import logging
import time
import uuid

logger = logging.getLogger(__name__)


class ClientSession:
    """Represents one connected frontend client."""

    def __init__(self, ws):
        self.id = uuid.uuid4().hex[:8]
        self.ws = ws
        self.is_voice_active = False
        self.is_muted = False
        self.connected_at = time.monotonic()
        self.last_activity = time.monotonic()

    async def send(self, message: dict):
        """Send a JSON message to this client. Handles disconnects gracefully."""
        try:
            await self.ws.send_json(message)
            self.last_activity = time.monotonic()
        except Exception:
            logger.debug(f"Client {self.id}: send failed (disconnected)")


class SessionManager:
    """Tracks connected clients and routes audio/events."""

    def __init__(self):
        self._clients: dict[str, ClientSession] = {}
        self._lock = asyncio.Lock()

    # --- Client lifecycle ---

    async def register(self, ws) -> ClientSession:
        session = ClientSession(ws)
        async with self._lock:
            self._clients[session.id] = session
        logger.info(f"Client {session.id} connected ({self.client_count} total)")
        return session

    async def unregister(self, session_id: str):
        async with self._lock:
            self._clients.pop(session_id, None)
        logger.info(f"Client {session_id} disconnected ({self.client_count} total)")

    # --- Broadcasting ---

    async def broadcast_json(self, message: dict):
        """Send a JSON message to every connected client.

        Dead clients are automatically cleaned up.
        """
        dead = []
        async with self._lock:
            sessions = list(self._clients.items())

        for sid, session in sessions:
            try:
                await session.ws.send_json(message)
            except Exception:
                dead.append(sid)

        for sid in dead:
            await self.unregister(sid)

    async def send_to(self, session_id: str, message: dict) -> bool:
        """Send a JSON message to a specific client. Returns True if delivered."""
        async with self._lock:
            session = self._clients.get(session_id)
        if session is None:
            return False
        try:
            await session.ws.send_json(message)
            return True
        except Exception:
            await self.unregister(session_id)
            return False

    # --- Properties ---

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def get_session(self, session_id: str) -> ClientSession | None:
        return self._clients.get(session_id)
