"""Session manager — tracks connected WebSocket clients and routes audio."""

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
        # Created lazily when voice is activated
        self.audio_source = None
        self.audio_sink = None

    async def send(self, message: dict):
        try:
            await self.ws.send_json(message)
            self.last_activity = time.monotonic()
        except Exception:
            logger.debug(f"Client {self.id}: send failed")


class SessionManager:
    """Tracks connected clients and routes audio/events."""

    def __init__(self):
        self._clients: dict[str, ClientSession] = {}
        self._active_voice_id: str | None = None
        self._lock = asyncio.Lock()
        # Set after construction by main.py
        self._state_machine = None
        self._local_audio_source = None
        self._local_muted = False

    def wire(self, state_machine, local_audio_source):
        """Called once at startup to give the manager references to the pipeline."""
        self._state_machine = state_machine
        self._local_audio_source = local_audio_source

    # --- Client lifecycle ---

    async def register(self, ws) -> ClientSession:
        session = ClientSession(ws)
        async with self._lock:
            self._clients[session.id] = session
        logger.info(f"Client {session.id} connected ({self.client_count} total)")
        return session

    async def unregister(self, session_id: str):
        # Deactivate voice if this client was the active voice source
        if self._active_voice_id == session_id:
            await self._deactivate_voice_locked(session_id)

        async with self._lock:
            self._clients.pop(session_id, None)

        logger.info(f"Client {session_id} disconnected ({self.client_count} total)")

    # --- Voice routing ---

    async def activate_voice(self, session_id: str) -> bool:
        """Route pipeline input to this client's mic. Returns True if accepted."""
        from daisy.audio.input_stream import NetworkAudioSource
        from daisy.audio.output_stream import NetworkAudioSink

        session = self._clients.get(session_id)
        if not session:
            return False

        async with self._lock:
            # Kick old voice client
            if self._active_voice_id and self._active_voice_id != session_id:
                old = self._clients.get(self._active_voice_id)
                if old:
                    old.is_voice_active = False
                    await old.send({"type": "voice_rejected", "reason": "another_client"})
                    if old.audio_sink and self._state_machine:
                        self._state_machine.remove_audio_sink(old.audio_sink)

            self._active_voice_id = session_id
            session.is_voice_active = True

        # Create audio source/sink for this session
        if session.audio_source is None:
            session.audio_source = NetworkAudioSource()
        if session.audio_sink is None:
            session.audio_sink = NetworkAudioSink(session.ws)

        # Swap pipeline input to this client's mic
        if self._state_machine:
            self._state_machine.set_audio_source(session.audio_source)
            self._state_machine.add_audio_sink(session.audio_sink)

        logger.info(f"Voice activated for client {session_id}")
        return True

    async def deactivate_voice(self, session_id: str):
        """Stop routing voice from this client."""
        if self._active_voice_id != session_id:
            return
        await self._deactivate_voice_locked(session_id)

    async def _deactivate_voice_locked(self, session_id: str):
        session = self._clients.get(session_id)
        if self._state_machine:
            # Revert to local audio source
            if self._local_audio_source:
                self._state_machine.set_audio_source(self._local_audio_source)
            # Remove network sink
            if session and session.audio_sink:
                self._state_machine.remove_audio_sink(session.audio_sink)

        if session:
            session.is_voice_active = False

        self._active_voice_id = None
        logger.info(f"Voice deactivated for client {session_id}")

    def route_mic_audio(self, session_id: str, pcm_chunk):
        """Push mic audio from a client into the pipeline. Only the active voice
        client's audio is processed."""
        if session_id != self._active_voice_id:
            return
        session = self._clients.get(session_id)
        if session and session.audio_source:
            session.audio_source.push(pcm_chunk)

    # --- Local mute ---

    async def set_local_muted(self, muted: bool):
        self._local_muted = muted
        if self._state_machine and self._local_audio_source:
            # When muted, we keep the source but the local sink is silenced
            # by removing/adding the local sink. Simpler: track it on the sink side.
            pass

    # --- Broadcasting ---

    async def broadcast_json(self, message: dict):
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
