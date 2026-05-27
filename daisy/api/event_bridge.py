"""Event bridge — relays internal pipeline events to WebSocket clients."""

import logging

logger = logging.getLogger(__name__)

# Orb animation parameters for each state.
# The frontend lerps smoothly between these when state changes.
ORB_STATE_PARAMS = {
    "idle": {
        "intensity": 0.15,
        "speed": 0.3,
        "color": [0.04, 0.52, 1.0],
        "bloom_strength": 0.5,
        "pulse_period": 3.0,
        "particle_count": 50,
        "particle_speed": 0.2,
    },
    "listening": {
        "intensity": 0.7,
        "speed": 1.0,
        "color": [0.0, 0.8, 1.0],
        "bloom_strength": 0.8,
        "pulse_period": 0.8,
        "particle_count": 150,
        "particle_speed": 0.8,
    },
    "processing": {
        "intensity": 0.9,
        "speed": 1.5,
        "color": [0.0, 0.7, 0.85],
        "bloom_strength": 1.0,
        "pulse_period": 0.5,
        "particle_count": 200,
        "particle_speed": 1.5,
    },
    "speaking": {
        "intensity": 1.0,
        "speed": 1.8,
        "color": [0.0, 0.85, 0.9],
        "bloom_strength": 1.5,
        "pulse_period": 0.3,
        "particle_count": 200,
        "particle_speed": 2.0,
    },
}


class EventBridge:
    """Subscribes to the internal EventBus and broadcasts pipeline events
    to all connected WebSocket clients via the SessionManager.

    Also provides callback hooks for the state machine to call directly
    when transcripts, sentences, and state changes occur.
    """

    def __init__(self, event_bus, session_manager):
        self._event_bus = event_bus
        self._session_manager = session_manager

        # Subscribe to pipeline events
        event_bus.subscribe("WAKE", self._on_wake)
        event_bus.subscribe("INTERRUPT", self._on_interrupt)

    # --- EventBus subscribers ---

    async def _on_wake(self, data=None):
        await self.broadcast_state("listening")

    async def _on_interrupt(self, data=None):
        await self.broadcast_state("listening")

    # --- State machine callback ---

    def on_state_enter(self, state_name: str):
        """Called by state machine on every state transition."""
        import asyncio

        asyncio.create_task(self.broadcast_state(state_name))

    # --- Broadcast helpers ---

    async def broadcast_state(self, state_name: str):
        params = ORB_STATE_PARAMS.get(state_name, ORB_STATE_PARAMS["idle"])
        await self._session_manager.broadcast_json({
            "type": "state",
            "state": state_name,
            **params,
        })

    async def broadcast_transcript(self, text: str, partial: bool = False, final: bool = True):
        await self._session_manager.broadcast_json({
            "type": "transcript",
            "text": text,
            "partial": partial,
            "final": final,
        })

    async def broadcast_sentence(self, text: str):
        await self._session_manager.broadcast_json({
            "type": "sentence",
            "text": text,
        })

    async def broadcast_response_complete(self, full_text: str):
        await self._session_manager.broadcast_json({
            "type": "response_complete",
            "full_text": full_text,
        })

    async def broadcast_error(self, message: str):
        await self._session_manager.broadcast_json({
            "type": "error",
            "message": message,
        })

    async def broadcast_audio_envelope(self, envelope: list[float], duration_s: float):
        """Send the amplitude envelope for the current TTS sentence."""
        await self._session_manager.broadcast_json({
            "type": "audio_amplitude",
            "envelope": envelope,
            "duration_s": round(duration_s, 3),
        })
