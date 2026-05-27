"""WebSocket endpoint — single bidirectional pipe per client."""

import asyncio
import json
import logging

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


def create_ws_handler(session_manager, event_bridge, state_machine):
    """Return an ASGI WebSocket endpoint wired to the live pipeline."""

    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        session = await session_manager.register(websocket)
        logger.info(f"WS client {session.id} connected")

        try:
            while True:
                message = await websocket.receive()

                if "text" in message:
                    await _handle_json(
                        websocket, json.loads(message["text"]), session,
                        state_machine, session_manager,
                    )
                elif "bytes" in message:
                    _handle_binary(message["bytes"], session, session_manager)
                else:
                    logger.debug(f"Unknown WS message type from {session.id}")

        except WebSocketDisconnect:
            logger.info(f"WS client {session.id} disconnected (clean)")
        except asyncio.CancelledError:
            logger.debug(f"WS handler for {session.id} cancelled")
        except Exception:
            logger.exception(f"WS handler error for {session.id}")
        finally:
            await session_manager.unregister(session.id)

    return ws_endpoint


async def _handle_json(websocket, msg, session, state_machine, session_manager):
    """Dispatch JSON control messages."""
    msg_type = msg.get("type", "")

    if msg_type == "text_input":
        text = (msg.get("text") or "").strip()
        if text:
            asyncio.create_task(state_machine.process_text(text))
            await websocket.send_json({"type": "text_accepted"})

    elif msg_type == "get_history":
        turns = state_machine.memory_manager.buffer.get_context()
        await websocket.send_json({"type": "history", "turns": turns})

    elif msg_type == "get_memory":
        facts = state_machine.memory_manager.store.get_all_facts()
        await websocket.send_json({"type": "memory", "facts": facts})

    elif msg_type == "get_config":
        c = state_machine.config
        await websocket.send_json({
            "type": "config",
            "settings": {
                "mode": c.mode,
                "vad": {
                    "silero_threshold": c.vad.silero_threshold,
                    "speech_start_frames": c.vad.speech_start_frames,
                    "speech_end_frames": c.vad.speech_end_frames,
                },
                "tts": {"voice": c.tts.kokoro.voice},
                "memory": {"max_turns": c.memory.max_turns},
                "tools": {"enabled": c.tools.enabled},
                "wake_word": {"threshold": c.wake_word.threshold},
            },
        })

    elif msg_type == "voice_start":
        accepted = await session_manager.activate_voice(session.id)
        if accepted:
            await websocket.send_json({"type": "voice_accepted"})
        else:
            await websocket.send_json({
                "type": "voice_rejected", "reason": "session_not_found",
            })

    elif msg_type == "voice_stop":
        await session_manager.deactivate_voice(session.id)

    elif msg_type == "mute_local":
        await session_manager.set_local_muted(True)

    elif msg_type == "unmute_local":
        await session_manager.set_local_muted(False)

    else:
        logger.debug(f"Unknown message type from {session.id}: {msg_type}")


def _handle_binary(data: bytes, session, session_manager):
    """Dispatch binary audio frames."""
    if len(data) <= 1:
        return
    marker = data[0]
    pcm_bytes = data[1:]

    if marker == 0x00:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        session_manager.route_mic_audio(session.id, samples)

    elif marker == 0x01:
        logger.debug(f"Unexpected TTS marker from client {session.id}")

    else:
        logger.debug(f"Unknown binary marker 0x{marker:02x} from {session.id}")
