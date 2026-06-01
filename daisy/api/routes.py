import asyncio
import logging

from fastapi import APIRouter, HTTPException

from daisy.utils.config_loader import serialize_config_for_client

logger = logging.getLogger(__name__)


def create_router(state_machine, memory_manager, config, session_manager) -> APIRouter:
    """Build REST endpoints backed by live pipeline objects."""
    router = APIRouter(prefix="/api")

    @router.get("/status")
    async def get_status():
        state = state_machine.current_state.id if state_machine.current_state else "init"
        active_clients = session_manager.client_count if session_manager else 0
        return {
            "state": state,
            "mode": config.mode,
            "tools_enabled": config.tools.enabled,
            "active_clients": active_clients,
        }

    @router.post("/message")
    async def send_message(body: dict):
        text = (body.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text required")
        asyncio.create_task(state_machine.process_text(text))
        return {"accepted": True}

    @router.get("/history")
    async def get_history():
        turns = memory_manager.buffer.get_context()
        return {"turns": turns}

    @router.get("/memory")
    async def get_memory():
        facts = await memory_manager.store.get_all_facts()
        return {"facts": facts}

    @router.delete("/memory/{key}")
    async def delete_fact(key: str):
        deleted = await memory_manager.store.delete_fact(key)
        if not deleted:
            raise HTTPException(status_code=404, detail="fact not found")
        return {"deleted": True}

    @router.get("/config")
    async def get_config():
        return serialize_config_for_client(config)

    @router.patch("/config")
    async def update_config(body: dict):
        """Update safe configuration keys in memory and persist to YAML."""
        import yaml
        from pathlib import Path

        changed = False

        if "mode" in body and body["mode"] in ("wake_word", "always_on", "push_to_talk"):
            config.mode = body["mode"]
            changed = True

        if "vad" in body and isinstance(body["vad"], dict):
            vad = body["vad"]
            if "silero_threshold" in vad:
                config.vad.silero_threshold = float(vad["silero_threshold"])
                changed = True
            if "speech_start_frames" in vad:
                config.vad.speech_start_frames = int(vad["speech_start_frames"])
                changed = True
            if "speech_end_frames" in vad:
                config.vad.speech_end_frames = int(vad["speech_end_frames"])
                changed = True

        if "tts" in body and isinstance(body["tts"], dict):
            if "voice" in body["tts"]:
                config.tts.kokoro.voice = body["tts"]["voice"]
                changed = True

        if "wake_word" in body and isinstance(body["wake_word"], dict):
            if "threshold" in body["wake_word"]:
                config.wake_word.threshold = float(body["wake_word"]["threshold"])
                changed = True

        if "memory" in body and isinstance(body["memory"], dict):
            if "max_turns" in body["memory"]:
                config.memory.max_turns = int(body["memory"]["max_turns"])
                changed = True

        if "tools" in body and isinstance(body["tools"], dict):
            if "enabled" in body["tools"]:
                config.tools.enabled = bool(body["tools"]["enabled"])
                changed = True

        if changed:
            # Persist to YAML
            try:
                cfg_path = Path("config.yaml")
                raw = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
                # Merge changes into the raw dict (simple top-level keys only)
                raw["mode"] = config.mode
                raw.setdefault("vad", {})["silero_threshold"] = config.vad.silero_threshold
                raw.setdefault("vad", {})["speech_start_frames"] = config.vad.speech_start_frames
                raw.setdefault("vad", {})["speech_end_frames"] = config.vad.speech_end_frames
                raw.setdefault("tts", {}).setdefault("kokoro", {})["voice"] = config.tts.kokoro.voice
                raw.setdefault("wake_word", {})["threshold"] = config.wake_word.threshold
                raw.setdefault("memory", {})["max_turns"] = config.memory.max_turns
                raw.setdefault("tools", {})["enabled"] = config.tools.enabled
                cfg_path.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False))
                logger.info("Config updated and persisted to config.yaml")
            except Exception as e:
                logger.error(f"Failed to persist config: {e}")

        return {"updated": True, "changed": changed}

    return router
