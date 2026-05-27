import asyncio
import signal
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from daisy.utils.config_loader import load_config
from daisy.core.event_bus import EventBus
from daisy.core.state_machine import DaisyStateMachine
from daisy.wake_word.detector import WakeWordDetector
from daisy.audio.input_stream import LocalAudioSource
from daisy.audio.output_stream import LocalAudioSink
from daisy.vad.silero_vad import SileroVAD
from daisy.stt.faster_whisper_stt import FasterWhisperSTT
from daisy.llm.groq_client import GroqClient
from daisy.memory.memory_manager import MemoryManager
from daisy.tts.kokoro_tts import KokoroTTS
from daisy.tools.task_tracker import TaskTracker
from daisy.tools.announcement_queue import AnnouncementQueue
from daisy.tools.tool_registry import build_handlers, TOOL_SCHEMAS
from daisy.api.server import create_app, run_api_server
from daisy.api.session_manager import SessionManager
from daisy.api.event_bridge import EventBridge

# Configure basic logging to see state transitions clearly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stderr)

async def main():
    load_dotenv()
    config = load_config("config.yaml")
    
    # --- Core Infrastructure ---
    event_bus = EventBus()
    
    # --- Pipeline Components ---
    audio_source = LocalAudioSource(config)
    local_sink = LocalAudioSink(config)
    audio_sinks = [local_sink]
    vad = SileroVAD(config)
    stt = FasterWhisperSTT(config)
    llm = GroqClient(config)
    tts = KokoroTTS(config)
    wake_word_detector = WakeWordDetector(config, event_bus)

    # --- Memory System ---
    memory_manager = MemoryManager(config)

    # --- Tool System ---
    task_tracker = TaskTracker() if config.tools.enabled else None
    announcement_queue = AnnouncementQueue() if config.tools.enabled else None
    tool_handlers = build_handlers(config, task_tracker, announcement_queue, llm) if config.tools.enabled else None
    tool_schemas = TOOL_SCHEMAS if config.tools.enabled else None

    # --- API Layer (created before state machine so event bridge can be injected) ---
    session_manager = SessionManager()
    event_bridge = EventBridge(event_bus, session_manager)

    # --- Initialize State Machine ---
    state_machine = DaisyStateMachine(
        config=config,
        event_bus=event_bus,
        audio_source=audio_source,
        audio_sinks=audio_sinks,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        wake_word_detector=wake_word_detector,
        memory_manager=memory_manager,
        task_tracker=task_tracker,
        announcement_queue=announcement_queue,
        tool_handlers=tool_handlers,
        tool_schemas=tool_schemas,
        event_bridge=event_bridge,
    )

    await audio_source.start()
    await local_sink.start()

    # Warm up all ML models concurrently so first inference doesn't pay cold-start
    logging.getLogger(__name__).info("Warming up models...")
    await asyncio.gather(
        vad.warmup(),
        stt.warmup(),
        tts.warmup(),
    )
    logging.getLogger(__name__).info("All models loaded.")

    # --- Graceful Shutdown Setup ---
    shutdown_event = asyncio.Event()

    def signal_handler():
        print("\nShutting down...", file=sys.stderr)
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    print("D.A.I.S.Y. v2 ready.", file=sys.stderr)

    # Wire event bridge to state machine callbacks
    state_machine.set_state_change_callback(event_bridge.on_state_enter)

    # --- Start API Server ---
    app = create_app(
        state_machine=state_machine,
        memory_manager=memory_manager,
        config=config,
        session_manager=session_manager,
        event_bridge=event_bridge,
    )
    api_task = asyncio.create_task(
        run_api_server(app, config.api.host, config.api.port)
    )

    # Boot the state machine (transitions init -> idle, starting wake word listening)
    await state_machine.boot()

    try:
        await shutdown_event.wait()
    finally:
        api_task.cancel()
        await state_machine.shutdown()
        wake_word_detector.stop()
        for sink in audio_sinks:
            sink.stop()
        await audio_source.stop()
        print("D.A.I.S.Y. stopped.", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
