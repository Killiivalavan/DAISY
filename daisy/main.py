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
from daisy.audio.input_stream import AudioInputStream
from daisy.audio.output_stream import AudioOutputStream
from daisy.vad.silero_vad import SileroVAD
from daisy.stt.faster_whisper_stt import FasterWhisperSTT
from daisy.llm.groq_client import GroqClient
from daisy.tts.kokoro_tts import KokoroTTS

# Configure basic logging to see state transitions clearly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stderr)

async def main():
    load_dotenv()
    config = load_config("config.yaml")
    
    # --- Core Infrastructure ---
    event_bus = EventBus()
    
    # --- Pipeline Components ---
    audio_in = AudioInputStream(config)
    audio_out = AudioOutputStream(config)
    vad = SileroVAD(config)
    stt = FasterWhisperSTT(config)
    llm = GroqClient(config)
    tts = KokoroTTS(config)
    wake_word_detector = WakeWordDetector(config, event_bus)

    # --- Initialize State Machine ---
    state_machine = DaisyStateMachine(
        config=config,
        event_bus=event_bus,
        audio_in=audio_in,
        audio_out=audio_out,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        wake_word_detector=wake_word_detector
    )

    await audio_in.start()
    await audio_out.start()

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
    
    # Boot the state machine (transitions init -> idle, starting wake word listening)
    await state_machine.boot()
    
    try:
        # The main task now simply waits for the shutdown signal.
        # Everything else is driven by the background tasks and event bus.
        await shutdown_event.wait()
    finally:
        await state_machine.shutdown()
        wake_word_detector.stop()
        audio_out.stop()
        await audio_in.stop()
        print("D.A.I.S.Y. stopped.", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
