import asyncio
import logging
from statemachine import StateMachine, State

logger = logging.getLogger(__name__)

class DaisyStateMachine(StateMachine):
    # States
    init = State(initial=True)
    idle = State()
    listening = State()
    processing = State()
    speaking = State()

    # Transitions
    boot = init.to(idle)
    wake_up = idle.to(listening) | listening.to(listening) | speaking.to(listening)
    speech_detected = listening.to(processing)
    response_ready = processing.to(speaking)
    turn_complete = speaking.to(listening) | processing.to(listening)
    timed_out = listening.to(idle)

    def __init__(self, config, event_bus, audio_in, audio_out, vad, stt, llm, tts, wake_word_detector):
        self.config = config
        self.event_bus = event_bus
        self.audio_in = audio_in
        self.audio_out = audio_out
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.wake_word_detector = wake_word_detector
        
        self.current_audio_buffer = None
        self.sentence_queue = None
        
        # Subscribe to global events
        self.event_bus.subscribe("WAKE", self.on_wake_event)
        
        # Must call super last
        super().__init__()

    async def on_wake_event(self):
        """Triggered by the event bus when wake word is detected."""
        if not self.listening.is_active:
            # We run the transition in a separate task to avoid deadlocks 
            # if the detector task is cancelled during the transition.
            asyncio.create_task(self._safe_wake_up())

    async def _safe_wake_up(self):
        try:
            await self.wake_up()
        except Exception as e:
            logger.error(f"Error during wake_up transition: {e}")

    async def _safe_speech_detected(self):
        try:
            await self.speech_detected()
        except Exception as e:
            logger.error(f"Error during speech_detected transition: {e}")

    async def _safe_turn_complete(self):
        try:
            await self.turn_complete()
        except Exception as e:
            logger.error(f"Error during turn_complete transition: {e}")

    async def _safe_timed_out(self):
        try:
            await self.timed_out()
        except Exception as e:
            logger.error(f"Error during timed_out transition: {e}")

    async def _safe_response_ready(self):
        try:
            await self.response_ready()
        except Exception as e:
            logger.error(f"Error during response_ready transition: {e}")

    # --- IDLE STATE ---
    async def on_enter_idle(self):
        logger.info("[State] Entering IDLE")
        if not self.wake_word_detector.is_listening:
            self.wake_word_detector.start(self.audio_in)

    async def on_exit_idle(self):
        logger.info("[State] Exiting IDLE")
        self.wake_word_detector.stop()

    # --- LISTENING STATE ---
    async def on_enter_listening(self):
        logger.info("[State] Entering LISTENING")
        
        # Only beep if coming from IDLE (a fresh wake word)
        # statemachine passes the event that triggered the transition. 
        # But a simpler way is to just beep. We'll beep for now to indicate mic is hot.
        print("  [system] *beep* (Mic is hot)", file=__import__("sys").stderr)
        
        # Start a background task for VAD listening to not block the state transition
        self._listening_task = asyncio.create_task(self._do_listen())

    async def on_exit_listening(self):
        if hasattr(self, '_listening_task') and not self._listening_task.done():
            self._listening_task.cancel()

    async def _do_listen(self):
        try:
            # We pass a timeout to the VAD so it doesn't hang forever
            audio_buffer = await self.vad.listen(self.audio_in, timeout=7.0)
            
            if audio_buffer is not None and len(audio_buffer) > 0:
                self.current_audio_buffer = audio_buffer
                asyncio.create_task(self._safe_speech_detected())
            else:
                logger.info("[State] Conversation timed out, returning to IDLE")
                asyncio.create_task(self._safe_timed_out())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[State] Error in LISTENING: {e}")
            asyncio.create_task(self._safe_timed_out())

    # --- PROCESSING STATE ---
    async def on_enter_processing(self):
        logger.info("[State] Entering PROCESSING")
        self._processing_task = asyncio.create_task(self._do_process())

    async def on_exit_processing(self):
        if hasattr(self, '_processing_task') and not self._processing_task.done():
            self._processing_task.cancel()

    async def _do_process(self):
        try:
            if self.current_audio_buffer is None:
                asyncio.create_task(self._safe_turn_complete())
                return

            text = await self.stt.transcribe(self.current_audio_buffer)
            logger.info(f"[State] Transcribed: '{text}'")
            print(f"You: {text}")
            
            if not text:
                logger.info("[State] Empty transcript, returning to IDLE")
                asyncio.create_task(self._safe_turn_complete())
                return
                
            from daisy.llm.sentence_splitter import SentenceSplitter
            self.sentence_queue = asyncio.Queue()
            splitter = SentenceSplitter()
            
            # Create the LLM streaming task
            async def llm_worker():
                try:
                    async for token in self.llm.stream_tokens(text):
                        sentence = splitter.process_token(token)
                        if sentence:
                            await self.sentence_queue.put(sentence)
                    remaining = splitter.flush()
                    if remaining:
                        await self.sentence_queue.put(remaining)
                except Exception as e:
                    logger.error(f"LLM Worker error: {e}")
                finally:
                    await self.sentence_queue.put(None) # EOF

            self._llm_task = asyncio.create_task(llm_worker())
            
            # Transition to SPEAKING state so TTS can start pulling from the queue immediately
            asyncio.create_task(self._safe_response_ready())
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[State] Error in PROCESSING: {e}")
            asyncio.create_task(self._safe_turn_complete())

    # --- SPEAKING STATE ---
    async def on_enter_speaking(self):
        logger.info("[State] Entering SPEAKING")
        self._speaking_task = asyncio.create_task(self._do_speak())

    async def on_exit_speaking(self):
        if hasattr(self, '_speaking_task') and not self._speaking_task.done():
            self._speaking_task.cancel()
        self.audio_out.clear()

    async def _do_speak(self):
        try:
            async def tts_worker():
                while True:
                    sentence = await self.sentence_queue.get()
                    if sentence is None:
                        break
                    audio = await self.tts.synthesize(sentence)
                    self.audio_out.play(audio)

            tts_task = asyncio.create_task(tts_worker())
            
            if hasattr(self, '_llm_task'):
                await self._llm_task
            await tts_task
            await self.audio_out.wait_until_done()
            
            logger.info("[State] Finished speaking.")
            asyncio.create_task(self._safe_turn_complete())
        except asyncio.CancelledError:
            logger.info("[State] Speaking interrupted.")
        except Exception as e:
            logger.error(f"[State] Error in SPEAKING: {e}")
            asyncio.create_task(self._safe_turn_complete())
