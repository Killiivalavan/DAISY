import asyncio
import json
import logging
from statemachine import StateMachine, State

from daisy.llm.sentence_splitter import SentenceSplitter
from daisy.llm.client import ToolCall
from daisy.tools.task_tracker import TaskTracker
from daisy.tools.announcement_queue import AnnouncementQueue

logger = logging.getLogger(__name__)


def _compute_envelope(audio, sample_rate: int, window_ms: int = 50):
    """Compute RMS amplitude envelope from PCM audio.

    Returns (envelope, duration_s) where envelope is a list of floats
    0-1 representing amplitude over ~50ms windows, and duration_s is
    the total audio length in seconds.
    """
    import numpy as np

    window_samples = max(1, int(sample_rate * window_ms / 1000))
    num_windows = max(1, len(audio) // window_samples)
    envelope = []

    for i in range(num_windows):
        start = i * window_samples
        end = start + window_samples
        window = audio[start:end].astype(np.float32)
        rms = float(np.sqrt(np.mean(window ** 2)))
        # Kokoro outputs float32 in [-1, 1]; typical speech RMS is ~0.05-0.25
        normalized = round(min(1.0, rms / 0.2), 3)
        envelope.append(normalized)

    duration_s = len(audio) / sample_rate
    return envelope, duration_s


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
    announce = idle.to(processing) | listening.to(processing)
    response_ready = processing.to(speaking)
    turn_complete = speaking.to(listening) | processing.to(listening)
    timed_out = listening.to(idle)

    def __init__(
        self, config, event_bus, audio_source, audio_sinks, vad, stt, llm_router, tts,
        wake_word_detector, memory_manager,
        task_tracker=None, announcement_queue=None,
        tool_handlers=None, tool_schemas=None,
        event_bridge=None,
    ):
        self.config = config
        self.event_bus = event_bus
        self.audio_source = audio_source
        self.audio_sinks = audio_sinks  # list[AudioSink] — local + remote clients
        self.vad = vad
        self.stt = stt
        self.llm_router = llm_router
        self.tts = tts
        self.wake_word_detector = wake_word_detector
        self.memory_manager = memory_manager
        self.task_tracker = task_tracker
        self.announcement_queue = announcement_queue
        self.tool_handlers = tool_handlers
        self.tool_schemas = tool_schemas
        self._event_bridge = event_bridge

        self.current_audio_buffer = None
        self.current_announcement = None
        self.sentence_queue = None
        self._last_response = None
        self._on_state_change = None  # callback(state_name) for event bridge

        # Subscribe to global events
        self.event_bus.subscribe("WAKE", self.on_wake_event)

        # Wire task tracker completion to announcement queue
        if self.task_tracker and self.announcement_queue:
            self.task_tracker.set_announce_callback(self._on_task_completed)

        # Must call super last
        super().__init__()

    def set_state_change_callback(self, callback):
        """Register a callback(state_name) called on every state entry.

        Used by the EventBridge to broadcast state to WebSocket clients.
        """
        self._on_state_change = callback

    def set_audio_source(self, source):
        """Swap the active audio input source (local mic → remote client or vice versa)."""
        self.audio_source = source

    def add_audio_sink(self, sink):
        """Add an audio output sink (e.g., new remote client)."""
        if sink not in self.audio_sinks:
            self.audio_sinks.append(sink)

    def remove_audio_sink(self, sink):
        """Remove an audio output sink (e.g., client disconnected)."""
        if sink in self.audio_sinks:
            self.audio_sinks.remove(sink)

    async def process_text(self, text: str):
        """Entry point for text input from API clients.

        Bypasses wake word + VAD + STT. Injects text directly into processing.
        """
        self._injected_text = text
        self.current_audio_buffer = None
        if self.idle.is_active:
            await self.announce()
        elif self.listening.is_active:
            await self.speech_detected()

    async def _on_task_completed(self, task_id: str, description: str, result):
        await self.announcement_queue.push({
            "task_id": task_id,
            "summary": f"'{description}' completed.",
            "type": "task_complete",
            "priority": 2,
        })

    async def shutdown(self):
        self.memory_manager.end_session()
        for name in ("_speaking_task", "_processing_task", "_listening_task", "_llm_task", "_summarize_task"):
            task = getattr(self, name, None)
            if task is not None and not task.done():
                task.cancel()

    async def on_wake_event(self, data=None):
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

    async def _safe_announce(self):
        try:
            await self.announce()
        except Exception as e:
            logger.error(f"Error during announce transition: {e}")

    # --- IDLE STATE ---
    async def on_enter_idle(self):
        logger.info("[State] Entering IDLE")

        if self._on_state_change:
            self._on_state_change("idle")

        if self.announcement_queue and self.announcement_queue.has_pending:
            logger.info("[State] Announcement pending, skipping wake word")
            await asyncio.sleep(2)
            asyncio.create_task(self._safe_announce())
            return

        self._summarize_task = asyncio.create_task(
            self.memory_manager.summarize_session(self.llm_router)
        )
        if not self.wake_word_detector.is_listening:
            self.wake_word_detector.start(self.audio_source)

    async def on_exit_idle(self):
        logger.info("[State] Exiting IDLE")
        self.wake_word_detector.stop()
        task = getattr(self, '_summarize_task', None)
        if task and not task.done():
            task.cancel()

    # --- LISTENING STATE ---
    async def on_enter_listening(self):
        logger.info("[State] Entering LISTENING")

        if self._on_state_change:
            self._on_state_change("listening")

        response = getattr(self, '_last_response', None)
        if response:
            self.memory_manager.record_turn("assistant", response)
            self._last_response = None

        if self.announcement_queue and self.announcement_queue.has_pending:
            logger.info("[State] Announcement pending, returning to IDLE")
            asyncio.create_task(self._safe_timed_out())
            return

        print("  [system] *beep* (Mic is hot)", file=__import__("sys").stderr)

        self._listening_task = asyncio.create_task(self._do_listen())

    async def on_exit_listening(self):
        if hasattr(self, '_listening_task') and not self._listening_task.done():
            self._listening_task.cancel()

    async def _do_listen(self):
        try:
            audio_buffer = await self.vad.listen(self.audio_source, timeout=self.config.pipeline.listening_timeout)

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
        if self._on_state_change:
            self._on_state_change("processing")
        if self.announcement_queue and self.announcement_queue.has_pending:
            self.current_announcement = await self.announcement_queue.pop()
            self._processing_task = asyncio.create_task(self._do_announce())
        else:
            self._processing_task = asyncio.create_task(self._do_process())

    async def on_exit_processing(self):
        if hasattr(self, '_processing_task') and not self._processing_task.done():
            self._processing_task.cancel()

    async def _do_process(self):
        try:
            text = getattr(self, '_injected_text', None)
            if text is not None:
                self._injected_text = None
                logger.info(f"[State] Text input: '{text}'")
            elif self.current_audio_buffer is not None:
                text = await self.stt.transcribe(self.current_audio_buffer)
                logger.info(f"[State] Transcribed: '{text}'")
            else:
                asyncio.create_task(self._safe_turn_complete())
                return

            if not text:
                logger.info("[State] Empty text, returning to IDLE")
                asyncio.create_task(self._safe_turn_complete())
                return

            # Broadcast transcript to frontend clients
            if self._event_bridge:
                asyncio.create_task(self._event_bridge.broadcast_transcript(text))

            if self.current_audio_buffer is not None:
                print(f"You: {text}")

            self.memory_manager.record_turn("user", text)
            messages = self.memory_manager.build_context(text)

            # Phase 1: Tool detection loop (non-streaming)
            tools = self.tool_schemas if self.tool_handlers else None
            tool_rounds = 0
            while tools and tool_rounds < 5:
                response = await self.llm_router.complete("main_agent", messages, tools=tools)
                if not response.tool_calls:
                    break

                for tool_call in response.tool_calls:
                    func_name = tool_call.name
                    handler = self.tool_handlers.get(func_name)
                    if not handler:
                        logger.warning(f"[State] No handler for tool: {func_name}")
                        continue

                    try:
                        args = json.loads(tool_call.arguments)
                        if args is None:
                            args = {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                    logger.info(f"[State] Tool call: {func_name}({args})")
                    print(f"  [Tool] {func_name}({args})", file=__import__("sys").stderr)

                    result = await handler(**args)
                    logger.info(f"[State] Tool result: {str(result)[:200]}")

                    assistant_msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                },
                            }
                            for tc in response.tool_calls
                        ],
                    }
                    messages.append(assistant_msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)[:2000],
                    })

                tool_rounds += 1

            # Phase 2: Stream final response
            self.sentence_queue = asyncio.Queue()
            splitter = SentenceSplitter()

            async def llm_worker():
                full_response = []
                try:
                    async for token in self.llm_router.stream_tokens("main_agent", messages):
                        full_response.append(token)
                        sentence = splitter.process_token(token)
                        if sentence:
                            await self.sentence_queue.put(sentence)
                            if self._event_bridge:
                                await self._event_bridge.broadcast_sentence(sentence)
                    remaining = splitter.flush()
                    if remaining:
                        await self.sentence_queue.put(remaining)
                        if self._event_bridge:
                            await self._event_bridge.broadcast_sentence(remaining)
                except Exception as e:
                    logger.error(f"LLM Worker error: {e}")
                    if self._event_bridge:
                        asyncio.create_task(
                            self._event_bridge.broadcast_error(f"LLM error: {e}")
                        )
                finally:
                    await self.sentence_queue.put(None)
                    self._last_response = "".join(full_response)
                    if self._event_bridge and self._last_response:
                        asyncio.create_task(
                            self._event_bridge.broadcast_response_complete(self._last_response)
                        )

            self._llm_task = asyncio.create_task(llm_worker())

            asyncio.create_task(self._safe_response_ready())

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[State] Error in PROCESSING: {e}")
            asyncio.create_task(self._safe_turn_complete())

    async def _do_announce(self):
        try:
            announcement = self.current_announcement
            if not announcement:
                asyncio.create_task(self._safe_turn_complete())
                return

            summary = announcement.get("summary", "Something completed.")
            prompt = (
                f"You need to proactively inform the user about a completed task. "
                f"Be natural and brief. The user is not in the middle of a conversation "
                f"with you, so greet them naturally. Here is the information: {summary}"
            )

            self.memory_manager.record_turn("user", f"[system notification: {summary}]")
            messages = self.memory_manager.build_context(prompt)

            self.sentence_queue = asyncio.Queue()
            splitter = SentenceSplitter()

            async def llm_worker():
                full_response = []
                try:
                    async for token in self.llm_router.stream_tokens("announcement", messages):
                        full_response.append(token)
                        sentence = splitter.process_token(token)
                        if sentence:
                            await self.sentence_queue.put(sentence)
                            if self._event_bridge:
                                await self._event_bridge.broadcast_sentence(sentence)
                    remaining = splitter.flush()
                    if remaining:
                        await self.sentence_queue.put(remaining)
                        if self._event_bridge:
                            await self._event_bridge.broadcast_sentence(remaining)
                except Exception as e:
                    logger.error(f"Announce LLM Worker error: {e}")
                    if self._event_bridge:
                        asyncio.create_task(
                            self._event_bridge.broadcast_error(f"LLM error: {e}")
                        )
                finally:
                    await self.sentence_queue.put(None)
                    self._last_response = "".join(full_response)
                    if self._event_bridge and self._last_response:
                        asyncio.create_task(
                            self._event_bridge.broadcast_response_complete(self._last_response)
                        )

            self._llm_task = asyncio.create_task(llm_worker())
            asyncio.create_task(self._safe_response_ready())

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[State] Error in ANNOUNCE: {e}")
            asyncio.create_task(self._safe_turn_complete())

    # --- SPEAKING STATE ---
    async def on_enter_speaking(self):
        logger.info("[State] Entering SPEAKING")
        if self._on_state_change:
            self._on_state_change("speaking")
        self._speaking_task = asyncio.create_task(self._do_speak())

    async def on_exit_speaking(self):
        if hasattr(self, '_speaking_task') and not self._speaking_task.done():
            self._speaking_task.cancel()
        if hasattr(self, '_llm_task') and not self._llm_task.done():
            self._llm_task.cancel()
        for sink in self.audio_sinks:
            sink.clear()
        if self.sentence_queue is not None:
            while not self.sentence_queue.empty():
                try:
                    self.sentence_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def _do_speak(self):
        try:
            async def tts_worker():
                while True:
                    sentence = await self.sentence_queue.get()
                    if sentence is None:
                        break
                    audio = await self.tts.synthesize(sentence)
                    for sink in self.audio_sinks:
                        sink.play(audio)
                    if self._event_bridge and len(audio) > 0:
                        try:
                            import numpy as np
                            sample_rate = getattr(
                                getattr(self.config.tts, 'kokoro', None), 'sample_rate', 24000
                            )
                            envelope, duration = _compute_envelope(audio, sample_rate)
                            asyncio.create_task(
                                self._event_bridge.broadcast_audio_envelope(envelope, duration)
                            )
                        except Exception:
                            pass  # Envelope is cosmetic — never break the pipeline

            tts_task = asyncio.create_task(tts_worker())

            if hasattr(self, '_llm_task'):
                await self._llm_task
            await tts_task
            for sink in self.audio_sinks:
                await sink.wait_until_done()

            logger.info("[State] Finished speaking.")
            asyncio.create_task(self._safe_turn_complete())
        except asyncio.CancelledError:
            logger.info("[State] Speaking interrupted.")
            tts_task.cancel()
        except Exception as e:
            logger.error(f"[State] Error in SPEAKING: {e}")
            asyncio.create_task(self._safe_turn_complete())
