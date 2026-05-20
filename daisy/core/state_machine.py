import asyncio
import json
import logging
from statemachine import StateMachine, State

from daisy.llm.sentence_splitter import SentenceSplitter
from daisy.tools.task_tracker import TaskTracker
from daisy.tools.announcement_queue import AnnouncementQueue

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
    announce = idle.to(processing) | listening.to(processing)
    response_ready = processing.to(speaking)
    turn_complete = speaking.to(listening) | processing.to(listening)
    timed_out = listening.to(idle)

    def __init__(
        self, config, event_bus, audio_in, audio_out, vad, stt, llm, tts,
        wake_word_detector, memory_manager,
        task_tracker=None, announcement_queue=None,
        tool_handlers=None, tool_schemas=None,
    ):
        self.config = config
        self.event_bus = event_bus
        self.audio_in = audio_in
        self.audio_out = audio_out
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.wake_word_detector = wake_word_detector
        self.memory_manager = memory_manager
        self.task_tracker = task_tracker
        self.announcement_queue = announcement_queue
        self.tool_handlers = tool_handlers
        self.tool_schemas = tool_schemas
        
        self.current_audio_buffer = None
        self.current_announcement = None
        self.sentence_queue = None
        self._last_response = None
        
        # Subscribe to global events
        self.event_bus.subscribe("WAKE", self.on_wake_event)

        # Wire task tracker completion to announcement queue
        if self.task_tracker and self.announcement_queue:
            self.task_tracker.set_announce_callback(self._on_task_completed)
        
        # Must call super last
        super().__init__()

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

        if self.announcement_queue and self.announcement_queue.has_pending:
            logger.info("[State] Announcement pending, skipping wake word")
            await asyncio.sleep(2)
            asyncio.create_task(self._safe_announce())
            return

        self._summarize_task = asyncio.create_task(
            self.memory_manager.summarize_session(self.llm)
        )
        if not self.wake_word_detector.is_listening:
            self.wake_word_detector.start(self.audio_in)

    async def on_exit_idle(self):
        logger.info("[State] Exiting IDLE")
        self.wake_word_detector.stop()
        task = getattr(self, '_summarize_task', None)
        if task and not task.done():
            task.cancel()

    # --- LISTENING STATE ---
    async def on_enter_listening(self):
        logger.info("[State] Entering LISTENING")

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
            # We pass a timeout to the VAD so it doesn't hang forever
            audio_buffer = await self.vad.listen(self.audio_in, timeout=self.config.pipeline.listening_timeout)
            
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

            self.memory_manager.record_turn("user", text)
            messages = self.memory_manager.build_context(text)

            # Phase 1: Tool detection loop (non-streaming)
            tools = self.tool_schemas if self.tool_handlers else None
            tool_rounds = 0
            while tools and tool_rounds < 5:
                response = await self.llm.complete(messages, tools=tools)
                if not response.tool_calls:
                    break

                for tool_call in response.tool_calls:
                    func_name = tool_call.function.name
                    handler = self.tool_handlers.get(func_name)
                    if not handler:
                        logger.warning(f"[State] No handler for tool: {func_name}")
                        continue

                    try:
                        args = json.loads(tool_call.function.arguments)
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
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
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
                    async for token in self.llm.stream_tokens(messages):
                        full_response.append(token)
                        sentence = splitter.process_token(token)
                        if sentence:
                            await self.sentence_queue.put(sentence)
                    remaining = splitter.flush()
                    if remaining:
                        await self.sentence_queue.put(remaining)
                except Exception as e:
                    logger.error(f"LLM Worker error: {e}")
                finally:
                    await self.sentence_queue.put(None)
                    self._last_response = "".join(full_response)

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
                    async for token in self.llm.stream_tokens(messages):
                        full_response.append(token)
                        sentence = splitter.process_token(token)
                        if sentence:
                            await self.sentence_queue.put(sentence)
                    remaining = splitter.flush()
                    if remaining:
                        await self.sentence_queue.put(remaining)
                except Exception as e:
                    logger.error(f"Announce LLM Worker error: {e}")
                finally:
                    await self.sentence_queue.put(None)
                    self._last_response = "".join(full_response)

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
        self._speaking_task = asyncio.create_task(self._do_speak())

    async def on_exit_speaking(self):
        if hasattr(self, '_speaking_task') and not self._speaking_task.done():
            self._speaking_task.cancel()
        if hasattr(self, '_llm_task') and not self._llm_task.done():
            self._llm_task.cancel()
        self.audio_out.clear()
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
            tts_task.cancel()
        except Exception as e:
            logger.error(f"[State] Error in SPEAKING: {e}")
            asyncio.create_task(self._safe_turn_complete())
