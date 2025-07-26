#!/usr/bin/env python
"""
Async Voice Processing Pipeline for DAISY - Prototype
=====================================================

This file implements a standalone async voice processing pipeline that:
1. Records audio continuously with energy-based VAD
2. Streams transcription as audio chunks become available
3. Generates LLM responses with streaming
4. Synthesizes and plays TTS audio sentence by sentence

This prototype excludes wake word detection and focuses on optimizing
the voice→text→LLM→TTS→audio pipeline for minimal latency.
"""

import asyncio
import logging
import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import aiohttp
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from faster_whisper import WhisperModel
from threading import Thread
import queue
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AudioChunk:
    """Container for audio data chunks."""
    data: np.ndarray
    timestamp: float
    chunk_id: int

@dataclass
class TranscriptSegment:
    """Container for transcription segments."""
    text: str
    confidence: float
    start_time: float
    end_time: float
    is_final: bool = True

@dataclass
class ResponseSegment:
    """Container for LLM response segments."""
    text: str
    sequence: int
    is_complete: bool = False

class AsyncAudioRecorder:
    """Async audio recorder with energy-based VAD."""
    
    def __init__(self, sample_rate=16000, channels=1, chunk_duration=0.1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        
        # VAD parameters
        self.energy_threshold = 0.01
        self.silence_threshold = 0.009 # Increased to ignore very low background noise more effectively
        self.min_speech_duration = 0.8 
        self.max_silence_duration = 2.0 # Adjusted for quicker end-of-speech detection
        
        # State for VAD logic
        self.speech_started = False
        self.silence_chunks = 0
        self.speech_chunks = 0
        self._max_silence_chunks = int(self.max_silence_duration / self.chunk_duration)
        self._chunk_queue: Optional[asyncio.Queue] = None
        
        self.is_recording = False
        self.chunk_counter = 0
        
        self._reset_vad_state_event: Optional[asyncio.Event] = None # Event to signal VAD reset

    def set_reset_event(self, event: asyncio.Event):
        self._reset_vad_state_event = event

    def _audio_input_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Calculate energy for VAD
        energy = np.mean(np.abs(indata))
        chunk_data = indata.copy().flatten()
        
        logger.debug(f"Chunk {self.chunk_counter} energy: {energy}")

        # Create audio chunk with metadata
        chunk = AudioChunk(
            data=chunk_data,
            timestamp=time_info.inputBufferAdcTime,
            chunk_id=self.chunk_counter
        )
        self.chunk_counter += 1
        
        # VAD logic
        if energy > self.energy_threshold:
            if not self.speech_started:
                logger.info("🎤 Speech detected - starting transcription")
                self.speech_started = True
            
            self.speech_chunks += 1
            self.silence_chunks = 0
            
            # Add to queue for processing
            try:
                if self._chunk_queue:
                    self._chunk_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                logger.warning("Audio chunk queue is full, dropping chunk")
                
        elif self.speech_started and energy < self.silence_threshold:
            self.silence_chunks += 1
            
            # Still add silence chunks to maintain audio continuity
            try:
                if self._chunk_queue:
                    self._chunk_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass
            
            # Check if speech has ended
            if self.silence_chunks >= self._max_silence_chunks:
                min_speech_chunks = int(self.min_speech_duration / self.chunk_duration)
                if self.speech_chunks >= min_speech_chunks:
                    logger.info(f"🔇 Speech ended ({self.speech_chunks} chunks, {self.speech_chunks * self.chunk_duration:.2f}s)")
                    # Signal end of speech
                    try:
                        if self._chunk_queue:
                            self._chunk_queue.put_nowait(None)  # End marker
                            # After signaling end of speech, reset VAD state and signal pipeline for next interaction
                            self.speech_started = False
                            self.silence_chunks = 0
                            self.speech_chunks = 0
                            if self._reset_vad_state_event:
                                self._reset_vad_state_event.set() # Signal pipeline for next interaction

                    except asyncio.QueueFull:
                        pass
                
                # Reset for next speech segment (if not already reset by the event logic above)
                if not self.speech_started:
                    self.speech_started = False
                    self.silence_chunks = 0
                    self.speech_chunks = 0

    async def record_audio_stream(self, chunk_queue: asyncio.Queue) -> None:
        """
        Continuously capture microphone input and push chunks to queue.
        Waits for a signal to start actively detecting speech.
        """
        logger.info("Starting async audio recording stream")
        self._chunk_queue = chunk_queue # Store queue for callback
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                callback=self._audio_input_callback,
                blocksize=self.chunk_size
            ):
                logger.info("Audio stream started - waiting for start signal...")
                while self.is_recording:
                    # Wait for the signal to reset VAD state and begin a new listening cycle
                    if self._reset_vad_state_event:
                        await self._reset_vad_state_event.wait()
                        self._reset_vad_state_event.clear() # Clear the event once processed

                    logger.info("Audio stream ready - speak now!")
                    # No need for explicit VAD reset here, as it's done in callback after EOS
                    await asyncio.sleep(0.1) # Keep coroutine alive while waiting for speech
                    
        except Exception as e:
            logger.error(f"Error in audio recording: {e}")
        finally:
            logger.info("Audio recording stream stopped")

class AsyncWhisperTranscriber:
    """Async transcriber using Faster-Whisper."""
    
    def __init__(self, model_size="base", device="cpu", chunk_duration=0.1):
        self.model_size = model_size
        self.device = device
        self.model = None
        self.recorder_chunk_duration = chunk_duration # Store the chunk duration
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize Whisper model."""
        try:
            logger.info(f"Loading Faster-Whisper model: {self.model_size}")
            compute_type = "int8" if self.device == "cpu" else "float16"
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    async def transcribe_stream(self, chunk_queue: asyncio.Queue, transcript_queue: asyncio.Queue) -> None:
        """
        Process audio chunks and generate transcriptions.
        Accumulates chunks and transcribes complete segments only after end-of-speech.
        """
        logger.info("Starting async transcription stream")
        
        audio_buffer = []
        sample_rate = 16000
        
        try:
            while True:
                # Get audio chunk
                try:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                
                # If it's an end marker (None) or we have accumulated some audio and recorder is no longer active (fallback)
                if chunk is None:
                    if audio_buffer:
                        # Transcribe accumulated audio
                        await self._process_audio_buffer(audio_buffer, sample_rate, transcript_queue)
                        audio_buffer = [] # Reset buffer after processing
                    continue # Wait for next speech segment
                
                # Accumulate audio chunks
                audio_buffer.append(chunk.data)
                    
        except Exception as e:
            logger.error(f"Error in transcription stream: {e}")
    
    async def _process_audio_buffer(self, audio_buffer: List[np.ndarray], sample_rate: int, transcript_queue: asyncio.Queue):
        """Process accumulated audio buffer for transcription."""
        if not audio_buffer or not self.model:
            return
            
        try:
            # Combine audio chunks
            combined_audio = np.concatenate(audio_buffer)
            
            # Normalize audio
            max_val = np.max(np.abs(combined_audio))
            if max_val > 0:
                combined_audio = combined_audio / max_val * 0.95
            
            # Save to temporary file for Whisper
            temp_file = f"temp_audio_{int(time.time() * 1000)}.wav"
            sf.write(temp_file, combined_audio, sample_rate)
            
            # Transcribe using thread executor to avoid blocking
            loop = asyncio.get_event_loop()
            transcript_text = await loop.run_in_executor(
                self.executor,
                self._transcribe_file,
                temp_file
            )
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
            
            if transcript_text and transcript_text.strip():
                logger.info(f"📝 Transcribed: '{transcript_text}'")
                
                segment = TranscriptSegment(
                    text=transcript_text.strip(),
                    confidence=0.9,  # Simplified confidence
                    start_time=time.time(),
                    end_time=time.time(),
                    is_final=True
                )
                
                await transcript_queue.put(segment)
                
        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}")
    
    def _transcribe_file(self, audio_file: str) -> Optional[str]:
        """Synchronously transcribe audio file."""
        try:
            segments, info = self.model.transcribe(
                audio_file,
                language="en",
                beam_size=1,  # Fast beam search
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                temperature=0.0,
                initial_prompt="This is a voice command or question."
            )
            
            # Combine segments
            text = " ".join([segment.text for segment in segments]).strip()
            return text if text else None
            
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            return None

class AsyncOllamaClient:
    """Async client for Ollama LLM API."""
    
    def __init__(self, base_url="http://localhost:11434", model="llama3.2:latest"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = None
        self._interaction_complete_event: Optional[asyncio.Event] = None

    def set_interaction_complete_event(self, event: asyncio.Event):
        self._interaction_complete_event = event

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate_response_stream(self, transcript_queue: asyncio.Queue, response_queue: asyncio.Queue) -> None:
        """
        Generate LLM responses from transcript segments.
        Supports streaming responses for lower latency.
        """
        logger.info("Starting LLM response generation stream")
        
        try:
            while True:
                # Wait for transcript
                try:
                    transcript = await asyncio.wait_for(transcript_queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    continue
                
                if not transcript or not transcript.text.strip():
                    continue
                
                logger.info(f"🧠 Generating response for: '{transcript.text}'")
                
                # Prepare messages (simplified - no chat history/RAG for prototype)
                messages = [
                    {
                        "role": "system",
                        "content": "You are DAISY, a helpful AI assistant. Provide concise, friendly responses."
                    },
                    {
                        "role": "user", 
                        "content": transcript.text
                    }
                ]
                
                # Generate response
                await self._stream_chat_completion(messages, response_queue)
                
                # After completing the response, signal that interaction is done
                if self._interaction_complete_event:
                    self._interaction_complete_event.set()
                    logger.info("Interaction completed - signaling pipeline for next turn.")

        except Exception as e:
            logger.error(f"Error in response generation: {e}")
    
    async def _stream_chat_completion(self, messages: List[Dict], response_queue: asyncio.Queue):
        """Stream chat completion from Ollama."""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    logger.error(f"LLM API error: {response.status}")
                    return
                
                full_response = ""
                sentence_buffer = ""
                sequence = 0
                
                async for line in response.content:
                    try:
                        line_text = line.decode('utf-8').strip()
                        if not line_text:
                            continue
                            
                        data = json.loads(line_text)
                        
                        if "message" in data and "content" in data["message"]:
                            content = data["message"]["content"]
                            full_response += content
                            sentence_buffer += content
                            
                            # Check for sentence boundaries
                            sentences = self._extract_complete_sentences(sentence_buffer)
                            
                            for sentence in sentences:
                                if sentence.strip():
                                    segment = ResponseSegment(
                                        text=sentence.strip(),
                                        sequence=sequence,
                                        is_complete=False
                                    )
                                    await response_queue.put(segment)
                                    sequence += 1
                            
                            # Keep remaining incomplete sentence
                            sentence_buffer = self._get_remaining_text(sentence_buffer, sentences)
                        
                        # Check if done
                        if data.get("done", False):
                            # Send any remaining text
                            if sentence_buffer.strip():
                                segment = ResponseSegment(
                                    text=sentence_buffer.strip(),
                                    sequence=sequence,
                                    is_complete=True
                                )
                                await response_queue.put(segment)
                            
                            logger.info(f"💬 Complete response: '{full_response}'")
                            break
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing stream chunk: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error in streaming chat completion: {e}")
    
    def _extract_complete_sentences(self, text: str) -> List[str]:
        """Extract complete sentences from text buffer."""
        import re
        # Split on sentence boundaries
        sentences = re.split(r'([.!?]+\s+)', text)
        complete = []
        
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                complete.append(sentences[i] + sentences[i + 1])
        
        return complete
    
    def _get_remaining_text(self, original: str, extracted_sentences: List[str]) -> str:
        """Get remaining incomplete text after extracting sentences."""
        remaining = original
        for sentence in extracted_sentences:
            remaining = remaining.replace(sentence, "", 1)
        return remaining

class AsyncTTSPlayer:
    """Async TTS synthesis and audio playback."""
    
    def __init__(self):
        self.tts_engine = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._initialize_tts()
        
    def _initialize_tts(self):
        """Initialize TTS engine."""
        try:
            # Try Coqui TTS first
            from TTS.api import TTS
            logger.info("Initializing Coqui TTS")
            self.tts_engine = TTS(
                model_name="tts_models/en/vctk/vits",
                progress_bar=False
            )
            logger.info("Coqui TTS initialized successfully")
            
        except Exception as e:
            logger.warning(f"Coqui TTS failed, falling back to pyttsx3: {e}")
            # Fallback to pyttsx3
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 180)
            self.tts_engine.setProperty('volume', 1.0)
    
    async def tts_and_play_loop(self, response_queue: asyncio.Queue) -> None:
        """
        Generate and play TTS audio sentence by sentence.
        Processes responses as they become available for minimal latency.
        """
        logger.info("Starting TTS and playback stream")
        
        try:
            while True:
                # Wait for response segment
                try:
                    segment = await asyncio.wait_for(response_queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    continue
                
                if not segment or not segment.text.strip():
                    continue
                
                logger.info(f"🔊 Synthesizing: '{segment.text}'")
                
                # Generate and play TTS
                await self._synthesize_and_play(segment.text)
                
        except Exception as e:
            logger.error(f"Error in TTS and playback: {e}")
    
    async def _synthesize_and_play(self, text: str):
        """Synthesize and play a single text segment."""
        try:
            # Run TTS synthesis in executor to avoid blocking
            loop = asyncio.get_event_loop()
            audio_file = await loop.run_in_executor(
                self.executor,
                self._synthesize_text,
                text
            )
            
            if audio_file:
                # Play audio in executor
                await loop.run_in_executor(
                    self.executor,
                    self._play_audio,
                    audio_file
                )
                
                # Clean up temp file
                try:
                    os.remove(audio_file)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error synthesizing/playing text: {e}")
    
    def _synthesize_text(self, text: str) -> Optional[str]:
        """Synchronously synthesize text to audio file."""
        try:
            temp_file = f"temp_tts_{int(time.time() * 1000)}.wav"
            
            if hasattr(self.tts_engine, 'tts_to_file'):
                # Coqui TTS
                self.tts_engine.tts_to_file(
                    text=text,
                    file_path=temp_file,
                    speaker="p277"  # British female voice
                )
            else:
                # pyttsx3 fallback (save to file not directly supported)
                # For prototype, we'll just play directly
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return None
            
            return temp_file
            
        except Exception as e:
            logger.error(f"Error synthesizing text: {e}")
            return None
    
    def _play_audio(self, audio_file: str):
        """Synchronously play audio file."""
        try:
            if os.name == 'nt':  # Windows
                import winsound
                winsound.PlaySound(audio_file, winsound.SND_FILENAME)
            else:
                # Cross-platform fallback
                import subprocess
                import platform
                
                system = platform.system()
                if system == 'Darwin':  # macOS
                    subprocess.call(['afplay', audio_file])
                else:  # Linux
                    subprocess.call(['aplay', audio_file])
                    
        except Exception as e:
            logger.error(f"Error playing audio: {e}")

class AsyncVoicePipeline:
    """Main async voice pipeline orchestrator."""
    
    def __init__(self):
        self.recorder = AsyncAudioRecorder()
        self.transcriber = AsyncWhisperTranscriber(
            chunk_duration=self.recorder.chunk_duration # Pass chunk duration
        )
        self.tts_player = AsyncTTSPlayer()
        
        # Communication queues
        self.chunk_queue = asyncio.Queue(maxsize=50)
        self.transcript_queue = asyncio.Queue(maxsize=10)
        self.response_queue = asyncio.Queue(maxsize=20)
        
        # Control events
        self.new_interaction_event = asyncio.Event() # Signal for new interaction
        
        self.running = False
        self.tasks = []

        # Set events in components
        self.recorder.set_reset_event(self.new_interaction_event)
        self.ollama_client = AsyncOllamaClient() # Initialize here for event setting
        self.ollama_client.set_interaction_complete_event(self.new_interaction_event)
    
    async def start_pipeline(self):
        """Start the complete async voice pipeline."""
        logger.info("🚀 Starting async voice pipeline")
        self.running = True
        self.recorder.is_recording = True
        
        # Initially set the event to allow the recorder to start listening for the first time
        self.new_interaction_event.set()

        try:
            # Start all pipeline stages concurrently
            async with self.ollama_client as ollama:
                self.tasks = [
                    asyncio.create_task(
                        self.recorder.record_audio_stream(self.chunk_queue),
                        name="audio_recorder"
                    ),
                    asyncio.create_task(
                        self.transcriber.transcribe_stream(self.chunk_queue, self.transcript_queue),
                        name="transcriber"
                    ),
                    asyncio.create_task(
                        ollama.generate_response_stream(self.transcript_queue, self.response_queue),
                        name="llm_generator"
                    ),
                    asyncio.create_task(
                        self.tts_player.tts_and_play_loop(self.response_queue),
                        name="tts_player"
                    )
                ]
                
                logger.info("✅ All pipeline stages started")
                logger.info("🎤 Speak now - the pipeline is listening!")
                
                # Keep the main pipeline alive and wait for tasks to finish
                await asyncio.gather(*self.tasks, return_exceptions=True)
                
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            await self.stop_pipeline()

async def main():
    """Main entry point for the async voice pipeline."""
    print("=" * 60)
    print("🤖 DAISY Async Voice Pipeline - Prototype")
    print("=" * 60)
    print()
    print("This prototype demonstrates an async voice processing pipeline:")
    print("1. 🎤 Continuous audio recording with VAD")
    print("2. 📝 Streaming transcription with Faster-Whisper")
    print("3. 🧠 Async LLM response generation with Ollama")
    print("4. 🔊 Real-time TTS synthesis and playback")
    print()
    print("Requirements:")
    print("- Ollama running with llama3.2:latest model")
    print("- Working microphone and speakers")
    print("- Python packages: faster-whisper, TTS, aiohttp")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    pipeline = AsyncVoicePipeline()
    
    try:
        await pipeline.start_pipeline()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main error: {e}")

if __name__ == "__main__":
    # Run the async pipeline
    asyncio.run(main()) 