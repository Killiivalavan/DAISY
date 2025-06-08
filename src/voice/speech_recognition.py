"""
Enhanced Speech Recognition for DAISY - Completely Rewritten for Reliability
"""
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import time
import logging
import re
from typing import Optional, Tuple, List
from faster_whisper import WhisperModel
from src.utils.config import (
    RECORDING_FILE, 
    TRANSCRIPTION_FILE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_PATH,
    PORCUPINE_ACCESS_KEY,
    PORCUPINE_MODEL_PATH,
    PORCUPINE_SENSITIVITY,
    PORCUPINE_ENABLED,
    USE_WAKE_WORD
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WakeWordDetector:
    """Simplified wake word detection using Picovoice Porcupine."""
    
    def __init__(self, sensitivity: float = None, model_path: str = None):
        """Initialize wake word detector with comprehensive error handling."""
        self.is_available = False
        self.initialization_error = None
        self.porcupine = None
        self.sample_rate = 16000
        self.frame_length = 512
        
        # Check if wake word detection should be enabled
        if not PORCUPINE_ENABLED or not PORCUPINE_ACCESS_KEY:
            self.initialization_error = "Wake word detection disabled (no access key provided)"
            logger.info(self.initialization_error)
            return
            
        try:
            import pvporcupine
            
            self.sensitivity = sensitivity or PORCUPINE_SENSITIVITY
            self.model_path = model_path or PORCUPINE_MODEL_PATH
            
            # Validate model file
            if not os.path.exists(self.model_path):
                self.initialization_error = f"Wake word model not found: {self.model_path}"
                logger.error(self.initialization_error)
                return
            
            # Initialize Porcupine
            logger.info("Initializing Porcupine wake word detector...")
            self.porcupine = pvporcupine.create(
                access_key=PORCUPINE_ACCESS_KEY,
                keyword_paths=[self.model_path],
                sensitivities=[self.sensitivity]
            )
            
            self.sample_rate = self.porcupine.sample_rate
            self.frame_length = self.porcupine.frame_length
            self.is_available = True
            
            logger.info(f"Wake word detector initialized (sensitivity={self.sensitivity})")
            
        except ImportError:
            self.initialization_error = "Porcupine library not installed"
            logger.error(self.initialization_error)
        except Exception as e:
            self.initialization_error = f"Porcupine initialization failed: {e}"
            logger.error(self.initialization_error)
    
    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """Process audio frame for wake word detection."""
        if not self.is_available or not self.porcupine:
            return False
            
        try:
            if audio_frame is None or len(audio_frame) != self.frame_length:
                return False
                
            # Ensure correct format
            if audio_frame.dtype != np.int16:
                audio_frame = (np.clip(audio_frame, -1, 1) * 32767).astype(np.int16)
                
            result = self.porcupine.process(audio_frame)
            if result >= 0:
                logger.info(f"Wake word detected! (result={result})")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error processing wake word frame: {e}")
            return False
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'porcupine') and self.porcupine:
            try:
                self.porcupine.delete()
            except Exception:
                pass

class SimplifiedSpeechRecognizer:
    """Completely rewritten speech recognizer with robust audio handling."""
    
    def __init__(self, model_name: str = "base", device: str = "cpu", use_wake_word: bool = None):
        """Initialize speech recognition with simplified, reliable approach."""
        logger.info(f"Initializing SimplifiedSpeechRecognizer with model: {model_name}")
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32
        
        # Recording state
        self.is_recording = False
        self.audio_data = []
        self.recording_thread = None
        
        # Initialize Whisper model
        self.model = None
        self.model_available = False
        self._initialize_whisper_model(model_name, device)
        
        # Initialize wake word detection
        self.use_wake_word = USE_WAKE_WORD if use_wake_word is None else use_wake_word
        self.wake_word_detector = None
        self.wake_word_fallback_reason = None
        
        if self.use_wake_word:
            self.wake_word_detector = WakeWordDetector()
            if not self.wake_word_detector.is_available:
                self.wake_word_fallback_reason = self.wake_word_detector.initialization_error
                logger.warning(f"Wake word detection disabled: {self.wake_word_fallback_reason}")
                self.use_wake_word = False
            else:
                logger.info("Wake word detection enabled")
        
        # Conversation state
        self.listening_for_command = False
        self.last_interaction_time = 0
        self.conversation_timeout = 10.0  # Increased timeout for better UX
        
        logger.info("SimplifiedSpeechRecognizer initialization completed")
    
    def _initialize_whisper_model(self, model_name: str, device: str):
        """Initialize Whisper model with proper error handling."""
        try:
            logger.info(f"Loading Faster-Whisper model: {model_name}")
            
            # Use appropriate compute type based on device
            compute_type = "float16" if device == "cuda" else "int8"
            
            self.model = WhisperModel(
                model_name, 
                device=device, 
                compute_type=compute_type,
                download_root=WHISPER_MODEL_PATH
            )
            self.model_available = True
            logger.info(f"Whisper model loaded successfully with {compute_type} precision")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            try:
                # Fallback to OpenAI Whisper
                import whisper
                self.model = whisper.load_model(model_name, download_root=WHISPER_MODEL_PATH)
                self.model_available = True
                logger.info("Fallback to OpenAI Whisper successful")
            except Exception as fallback_error:
                logger.error(f"Failed to load any Whisper model: {fallback_error}")
                self.model = None
                self.model_available = False
    
    def _validate_transcription(self, text: str) -> bool:
        """Validate transcription to detect obvious errors."""
        if not text or len(text.strip()) < 3:
            return False
        
        # Clean the text
        cleaned = text.strip().lower()
        
        # Check for repetitive patterns
        words = cleaned.split()
        if len(words) > 5:
            # Check if more than 70% of words are the same
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:
                logger.warning(f"Detected repetitive transcription: {text[:100]}...")
                return False
        
        # Check for common transcription artifacts
        artifacts = [
            "thank you for watching",
            "thanks for watching", 
            "please subscribe",
            "don't forget to like",
            "music playing",
            "applause",
            "laughter"
        ]
        
        if any(artifact in cleaned for artifact in artifacts):
            logger.warning(f"Detected transcription artifact: {text[:50]}...")
            return False
        
        # Check for excessive repetition of short phrases
        if len(text) > 50:
            # Split into chunks and check for repetition
            chunk_size = min(10, len(text) // 4)
            chunks = [text[i:i+chunk_size] for i in range(0, len(text)-chunk_size, chunk_size)]
            if len(chunks) > 3:
                most_common = max(set(chunks), key=chunks.count)
                if chunks.count(most_common) > len(chunks) * 0.6:
                    logger.warning(f"Detected chunk repetition: {most_common}")
                    return False
        
        return True
    
    def transcribe(self, audio_filename: str, text_filename: str = None) -> Optional[str]:
        """Transcribe audio file with robust error handling and validation."""
        if not audio_filename or not os.path.exists(audio_filename):
            logger.error(f"Audio file not found: {audio_filename}")
            return None
            
        if not self.model_available:
            logger.error("Whisper model not available")
            return None
        
        # Validate audio file
        try:
            audio_data, sr = sf.read(audio_filename)
            duration = len(audio_data) / sr
            logger.info(f"Processing audio with duration {duration:.2f}s")
            
            # Skip files that are too short or too long
            if duration < 0.5:
                logger.warning("Audio too short for reliable transcription")
                return None
            if duration > 30:
                logger.warning("Audio too long, truncating to 30 seconds")
                # Truncate to 30 seconds
                max_samples = int(30 * sr)
                audio_data = audio_data[:max_samples]
                sf.write(audio_filename, audio_data, sr)
                
        except Exception as e:
            logger.error(f"Error reading audio file: {e}")
            return None
        
        try:
            logger.info(f"Transcribing audio file: {audio_filename}")
            
            # Use faster-whisper with optimized settings
            if isinstance(self.model, WhisperModel):
                logger.info("Using Faster-Whisper for transcription")
                
                # Optimized transcription parameters
                segments, info = self.model.transcribe(
                    audio_filename,
                    language=WHISPER_LANGUAGE if WHISPER_LANGUAGE != "auto" else None,
                    vad_filter=False,  # Disable aggressive VAD filtering
                    beam_size=1,  # Fast beam search
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,  # Higher threshold to avoid false transcriptions
                    temperature=0.0,
                    initial_prompt="This is a voice command or question.",  # Guide the model
                    word_timestamps=False  # Disable for speed
                )
                
                # Combine segments
                transcribed_text = " ".join([segment.text for segment in segments]).strip()
                
                # Log detected language and confidence
                if hasattr(info, 'language'):
                    logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
                
            else:
                # Fallback to OpenAI Whisper
                logger.info("Using OpenAI Whisper for transcription")
                result = self.model.transcribe(
                    audio_filename,
                    language=WHISPER_LANGUAGE if WHISPER_LANGUAGE != "auto" else None,
                    condition_on_previous_text=False,
                    temperature=0.0,
                    initial_prompt="This is a voice command or question."
                )
                transcribed_text = result["text"].strip()
            
            # Validate transcription
            if not self._validate_transcription(transcribed_text):
                logger.warning("Transcription failed validation")
                return None
            
            if transcribed_text:
                logger.info(f"Transcription successful: '{transcribed_text}'")
                print(f"User: {transcribed_text}")
                
                # Save transcription
                if text_filename:
                    os.makedirs(os.path.dirname(text_filename), exist_ok=True)
                    with open(text_filename, 'w', encoding='utf-8') as f:
                        f.write(transcribed_text)
                
                return transcribed_text
            else:
                logger.warning("Empty transcription result")
                return None
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return None
    
    def _record_with_energy_detection(self, timeout: float = 10.0, phrase_timeout: float = 2.0) -> Optional[str]:
        """Record audio using energy-based voice activity detection."""
        logger.info("Starting energy-based recording")
        
        # Audio recording parameters
        chunk_duration = 0.1  # 100ms chunks
        chunk_size = int(self.sample_rate * chunk_duration)
        
        # Energy detection parameters
        energy_threshold = 0.01  # Adjust based on environment
        silence_threshold = 0.005
        min_phrase_length = 0.5  # Minimum phrase duration
        
        audio_buffer = []
        silence_chunks = 0
        speech_chunks = 0
        max_silence_chunks = int(phrase_timeout / chunk_duration)
        max_total_chunks = int(timeout / chunk_duration)
        
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            audio_buffer.append(indata.copy())
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=audio_callback,
                blocksize=chunk_size
            ):
                logger.info("Listening for speech...")
                
                speech_started = False
                speech_audio = []
                
                for chunk_idx in range(max_total_chunks):
                    time.sleep(chunk_duration)
                    
                    if len(audio_buffer) > chunk_idx:
                        chunk = audio_buffer[chunk_idx]
                        
                        # Calculate energy
                        energy = np.mean(np.abs(chunk))
                        
                        if energy > energy_threshold:
                            # Speech detected
                            if not speech_started:
                                logger.info("Speech started")
                                speech_started = True
                            
                            speech_audio.append(chunk)
                            speech_chunks += 1
                            silence_chunks = 0
                            
                        elif speech_started and energy < silence_threshold:
                            # Silence during speech
                            silence_chunks += 1
                            speech_audio.append(chunk)  # Include silence in recording
                            
                            if silence_chunks >= max_silence_chunks:
                                logger.info("Speech ended (silence detected)")
                                break
                        
                        elif speech_started:
                            # Low energy but not silent
                            speech_audio.append(chunk)
                
                # Check if we have enough speech
                if speech_started and len(speech_audio) >= int(min_phrase_length / chunk_duration):
                    # Combine audio chunks
                    audio_data = np.concatenate(speech_audio)
                    
                    # Normalize audio
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 0:
                        audio_data = audio_data / max_val * 0.95
                    
                    # Save audio
                    os.makedirs(os.path.dirname(RECORDING_FILE), exist_ok=True)
                    sf.write(RECORDING_FILE, audio_data, self.sample_rate)
                    
                    duration = len(audio_data) / self.sample_rate
                    logger.info(f"Audio saved: {duration:.2f}s duration")
                    print(f"Audio saved to {RECORDING_FILE}")
                    
                    return RECORDING_FILE
                else:
                    logger.info("No speech detected or phrase too short")
                    return None
                    
        except Exception as e:
            logger.error(f"Error during recording: {e}")
            return None
    
    def listen_for_wake_word(self) -> bool:
        """Listen for wake word with simplified implementation."""
        if not self.use_wake_word or not self.wake_word_detector or not self.wake_word_detector.is_available:
            return True
        
        logger.info("Listening for wake word...")
        print("Listening for wake word...")
        
        frame_length = self.wake_word_detector.frame_length
        frame_buffer = []
        
        def wake_word_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Wake word callback status: {status}")
            frame_buffer.extend(indata.flatten())
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=wake_word_callback,
                blocksize=frame_length
            ):
                while True:
                    if len(frame_buffer) >= frame_length:
                        # Extract frame
                        frame_data = np.array(frame_buffer[:frame_length])
                        frame_buffer = frame_buffer[frame_length:]
                        
                        # Convert to int16 for Porcupine
                        frame_int16 = (np.clip(frame_data, -1, 1) * 32767).astype(np.int16)
                        
                        # Process frame
                        if self.wake_word_detector.process_frame(frame_int16):
                            print("Wake word detected!")
                            return True
                    
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            logger.info("Wake word detection interrupted")
            return False
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            return False
    
    def listen(self, listen_for_wake_word: bool = None) -> Optional[str]:
        """
        Main listening method with simplified logic.
        
        Args:
            listen_for_wake_word: Override wake word setting
            
        Returns:
            Path to recorded audio file or None
        """
        current_time = time.time()
        
        # Determine if wake word is needed
        should_listen_for_wake_word = self.use_wake_word
        if listen_for_wake_word is not None:
            should_listen_for_wake_word = listen_for_wake_word
        
        # Handle wake word detection
        if should_listen_for_wake_word and not self.listening_for_command:
            # Check conversation timeout
            if current_time - self.last_interaction_time > self.conversation_timeout:
                if not self.listen_for_wake_word():
                    return None
                self.listening_for_command = True
        
        # Prevent concurrent recordings
        if self.is_recording:
            logger.warning("Already recording")
            return None
        
        self.is_recording = True
        try:
            # Record audio
            audio_file = self._record_with_energy_detection()
            
            # Update interaction time
            if audio_file:
                self.last_interaction_time = current_time
            else:
                # Reset command listening if no speech detected
                if self.use_wake_word:
                    self.listening_for_command = False
            
            return audio_file
            
        finally:
            self.is_recording = False
    
    def get_device_info(self) -> dict:
        """Get audio device information for debugging."""
        try:
            devices = sd.query_devices()
            default_input = sd.query_devices(kind='input')
            
            return {
                'devices': devices,
                'default_input': default_input,
                'api': sd.query_hostapis()[0].get('name', 'Unknown'),
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'error': None
            }
        except Exception as e:
            return {
                'devices': [],
                'default_input': 'Unknown',
                'api': 'Unknown',
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'error': str(e)
            }

# Maintain compatibility with existing code
SpeechRecognizer = SimplifiedSpeechRecognizer 