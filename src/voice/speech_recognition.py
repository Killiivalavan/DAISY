"""
Speech recognition functionality for DAISY using faster-whisper and webrtcvad.
"""
import os
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time
import wave
import webrtcvad
import collections
from array import array
from struct import pack
from faster_whisper import WhisperModel
from src.utils.config import (
    RECORDING_FILE, 
    TRANSCRIPTION_FILE,
    WHISPER_LANGUAGE,
    WHISPER_VAD_FILTER,
    WHISPER_VAD_PARAMETERS,
    WEBRTC_VAD_MODE,
    WEBRTC_FRAME_DURATION_MS,
    WEBRTC_SPEECH_START_FRAMES,
    WEBRTC_SPEECH_END_FRAMES,
    WEBRTC_MAX_RECORDING_SECS,
    PORCUPINE_ACCESS_KEY,
    PORCUPINE_MODEL_PATH,
    PORCUPINE_SENSITIVITY,
    PORCUPINE_ENABLED,
    USE_WAKE_WORD
)

class WakeWordDetector:
    """Wake word detection using Picovoice Porcupine."""
    
    def __init__(self, sensitivity=None, model_path=None):
        """
        Initialize the wake word detector.
        
        Args:
            sensitivity: Detection sensitivity (0.0-1.0)
            model_path: Path to the Porcupine model file (.ppn)
        """
        self.is_available = PORCUPINE_ENABLED
        
        if not self.is_available:
            print("Wake word detection disabled (no access key provided)")
            return
            
        try:
            # Import here to avoid dependency if not used
            import pvporcupine
            
            self.sensitivity = sensitivity if sensitivity is not None else PORCUPINE_SENSITIVITY
            self.model_path = model_path if model_path is not None else PORCUPINE_MODEL_PATH
            
            if not os.path.exists(self.model_path):
                print(f"Wake word model not found at: {self.model_path}")
                self.is_available = False
                return
                
            # Initialize Porcupine
            self.porcupine = pvporcupine.create(
                access_key=PORCUPINE_ACCESS_KEY,
                keyword_paths=[self.model_path],
                sensitivities=[self.sensitivity]
            )
            
            # Audio settings must match Porcupine requirements
            self.sample_rate = self.porcupine.sample_rate
            self.frame_length = self.porcupine.frame_length
            
            print(f"Wake word detector initialized (sensitivity={self.sensitivity})")
        except Exception as e:
            print(f"Error initializing wake word detector: {e}")
            self.is_available = False
    
    def process_frame(self, audio_frame):
        """
        Process an audio frame to detect wake word.
        
        Args:
            audio_frame: Audio frame as a numpy array (int16, must match porcupine.frame_length)
            
        Returns:
            True if wake word detected, False otherwise
        """
        if not self.is_available:
            return False
            
        try:
            # Make sure the audio is in the correct format (int16)
            if audio_frame.dtype != np.int16:
                audio_frame = (np.clip(audio_frame, -1, 1) * 32767).astype(np.int16)
                
            # Make sure we have the right number of samples
            if len(audio_frame) != self.frame_length:
                print(f"Warning: Audio frame length mismatch. Got {len(audio_frame)}, expected {self.frame_length}")
                return False
                
            # Process the frame
            result = self.porcupine.process(audio_frame)
            return result >= 0  # Result is the keyword index or -1 if not detected
            
        except Exception as e:
            print(f"Error processing audio frame for wake word: {e}")
            return False
            
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'porcupine') and self.porcupine:
            self.porcupine.delete()

class SpeechRecognizer:
    def __init__(self, model_size="base", compute_type=None, device=None, beam_size=5, use_wake_word=None):
        """
        Initialize the speech recognizer with faster-whisper and webrtcvad.
        
        Args:
            model_size: Size of the Whisper model ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
            compute_type: Type of compute to use ('float16', 'float32', 'int8'). If None, auto-detected.
            device: Device to use for inference ('cuda', 'cpu'). If None, auto-detected.
            beam_size: Beam size for decoding (higher = better quality, slower)
            use_wake_word: Whether to use wake word detection. If None, uses config value.
        """
        # Audio settings - Whisper works best with 16kHz mono audio
        # webrtcvad requires 16kHz sample rate
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = 'float32'
        self.frame_duration_ms = WEBRTC_FRAME_DURATION_MS
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        # VAD parameters
        self.vad = webrtcvad.Vad(WEBRTC_VAD_MODE)
        self.current_vad_mode = WEBRTC_VAD_MODE  # Store the current mode
        self.speech_start_frames = WEBRTC_SPEECH_START_FRAMES
        self.speech_end_frames = WEBRTC_SPEECH_END_FRAMES
        self.max_recording_seconds = WEBRTC_MAX_RECORDING_SECS
        
        # Audio data buffers
        self.audio_queue = queue.Queue()
        self.audio_buffer = collections.deque(maxlen=int(self.sample_rate * self.max_recording_seconds / 480))
        
        # Recording state
        self.is_recording = False
        self.recording_thread = None
        self.speech_detected = False
        self.voiced_frames = 0
        self.unvoiced_frames = 0
        self.recording_start_time = 0
        
        # Wake word detection
        self.use_wake_word = USE_WAKE_WORD if use_wake_word is None else use_wake_word
        self.wake_word_detector = None
        if self.use_wake_word:
            self.wake_word_detector = WakeWordDetector()
            if not self.wake_word_detector.is_available:
                print("Wake word detection not available, falling back to normal mode")
                self.use_wake_word = False
        
        # Wake word detection state
        self.wake_word_detected = False
        self.listening_for_command = False
        
        # Auto-detect device and compute type if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        
        # Initialize the faster-whisper model
        print(f"Loading Whisper model (size={model_size}, device={device}, compute_type={compute_type})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.beam_size = beam_size
        print("Whisper model loaded successfully")
        
    def _audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice to process incoming audio chunks."""
        if status:
            print(f"Audio status: {status}")
        
        # Add audio data to queue
        self.audio_queue.put(indata.copy())
        
    def _process_audio_frame(self, frame_data):
        """Process audio frame through VAD"""
        # Convert float32 audio to int16 PCM for webrtcvad
        pcm_data = (np.clip(frame_data, -1.0, 1.0) * 32767).astype(np.int16)
        pcm_bytes = pcm_data.tobytes()
        
        # Check if this frame is voiced
        try:
            is_speech = self.vad.is_speech(pcm_bytes, self.sample_rate)
        except Exception as e:
            print(f"VAD error: {e}")
            return False
            
        # Update frame counters
        if is_speech:
            self.voiced_frames += 1
            self.unvoiced_frames = 0
            
            # Detect speech start
            if self.voiced_frames >= self.speech_start_frames and not self.speech_detected:
                self.speech_detected = True
                print("Speech detected")
        else:
            self.unvoiced_frames += 1
            
            # Don't reset voiced frames immediately to prevent cutting during short pauses
            if self.unvoiced_frames > 3:
                self.voiced_frames = 0
                
            # Detect speech end if we've had a significant number of silent frames
            # But only if speech was previously detected
            if (self.speech_detected and 
                self.unvoiced_frames >= self.speech_end_frames):
                print("Speech ended")
                return True  # Signal to stop recording
                
        # Check for maximum recording time as a safety measure
        if (time.time() - self.recording_start_time > self.max_recording_seconds):
            print(f"Maximum recording time of {self.max_recording_seconds}s reached")
            return True
                
        return False  # Continue recording
        
    def _recording_thread_func(self):
        """Thread function to handle continuous recording with VAD."""
        # Reset state
        self.audio_buffer.clear()
        self.voiced_frames = 0
        self.unvoiced_frames = 0
        self.speech_detected = False
        self.recording_start_time = time.time()
        speech_ended = False
        
        # Keep track of frames for VAD
        frame_buffer = []
        frame_samples = 0
        
        try:
            # Start the input stream
            with sd.InputStream(
                samplerate=self.sample_rate, 
                channels=self.channels,
                dtype=self.dtype,
                blocksize=512,  # Small blocks for responsive VAD
                callback=self._audio_callback
            ):
                print("Listening...")
                
                # Wait for audio to be captured and processed
                while self.is_recording:
                    if not self.audio_queue.empty():
                        # Get the next chunk of audio data
                        audio_chunk = self.audio_queue.get()
                        
                        # Store audio in buffer for saving later
                        self.audio_buffer.append(audio_chunk)
                        
                        # Add to frame buffer for VAD processing
                        frame_buffer.extend(audio_chunk)
                        frame_samples += len(audio_chunk)
                        
                        # Process complete frames for VAD
                        while frame_samples >= self.frame_size:
                            # Extract a single frame
                            frame = np.array(frame_buffer[:self.frame_size])
                            frame_buffer = frame_buffer[self.frame_size:]
                            frame_samples -= self.frame_size
                            
                            # Process frame through VAD
                            speech_ended = self._process_audio_frame(frame)
                            
                            # Stop recording if speech has ended
                            if speech_ended and self.speech_detected:
                                print("Stopping recording due to speech end or timeout")
                                self.is_recording = False
                                break
                                
                    # Small sleep to prevent high CPU usage
                    time.sleep(0.005)
                    
        except Exception as e:
            print(f"Error during recording: {e}")
            self.is_recording = False
        
        # If we have detected speech and it has ended (or recording stopped),
        # save the audio buffer to a file
        if self.speech_detected:
            print("Recording complete, saving audio...")
            return self._save_audio()
        else:
            print("No speech detected")
            return None
            
    def _save_audio(self):
        """Save the recorded audio to a WAV file."""
        if not self.audio_buffer:
            return None
            
        # Combine audio chunks
        audio_data = np.vstack(self.audio_buffer)
        audio_data = audio_data.flatten()
        
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(RECORDING_FILE), exist_ok=True)
            
        # Save as WAV file
        sf.write(RECORDING_FILE, audio_data, self.sample_rate)
        
        print(f"Audio saved to {RECORDING_FILE}")
        return RECORDING_FILE
    
    def listen_for_wake_word(self):
        """
        Continuously listen for wake word until detected.
        Returns True if wake word detected, False on error or if wake word detection is disabled.
        """
        if not self.use_wake_word or not self.wake_word_detector or not self.wake_word_detector.is_available:
            return True  # If wake word detection is disabled, always return True
            
        # Reset wake word state
        self.wake_word_detected = False
        
        # Start recording in wake word detection mode
        try:
            # Buffer for Porcupine frames
            frame_buffer = []
            frame_length = self.wake_word_detector.frame_length
            
            with sd.InputStream(
                samplerate=self.sample_rate, 
                channels=self.channels,
                dtype=self.dtype,
                blocksize=512,
                callback=self._audio_callback
            ):
                print("Listening for wake word...")
                
                while not self.wake_word_detected:
                    if not self.audio_queue.empty():
                        # Get audio chunk
                        audio_chunk = self.audio_queue.get()
                        
                        # Add to frame buffer
                        frame_buffer.extend(audio_chunk.flatten())
                        
                        # Process chunks of frame_length samples
                        while len(frame_buffer) >= frame_length:
                            # Extract a frame
                            frame_audio = np.array(frame_buffer[:frame_length])
                            frame_buffer = frame_buffer[frame_length:]
                            
                            # Convert to int16 for Porcupine
                            frame_int16 = (np.clip(frame_audio, -1, 1) * 32767).astype(np.int16)
                            
                            # Process frame for wake word detection
                            self.wake_word_detected = self.wake_word_detector.process_frame(frame_int16)
                            
                            if self.wake_word_detected:
                                print("Wake word detected!")
                                return True
                    
                    # Small sleep to prevent high CPU usage
                    time.sleep(0.01)
                    
            return self.wake_word_detected
            
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            return False
            
    def listen(self, listen_for_wake_word=None):
        """
        Listen for speech using webrtcvad for voice activity detection.
        If wake word detection is enabled, first listen for wake word.
        
        Args:
            listen_for_wake_word: Whether to listen for wake word first (overrides instance setting)
            
        Returns the path to the WAV file when speech is detected and finished.
        """
        # Determine if we should listen for wake word
        should_listen_for_wake_word = self.use_wake_word
        if listen_for_wake_word is not None:
            should_listen_for_wake_word = listen_for_wake_word
        
        # If wake word detection is enabled and we haven't detected it yet, listen for it
        if should_listen_for_wake_word and not self.listening_for_command:
            wake_word_detected = self.listen_for_wake_word()
            if not wake_word_detected:
                return None
            self.listening_for_command = True
        
        # Prevent multiple recording sessions
        if self.is_recording:
            print("Already recording")
            return None
            
        # Start recording in a separate thread
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._recording_thread_func)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Wait for recording to complete
        self.recording_thread.join()
        
        # Reset command listening state after recording is complete
        if self.use_wake_word:
            self.listening_for_command = False
        
        # Return the file path if speech was detected
        if self.speech_detected:
            return RECORDING_FILE
        return None
        
    def transcribe(self, audio_filename, text_filename=None):
        """
        Transcribes audio to text using faster-whisper.
        
        Args:
            audio_filename: Path to audio file
            text_filename: Path to save transcription text (if None, uses default)
            
        Returns:
            Transcribed text as string
        """
        if not audio_filename or not os.path.exists(audio_filename):
            print(f"Audio file not found: {audio_filename}")
            return None
            
        if text_filename is None:
            text_filename = TRANSCRIPTION_FILE
            
        # Make sure the directory exists for the transcription file
        os.makedirs(os.path.dirname(text_filename), exist_ok=True)
        
        try:
            # Transcribe using faster-whisper
            print("Transcribing audio...")
            segments, info = self.model.transcribe(
                audio_filename, 
                beam_size=self.beam_size,
                language=WHISPER_LANGUAGE,
                vad_filter=WHISPER_VAD_FILTER,
                vad_parameters=WHISPER_VAD_PARAMETERS
            )
            
            # Combine all segments into one text
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            
            if transcribed_text:
                print(f"User: {transcribed_text}")
                
                # Save transcription to file
                with open(text_filename, 'w') as inputtext:
                    inputtext.write(transcribed_text)
                    
                return transcribed_text
            else:
                print("No speech detected or transcription empty")
                return None
                
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None
            
    def get_device_info(self):
        """Get information about audio devices for debugging."""
        result = {
            'devices': [],
            'default_input': 'Unknown',
            'api': 'Unknown',
            'vad_mode': self.current_vad_mode,
            'error': None
        }
        
        try:
            devices = sd.query_devices()
            default_input = sd.query_devices(kind='input')
            
            result.update({
                'devices': devices,
                'default_input': default_input,
                'api': sd.query_hostapis()[0].get('name', 'Unknown'),
            })
            
        except Exception as e:
            # Log the error but return a valid dictionary
            print(f"Error getting audio device info: {e}")
            result['error'] = str(e)
            
        return result
            
    def set_vad_aggressiveness(self, mode):
        """
        Set the VAD aggressiveness level.
        
        Args:
            mode: Integer between 0 and 3. 
                 0 is least aggressive (more false positives),
                 3 is most aggressive (more false negatives)
        """
        if 0 <= mode <= 3:
            self.vad.set_mode(mode)
            self.current_vad_mode = mode
            return True
        return False

    def adjust_detection_parameters(self, speech_start_frames=None, speech_end_frames=None):
        """
        Adjust speech detection parameters dynamically.
        
        Args:
            speech_start_frames: Number of consecutive voice frames to consider speech has started
            speech_end_frames: Number of consecutive silent frames to consider speech has ended
        
        Returns:
            Dictionary with current parameter values
        """
        if speech_start_frames is not None and speech_start_frames > 0:
            self.speech_start_frames = speech_start_frames
            
        if speech_end_frames is not None and speech_end_frames > 0:
            self.speech_end_frames = speech_end_frames
            
        return {
            "vad_mode": self.current_vad_mode,
            "speech_start_frames": self.speech_start_frames,
            "speech_end_frames": self.speech_end_frames,
            "frame_duration_ms": self.frame_duration_ms
        } 