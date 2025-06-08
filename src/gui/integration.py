"""
Integration module to connect the GUI with the voice assistant logic.
"""
import numpy as np
import threading
import time
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread

from src.gui.main_window import JarvisGUI
from src.core.assistant import VoiceAssistant
from src.voice.speech_recognition import SpeechRecognizer
from src.voice.text_to_speech import TextToSpeech
from src.utils.config import (
    TTS_RATE, TTS_VOLUME, TTS_VOICE_ID, TTS_SPEAKER_IDX,
    WHISPER_MODEL_SIZE
)


class AudioThread(QThread):
    """Thread to continuously get audio data for visualization."""
    
    # Signal to send audio data to GUI
    audio_data_signal = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        
    def run(self):
        """Thread main loop to get audio data."""
        try:
            import sounddevice as sd
            
            # Set up audio parameters
            sample_rate = 44100  # Hz
            block_size = 1024
            channels = 1
            
            # Start stream
            with sd.InputStream(samplerate=sample_rate, blocksize=block_size, 
                              channels=channels, dtype='float32') as stream:
                while self.running:
                    # Get audio data
                    data, overflowed = stream.read(block_size)
                    if not overflowed:
                        # Process data (convert to mono if necessary)
                        if channels > 1:
                            data = np.mean(data, axis=1)
                        else:
                            data = data.flatten()
                        
                        # Send data to GUI
                        self.audio_data_signal.emit(data)
                    
                    # Sleep to avoid high CPU usage
                    time.sleep(0.01)
        except Exception as e:
            print(f"Audio thread error: {e}")
            # If audio device not available, emit synthetic data
            while self.running:
                # Generate synthetic audio data
                t = time.time()
                data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1024) + t) * 0.1
                data += np.sin(2 * np.pi * 880 * np.linspace(0, 1, 1024) + t * 2) * 0.05
                data += np.random.normal(0, 0.01, 1024)  # Add noise
                
                # Send data to GUI
                self.audio_data_signal.emit(data)
                
                # Sleep to control update rate
                time.sleep(0.05)
                
    def stop(self):
        """Stop the audio thread."""
        self.running = False
        self.wait()


class DaisyGuiIntegration(QObject):
    """Main integration class connecting the GUI with the voice assistant logic."""
    
    # Signals for communication between the backend and GUI
    transcription_signal = pyqtSignal(str)
    user_message_signal = pyqtSignal(str)
    assistant_response_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    
    def __init__(self, use_rag=True, model_name="llama3.2:latest", whisper_model_size=None, debug=False):
        super().__init__()
        
        # Debug mode
        self.debug = debug
        
        # Initialize the GUI
        self.gui = JarvisGUI()
        
        # Initialize voice assistant components
        self.voice_ai = VoiceAssistant(model_name=model_name, use_rag=use_rag)
        
        try:
            self.speech_recognizer = SpeechRecognizer(
                model_name=whisper_model_size or WHISPER_MODEL_SIZE,
                device="cpu"
            )
            
            # Debug audio devices if needed
            if debug:
                print("Audio device information:")
                device_info = self.speech_recognizer.get_device_info()
                print(device_info)
                
        except Exception as e:
            print(f"Error initializing speech recognizer: {e}")
            self.status_signal.emit(f"Error initializing speech: {str(e)}")
            raise
            
        self.tts = TextToSpeech(
            rate=TTS_RATE, 
            volume=TTS_VOLUME, 
            voice_id=TTS_VOICE_ID,
            speaker_idx=TTS_SPEAKER_IDX
        )
        
        # Set up audio thread for visualization
        self.audio_thread = AudioThread()
        
        # Connect signals and slots
        self.setup_connections()
        
        # State tracking
        self.listening = False
        self.processing_thread = None
        
    def setup_connections(self):
        """Set up signal/slot connections between GUI and backend."""
        # Connect GUI buttons to backend logic
        self.gui.start_listening_signal.connect(self.start_listening)
        self.gui.stop_listening_signal.connect(self.stop_listening)
        
        # Connect backend signals to GUI updates
        self.transcription_signal.connect(self.gui.update_transcription)
        self.user_message_signal.connect(self.gui.add_user_input)
        self.assistant_response_signal.connect(self.gui.add_assistant_response)
        self.status_signal.connect(self.gui.update_status_message)
        
        # Connect audio visualization
        self.audio_thread.audio_data_signal.connect(self.gui.update_audio_visualization)
        
    def start(self):
        """Start the GUI and background threads."""
        # Start audio visualization thread
        self.audio_thread.start()
        
        # Check Ollama connection
        if not self.check_ollama_connection():
            self.gui.llm_status.setText("LLM Connection: Offline")
            self.gui.llm_status.setStyleSheet("color: #FF0000;")
        
        # Show the GUI
        self.gui.show()
        
    def stop(self):
        """Stop all threads and cleanup."""
        self.stop_listening()
        self.audio_thread.stop()
        
    def check_ollama_connection(self):
        """Check if Ollama server is running."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
        
    @pyqtSlot()
    def start_listening(self):
        """Start the voice recognition process."""
        if self.listening:
            return
            
        self.listening = True
        self.transcription_signal.emit("Listening...")
        self.gui.audio_visualizer.set_state(listening=True)
        
        # Start listening in a separate thread to avoid blocking GUI
        self.listening_thread = threading.Thread(target=self._listening_worker)
        self.listening_thread.daemon = True
        self.listening_thread.start()
        
    def _listening_worker(self):
        """Worker function to handle voice recognition in a continuous loop."""
        while self.listening:
            try:
                # Listen for audio
                if self.debug:
                    print("Starting listening...")
                    
                audio_file = self.speech_recognizer.listen()
                
                if audio_file and self.listening:
                    # Set processing state in GUI
                    self.gui.set_processing_state()
                    
                    if self.debug:
                        print(f"Transcribing audio from {audio_file}...")
                        
                    # Transcribe audio to text
                    user_command = self.speech_recognizer.transcribe(audio_file)
                    
                    if user_command and self.listening:
                        # Update transcription in GUI
                        self.transcription_signal.emit(user_command)
                        self.user_message_signal.emit(user_command)
                        
                        # Process the command in another thread
                        self.processing_thread = threading.Thread(
                            target=self._process_command, 
                            args=(user_command,)
                        )
                        self.processing_thread.daemon = True
                        self.processing_thread.start()
                        
                        # Wait for processing to complete before listening again
                        if self.processing_thread.is_alive():
                            self.processing_thread.join()
                    else:
                        # If no command was recognized, reset to listening state
                        if self.listening:
                            self.gui.audio_visualizer.set_state(listening=True)
                            self.transcription_signal.emit("Listening...")
                else:
                    # Reset to listening state if still listening
                    if self.listening:
                        self.gui.audio_visualizer.set_state(listening=True)
                        self.transcription_signal.emit("Listening...")
            except Exception as e:
                print(f"Error in listening worker: {e}")
                self.status_signal.emit(f"Error: {str(e)}")
                # Reset state after a brief pause
                time.sleep(2)
                if self.listening:
                    self.gui.audio_visualizer.set_state(listening=True)
                    self.transcription_signal.emit("Listening...")
                    
            # Short delay to prevent CPU overuse
            time.sleep(0.1)
        
    def _process_command(self, user_command):
        """Process the recognized command and generate a response."""
        try:
            # Generate response
            response_text = self.voice_ai.get_ai_response(user_command)
            
            # Update GUI with response
            if response_text and self.listening:
                self.assistant_response_signal.emit(response_text)
                
                # Set speaking state
                self.gui.set_speaking_state()
                
                # Speak the response
                self.tts.speak(response_text)
                
                # Return to listening state if still active
                if self.listening:
                    self.gui.audio_visualizer.set_state(listening=True)
                    self.transcription_signal.emit("Listening...")
        except Exception as e:
            print(f"Error processing command: {e}")
            self.status_signal.emit(f"Error: {str(e)}")
            # Reset state
            if self.listening:
                self.gui.audio_visualizer.set_state(listening=True)
                self.transcription_signal.emit("Listening...")
                
    @pyqtSlot()
    def stop_listening(self):
        """Stop the listening process."""
        self.listening = False
        
        # Reset GUI state
        self.gui.audio_visualizer.set_state(listening=False, 
                                        processing=False, 
                                        speaking=False)
        self.transcription_signal.emit("") 