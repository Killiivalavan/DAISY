"""
Main entry point for the DAISY voice assistant.
"""
import time
import requests
import argparse

from src.utils.config import (
    ASSISTANT_NAME, TRIGGER_WORD, RECORDING_FILE, 
    TRANSCRIPTION_FILE, TTS_RATE, TTS_VOLUME, TTS_VOICE_ID, TTS_SPEAKER_IDX,
    WHISPER_MODEL_SIZE, WHISPER_BEAM_SIZE, WHISPER_LANGUAGE, WHISPER_VAD_FILTER,
    WHISPER_VAD_PARAMETERS, USE_WAKE_WORD, PORCUPINE_ENABLED, PORCUPINE_SENSITIVITY
)
from src.core.assistant import VoiceAssistant
from src.voice.speech_recognition import SpeechRecognizer
from src.voice.text_to_speech import TextToSpeech

class DaisyAssistant:
    """Main DAISY voice assistant class."""
    
    def __init__(self, use_rag=True, model_name="llama3.2:latest", whisper_model_size=None, 
                 debug=False, use_wake_word=None, wake_word_sensitivity=None):
        """Initialize components."""
        self.should_run = True
        self.debug = debug
        
        # Initialize text-to-speech and speech recognition components first
        # so we can provide audio feedback during initialization
        self.tts = TextToSpeech(
            rate=TTS_RATE, 
            volume=TTS_VOLUME, 
            voice_id=TTS_VOICE_ID,
            speaker_idx=TTS_SPEAKER_IDX
        )
        
        # Use provided whisper_model_size or default from config
        self.speech_recognizer = SpeechRecognizer(
            model_size=whisper_model_size or WHISPER_MODEL_SIZE,
            beam_size=WHISPER_BEAM_SIZE,
            use_wake_word=use_wake_word
        )
        
        # Configure wake word
        self.use_wake_word = use_wake_word if use_wake_word is not None else USE_WAKE_WORD
        if self.use_wake_word:
            # Update wake word sensitivity if provided
            if wake_word_sensitivity is not None and self.speech_recognizer.wake_word_detector:
                self.speech_recognizer.wake_word_detector.sensitivity = wake_word_sensitivity
        
        # First check if Ollama is accessible
        if not self.check_ollama_server():
            print("Warning: Could not connect to Ollama server. Will retry when needed.")
        
        # Initialize voice assistant with Ollama connection
        print("Initializing voice assistant...")
        self.voice_ai = VoiceAssistant(model_name=model_name, use_rag=use_rag)
        
        # In debug mode, print audio device information
        if debug:
            print("Audio device information:")
            device_info = self.speech_recognizer.get_device_info()
            
            # Check if there was an error getting device info
            if device_info.get('error'):
                print(f"Could not retrieve complete audio device info: {device_info['error']}")
            
            # Safely access device info with fallbacks
            default_input = "Unknown"
            if isinstance(device_info.get('default_input'), dict):
                default_input = device_info['default_input'].get('name', "Unknown")
            elif device_info.get('default_input'):
                default_input = str(device_info['default_input'])
                
            print(f"Default input device: {default_input}")
            print(f"Audio API: {device_info.get('api', 'Unknown')}")
            print(f"Sample rate: {self.speech_recognizer.sample_rate}")
            print(f"Whisper model size: {whisper_model_size or WHISPER_MODEL_SIZE}")
            
            if self.use_wake_word:
                print(f"Wake word detection: {'enabled' if self.speech_recognizer.use_wake_word else 'disabled'}")
                if self.speech_recognizer.wake_word_detector:
                    print(f"Wake word sensitivity: {self.speech_recognizer.wake_word_detector.sensitivity}")
        
    def check_ollama_server(self):
        """Check if Ollama server is running."""
        try:
            test_response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return test_response.status_code == 200
        except Exception:
            return False
    
    def process_command(self, user_command):
        """Process user command and generate response."""
        try:
            # Give immediate feedback that we're processing
            self.tts.speak("Let me think about that", block=False)
            print('Thinking...')
            
            # Check for special commands
            if "process documents" in user_command.lower() or "index documents" in user_command.lower():
                force_reprocess = "force" in user_command.lower() or "reprocess all" in user_command.lower()
                print(f"Processing documents for RAG (force_reprocess={force_reprocess})...")
                response_text = self.voice_ai.process_documents(force_reprocess)
            else:
                # Normal response generation
                response_text = self.voice_ai.get_ai_response(user_command)
                
            if response_text:
                print(f"{ASSISTANT_NAME.upper()}: {response_text}")
                self.tts.speak(response_text)
            else:
                error_msg = "I'm having trouble connecting to my brain right now. Please try again."
                print(f"{ASSISTANT_NAME.upper()}: {error_msg}")
                self.tts.speak(error_msg)
        except Exception as e:
            print(f"Error generating response: {e}")
            error_msg = "I encountered an error while processing your request."
            print(f"{ASSISTANT_NAME.upper()}: {error_msg}")
            self.tts.speak(error_msg)
            if self.debug:
                print("Make sure Ollama is running with: 'ollama serve'")
    
    def run(self):
        """Main loop for the voice assistant."""
        # Check if Ollama was pre-initialized successfully
        ollama_status = "available" if self.voice_ai.ollama_available else "unavailable"
        
        wake_word_status = "enabled" if self.use_wake_word and self.speech_recognizer.use_wake_word else "disabled"
        print(f"Voice assistant started. Wake word detection is {wake_word_status}.")
        print(f"Ollama connection is {ollama_status}.")
        
        # If Ollama or model connection failed, try to get available models for suggestion
        if not self.voice_ai.ollama_available:
            print("Will attempt to connect to Ollama when needed.")
            print("Make sure Ollama is running with: 'ollama serve'")
            
            # Try to get a list of available models
            try:
                response = requests.get('http://localhost:11434/api/tags', timeout=2)
                if response.status_code == 200:
                    tags_data = response.json()
                    available_models = [model["name"] for model in tags_data.get("models", [])]
                    if available_models:
                        print("\nAvailable models:")
                        for i, model in enumerate(available_models):
                            print(f"  {i+1}. {model}")
                        print(f"\nTo use a specific model, restart with: python daisy.py --model MODEL_NAME")
            except:
                # Silently fail if we can't get the model list
                pass
        
        if not self.use_wake_word:
            print(f"Say '{TRIGGER_WORD}' to begin...")
        
        # Provide audio feedback that we're ready
        if self.voice_ai.ollama_available:
            self.tts.speak("I'm ready to assist you")
        
        while self.should_run:
            # Listen for audio
            if self.debug:
                print("Starting listening...")
                
            audio_file = self.speech_recognizer.listen()
            
            if audio_file:
                # Transcribe audio to text
                if self.debug:
                    print(f"Transcribing audio from {audio_file}...")
                    
                user_command = self.speech_recognizer.transcribe(audio_file)
                
                if user_command:
                    if self.use_wake_word:
                        # With wake word, we can process the command directly
                        self.process_command(user_command)
                    else:
                        # Without wake word, check for trigger word in the command
                        if TRIGGER_WORD.lower() in user_command.lower():
                            self.tts.speak("I'm listening!")
                        else:
                            # Process the command
                            self.process_command(user_command)
            
            # Short delay to prevent CPU overuse
            time.sleep(0.1)
        
        self.tts.speak('See you later, alligator!')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DAISY Voice Assistant")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--model", type=str, default="llama3.2:latest", 
                      help="Specify the Ollama model to use")
    parser.add_argument("--no-rag", action="store_true", 
                      help="Disable RAG (Retrieval-Augmented Generation)")
    parser.add_argument("--process-docs", action="store_true",
                      help="Process documents and exit")
    parser.add_argument("--force-reprocess", action="store_true",
                      help="Force reprocessing of all documents")
    parser.add_argument("--whisper-model", type=str, choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                      help="Specify the Whisper model size (default: from config)")
    parser.add_argument("--audio-info", action="store_true",
                      help="Print audio device information and exit")
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], 
                      help="Set VAD aggressiveness (0=least aggressive, 3=most aggressive)")
    parser.add_argument("--speech-start", type=int, 
                      help="Number of voice frames to consider speech started (default: 2)")
    parser.add_argument("--speech-end", type=int, 
                      help="Number of silent frames to consider speech ended (default: 15)")
    parser.add_argument("--no-wake-word", action="store_true",
                      help="Disable wake word detection even if available")
    parser.add_argument("--wake-word-sensitivity", type=float, 
                      help="Wake word detection sensitivity (0.0-1.0)")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    use_rag = not args.no_rag
    use_wake_word = None if not args.no_wake_word else False
    
    # Create assistant
    assistant = DaisyAssistant(
        use_rag=use_rag, 
        model_name=args.model,
        whisper_model_size=args.whisper_model,
        debug=args.debug,
        use_wake_word=use_wake_word,
        wake_word_sensitivity=args.wake_word_sensitivity
    )
    
    # Handle audio info request
    if args.audio_info:
        print("Audio device information:")
        device_info = assistant.speech_recognizer.get_device_info()
        
        if device_info.get('error'):
            print(f"Warning: {device_info['error']}")
        
        # Print device list in a readable format
        print("\nAvailable audio devices:")
        devices = device_info.get('devices', [])
        if not devices:
            print("  No devices found or could not query device list")
        else:
            for i, device in enumerate(devices):
                device_type = "Input" if device.get('max_input_channels', 0) > 0 else "Output"
                name = device.get('name', f"Device {i}")
                print(f"  {i}: {name} ({device_type})")
        
        # Print default input device
        default_input = "Unknown"
        if isinstance(device_info.get('default_input'), dict):
            default_input = device_info['default_input'].get('name', "Unknown")
        elif device_info.get('default_input'):
            default_input = str(device_info['default_input'])
            
        print(f"\nDefault input device: {default_input}")
        print(f"Audio API: {device_info.get('api', 'Unknown')}")
        print(f"VAD mode: {device_info.get('vad_mode', 'Unknown')}")
        return
    
    # Set VAD aggressiveness if specified
    if args.vad_mode is not None:
        assistant.speech_recognizer.set_vad_aggressiveness(args.vad_mode)
        if args.debug:
            print(f"VAD aggressiveness set to {args.vad_mode}")
    
    # Adjust speech detection parameters if specified
    if args.speech_start is not None or args.speech_end is not None:
        params = assistant.speech_recognizer.adjust_detection_parameters(
            speech_start_frames=args.speech_start,
            speech_end_frames=args.speech_end
        )
        if args.debug:
            print(f"Speech detection parameters: {params}")
    
    # Handle document processing option
    if args.process_docs:
        print("Processing documents for RAG...")
        result = assistant.voice_ai.process_documents(force_reprocess=args.force_reprocess)
        print(result)
        return
    
    # Run the assistant
    assistant.run()

if __name__ == "__main__":
    main() 