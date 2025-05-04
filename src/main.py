"""
Main entry point for the DAISY voice assistant.
"""
import time
import requests
import argparse

from src.utils.config import (
    ASSISTANT_NAME, TRIGGER_WORD, RECORDING_FILE, 
    TRANSCRIPTION_FILE, TTS_RATE, TTS_VOLUME, TTS_VOICE_ID, TTS_SPEAKER_IDX
)
from src.core.assistant import VoiceAssistant
from src.voice.speech_recognition import SpeechRecognizer
from src.voice.text_to_speech import TextToSpeech

class DaisyAssistant:
    """Main DAISY voice assistant class."""
    
    def __init__(self, use_rag=True, model_name="llama3.2:latest"):
        """Initialize components."""
        self.should_run = True
        self.voice_ai = VoiceAssistant(model_name=model_name, use_rag=use_rag)
        self.speech_recognizer = SpeechRecognizer()
        self.tts = TextToSpeech(
            rate=TTS_RATE, 
            volume=TTS_VOLUME, 
            voice_id=TTS_VOICE_ID,
            speaker_idx=TTS_SPEAKER_IDX
        )
        
    def check_ollama_server(self):
        """Check if Ollama server is running."""
        try:
            test_response = requests.get("http://localhost:11434/api/tags")
            return test_response.status_code == 200
        except Exception:
            return False
    
    def process_command(self, user_command):
        """Process user command and generate response."""
        print('Thinking...')
        try:
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
            print("Make sure Ollama is running with: 'ollama serve'")
    
    def run(self):
        """Main loop for the voice assistant."""
        # Check if Ollama is running
        if not self.check_ollama_server():
            print("Error: Could not connect to Ollama server.")
            print("Please make sure Ollama is running with: 'ollama serve'")
            return
        
        print(f"Voice assistant started. Say '{TRIGGER_WORD}' to begin...")
        
        while self.should_run:
            # Listen for audio
            audio_file = self.speech_recognizer.listen()
            
            if audio_file:
                # Transcribe audio to text
                user_command = self.speech_recognizer.transcribe(audio_file)
                
                if user_command:
                    # Check for trigger word
                    if TRIGGER_WORD.lower() in user_command.lower():
                        self.tts.speak("I'm listening!")
                    else:
                        # Process the command
                        self.process_command(user_command)
            
            # Short delay to prevent CPU overuse
            time.sleep(0.5)
        
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
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    use_rag = not args.no_rag
    
    # Create assistant
    assistant = DaisyAssistant(use_rag=use_rag, model_name=args.model)
    
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