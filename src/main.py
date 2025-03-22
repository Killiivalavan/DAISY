"""
Main entry point for the DAISY voice assistant.
"""
import time
import requests

from src.utils.config import (
    ASSISTANT_NAME, TRIGGER_WORD, RECORDING_FILE, 
    TRANSCRIPTION_FILE
)
from src.core.assistant import VoiceAssistant
from src.voice.speech_recognition import SpeechRecognizer
from src.voice.text_to_speech import TextToSpeech

class DaisyAssistant:
    """Main DAISY voice assistant class."""
    
    def __init__(self):
        """Initialize components."""
        self.should_run = True
        self.voice_ai = VoiceAssistant()
        self.speech_recognizer = SpeechRecognizer()
        self.tts = TextToSpeech()
        
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

def main():
    """Main entry point."""
    assistant = DaisyAssistant()
    assistant.run()

if __name__ == "__main__":
    main() 