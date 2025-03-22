"""
Main voice assistant class for DAISY.
"""
import requests
from src.core.personality import PersonalityManager
from src.data.chat_history import ChatHistory

class VoiceAssistant:
    def __init__(self, model_name="llama3.2:latest"):
        self.chat_history = ChatHistory()
        self.personality = PersonalityManager()
        self.model_name = model_name
        
    def get_ai_response(self, user_input):
        # Add user message to history
        self.chat_history.add_message("user", user_input)
        
        # Prepare the messages with personality as system message
        messages = [
            {"role": "system", "content": self.personality.get_personality()}
        ] + self.chat_history.get_formatted_history()
        
        # Get response from Ollama
        try: 
            print("Attempting to connect to Ollama...")
            response = requests.post(
                'http://localhost:11434/api/chat',
                json={
                    'model': self.model_name,
                    'messages': messages,
                    'stream': False
                },
                timeout=10  # Add timeout to avoid hanging
            )
            print(f"Response status code: {response.status_code}")
            response_json = response.json()
            ai_response = response_json['message']['content']
            self.chat_history.add_message("assistant", ai_response)
            return ai_response
            
        except requests.exceptions.ConnectionError:
            error_msg = "Could not connect to Ollama. Is it running?"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error getting AI response: {str(e)}"
            print(error_msg)
            return error_msg 