"""
Manages chat history for the DAISY assistant.
"""
import os
import json
from datetime import datetime
from src.utils.config import CHAT_HISTORY_FILE

class ChatHistory:
    def __init__(self, history_file=None):
        self.history_file = history_file or CHAT_HISTORY_FILE
        self.messages = self.load_history()
    
    def load_history(self):
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_history(self):
        # Make sure the directory exists
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        
        with open(self.history_file, 'w') as f:
            json.dump(self.messages, f, indent=2)
    
    def add_message(self, role, content):
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
        self.save_history()
    
    def get_formatted_history(self):
        return [{"role": msg["role"], "content": msg["content"]} 
                for msg in self.messages]
    
    def clear_history(self):
        self.messages = []
        self.save_history() 