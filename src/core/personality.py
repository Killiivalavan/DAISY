"""
Manages the DAISY assistant's personality.
"""
from src.utils.config import PERSONALITY_FILE

class PersonalityManager:
    def __init__(self, personality_file=None):
        self.personality_file = personality_file or PERSONALITY_FILE
        self.personality = self.load_personality()
    
    def load_personality(self):
        try:
            with open(self.personality_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Personality file {self.personality_file} not found. Using default personality.")
            return "You are an intelligent assistant named DAISY. You provide helpful and concise responses."
    
    def get_personality(self):
        return self.personality 