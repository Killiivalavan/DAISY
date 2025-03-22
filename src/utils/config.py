"""
Configuration settings for DAISY.
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Assistant configuration
ASSISTANT_NAME = "daisy"
TRIGGER_WORD = "hey daisy"
DEFAULT_MODEL = "llama3.2:latest"

# File paths
PERSONALITY_FILE = os.path.join(BASE_DIR, "personality.txt")
CHAT_HISTORY_FILE = os.path.join(DATA_DIR, "chat_history.json")
RECORDING_FILE = os.path.join(DATA_DIR, "recording.wav")
TRANSCRIPTION_FILE = os.path.join(DATA_DIR, "transcription.txt")

# Text-to-speech settings
TTS_RATE = 200
TTS_VOLUME = 1.0
TTS_VOICE_ID = 1  # Female voice 