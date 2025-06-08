"""
Configuration settings for DAISY.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
if os.path.exists(env_file):
    load_dotenv(env_file)

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
TTS_RATE = 180
TTS_VOLUME = 1.0
TTS_VOICE_ID = 1  # Female voice for pyttsx3
TTS_SPEAKER_IDX = "p277"  # Voice ID for Coqui TTS (VCTK dataset)

# Speech recognition settings (faster-whisper)
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large-v2, large-v3
WHISPER_BEAM_SIZE = 3  # Reduced beam size for faster processing
WHISPER_LANGUAGE = "en"  # Language code (None for auto-detection)
WHISPER_VAD_FILTER = True  # Voice Activity Detection to filter out non-speech
WHISPER_VAD_PARAMETERS = {
    "min_silence_duration_ms": 1000,  # Increased for better silence detection
    "speech_pad_ms": 400,  # Padding around speech segments
}
WHISPER_MODEL_PATH = os.path.join(BASE_DIR, "models")  # Model cache directory

# WebRTC VAD settings for speech detection - Optimized based on best practices
WEBRTC_VAD_MODE = 1  # Aggressiveness (0-3): 1 provides good balance between sensitivity and false positives
WEBRTC_FRAME_DURATION_MS = 30  # Frame size in ms (10, 20, or 30) - 30ms provides good balance
WEBRTC_SPEECH_START_FRAMES = 3  # Number of voiced frames to consider speech started
WEBRTC_SPEECH_END_FRAMES = 20  # Number of silent frames to consider speech ended (reduced from 25)
WEBRTC_MAX_RECORDING_SECS = 30  # Maximum recording time - reduced for faster processing

# Porcupine Wake Word settings
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY", "")
PORCUPINE_MODEL_PATH = os.getenv("WAKE_WORD_MODEL_PATH", os.path.join(BASE_DIR, "models", "hey-daisy_en_windows_v3_0_0.ppn"))
PORCUPINE_SENSITIVITY = 0.65  # Detection sensitivity (0.0-1.0)
PORCUPINE_ENABLED = bool(PORCUPINE_ACCESS_KEY)  # Only enable if access key is available
USE_WAKE_WORD = True  # Set to False to disable wake word detection even if available

# RAG settings
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
CHUNK_SIZE = 500  # Reduced chunk size for quicker processing
CHUNK_OVERLAP = 50  # Reduced overlap for quicker processing
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_DOCS_TO_RETRIEVE = 3 
DOCUMENT_TRACKING_FILE = os.path.join(DATA_DIR, "document_tracking.json") 