"""
Centralized configuration management for DAISY.
"""
import os
import json
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path
from src.utils.error_handler import ConfigurationError, safe_execute

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages all configuration settings for DAISY."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent.parent
        self.config_cache = {}
        self.env_cache = {}
        
        # Default configuration values
        self.defaults = {
            # Assistant settings
            'ASSISTANT_NAME': 'DAISY',
            'TRIGGER_WORD': 'hey daisy',
            
            # File paths
            'RECORDING_FILE': 'data/recording.wav',
            'TRANSCRIPTION_FILE': 'data/transcription.txt',
            'CHAT_HISTORY_FILE': 'data/chat_history.json',
            'PERSONALITY_FILE': 'personality.txt',
            'DOCUMENT_TRACKING_FILE': 'data/document_tracking.json',
            'CACHE_DIR': 'cache',
            'VECTOR_DB_DIR': 'data/vector_db',
            'DOCUMENTS_DIR': 'data/documents',
            
            # TTS settings
            'TTS_RATE': 180,
            'TTS_VOLUME': 1.0,
            'TTS_VOICE_ID': 1,
            'TTS_SPEAKER_IDX': 'p277',
            'USE_COQUI_TTS': True,
            
            # Whisper settings
            'WHISPER_MODEL_SIZE': 'base',
            'WHISPER_BEAM_SIZE': 5,
            'WHISPER_LANGUAGE': 'en',
            'WHISPER_VAD_FILTER': True,
            'WHISPER_VAD_PARAMETERS': {
                'threshold': 0.5,
                'min_speech_duration_ms': 250,
                'max_speech_duration_s': 30,
                'min_silence_duration_ms': 2000,
                'window_size_samples': 1024,
                'speech_pad_ms': 400
            },
            
            # WebRTC VAD settings
            'WEBRTC_VAD_MODE': 2,
            'WEBRTC_FRAME_DURATION_MS': 30,
            'WEBRTC_SPEECH_START_FRAMES': 3,
            'WEBRTC_SPEECH_END_FRAMES': 10,
            'WEBRTC_MAX_RECORDING_SECS': 30,
            
            # Wake word settings
            'USE_WAKE_WORD': False,
            'PORCUPINE_ENABLED': False,
            'PORCUPINE_ACCESS_KEY': '',
            'PORCUPINE_MODEL_PATH': 'models/hey_computer_windows.ppn',
            'PORCUPINE_SENSITIVITY': 0.5,
            
            # RAG settings
            'MAX_DOCS_TO_RETRIEVE': 5,
            'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
            'CHUNK_SIZE': 1000,
            'CHUNK_OVERLAP': 200,
        }
        
        # Initialize configuration
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration from various sources."""
        # Load from environment variables
        self._load_env_config()
        
        # Load from config file if it exists
        config_file = self.config_dir / 'config.json'
        if config_file.exists():
            self._load_json_config(config_file)
    
    def _load_env_config(self):
        """Load configuration from environment variables."""
        for key in self.defaults:
            env_value = os.getenv(key)
            if env_value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(env_value, self.defaults[key])
                self.env_cache[key] = converted_value
                logger.debug(f"Loaded {key} from environment: {converted_value}")
    
    def _convert_env_value(self, env_value: str, default_value: Any) -> Any:
        """Convert environment variable string to appropriate type."""
        if isinstance(default_value, bool):
            return env_value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(default_value, int):
            try:
                return int(env_value)
            except ValueError:
                logger.warning(f"Could not convert {env_value} to int, using string")
                return env_value
        elif isinstance(default_value, float):
            try:
                return float(env_value)
            except ValueError:
                logger.warning(f"Could not convert {env_value} to float, using string")
                return env_value
        elif isinstance(default_value, dict):
            try:
                return json.loads(env_value)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse {env_value} as JSON, using string")
                return env_value
        else:
            return env_value
    
    def _load_json_config(self, config_file: Path):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self.config_cache.update(file_config)
                logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with precedence: env > config file > defaults.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Check environment variables first
        if key in self.env_cache:
            return self.env_cache[key]
        
        # Check config file cache
        if key in self.config_cache:
            return self.config_cache[key]
        
        # Check defaults
        if key in self.defaults:
            return self.defaults[key]
        
        # Return provided default or None
        return default
    
    def set(self, key: str, value: Any, persist: bool = False):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            persist: Whether to save to config file
        """
        self.config_cache[key] = value
        
        if persist:
            self.save_config()
    
    def save_config(self):
        """Save current configuration to file."""
        config_file = self.config_dir / 'config.json'
        
        try:
            # Ensure directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(self.config_cache, f, indent=2)
            
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def validate_paths(self):
        """Validate that all required paths exist or can be created."""
        path_keys = [
            'RECORDING_FILE', 'TRANSCRIPTION_FILE', 'CHAT_HISTORY_FILE',
            'DOCUMENT_TRACKING_FILE', 'CACHE_DIR', 'VECTOR_DB_DIR', 'DOCUMENTS_DIR'
        ]
        
        for key in path_keys:
            path_value = self.get(key)
            if path_value:
                path = Path(path_value)
                
                # Create directory if it's a directory or parent directory for files
                if key.endswith('_DIR'):
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
    
    def validate_wake_word_config(self) -> bool:
        """Validate wake word configuration."""
        if not self.get('USE_WAKE_WORD'):
            return True
        
        if not self.get('PORCUPINE_ENABLED'):
            logger.warning("Wake word requested but Porcupine is not enabled")
            return False
        
        access_key = self.get('PORCUPINE_ACCESS_KEY')
        if not access_key:
            logger.warning("Wake word enabled but no Porcupine access key provided")
            return False
        
        model_path = Path(self.get('PORCUPINE_MODEL_PATH'))
        if not model_path.exists():
            logger.warning(f"Wake word model not found at {model_path}")
            return False
        
        return True
    
    def validate_personality_file(self) -> bool:
        """Validate personality file exists."""
        personality_file = Path(self.get('PERSONALITY_FILE'))
        if not personality_file.exists():
            logger.warning(f"Personality file not found at {personality_file}")
            # Create default personality file
            try:
                personality_file.parent.mkdir(parents=True, exist_ok=True)
                with open(personality_file, 'w') as f:
                    f.write("You are DAISY, a helpful voice assistant. You provide concise, friendly responses.")
                logger.info(f"Created default personality file at {personality_file}")
                return True
            except Exception as e:
                logger.error(f"Could not create personality file: {e}")
                return False
        return True
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all current settings."""
        settings = {}
        
        # Start with defaults
        settings.update(self.defaults)
        
        # Override with config file values
        settings.update(self.config_cache)
        
        # Override with environment variables
        settings.update(self.env_cache)
        
        return settings
    
    def validate_all(self) -> bool:
        """Validate all configuration settings."""
        all_valid = True
        
        # Validate paths
        try:
            self.validate_paths()
        except Exception as e:
            logger.error(f"Path validation failed: {e}")
            all_valid = False
        
        # Validate wake word configuration
        if not self.validate_wake_word_config():
            all_valid = False
        
        # Validate personality file
        if not self.validate_personality_file():
            all_valid = False
        
        return all_valid

# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value using global config manager."""
    return get_config_manager().get(key, default)

def set_config(key: str, value: Any, persist: bool = False):
    """Set configuration value using global config manager."""
    get_config_manager().set(key, value, persist)

def validate_config() -> bool:
    """Validate all configuration using global config manager."""
    return get_config_manager().validate_all() 