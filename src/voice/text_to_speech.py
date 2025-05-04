"""
Text-to-speech functionality for DAISY.
"""
import os
import logging
import re
import torch
import pyttsx3
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add espeak-ng directory to PATH if not already there
espeak_path = "C:\\Program Files\\eSpeak NG"
if os.path.exists(espeak_path) and espeak_path not in os.environ.get("PATH", ""):
    logger.info(f"Adding {espeak_path} to PATH for espeak-ng")
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + espeak_path

class TextToSpeech:
    def __init__(self, rate=150, volume=1.0, voice_id=1, use_coqui=True, 
                 model_name="tts_models/en/vctk/vits", speaker_idx="p250"):
        """
        Initialize the TTS engine with Coqui-AI TTS (primary) and pyttsx3 (fallback).
        
        Args:
            rate: Speech rate for pyttsx3 (words per minute)
            volume: Volume level (0.0 to 1.0)
            voice_id: Voice ID for pyttsx3
            use_coqui: Whether to try using Coqui-AI TTS (set to False to force pyttsx3)
            model_name: Coqui TTS model to use (default is VITS with VCTK voices)
            speaker_idx: Speaker ID for multi-speaker models (p250 is the selected voice)
        """
        # Flag to track which engine is active
        self.using_coqui = False
        self.coqui_tts = None
        self.speaker_idx = speaker_idx
        
        # Try to initialize Coqui-AI TTS if requested
        if use_coqui:
            try:
                logger.info(f"Initializing Coqui-AI TTS with model {model_name}...")
                # Explicitly check if espeak-ng is accessible
                espeak_found = False
                for path_dir in os.environ.get("PATH", "").split(os.pathsep):
                    if os.path.exists(os.path.join(path_dir, "espeak-ng.exe")):
                        espeak_found = True
                        logger.info(f"Found espeak-ng at: {os.path.join(path_dir, 'espeak-ng.exe')}")
                        break
                
                if not espeak_found:
                    logger.warning("espeak-ng not found in PATH. Trying direct check...")
                    if os.path.exists(os.path.join(espeak_path, "espeak-ng.exe")):
                        logger.info(f"Found espeak-ng at expected location: {os.path.join(espeak_path, 'espeak-ng.exe')}")
                        # Add it to PATH again to be sure
                        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + espeak_path
                    else:
                        logger.error(f"espeak-ng not found at {os.path.join(espeak_path, 'espeak-ng.exe')}")
                        raise FileNotFoundError("espeak-ng.exe not found in PATH or expected location")
                
                # Lazy import to avoid import errors if TTS is not installed
                from TTS.api import TTS
                
                # Get device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                
                # Initialize TTS
                self.coqui_tts = TTS(model_name).to(device)
                
                # Validate speaker_idx if it's a multi-speaker model
                if hasattr(self.coqui_tts, "speakers") and self.coqui_tts.speakers:
                    if speaker_idx not in self.coqui_tts.speakers:
                        available_speakers = self.coqui_tts.speakers
                        logger.warning(f"Speaker {speaker_idx} not found. Available speakers: {available_speakers}")
                        # Use the first available speaker if the requested one isn't available
                        self.speaker_idx = available_speakers[0]
                        logger.info(f"Using speaker {self.speaker_idx} instead")
                    else:
                        logger.info(f"Using speaker: {speaker_idx}")
                
                self.using_coqui = True
                logger.info("Coqui-AI TTS initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Coqui-AI TTS: {e}")
                logger.info("Falling back to pyttsx3")
                self.using_coqui = False
        
        # Initialize pyttsx3 as fallback
        if not self.using_coqui:
            logger.info("Initializing pyttsx3...")
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            voices = self.engine.getProperty('voices')
            self.engine.setProperty('voice', voices[voice_id].id)
            logger.info("pyttsx3 initialized successfully")
        
        # Define patterns for text cleaning
        self.cleaning_patterns = [
            # Remove Markdown formatting
            (r'\*\*(.*?)\*\*', r'\1'),  # Bold: **text** -> text
            (r'\*(.*?)\*', r'\1'),      # Italic: *text* -> text
            (r'__(.*?)__', r'\1'),      # Bold: __text__ -> text
            (r'_(.*?)_', r'\1'),        # Italic: _text_ -> text
            (r'~~(.*?)~~', r'\1'),      # Strikethrough: ~~text~~ -> text
            
            # Handle code blocks and inline code
            (r'```[a-z]*\n(.*?)\n```', r'\1'),  # Code blocks
            (r'`(.*?)`', r'\1'),                # Inline code
            
            # Handle URLs
            (r'https?://[^\s]+', r'link'),      # URLs -> "link"
            
            # Handle list markers
            (r'^\s*[-*+]\s+', r'• '),           # List item markers
            (r'^\s*\d+\.\s+', r'• '),           # Numbered list markers
            
            # Handle quotes
            (r'^\s*>\s+', r'quote: '),          # Quote markers
            
            # Clean up extra whitespace
            (r'\n{3,}', r'\n\n'),               # Multiple newlines
            (r'\s{2,}', r' '),                  # Multiple spaces
            
            # Handle special characters
            (r'&', r'and'),                    # & -> "and"
            (r'@', r'at'),                     # @ -> "at"
            (r'#', r'hashtag'),                # # -> "hashtag"
            (r'[$€£¥]', r'currency'),          # Currency symbols
            
            # Document citation format from RAG
            (r'\[Document \d+.*?\]', r'According to the document:'),  # Replace citation tags
        ]
        
        # Common abbreviations for expansion
        self.abbreviations = {
            "e.g.": "for example",
            "i.e.": "that is",
            "etc.": "etcetera",
            "vs.": "versus",
            "fig.": "figure",
            "Dr.": "Doctor",
            "Mr.": "Mister",
            "Mrs.": "Misses",
            "Prof.": "Professor",
            "PhD": "P H D",
            "URL": "U R L",
            "API": "A P I",
            "HTML": "H T M L",
            "CSS": "C S S",
            "NASA": "NASA",
        }
    
    def list_available_voices(self):
        """List available voices from the current TTS engine."""
        if self.using_coqui:
            if hasattr(self.coqui_tts, "speakers") and self.coqui_tts.speakers:
                return self.coqui_tts.speakers
            else:
                return ["Single speaker model"]
        else:
            voices = self.engine.getProperty('voices')
            return [voice.name for voice in voices]
    
    def set_voice(self, speaker_idx=None, voice_id=None):
        """
        Change the voice used by the TTS engine.
        
        Args:
            speaker_idx: Speaker ID for Coqui-AI TTS (e.g., "p250" for VCTK dataset)
            voice_id: Voice ID index for pyttsx3 (e.g., 0 for male, 1 for female in Windows)
            
        Returns:
            bool: True if voice was changed successfully, False otherwise
        """
        if self.using_coqui and speaker_idx:
            if hasattr(self.coqui_tts, "speakers") and self.coqui_tts.speakers:
                if speaker_idx in self.coqui_tts.speakers:
                    self.speaker_idx = speaker_idx
                    logger.info(f"Voice changed to: {speaker_idx}")
                    return True
                else:
                    available_speakers = self.coqui_tts.speakers
                    logger.warning(f"Speaker {speaker_idx} not found. Available speakers: {available_speakers}")
                    return False
            else:
                logger.warning("Current TTS model doesn't support multiple speakers")
                return False
        elif not self.using_coqui and voice_id is not None:
            try:
                voices = self.engine.getProperty('voices')
                if 0 <= voice_id < len(voices):
                    self.engine.setProperty('voice', voices[voice_id].id)
                    logger.info(f"Voice changed to: {voices[voice_id].name}")
                    return True
                else:
                    logger.warning(f"Voice ID {voice_id} out of range. Available voices: 0-{len(voices)-1}")
                    return False
            except Exception as e:
                logger.error(f"Error changing pyttsx3 voice: {e}")
                return False
        else:
            logger.warning("No valid voice specified for the current TTS engine")
            return False
        
    def clean_text_for_speech(self, text: str) -> str:
        """
        Clean and preprocess text for better speech synthesis.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text ready for speech synthesis
        """
        if not text:
            return ""
        
        # Apply all regex cleaning patterns
        cleaned_text = text
        for pattern, replacement in self.cleaning_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.MULTILINE)
        
        # Expand common abbreviations
        words = cleaned_text.split()
        for i, word in enumerate(words):
            # Check if the word (stripped of punctuation) is in abbreviations
            clean_word = word.strip('.,;:!?()[]{}')
            if clean_word in self.abbreviations:
                # Replace while preserving trailing punctuation
                trailing_punct = word[len(clean_word):]
                words[i] = self.abbreviations[clean_word] + trailing_punct
        
        # Rejoin the text
        cleaned_text = ' '.join(words)
        
        # Additional cleanup for readability
        cleaned_text = cleaned_text.replace(' ,', ',')
        cleaned_text = cleaned_text.replace(' .', '.')
        cleaned_text = cleaned_text.replace(' !', '!')
        cleaned_text = cleaned_text.replace(' ?', '?')
        cleaned_text = cleaned_text.replace(' :', ':')
        cleaned_text = cleaned_text.replace(' ;', ';')
        
        return cleaned_text
    
    def speak(self, text: str):
        """
        Converts text to speech and plays it.
        
        Args:
            text: The text to convert to speech
        """
        # Preprocess text before speaking
        cleaned_text = self.clean_text_for_speech(text)
        
        if not cleaned_text:
            logger.warning("Empty text provided to speak method, skipping")
            return
            
        try:
            # Use Coqui-AI TTS if available
            if self.using_coqui:
                logger.debug(f"Speaking with Coqui-AI TTS: {cleaned_text[:30]}...")
                
                # Create a temporary file for the audio
                import tempfile
                import os
                import sounddevice as sd
                import soundfile as sf
                
                # Create a temporary WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                logger.info(f"Saving audio to temporary file: {temp_path}")
                
                # Use the proper TTS API method to generate audio directly to a file
                if hasattr(self.coqui_tts, "tts_to_file"):
                    # Determine arguments based on model capabilities
                    if hasattr(self.coqui_tts, "speakers") and self.coqui_tts.speakers:
                        # Multi-speaker model
                        self.coqui_tts.tts_to_file(
                            text=cleaned_text, 
                            speaker=self.speaker_idx,
                            file_path=temp_path
                        )
                    else:
                        # Single-speaker model
                        self.coqui_tts.tts_to_file(
                            text=cleaned_text,
                            file_path=temp_path
                        )
                else:
                    # Generate audio with tts() method and save it manually
                    if hasattr(self.coqui_tts, "speakers") and self.coqui_tts.speakers:
                        # Multi-speaker model
                        wav = self.coqui_tts.tts(text=cleaned_text, speaker=self.speaker_idx)
                    else:
                        # Single-speaker model
                        wav = self.coqui_tts.tts(text=cleaned_text)
                    
                    # Save the audio manually
                    import numpy as np
                    sample_rate = 22050  # Default for TTS
                    
                    # Process the returned audio data
                    if isinstance(wav, np.ndarray):
                        sf.write(temp_path, wav, sample_rate)
                    elif isinstance(wav, list) and len(wav) > 0:
                        # If it's a list of audio arrays, join them
                        try:
                            wav_array = np.concatenate([np.array(chunk) for chunk in wav])
                            sf.write(temp_path, wav_array, sample_rate)
                        except:
                            logger.warning("Could not concatenate audio chunks, using first chunk")
                            sf.write(temp_path, np.array(wav[0]), sample_rate)
                
                # Play the audio file - this will block until audio is finished
                try:
                    logger.info(f"Playing audio from {temp_path}")
                    data, fs = sf.read(temp_path)
                    sd.play(data, fs)
                    sd.wait()  # This blocks until audio is finished
                    logger.info("Audio playback completed")
                finally:
                    # Clean up the temporary file
                    try:
                        os.unlink(temp_path)
                        logger.debug(f"Removed temporary file: {temp_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove temporary file: {e}")
            else:
                # Fall back to pyttsx3
                logger.debug(f"Speaking with pyttsx3: {cleaned_text[:30]}...")
                self.engine.say(cleaned_text)
                self.engine.runAndWait()
                
        except Exception as e:
            logger.error(f"Error in TTS: {e}")
            
            # If Coqui fails, fallback to pyttsx3 for this particular text
            if self.using_coqui:
                logger.info("Temporarily falling back to pyttsx3 for this text")
                try:
                    if not hasattr(self, 'engine'):
                        self.engine = pyttsx3.init()
                        self.engine.setProperty('rate', 200)
                        self.engine.setProperty('volume', 1.0)
                        voices = self.engine.getProperty('voices')
                        self.engine.setProperty('voice', voices[1].id)
                    
                    self.engine.say(cleaned_text)
                    self.engine.runAndWait()
                except Exception as fallback_error:
                    logger.error(f"Fallback TTS also failed: {fallback_error}") 