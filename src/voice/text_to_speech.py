"""
Text-to-speech functionality for DAISY.
"""
import os
import logging
import re
import torch
import pyttsx3
from typing import Dict, List, Tuple, Optional
import time
import soundfile as sf
from src.utils.resource_manager import managed_temp_file, get_resource_tracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add espeak-ng directory to PATH if not already there
espeak_path = "C:\\Program Files\\eSpeak NG"
if os.path.exists(espeak_path) and espeak_path not in os.environ.get("PATH", ""):
    logger.info(f"Adding {espeak_path} to PATH for espeak-ng")
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + espeak_path

class TextToSpeech:
    def __init__(self, rate=180, volume=1.0, voice_id=1, use_coqui=True, 
                 speaker_idx=None, cache_size=100):
        """Initialize TTS with caching and optimization."""
        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id
        self.use_coqui = use_coqui
        self.speaker_idx = speaker_idx
        
        # TTS caching
        self.cache_dir = os.path.join(os.path.dirname(__file__), "cache", "tts")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_size = cache_size
        self.cache = {}
        
        # Initialize TTS engine
        if use_coqui:
            try:
                from TTS.api import TTS
                logger.info("Initializing Coqui-AI TTS with model tts_models/en/vctk/vits...")
                
                # Check for espeak-ng
                espeak_path = self._find_espeak()
                if espeak_path:
                    logger.info(f"Found espeak-ng at: {espeak_path}")
                    os.environ["PHONEMIZER_ESPEAK_PATH"] = espeak_path
                
                # Initialize TTS with optimized settings
                self.tts = TTS(
                    model_name="tts_models/en/vctk/vits",
                    progress_bar=False,  # Disable progress bar for faster initialization
                    gpu=torch.cuda.is_available()  # Use GPU if available
                )
                
                # Set speaker
                if speaker_idx is not None:
                    logger.info(f"Using speaker: p{speaker_idx}")
                    self.speaker_idx = speaker_idx
                
                logger.info("Coqui-AI TTS initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Coqui TTS: {e}")
                self.use_coqui = False
                self._init_pyttsx3()
        else:
            self._init_pyttsx3()
        
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
        if self.use_coqui:
            if hasattr(self.tts, "speakers") and self.tts.speakers:
                return self.tts.speakers
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
        if self.use_coqui and speaker_idx:
            if hasattr(self.tts, "speakers") and self.tts.speakers:
                if speaker_idx in self.tts.speakers:
                    self.speaker_idx = speaker_idx
                    logger.info(f"Voice changed to: {speaker_idx}")
                    return True
                else:
                    available_speakers = self.tts.speakers
                    logger.warning(f"Speaker {speaker_idx} not found. Available speakers: {available_speakers}")
                    return False
            else:
                logger.warning("Current TTS model doesn't support multiple speakers")
                return False
        elif not self.use_coqui and voice_id is not None:
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
    
    def _get_cache_key(self, text, rate=None, volume=None):
        """Generate a cache key for TTS output."""
        # Normalize text and parameters
        text = text.strip().lower()
        rate = rate or self.rate
        volume = volume or self.volume
        return hash(f"{text}:{rate}:{volume}:{self.voice_id}:{self.speaker_idx}")
        
    def _get_cached_audio(self, cache_key):
        """Get cached audio file if it exists."""
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.wav")
        if os.path.exists(cache_file):
            self.cache[cache_key] = cache_file
            return cache_file
        return None
        
    def _cache_audio(self, cache_key, audio_file):
        """Cache audio file."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            try:
                os.remove(self.cache[oldest_key])
            except:
                pass
            del self.cache[oldest_key]
            
        self.cache[cache_key] = audio_file
        
    def speak(self, text, block=True, rate=None, volume=None):
        """Optimized speech synthesis with caching."""
        if not text:
            return
            
        # Generate cache key
        cache_key = self._get_cache_key(text, rate, volume)
        
        # Check cache first
        cached_file = self._get_cached_audio(cache_key)
        if cached_file:
            logger.debug("Using cached TTS output")
            self._play_audio_file(cached_file, block)
            return
            
        # Generate new audio
        try:
            # Split text into sentences for better processing
            sentences = self._split_into_sentences(text)
            logger.debug(f"Text splitted to sentences.\n{sentences}")
            
            # Process sentences in parallel if not blocking
            if not block and len(sentences) > 1:
                import threading
                threads = []
                for sentence in sentences:
                    thread = threading.Thread(
                        target=self._process_sentence,
                        args=(sentence, rate, volume)
                    )
                    thread.daemon = True
                    thread.start()
                    threads.append(thread)
                    
                # Wait for all threads if blocking
                if block:
                    for thread in threads:
                        thread.join()
                return
                
            # Process single sentence or blocking mode
            audio_file = self._process_sentence(text, rate, volume)
            
            # Cache the result
            if audio_file:
                self._cache_audio(cache_key, audio_file)
                
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            # Fallback to pyttsx3
            if self.use_coqui:
                logger.info("Falling back to pyttsx3")
                self.use_coqui = False
                self._init_pyttsx3()
                self.speak(text, block, rate, volume)
                
    def _process_sentence(self, text, rate=None, volume=None):
        """Process a single sentence with timing information."""
        start_time = time.time()
        
        try:
            if self.use_coqui:
                # Generate unique filename
                temp_file = os.path.join(
                    self.cache_dir,
                    f"temp_{int(time.time() * 1000)}_{hash(text)}.wav"
                )
                
                # Ensure rate is not None and is a valid number
                rate = rate or self.rate
                if not isinstance(rate, (int, float)) or rate <= 0:
                    rate = self.rate
                
                # Generate speech with optimized settings
                self.tts.tts_to_file(
                    text=text,
                    file_path=temp_file,
                    speaker=self.speaker_idx,
                    speed=max(0.5, min(2.0, rate/180.0))  # Clamp speed between 0.5x and 2.0x
                )
                
                # Play the audio
                self._play_audio_file(temp_file, True)
                
                # Log timing information
                processing_time = time.time() - start_time
                audio_duration = len(sf.read(temp_file)[0]) / self.tts.synthesizer.output_sample_rate
                rtf = processing_time / audio_duration
                logger.debug(f"Processing time: {processing_time}")
                logger.debug(f"Real-time factor: {rtf}")
                
                return temp_file
            else:
                # Use pyttsx3
                rate = rate or self.rate
                if not isinstance(rate, (int, float)) or rate <= 0:
                    rate = self.rate
                    
                self.engine.setProperty('rate', rate)
                self.engine.setProperty('volume', volume or self.volume)
                self.engine.setProperty('voice', self.voices[self.voice_id].id)
                
                if block:
                    self.engine.say(text)
                    self.engine.runAndWait()
                else:
                    self.engine.say(text)
                    self.engine.startLoop(False)
                    
        except Exception as e:
            logger.error(f"Error processing sentence: {e}")
            return None
    
    def _play_audio_file(self, file_path, block=True):
        """
        Play audio file using appropriate platform-specific method.
        
        Args:
            file_path: Path to audio file
            block: Whether to block until audio is complete
        """
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return
            
        try:
            # Use platform-specific audio playback
            if os.name == 'nt':  # Windows
                # Use winsound for Windows
                import winsound
                if block:
                    winsound.PlaySound(file_path, winsound.SND_FILENAME)
                else:
                    import threading
                    threading.Thread(
                        target=winsound.PlaySound, 
                        args=(file_path, winsound.SND_FILENAME),
                        daemon=True
                    ).start()
            else:  # macOS, Linux, etc.
                # Try to use playsound, fallback to sox/aplay
                try:
                    from playsound import playsound
                    if block:
                        playsound(file_path)
                    else:
                        import threading
                        threading.Thread(target=playsound, args=(file_path,), daemon=True).start()
                except ImportError:
                    # Fallback to system commands
                    import subprocess
                    import platform
                    
                    system = platform.system()
                    if system == 'Darwin':  # macOS
                        cmd = ['afplay', file_path]
                    else:  # Linux and others
                        # Try aplay, then mpg123, then fall back to sox
                        if self._command_exists('aplay'):
                            cmd = ['aplay', file_path]
                        elif self._command_exists('mpg123'):
                            cmd = ['mpg123', file_path]
                        else:
                            cmd = ['play', file_path]  # sox command
                    
                    if block:
                        subprocess.call(cmd)
                    else:
                        subprocess.Popen(cmd)
        except Exception as e:
            logger.error(f"Error playing audio file: {e}")
    
    def _command_exists(self, cmd):
        """Check if a command exists by trying to run it."""
        try:
            subprocess.check_call(['which', cmd], 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _find_espeak(self):
        """Find espeak-ng installation path."""
        # Check common installation paths
        possible_paths = [
            "C:\\Program Files\\eSpeak NG",
            "C:\\Program Files (x86)\\eSpeak NG",
            "/usr/bin",
            "/usr/local/bin",
            "/opt/espeak-ng"
        ]
        
        # Check PATH first
        for path_dir in os.environ.get("PATH", "").split(os.pathsep):
            if os.path.exists(os.path.join(path_dir, "espeak-ng.exe")):
                return os.path.join(path_dir, "espeak-ng.exe")
            elif os.path.exists(os.path.join(path_dir, "espeak-ng")):
                return os.path.join(path_dir, "espeak-ng")
                
        # Check common installation paths
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "espeak-ng.exe")):
                return os.path.join(path, "espeak-ng.exe")
            elif os.path.exists(os.path.join(path, "espeak-ng")):
                return os.path.join(path, "espeak-ng")
                
        logger.warning("espeak-ng not found in common locations")
        return None
        
    def _init_pyttsx3(self):
        """Initialize pyttsx3 as fallback TTS engine."""
        try:
            logger.info("Initializing pyttsx3...")
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            self.voices = self.engine.getProperty('voices')
            self.engine.setProperty('voice', self.voices[self.voice_id].id)
            logger.info("pyttsx3 initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing pyttsx3: {e}")
            raise
            
    def _split_into_sentences(self, text):
        """Split text into sentences for better processing."""
        # Simple sentence splitting on common punctuation
        sentences = []
        current = []
        
        for char in text:
            current.append(char)
            if char in '.!?':
                sentences.append(''.join(current).strip())
                current = []
                
        if current:  # Add any remaining text
            sentences.append(''.join(current).strip())
            
        return [s for s in sentences if s]  # Remove empty sentences 