"""
Text-to-speech functionality for DAISY.
"""
import pyttsx3
import re
from typing import Dict, List, Tuple

class TextToSpeech:
    def __init__(self, rate=200, volume=1.0, voice_id=1):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[voice_id].id)
        
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
        
        # Speak the cleaned text
        self.engine.say(cleaned_text)
        self.engine.runAndWait() 