"""
Text-to-speech functionality for DAISY.
"""
import pyttsx3

class TextToSpeech:
    def __init__(self, rate=200, volume=1.0, voice_id=1):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[voice_id].id)
        
    def speak(self, text):
        """Converts text to speech and plays it."""
        self.engine.say(text)
        self.engine.runAndWait() 