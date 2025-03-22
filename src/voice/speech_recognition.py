"""
Speech recognition functionality for DAISY.
"""
import os
import speech_recognition as sr
from src.utils.config import RECORDING_FILE, TRANSCRIPTION_FILE

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.source = sr.Microphone()
        
    def listen(self):
        """Listens for audio input and saves it to a WAV file."""
        with self.source as s:
            print('Listening...')
            self.recognizer.adjust_for_ambient_noise(s)
            audio = self.recognizer.listen(s)
            print("Audio captured successfully.")
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(RECORDING_FILE), exist_ok=True)
            
            # Use the configured path from config.py
            with open(RECORDING_FILE, "wb") as f:
                f.write(audio.get_wav_data())
            print(f"Audio written to {RECORDING_FILE} successfully")
            return RECORDING_FILE
        except FileNotFoundError as file_not_found_error:
            print(f"FileNotFoundError: {file_not_found_error.filename} not found")
            return None
        except Exception as file_write_error:
            print(f"Error writing to {RECORDING_FILE}: {file_write_error}")
            return None
        
    def transcribe(self, audio_filename, text_filename=None):
        """Transcribes audio to text using Google's speech recognition API."""
        if text_filename is None:
            text_filename = TRANSCRIPTION_FILE
            
        # Make sure the directory exists for the transcription file
        os.makedirs(os.path.dirname(text_filename), exist_ok=True)
        
        with sr.AudioFile(audio_filename) as audf:
            audio_data = self.recognizer.record(audf, duration=None)
            try:
                transcribed_text = self.recognizer.recognize_google(audio_data)
                print(f"User: {transcribed_text}")
                with open(text_filename, 'w') as inputtext:
                    inputtext.write(transcribed_text)
                return transcribed_text
            except sr.UnknownValueError:
                print("I didn't get that...")
                return None
            except sr.RequestError as e:
                print(f"There has been a trouble accessing SpeechRecognition module: {e}")
                return None 