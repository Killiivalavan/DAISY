import speech_recognition as sr
import os
import pyttsx3
import time
import requests
import json
from datetime import datetime

assistant_name = "daisy"
should_run = True
source = sr.Microphone()
recognizer = sr.Recognizer()
trigger_word = "hey daisy"

class PersonalityManager:
    def __init__(self, personality_file="personality.txt"):
        self.personality_file = personality_file
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

class ChatHistory:
    def __init__(self, history_file="chat_history.json"):
        self.history_file = history_file
        self.messages = self.load_history()
    
    def load_history(self):
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_history(self):
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

class VoiceAssistant:
    def __init__(self):
        self.chat_history = ChatHistory()
        self.personality = PersonalityManager()
        
    def get_ai_response(self, user_input):
        # Add user message to history
        self.chat_history.add_message("user", user_input)
        
        # Prepare the messages with personality as system message
        messages = [
            {"role": "system", "content": self.personality.get_personality()}
        ] + self.chat_history.get_formatted_history()
        
        # Get response from Ollama
        try:
            response = requests.post(
                'http://localhost:11434/api/chat',
                json={
                    'model': 'llama3.2',
                    'messages': messages,
                    'stream': False
                }
            ).json()
            
            ai_response = response['message']['content']
            self.chat_history.add_message("assistant", ai_response)
            return ai_response
            
        except Exception as e:
            error_msg = f"Error getting AI response: {str(e)}"
            print(error_msg)
            return error_msg

def response(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.setProperty('volume', 1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say(text)
    engine.runAndWait()

def listen():
    with source as s:
        print('Listening...')
        recognizer.adjust_for_ambient_noise(s)
        audio = recognizer.listen(s)
        print("Audio captured successfully.")
    try:
        file_path = os.path.join(os.getcwd(), "recording.wav")
        with open(file_path, "wb") as f:
            f.write(audio.get_wav_data())
        print("audio written to recording.wav successfully")
        return file_path
    except FileNotFoundError as file_not_found_error:
        print(f"FileNotFoundError: {file_not_found_error.filename} not found")
        return None
    except Exception as file_write_error:
        print(f"Error writing to recording.wav: {file_write_error}")
        return None

def transcribe(audio_filename, text_filename):
    with sr.AudioFile(audio_filename) as audf:
        audio_data = recognizer.record(audf, duration=None)
        try:
            transcribed_text = recognizer.recognize_google(audio_data)
            print(f"User: {transcribed_text}")
            with open(text_filename, 'w') as inputtext:
                inputtext.write(transcribed_text)
            return transcribed_text
        except sr.UnknownValueError:
            print("I didn't get that...")
        except sr.RequestError as e:
            print(f"There has been a trouble accessing SpeechRecognition module: {e}")

def responseGeneration(userCmd):
    print('Thinking...')
    try:
        # Initialize VoiceAssistant if not already done
        if not hasattr(responseGeneration, 'assistant'):
            responseGeneration.assistant = VoiceAssistant()
            
        response_text = responseGeneration.assistant.get_ai_response(userCmd)
        if response_text:
            print(f"DAISY: {response_text}")
            response(response_text)
        else:
            error_msg = "I'm having trouble connecting to my brain right now. Please try again."
            print(f"DAISY: {error_msg}")
            response(error_msg)
    except Exception as e:
        print(f"Error generating response: {e}")
        print("Make sure Ollama is running with: 'ollama serve'")

def main():
    audio_file = "recording.wav"
    transcription_file = "transcription.txt"
    
    # Check if Ollama is running and model is available
    try:
        test_response = requests.get("http://localhost:11434/api/tags")
        if test_response.status_code != 200:
            raise Exception("Ollama server not responding")
    except Exception as e:
        print("Error: Could not connect to Ollama server.")
        print("Please make sure Ollama is running with: 'ollama serve'")
        return

    print("Voice assistant started. Say 'hey daisy' to begin...")
    
    while should_run:
        command = listen()
        if command:
            user_command = transcribe(audio_file, transcription_file)
            if user_command:
                if trigger_word.lower() in user_command.lower():
                    response("I'm listening!")
                else:
                    responseGeneration(user_command)
        time.sleep(1)
    response('See you later, alligator!')

if __name__ == "__main__":
    main()
