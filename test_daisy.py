#!/usr/bin/env python
"""
Test script for DAISY voice assistant.
Used to verify that all components are working properly.
"""
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

try:
    # Test imports
    from src.core.assistant import VoiceAssistant
    from src.voice.speech_recognition import SpeechRecognizer
    from src.voice.text_to_speech import TextToSpeech
    from src.data.chat_history import ChatHistory
    from src.core.personality import PersonalityManager
    from src.utils.config import ASSISTANT_NAME, TRIGGER_WORD
    
    print("✓ All modules imported successfully")
    
    # Test initialization
    print("\nInitializing components...")
    va = VoiceAssistant()
    sr = SpeechRecognizer()
    tts = TextToSpeech()
    ch = ChatHistory()
    pm = PersonalityManager()
    
    print("✓ All components initialized successfully")
    
    # Print personality
    print(f"\nPersonality loaded: {pm.get_personality()[:60]}...")
    
    # Print config
    print(f"\nAssistant name: {ASSISTANT_NAME}")
    print(f"Trigger word: {TRIGGER_WORD}")
    
    print("\nAll tests passed! The DAISY voice assistant structure is working properly.")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 