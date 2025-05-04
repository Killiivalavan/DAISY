#!/usr/bin/env python
"""
Test script for the enhanced TextToSpeech functionality.
"""
import sys
import os
import logging
import argparse
import time

# Add the project root to Python path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.voice.text_to_speech import TextToSpeech

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TTS_Test")

# Print environment info
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PATH environment variable: {os.environ.get('PATH', '')}")

def main():
    """Main entry point for the TTS test script."""
    parser = argparse.ArgumentParser(description="Test the TTS functionality")
    parser.add_argument("--text", type=str, default="Hello, I am DAISY. I'm using the new VITS text to speech engine which has a more natural voice.", 
                      help="Text to synthesize")
    parser.add_argument("--model", type=str, default="tts_models/en/vctk/vits", 
                      help="Coqui TTS model to use")
    parser.add_argument("--speaker", type=str, default="p250", 
                      help="Speaker ID for multi-speaker models (p250 is the selected voice)")
    parser.add_argument("--no-coqui", action="store_true",
                      help="Disable Coqui TTS and use pyttsx3 only")
    parser.add_argument("--list-voices", action="store_true",
                      help="List available voices and exit")
    parser.add_argument("--voice-type", type=str, choices=['male', 'female', 'all'],
                      help="Try a selection of voices of a particular type")
    
    args = parser.parse_args()
    
    try:
        # Initialize TextToSpeech
        logger.info("Initializing TextToSpeech...")
        tts = TextToSpeech(
            use_coqui=not args.no_coqui,
            model_name=args.model,
            speaker_idx=args.speaker
        )
        
        # List available voices if requested
        if args.list_voices:
            voices = tts.list_available_voices()
            logger.info("Available voices:")
            for voice in voices:
                logger.info(f"- {voice}")
            return
        
        # Voice type selection feature
        if args.voice_type:
            # Common female voices in VCTK
            female_voices = ["p225", "p228", "p229", "p231", "p276", "p294", "p302"]
            # Common male voices in VCTK
            male_voices = ["p236", "p241", "p245", "p260", "p270", "p304", "p311"]
            
            selected_voices = []
            if args.voice_type == 'male':
                selected_voices = male_voices
                logger.info("Testing male voices...")
            elif args.voice_type == 'female':
                selected_voices = female_voices
                logger.info("Testing female voices...")
            else:  # 'all'
                selected_voices = male_voices + female_voices
                logger.info("Testing all voice types...")
            
            for voice in selected_voices:
                if tts.set_voice(speaker_idx=voice):
                    logger.info(f"Speaking with voice: {voice}")
                    tts.speak(f"Hello, this is voice sample {voice}.")
                    time.sleep(0.5)
            return
        
        # Log which engine is being used
        engine_name = "Coqui-AI TTS" if tts.using_coqui else "pyttsx3"
        logger.info(f"Using TTS engine: {engine_name}")
        
        if tts.using_coqui and hasattr(tts, "speaker_idx"):
            logger.info(f"Using speaker: {tts.speaker_idx}")
        
        # Record start time
        start_time = time.time()
        
        # Synthesize speech
        logger.info(f"Speaking: {args.text}")
        tts.speak(args.text)
        
        # Calculate time taken
        elapsed_time = time.time() - start_time
        # Calculate the real-time factor (processing time / audio duration)
        # Assuming average speech rate of 150 words per minute, and 5 characters per word
        char_count = len(args.text)
        word_count = char_count / 5
        est_duration = (word_count / 150) * 60  # Duration in seconds
        rtf = elapsed_time / est_duration
        
        logger.info(f"Processing time: {elapsed_time:.2f} seconds")
        logger.info(f"Estimated audio duration: {est_duration:.2f} seconds")
        logger.info(f"Real-time factor: {rtf:.2f}")
        logger.info(f"Using Coqui TTS: {tts.using_coqui}")
        
    except Exception as e:
        logger.error(f"Error in TTS test: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 