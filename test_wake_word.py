#!/usr/bin/env python
"""
Test script for wake word detection in DAISY.
"""
import time
import argparse
from src.voice.speech_recognition import WakeWordDetector
from src.voice.text_to_speech import TextToSpeech

def main():
    """Test wake word detection."""
    parser = argparse.ArgumentParser(description="Test Wake Word Detection")
    parser.add_argument("--sensitivity", type=float, default=0.65,
                      help="Wake word detection sensitivity (0.0-1.0)")
    args = parser.parse_args()
    
    # Initialize wake word detector
    wake_detector = WakeWordDetector(sensitivity=args.sensitivity)
    
    if not wake_detector.is_available:
        print("ERROR: Wake word detection is not available.")
        print("Make sure you have:")
        print("1. Created a .env file with your Porcupine access key")
        print("2. Installed pvporcupine package")
        print("3. Have the wake word model file in the models directory")
        return
    
    # Initialize TTS for feedback
    tts = TextToSpeech()
    
    print(f"Wake word detector initialized with sensitivity {wake_detector.sensitivity}")
    print("Say 'Hey DAISY' to trigger the wake word detection")
    print("Press Ctrl+C to exit")
    
    try:
        import sounddevice as sd
        import numpy as np
        
        # Create audio stream
        with sd.InputStream(
            samplerate=wake_detector.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=512,
            callback=None
        ) as stream:
            print("Listening for wake word...")
            
            # Buffer to accumulate frames
            frame_buffer = []
            
            while True:
                # Read audio data
                audio_chunk, overflowed = stream.read(512)
                if overflowed:
                    print("Audio buffer overflowed")
                
                # Add to frame buffer
                frame_buffer.extend(audio_chunk.flatten())
                
                # Process complete frames
                while len(frame_buffer) >= wake_detector.frame_length:
                    # Extract a frame
                    frame = np.array(frame_buffer[:wake_detector.frame_length])
                    frame_buffer = frame_buffer[wake_detector.frame_length:]
                    
                    # Process frame through wake word detector
                    wake_word_detected = wake_detector.process_frame(frame)
                    
                    if wake_word_detected:
                        print("\nWake word detected! 🎉")
                        tts.speak("I heard you! How can I help?")
                        print("\nListening for wake word again...")
                
                # Small sleep to prevent high CPU usage
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        print("Test complete.")

if __name__ == "__main__":
    main() 