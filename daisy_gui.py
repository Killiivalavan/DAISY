#!/usr/bin/env python
"""
DAISY Voice Assistant - GUI Launcher
This script launches the JARVIS-inspired GUI for the DAISY voice assistant.
"""
import sys
import argparse
from PyQt6.QtWidgets import QApplication
from src.gui.integration import DaisyGuiIntegration
from src.voice.speech_recognition import SpeechRecognizer

def main():
    """Main entry point for the DAISY GUI launcher."""
    parser = argparse.ArgumentParser(description="DAISY Voice Assistant GUI")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--model", type=str, default="llama3.2:latest", 
                        help="Specify the Ollama model to use")
    parser.add_argument("--no-rag", action="store_true", 
                        help="Disable RAG (Retrieval-Augmented Generation)")
    parser.add_argument("--whisper-model", type=str, 
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Specify the Whisper model size for speech recognition")
    parser.add_argument("--audio-info", action="store_true",
                        help="Print audio device information and exit")
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], 
                        help="Set VAD aggressiveness (0=least aggressive, 3=most aggressive)")
    parser.add_argument("--speech-start", type=int, 
                        help="Number of voice frames to consider speech started (default: 2)")
    parser.add_argument("--speech-end", type=int, 
                        help="Number of silent frames to consider speech ended (default: 15)")
    args = parser.parse_args()
    
    # Handle audio info request
    if args.audio_info:
        print("Audio device information:")
        # Create a temporary recognizer just to get device info
        recognizer = SpeechRecognizer()
        print(recognizer.get_device_info())
        return
    
    # Print debug info if requested
    if args.debug:
        print(f"Debug mode enabled")
        print(f"Using model: {args.model}")
        print(f"RAG enabled: {not args.no_rag}")
        if args.whisper_model:
            print(f"Using Whisper model: {args.whisper_model}")
        if args.vad_mode is not None:
            print(f"VAD aggressiveness: {args.vad_mode}")
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("DAISY Voice Assistant")
    
    # Create and start the integration
    integration = DaisyGuiIntegration(
        use_rag=not args.no_rag,
        model_name=args.model,
        whisper_model_size=args.whisper_model,
        debug=args.debug
    )
    
    # Set VAD aggressiveness if specified
    if args.vad_mode is not None:
        integration.speech_recognizer.set_vad_aggressiveness(args.vad_mode)
    
    # Adjust speech detection parameters if specified
    if args.speech_start is not None or args.speech_end is not None:
        params = integration.speech_recognizer.adjust_detection_parameters(
            speech_start_frames=args.speech_start,
            speech_end_frames=args.speech_end
        )
        if args.debug:
            print(f"Speech detection parameters: {params}")
    
    integration.start()
    
    # Run the application event loop
    exit_code = app.exec()
    
    # Clean up
    integration.stop()
    
    # Exit with the application exit code
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 