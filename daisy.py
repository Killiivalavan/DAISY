#!/usr/bin/env python
"""
DAISY Voice Assistant - Run Script
"""
import os
import sys
import argparse

def main():
    """Main entry point for the DAISY voice assistant launcher."""
    parser = argparse.ArgumentParser(description="DAISY Voice Assistant")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--model", type=str, default="llama3.2:latest", 
                        help="Specify the Ollama model to use")
    parser.add_argument("--no-rag", action="store_true", 
                        help="Disable RAG (Retrieval-Augmented Generation)")
    parser.add_argument("--process-docs", action="store_true",
                        help="Process documents and exit")
    parser.add_argument("--no-wake-word", action="store_true",
                        help="Disable wake word detection even if available")
    parser.add_argument("--wake-word-sensitivity", type=float, 
                        help="Wake word detection sensitivity (0.0-1.0)")
    parser.add_argument("--whisper-model", type=str, 
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Specify the Whisper model size (default: from config)")
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], 
                        help="Set VAD aggressiveness (0=least aggressive, 3=most aggressive)")
    args = parser.parse_args()
    
    # Add src directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, "src")
    sys.path.insert(0, current_dir)
    
    if args.debug:
        print(f"Debug mode enabled")
        print(f"Using model: {args.model}")
        print(f"RAG enabled: {not args.no_rag}")
        print(f"Wake word detection: {'disabled' if args.no_wake_word else 'enabled'}")
        if args.wake_word_sensitivity:
            print(f"Wake word sensitivity: {args.wake_word_sensitivity}")
        if args.whisper_model:
            print(f"Whisper model: {args.whisper_model}")
        print(f"Current directory: {current_dir}")
        print(f"Source directory: {src_dir}")
        print(f"Python path: {sys.path}")
    
    from src.main import main as daisy_main
    daisy_main()

if __name__ == "__main__":
    main() 