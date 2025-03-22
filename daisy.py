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
    args = parser.parse_args()
    
    # Add src directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, "src")
    sys.path.insert(0, current_dir)
    
    if args.debug:
        print(f"Debug mode enabled")
        print(f"Using model: {args.model}")
        print(f"Current directory: {current_dir}")
        print(f"Source directory: {src_dir}")
        print(f"Python path: {sys.path}")
    
    from src.main import main as daisy_main
    daisy_main()

if __name__ == "__main__":
    main() 