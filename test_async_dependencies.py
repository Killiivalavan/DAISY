#!/usr/bin/env python
"""
Test script to verify dependencies for DAISY async pipeline.
"""

import sys
import subprocess

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - Available")
        return True
    except ImportError as e:
        package = package_name or module_name
        print(f"❌ {module_name} - Missing (install with: pip install {package})")
        return False

def test_ollama_connection():
    """Test connection to Ollama server."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            print(f"✅ Ollama server - Connected ({len(models)} models available)")
            if "llama3.2:latest" in model_names:
                print("  ✅ llama3.2:latest model found")
            else:
                print("  ⚠️  llama3.2:latest model not found")
                print("     Run: ollama pull llama3.2:latest")
            return True
        else:
            print(f"❌ Ollama server - HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama server - Not reachable ({e})")
        print("   Start with: ollama serve")
        return False

def test_audio_devices():
    """Test audio device availability."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        
        print(f"✅ Audio devices - {len(input_devices)} input, {len(output_devices)} output")
        
        default_input = sd.query_devices(kind='input')
        default_output = sd.query_devices(kind='output')
        print(f"   Default input: {default_input['name']}")
        print(f"   Default output: {default_output['name']}")
        
        return len(input_devices) > 0 and len(output_devices) > 0
    except Exception as e:
        print(f"❌ Audio devices - Error ({e})")
        return False

def main():
    """Test all dependencies for DAISY async pipeline."""
    print("🔍 Testing DAISY Async Pipeline Dependencies")
    print("=" * 50)
    
    all_good = True
    
    # Core Python modules
    print("\n📦 Core Dependencies:")
    all_good &= test_import("asyncio")
    all_good &= test_import("numpy")
    all_good &= test_import("sounddevice")
    all_good &= test_import("soundfile")
    all_good &= test_import("aiohttp")
    
    # AI/ML modules
    print("\n🤖 AI/ML Dependencies:")
    all_good &= test_import("faster_whisper", "faster-whisper")
    
    # TTS modules
    print("\n🔊 TTS Dependencies:")
    tts_available = test_import("TTS")
    if not tts_available:
        print("   Fallback to pyttsx3...")
        all_good &= test_import("pyttsx3")
    
    # System tests
    print("\n🔧 System Tests:")
    all_good &= test_audio_devices()
    all_good &= test_ollama_connection()
    
    print("\n" + "=" * 50)
    if all_good:
        print("✅ All dependencies are ready!")
        print("🚀 You can run: python daisy_async.py")
    else:
        print("❌ Some dependencies are missing")
        print("📋 Install missing packages and ensure Ollama is running")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main()) 