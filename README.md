# D.A.I.S.Y. Voice Assistant

**D**omestic **A** **I** **S** **Y**stem - A modular voice assistant powered by Ollama.

## Overview

DAISY is a voice assistant that uses local AI models through Ollama to provide a privacy-focused, customizable voice assistant experience. It features:

- Voice interaction using speech recognition and text-to-speech
- High-quality speech synthesis with Coqui-AI TTS (with fallback to pyttsx3)
- Integration with Ollama for AI responses
- Modular and extensible architecture

## Requirements

- Python 3.10 (recommended for optimal compatibility with TTS library)
- Ollama installed and running (for AI responses)
- PyTorch (installed automatically with requirements.txt)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/daisy.git
   cd daisy
   ```

2. Create a Python 3.10 virtual environment (recommended):
   ```
   python3.10 -m venv venv_py310
   # On Windows
   venv_py310\Scripts\activate
   # On macOS/Linux
   source venv_py310/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Make sure Ollama is installed and running with the required model:
   ```
   ollama serve
   ollama pull llama3.2
   ```

5. Install espeak-NG:
   ```
   navigate to https://github.com/espeak-ng/espeak-ng/releases
   download the latest release for your system. (usually the .msi file)
   run the installer
   make sure to add it to the environment variables
   ```

## Usage

To start DAISY:

```
python daisy.py
```

Or, for additional options:

```
python daisy.py --debug --model llama3.2:latest
```

Once running, activate DAISY by saying "Hey DAISY" and then ask your question.

### Wake Word Detection

DAISY supports wake word detection using Picovoice's Porcupine engine, allowing a true hands-free experience:

1. **Setup**:
   - Copy `.env.example` to `.env`
   - Get a free Porcupine access key from [Picovoice Console](https://console.picovoice.ai/)
   - Add your access key to the `.env` file

2. **Usage**:
   - Start DAISY normally
   - Say "Hey DAISY" to activate (no need to press any keys)
   - DAISY will indicate when it's listening for your command
   - Speak your command after wake word detection

3. **Command-line Options**:
   - `--no-wake-word`: Disable wake word detection (use manual activation)
   - `--wake-word-sensitivity=0.7`: Adjust wake word detection sensitivity (0.0-1.0)

The wake word model is already included in the `models` directory.

### Testing the TTS Engine

To test the TTS engine separately:

```
python test_tts.py --text "Hello, I am testing the TTS engine."
```

Additional options:
- `--no-coqui`: Force using pyttsx3 instead of Coqui-AI TTS
- `--list-voices`: List available voices
- `--model MODEL_NAME`: Specify a different Coqui-AI TTS model
- `--speaker SPEAKER_ID`: Specify a different speaker for multi-speaker models

## Text-to-Speech Options

DAISY now supports two TTS engines:

1. **Coqui-AI TTS (Primary)**: High-quality, natural-sounding speech using deep learning models
   - Default model: VITS with VCTK dataset (provides British accent options)
   - Automatically uses GPU acceleration when available
   - Multiple voices available (British and other accents)

2. **pyttsx3 (Fallback)**: Traditional TTS engine
   - Used as a fallback if Coqui-AI TTS fails to initialize or encounters an error
   - Lower quality but more reliable on systems with limited resources

The system automatically attempts to use Coqui-AI TTS first and falls back to pyttsx3 if needed.

## Project Structure

```
daisy/
├── daisy.py                 # Main CLI entry script
├── daisy_gui.py            # GUI entry script
├── test_*.py               # Test scripts
├── test_comprehensive.py   # Comprehensive test suite
├── setup.py                # Setup script for installation
├── requirements.txt        # Dependencies
├── personality.txt         # Personality definition
├── README.md               # Documentation
├── src/                    # Source code
│   ├── core/               # Core assistant functionality
│   │   ├── assistant.py         # Main assistant logic
│   │   └── personality.py       # Personality management
│   ├── data/               # Data management
│   │   └── chat_history.py      # Chat history tracking
│   ├── voice/              # Voice I/O
│   │   ├── speech_recognition.py # Speech recognition
│   │   └── text_to_speech.py     # Text-to-speech
│   ├── rag/                # RAG (Retrieval-Augmented Generation)
│   │   ├── document_loader.py    # Document loading
│   │   ├── document_processor.py # Document processing
│   │   ├── embedding_generator.py # Embedding generation
│   │   ├── vector_store.py       # Vector storage
│   │   ├── retriever.py          # Document retrieval
│   │   └── document_tracker.py   # Document tracking
│   ├── gui/                # GUI components
│   │   ├── main_window.py        # Main GUI window
│   │   └── integration.py        # GUI integration layer
│   ├── utils/              # Utilities
│   │   ├── config.py             # Configuration settings
│   │   ├── config_manager.py     # Configuration management
│   │   ├── connection_manager.py # Ollama connection management
│   │   ├── resource_manager.py   # Resource cleanup management
│   │   └── error_handler.py      # Error handling utilities
│   └── main.py             # Application entry point
```

## Testing

Run the comprehensive test suite to verify all components:

```
python test_comprehensive.py
```

Individual component tests are also available:
- `python test_daisy.py` - Basic functionality test
- `python test_tts.py` - Text-to-speech testing
- `python test_wake_word.py` - Wake word detection testing

## Extension and Customization

### Personality

Edit the `personality.txt` file to customize DAISY's personality and behavior.

### TTS Voice Selection

To change the default voice:
- Edit `src/voice/text_to_speech.py` and change the `speaker_idx` parameter in the TextToSpeech initialization
- Use `test_tts.py --list-voices` to see available voices

## License

[Specify license information here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
