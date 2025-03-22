# D.A.I.S.Y. Voice Assistant

**D**omestic **A** **I** **S****Y**stem - A modular voice assistant powered by Ollama.

## Overview

DAISY is a voice assistant that uses local AI models through Ollama to provide a privacy-focused, customizable voice assistant experience. It features:

- Voice interaction using speech recognition and text-to-speech
- Integration with Ollama for AI responses
- Modular and extensible architecture

## Requirements

- Python 3.8 or higher
- Ollama installed and running (for AI responses)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/daisy.git
   cd daisy
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure Ollama is installed and running with the required model:
   ```
   ollama serve
   ollama pull llama3.2
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

## Project Structure

```
daisy/
├── daisy.py             # Main entry script
├── setup.py             # Setup script for installation
├── requirements.txt     # Dependencies
├── personality.txt      # Personality definition
├── README.md            # Documentation
├── src/                 # Source code
│   ├── core/            # Core assistant functionality
│   │   ├── assistant.py     # Main assistant logic
│   │   └── personality.py   # Personality management
│   ├── data/            # Data management
│   │   └── chat_history.py  # Chat history tracking
│   ├── voice/           # Voice I/O
│   │   ├── speech_recognition.py  # Speech recognition
│   │   └── text_to_speech.py      # Text-to-speech
│   ├── utils/           # Utilities
│   │   └── config.py    # Configuration settings
│   └── main.py          # Application entry point
```

## Extension and Customization

### Personality

Edit the `personality.txt` file to customize DAISY's personality and behavior.

## License

[Specify license information here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 