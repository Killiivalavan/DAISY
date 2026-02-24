# D.A.I.S.Y. Voice Assistant

**D**omestic **A**rtificial **I**ntelligence **SY**stem

*An advanced, privacy-focused voice assistant inspired by J.A.R.V.I.S., powered by local AI models*

---

## Overview

DAISY is a sophisticated voice assistant that combines the power of local AI models with advanced speech processing to provide a privacy-focused, customizable assistant experience. Built with a modular architecture, DAISY offers enterprise-grade features while maintaining complete data privacy through local processing.

### Key Features

- **Advanced Voice Recognition**: Faster-Whisper models with WebRTC VAD for precise speech detection
- **High-Quality Text-to-Speech**: Coqui-AI TTS with British accent and natural voice synthesis
- **Wake Word Detection**: Hands-free activation using Picovoice Porcupine engine
- **RAG-Powered Intelligence**: Document-aware responses using local vector database
- **Complete Privacy**: All processing happens locally - no data leaves your machine
- **Customizable Personality**: J.A.R.V.I.S.-inspired with configurable behavior
- **Multi-Interface**: Command-line and GUI support
- **Highly Configurable**: Extensive customization options for all components

---

## Architecture Overview

DAISY follows a modular, layered architecture designed for extensibility and maintainability:

```
┌─────────────────────────────────────────────┐
│                 User Interface              │
│           (CLI / GUI / Voice)               │
├─────────────────────────────────────────────┤
│               Core Assistant                │
│        (Personality, Chat History)          │
├─────────────────────────────────────────────┤
│     Voice Processing     │    AI & RAG      │
│  • Speech Recognition   │  • Ollama LLM     │
│  • Text-to-Speech      │  • Vector Store   │
│  • Wake Word Detection │  • Doc Processing │
├─────────────────────────────────────────────┤
│              Utilities & Config             │
│   (Error Handling, Resource Management)     │
└─────────────────────────────────────────────┘
```

### Core Components

#### **Voice Processing Pipeline**
- **Speech Recognition**: Faster-Whisper models (tiny/base/small/medium/large)
- **Voice Activity Detection**: WebRTC VAD with configurable aggressiveness
- **Wake Word Engine**: Porcupine with custom "Hey DAISY" model
- **Text-to-Speech**: Coqui-AI TTS with VCTK dataset (British voices)

#### **AI & Knowledge System**
- **Language Model**: Ollama integration (Llama 3.2, Mistral, etc.)
- **RAG System**: FAISS vector database with sentence-transformers
- **Document Processing**: PDF extraction with chunking and embedding
- **Personality Engine**: Configurable J.A.R.V.I.S.-inspired character

#### **Infrastructure**
- **Resource Management**: Automatic cleanup and memory optimization
- **Error Handling**: Graceful degradation with fallback mechanisms
- **Configuration**: Environment-based settings with validation
- **Caching**: TTS and model caching for performance

---

## Installation & Setup

### Prerequisites

- **Python 3.10+** (recommended for optimal TTS compatibility)
- **Ollama** (for AI model hosting)
- **espeak-NG** (for phoneme generation)
- **FFmpeg** (for audio processing)
- **Git LFS** (if using model files)

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space (for models and cache)
- **Audio**: Microphone and speakers/headphones
- **OS**: Windows 10/11, macOS 10.15+, Linux (Ubuntu 20.04+)

### Installation Steps

#### 1. **Clone Repository**
```bash
git clone https://github.com/Killiivalavan/DAISY.git
cd DAISY
```

#### 2. **Set Up Python Environment**
```bash
# Create virtual environment with Python 3.10
python3.10 -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

#### 3. **Install Dependencies**
```bash
# Install core dependencies
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
```

#### 4. **Install System Dependencies**

**Windows:**
```bash
# Download and install espeak-NG from:
# https://github.com/espeak-ng/espeak-ng/releases
# Add to PATH: C:\Program Files\eSpeak NG

# Install FFmpeg:
# https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install espeak-ng ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install espeak-ng ffmpeg libportaudio2 libasound2-dev
```

#### 5. **Install & Configure Ollama**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull recommended model (in new terminal)
ollama pull llama3.2:latest
```

#### 6. **Configure Wake Word Detection** *(Optional)*
```bash
# Copy environment template
cp .env.example .env

# Get free Porcupine access key from:
# https://console.picovoice.ai/

# Add to .env file:
echo "PORCUPINE_ACCESS_KEY=your_access_key_here" >> .env
```

---

## Configuration

### Environment Variables (`.env`)

```bash
# Porcupine Wake Word Settings
PORCUPINE_ACCESS_KEY=your_access_key_here
WAKE_WORD_MODEL_PATH=models/hey-daisy_en_windows_v3_0_0.ppn

# Optional: Custom model paths
WHISPER_MODEL_PATH=models/
OLLAMA_BASE_URL=http://localhost:11434
```

### Configuration Files

#### **`src/utils/config.py`** - Main Configuration
```python
# Assistant Settings
ASSISTANT_NAME = "daisy"
TRIGGER_WORD = "hey daisy"
DEFAULT_MODEL = "llama3.2:latest"

# Speech Recognition
WHISPER_MODEL_SIZE = "base"  # tiny, base, small, medium, large-v2, large-v3
WHISPER_LANGUAGE = "en"
WEBRTC_VAD_MODE = 1  # 0-3 (aggressiveness)

# Text-to-Speech
TTS_RATE = 180
TTS_VOLUME = 1.0
TTS_SPEAKER_IDX = "p277"  # British female voice

# RAG Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_DOCS_TO_RETRIEVE = 3
```

#### **`personality.txt`** - Assistant Personality
```text
You are DAISY, an elite, highly intelligent AI assistant inspired by J.A.R.V.I.S. 
from the Marvel universe. You speak with calm precision, dry wit, and unwavering 
professionalism. Your tone is distinctly British—polite, articulate, and 
occasionally clever...
```

---

## Usage Guide

### Basic Usage

#### **Start DAISY (Standard Mode)**
```bash
python daisy.py
```

#### **Start with Wake Word Detection**
```bash
python daisy.py --wake-word-sensitivity 0.7
```

#### **Start with Custom Model**
```bash
python daisy.py --model mistral:latest --whisper-model small
```

#### **Debug Mode**
```bash
python daisy.py --debug --audio-info
```

### 🎯 Command Line Options

```bash
# Core Options
--model MODEL_NAME          # Ollama model (default: llama3.2:latest)
--whisper-model SIZE         # Whisper model size (tiny/base/small/medium/large)
--debug                      # Enable verbose logging

# Wake Word Options
--no-wake-word              # Disable wake word detection
--wake-word-sensitivity 0.7 # Sensitivity (0.0-1.0)

# Voice Activity Detection
--vad-mode 2                # VAD aggressiveness (0-3)
--speech-start 3            # Frames to start speech detection
--speech-end 20             # Frames to end speech detection

# RAG & Documents
--no-rag                    # Disable document retrieval
--process-docs              # Process documents and exit
--force-reprocess           # Reprocess all documents

# System Information
--audio-info                # Show audio device information
```

### Voice Interaction Modes

#### **Mode 1: Wake Word (Hands-Free)**
1. Start DAISY with wake word enabled
2. Say "Hey DAISY" to activate
3. Speak your command when prompted
4. DAISY responds and returns to listening

#### **Mode 2: Manual Activation**
1. Start DAISY without wake word
2. Say "Hey DAISY" followed by your command
3. DAISY processes and responds
4. Repeat for each interaction

### Document Management (RAG)

#### **Adding Documents**
```bash
# Place PDF files in the documents/ directory
cp your_document.pdf documents/

# Process documents for RAG
python daisy.py --process-docs

# Force reprocessing of all documents
python daisy.py --force-reprocess
```

#### **Document Organization**
```
documents/
├── Books/
│   ├── Fiction/
│   └── Non-Fiction/
├── Manuals/
├── Research/
└── README.md
```

---

## Testing & Validation

### Comprehensive Testing
```bash
# Run full test suite
python test_comprehensive.py

# Test individual components
python test_daisy.py          # Basic functionality
python test_tts.py           # Text-to-speech engine
python test_wake_word.py     # Wake word detection
python test_audio_analysis.py # Audio processing
```

### TTS Testing & Voice Selection
```bash
# Test TTS with different voices
python test_tts.py --text "Hello, this is a voice test"

# List available voices
python test_tts.py --list-voices

# Test specific speaker
python test_tts.py --speaker p280 --text "Testing British male voice"

# Force pyttsx3 fallback
python test_tts.py --no-coqui --text "Testing fallback engine"
```

### Audio Device Configuration
```bash
# Show audio device information
python daisy.py --audio-info

# Test VAD settings
python daisy.py --vad-mode 2 --speech-start 3 --speech-end 15 --debug
```

---

## Customization Guide

### Voice Customization

#### **Available TTS Voices (VCTK Dataset)**
- **p225**: British English, Female, Southern England
- **p226**: British English, Male, Yorkshire  
- **p227**: British English, Male, Belfast
- **p228**: British English, Female, Southern England
- **p229**: British English, Female, Southern England
- **p230**: British English, Female, Stockton-on-Tees
- **p231**: British English, Female, Southern England
- **p232**: British English, Male, Southern England
- **p233**: British English, Female, Staffordshire
- **p234**: British English, Female, Newcastle
- **p236**: British English, Female, Manchester
- **p237**: British English, Male, Yorkshire
- **p238**: British English, Female, Liverpool
- **p239**: British English, Female, Belfast
- **p240**: British English, Female, Birmingham
- **p241**: British English, Male, Aberdeen
- **p243**: British English, Male, London
- **p244**: British English, Female, London
- **p245**: British English, Male, Ireland
- **p246**: British English, Male, Yorkshire
- **p247**: British English, Male, Scotland
- **p248**: British English, Female, Ireland
- **p249**: British English, Female, Birmingham
- **p250**: British English, Female, Hertfordshire
- **p251**: British English, Male, Central Scotland
- **p252**: British English, Male, London
- **p253**: British English, Female, Cardiff
- **p254**: British English, Male, Surrey
- **p255**: British English, Male, England
- **p256**: British English, Male, Birmingham
- **p257**: British English, Female, Surrey
- **p258**: British English, Male, Ireland
- **p259**: British English, Male, Nottingham
- **p260**: British English, Male, Scotland
- **p261**: British English, Female, Northern Ireland
- **p262**: British English, Female, Edinburgh
- **p263**: British English, Male, Gloucester
- **p264**: British English, Female, Gloucester
- **p265**: British English, Female, Wales
- **p266**: British English, Female, Ireland
- **p267**: British English, Female, Yorkshire
- **p268**: British English, Female, Northern Ireland
- **p269**: British English, Female, Tyneside
- **p270**: British English, Male, Yorkshire
- **p271**: British English, Male, Yorkshire
- **p272**: British English, Male, Edinburgh
- **p273**: British English, Male, Cambridge
- **p274**: British English, Male, Newcastle
- **p275**: British English, Male, Gloucester
- **p276**: British English, Female, Gloucester
- **p277**: British English, Female, Newcastle *(Default)*
- **p278**: British English, Male, Liverpool
- **p279**: British English, Male, Derby
- **p280**: British English, Male, Yorkshire
- **p281**: British English, Male, Sheffield
- **p282**: British English, Female, Yorkshire
- **p283**: British English, Female, Gloucester
- **p284**: British English, Male, Gloucester
- **p285**: British English, Male, Newcastle
- **p286**: British English, Male, Newcastle
- **p287**: British English, Male, York

#### **Change Default Voice**
Edit `src/utils/config.py`:
```python
TTS_SPEAKER_IDX = "p280"  # Change to desired speaker ID
```

### Personality Customization

Edit `personality.txt` to modify DAISY's:
- **Communication style** (formal, casual, humorous)
- **Response length** (brief, detailed, conversational)
- **Expertise areas** (technical, creative, analytical)
- **Cultural references** (British, American, international)

### Wake Word Customization

#### **Train Custom Wake Word** *(Advanced)*
1. Visit [Picovoice Console](https://console.picovoice.ai/)
2. Create custom wake word model
3. Download `.ppn` file
4. Update `WAKE_WORD_MODEL_PATH` in `.env`

### Advanced Configuration

#### **Whisper Model Selection**
- **tiny**: 39MB, fastest, lowest quality
- **base**: 74MB, balanced speed/quality *(default)*
- **small**: 244MB, better quality
- **medium**: 769MB, high quality
- **large-v2/v3**: 1550MB, highest quality

#### **VAD Tuning**
```python
# In src/utils/config.py
WEBRTC_VAD_MODE = 2           # 0=least aggressive, 3=most aggressive
WEBRTC_SPEECH_START_FRAMES = 3 # Frames to detect speech start
WEBRTC_SPEECH_END_FRAMES = 20  # Frames to detect speech end
```

---

## Project Structure

```
DAISY/
├── daisy.py                      # Main CLI entry point
├── daisy_gui.py                  # GUI application
├── setup.py                     # Installation setup
├── requirements.txt             # Python dependencies
├── personality.txt              # Assistant personality
├── README.md                    # This documentation
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
│
├── src/                         # Source code
│   ├── core/                    # Core assistant logic
│   │   ├── assistant.py         # Main assistant class
│   │   └── personality.py       # Personality management
│   │
│   ├── data/                    # Data management
│   │   └── chat_history.py      # Conversation history
│   │
│   ├── voice/                   # Voice processing
│   │   ├── speech_recognition.py # Whisper + VAD
│   │   ├── text_to_speech.py     # Coqui-AI TTS
│   │   └── cache/               # TTS audio cache
│   │
│   ├── rag/                     # RAG system
│   │   ├── document_loader.py   # PDF document loading
│   │   ├── document_processor.py # Text extraction/chunking
│   │   ├── embedding_generator.py # Sentence embeddings
│   │   ├── vector_store.py      # FAISS vector database
│   │   ├── retriever.py         # Document retrieval
│   │   └── document_tracker.py  # File change tracking
│   │
│   ├── gui/                     # GUI components
│   │   ├── main_window.py       # Main window
│   │   └── integration.py       # GUI-core integration
│   │
│   └── utils/                   # Utilities
│       ├── config.py            # Configuration settings
│       ├── config_manager.py    # Config management
│       ├── connection_manager.py # Ollama connection
│       ├── resource_manager.py  # Memory/resource cleanup
│       └── error_handler.py     # Error handling
│
├── models/                      # AI models
│   ├── hey-daisy_en_windows_v3_0_0.ppn  # Wake word model
│   └── [Whisper models auto-download]
│
├── documents/                   # RAG document store
│   ├── Books/
│   ├── Manuals/
│   └── README.md
│
├── data/                        # Application data
│   ├── chat_history.json        # Conversation history
│   ├── document_tracking.json   # Document processing state
│   └── .gitkeep
│
├── vector_db/                   # Vector database (auto-generated)
│   ├── faiss_index.bin         # FAISS index
│   ├── metadata.json           # Document metadata
│   └── id_mapping.pkl          # ID mappings
│
└── test_*.py                   # Test scripts
    ├── test_comprehensive.py    # Full system test
    ├── test_daisy.py           # Basic functionality
    ├── test_tts.py             # TTS engine test
    ├── test_wake_word.py       # Wake word test
    └── test_audio_analysis.py  # Audio processing test
```

---

## Troubleshooting

### Common Issues

#### **Audio Issues**
```bash
# Check audio devices
python daisy.py --audio-info

# Test microphone access
python -c "import sounddevice as sd; print(sd.query_devices())"

# Windows: Enable microphone permissions in Settings
# macOS: Grant Terminal microphone access in Security & Privacy
# Linux: Check ALSA/PulseAudio configuration
```

#### **Ollama Connection Issues**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama service
ollama serve

# Check available models
ollama list

# Pull required model
ollama pull llama3.2:latest
```

#### **TTS Problems**
```bash
# Test TTS engine
python test_tts.py --text "Testing TTS engine"

# Check espeak-ng installation
espeak-ng --version

# Windows: Verify espeak-ng is in PATH
# macOS: brew install espeak-ng
# Linux: sudo apt install espeak-ng
```

#### **Wake Word Issues**
```bash
# Verify Porcupine access key
cat .env | grep PORCUPINE_ACCESS_KEY

# Test wake word detection
python test_wake_word.py

# Check microphone sensitivity
python daisy.py --wake-word-sensitivity 0.8 --debug
```

#### **RAG/Document Issues**
```bash
# Check document directory
ls -la documents/

# Process documents manually
python daisy.py --process-docs --debug

# Verify vector database
ls -la vector_db/

# Force document reprocessing
python daisy.py --force-reprocess
```

### Error Messages

#### **ModuleNotFoundError: No module named 'torch'**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### **"Could not find espeak-ng"**
```bash
# Windows: Download from https://github.com/espeak-ng/espeak-ng/releases
# Add C:\Program Files\eSpeak NG to PATH

# macOS:
brew install espeak-ng

# Linux:
sudo apt install espeak-ng
```

#### **"Ollama connection failed"**
```bash
# Start Ollama in background
ollama serve &

# Or check if already running:
ps aux | grep ollama
```

#### **"No wake word model found"**
```bash
# Check model file exists
ls -la models/hey-daisy_en_windows_v3_0_0.ppn

# Verify access key in .env
echo $PORCUPINE_ACCESS_KEY
```

### Performance Optimization

#### **Memory Usage**
- Use `tiny` or `base` Whisper models for lower memory usage
- Reduce TTS cache size in configuration
- Limit RAG document collection size

#### **Speed Optimization**
- Use GPU acceleration if available (`CUDA_VISIBLE_DEVICES=0`)
- Adjust VAD sensitivity for faster response
- Pre-load models with warm-up commands

#### **Storage Management**
```bash
# Clean TTS cache
rm -rf src/voice/cache/tts/*

# Rebuild vector database
python daisy.py --force-reprocess

# Check disk usage
du -sh models/ vector_db/ src/voice/cache/
```

---

## Contributing

### Development Setup

```bash
# Clone with development dependencies
git clone https://github.com/Killiivalavan/DAISY.git
cd DAISY

# Create development environment
python -m venv venv-dev
source venv-dev/bin/activate  # or venv-dev\Scripts\activate on Windows

# Install with development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
python -m pytest test_comprehensive.py -v
```

### Contribution Guidelines

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Test** thoroughly (`python test_comprehensive.py`)
5. **Push** to branch (`git push origin feature/amazing-feature`)
6. **Open** a Pull Request

### Testing Requirements

- All new features must include tests
- Maintain test coverage above 80%
- Test on multiple platforms (Windows, macOS, Linux)
- Verify audio functionality with different devices

---

## Changelog

### Recent Updates

#### **v2.0.0** - Wake Word Integration
- Added Picovoice Porcupine wake word detection
- Hands-free voice activation with "Hey DAISY"
- Configurable wake word sensitivity
- Enhanced audio processing pipeline

#### **v1.9.0** - Faster-Whisper Integration  
- Switched to Faster-Whisper for improved performance
- Multiple model size options (tiny to large-v3)
- Enhanced WebRTC VAD integration
- Optimized memory usage and speed

#### **v1.8.0** - RAG System Implementation
- FAISS vector database integration
- PDF document processing and indexing
- Sentence-transformers embeddings
- Context-aware responses

#### **v1.7.0** - TTS Enhancement
- Coqui-AI TTS with British voices
- Multiple speaker options (VCTK dataset)
- TTS caching for performance
- pyttsx3 fallback engine

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Ollama Team** - Local LLM hosting platform
- **Hugging Face** - Faster-Whisper and transformers
- **Picovoice** - Porcupine wake word engine  
- **Coqui-AI** - High-quality TTS engine
- **OpenAI** - Original Whisper model
- **Facebook Research** - FAISS vector database
- **Marvel Studios** - J.A.R.V.I.S. inspiration

---

## Support

- **Issues**: [GitHub Issues](https://github.com/Killiivalavan/DAISY/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Killiivalavan/DAISY/discussions)
- **Documentation**: This README and inline code documentation

---

*Built with love by [Killiivalavan](https://github.com/Killiivalavan)*

*"I'm ready to assist you" - DAISY*
