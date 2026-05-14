# D.A.I.S.Y. mark-II (Dialogue-Driven Agentic Intelligence for Seamless Yield)

D.A.I.S.Y. mark-II is a highly responsive, always-on personal AI assistant built for sub-second latency. It leverages local Voice Activity Detection (Silero VAD), local streaming Speech-to-Text (Faster-Whisper), fast LLM inference (Groq API), and local Text-to-Speech (Kokoro).

## Prerequisites

- **Python 3.11 or higher**
- A working microphone and speaker
- A [Groq API Key](https://console.groq.com/keys) (Free Tier works great)

### OS-Specific System Dependencies

The `sounddevice` library requires **PortAudio** for microphone/speaker access. The Kokoro TTS engine requires **espeak-ng** for text phonemization.

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install libportaudio2 portaudio19-dev espeak-ng
```

#### Linux (Arch)
```bash
sudo pacman -S portaudio espeak-ng
```

#### macOS
```bash
# Using Homebrew
brew install portaudio espeak
```

#### 🪟 Windows
Windows users generally do not need to install PortAudio manually as the `sounddevice` pip wheel comes with it pre-packaged. However, you MUST install `espeak-ng` manually.

1. Download the latest `.msi` installer from the [espeak-ng releases page](https://github.com/espeak-ng/espeak-ng/releases).
2. Install it.
3. Make sure the installation path (usually `C:\Program Files\eSpeak NG`) is added to your system's PATH variable.

*(If you run into audio backend errors, you can optionally install PortAudio via Anaconda: `conda install -c conda-forge python-sounddevice`)*

---

## Installation

1. **Clone the repository** and navigate into it:
   ```bash
   git clone <your-repo-url>
   cd DAISY
   ```

2. **Create a Python Virtual Environment** (Highly Recommended):
   ```bash
   python -m venv .venv
   
   # Linux/macOS:
   source .venv/bin/activate
   
   # Windows:
   .venv\Scripts\activate
   ```

3. **Install the Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Environment Variables**:
   Copy the example environment file and add your API key.
   ```bash
   cp .env.example .env
   ```
   Open `.env` and paste your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

2. **Application Settings (`config.yaml`)**:
   The system's behavior is fully configurable via `config.yaml`. 
   - You can adjust the VAD threshold under `vad -> silero_threshold` (lower it if it doesn't pick up your voice, raise it if it triggers on noise).
   - You can change the TTS voice under `tts -> kokoro -> voice`.

## Running D.A.I.S.Y.

Start the assistant by running the main pipeline script:

```bash
python daisy/main.py
```

You should see: `D.A.I.S.Y. v2 ready. Listening...`
Speak into your microphone, and D.A.I.S.Y. will respond! Press `Ctrl+C` to gracefully shut down the daemon.
