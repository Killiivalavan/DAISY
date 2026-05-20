# D.A.I.S.Y. mark-II

**Dialogue-driven Agentic Intelligence for Seamless Yield** — a voice-activated personal AI assistant with sub-second latency. Always-on, interruptible, conversational, and capable of executing real-world actions through tool calling.

D.A.I.S.Y. runs as a persistent daemon on a home server/ personal laptop, using local models for wake word detection and speech-to-text, a cloud API (Groq) for fast LLM inference, and local neural text-to-speech (Kokoro). Inspired by J.A.R.V.I.S.

---

## Features

- **Wake word activation** — custom-trained "Daisy" wake word via OpenWakeWord
- **Voice Activity Detection** — Silero VAD with configurable thresholds and noise-floor tracking
- **Streaming Speech-to-Text** — Faster-Whisper (`small.en`, int8 quantized)
- **Fast LLM inference** — Groq API (`llama-3.3-70b-versatile`), streaming tokens for TTS pipelining
- **Neural Text-to-Speech** — Kokoro with multiple voice options
- **State machine pipeline** — `init → idle → listening → processing → speaking` with proper lifecycle management
- **Tool calling** — LLM function calling with 12 tools across 5 categories:
  - **System**: time/date, system info, sandboxed shell commands, reminders
  - **Web**: DuckDuckGo search, URL browsing with content extraction
  - **Files**: path-validated file read/write with size limits
  - **Background tasks**: spawn shell subprocesses, LLM sub-agents, and OpenCode coding tasks
  - **Task management**: list, check status, and cancel background tasks
- **Multi-round tool loop** — up to 5 rounds of tool calling before generating a natural-language response
- **Background task infrastructure** — async task lifecycle with UUID tracking, status reporting, and proactive announcements
- **Proactive announcements** — D.A.I.S.Y. speaks up when background tasks complete or reminders fire
- **Persistent memory** — SQLite-backed fact storage with FTS5 full-text search, session tracking, and automatic summarization
- **Conversation context** — in-memory FIFO turn buffer with configurable depth
- **Automatic fact extraction** — "remember" commands are parsed and stored automatically
- **Async-first architecture** — event-driven pipeline via asyncio pub/sub event bus
- **Comprehensive test suite** — 11 test files covering all major subsystems
- **System diagnostics** — built-in diagnostic runner and VAD debugger
- **Fully configurable** — YAML-based config with Pydantic validation

---

## Architecture

```
┌─────────────┐    ┌──────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐
│  Audio In   │───▶│   VAD    │───▶│    STT    │───▶│    LLM     │───▶│   TTS    │───▶ Audio Out
│ (sounddevice)│   │ (Silero) │    │(Whisper)  │    │  (Groq)    │    │ (Kokoro) │
└─────────────┘    └──────────┘    └───────────┘    └────────────┘    └──────────┘
       │                                                  │
       ▼                                                  ▼
┌─────────────┐                                  ┌──────────────┐
│Wake Word    │                                  │  Tool System │
│(OpenWakeWord)│                                  │  12 tools    │
└─────────────┘                                  │ + loop       │
       │                                         └──────────────┘
       ▼                                                │
┌──────────────────┐                           ┌───────────────┐
│   Event Bus      │                           │  Memory       │
│ (pub/sub async)  │                           │ SQLite + FTS5 │
└──────────────────┘                           │ + Summarizer  │
       │                                        └───────────────┘
       ▼
┌──────────────────┐
│  State Machine   │
│  idle → listening│
│  → processing →  │
│  speaking        │
└──────────────────┘
```

---

## Tool System

D.A.I.S.Y. exposes 13 function-calling tools to the LLM using OpenAI-compatible JSON schemas. When the LLM decides it needs to perform an action, the system executes the tool, injects the result back as a `tool` role message, and continues for up to 5 rounds before generating the final spoken response.

| Tool | Description |
|---|---|
| `get_time_date` | Current time, date, and timezone |
| `get_system_info` | CPU, RAM, and disk usage |
| `run_command` | Sandboxed shell (pre-approved commands only) |
| `set_reminder` | Async timer with proactive announcement |
| `web_search` | DuckDuckGo search with snippets |
| `browse_url` | Fetch URL and extract readable content via trafilatura |
| `read_file` | Read files from allowed directories (1 MB limit) |
| `write_file` | Write files to allowed directories |
| `spawn_task` | Background shell or sub-agent task |
| `spawn_opencode_task` | Background OpenCode coding task |
| `get_task_status` | Check a background task's progress |
| `list_tasks` | List all recent background tasks |
| `cancel_task` | Cancel a running background task |

### Safety

- Shell commands restricted to a whitelist (`df`, `free`, `uptime`, `uname`, `whoami`, `ls`, `cat`, `ps`, `ping`, `systemctl`)
- File paths validated against allowed directories
- File size capped at 1 MB
- Default command timeout of 30 s (max 300 s)
- Tools can be disabled entirely via `config.yaml`

---

## Memory System

D.A.I.S.Y. uses a two-tier memory architecture:

1. **Conversation Buffer** (in-memory) — FIFO queue of recent turns, injected into the LLM context for immediate recall
2. **SQLite Store** (persistent) — stores facts extracted from "remember" commands with FTS5 full-text search, tracks sessions with automatic summarization, and injects relevant facts into the LLM context

The system automatically parses utterances like *"remember my favorite color is blue"*, *"remember that the server IP is 192.168.1.1"*, or *"remember this"*.

---

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
brew install portaudio espeak
```

#### Windows
Windows users generally do not need to install PortAudio manually as the `sounddevice` pip wheel comes with it pre-packaged. However, you **must** install `espeak-ng` manually.

1. Download the latest `.msi` installer from the [espeak-ng releases page](https://github.com/espeak-ng/espeak-ng/releases).
2. Install it.
3. Make sure the installation path (usually `C:\Program Files\eSpeak NG`) is added to your system's PATH variable.

---

## Installation

1. **Clone the repository** and navigate into it:
   ```bash
   git clone <your-repo-url>
   cd DAISY
   ```

2. **Create a Python virtual environment**:
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

---

## Configuration

### 1. Environment Variables

Copy the example environment file and add your API key.

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Application Settings (`config.yaml`)

The system's behavior is fully configurable via `config.yaml`. Key sections:

| Section | Key Settings |
|---|---|
| **audio** | sample rate, channels, chunk size, I/O devices |
| **vad** | silence threshold, speech start/end frame counts, max recording duration |
| **stt** | Whisper model size, device, compute type |
| **llm** | Groq model, temperature, max tokens, system prompt path |
| **tts** | Kokoro voice, sample rate, speed |
| **pipeline** | listening and processing timeout durations |
| **memory** | max conversation turns, database path, fact injection settings |
| **wake_word** | model path, detection threshold |
| **tools** | enable/disable, allowed directories, allowed commands, timeouts, OpenCode integration |
| **mode** | `wake_word`, `always_on`, or `push_to_talk` |

Adjust the VAD threshold (`vad → silero_threshold`) — lower it if D.A.I.S.Y. doesn't pick up your voice, raise it if it triggers on background noise. Change the TTS voice under `tts → kokoro → voice`.

---

## Wake Word Engine

1. Create a `models/` folder in the root directory
2. Generate custom wake word models using [this Colab notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb)
3. Place the generated `.onnx` and `.tflite` files in `models/`
4. Reference them in `config.yaml` under `wake_word → model`

Pre-trained "daisy" models are included for quick start.

---

## Running D.A.I.S.Y.

Start the assistant by running the main entry point:

```bash
python daisy/main.py
```

You should see: `D.A.I.S.Y. v2 ready. Listening...`

Speak the wake word ("Daisy"), then ask your question. D.A.I.S.Y. will process, potentially use tools, and respond. Press `Ctrl+C` to gracefully shut down.

### Diagnostics

Run the system diagnostic to verify all components are working:

```bash
python -m daisy.tests.diagnostic
```

For VAD debugging:

```bash
python -m daisy.tests.debug_vad
```

---

## Running Tests

```bash
pytest daisy/tests/ -v
```

---

## Project Structure

```
├── config.yaml              # Master configuration (Pydantic-validated)
├── SOUL.md                  # D.A.I.S.Y. personality system prompt
├── requirements.txt         # Python dependencies
├── models/                  # Wake word models (.onnx, .tflite)
├── daisy/
│   ├── main.py              # Single entry point
│   ├── core/
│   │   ├── event_bus.py     # Async pub/sub event system
│   │   └── state_machine.py # DaisyStateMachine (python-statemachine)
│   ├── audio/
│   │   ├── input_stream.py  # Microphone input (sounddevice async)
│   │   ├── output_stream.py # Speaker output (sounddevice async)
│   │   └── audio_utils.py   # Format conversion, resampling
│   ├── wake_word/
│   │   └── detector.py      # OpenWakeWord detector
│   ├── vad/
│   │   └── silero_vad.py    # Silero VAD (ONNX)
│   ├── stt/
│   │   └── faster_whisper_stt.py # Faster-Whisper transcription
│   ├── llm/
│   │   ├── groq_client.py   # Groq API client (streaming + non-streaming)
│   │   └── sentence_splitter.py # Token stream → sentence chunks
│   ├── tts/
│   │   └── kokoro_tts.py    # Kokoro TTS synthesis
│   ├── memory/
│   │   ├── conversation_buffer.py # In-memory FIFO turn history
│   │   ├── sqlite_store.py  # Persistent fact/session storage with FTS5
│   │   └── memory_manager.py # Unified memory orchestrator
│   ├── tools/
│   │   ├── tool_registry.py # LLM function-call schemas + handler builder
│   │   ├── system_tools.py  # Time, system info, shell, reminders
│   │   ├── web_tools.py     # Web search (DuckDuckGo) + URL browsing
│   │   ├── file_tools.py    # Safe file read/write
│   │   ├── background_tools.py # Background task spawning
│   │   ├── task_tracker.py  # Background task lifecycle management
│   │   └── announcement_queue.py # Proactive announcement queue
│   ├── tests/               # 11 test files + diagnostic tools
│   └── utils/
│       └── config_loader.py # Pydantic-validated YAML config loading
```

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Async Runtime | asyncio |
| Audio I/O | sounddevice |
| Voice Activity Detection | Silero VAD (ONNX) |
| Speech-to-Text | Faster-Whisper (`small.en`) |
| LLM API | Groq (`llama-3.3-70b-versatile`) via OpenAI SDK |
| Text-to-Speech | Kokoro |
| Wake Word | OpenWakeWord |
| State Machine | python-statemachine |
| Web Search | DuckDuckGo Search (ddgs) |
| Content Extraction | trafilatura + httpx |
| Persistence | SQLite with FTS5 |
| Configuration | pydantic + PyYAML |
| Testing | pytest, pytest-asyncio, pytest-mock |
