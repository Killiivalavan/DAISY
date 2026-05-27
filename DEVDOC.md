# D.A.I.S.Y. v2 — Project Requirements Document

**D**ialoue-driven **A**gentic **I**ntelligence for **S**eamless **Y**ield  
**Version**: 2.0.0  
**Status**: Pre-Development  
**Author**: Killiivalavan  
**Target Platform**: Andromeda (Ubuntu Server 24.04 LTS) + Niggatron (dev)  
**Access**: Tailscale VPN, PWA client  
**Personality Model**: F.R.I.D.A.Y. / J.A.R.V.I.S. hybrid — sharp, efficient, addresses user as "Boss"  
**Budget**: Zero-spend. Every component must be free and self-hostable.

---

## Table of Contents

1. [Vision & Goals](#1-vision--goals)
2. [Why v2 — Lessons from v1](#2-why-v2--lessons-from-v1)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Technology Stack](#4-technology-stack)
5. [Component Specifications](#5-component-specifications)
   - 5.1 Wake Word Engine
   - 5.2 Voice Activity Detection (VAD)
   - 5.3 Speech-to-Text (STT)
   - 5.4 Large Language Model (LLM)
   - 5.5 Text-to-Speech (TTS)
   - 5.6 Barge-in / Interruption Engine
   - 5.7 Memory System
   - 5.8 Tool / Agent Layer
   - 5.9 PWA Client
6. [Asyncio Event Architecture](#6-asyncio-event-architecture)
7. [State Machine](#7-state-machine)
8. [Module Structure](#8-module-structure)
9. [Latency Strategy](#9-latency-strategy)
10. [Build Phases](#10-build-phases)
11. [Non-Functional Requirements](#11-non-functional-requirements)
12. [Out of Scope (for now)](#12-out-of-scope-for-now)

---

## 1. Vision & Goals

### The Vision

Build a personal AI assistant that genuinely feels like J.A.R.V.I.S. from Iron Man — not a voice chatbot with a wake word bolted on, but a system that feels alive, responsive, and present. The user should be able to speak naturally, get interrupted responses back in under a second, interrupt D.A.I.S.Y. mid-sentence, and have the system remember context across sessions.

### Core Goals

- **Sub-second perceived response latency** from end of user speech to start of D.A.I.S.Y. speaking
- **Natural interruption** — user can barge in while D.A.I.S.Y. is speaking and she stops immediately
- **Persistent memory** — D.A.I.S.Y. remembers facts, preferences, and ongoing projects across sessions
- **Tool execution** — can browse the web, manage files, run code, answer real-time questions
- **Zero ongoing cost** — no paid APIs, no subscriptions, no metered services
- **Self-hosted on Andromeda** — runs as persistent daemon, accessible globally via Tailscale
- **Personality** — consistent F.R.I.D.A.Y.-inspired character, addresses user as "Boss", dry wit, sharp and efficient

### What "JARVIS Feel" Actually Means (Engineering Translation)

| JARVIS Trait | Engineering Requirement |
|---|---|
| Always ready, no delay | Wake word + persistent loaded models |
| Responds while still forming thought | Streaming LLM → streaming TTS |
| Knows when you're talking to it | OpenWakeWord custom trigger |
| Can be interrupted mid-sentence | Barge-in with AEC + hard stop + task cancel |
| Remembers everything | Multi-tier memory system |
| Can actually do things | Custom tool layer with 13 built-in tools |
| Consistent personality | SOUL.md system prompt, locked |

---

## 2. Why v2 — Lessons from v1

The original D.A.I.S.Y. (built circa Claude 3.5 Sonnet era) had the right instincts but wrong execution. These are the specific failure modes to avoid in v2.

### v1 Failure Modes

**Synchronous core with async patches**  
v1 started synchronous and `daisy_async.py` was a retrofit. The `ASYNC_PIPELINE_README.md` and `CLEANUP_REPORT.md` in the repo are evidence of a migration that never completed cleanly. v2 must be async-first from line one — no sync calls anywhere in the hot path.

**Model loading per-call**  
Whisper and TTS models were being loaded and unloaded. First-inference cold start on Whisper large is 1-2 seconds. v2 loads all models once at startup and keeps them warm in memory forever.

**No streaming**  
v1 waited for the full LLM response before starting TTS synthesis. This adds the full generation time (1-4 seconds) to perceived latency. v2 streams token-by-token, synthesizes sentence-by-sentence, and starts playing before generation is complete.

**No interruption**  
Once D.A.I.S.Y. started speaking, you had to wait. This is the single biggest UX failure for a voice assistant.

**Local-only LLM (Ollama)**  
Ollama on consumer hardware without a GPU is too slow for natural conversation. v2 uses cloud inference APIs (free tier) which are orders of magnitude faster.

**Coqui TTS quality and speed**  
Coqui was slow and voice quality was inconsistent. v2 uses Kokoro-TTS (faster, more natural) with Piper as a speed fallback.

**Too many entry points**  
`daisy.py`, `daisy_async.py`, `daisy_gui.py` — three different ways to start the system. v2 has one entry point: `main.py`. One daemon. One process.

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         ANDROMEDA                               │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │  AUDIO IN    │    │           D.A.I.S.Y. CORE            │  │
│  │              │    │                                      │  │
│  │  Microphone  │───▶│  Wake Word ──▶ VAD ──▶ STT          │  │
│  │  (always on) │    │                         │            │  │
│  └──────────────┘    │                         ▼            │  │
│                       │                    LLM (stream)     │  │
│  ┌──────────────┐    │                         │            │  │
│  │  AUDIO OUT   │    │                         ▼            │  │
│  │              │◀───│              TTS (stream chunks)     │  │
│  │  Speakers    │    │                         │            │  │
│  └──────────────┘    │                         ▼            │  │
│        │              │                  Audio Player       │  │
│        │ (echo)       │                    + AEC            │  │
│        ▼              │                         │            │  │
│  ┌──────────────┐    │                    Barge-in Listener │  │
│  │  AEC Module  │────│─────────────────────────┘            │  │
│  └──────────────┘    │                                      │  │
│                       │  ┌────────────┐  ┌───────────────┐  │  │
│                       │  │  Memory    │  │  Tool Layer   │  │  │
│                       │  │  SQLite    │  │  Custom Tools │  │  │
│                       │  │  ChromaDB  │  │  (Python)     │  │  │
│                       │  └────────────┘  └───────────────┘  │  │
│                       └──────────────────────────────────────┘  │
│                                    │                             │
│                               FastAPI + WS                       │
│                                    │                             │
└────────────────────────────────────┼─────────────────────────────┘
                                     │ Tailscale
                          ┌──────────┴──────────┐
                          │     PWA Client       │
                          │  (Browser / Mobile)  │
                          └──────────────────────┘
```

### Data Flow (Happy Path)

```
User speaks "DAISY, what's the weather?"

1. OpenWakeWord detects "DAISY" → fires WAKE event
2. Audio cue plays (500ms beep/click)
3. Silero VAD activates, buffers audio
4. User finishes speaking → VAD detects silence endpoint
5. Audio buffer sent to Faster-Whisper → transcript: "what's the weather?"
6. Transcript sent to LLM (Groq) with conversation history + system prompt
7. LLM streams tokens → sentence splitter watches for [.!?,]
8. First complete sentence detected → Kokoro TTS synthesizes it
9. Audio chunk 1 starts playing → TTS synthesizes chunk 2 in parallel
10. Barge-in listener running concurrently throughout step 9
11. D.A.I.S.Y. finishes speaking → returns to IDLE/listening state
```

---

## 4. Technology Stack

### Overview Table

| Layer | Technology | Why | Cost |
|---|---|---|---|
| Language | Python 3.11+ | Async-first, all AI libs native, orchestration glue | Free |
| Runtime | asyncio | Native Python async, no extra framework needed | Free |
| Wake Word | OpenWakeWord | Open source, trainable custom words, Python native | Free |
| VAD | Silero VAD (ONNX) | Fast, accurate, tiny memory footprint, no network | Free |
| STT | RealtimeSTT + Faster-Whisper | Streaming support, small.en int8 is fast and accurate | Free |
| LLM Primary | Groq API (free tier) | Fastest inference available for free, function calling | Free |
| LLM Fallback | Gemini 2.0 Flash (free tier) | Long context, function calling, reliable fallback | Free |
| TTS Primary | Kokoro-TTS | Natural voice, fast on CPU, MIT licensed | Free |
| TTS Fallback | Piper TTS | Extremely fast, minimal memory, runs on anything | Free |
| AEC | WebRTC AEC (via webrtcvad) | Cancels D.A.I.S.Y.'s own voice from mic input | Free |
| Memory (short) | In-process conversation buffer | Fast, no overhead | Free |
| Memory (medium) | SQLite | Session summaries, fact store, zero setup | Free |
| Memory (long) | ChromaDB | Vector semantic recall, embedded Python, no server | Free |
| Tools / Agent | Custom Python tool framework | System, web, file, and background task tools via direct Python | Free |
| API Server | FastAPI + uvicorn | Async, fast, WebSocket support, you know it | Free |
| Transport | WebSockets | Real-time bidirectional for PWA client | Free |
| Secure Access | Tailscale | Already deployed on Andromeda | Free |
| Deployment | systemd user service | Persistent daemon, auto-restart, logging | Free |

### Language Choice Rationale

Python is the correct choice for D.A.I.S.Y. The performance concern is a red herring for this use case:

- D.A.I.S.Y.'s pipeline is **I/O-bound**, not CPU-bound. Waiting on network, audio, and model inference — not Python computation.
- All compute-heavy work (VAD, Whisper, TTS inference) runs in C/C++ under the hood regardless of the calling language.
- `asyncio` handles concurrent waiting perfectly. When awaiting a Groq response, Python isn't "running slowly" — it's just not running at all, which is correct.
- The one Python weakness (startup time) is irrelevant for a persistent daemon that starts once.
- Switching to TypeScript or Rust would cost months of rework for zero perceptible gain.

### LLM Choice Rationale

**Groq** runs Llama 3.3 70B on custom LPU (Language Processing Unit) hardware. It produces inference speeds of 500-800 tokens/second on free tier, which is faster than any other free option by a significant margin. For a voice assistant where the LLM is in the latency-critical path, this matters enormously.

**Gemini 2.0 Flash** as fallback because it has a generous free tier, reliable uptime, and strong function calling support. When Groq is rate-limited or down, Gemini takes over.

Both support streaming and function calling (tool use), which are required features.

---

## 5. Component Specifications

### 5.1 Wake Word Engine

**Library**: OpenWakeWord  
**Custom trigger word**: "DAISY" or "Hey DAISY" (to be decided and trained)  
**Always-on CPU load**: < 5% on Andromeda  
**False positive rate target**: < 1 per hour of ambient noise  
**False negative rate target**: < 5% (misses your trigger)

**Implementation details**:
- Runs as a coroutine in the main asyncio event loop
- Listens to raw audio stream continuously via `sounddevice` input stream
- On detection: fires `WAKE` asyncio Event, plays audio acknowledgment cue
- Audio acknowledgment cue: short 500ms tone or subtle click (not a full "Yes, Boss?" — just a ready indicator)
- After the full conversation ends, automatically resets to wake-word listening mode

**Custom wake word training**:
- OpenWakeWord requires ~20-30 audio samples of the trigger phrase spoken by the user
- Training process: record samples using provided script, run training, output is a `.tflite` model file
- This training happens once during setup, takes ~10 minutes

**Activation modes** (configurable via config.yaml):
- `wake_word` — default, requires saying "DAISY" to activate
- `always_on` — VAD triggers on any speech, D.A.I.S.Y. responds to everything (suitable for use alone)
- `push_to_talk` — keyboard shortcut activates listening (for silent environments)

---

### 5.2 Voice Activity Detection (VAD)

**Library**: Silero VAD via ONNX Runtime  
**Model size**: ~1MB (tiny)  
**Purpose**: Detect precise start and end of user speech after wake word triggers

**Configuration targets**:
- Speech start detection: 3 consecutive positive frames (~96ms)
- Speech end detection: 15-20 frames of silence (~480-640ms) — tunable
- Sampling rate: 16kHz mono (Silero's native rate)
- Frame size: 512 samples (32ms)

**Role in pipeline**:
- OpenWakeWord handles "is this my trigger word"
- Silero VAD handles "has the user started/stopped speaking"
- They are separate concerns and run on separate audio buffers

**Silence threshold tuning**: The speech-end threshold is the primary knob for perceived latency. Shorter = faster response but more false endpoints (cuts off mid-sentence). Start at 600ms, tune down based on feel.

---

### 5.3 Speech-to-Text (STT)

**Library**: RealtimeSTT (wraps Faster-Whisper)  
**Model**: `small.en` with int8 quantization  
**Reason for `small.en`**: Best balance of accuracy and speed for English-only use. `base.en` is faster but noticeably less accurate. `medium.en` is more accurate but significantly slower.

**Performance targets on Andromeda**:
- Transcription latency for 3-5 second utterance: < 300ms
- Transcription latency for 1-2 second utterance: < 150ms

**RealtimeSTT advantages over raw Faster-Whisper**:
- Partial transcript streaming — starts processing while user is still speaking
- Handles the VAD→STT handoff cleanly
- Built-in audio buffering

**Model loading**: Loaded once at daemon startup, kept in memory. First inference may be slow due to ONNX graph compilation; subsequent calls are fast.

**Language**: English only for now. Multilingual is a future phase consideration.

---

### 5.4 Large Language Model (LLM)

**Primary**: Groq API — `llama-3.3-70b-versatile` model  
**Fallback**: Google Gemini 2.0 Flash via `google-generativeai` SDK  
**Failover logic**: If Groq returns 429 (rate limit) or 5xx, automatically retry with Gemini on same request

**Required capabilities**:
- Streaming token output (mandatory — no streaming = no streaming TTS = high latency)
- Function calling / tool use (for agent tool execution)
- Sufficient context window for conversation history (Groq Llama 3.3: 128k tokens)

**System prompt (SOUL.md)**:
- D.A.I.S.Y. personality: F.R.I.D.A.Y.-inspired, sharp, dry wit, efficient
- Always addresses user as "Boss"
- British-adjacent precision in language
- Never verbose when brief is sufficient
- Aware she is running on Andromeda, aware of the user's projects
- Injected at every conversation turn as the system message

**Conversation history management**:
- Last N turns kept in memory buffer (N = configurable, default 20)
- When buffer approaches context limit, oldest turns are summarized and compressed
- Summaries stored in SQLite, retrieved and injected as compressed history context

**Streaming implementation**:
- Use async generators to yield tokens as they arrive
- Sentence splitter watches the token stream for sentence boundaries (`.`, `!`, `?`, `;` followed by whitespace or end)
- Each complete sentence is immediately dispatched to the TTS queue
- Generation can be cancelled mid-stream via asyncio task cancellation (for barge-in)

---

### 5.5 Text-to-Speech (TTS)

**Primary**: Kokoro-TTS  
**Fallback**: Piper TTS  
**Voice target**: Natural, clear, slightly formal — appropriate for an AI assistant character

**Kokoro-TTS specs**:
- 82M parameter model — small enough for CPU, fast enough for real-time
- Real-time factor (RTF) on CPU: ~0.3-0.5 (synthesizes 2-3x faster than playback speed)
- MIT licensed, fully local, no API calls
- Quality: noticeably more natural than Coqui or Piper

**Piper TTS specs**:
- RTF on CPU: ~0.05-0.1 (extremely fast, 10-20x real-time)
- Lower voice quality than Kokoro but excellent for low-latency scenarios
- Runs on anything, minimal memory

**Streaming TTS architecture**:
- TTS does NOT wait for the full LLM response
- Sentence queue: LLM sentence splitter → async queue → TTS worker coroutine
- TTS worker: pulls sentence from queue → synthesizes → pushes audio chunk to playback queue
- Playback worker: pulls audio chunks → plays via sounddevice OutputStream
- Double-buffering: while chunk N is playing, chunk N+1 is being synthesized
- Result: user hears first sentence ~300-500ms after LLM starts generating

**Audio format**: 22050Hz or 24000Hz mono WAV in memory (no disk writes for TTS cache in hot path)

---

### 5.6 Barge-in / Interruption Engine

This is one of the most technically complex components. It must solve the acoustic echo cancellation problem — D.A.I.S.Y.'s own voice playing through speakers will trigger VAD and cause self-interruption without AEC.

**Acoustic Echo Cancellation (AEC)**:
- Library: `webrtcvad` includes WebRTC's AEC implementation
- Alternative: `pyaudio` with echo cancellation preprocessing
- Process: the playback audio signal is subtracted from the microphone input in real-time before VAD processing
- This makes the system deaf to its own voice while remaining sensitive to the user's voice

**Barge-in detection flow**:
1. When system enters `SPEAKING` state, barge-in listener coroutine activates
2. Barge-in listener receives AEC-processed audio (D.A.I.S.Y.'s voice already removed)
3. Silero VAD runs on cleaned audio with slightly higher threshold than normal (extra safety margin)
4. If VAD fires → INTERRUPT event dispatched
5. On INTERRUPT event:
   - Audio player: `stop()` immediately (hard stop, no fade)
   - TTS queue: `clear()` — discard all pending chunks
   - LLM generation task: `task.cancel()` — stop generating
   - Short audio cue plays (indicates she heard the interruption)
   - System transitions to `LISTENING` state
6. Entire stop-and-pivot must complete in < 200ms

**False interruption protection**:
- Minimum speech duration before interruption registers: 300ms
- Prevents coughs, short sounds from triggering barge-in
- Configurable threshold

---

### 5.7 Memory System

Three-tier memory architecture. Each tier serves a different time horizon and retrieval pattern.

**Tier 1 — Conversational Buffer (in-process)**
- What: Last N conversation turns (user + D.A.I.S.Y. pairs)
- Where: Python list in memory, lost on restart
- Capacity: 20 turns default, configurable
- Purpose: Immediate context, pronoun resolution, follow-up questions
- Management: FIFO, when full oldest turn is summarized before dropping

**Tier 2 — Session Memory (SQLite)**
- What: Session summaries, extracted facts, user preferences, ongoing project notes
- Where: `~/.daisy/memory.db` on Andromeda
- Persists: Across restarts, indefinitely
- Schema:
  ```
  sessions(id, timestamp, summary, duration_minutes)
  facts(id, category, key, value, confidence, timestamp, last_accessed)
  preferences(key, value, updated_at)
  ```
- Fact categories: `user_preference`, `ongoing_project`, `technical_context`, `personal_info`
- Retrieval: At conversation start, recent session summary + relevant facts injected into system prompt

**Tier 3 — Semantic Memory (ChromaDB)**
- What: Longer-form memories, document context, project documentation
- Where: Embedded ChromaDB instance at `~/.daisy/chromadb/`
- Retrieval: Semantic similarity search on conversation turn to find relevant past context
- Embedding model: `all-MiniLM-L6-v2` (fast, small, good quality)
- Populated by: Summarization of long sessions, explicit "remember this" commands, document ingestion

**Memory injection into LLM context**:
```
[System: SOUL.md personality]
[System: Relevant facts from SQLite]
[System: Semantically relevant past context from ChromaDB (top 3)]
[Conversation: Last N turns from buffer]
[User: Current message]
```

---

### 5.8 Tool / Agent Layer

**Framework**: Custom Python tool layer built directly into D.A.I.S.Y.'s process  
**Integration pattern**: Tool handlers are async Python functions registered via a shared schema registry. The LLM's function-calling mechanism routes directly to in-process handlers — no external daemon, no subprocess overhead.

**Why custom tools instead of OpenClaw**:
- OpenClaw (a Node.js daemon) was originally evaluated as the tool executor, but the integration complexity — managing a separate service, cross-process IPC, state synchronization, and debugging across two runtimes — outweighed the benefits for D.A.I.S.Y.'s use case
- D.A.I.S.Y.'s tool needs are well-defined and finite: web search, file ops, system info, reminders, background tasks. These are straightforward to implement in pure Python
- Keeping everything in one process eliminates IPC latency, simplifies deployment, and makes error handling deterministic
- The OpenClaw approach can still be revisited in a future phase if multi-channel or browser automation becomes necessary

**Integration architecture**:
```
LLM generates function call → tool_registry dispatches to handler
    │
    ├── Web search   → web_tools.py        (httpx + DuckDuckGo + trafilatura)
    ├── File ops     → file_tools.py        (path-validated read/write)
    ├── Background   → background_tools.py  (asyncio task spawning)
    ├── System info  → system_tools.py      (psutil, datetime, subprocess)
    └── Reminders    → system_tools.py      (asyncio timer + announcement_queue)
```

**Tool call latency**: Tool calls are NOT in the latency-critical voice path. D.A.I.S.Y. can say "Let me check that for you, Boss" (immediate TTS) while the tool call executes in background. Result comes back, D.A.I.S.Y. speaks it.

**Tools available to LLM via function calling**:

| Tool | Description | Executor |
|---|---|---|
| `get_time_date` | Current time, date, and timezone | Python direct (datetime) |
| `get_system_info` | CPU, RAM, disk, network status | Python direct (psutil) |
| `run_command` | Execute a shell command (sandboxed, whitelisted) | Python direct (asyncio.subprocess) |
| `set_reminder` | Schedule a future notification | Python direct (asyncio) |
| `web_search` | Search the web via DuckDuckGo | Python direct (ddgs) |
| `browse_url` | Fetch and summarize a specific URL | Python direct (httpx + trafilatura) |
| `read_file` | Read a file from allowed directories | Python direct (path-validated) |
| `write_file` | Write content to a file | Python direct (path-validated) |
| `spawn_task` | Run a shell command or sub-agent in background | Python direct (asyncio.create_task) |
| `spawn_opencode_task` | Launch an OpenCode coding task in background | Python direct (asyncio.subprocess) |
| `get_task_status` | Check status of a background task by UUID | Python direct (task_tracker) |
| `list_tasks` | List recent background tasks | Python direct (task_tracker) |
| `cancel_task` | Cancel a running background task | Python direct (task_tracker) |

**Safety**: 
- Shell commands restricted to a whitelist (`df`, `free`, `uptime`, `uname`, `whoami`, `ls`, `cat`, `ps`, `ping`, `systemctl`)
- File access restricted to configured allowed directories (`/home/bashman`, `/tmp`, `/home/bashman/Code`)
- File reads capped at 1MB, tool content capped at 5KB
- Default command timeout 30s, maximum 300s
- Tools can be fully disabled via `config.yaml`

---

### 5.9 PWA Client

**Technology**: FastAPI backend + vanilla JS PWA frontend (or React if preferred)  
**Transport**: WebSockets for real-time audio streaming and status updates  
**Purpose**: Browser-based interface when voice isn't appropriate (typing, reviewing, etc.)

**Features**:
- Text input as alternative to voice
- Live transcript display (STT output shown as D.A.I.S.Y. processes)
- D.A.I.S.Y. response displayed as text alongside audio
- System status indicator (IDLE / LISTENING / PROCESSING / SPEAKING)
- Conversation history view
- Memory viewer (what facts D.A.I.S.Y. knows about you)
- Settings panel (voice mode, wake word toggle, model selection)

**Access**: Available on local network and globally via Tailscale. No public internet exposure.

---

## 6. Asyncio Event Architecture

The entire system runs in a single asyncio event loop. Everything is a coroutine or an asyncio-aware stream. No `time.sleep()`. No `threading.Thread()`. No blocking calls in the hot path.

### Core asyncio Primitives Used

| Primitive | Used For |
|---|---|
| `asyncio.Event` | Wake word trigger, barge-in signal, interrupt signal |
| `asyncio.Queue` | TTS sentence queue, audio chunk playback queue |
| `asyncio.Task` | LLM generation (cancellable), TTS synthesis, tool calls |
| `asyncio.Lock` | Audio device access, state machine transitions |
| `asyncio.gather` | Running concurrent coroutines (VAD + barge-in listener) |
| `async generators` | LLM token streaming, audio chunk streaming |

### Concurrent Coroutines

At any point in time, multiple coroutines are running concurrently in the event loop:

```
Always running:
├── audio_input_stream()         — reads from microphone continuously
├── wake_word_listener()         — processes audio for wake word
├── system_health_monitor()      — logs CPU/RAM, checks model health
└── fastapi_server()             — serves PWA client

During LISTENING state (added):
├── vad_processor()              — detects speech start/end
└── audio_buffer_accumulator()   — collects speech audio

During PROCESSING state (added):
├── llm_streamer()               — async generator yielding tokens
├── sentence_splitter()          — watches token stream, fires on sentence end
└── tts_synthesizer()            — synthesizes sentences from queue

During SPEAKING state (added):
├── audio_player()               — plays audio chunks from queue
├── barge_in_listener()          — AEC-cleaned VAD watching for interruption
└── tts_synthesizer()            — continues synthesizing ahead of playback
```

### Event Flow Diagram

```
WAKE EVENT
    │
    ▼
[play_ack_cue()]  →  vad_processor() starts
                              │
                    speech_start detected
                              │
                    audio accumulates...
                              │
                    speech_end detected
                              │
                    [stop vad_processor()]
                              │
                    [start llm_streamer()]
                              │
                    tokens stream in...
                              │
                    sentence boundary detected
                              │
                    [tts_queue.put(sentence)]
                              │
                    [tts_synthesizer() pulls sentence]
                              │
                    [audio_queue.put(chunk)]
                              │
                    [audio_player() plays chunk]
                              │
                    [barge_in_listener() running concurrently]
                              │
                    ┌─── VAD fires? ──────────────────────┐
                    │ YES                                  │ NO
                    ▼                                      ▼
            INTERRUPT EVENT                     continue until queue empty
                    │                                      │
            [audio_player.stop()]                 [return to IDLE]
            [tts_queue.clear()]
            [llm_task.cancel()]
            [play_ack_cue()]
            [return to LISTENING]
```

---

## 7. State Machine

D.A.I.S.Y. is always in exactly one state. State transitions are atomic (protected by asyncio.Lock).

```
                    ┌─────────┐
                    │  IDLE   │◀────────────────────────────┐
                    └────┬────┘                             │
                         │ WAKE event                       │
                         ▼                                  │
                   ┌──────────┐                             │
                   │LISTENING │                             │
                   └────┬─────┘                             │
                        │ VAD endpoint detected             │
                        ▼                                   │
                  ┌────────────┐                            │
                  │ PROCESSING │                            │
                  └─────┬──────┘                            │
                        │ First TTS chunk ready             │
                        ▼                                   │
                  ┌──────────┐                              │
                  │ SPEAKING │──── speech ends ─────────────┘
                  └────┬─────┘
                       │ INTERRUPT event
                       ▼
                  ┌──────────┐
                  │LISTENING │ (immediately, user is talking)
                  └──────────┘
```

**State definitions**:

| State | Active Coroutines | Can Transition To |
|---|---|---|
| IDLE | wake_word_listener | LISTENING |
| LISTENING | vad_processor, audio_buffer | PROCESSING, IDLE (timeout) |
| PROCESSING | llm_streamer, sentence_splitter, tts_synthesizer | SPEAKING |
| SPEAKING | audio_player, barge_in_listener, tts_synthesizer | IDLE, LISTENING |

**Timeout rules**:
- LISTENING → IDLE after 10 seconds of no speech detected (configurable)
- PROCESSING → IDLE after 30 seconds if no LLM response (error handling)

---

## 8. Module Structure

```
daisy/
│
├── main.py                          # Single entry point. Starts event loop, loads all components.
├── config.yaml                      # All configuration. No hardcoded values anywhere else.
├── SOUL.md                          # D.A.I.S.Y. personality / system prompt. Never hardcoded inline.
│
├── core/
│   ├── __init__.py
│   ├── state_machine.py             # DaisyState enum, StateManager class, transition logic
│   ├── event_bus.py                 # Central asyncio.Event registry, WAKE/INTERRUPT/STOP events
│   └── pipeline.py                  # Orchestrates full conversation turn: wake→vad→stt→llm→tts
│
├── audio/
│   ├── __init__.py
│   ├── input_stream.py              # sounddevice async input, raw audio yielder
│   ├── output_stream.py             # sounddevice async output, chunk player, hard-stop
│   ├── aec.py                       # Acoustic echo cancellation wrapper (WebRTC AEC)
│   └── audio_utils.py              # Resampling, format conversion, level normalization
│
├── wake_word/
│   ├── __init__.py
│   └── detector.py                  # OpenWakeWord wrapper, fires WAKE event on detection
│
├── vad/
│   ├── __init__.py
│   └── silero_vad.py               # Silero VAD ONNX wrapper, speech start/end detection
│
├── stt/
│   ├── __init__.py
│   └── faster_whisper_stt.py       # RealtimeSTT wrapper, returns transcript async
│
├── llm/
│   ├── __init__.py
│   ├── groq_client.py              # Groq API async streaming client
│   ├── gemini_client.py            # Gemini fallback async streaming client  
│   ├── router.py                   # Primary/fallback logic, rate limit handling
│   ├── sentence_splitter.py        # Async generator: tokens in → sentences out
│   └── tool_dispatcher.py          # Handles function call responses, routes to tools
│
├── tts/
│   ├── __init__.py
│   ├── kokoro_tts.py               # Kokoro TTS wrapper, async synthesis
│   ├── piper_tts.py                # Piper TTS wrapper, async synthesis
│   └── tts_router.py               # Primary/fallback, routes to kokoro or piper
│
├── barge_in/
│   ├── __init__.py
│   └── detector.py                  # AEC-cleaned VAD during SPEAKING state, fires INTERRUPT
│
├── memory/
│   ├── __init__.py
│   ├── conversation_buffer.py       # In-process turn history, FIFO with summarization
│   ├── sqlite_store.py              # Facts, preferences, session summaries
│   ├── chroma_store.py             # Vector semantic memory (ChromaDB)
│   └── memory_manager.py           # Unified interface: store, recall, inject into context
│
├── tools/
│   ├── __init__.py
│   ├── tool_registry.py            # LLM function call schemas + handler builder
│   ├── system_tools.py             # Time, system info, shell, reminders
│   ├── web_tools.py                # Web search (DuckDuckGo) + URL browsing
│   ├── file_tools.py               # Safe file read/write
│   ├── background_tools.py         # Background task spawning
│   ├── task_tracker.py             # Background task lifecycle management
│   └── announcement_queue.py       # Proactive announcement queue
│
├── api/
│   ├── __init__.py
│   ├── server.py                   # FastAPI app, WebSocket endpoint
│   ├── routes.py                   # REST endpoints (status, memory, config)
│   └── ws_handler.py               # WebSocket: audio streaming, transcript push, status
│
├── utils/
│   ├── __init__.py
│   ├── config_loader.py            # Loads config.yaml, validates, provides typed access
│   ├── logger.py                   # Structured logging, log levels, file rotation
│   └── model_loader.py             # Loads all ML models at startup, health checks
│
└── tests/
    ├── test_vad.py
    ├── test_stt.py
    ├── test_tts.py
    ├── test_pipeline.py
    ├── test_barge_in.py
    └── test_memory.py
```

**Key principles**:
- Each module has exactly one responsibility
- All inter-module communication goes through `event_bus.py` or explicit async queues
- No module imports from another module's internals (only from `__init__.py`)
- `config.yaml` is the single source of truth for all tunables
- `SOUL.md` is the single source of truth for personality — never inline strings

---

## 9. Latency Strategy

Latency is the primary UX metric. Every architectural decision should be evaluated against its latency impact.

### Latency Budget (Target End-to-End)

```
User stops speaking → D.A.I.S.Y. starts speaking first word

VAD endpoint detection:     ~150-300ms   (silence threshold)
STT transcription:          ~150-300ms   (small.en int8)
LLM first token:            ~200-400ms   (Groq LPU)
Sentence 1 complete:        ~300-600ms   (depends on sentence length)
TTS synthesis sentence 1:   ~100-200ms   (Kokoro RTF ~0.3)
Audio buffer + play start:  ~20-50ms

Total target:               ~1.0-1.5 seconds
Acceptable maximum:         2.0 seconds
```

### Latency Reduction Techniques (in order of impact)

**1. Streaming LLM + streaming TTS (highest impact)**  
Never wait for full LLM response. First sentence out of LLM → immediately to TTS → immediately to audio. This alone cuts perceived latency by 60-80% compared to batch processing.

**2. Pre-loaded warm models**  
All models (Silero, Whisper, Kokoro) loaded at startup and kept in memory. Zero cold-start penalty on inference.

**3. Audio double-buffering**  
While chunk N plays, chunk N+1 is synthesized. No gap between audio chunks.

**4. Parallel pipeline stages**  
VAD and barge-in detection run concurrently (not sequentially) with TTS playback. LLM generation overlaps TTS synthesis overlaps audio playback.

**5. VAD endpoint tuning**  
Silence endpoint threshold is the biggest latency knob. Default 600ms, tune to taste. Going below 400ms risks false endpoints (cutting off mid-sentence).

**6. Groq inference speed**  
Groq's LPU produces 500-800 tokens/second vs ~50-100 for typical GPU inference. First token arrives in ~200ms. This is why Groq is the primary choice.

**7. Short sentences first**  
Instruct the LLM in SOUL.md to lead responses with short, direct sentences. The first thing D.A.I.S.Y. says should be short — the elaboration can come after. This means the first audio chunk is synthesized faster.

**8. Acknowledgment cue as latency masking**  
After wake word or barge-in, an immediate audio cue (< 100ms) plays. This tells the user the system heard them. The user's perception of "waiting" starts from the cue, not from when they stopped speaking. This buys 200-400ms of processing time without the user noticing.

---

## 10. Build Phases

Build strictly in order. Do not skip phases. Do not add Phase N+1 features to Phase N.

### Phase 1 — Core Voice Loop (MVP)
**Goal**: End-to-end voice conversation working. No tools, no memory, no wake word.

Deliverables:
- [x] Project repo initialized, config.yaml defined, SOUL.md written
- [x] `audio/input_stream.py` — microphone audio stream
- [x] `vad/silero_vad.py` — VAD detecting speech start/end
- [x] `stt/faster_whisper_stt.py` — transcript from audio
- [x] `llm/groq_client.py` — streaming LLM response
- [x] `llm/sentence_splitter.py` — sentence chunking from token stream
- [x] `tts/kokoro_tts.py` — speech synthesis
- [x] `audio/output_stream.py` — audio playback
- [x] `core/pipeline.py` — wires all of above into one conversation turn
- [x] `main.py` — simple loop: VAD → STT → LLM → TTS → repeat

**Success criterion**: You can speak to D.A.I.S.Y. and she responds naturally within 2 seconds. The streaming pipeline is working. No crashes on 10 consecutive turns.

---

### Phase 2 — Wake Word + State Machine
**Goal**: Proper activation flow, clean state management.

Deliverables:
- [x] `wake_word/detector.py` — OpenWakeWord running continuously
- [x] Custom wake word trained ("DAISY")
- [x] `core/state_machine.py` — IDLE/LISTENING/PROCESSING/SPEAKING states
- [x] `core/event_bus.py` — WAKE event, state transition events
- [x] Timeout handling (LISTENING → IDLE after silence)
- [ ] Audio acknowledgment cues at state transitions
- [x] Refactor `main.py` to be event-driven rather than loop-based

**Success criterion**: D.A.I.S.Y. only activates when she hears "DAISY". Returns to IDLE correctly. Audio cues feel natural.

---

### Phase 3 — Barge-in Interruption
**Goal**: User can interrupt D.A.I.S.Y. mid-sentence.

Deliverables:
- [ ] `audio/aec.py` — acoustic echo cancellation on mic input
- [ ] `barge_in/detector.py` — VAD on AEC-cleaned audio during SPEAKING
- [ ] INTERRUPT event in event bus
- [ ] Hard-stop logic in audio player
- [ ] TTS queue flush on interruption
- [ ] LLM task cancellation on interruption
- [ ] State transition: SPEAKING → LISTENING on interrupt

**Success criterion**: During D.A.I.S.Y. speaking, user says anything → she stops within 200ms and starts listening. No self-interruption (AEC working).

---

### Phase 4 — Memory System
**Goal**: D.A.I.S.Y. remembers context across conversations.

Deliverables:
- [ ] `memory/conversation_buffer.py` — in-process turn history
- [ ] `memory/sqlite_store.py` — fact/preference/session storage
- [ ] `memory/chroma_store.py` — vector semantic store
- [ ] `memory/memory_manager.py` — unified interface
- [ ] Memory injection into LLM system prompt
- [ ] Conversation summarization before buffer overflow
- [ ] "Remember this, DAISY" explicit memory command
- [ ] Gemini fallback LLM for long-context tasks

**Success criterion**: After a restart, D.A.I.S.Y. remembers your name, recent project context, and key facts from the last session. Conversation history feels continuous.

---

### Phase 5 — Tool Integration
**Goal**: D.A.I.S.Y. can actually do things.

Deliverables:
- [x] `tools/tool_registry.py` — LLM function call schemas + handler builder (13 tools)
- [x] `tools/system_tools.py` — time, system info, sandboxed shell, reminders
- [x] `tools/web_tools.py` — DuckDuckGo search + URL browsing via httpx + trafilatura
- [x] `tools/file_tools.py` — path-validated safe file read/write
- [x] `tools/background_tools.py` — background shell and OpenCode task spawning
- [x] `tools/task_tracker.py` — background task lifecycle management (create, list, cancel, status)
- [x] `tools/announcement_queue.py` — proactive announcement queue for reminders and task completion
- [x] Tool result injection back into conversation stream (tool loop up to 5 rounds, then streaming response)

**Note**: Original plan called for OpenClaw (a Node.js daemon) as the tool executor. After evaluation, OpenClaw was abandoned in favour of direct Python tool implementations. The integration complexity of a separate daemon with cross-process IPC was not justified for D.A.I.S.Y.'s well-defined tool set. All 13 tools are in-process async Python functions.

**Success criterion**: "DAISY, what's the weather in Chennai?" → she searches → answers. "DAISY, what time is it?" → immediate answer. "DAISY, what's 2+2?" → answers without tool call (LLM knowledge).

---

### Phase 6 — PWA Client + Remote Access
**Goal**: Browser interface accessible anywhere via Tailscale.

Deliverables:
- [ ] `api/server.py` — FastAPI + uvicorn
- [ ] `api/ws_handler.py` — WebSocket for real-time
- [ ] PWA frontend (text input, transcript display, status)
- [ ] Conversation history view
- [ ] Settings panel
- [ ] systemd user service for auto-start on Andromeda boot

**Success criterion**: From Niggatron browser over Tailscale, can text-chat with D.A.I.S.Y. and see live responses. Status indicator shows correct state. System auto-starts after Andromeda reboot.

---

### Phase 7 — Polish + JARVIS Features (Future)
- Proactive alerts (D.A.I.S.Y. speaks without being asked — reminders, system alerts)
- Nextcloud integration for calendar, contacts, notes
- Multi-device audio (phone as mic/speaker node via PWA)
- "Always listening" mode with contextual awareness
- Voice cloning for a more distinctive D.A.I.S.Y. voice

---

## 11. Non-Functional Requirements

### Performance
- End-to-end latency (speech end → first audio out): < 2 seconds (target < 1.5s)
- Wake word detection latency: < 100ms
- Barge-in stop response: < 200ms
- Tool call response (with verbal acknowledgment): < 5 seconds
- Memory retrieval (SQLite): < 50ms
- Memory retrieval (ChromaDB): < 200ms

### Reliability
- Daemon must not crash on LLM API errors — graceful fallback to Gemini, then error speech
- Audio device disconnection must be handled gracefully
- All tool call failures must be caught and reported to user verbally
- Automatic restart via systemd on unexpected crash

### Resource Usage (on Andromeda)
- Idle CPU: < 10% (wake word + VAD continuously running)
- Active CPU: < 60% during full pipeline run
- RAM: < 2GB total (all models loaded)
- Disk: < 5GB total (models + ChromaDB + SQLite)

### Security
- Tailscale-only access — no public internet exposure
- No credentials stored in code — all in environment variables
- Custom tool sandbox: command whitelist, path validation, file size caps, timeouts
- File system access restricted to configured directories
- No conversation data leaves Andromeda (LLM API sends text, not audio)

### Code Quality
- Type hints on all function signatures
- Docstrings on all public classes and methods
- No function longer than 50 lines (refactor if needed)
- No `time.sleep()` anywhere — `await asyncio.sleep()` only
- All configuration in `config.yaml` — zero hardcoded values
- Logging at appropriate levels (DEBUG for audio frames, INFO for state transitions, ERROR for failures)

---

## 12. Out of Scope (for now)

These are acknowledged goals but explicitly deferred to avoid scope creep:

- **GPU acceleration** — will work on CPU, GPU support can be added later
- **Multi-language support** — English only for v2
- **Conversation export / search** — future feature
- **Multi-user support** — single-user system, no authentication
- **Mobile companion app** — PWA covers mobile via browser
- **Proactive / autonomous behavior** — D.A.I.S.Y. only speaks when addressed
- **Computer vision** — screen capture, camera integration
- **Home automation** — smart device control
- **Nextcloud integration** — Phase 7+
- **Custom voice cloning** — Phase 7+
- **Wake word "just knows you're talking to it"** — Phase 7+, requires contextual classifier

---

*Document version 2.0.0 — Last updated May 2026*  
*D.A.I.S.Y. v2 — Built for Andromeda. Built to feel real.*
