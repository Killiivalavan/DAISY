from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512
    input_device: Optional[int] = None
    output_device: Optional[int] = None


class VADConfig(BaseModel):
    speech_start_frames: int = 8
    speech_end_frames: int = 20
    max_recording_seconds: int = 15
    startup_ignore_ms: int = 500
    silero_threshold: float = 0.5


class STTConfig(BaseModel):
    model: str = "small.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"


class GroqConfig(BaseModel):
    api_key_env: str = "GROQ_API_KEY"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 1024
    base_url: str = "https://api.groq.com/openai/v1"


class LLMConfig(BaseModel):
    primary: str = "groq"
    groq: GroqConfig = GroqConfig()
    system_prompt_path: str = "SOUL.md"


class KokoroConfig(BaseModel):
    voice: str = "af_heart"
    sample_rate: int = 24000
    speed: float = 1.0
    lang_code: str = "a"


class TTSConfig(BaseModel):
    primary: str = "kokoro"
    kokoro: KokoroConfig = KokoroConfig()


class WakeWordConfig(BaseModel):
    model: str = "models/daisy.onnx"
    threshold: float = 0.5

class PipelineConfig(BaseModel):
    listening_timeout: int = 10
    processing_timeout: int = 30

class OpenCodeConfig(BaseModel):
    enabled: bool = True
    project_root: str = "/home/bashman/Code"

class ToolsConfig(BaseModel):
    enabled: bool = True
    allowed_directories: list[str] = ["/home/bashman", "/tmp"]
    file_max_size_bytes: int = 1048576
    allowed_commands: list[str] = ["df", "free", "uptime", "uname", "whoami", "ls", "cat", "ps", "ping", "systemctl"]
    default_timeout: int = 30
    max_timeout: int = 300
    opencode: OpenCodeConfig = OpenCodeConfig()

class MemoryConfig(BaseModel):
    max_turns: int = 20
    db_path: str = "~/.daisy/memory.db"
    inject_facts: bool = True
    max_facts_to_inject: int = 15

class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8443

class Config(BaseModel):
    audio: AudioConfig = AudioConfig()
    vad: VADConfig = VADConfig()
    stt: STTConfig = STTConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    pipeline: PipelineConfig = PipelineConfig()
    wake_word: WakeWordConfig = WakeWordConfig()
    memory: MemoryConfig = MemoryConfig()
    tools: ToolsConfig = ToolsConfig()
    api: ApiConfig = ApiConfig()
    mode: str = "wake_word"


def load_config(path: str = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        return Config()

    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(**raw) if raw else Config()
