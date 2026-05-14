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


class TTSConfig(BaseModel):
    primary: str = "kokoro"
    kokoro: KokoroConfig = KokoroConfig()


class WakeWordConfig(BaseModel):
    model: str = "models/daisy.onnx"
    threshold: float = 0.5

class PipelineConfig(BaseModel):
    listening_timeout: int = 10
    processing_timeout: int = 30

class Config(BaseModel):
    audio: AudioConfig = AudioConfig()
    vad: VADConfig = VADConfig()
    stt: STTConfig = STTConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    pipeline: PipelineConfig = PipelineConfig()
    wake_word: WakeWordConfig = WakeWordConfig()
    mode: str = "wake_word"


def load_config(path: str = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        return Config()

    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(**raw) if raw else Config()
