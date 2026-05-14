import numpy as np
from kokoro import KPipeline


class KokoroTTS:
    def __init__(self, config):
        self._pipeline = None
        self.voice = config.tts.kokoro.voice
        self.sample_rate = config.tts.kokoro.sample_rate

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = KPipeline(lang_code="a")
        return self._pipeline

    def synthesize(self, text: str) -> np.ndarray:
        audio_chunks = []
        for result in self.pipeline(text, voice=self.voice, speed=1.0):
            audio_chunks.append(result.audio)
        if not audio_chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(audio_chunks)
