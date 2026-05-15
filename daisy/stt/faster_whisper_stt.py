import asyncio
import numpy as np
from faster_whisper import WhisperModel


class FasterWhisperSTT:
    def __init__(self, config):
        self._model = None
        self._config = config

    @property
    def model(self):
        if self._model is None:
            self._model = WhisperModel(
                self._config.stt.model,
                device=self._config.stt.device,
                compute_type=self._config.stt.compute_type,
            )
        return self._model

    async def warmup(self):
        await asyncio.to_thread(lambda: self.model)

    async def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = await asyncio.to_thread(self.model.transcribe, audio, language=self._config.stt.language)
        text = "".join(segment.text for segment in segments)
        return text.strip()
