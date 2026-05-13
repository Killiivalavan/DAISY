import numpy as np
from faster_whisper import WhisperModel


class FasterWhisperSTT:
    def __init__(self, config):
        self.model = WhisperModel(
            config.stt.model,
            device=config.stt.device,
            compute_type=config.stt.compute_type,
        )

    async def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language="en")
        text = "".join(segment.text for segment in segments)
        return text.strip()
