import numpy as np
import torch
from silero_vad import load_silero_vad


class SileroVAD:
    def __init__(self, config):
        self.model = load_silero_vad()
        self.sample_rate = config.vad.sample_rate
        self.frame_size = config.vad.frame_size
        self.threshold = config.vad.threshold
        self.speech_start_frames = config.vad.speech_start_frames
        self.speech_end_frames = config.vad.speech_end_frames

    def get_speech_prob(self, frame: np.ndarray) -> float:
        frame_tensor = torch.from_numpy(frame).float()
        with torch.no_grad():
            prob = self.model(frame_tensor, self.sample_rate).item()
        return prob

    async def listen(self, audio_input) -> np.ndarray:
        buffer = []
        speech_frames = 0
        silence_frames = 0
        is_speaking = False

        while True:
            chunk = await audio_input.read()
            frame = chunk.flatten()
            prob = self.get_speech_prob(frame)

            if not is_speaking:
                if prob > self.threshold:
                    speech_frames += 1
                    buffer.append(frame)
                    if speech_frames >= self.speech_start_frames:
                        is_speaking = True
                        speech_frames = 0
                        silence_frames = 0
                else:
                    speech_frames = 0
                    buffer = []
            else:
                buffer.append(frame)
                if prob > self.threshold:
                    silence_frames = 0
                else:
                    silence_frames += 1
                    if silence_frames >= self.speech_end_frames:
                        break

        return np.concatenate(buffer)
