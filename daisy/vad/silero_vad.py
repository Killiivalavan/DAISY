import asyncio
import logging
import sys
import time
import numpy as np
import torch
from silero_vad import load_silero_vad

logger = logging.getLogger(__name__)


class SileroVAD:
    def __init__(self, config):
        self._model = None
        self.sample_rate = config.audio.sample_rate
        # Silero needs chunks of 512 samples for 16kHz
        self.chunk_size = config.audio.chunk_size 
        self.threshold = config.vad.silero_threshold
        self.speech_start_frames = config.vad.speech_start_frames
        self.speech_end_frames = config.vad.speech_end_frames
        self.max_recording_seconds = config.vad.max_recording_seconds
        self.startup_ignore_ms = config.vad.startup_ignore_ms

    async def warmup(self):
        await asyncio.to_thread(lambda: self.model)

    @property
    def model(self):
        if self._model is None:
            self._model = load_silero_vad()
        return self._model

    def get_speech_prob(self, frame: np.ndarray) -> float:
        frame_tensor = torch.from_numpy(frame).float()
        with torch.no_grad():
            prob = self.model(frame_tensor, self.sample_rate).item()
        return prob

    async def listen(self, audio_input, timeout: float = None) -> np.ndarray:
        import asyncio

        # Flush the old audio queue so we don't process audio from 
        # while D.A.I.S.Y. was speaking or processing.
        audio_input.flush()

        # Skip audio for startup_ignore_ms to avoid processing stale audio
        # from the wake word period
        startup_chunks = int(self.startup_ignore_ms / (self.chunk_size / self.sample_rate * 1000))
        for _ in range(startup_chunks):
            await audio_input.read()

        ring_buffer_size = 15
        ring_buffer = []
        
        buffer = []
        silence_frames = 0
        is_speaking = False
        recording_start = 0.0
        listen_start = time.monotonic()
        
        window_size = 10
        speech_history = [False] * window_size
        history_idx = 0

        while True:
            chunk = await audio_input.read()
            frame = chunk.flatten()
            
            # Silero expects float32 in range [-1, 1]
            if frame.dtype != np.float32:
                # If it's int16, normalize to [-1, 1]
                if frame.dtype == np.int16:
                    frame = frame.astype(np.float32) / 32768.0
                else:
                    frame = frame.astype(np.float32)

            prob = self.get_speech_prob(frame)
            is_speech = prob > self.threshold

            # --- Diagnostic Logging ---
            rms = np.sqrt(np.mean(frame**2))
            if not hasattr(self, '_log_counter'): self._log_counter = 0
            self._log_counter += 1
            if self._log_counter % 20 == 0:
                logger.debug(f"  [VAD Debug] Vol: {rms:.3f} | Prob: {prob:.3f} | Speaking: {is_speaking}")
            # --------------------------

            if not is_speaking:
                # Timeout check: If we haven't started speaking and the timer expires
                if timeout is not None and (time.monotonic() - listen_start) > timeout:
                    logger.info("  [VAD] Listening timed out.")
                    return None

                ring_buffer.append(frame)
                if len(ring_buffer) > ring_buffer_size:
                    ring_buffer.pop(0)

                speech_history[history_idx] = is_speech
                history_idx = (history_idx + 1) % window_size
                
                if sum(speech_history) >= self.speech_start_frames:
                    is_speaking = True
                    silence_frames = 0
                    buffer = list(ring_buffer)
                    recording_start = time.monotonic()
            else:
                buffer.append(frame)
                if is_speech:
                    silence_frames = 0
                else:
                    silence_frames += 1
                    if silence_frames >= self.speech_end_frames:
                        # Remove trailing silence frames
                        if silence_frames <= len(buffer):
                            buffer = buffer[:-silence_frames]
                        logger.debug(f"  [VAD Debug] Reached {silence_frames} silence frames, returning audio.")
                        return np.concatenate(buffer)

                if time.monotonic() - recording_start > self.max_recording_seconds:
                    logger.debug(f"  [VAD Debug] Reached max recording seconds, returning audio.")
                    return np.concatenate(buffer)
