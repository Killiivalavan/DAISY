import asyncio
import logging
import threading
import sounddevice as sd
import numpy as np
from collections import deque


class AudioOutputStream:
    def __init__(self, config):
        self.sample_rate = config.tts.kokoro.sample_rate
        self.channels = 1
        self.device = config.audio.output_device
        self._buffer = deque()
        self._lock = threading.Lock()
        self._stream = None

    def _callback(self, outdata, frames, time, status):
        if status:
            logging.warning(f"Audio output underflow/error: {status}")
        with self._lock:
            total = 0
            while total < frames and self._buffer:
                chunk = self._buffer[0]
                needed = frames - total
                if len(chunk) <= needed:
                    outdata[total : total + len(chunk)] = chunk.reshape(-1, 1)
                    total += len(chunk)
                    self._buffer.popleft()
                else:
                    outdata[total:frames] = chunk[:needed].reshape(-1, 1)
                    self._buffer[0] = chunk[needed:]
                    total = frames
            if total < frames:
                outdata[total:frames] = 0

    async def start(self):
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def play(self, audio: np.ndarray):
        with self._lock:
            self._buffer.append(audio)

    def clear(self):
        with self._lock:
            self._buffer.clear()

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return len(self._buffer) > 0

    async def wait_until_done(self):
        while self.is_playing:
            await asyncio.sleep(0.05)
