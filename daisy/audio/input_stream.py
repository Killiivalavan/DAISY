import asyncio
import sounddevice as sd
import numpy as np


class AudioInputStream:
    def __init__(self, config):
        self.sample_rate = config.audio.sample_rate
        self.channels = config.audio.channels
        self.chunk_size = config.audio.chunk_size
        self.device = config.audio.input_device
        self._queue = asyncio.Queue()
        self._stream = None

    def _callback(self, indata, frames, time, status):
        self._loop.call_soon_threadsafe(self._queue.put_nowait, indata.copy())

    async def start(self):
        while not self._queue.empty():
            self._queue.get_nowait()
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.chunk_size,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    async def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def read(self) -> np.ndarray:
        return await self._queue.get()
