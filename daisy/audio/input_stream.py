import asyncio
from abc import ABC, abstractmethod

import numpy as np
import sounddevice as sd


class AudioSource(ABC):
    """Abstract audio input source."""

    @abstractmethod
    async def start(self):
        """Begin capturing audio."""

    @abstractmethod
    async def stop(self):
        """Stop capturing audio."""

    @abstractmethod
    def flush(self):
        """Synchronously discard any buffered audio."""

    @abstractmethod
    async def read(self) -> np.ndarray:
        """Read the next chunk of audio. Blocks until available."""


class LocalAudioSource(AudioSource):
    """Audio source wrapping the local microphone via sounddevice."""

    def __init__(self, config):
        self._sample_rate = config.audio.sample_rate
        self._channels = config.audio.channels
        self._chunk_size = config.audio.chunk_size
        self._device = config.audio.input_device
        self._queue: asyncio.Queue = asyncio.Queue()
        self._stream = None

    def _callback(self, indata, frames, time, status):
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, indata.copy())
        except asyncio.QueueFull:
            pass  # drop chunk; consumer is falling behind

    async def start(self):
        while not self._queue.empty():
            self._queue.get_nowait()
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            blocksize=self._chunk_size,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    async def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def flush(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def read(self) -> np.ndarray:
        return await self._queue.get()


class NetworkAudioSource(AudioSource):
    """Audio source that receives PCM chunks from a remote client via WebSocket.

    Call ``push()`` from the WebSocket handler to feed audio in.
    The wake word detector and VAD consume via ``read()``.
    """

    def __init__(self, maxsize: int = 200):
        self._buffer: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    async def start(self):
        pass

    async def stop(self):
        self.flush()

    def flush(self):
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def read(self) -> np.ndarray:
        return await self._buffer.get()

    def push(self, pcm_chunk: np.ndarray):
        """Feed a chunk of int16 PCM audio from the network."""
        try:
            self._buffer.put_nowait(pcm_chunk)
        except asyncio.QueueFull:
            # Drop oldest to make room — better to lose old audio than build latency
            try:
                self._buffer.get_nowait()
                self._buffer.put_nowait(pcm_chunk)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass
