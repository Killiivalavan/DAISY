import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import sounddevice as sd


class AudioSink(ABC):
    """Abstract audio output sink."""

    @abstractmethod
    async def start(self):
        """Initialize the output device."""

    @abstractmethod
    def stop(self):
        """Shut down the output device. Synchronous — may be called from signal handler."""

    @abstractmethod
    def play(self, audio: np.ndarray):
        """Queue audio for playback. Non-blocking."""

    @abstractmethod
    def clear(self):
        """Flush the playback queue immediately (barge-in)."""

    @property
    @abstractmethod
    def is_playing(self) -> bool:
        """True if audio is currently queued or playing."""

    @abstractmethod
    async def wait_until_done(self):
        """Block until all queued audio has finished playing."""


class LocalAudioSink(AudioSink):
    """Audio sink wrapping the local speakers via sounddevice."""

    def __init__(self, config):
        self._sample_rate = config.tts.kokoro.sample_rate
        self._channels = 1
        self._device = config.audio.output_device
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
            samplerate=self._sample_rate,
            channels=self._channels,
            device=self._device,
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


class NetworkAudioSink(AudioSink):
    """Audio sink that sends TTS audio to a remote client via WebSocket.

    Each ``play()`` call sends a binary WebSocket frame (0x01 prefix + PCM).
    ``clear()`` sends a JSON ``audio_clear`` message so the client stops playback.
    """

    def __init__(self, ws=None):
        self._ws = ws  # Set after construction when WS connection is established
        self._playing = False

    def set_ws(self, ws):
        self._ws = ws

    async def start(self):
        pass

    def stop(self):
        pass

    def play(self, audio: np.ndarray):
        if self._ws is None:
            return
        self._playing = True
        frame = bytes([0x01]) + audio.astype(np.int16).tobytes()
        asyncio.create_task(self._send_frame(frame))

    async def _send_frame(self, frame: bytes):
        try:
            await self._ws.send_bytes(frame)
        except Exception:
            self._playing = False

    def clear(self):
        self._playing = False
        if self._ws is not None:
            asyncio.create_task(self._ws.send_json({"type": "audio_clear"}))

    @property
    def is_playing(self) -> bool:
        return self._playing

    async def wait_until_done(self):
        # Network sink can't truly know when the client finishes playback
        # without an ACK. Approximate with a short settle time.
        await asyncio.sleep(0.1)
        self._playing = False
