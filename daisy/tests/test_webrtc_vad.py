import numpy as np
import pytest
from daisy.vad.webrtc_vad import WebRTCVAD


class FakeAudioConfig:
    sample_rate = 16000
    channels = 1
    chunk_size = 512
    input_device = None
    output_device = None


class FakeVADConfig:
    webrtc_mode = 1
    frame_ms = 10
    speech_start_frames = 3
    speech_end_frames = 10
    max_recording_seconds = 15


class FakeConfig:
    audio = FakeAudioConfig()
    vad = FakeVADConfig()


def test_vad_initializes():
    vad = WebRTCVAD(FakeConfig())
    assert vad.sample_rate == 16000
    assert vad.frame_bytes == 320


def test_speech_frame_detected():
    vad = WebRTCVAD(FakeConfig())
    frame_size = 160
    t = np.linspace(0, frame_size / 16000, frame_size, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    audio_int16 = (audio * 32767).astype(np.int16)
    frame = audio_int16.tobytes()
    assert vad.vad.is_speech(frame, 16000) is True


def test_silence_frame_not_detected():
    vad = WebRTCVAD(FakeConfig())
    frame = np.zeros(160, dtype=np.int16).tobytes()
    assert vad.vad.is_speech(frame, 16000) is False


@pytest.mark.asyncio
async def test_listen_returns_audio_on_speech():
    vad = WebRTCVAD(FakeConfig())

    class FakeAudioInput:
        def __init__(self):
            self._call_count = 0
            self._speech_chunks = []

            frame_len = 160
            t = np.linspace(0, frame_len / 16000, frame_len, endpoint=False)
            speech = np.sin(2 * np.pi * 440 * t) * 0.3

            for _ in range(50):
                chunk = np.tile(speech, (512 // frame_len, 1)).flatten()
                self._speech_chunks.append(
                    chunk.reshape(-1, 1).astype(np.float32)
                )

            silence = np.zeros((512, 1), dtype=np.float32)
            for _ in range(30):
                self._speech_chunks.append(silence)

        async def read(self):
            if self._call_count < len(self._speech_chunks):
                result = self._speech_chunks[self._call_count]
                self._call_count += 1
                return result
            await asyncio.sleep(0.1)
            return np.zeros((512, 1), dtype=np.float32)

    import asyncio
    result = await vad.listen(FakeAudioInput())
    assert isinstance(result, np.ndarray)
    assert len(result) > 0
    assert result.dtype == np.float32
