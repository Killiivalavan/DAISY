import numpy as np
import pytest
from daisy.audio.audio_utils import int16_to_float32, float32_to_int16, resample, normalize


def test_int16_to_float32_identity():
    assert int16_to_float32(np.int16(0)) == 0.0


def test_int16_to_float32_max_value():
    assert int16_to_float32(np.int16(32767)) == pytest.approx(0.99997, abs=1e-4)


def test_int16_to_float32_min_value():
    assert int16_to_float32(np.int16(-32768)) == pytest.approx(-1.0, abs=1e-5)


def test_float32_to_int16_identity():
    assert float32_to_int16(np.float32(0.0)) == 0


def test_float32_to_int16_roundtrip():
    original = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    as_int = float32_to_int16(original)
    back = int16_to_float32(as_int)
    assert np.allclose(original, back, atol=1e-4)


def test_resample_same_rate():
    audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    result = resample(audio, 16000, 16000)
    assert len(result) == len(audio)
    assert np.allclose(result, audio)


def test_resample_downsample():
    audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    result = resample(audio, 16000, 8000)
    assert len(result) < len(audio)


def test_resample_upsample():
    audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    result = resample(audio, 8000, 16000)
    assert len(result) > len(audio)


def test_normalize_silence():
    audio = np.zeros(100, dtype=np.float32)
    result = normalize(audio)
    assert np.allclose(result, np.zeros(100))


def test_normalize_peak():
    audio = np.array([0.5, -0.3, 0.8, -0.2], dtype=np.float32)
    result = normalize(audio, target_level=0.95)
    assert np.max(np.abs(result)) == pytest.approx(0.95, abs=1e-5)


def test_normalize_clips():
    audio = np.array([2.0, -1.5], dtype=np.float32)
    result = normalize(audio)
    assert np.all(np.abs(result) <= 1.0)
