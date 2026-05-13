import numpy as np


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    return audio.astype(np.float32) / 32768.0


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    return (audio * 32768.0).astype(np.int16)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    new_len = int(len(audio) * ratio)
    return np.interp(
        np.linspace(0, len(audio) - 1, new_len),
        np.arange(len(audio)),
        audio,
    )


def normalize(audio: np.ndarray, target_level: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * target_level
    return np.clip(audio, -1.0, 1.0)
