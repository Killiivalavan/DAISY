import numpy as np
import pytest
from daisy.tts.kokoro_tts import KokoroTTS


class FakeConfig:
    class KokoroConfig:
        voice = "af_heart"
        sample_rate = 24000
        speed = 1.0
        lang_code = "a"

    tts = type("TTSConfig", (), {"kokoro": KokoroConfig(), "primary": "kokoro"})()


def test_pipeline_not_loaded_at_init():
    tts = KokoroTTS(FakeConfig())
    assert tts._pipeline is None


def test_pipeline_loaded_on_first_call(mocker):
    mock_result = mocker.MagicMock()
    mock_result.audio = np.ones(100, dtype=np.float32)
    mock_pipeline = mocker.MagicMock()
    mock_pipeline.return_value = [mock_result]
    mocker.patch("daisy.tts.kokoro_tts.KPipeline", return_value=mock_pipeline)

    tts = KokoroTTS(FakeConfig())
    _ = tts.pipeline
    assert tts._pipeline is not None


@pytest.mark.asyncio
async def test_synthesize_returns_float32_array(mocker):
    mock_result = mocker.MagicMock()
    mock_result.audio = np.ones(100, dtype=np.float32)
    mock_pipeline = mocker.MagicMock()
    mock_pipeline.return_value = [mock_result]
    mocker.patch("daisy.tts.kokoro_tts.KPipeline", return_value=mock_pipeline)

    tts = KokoroTTS(FakeConfig())
    result = await tts.synthesize("Hello.")
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) > 0


@pytest.mark.asyncio
async def test_synthesize_empty_text_returns_empty(mocker):
    mock_pipeline = mocker.MagicMock()
    mock_pipeline.return_value = []
    mocker.patch("daisy.tts.kokoro_tts.KPipeline", return_value=mock_pipeline)

    tts = KokoroTTS(FakeConfig())
    result = await tts.synthesize("")
    assert isinstance(result, np.ndarray)
    assert len(result) == 0
