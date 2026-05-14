import numpy as np
import pytest
from daisy.stt.faster_whisper_stt import FasterWhisperSTT


class FakeConfig:
    class SttConfig:
        model = "tiny.en"
        device = "cpu"
        compute_type = "int8"

    stt = SttConfig()


def test_model_not_loaded_at_init():
    stt = FasterWhisperSTT(FakeConfig())
    assert stt._model is None


def test_model_loaded_on_first_call(mocker):
    mock_model = mocker.MagicMock()
    mocker.patch(
        "daisy.stt.faster_whisper_stt.WhisperModel", return_value=mock_model
    )
    stt = FasterWhisperSTT(FakeConfig())
    _ = stt.model
    assert stt._model is not None


@pytest.mark.asyncio
async def test_transcribe_returns_text(mocker):
    mock_segment = mocker.MagicMock()
    mock_segment.text = "hello world"
    mock_model = mocker.MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], None)
    mocker.patch(
        "daisy.stt.faster_whisper_stt.WhisperModel", return_value=mock_model
    )
    stt = FasterWhisperSTT(FakeConfig())
    result = await stt.transcribe(np.zeros(16000, dtype=np.float32))
    assert result == "hello world"


@pytest.mark.asyncio
async def test_transcribe_strips_whitespace(mocker):
    mock_segment = mocker.MagicMock()
    mock_segment.text = "  hello  "
    mock_model = mocker.MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], None)
    mocker.patch(
        "daisy.stt.faster_whisper_stt.WhisperModel", return_value=mock_model
    )
    stt = FasterWhisperSTT(FakeConfig())
    result = await stt.transcribe(np.zeros(16000, dtype=np.float32))
    assert result == "hello"
