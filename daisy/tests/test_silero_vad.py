import numpy as np
from daisy.vad.silero_vad import SileroVAD


class FakeConfig:
    class VadConfig:
        sample_rate = 16000
        frame_size = 512
        threshold = 0.5
        speech_start_frames = 3
        speech_end_frames = 18

    vad = VadConfig()


def test_model_not_loaded_at_init():
    vad = SileroVAD(FakeConfig())
    assert vad._model is None


def test_model_loaded_on_first_call(mocker):
    mock_model = mocker.MagicMock()
    mocker.patch("daisy.vad.silero_vad.load_silero_vad", return_value=mock_model)
    vad = SileroVAD(FakeConfig())
    assert vad._model is None
    _ = vad.model
    assert vad._model is mock_model


def test_get_speech_prob_returns_value(mocker):
    mock_model = mocker.MagicMock()
    mock_model.return_value.item.return_value = 0.8
    mocker.patch("daisy.vad.silero_vad.load_silero_vad", return_value=mock_model)
    vad = SileroVAD(FakeConfig())
    frame = np.zeros(512, dtype=np.float32)
    prob = vad.get_speech_prob(frame)
    assert prob == 0.8


def test_get_speech_prob_passes_tensor_and_sr(mocker):
    mock_model = mocker.MagicMock()
    mock_model.return_value.item.return_value = 0.5
    mocker.patch("daisy.vad.silero_vad.load_silero_vad", return_value=mock_model)
    vad = SileroVAD(FakeConfig())
    frame = np.ones(512, dtype=np.float32) * 0.5
    vad.get_speech_prob(frame)
    mock_model.assert_called_once()
    args, _ = mock_model.call_args
    assert args[1] == 16000
