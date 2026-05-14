import yaml
from pathlib import Path
from daisy.utils.config_loader import load_config, Config


def test_missing_file_returns_defaults(tmp_path):
    path = tmp_path / "nonexistent.yaml"
    config = load_config(str(path))
    assert isinstance(config, Config)
    assert config.audio.sample_rate == 16000
    assert config.vad.webrtc_mode == 2
    assert config.stt.model == "small.en"
    assert config.mode == "always_on"


def test_load_with_overrides(tmp_path):
    data = {
        "audio": {"sample_rate": 22050},
        "vad": {"webrtc_mode": 2, "speech_start_frames": 8},
        "stt": {"model": "tiny.en"},
        "mode": "wake_word",
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)

    config = load_config(str(path))
    assert config.audio.sample_rate == 22050
    assert config.vad.webrtc_mode == 2
    assert config.vad.speech_start_frames == 8
    assert config.stt.model == "tiny.en"
    assert config.mode == "wake_word"


def test_defaults_preserved_for_missing_keys(tmp_path):
    data = {"mode": "push_to_talk"}
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)

    config = load_config(str(path))
    assert config.mode == "push_to_talk"
    assert config.audio.sample_rate == 16000


def test_empty_yaml_returns_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    config = load_config(str(path))
    assert isinstance(config, Config)
