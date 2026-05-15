"""
D.A.I.S.Y. v2 — System Diagnostic Script

Run: python -m daisy.tests.diagnostic
"""

import importlib
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def check(ok: bool, msg: str, hint: str = ""):
    icon = PASS if ok else FAIL
    print(f"  {icon} {msg}")
    if not ok and hint:
        print(f"      {hint}")
    return ok


def section(title: str):
    print(f"\n\033[1m{title}\033[0m")


def main():
    errors = 0
    print("\033[96mD.A.I.S.Y. v2 Diagnostic\033[0m")
    print(f"Python: {sys.version}")

    section("1. Python Version")
    ok = check(
        sys.version_info >= (3, 11),
        f"Python \u2265 3.11 ({sys.version_info.major}.{sys.version_info.minor})",
    )
    if not ok:
        errors += 1

    section("2. Python Packages")
    packages = [
        "sounddevice", "numpy", "webrtcvad",
        "faster_whisper", "openai", "kokoro", "pydantic", "yaml",
    ]
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            check(True, pkg)
        except ImportError as e:
            check(False, pkg, str(e))
            errors += 1

    section("3. Environment Variables")
    for label, var in [("LLM", "GROQ_API_KEY"), ("Fallback", "GEMINI_API_KEY")]:
        ok = check(
            os.environ.get(var) is not None,
            f"{label}: {var} is set",
            f"export {var}='your-key'",
        )
        if not ok:
            errors += 1

    section("4. Audio Devices")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        if len(devices) > 0:
            check(True, f"Found {len(devices)} audio device(s)")
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            input_name = (
                sd.query_devices(default_input)["name"]
                if default_input is not None else "none"
            )
            output_name = (
                sd.query_devices(default_output)["name"]
                if default_output is not None else "none"
            )
            print(f"      Default input:  [{default_input}] {input_name}")
            print(f"      Default output: [{default_output}] {output_name}")
        else:
            check(False, "No audio devices found")
            errors += 1
    except Exception as e:
        check(False, f"Failed to query audio devices: {e}")
        errors += 1

    section("5. WebRTC VAD")
    try:
        import webrtcvad
        import numpy as np
        vad = webrtcvad.Vad()
        vad.set_mode(0)
        frame = (np.ones(320, dtype=np.int16) * 8000).tobytes()
        result = vad.is_speech(frame, 16000)
        check(result, "WebRTC VAD detects speech in test frame")
    except Exception as e:
        check(False, "WebRTC VAD", str(e))
        errors += 1

    section("6. Model Loading")
    model_ok = True

    if model_ok:
        try:
            from kokoro import KPipeline
            KPipeline(lang_code="a")
            check(True, "Kokoro TTS pipeline created")
        except Exception as e:
            check(False, "Kokoro TTS pipeline", str(e))
            errors += 1

    if model_ok:
        try:
            from faster_whisper import WhisperModel
            m = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            check(True, "Faster-Whisper tiny.en loaded")
        except Exception as e:
            check(False, "Faster-Whisper model", str(e))
            errors += 1

    section("7. Configuration")
    try:
        from daisy.utils.config_loader import load_config
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        config = load_config(str(config_path))
        check(True, f"config.yaml loaded ({config_path})")
        print(f"      Mode: {config.mode}")
        print(f"      VAD threshold: {config.vad.silero_threshold}")
        print(f"      STT model: {config.stt.model}")
        print(f"      TTS voice: {config.tts.kokoro.voice}")
    except Exception as e:
        check(False, "Config loading", str(e))
        errors += 1

    section("\nSummary")
    if errors == 0:
        print(f"  {PASS} All checks passed - system is ready.")
    else:
        print(f"  {FAIL} {errors} check(s) failed - review above.")

    return errors


if __name__ == "__main__":
    sys.exit(main())
