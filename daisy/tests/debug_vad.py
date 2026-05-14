"""
D.A.I.S.Y. VAD Debug Script
Tests the real microphone -> WebRTC VAD pipeline to diagnose
why speech never reaches the transcriber.

Run: python -m daisy.tests.debug_vad
"""

import asyncio
import time
import sys
import numpy as np
import sounddevice as sd


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def test_audio_capture(duration=3):
    """Test if the microphone is actually producing audio data."""
    section("1. Microphone Audio Capture Test")
    print(f"  Capturing {duration}s of audio from default input device...")

    device_info = sd.query_devices(None, 'input')
    print(f"  Default input device: [{device_info['index']}] {device_info['name']}")
    print(f"  Default samplerate: {device_info['default_samplerate']} Hz")
    print(f"  Max input channels: {device_info['max_input_channels']}")

    sample_rate = int(device_info['default_samplerate'])
    chunk_size = 512
    chunks = []
    peaks = []

    def callback(indata, frames, time_info, status):
        chunk = indata.copy()
        chunks.append(chunk)
        peak = np.max(np.abs(chunk))
        peaks.append(peak)
        if status:
            print(f"  [WARN] Stream status: {status}", file=sys.stderr)

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        blocksize=chunk_size,
        callback=callback,
    )

    with stream:
        print(f"  Recording... (say something or make noise)")
        total_chunks = int(sample_rate / chunk_size * duration)
        waited = 0
        while len(chunks) < total_chunks and waited < duration + 1:
            await asyncio.sleep(0.1)
            waited += 0.1

    if not chunks:
        print("  [FAIL] No audio chunks received!")
        return None, None

    all_audio = np.concatenate([c.flatten() for c in chunks])
    max_peak = max(peaks) if peaks else 0
    mean_peak = np.mean(peaks) if peaks else 0

    print(f"  Chunks received: {len(chunks)}")
    print(f"  Audio duration: {len(all_audio) / sample_rate:.2f}s")
    print(f"  Max peak: {max_peak:.6f}")
    print(f"  Mean peak: {mean_peak:.6f}")
    print(f"  RMS level: {np.sqrt(np.mean(all_audio**2)):.6f}")
    print(f"  Sample range: [{all_audio.min():.6f}, {all_audio.max():.6f}]")
    print(f"  Dtype: {all_audio.dtype}")

    if max_peak < 0.001:
        print("  [FAIL] Audio level extremely low - microphone may not be working!")
        print("          Check: arecord -l, alsamixer, pavucontrol")
    elif max_peak > 0.01:
        print("  [OK] Audio level looks healthy")
    else:
        print("  [WARN] Audio level is low, try speaking louder or increasing mic gain")

    return all_audio, sample_rate


def test_webrtc_vad(audio, sample_rate, mode=2, frame_ms=20):
    """Test WebRTC VAD on captured audio."""
    section(f"2. WebRTC VAD Test (mode={mode}, frame={frame_ms}ms)")

    import webrtcvad
    vad = webrtcvad.Vad()
    vad.set_mode(mode)

    frame_size = int(sample_rate * frame_ms / 1000)
    frame_bytes = frame_size * 2  # 16-bit PCM

    # Convert float32/int16 audio to int16 bytes
    if audio.dtype in (np.float32, np.float64):
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    else:
        audio_int16 = audio.astype(np.int16)

    audio_bytes = audio_int16.tobytes()

    total_frames = 0
    speech_frames = 0
    current_streak = 0
    max_streak = 0
    in_speech = False
    segments = []

    for offset in range(0, len(audio_bytes) - frame_bytes + 1, frame_bytes):
        frame = audio_bytes[offset:offset + frame_bytes]
        total_frames += 1
        try:
            is_speech = vad.is_speech(frame, sample_rate)
        except Exception as e:
            print(f"  [ERROR] VAD exception at frame {total_frames}: {e}")
            continue

        if is_speech:
            speech_frames += 1
            current_streak += 1
            max_streak = max(max_streak, current_streak)
            if not in_speech:
                in_speech = True
                segments.append({'start': offset, 'frames': []})
        else:
            current_streak = 0
            if in_speech:
                in_speech = False

        if in_speech and segments:
            segments[-1]['frames'].append(frame)

    speech_pct = (speech_frames / total_frames * 100) if total_frames > 0 else 0

    print(f"  Total frames ({frame_ms}ms each): {total_frames}")
    print(f"  Frames classified as speech: {speech_frames} ({speech_pct:.1f}%)")
    print(f"  Max consecutive speech frames: {max_streak}")
    print(f"  Speech segments detected: {len(segments)}")

    print(f"\n  Segment details:")
    for i, seg in enumerate(segments[:10]):
        dur = len(seg['frames']) * frame_ms / 1000
        start_time = seg['start'] / (sample_rate * 2)
        print(f"    [{i}] start={start_time:.2f}s, duration={dur:.2f}s, frames={len(seg['frames'])}")

    return speech_frames, total_frames


def test_noise_floor_adaptation():
    """Test the _above_noise_floor logic in isolation."""
    section("3. Noise Floor Adaptation Analysis")

    # Simulate the _above_noise_floor logic
    noise_floor = 0.001
    noise_floor_min = 0.001
    frames_seen = 0

    def above_noise_floor(peak):
        nonlocal noise_floor, frames_seen

        if frames_seen < 50:
            if peak > 0:
                noise_floor = min(noise_floor, peak)
            frames_seen += 1
            noise_floor = max(noise_floor, noise_floor_min)
            return True

        if peak < noise_floor:
            noise_floor += (peak - noise_floor) * 0.1
        elif peak > noise_floor:
            noise_floor += (peak - noise_floor) * 0.001

        noise_floor = max(noise_floor, noise_floor_min)
        return peak > noise_floor * 3.0

    print("  Simulating ambient noise at peak=0.005 for 1000 frames...")
    print(f"  {'Frame':>6} {'Peak':>8} {'NF':>8} {'Thresh':>8} {'Pass':>6}")

    for i in range(1050):
        peak = 0.005
        if i >= 950:  # speech starts at frame 950
            peak = 0.3
        passed = above_noise_floor(peak)
        if i < 55 or i % 100 == 0 or i == 950 or i == 951:
            threshold = noise_floor * 3.0
            print(f"  {i:>6} {peak:>8.4f} {noise_floor:>8.6f} {threshold:>8.6f} {str(passed):>6}")

    print(f"\n  Final noise_floor: {noise_floor:.6f}")
    print(f"  Final threshold (NF*3): {noise_floor * 3:.6f}")

    # Check: at this point, does speech pass?
    speech_peak = 0.3
    speech_pass = speech_peak > noise_floor * 3.0
    print(f"  Speech at peak=0.3: {speech_peak:.1f} > {noise_floor * 3:.6f} = {speech_pass}")

    # Check: what if we then go silent again?
    noise_floor_save = noise_floor
    for i in range(100):
        peak = 0.005
        above_noise_floor(peak)
    print(f"\n  After 100 more silence frames:")
    print(f"  Noise floor dropped from {noise_floor_save:.6f} to {noise_floor:.6f}")
    print(f"  Threshold: {noise_floor * 3:.6f}")
    print(f"  Ambient noise (0.005) passes: {0.005 > noise_floor * 3.0}")

    return noise_floor


def test_frame_alignment():
    """Test the frame alignment logic."""
    section("4. Frame Alignment Test")

    sample_rate = 16000
    chunk_size = 512  # from config
    frame_ms = 20
    frame_bytes = frame_ms * sample_rate // 1000 * 2  # 640

    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Chunk size: {chunk_size} samples")
    print(f"  Chunk duration: {chunk_size / sample_rate * 1000:.0f} ms")
    print(f"  VAD frame: {frame_ms}ms = {frame_bytes} bytes")
    print(f"  Chunk bytes: {chunk_size * 2} bytes")

    # Simulate the byte-level accumulation
    raw = bytearray()
    frames_processed = 0
    chunks_needed = 20  # process 20 chunks

    for chunk_idx in range(chunks_needed):
        # Simulate a 512-sample chunk of noise with speech-like peaks
        audio = np.random.randn(chunk_size) * 0.01  # low noise
        if chunk_idx >= 10:  # "speech" starts at chunk 10
            audio += np.sin(2 * np.pi * 440 * np.arange(chunk_size) / sample_rate) * 0.3

        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        raw.extend(audio_int16.tobytes())

        while len(raw) >= frame_bytes:
            frame = bytes(raw[:frame_bytes])
            raw = raw[frame_bytes:]
            frames_processed += 1

    frame_duration_actual = frames_processed * frame_ms / 1000
    print(f"\n  Chunks fed: {chunks_needed}")
    print(f"  VAD frames extracted: {frames_processed}")
    print(f"  Total audio duration: {frame_duration_actual:.2f}s")
    print(f"  Expected duration: {chunks_needed * chunk_size / sample_rate:.2f}s")

    if abs(frame_duration_actual - chunks_needed * chunk_size / sample_rate) < 0.001:
        print("  [OK] Frame alignment is correct")
    else:
        print(f"  [INFO] Small alignment diff: {abs(frame_duration_actual - chunks_needed * chunk_size / sample_rate)*1000:.1f}ms")


async def main():
    print("\n" + "█" * 60)
    print("  D.A.I.S.Y. VAD Diagnostic Tool")
    print("█" * 60)

    # Test 1: Audio capture
    audio, sample_rate = await test_audio_capture(duration=3)

    if audio is None:
        print("\n  Cannot proceed with VAD tests - no audio captured.")
        print("  Please check your microphone configuration.")
        return

    # If sample_rate != 16000, resample for WebRTC VAD
    if sample_rate != 16000:
        print(f"\n  [INFO] Resampling from {sample_rate}Hz to 16000Hz for VAD test")
        from daisy.audio.audio_utils import resample
        audio = resample(audio, sample_rate, 16000)
        sample_rate = 16000

    # Normalize to float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Test 2: WebRTC VAD at various modes
    for mode in [0, 1, 2, 3]:
        for frame_ms in [10, 20, 30]:
            frame_size = frame_ms * sample_rate // 1000 * 2
            if len(audio.tobytes()) < frame_size:
                print(f"  [SKIP] Audio too short for {frame_ms}ms frames")
                continue
            test_webrtc_vad(audio, sample_rate, mode=mode, frame_ms=frame_ms)
            break  # only test with one frame_ms per mode for brevity

    # Test 3: Noise floor analysis
    test_noise_floor_adaptation()

    # Test 4: Frame alignment
    test_frame_alignment()

    section("5. Recommendations")
    print("  Based on the test results above, the likely issue is:")
    print()
    print("  1) If Audio Capture shows very low peaks (< 0.001):")
    print("     -> Microphone not working with sounddevice. Check alsamixer/pavucontrol.")
    print()
    print("  2) If Audio Capture shows healthy peaks but VAD shows HIGH speech")
    print("     percentage during silence (e.g., > 50%):")
    print("     -> WebRTC VAD mode 2 is too aggressive for this microphone.")
    print("     -> Solution: set 'webrtc_mode: 3' in config.yaml")
    print()
    print("  3) If Audio Capture shows healthy peaks and VAD shows LOW speech")
    print("     percentage during speech:")
    print("     -> WebRTC VAD at current settings is missing speech.")
    print("     -> Solution: try mode 1 or 0, or switch to Silero VAD")
    print()
    print("  4) If Audio Capture shows healthy peaks but the noise floor")
    print("     adaptation is blocking detection:")
    print("     -> The _above_noise_floor function may need tuning.")
    print("     -> The noise floor only increases during sound and never fully")
    print("        resets between listen() calls.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
