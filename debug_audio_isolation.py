#!/usr/bin/env python
"""
Audio Isolation Debug Script for DAISY
Implements systematic debugging approach to isolate transcription issues
"""
import os
import sys
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
import wave
import scipy.io.wavfile

# Add src to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def step1_analyze_wav_file(filepath):
    """Step 1: Extract and analyze WAV files directly"""
    print("🔍 STEP 1: Analyzing WAV file directly")
    print("=" * 50)
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    # Method 1: soundfile
    print("📊 Method 1: soundfile analysis")
    try:
        audio_data, sample_rate = sf.read(filepath)
        print(f"   Sample rate: {sample_rate}")
        print(f"   Shape: {audio_data.shape}")
        print(f"   Dtype: {audio_data.dtype}")
        print(f"   Duration: {len(audio_data)/sample_rate:.3f}s")
        print(f"   Min/Max: {np.min(audio_data):.6f} / {np.max(audio_data):.6f}")
        print(f"   RMS: {np.sqrt(np.mean(audio_data**2)):.6f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 2: scipy.io.wavfile
    print("\n📊 Method 2: scipy.io.wavfile analysis")
    try:
        rate, data = scipy.io.wavfile.read(filepath)
        print(f"   Sample rate: {rate}")
        print(f"   Shape: {data.shape}")
        print(f"   Dtype: {data.dtype}")
        if data.dtype == np.int16:
            data_float = data.astype(np.float32) / 32767.0
        else:
            data_float = data.astype(np.float32)
        print(f"   Duration: {len(data)/rate:.3f}s")
        print(f"   Min/Max (normalized): {np.min(data_float):.6f} / {np.max(data_float):.6f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 3: wave module
    print("\n📊 Method 3: wave module analysis")
    try:
        with wave.open(filepath, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            print(f"   Frames: {frames}")
            print(f"   Sample rate: {sample_rate}")
            print(f"   Channels: {channels}")
            print(f"   Sample width: {sample_width} bytes")
            print(f"   Duration: {frames/sample_rate:.3f}s")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Generate waveform plot
    print("\n📈 Generating waveform plot...")
    try:
        audio_data, sr = librosa.load(filepath, sr=None)
        
        plt.figure(figsize=(12, 8))
        
        # Waveform
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(audio_data, sr=sr)
        plt.title('Waveform')
        plt.ylabel('Amplitude')
        
        # Spectrogram
        plt.subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        output_plot = filepath.replace('.wav', '_analysis.png')
        plt.tight_layout()
        plt.savefig(output_plot)
        plt.close()
        print(f"   📊 Plot saved to: {output_plot}")
        
        # Audio quality analysis
        print(f"\n🎵 Audio Quality Analysis:")
        print(f"   Zero crossings: {librosa.zero_crossings(audio_data).sum()}")
        print(f"   Spectral centroid mean: {np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr)):.2f}")
        
        # Check for silence
        energy = np.sum(audio_data ** 2)
        print(f"   Total energy: {energy:.6f}")
        if energy < 1e-6:
            print("   ⚠️ WARNING: Very low energy - likely silence")
        
    except Exception as e:
        print(f"   ❌ Error generating plots: {e}")
    
    return audio_data, sample_rate

def step2_test_baseline_whisper(filepath):
    """Step 2: Test with original OpenAI Whisper"""
    print("\n🔍 STEP 2: Testing with baseline OpenAI Whisper")
    print("=" * 50)
    
    try:
        import whisper
        print("✅ OpenAI Whisper available")
        
        # Load model
        model = whisper.load_model("base")
        print("✅ Model loaded")
        
        # Transcribe
        result = model.transcribe(filepath, language="en", fp16=False)
        print(f"📝 OpenAI Whisper Result: '{result['text']}'")
        print(f"📝 Language: {result.get('language', 'unknown')}")
        
        return result['text']
        
    except ImportError:
        print("❌ OpenAI Whisper not available - installing...")
        os.system("pip install openai-whisper")
        return None
    except Exception as e:
        print(f"❌ Error with OpenAI Whisper: {e}")
        return None

def step3_test_faster_whisper_no_vad(filepath):
    """Step 3: Test faster-whisper without VAD"""
    print("\n🔍 STEP 3: Testing faster-whisper WITHOUT VAD")
    print("=" * 50)
    
    try:
        from faster_whisper import WhisperModel
        
        # Create fresh model instance
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print("✅ Faster-whisper model loaded")
        
        # Test without VAD
        print("🔸 Test 1: No VAD filtering")
        segments, info = model.transcribe(
            filepath,
            language="en",
            temperature=0.0,
            vad_filter=False  # DISABLED
        )
        
        segments_list = list(segments)
        text_no_vad = " ".join([segment.text for segment in segments_list]).strip()
        print(f"   Result: '{text_no_vad}'")
        print(f"   Segments: {len(segments_list)}")
        
        # Test with minimal VAD
        print("\n🔸 Test 2: Minimal VAD filtering")
        segments, info = model.transcribe(
            filepath,
            language="en",
            temperature=0.0,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 50,  # Very short
                "speech_pad_ms": 100
            }
        )
        
        segments_list = list(segments)
        text_minimal_vad = " ".join([segment.text for segment in segments_list]).strip()
        print(f"   Result: '{text_minimal_vad}'")
        print(f"   Segments: {len(segments_list)}")
        
        return text_no_vad, text_minimal_vad
        
    except Exception as e:
        print(f"❌ Error with faster-whisper: {e}")
        return None, None

def step4_validate_audio_format(filepath):
    """Step 4: Validate audio format consistency"""
    print("\n🔍 STEP 4: Validating audio format consistency")
    print("=" * 50)
    
    try:
        # Read with soundfile
        audio_data, sample_rate = sf.read(filepath)
        
        print("🔸 Format validation:")
        print(f"   Dtype: {audio_data.dtype}")
        print(f"   Shape: {audio_data.shape}")
        print(f"   Sample rate: {sample_rate}")
        print(f"   Max amplitude: {np.max(np.abs(audio_data)):.6f}")
        
        # Assertions
        try:
            assert len(audio_data.shape) == 1, f"Expected 1D array, got {audio_data.shape}"
            assert sample_rate == 16000, f"Expected 16000 Hz, got {sample_rate}"
            assert np.max(np.abs(audio_data)) <= 1.0, f"Amplitude exceeds 1.0: {np.max(np.abs(audio_data))}"
            print("✅ All format validations passed")
        except AssertionError as e:
            print(f"⚠️ Format validation failed: {e}")
        
        # Check for common corruption patterns
        unique_values = len(np.unique(audio_data))
        print(f"   Unique values: {unique_values}")
        if unique_values < 100:
            print("   ⚠️ WARNING: Very few unique values - potential corruption")
        
        nan_count = np.sum(np.isnan(audio_data))
        inf_count = np.sum(np.isinf(audio_data))
        print(f"   NaN values: {nan_count}")
        print(f"   Infinite values: {inf_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating format: {e}")
        return False

def step5_test_fresh_model_instances(filepath):
    """Step 5: Test with fresh model instances"""
    print("\n🔍 STEP 5: Testing with fresh model instances")
    print("=" * 50)
    
    try:
        from faster_whisper import WhisperModel
        
        # Test 1: Fresh model each time
        print("🔸 Test 1: Fresh model instance")
        model1 = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model1.transcribe(filepath, language="en", temperature=0.0, vad_filter=False)
        text1 = " ".join([segment.text for segment in segments]).strip()
        print(f"   Result 1: '{text1}'")
        del model1  # Explicit cleanup
        
        # Test 2: Another fresh model
        print("\n🔸 Test 2: Another fresh model instance")
        model2 = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model2.transcribe(filepath, language="en", temperature=0.0, vad_filter=False)
        text2 = " ".join([segment.text for segment in segments]).strip()
        print(f"   Result 2: '{text2}'")
        del model2
        
        # Test 3: Different compute type
        print("\n🔸 Test 3: Different compute type (float32)")
        model3 = WhisperModel("base", device="cpu", compute_type="float32")
        segments, info = model3.transcribe(filepath, language="en", temperature=0.0, vad_filter=False)
        text3 = " ".join([segment.text for segment in segments]).strip()
        print(f"   Result 3: '{text3}'")
        del model3
        
        print(f"\n📊 Consistency check:")
        print(f"   All results identical: {text1 == text2 == text3}")
        
        return text1, text2, text3
        
    except Exception as e:
        print(f"❌ Error testing fresh models: {e}")
        return None, None, None

def step6_simple_recording_test():
    """Step 6: Simple single-threaded recording test"""
    print("\n🔍 STEP 6: Simple single-threaded recording test")
    print("=" * 50)
    
    try:
        print("🎤 Recording 3 seconds of audio...")
        print("Say something now!")
        
        # Simple recording
        duration = 3  # seconds
        sample_rate = 16000
        
        audio = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1, 
                      dtype='float32')
        sd.wait()  # Wait until recording is finished
        
        print("✅ Recording complete")
        
        # Save directly
        simple_file = "data/simple_recording.wav"
        os.makedirs("data", exist_ok=True)
        sf.write(simple_file, audio.flatten(), sample_rate, subtype='PCM_16')
        
        print(f"💾 Saved to: {simple_file}")
        
        # Analyze this simple recording
        step1_analyze_wav_file(simple_file)
        step2_test_baseline_whisper(simple_file)
        step3_test_faster_whisper_no_vad(simple_file)
        
        return simple_file
        
    except Exception as e:
        print(f"❌ Error in simple recording: {e}")
        return None

def main():
    """Run all diagnostic steps"""
    print("🚀 DAISY Audio Isolation Debug Script")
    print("=" * 60)
    
    # Check for existing recording
    from src.utils.config import RECORDING_FILE
    
    if os.path.exists(RECORDING_FILE):
        print(f"📁 Found existing recording: {RECORDING_FILE}")
        
        # Run all steps on existing file
        step1_analyze_wav_file(RECORDING_FILE)
        step2_test_baseline_whisper(RECORDING_FILE)
        step3_test_faster_whisper_no_vad(RECORDING_FILE)
        step4_validate_audio_format(RECORDING_FILE)
        step5_test_fresh_model_instances(RECORDING_FILE)
    else:
        print(f"❌ No existing recording found at {RECORDING_FILE}")
    
    # Always do the simple recording test
    print("\n" + "="*60)
    simple_file = step6_simple_recording_test()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Existing file analysis: {'✅' if os.path.exists(RECORDING_FILE) else '❌'}")
    print(f"   Simple recording test: {'✅' if simple_file else '❌'}")
    print(f"\n📋 Check the generated plots and logs above for insights!")

if __name__ == "__main__":
    main() 