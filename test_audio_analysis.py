#!/usr/bin/env python
"""
Audio Analysis Test for DAISY - Diagnose transcription issues
"""
import os
import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(current_dir))

def analyze_audio_file(filepath):
    """Analyze an audio file for potential issues."""
    print(f"\n=== Analyzing {filepath} ===")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    try:
        # Load audio file
        audio_data, sample_rate = sf.read(filepath)
        duration = len(audio_data) / sample_rate
        
        print(f"📊 Basic Info:")
        print(f"   Duration: {duration:.3f} seconds")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Channels: {audio_data.ndim}")
        print(f"   Data type: {audio_data.dtype}")
        print(f"   Shape: {audio_data.shape}")
        print(f"   File size: {os.path.getsize(filepath)} bytes")
        
        # Audio statistics
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()
            
        print(f"\n📈 Audio Statistics:")
        print(f"   Min value: {np.min(audio_data):.6f}")
        print(f"   Max value: {np.max(audio_data):.6f}")
        print(f"   Mean: {np.mean(audio_data):.6f}")
        print(f"   RMS: {np.sqrt(np.mean(audio_data**2)):.6f}")
        print(f"   Peak amplitude: {np.max(np.abs(audio_data)):.6f}")
        
        # Check for silence
        silence_threshold = 0.001
        non_silent_samples = np.sum(np.abs(audio_data) > silence_threshold)
        silence_ratio = 1.0 - (non_silent_samples / len(audio_data))
        
        print(f"\n🔇 Silence Analysis:")
        print(f"   Silence threshold: {silence_threshold}")
        print(f"   Non-silent samples: {non_silent_samples}/{len(audio_data)}")
        print(f"   Silence ratio: {silence_ratio:.2%}")
        
        # Check for clipping
        clipping_threshold = 0.95
        clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
        clipping_ratio = clipped_samples / len(audio_data)
        
        print(f"\n✂️ Clipping Analysis:")
        print(f"   Clipping threshold: {clipping_threshold}")
        print(f"   Clipped samples: {clipped_samples}/{len(audio_data)}")
        print(f"   Clipping ratio: {clipping_ratio:.2%}")
        
        # Check for suspicious patterns
        print(f"\n🔍 Pattern Analysis:")
        
        # Check for repeated values (potential corruption)
        unique_values = len(np.unique(audio_data))
        print(f"   Unique values: {unique_values}")
        if unique_values < 100:
            print(f"   ⚠️ WARNING: Very few unique values - potential corruption")
            
        # Check for NaN or infinite values
        nan_count = np.sum(np.isnan(audio_data))
        inf_count = np.sum(np.isinf(audio_data))
        print(f"   NaN values: {nan_count}")
        print(f"   Infinite values: {inf_count}")
        if nan_count > 0 or inf_count > 0:
            print(f"   ⚠️ WARNING: Invalid values detected")
            
        # Check for DC offset
        dc_offset = np.mean(audio_data)
        print(f"   DC offset: {dc_offset:.6f}")
        if abs(dc_offset) > 0.1:
            print(f"   ⚠️ WARNING: Large DC offset detected")
            
        # Energy distribution
        energy = audio_data ** 2
        energy_percentiles = np.percentile(energy, [25, 50, 75, 90, 95, 99])
        print(f"\n⚡ Energy Distribution:")
        for i, p in enumerate([25, 50, 75, 90, 95, 99]):
            print(f"   {p}th percentile: {energy_percentiles[i]:.8f}")
            
        # Frequency analysis (basic)
        if len(audio_data) > 1024:
            fft = np.fft.fft(audio_data[:1024])
            freqs = np.fft.fftfreq(1024, 1/sample_rate)
            magnitude = np.abs(fft)
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
            dominant_freq = freqs[dominant_freq_idx]
            
            print(f"\n🎵 Frequency Analysis (first 1024 samples):")
            print(f"   Dominant frequency: {abs(dominant_freq):.1f} Hz")
            print(f"   Magnitude at dominant freq: {magnitude[dominant_freq_idx]:.2f}")
        
        return {
            'duration': duration,
            'sample_rate': sample_rate,
            'peak_amplitude': np.max(np.abs(audio_data)),
            'rms': np.sqrt(np.mean(audio_data**2)),
            'silence_ratio': silence_ratio,
            'clipping_ratio': clipping_ratio,
            'unique_values': unique_values,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'dc_offset': dc_offset
        }
        
    except Exception as e:
        print(f"❌ Error analyzing audio: {e}")
        return None

def test_whisper_directly():
    """Test Whisper transcription directly on the recording file."""
    print(f"\n=== Testing Whisper Transcription ===")
    
    try:
        from src.voice.speech_recognition import SpeechRecognizer
        from src.utils.config import RECORDING_FILE
        
        print(f"Initializing SpeechRecognizer...")
        recognizer = SpeechRecognizer(model_name="base", device="cpu", use_wake_word=False)
        
        if os.path.exists(RECORDING_FILE):
            print(f"Transcribing {RECORDING_FILE}...")
            
            # Test with different configurations
            configurations = [
                {"vad_filter": False, "description": "No VAD filtering"},
                {"vad_filter": True, "vad_parameters": {"min_silence_duration_ms": 50}, "description": "Light VAD filtering"},
                {"vad_filter": True, "vad_parameters": {"min_silence_duration_ms": 100}, "description": "Current VAD settings"},
                {"vad_filter": True, "vad_parameters": {"min_silence_duration_ms": 500}, "description": "Heavy VAD filtering"}
            ]
            
            for i, config in enumerate(configurations):
                print(f"\n--- Test {i+1}: {config['description']} ---")
                
                try:
                    # Temporarily modify the model's transcribe method
                    segments, info = recognizer.model.transcribe(
                        RECORDING_FILE,
                        language="en",
                        **{k: v for k, v in config.items() if k != "description"}
                    )
                    
                    # Convert segments to list and get text
                    segments_list = list(segments)
                    transcribed_text = " ".join([segment.text for segment in segments_list]).strip()
                    
                    print(f"   Result: '{transcribed_text}'")
                    print(f"   Segments: {len(segments_list)}")
                    
                    if hasattr(info, 'language'):
                        print(f"   Detected language: {info.language}")
                    if hasattr(info, 'language_probability'):
                        print(f"   Language probability: {info.language_probability:.3f}")
                        
                except Exception as e:
                    print(f"   ❌ Error: {e}")
        else:
            print(f"❌ Recording file not found: {RECORDING_FILE}")
            
    except Exception as e:
        print(f"❌ Error setting up Whisper test: {e}")

def test_vad_processing():
    """Test VAD processing on the recorded audio."""
    print(f"\n=== Testing VAD Processing ===")
    
    try:
        import webrtcvad
        from src.utils.config import RECORDING_FILE, WEBRTC_VAD_MODE
        
        if not os.path.exists(RECORDING_FILE):
            print(f"❌ Recording file not found: {RECORDING_FILE}")
            return
            
        # Load audio
        audio_data, sample_rate = sf.read(RECORDING_FILE)
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()
            
        print(f"Testing WebRTC VAD with mode {WEBRTC_VAD_MODE}")
        print(f"Audio duration: {len(audio_data)/sample_rate:.3f} seconds")
        
        # Initialize VAD
        vad = webrtcvad.Vad(WEBRTC_VAD_MODE)
        
        # Convert to 16kHz if needed
        if sample_rate != 16000:
            print(f"Converting from {sample_rate}Hz to 16000Hz...")
            from scipy.signal import resample
            target_length = int(len(audio_data) * 16000 / sample_rate)
            audio_data = resample(audio_data, target_length)
            sample_rate = 16000
        
        # Convert to int16 PCM
        pcm_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        
        # Test different frame sizes
        frame_durations = [10, 20, 30]  # ms
        
        for frame_duration_ms in frame_durations:
            frame_length = int(sample_rate * frame_duration_ms / 1000)
            print(f"\n--- Frame duration: {frame_duration_ms}ms ({frame_length} samples) ---")
            
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(pcm_data) - frame_length + 1, frame_length):
                frame = pcm_data[i:i + frame_length]
                if len(frame) == frame_length:
                    try:
                        is_speech = vad.is_speech(frame.tobytes(), sample_rate)
                        if is_speech:
                            speech_frames += 1
                        total_frames += 1
                    except Exception as e:
                        print(f"   ❌ VAD error on frame {total_frames}: {e}")
                        
            speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
            print(f"   Speech frames: {speech_frames}/{total_frames}")
            print(f"   Speech ratio: {speech_ratio:.2%}")
            
    except Exception as e:
        print(f"❌ Error in VAD testing: {e}")

def main():
    """Run comprehensive audio analysis."""
    print("🎤 DAISY Audio Analysis Tool")
    print("=" * 50)
    
    # Check for existing recording
    from src.utils.config import RECORDING_FILE
    
    if os.path.exists(RECORDING_FILE):
        # Analyze the existing recording
        result = analyze_audio_file(RECORDING_FILE)
        
        # Test VAD processing
        test_vad_processing()
        
        # Test Whisper transcription with different settings
        test_whisper_directly()
        
    else:
        print(f"❌ No recording file found at {RECORDING_FILE}")
        print("Please run the voice assistant first to create a recording.")
    
    print(f"\n✅ Audio analysis complete!")

if __name__ == "__main__":
    main() 