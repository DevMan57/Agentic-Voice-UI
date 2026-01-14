"""
Standalone SER Calibration Tool - Captures YOUR voice's emotional baseline.

This tool works INDEPENDENTLY - no need to run Voice Chat first.

Usage:
    # Two-step process (Windows + WSL):
    1. Run on Windows: python tools/calibrate_record_windows.py
    2. Run in WSL:     python tools/calibrate_emotion_standalone.py --from-files

    # Or if running natively (not WSL):
    python tools/calibrate_emotion_standalone.py

Requirements:
    - Run AFTER main installation is complete
    - For WSL: Use --from-files with pre-recorded audio
"""

import time
import sys
import os
import re
import shutil
import wave
import tempfile
from pathlib import Path
from datetime import datetime
import argparse

# Add parent dir to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

EMOTION_DETECTOR_PATH = SCRIPT_DIR / "audio" / "emotion_detector.py"
TEMP_AUDIO_DIR = SCRIPT_DIR / "recordings"


def record_audio(duration: float = 3.0, sample_rate: int = 16000) -> str:
    """Record audio from microphone for specified duration. Returns path to temp wav file."""
    try:
        import pyaudio
        import numpy as np
    except ImportError:
        print("ERROR: pyaudio not installed. Run: pip install pyaudio")
        return None

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = sample_rate

    p = pyaudio.PyAudio()

    # Find default input device
    try:
        default_device = p.get_default_input_device_info()
        print(f"   Using microphone: {default_device['name']}")
    except Exception as e:
        print(f"   Warning: Could not get default device info: {e}")

    print(f"   Recording for {duration} seconds...")
    print("   >>> Speak now! <<<")

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []
    num_chunks = int(RATE / CHUNK * duration)

    for i in range(num_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        # Show progress
        progress = int((i + 1) / num_chunks * 20)
        bar = '█' * progress + '░' * (20 - progress)
        print(f"\r   [{bar}] {(i+1)/num_chunks*100:.0f}%", end="", flush=True)

    print()  # New line after progress bar

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save to temp file
    TEMP_AUDIO_DIR.mkdir(exist_ok=True)
    temp_path = TEMP_AUDIO_DIR / f"calibration_{int(time.time())}.wav"

    with wave.open(str(temp_path), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT) if hasattr(p, 'get_sample_size') else 2)
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"   Recorded: {temp_path.name}")
    return str(temp_path)


def analyze_recording(filepath: str) -> dict:
    """Analyze a recording with the SER model. Returns VAD dict or None."""
    try:
        from audio.emotion_detector import EmotionDetector

        detector = EmotionDetector()
        result = detector.analyze(audio_path=filepath)

        if result:
            return {
                'valence': result.valence,
                'arousal': result.arousal,
                'dominance': result.dominance,
                'label': result.label
            }
        return None
    except RuntimeError as e:
        if "CUDA" in str(e) or "out of memory" in str(e).lower():
            print(f"   [GPU busy, trying CPU...]")
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                # Re-import to force CPU
                import importlib
                import audio.emotion_detector
                importlib.reload(audio.emotion_detector)
                from audio.emotion_detector import EmotionDetector
                detector = EmotionDetector()
                result = detector.analyze(audio_path=filepath)
                if result:
                    return {
                        'valence': result.valence,
                        'arousal': result.arousal,
                        'dominance': result.dominance,
                        'label': result.label
                    }
            except Exception as e2:
                print(f"   [CPU fallback failed: {e2}]")
        else:
            print(f"   [Analysis error: {e}]")
        return None
    except Exception as e:
        print(f"   [Analysis error: {e}]")
        import traceback
        traceback.print_exc()
        return None


def generate_threshold_code(neutral: dict, happy: dict = None, frustrated: dict = None) -> str:
    """Generate the _vad_to_label method code based on calibration results."""
    # Calculate thresholds from neutral baseline
    calm_a_max = min(neutral['arousal'] + 0.15, 0.30)
    high_a_min = max(neutral['arousal'] + 0.25, 0.35)
    low_v = max(neutral['valence'] - 0.15, 0.25)
    high_v = min(neutral['valence'] + 0.20, 0.65)

    if happy:
        high_v = max((neutral['valence'] + happy['valence']) / 2, 0.55)

    if frustrated:
        high_a_min = max((neutral['arousal'] + frustrated['arousal']) / 2, 0.30)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f'''    def _vad_to_label(self, v: float, a: float, d: float) -> str:
        """Map VAD values to emotion label (Personalized via calibration)"""
        # Calibrated: {timestamp}
        # Neutral baseline: V={neutral['valence']:.2f}, A={neutral['arousal']:.2f}

        # 1. Low arousal zone (calm/tired)
        if a < {calm_a_max:.2f}:
            if v < {low_v:.2f}:
                return 'tired'
            return 'calm'

        # 2. High arousal zone (excited/frustrated)
        if a > {high_a_min:.2f}:
            if v < 0.40:
                return 'frustrated'
            if v > {high_v:.2f}:
                return 'excited'
            return 'anxious'

        # 3. Mid arousal zone (happy/annoyed/neutral)
        if v > {high_v:.2f}:
            return 'happy'
        if v < {low_v:.2f}:
            return 'annoyed'

        return 'neutral'
'''


def apply_thresholds(new_code: str) -> bool:
    """Replace _vad_to_label method in emotion_detector.py with new thresholds."""
    if not EMOTION_DETECTOR_PATH.exists():
        print(f"   [Error: {EMOTION_DETECTOR_PATH} not found]")
        return False

    # Create backup
    backup_path = EMOTION_DETECTOR_PATH.with_suffix('.py.bak')
    shutil.copy2(EMOTION_DETECTOR_PATH, backup_path)
    print(f"   [Backup created: {backup_path.name}]")

    # Read current content
    content = EMOTION_DETECTOR_PATH.read_text(encoding='utf-8')

    # Find and replace the _vad_to_label method
    # Pattern matches from "def _vad_to_label" to the next method or end of class
    pattern = r'(    def _vad_to_label\(self, v: float, a: float, d: float\) -> str:.*?)((?=\n    def )|(?=\nclass )|(?=\n[^\s]))'

    match = re.search(pattern, content, re.DOTALL)
    if not match:
        # Try simpler pattern
        pattern = r'    def _vad_to_label\(self[^)]*\)[^:]*:.*?(?=\n    def |\n\nclass |\Z)'
        match = re.search(pattern, content, re.DOTALL)

    if match:
        # Replace the method
        new_content = content[:match.start()] + new_code + content[match.end():]
        EMOTION_DETECTOR_PATH.write_text(new_content, encoding='utf-8')
        print(f"   [SUCCESS: Thresholds applied to {EMOTION_DETECTOR_PATH.name}]")
        return True
    else:
        print("   [Error: Could not find _vad_to_label method in file]")
        print("   [Thresholds NOT applied - manual copy required]")
        return False


def calibrate(dry_run: bool = False, duration: float = 3.0, from_files: bool = False):
    """Main calibration routine."""
    print()
    print("=" * 65)
    print("   STANDALONE SER CALIBRATION TOOL")
    print("   Captures YOUR voice's emotional baseline")
    print("=" * 65)
    print()
    print("This tool calibrates emotion detection to YOUR voice.")
    if from_files:
        print("Mode: Analyzing pre-recorded files from recordings/ folder")
    else:
        print("You'll be asked to speak with different emotions.")
        print(f"Each recording lasts {duration} seconds.")
        print()
        print("Make sure your microphone is connected and working.")
    print()
    print(f"Dry run mode: {'YES' if dry_run else 'NO (will auto-apply)'}")
    print()
    print("-" * 65)

    # Pre-load the model
    print("[1/2] Loading SER model (this may take a moment)...")
    try:
        import numpy as np
        from audio.emotion_detector import EmotionDetector
        detector = EmotionDetector()
        # Force load with dummy audio
        detector.analyze(audio_array=np.zeros(16000, dtype=np.float32), sample_rate=16000)
        print("      Model loaded successfully.")
    except RuntimeError as e:
        if "CUDA" in str(e) or "out of memory" in str(e).lower():
            print(f"      GPU unavailable, will use CPU for analysis")
        else:
            print(f"ERROR loading model: {e}")
            return
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("-" * 65)
    print("[2/2] Analyzing emotional samples...")
    print("-" * 65)
    print()

    # Emotions to capture/analyze
    if from_files:
        # Use pre-recorded files from Windows
        emotion_files = [
            ("NEUTRAL", TEMP_AUDIO_DIR / "calibration_neutral.wav"),
            ("HAPPY", TEMP_AUDIO_DIR / "calibration_happy.wav"),
            ("FRUSTRATED", TEMP_AUDIO_DIR / "calibration_frustrated.wav"),
            ("TIRED/SAD", TEMP_AUDIO_DIR / "calibration_tired.wav"),
        ]

        results = {}
        for emotion_name, filepath in emotion_files:
            print(f">>> {emotion_name}")
            if not filepath.exists():
                print(f"    File not found: {filepath.name}")
                print(f"    Run calibrate_record_windows.py first!")
                print()
                continue

            print(f"    Analyzing: {filepath.name}")
            vad = analyze_recording(str(filepath))

            if vad:
                print(f"    Result: V={vad['valence']:.3f}  A={vad['arousal']:.3f}  D={vad['dominance']:.3f}")
                print(f"    Detected as: {vad['label']}")
                results[emotion_name] = vad
            else:
                print(f"    Analysis failed")
            print()
    else:
        # Record live (only works outside WSL)
        emotions = [
            ("NEUTRAL", "Speak normally, like reading a news article:\n    'The weather today is partly cloudy with temperatures around 70 degrees.'"),
            ("HAPPY", "Say something with genuine happiness or excitement:\n    'I just got amazing news! This is the best day ever!'"),
            ("FRUSTRATED", "Say with frustration or annoyance:\n    'Why does this never work? This is so frustrating!'"),
            ("TIRED/SAD", "Say with low energy or sadness:\n    'I'm so tired. I just want to rest.'"),
        ]

        results = {}
        for emotion_name, prompt in emotions:
            print(f">>> {emotion_name}")
            print(f"    {prompt}")
            print()

            input("    Press ENTER when ready to record...")

            filepath = record_audio(duration=duration)

            if not filepath:
                print(f"    Recording failed, skipping...\n")
                continue

            print("   Analyzing...")
            vad = analyze_recording(filepath)

            if vad:
                print(f"    Result: V={vad['valence']:.3f}  A={vad['arousal']:.3f}  D={vad['dominance']:.3f}")
                print(f"    Detected as: {vad['label']}")
                results[emotion_name] = vad
            else:
                print(f"    Analysis failed")

            print()

    # Generate thresholds
    print("=" * 65)
    print("   CALIBRATION RESULTS")
    print("=" * 65)
    print()

    if not results:
        print("No samples captured successfully.")
        print("Check your microphone and try again.")
        return

    # Show raw values
    print("Your Voice Baselines:")
    print("-" * 40)
    for emotion, vad in results.items():
        print(f"  {emotion:12} V={vad['valence']:.3f}  A={vad['arousal']:.3f}  D={vad['dominance']:.3f}")
    print()

    if "NEUTRAL" not in results:
        print("WARNING: No NEUTRAL sample captured. Cannot calculate thresholds.")
        print("Run calibration again and make sure to record the NEUTRAL sample.")
        return

    neutral = results["NEUTRAL"]
    happy = results.get("HAPPY")
    frustrated = results.get("FRUSTRATED")

    print(f"Your NEUTRAL baseline: V={neutral['valence']:.2f}, A={neutral['arousal']:.2f}")
    print()

    # Generate threshold code
    new_code = generate_threshold_code(neutral, happy, frustrated)

    print("-" * 65)
    print("GENERATED THRESHOLDS:")
    print("-" * 65)
    print(new_code)
    print("-" * 65)

    if dry_run:
        print()
        print("DRY RUN MODE - Thresholds NOT applied.")
        print("To apply, run without --dry-run flag.")
    else:
        print()
        print("Applying thresholds to emotion_detector.py...")
        success = apply_thresholds(new_code)
        if success:
            print()
            print("=" * 65)
            print("   CALIBRATION COMPLETE!")
            print("=" * 65)
            print()
            print("Emotion detection is now calibrated to your voice.")
            print("Enable 'Emotion Detection' in the Voice Agent settings to use it.")
        else:
            print()
            print("AUTO-APPLY FAILED. Manual steps:")
            print("  1. Open audio/emotion_detector.py")
            print("  2. Find the _vad_to_label method")
            print("  3. Replace it with the code above")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone SER Calibration Tool")
    parser.add_argument("--dry-run", action="store_true", help="Preview thresholds without applying")
    parser.add_argument("--duration", type=float, default=3.0, help="Recording duration in seconds (default: 3)")
    parser.add_argument("--from-files", action="store_true", help="Use pre-recorded files from recordings/ folder (for WSL)")
    args = parser.parse_args()

    try:
        calibrate(dry_run=args.dry_run, duration=args.duration, from_files=args.from_files)
    except KeyboardInterrupt:
        print("\n\nCalibration cancelled.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    input("\nPress ENTER to exit...")
