"""
SER Calibration Tool - Captures YOUR voice's emotional baseline.

Run this WHILE Voice Chat is running (so VAD is active).
Uses PTT (Right Shift) to record - same as normal voice chat.

Usage:
    1. Start Voice Chat normally
    2. In a separate terminal: python tools/calibrate_emotion.py
    3. Follow prompts - hold Right Shift and speak each emotion
    4. Tool automatically applies personalized thresholds to emotion_detector.py

Options:
    --dry-run    Preview thresholds without applying them
"""

import time
import sys
import os
import re
import shutil
from pathlib import Path
from datetime import datetime
import argparse

# Add parent dir to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

RECORDINGS_DIR = SCRIPT_DIR / "recordings"
EMOTION_DETECTOR_PATH = SCRIPT_DIR / "audio" / "emotion_detector.py"


def get_newest_recording(after_time: float = None) -> Path:
    """Get the newest .wav file, optionally only those created after a timestamp."""
    wavs = list(RECORDINGS_DIR.glob("*.wav"))
    if not wavs:
        return None

    if after_time:
        wavs = [w for w in wavs if w.stat().st_mtime > after_time]
        if not wavs:
            return None

    return max(wavs, key=lambda p: p.stat().st_mtime)


def wait_for_file_stable(filepath: Path, stability_time: float = 0.5) -> bool:
    """Wait until file size stops changing (indicating write is complete)."""
    last_size = -1
    stable_since = None

    for _ in range(20):  # Max 10 seconds (20 * 0.5s)
        try:
            current_size = filepath.stat().st_size
            if current_size == last_size and current_size > 0:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= stability_time:
                    return True
            else:
                stable_since = None
                last_size = current_size
        except (OSError, FileNotFoundError):
            pass
        time.sleep(0.5)

    return False


def wait_for_new_recording(timeout: int = 60) -> Path:
    """Wait for a new recording to appear. Returns path or None on timeout."""
    print("   [Waiting for recording...]")
    print("   >>> Hold RIGHT SHIFT and speak now <<<")

    existing_files = set(RECORDINGS_DIR.glob("*.wav"))
    start_time = time.time()
    baseline_time = start_time

    current_newest = get_newest_recording()
    if current_newest:
        baseline_time = current_newest.stat().st_mtime

    while (time.time() - start_time) < timeout:
        elapsed = int(time.time() - start_time)
        print(f"\r   [Listening... {elapsed}s / {timeout}s]    ", end="", flush=True)

        newest = get_newest_recording(after_time=baseline_time)
        if newest:
            print(f"\n   [Detected: {newest.name}]")

            # Wait for file to be fully written (size stabilizes)
            if wait_for_file_stable(newest):
                size = newest.stat().st_size
                print(f"   [Captured: {size} bytes]")
                return newest
            else:
                print("   [Warning: File may still be writing]")
                return newest

        time.sleep(0.3)

    print("\n   [Timeout - no recording detected]")
    return None


def analyze_recording(filepath: Path) -> dict:
    """Analyze a recording with the SER model. Returns VAD dict or None."""
    try:
        from audio.emotion_detector import EmotionDetector

        detector = EmotionDetector()
        result = detector.analyze(audio_path=str(filepath))

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
            print(f"   [GPU out of memory, trying CPU...]")
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                from audio.emotion_detector import EmotionDetector
                detector = EmotionDetector()
                result = detector.analyze(audio_path=str(filepath))
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


def calibrate(dry_run: bool = False):
    """Main calibration routine."""
    print()
    print("=" * 65)
    print("   SER CALIBRATION TOOL")
    print("   Captures YOUR voice's emotional baseline")
    print("=" * 65)
    print()
    print("REQUIREMENTS:")
    print("  1. Voice Chat must be running (with VAD/PTT active)")
    print("  2. Run this from WINDOWS CMD (not WSL)")
    print("  3. Use RIGHT SHIFT to record (same as chatting)")
    print("  4. Speak naturally - don't exaggerate emotions")
    print()
    print(f"Recordings directory: {RECORDINGS_DIR}")
    print(f"Dry run mode: {'YES' if dry_run else 'NO (will auto-apply)'}")
    print()
    print("-" * 65)

    if not RECORDINGS_DIR.exists():
        print(f"ERROR: Recordings directory not found: {RECORDINGS_DIR}")
        print("Make sure Voice Chat is running first.")
        return

    # Pre-load the model
    print("[1/2] Loading SER model...")
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
        return

    print()
    print("-" * 65)
    print("[2/2] Capturing emotional samples...")
    print("-" * 65)
    print()

    # Emotions to capture with sample phrases
    emotions = [
        ("NEUTRAL", "Read this normally: 'The quick brown fox jumps over the lazy dog.'"),
        ("HAPPY", "Say something that makes you smile: 'I just won the lottery!' or laugh a bit."),
        ("FRUSTRATED", "Say with frustration: 'Why does this never work when I need it to?!'"),
        ("TIRED/SAD", "Say with low energy: 'I'm exhausted and just want to sleep.'"),
    ]

    results = {}

    for emotion_name, prompt in emotions:
        print(f">>> {emotion_name}")
        print(f"    {prompt}")
        print()

        filepath = wait_for_new_recording(timeout=60)

        if not filepath:
            print(f"    Skipped (no recording)\n")
            continue

        vad = analyze_recording(filepath)

        if vad:
            print(f"    Result: V={vad['valence']:.3f}  A={vad['arousal']:.3f}  D={vad['dominance']:.3f}")
            print(f"    Detected: {vad['label']}")
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
        print("No samples captured. Make sure:")
        print("  1. Voice Chat is running with VAD enabled")
        print("  2. You're holding RIGHT SHIFT while speaking")
        print("  3. The recording is at least 0.5 seconds long")
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
            print("CALIBRATION COMPLETE!")
            print("Restart Voice Chat to use new thresholds.")
        else:
            print()
            print("AUTO-APPLY FAILED. Manual steps:")
            print("  1. Open audio/emotion_detector.py")
            print("  2. Find the _vad_to_label method")
            print("  3. Replace it with the code above")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SER Calibration Tool")
    parser.add_argument("--dry-run", action="store_true", help="Preview thresholds without applying")
    args = parser.parse_args()

    try:
        calibrate(dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n\nCalibration cancelled.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
