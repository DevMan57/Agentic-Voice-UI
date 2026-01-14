"""
Windows Audio Recorder for SER Calibration
Records 4 emotion samples on Windows where microphone is accessible.
Run this FIRST, then run calibrate_emotion_standalone.py --from-files in WSL.
"""

import time
import wave
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent
RECORDINGS_DIR = SCRIPT_DIR / "recordings"

def record_audio(filename: str, duration: float = 3.0, sample_rate: int = 16000) -> str:
    """Record audio from microphone. Returns path to wav file."""
    try:
        import pyaudio
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

    # Save to file
    RECORDINGS_DIR.mkdir(exist_ok=True)
    filepath = RECORDINGS_DIR / filename

    with wave.open(str(filepath), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"   Saved: {filepath.name}")
    return str(filepath)


def main():
    print()
    print("=" * 65)
    print("   WINDOWS AUDIO RECORDER FOR SER CALIBRATION")
    print("=" * 65)
    print()
    print("This records 4 emotion samples using your Windows microphone.")
    print("After recording, run the calibration analysis in WSL.")
    print()
    print("-" * 65)

    emotions = [
        ("calibration_neutral.wav", "NEUTRAL",
         "Speak normally, like reading a news article:\n    'The weather today is partly cloudy with temperatures around 70 degrees.'"),
        ("calibration_happy.wav", "HAPPY",
         "Say something with genuine happiness or excitement:\n    'I just got amazing news! This is the best day ever!'"),
        ("calibration_frustrated.wav", "FRUSTRATED",
         "Say with frustration or annoyance:\n    'Why does this never work? This is so frustrating!'"),
        ("calibration_tired.wav", "TIRED/SAD",
         "Say with low energy or sadness:\n    'I'm so tired. I just want to rest.'"),
    ]

    recorded = []
    for filename, emotion_name, prompt in emotions:
        print()
        print(f">>> {emotion_name}")
        print(f"    {prompt}")
        print()

        input("    Press ENTER when ready to record...")

        result = record_audio(filename, duration=3.0)
        if result:
            recorded.append(filename)
        else:
            print(f"    Recording failed!")

        print()

    print("=" * 65)
    print(f"   RECORDING COMPLETE - {len(recorded)}/4 samples captured")
    print("=" * 65)
    print()

    if recorded:
        print("Files saved to recordings/ folder.")
        print()
        print("Next step: Run calibration analysis:")
        print("  VoiceChat.bat -> Option [5] (or run with --from-files)")
        print()
    else:
        print("No recordings captured. Check your microphone.")

    input("Press ENTER to exit...")


if __name__ == "__main__":
    main()
