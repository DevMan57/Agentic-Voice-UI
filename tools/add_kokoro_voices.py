import os
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

# Standard Kokoro v0.19 voices
VOICES = [
    # American Female
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", 
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    
    # American Male
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
    "am_michael", "am_onyx", "am_puck",
    
    # British Female
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    
    # British Male
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis"
]

def create_dummy_voices():
    # Target directory: ../voice_reference (relative to this script)
    base_dir = Path(__file__).parent.parent
    voice_dir = base_dir / "voice_reference"
    voice_dir.mkdir(exist_ok=True)
    
    print(f"Creating {len(VOICES)} dummy voice files in {voice_dir}...")
    
    # Create 1 second of silence
    sample_rate = 24000
    duration = 1.0
    # Create near-silence (very low amplitude noise) instead of pure zero
    # to avoid potential "empty audio" checks failing in some tools
    audio = np.random.normal(0, 0.001, int(sample_rate * duration)).astype(np.float32)
    
    created_count = 0
    for voice in VOICES:
        filename = f"{voice}.wav"
        filepath = voice_dir / filename
        
        if not filepath.exists():
            wav.write(str(filepath), sample_rate, audio)
            print(f"  + Created: {filename}")
            created_count += 1
        else:
            print(f"  . Exists:  {filename}")
            
    print(f"\nDone! Created {created_count} new voice files.")
    print("Restart the application to see them in the dropdown.")

if __name__ == "__main__":
    create_dummy_voices()
