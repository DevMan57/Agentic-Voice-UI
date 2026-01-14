#!/usr/bin/env python3
"""
Voice Activity Detection (VAD) Listener for Linux

This is a standalone VAD listener for Linux systems.
It listens for speech using VAD and writes recordings to the shared directory.

Works alongside ptt_linux.py - you can run both:
- PTT: Hold Shift to record manually (requires sudo or input group)
- VAD: Automatic speech detection (hands-free)

Usage:
    python vad_linux.py              # Default settings
    python vad_linux.py --threshold 0.02   # More sensitive
    python vad_linux.py --silence 1.0      # Longer pause before cutoff

Requirements:
    pip install pyaudio numpy

Note: On Linux, you may need to install portaudio:
    sudo apt-get install portaudio19-dev
"""

import os
import sys
import time
import wave
import threading
import argparse
from pathlib import Path
from datetime import datetime

# Check platform
if sys.platform == 'win32':
    print("WARNING: This script is for Linux. Use vad_windows.py on Windows.")

try:
    import pyaudio
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pyaudio numpy")
    print("On Debian/Ubuntu: sudo apt-get install portaudio19-dev")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent  # Go up from audio/ to project root
RECORDINGS_DIR = PROJECT_DIR / "recordings"
STATUS_FILE = RECORDINGS_DIR / "ptt_status.txt"
TRIGGER_FILE = RECORDINGS_DIR / "latest.txt"

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 30  # 30ms chunks for VAD
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
FORMAT = pyaudio.paInt16

# Default VAD settings
DEFAULT_ENERGY_THRESHOLD = 0.015  # RMS threshold for speech
DEFAULT_SILENCE_DURATION = 0.8    # Seconds of silence before stopping
DEFAULT_MIN_SPEECH_DURATION = 0.3 # Minimum speech to be valid
DEFAULT_MAX_DURATION = 30.0       # Maximum recording duration


# ============================================================================
# VAD Backend (Energy-based for portability)
# ============================================================================

class EnergyVAD:
    """Simple energy-based VAD - works everywhere"""
    
    def __init__(self, threshold: float = DEFAULT_ENERGY_THRESHOLD):
        self.threshold = threshold
        
    def is_speech(self, audio_bytes: bytes) -> bool:
        """Check if audio chunk contains speech based on energy"""
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio ** 2))
        return rms > self.threshold
    
    def reset(self):
        pass


# ============================================================================
# VAD Recorder
# ============================================================================

class LinuxVADRecorder:
    """
    Voice Activity Detection recorder for Linux.
    Automatically starts/stops recording based on speech detection.
    """
    
    def __init__(
        self,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
        silence_duration: float = DEFAULT_SILENCE_DURATION,
        min_speech_duration: float = DEFAULT_MIN_SPEECH_DURATION,
        max_duration: float = DEFAULT_MAX_DURATION
    ):
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_duration = max_duration
        
        # Initialize VAD backend
        self.vad = EnergyVAD(threshold=energy_threshold)
        print(f"[VAD] Using Energy-based VAD (threshold={energy_threshold})")
        
        # PyAudio setup
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        # State
        self.is_running = False
        self.is_recording = False
        self.audio_buffer = []
        self.pre_speech_buffer = []
        self.PRE_SPEECH_CHUNKS = 10  # ~300ms before speech
        self.recording_start_time = 0.0
        self.utterance_count = 0
        
        # Ensure recordings directory exists
        RECORDINGS_DIR.mkdir(exist_ok=True)
        
    def _update_status(self, status: str, duration: float = 0.0, extra: str = ""):
        """Update status file for UI feedback"""
        try:
            STATUS_FILE.write_text(f"{status}|{duration:.1f}|{extra}")
        except:
            pass
    
    def _save_recording(self) -> str:
        """Save recorded audio to file and update trigger"""
        if not self.audio_buffer:
            return None
            
        timestamp = int(time.time() * 1000) % 10000000000
        filename = f"vad_{timestamp}.wav"
        filepath = RECORDINGS_DIR / filename
        
        try:
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(self.audio_buffer))
            
            # Calculate duration
            duration = len(self.audio_buffer) * CHUNK_SIZE / SAMPLE_RATE
            print(f"[VAD] Saved: {filename} ({duration:.1f}s)")
            
            # Update trigger file for main app to pick up
            TRIGGER_FILE.write_text(str(filepath))
            
            self.utterance_count += 1
            return str(filepath)
            
        except Exception as e:
            print(f"[VAD] Error saving: {e}")
            return None
    
    def _find_input_device(self):
        """Find the best available input device"""
        info = self.pyaudio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount', 0)
        
        default_device = None
        pulse_device = None
        
        for i in range(num_devices):
            try:
                device_info = self.pyaudio.get_device_info_by_index(i)
                if device_info.get('maxInputChannels', 0) > 0:
                    name = device_info.get('name', '').lower()
                    
                    # Prefer PulseAudio or default device
                    if 'pulse' in name:
                        pulse_device = i
                    if 'default' in name:
                        default_device = i
                        
            except Exception:
                continue
        
        # Return best device or None for system default
        if pulse_device is not None:
            return pulse_device
        return default_device
    
    def start(self):
        """Start VAD recording loop"""
        if self.is_running:
            return
            
        print("\n" + "="*50)
        print("  VAD Listener Active - Speak to record")
        print("  Press Ctrl+C to stop")
        print("="*50 + "\n")
        
        try:
            # Find input device
            input_device = self._find_input_device()
            if input_device is not None:
                print(f"[VAD] Using input device: {input_device}")
            
            self.stream = self.pyaudio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=CHUNK_SIZE
            )
            
            self.is_running = True
            self._update_status("ready", extra="VAD Listening")
            
            silence_chunks = 0
            silence_threshold_chunks = int(self.silence_duration * SAMPLE_RATE / CHUNK_SIZE)
            
            while self.is_running:
                try:
                    audio_data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                except Exception as e:
                    print(f"[VAD] Read error: {e}")
                    continue
                
                # Check for speech
                is_speech = self.vad.is_speech(audio_data)
                
                # Calculate level for display
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                level = np.sqrt(np.mean(audio_np ** 2))
                
                if is_speech:
                    silence_chunks = 0
                    
                    if not self.is_recording:
                        # Start recording
                        self.is_recording = True
                        self.recording_start_time = time.time()
                        self.audio_buffer = list(self.pre_speech_buffer)
                        self.pre_speech_buffer = []
                        print("[VAD] 🎤 Speech detected - recording...")
                        self._update_status("recording", 0.0, "Speaking...")
                    
                    self.audio_buffer.append(audio_data)
                    
                    # Update duration display
                    duration = time.time() - self.recording_start_time
                    self._update_status("recording", duration)
                    
                    # Check max duration
                    if duration >= self.max_duration:
                        print(f"[VAD] Max duration reached ({duration:.1f}s)")
                        self._end_recording()
                        
                else:
                    if self.is_recording:
                        # Still recording but silence detected
                        self.audio_buffer.append(audio_data)
                        silence_chunks += 1
                        
                        duration = time.time() - self.recording_start_time
                        self._update_status("recording", duration, "Waiting...")
                        
                        if silence_chunks >= silence_threshold_chunks:
                            # End of speech
                            if duration >= self.min_speech_duration:
                                self._end_recording()
                            else:
                                print(f"[VAD] Too short ({duration:.2f}s), discarding")
                                self.is_recording = False
                                self.audio_buffer = []
                                self._update_status("ready", extra="VAD Listening")
                    else:
                        # Not recording - maintain pre-speech buffer
                        self.pre_speech_buffer.append(audio_data)
                        if len(self.pre_speech_buffer) > self.PRE_SPEECH_CHUNKS:
                            self.pre_speech_buffer.pop(0)
                        
                        # Show level indicator
                        bars = int(level * 40)
                        bar_str = "█" * bars + "░" * (40 - bars)
                        print(f"\r[VAD] {bar_str} {level:.3f}", end="", flush=True)
                        
        except KeyboardInterrupt:
            print("\n[VAD] Stopping...")
        finally:
            self.stop()
    
    def _end_recording(self):
        """End current recording and save"""
        if not self.is_recording:
            return
            
        duration = time.time() - self.recording_start_time
        print(f"\n[VAD] ✓ Speech ended ({duration:.1f}s)")
        
        self._update_status("processing")
        self._save_recording()
        
        self.is_recording = False
        self.audio_buffer = []
        self.vad.reset()
        
        self._update_status("ready", extra=f"VAD Ready (#{self.utterance_count})")
        print(f"[VAD] Ready for next utterance (#{self.utterance_count + 1})")
    
    def stop(self):
        """Stop VAD recording"""
        self.is_running = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except:
                pass
        
        self._update_status("ready", extra="VAD Stopped")
        print("[VAD] Stopped")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Voice Activity Detection Listener for Linux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vad_linux.py                    # Default settings
  python vad_linux.py --threshold 0.02   # More sensitive (noisier environments)
  python vad_linux.py --threshold 0.01   # Very sensitive
  python vad_linux.py --silence 1.5      # Wait longer before stopping
        """
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=DEFAULT_ENERGY_THRESHOLD,
        help=f"Energy threshold for speech detection (default: {DEFAULT_ENERGY_THRESHOLD})"
    )
    parser.add_argument(
        "--silence", "-s",
        type=float,
        default=DEFAULT_SILENCE_DURATION,
        help=f"Silence duration before stopping (default: {DEFAULT_SILENCE_DURATION}s)"
    )
    parser.add_argument(
        "--min-speech", "-m",
        type=float,
        default=DEFAULT_MIN_SPEECH_DURATION,
        help=f"Minimum speech duration to save (default: {DEFAULT_MIN_SPEECH_DURATION}s)"
    )
    parser.add_argument(
        "--max-duration", "-d",
        type=float,
        default=DEFAULT_MAX_DURATION,
        help=f"Maximum recording duration (default: {DEFAULT_MAX_DURATION}s)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("   IndexTTS2 Voice Chat - VAD Listener (Linux)")
    print("="*60)
    print(f"\n  Recordings: {RECORDINGS_DIR}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Silence:    {args.silence}s")
    print(f"  Min Speech: {args.min_speech}s")
    print(f"  Max Duration: {args.max_duration}s")
    
    recorder = LinuxVADRecorder(
        energy_threshold=args.threshold,
        silence_duration=args.silence,
        min_speech_duration=args.min_speech,
        max_duration=args.max_duration
    )
    
    try:
        recorder.start()
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        print("\nDone!")


if __name__ == "__main__":
    main()
