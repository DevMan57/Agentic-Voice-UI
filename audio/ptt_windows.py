#!/usr/bin/env python3
"""
Push-to-Talk Recorder for IndexTTS2 Voice Chat
Runs on Windows, communicates with WSL app via files

Features:
- Hold Right Shift OR Left Shift to record
- Writes status file for UI indicator
- Low-latency always-open microphone stream
- Visual feedback via status file
"""

import time
import wave
import keyboard
import pyaudio
import pathlib
import sys

# Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000

# Use ONLY Right Shift (Left Shift is for typing capital letters!)
PTT_KEYS = ['right shift']

# Paths - go up one level from audio/ to project root
CURRENT_DIR = pathlib.Path(__file__).parent.absolute()
PROJECT_DIR = CURRENT_DIR.parent  # Go up from audio/ to project root
RECORDINGS_DIR = PROJECT_DIR / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

# Status file for UI communication
STATUS_FILE = RECORDINGS_DIR / "ptt_status.txt"
VAD_CONTROL_FILE = RECORDINGS_DIR / "vad_control.txt"  # App writes: enabled|tts_playing

def is_tts_playing() -> bool:
    """Check if TTS is currently playing (via control file from app)"""
    try:
        if VAD_CONTROL_FILE.exists():
            content = VAD_CONTROL_FILE.read_text().strip()
            parts = content.split("|")
            # Format: enabled|tts_playing
            # Return True if second part is "1" (TTS is playing)
            return len(parts) >= 2 and parts[1] == "1"
        return False
    except:
        return False

# Track TTS stop time for cooldown
_tts_stop_time = 0

def should_block_recording() -> bool:
    """Check if recording should be blocked (TTS playing or in cooldown).
    
    Returns True if we should NOT record (TTS playing or just stopped).
    Includes a 3-second cooldown after TTS stops to prevent self-hearing.
    """
    global _tts_stop_time
    
    if is_tts_playing():
        _tts_stop_time = time.time()
        return True
    
    # Check cooldown period
    if _tts_stop_time > 0:
        elapsed = time.time() - _tts_stop_time
        if elapsed < 3.0:
            return True
        else:
            _tts_stop_time = 0
    
    return False

def is_vad_enabled() -> bool:
    """Check if VAD hands-free mode is enabled (via control file from app)"""
    try:
        if VAD_CONTROL_FILE.exists():
            content = VAD_CONTROL_FILE.read_text().strip()
            parts = content.split("|")
            # Format: enabled|tts_playing - VAD is enabled if first part is "1"
            return len(parts) >= 1 and parts[0] == "1"
        return False
    except:
        return False

def write_status(status: str, duration: float = 0.0, extra: str = ""):
    """Write current PTT status for UI to read (skips if VAD mode is active)"""
    # Don't overwrite status when hands-free mode is enabled
    if is_vad_enabled() and status == "ready":
        return  # Let VAD control the status display
    try:
        content = f"{status}|{duration:.1f}"
        if extra:
            content += f"|{extra}"
        STATUS_FILE.write_text(content)
    except Exception as e:
        print(f"[PTT] Error writing status: {e}")

class PTTRecorder:
    def __init__(self, quiet_mode: bool = False):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.recording_start_time = 0
        self.quiet_mode = quiet_mode
        self.input_device_index = self.find_microphone_device()
        self.setup_stream()
        write_status("ready", 0.0, "Hold Right Shift to record")

    def find_microphone_device(self) -> int:
        """Find the correct microphone input device"""
        try:
            # List all devices
            device_count = self.audio.get_device_count()
            self.log(f"[PTT] Found {device_count} audio devices:")

            mic_devices = []
            for i in range(device_count):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    name = info['name']
                    mic_devices.append((i, name))
                    # Check if this looks like a real microphone
                    if any(keyword in name.lower() for keyword in ['microphone', 'mic', 'realtek', 'yeti', 'blue']):
                        self.log(f"[PTT]   [{i}] {name} <-- Selected (looks like a mic)")
                        return i
                    else:
                        self.log(f"[PTT]   [{i}] {name}")

            if mic_devices:
                # Use first mic device if no obvious match found
                selected = mic_devices[0][0]
                if not self.quiet_mode:
                    print(f"[PTT] Using device [{selected}]: {mic_devices[0][1]}")
                return selected
            else:
                self.log("[PTT] WARNING: No input devices found, using default")
                return None  # Use default
        except Exception as e:
            self.log(f"[PTT] Error listing devices: {e}")
            return None  # Use default

    def log(self, msg: str, end="\n", flush=False):
        """Print only if not in quiet mode"""
        if not self.quiet_mode:
            print(msg, end=end, flush=flush)

    def setup_stream(self):
        """Initialize and keep the stream open to avoid startup latency"""
        try:
            stream_kwargs = {
                'format': pyaudio.paInt16,
                'channels': CHANNELS,
                'rate': SAMPLE_RATE,
                'input': True,
                'frames_per_buffer': CHUNK
            }
            # Use the selected device if found
            if self.input_device_index is not None:
                stream_kwargs['input_device_index'] = self.input_device_index
                self.log(f"[PTT] Opening audio stream with device index: {self.input_device_index}")
            else:
                self.log("[PTT] Opening audio stream with DEFAULT device")

            self.stream = self.audio.open(**stream_kwargs)

            # TEST: Read one chunk to verify audio is working
            import numpy as np
            test_data = self.stream.read(CHUNK, exception_on_overflow=False)
            if test_data:
                test_audio = np.frombuffer(test_data, dtype=np.int16)
                test_max = np.abs(test_audio).max()
                test_rms = np.sqrt(np.mean(test_audio.astype(np.float32) ** 2))
                self.log(f"[PTT] Microphone TEST: max={test_max}, RMS={test_rms:.6f}")
                if test_max < 100:
                    self.log(f"[PTT] WARNING: Test audio is very quiet! Check:")
                    self.log(f"[PTT]   - Is the correct microphone selected?")
                    self.log(f"[PTT]   - Is the microphone unmuted in Windows Sound settings?")
                    self.log(f"[PTT]   - Is another app using the microphone?")
                else:
                    self.log("[PTT] Microphone test PASSED - audio detected!")
            else:
                self.log("[PTT] WARNING: No data read from microphone!")

            self.log("[PTT] Microphone ready")
        except Exception as e:
            self.log(f"[PTT] ERROR: Failed to open microphone: {e}")
            write_status("error", 0.0, "Microphone not found")

    def start_recording(self):
        if self.is_recording:
            return
        
        self.is_recording = True
        self.frames = []
        self.recording_start_time = time.time()
        write_status("recording", 0.0, "Recording...")
        self.log("[REC] ", end="", flush=True)

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        duration = time.time() - self.recording_start_time
        frame_count = len(self.frames)
        write_status("processing", duration, "Sending to AI...")
        self.log(f" {duration:.1f}s ({frame_count} frames)")

        if not self.frames:
            self.log("(no audio frames captured!)")
            write_status("ready", 0.0, "Hold Right Shift to record")
            return

        if duration < 0.5:
            self.log(f"(too short: {duration:.2f}s, need 0.5s+)")
            write_status("ready", 0.0, "Hold longer (0.5s+)")
            time.sleep(0.5)
            write_status("ready", 0.0, "Hold Right Shift to record")
            return

        # Check audio level before saving
        import numpy as np
        audio_bytes = b''.join(self.frames)
        if audio_bytes:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            max_level = np.abs(audio_array).max()
            rms_level = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            self.log(f"  Audio levels: max={max_level}, RMS={rms_level:.6f}")
            if max_level < 100:
                self.log(f"  WARNING: Audio seems very quiet (max={max_level})")
        else:
            self.log(f"  ERROR: No audio data captured!")

        # Save to file
        filename = RECORDINGS_DIR / f"rec_{int(time.time())}.wav"

        try:
            wf = wave.open(str(filename), 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)
            wf.close()

            self.log(f"-> {filename.name}")

            # Trigger file for WSL to detect
            trigger_file = RECORDINGS_DIR / "latest.txt"
            trigger_file.write_text(str(filename))
            
            write_status("sent", duration, "Sent!")
            # Brief delay then back to ready
            time.sleep(0.3)
            write_status("ready", 0.0, "Hold Right Shift to record")
            
        except Exception as e:
            self.log(f"Error: {e}")
            write_status("error", 0.0, str(e)[:30])

    def is_any_shift_pressed(self) -> bool:
        """Check if any shift key is pressed"""
        return any(keyboard.is_pressed(key) for key in PTT_KEYS)

    def run(self):
        self.log("=" * 50)
        self.log("  PTT Recorder - Hold [Right Shift] to talk")
        self.log("  (Right Shift only - Left Shift is for typing)")
        self.log("  Press [Esc] to exit")
        self.log("=" * 50)

        # Track state to detect press/release with debouncing
        was_pressed = False
        press_start_time = 0
        DEBOUNCE_MS = 100  # Require 100ms hold before activating

        # Real-time audio level monitoring
        level_update_counter = 0
        import numpy as np

        try:
            while True:
                # Keep reading to maintain fresh buffer
                if self.stream:
                    try:
                        data = self.stream.read(CHUNK, exception_on_overflow=False)

                        # Monitor audio levels even when not recording
                        level_update_counter += 1
                        if level_update_counter >= 43:  # Update ~every second (43 * 23ms ≈ 1s)
                            audio_array = np.frombuffer(data, dtype=np.int16)
                            level = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
                            bars = int(level * 100)
                            bar_str = "█" * min(bars, 40) + "░" * max(0, 40 - bars)
                            self.log(f"\r[PTT] Mic Level: {bar_str} {level:.4f}  (Hold Right Shift to record)", end="", flush=True)
                            level_update_counter = 0

                        if self.is_recording:
                            self.frames.append(data)
                            # Update duration while recording
                            duration = time.time() - self.recording_start_time
                            # Update every 100ms for smooth UI
                            write_status("recording", duration, f"Recording {duration:.1f}s")
                    except Exception as e:
                        self.log(f"[PTT] Stream error: {e}")

                # Check shift key state
                is_pressed = self.is_any_shift_pressed()

                # Prevent recording during TTS playback and cooldown period
                if should_block_recording():
                    # TTS is playing or just stopped - don't allow recording even if shift is pressed
                    if self.is_recording:
                        self.stop_recording()
                    was_pressed = False
                    time.sleep(0.05)  # Short sleep to reduce CPU usage during TTS
                    continue

                # Detect transitions with debouncing
                if is_pressed and not was_pressed:
                    # Key just pressed - start debounce timer
                    press_start_time = time.time()
                elif is_pressed and was_pressed:
                    # Key still held - check if debounce period passed
                    if not self.is_recording and (time.time() - press_start_time) >= (DEBOUNCE_MS / 1000.0):
                        self.start_recording()
                elif not is_pressed and was_pressed:
                    # Key just released
                    if self.is_recording:
                        self.stop_recording()
                    press_start_time = 0  # Reset debounce timer

                was_pressed = is_pressed
                
                if keyboard.is_pressed('esc'):
                    break
                
                # Small delay to prevent CPU spinning
                time.sleep(0.01)
                    
        except KeyboardInterrupt:
            pass
        finally:
            write_status("offline", 0.0, "PTT stopped")
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.audio.terminate()
            self.log("\n[PTT] Stopped")

def main():
    # Check for quiet mode flag
    quiet_mode = "--quiet" in sys.argv or "-q" in sys.argv
    
    try:
        recorder = PTTRecorder(quiet_mode=quiet_mode)
        recorder.run()
    except Exception as e:
        error_msg = str(e).lower()
        if "administrator" in error_msg or "access" in error_msg:
            print("\nERROR: Please run as Administrator!")
            print("Or try: Run the VoiceChat.bat file instead")
        else:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
