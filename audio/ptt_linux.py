#!/usr/bin/env python3
"""
Push-to-Talk Recorder for IndexTTS2 Voice Chat - Linux Version

Features:
- Hold Right Shift OR Left Shift to record
- Writes status file for UI indicator
- Low-latency always-open microphone stream
- Requires root/sudo for keyboard access on Linux

Usage:
    sudo python ptt_linux.py
    
Or add your user to the input group:
    sudo usermod -a -G input $USER
    # Log out and back in, then:
    python ptt_linux.py
"""

import time
import wave
import pathlib
import sys
import os

# Check for root on Linux (required for keyboard access)
if os.geteuid() != 0:
    print("\n⚠️  Warning: Running without root privileges.")
    print("    Keyboard access may not work.")
    print("    Run with: sudo python ptt_linux.py")
    print("    Or add yourself to input group: sudo usermod -a -G input $USER\n")

try:
    import keyboard
except ImportError:
    print("ERROR: keyboard module not installed")
    print("Install with: pip install keyboard")
    sys.exit(1)

try:
    import pyaudio
except ImportError:
    print("ERROR: pyaudio module not installed")
    print("Install with: pip install pyaudio")
    print("On Ubuntu/Debian, first run: sudo apt install portaudio19-dev")
    sys.exit(1)

# Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000

# Support both shift keys
PTT_KEYS = ['right shift', 'left shift', 'shift']

# Paths - go up one level from audio/ to project root
CURRENT_DIR = pathlib.Path(__file__).parent.absolute()
PROJECT_DIR = CURRENT_DIR.parent  # Go up from audio/ to project root
RECORDINGS_DIR = PROJECT_DIR / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

STATUS_FILE = RECORDINGS_DIR / "ptt_status.txt"

def write_status(status: str, duration: float = 0.0, extra: str = ""):
    """Write current PTT status for UI to read"""
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
        self.setup_stream()
        write_status("ready", 0.0, "Hold Shift to record")

    def log(self, msg: str, end="\n", flush=False):
        """Print only if not in quiet mode"""
        if not self.quiet_mode:
            print(msg, end=end, flush=flush)

    def setup_stream(self):
        """Initialize and keep the stream open"""
        try:
            # Find default input device
            default_device = self.audio.get_default_input_device_info()
            self.log(f"[PTT] Using microphone: {default_device['name']}")
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
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
        write_status("processing", duration, "Sending to AI...")
        self.log(f" {duration:.1f}s")

        if not self.frames:
            self.log("(no audio)")
            write_status("ready", 0.0, "Hold Shift to record")
            return

        if duration < 0.3:
            self.log("(too short, ignored)")
            write_status("ready", 0.0, "Too short - hold longer")
            time.sleep(0.5)
            write_status("ready", 0.0, "Hold Shift to record")
            return

        # Save to file
        filename = RECORDINGS_DIR / f"rec_{int(time.time())}.wav"
        
        try:
            wf = wave.open(str(filename), 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()

            self.log(f"-> {filename.name}")

            # Trigger file for app to detect
            trigger_file = RECORDINGS_DIR / "latest.txt"
            trigger_file.write_text(str(filename))
            
            write_status("sent", duration, "Sent!")
            time.sleep(0.3)
            write_status("ready", 0.0, "Hold Shift to record")
            
        except Exception as e:
            self.log(f"Error: {e}")
            write_status("error", 0.0, str(e)[:30])

    def is_any_shift_pressed(self) -> bool:
        """Check if any shift key is pressed"""
        try:
            return any(keyboard.is_pressed(key) for key in PTT_KEYS)
        except Exception:
            return False

    def run(self):
        self.log("=" * 50)
        self.log("  PTT Recorder - Hold [Shift] to talk")
        self.log("  (Left Shift or Right Shift)")
        self.log("  Press [Esc] to exit")
        self.log("=" * 50)
        
        was_pressed = False
        
        try:
            while True:
                # Keep reading to maintain fresh buffer
                if self.stream:
                    try:
                        data = self.stream.read(CHUNK, exception_on_overflow=False)
                        if self.is_recording:
                            self.frames.append(data)
                            duration = time.time() - self.recording_start_time
                            write_status("recording", duration, f"Recording {duration:.1f}s")
                    except Exception as e:
                        self.log(f"[PTT] Stream error: {e}")
                
                # Check shift key state
                is_pressed = self.is_any_shift_pressed()
                
                # Detect transitions
                if is_pressed and not was_pressed:
                    self.start_recording()
                elif not is_pressed and was_pressed:
                    self.stop_recording()
                
                was_pressed = is_pressed
                
                try:
                    if keyboard.is_pressed('esc'):
                        break
                except Exception:
                    pass
                
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
    quiet_mode = "--quiet" in sys.argv or "-q" in sys.argv
    
    try:
        recorder = PTTRecorder(quiet_mode=quiet_mode)
        recorder.run()
    except Exception as e:
        error_msg = str(e).lower()
        if "permission" in error_msg or "access" in error_msg:
            print("\nERROR: Permission denied for keyboard access")
            print("Run with: sudo python ptt_linux.py")
            print("Or add to input group: sudo usermod -a -G input $USER")
        else:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
