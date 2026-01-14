#!/usr/bin/env python3
"""
Voice Activity Detection (VAD) Recorder for IndexTTS2 Voice Chat

Provides hands-free voice recording using VAD to detect speech.
Supports multiple VAD backends:
- Silero VAD (best quality, requires torch)
- WebRTC VAD (lightweight, requires webrtcvad)
- Energy-based (fallback, no dependencies)

Usage:
    recorder = VADRecorder(backend='silero')
    recorder.start()
    # ... speech detected automatically ...
    recorder.stop()
"""

import os
import time
import threading
import queue
import wave
from collections import deque
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass
import numpy as np

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 30  # 30ms chunks for VAD
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
FORMAT_BYTES = 2  # 16-bit audio

# VAD settings
DEFAULT_SILENCE_THRESHOLD = 0.8  # Seconds of silence before stopping
DEFAULT_MIN_SPEECH_DURATION = 0.3  # Minimum speech duration to consider valid
DEFAULT_MAX_RECORDING_DURATION = 60.0  # Maximum recording duration


@dataclass
class VADConfig:
    """Configuration for VAD recorder"""
    backend: str = "energy"  # 'silero', 'webrtc', or 'energy'
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD
    min_speech_duration: float = DEFAULT_MIN_SPEECH_DURATION
    max_recording_duration: float = DEFAULT_MAX_RECORDING_DURATION
    
    # Energy-based VAD settings
    energy_threshold: float = 0.022  # RMS energy threshold for speech (raised from 0.015)

    # Consecutive frame detection (prevents cough/bump triggers)
    consecutive_frames_required: int = 5  # Require 5 frames (150ms) of speech before recording
    
    # WebRTC VAD settings
    webrtc_aggressiveness: int = 2  # 0-3, higher = more aggressive
    
    # Silero VAD settings
    silero_threshold: float = 0.6  # Probability threshold for speech (raised from 0.5)


class SileroVAD:
    """Silero VAD backend - highest accuracy"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self.utils = None
        self._load_model()
    
    def _load_model(self):
        try:
            import torch
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            # Force CPU to reserve GPU memory for IndexTTS2
            # Silero VAD is tiny (~30MB) and runs fast on CPU
            self.model = self.model.to('cpu')
            self.model.eval()
            print("[VAD] Silero VAD loaded on CPU")
        except Exception as e:
            print(f"[VAD] Failed to load Silero VAD: {e}")
            raise
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk contains speech"""
        import torch
        
        # Ensure float32 in range [-1, 1]
        if audio_chunk.dtype == np.int16:
            audio = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio = audio_chunk.astype(np.float32)
        
        # Model expects specific sample count
        tensor = torch.from_numpy(audio)
        if len(tensor) != 512:
            # Resample or pad to 512 samples
            if len(tensor) < 512:
                tensor = torch.nn.functional.pad(tensor, (0, 512 - len(tensor)))
            else:
                tensor = tensor[:512]
        
        with torch.no_grad():
            speech_prob = self.model(tensor, SAMPLE_RATE).item()
        
        return speech_prob > self.threshold
    
    def reset(self):
        """Reset internal state"""
        if self.model:
            self.model.reset_states()


class WebRTCVAD:
    """WebRTC VAD backend - lightweight and fast"""
    
    def __init__(self, aggressiveness: int = 2):
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
            print(f"[VAD] WebRTC VAD loaded (aggressiveness={aggressiveness})")
        except ImportError:
            raise ImportError("webrtcvad not installed. Run: pip install webrtcvad")
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk contains speech"""
        # WebRTC VAD needs int16 bytes
        if audio_chunk.dtype != np.int16:
            audio = (audio_chunk * 32768).astype(np.int16)
        else:
            audio = audio_chunk
        
        # WebRTC VAD needs exactly 10, 20, or 30ms of audio
        samples_per_frame = int(SAMPLE_RATE * 0.03)  # 30ms
        if len(audio) < samples_per_frame:
            audio = np.pad(audio, (0, samples_per_frame - len(audio)))
        elif len(audio) > samples_per_frame:
            audio = audio[:samples_per_frame]
        
        return self.vad.is_speech(audio.tobytes(), SAMPLE_RATE)
    
    def reset(self):
        pass


class EnergyVAD:
    """Energy-based VAD - no dependencies, fallback option"""
    
    def __init__(self, threshold: float = 0.015):
        self.threshold = threshold
        print(f"[VAD] Energy-based VAD initialized (threshold={threshold})")
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk contains speech based on energy"""
        if audio_chunk.dtype == np.int16:
            audio = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio = audio_chunk
        
        # Compute RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        return rms > self.threshold
    
    def reset(self):
        pass


def create_vad(config: VADConfig):
    """Factory function to create appropriate VAD backend"""
    if config.backend == "silero":
        try:
            return SileroVAD(threshold=config.silero_threshold)
        except Exception as e:
            print(f"[VAD] Silero failed, falling back to energy: {e}")
            return EnergyVAD(threshold=config.energy_threshold)
    
    elif config.backend == "webrtc":
        try:
            return WebRTCVAD(aggressiveness=config.webrtc_aggressiveness)
        except ImportError as e:
            print(f"[VAD] WebRTC failed, falling back to energy: {e}")
            return EnergyVAD(threshold=config.energy_threshold)
    
    else:
        return EnergyVAD(threshold=config.energy_threshold)


class VADRecorder:
    """
    Voice Activity Detection based recorder.
    Automatically starts recording when speech is detected
    and stops after silence threshold.
    """
    
    def __init__(
        self,
        config: VADConfig = None,
        output_dir: Path = None,
        on_speech_start: Callable[[], None] = None,
        on_speech_end: Callable[[str, float], None] = None,
        on_audio_level: Callable[[float], None] = None
    ):
        self.config = config or VADConfig()
        self.output_dir = Path(output_dir) if output_dir else Path("./recordings")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_audio_level = on_audio_level
        
        # State
        self.vad = None
        self.pyaudio = None
        self.stream = None
        self.is_running = False
        self.is_recording = False
        self.recording_start_time = 0.0
        
        # Audio buffers
        self.audio_buffer = []
        self.PRE_SPEECH_CHUNKS = 10  # ~300ms of audio before speech
        self.pre_speech_buffer = deque(maxlen=self.PRE_SPEECH_CHUNKS)  # Auto-evicts oldest
        
        # Threading
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        
        # Pause state (for TTS playback)
        self.is_paused = False
        self.paused_until = 0.0
        
        # Initialize VAD
        self._init_vad()
    
    def pause(self, duration: float = None):
        """Pause VAD (e.g. during TTS playback)"""
        self.is_paused = True
        if duration:
            self.paused_until = time.time() + duration
            print(f"[VAD] Paused for {duration:.1f}s")
        else:
            self.paused_until = 0.0
            print("[VAD] Paused indefinitely")

    def resume(self):
        """Resume VAD"""
        self.is_paused = False
        self.paused_until = 0.0
        print("[VAD] Resumed")

    def _init_vad(self):
        """Initialize the VAD backend"""
        self.vad = create_vad(self.config)
    
    def _init_audio(self):
        """Initialize PyAudio"""
        try:
            import pyaudio
            self.pyaudio = pyaudio.PyAudio()
            print("[VAD] PyAudio initialized")
            return True
        except ImportError:
            print("[VAD] PyAudio not installed. Run: pip install pyaudio")
            return False
        except Exception as e:
            print(f"[VAD] Failed to initialize PyAudio: {e}")
            return False
    
    def start(self) -> bool:
        """Start VAD-based recording"""
        if self.is_running:
            print("[VAD] Already running")
            return True
        
        if not self._init_audio():
            return False
        
        try:
            import pyaudio
            
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            
            self.is_running = True
            self.is_recording = False
            self.audio_buffer = []
            self.pre_speech_buffer.clear()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
            self.processing_thread.start()
            
            print("[VAD] Recording started - listening for speech...")
            return True
            
        except Exception as e:
            print(f"[VAD] Failed to start recording: {e}")
            return False
    
    def stop(self) -> Optional[str]:
        """Stop VAD-based recording"""
        if not self.is_running:
            return None
        
        self.is_running = False
        
        # If currently recording, save the audio
        output_path = None
        if self.is_recording and self.audio_buffer:
            output_path = self._save_audio()
        
        # Cleanup
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        
        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except:
                pass
        
        self.stream = None
        self.pyaudio = None
        
        print("[VAD] Recording stopped")
        return output_path
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - runs in audio thread"""
        import pyaudio
        if self.is_running:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def _process_loop(self):
        """Main processing loop - runs in separate thread"""
        silence_frames = 0
        consecutive_speech_frames = 0  # Track consecutive speech frames
        silence_threshold_frames = int(self.config.silence_threshold * SAMPLE_RATE / CHUNK_SIZE)

        while self.is_running:
            try:
                # Check auto-resume
                if self.is_paused and self.paused_until > 0 and time.time() > self.paused_until:
                    self.is_paused = False
                    self.paused_until = 0.0
                    print("[VAD] Auto-resumed after timeout")

                # Get audio data with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # If paused, just consume audio to keep queue clear, but don't process
                if self.is_paused:
                    continue

                audio_chunk = np.frombuffer(audio_data, dtype=np.int16)

                # Compute audio level for visualization
                if self.on_audio_level:
                    level = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)) / 32768.0
                    self.on_audio_level(level)

                # Check for speech
                is_speech = self.vad.is_speech(audio_chunk)

                if is_speech:
                    silence_frames = 0
                    consecutive_speech_frames += 1

                    # Debug logging for false trigger investigation
                    if consecutive_speech_frames <= self.config.consecutive_frames_required:
                        print(f"[VAD DEBUG] Speech frame {consecutive_speech_frames}/{self.config.consecutive_frames_required}")

                    if not self.is_recording:
                        # Require consecutive speech frames before starting to record
                        # This prevents single-event triggers like coughs or mic bumps
                        if consecutive_speech_frames >= self.config.consecutive_frames_required:
                            # Start recording
                            self.is_recording = True
                            self.recording_start_time = time.time()

                            # Include pre-speech buffer
                            self.audio_buffer = list(self.pre_speech_buffer)
                            self.pre_speech_buffer = []

                            print(f"[VAD] Speech detected ({consecutive_speech_frames} frames) - recording started")
                            if self.on_speech_start:
                                self.on_speech_start()
                        else:
                            # Still building up consecutive frames, keep buffering
                            self.pre_speech_buffer.append(audio_data)  # deque auto-evicts oldest
                            continue

                    self.audio_buffer.append(audio_data)
                    
                    # Check max duration
                    duration = time.time() - self.recording_start_time
                    if duration >= self.config.max_recording_duration:
                        print(f"[VAD] Max duration reached ({duration:.1f}s)")
                        self._end_recording()
                
                else:
                    # Silence detected - reset consecutive speech counter
                    consecutive_speech_frames = 0

                    if self.is_recording:
                        # Still recording but silence detected
                        self.audio_buffer.append(audio_data)
                        silence_frames += 1

                        if silence_frames >= silence_threshold_frames:
                            # End of speech
                            duration = time.time() - self.recording_start_time

                            if duration >= self.config.min_speech_duration:
                                print(f"[VAD] Speech ended ({duration:.1f}s)")
                                self._end_recording()
                            else:
                                # Too short, discard
                                print(f"[VAD] Speech too short ({duration:.1f}s), discarding")
                                self.is_recording = False
                                self.audio_buffer = []
                                self.vad.reset()
                    else:
                        # Not recording - maintain pre-speech buffer
                        self.pre_speech_buffer.append(audio_data)  # deque auto-evicts oldest
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VAD] Processing error: {e}")
    
    def _end_recording(self):
        """End current recording and save audio"""
        if not self.is_recording:
            return
        
        duration = time.time() - self.recording_start_time
        output_path = self._save_audio()
        
        self.is_recording = False
        self.audio_buffer = []
        self.vad.reset()
        
        if self.on_speech_end and output_path:
            self.on_speech_end(output_path, duration)
    
    def _save_audio(self) -> Optional[str]:
        """Save recorded audio to file"""
        if not self.audio_buffer:
            return None
        
        try:
            # Generate filename
            timestamp = int(time.time() * 1000)
            filename = f"vad_{timestamp}.wav"
            output_path = self.output_dir / filename
            
            # Write WAV file
            with wave.open(str(output_path), 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(FORMAT_BYTES)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(self.audio_buffer))
            
            duration = len(self.audio_buffer) * CHUNK_SIZE / SAMPLE_RATE
            print(f"[VAD] Saved: {filename} ({duration:.1f}s)")
            
            # Update latest.txt for compatibility with PTT system
            latest_file = self.output_dir / "latest.txt"
            latest_file.write_text(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            print(f"[VAD] Failed to save audio: {e}")
            return None
    
    @property
    def recording_duration(self) -> float:
        """Get current recording duration"""
        if self.is_recording:
            return time.time() - self.recording_start_time
        return 0.0


class ContinuousVADRecorder(VADRecorder):
    """
    Continuous VAD recorder that keeps listening after each utterance.
    Useful for conversation mode where multiple utterances are expected.
    """
    
    def __init__(self, *args, pause_between_utterances: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pause_between_utterances = pause_between_utterances
        self.utterance_count = 0
        self.last_utterance_end = 0.0
    
    def _end_recording(self):
        """End current recording and prepare for next utterance"""
        if not self.is_recording:
            return
        
        duration = time.time() - self.recording_start_time
        output_path = self._save_audio()
        
        self.is_recording = False
        self.audio_buffer = []
        self.vad.reset()
        self.utterance_count += 1
        self.last_utterance_end = time.time()
        
        if self.on_speech_end and output_path:
            self.on_speech_end(output_path, duration)
        
        # Continue listening for next utterance
        print(f"[VAD] Ready for next utterance (#{self.utterance_count + 1})")


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_record(
    output_path: str = None,
    max_duration: float = 30.0,
    silence_threshold: float = 1.0
) -> Optional[str]:
    """
    Quick one-shot VAD recording.
    Blocks until speech is detected and ends.
    
    Returns path to recorded audio file.
    """
    result = [None]
    done = threading.Event()
    
    def on_end(path: str, duration: float):
        result[0] = path
        done.set()
    
    config = VADConfig(
        silence_threshold=silence_threshold,
        max_recording_duration=max_duration
    )
    
    recorder = VADRecorder(
        config=config,
        output_dir=Path(output_path).parent if output_path else Path("./recordings"),
        on_speech_end=on_end
    )
    
    if recorder.start():
        done.wait(timeout=max_duration + 5)
        recorder.stop()
    
    return result[0]


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("VAD Recorder Test")
    print("=" * 40)
    print("Speak to test voice activity detection...")
    print("Press Ctrl+C to stop")
    print()
    
    def on_start():
        print("🎤 Recording...")
    
    def on_end(path: str, duration: float):
        print(f"✅ Saved: {path} ({duration:.1f}s)")
    
    def on_level(level: float):
        bars = int(level * 50)
        print(f"\r{'█' * bars}{' ' * (50 - bars)} {level:.2f}", end="", flush=True)
    
    config = VADConfig(
        backend="energy",  # Try 'silero' for better accuracy
        silence_threshold=0.8,
        min_speech_duration=0.3
    )
    
    recorder = ContinuousVADRecorder(
        config=config,
        on_speech_start=on_start,
        on_speech_end=on_end,
        on_audio_level=on_level
    )
    
    try:
        if recorder.start():
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        recorder.stop()
        print("Done!")
