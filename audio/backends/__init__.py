"""
Audio Backend Implementations

This package contains concrete implementations of the audio interfaces:
- TTS backends: IndexTTS2Backend, KokoroBackend, SupertonicBackend
- STT backends: FasterWhisperBackend, SenseVoiceBackend, FunASRBackend
- VAD backends: SileroVADBackend, WebRTCVADBackend, EnergyVADBackend

Usage:
    from audio.backends import IndexTTS2Backend, KokoroBackend
    from audio.backends import FasterWhisperBackend
    from audio.backends import SileroVADBackend, WebRTCVADBackend, EnergyVADBackend

    # Optional backends (require additional packages)
    from audio.backends import SupertonicBackend  # requires: pip install supertonic
    from audio.backends import SenseVoiceBackend, FunASRBackend  # requires: pip install funasr
"""

from audio.backends.tts import IndexTTS2Backend, KokoroBackend
from audio.backends.stt import FasterWhisperBackend
from audio.backends.vad import (
    SileroVAD,
    WebRTCVAD,
    EnergyVAD,
    SileroVADBackend,
    WebRTCVADBackend,
    EnergyVADBackend,
)

# Optional TTS backends
try:
    from audio.backends.supertonic import SupertonicBackend
except ImportError:
    SupertonicBackend = None

# Optional STT backends
try:
    from audio.backends.sensevoice import SenseVoiceBackend
except ImportError:
    SenseVoiceBackend = None

try:
    from audio.backends.funasr_backend import FunASRBackend
except ImportError:
    FunASRBackend = None

__all__ = [
    # TTS
    'IndexTTS2Backend',
    'KokoroBackend',
    'SupertonicBackend',
    # STT
    'FasterWhisperBackend',
    'SenseVoiceBackend',
    'FunASRBackend',
    # VAD (original classes)
    'SileroVAD',
    'WebRTCVAD',
    'EnergyVAD',
    # VAD (interface-conformant wrappers)
    'SileroVADBackend',
    'WebRTCVADBackend',
    'EnergyVADBackend',
]
