# Audio input modules for IndexTTS2 Voice Chat
#
# Modules:
#   - interface.py    - Abstract base classes for audio backends
#   - registry.py     - Backend discovery and instantiation
#   - backends/       - Concrete backend implementations
#   - ptt_windows.py  - Push-to-talk listener for Windows
#   - ptt_linux.py    - Push-to-talk listener for Linux
#   - vad_recorder.py - Voice Activity Detection recording
#   - vad_windows.py  - VAD standalone script for Windows
#   - vad_linux.py    - VAD standalone script for Linux

from pathlib import Path

AUDIO_DIR = Path(__file__).parent

# Export interfaces
from audio.interface import (
    TTSBackend,
    STTBackend,
    VADBackend,
    EmotionAnalyzer,
    EmotionResult,
    TranscriptionResult,
)

# Export registry
from audio.registry import AudioRegistry

__all__ = [
    'AUDIO_DIR',
    # Interfaces
    'TTSBackend',
    'STTBackend',
    'VADBackend',
    'EmotionAnalyzer',
    'EmotionResult',
    'TranscriptionResult',
    # Registry
    'AudioRegistry',
]
