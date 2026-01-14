"""
Audio Interface Abstract Base Classes

Defines standard interfaces for swappable audio backends:
- TTSBackend: Text-to-Speech synthesis
- STTBackend: Speech-to-Text transcription
- VADBackend: Voice Activity Detection
- EmotionAnalyzer: Speech emotion recognition

These interfaces enable backend swapping via configuration without code changes.
Preparing for FunASR, Soprano TTS, and other future backends.

Usage:
    from audio.interface import TTSBackend, STTBackend, VADBackend

    class MyTTSBackend(TTSBackend):
        def synthesize(self, text, voice, emotion=None, **kwargs):
            # Implementation
            pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Any
from threading import Thread
import numpy as np


# Re-export EmotionResult for convenience (canonical location: emotion_detector.py)
from audio.emotion_detector import EmotionResult


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription"""
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None  # Word/segment timings if available

    def __str__(self) -> str:
        return self.text


class TTSBackend(ABC):
    """
    Abstract base class for Text-to-Speech backends.

    Implementations: IndexTTS2Backend, KokoroBackend, (future) SopranoBackend
    """

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str,
        emotion: Optional[EmotionResult] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Synthesize speech from text.

        Args:
            text: Text to convert to speech
            voice: Voice identifier (backend-specific)
            emotion: Optional emotion to express in speech
            output_path: Optional path for output file. If None, use temp file.
            **kwargs: Backend-specific parameters (speed, pitch, etc.)

        Returns:
            Path to the generated audio file
        """
        pass

    @abstractmethod
    def get_available_voices(self) -> List[str]:
        """Return list of available voice identifiers"""
        pass

    @abstractmethod
    def supports_emotion(self) -> bool:
        """Whether this backend can express emotion in speech"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'indextts2', 'kokoro')"""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Device backend runs on ('cuda', 'cpu', 'mps')"""
        pass

    def is_available(self) -> bool:
        """Check if backend is properly initialized and ready"""
        return True


class STTBackend(ABC):
    """
    Abstract base class for Speech-to-Text backends.

    Implementations: FasterWhisperBackend, (future) FunASRBackend
    """

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'ja') or None for auto-detect
            **kwargs: Backend-specific parameters

        Returns:
            TranscriptionResult with text and metadata
        """
        pass

    @abstractmethod
    def transcribe_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio from numpy array.

        Args:
            audio_array: Audio samples as numpy array
            sample_rate: Sample rate of audio
            language: Language code or None for auto-detect
            **kwargs: Backend-specific parameters

        Returns:
            TranscriptionResult with text and metadata
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'faster_whisper', 'funasr')"""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Device backend runs on ('cuda', 'cpu')"""
        pass

    def is_available(self) -> bool:
        """Check if backend is properly initialized and ready"""
        return True


class VADBackend(ABC):
    """
    Abstract base class for Voice Activity Detection backends.

    Implementations: SileroVAD, WebRTCVAD, EnergyVAD
    """

    @abstractmethod
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect if audio chunk contains speech.

        Args:
            audio_chunk: Audio samples (typically 30ms at 16kHz)

        Returns:
            True if speech detected, False otherwise
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (for stateful models like Silero)"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'silero', 'webrtc', 'energy')"""
        pass

    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """
        Get speech probability (0-1) for audio chunk.

        Default implementation returns 1.0 or 0.0 based on is_speech().
        Override for backends that provide continuous probabilities.
        """
        return 1.0 if self.is_speech(audio_chunk) else 0.0


class EmotionAnalyzer(ABC):
    """
    Abstract base class for Speech Emotion Recognition backends.

    Implementations: Wav2Vec2EmotionAnalyzer (existing EmotionDetector)
    """

    @abstractmethod
    def analyze(
        self,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None
    ) -> Optional[EmotionResult]:
        """
        Analyze emotion from audio.

        Args:
            audio_path: Path to audio file (preferred)
            audio_array: Audio samples as numpy array (alternative)
            sample_rate: Sample rate if using audio_array

        Returns:
            EmotionResult with VAD scores and label, or None on failure
        """
        pass

    @abstractmethod
    def analyze_async(
        self,
        audio_path: str,
        callback: Callable[[Optional[EmotionResult]], None]
    ) -> Thread:
        """
        Analyze emotion asynchronously.

        Args:
            audio_path: Path to audio file
            callback: Function called with result when complete

        Returns:
            Running thread (already started)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'wav2vec2')"""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Device backend runs on ('cuda', 'cpu')"""
        pass

    def is_available(self) -> bool:
        """Check if backend is properly initialized and ready"""
        return True


# Type aliases for convenience
TTSProvider = TTSBackend
STTProvider = STTBackend
VADProvider = VADBackend
SERProvider = EmotionAnalyzer  # Speech Emotion Recognition


__all__ = [
    # Abstract base classes
    'TTSBackend',
    'STTBackend',
    'VADBackend',
    'EmotionAnalyzer',
    # Data classes
    'EmotionResult',
    'TranscriptionResult',
    # Type aliases
    'TTSProvider',
    'STTProvider',
    'VADProvider',
    'SERProvider',
]
