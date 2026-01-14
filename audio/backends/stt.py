"""
STT Backend Implementations

Concrete implementations of STTBackend for faster-whisper.

Usage:
    from audio.backends.stt import FasterWhisperBackend

    stt = FasterWhisperBackend(model_size="base", device="cpu")
    result = stt.transcribe("audio.wav")
    print(result.text)
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from audio.interface import STTBackend, TranscriptionResult


class FasterWhisperBackend(STTBackend):
    """
    faster-whisper STT backend.

    Uses CTranslate2-optimized Whisper model for efficient transcription.
    Falls back to OpenAI Whisper if faster-whisper is unavailable.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        """
        Initialize faster-whisper backend.

        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v2')
            device: Device to run on ('cuda' or 'cpu')
            compute_type: Compute type ('float16', 'int8', 'int8_float16')
        """
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._backend = None  # 'faster_whisper' or 'openai_whisper'
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy load the Whisper model"""
        if self._initialized:
            return

        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel

            print(f"[STT] Loading faster-whisper ({self._model_size}) on {self._device}...")
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type
            )
            self._backend = 'faster_whisper'
            self._initialized = True
            print(f"[STT] faster-whisper loaded successfully")
            return

        except ImportError:
            print("[STT] faster-whisper not available, trying openai-whisper...")

        # Fallback to OpenAI Whisper
        try:
            import whisper

            print(f"[STT] Loading OpenAI Whisper ({self._model_size})...")
            self._model = whisper.load_model(self._model_size)
            self._backend = 'openai_whisper'
            self._initialized = True
            print(f"[STT] OpenAI Whisper loaded successfully")

        except ImportError:
            print("[STT] Neither faster-whisper nor openai-whisper available")
            self._model = None
            self._backend = None

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code ('en', 'ja', etc.) or None for auto-detect
            **kwargs: Additional backend-specific parameters

        Returns:
            TranscriptionResult with transcribed text
        """
        self._ensure_initialized()

        if self._model is None:
            return TranscriptionResult(
                text="[Whisper not available - install faster-whisper]",
                confidence=0.0
            )

        try:
            if self._backend == 'faster_whisper':
                return self._transcribe_faster_whisper(audio_path, language, **kwargs)
            else:
                return self._transcribe_openai_whisper(audio_path, language, **kwargs)

        except Exception as e:
            print(f"[STT] Transcription failed: {e}")
            return TranscriptionResult(
                text=f"[Transcription error: {e}]",
                confidence=0.0
            )

    def _transcribe_faster_whisper(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe using faster-whisper"""
        segments, info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=kwargs.get('beam_size', 5),
            vad_filter=kwargs.get('vad_filter', True)
        )

        # Collect all segments
        text_parts = []
        segment_list = []
        for segment in segments:
            text_parts.append(segment.text.strip())
            segment_list.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })

        full_text = ' '.join(text_parts)

        return TranscriptionResult(
            text=full_text,
            language=info.language if info else language,
            confidence=info.language_probability if info else None,
            segments=segment_list if segment_list else None
        )

    def _transcribe_openai_whisper(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper"""
        result = self._model.transcribe(
            audio_path,
            language=language,
            fp16=kwargs.get('fp16', False)
        )

        # Build segment list if available
        segments = None
        if 'segments' in result:
            segments = [
                {
                    'start': s['start'],
                    'end': s['end'],
                    'text': s['text'].strip()
                }
                for s in result['segments']
            ]

        return TranscriptionResult(
            text=result['text'].strip(),
            language=result.get('language'),
            confidence=None,  # OpenAI Whisper doesn't provide this easily
            segments=segments
        )

    def transcribe_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio from numpy array.

        Writes array to temp file and transcribes.
        """
        import scipy.io.wavfile as wav

        # Ensure correct format
        if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
            audio_int16 = (audio_array * 32767).astype(np.int16)
        elif audio_array.dtype == np.int32:
            audio_int16 = (audio_array / 65536).astype(np.int16)
        else:
            audio_int16 = audio_array.astype(np.int16)

        # Write to temp file
        fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="stt_")
        os.close(fd)

        try:
            wav.write(temp_path, sample_rate, audio_int16)
            result = self.transcribe(temp_path, language=language, **kwargs)
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        return result

    @property
    def name(self) -> str:
        if self._backend:
            return self._backend
        return "faster_whisper"

    @property
    def device(self) -> str:
        return self._device

    def is_available(self) -> bool:
        """Check if Whisper is initialized"""
        self._ensure_initialized()
        return self._model is not None


__all__ = ['FasterWhisperBackend']
