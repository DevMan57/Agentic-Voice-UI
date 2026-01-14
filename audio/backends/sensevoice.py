"""
SenseVoice STT Backend

Unified Speech-to-Text + VAD + Emotion using FunAudioLLM/SenseVoiceSmall.
Provides 15x speedup over Whisper with built-in emotion detection.

Features:
- 50+ languages with auto-detection
- Built-in VAD (no separate Silero needed)
- Emotion detection (Happy/Sad/Angry/Neutral) embedded in output
- Audio event detection (music, laughter, applause)

Usage:
    from audio.backends.sensevoice import SenseVoiceBackend

    stt = SenseVoiceBackend(device="cuda:0")
    result = stt.transcribe("audio.wav")
    print(result.text)

    # Get emotion with transcription
    result, emotion = stt.transcribe_with_emotion("audio.wav")
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from audio.interface import STTBackend, TranscriptionResult, EmotionResult


# Emoji to emotion mapping for SenseVoice output parsing
EMOTION_EMOJI_MAP = {
    "😊": ("happy", 0.8, 0.7, 0.6),    # (label, valence, arousal, dominance)
    "😡": ("angry", 0.2, 0.9, 0.8),
    "😔": ("sad", 0.2, 0.3, 0.3),
    "😐": ("neutral", 0.5, 0.5, 0.5),
}

# Audio event emoji (informational, not emotion)
AUDIO_EVENT_MAP = {
    "🎼": "music",
    "😀": "laughter",
    "👏": "applause",
}


def parse_sensevoice_output(text: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse SenseVoice output to extract clean text, emotion, and audio events.

    Args:
        text: Raw SenseVoice output with emoji tags

    Returns:
        Tuple of (clean_text, emotion_label, audio_event)
    """
    clean_text = text
    detected_emotion = None
    detected_event = None

    # Check for emotion emoji
    for emoji, (label, _, _, _) in EMOTION_EMOJI_MAP.items():
        if emoji in text:
            detected_emotion = label
            clean_text = clean_text.replace(emoji, "").strip()
            break

    # Check for audio events
    for emoji, event in AUDIO_EVENT_MAP.items():
        if emoji in text:
            detected_event = event
            clean_text = clean_text.replace(emoji, "").strip()

    return clean_text.strip(), detected_emotion, detected_event


def emotion_to_result(emotion_label: Optional[str]) -> Optional[EmotionResult]:
    """
    Convert SenseVoice emotion label to EmotionResult.

    Args:
        emotion_label: Emotion label from parsing

    Returns:
        EmotionResult or None if no emotion detected
    """
    if emotion_label is None:
        return None

    # Find matching emotion data
    for emoji, (label, valence, arousal, dominance) in EMOTION_EMOJI_MAP.items():
        if label == emotion_label:
            return EmotionResult(
                valence=valence,
                arousal=arousal,
                dominance=dominance,
                label=label,
                confidence=0.8  # SenseVoice doesn't provide confidence
            )

    return None


class SenseVoiceBackend(STTBackend):
    """
    SenseVoice-Small unified STT backend.

    Provides STT + VAD + Emotion in a single model pass.
    15x faster than Whisper-Large with comparable accuracy.

    Model: FunAudioLLM/SenseVoiceSmall (~400MB)
    Languages: 50+ (zh, en, yue, ja, ko primary)
    """

    def __init__(
        self,
        device: str = "cuda:0",
        use_vad: bool = True,
        max_segment_time: int = 30000,
        batch_size_s: int = 60
    ):
        """
        Initialize SenseVoice backend.

        Args:
            device: Device to run on ('cuda:0', 'cpu')
            use_vad: Enable built-in VAD for segmentation
            max_segment_time: Maximum segment length in ms (for VAD)
            batch_size_s: Batch size in seconds for processing
        """
        self._device = device
        self._use_vad = use_vad
        self._max_segment_time = max_segment_time
        self._batch_size_s = batch_size_s
        self._model = None
        self._initialized = False
        self._postprocess = None

    def _ensure_initialized(self):
        """Lazy load the SenseVoice model"""
        if self._initialized:
            return

        try:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess

            print(f"[SenseVoice] Loading model on {self._device}...")

            vad_kwargs = {"max_single_segment_time": self._max_segment_time} if self._use_vad else None
            vad_model = "fsmn-vad" if self._use_vad else None

            self._model = AutoModel(
                model="FunAudioLLM/SenseVoiceSmall",
                vad_model=vad_model,
                vad_kwargs=vad_kwargs,
                device=self._device,
                hub="hf",
            )
            self._postprocess = rich_transcription_postprocess
            self._initialized = True
            print("[SenseVoice] Model loaded successfully")

        except ImportError as e:
            print(f"[SenseVoice] FunASR not available: {e}")
            print("[SenseVoice] Install with: pip install funasr")
            self._model = None

        except Exception as e:
            print(f"[SenseVoice] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self._model = None

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
            language: Language code ('en', 'zh', 'ja', 'ko', 'yue') or 'auto'
            **kwargs: Additional parameters (use_itn, merge_vad)

        Returns:
            TranscriptionResult with clean text (emotion stripped)
        """
        self._ensure_initialized()

        if self._model is None:
            return TranscriptionResult(
                text="[SenseVoice not available - install funasr]",
                confidence=0.0
            )

        try:
            # Default to auto language detection
            lang = language if language else "auto"

            result = self._model.generate(
                input=audio_path,
                language=lang,
                use_itn=kwargs.get("use_itn", True),
                batch_size_s=self._batch_size_s,
                merge_vad=kwargs.get("merge_vad", True),
            )

            # Get raw text and postprocess
            raw_text = result[0]["text"] if result else ""
            if self._postprocess:
                raw_text = self._postprocess(raw_text)

            # Parse out emotion and events, return clean text
            clean_text, emotion_label, audio_event = parse_sensevoice_output(raw_text)

            return TranscriptionResult(
                text=clean_text,
                language=lang if lang != "auto" else None,
                confidence=None,  # SenseVoice doesn't provide confidence scores
                segments=None
            )

        except Exception as e:
            print(f"[SenseVoice] Transcription failed: {e}")
            return TranscriptionResult(
                text=f"[SenseVoice error: {e}]",
                confidence=0.0
            )

    def transcribe_with_emotion(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs
    ) -> Tuple[TranscriptionResult, Optional[EmotionResult]]:
        """
        Transcribe audio and extract emotion.

        This is the unified method that returns both transcription and emotion
        in a single model pass (no separate emotion analysis needed).

        Args:
            audio_path: Path to audio file
            language: Language code or 'auto'
            **kwargs: Additional parameters

        Returns:
            Tuple of (TranscriptionResult, EmotionResult or None)
        """
        self._ensure_initialized()

        if self._model is None:
            return (
                TranscriptionResult(
                    text="[SenseVoice not available - install funasr]",
                    confidence=0.0
                ),
                None
            )

        try:
            lang = language if language else "auto"

            result = self._model.generate(
                input=audio_path,
                language=lang,
                use_itn=kwargs.get("use_itn", True),
                batch_size_s=self._batch_size_s,
                merge_vad=kwargs.get("merge_vad", True),
            )

            raw_text = result[0]["text"] if result else ""
            if self._postprocess:
                raw_text = self._postprocess(raw_text)

            # Parse text and emotion
            clean_text, emotion_label, audio_event = parse_sensevoice_output(raw_text)
            emotion = emotion_to_result(emotion_label)

            transcription = TranscriptionResult(
                text=clean_text,
                language=lang if lang != "auto" else None,
                confidence=None,
                segments=None
            )

            return transcription, emotion

        except Exception as e:
            print(f"[SenseVoice] Transcription failed: {e}")
            return (
                TranscriptionResult(text=f"[SenseVoice error: {e}]", confidence=0.0),
                None
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

        # Ensure correct format for WAV writing
        if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
            audio_int16 = (audio_array * 32767).astype(np.int16)
        elif audio_array.dtype == np.int32:
            audio_int16 = (audio_array / 65536).astype(np.int16)
        else:
            audio_int16 = audio_array.astype(np.int16)

        # Write to temp file
        fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="sensevoice_")
        os.close(fd)

        try:
            wav.write(temp_path, sample_rate, audio_int16)
            result = self.transcribe(temp_path, language=language, **kwargs)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        return result

    def supports_emotion(self) -> bool:
        """SenseVoice includes built-in emotion detection"""
        return True

    def supports_vad(self) -> bool:
        """SenseVoice includes built-in VAD"""
        return self._use_vad

    @property
    def name(self) -> str:
        return "sensevoice"

    @property
    def device(self) -> str:
        return self._device

    def is_available(self) -> bool:
        """Check if SenseVoice is initialized"""
        self._ensure_initialized()
        return self._model is not None


__all__ = ['SenseVoiceBackend', 'parse_sensevoice_output', 'emotion_to_result']
