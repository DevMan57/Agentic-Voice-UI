"""
Fun-ASR STT Backend

High-accuracy Speech-to-Text using Paraformer models.
Optimized for accuracy with VAD and punctuation support.

Features:
- Paraformer: Non-autoregressive end-to-end ASR
- High accuracy with low latency
- VAD support for long audio segmentation
- Model variants: zh (Chinese), en (English)

Note: Does NOT include emotion detection - use with EmotionDetector.

Usage:
    from audio.backends.funasr_backend import FunASRBackend

    stt = FunASRBackend(device="cuda:0", model_variant="zh")
    result = stt.transcribe("audio.wav")
    print(result.text)
"""

import os
import tempfile
from typing import Optional, List

import numpy as np

from audio.interface import STTBackend, TranscriptionResult


class FunASRBackend(STTBackend):
    """
    Fun-ASR high-accuracy STT backend.

    Uses Paraformer models for high-quality transcription.
    Supports VAD for long audio segmentation.

    Model variants:
    - zh: paraformer-zh (Chinese)
    - en: paraformer-en (English)
    """

    LANGUAGE_MAP = {
        "zh": ["zh", "cn", "chinese"],
        "en": ["en", "english"],
    }

    def __init__(
        self,
        device: str = "cuda:0",
        model_variant: str = "zh",
        use_vad: bool = True
    ):
        """
        Initialize Fun-ASR backend.

        Args:
            device: Device to run on ('cuda:0', 'cpu')
            model_variant: Model to use ('zh' for Chinese, 'en' for English)
            use_vad: Enable VAD segmentation
        """
        self._device = device
        self._model_variant = model_variant
        self._use_vad = use_vad
        self._model = None
        self._initialized = False

    def _get_model_name(self) -> str:
        """Get model name based on variant"""
        if self._model_variant == "en":
            return "paraformer-en"
        return "paraformer-zh"

    def _ensure_initialized(self):
        """Lazy load the Fun-ASR model"""
        if self._initialized:
            return

        try:
            from funasr import AutoModel

            model_name = self._get_model_name()
            print(f"[FunASR] Loading {model_name} on {self._device}...")

            vad_model = "fsmn-vad" if self._use_vad else None
            vad_kwargs = {"max_single_segment_time": 30000} if self._use_vad else None

            self._model = AutoModel(
                model=model_name,
                vad_model=vad_model,
                vad_kwargs=vad_kwargs,
                device=self._device,
            )
            self._initialized = True
            print(f"[FunASR] Model loaded successfully ({model_name})")

        except ImportError as e:
            print(f"[FunASR] FunASR not available: {e}")
            print("[FunASR] Install with: pip install funasr")
            self._model = None

        except Exception as e:
            print(f"[FunASR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self._model = None

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        hotwords: Optional[List[str]] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language hint ('zh', 'en', 'ja', etc.)
            hotwords: List of words to boost in recognition
            **kwargs: Additional parameters (itn, cache)

        Returns:
            TranscriptionResult with transcribed text
        """
        self._ensure_initialized()

        if self._model is None:
            return TranscriptionResult(
                text="[FunASR not available - install funasr]",
                confidence=0.0
            )

        try:
            # Map language code to Fun-ASR format
            lang_map = {
                "zh": "中文",
                "en": "English",
                "ja": "日本語",
                "ko": "한국어",
            }
            funasr_lang = lang_map.get(language, language) if language else None

            # Build generation kwargs
            gen_kwargs = {
                "input": [audio_path],
                "cache": kwargs.get("cache", {}),
                "batch_size": kwargs.get("batch_size", 1),
                "itn": kwargs.get("itn", True),
            }

            if funasr_lang:
                gen_kwargs["language"] = funasr_lang

            if hotwords:
                gen_kwargs["hotwords"] = hotwords

            result = self._model.generate(**gen_kwargs)

            # Extract text from result
            text = result[0]["text"] if result else ""

            return TranscriptionResult(
                text=text.strip(),
                language=language,
                confidence=None,  # Fun-ASR doesn't provide confidence
                segments=None
            )

        except Exception as e:
            print(f"[FunASR] Transcription failed: {e}")
            return TranscriptionResult(
                text=f"[FunASR error: {e}]",
                confidence=0.0
            )

    def transcribe_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
        hotwords: Optional[List[str]] = None,
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
        fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="funasr_")
        os.close(fd)

        try:
            wav.write(temp_path, sample_rate, audio_int16)
            result = self.transcribe(
                temp_path,
                language=language,
                hotwords=hotwords,
                **kwargs
            )
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        return result

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes for current model variant"""
        return self.LANGUAGE_MAP.get(self._model_variant, [])

    @property
    def name(self) -> str:
        return f"funasr_{self._model_variant}"

    @property
    def device(self) -> str:
        return self._device

    def is_available(self) -> bool:
        """Check if Fun-ASR is initialized"""
        self._ensure_initialized()
        return self._model is not None


__all__ = ['FunASRBackend']
