"""
Soprano TTS Backend

Ultra-fast 80M parameter TTS using ekwek/Soprano-80M.
Features streaming synthesis with <15ms latency.

Usage:
    from audio.backends.soprano import SopranoBackend

    tts = SopranoBackend(device="cuda")
    path = tts.synthesize("Hello world", voice="default")

Note: Requires CUDA GPU. CPU support coming soon.
      Install with: pip install soprano-tts
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, List

from audio.interface import TTSBackend, EmotionResult


class SopranoBackend(TTSBackend):
    """
    Soprano ultra-fast TTS backend.

    Uses ekwek/Soprano-80M for GPU-accelerated synthesis.
    Currently no voice cloning - uses default voice.

    Key specs:
    - 80M parameters
    - 32kHz output quality
    - ~2000x realtime speed
    - <15ms streaming latency
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize Soprano backend.

        Args:
            device: Device to run on ('cuda' required for now)
        """
        self._device = device
        self._model = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy load the Soprano model"""
        if self._initialized:
            return

        try:
            from soprano import SopranoTTS

            print(f"[Soprano] Loading model on {self._device}...")
            self._model = SopranoTTS()
            self._initialized = True
            print("[Soprano] Model loaded successfully")

        except ImportError as e:
            print(f"[Soprano] soprano-tts not available: {e}")
            print("[Soprano] Install with: pip install soprano-tts")
            self._model = None

        except Exception as e:
            print(f"[Soprano] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self._model = None

    def synthesize(
        self,
        text: str,
        voice: str,
        emotion: Optional[EmotionResult] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Synthesize speech using Soprano.

        Args:
            text: Text to synthesize
            voice: Voice identifier (ignored - Soprano has one voice)
            emotion: Optional emotion (not supported yet)
            output_path: Output file path (optional, creates temp file if None)
            **kwargs: Additional params (temperature, top_p, repetition_penalty)

        Returns:
            Path to generated audio file
        """
        self._ensure_initialized()

        if self._model is None:
            raise RuntimeError("Soprano model not initialized")

        # Generate output path if not provided
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="soprano_")
            os.close(fd)

        try:
            # Get optional sampling parameters
            temperature = kwargs.get('temperature', 0.3)
            top_p = kwargs.get('top_p', 0.95)
            repetition_penalty = kwargs.get('repetition_penalty', 1.2)

            # Synthesize audio using Soprano's infer method
            # When output_path is provided, it saves to file
            self._model.infer(
                text,
                output_path,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

            return output_path

        except Exception as e:
            raise RuntimeError(f"Soprano synthesis failed: {e}")

    def get_available_voices(self) -> List[str]:
        """Return list of available Soprano voices (currently just default)"""
        return ["default"]

    def supports_emotion(self) -> bool:
        """Soprano doesn't support emotion control yet"""
        return False

    @property
    def name(self) -> str:
        return "soprano"

    @property
    def device(self) -> str:
        return self._device

    def is_available(self) -> bool:
        """Check if Soprano is properly initialized"""
        self._ensure_initialized()
        return self._model is not None


__all__ = ['SopranoBackend']
