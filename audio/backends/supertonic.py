"""
Supertonic TTS Backend

Ultra-fast ONNX-based Text-to-Speech using Supertone/supertonic.
Optimized for CPU inference with minimal latency.

Features:
- 66M parameters (lightweight)
- 167x realtime on CPU (M4 Pro), 1000x on GPU
- 6 preset voices (M3-M5, F3-F5)
- Languages: en, ko, es, pt, fr

Usage:
    from audio.backends.supertonic import SupertonicBackend

    tts = SupertonicBackend(model_path="assets/supertonic")
    path = tts.synthesize("Hello world", voice="female_1")

Note: Requires model download:
    git lfs install
    git clone https://huggingface.co/Supertone/supertonic assets/supertonic
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, List

import paths
from audio.interface import TTSBackend, EmotionResult


class SupertonicBackend(TTSBackend):
    """
    Supertonic ultra-fast ONNX TTS backend.

    Provides extremely fast CPU-based synthesis with preset voices.
    No voice cloning - uses 6 predefined voice styles.

    Voices:
    - male_1 (M3), male_2 (M4), male_3 (M5)
    - female_1 (F3), female_2 (F4), female_3 (F5)
    """

    # Voice name to internal ID mapping
    VOICE_MAP = {
        "male_1": "M3",
        "male_2": "M4",
        "male_3": "M5",
        "female_1": "F3",
        "female_2": "F4",
        "female_3": "F5",
    }

    # Reverse mapping for listing
    VOICE_NAMES = list(VOICE_MAP.keys())

    # Supported languages
    SUPPORTED_LANGUAGES = ["en", "ko", "es", "pt", "fr"]

    def __init__(
        self,
        model_path: str = None,
        device: str = "cpu"
    ):
        """
        Initialize Supertonic backend.

        Args:
            model_path: Path to Supertonic model directory.
                       If None, tries ./assets/supertonic relative to project.
            device: Device to run on ('cpu' recommended, 'cuda' untested)
        """
        if model_path is None:
            # Default to models/supertonic
            model_path = paths.SUPERTONIC_MODELS_DIR

        self._model_path = Path(model_path)
        self._device = device
        self._model = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy load the Supertonic model"""
        if self._initialized:
            return

        try:
            from supertonic import TTS

            print(f"[Supertonic] Loading model...")

            # Initialize TTS with auto_download (supertonic handles caching)
            self._model = TTS()

            self._initialized = True
            print("[Supertonic] Model loaded successfully")

        except ImportError as e:
            print(f"[Supertonic] supertonic not available: {e}")
            print("[Supertonic] Install with: pip install supertonic")
            self._model = None

        except Exception as e:
            print(f"[Supertonic] Failed to load model: {e}")
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
        Synthesize speech using Supertonic.

        Args:
            text: Text to synthesize
            voice: Voice identifier ('male_1', 'female_1', etc. or 'M3', 'F3', etc.)
            emotion: Optional emotion (affects speed only - limited support)
            output_path: Output file path (optional, creates temp file if None)
            **kwargs: Additional params (speed, language)

        Returns:
            Path to generated audio file
        """
        self._ensure_initialized()

        if self._model is None:
            raise RuntimeError("Supertonic model not initialized")

        # Generate output path if not provided
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="supertonic_")
            os.close(fd)

        # Map voice name to internal ID
        voice_id = self.VOICE_MAP.get(voice, voice)

        # Get speed (emotion has limited effect - speed only)
        speed = kwargs.get('speed', 1.0)
        if emotion is not None:
            # Handle both EmotionResult objects and string labels
            if hasattr(emotion, 'arousal'):
                arousal = emotion.arousal
            else:
                # String label - map to arousal value
                emotion_arousal_map = {
                    'happy': 0.7, 'excited': 0.8, 'angry': 0.9, 'anxious': 0.7,
                    'sad': 0.3, 'calm': 0.4, 'neutral': 0.5, 'fear': 0.6
                }
                arousal = emotion_arousal_map.get(str(emotion).lower(), 0.5)
            # Higher arousal = slightly faster
            arousal_adjustment = (arousal - 0.5) * 0.2
            speed = 1.0 + arousal_adjustment
            speed = max(0.8, min(1.2, speed))  # Clamp to reasonable range

        try:
            # Get voice style from the model
            voice_style = self._model.get_voice_style(voice_name=voice_id)

            # Synthesize audio - returns (wav_array, duration)
            wav, duration = self._model.synthesize(text, voice_style=voice_style)

            # Save audio to file using Supertonic's built-in method
            self._model.save_audio(wav, output_path)

            return output_path

        except Exception as e:
            raise RuntimeError(f"Supertonic synthesis failed: {e}")

    def get_available_voices(self) -> List[str]:
        """Return list of available Supertonic voices"""
        return self.VOICE_NAMES.copy()

    def supports_emotion(self) -> bool:
        """
        Supertonic has limited emotion support via speed adjustment.
        Returns False as it doesn't have native emotion vectors.
        """
        return False

    @property
    def name(self) -> str:
        return "supertonic"

    @property
    def device(self) -> str:
        return self._device

    def is_available(self) -> bool:
        """Check if Supertonic is properly initialized"""
        self._ensure_initialized()
        return self._model is not None


__all__ = ['SupertonicBackend']
