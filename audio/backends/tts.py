"""
TTS Backend Implementations

Concrete implementations of TTSBackend for IndexTTS2 and Kokoro.
These wrappers preserve existing behavior while conforming to the interface.

Usage:
    from audio.backends.tts import IndexTTS2Backend, KokoroBackend

    # Kokoro (fast, CPU-based)
    kokoro = KokoroBackend(checkpoints_dir="./checkpoints")
    path = kokoro.synthesize("Hello world", voice="af_sarah")

    # IndexTTS2 (high quality, GPU-based)
    indextts = IndexTTS2Backend()  # Uses external IndexTTS2 model
    path = indextts.synthesize("Hello world", voice="custom_voice")
"""

import os
import tempfile
import paths
from pathlib import Path
from typing import Optional, List

from audio.interface import TTSBackend, EmotionResult
from audio.emotion_tts import get_tts_params_for_emotion, get_indextts_emotion_params


class KokoroBackend(TTSBackend):
    """
    Kokoro TTS backend - fast ONNX-based synthesis.

    Uses kokoro-onnx for CPU-based text-to-speech with preset voices.
    No voice cloning - uses predefined voice embeddings.
    """

    def __init__(self, checkpoints_dir: str = None):
        """
        Initialize Kokoro backend.

        Args:
            checkpoints_dir: Path to checkpoint directory.
                            If None, uses ./checkpoints relative to project root.
        """
        if checkpoints_dir is None:
            # Default to models directory
            checkpoints_dir = paths.MODELS_DIR

        self._checkpoints_dir = Path(checkpoints_dir)
        self._model = None
        self._device = "cpu"  # Kokoro ONNX is CPU-only

        # Lazy initialization
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy load the Kokoro model"""
        if self._initialized:
            return

        try:
            from audio.tts_kokoro import KokoroTTS
            self._model = KokoroTTS(str(self._checkpoints_dir))
            self._initialized = True
        except Exception as e:
            print(f"[KokoroBackend] Failed to initialize: {e}")
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
        Synthesize speech using Kokoro.

        Args:
            text: Text to synthesize
            voice: Voice identifier (e.g., 'af_sarah', 'am_adam')
            emotion: Optional emotion to apply (affects speed/pitch)
            output_path: Output file path (optional, creates temp file if None)
            **kwargs: Additional params (speed, pitch)

        Returns:
            Path to generated audio file
        """
        self._ensure_initialized()

        if self._model is None:
            raise RuntimeError("Kokoro model not initialized")

        # Generate output path if not provided
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="kokoro_")
            os.close(fd)

        # Get emotion parameters
        speed = kwargs.get('speed', 1.0)
        if emotion is not None:
            emo_params = get_tts_params_for_emotion(emotion)
            speed = emo_params.get('speed', 1.0)

        # Clamp speed to valid range
        speed = max(0.5, min(2.0, speed))

        # Synthesize
        success = self._model.infer(
            voice_name=voice,
            text=text,
            output_path=output_path,
            speed=speed
        )

        if not success:
            raise RuntimeError("Kokoro synthesis failed")

        return output_path

    def get_available_voices(self) -> List[str]:
        """Return list of available Kokoro voices"""
        self._ensure_initialized()

        if self._model is None:
            return []

        return self._model.get_available_voices()

    def supports_emotion(self) -> bool:
        """Kokoro supports emotion via speed/pitch adjustment"""
        return True

    @property
    def name(self) -> str:
        return "kokoro"

    @property
    def device(self) -> str:
        return self._device

    def is_available(self) -> bool:
        """Check if Kokoro is properly initialized"""
        self._ensure_initialized()
        return self._model is not None


class IndexTTS2Backend(TTSBackend):
    """
    IndexTTS2 TTS backend - high-fidelity voice cloning.

    Uses the external IndexTTS2 model for GPU-accelerated synthesis.
    Supports voice cloning and 8-emotion vector control.

    Note: IndexTTS2 is imported from the external indextts package,
    not from this codebase.
    """

    def __init__(self, model=None, device: str = "cuda"):
        """
        Initialize IndexTTS2 backend.

        Args:
            model: Pre-loaded IndexTTS2 model instance (optional)
            device: Device to use ('cuda' recommended)
        """
        self._model = model
        self._device = device
        self._initialized = model is not None

    def set_model(self, model):
        """
        Set the IndexTTS2 model instance.

        This is used when the model is loaded externally (e.g., in voice_chat_app.py)
        and needs to be shared with this backend.
        """
        self._model = model
        self._initialized = model is not None

    def synthesize(
        self,
        text: str,
        voice: str,
        emotion: Optional[EmotionResult] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Synthesize speech using IndexTTS2.

        Args:
            text: Text to synthesize
            voice: Voice identifier (character voice reference)
            emotion: Optional emotion to express
            output_path: Output file path (optional, model generates its own)
            **kwargs: Additional params (emo_text, emo_alpha, use_emo_text)

        Returns:
            Path to generated audio file
        """
        if self._model is None:
            raise RuntimeError("IndexTTS2 model not set. Call set_model() first.")

        # Get emotion parameters for IndexTTS2
        emo_text = kwargs.get('emo_text')
        emo_alpha = kwargs.get('emo_alpha', 0.6)
        use_emo_text = kwargs.get('use_emo_text', True)

        if emotion is not None and emo_text is None:
            emo_params = get_indextts_emotion_params(emotion, use_vector=False)
            emo_text = emo_params.get('emo_text')
            emo_alpha = emo_params.get('emo_alpha', 0.6)
            use_emo_text = emo_params.get('use_emo_text', True)

        # Call IndexTTS2 infer
        # Returns (audio_path, duration)
        try:
            result = self._model.infer(
                text=text,
                voice=voice,
                emo_text=emo_text,
                emo_alpha=emo_alpha,
                use_emo_text=use_emo_text,
                output_path=output_path
            )

            # Handle different return formats
            if isinstance(result, tuple):
                audio_path = result[0]
            else:
                audio_path = result

            return str(audio_path)

        except Exception as e:
            raise RuntimeError(f"IndexTTS2 synthesis failed: {e}")

    def get_available_voices(self) -> List[str]:
        """
        Return list of available voices.

        For IndexTTS2, voices are character-specific references stored externally.
        This method returns an empty list; voice availability depends on
        the character's voice reference file.
        """
        # IndexTTS2 voices are dynamic (character voice references)
        return []

    def supports_emotion(self) -> bool:
        """IndexTTS2 supports native 8-emotion vectors"""
        return True

    @property
    def name(self) -> str:
        return "indextts2"

    @property
    def device(self) -> str:
        return self._device

    def is_available(self) -> bool:
        """Check if IndexTTS2 model is loaded"""
        return self._model is not None


__all__ = ['IndexTTS2Backend', 'KokoroBackend']
