"""
Audio Backend Registry

Central registry for discovering and instantiating audio backends.
Allows backend selection via configuration without code changes.

Usage:
    from audio.registry import AudioRegistry

    # Register backends (done automatically on import)
    AudioRegistry.register_tts('kokoro', KokoroBackend)
    AudioRegistry.register_tts('indextts2', IndexTTS2Backend)

    # Get backend by name
    tts = AudioRegistry.get_tts('kokoro')

    # List available backends
    print(AudioRegistry.list_tts_backends())  # ['kokoro', 'indextts2']
"""

from typing import Dict, Type, Optional, Any, List

from audio.interface import TTSBackend, STTBackend, VADBackend, EmotionAnalyzer


class AudioRegistry:
    """
    Central registry for audio backends.

    Provides discovery and instantiation of audio components.
    Backends are registered at import time and can be retrieved by name.
    """

    # Class-level registries
    _tts_backends: Dict[str, Type[TTSBackend]] = {}
    _stt_backends: Dict[str, Type[STTBackend]] = {}
    _vad_backends: Dict[str, Type[VADBackend]] = {}
    _emotion_backends: Dict[str, Type[EmotionAnalyzer]] = {}

    # Cached instances (for singleton-like behavior)
    _tts_instances: Dict[str, TTSBackend] = {}
    _stt_instances: Dict[str, STTBackend] = {}
    _vad_instances: Dict[str, VADBackend] = {}
    _emotion_instances: Dict[str, EmotionAnalyzer] = {}

    # =========================================================================
    # TTS Registry
    # =========================================================================

    @classmethod
    def register_tts(cls, name: str, backend_class: Type[TTSBackend]) -> None:
        """
        Register a TTS backend class.

        Args:
            name: Backend identifier (e.g., 'kokoro', 'indextts2')
            backend_class: Class implementing TTSBackend
        """
        cls._tts_backends[name.lower()] = backend_class

    @classmethod
    def get_tts(
        cls,
        name: str = None,
        cached: bool = True,
        **kwargs
    ) -> Optional[TTSBackend]:
        """
        Get a TTS backend instance.

        Args:
            name: Backend name (default: first registered)
            cached: If True, return cached instance; else create new
            **kwargs: Arguments to pass to backend constructor

        Returns:
            TTSBackend instance or None if not found
        """
        if name is None:
            # Return first available
            if not cls._tts_backends:
                return None
            name = next(iter(cls._tts_backends))

        name = name.lower()
        if name not in cls._tts_backends:
            raise ValueError(f"Unknown TTS backend: {name}. Available: {list(cls._tts_backends.keys())}")

        if cached and name in cls._tts_instances:
            return cls._tts_instances[name]

        backend_class = cls._tts_backends[name]
        instance = backend_class(**kwargs)

        if cached:
            cls._tts_instances[name] = instance

        return instance

    @classmethod
    def list_tts_backends(cls) -> List[str]:
        """Return list of registered TTS backend names"""
        return list(cls._tts_backends.keys())

    # =========================================================================
    # STT Registry
    # =========================================================================

    @classmethod
    def register_stt(cls, name: str, backend_class: Type[STTBackend]) -> None:
        """Register an STT backend class"""
        cls._stt_backends[name.lower()] = backend_class

    @classmethod
    def get_stt(
        cls,
        name: str = None,
        cached: bool = True,
        **kwargs
    ) -> Optional[STTBackend]:
        """Get an STT backend instance"""
        if name is None:
            if not cls._stt_backends:
                return None
            name = next(iter(cls._stt_backends))

        name = name.lower()
        if name not in cls._stt_backends:
            raise ValueError(f"Unknown STT backend: {name}. Available: {list(cls._stt_backends.keys())}")

        if cached and name in cls._stt_instances:
            return cls._stt_instances[name]

        backend_class = cls._stt_backends[name]
        instance = backend_class(**kwargs)

        if cached:
            cls._stt_instances[name] = instance

        return instance

    @classmethod
    def list_stt_backends(cls) -> List[str]:
        """Return list of registered STT backend names"""
        return list(cls._stt_backends.keys())

    # =========================================================================
    # VAD Registry
    # =========================================================================

    @classmethod
    def register_vad(cls, name: str, backend_class: Type[VADBackend]) -> None:
        """Register a VAD backend class"""
        cls._vad_backends[name.lower()] = backend_class

    @classmethod
    def get_vad(
        cls,
        name: str = None,
        cached: bool = True,
        **kwargs
    ) -> Optional[VADBackend]:
        """Get a VAD backend instance"""
        if name is None:
            if not cls._vad_backends:
                return None
            name = next(iter(cls._vad_backends))

        name = name.lower()
        if name not in cls._vad_backends:
            raise ValueError(f"Unknown VAD backend: {name}. Available: {list(cls._vad_backends.keys())}")

        if cached and name in cls._vad_instances:
            return cls._vad_instances[name]

        backend_class = cls._vad_backends[name]
        instance = backend_class(**kwargs)

        if cached:
            cls._vad_instances[name] = instance

        return instance

    @classmethod
    def list_vad_backends(cls) -> List[str]:
        """Return list of registered VAD backend names"""
        return list(cls._vad_backends.keys())

    # =========================================================================
    # Emotion Registry
    # =========================================================================

    @classmethod
    def register_emotion(cls, name: str, backend_class: Type[EmotionAnalyzer]) -> None:
        """Register an emotion analyzer backend class"""
        cls._emotion_backends[name.lower()] = backend_class

    @classmethod
    def get_emotion(
        cls,
        name: str = None,
        cached: bool = True,
        **kwargs
    ) -> Optional[EmotionAnalyzer]:
        """Get an emotion analyzer instance"""
        if name is None:
            if not cls._emotion_backends:
                return None
            name = next(iter(cls._emotion_backends))

        name = name.lower()
        if name not in cls._emotion_backends:
            raise ValueError(f"Unknown emotion backend: {name}. Available: {list(cls._emotion_backends.keys())}")

        if cached and name in cls._emotion_instances:
            return cls._emotion_instances[name]

        backend_class = cls._emotion_backends[name]
        instance = backend_class(**kwargs)

        if cached:
            cls._emotion_instances[name] = instance

        return instance

    @classmethod
    def list_emotion_backends(cls) -> List[str]:
        """Return list of registered emotion backend names"""
        return list(cls._emotion_backends.keys())

    # =========================================================================
    # Utilities
    # =========================================================================

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached backend instances"""
        cls._tts_instances.clear()
        cls._stt_instances.clear()
        cls._vad_instances.clear()
        cls._emotion_instances.clear()

    @classmethod
    def status(cls) -> Dict[str, Any]:
        """Return registry status for debugging"""
        return {
            'tts': {
                'registered': list(cls._tts_backends.keys()),
                'cached': list(cls._tts_instances.keys()),
            },
            'stt': {
                'registered': list(cls._stt_backends.keys()),
                'cached': list(cls._stt_instances.keys()),
            },
            'vad': {
                'registered': list(cls._vad_backends.keys()),
                'cached': list(cls._vad_instances.keys()),
            },
            'emotion': {
                'registered': list(cls._emotion_backends.keys()),
                'cached': list(cls._emotion_instances.keys()),
            },
        }


# =============================================================================
# Auto-register built-in backends on import
# =============================================================================

def _register_builtin_backends():
    """Register all built-in backends"""
    # TTS backends
    try:
        from audio.backends.tts import KokoroBackend, IndexTTS2Backend
        AudioRegistry.register_tts('kokoro', KokoroBackend)
        AudioRegistry.register_tts('indextts2', IndexTTS2Backend)
    except ImportError as e:
        print(f"[AudioRegistry] Could not register TTS backends: {e}")

    try:
        from audio.backends.supertonic import SupertonicBackend
        AudioRegistry.register_tts('supertonic', SupertonicBackend)
    except ImportError:
        pass  # Optional backend - supertonic not installed

    try:
        from audio.backends.soprano import SopranoBackend
        AudioRegistry.register_tts('soprano', SopranoBackend)
    except ImportError:
        pass  # Optional backend - soprano-tts not installed

    # STT backends
    try:
        from audio.backends.stt import FasterWhisperBackend
        AudioRegistry.register_stt('faster_whisper', FasterWhisperBackend)
    except ImportError as e:
        print(f"[AudioRegistry] Could not register STT backends: {e}")

    try:
        from audio.backends.sensevoice import SenseVoiceBackend
        AudioRegistry.register_stt('sensevoice', SenseVoiceBackend)
    except ImportError:
        pass  # Optional backend - funasr not installed

    try:
        from audio.backends.funasr_backend import FunASRBackend
        AudioRegistry.register_stt('funasr', FunASRBackend)
    except ImportError:
        pass  # Optional backend - funasr not installed

    # VAD backends
    try:
        from audio.backends.vad import SileroVADBackend, WebRTCVADBackend, EnergyVADBackend
        AudioRegistry.register_vad('silero', SileroVADBackend)
        AudioRegistry.register_vad('webrtc', WebRTCVADBackend)
        AudioRegistry.register_vad('energy', EnergyVADBackend)
    except ImportError as e:
        print(f"[AudioRegistry] Could not register VAD backends: {e}")


# Register on import
_register_builtin_backends()


__all__ = ['AudioRegistry']
