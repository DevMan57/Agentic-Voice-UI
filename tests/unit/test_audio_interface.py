"""
Unit tests for Audio Interface and Registry

Tests the abstract base classes, backend registry, and backend wrappers.
These tests validate interface contracts and registry operations.

Run with: pytest tests/unit/test_audio_interface.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from abc import ABC
import numpy as np


# ============================================================================
# Interface Contract Tests
# ============================================================================

class TestTTSBackendInterface:
    """Test TTSBackend abstract interface"""

    def test_tts_backend_is_abstract(self):
        """TTSBackend cannot be instantiated directly"""
        from audio.interface import TTSBackend

        assert issubclass(TTSBackend, ABC)

        with pytest.raises(TypeError):
            TTSBackend()

    def test_tts_backend_requires_synthesize(self):
        """TTSBackend subclass must implement synthesize()"""
        from audio.interface import TTSBackend

        class IncompleteTTS(TTSBackend):
            def get_available_voices(self):
                return []

            def supports_emotion(self):
                return False

            @property
            def name(self):
                return "incomplete"

            @property
            def device(self):
                return "cpu"

        with pytest.raises(TypeError):
            IncompleteTTS()

    def test_tts_backend_concrete_implementation(self):
        """Complete TTSBackend implementation can be instantiated"""
        from audio.interface import TTSBackend

        class ConcreteTTS(TTSBackend):
            def synthesize(self, text, voice, emotion=None, output_path=None, **kwargs):
                return "/tmp/test.wav"

            def get_available_voices(self):
                return ["voice1", "voice2"]

            def supports_emotion(self):
                return True

            @property
            def name(self):
                return "test_tts"

            @property
            def device(self):
                return "cpu"

        tts = ConcreteTTS()
        assert tts.name == "test_tts"
        assert tts.device == "cpu"
        assert tts.supports_emotion() is True
        assert "voice1" in tts.get_available_voices()


class TestSTTBackendInterface:
    """Test STTBackend abstract interface"""

    def test_stt_backend_is_abstract(self):
        """STTBackend cannot be instantiated directly"""
        from audio.interface import STTBackend

        assert issubclass(STTBackend, ABC)

        with pytest.raises(TypeError):
            STTBackend()

    def test_stt_backend_requires_transcribe(self):
        """STTBackend subclass must implement transcribe()"""
        from audio.interface import STTBackend

        class IncompleteSTT(STTBackend):
            def transcribe_array(self, audio_array, sample_rate, language=None, **kwargs):
                pass

            @property
            def name(self):
                return "incomplete"

            @property
            def device(self):
                return "cpu"

        with pytest.raises(TypeError):
            IncompleteSTT()


class TestVADBackendInterface:
    """Test VADBackend abstract interface"""

    def test_vad_backend_is_abstract(self):
        """VADBackend cannot be instantiated directly"""
        from audio.interface import VADBackend

        assert issubclass(VADBackend, ABC)

        with pytest.raises(TypeError):
            VADBackend()

    def test_vad_backend_requires_is_speech(self):
        """VADBackend subclass must implement is_speech()"""
        from audio.interface import VADBackend

        class IncompleteVAD(VADBackend):
            def reset(self):
                pass

            @property
            def name(self):
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteVAD()

    def test_vad_backend_default_probability(self):
        """VADBackend provides default get_speech_probability()"""
        from audio.interface import VADBackend

        class ConcreteVAD(VADBackend):
            def __init__(self):
                self._last_result = False

            def is_speech(self, audio_chunk):
                return self._last_result

            def reset(self):
                pass

            @property
            def name(self):
                return "test_vad"

        vad = ConcreteVAD()
        vad._last_result = True
        assert vad.get_speech_probability(np.zeros(512)) == 1.0

        vad._last_result = False
        assert vad.get_speech_probability(np.zeros(512)) == 0.0


class TestEmotionAnalyzerInterface:
    """Test EmotionAnalyzer abstract interface"""

    def test_emotion_analyzer_is_abstract(self):
        """EmotionAnalyzer cannot be instantiated directly"""
        from audio.interface import EmotionAnalyzer

        assert issubclass(EmotionAnalyzer, ABC)

        with pytest.raises(TypeError):
            EmotionAnalyzer()


# ============================================================================
# Data Class Tests
# ============================================================================

class TestEmotionResult:
    """Test EmotionResult dataclass"""

    def test_emotion_result_creation(self):
        """EmotionResult can be created with all fields"""
        from audio.interface import EmotionResult

        result = EmotionResult(
            valence=0.7,
            arousal=0.5,
            dominance=0.6,
            label="happy",
            confidence=0.9
        )

        assert result.valence == 0.7
        assert result.arousal == 0.5
        assert result.dominance == 0.6
        assert result.label == "happy"
        assert result.confidence == 0.9

    def test_emotion_result_to_dict(self):
        """EmotionResult.to_dict() returns correct dictionary"""
        from audio.interface import EmotionResult

        result = EmotionResult(
            valence=0.7,
            arousal=0.5,
            dominance=0.6,
            label="happy",
            confidence=0.9
        )

        d = result.to_dict()
        assert d['valence'] == 0.7
        assert d['label'] == "happy"

    def test_emotion_result_to_prompt_context(self):
        """EmotionResult.to_prompt_context() returns formatted string"""
        from audio.interface import EmotionResult

        result = EmotionResult(
            valence=0.7,
            arousal=0.5,
            dominance=0.6,
            label="happy",
            confidence=0.9
        )

        context = result.to_prompt_context()
        assert "happy" in context
        assert "0.70" in context or "0.7" in context


class TestTranscriptionResult:
    """Test TranscriptionResult dataclass"""

    def test_transcription_result_creation(self):
        """TranscriptionResult can be created"""
        from audio.interface import TranscriptionResult

        result = TranscriptionResult(
            text="Hello world",
            language="en",
            confidence=0.95
        )

        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.confidence == 0.95

    def test_transcription_result_str(self):
        """TranscriptionResult __str__ returns text"""
        from audio.interface import TranscriptionResult

        result = TranscriptionResult(text="Hello world")
        assert str(result) == "Hello world"


# ============================================================================
# Registry Tests
# ============================================================================

class TestAudioRegistry:
    """Test AudioRegistry operations"""

    def test_registry_register_tts(self):
        """Can register a TTS backend"""
        from audio.registry import AudioRegistry
        from audio.interface import TTSBackend

        # Save original state
        original_backends = AudioRegistry._tts_backends.copy()

        try:
            class MockTTS(TTSBackend):
                def synthesize(self, text, voice, emotion=None, output_path=None, **kwargs):
                    return "/tmp/mock.wav"

                def get_available_voices(self):
                    return []

                def supports_emotion(self):
                    return False

                @property
                def name(self):
                    return "mock"

                @property
                def device(self):
                    return "cpu"

            AudioRegistry.register_tts('mock_test', MockTTS)
            assert 'mock_test' in AudioRegistry.list_tts_backends()

        finally:
            # Restore original state
            AudioRegistry._tts_backends = original_backends

    def test_registry_get_tts_unknown(self):
        """Getting unknown backend raises ValueError"""
        from audio.registry import AudioRegistry

        with pytest.raises(ValueError) as excinfo:
            AudioRegistry.get_tts('nonexistent_backend_xyz')

        assert "Unknown TTS backend" in str(excinfo.value)

    def test_registry_list_backends(self):
        """Can list registered backends"""
        from audio.registry import AudioRegistry

        tts_list = AudioRegistry.list_tts_backends()
        stt_list = AudioRegistry.list_stt_backends()
        vad_list = AudioRegistry.list_vad_backends()

        assert isinstance(tts_list, list)
        assert isinstance(stt_list, list)
        assert isinstance(vad_list, list)

    def test_registry_builtin_backends_registered(self):
        """Built-in backends are auto-registered on import"""
        from audio.registry import AudioRegistry

        # TTS backends should be registered
        tts_backends = AudioRegistry.list_tts_backends()
        assert 'kokoro' in tts_backends or 'indextts2' in tts_backends

        # STT backends should be registered
        stt_backends = AudioRegistry.list_stt_backends()
        assert 'faster_whisper' in stt_backends

    def test_registry_status(self):
        """Registry status returns correct structure"""
        from audio.registry import AudioRegistry

        status = AudioRegistry.status()

        assert 'tts' in status
        assert 'stt' in status
        assert 'vad' in status
        assert 'emotion' in status

        assert 'registered' in status['tts']
        assert 'cached' in status['tts']

    def test_registry_clear_cache(self):
        """Clear cache removes all cached instances"""
        from audio.registry import AudioRegistry

        # Clear cache
        AudioRegistry.clear_cache()

        status = AudioRegistry.status()
        assert len(status['tts']['cached']) == 0
        assert len(status['stt']['cached']) == 0


# ============================================================================
# Backend Wrapper Tests (with mocking)
# ============================================================================

class TestKokoroBackend:
    """Test KokoroBackend wrapper"""

    def test_kokoro_backend_name(self):
        """KokoroBackend has correct name"""
        from audio.backends.tts import KokoroBackend

        backend = KokoroBackend(checkpoints_dir="/tmp/test")
        assert backend.name == "kokoro"
        assert backend.device == "cpu"

    def test_kokoro_backend_supports_emotion(self):
        """KokoroBackend supports emotion via speed"""
        from audio.backends.tts import KokoroBackend

        backend = KokoroBackend(checkpoints_dir="/tmp/test")
        assert backend.supports_emotion() is True


class TestIndexTTS2Backend:
    """Test IndexTTS2Backend wrapper"""

    def test_indextts2_backend_name(self):
        """IndexTTS2Backend has correct name"""
        from audio.backends.tts import IndexTTS2Backend

        backend = IndexTTS2Backend(device="cuda")
        assert backend.name == "indextts2"
        assert backend.device == "cuda"

    def test_indextts2_backend_requires_model(self):
        """IndexTTS2Backend requires model to be set"""
        from audio.backends.tts import IndexTTS2Backend

        backend = IndexTTS2Backend()
        assert backend.is_available() is False

        # Mock model
        mock_model = Mock()
        backend.set_model(mock_model)
        assert backend.is_available() is True

    def test_indextts2_backend_supports_emotion(self):
        """IndexTTS2Backend supports native emotion vectors"""
        from audio.backends.tts import IndexTTS2Backend

        backend = IndexTTS2Backend()
        assert backend.supports_emotion() is True


class TestFasterWhisperBackend:
    """Test FasterWhisperBackend wrapper"""

    def test_faster_whisper_backend_name(self):
        """FasterWhisperBackend has correct name"""
        from audio.backends.stt import FasterWhisperBackend

        backend = FasterWhisperBackend(model_size="base", device="cpu")
        # Before initialization, name returns default
        assert backend.name == "faster_whisper"
        assert backend.device == "cpu"


# ============================================================================
# Config Integration Tests
# ============================================================================

class TestConfigTTSSTT:
    """Test TTSConfig and STTConfig in config manager"""

    def test_config_has_tts_config(self):
        """Config has TTSConfig section"""
        from config.manager import Config

        cfg = Config()
        assert hasattr(cfg, 'tts')
        assert hasattr(cfg.tts, 'backend')
        assert hasattr(cfg.tts, 'device')

    def test_config_has_stt_config(self):
        """Config has STTConfig section"""
        from config.manager import Config

        cfg = Config()
        assert hasattr(cfg, 'stt')
        assert hasattr(cfg.stt, 'backend')
        assert hasattr(cfg.stt, 'device')

    def test_config_tts_defaults(self):
        """TTSConfig has sensible defaults"""
        from config.manager import Config

        cfg = Config()
        assert cfg.tts.backend == 'indextts2'
        assert cfg.tts.device == 'cuda'
        assert cfg.tts.kokoro_speed == 1.0

    def test_config_stt_defaults(self):
        """STTConfig has sensible defaults"""
        from config.manager import Config

        cfg = Config()
        assert cfg.stt.backend == 'faster_whisper'
        assert cfg.stt.device == 'cpu'
        assert cfg.stt.whisper_model_size == 'base'

    def test_config_to_dict_includes_tts_stt(self):
        """Config.to_dict() includes tts and stt"""
        from config.manager import Config

        cfg = Config()
        d = cfg.to_dict()

        assert 'tts' in d
        assert 'stt' in d
        assert d['tts']['backend'] == 'indextts2'
        assert d['stt']['backend'] == 'faster_whisper'
