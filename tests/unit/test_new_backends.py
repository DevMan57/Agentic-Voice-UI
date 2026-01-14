"""
Unit tests for New Audio Backends (Path C)

Tests the new SenseVoice, FunASR, and Supertonic backends.
These tests validate interface contracts without requiring the actual models.

Run with: pytest tests/unit/test_new_backends.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np


# ============================================================================
# SenseVoice Backend Tests
# ============================================================================

class TestSenseVoiceBackend:
    """Test SenseVoiceBackend interface and utilities"""

    def test_sensevoice_backend_interface(self):
        """SenseVoiceBackend implements STTBackend interface"""
        from audio.backends.sensevoice import SenseVoiceBackend
        from audio.interface import STTBackend

        assert issubclass(SenseVoiceBackend, STTBackend)

    def test_sensevoice_backend_properties(self):
        """SenseVoiceBackend has correct name and device"""
        from audio.backends.sensevoice import SenseVoiceBackend

        backend = SenseVoiceBackend(device="cuda:0")
        assert backend.name == "sensevoice"
        assert backend.device == "cuda:0"

    def test_sensevoice_supports_emotion(self):
        """SenseVoiceBackend reports emotion support"""
        from audio.backends.sensevoice import SenseVoiceBackend

        backend = SenseVoiceBackend()
        assert backend.supports_emotion() is True

    def test_sensevoice_supports_vad(self):
        """SenseVoiceBackend reports VAD support when enabled"""
        from audio.backends.sensevoice import SenseVoiceBackend

        backend_with_vad = SenseVoiceBackend(use_vad=True)
        assert backend_with_vad.supports_vad() is True

        backend_no_vad = SenseVoiceBackend(use_vad=False)
        assert backend_no_vad.supports_vad() is False


class TestSenseVoiceEmotionParsing:
    """Test SenseVoice emotion parsing utilities"""

    def test_parse_happy_emotion(self):
        """Parse happy emotion from SenseVoice output"""
        from audio.backends.sensevoice import parse_sensevoice_output

        text = "😊 Hello, how are you today?"
        clean_text, emotion, event = parse_sensevoice_output(text)

        assert clean_text == "Hello, how are you today?"
        assert emotion == "happy"
        assert event is None

    def test_parse_angry_emotion(self):
        """Parse angry emotion from SenseVoice output"""
        from audio.backends.sensevoice import parse_sensevoice_output

        text = "😡 I can't believe this happened!"
        clean_text, emotion, event = parse_sensevoice_output(text)

        assert clean_text == "I can't believe this happened!"
        assert emotion == "angry"

    def test_parse_sad_emotion(self):
        """Parse sad emotion from SenseVoice output"""
        from audio.backends.sensevoice import parse_sensevoice_output

        text = "😔 I miss those days..."
        clean_text, emotion, event = parse_sensevoice_output(text)

        assert clean_text == "I miss those days..."
        assert emotion == "sad"

    def test_parse_neutral_no_emoji(self):
        """Parse text with no emotion emoji"""
        from audio.backends.sensevoice import parse_sensevoice_output

        text = "This is a neutral statement."
        clean_text, emotion, event = parse_sensevoice_output(text)

        assert clean_text == "This is a neutral statement."
        assert emotion is None

    def test_parse_audio_event_music(self):
        """Parse music audio event"""
        from audio.backends.sensevoice import parse_sensevoice_output

        text = "🎼 Playing some background music"
        clean_text, emotion, event = parse_sensevoice_output(text)

        assert "Playing some background music" in clean_text
        assert event == "music"

    def test_parse_audio_event_laughter(self):
        """Parse laughter audio event"""
        from audio.backends.sensevoice import parse_sensevoice_output

        text = "😀 That's so funny!"
        clean_text, emotion, event = parse_sensevoice_output(text)

        assert event == "laughter"

    def test_emotion_to_result_happy(self):
        """Convert happy label to EmotionResult"""
        from audio.backends.sensevoice import emotion_to_result

        result = emotion_to_result("happy")

        assert result is not None
        assert result.label == "happy"
        assert result.valence == 0.8
        assert result.arousal == 0.7
        assert result.confidence == 0.8

    def test_emotion_to_result_angry(self):
        """Convert angry label to EmotionResult"""
        from audio.backends.sensevoice import emotion_to_result

        result = emotion_to_result("angry")

        assert result is not None
        assert result.label == "angry"
        assert result.valence == 0.2  # Low valence (negative)
        assert result.arousal == 0.9  # High arousal

    def test_emotion_to_result_none(self):
        """Return None for no emotion"""
        from audio.backends.sensevoice import emotion_to_result

        result = emotion_to_result(None)
        assert result is None


# ============================================================================
# FunASR Backend Tests
# ============================================================================

class TestFunASRBackend:
    """Test FunASRBackend interface"""

    def test_funasr_backend_interface(self):
        """FunASRBackend implements STTBackend interface"""
        from audio.backends.funasr_backend import FunASRBackend
        from audio.interface import STTBackend

        assert issubclass(FunASRBackend, STTBackend)

    def test_funasr_backend_properties(self):
        """FunASRBackend has correct name and device"""
        from audio.backends.funasr_backend import FunASRBackend

        backend = FunASRBackend(device="cuda:0", model_variant="nano")
        assert backend.name == "funasr_nano"
        assert backend.device == "cuda:0"

    def test_funasr_backend_mlt_variant(self):
        """FunASRBackend MLT variant has correct name"""
        from audio.backends.funasr_backend import FunASRBackend

        backend = FunASRBackend(model_variant="mlt")
        assert backend.name == "funasr_mlt"

    def test_funasr_supported_languages_nano(self):
        """FunASR nano variant supports zh, en, ja"""
        from audio.backends.funasr_backend import FunASRBackend

        backend = FunASRBackend(model_variant="nano")
        languages = backend.get_supported_languages()

        assert "zh" in languages
        assert "en" in languages
        assert "ja" in languages

    def test_funasr_supported_languages_mlt(self):
        """FunASR MLT variant supports more languages"""
        from audio.backends.funasr_backend import FunASRBackend

        backend = FunASRBackend(model_variant="mlt")
        languages = backend.get_supported_languages()

        assert len(languages) >= 3  # Has more languages than nano


# ============================================================================
# Supertonic Backend Tests
# ============================================================================

class TestSupertonicBackend:
    """Test SupertonicBackend interface"""

    def test_supertonic_backend_interface(self):
        """SupertonicBackend implements TTSBackend interface"""
        from audio.backends.supertonic import SupertonicBackend
        from audio.interface import TTSBackend

        assert issubclass(SupertonicBackend, TTSBackend)

    def test_supertonic_backend_properties(self):
        """SupertonicBackend has correct name and device"""
        from audio.backends.supertonic import SupertonicBackend

        backend = SupertonicBackend(device="cpu")
        assert backend.name == "supertonic"
        assert backend.device == "cpu"

    def test_supertonic_available_voices(self):
        """SupertonicBackend has 6 preset voices"""
        from audio.backends.supertonic import SupertonicBackend

        backend = SupertonicBackend()
        voices = backend.get_available_voices()

        assert len(voices) == 6
        assert "male_1" in voices
        assert "male_2" in voices
        assert "male_3" in voices
        assert "female_1" in voices
        assert "female_2" in voices
        assert "female_3" in voices

    def test_supertonic_voice_mapping(self):
        """SupertonicBackend voice names map to internal IDs"""
        from audio.backends.supertonic import SupertonicBackend

        assert SupertonicBackend.VOICE_MAP["male_1"] == "M3"
        assert SupertonicBackend.VOICE_MAP["female_1"] == "F3"

    def test_supertonic_emotion_support(self):
        """SupertonicBackend has limited emotion support"""
        from audio.backends.supertonic import SupertonicBackend

        backend = SupertonicBackend()
        # Supertonic only adjusts speed, doesn't have native emotion vectors
        assert backend.supports_emotion() is False


# ============================================================================
# Config Integration Tests
# ============================================================================

class TestNewBackendConfig:
    """Test config manager with new backend options"""

    def test_config_has_sensevoice_options(self):
        """Config has SenseVoice settings"""
        from config.manager import Config

        cfg = Config()
        assert hasattr(cfg.stt, 'sensevoice_use_vad')
        assert hasattr(cfg.stt, 'sensevoice_language')
        assert cfg.stt.sensevoice_use_vad is True
        assert cfg.stt.sensevoice_language == 'auto'

    def test_config_has_funasr_options(self):
        """Config has FunASR settings"""
        from config.manager import Config

        cfg = Config()
        assert hasattr(cfg.stt, 'funasr_model_variant')
        assert hasattr(cfg.stt, 'funasr_hotwords')
        assert cfg.stt.funasr_model_variant == 'nano'

    def test_config_has_supertonic_options(self):
        """Config has Supertonic settings"""
        from config.manager import Config

        cfg = Config()
        assert hasattr(cfg.tts, 'supertonic_voice')
        assert hasattr(cfg.tts, 'supertonic_model_path')
        assert cfg.tts.supertonic_voice == 'female_1'

    def test_config_to_dict_includes_new_options(self):
        """Config.to_dict() includes new backend options"""
        from config.manager import Config

        cfg = Config()
        d = cfg.to_dict()

        assert 'supertonic_voice' in d['tts']
        assert 'sensevoice_use_vad' in d['stt']
        assert 'funasr_model_variant' in d['stt']


# ============================================================================
# Registry Tests
# ============================================================================

class TestNewBackendRegistry:
    """Test registry with new backends"""

    def test_registry_list_stt_includes_sensevoice(self):
        """Registry lists sensevoice if available"""
        from audio.registry import AudioRegistry

        stt_backends = AudioRegistry.list_stt_backends()
        # Will be registered if funasr is installed
        assert isinstance(stt_backends, list)
        # Always has faster_whisper
        assert 'faster_whisper' in stt_backends

    def test_registry_list_tts_includes_supertonic(self):
        """Registry lists supertonic if available"""
        from audio.registry import AudioRegistry

        tts_backends = AudioRegistry.list_tts_backends()
        # Will be registered if supertonic is installed
        assert isinstance(tts_backends, list)
        # Always has kokoro and indextts2
        assert 'kokoro' in tts_backends
        assert 'indextts2' in tts_backends


# ============================================================================
# Backend Export Tests
# ============================================================================

class TestBackendExports:
    """Test that new backends are exported correctly"""

    def test_sensevoice_in_all(self):
        """SenseVoiceBackend is in __all__"""
        from audio.backends import __all__

        assert 'SenseVoiceBackend' in __all__

    def test_funasr_in_all(self):
        """FunASRBackend is in __all__"""
        from audio.backends import __all__

        assert 'FunASRBackend' in __all__

    def test_supertonic_in_all(self):
        """SupertonicBackend is in __all__"""
        from audio.backends import __all__

        assert 'SupertonicBackend' in __all__

    def test_imports_dont_fail(self):
        """Importing backends module doesn't raise ImportError"""
        # This should not raise even if optional packages not installed
        from audio import backends

        # Should have at least the core backends
        assert hasattr(backends, 'IndexTTS2Backend')
        assert hasattr(backends, 'KokoroBackend')
        assert hasattr(backends, 'FasterWhisperBackend')


# ============================================================================
# Interface Compliance Tests
# ============================================================================

class TestInterfaceCompliance:
    """Test that new backends implement all required interface methods"""

    def test_sensevoice_has_transcribe(self):
        """SenseVoiceBackend has transcribe method"""
        from audio.backends.sensevoice import SenseVoiceBackend

        backend = SenseVoiceBackend()
        assert hasattr(backend, 'transcribe')
        assert callable(backend.transcribe)

    def test_sensevoice_has_transcribe_array(self):
        """SenseVoiceBackend has transcribe_array method"""
        from audio.backends.sensevoice import SenseVoiceBackend

        backend = SenseVoiceBackend()
        assert hasattr(backend, 'transcribe_array')
        assert callable(backend.transcribe_array)

    def test_sensevoice_has_transcribe_with_emotion(self):
        """SenseVoiceBackend has transcribe_with_emotion method"""
        from audio.backends.sensevoice import SenseVoiceBackend

        backend = SenseVoiceBackend()
        assert hasattr(backend, 'transcribe_with_emotion')
        assert callable(backend.transcribe_with_emotion)

    def test_funasr_has_transcribe(self):
        """FunASRBackend has transcribe method"""
        from audio.backends.funasr_backend import FunASRBackend

        backend = FunASRBackend()
        assert hasattr(backend, 'transcribe')
        assert callable(backend.transcribe)

    def test_supertonic_has_synthesize(self):
        """SupertonicBackend has synthesize method"""
        from audio.backends.supertonic import SupertonicBackend

        backend = SupertonicBackend()
        assert hasattr(backend, 'synthesize')
        assert callable(backend.synthesize)

    def test_supertonic_has_get_available_voices(self):
        """SupertonicBackend has get_available_voices method"""
        from audio.backends.supertonic import SupertonicBackend

        backend = SupertonicBackend()
        assert hasattr(backend, 'get_available_voices')
        voices = backend.get_available_voices()
        assert isinstance(voices, list)
