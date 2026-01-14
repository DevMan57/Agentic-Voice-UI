"""
Unit tests for ConfigManager (config/manager.py)

Tests cover:
1. Environment variable parsing functions (_env_float, _env_int, _env_bool, _env_str)
2. Config dataclass initialization and defaults
3. Per-character override system

Run with: pytest tests/unit/test_config_manager.py -v
"""

import pytest
import os
from unittest.mock import patch
from pathlib import Path

from config.manager import (
    _env_float, _env_int, _env_bool, _env_str,
    AudioConfig, MemoryConfig, ServerConfig, LLMConfig, EmotionConfig, PathsConfig,
    Config
)


# ============================================================================
# Test Class 1: Environment Variable Parsing
# ============================================================================

class TestEnvironmentVariableParsing:
    """Test environment variable parsing helper functions."""

    def test_env_float_valid_values(self):
        """Test _env_float with valid float strings."""
        with patch.dict(os.environ, {'TEST_FLOAT': '3.14'}):
            assert _env_float('TEST_FLOAT', 1.0) == 3.14

        with patch.dict(os.environ, {'TEST_FLOAT': '1.5e-3'}):
            result = _env_float('TEST_FLOAT', 1.0)
            assert abs(result - 0.0015) < 1e-6  # Scientific notation

        with patch.dict(os.environ, {'TEST_FLOAT': '-2.5'}):
            assert _env_float('TEST_FLOAT', 1.0) == -2.5

        with patch.dict(os.environ, {'TEST_FLOAT': '0.0'}):
            assert _env_float('TEST_FLOAT', 1.0) == 0.0

    def test_env_float_invalid_returns_default(self):
        """Test _env_float returns default for invalid values."""
        with patch.dict(os.environ, {'TEST_FLOAT': 'invalid_float'}):
            assert _env_float('TEST_FLOAT', 99.9) == 99.9

        with patch.dict(os.environ, {'TEST_FLOAT': ''}):
            assert _env_float('TEST_FLOAT', 99.9) == 99.9

        # Unset environment variable
        with patch.dict(os.environ, {}, clear=True):
            assert _env_float('TEST_FLOAT', 99.9) == 99.9

    def test_env_int_valid_values(self):
        """Test _env_int with valid integer strings."""
        with patch.dict(os.environ, {'TEST_INT': '42'}):
            assert _env_int('TEST_INT', 10) == 42

        with patch.dict(os.environ, {'TEST_INT': '-10'}):
            assert _env_int('TEST_INT', 10) == -10

        with patch.dict(os.environ, {'TEST_INT': '0'}):
            assert _env_int('TEST_INT', 10) == 0

    def test_env_int_invalid_returns_default(self):
        """Test _env_int returns default for invalid values."""
        # Float string (not an int)
        with patch.dict(os.environ, {'TEST_INT': '3.14'}):
            assert _env_int('TEST_INT', 99) == 99

        with patch.dict(os.environ, {'TEST_INT': 'not_a_number'}):
            assert _env_int('TEST_INT', 99) == 99

        # Unset environment variable
        with patch.dict(os.environ, {}, clear=True):
            assert _env_int('TEST_INT', 99) == 99

    def test_env_bool_true_values(self):
        """Test _env_bool recognizes true values (case insensitive)."""
        for true_value in ['true', '1', 'yes', 'on', 'True', 'TRUE', 'YES']:
            with patch.dict(os.environ, {'TEST_BOOL': true_value}):
                assert _env_bool('TEST_BOOL', False) is True, f"Failed for: {true_value}"

    def test_env_bool_false_or_default(self):
        """Test _env_bool returns False for non-true values (not default)."""
        # When env var is set to non-true values, function returns False (not default)
        for false_value in ['false', '0', 'no', 'off', 'False', 'random']:
            with patch.dict(os.environ, {'TEST_BOOL': false_value}):
                # Should return False (actual behavior)
                assert _env_bool('TEST_BOOL', True) is False, f"Failed for: {false_value}"

        # Unset environment variable returns default
        with patch.dict(os.environ, {}, clear=True):
            assert _env_bool('TEST_BOOL', True) is True
            assert _env_bool('TEST_BOOL', False) is False

    def test_env_str_passthrough(self):
        """Test _env_str passes through string values."""
        with patch.dict(os.environ, {'TEST_STR': 'hello'}):
            assert _env_str('TEST_STR', 'default') == 'hello'

        # Empty string is valid (not default)
        with patch.dict(os.environ, {'TEST_STR': ''}):
            assert _env_str('TEST_STR', 'default') == ''

        # Unset returns default
        with patch.dict(os.environ, {}, clear=True):
            assert _env_str('TEST_STR', 'default') == 'default'


# ============================================================================
# Test Class 2: Config Dataclasses
# ============================================================================

class TestConfigDataclasses:
    """Test config dataclass initialization and defaults."""

    def test_audio_config_defaults(self):
        """Test AudioConfig loads with correct default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = AudioConfig()

            assert config.vad_energy_threshold == 0.015
            assert config.silero_threshold == 0.6
            assert config.consecutive_frames_required == 5
            assert config.sample_rate == 16000
            assert config.chunk_size == 512
            assert config.channels == 1
            assert config.tts_backend == 'indextts'
            assert config.kokoro_speed == 1.0

    def test_audio_config_env_overrides(self):
        """Test AudioConfig respects environment variable overrides."""
        with patch.dict(os.environ, {
            'VAD_ENERGY_THRESHOLD': '0.02',
            'SILERO_THRESHOLD': '0.7',
            'AUDIO_SAMPLE_RATE': '48000'
        }):
            config = AudioConfig()

            # Overridden values
            assert config.vad_energy_threshold == 0.02
            assert config.silero_threshold == 0.7
            assert config.sample_rate == 48000

            # Default values (not overridden)
            assert config.chunk_size == 512
            assert config.tts_backend == 'indextts'

    def test_memory_config_defaults(self):
        """Test MemoryConfig loads with correct default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = MemoryConfig()

            # Weighted retrieval weights
            assert config.recency_weight == 0.2
            assert config.relevance_weight == 0.5
            assert config.importance_weight == 0.3

            # Embedding settings
            assert config.embedding_model == 'Svenni551/Qwen3-Embedding-0.6B-ONNX-INT8'
            assert config.embedding_dimension == 1024
            assert config.embedding_cache_size == 1000

            # Context limits
            assert config.max_context_tokens == 2000
            assert config.summary_interval == 10

    def test_server_config_defaults(self):
        """Test ServerConfig loads with correct default ports."""
        with patch.dict(os.environ, {}, clear=True):
            config = ServerConfig()

            assert config.port == 7861
            assert config.character_manager_port == 7863
            assert config.mcp_manager_port == 7864
            assert config.lm_studio_port == 1235
            assert config.llm_timeout == 120
            assert config.tool_timeout == 30

    def test_llm_config_defaults(self):
        """Test LLMConfig loads with correct defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig()

            assert config.provider == 'openrouter'
            assert config.default_model == 'anthropic/claude-sonnet-4'
            assert config.temperature == 0.7
            assert config.max_tokens == 2000
            assert config.top_p == 1.0

    def test_emotion_config_defaults(self):
        """Test EmotionConfig loads with correct defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = EmotionConfig()

            assert config.enabled is True
            assert config.timeout == 2.0
            assert 'calm' in config.emotion_speed_map
            assert config.emotion_speed_map['happy'] == 1.1
            assert config.emotion_pitch_map['sad'] == -0.15

    def test_paths_config_computed_properties(self):
        """Test PathsConfig computes paths correctly."""
        config = PathsConfig()

        # root_dir should be the voice_chat directory
        assert isinstance(config.root_dir, Path)
        assert config.root_dir.exists()
        assert config.root_dir.name == 'voice_chat'

        # Computed properties should be Path objects
        assert isinstance(config.sessions_dir, Path)
        assert isinstance(config.recordings_dir, Path)
        assert isinstance(config.memory_db, Path)

        # Verify path relationships
        assert config.sessions_dir == config.root_dir / "sessions"
        assert config.recordings_dir == config.root_dir / "recordings"
        assert config.checkpoints_dir == config.root_dir / "checkpoints"
        assert config.skills_dir == config.root_dir / "skills"
        assert config.memory_db == config.sessions_dir / "memory.db"


# ============================================================================
# Test Class 3: Per-Character Overrides
# ============================================================================

class TestPerCharacterOverrides:
    """Test per-character configuration override system."""

    def test_get_for_character_with_override(self):
        """Test retrieving an existing character override."""
        config = Config()
        config.set_character_override('hermione', 'audio.kokoro_speed', 1.2)

        result = config.get_for_character('hermione', 'audio.kokoro_speed', 1.0)
        assert result == 1.2

    def test_get_for_character_without_override(self):
        """Test retrieving default when no override exists."""
        config = Config()

        # No override set, should return default
        result = config.get_for_character('hermione', 'audio.kokoro_speed', 1.0)
        assert result == 1.0

    def test_get_for_character_nonexistent_character(self):
        """Test retrieving from non-existent character returns default."""
        config = Config()

        # Character doesn't exist in overrides
        result = config.get_for_character('nonexistent', 'some.key', 'default_value')
        assert result == 'default_value'

    def test_set_character_override_creates_entry(self):
        """Test setting an override creates character entry if needed."""
        config = Config()

        # Verify character not in overrides initially
        assert 'hermione' not in config.character_overrides

        # Set override
        config.set_character_override('hermione', 'test_key', 'test_value')

        # Verify character now exists
        assert 'hermione' in config.character_overrides
        assert config.character_overrides['hermione']['test_key'] == 'test_value'

    def test_multiple_overrides_per_character(self):
        """Test setting multiple overrides for same character."""
        config = Config()

        config.set_character_override('hermione', 'key1', 'value1')
        config.set_character_override('hermione', 'key2', 'value2')
        config.set_character_override('hermione', 'key3', 'value3')

        # All overrides should exist
        assert len(config.character_overrides['hermione']) == 3
        assert config.character_overrides['hermione']['key1'] == 'value1'
        assert config.character_overrides['hermione']['key2'] == 'value2'
        assert config.character_overrides['hermione']['key3'] == 'value3'

    def test_overrides_for_different_characters(self):
        """Test overrides are isolated per character."""
        config = Config()

        config.set_character_override('hermione', 'mood', 'studious')
        config.set_character_override('lisbeth', 'mood', 'hacker')

        assert config.get_for_character('hermione', 'mood', 'neutral') == 'studious'
        assert config.get_for_character('lisbeth', 'mood', 'neutral') == 'hacker'
        assert config.get_for_character('other', 'mood', 'neutral') == 'neutral'


# ============================================================================
# Test Class 4: Config Integration
# ============================================================================

class TestConfigIntegration:
    """Test Config master container and utility functions."""

    def test_config_instantiation(self):
        """Test Config dataclass instantiates with all sub-configs."""
        config = Config()

        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.emotion, EmotionConfig)
        assert isinstance(config.paths, PathsConfig)

    def test_config_to_dict_export(self):
        """Test Config.to_dict() exports configuration."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            config_dict = config.to_dict()

            # Verify main sections exist
            assert 'audio' in config_dict
            assert 'memory' in config_dict
            assert 'server' in config_dict
            assert 'llm' in config_dict
            assert 'emotion' in config_dict

            # Verify some values are correct
            assert config_dict['audio']['vad_energy_threshold'] == 0.015
            assert config_dict['memory']['recency_weight'] == 0.2
            assert config_dict['server']['port'] == 7861
            assert config_dict['llm']['provider'] == 'openrouter'

    def test_character_overrides_dictionary(self):
        """Test character_overrides is initialized as empty dict."""
        config = Config()

        assert isinstance(config.character_overrides, dict)
        assert len(config.character_overrides) == 0
