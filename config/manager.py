"""
Unified Configuration Manager for IndexTTS2 Voice Agent

Centralizes all configurable settings that were previously scattered
across multiple files. Supports environment variable overrides and
per-character configuration.

Usage:
    from config.manager import config
    
    # Access settings
    threshold = config.audio.vad_energy_threshold
    port = config.server.port
    
    # Environment overrides (optional)
    # Set VAD_ENERGY_THRESHOLD=0.02 in environment
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


def _env_float(key: str, default: float) -> float:
    """Get float from environment or return default."""
    val = os.environ.get(key)
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    return default


def _env_int(key: str, default: int) -> int:
    """Get int from environment or return default."""
    val = os.environ.get(key)
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def _env_bool(key: str, default: bool) -> bool:
    """Get bool from environment or return default."""
    val = os.environ.get(key)
    if val:
        return val.lower() in ('true', '1', 'yes', 'on')
    return default


def _env_str(key: str, default: str) -> str:
    """Get string from environment or return default."""
    return os.environ.get(key, default)


@dataclass
class AudioConfig:
    """Audio pipeline configuration."""
    
    # VAD Settings
    vad_energy_threshold: float = field(default_factory=lambda: _env_float('VAD_ENERGY_THRESHOLD', 0.015))
    silero_threshold: float = field(default_factory=lambda: _env_float('SILERO_THRESHOLD', 0.6))
    consecutive_frames_required: int = field(default_factory=lambda: _env_int('VAD_CONSECUTIVE_FRAMES', 5))
    
    # Audio Format
    sample_rate: int = field(default_factory=lambda: _env_int('AUDIO_SAMPLE_RATE', 16000))
    chunk_size: int = field(default_factory=lambda: _env_int('AUDIO_CHUNK_SIZE', 512))
    channels: int = 1
    
    # TTS Settings
    tts_backend: str = field(default_factory=lambda: _env_str('TTS_BACKEND', 'indextts'))
    kokoro_speed: float = field(default_factory=lambda: _env_float('KOKORO_SPEED', 1.0))
    
    # VAD cooldown after TTS (seconds)
    tts_vad_cooldown: float = field(default_factory=lambda: _env_float('TTS_VAD_COOLDOWN', 3.0))
    tts_vad_buffer_multiplier: float = field(default_factory=lambda: _env_float('TTS_VAD_BUFFER_MULTIPLIER', 2.5))
    tts_vad_buffer_seconds: float = field(default_factory=lambda: _env_float('TTS_VAD_BUFFER_SECONDS', 8.0))


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    
    # Weighted retrieval scores (must sum to 1.0)
    recency_weight: float = field(default_factory=lambda: _env_float('MEMORY_RECENCY_WEIGHT', 0.2))
    relevance_weight: float = field(default_factory=lambda: _env_float('MEMORY_RELEVANCE_WEIGHT', 0.5))
    importance_weight: float = field(default_factory=lambda: _env_float('MEMORY_IMPORTANCE_WEIGHT', 0.3))
    
    # Recency decay
    recency_decay_base: float = field(default_factory=lambda: _env_float('MEMORY_RECENCY_DECAY', 0.995))
    
    # Embedding
    embedding_model: str = field(default_factory=lambda: _env_str('EMBEDDING_MODEL', 'Svenni551/Qwen3-Embedding-0.6B-ONNX-INT8'))
    embedding_dimension: int = field(default_factory=lambda: _env_int('EMBEDDING_DIMENSION', 1024))
    embedding_cache_size: int = field(default_factory=lambda: _env_int('EMBEDDING_CACHE_SIZE', 1000))
    
    # Context limits
    max_context_tokens: int = field(default_factory=lambda: _env_int('MAX_CONTEXT_TOKENS', 2000))
    summary_interval: int = field(default_factory=lambda: _env_int('SUMMARY_INTERVAL', 10))
    
    # Graph extraction
    nuextract_context_window: int = field(default_factory=lambda: _env_int('NUEXTRACT_CONTEXT_WINDOW', 4096))
    nuextract_max_input_tokens: int = field(default_factory=lambda: _env_int('NUEXTRACT_MAX_INPUT_TOKENS', 3234))


@dataclass
class ServerConfig:
    """Server and network configuration."""
    
    # Ports
    port: int = field(default_factory=lambda: _env_int('SERVER_PORT', 7861))
    character_manager_port: int = field(default_factory=lambda: _env_int('CHARACTER_MANAGER_PORT', 7863))
    mcp_manager_port: int = field(default_factory=lambda: _env_int('MCP_MANAGER_PORT', 7864))
    
    # LM Studio
    lm_studio_port: int = field(default_factory=lambda: _env_int('LM_STUDIO_PORT', 1235))
    lm_studio_url: str = field(default_factory=lambda: _env_str('LM_STUDIO_URL', ''))
    
    # Timeouts
    llm_timeout: int = field(default_factory=lambda: _env_int('LLM_TIMEOUT', 120))
    tool_timeout: int = field(default_factory=lambda: _env_int('TOOL_TIMEOUT', 30))


@dataclass
class LLMConfig:
    """LLM configuration defaults."""
    
    provider: str = field(default_factory=lambda: _env_str('LLM_PROVIDER', 'openrouter'))
    default_model: str = field(default_factory=lambda: _env_str('DEFAULT_MODEL', 'anthropic/claude-sonnet-4'))
    temperature: float = field(default_factory=lambda: _env_float('LLM_TEMPERATURE', 0.7))
    max_tokens: int = field(default_factory=lambda: _env_int('LLM_MAX_TOKENS', 2000))
    top_p: float = field(default_factory=lambda: _env_float('LLM_TOP_P', 1.0))
    frequency_penalty: float = field(default_factory=lambda: _env_float('LLM_FREQ_PENALTY', 0.0))
    presence_penalty: float = field(default_factory=lambda: _env_float('LLM_PRES_PENALTY', 0.0))


@dataclass
class TTSConfig:
    """Text-to-Speech backend configuration."""

    # Backend selection ('indextts2', 'kokoro', 'supertonic')
    backend: str = field(default_factory=lambda: _env_str('TTS_BACKEND', 'indextts2'))
    device: str = field(default_factory=lambda: _env_str('TTS_DEVICE', 'cuda'))

    # IndexTTS2 settings
    indextts_emo_alpha: float = field(default_factory=lambda: _env_float('INDEXTTS_EMO_ALPHA', 0.6))
    indextts_use_emo_text: bool = field(default_factory=lambda: _env_bool('INDEXTTS_USE_EMO_TEXT', True))

    # Kokoro settings
    kokoro_speed: float = field(default_factory=lambda: _env_float('KOKORO_SPEED', 1.0))
    kokoro_default_voice: str = field(default_factory=lambda: _env_str('KOKORO_DEFAULT_VOICE', 'af_sarah'))

    # Supertonic settings
    supertonic_voice: str = field(default_factory=lambda: _env_str('SUPERTONIC_VOICE', 'female_1'))
    supertonic_model_path: str = field(default_factory=lambda: _env_str('SUPERTONIC_MODEL_PATH', 'assets/supertonic'))


@dataclass
class STTConfig:
    """Speech-to-Text backend configuration."""

    # Backend selection ('faster_whisper', 'sensevoice', 'funasr')
    backend: str = field(default_factory=lambda: _env_str('STT_BACKEND', 'faster_whisper'))
    device: str = field(default_factory=lambda: _env_str('STT_DEVICE', 'cpu'))

    # Whisper settings
    whisper_model_size: str = field(default_factory=lambda: _env_str('WHISPER_MODEL_SIZE', 'base'))
    whisper_compute_type: str = field(default_factory=lambda: _env_str('WHISPER_COMPUTE_TYPE', 'int8'))
    whisper_language: str = field(default_factory=lambda: _env_str('WHISPER_LANGUAGE', ''))  # Empty for auto-detect

    # SenseVoice settings (unified STT+VAD+Emotion)
    sensevoice_use_vad: bool = field(default_factory=lambda: _env_bool('SENSEVOICE_USE_VAD', True))
    sensevoice_language: str = field(default_factory=lambda: _env_str('SENSEVOICE_LANGUAGE', 'auto'))

    # FunASR settings (high-accuracy STT)
    funasr_model_variant: str = field(default_factory=lambda: _env_str('FUNASR_MODEL_VARIANT', 'nano'))  # 'nano' or 'mlt'
    funasr_hotwords: str = field(default_factory=lambda: _env_str('FUNASR_HOTWORDS', ''))  # Comma-separated


@dataclass
class EmotionConfig:
    """Emotion detection and feedback configuration."""

    enabled: bool = field(default_factory=lambda: _env_bool('EMOTION_ENABLED', True))
    timeout: float = field(default_factory=lambda: _env_float('EMOTION_TIMEOUT', 2.0))

    # SER backend (future: could be swappable)
    backend: str = field(default_factory=lambda: _env_str('EMOTION_BACKEND', 'wav2vec2'))
    device: str = field(default_factory=lambda: _env_str('EMOTION_DEVICE', 'cpu'))
    
    # Emotion to TTS mapping (for future emotional feedback loop)
    # These affect TTS parameters when emotion is detected
    emotion_speed_map: Dict[str, float] = field(default_factory=lambda: {
        'calm': 1.0,
        'happy': 1.1,
        'sad': 0.9,
        'angry': 1.15,
        'fearful': 1.05,
        'surprised': 1.1,
        'disgusted': 0.95,
        'neutral': 1.0,
        'excited': 1.15,
        'tired': 0.85,
        'frustrated': 1.1,
        'annoyed': 1.05,
        'anxious': 1.1,
        'content': 0.95,
        'relaxed': 0.95,
        'pleased': 1.0,
        'upset': 1.05,
    })

    emotion_pitch_map: Dict[str, float] = field(default_factory=lambda: {
        'calm': 0.0,
        'happy': 0.1,
        'sad': -0.15,
        'angry': 0.2,
        'fearful': 0.1,
        'surprised': 0.15,
        'disgusted': -0.1,
        'neutral': 0.0,
        'excited': 0.15,
        'tired': -0.1,
        'frustrated': 0.1,
        'annoyed': 0.05,
        'anxious': 0.1,
        'content': 0.0,
        'relaxed': -0.05,
        'pleased': 0.05,
        'upset': 0.05,
    })


@dataclass
class PathsConfig:
    """File and directory paths."""
    
    # Base paths (computed at initialization)
    root_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def sessions_dir(self) -> Path:
        return self.root_dir / "sessions"
    
    @property
    def recordings_dir(self) -> Path:
        return self.root_dir / "recordings"
    
    @property
    def models_dir(self) -> Path:
        return self.root_dir / "models"

    @property
    def checkpoints_dir(self) -> Path:
        """Backward compatibility alias for models_dir."""
        return self.models_dir

    @property
    def voices_dir(self) -> Path:
        return self.root_dir / "voices"
    
    @property
    def skills_dir(self) -> Path:
        return self.root_dir / "skills"
    
    @property
    def memory_db(self) -> Path:
        return self.sessions_dir / "memory.db"


@dataclass
class Config:
    """
    Master configuration container.

    All settings centralized here, with environment variable overrides.
    """
    audio: AudioConfig = field(default_factory=AudioConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    # Per-character overrides (character_id -> config overrides)
    character_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def get_for_character(self, character_id: str, key: str, default: Any = None) -> Any:
        """
        Get a config value with per-character override support.
        
        Example:
            speed = config.get_for_character('hermione', 'audio.kokoro_speed', 1.0)
        """
        overrides = self.character_overrides.get(character_id, {})
        return overrides.get(key, default)
    
    def set_character_override(self, character_id: str, key: str, value: Any):
        """Set a per-character config override."""
        if character_id not in self.character_overrides:
            self.character_overrides[character_id] = {}
        self.character_overrides[character_id][key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config to dictionary for debugging/display."""
        return {
            'audio': {
                'vad_energy_threshold': self.audio.vad_energy_threshold,
                'silero_threshold': self.audio.silero_threshold,
                'consecutive_frames_required': self.audio.consecutive_frames_required,
                'sample_rate': self.audio.sample_rate,
                'tts_backend': self.audio.tts_backend,
            },
            'memory': {
                'recency_weight': self.memory.recency_weight,
                'relevance_weight': self.memory.relevance_weight,
                'importance_weight': self.memory.importance_weight,
                'embedding_model': self.memory.embedding_model,
            },
            'server': {
                'port': self.server.port,
                'lm_studio_port': self.server.lm_studio_port,
                'llm_timeout': self.server.llm_timeout,
            },
            'llm': {
                'provider': self.llm.provider,
                'default_model': self.llm.default_model,
                'temperature': self.llm.temperature,
            },
            'tts': {
                'backend': self.tts.backend,
                'device': self.tts.device,
                'kokoro_speed': self.tts.kokoro_speed,
                'supertonic_voice': self.tts.supertonic_voice,
            },
            'stt': {
                'backend': self.stt.backend,
                'device': self.stt.device,
                'whisper_model_size': self.stt.whisper_model_size,
                'sensevoice_use_vad': self.stt.sensevoice_use_vad,
                'funasr_model_variant': self.stt.funasr_model_variant,
            },
            'emotion': {
                'enabled': self.emotion.enabled,
                'backend': self.emotion.backend,
            }
        }


# Singleton instance - import this in other modules
config = Config()


# Convenience exports for backwards compatibility
# These allow gradual migration from hardcoded values
def get_vad_energy_threshold() -> float:
    return config.audio.vad_energy_threshold

def get_silero_threshold() -> float:
    return config.audio.silero_threshold

def get_memory_weights() -> tuple:
    return (
        config.memory.recency_weight,
        config.memory.relevance_weight,
        config.memory.importance_weight
    )

def get_server_port() -> int:
    return config.server.port


if __name__ == "__main__":
    # Test config loading
    print("Config loaded successfully:")
    print(f"  VAD Energy Threshold: {config.audio.vad_energy_threshold}")
    print(f"  Silero Threshold: {config.audio.silero_threshold}")
    print(f"  Memory Weights: {get_memory_weights()}")
    print(f"  Server Port: {config.server.port}")
    print(f"  Sessions Dir: {config.paths.sessions_dir}")
