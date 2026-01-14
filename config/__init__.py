"""
Configuration module for IndexTTS2 Voice Agent.

Usage:
    from config import config
    
    # Access any setting
    threshold = config.audio.vad_energy_threshold
    port = config.server.port
"""

from .manager import (
    config,
    Config,
    AudioConfig,
    MemoryConfig,
    ServerConfig,
    LLMConfig,
    EmotionConfig,
    PathsConfig,
    # Backwards compatibility helpers
    get_vad_energy_threshold,
    get_silero_threshold,
    get_memory_weights,
    get_server_port,
)

__all__ = [
    'config',
    'Config',
    'AudioConfig',
    'MemoryConfig', 
    'ServerConfig',
    'LLMConfig',
    'EmotionConfig',
    'PathsConfig',
    'get_vad_energy_threshold',
    'get_silero_threshold',
    'get_memory_weights',
    'get_server_port',
]
