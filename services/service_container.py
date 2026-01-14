"""
Service Container for Dependency Injection

Wraps global services (MEMORY_MANAGER, CHARACTER_MANAGER, TTS_MODEL, etc.)
to enable dependency injection without removing the globals.

This is a transitional pattern - Phase 1 wraps globals, Phase 2 will replace them.
"""

from typing import Optional, Any


class ServiceContainer:
    """
    Singleton container that wraps global services for dependency injection.
    
    Usage:
        # After globals are initialized (voice_chat_app.py ~line 1600)
        container = ServiceContainer.get_instance()
        container.initialize(
            memory_manager=MEMORY_MANAGER,
            character_manager=CHARACTER_MANAGER,
            tts_model=TTS_MODEL,
            settings=SETTINGS
        )
        
        # Later, in UI callbacks:
        chat_service = container.get_chat_service()
        result = chat_service.process_message(...)
    """
    
    _instance: Optional['ServiceContainer'] = None
    
    def __init__(self):
        """Private constructor - use get_instance() instead."""
        self._memory_manager = None
        self._character_manager = None
        self._tts_model = None
        self._whisper_model = None
        self._settings = None
        self._vad_manager = None
        self._initialized = False
    
    @classmethod
    def get_instance(cls) -> 'ServiceContainer':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton (useful for testing)."""
        cls._instance = None
    
    def initialize(
        self,
        memory_manager: Any,
        character_manager: Any,
        tts_model: Any,
        settings: dict,
        whisper_model: Any = None,
        vad_manager: Any = None
    ):
        """
        Initialize the container with service instances.
        
        Args:
            memory_manager: MultiCharacterMemoryManager instance
            character_manager: CharacterManager instance
            tts_model: TTS model instance (IndexTTS2 or Kokoro)
            settings: Settings dictionary
            whisper_model: Optional Whisper model for STT
            vad_manager: Optional VAD manager instance
        """
        self._memory_manager = memory_manager
        self._character_manager = character_manager
        self._tts_model = tts_model
        self._settings = settings
        self._whisper_model = whisper_model
        self._vad_manager = vad_manager
        self._initialized = True
        
        print("[ServiceContainer] Initialized with all services")
    
    @property
    def memory_manager(self):
        """Get memory manager instance."""
        if not self._initialized:
            raise RuntimeError("ServiceContainer not initialized. Call initialize() first.")
        return self._memory_manager
    
    @property
    def character_manager(self):
        """Get character manager instance."""
        if not self._initialized:
            raise RuntimeError("ServiceContainer not initialized. Call initialize() first.")
        return self._character_manager
    
    @property
    def tts_model(self):
        """Get TTS model instance."""
        if not self._initialized:
            raise RuntimeError("ServiceContainer not initialized. Call initialize() first.")
        return self._tts_model
    
    @property
    def settings(self):
        """Get settings dictionary."""
        if not self._initialized:
            raise RuntimeError("ServiceContainer not initialized. Call initialize() first.")
        return self._settings
    
    @property
    def whisper_model(self):
        """Get Whisper model instance (may be None)."""
        return self._whisper_model
    
    @property
    def vad_manager(self):
        """Get VAD manager instance (may be None)."""
        return self._vad_manager
    
    def get_chat_service(self):
        """
        Create a ChatService instance with injected dependencies.
        
        Returns:
            ChatService instance ready to process messages
        """
        from .chat_service import ChatService
        
        return ChatService(
            memory_manager=self._memory_manager,
            character_manager=self._character_manager,
            tts_model=self._tts_model,
            settings=self._settings
        )
    
    def is_initialized(self) -> bool:
        """Check if container has been initialized."""
        return self._initialized
