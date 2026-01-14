"""
Service Layer for IndexTTS2 Voice Chat

Separates business logic from UI layer for better testability and maintainability.
"""

from .chat_service import ChatService, ChatResult, ToolCallInfo
from .service_container import ServiceContainer

__all__ = [
    'ChatService',
    'ChatResult',
    'ToolCallInfo',
    'ServiceContainer',
]
