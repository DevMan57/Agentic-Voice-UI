"""
UI Formatting Layer for IndexTTS2 Voice Chat

Handles conversion of service layer data to UI-ready formats.
"""

from .formatters import (
    format_memory_for_ui,
    format_tool_call_html,
    format_message_with_extras,
    format_memory_recall_html,
    filter_for_speech,
    format_chat_result_for_ui,
)

__all__ = [
    'format_memory_for_ui',
    'format_tool_call_html',
    'format_message_with_extras',
    'format_memory_recall_html',
    'filter_for_speech',
    'format_chat_result_for_ui',
]
