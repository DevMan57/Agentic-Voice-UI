"""
Enhanced Utilities for IndexTTS2 Voice Chat

Includes:
- Dark theme for Gradio UI
- Keyboard shortcut handling
- Conversation search and export
- Audio visualization helpers
- Message formatting utilities
- Settings management
"""

import gradio as gr
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib

# ============================================================================
# Dark Theme
# ============================================================================

def create_dark_theme():
    """Create Ghost White theme - Clean Monochrome"""
    return gr.themes.Soft(
        primary_hue="gray",
        secondary_hue="gray",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Fira Code"),
    ).set(
        # Base colors - Pure black
        body_background_fill="#000000",
        body_background_fill_dark="#000000",
        background_fill_primary="#000000",
        background_fill_primary_dark="#000000",
        background_fill_secondary="#000000",
        background_fill_secondary_dark="#000000",

        # Block colors - Black with white borders
        block_background_fill="#000000",
        block_background_fill_dark="#000000",
        block_border_color="#FFFFFF",
        block_border_color_dark="#FFFFFF",
        block_label_background_fill="#000000",
        block_label_background_fill_dark="#000000",
        block_title_text_color="#FFFFFF",
        block_title_text_color_dark="#FFFFFF",

        # Input colors - Black
        input_background_fill="#000000",
        input_background_fill_dark="#000000",
        input_border_color="#FFFFFF",
        input_border_color_dark="#FFFFFF",
        input_placeholder_color="#B3B3B3",
        input_placeholder_color_dark="#B3B3B3",

        # Button colors - Ghost White
        button_primary_background_fill="#FFFFFF",
        button_primary_background_fill_dark="#FFFFFF",
        button_primary_background_fill_hover="#E6E6E6",
        button_primary_background_fill_hover_dark="#E6E6E6",
        button_primary_text_color="#000000",
        button_primary_text_color_dark="#000000",
        # Secondary buttons
        button_secondary_background_fill="#000000",
        button_secondary_background_fill_dark="#000000",
        button_secondary_background_fill_hover="#FFFFFF",
        button_secondary_background_fill_hover_dark="#FFFFFF",
        button_secondary_text_color="#FFFFFF",
        button_secondary_text_color_dark="#FFFFFF",

        # Slider colors - Ghost White
        slider_color="#FFFFFF",
        slider_color_dark="#FFFFFF",

        # Checkbox colors - Ghost White
        checkbox_background_color_selected="#FFFFFF",
        checkbox_background_color_selected_dark="#FFFFFF",
        checkbox_border_color_selected="#FFFFFF",
        checkbox_border_color_selected_dark="#FFFFFF",

        # Text colors - Ghost White
        body_text_color="#FFFFFF",
        body_text_color_dark="#FFFFFF",
        body_text_color_subdued="#B3B3B3",
        body_text_color_subdued_dark="#B3B3B3",

        # Border radius - CYBERDECK: Sharp corners
        block_radius="0px",
        container_radius="0px",
        input_radius="0px",
        button_large_radius="0px",
        button_small_radius="0px",
        checkbox_border_radius="0px",
        table_radius="0px",

        # Spacing
        block_padding="16px",
    )


# ============================================================================
# Keyboard Shortcuts
# ============================================================================

KEYBOARD_SHORTCUTS_JS = """
<script>
(function() {
    // Keyboard shortcut handler
    document.addEventListener('keydown', function(e) {
        // Ctrl+Enter to send message
        if (e.ctrlKey && e.key === 'Enter') {
            const sendBtn = document.querySelector('#send-btn');
            if (sendBtn) {
                sendBtn.click();
                e.preventDefault();
            }
        }
        
        // Ctrl+N for new conversation
        if (e.ctrlKey && e.key === 'n') {
            const newBtn = document.querySelector('#new-conv-btn');
            if (newBtn) {
                newBtn.click();
                e.preventDefault();
            }
        }
        
        // Escape to clear input
        if (e.key === 'Escape') {
            const input = document.querySelector('#user-input textarea');
            if (input && document.activeElement === input) {
                input.value = '';
                input.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }
        
        // Ctrl+/ to show shortcuts help
        if (e.ctrlKey && e.key === '/') {
            const helpModal = document.querySelector('#shortcuts-modal');
            if (helpModal) {
                helpModal.style.display = helpModal.style.display === 'none' ? 'block' : 'none';
                e.preventDefault();
            }
        }
    });
    
    console.log('[VoiceChat] Keyboard shortcuts loaded');
})();
</script>
"""

KEYBOARD_SHORTCUTS_HTML = """
<div id="shortcuts-modal" style="display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
     background: #000000; padding: 24px; border-radius: 0px; border: 2px solid var(--theme-primary, #FFFFFF); z-index: 1000; box-shadow: 0 0 30px var(--theme-glow, rgba(255,255,255,0.3));">
    <h3 style="margin-top: 0; color: var(--theme-primary, #FFFFFF);">⌨️ Keyboard Shortcuts</h3>
    <table style="color: var(--theme-primary, #FFFFFF); border-collapse: collapse;">
        <tr><td style="padding: 8px;"><kbd style="background: #000000; border: 1px solid var(--theme-primary, #FFFFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #FFFFFF);">Shift</kbd></td>
            <td style="padding: 8px;">Hold to record (Push-to-Talk)</td></tr>
        <tr><td style="padding: 8px;"><kbd style="background: #000000; border: 1px solid var(--theme-primary, #FFFFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #FFFFFF);">Ctrl</kbd>+<kbd style="background: #000000; border: 1px solid var(--theme-primary, #FFFFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #FFFFFF);">Enter</kbd></td>
            <td style="padding: 8px;">Send message</td></tr>
        <tr><td style="padding: 8px;"><kbd style="background: #000000; border: 1px solid var(--theme-primary, #FFFFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #FFFFFF);">Ctrl</kbd>+<kbd style="background: #000000; border: 1px solid var(--theme-primary, #FFFFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #FFFFFF);">N</kbd></td>
            <td style="padding: 8px;">New conversation</td></tr>
        <tr><td style="padding: 8px;"><kbd style="background: #000000; border: 1px solid var(--theme-primary, #FFFFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #FFFFFF);">Esc</kbd></td>
            <td style="padding: 8px;">Clear input</td></tr>
        <tr><td style="padding: 8px;"><kbd style="background: #000000; border: 1px solid var(--theme-primary, #FFFFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #FFFFFF);">Ctrl</kbd>+<kbd style="background: #000000; border: 1px solid var(--theme-primary, #FFFFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #FFFFFF);">/</kbd></td>
            <td style="padding: 8px;">Toggle this help</td></tr>
    </table>
    <button onclick="this.parentElement.style.display='none'"
            style="margin-top: 16px; padding: 8px 16px; background: var(--theme-primary, #FFFFFF); color: #000000; border: none; border-radius: 0px; cursor: pointer;">
        Close
    </button>
</div>
"""


# ============================================================================
# Audio Visualization (Web Audio API)
# ============================================================================

AUDIO_VISUALIZER_JS = """
<script>
(function() {
    // Audio visualizer for recording feedback
    class AudioVisualizer {
        constructor(canvasId) {
            this.canvas = document.getElementById(canvasId);
            if (!this.canvas) return;
            
            this.ctx = this.canvas.getContext('2d');
            this.isActive = false;
        }
        
        start() {
            this.isActive = true;
            this.animate();
        }
        
        stop() {
            this.isActive = false;
        }
        
        animate() {
            if (!this.isActive || !this.ctx) return;
            
            const width = this.canvas.width;
            const height = this.canvas.height;
            
            // Clear
            this.ctx.fillStyle = '#000000';
            this.ctx.fillRect(0, 0, width, height);

            // Draw fake waveform (would connect to real audio data)
            this.ctx.strokeStyle = '#FFFFFF';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            
            const sliceWidth = width / 50;
            let x = 0;
            
            for (let i = 0; i < 50; i++) {
                const v = Math.sin(Date.now() / 100 + i * 0.3) * 0.5 + 0.5;
                const y = v * height;
                
                if (i === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
                x += sliceWidth;
            }
            
            this.ctx.stroke();
            
            requestAnimationFrame(() => this.animate());
        }
    }
    
    window.AudioVisualizer = AudioVisualizer;
})();
</script>
"""


# ============================================================================
# Message Formatting
# ============================================================================

def format_message_html(role: str, content: str, timestamp: str = None, character_name: str = None) -> str:
    """Format a chat message as styled HTML"""

    if role == "user":
        bg_color = "#000000"
        border_color = "var(--theme-primary, #FFFFFF)"
        icon = "👤"
        name = "You"
        align = "right"
    else:
        bg_color = "#000000"
        border_color = "var(--theme-dim, #B3B3B3)"
        icon = "🤖"
        name = character_name or "Assistant"
        align = "left"

    time_str = ""
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = f'<span style="font-size: 0.75em; color: var(--theme-dim, #B3B3B3); margin-left: 8px;">{dt.strftime("%H:%M")}</span>'
        except:
            pass

    # Escape HTML but preserve basic markdown
    content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
    content = content.replace("\n", "<br>")

    return f'''
    <div style="display: flex; justify-content: flex-{align.replace('right', 'end').replace('left', 'start')}; margin: 8px 0;">
        <div style="background: {bg_color}; border: 1px solid {border_color}; padding: 12px 16px; border-radius: 0px; max-width: 80%;">
            <div style="font-size: 0.85em; color: var(--theme-dim, #B3B3B3); margin-bottom: 4px;">
                {icon} {name}{time_str}
            </div>
            <div style="color: var(--theme-primary, #FFFFFF); line-height: 1.5;">
                {content}
            </div>
        </div>
    </div>
    '''


def format_chat_history_html(history: List[Dict[str, str]], character_name: str = "Assistant") -> str:
    """Format entire chat history as HTML"""
    if not history:
        return '<div style="color: var(--theme-dim, #B3B3B3); text-align: center; padding: 40px;">Start a conversation...</div>'
    
    html_parts = []
    for msg in history:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        timestamp = msg.get('timestamp')
        
        if role in ['user', 'assistant']:
            html_parts.append(format_message_html(
                role=role,
                content=content,
                timestamp=timestamp,
                character_name=character_name if role == 'assistant' else None
            ))
    
    return '\n'.join(html_parts)


def format_typing_indicator(character_name: str = "Assistant") -> str:
    """Create a typing indicator HTML"""
    return f'''
    <div style="display: flex; align-items: center; padding: 12px; color: var(--theme-primary, #FFFFFF);">
        <span style="margin-right: 8px;">🤖</span>
        <span>{character_name} is thinking</span>
        <span class="typing-dots" style="margin-left: 4px;">
            <span style="animation: blink 1.4s infinite;">.</span>
            <span style="animation: blink 1.4s infinite 0.2s;">.</span>
            <span style="animation: blink 1.4s infinite 0.4s;">.</span>
        </span>
    </div>
    <style>
        @keyframes blink {{ 0%, 20% {{ opacity: 0; }} 50% {{ opacity: 1; }} 100% {{ opacity: 0; }} }}
    </style>
    '''


# ============================================================================
# Conversation Search & Export
# ============================================================================

@dataclass
class ConversationSearchResult:
    """Search result for conversation lookup"""
    conversation_id: str
    character_id: str
    title: str
    snippet: str
    timestamp: str
    message_count: int
    match_score: float = 0.0


def search_conversations(
    conversations_dir: Path,
    query: str,
    character_id: str = None,
    limit: int = 20
) -> List[ConversationSearchResult]:
    """
    Search through saved conversations using keyword matching.
    Returns list of matching conversations with snippets.
    """
    results = []
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Iterate through character directories
    for char_dir in conversations_dir.iterdir():
        if not char_dir.is_dir():
            continue
        
        if character_id and char_dir.name != character_id:
            continue
        
        # Search conversation files
        for conv_file in char_dir.glob("*.json"):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract text content for searching
                history = data.get('history', [])
                all_text = ""
                for msg in history:
                    if isinstance(msg, dict):
                        all_text += " " + msg.get('content', '')
                    elif isinstance(msg, (list, tuple)):
                        all_text += " " + str(msg[0] or '') + " " + str(msg[1] or '')
                
                all_text_lower = all_text.lower()
                
                # Calculate match score
                matches = sum(1 for word in query_words if word in all_text_lower)
                if matches == 0:
                    continue
                
                score = matches / len(query_words) if query_words else 0
                
                # Find snippet containing query
                snippet = ""
                idx = all_text_lower.find(query_lower)
                if idx >= 0:
                    start = max(0, idx - 40)
                    end = min(len(all_text), idx + len(query) + 100)
                    snippet = "..." + all_text[start:end].strip() + "..."
                else:
                    # Use first query word match
                    for word in query_words:
                        idx = all_text_lower.find(word)
                        if idx >= 0:
                            start = max(0, idx - 40)
                            end = min(len(all_text), idx + 100)
                            snippet = "..." + all_text[start:end].strip() + "..."
                            break
                
                results.append(ConversationSearchResult(
                    conversation_id=conv_file.stem,
                    character_id=char_dir.name,
                    title=data.get('title', 'Untitled'),
                    snippet=snippet,
                    timestamp=data.get('updated_at', ''),
                    message_count=len(history),
                    match_score=score
                ))
                
            except Exception as e:
                print(f"[Search] Error reading {conv_file}: {e}")
                continue
    
    # Sort by score and recency
    results.sort(key=lambda x: (x.match_score, x.timestamp), reverse=True)
    return results[:limit]


def export_conversation_markdown(
    conversation: Dict[str, Any],
    character_name: str = "Assistant"
) -> str:
    """Export a conversation as formatted Markdown"""
    lines = []
    
    # Header
    lines.append(f"# {conversation.get('title', 'Conversation')}")
    lines.append("")
    lines.append(f"**Character:** {character_name}")
    lines.append(f"**Date:** {conversation.get('created_at', 'Unknown')}")
    lines.append(f"**Messages:** {len(conversation.get('history', []))}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Messages
    for msg in conversation.get('history', []):
        if isinstance(msg, dict):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
        elif isinstance(msg, (list, tuple)):
            # Legacy format
            if msg[0]:
                lines.append(f"**You:** {msg[0]}")
                lines.append("")
            if msg[1]:
                lines.append(f"**{character_name}:** {msg[1]}")
                lines.append("")
            continue
        else:
            continue
        
        if role == 'user':
            lines.append(f"**You:** {content}")
        elif role == 'assistant':
            lines.append(f"**{character_name}:** {content}")
        lines.append("")
    
    return "\n".join(lines)


def export_conversation_json(conversation: Dict[str, Any]) -> str:
    """Export a conversation as formatted JSON"""
    return json.dumps(conversation, indent=2, ensure_ascii=False)


# ============================================================================
# Settings Management
# ============================================================================

@dataclass
class AppSettings:
    """Application settings with defaults"""
    # LLM Settings
    model: str = "x-ai/grok-4.1-fast"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    llm_provider: str = "openrouter"
    
    # Character Settings
    last_character: str = "hermione"
    last_voice: str = "reference.wav"
    current_conversation_id: Optional[str] = None
    
    # Audio Settings
    auto_play: bool = True
    tts_enabled: bool = True
    tts_backend: str = "indextts"
    sample_rate: int = 16000
    vad_enabled: bool = False
    vad_threshold: float = 0.5
    
    # UI Settings
    show_timestamps: bool = True
    show_typing_indicator: bool = True
    theme: str = "dark"
    compact_mode: bool = False
    
    # Memory Settings
    memory_enabled: bool = True
    memory_context_size: int = 10
    auto_summarize: bool = True
    
    # Advanced
    debug_mode: bool = False
    log_conversations: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'llm_provider': self.llm_provider,
            'last_character': self.last_character,
            'last_voice': self.last_voice,
            'current_conversation_id': self.current_conversation_id,
            'auto_play': self.auto_play,
            'tts_enabled': self.tts_enabled,
            'tts_backend': self.tts_backend,
            'sample_rate': self.sample_rate,
            'vad_enabled': self.vad_enabled,
            'vad_threshold': self.vad_threshold,
            'show_timestamps': self.show_timestamps,
            'show_typing_indicator': self.show_typing_indicator,
            'theme': self.theme,
            'compact_mode': self.compact_mode,
            'memory_enabled': self.memory_enabled,
            'memory_context_size': self.memory_context_size,
            'auto_summarize': self.auto_summarize,
            'debug_mode': self.debug_mode,
            'log_conversations': self.log_conversations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppSettings':
        return cls(
            model=data.get('model', "x-ai/grok-4.1-fast"),
            temperature=data.get('temperature', 0.7),
            max_tokens=data.get('max_tokens', 2000),
            top_p=data.get('top_p', 1.0),
            frequency_penalty=data.get('frequency_penalty', 0.0),
            presence_penalty=data.get('presence_penalty', 0.0),
            llm_provider=data.get('llm_provider', 'openrouter'),
            last_character=data.get('last_character', 'hermione'),
            last_voice=data.get('last_voice', 'reference.wav'),
            current_conversation_id=data.get('current_conversation_id'),
            auto_play=data.get('auto_play', True),
            tts_enabled=data.get('tts_enabled', True),
            tts_backend=data.get('tts_backend', "indextts"),
            sample_rate=data.get('sample_rate', 16000),
            vad_enabled=data.get('vad_enabled', False),
            vad_threshold=data.get('vad_threshold', 0.5),
            show_timestamps=data.get('show_timestamps', True),
            show_typing_indicator=data.get('show_typing_indicator', True),
            theme=data.get('theme', 'dark'),
            compact_mode=data.get('compact_mode', False),
            memory_enabled=data.get('memory_enabled', True),
            memory_context_size=data.get('memory_context_size', 10),
            auto_summarize=data.get('auto_summarize', True),
            debug_mode=data.get('debug_mode', False),
            log_conversations=data.get('log_conversations', True)
        )


class SettingsManager:
    """Manages loading and saving application settings"""
    
    def __init__(self, settings_path: Path):
        self.settings_path = settings_path
        self.settings = self.load()
    
    def load(self) -> AppSettings:
        """Load settings from file"""
        if self.settings_path.exists():
            try:
                with open(self.settings_path, 'r') as f:
                    data = json.load(f)
                return AppSettings.from_dict(data)
            except Exception as e:
                print(f"[Settings] Error loading: {e}")
        return AppSettings()
    
    def save(self):
        """Save current settings to file"""
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(self.settings.to_dict(), f, indent=2)
        except Exception as e:
            print(f"[Settings] Error saving: {e}")
    
    def update(self, **kwargs):
        """Update settings with new values"""
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        self.save()
    
    def reset(self):
        """Reset to default settings"""
        self.settings = AppSettings()
        self.save()


# ============================================================================
# Status Display Helpers
# ============================================================================

def create_status_html(
    ptt_status: str = "ready",
    recording_duration: float = 0.0,
    character_name: str = "",
    model_name: str = "",
    is_thinking: bool = False
) -> str:
    """Create status bar HTML"""

    # PTT indicator - All lapis lazuli
    if ptt_status == "recording":
        ptt_html = f'<span style="color: var(--theme-primary, #FFFFFF);">🔴 Recording ({recording_duration:.1f}s)</span>'
    elif ptt_status == "processing":
        ptt_html = '<span style="color: var(--theme-primary, #FFFFFF);">⏳ Processing...</span>'
    elif ptt_status == "offline":
        ptt_html = '<span style="color: var(--theme-dim, #B3B3B3);">⭕ PTT Offline</span>'
    else:
        ptt_html = '<span style="color: var(--theme-primary, #FFFFFF);">🟢 Ready (Hold Shift)</span>'

    # Character
    char_html = f'<span style="color: var(--theme-primary, #FFFFFF);">👤 {character_name}</span>' if character_name else ''

    # Model
    model_html = f'<span style="color: var(--theme-dim, #B3B3B3);">🤖 {model_name[:30]}...</span>' if len(model_name) > 30 else f'<span style="color: var(--theme-dim, #B3B3B3);">🤖 {model_name}</span>' if model_name else ''

    # Thinking
    thinking_html = '<span style="color: var(--theme-primary, #FFFFFF);">💭 Thinking...</span>' if is_thinking else ''

    parts = [p for p in [ptt_html, char_html, model_html, thinking_html] if p]

    return f'''
    <div style="display: flex; justify-content: space-between; align-items: center;
                padding: 8px 16px; background: #000000; border: 1px solid var(--theme-dim, #B3B3B3); border-radius: 0px; font-size: 0.9em;">
        {' | '.join(parts)}
    </div>
    '''


def create_memory_stats_html(stats: Dict[str, Any]) -> str:
    """Create memory statistics display HTML"""
    return f'''
    <div style="background: #000000; padding: 16px; border: 1px solid var(--theme-primary, #FFFFFF); border-radius: 0px; color: var(--theme-primary, #FFFFFF);">
        <h4 style="margin-top: 0; color: var(--theme-primary, #FFFFFF);">🧠 Memory Statistics</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 4px 8px;">Episodic memories:</td>
                <td style="padding: 4px 8px; text-align: right;">{stats.get('episodic_count', 0)}</td>
            </tr>
            <tr>
                <td style="padding: 4px 8px;">Semantic memories:</td>
                <td style="padding: 4px 8px; text-align: right;">{stats.get('semantic_count', 0)}</td>
            </tr>
            <tr>
                <td style="padding: 4px 8px;">Procedural memories:</td>
                <td style="padding: 4px 8px; text-align: right;">{stats.get('procedural_count', 0)}</td>
            </tr>
            <tr>
                <td style="padding: 4px 8px;">Session summaries:</td>
                <td style="padding: 4px 8px; text-align: right;">{stats.get('summary_count', 0)}</td>
            </tr>
            <tr style="border-top: 1px solid var(--theme-dim, #B3B3B3);">
                <td style="padding: 8px 8px 4px;">Total interactions:</td>
                <td style="padding: 8px 8px 4px; text-align: right; font-weight: bold;">{stats.get('total_interactions', 0)}</td>
            </tr>
        </table>
        <div style="margin-top: 12px; font-size: 0.85em; color: var(--theme-dim, #B3B3B3);">
            Embedding model: {stats.get('embedding_model', 'N/A')}<br>
            Retrieval weights: R={stats.get('retrieval_weights', {}).get('recency', 0.2)}
                              V={stats.get('retrieval_weights', {}).get('relevance', 0.5)}
                              I={stats.get('retrieval_weights', {}).get('importance', 0.3)}
        </div>
    </div>
    '''


# ============================================================================
# Character Card HTML
# ============================================================================

def create_character_card_html(
    character_id: str,
    name: str,
    display_name: str,
    description: str = "",
    tags: List[str] = None,
    is_selected: bool = False
) -> str:
    """Create a character selection card"""

    border_color = "var(--theme-primary, #FFFFFF)" if is_selected else "var(--theme-dim, #B3B3B3)"
    bg_color = "#000000"

    tags_html = ""
    if tags:
        tags_html = '<div style="margin-top: 8px;">' + ' '.join(
            f'<span style="background: #000000; border: 1px solid var(--theme-dim, #B3B3B3); padding: 2px 8px; border-radius: 0px; font-size: 0.75em; color: var(--theme-primary, #FFFFFF);">{tag}</span>'
            for tag in tags[:3]
        ) + '</div>'

    return f'''
    <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 12px;
                padding: 16px; cursor: pointer; transition: all 0.2s;"
         class="character-card" data-id="{character_id}">
        <div style="font-size: 1.2em; margin-bottom: 4px;">{display_name}</div>
        <div style="color: #888; font-size: 0.9em;">{description[:100]}...</div>
        {tags_html}
    </div>
    '''


# ============================================================================
# Audio Helpers
# ============================================================================

def get_audio_duration(filepath: str) -> float:
    """Get duration of an audio file in seconds"""
    try:
        import wave
        with wave.open(filepath, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except:
        return 0.0


def normalize_audio_level(filepath: str) -> Optional[str]:
    """Normalize audio levels and return path to normalized file"""
    try:
        import numpy as np
        from scipy.io import wavfile
        
        rate, data = wavfile.read(filepath)
        
        # Convert to float
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        
        # Normalize to 0.9 peak
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data * (0.9 / peak)
        
        # Convert back to int16
        data = (data * 32767).astype(np.int16)
        
        # Save to temp file
        output_path = filepath.replace('.wav', '_normalized.wav')
        wavfile.write(output_path, rate, data)
        
        return output_path
        
    except Exception as e:
        print(f"[Audio] Normalization failed: {e}")
        return None


# ============================================================================
# Model Lists
# ============================================================================

POPULAR_MODELS = [
    # Anthropic
    ("claude-sonnet-4-20250514", "Claude Sonnet 4 (Latest)"),
    ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet"),
    ("anthropic/claude-3-opus", "Claude 3 Opus"),
    
    # OpenAI
    ("openai/gpt-4o", "GPT-4o"),
    ("openai/gpt-4o-mini", "GPT-4o Mini"),
    ("openai/gpt-4-turbo", "GPT-4 Turbo"),
    
    # Google
    ("google/gemini-2.5-pro-preview", "Gemini 2.5 Pro"),
    ("google/gemini-2.0-flash-001", "Gemini 2.0 Flash"),
    
    # X.AI
    ("x-ai/grok-4.1-fast", "Grok 4.1 Fast"),
    ("x-ai/grok-3-beta", "Grok 3 Beta"),
    
    # Meta
    ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B"),
    ("meta-llama/llama-3.1-405b-instruct", "Llama 3.1 405B"),
    
    # DeepSeek
    ("deepseek/deepseek-chat-v3-0324", "DeepSeek V3"),
    ("deepseek/deepseek-r1", "DeepSeek R1"),
    
    # Qwen
    ("qwen/qwen3-235b-a22b", "Qwen 3 235B"),
    ("qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B"),
    
    # Open Source
    ("mistralai/mistral-large-2411", "Mistral Large"),
    ("nvidia/llama-3.1-nemotron-70b-instruct", "Nemotron 70B"),
]

VISION_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini", 
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "google/gemini-2.5-pro-preview",
    "google/gemini-2.0-flash-001",
    "x-ai/grok-2-vision-1212",
]


def get_model_display_name(model_id: str) -> str:
    """Get human-readable name for a model ID"""
    for mid, name in POPULAR_MODELS:
        if mid == model_id:
            return name
    # Fallback: clean up the ID
    return model_id.split("/")[-1].replace("-", " ").title()
