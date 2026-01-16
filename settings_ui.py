#!/usr/bin/env python3
"""
Settings UI Component for IndexTTS2 Voice Chat

Provides a Gradio-based settings panel that can be integrated
into the main application or run standalone.
"""

import gradio as gr
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Any
import json

from utils import AppSettings, SettingsManager, POPULAR_MODELS, create_dark_theme


def create_settings_panel(
    settings_manager: SettingsManager,
    on_save: Callable[[AppSettings], None] = None,
    voice_references: List[str] = None
) -> Tuple[gr.Blocks, Callable]:
    """
    Create a settings panel component.
    
    Returns:
        (gr.Blocks, refresh_callback) - The panel and a function to refresh it
    """
    
    voice_refs = voice_references or ["reference.wav"]
    
    with gr.Blocks() as panel:
        gr.Markdown("## ⚙️ Settings")
        
        with gr.Tabs():
            # ==================== LLM Settings ====================
            with gr.Tab("🤖 LLM"):
                with gr.Row():
                    with gr.Column(scale=2):
                        model_dropdown = gr.Dropdown(
                            choices=[m[0] for m in POPULAR_MODELS],
                            value=settings_manager.settings.model,
                            label="Model",
                            info="Select the LLM model to use"
                        )
                    with gr.Column(scale=1):
                        llm_provider = gr.Radio(
                            choices=["openrouter", "lmstudio", "openai"],
                            value=settings_manager.settings.llm_provider,
                            label="Provider"
                        )
                
                with gr.Row():
                    temperature = gr.Slider(
                        0, 2, value=settings_manager.settings.temperature,
                        step=0.1, label="Temperature",
                        info="Higher = more creative, Lower = more focused"
                    )
                    max_tokens = gr.Slider(
                        100, 8000, value=settings_manager.settings.max_tokens,
                        step=100, label="Max Tokens"
                    )
                
                with gr.Row():
                    top_p = gr.Slider(
                        0, 1, value=settings_manager.settings.top_p,
                        step=0.05, label="Top P",
                        info="Nucleus sampling threshold"
                    )
                    freq_penalty = gr.Slider(
                        -2, 2, value=settings_manager.settings.frequency_penalty,
                        step=0.1, label="Frequency Penalty"
                    )
                    pres_penalty = gr.Slider(
                        -2, 2, value=settings_manager.settings.presence_penalty,
                        step=0.1, label="Presence Penalty"
                    )
            
            # ==================== Audio Settings ====================
            with gr.Tab("🎤 Audio"):
                with gr.Row():
                    auto_play = gr.Checkbox(
                        value=settings_manager.settings.auto_play,
                        label="Auto-play audio",
                        info="Automatically play TTS responses"
                    )
                    tts_enabled = gr.Checkbox(
                        value=settings_manager.settings.tts_enabled,
                        label="Enable TTS",
                        info="Enable text-to-speech"
                    )
                
                tts_backend = gr.Radio(
                    choices=["indextts", "kokoro"],
                    value=settings_manager.settings.tts_backend,
                    label="TTS Backend",
                    info="Select speech synthesis engine (IndexTTS2: High Quality/GPU, Kokoro: Fast/Low VRAM)"
                )
                
                with gr.Row():
                    vad_enabled = gr.Checkbox(
                        value=settings_manager.settings.vad_enabled,
                        label="Voice Activity Detection (VAD)",
                        info="Hands-free recording mode"
                    )
                    vad_threshold = gr.Slider(
                        0.3, 1.5, value=settings_manager.settings.vad_threshold,
                        step=0.1, label="VAD Silence Threshold (seconds)",
                        info="Silence duration before stopping recording"
                    )
                
                sample_rate = gr.Dropdown(
                    choices=[8000, 16000, 22050, 44100, 48000],
                    value=settings_manager.settings.sample_rate,
                    label="Sample Rate",
                    info="Audio sample rate for recording"
                )
            
            # ==================== Memory Settings ====================
            with gr.Tab("🧠 Memory"):
                memory_enabled = gr.Checkbox(
                    value=settings_manager.settings.memory_enabled,
                    label="Enable Memory System",
                    info="Persistent character memory across conversations"
                )
                
                with gr.Row():
                    memory_context_size = gr.Slider(
                        1, 50, value=settings_manager.settings.memory_context_size,
                        step=1, label="Memory Context Size",
                        info="Number of memories to include in context"
                    )
                    auto_summarize = gr.Checkbox(
                        value=settings_manager.settings.auto_summarize,
                        label="Auto-summarize conversations",
                        info="Automatically create summaries for long conversations"
                    )
            
            # ==================== UI Settings ====================
            with gr.Tab("🎨 Interface"):
                with gr.Row():
                    show_timestamps = gr.Checkbox(
                        value=settings_manager.settings.show_timestamps,
                        label="Show message timestamps"
                    )
                    show_typing = gr.Checkbox(
                        value=settings_manager.settings.show_typing_indicator,
                        label="Show typing indicator"
                    )
                    compact_mode = gr.Checkbox(
                        value=settings_manager.settings.compact_mode,
                        label="Compact mode"
                    )
                
                theme = gr.Radio(
                    choices=["dark", "light"],
                    value=settings_manager.settings.theme,
                    label="Theme"
                )
            
            # ==================== Advanced ====================
            with gr.Tab("🔧 Advanced"):
                with gr.Row():
                    debug_mode = gr.Checkbox(
                        value=settings_manager.settings.debug_mode,
                        label="Debug mode",
                        info="Show detailed logging in console"
                    )
                    log_conversations = gr.Checkbox(
                        value=settings_manager.settings.log_conversations,
                        label="Log conversations",
                        info="Save conversation logs"
                    )
                
                gr.Markdown("### 📊 System Info")
                system_info = gr.HTML(
                    value=_get_system_info_html()
                )
        
        # ==================== Actions ====================
        with gr.Row():
            save_btn = gr.Button("💾 Save Settings", variant="primary")
            reset_btn = gr.Button("🔄 Reset to Defaults", variant="secondary")
        
        status_text = gr.Textbox(label="Status", interactive=False, visible=False)
        
        # ==================== Event Handlers ====================
        
        def save_settings(
            model, llm_prov, temp, max_tok, top, freq, pres,
            auto_p, tts, tts_back, vad, vad_thresh, sample,
            mem, mem_ctx, auto_sum,
            timestamps, typing, compact, theme_val,
            debug, log_conv
        ):
            settings_manager.update(
                model=model,
                llm_provider=llm_prov,
                temperature=temp,
                max_tokens=int(max_tok),
                top_p=top,
                frequency_penalty=freq,
                presence_penalty=pres,
                auto_play=auto_p,
                tts_enabled=tts,
                tts_backend=tts_back,
                vad_enabled=vad,
                vad_threshold=vad_thresh,
                sample_rate=sample,
                memory_enabled=mem,
                memory_context_size=int(mem_ctx),
                auto_summarize=auto_sum,
                show_timestamps=timestamps,
                show_typing_indicator=typing,
                compact_mode=compact,
                theme=theme_val,
                debug_mode=debug,
                log_conversations=log_conv
            )
            
            if on_save:
                on_save(settings_manager.settings)
            
            return gr.update(value="✅ Settings saved!", visible=True)
        
        def reset_settings():
            settings_manager.reset()
            s = settings_manager.settings
            return [
                s.model, s.llm_provider, s.temperature, s.max_tokens,
                s.top_p, s.frequency_penalty, s.presence_penalty,
                s.auto_play, s.tts_enabled, s.tts_backend, s.vad_enabled, s.vad_threshold, s.sample_rate,
                s.memory_enabled, s.memory_context_size, s.auto_summarize,
                s.show_timestamps, s.show_typing_indicator, s.compact_mode, s.theme,
                s.debug_mode, s.log_conversations,
                gr.update(value="🔄 Settings reset to defaults!", visible=True)
            ]
        
        save_btn.click(
            save_settings,
            inputs=[
                model_dropdown, llm_provider, temperature, max_tokens,
                top_p, freq_penalty, pres_penalty,
                auto_play, tts_enabled, tts_backend, vad_enabled, vad_threshold, sample_rate,
                memory_enabled, memory_context_size, auto_summarize,
                show_timestamps, show_typing, compact_mode, theme,
                debug_mode, log_conversations
            ],
            outputs=[status_text]
        )
        
        reset_btn.click(
            reset_settings,
            outputs=[
                model_dropdown, llm_provider, temperature, max_tokens,
                top_p, freq_penalty, pres_penalty,
                auto_play, tts_enabled, tts_backend, vad_enabled, vad_threshold, sample_rate,
                memory_enabled, memory_context_size, auto_summarize,
                show_timestamps, show_typing, compact_mode, theme,
                debug_mode, log_conversations,
                status_text
            ]
        )
    
    def refresh():
        """Refresh settings from file"""
        settings_manager.settings = settings_manager.load()
    
    return panel, refresh


def _get_system_info_html() -> str:
    """Get system information HTML"""
    import platform
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        cuda_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if cuda_available else "N/A"
    except:
        cuda_available = False
        cuda_device = "N/A"
        cuda_memory = "N/A"
    
    return f"""
    <div style="background: #000000; padding: 12px; border: 1px solid var(--theme-primary, #00BFFF); border-radius: 0px; font-family: monospace; font-size: 0.85em;">
        <div style="color: var(--theme-dim, #006699);">Platform: <span style="color: var(--theme-primary, #00BFFF);">{platform.system()} {platform.release()}</span></div>
        <div style="color: var(--theme-dim, #006699);">Python: <span style="color: var(--theme-primary, #00BFFF);">{platform.python_version()}</span></div>
        <div style="color: var(--theme-dim, #006699);">CUDA: <span style="color: var(--theme-primary, #00BFFF);">{'✓ Available' if cuda_available else '✗ Not available'}</span></div>
        <div style="color: var(--theme-dim, #006699);">GPU: <span style="color: var(--theme-primary, #00BFFF);">{cuda_device}</span></div>
        <div style="color: var(--theme-dim, #006699);">VRAM: <span style="color: var(--theme-primary, #00BFFF);">{cuda_memory}</span></div>
    </div>
    """


# ============================================================================
# Keyboard Shortcuts Modal
# ============================================================================

def create_shortcuts_modal() -> str:
    """Create keyboard shortcuts modal HTML"""
    return """
    <div id="shortcuts-help" style="
        position: fixed;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        background: #000000;
        border: 2px solid var(--theme-primary, #00BFFF);
        border-radius: 0px;
        padding: 24px;
        z-index: 1000;
        display: none;
        box-shadow: 0 0 30px var(--theme-glow, rgba(0,191,255,0.3));
        max-width: 400px;
    ">
        <h3 style="margin-top: 0; color: var(--theme-primary, #00BFFF); display: flex; align-items: center; gap: 8px;">
            <span>⌨️</span> Keyboard Shortcuts
        </h3>
        <table style="width: 100%; color: var(--theme-primary, #00BFFF); border-collapse: collapse;">
            <tr>
                <td style="padding: 8px 0;">
                    <kbd style="background: #000000; border: 1px solid var(--theme-primary, #00BFFF); padding: 4px 8px; border-radius: 0px; font-family: monospace; color: var(--theme-primary, #00BFFF);">Shift</kbd>
                </td>
                <td style="padding: 8px 12px;">Hold to record (Push-to-Talk)</td>
            </tr>
            <tr>
                <td style="padding: 8px 0;">
                    <kbd style="background: #000000; border: 1px solid var(--theme-primary, #00BFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #00BFFF);">Ctrl</kbd>
                    +
                    <kbd style="background: #000000; border: 1px solid var(--theme-primary, #00BFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #00BFFF);">Enter</kbd>
                </td>
                <td style="padding: 8px 12px;">Send message</td>
            </tr>
            <tr>
                <td style="padding: 8px 0;">
                    <kbd style="background: #000000; border: 1px solid var(--theme-primary, #00BFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #00BFFF);">Ctrl</kbd>
                    +
                    <kbd style="background: #000000; border: 1px solid var(--theme-primary, #00BFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #00BFFF);">N</kbd>
                </td>
                <td style="padding: 8px 12px;">New conversation</td>
            </tr>
            <tr>
                <td style="padding: 8px 0;">
                    <kbd style="background: #000000; border: 1px solid var(--theme-primary, #00BFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #00BFFF);">Esc</kbd>
                </td>
                <td style="padding: 8px 12px;">Clear input</td>
            </tr>
            <tr>
                <td style="padding: 8px 0;">
                    <kbd style="background: #000000; border: 1px solid var(--theme-primary, #00BFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #00BFFF);">Ctrl</kbd>
                    +
                    <kbd style="background: #000000; border: 1px solid var(--theme-primary, #00BFFF); padding: 4px 8px; border-radius: 0px; color: var(--theme-primary, #00BFFF);">/</kbd>
                </td>
                <td style="padding: 8px 12px;">Toggle this help</td>
            </tr>
        </table>
        <button onclick="this.parentElement.style.display='none'" style="
            margin-top: 16px;
            width: 100%;
            padding: 10px;
            background: var(--theme-primary, #00BFFF);
            color: #000000;
            border: none;
            border-radius: 0px;
            cursor: pointer;
            font-size: 0.9em;
        ">Close</button>
    </div>
    <script>
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === '/') {
            var modal = document.getElementById('shortcuts-help');
            modal.style.display = modal.style.display === 'none' ? 'block' : 'none';
            e.preventDefault();
        }
        if (e.key === 'Escape') {
            document.getElementById('shortcuts-help').style.display = 'none';
        }
    });
    </script>
    """


# ============================================================================
# Status Bar Component
# ============================================================================

def create_status_bar(
    ptt_status: str = "ready",
    character_name: str = "",
    model_name: str = "",
    memory_count: int = 0
) -> str:
    """Create status bar HTML"""

    # Use CSS variables for theme support
    status_colors = {
        "ready": "var(--theme-primary, #00BFFF)",
        "recording": "var(--theme-primary, #00BFFF)",
        "processing": "var(--theme-primary, #00BFFF)",
        "offline": "var(--theme-dim, #006699)"
    }

    status_icons = {
        "ready": "🟢",
        "recording": "🔴",
        "processing": "⏳",
        "offline": "⭕"
    }

    ptt_color = status_colors.get(ptt_status, "var(--theme-dim, #006699)")
    ptt_icon = status_icons.get(ptt_status, "⭕")
    ptt_text = {
        "ready": "Ready (Hold Shift)",
        "recording": "Recording...",
        "processing": "Processing...",
        "offline": "PTT Offline"
    }.get(ptt_status, "Unknown")

    return f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 16px;
        background: #000000;
        border: 1px solid var(--theme-dim, #006699);
        border-radius: 0px;
        font-size: 0.85em;
        color: var(--theme-primary, #00BFFF);
    ">
        <div style="display: flex; gap: 16px; align-items: center;">
            <span style="color: {ptt_color};">{ptt_icon} {ptt_text}</span>
            {f'<span style="color: var(--theme-primary, #00BFFF);">👤 {character_name}</span>' if character_name else ''}
        </div>
        <div style="display: flex; gap: 16px; align-items: center;">
            {f'<span>🧠 {memory_count} memories</span>' if memory_count else ''}
            {f'<span style="color: var(--theme-dim, #006699);">🤖 {model_name[:25]}...</span>' if len(model_name) > 25 else f'<span style="color: var(--theme-dim, #006699);">🤖 {model_name}</span>' if model_name else ''}
        </div>
    </div>
    """


# ============================================================================
# Standalone Settings App
# ============================================================================

def run_standalone(settings_path: str = "./settings.json", port: int = 7864):
    """Run settings panel as standalone app"""
    
    settings_manager = SettingsManager(Path(settings_path))
    panel, _ = create_settings_panel(settings_manager)
    
    panel.launch(
        server_port=port,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7864
    run_standalone(port=port)
