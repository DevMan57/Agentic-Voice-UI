"""
UI Formatting Functions for IndexTTS2 Voice Chat

Extracted from voice_chat_app.py to separate UI concerns from business logic.
These functions convert service layer data into UI-ready HTML/text formats.
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime


def format_memory_for_ui(context: Dict[str, Any]) -> str:
    """Format memory context for display in the UI"""
    parts = []

    if context.get('character_state'):
        state = context['character_state']
        parts.append(f"**Session:** {state['total_interactions']} interactions | Mood: {state['mood']}")

    semantic = context.get('semantic_memories', [])
    if semantic:
        parts.append("\n**Character Knowledge:**")
        for mem in semantic[:3]:
            parts.append(f"• {mem[:80]}...")

    episodic = context.get('episodic_memories', [])
    if episodic:
        parts.append("\n**Relevant Past:**")
        for mem in episodic[:2]:
            lines = mem.split('\n')
            if lines:
                parts.append(f"• {lines[0][:60]}...")

    return '\n'.join(parts) if parts else "*No memories yet*"


def format_tool_call_html(tool_name: str, arguments: Dict[str, Any], result: str = None,
                          step: int = None, total: int = None, status: str = "complete",
                          use_tree_format: bool = False) -> str:
    """Format a tool call as HTML for CYBERDECK System Protocol display.

    Args:
        tool_name: Name of the tool being called
        arguments: Dict of argument key-value pairs
        result: Result string from the tool (optional)
        step: Current step number in a chain (optional)
        total: Total steps in chain (optional)
        status: Status indicator - "pending", "complete", or "failed"
        use_tree_format: If True, use tree/thread structure instead of block format
    """
    args_str = ", ".join([f"{k}: {v}" for k, v in arguments.items()])

    # Status indicator - Military style
    status_icons = {"pending": "[ EXEC ]", "complete": "[ OK ]", "failed": "[ FAIL ]"}
    status_icon = status_icons.get(status, "[ OK ]")

    # Status class for success indicator
    status_class = "thread-step-success" if status == "complete" else ""

    # Protocol indicator for all tool calls
    step_html = ""
    chain_class = ""
    if step is not None and total is not None:
        if total > 1:
            step_html = f'<span class="tool-step-badge">PROTOCOL {step:02d}</span>'
            if step < total:
                chain_class = " tool-chain-continues"
            if step > 1:
                chain_class += " tool-chain-continued"
        else:
            # Single tool call - SYS_CALL
            step_html = f'<span class="tool-step-badge">SYS_CALL</span>'

    # Tree format - uses conversation thread structure
    if use_tree_format:
        html = f'<div class="thread-step thread-step-exec">'
        html += f'<span class="tool-step-badge">EXEC</span> {tool_name}'
        if args_str:
            html += f'<div class="tool-call-args">└─ {args_str}</div>'
        html += '</div>'

        if result:
            display_result = result[:200] + "..." if len(result) > 200 else result
            display_result = display_result.replace('<', '&lt;').replace('>', '&gt;')
            html += f'<div class="thread-step thread-step-data">'
            html += f'<span class="tool-step-badge">DATA</span>'
            html += f'<div class="tool-call-result">└─ {display_result}</div>'
            html += '</div>'

        return html

    # Standard block format
    html = f'<div class="tool-call-block{chain_class}">'
    html += f'<div class="tool-call-header">{step_html}{status_icon} {tool_name}</div>'
    html += f'<div class="tool-call-args">{args_str}</div>'

    if result:
        # Truncate result if too long
        display_result = result[:200] + "..." if len(result) > 200 else result
        # Escape HTML in result
        display_result = display_result.replace('<', '&lt;').replace('>', '&gt;')
        html += f'<div class="tool-call-result">{display_result}</div>'

    html += '</div>'
    return html


def format_message_with_extras(content: str, add_timestamp: bool = False, expandable_threshold: int = 800) -> str:
    """Wrap message content with expandable container (if long).

    Note: add_timestamp defaults to False since Gradio's chatbot component
    handles timestamps automatically. Adding them here causes duplicates.
    """
    result = content

    # Wrap in expandable container if content is long
    if len(content) > expandable_threshold:
        result = f'<div class="message-expandable">{content}</div>'
        result += '<div class="expand-btn">▼ Show more</div>'

    # Timestamps disabled - Gradio chatbot handles them
    # This prevents duplicate timestamps and timestamps being spoken by TTS

    return result


def format_memory_recall_html(context: Dict[str, Any]) -> str:
    """Format memory recall as HTML for Claude Desktop-style display"""
    if not context:
        return ""

    items = []

    # Extract memories for display
    semantic = context.get('semantic_memories', [])
    episodic = context.get('episodic_memories', [])

    for mem in semantic[:2]:
        items.append(f'<div class="memory-item">{mem[:100]}...</div>')

    for mem in episodic[:2]:
        lines = mem.split('\n')
        if lines:
            items.append(f'<div class="memory-item">{lines[0][:80]}...</div>')

    if not items:
        return ""

    html = '<div class="memory-recall-block">'
    html += f'<div class="memory-recall-header">🧠 Recalled {len(items)} memories:</div>'
    html += '\n'.join(items)
    html += '</div>'

    return html


def filter_for_speech(full_message: str, has_tools: bool = False) -> str:
    """
    Extract only conversational narration for TTS.
    Removes HTML blocks (tool calls, memory recalls) and keeps only spoken text.

    Args:
        full_message: Full assistant message with HTML blocks
        has_tools: Whether this message involved tool calls (deprecated - now auto-detects)

    Returns:
        Clean text suitable for TTS
    """
    # Auto-detect if message contains ANY HTML blocks (check for both quote styles)
    has_html_blocks = ('<div class="' in full_message) or ('<div class=\'' in full_message)

    # If no HTML blocks present, return as-is
    if not has_html_blocks:
        return full_message

    print(f"[TTS Filter] Detected HTML blocks, filtering...")
    print(f"[TTS Filter] Original length: {len(full_message)} chars")

    # AGGRESSIVE APPROACH: Remove ALL <div...>...</div> tags and keep only plain text
    # This handles nested divs properly
    text = full_message

    # Remove all div blocks (handles nesting by removing from innermost to outermost)
    while '<div' in text and '</div>' in text:
        text = re.sub(r'<div[^>]*>.*?</div>', '', text, flags=re.DOTALL, count=1)

    # Also remove any remaining orphaned tags
    text = re.sub(r'<[^>]+>', '', text)

    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = text.strip()

    print(f"[TTS Filter] Filtered length: {len(text)} chars")
    print(f"[TTS Filter] Will speak: {text[:100]}...")

    # If filtering removed everything, provide a fallback
    if not text or len(text) < 10:
        print("[TTS Filter] WARNING: Filtering removed all text, using fallback")
        return "Working on that now."

    return text


def format_conversation_thread(tool_calls: list, response_text: str) -> str:
    """Format tool calls and response as a conversation thread with tree structure.

    Creates a visual tree showing:
    ├── [ SYSTEM :: PLANNER ]
    │   ├── > EXEC: tool_name
    │   │     └─ args
    │   ├── < DATA: result
    │   └── √ STATUS: Success
    └── [ AI :: RESPONSE ]
        Response text here

    Args:
        tool_calls: List of tool call objects with name, arguments, result
        response_text: Final response text

    Returns:
        HTML string with thread visualization
    """
    if not tool_calls:
        return response_text

    total_tools = len(tool_calls)
    html = '<div class="conversation-thread">'

    # System/Planner block
    html += '<div class="system-planner-block">'
    html += '<div class="system-planner-header">SYSTEM :: PLANNER</div>'
    html += '<div class="system-planner-content">'

    for i, tc in enumerate(tool_calls, 1):
        # Exec step
        html += f'<div class="thread-step thread-step-exec">'
        html += f'> EXEC: {tc.name}'
        if tc.arguments:
            args_str = ", ".join([f"{k}: {v}" for k, v in tc.arguments.items()])
            html += f'<div style="padding-left: 20px; color: var(--theme-dim);">└─ {args_str[:100]}</div>'
        html += '</div>'

        # Data step (result)
        if tc.result:
            display_result = tc.result[:150] + "..." if len(tc.result) > 150 else tc.result
            display_result = display_result.replace('<', '&lt;').replace('>', '&gt;')
            html += f'<div class="thread-step thread-step-data">'
            html += f'&lt; DATA: (Click to expand)'
            html += f'<div style="padding-left: 20px; color: var(--theme-medium);">└─ {display_result}</div>'
            html += '</div>'

    # Success status
    html += '<div class="thread-step thread-step-success">'
    html += f'√ STATUS: Success ({total_tools} {"calls" if total_tools > 1 else "call"})'
    html += '</div>'

    html += '</div></div>'  # Close planner content and block

    # AI Response block
    html += '<div class="system-planner-block" style="margin-top: 8px;">'
    html += '<div class="system-planner-header">AI :: RESPONSE</div>'
    html += f'<div class="system-planner-content">{response_text}</div>'
    html += '</div>'

    html += '</div>'  # Close conversation thread
    return html


def format_chat_result_for_ui(result, chat_history: list, use_thread_format: bool = False) -> list:
    """
    Convert ChatResult from service layer into Gradio chatbot format.

    Args:
        result: ChatResult object from ChatService
        chat_history: Existing chat history
        use_thread_format: If True, use thread/tree visualization for tool chains

    Returns:
        Updated chat history with new messages formatted for UI
    """

    # Build enhanced assistant message with tool call visualizations
    assistant_message_content = ""

    if result.tool_calls:
        total_tools = len(result.tool_calls)
        print(f"[UI] Adding {total_tools} tool call blocks with chain visualization")

        # Use thread format for complex multi-tool chains
        if use_thread_format and total_tools > 1:
            assistant_message_content = format_conversation_thread(
                result.tool_calls, result.response_text
            )
        else:
            # Standard block format
            formatted_tools = []
            for tc in result.tool_calls:
                formatted_tools.append(format_tool_call_html(
                    tc.name, tc.arguments, tc.result,
                    step=tc.step, total=total_tools
                ))

            # Wrap in tool chain container if multiple tools
            if total_tools > 1:
                assistant_message_content += '<div class="tool-chain-container">'
                assistant_message_content += "\n".join(formatted_tools)
                assistant_message_content += '</div>\n\n'
            else:
                assistant_message_content += formatted_tools[0] + "\n\n"

            assistant_message_content += result.response_text
    else:
        assistant_message_content = result.response_text

    # Format with timestamps and expandable containers
    user_formatted = format_message_with_extras(result.user_display_message)
    assistant_formatted = format_message_with_extras(assistant_message_content)

    # Append to history
    chat_history.append({"role": "user", "content": user_formatted})
    chat_history.append({"role": "assistant", "content": assistant_formatted})

    return chat_history
