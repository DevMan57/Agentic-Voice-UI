"""
Chat Service - Core Business Logic

Extracted from voice_chat_app.py:process_message_with_memory() (lines 2493-2810).
Separates message processing logic from UI formatting.
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class ToolCallInfo:
    """Information about a single tool call."""
    name: str
    arguments: dict
    result: str
    step: int


@dataclass
class ChatResult:
    """
    Result of processing a chat message.

    This is pure data - no UI formatting. The UI layer converts this to Gradio format.
    """
    response_text: str
    audio_data: Optional[Tuple[int, Any]]  # (sample_rate, np_array) or None
    tool_calls: List[ToolCallInfo]
    memory_context: dict
    conversation_id: str
    user_display_message: str  # What to show in UI for user message
    was_long_response: bool = False
    tts_warning: str = ""
    # HUD metrics
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    memory_nodes: int = 0
    emotion: str = "neutral"
    tts_speed: float = 0.0


class ChatService:
    """
    Core chat processing service.
    
    Handles:
    - Memory retrieval and context building
    - LLM API calls with tool execution
    - TTS generation
    - Conversation persistence
    
    Does NOT handle:
    - UI formatting (that's in ui/formatters.py)
    - Gradio state management
    """
    
    def __init__(
        self,
        memory_manager,
        character_manager,
        tts_model,
        settings: dict
    ):
        """
        Initialize chat service with injected dependencies.
        
        Args:
            memory_manager: MultiCharacterMemoryManager instance
            character_manager: CharacterManager instance
            tts_model: TTS model (IndexTTS2 or Kokoro)
            settings: Settings dictionary
        """
        self.memory = memory_manager
        self.characters = character_manager
        self.tts = tts_model
        self.settings = settings
    
    def process_message(
        self,
        user_message: str,
        character_id: str,
        voice_file: str,
        model: str,
        conversation_id: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 1.0,
        freq_penalty: float = 0.0,
        pres_penalty: float = 0.0,
        llm_provider: str = "openrouter",
        image_data: Optional[Tuple] = None,
        incognito: bool = False,
        chat_history: Optional[list] = None,
        emotion_context: Optional[str] = None,
        emotion_label: Optional[str] = None,  # Emotion label for TTS modulation
        skill_context: Optional[str] = None,
        # Import functions we need from voice_chat_app
        chat_with_llm_func = None,
        execute_tool_func = None,
        get_tools_schema_func = None,
        should_load_tools_func = None,
        should_load_mcp_tools_func = None,
        generate_speech_func = None,
        save_conversation_func = None,
        generate_conversation_id_func = None,
    ) -> ChatResult:
        """
        Process a user message and return structured result.

        This is the core logic extracted from process_message_with_memory().
        Returns pure data - UI layer handles formatting.

        Args:
            user_message: User's input text
            character_id: Active character ID
            voice_file: Voice reference file for TTS
            model: LLM model name
            conversation_id: Current conversation ID ("new" for new conversation)
            temperature: LLM temperature
            max_tokens: Max tokens for LLM
            top_p: LLM top_p
            freq_penalty: LLM frequency penalty
            pres_penalty: LLM presence penalty
            llm_provider: "openrouter" or "lmstudio"
            image_data: Optional (base64_str, mime_type) tuple
            incognito: If True, don't save to memory
            chat_history: Existing chat history
            emotion_context: Optional emotion context string for LLM prompt
            emotion_label: Optional emotion label (e.g., 'happy') for TTS modulation
            skill_context: Optional skill context string
            *_func: Function references from voice_chat_app (dependency injection)

        Returns:
            ChatResult with response, audio, tool calls, etc.
        """
        # Handle conversation ID
        actual_conversation_id = conversation_id
        if conversation_id == "new" or not conversation_id:
            actual_conversation_id = generate_conversation_id_func()
            if not incognito:
                self.settings["current_conversation_id"] = actual_conversation_id
        
        # Activate character memory (skip in incognito mode)
        if not incognito:
            self.memory.activate_character(character_id)

        # Get character
        character = self.characters.get_character(character_id)
        if not character:
            return ChatResult(
                response_text="Character not found!",
                audio_data=None,
                tool_calls=[],
                memory_context={},
                conversation_id=conversation_id,
                user_display_message=user_message,
                tts_warning="❌ Character not found"
            )

        # Build memory context (skip entirely in incognito mode)
        if incognito:
            print("[Memory] Incognito: Bypassing memory and knowledge graph queries")
            memory_context = {}
            formatted_memory = ""
        else:
            memory_context = self.memory.build_context(character_id, user_message)
            formatted_memory = self.memory.format_context_for_prompt(memory_context)
        
        # Build system prompt
        system_prompt = self._build_system_prompt(
            character=character,
            formatted_memory=formatted_memory,
            emotion_context=emotion_context,
            skill_context=skill_context
        )
        
        # Build messages for LLM
        messages = self._build_llm_messages(
            system_prompt=system_prompt,
            user_message=user_message,
            chat_history=chat_history or []
        )
        
        # Prepare tools (progressive disclosure)
        tools_schema = []
        if character.allowed_tools and should_load_tools_func(user_message):
            include_mcp = should_load_mcp_tools_func(user_message)
            tools_schema = get_tools_schema_func(character.allowed_tools, include_mcp=include_mcp)
            mcp_status = "+MCP" if include_mcp else "local only"
            print(f"[Progressive Disclosure] Loaded {len(tools_schema)} tool schemas ({mcp_status})")
        else:
            print(f"[Progressive Disclosure] Skipped tool schemas (casual message or no tools)")
        
        # Execute chat loop with tool handling (track timing for HUD)
        import time
        llm_start = time.time()
        final_response, tool_calls_info, usage_stats = self._execute_chat_loop(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            freq_penalty=freq_penalty,
            pres_penalty=pres_penalty,
            llm_provider=llm_provider,
            image_data=image_data,
            tools_schema=tools_schema,
            character=character,
            chat_with_llm_func=chat_with_llm_func,
            execute_tool_func=execute_tool_func
        )
        latency_ms = int((time.time() - llm_start) * 1000)

        # Get memory stats for HUD
        memory_stats = self.memory.get_stats(character_id)
        memory_nodes = (
            memory_stats.get("episodic_count", 0) +
            memory_stats.get("semantic_count", 0) +
            memory_stats.get("procedural_count", 0)
        )
        
        # Save to memory (unless incognito)
        if not incognito:
            self.memory.add_interaction(
                character_id=character_id,
                user_message=user_message,
                assistant_response=final_response
            )
        else:
            print("[Memory] Incognito: Skipping memory save")
        
        # Generate TTS audio
        audio_data = None
        tts_warning = ""
        was_long = False
        
        if self.settings.get("tts_enabled", True):
            # Note: TTS filtering happens in UI layer via filter_for_speech()
            # generate_speech_func handles lazy initialization internally
            audio_result, was_long = generate_speech_func(final_response, voice_file, emotion=emotion_label)
            audio_data = audio_result
            
            if was_long:
                tts_warning = "⚠️ Long response - TTS may take longer"
            elif audio_result is None:
                tts_warning = "ℹ️ TTS not available"
        else:
            tts_warning = "ℹ️ TTS disabled in settings"
        
        # Save conversation (unless incognito)
        if not incognito and save_conversation_func:
            # Note: Actual saving happens in UI layer after formatting
            pass
        
        # Prepare user display message
        user_display = user_message.strip() if user_message else ""
        if image_data and not user_message:
            user_display = "📷 [Sent an image]"
        elif image_data and user_message:
            user_display = f"📷 {user_message.strip()}"
        
        return ChatResult(
            response_text=final_response,
            audio_data=audio_data,
            tool_calls=tool_calls_info,
            memory_context=memory_context,
            conversation_id=actual_conversation_id,
            user_display_message=user_display,
            was_long_response=was_long,
            tts_warning=tts_warning,
            # HUD metrics
            latency_ms=latency_ms,
            tokens_in=usage_stats.get("prompt_tokens", 0),
            tokens_out=usage_stats.get("completion_tokens", 0),
            memory_nodes=memory_nodes,
            emotion=emotion_label or "neutral"
        )
    
    def _build_system_prompt(
        self,
        character,
        formatted_memory: str,
        emotion_context: Optional[str] = None,
        skill_context: Optional[str] = None
    ) -> str:
        """Build system prompt with character, memory, emotion, and skill context."""
        system_prompt = character.system_prompt
        
        # Add TTS guidance
        tts_guidance = """

VOICE RESPONSE FORMAT (Critical for natural speech):
- Respond as natural spoken dialogue ONLY - no written formatting
- DO NOT use asterisks for actions (*sighs*, *thinks*) - just speak naturally
- DO NOT use markdown, bullet points, headers, or special characters
- Use punctuation to create natural pauses: commas, periods, ellipses
- Keep sentences flowing and conversational
- Moderate response length - aim for 2-4 natural sentences unless more detail requested
- Express emotions through word choice and phrasing, not action tags"""
        
        system_prompt += tts_guidance
        
        # Add tool guidance if tools available
        if character.allowed_tools:
            tool_names = ", ".join(character.allowed_tools)
            tool_guidance = f"""

TOOLS AVAILABLE:
You have access to these tools: {tool_names}
- IMPORTANT: You must ACTUALLY call the tool function. Do not just say "I'm searching" - emit the tool call.
- When asked to search the web, USE the web_search tool - don't just talk about it
- When asked what time it is, USE the get_current_time tool
- When asked to read or write files, USE the appropriate file tools
- Call tools proactively when they would help answer the question

TOOL NARRATION (Critical for voice responses):
When using tools, provide brief spoken updates and summaries:
- BEFORE using a tool: "Let me check that file for you" or "I'll search for that information"
- AFTER getting results: Summarize key findings conversationally, don't read raw data
- Keep narration natural and concise: "I found 3 matches in the config file" not "read_file returned: {{'matches': 3}}"
- Tool details will be shown visually to the user - your job is to narrate what's happening
- Example: "Let me search for recent AI news. [TOOL CALL] Okay, I found several articles..."
"""
            system_prompt += tool_guidance
        
        # Add memory context
        if formatted_memory.strip():
            system_prompt += f"\n\n--- YOUR MEMORIES ---\n{formatted_memory}"
        
        # Add emotion context
        if emotion_context:
            system_prompt += f"\n\n--- USER EMOTIONAL STATE ---\n{emotion_context}\nConsider this emotional state when crafting your response."
        
        # Add skill context
        if skill_context:
            system_prompt += skill_context
            print(f"[Skills] Injected skill context ({len(skill_context)} chars)")
        
        return system_prompt
    
    def _build_llm_messages(
        self,
        system_prompt: str,
        user_message: str,
        chat_history: list
    ) -> list:
        """Build message list for LLM API."""
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 10 messages)
        recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
        for item in recent_history:
            if isinstance(item, dict):
                # Already in correct format (type="messages")
                messages.append(item)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                # Legacy tuple format [user, bot]
                if item[0]: messages.append({"role": "user", "content": item[0]})
                if item[1]: messages.append({"role": "assistant", "content": item[1]})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _execute_chat_loop(
        self,
        messages: list,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        freq_penalty: float,
        pres_penalty: float,
        llm_provider: str,
        image_data: Optional[Tuple],
        tools_schema: list,
        character,
        chat_with_llm_func,
        execute_tool_func,
        max_turns: int = 5
    ) -> Tuple[str, List[ToolCallInfo], dict]:
        """
        Execute chat loop with recursive tool handling.

        Returns:
            (final_response, tool_calls_info, usage_stats)
        """
        current_turn = 0
        final_response = ""
        tool_calls_info = []
        tool_call_count = 0
        # Accumulate usage across turns
        usage_stats = {"prompt_tokens": 0, "completion_tokens": 0}
        
        while current_turn < max_turns:
            print(f"[Chat] Turn {current_turn+1} - Sending {len(messages)} messages (Tools: {len(tools_schema)})")
            
            # Call LLM
            response_msg = chat_with_llm_func(
                messages, model, temperature, max_tokens, top_p,
                freq_penalty, pres_penalty, provider=llm_provider,
                image_data=image_data, tools=tools_schema
            )

            # Accumulate usage stats if present
            if isinstance(response_msg, dict) and "_usage" in response_msg:
                msg_usage = response_msg["_usage"]
                usage_stats["prompt_tokens"] += msg_usage.get("prompt_tokens", 0)
                usage_stats["completion_tokens"] += msg_usage.get("completion_tokens", 0)

            # Check for tool calls
            if isinstance(response_msg, dict) and response_msg.get("tool_calls"):
                tool_calls = response_msg["tool_calls"]
                num_tools_this_batch = len(tool_calls)
                print(f"[Chat] Model requested {num_tools_this_batch} tools")
                
                # Append assistant's request to history
                messages.append(response_msg)
                
                # Execute each tool
                for idx, tool_call in enumerate(tool_calls):
                    tool_call_count += 1
                    call_id = tool_call["id"]
                    tool_name = tool_call.get("function", {}).get("name", "unknown")
                    tool_args = tool_call.get("function", {}).get("arguments", {})
                    
                    # Parse arguments if string
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {"raw": tool_args}
                    
                    result = execute_tool_func(tool_call)
                    
                    # Truncate result if too long
                    if len(result) > 2000:
                        result = result[:2000] + "... (truncated)"
                    
                    # Store tool call info
                    tool_calls_info.append(ToolCallInfo(
                        name=tool_name,
                        arguments=tool_args,
                        result=result,
                        step=tool_call_count
                    ))
                    
                    # Append tool result to messages
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": call_id
                    })
                
                # Loop again to get interpretation
                current_turn += 1
                image_data = None  # Don't resend image in subsequent turns
                continue
            
            else:
                # No tool calls, just text response
                final_response = response_msg.get("content", "")
                
                # Cleanup: Remove common prefixes that models might hallucinate
                if final_response:
                    clean_pattern = r'^(Assistant|AI|'+re.escape(character.name)+r'):?\s*'
                    final_response = re.sub(clean_pattern, '', final_response, flags=re.IGNORECASE).strip()
                    
                    # Remove quotes if wrapped
                    if final_response.startswith('"') and final_response.endswith('"'):
                        final_response = final_response[1:-1].strip()
                break
        
        # Fallback if loop exhausted
        if not final_response:
            final_response = "I processed the information but couldn't generate a final response."

        print(f"[Chat] [{character.id}] Response: {final_response[:100]}...")

        return final_response, tool_calls_info, usage_stats
