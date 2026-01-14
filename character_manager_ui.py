#!/usr/bin/env python3
"""
IndexTTS2 Voice Chat - Character & Memory Management UI

Manage:
- Characters (create, edit, delete) with full-featured wizard
- Voices (upload, manage, auto-emoji assignment)
- Memories (view, edit, delete - episodic, semantic, procedural)
- Conversations (browse, export, delete)

Runs on port 7863
Cross-platform: Windows, Linux, WSL
"""

import os
import sys
import json
import platform
import shutil
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import requests

import gradio as gr

# Import shared memory system
from memory.characters import CharacterManager, Character, create_character_manager
from memory.memory_manager import MultiCharacterMemoryManager, create_memory_manager

# Import shared utilities
from utils import create_dark_theme
from tools import REGISTRY  # Import tool registry


# ============================================================================
# Platform Detection
# ============================================================================

PLATFORM = platform.system().lower()
IS_WINDOWS = PLATFORM == 'windows'
IS_LINUX = PLATFORM == 'linux'
IS_MAC = PLATFORM == 'darwin'
IS_WSL = IS_LINUX and 'microsoft' in platform.uname().release.lower()


# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
SESSIONS_DIR = SCRIPT_DIR / "sessions"
CONVERSATIONS_DIR = SESSIONS_DIR / "conversations"  # Consolidated under sessions/
VOICE_REF_DIR = SCRIPT_DIR / "voice_reference"
SKILLS_DIR = SCRIPT_DIR / "skills"

# Ensure directories exist
SESSIONS_DIR.mkdir(exist_ok=True)
CONVERSATIONS_DIR.mkdir(exist_ok=True)
VOICE_REF_DIR.mkdir(exist_ok=True)
SKILLS_DIR.mkdir(exist_ok=True)

CONFIG_FILE = SCRIPT_DIR / "config.env"


def load_config_key() -> str:
    """Load API Key from config"""
    if CONFIG_FILE.exists():
        text = CONFIG_FILE.read_text()
        for line in text.splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


API_KEY = load_config_key()


# ============================================================================
# Voice Emoji Mapping
# ============================================================================

# Auto-assign emojis based on voice file names or characteristics
VOICE_EMOJI_MAP = {
    # Character/celebrity names
    "emma": "🧙‍♀️",
    "watson": "🧙‍♀️",
    "hermione": "🧙‍♀️",
    "gandalf": "🧙",
    "wizard": "🧙",
    "rooney": "🖤",
    "mara": "🖤",
    "lisbeth": "🖤",
    "samantha": "💝",
    "her": "💝",
    "scarlett": "💋",
    "morgan": "🎬",
    "freeman": "🎬",
    "david": "🎙️",
    "attenborough": "🌍",
    
    # Voice types
    "male": "🎤",
    "female": "🎵",
    "deep": "🔊",
    "soft": "🌸",
    "robotic": "🤖",
    "child": "👶",
    "old": "👴",
    "young": "👱",
    
    # Accents/regions
    "british": "🇬🇧",
    "american": "🇺🇸",
    "australian": "🇦🇺",
    "french": "🇫🇷",
    "german": "🇩🇪",
    "japanese": "🇯🇵",
    "spanish": "🇪🇸",
    
    # Moods/styles
    "calm": "😌",
    "energetic": "⚡",
    "mysterious": "🌙",
    "cheerful": "😊",
    "serious": "😐",
    "warm": "🌞",
    "cool": "❄️",
    
    # Generic
    "reference": "🎭",
    "custom": "🎤",
    "voice": "🔊",
    "default": "🔊",
    "me": "🎤",
    "my": "🎤",
    "maya": "🌸",
}

# Default emojis to cycle through for unknown voices
DEFAULT_EMOJIS = ["🎤", "🎵", "🔊", "🎙️", "🎶", "🗣️", "📢", "🔉", "🎧", "🎼"]


def get_emoji_for_voice(filename: str) -> str:
    """
    Automatically assign an emoji based on voice file name.
    Uses keyword matching to find the best emoji.
    """
    if not filename:
        return "🔊"
    
    name_lower = Path(filename).stem.lower()
    
    # Check for exact or partial matches
    for keyword, emoji in VOICE_EMOJI_MAP.items():
        if keyword in name_lower:
            return emoji
    
    # Use a hash-based selection for consistency
    hash_val = sum(ord(c) for c in name_lower)
    return DEFAULT_EMOJIS[hash_val % len(DEFAULT_EMOJIS)]


def suggest_emoji_options(filename: str) -> List[str]:
    """
    Suggest multiple emoji options for a voice file.
    Returns a list of emojis that might fit.
    """
    suggestions = []
    name_lower = Path(filename).stem.lower()
    
    # Add matches from keyword map
    for keyword, emoji in VOICE_EMOJI_MAP.items():
        if keyword in name_lower and emoji not in suggestions:
            suggestions.append(emoji)
    
    # Add default if no matches
    if not suggestions:
        suggestions.append(get_emoji_for_voice(filename))
    
    # Add some general options
    general = ["🎤", "🎵", "🔊", "🎙️", "🗣️", "🌟", "✨", "💫"]
    for emoji in general:
        if emoji not in suggestions:
            suggestions.append(emoji)
        if len(suggestions) >= 8:
            break
    
    return suggestions


# ============================================================================
# Initialize Systems
# ============================================================================

CHARACTER_MANAGER = create_character_manager()
MEMORY_MANAGER = create_memory_manager(use_local=True)


# ============================================================================
# Voice Management Functions
# ============================================================================

def get_available_voices() -> List[tuple]:
    """Get available voice reference files as dropdown choices with emojis"""
    VOICE_REF_DIR.mkdir(exist_ok=True)
    voices = []
    for f in VOICE_REF_DIR.glob("*.wav"):
        name = f.stem
        emoji = get_emoji_for_voice(name)
        display_name = f"{emoji} {name.title()}"
        voices.append((display_name, f.name))
    voices.sort(key=lambda x: x[0])
    if not voices:
        voices = [("🔊 Reference", "reference.wav")]
    return voices


def get_voice_metadata_path() -> Path:
    """Get path to voice metadata file"""
    return VOICE_REF_DIR / "voice_metadata.json"


def load_voice_metadata() -> Dict[str, Dict]:
    """Load voice metadata (custom emojis, descriptions, etc.)"""
    meta_path = get_voice_metadata_path()
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}


def save_voice_metadata(metadata: Dict[str, Dict]):
    """Save voice metadata"""
    meta_path = get_voice_metadata_path()
    try:
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"[Voice] Error saving metadata: {e}")


def upload_voice_file(file_obj, custom_name: str = None, custom_emoji: str = None) -> Tuple[str, Any]:
    """
    Upload a voice file to the voice_reference directory.
    
    Args:
        file_obj: Gradio file upload object
        custom_name: Optional custom name for the voice
        custom_emoji: Optional custom emoji
    
    Returns:
        (status_message, updated_voice_choices)
    """
    if file_obj is None:
        return "❌ No file uploaded", gr.update()
    
    try:
        # Handle different file object types (Gradio versions vary)
        if hasattr(file_obj, 'name'):
            source_path = Path(file_obj.name)
        elif isinstance(file_obj, str):
            source_path = Path(file_obj)
        else:
            return "❌ Invalid file object", gr.update()
        
        # Validate it's a WAV file
        if source_path.suffix.lower() != '.wav':
            return "❌ Only WAV files are supported. Please convert your audio to WAV format.", gr.update()
        
        # Determine destination name
        if custom_name and custom_name.strip():
            # Sanitize custom name
            safe_name = re.sub(r'[^\w\s-]', '', custom_name.strip())
            safe_name = re.sub(r'[\s-]+', '_', safe_name).lower()
            dest_name = f"{safe_name}.wav"
        else:
            dest_name = source_path.name.lower().replace(' ', '_')
        
        dest_path = VOICE_REF_DIR / dest_name
        
        # Check if file already exists
        if dest_path.exists():
            # Add number suffix
            base = dest_path.stem
            counter = 1
            while dest_path.exists():
                dest_path = VOICE_REF_DIR / f"{base}_{counter}.wav"
                counter += 1
            dest_name = dest_path.name
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        
        # Determine emoji
        emoji = custom_emoji if custom_emoji else get_emoji_for_voice(dest_name)
        
        # Save metadata
        metadata = load_voice_metadata()
        metadata[dest_name] = {
            "emoji": emoji,
            "uploaded_at": datetime.now().isoformat(),
            "original_name": source_path.name,
            "platform": PLATFORM
        }
        save_voice_metadata(metadata)
        
        # Get updated choices
        updated_choices = get_available_voices_with_metadata()
        
        return f"✅ Voice '{dest_name}' uploaded successfully! Emoji: {emoji}", gr.update(choices=updated_choices)
        
    except PermissionError:
        return "❌ Permission denied. Cannot write to voice_reference directory.", gr.update()
    except Exception as e:
        return f"❌ Upload failed: {str(e)}", gr.update()


def delete_voice_file(voice_name: str) -> Tuple[str, Any]:
    """Delete a voice file"""
    if not voice_name or voice_name == "reference.wav":
        return "❌ Cannot delete the default reference voice", gr.update()
    
    try:
        voice_path = VOICE_REF_DIR / voice_name
        if voice_path.exists():
            voice_path.unlink()
            
            # Remove from metadata
            metadata = load_voice_metadata()
            if voice_name in metadata:
                del metadata[voice_name]
                save_voice_metadata(metadata)
            
            updated_choices = get_available_voices_with_metadata()
            return f"✅ Voice '{voice_name}' deleted", gr.update(choices=updated_choices)
        else:
            return "❌ Voice file not found", gr.update()
    except Exception as e:
        return f"❌ Error deleting voice: {str(e)}", gr.update()


def update_voice_emoji(voice_name: str, new_emoji: str) -> Tuple[str, Any]:
    """Update the emoji for a voice"""
    if not voice_name:
        return "❌ No voice selected", gr.update()
    
    if not new_emoji:
        return "❌ No emoji provided", gr.update()
    
    try:
        metadata = load_voice_metadata()
        if voice_name not in metadata:
            metadata[voice_name] = {}
        metadata[voice_name]["emoji"] = new_emoji
        save_voice_metadata(metadata)
        
        updated_choices = get_available_voices_with_metadata()
        return f"✅ Emoji updated to {new_emoji}", gr.update(choices=updated_choices)
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update()


def get_available_voices_with_metadata() -> List[tuple]:
    """Get voice choices with custom metadata (emojis)"""
    VOICE_REF_DIR.mkdir(exist_ok=True)
    metadata = load_voice_metadata()
    voices = []
    
    for f in VOICE_REF_DIR.glob("*.wav"):
        name = f.name
        stem = f.stem
        
        # Get emoji from metadata or auto-assign
        if name in metadata and "emoji" in metadata[name]:
            emoji = metadata[name]["emoji"]
        else:
            emoji = get_emoji_for_voice(stem)
        
        display_name = f"{emoji} {stem.title().replace('_', ' ')}"
        voices.append((display_name, name))
    
    voices.sort(key=lambda x: x[0])
    if not voices:
        voices = [("🔊 Reference", "reference.wav")]
    
    return voices


def preview_voice_emoji(filename: str) -> str:
    """Preview what emoji would be assigned to a filename"""
    if not filename:
        return "Upload a file to see emoji preview"
    
    emoji = get_emoji_for_voice(filename)
    suggestions = suggest_emoji_options(filename)
    
    return f"**Auto-assigned:** {emoji}\n\n**Suggestions:** {' '.join(suggestions)}"


# ============================================================================
# Wizard Generation
# ============================================================================

def generate_character_profile(concept: str) -> List[Any]:
    """Generate profile from concept using LLM"""
    if not API_KEY:
        return ["❌ API Key not found in config.env"] + [gr.update()] * 9

    if not concept:
        return ["❌ Please enter a concept"] + [gr.update()] * 9

    prompt = f"""You are an expert character designer. Create a detailed character profile based on this concept: "{concept}".
    
    Return ONLY a JSON object with this structure (no markdown, no explanation, just the JSON):
    {{
      "id": "short_id_lowercase_no_spaces",
      "name": "Full Character Name",
      "display_name": "Emoji FullName",
      "system_prompt": "Detailed roleplay instructions. Include: personality description, how they speak, their background, and behavioral guidelines. Make it at least 3-4 paragraphs for depth.",
      "personality_traits": ["trait1", "trait2", "trait3", "trait4", "trait5"],
      "initial_memories": ["memory1 - background fact", "memory2 - personality aspect", "memory3 - current situation"],
      "speech_patterns": ["pattern1", "pattern2", "pattern3"],
      "metadata": {{"setting": "where they are", "mood": "current emotional state", "background": "brief history"}},
      "allowed_tools": [],
      "default_voice": "reference.wav"
    }}
    
    Make the character deep, interesting, and well-developed. The system_prompt should be comprehensive enough for immersive roleplay.
    """

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": "x-ai/grok-4.1-fast",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=60
        )
        data = resp.json()['choices'][0]['message']['content']
        
        # Parse JSON (handle markdown fences)
        if "```json" in data:
            data = data.split("```json")[1].split("```")[0]
        elif "```" in data:
            data = data.split("```")[1].split("```")[0]
            
        profile = json.loads(data.strip())
        
        # Format speech patterns for display
        speech_patterns = profile.get('speech_patterns', [])
        speech_text = "\n".join(speech_patterns) if speech_patterns else ""
        
        return [
            f"✅ Generated: {profile.get('name', 'Character')}",
            profile.get('id', ''),
            profile.get('name', ''),
            profile.get('system_prompt', ''),
            profile.get('default_voice', 'reference.wav'),
            "\n".join(profile.get('personality_traits', [])),
            "\n".join(profile.get('initial_memories', [])),
            speech_text,
            profile.get('metadata', {}).get('setting', ''),
            json.dumps(profile.get('metadata', {}), indent=2),
            profile.get('allowed_tools', [])
        ]
        
    except json.JSONDecodeError as e:
        return [f"❌ JSON parsing error: {str(e)}"] + [gr.update()] * 10
    except Exception as e:
        return [f"❌ Error: {str(e)}"] + [gr.update()] * 10


# ============================================================================
# Character Management Functions
# ============================================================================

def get_character_list() -> List[str]:
    """Get list of character IDs"""
    return list(CHARACTER_MANAGER.characters.keys())


def get_character_display_list() -> List[tuple]:
    """Get list of (display_name, character_id) tuples with built-in indicator"""
    result = []
    for cid, c in CHARACTER_MANAGER.characters.items():
        if c.is_builtin:
            # Add lock emoji for built-in characters
            display = f"🔒 {c.display_name}"
        else:
            display = c.display_name
        result.append((display, cid))
    return result


def load_character_details(character_id: str) -> tuple:
    """Load character details for editing
    
    Returns tuple of form values. For built-in characters, the system_prompt
    includes a notice about where to edit.
    """
    char = CHARACTER_MANAGER.get_character(character_id)
    if not char:
        return "", "", "", "", "", "", "", "", "", []
    
    # Get speech patterns from character if available
    speech_patterns = getattr(char, 'speech_patterns', [])
    if not speech_patterns:
        speech_patterns = char.metadata.get('speech_patterns', [])
    speech_text = "\n".join(speech_patterns) if speech_patterns else ""
    
    # For built-in characters, add notice to system prompt display
    system_prompt = char.system_prompt
    if char.is_builtin:
        skill_path = char.metadata.get('skill_path', f'skills/{character_id}-*/SKILL.md')
        system_prompt = f"# 🔒 BUILT-IN CHARACTER (Read-Only)\n# Edit at: {skill_path}\n\n{system_prompt}"
    
    return (
        character_id,
        char.display_name,
        system_prompt,
        char.default_voice,
        "\n".join(char.personality_traits),
        "\n".join(char.initial_memories),
        speech_text,
        char.metadata.get("setting", ""),
        json.dumps(char.metadata, indent=2) if char.metadata else "{}",
        char.allowed_tools or []
    )


def save_character(char_id: str, display_name: str, system_prompt: str,
                  default_voice: str, traits: str, memories: str,
                  speech_patterns: str, setting: str, metadata_json: str, 
                  allowed_tools: List[str]) -> tuple:
    """Save or update a character. Returns (status_message, updated_character_list)
    
    Note: Built-in characters (loaded from skills/) cannot be modified here.
    Use the skills/ directory to edit built-in character definitions.
    """
    try:
        if not char_id or not char_id.strip():
            return "❌ Character ID is required", gr.update()
        if not display_name or not display_name.strip():
            return "❌ Display name is required", gr.update()
        
        # Sanitize character ID
        char_id = re.sub(r'[^\w]', '_', char_id.strip().lower())
        
        # Check if trying to modify a built-in (skill-based) character
        existing = CHARACTER_MANAGER.get_character(char_id)
        if existing and existing.is_builtin:
            return f"❌ '{display_name}' is a built-in character (from skills/). Edit skills/{char_id}-*/SKILL.md instead.", gr.update()
        
        # Parse inputs
        personality_traits = [t.strip() for t in traits.split("\n") if t.strip()]
        initial_memories = [m.strip() for m in memories.split("\n") if m.strip()]
        speech_pattern_list = [s.strip() for s in speech_patterns.split("\n") if s.strip()]
        
        try:
            metadata = json.loads(metadata_json) if metadata_json.strip() else {}
        except json.JSONDecodeError:
            return "❌ Invalid JSON in metadata", gr.update()
        
        # Store setting in metadata
        if setting and setting.strip():
            metadata["setting"] = setting.strip()
        
        # Store speech patterns in metadata (Character class may not have this field)
        if speech_pattern_list:
            metadata["speech_patterns"] = speech_pattern_list
        
        # Create character
        character = Character(
            id=char_id,
            name=display_name.strip().split(" ", 1)[-1] if " " in display_name else display_name.strip(),
            display_name=display_name.strip(),
            system_prompt=system_prompt,
            default_voice=default_voice if default_voice else "reference.wav",
            personality_traits=personality_traits,
            initial_memories=initial_memories,
            metadata=metadata,
            allowed_tools=allowed_tools,
            is_builtin=False  # User-created characters are not built-in
        )
        
        # Save to characters dict (user-created go to characters.json)
        CHARACTER_MANAGER.characters[char_id] = character
        CHARACTER_MANAGER._save_characters()
        
        # Return updated character list
        updated_choices = get_character_display_list()
        return f"✅ Character '{display_name}' saved successfully!", gr.update(choices=updated_choices)
        
    except Exception as e:
        return f"❌ Error saving character: {str(e)}", gr.update()


def save_character_yaml(character: Character, speech_patterns: List[str] = None):
    """Save character as YAML file for easy editing"""
    import yaml
    
    yaml_path = CHARACTERS_DIR / f"{character.id}.yaml"
    
    data = {
        'id': character.id,
        'name': character.name,
        'display_name': character.display_name,
        'default_voice': character.default_voice,
        'avatar': f"{character.id}.png",
        'system_prompt': character.system_prompt,
        'personality_traits': character.personality_traits,
        'speech_patterns': speech_patterns or [],
        'initial_memories': character.initial_memories,
        'allowed_tools': character.allowed_tools or [],
        'metadata': character.metadata
    }
    
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except ImportError:
        # Fallback if PyYAML not installed - write as formatted text
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"# Character Definition: {character.name}\n\n")
            f.write(f"id: {character.id}\n")
            f.write(f"name: {character.name}\n")
            f.write(f'display_name: "{character.display_name}"\n')
            f.write(f"default_voice: {character.default_voice}\n\n")
            f.write("system_prompt: |\n")
            for line in character.system_prompt.split('\n'):
                f.write(f"  {line}\n")
            f.write("\npersonality_traits:\n")
            for trait in character.personality_traits:
                f.write(f"  - {trait}\n")
            f.write("\ninitial_memories:\n")
            for mem in character.initial_memories:
                f.write(f"  - {mem}\n")
            f.write("\nallowed_tools:\n")
            for tool in (character.allowed_tools or []):
                f.write(f"  - {tool}\n")
    except Exception as e:
        print(f"[Character] Error saving YAML: {e}")


def delete_character(character_id: str) -> tuple:
    """Delete a character. Returns (status_message, updated_character_list)
    
    Note: Built-in characters (from skills/) cannot be deleted from here.
    """
    try:
        if character_id not in CHARACTER_MANAGER.characters:
            return "❌ Character not found", gr.update()
        
        char = CHARACTER_MANAGER.characters[character_id]
        
        # Protect built-in characters
        if char.is_builtin:
            return f"❌ '{char.display_name}' is a built-in character (from skills/). Cannot delete built-in characters.", gr.update()
        
        char_name = char.display_name
        
        # Remove from dict
        del CHARACTER_MANAGER.characters[character_id]
        CHARACTER_MANAGER._save_characters()
        
        # Delete conversations directory (keep conversations for history)
        # conv_dir = CONVERSATIONS_DIR / character_id
        # if conv_dir.exists():
        #     shutil.rmtree(conv_dir)
        
        # Delete session file
        session_file = SESSIONS_DIR / f"{character_id}_session.json"
        if session_file.exists():
            session_file.unlink()
        
        # Return updated character list
        updated_choices = get_character_display_list()
        return f"✅ Character '{char_name}' deleted successfully!", gr.update(choices=updated_choices)
        
    except Exception as e:
        return f"❌ Error deleting character: {str(e)}", gr.update()


def create_new_character_form() -> tuple:
    """Reset form for creating a new character"""
    return (
        "",  # char_id
        "",  # display_name
        "",  # system_prompt
        "reference.wav",  # voice
        "",  # traits
        "",  # memories
        "",  # speech_patterns
        "",  # setting
        "{}",  # metadata
        []   # tools
    )


def duplicate_character(source_id: str) -> tuple:
    """Duplicate an existing character as a starting point"""
    if not source_id:
        return ("❌ No character selected",) + tuple([gr.update()] * 9)
    
    char = CHARACTER_MANAGER.get_character(source_id)
    if not char:
        return ("❌ Character not found",) + tuple([gr.update()] * 9)
    
    # Create new ID
    new_id = f"{source_id}_copy"
    counter = 1
    while new_id in CHARACTER_MANAGER.characters:
        new_id = f"{source_id}_copy{counter}"
        counter += 1
    
    speech_patterns = char.metadata.get('speech_patterns', [])
    speech_text = "\n".join(speech_patterns) if speech_patterns else ""
    
    return (
        f"✅ Duplicated '{char.display_name}' - modify and save as new character",
        new_id,
        f"{char.display_name} (Copy)",
        char.system_prompt,
        char.default_voice,
        "\n".join(char.personality_traits),
        "\n".join(char.initial_memories),
        speech_text,
        char.metadata.get("setting", ""),
        json.dumps(char.metadata, indent=2),
        char.allowed_tools or []
    )


# ============================================================================
# Memory Management Functions
# ============================================================================

def get_character_memory_stats(character_id: str) -> str:
    """Get memory statistics for a character"""
    if not character_id:
        return "*Select a character to view memory stats*"
    
    try:
        MEMORY_MANAGER.activate_character(character_id)
        stats = MEMORY_MANAGER.get_stats(character_id)
        
        return f"""**Character:** {CHARACTER_MANAGER.get_character(character_id).display_name}
**Total Interactions:** {stats.get('total_interactions', 0)}
**Episodic Memories:** {stats.get('episodic_count', 0)}
**Semantic Memories:** {stats.get('semantic_count', 0)}
**Procedural Memories:** {stats.get('procedural_count', 0)}
**Embedding Model:** {stats.get('embedding_model', 'unknown')}
**Recent Activity:** {stats.get('last_interaction', 'Never')}"""
        
    except Exception as e:
        return f"❌ Error loading stats: {str(e)}"


def get_episodic_memories(character_id: str) -> str:
    """Get episodic memories as markdown table"""
    if not character_id:
        return "| Timestamp | Content | Importance |\n|-----------|---------|------------|"
    
    try:
        MEMORY_MANAGER.activate_character(character_id)
        memories = MEMORY_MANAGER.storage.get_memories_by_character(
            character_id=character_id,
            memory_type='episodic',
            limit=50
        )
        
        if not memories:
            return "| Timestamp | Content | Importance |\n|-----------|---------|------------|\n| *No episodic memories yet* | | |"
        
        rows = []
        for mem in memories:
            timestamp = mem.created_at.strftime("%Y-%m-%d %H:%M") if mem.created_at else 'Unknown'
            content = mem.content[:80].replace('\n', ' ')
            importance = f"{mem.importance_score:.2f}"
            rows.append(f"| {timestamp} | {content}... | {importance} |")
        
        header = "| Timestamp | Content | Importance |\n|-----------|---------|------------|"
        return header + "\n" + "\n".join(rows)
        
    except Exception as e:
        return f"Error: {str(e)}"


def get_semantic_memories(character_id: str) -> str:
    """Get semantic memories as markdown list"""
    if not character_id:
        return "*No semantic memories*"
    
    try:
        MEMORY_MANAGER.activate_character(character_id)
        memories = MEMORY_MANAGER.storage.get_memories_by_character(
            character_id=character_id,
            memory_type='semantic',
            limit=100
        )
        
        if not memories:
            return "*No semantic memories yet*"
        
        items = []
        for i, mem in enumerate(memories, 1):
            importance = mem.importance_score
            content = mem.content
            items.append(f"{i}. **[{importance:.2f}]** {content}")
        
        return "\n".join(items)
        
    except Exception as e:
        return f"Error: {str(e)}"


def get_procedural_memories(character_id: str) -> str:
    """Get procedural memories as markdown list"""
    if not character_id:
        return "*No procedural memories*"
    
    try:
        MEMORY_MANAGER.activate_character(character_id)
        memories = MEMORY_MANAGER.storage.get_memories_by_character(
            character_id=character_id,
            memory_type='procedural',
            limit=50
        )
        
        if not memories:
            return "*No procedural memories yet*"
        
        items = []
        for i, mem in enumerate(memories, 1):
            content = mem.content
            items.append(f"{i}. {content}")
        
        return "\n".join(items)
        
    except Exception as e:
        return f"Error: {str(e)}"


def clear_character_memories(character_id: str) -> str:
    """Clear all memories for a character"""
    if not character_id:
        return "❌ No character selected"
    
    try:
        char_name = CHARACTER_MANAGER.get_character(character_id).display_name
        MEMORY_MANAGER.clear_character_memory(character_id)
        return f"✅ All memories cleared for '{char_name}'"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def reinitialize_base_memories(character_id: str) -> str:
    """Reinitialize the default semantic and procedural memories for a character"""
    if not character_id:
        return "❌ No character selected"
    
    try:
        character = CHARACTER_MANAGER.get_character(character_id)
        if not character:
            return "❌ Character not found"
        
        MEMORY_MANAGER.activate_character(character_id)
        
        added_semantic = 0
        added_procedural = 0
        
        # Add semantic memories (facts/background)
        for mem in character.initial_memories:
            MEMORY_MANAGER.add_semantic_memory(character_id, mem, importance=0.9)
            added_semantic += 1
        
        # Add procedural memories (personality traits)
        for trait in character.personality_traits:
            MEMORY_MANAGER.add_procedural_memory(character_id, f"Personality: {trait}")
            added_procedural += 1
        
        return f"✅ Reinitialized '{character.display_name}': {added_semantic} semantic, {added_procedural} procedural memories added"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def add_semantic_memory(character_id: str, content: str, importance: float) -> str:
    """Add a new semantic memory for a character"""
    if not character_id:
        return "❌ No character selected"
    if not content or not content.strip():
        return "❌ Memory content is required"
    
    try:
        MEMORY_MANAGER.activate_character(character_id)
        MEMORY_MANAGER.add_semantic_memory(character_id, content.strip(), importance=importance)
        char_name = CHARACTER_MANAGER.get_character(character_id).display_name
        return f"✅ Semantic memory added for '{char_name}'"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def add_episodic_memory(character_id: str, content: str) -> str:
    """Add a new episodic memory for a character"""
    if not character_id:
        return "❌ No character selected"
    if not content or not content.strip():
        return "❌ Memory content is required"
    
    try:
        MEMORY_MANAGER.activate_character(character_id)
        MEMORY_MANAGER.add_interaction(character_id, content.strip(), "[Manual entry]")
        char_name = CHARACTER_MANAGER.get_character(character_id).display_name
        return f"✅ Episodic memory added for '{char_name}'"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def add_procedural_memory(character_id: str, content: str) -> str:
    """Add a new procedural memory for a character"""
    if not character_id:
        return "❌ No character selected"
    if not content or not content.strip():
        return "❌ Memory content is required"
    
    try:
        MEMORY_MANAGER.activate_character(character_id)
        MEMORY_MANAGER.add_procedural_memory(character_id, content.strip())
        char_name = CHARACTER_MANAGER.get_character(character_id).display_name
        return f"✅ Procedural memory added for '{char_name}'"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def delete_memory_by_index(character_id: str, memory_type: str, index: int) -> str:
    """Delete a specific memory by its index (1-based)"""
    if not character_id:
        return "❌ No character selected"
    if index < 1:
        return "❌ Invalid index (must be 1 or greater)"
    
    try:
        MEMORY_MANAGER.activate_character(character_id)
        memories = MEMORY_MANAGER.storage.get_memories_by_character(
            character_id=character_id,
            memory_type=memory_type,
            limit=1000
        )
        
        if index > len(memories):
            return f"❌ Index {index} out of range (only {len(memories)} memories)"
        
        memory_to_delete = memories[index - 1]
        MEMORY_MANAGER.storage.delete_memory(memory_to_delete.id)
        
        return f"✅ Deleted {memory_type} memory #{index}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def generate_graph_ui(character_id: str) -> Tuple[str, str]:
    """Generate memory graph and return HTML content + status"""
    if not character_id:
        return None, "❌ Select a character first"
    
    try:
        path = MEMORY_MANAGER.generate_memory_graph(character_id)
        if not path:
            return None, "❌ Graph generation failed or PyVis not installed"
        return path, "✅ Graph generated"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def get_memory_choices(character_id: str, memory_type: str) -> List[tuple]:
    """Get memories as dropdown choices for editing"""
    if not character_id:
        return []
        
    try:
        MEMORY_MANAGER.activate_character(character_id)
        memories = MEMORY_MANAGER.storage.get_memories_by_character(
            character_id=character_id,
            memory_type=memory_type,
            limit=100
        )
        return [(f"[{m.importance_score:.2f}] {m.content[:40]}...", m.id) for m in memories]
    except:
        return []


def load_memory_for_edit(character_id: str, memory_id: str) -> Tuple[str, float]:
    """Load memory content for editing"""
    if not memory_id:
        return "", 0.5
        
    mem = MEMORY_MANAGER.get_memory(memory_id)
    if mem:
        return mem.content, mem.importance_score
    return "", 0.5


def update_memory_ui(character_id: str, memory_id: str, content: str, importance: float) -> str:
    """Update memory content"""
    if not memory_id:
        return "❌ No memory selected"
        
    try:
        success = MEMORY_MANAGER.update_memory(memory_id, content, importance)
        if success:
            return "✅ Memory updated successfully"
        return "❌ Update failed"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def delete_memory_from_ui(character_id: str, memory_id: str) -> str:
    """Delete a specific memory"""
    if not memory_id:
        return "❌ No memory selected"
        
    try:
        success = MEMORY_MANAGER.storage.delete_memory(memory_id)
        if success:
            return "✅ Memory deleted"
        return "❌ Delete failed"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def export_memories_ui(character_id: str) -> str:
    """Export memories to JSON file"""
    if not character_id:
        return None
        
    try:
        data = MEMORY_MANAGER.export_all_memories(character_id)
        
        filename = f"{character_id}_memories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_dir = SESSIONS_DIR / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        return str(path)
    except Exception as e:
        print(f"Export error: {e}")
        return None


def import_memories_ui(file_obj) -> str:
    """Import memories from JSON"""
    if not file_obj:
        return "❌ No file uploaded"
        
    try:
        with open(file_obj.name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        mem_count, sum_count = MEMORY_MANAGER.import_memories(data)
        return f"✅ Imported {mem_count} memories, {sum_count} summaries"
    except Exception as e:
        return f"❌ Import error: {str(e)}"


# ============================================================================
# Conversation Management Functions
# ============================================================================

def list_conversations(character_id: str) -> List[tuple]:
    """List all conversations for a character"""
    if not character_id:
        return []
    
    char_dir = CONVERSATIONS_DIR / character_id
    if not char_dir.exists():
        return []
    
    conversations = []
    for f in char_dir.glob("*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                conversations.append((
                    data.get('title', 'Untitled'),
                    f.stem,
                    data.get('updated_at', '')[:10],
                    len(data.get('history', []))
                ))
        except:
            pass
    
    conversations.sort(key=lambda x: x[2], reverse=True)
    return conversations


def get_conversation_table(character_id: str) -> str:
    """Get conversations as markdown table"""
    convs = list_conversations(character_id)
    
    if not convs:
        return "| Title | ID | Date | Messages |\n|-------|----|----|----------|"
    
    header = "| Title | ID | Date | Messages |\n|-------|-----|------|----------|"
    rows = [f"| {title[:30]} | {cid[:15]} | {date} | {msgs} |" 
            for title, cid, date, msgs in convs]
    
    return header + "\n" + "\n".join(rows)


def view_conversation(character_id: str, conversation_id: str) -> str:
    """View conversation history"""
    if not character_id or not conversation_id:
        return "*No conversation selected*"
    
    try:
        filepath = CONVERSATIONS_DIR / character_id / f"{conversation_id}.json"
        if not filepath.exists():
            return "❌ Conversation not found"
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        title = data.get('title', 'Untitled')
        history = data.get('history', [])
        
        output = [f"# {title}\n"]
        for item in history:
            if isinstance(item, dict):
                role = item.get('role', 'user')
                content = item.get('content', '')
                if role == 'user':
                    output.append(f"**User:** {content}\n")
                else:
                    output.append(f"**AI:** {content}\n")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                user_msg, ai_msg = item[0], item[1]
                if user_msg:
                    output.append(f"**User:** {user_msg}\n")
                if ai_msg:
                    output.append(f"**AI:** {ai_msg}\n")
            output.append("---\n")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


def export_conversation(character_id: str, conversation_id: str) -> str:
    """Export conversation as JSON"""
    if not character_id or not conversation_id:
        return "{}"
    
    try:
        filepath = CONVERSATIONS_DIR / character_id / f"{conversation_id}.json"
        if filepath.exists():
            return filepath.read_text(encoding='utf-8')
        return "{}"
    except:
        return "{}"


def delete_conversation(character_id: str, conversation_id: str) -> str:
    """Delete a conversation"""
    if not character_id or not conversation_id:
        return "❌ No conversation selected"
    
    try:
        filepath = CONVERSATIONS_DIR / character_id / f"{conversation_id}.json"
        if filepath.exists():
            filepath.unlink()
            return f"✅ Conversation deleted"
        return "❌ Conversation not found"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def rename_conversation(character_id: str, conversation_id: str, new_title: str) -> str:
    """Rename a conversation"""
    if not character_id or not conversation_id:
        return "❌ No conversation selected"
    if not new_title or not new_title.strip():
        return "❌ New title is required"
    
    try:
        filepath = CONVERSATIONS_DIR / character_id / f"{conversation_id}.json"
        if not filepath.exists():
            return "❌ Conversation not found"
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['title'] = new_title.strip()
        data['updated_at'] = datetime.now().isoformat()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return f"✅ Conversation renamed to '{new_title.strip()}'"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# Skills Management Functions
# ============================================================================

def list_available_skills() -> List[Dict[str, Any]]:
    """List all available skills with metadata"""
    skills = []
    
    if not SKILLS_DIR.exists():
        return skills
    
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue
        
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        
        try:
            content = skill_file.read_text(encoding='utf-8')
            
            # Parse frontmatter
            import re
            pattern = r'^---\s*\n(.*?)\n---\s*\n'
            match = re.match(pattern, content, re.DOTALL)
            
            if match:
                try:
                    import yaml
                    frontmatter = yaml.safe_load(match.group(1)) or {}
                except:
                    frontmatter = {}
            else:
                frontmatter = {}
            
            skills.append({
                'dir_name': skill_dir.name,
                'path': str(skill_file),
                'id': frontmatter.get('id', skill_dir.name),
                'name': frontmatter.get('name', skill_dir.name),
                'display_name': frontmatter.get('display_name', skill_dir.name),
                'description': frontmatter.get('description', 'No description'),
                'allowed_tools': frontmatter.get('allowed_tools', []),
                'has_scripts': (skill_dir / 'scripts').exists(),
                'has_references': (skill_dir / 'references').exists(),
            })
        except Exception as e:
            print(f"[Skills] Error loading {skill_dir.name}: {e}")
    
    return skills


def get_skill_content(skill_path: str) -> str:
    """Get full content of a skill file"""
    try:
        path = Path(skill_path)
        if path.exists():
            return path.read_text(encoding='utf-8')
        return "*Skill file not found*"
    except Exception as e:
        return f"Error reading skill: {e}"


def get_skill_files(skill_dir_name: str) -> str:
    """Get list of files in a skill directory as markdown"""
    skill_path = SKILLS_DIR / skill_dir_name
    if not skill_path.exists():
        return "*Skill directory not found*"
    
    output = [f"## {skill_dir_name}/\n"]
    
    # List all files recursively
    for item in sorted(skill_path.rglob('*')):
        if item.is_file():
            rel_path = item.relative_to(skill_path)
            size = item.stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            else:
                size_str = f"{size/1024:.1f}KB"
            output.append(f"- `{rel_path}` ({size_str})")
    
    return "\n".join(output) if len(output) > 1 else "*Empty skill directory*"


def get_skills_dropdown_choices() -> List[tuple]:
    """Get skills as dropdown choices"""
    skills = list_available_skills()
    return [(f"📜 {s['display_name']}", s['path']) for s in skills]


def get_skills_summary_table() -> str:
    """Get skills as markdown table"""
    skills = list_available_skills()

    if not skills:
        return "*No skills found in skills/ directory*"

    header = "| Skill | Character ID | Tools | Scripts | References |\n|-------|--------------|-------|---------|------------|\n"
    rows = []

    for s in skills:
        tools = len(s.get('allowed_tools', []))
        scripts = "✅" if s.get('has_scripts') else "❌"
        refs = "✅" if s.get('has_references') else "❌"
        rows.append(f"| {s['display_name']} | `{s['id']}` | {tools} | {scripts} | {refs} |")

    return header + "\n".join(rows)


def get_pending_skills() -> List[Dict[str, Any]]:
    """Get skills from sessions/files that haven't been installed"""
    pending = []
    files_dir = SESSIONS_DIR / "files"

    if not files_dir.exists():
        return pending

    for f in files_dir.glob("*_SKILL.md"):
        try:
            content = f.read_text(encoding='utf-8')

            # Parse frontmatter
            pattern = r'^---\s*\n(.*?)\n---\s*\n'
            match = re.match(pattern, content, re.DOTALL)

            if match:
                try:
                    import yaml
                    frontmatter = yaml.safe_load(match.group(1)) or {}
                except:
                    frontmatter = {}
            else:
                frontmatter = {}

            skill_id = frontmatter.get('name', f.stem.replace('_SKILL', ''))

            # Check if already installed
            if not (SKILLS_DIR / skill_id).exists():
                pending.append({
                    'file': str(f),
                    'filename': f.name,
                    'id': skill_id,
                    'name': frontmatter.get('name', skill_id),
                    'description': frontmatter.get('description', 'No description')
                })
        except Exception as e:
            print(f"[Skills] Error reading {f.name}: {e}")

    return pending


def get_pending_skills_table() -> str:
    """Get pending skills as markdown table"""
    pending = get_pending_skills()

    if not pending:
        return "*No pending skills in sessions/files/*"

    header = "| File | ID | Description |\n|------|-----|-------------|\n"
    rows = []

    for s in pending:
        desc = s['description'][:50] + "..." if len(s['description']) > 50 else s['description']
        rows.append(f"| `{s['filename']}` | {s['id']} | {desc} |")

    return header + "\n".join(rows)


def get_pending_skills_choices() -> List[Tuple[str, str]]:
    """Get pending skills as dropdown choices"""
    pending = get_pending_skills()
    return [(f"{s['name']} ({s['filename']})", s['file']) for s in pending]


def install_skill_from_file(skill_file: str) -> str:
    """Install a skill from sessions/files to skills/"""
    if not skill_file:
        return "❌ No skill file selected"

    try:
        skill_path = Path(skill_file)
        if not skill_path.exists():
            return f"❌ File not found: {skill_file}"

        content = skill_path.read_text(encoding='utf-8')

        # Parse frontmatter for name
        pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(pattern, content, re.DOTALL)

        if match:
            try:
                import yaml
                frontmatter = yaml.safe_load(match.group(1)) or {}
            except:
                frontmatter = {}
        else:
            frontmatter = {}

        skill_id = frontmatter.get('name', skill_path.stem.replace('_SKILL', ''))

        # Create skill directory
        skill_dir = SKILLS_DIR / skill_id
        skill_dir.mkdir(exist_ok=True)

        # Copy SKILL.md
        (skill_dir / "SKILL.md").write_text(content, encoding='utf-8')

        # Look for matching references and scripts
        base_name = skill_path.stem.replace('_SKILL', '')
        files_dir = skill_path.parent

        refs_file = files_dir / f"{base_name}_references.md"
        if refs_file.exists():
            refs_dir = skill_dir / "references"
            refs_dir.mkdir(exist_ok=True)
            shutil.copy2(refs_file, refs_dir / "knowledge.md")

        script_files = list(files_dir.glob(f"{base_name}_*.py"))
        if script_files:
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            for sf in script_files:
                shutil.copy2(sf, scripts_dir / sf.name.replace(f"{base_name}_", ""))

        return f"✅ Skill '{skill_id}' installed to skills/{skill_id}/"

    except Exception as e:
        return f"❌ Error: {str(e)}"


def create_new_skill(skill_id: str, display_name: str, description: str,
                    system_prompt: str, allowed_tools: List[str]) -> str:
    """Create a new skill from scratch"""
    if not skill_id or not skill_id.strip():
        return "❌ Skill ID is required"

    skill_id = re.sub(r'[^\w-]', '-', skill_id.strip().lower())

    skill_dir = SKILLS_DIR / skill_id
    if skill_dir.exists():
        return f"❌ Skill '{skill_id}' already exists"

    try:
        skill_dir.mkdir(parents=True)

        # Create SKILL.md
        tools_yaml = "\n  - ".join(allowed_tools) if allowed_tools else ""
        if tools_yaml:
            tools_yaml = "\n  - " + tools_yaml

        skill_content = f"""---
name: {skill_id}
description: {description or 'Custom skill'}
---

# {display_name or skill_id.title()}

{system_prompt or 'Add skill instructions here...'}

## Configuration

allowed_tools:{tools_yaml if tools_yaml else ' []'}

## Usage

This skill is loaded when the character is activated.
"""

        (skill_dir / "SKILL.md").write_text(skill_content, encoding='utf-8')

        # Create empty directories
        (skill_dir / "references").mkdir(exist_ok=True)
        (skill_dir / "scripts").mkdir(exist_ok=True)

        return f"✅ Skill '{skill_id}' created at skills/{skill_id}/"

    except Exception as e:
        return f"❌ Error: {str(e)}"


def delete_skill(skill_dir_name: str) -> str:
    """Delete a skill directory"""
    if not skill_dir_name:
        return "❌ No skill selected"

    skill_path = SKILLS_DIR / skill_dir_name
    if not skill_path.exists():
        return f"❌ Skill '{skill_dir_name}' not found"

    try:
        shutil.rmtree(skill_path)
        return f"✅ Skill '{skill_dir_name}' deleted"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def save_skill_content(skill_path: str, content: str) -> str:
    """Save edited skill content"""
    if not skill_path:
        return "❌ No skill selected"

    try:
        Path(skill_path).write_text(content, encoding='utf-8')
        return "✅ Skill saved"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_skill_dir_choices() -> List[Tuple[str, str]]:
    """Get skill directories for dropdown"""
    skills = list_available_skills()
    return [(f"📜 {s['display_name']}", s['dir_name']) for s in skills]


# ============================================================================
# Gradio UI
# ============================================================================

def create_management_ui():
    
    with gr.Blocks(title="Character & Memory Manager", theme=create_dark_theme()) as app:
        
        gr.Markdown(f"""
        # 🎭 Character & Memory Manager
        ### Create Characters • Manage Voices • Edit Memories
        *Running on {PLATFORM.title()}{" (WSL)" if IS_WSL else ""}*
        """)
        
        with gr.Tabs():
            
            # ==================== CHARACTER MANAGEMENT ====================
            with gr.Tab("🎭 Characters"):
                
                # AI Wizard Section
                with gr.Accordion("✨ AI Character Generator (Wizard)", open=False):
                    gr.Markdown("""
                    ### Create a character from a concept
                    *Describe your character idea and let AI generate a complete profile*
                    """)
                    with gr.Row():
                        wizard_concept = gr.Textbox(
                            label="Character Concept", 
                            placeholder="e.g., A grumpy cyberpunk taxi driver who secretly writes poetry",
                            lines=2,
                            scale=4
                        )
                        wizard_btn = gr.Button("✨ Generate Profile", variant="primary", scale=1)
                    wizard_status = gr.Textbox(label="Status", interactive=False)
                    gr.Markdown("*Generated profiles appear in the Editor below*")
                
                with gr.Row():
                    # Left - Character List
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Existing Characters")
                        character_list = gr.Dropdown(
                            choices=get_character_display_list(),
                            label="Select Character to Edit",
                            interactive=True
                        )
                        
                        with gr.Row():
                            new_char_btn = gr.Button("➕ New", size="sm")
                            duplicate_btn = gr.Button("📋 Duplicate", size="sm")
                        
                        gr.Markdown("---")
                        delete_char_btn = gr.Button("🗑️ Delete Character", variant="stop")
                        delete_status = gr.Textbox(label="Status", interactive=False)
                    
                    # Right - Character Editor
                    with gr.Column(scale=2):
                        gr.Markdown("### ✏️ Character Editor")
                        
                        with gr.Row():
                            char_id_input = gr.Textbox(
                                label="Character ID",
                                placeholder="lowercase_no_spaces",
                                info="Unique identifier (auto-generated from name if empty)",
                                scale=1
                            )
                            display_name_input = gr.Textbox(
                                label="Display Name",
                                placeholder="🎭 Character Name",
                                info="Include emoji for visual identification",
                                scale=2
                            )
                        
                        system_prompt_input = gr.Textbox(
                            label="System Prompt",
                            placeholder="You are [Character Name], a [description]...\n\nPERSONALITY:\n- Trait 1\n- Trait 2\n\nHOW YOU SPEAK:\n- Pattern 1\n- Pattern 2\n\nSETTING:\n[Describe the scenario]",
                            lines=8,
                            info="The main instruction that defines your character's behavior"
                        )
                        
                        with gr.Row():
                            voice_input = gr.Dropdown(
                                choices=get_available_voices_with_metadata(),
                                label="Default Voice",
                                value="reference.wav",
                                interactive=True,
                                allow_custom_value=True,
                                scale=2
                            )
                            refresh_voices_btn = gr.Button("🔄", scale=0, size="sm")
                        
                        with gr.Row():
                            with gr.Column():
                                traits_input = gr.Textbox(
                                    label="Personality Traits (one per line)",
                                    placeholder="Intelligent\nCurious\nBrave\nWitty",
                                    lines=4
                                )
                            with gr.Column():
                                memories_input = gr.Textbox(
                                    label="Initial Memories (one per line)",
                                    placeholder="I am a wizard at Hogwarts\nI love learning new spells\nMy best friends are...",
                                    lines=4
                                )
                        
                        speech_patterns_input = gr.Textbox(
                            label="Speech Patterns (one per line)",
                            placeholder="Uses British expressions\nSpeaks formally when nervous\nMakes book references",
                            lines=3,
                            info="How the character speaks - helps with voice consistency"
                        )
                        
                        # Tools Selection
                        available_tools = [t['function']['name'] for t in REGISTRY.list_tools()]
                        tools_input = gr.CheckboxGroup(
                            choices=available_tools,
                            label="Allowed Tools",
                            info="Select tools this character can use (leave empty for pure roleplay)"
                        )

                        with gr.Row():
                            setting_input = gr.Textbox(
                                label="Setting/Location",
                                placeholder="e.g., Hog's Head pub, evening",
                                scale=1
                            )
                            metadata_input = gr.Textbox(
                                label="Metadata (JSON)",
                                placeholder='{"mood": "neutral", "age": 25}',
                                scale=2
                            )
                        
                        with gr.Row():
                            save_char_btn = gr.Button("💾 Save Character", variant="primary", size="lg", scale=2)
                            save_status = gr.Textbox(label="Status", interactive=False, scale=1)
            
            # ==================== VOICE MANAGEMENT ====================
            with gr.Tab("🎤 Voices"):
                gr.Markdown("""
                ### Voice Reference Management
                Upload voice samples for TTS cloning. WAV format required (~5-30 seconds of clear speech).
                """)
                
                with gr.Row():
                    # Left - Upload
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Upload New Voice")
                        
                        voice_upload = gr.File(
                            label="Drop WAV file here or click to upload",
                            file_types=[".wav"],
                            type="filepath"
                        )
                        
                        with gr.Row():
                            voice_custom_name = gr.Textbox(
                                label="Custom Name (optional)",
                                placeholder="my_voice",
                                scale=2
                            )
                            voice_custom_emoji = gr.Textbox(
                                label="Emoji",
                                placeholder="🎤",
                                max_lines=1,
                                scale=1
                            )
                        
                        emoji_preview = gr.Markdown("*Upload a file to see emoji suggestions*")
                        
                        upload_voice_btn = gr.Button("📤 Upload Voice", variant="primary")
                        upload_status = gr.Textbox(label="Status", interactive=False)
                        
                        gr.Markdown("---")
                        gr.Markdown("""
                        **Tips for good voice samples:**
                        - Use clear audio without background noise
                        - 5-30 seconds of natural speech
                        - WAV format, 16kHz or 22kHz sample rate
                        - Consistent volume and tone
                        """)
                    
                    # Right - Manage
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Existing Voices")
                        
                        voice_list = gr.Dropdown(
                            choices=get_available_voices_with_metadata(),
                            label="Select Voice",
                            interactive=True
                        )
                        
                        gr.Markdown("#### Update Emoji")
                        with gr.Row():
                            new_emoji_input = gr.Textbox(
                                label="New Emoji",
                                placeholder="🎤",
                                max_lines=1,
                                scale=1
                            )
                            update_emoji_btn = gr.Button("Update", size="sm", scale=1)
                        
                        emoji_suggestions = gr.Markdown("")
                        
                        gr.Markdown("---")
                        
                        delete_voice_btn = gr.Button("🗑️ Delete Voice", variant="stop")
                        voice_manage_status = gr.Textbox(label="Status", interactive=False)
                        
                        gr.Markdown("---")
                        gr.Markdown("""
                        **Auto-assigned emojis based on name:**
                        - Names with "wizard/gandalf" → 🧙
                        - Names with "female/soft" → 🌸
                        - Names with "male/deep" → 🎤
                        - Region names → 🇬🇧 🇺🇸 etc.
                        """)
            
            # ==================== MEMORY MANAGEMENT ====================
            with gr.Tab("🧠 Memories"):
                with gr.Row():
                    # Left - Controls
                    with gr.Column(scale=1):
                        memory_char_select = gr.Dropdown(
                            choices=get_character_display_list(),
                            label="Select Character",
                            interactive=True
                        )
                        memory_stats = gr.Markdown("*Select a character*")
                        refresh_stats_btn = gr.Button("🔄 Refresh Stats")
                        
                        gr.Markdown("---")
                        gr.Markdown("### ➕ Add New Memory")
                        new_memory_content = gr.Textbox(
                            label="Memory Content",
                            placeholder="Enter memory content...",
                            lines=2
                        )
                        new_memory_importance = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                            label="Importance (for semantic)"
                        )
                        with gr.Row():
                            add_semantic_btn = gr.Button("💡 Semantic", size="sm")
                            add_episodic_btn = gr.Button("📚 Episodic", size="sm")
                            add_procedural_btn = gr.Button("🔧 Procedural", size="sm")
                        
                        gr.Markdown("---")
                        gr.Markdown("### 🗑️ Delete Memory")
                        delete_memory_type = gr.Dropdown(
                            choices=[("📚 Episodic", "episodic"), ("💡 Semantic", "semantic"), ("🔧 Procedural", "procedural")],
                            label="Memory Type",
                            value="episodic"
                        )
                        delete_memory_index = gr.Number(
                            label="Memory # to Delete",
                            value=1,
                            minimum=1,
                            precision=0
                        )
                        delete_memory_btn = gr.Button("🗑️ Delete Memory", size="sm")
                        
                        gr.Markdown("---")
                        reinit_memories_btn = gr.Button("🔄 Reinitialize Base Memories", variant="secondary")
                        clear_memories_btn = gr.Button("🗑️ Clear ALL Memories", variant="stop")
                        memory_status = gr.Textbox(label="Status", interactive=False)
                    
                    # Right - Memory Display
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("📚 Episodic"):
                                episodic_display = gr.Markdown("*No memories*")
                                refresh_episodic_btn = gr.Button("🔄 Refresh")
                            
                            with gr.Tab("💡 Semantic"):
                                semantic_display = gr.Markdown("*No memories*")
                                refresh_semantic_btn = gr.Button("🔄 Refresh")
                            
                            with gr.Tab("🔧 Procedural"):
                                procedural_display = gr.Markdown("*No procedural memories*")
                                refresh_procedural_btn = gr.Button("🔄 Refresh")
                            
                            with gr.Tab("🕸️ Graph"):
                                graph_status = gr.Textbox(label="Status", interactive=False)
                                generate_graph_btn = gr.Button("🕸️ Generate Graph")
                                graph_file = gr.File(label="Interactive Graph (HTML)", interactive=False)

                        gr.Markdown("### ✏️ Edit Memory")
                        with gr.Row():
                            edit_memory_type = gr.Dropdown(
                                choices=[("📚 Episodic", "episodic"), ("💡 Semantic", "semantic"), ("🔧 Procedural", "procedural")],
                                label="Type to Edit",
                                value="episodic"
                            )
                            edit_memory_select = gr.Dropdown(
                                label="Select Memory to Edit",
                                interactive=True
                            )
                            refresh_edit_list_btn = gr.Button("🔄", size="sm")
                        
                        edit_content = gr.Textbox(label="Content", lines=3)
                        edit_importance = gr.Slider(minimum=0.0, maximum=1.0, label="Importance")
                        with gr.Row():
                            update_memory_btn = gr.Button("💾 Update Memory", variant="primary")
                            delete_mem_ui_btn = gr.Button("🗑️ Delete Memory", variant="stop")
                        
                        gr.Markdown("---")
                        gr.Markdown("### 📤 Import / Export")
                        with gr.Row():
                            with gr.Column():
                                export_mem_btn = gr.Button("📤 Export All Memories (JSON)")
                                export_file = gr.File(label="Download Export", interactive=False, height=100)
                            with gr.Column():
                                import_file = gr.File(label="Upload JSON Backup", height=100)
                                import_mem_btn = gr.Button("📥 Restore/Import Memories")
            
            # ==================== CONVERSATION MANAGEMENT ====================
            with gr.Tab("💬 Conversations"):
                with gr.Row():
                    # Left - Conversation List
                    with gr.Column(scale=1):
                        conv_char_select = gr.Dropdown(
                            choices=get_character_display_list(),
                            label="Select Character",
                            interactive=True
                        )
                        conv_table = gr.Markdown("*Select a character*")
                        refresh_convs_btn = gr.Button("🔄 Refresh List")
                        
                        conversation_id_input = gr.Textbox(
                            label="Conversation ID",
                            placeholder="Paste ID from table above"
                        )
                        
                        with gr.Row():
                            view_conv_btn = gr.Button("👁️ View", scale=1)
                            export_conv_btn = gr.Button("📥 Export", scale=1)
                            delete_conv_btn = gr.Button("🗑️ Delete", scale=1)
                        
                        gr.Markdown("---")
                        gr.Markdown("### ✏️ Rename Conversation")
                        new_title_input = gr.Textbox(
                            label="New Title",
                            placeholder="Enter new title..."
                        )
                        rename_conv_btn = gr.Button("✏️ Rename", size="sm")
                        conv_status = gr.Textbox(label="Status", interactive=False)
                    
                    # Right - Conversation Viewer
                    with gr.Column(scale=2):
                        conv_display = gr.Markdown("*Select and view a conversation*")
                        
                        with gr.Accordion("📥 Export JSON", open=False):
                            conv_export = gr.Textbox(
                                label="JSON Content",
                                lines=10,
                                interactive=False
                            )
                            download_conv_btn = gr.Button("📁 Download JSON File", size="sm")
                            conv_file_download = gr.File(label="Download", visible=False)

            # ==================== SKILLS MANAGEMENT ====================
            with gr.Tab("📜 Skills"):
                gr.Markdown("""
                ### Agent Skills
                *Manage skill packages that define character capabilities*
                """)

                with gr.Row():
                    # Left - Installed skills
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Installed Skills")
                        skills_table = gr.Markdown(get_skills_summary_table())
                        refresh_skills_btn = gr.Button("🔄 Refresh", size="sm")

                        gr.Markdown("---")
                        gr.Markdown("### 🔧 Manage Skill")

                        skill_select = gr.Dropdown(
                            choices=get_skill_dir_choices(),
                            label="Select Skill",
                            interactive=True
                        )

                        skill_files_display = gr.Markdown("*Select a skill to see files*")

                        with gr.Row():
                            view_skill_btn = gr.Button("👁️ View", size="sm")
                            delete_skill_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")

                        skill_manage_status = gr.Textbox(label="Status", interactive=False)

                        gr.Markdown("---")
                        gr.Markdown("### 📥 Agent-Created Skills")
                        gr.Markdown("*Skills created by agents in sessions/files/*")

                        pending_table = gr.Markdown(get_pending_skills_table())
                        refresh_pending_btn = gr.Button("🔄 Refresh Pending", size="sm")

                        pending_select = gr.Dropdown(
                            choices=get_pending_skills_choices(),
                            label="Select to Install",
                            interactive=True
                        )
                        install_skill_btn = gr.Button("📦 Install Skill", variant="primary")
                        install_status = gr.Textbox(label="Status", interactive=False)

                    # Right - Skill editor
                    with gr.Column(scale=2):
                        gr.Markdown("### ✏️ Skill Editor")

                        skill_content_editor = gr.Code(
                            label="SKILL.md Content",
                            language="markdown",
                            lines=20
                        )

                        with gr.Row():
                            save_skill_btn = gr.Button("💾 Save Changes", variant="primary")
                            edit_skill_status = gr.Textbox(label="Status", interactive=False, scale=2)

                        gr.Markdown("---")
                        gr.Markdown("### ➕ Create New Skill")

                        with gr.Row():
                            new_skill_id = gr.Textbox(
                                label="Skill ID",
                                placeholder="my-custom-skill",
                                scale=1
                            )
                            new_skill_name = gr.Textbox(
                                label="Display Name",
                                placeholder="🎯 My Custom Skill",
                                scale=2
                            )

                        new_skill_desc = gr.Textbox(
                            label="Description",
                            placeholder="Brief description of what this skill does..."
                        )

                        new_skill_prompt = gr.Textbox(
                            label="System Prompt / Instructions",
                            placeholder="You are a specialized assistant that...",
                            lines=4
                        )

                        available_tools = [t['function']['name'] for t in REGISTRY.list_tools()]
                        new_skill_tools = gr.CheckboxGroup(
                            choices=available_tools,
                            label="Allowed Tools"
                        )

                        create_skill_btn = gr.Button("✨ Create Skill", variant="primary")
                        create_skill_status = gr.Textbox(label="Status", interactive=False)

        # ==================== EVENT HANDLERS ====================
        
        # === Character Tab ===
        
        # Wizard
        wizard_btn.click(
            fn=generate_character_profile,
            inputs=[wizard_concept],
            outputs=[
                wizard_status,
                char_id_input, display_name_input, system_prompt_input, voice_input,
                traits_input, memories_input, speech_patterns_input, setting_input, 
                metadata_input, tools_input
            ]
        )
        
        # Character list selection - auto-load
        character_list.change(
            fn=load_character_details,
            inputs=[character_list],
            outputs=[char_id_input, display_name_input, system_prompt_input, voice_input,
                    traits_input, memories_input, speech_patterns_input, setting_input, 
                    metadata_input, tools_input]
        )
        
        # New character button
        new_char_btn.click(
            fn=create_new_character_form,
            outputs=[char_id_input, display_name_input, system_prompt_input, voice_input,
                    traits_input, memories_input, speech_patterns_input, setting_input, 
                    metadata_input, tools_input]
        )
        
        # Duplicate character
        duplicate_btn.click(
            fn=duplicate_character,
            inputs=[character_list],
            outputs=[delete_status, char_id_input, display_name_input, system_prompt_input, 
                    voice_input, traits_input, memories_input, speech_patterns_input, 
                    setting_input, metadata_input, tools_input]
        )
        
        # Save character
        save_char_btn.click(
            fn=save_character,
            inputs=[char_id_input, display_name_input, system_prompt_input,
                   voice_input, traits_input, memories_input, speech_patterns_input,
                   setting_input, metadata_input, tools_input],
            outputs=[save_status, character_list]
        )
        
        # Delete character
        delete_char_btn.click(
            fn=delete_character,
            inputs=[character_list],
            outputs=[delete_status, character_list]
        )
        
        # Refresh voices button
        refresh_voices_btn.click(
            fn=lambda: gr.update(choices=get_available_voices_with_metadata()),
            outputs=[voice_input]
        )
        
        # === Voice Tab ===
        
        # Preview emoji on file upload
        voice_upload.change(
            fn=lambda f: preview_voice_emoji(Path(f).name if f else ""),
            inputs=[voice_upload],
            outputs=[emoji_preview]
        )
        
        # Upload voice
        upload_voice_btn.click(
            fn=upload_voice_file,
            inputs=[voice_upload, voice_custom_name, voice_custom_emoji],
            outputs=[upload_status, voice_list]
        ).then(
            fn=lambda: gr.update(choices=get_available_voices_with_metadata()),
            outputs=[voice_input]
        )
        
        # Show emoji suggestions when voice selected
        def show_suggestions(voice_name):
            if not voice_name:
                return ""
            suggestions = suggest_emoji_options(voice_name)
            return f"**Suggestions:** {' '.join(suggestions)}"
        
        voice_list.change(
            fn=show_suggestions,
            inputs=[voice_list],
            outputs=[emoji_suggestions]
        )
        
        # Update emoji
        update_emoji_btn.click(
            fn=update_voice_emoji,
            inputs=[voice_list, new_emoji_input],
            outputs=[voice_manage_status, voice_list]
        ).then(
            fn=lambda: gr.update(choices=get_available_voices_with_metadata()),
            outputs=[voice_input]
        )
        
        # Delete voice
        delete_voice_btn.click(
            fn=delete_voice_file,
            inputs=[voice_list],
            outputs=[voice_manage_status, voice_list]
        ).then(
            fn=lambda: gr.update(choices=get_available_voices_with_metadata()),
            outputs=[voice_input]
        )
        
        # === Memory Tab ===
        
        # Auto-load memories when character selected
        memory_char_select.change(
            fn=get_character_memory_stats,
            inputs=[memory_char_select],
            outputs=[memory_stats]
        ).then(
            fn=get_episodic_memories,
            inputs=[memory_char_select],
            outputs=[episodic_display]
        ).then(
            fn=get_semantic_memories,
            inputs=[memory_char_select],
            outputs=[semantic_display]
        ).then(
            fn=get_procedural_memories,
            inputs=[memory_char_select],
            outputs=[procedural_display]
        )
        
        refresh_stats_btn.click(
            fn=get_character_memory_stats,
            inputs=[memory_char_select],
            outputs=[memory_stats]
        )
        
        refresh_episodic_btn.click(
            fn=get_episodic_memories,
            inputs=[memory_char_select],
            outputs=[episodic_display]
        )
        
        refresh_semantic_btn.click(
            fn=get_semantic_memories,
            inputs=[memory_char_select],
            outputs=[semantic_display]
        )
        
        refresh_procedural_btn.click(
            fn=get_procedural_memories,
            inputs=[memory_char_select],
            outputs=[procedural_display]
        )
        
        # Add memory buttons
        add_semantic_btn.click(
            fn=add_semantic_memory,
            inputs=[memory_char_select, new_memory_content, new_memory_importance],
            outputs=[memory_status]
        ).then(
            fn=get_semantic_memories,
            inputs=[memory_char_select],
            outputs=[semantic_display]
        )
        
        add_episodic_btn.click(
            fn=add_episodic_memory,
            inputs=[memory_char_select, new_memory_content],
            outputs=[memory_status]
        ).then(
            fn=get_episodic_memories,
            inputs=[memory_char_select],
            outputs=[episodic_display]
        )
        
        add_procedural_btn.click(
            fn=add_procedural_memory,
            inputs=[memory_char_select, new_memory_content],
            outputs=[memory_status]
        ).then(
            fn=get_procedural_memories,
            inputs=[memory_char_select],
            outputs=[procedural_display]
        )
        
        # Delete memory by index
        delete_memory_btn.click(
            fn=delete_memory_by_index,
            inputs=[memory_char_select, delete_memory_type, delete_memory_index],
            outputs=[memory_status]
        ).then(
            fn=get_episodic_memories,
            inputs=[memory_char_select],
            outputs=[episodic_display]
        ).then(
            fn=get_semantic_memories,
            inputs=[memory_char_select],
            outputs=[semantic_display]
        ).then(
            fn=get_procedural_memories,
            inputs=[memory_char_select],
            outputs=[procedural_display]
        )
        
        # Clear/reinit memories
        clear_memories_btn.click(
            fn=clear_character_memories,
            inputs=[memory_char_select],
            outputs=[memory_status]
        ).then(
            fn=get_character_memory_stats,
            inputs=[memory_char_select],
            outputs=[memory_stats]
        ).then(
            fn=get_episodic_memories,
            inputs=[memory_char_select],
            outputs=[episodic_display]
        ).then(
            fn=get_semantic_memories,
            inputs=[memory_char_select],
            outputs=[semantic_display]
        ).then(
            fn=get_procedural_memories,
            inputs=[memory_char_select],
            outputs=[procedural_display]
        )
        
        reinit_memories_btn.click(
            fn=reinitialize_base_memories,
            inputs=[memory_char_select],
            outputs=[memory_status]
        ).then(
            fn=get_character_memory_stats,
            inputs=[memory_char_select],
            outputs=[memory_stats]
        ).then(
            fn=get_semantic_memories,
            inputs=[memory_char_select],
            outputs=[semantic_display]
        ).then(
            fn=get_procedural_memories,
            inputs=[memory_char_select],
            outputs=[procedural_display]
        )
        
        # Graph generation
        generate_graph_btn.click(
            fn=generate_graph_ui,
            inputs=[memory_char_select],
            outputs=[graph_file, graph_status]
        )
        
        # Edit memory logic
        def update_edit_choices(char_id, mem_type):
            return gr.update(choices=get_memory_choices(char_id, mem_type))
        
        memory_char_select.change(
            fn=update_edit_choices,
            inputs=[memory_char_select, edit_memory_type],
            outputs=[edit_memory_select]
        )
        
        edit_memory_type.change(
            fn=update_edit_choices,
            inputs=[memory_char_select, edit_memory_type],
            outputs=[edit_memory_select]
        )
        
        refresh_edit_list_btn.click(
            fn=update_edit_choices,
            inputs=[memory_char_select, edit_memory_type],
            outputs=[edit_memory_select]
        )
        
        edit_memory_select.change(
            fn=load_memory_for_edit,
            inputs=[memory_char_select, edit_memory_select],
            outputs=[edit_content, edit_importance]
        )
        
        update_memory_btn.click(
            fn=update_memory_ui,
            inputs=[memory_char_select, edit_memory_select, edit_content, edit_importance],
            outputs=[memory_status]
        ).then(
            fn=update_edit_choices,
            inputs=[memory_char_select, edit_memory_type],
            outputs=[edit_memory_select]
        ).then(
            fn=get_episodic_memories,
            inputs=[memory_char_select],
            outputs=[episodic_display]
        ).then(
            fn=get_semantic_memories,
            inputs=[memory_char_select],
            outputs=[semantic_display]
        )

        delete_mem_ui_btn.click(
            fn=delete_memory_from_ui,
            inputs=[memory_char_select, edit_memory_select],
            outputs=[memory_status]
        ).then(
            fn=update_edit_choices,
            inputs=[memory_char_select, edit_memory_type],
            outputs=[edit_memory_select]
        ).then(
            fn=get_episodic_memories,
            inputs=[memory_char_select],
            outputs=[episodic_display]
        ).then(
            fn=get_semantic_memories,
            inputs=[memory_char_select],
            outputs=[semantic_display]
        )
        
        # Import/Export
        export_mem_btn.click(
            fn=export_memories_ui,
            inputs=[memory_char_select],
            outputs=[export_file]
        )
        
        import_mem_btn.click(
            fn=import_memories_ui,
            inputs=[import_file],
            outputs=[memory_status]
        ).then(
            fn=get_character_memory_stats,
            inputs=[memory_char_select],
            outputs=[memory_stats]
        )

        # === Conversation Tab ===
        
        conv_char_select.change(
            fn=get_conversation_table,
            inputs=[conv_char_select],
            outputs=[conv_table]
        )
        
        refresh_convs_btn.click(
            fn=get_conversation_table,
            inputs=[conv_char_select],
            outputs=[conv_table]
        )
        
        view_conv_btn.click(
            fn=view_conversation,
            inputs=[conv_char_select, conversation_id_input],
            outputs=[conv_display]
        )
        
        export_conv_btn.click(
            fn=export_conversation,
            inputs=[conv_char_select, conversation_id_input],
            outputs=[conv_export]
        )
        
        delete_conv_btn.click(
            fn=delete_conversation,
            inputs=[conv_char_select, conversation_id_input],
            outputs=[conv_status]
        ).then(
            fn=get_conversation_table,
            inputs=[conv_char_select],
            outputs=[conv_table]
        )
        
        rename_conv_btn.click(
            fn=rename_conversation,
            inputs=[conv_char_select, conversation_id_input, new_title_input],
            outputs=[conv_status]
        ).then(
            fn=get_conversation_table,
            inputs=[conv_char_select],
            outputs=[conv_table]
        )
        
        def prepare_download(character_id, conversation_id):
            if not character_id or not conversation_id:
                return gr.update(visible=False)
            filepath = CONVERSATIONS_DIR / character_id / f"{conversation_id}.json"
            if filepath.exists():
                return gr.update(value=str(filepath), visible=True)
            return gr.update(visible=False)
        
        download_conv_btn.click(
            fn=prepare_download,
            inputs=[conv_char_select, conversation_id_input],
            outputs=[conv_file_download]
        )

        # === Skills Tab ===

        # Store current skill path for editing
        current_skill_path = gr.State("")

        # Refresh installed skills
        refresh_skills_btn.click(
            fn=get_skills_summary_table,
            outputs=[skills_table]
        ).then(
            fn=lambda: gr.update(choices=get_skill_dir_choices()),
            outputs=[skill_select]
        )

        # Show skill files when selected
        skill_select.change(
            fn=get_skill_files,
            inputs=[skill_select],
            outputs=[skill_files_display]
        )

        # View skill content
        def load_skill_for_edit(skill_dir_name):
            if not skill_dir_name:
                return "", ""
            skill_path = SKILLS_DIR / skill_dir_name / "SKILL.md"
            if skill_path.exists():
                return skill_path.read_text(encoding='utf-8'), str(skill_path)
            return "*SKILL.md not found*", ""

        view_skill_btn.click(
            fn=load_skill_for_edit,
            inputs=[skill_select],
            outputs=[skill_content_editor, current_skill_path]
        )

        # Save skill content
        save_skill_btn.click(
            fn=save_skill_content,
            inputs=[current_skill_path, skill_content_editor],
            outputs=[edit_skill_status]
        )

        # Delete skill
        delete_skill_btn.click(
            fn=delete_skill,
            inputs=[skill_select],
            outputs=[skill_manage_status]
        ).then(
            fn=get_skills_summary_table,
            outputs=[skills_table]
        ).then(
            fn=lambda: gr.update(choices=get_skill_dir_choices()),
            outputs=[skill_select]
        )

        # Refresh pending skills
        refresh_pending_btn.click(
            fn=get_pending_skills_table,
            outputs=[pending_table]
        ).then(
            fn=lambda: gr.update(choices=get_pending_skills_choices()),
            outputs=[pending_select]
        )

        # Install skill from pending
        install_skill_btn.click(
            fn=install_skill_from_file,
            inputs=[pending_select],
            outputs=[install_status]
        ).then(
            fn=get_skills_summary_table,
            outputs=[skills_table]
        ).then(
            fn=lambda: gr.update(choices=get_skill_dir_choices()),
            outputs=[skill_select]
        ).then(
            fn=get_pending_skills_table,
            outputs=[pending_table]
        ).then(
            fn=lambda: gr.update(choices=get_pending_skills_choices()),
            outputs=[pending_select]
        )

        # Create new skill
        create_skill_btn.click(
            fn=create_new_skill,
            inputs=[new_skill_id, new_skill_name, new_skill_desc, new_skill_prompt, new_skill_tools],
            outputs=[create_skill_status]
        ).then(
            fn=get_skills_summary_table,
            outputs=[skills_table]
        ).then(
            fn=lambda: gr.update(choices=get_skill_dir_choices()),
            outputs=[skill_select]
        )

        # Auto-refresh timer (if supported)
        try:
            memory_refresh_timer = gr.Timer(value=5)
            memory_refresh_timer.tick(
                fn=lambda char_id: (
                    get_character_memory_stats(char_id),
                    get_episodic_memories(char_id),
                    get_semantic_memories(char_id),
                    get_procedural_memories(char_id)
                ) if char_id else ("*Select a character*", "*No memories*", "*No memories*", "*No procedural memories*"),
                inputs=[memory_char_select],
                outputs=[memory_stats, episodic_display, semantic_display, procedural_display]
            )
        except (AttributeError, TypeError):
            print("[Warning] gr.Timer not available for auto-refresh")
    
    return app


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   [*] Character & Memory Manager")
    print("   Create Characters - Manage Voices - Edit Memories")
    print(f"   Platform: {PLATFORM.title()}" + (" (WSL)" if IS_WSL else ""))
    print("="*60)
    
    print(f"\n✓ Characters loaded: {len(CHARACTER_MANAGER.characters)}")
    print(f"✓ Voices available: {len(get_available_voices_with_metadata())}")
    print(f"✓ Starting on http://127.0.0.1:7863")
    print("✓ Press Ctrl+C to stop\n")
    
    app = create_management_ui()
    app.launch(server_port=7863, server_name="127.0.0.1", inbrowser=True, share=False)
