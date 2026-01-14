"""
Enhanced Character Manager for IndexTTS2 Voice Chat

Features:
- Dynamic character loading from Agent Skills (SKILL.md)
- Persistent character state (mood, relationship)
- Character import/export
- Custom character creation via UI
- Voice reference management
- Topic tracking across conversations

Characters are loaded from (in priority order):
1. skills/*/SKILL.md (Agent Skills - primary source)
2. characters.json (user-created via UI)
3. characters/*.yaml (legacy, for backwards compatibility)

Agent Skills Format:
- YAML frontmatter contains character identity metadata
- Markdown body contains system prompt and instructions
- Progressive disclosure: metadata always loaded, full content on activation
"""

import re
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import shutil

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("[Characters] PyYAML not installed. YAML character loading disabled.")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CharacterVoice:
    """Voice configuration for a character"""
    reference_file: str  # Path to voice reference WAV
    speaking_rate: float = 1.0  # TTS speed multiplier
    pitch_shift: float = 0.0  # Pitch adjustment
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterVoice':
        return cls(**{k: v for k, v in data.items() if k in ['reference_file', 'speaking_rate', 'pitch_shift']})


@dataclass
class CharacterPersonality:
    """Personality traits and behavioral patterns"""
    traits: List[str] = field(default_factory=list)
    speech_patterns: List[str] = field(default_factory=list)
    likes: List[str] = field(default_factory=list)
    dislikes: List[str] = field(default_factory=list)
    quirks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterPersonality':
        return cls(
            traits=data.get('traits', []),
            speech_patterns=data.get('speech_patterns', []),
            likes=data.get('likes', []),
            dislikes=data.get('dislikes', []),
            quirks=data.get('quirks', [])
        )


@dataclass
class PersistentCharacterState:
    """State that persists across sessions"""
    character_id: str
    current_mood: str = "neutral"
    mood_history: List[Tuple[str, str]] = field(default_factory=list)
    relationship_level: float = 0.0  # -1 to 1 (hostile to intimate)
    trust_level: float = 0.5  # 0 to 1
    familiarity: float = 0.0  # 0 to 1 (stranger to well-known)
    topics_discussed: Dict[str, int] = field(default_factory=dict)
    user_facts_learned: List[str] = field(default_factory=list)
    last_interaction: Optional[str] = None
    total_interactions: int = 0
    total_tokens_exchanged: int = 0
    favorite_topics: List[str] = field(default_factory=list)
    avoided_topics: List[str] = field(default_factory=list)
    
    def update_mood(self, new_mood: str):
        if new_mood != self.current_mood:
            self.mood_history.append((datetime.now().isoformat(), new_mood))
            if len(self.mood_history) > 50:
                self.mood_history = self.mood_history[-50:]
            self.current_mood = new_mood
    
    def record_topic(self, topic: str):
        self.topics_discussed[topic] = self.topics_discussed.get(topic, 0) + 1
        sorted_topics = sorted(self.topics_discussed.items(), key=lambda x: x[1], reverse=True)
        self.favorite_topics = [t[0] for t in sorted_topics[:5]]
    
    def learn_user_fact(self, fact: str):
        if fact not in self.user_facts_learned:
            self.user_facts_learned.append(fact)
            if len(self.user_facts_learned) > 100:
                self.user_facts_learned = self.user_facts_learned[-100:]
    
    def update_relationship(self, delta: float):
        self.relationship_level = max(-1.0, min(1.0, self.relationship_level + delta))
    
    def update_trust(self, delta: float):
        self.trust_level = max(0.0, min(1.0, self.trust_level + delta))
    
    def update_familiarity(self, delta: float):
        self.familiarity = max(0.0, min(1.0, self.familiarity + delta))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'character_id': self.character_id,
            'current_mood': self.current_mood,
            'mood_history': self.mood_history,
            'relationship_level': self.relationship_level,
            'trust_level': self.trust_level,
            'familiarity': self.familiarity,
            'topics_discussed': self.topics_discussed,
            'user_facts_learned': self.user_facts_learned,
            'last_interaction': self.last_interaction,
            'total_interactions': self.total_interactions,
            'total_tokens_exchanged': self.total_tokens_exchanged,
            'favorite_topics': self.favorite_topics,
            'avoided_topics': self.avoided_topics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersistentCharacterState':
        return cls(
            character_id=data.get('character_id', 'unknown'),
            current_mood=data.get('current_mood', 'neutral'),
            mood_history=[(m[0], m[1]) for m in data.get('mood_history', [])],
            relationship_level=data.get('relationship_level', 0.0),
            trust_level=data.get('trust_level', 0.5),
            familiarity=data.get('familiarity', 0.0),
            topics_discussed=data.get('topics_discussed', {}),
            user_facts_learned=data.get('user_facts_learned', []),
            last_interaction=data.get('last_interaction'),
            total_interactions=data.get('total_interactions', 0),
            total_tokens_exchanged=data.get('total_tokens_exchanged', 0),
            favorite_topics=data.get('favorite_topics', []),
            avoided_topics=data.get('avoided_topics', [])
        )
    
    def get_relationship_description(self) -> str:
        if self.relationship_level < -0.5:
            return "hostile"
        elif self.relationship_level < -0.2:
            return "cold"
        elif self.relationship_level < 0.2:
            return "neutral"
        elif self.relationship_level < 0.5:
            return "friendly"
        elif self.relationship_level < 0.8:
            return "close"
        else:
            return "intimate"
    
    def get_familiarity_description(self) -> str:
        if self.familiarity < 0.2:
            return "stranger"
        elif self.familiarity < 0.4:
            return "acquaintance"
        elif self.familiarity < 0.6:
            return "familiar"
        elif self.familiarity < 0.8:
            return "well-known"
        else:
            return "deeply known"


@dataclass
class Character:
    """Complete character profile"""
    id: str
    name: str
    display_name: str
    system_prompt: str
    default_voice: str = "reference.wav"
    personality_traits: List[str] = field(default_factory=list)
    speech_patterns: List[str] = field(default_factory=list)
    voice_reference: Optional[str] = None
    initial_memories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    allowed_tools: List[str] = field(default_factory=list)
    
    # Extended fields
    personality: Optional[CharacterPersonality] = None
    voice_config: Optional[CharacterVoice] = None
    greeting: str = ""
    scenario: str = ""
    example_dialogue: List[Dict[str, str]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    author: str = ""
    version: str = "1.0"
    created_at: str = ""
    updated_at: str = ""
    is_builtin: bool = False
    is_nsfw: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'system_prompt': self.system_prompt,
            'default_voice': self.default_voice,
            'personality_traits': self.personality_traits,
            'speech_patterns': self.speech_patterns,
            'voice_reference': self.voice_reference,
            'initial_memories': self.initial_memories,
            'metadata': self.metadata,
            'allowed_tools': self.allowed_tools,
            'greeting': self.greeting,
            'scenario': self.scenario,
            'example_dialogue': self.example_dialogue,
            'tags': self.tags,
            'author': self.author,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'is_builtin': self.is_builtin,
            'is_nsfw': self.is_nsfw
        }
        if self.personality:
            data['personality'] = self.personality.to_dict()
        if self.voice_config:
            data['voice_config'] = self.voice_config.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Character':
        personality = None
        if 'personality' in data:
            personality = CharacterPersonality.from_dict(data['personality'])
        
        voice_config = None
        if 'voice_config' in data:
            voice_config = CharacterVoice.from_dict(data['voice_config'])
        
        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            display_name=data.get('display_name', data.get('name', '')),
            system_prompt=data.get('system_prompt', ''),
            default_voice=data.get('default_voice', 'reference.wav'),
            personality_traits=data.get('personality_traits', []),
            speech_patterns=data.get('speech_patterns', []),
            voice_reference=data.get('voice_reference'),
            initial_memories=data.get('initial_memories', []),
            metadata=data.get('metadata', {}),
            allowed_tools=data.get('allowed_tools', []),
            personality=personality,
            voice_config=voice_config,
            greeting=data.get('greeting', ''),
            scenario=data.get('scenario', ''),
            example_dialogue=data.get('example_dialogue', []),
            tags=data.get('tags', []),
            author=data.get('author', ''),
            version=data.get('version', '1.0'),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            is_builtin=data.get('is_builtin', False),
            is_nsfw=data.get('is_nsfw', False)
        )
    
    def get_effective_system_prompt(self, state: Optional[PersistentCharacterState] = None) -> str:
        """Generate the full system prompt with state-aware modifications."""
        prompt = self.system_prompt
        
        if self.scenario:
            prompt = f"SCENARIO:\n{self.scenario}\n\n{prompt}"
        
        if self.personality:
            personality_section = "\n\nPERSONALITY DETAILS:\n"
            if self.personality.traits:
                personality_section += f"- Core traits: {', '.join(self.personality.traits)}\n"
            if self.personality.quirks:
                personality_section += f"- Quirks: {', '.join(self.personality.quirks)}\n"
            if self.personality.likes:
                personality_section += f"- Likes: {', '.join(self.personality.likes)}\n"
            if self.personality.dislikes:
                personality_section += f"- Dislikes: {', '.join(self.personality.dislikes)}\n"
            prompt += personality_section
        
        if state:
            context_section = f"\n\nRELATIONSHIP CONTEXT:\n"
            context_section += f"- Current mood: {state.current_mood}\n"
            context_section += f"- Relationship: {state.get_relationship_description()}\n"
            context_section += f"- Familiarity: {state.get_familiarity_description()}\n"
            context_section += f"- Total interactions: {state.total_interactions}\n"
            
            if state.favorite_topics:
                context_section += f"- Favorite discussion topics: {', '.join(state.favorite_topics[:3])}\n"
            
            if state.user_facts_learned:
                context_section += f"- Known about user: {'; '.join(state.user_facts_learned[-5:])}\n"
            
            prompt += context_section
        
        return prompt


# ============================================================================
# Character Manager
# ============================================================================

class CharacterManager:
    """
    Enhanced character manager with:
    - Dynamic loading from Agent Skills (SKILL.md) - primary source
    - Persistent state management
    - Character import/export
    - Legacy YAML file support (backwards compatibility)
    """
    
    def __init__(self, config_path: str = None, characters_dir: str = None, skills_dir: str = None):
        """
        Initialize character manager.
        
        Args:
            config_path: Path to characters.json for user-created characters
            characters_dir: Directory for legacy YAML character files
            skills_dir: Directory for Agent Skills (primary character source)
        """
        self.characters: Dict[str, Character] = {}
        self.states: Dict[str, PersistentCharacterState] = {}
        
        # Paths
        root_dir = Path(__file__).parent.parent
        self.config_path = Path(config_path) if config_path else root_dir / "characters.json"
        self.characters_dir = Path(characters_dir) if characters_dir else root_dir / "characters"
        self.skills_dir = Path(skills_dir) if skills_dir else root_dir / "skills"
        self.states_path = root_dir / "sessions" / "character_states.json"
        self.voice_ref_dir = root_dir / "voice_reference"
        
        # Ensure directories exist
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        self.states_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load characters from all sources (priority order)
        # Skills overwrite everything, then JSON, then YAML
        self._load_skill_characters()    # Primary source
        self._load_json_characters()     # User-created (won't overwrite skills)
        self._load_yaml_characters()     # Legacy support (won't overwrite)
        self._load_states()
        
        print(f"[Characters] Loaded {len(self.characters)} characters: {list(self.characters.keys())}")
    
    def _parse_skill_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """
        Parse YAML frontmatter and markdown body from SKILL.md content.
        
        Returns:
            Tuple of (frontmatter_dict, markdown_body)
        """
        # Match YAML frontmatter between --- delimiters
        pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
        match = re.match(pattern, content, re.DOTALL)
        
        if not match:
            return {}, content
        
        frontmatter_str = match.group(1)
        markdown_body = match.group(2)
        
        try:
            frontmatter = yaml.safe_load(frontmatter_str) or {}
        except Exception as e:
            print(f"[Characters] Error parsing SKILL.md frontmatter: {e}")
            frontmatter = {}
        
        return frontmatter, markdown_body
    
    def _extract_system_prompt(self, markdown_body: str) -> str:
        """
        Extract system prompt from markdown body.
        Looks for content under '## System Prompt' heading.
        """
        # Try to find ## System Prompt section
        pattern = r'##\s*System Prompt\s*\n\n(.*?)(?=\n##|$)'
        match = re.search(pattern, markdown_body, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: use everything after the first heading
        lines = markdown_body.strip().split('\n')
        content_lines = []
        skip_first_heading = True
        
        for line in lines:
            if skip_first_heading and line.startswith('# '):
                skip_first_heading = False
                continue
            content_lines.append(line)
        
        return '\n'.join(content_lines).strip()
    
    def _load_skill_characters(self):
        """
        Load characters from Agent Skills (skills/*/SKILL.md).
        This is the primary source for character definitions.
        """
        if not YAML_AVAILABLE:
            print("[Characters] PyYAML required for skill loading")
            return
        
        if not self.skills_dir.exists():
            print(f"[Characters] Skills directory not found: {self.skills_dir}")
            return
        
        skill_count = 0
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            
            try:
                with open(skill_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                frontmatter, markdown_body = self._parse_skill_frontmatter(content)
                
                # Skip if no character ID defined (might be a non-character skill)
                if 'id' not in frontmatter:
                    continue
                
                # Extract system prompt from markdown
                system_prompt = self._extract_system_prompt(markdown_body)
                
                # Build character from frontmatter
                char_data = {
                    'id': frontmatter.get('id'),
                    'name': frontmatter.get('name', frontmatter.get('id')),
                    'display_name': frontmatter.get('display_name', frontmatter.get('name', frontmatter.get('id'))),
                    'system_prompt': system_prompt,
                    'default_voice': frontmatter.get('voice', 'reference.wav'),
                    'personality_traits': frontmatter.get('personality_traits', []),
                    'speech_patterns': frontmatter.get('speech_patterns', []),
                    'initial_memories': frontmatter.get('initial_memories', []),
                    'metadata': frontmatter.get('metadata', {}),
                    'allowed_tools': frontmatter.get('allowed_tools', []),
                    'tags': frontmatter.get('tags', []),
                    'is_builtin': True,  # Skills are treated as built-in
                }
                
                # Add skill-specific metadata
                char_data['metadata']['skill_path'] = str(skill_file)
                char_data['metadata']['skill_description'] = frontmatter.get('description', '')
                
                char = Character.from_dict(char_data)
                self.characters[char.id] = char
                skill_count += 1
                print(f"[Characters] Loaded from skill: {char.id}")
                
            except Exception as e:
                print(f"[Characters] Error loading skill {skill_dir.name}: {e}")
        
        if skill_count > 0:
            print(f"[Characters] Loaded {skill_count} characters from skills")
    
    def _load_json_characters(self):
        """Load user-created characters from characters.json"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                loaded = 0
                for char_data in config.get('characters', []):
                    char = Character.from_dict(char_data)
                    char.is_builtin = False
                    # Don't overwrite skill-loaded characters
                    if char.id not in self.characters:
                        self.characters[char.id] = char
                        loaded += 1
                    
                if loaded > 0:
                    print(f"[Characters] Loaded {loaded} from JSON")
            except Exception as e:
                print(f"[Characters] Error loading JSON config: {e}")
    
    def _load_yaml_characters(self):
        """Load characters from individual YAML files (legacy support)"""
        if not YAML_AVAILABLE:
            return
        
        yaml_files = list(self.characters_dir.glob("*.yaml")) + list(self.characters_dir.glob("*.yml"))
        
        loaded = 0
        for yaml_file in yaml_files:
            # Skip template file
            if yaml_file.name.startswith('_'):
                continue
                
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if data:
                    char = Character.from_dict(data)
                    char.is_builtin = False
                    if not char.id:
                        char.id = yaml_file.stem
                    
                    # Don't overwrite skill-loaded characters
                    if char.id not in self.characters:
                        self.characters[char.id] = char
                        loaded += 1
                        print(f"[Characters] Loaded from YAML (legacy): {char.id}")
                    
            except Exception as e:
                print(f"[Characters] Error loading {yaml_file}: {e}")
        
        if loaded > 0:
            print(f"[Characters] Loaded {loaded} from legacy YAML files")
    
    def _load_states(self):
        """Load persistent character states"""
        if self.states_path.exists():
            try:
                with open(self.states_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for char_id, state_data in data.items():
                    self.states[char_id] = PersistentCharacterState.from_dict(state_data)
                    
                print(f"[Characters] Loaded {len(self.states)} character states")
            except Exception as e:
                print(f"[Characters] Error loading states: {e}")
    
    def _save_characters(self):
        """Save user-created characters to JSON"""
        try:
            custom_chars = [
                char.to_dict() 
                for char in self.characters.values() 
                if not char.is_builtin
            ]
            
            config = {'characters': custom_chars}
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"[Characters] Saved {len(custom_chars)} custom characters")
        except Exception as e:
            print(f"[Characters] Error saving: {e}")
    
    def _save_states(self):
        """Save all character states"""
        try:
            data = {
                char_id: state.to_dict()
                for char_id, state in self.states.items()
            }
            
            with open(self.states_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"[Characters] Error saving states: {e}")
    
    # ==================== Character Operations ====================
    
    def get_character(self, character_id: str) -> Optional[Character]:
        """Get a character by ID"""
        return self.characters.get(character_id)
    
    def list_characters(self) -> List[str]:
        """List all character IDs"""
        return list(self.characters.keys())
    
    def get_display_names(self) -> Dict[str, str]:
        """Get mapping of ID -> display name"""
        return {cid: c.display_name for cid, c in self.characters.items()}
    
    def get_dropdown_choices(self) -> List[Tuple[str, str]]:
        """Get (display_name, id) tuples for dropdown menus"""
        return [(c.display_name, cid) for cid, c in self.characters.items()]
    
    def get_characters_by_tag(self, tag: str) -> List[Character]:
        """Get all characters with a specific tag"""
        return [c for c in self.characters.values() if tag in c.tags]
    
    def add_character(self, character: Character) -> bool:
        """Add a new character"""
        if character.id in self.characters:
            print(f"[Characters] Character {character.id} already exists")
            return False
        
        character.created_at = datetime.now().isoformat()
        character.updated_at = character.created_at
        character.is_builtin = False
        
        self.characters[character.id] = character
        self._save_characters()
        return True
    
    def update_character(self, character: Character) -> bool:
        """Update an existing character"""
        if character.id not in self.characters:
            return False
        
        if self.characters[character.id].is_builtin:
            print(f"[Characters] Cannot modify built-in character: {character.id}")
            return False
        
        character.updated_at = datetime.now().isoformat()
        self.characters[character.id] = character
        self._save_characters()
        return True
    
    def delete_character(self, character_id: str) -> bool:
        """Delete a character"""
        if character_id not in self.characters:
            return False
        
        if self.characters[character_id].is_builtin:
            print(f"[Characters] Cannot delete built-in character: {character_id}")
            return False
        
        del self.characters[character_id]
        
        if character_id in self.states:
            del self.states[character_id]
            self._save_states()
        
        self._save_characters()
        return True
    
    def duplicate_character(self, character_id: str, new_id: str) -> Optional[Character]:
        """Create a copy of a character with a new ID"""
        if character_id not in self.characters:
            return None
        
        if new_id in self.characters:
            return None
        
        original = self.characters[character_id]
        new_char = Character.from_dict(original.to_dict())
        new_char.id = new_id
        new_char.name = f"{original.name} (Copy)"
        new_char.display_name = f"{original.display_name} (Copy)"
        new_char.is_builtin = False
        new_char.created_at = datetime.now().isoformat()
        new_char.updated_at = new_char.created_at
        
        self.characters[new_id] = new_char
        self._save_characters()
        return new_char
    
    # ==================== State Operations ====================
    
    def get_state(self, character_id: str) -> PersistentCharacterState:
        """Get or create persistent state for a character"""
        if character_id not in self.states:
            self.states[character_id] = PersistentCharacterState(character_id=character_id)
        return self.states[character_id]
    
    def update_state(self, character_id: str, state: PersistentCharacterState):
        """Update and save character state"""
        state.last_interaction = datetime.now().isoformat()
        self.states[character_id] = state
        self._save_states()
    
    def record_interaction(self, character_id: str, user_tokens: int = 0, assistant_tokens: int = 0):
        """Record an interaction and update state"""
        state = self.get_state(character_id)
        state.total_interactions += 1
        state.total_tokens_exchanged += user_tokens + assistant_tokens
        state.last_interaction = datetime.now().isoformat()
        
        if state.familiarity < 1.0:
            state.update_familiarity(0.01)
        
        self._save_states()
    
    def reset_state(self, character_id: str):
        """Reset character state to defaults"""
        self.states[character_id] = PersistentCharacterState(character_id=character_id)
        self._save_states()
    
    # ==================== Import/Export ====================
    
    def export_character(self, character_id: str, include_state: bool = True) -> Optional[Dict[str, Any]]:
        """Export a character with optional state"""
        if character_id not in self.characters:
            return None
        
        char = self.characters[character_id]
        data = {
            'version': '2.0',
            'exported_at': datetime.now().isoformat(),
            'character': char.to_dict()
        }
        
        if include_state and character_id in self.states:
            data['state'] = self.states[character_id].to_dict()
        
        return data
    
    def export_character_yaml(self, character_id: str) -> Optional[str]:
        """Export a character as YAML string"""
        if not YAML_AVAILABLE:
            return None
        
        if character_id not in self.characters:
            return None
        
        char = self.characters[character_id]
        return yaml.dump(char.to_dict(), default_flow_style=False, allow_unicode=True)
    
    def import_character(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Import a character from export data."""
        try:
            char_data = data.get('character', data)
            char = Character.from_dict(char_data)
            
            if char.id in self.characters:
                base_id = char.id
                counter = 1
                while f"{base_id}_{counter}" in self.characters:
                    counter += 1
                char.id = f"{base_id}_{counter}"
                char.display_name = f"{char.display_name} (Imported)"
            
            char.is_builtin = False
            char.created_at = datetime.now().isoformat()
            char.updated_at = char.created_at
            
            self.characters[char.id] = char
            
            if 'state' in data:
                state = PersistentCharacterState.from_dict(data['state'])
                state.character_id = char.id
                self.states[char.id] = state
                self._save_states()
            
            self._save_characters()
            return True, f"Imported character as '{char.id}'"
            
        except Exception as e:
            return False, f"Import failed: {e}"
    
    def save_character_as_yaml(self, character_id: str) -> Optional[Path]:
        """Save a character to a YAML file"""
        if not YAML_AVAILABLE:
            return None
        
        if character_id not in self.characters:
            return None
        
        char = self.characters[character_id]
        filepath = self.characters_dir / f"{character_id}.yaml"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(char.to_dict(), f, default_flow_style=False, allow_unicode=True)
            return filepath
        except Exception as e:
            print(f"[Characters] Error saving YAML: {e}")
            return None
    
    # ==================== Voice Reference Management ====================
    
    def list_voice_references(self) -> List[str]:
        """List available voice reference files"""
        if not self.voice_ref_dir.exists():
            return []
        
        return [
            f.name for f in self.voice_ref_dir.iterdir()
            if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']
        ]
    
    def add_voice_reference(self, source_path: str, filename: str = None) -> Optional[str]:
        """Copy a voice reference file to the voice_reference directory"""
        source = Path(source_path)
        if not source.exists():
            return None
        
        if filename is None:
            filename = source.name
        
        dest = self.voice_ref_dir / filename
        
        try:
            shutil.copy2(source, dest)
            return filename
        except Exception as e:
            print(f"[Characters] Error copying voice reference: {e}")
            return None
    
    def get_character_voice_path(self, character_id: str) -> Optional[Path]:
        """Get the full path to a character's voice reference"""
        if character_id not in self.characters:
            return None
        
        char = self.characters[character_id]
        voice_file = char.voice_reference or char.default_voice
        
        voice_path = self.voice_ref_dir / voice_file
        if voice_path.exists():
            return voice_path
        
        default_path = self.voice_ref_dir / "reference.wav"
        return default_path if default_path.exists() else None


# ============================================================================
# Factory Function
# ============================================================================

def create_character_manager(config_path: str = None, characters_dir: str = None, skills_dir: str = None) -> CharacterManager:
    """Factory function to create a character manager"""
    return CharacterManager(config_path, characters_dir, skills_dir)
