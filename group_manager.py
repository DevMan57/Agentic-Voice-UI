"""
Group Chat Manager for IndexTTS2 Voice Chat

Manages natural multi-character conversations where:
- User can address any character by name
- Characters decide whether to respond based on context
- Multiple characters can respond in sequence
- Each character uses their own voice

Example flow:
  User: "Hey Hermione, what do you think about this code?"
  → Hermione responds (she was addressed)
  
  User: "What do you both think about going to the beach?"
  → Hermione responds, then Lisbeth responds (both addressed)
  
  User: "That's interesting"
  → Primary character responds (no specific address)
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class SpeakerDecision:
    """Decision about who should speak next"""
    character_id: str
    confidence: float  # 0.0 to 1.0
    reason: str


class GroupChatManager:
    """
    Manages natural multi-character conversation flow.
    
    Key features:
    - Detects who is being addressed in user messages
    - Allows multiple characters to respond naturally
    - Maintains conversation context for all characters
    """
    
    def __init__(self):
        self.active_characters: List[str] = []
        self.character_names: Dict[str, List[str]] = {}  # char_id -> [name, aliases]
        self.max_responses: int = 2  # Max characters that can respond per user message
        self.is_enabled: bool = False
        self.last_speakers: List[str] = []  # Track who spoke recently
        
    def set_active_characters(self, char_ids: List[str], char_manager=None):
        """
        Set the list of characters in the group and cache their names.
        
        Args:
            char_ids: List of character IDs
            char_manager: CharacterManager instance to get names
        """
        self.active_characters = [cid for cid in char_ids if cid]
        self.character_names = {}
        
        if char_manager:
            for cid in self.active_characters:
                char = char_manager.get_character(cid)
                if char:
                    # Store name and potential aliases (first name, nickname, etc.)
                    names = [char.name.lower()]
                    # Add first name if multi-word
                    if ' ' in char.name:
                        names.append(char.name.split()[0].lower())
                    # Add character ID as fallback
                    names.append(cid.lower())
                    self.character_names[cid] = names
                    
        print(f"[Group] Active characters: {self.active_characters}")
        print(f"[Group] Name mappings: {self.character_names}")
        
    def enable(self, enabled: bool = True):
        self.is_enabled = enabled
        if not enabled:
            self.last_speakers = []
        
    def detect_addressed_characters(self, user_message: str) -> List[str]:
        """
        Detect which characters are being addressed in the user's message.
        
        Returns list of character IDs that appear to be addressed, in order.
        Empty list means no specific character addressed (use primary).
        """
        addressed = []
        message_lower = user_message.lower()
        
        for char_id, names in self.character_names.items():
            for name in names:
                # Check for direct address patterns
                patterns = [
                    rf'\b{name}\b',  # Name mentioned anywhere
                    rf'^{name}[,:]',  # "Hermione, ..." or "Hermione: ..."
                    rf'hey {name}',  # "hey Hermione"
                    rf'@{name}',  # "@hermione"
                ]
                
                for pattern in patterns:
                    if re.search(pattern, message_lower):
                        if char_id not in addressed:
                            addressed.append(char_id)
                        break
                        
        return addressed
    
    def detect_group_address(self, user_message: str) -> bool:
        """Check if the message addresses the group as a whole."""
        group_patterns = [
            r'\b(you both|both of you|everyone|you all|y\'all)\b',
            r'\b(what do you (both|all) think)\b',
            r'\b(anyone|either of you)\b',
        ]
        
        message_lower = user_message.lower()
        for pattern in group_patterns:
            if re.search(pattern, message_lower):
                return True
        return False
    
    def get_speakers_for_message(
        self, 
        user_message: str, 
        primary_character: str,
        max_responses: int = 2
    ) -> List[Tuple[str, str]]:
        """
        Determine who should speak in response to a user message.
        
        Returns list of (character_id, context_prompt) tuples.
        The context_prompt helps the character understand their role.
        """
        if not self.is_enabled or len(self.active_characters) < 2:
            return [(primary_character, "")]
        
        speakers = []
        addressed = self.detect_addressed_characters(user_message)
        group_addressed = self.detect_group_address(user_message)
        
        if group_addressed:
            # Everyone responds (up to max)
            for char_id in self.active_characters[:max_responses]:
                context = f"[The user addressed the whole group. Respond naturally as yourself.]"
                speakers.append((char_id, context))
                
        elif addressed:
            # Specific characters addressed
            for char_id in addressed[:max_responses]:
                if char_id in self.active_characters:
                    context = f"[The user addressed you directly. Respond to them.]"
                    speakers.append((char_id, context))
                    
            # If addressed character isn't in active list, fall back to primary
            if not speakers:
                speakers.append((primary_character, ""))
                
        else:
            # No specific address - primary character responds
            # But if another character was the last speaker, they might chime in
            speakers.append((primary_character, ""))
            
        return speakers
    
    def build_group_context(
        self,
        chat_history: list,
        current_speaker: str,
        all_characters: Dict[str, any]  # char_id -> Character object
    ) -> str:
        """
        Build context for a character about the ongoing group conversation.
        
        This helps the character understand:
        - Who else is in the conversation
        - What others have said recently
        - Their relationship to the conversation
        """
        context_parts = []
        
        # List other participants
        others = [cid for cid in self.active_characters if cid != current_speaker]
        if others and all_characters:
            other_names = []
            for cid in others:
                char = all_characters.get(cid)
                if char:
                    other_names.append(char.name)
            if other_names:
                context_parts.append(f"[Other participants: {', '.join(other_names)}]")
        
        return "\n".join(context_parts)
    
    def format_response_attribution(
        self, 
        character_id: str, 
        response: str,
        char_manager=None
    ) -> str:
        """
        Format a response with character attribution for the chat display.
        Used in group chat to show who said what.
        """
        if char_manager:
            char = char_manager.get_character(character_id)
            if char:
                return f"**{char.name}:** {response}"
        return f"**{character_id}:** {response}"
    
    def get_next_in_round_robin(self, current_speaker: str) -> Optional[str]:
        """Get next speaker in round-robin order (fallback method)."""
        if not self.active_characters or len(self.active_characters) < 2:
            return None
            
        try:
            curr_idx = self.active_characters.index(current_speaker)
            next_idx = (curr_idx + 1) % len(self.active_characters)
            return self.active_characters[next_idx]
        except ValueError:
            return self.active_characters[0] if self.active_characters else None


# Global Instance
GROUP_MANAGER = GroupChatManager()
