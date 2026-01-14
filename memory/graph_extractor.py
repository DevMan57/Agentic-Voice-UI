import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Try to import local NuExtract model
try:
    from .nuextract import nuextract_llm_client, is_available as nuextract_available
    NUEXTRACT_IMPORTED = True
except ImportError:
    NUEXTRACT_IMPORTED = False
    nuextract_available = lambda: False
    nuextract_llm_client = None

@dataclass
class GraphUpdate:
    entities: List[Dict[str, str]]
    relationships: List[Dict[str, Any]]

class GraphExtractor:
    """
    "The Scribe": A background process that reads conversation history
    and converts it into a structured Knowledge Graph.

    By default uses NuExtract (local, specialized) for extraction.
    Falls back to provided llm_client if NuExtract unavailable.
    """

    def __init__(self, llm_client=None, use_local: bool = True, max_workers: int = 3):
        self.llm_client = llm_client
        self.use_local = use_local and NUEXTRACT_IMPORTED and nuextract_available()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="graph_")

        if self.use_local:
            print("[Graph] Using NuExtract for local extraction")
        elif llm_client:
            print("[Graph] Using remote LLM for extraction")
        else:
            print("[Graph] Warning: No extraction backend available")

    def extract(self, character_id: str, user_text: str, assistant_text: str) -> GraphUpdate:
        """
        Analyze the interaction and return structured graph updates.
        This is designed to run in a background thread to avoid blocking TTS.
        """
        # If texts are too short, skip extraction to save tokens/time
        if len(user_text) < 10 and len(assistant_text) < 10:
            return GraphUpdate([], [])

        # Choose extraction backend
        if self.use_local:
            return self._extract_with_nuextract(character_id, user_text, assistant_text)
        elif self.llm_client:
            return self._extract_with_llm(character_id, user_text, assistant_text)
        else:
            return GraphUpdate([], [])

    def _extract_with_nuextract(self, character_id: str, user_text: str, assistant_text: str) -> GraphUpdate:
        """Extract using local NuExtract model."""
        try:
            from .nuextract import extract_entities_and_relationships

            result = extract_entities_and_relationships(
                user_text=user_text,
                assistant_text=assistant_text,
                character_id=character_id
            )

            return GraphUpdate(
                entities=result.get('entities', []),
                relationships=result.get('relationships', [])
            )
        except Exception as e:
            print(f"[Graph] NuExtract extraction failed: {e}")
            # Fall back to LLM if available
            if self.llm_client:
                return self._extract_with_llm(character_id, user_text, assistant_text)
            return GraphUpdate([], [])

    def _extract_with_llm(self, character_id: str, user_text: str, assistant_text: str) -> GraphUpdate:
        """Extract using remote LLM client."""
        prompt = f"""
        You are a Knowledge Graph extraction system.
        Analyze this conversation for PERMANENT facts to add to the character's long-term memory.

        CRITICAL RULES:
        1. Extract Entities: People, Locations, Concepts, Items, Projects.
        2. Extract Relationships: How these entities connect (KNOWS, OWNS, LOCATED_IN, DEFINED_AS).
        3. Resolve Pronouns: "I live in London" -> Entity: "User", Relation: "LIVES_IN", Target: "London".
        4. IGNORE casual chatter, greetings, or temporary states ("I am tired").
        5. Use consistent naming: "Harry", "Potter", "Mr. Potter" -> "Harry Potter".

        Conversation:
        User: {user_text}
        Assistant ({character_id}): {assistant_text}

        Return ONLY valid JSON in this format (no markdown):
        {{
            "entities": [
                {{"name": "Entity Name", "type": "Person|Location|Concept|Item", "description": "Context"}}
            ],
            "relationships": [
                {{"source": "Entity Name", "target": "Entity Name", "relation": "RELATION_TYPE", "strength": 0.1_to_1.0}}
            ]
        }}
        """

        try:
            # Low temperature for deterministic JSON
            response = self.llm_client(prompt, temperature=0.0)

            # Extract JSON from potential markdown wrappers
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return GraphUpdate([], [])

            data = json.loads(json_match.group())

            return GraphUpdate(
                entities=data.get('entities', []),
                relationships=data.get('relationships', [])
            )
        except Exception as e:
            print(f"[Graph] LLM extraction failed: {e}")
            return GraphUpdate([], [])

    def extract_async(self, storage, character_id: str, user_text: str, assistant_text: str):
        """
        Fire-and-forget method to run extraction in the background.
        Uses thread pool to limit concurrent extractions (default: 3 workers).
        """
        def _run():
            try:
                updates = self.extract(character_id, user_text, assistant_text)

                if not updates.entities and not updates.relationships:
                    return

                # Write to Storage (SQLite is thread-safe for WAL mode)
                count_nodes = 0
                count_edges = 0

                for entity in updates.entities:
                    storage.add_graph_node(
                        character_id,
                        entity['name'],
                        entity.get('type', 'Concept'),
                        entity.get('description', '')
                    )
                    count_nodes += 1

                for rel in updates.relationships:
                    storage.add_graph_edge(
                        character_id,
                        rel['source'],
                        rel['target'],
                        rel['relation']
                    )
                    count_edges += 1

                if count_nodes > 0:
                    print(f"[Graph] Background update: +{count_nodes} Nodes, +{count_edges} Edges for {character_id}")
            except Exception as e:
                print(f"[Graph] Async update error: {e}")

        # Submit to thread pool (limits concurrent extractions)
        self._executor.submit(_run)

    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool. Call this when application exits."""
        self._executor.shutdown(wait=wait)