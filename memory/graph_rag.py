"""
GraphRAG Community Detection and Global Query Routing

Implements Microsoft GraphRAG-style community detection for answering
"global" questions like "What are the main themes?" or "How has my mood changed?"

Uses networkx for graph operations and Leiden algorithm for community detection.
"""

import hashlib
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("[GraphRAG] WARNING: networkx not installed. Community detection disabled.")

# cdlib is optional - we fall back to networkx community detection if unavailable
try:
    from cdlib import algorithms as cd_algorithms
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False
    print("[GraphRAG] Note: cdlib not installed. Using networkx community detection.")


@dataclass
class Community:
    """A community of related entities"""
    id: str
    level: int
    entity_ids: List[str]
    entity_names: List[str]
    summary: str
    keywords: List[str]
    embedding: Optional[List[float]] = None


class GraphRAGProcessor:
    """
    Handles community detection and global query processing.

    Architecture:
    1. Build networkx graph from entities and relationships
    2. Detect communities using Leiden (or Louvain fallback)
    3. Generate summaries for each community via LLM
    4. Route queries to appropriate retrieval method
    """

    # Patterns that indicate a "global" query (needs community summaries)
    GLOBAL_QUERY_PATTERNS = [
        r"\b(what are the|main|overall|general|recurring|common)\s+(themes?|topics?|patterns?)\b",
        r"\b(how has|how have|how did)\s+(my|our|the)\s+\w+\s+(change|evolve|progress)\b",
        r"\b(summarize|summary|overview|recap)\s+(of|our|the|all)\b",
        r"\bwhat do (we|i) (usually|often|typically)\s+(talk|discuss|chat)\s+about\b",
        r"\b(most|frequently|common)\s+(discussed|mentioned|talked about)\b",
        r"\brelationship\s+(over time|history|progression)\b",
        r"\ball (of )?our (conversations?|chats?|interactions?)\b",
        r"\b(entire|whole|full)\s+(history|conversation|chat)\b",
    ]

    MIN_ENTITIES_FOR_COMMUNITIES = 5

    def __init__(
        self,
        storage,
        embedding_manager,
        llm_client=None
    ):
        """
        Initialize GraphRAG processor.

        Args:
            storage: SQLiteStorage instance
            embedding_manager: EmbeddingManager for generating embeddings
            llm_client: Optional LLM callable for summary generation
        """
        self.storage = storage
        self.embeddings = embedding_manager
        self.llm_client = llm_client

    def is_global_query(self, query: str) -> bool:
        """
        Determine if a query requires global (community-based) retrieval.

        Args:
            query: The user's query

        Returns:
            True if this is a global query needing community summaries
        """
        query_lower = query.lower()

        for pattern in self.GLOBAL_QUERY_PATTERNS:
            if re.search(pattern, query_lower):
                return True

        return False

    def build_graph(self, character_id: str) -> Optional['nx.Graph']:
        """
        Build a networkx graph from the knowledge graph.

        Args:
            character_id: Character to build graph for

        Returns:
            networkx Graph or None if not enough data
        """
        if not NETWORKX_AVAILABLE:
            return None

        entities = self.storage.get_all_entities(character_id)
        relationships = self.storage.get_all_relationships(character_id)

        if len(entities) < self.MIN_ENTITIES_FOR_COMMUNITIES:
            return None

        G = nx.Graph()

        # Add nodes with attributes
        for entity in entities:
            G.add_node(
                entity['name'],
                entity_type=entity.get('entity_type', 'unknown'),
                description=entity.get('description', '')
            )

        # Add edges with relationship type as attribute
        for rel in relationships:
            source = rel['source_entity']
            target = rel['target_entity']
            rel_type = rel['relation_type']
            strength = rel.get('strength', 1.0)

            if G.has_node(source) and G.has_node(target):
                G.add_edge(source, target, relation=rel_type, weight=strength)

        return G

    def detect_communities(self, G: 'nx.Graph') -> List[List[str]]:
        """
        Detect communities in the graph.

        Uses Leiden algorithm if cdlib available, otherwise falls back to Louvain.

        Args:
            G: networkx Graph

        Returns:
            List of communities (each community is a list of entity names)
        """
        if G is None or len(G.nodes()) < 3:
            return []

        # Try Leiden first (better quality), fall back to Louvain
        if CDLIB_AVAILABLE:
            try:
                result = cd_algorithms.leiden(G)
                communities = result.communities
            except Exception as e:
                print(f"[GraphRAG] Leiden failed, using Louvain: {e}")
                communities = self._louvain_fallback(G)
        else:
            communities = self._louvain_fallback(G)

        # Filter out tiny communities (less than 2 nodes)
        return [c for c in communities if len(c) >= 2]

    def _louvain_fallback(self, G: 'nx.Graph') -> List[List[str]]:
        """Fallback community detection using networkx Louvain"""
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, seed=42)
            return [list(c) for c in communities]
        except Exception as e:
            print(f"[GraphRAG] Louvain also failed: {e}")
            # Ultimate fallback: treat connected components as communities
            return [list(c) for c in nx.connected_components(G)]

    def generate_community_summary(
        self,
        community_entities: List[str],
        character_id: str
    ) -> Tuple[str, List[str]]:
        """
        Generate a summary for a community of entities.

        Args:
            community_entities: List of entity names in the community
            character_id: Character this community belongs to

        Returns:
            Tuple of (summary_text, keywords_list)
        """
        # Get entity details
        all_entities = self.storage.get_all_entities(character_id)
        entity_details = [
            e for e in all_entities if e['name'] in community_entities
        ]

        # Get relationships within community
        all_rels = self.storage.get_all_relationships(character_id)
        community_set = set(community_entities)
        internal_rels = [
            r for r in all_rels
            if r['source_entity'] in community_set and r['target_entity'] in community_set
        ]

        # Build context for LLM
        entities_text = "\n".join([
            f"- {e['name']} ({e.get('entity_type', 'unknown')}): {e.get('description', 'no description')}"
            for e in entity_details
        ])

        rels_text = "\n".join([
            f"- {r['source_entity']} --[{r['relation_type']}]--> {r['target_entity']}"
            for r in internal_rels[:20]  # Limit to 20 relationships
        ])

        if self.llm_client:
            return self._llm_summarize_community(entities_text, rels_text)
        else:
            return self._heuristic_summarize(entity_details, internal_rels)

    def _llm_summarize_community(
        self,
        entities_text: str,
        rels_text: str
    ) -> Tuple[str, List[str]]:
        """Use LLM to generate community summary"""
        prompt = f"""Analyze this cluster of related concepts from a conversation history:

ENTITIES:
{entities_text}

RELATIONSHIPS:
{rels_text}

Generate a brief summary (2-3 sentences) that captures:
1. The main theme or topic of this cluster
2. Key relationships between entities
3. Any notable patterns

Also provide 3-5 keywords that best describe this cluster.

Respond in JSON:
{{"summary": "...", "keywords": ["keyword1", "keyword2", ...]}}
"""

        try:
            response = self.llm_client(prompt)
            import json
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('summary', ''), data.get('keywords', [])
        except Exception as e:
            print(f"[GraphRAG] LLM summarization failed: {e}")

        return self._heuristic_summarize_from_text(entities_text)

    def _heuristic_summarize(
        self,
        entities: List[Dict],
        relationships: List[Dict]
    ) -> Tuple[str, List[str]]:
        """Simple heuristic summary without LLM"""
        entity_types = {}
        for e in entities:
            t = e.get('entity_type', 'unknown')
            entity_types[t] = entity_types.get(t, 0) + 1

        main_type = max(entity_types, key=entity_types.get) if entity_types else 'concepts'
        entity_names = [e['name'] for e in entities[:5]]

        summary = f"A cluster about {main_type} including: {', '.join(entity_names)}"
        if len(entities) > 5:
            summary += f" and {len(entities) - 5} more"

        keywords = list(entity_types.keys()) + entity_names[:3]

        return summary, keywords

    def _heuristic_summarize_from_text(self, entities_text: str) -> Tuple[str, List[str]]:
        """Extract simple summary from entities text"""
        lines = entities_text.strip().split('\n')
        names = []
        for line in lines[:5]:
            if line.startswith('- '):
                name = line[2:].split('(')[0].strip()
                names.append(name)

        summary = f"Cluster containing: {', '.join(names)}"
        return summary, names

    def update_communities(self, character_id: str) -> int:
        """
        Rebuild communities for a character.

        This should be called periodically (e.g., after N interactions)
        or during "sleep time" processing.

        Args:
            character_id: Character to update communities for

        Returns:
            Number of communities created
        """
        # Build graph
        G = self.build_graph(character_id)
        if G is None:
            return 0

        # Detect communities
        communities = self.detect_communities(G)
        if not communities:
            return 0

        # Clear old communities
        self.storage.clear_communities(character_id)

        # Generate and store new communities
        count = 0
        for i, comm_entities in enumerate(communities):
            # Generate summary
            summary, keywords = self.generate_community_summary(comm_entities, character_id)

            # Generate embedding for the summary
            embedding = self.embeddings.embed(summary, is_query=False)

            # Create community ID
            comm_id = hashlib.md5(
                f"{character_id}:community:{i}:{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()

            # Store
            self.storage.store_community(
                community_id=comm_id,
                character_id=character_id,
                level=0,  # Single level for now
                entity_ids=comm_entities,
                summary=summary,
                keywords=keywords,
                embedding=embedding
            )
            count += 1

        print(f"[GraphRAG] Created {count} communities for {character_id}")
        return count

    def answer_global_query(
        self,
        character_id: str,
        query: str
    ) -> Optional[str]:
        """
        Answer a global query using community summaries.

        Args:
            character_id: Character to query
            query: The global query

        Returns:
            Aggregated answer from community summaries, or None if no communities
        """
        # Get query embedding
        query_embedding = self.embeddings.embed(query, is_query=True)

        # Search communities
        results = self.storage.search_communities(
            character_id=character_id,
            query_embedding=query_embedding,
            limit=5,
            min_similarity=0.2
        )

        if not results:
            # No communities - maybe trigger community generation
            count = self.update_communities(character_id)
            if count > 0:
                results = self.storage.search_communities(
                    character_id=character_id,
                    query_embedding=query_embedding,
                    limit=5,
                    min_similarity=0.2
                )

        if not results:
            return None

        # Aggregate community summaries
        summaries = []
        for community, score in results:
            summaries.append({
                'summary': community['summary'],
                'keywords': community['keywords'],
                'score': score
            })

        # If LLM available, synthesize answer
        if self.llm_client:
            return self._synthesize_global_answer(query, summaries)
        else:
            # Return formatted summaries
            lines = ["Based on our conversation history:\n"]
            for i, s in enumerate(summaries, 1):
                lines.append(f"{i}. {s['summary']}")
                if s['keywords']:
                    lines.append(f"   Keywords: {', '.join(s['keywords'])}")
            return "\n".join(lines)

    def _synthesize_global_answer(
        self,
        query: str,
        summaries: List[Dict]
    ) -> str:
        """Use LLM to synthesize answer from community summaries"""
        context = "\n\n".join([
            f"Theme {i+1} (relevance: {s['score']:.2f}):\n{s['summary']}\nKeywords: {', '.join(s['keywords'])}"
            for i, s in enumerate(summaries)
        ])

        prompt = f"""Based on these themes from the conversation history, answer the question.

QUESTION: {query}

CONVERSATION THEMES:
{context}

Provide a helpful, synthesized answer that draws on these themes. Be conversational and specific."""

        try:
            return self.llm_client(prompt)
        except Exception as e:
            print(f"[GraphRAG] Failed to synthesize answer: {e}")
            return f"Based on our conversations, the main themes are:\n{context}"
