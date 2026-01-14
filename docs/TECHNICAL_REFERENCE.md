# IndexTTS2 Voice Agent — Technical Reference

<div align="center">

**A State-of-the-Art Multi-Character AI Voice Agent System**

*Combining Weighted Memory Retrieval, Voice Cloning, and Modular Tool Execution*

[![Version](https://img.shields.io/badge/version-2.3.1-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)]()

</div>

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Memory System — Deep Dive](#3-memory-system--deep-dive)
   - [3.5 Knowledge Graph (Layer 4)](#35-knowledge-graph-layer-4)
4. [Embedding System](#4-embedding-system)
5. [Weighted Retrieval Algorithm](#5-weighted-retrieval-algorithm)
6. [MMR Re-ranking for Diversity](#6-mmr-re-ranking-for-diversity)
7. [Importance Scoring Heuristics](#7-importance-scoring-heuristics)
8. [Streaming TTS Pipeline](#8-streaming-tts-pipeline)
9. [Voice Activity Detection](#9-voice-activity-detection)
10. [MCP and Agent Tools](#10-mcp-and-agent-tools)
11. [Character System](#11-character-system)
12. [LLM Integration](#12-llm-integration)
13. [Data Flow & Processing Pipeline](#13-data-flow--processing-pipeline)
14. [Token Budget Management](#14-token-budget-management)
15. [Performance Characteristics](#15-performance-characteristics)
16. [Configuration Reference](#16-configuration-reference)
17. [API Reference](#17-api-reference)
18. [Future Research Directions](#18-future-research-directions)

---

## 1. Executive Summary

IndexTTS2 Voice Agent is a sophisticated multi-character AI voice conversation system that combines several cutting-edge techniques:

| Component | Technology | Key Innovation |
|-----------|------------|----------------|
| **Memory** | Qwen3-Embedding-0.6B ONNX + SQLite Graph | 4-layer architecture: Episodic, Semantic, Procedural, Knowledge Graph |
| **Tools** | Model Context Protocol (MCP) | Standardized agent tool interface, async capable |
| **TTS** | IndexTTS2 voice cloning | 5-second sample → high-fidelity synthesis |
| **STT** | faster-whisper (CTranslate2) | CPU-optimized with beam search |
| **VAD** | Silero/WebRTC/Energy backends | Adaptive speech boundary detection |
| **LLM** | OpenRouter + LM Studio | 100+ cloud models + local inference |

### Core Differentiators

1. **Weighted Retrieval Formula**: `Score = 0.2 × Recency + 0.5 × Relevance + 0.3 × Importance`
2. **MMR Re-ranking**: Balances relevance with diversity using λ=0.7
3. **Four-Layer Memory**: Episodic (decaying), Semantic (permanent facts), Procedural (behaviors), Knowledge Graph (entities)
4. **Character Isolation**: Complete memory separation between characters
5. **Streaming TTS**: Sentence-boundary detection for sub-second perceived latency

---

## 2. System Architecture

### 2. System Architecture

> **See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed visual diagrams of the Hybrid System, Memory Layers, and Voice Pipeline.**

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Gradio Web Interface (7861)                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │  Chat UI    │  │  Voice      │  │  Character  │  │  Settings  │ │   │
│  │  │  Component  │  │  Controls   │  │  Selector   │  │  Panel     │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│    INPUT LAYER       │  │   CORE ENGINE    │  │    OUTPUT LAYER      │
├──────────────────────┤  ├──────────────────┤  ├──────────────────────┤
│ ┌──────────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────────┐ │
│ │ PTT Listener     │ │  │ │   Memory     │ │  │ │ Streaming TTS    │ │
│ │ (Windows/Linux)  │ │  │ │   Manager    │ │  │ │ (IndexTTS2)      │ │
│ └──────────────────┘ │  │ └──────────────┘ │  │ └──────────────────┘ │
│ ┌──────────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────────┐ │
│ │ VAD Recorder     │ │  │ │  Character   │ │  │ │ Audio Queue      │ │
│ │ (Silero/WebRTC)  │ │  │ │  Manager     │ │  │ │ Manager          │ │
│ └──────────────────┘ │  │ └──────────────┘ │  │ └──────────────────┘ │
│ ┌──────────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────────┐ │
│ │ faster-whisper   │ │  │ │    Tool      │ │  │ │ Phrase Cache     │ │
│ │ STT (CPU)        │ │  │ │  Registry    │ │  │ │ System           │ │
│ └──────────────────┘ │  │ └──────────────┘ │  │ └──────────────────┘ │
└──────────────────────┘  │ ┌──────────────┐ │  └──────────────────────┘
                          │ │  Streaming   │ │
                          │ │  LLM Client  │ │
                          │ └──────────────┘ │
                          └──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │   OpenRouter     │  │   LM Studio      │  │   Local Storage  │
    │   (100+ models)  │  │   (localhost)    │  │   (JSON/SQLite)  │
    └──────────────────┘  └──────────────────┘  └──────────────────┘
```

### 2.2 Module Dependency Graph

```
voice_chat_app.py (Main Application)
├── memory/
│   ├── memory_manager.py      # Core memory system
│   │   ├── EmbeddingManager   # Qwen3-Embedding-0.6B ONNX (1024-dim)
│   │   ├── WeightedRetrievalScorer
│   │   ├── LocalStorage / SupabaseStorage
│   │   └── MultiCharacterMemoryManager
│   └── characters.py          # Character definitions
├── streaming.py               # Streaming TTS pipeline
│   ├── SentenceBuffer         # Sentence boundary detection
│   ├── StreamingLLMClient     # Streaming API handler
│   └── AudioQueue             # Gapless playback
├── audio/
│   ├── vad_recorder.py        # Voice activity detection
│   ├── ptt_windows.py         # Push-to-talk (Windows)
│   └── ptt_linux.py           # Push-to-talk (Linux)
├── tools/
│   └── __init__.py            # Tool registry and implementations
└── utils.py                   # Shared utilities
```

---

## 3. Memory System — Deep Dive

The memory system is the cognitive core of IndexTTS2 Voice Agent, implementing a three-layer architecture inspired by human memory systems and modern retrieval-augmented generation (RAG) research.

### 3.1 Three-Layer Memory Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MEMORY ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 1: EPISODIC MEMORY                      │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  • Past conversations and interactions                           │   │
│  │  • Subject to temporal decay: factor = 0.995^hours               │   │
│  │  • Uses MMR re-ranking for diverse recall                        │   │
│  │  • Hierarchical summarization after N interactions               │   │
│  │  • Token budget: ~2,000 tokens                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 2: SEMANTIC MEMORY                      │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  • Permanent facts about user and world                          │   │
│  │  • No decay (decay_factor = 1.0 always)                         │   │
│  │  • High importance threshold (default 0.8)                       │   │
│  │  • Examples: "User's name is Henri", "User lives in Tasmania"   │   │
│  │  • Token budget: ~3,000 tokens                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   LAYER 3: PROCEDURAL MEMORY                     │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  • Behavioral patterns and interaction styles                    │   │
│  │  • No decay (permanent behavioral knowledge)                     │   │
│  │  • Default importance: 0.9                                       │   │
│  │  • Examples: "Speaks with British accent", "Uses formal tone"   │   │
│  │  • Token budget: ~500 tokens                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Memory Data Structure

```python
@dataclass
class Memory:
    id: str                           # MD5 hash of character_id + timestamp + content
    character_id: str                 # Owning character (isolation boundary)
    memory_type: str                  # 'episodic' | 'semantic' | 'procedural'
    content: str                      # The actual memory text
    embedding: List[float]            # 384-dim E5-small-v2 vector
    importance_score: float           # 0.0 to 1.0 (multi-factor heuristic)
    decay_factor: float               # Current decay multiplier
    tags: Dict[str, Any]              # Metadata (emotion, entities, etc.)
    created_at: datetime              # Creation timestamp
    updated_at: datetime              # Last modification
    last_accessed: datetime           # For recency calculation
    access_count: int                 # Retrieval frequency
```

### 3.3 Character State Tracking

```python
@dataclass
class CharacterState:
    character_id: str
    current_mood: str                 # "neutral", "happy", "concerned", etc.
    emotional_state: Dict[str, float] # VAD model (Valence, Arousal, Dominance)
    #   valence: -1 (negative) to 1 (positive)
    #   arousal: -1 (calm) to 1 (excited)  
    #   dominance: -1 (submissive) to 1 (dominant)
    relationship_with_user: Dict[str, float]
    #   familiarity: 0 (stranger) to 1 (intimate)
    #   trust: 0 to 1
    #   affection: -1 to 1
    session_context: List[Dict]       # Rolling window of recent exchanges
    interaction_count: int            # Total interactions with this character
```

### 3.4 Hierarchical Summarization

When `interaction_count % SUMMARY_INTERVAL == 0` (default: every 10 interactions), the system generates hierarchical summaries:

```python
@dataclass
class EpisodicSummary:
    summary_short: str    # <100 tokens - Quick context snippet
    summary_medium: str   # <300 tokens - Detailed recap
    summary_long: str     # <800 tokens - Full narrative summary
    key_entities: List[str]           # Named entities mentioned
    emotional_arc: str                # "started anxious, ended relieved"
    plot_points: List[str]            # Story-relevant developments
    relationship_delta: Dict[str, str] # {"trust": "+0.1", "familiarity": "+0.2"}
    embedding: List[float]            # For similarity search on summaries
```
### 3.5 Knowledge Graph (Layer 4)

The Knowledge Graph is the newest layer, implementing persistent entity-relationship storage for deep reasoning about the user's world.

#### 3.5.1 Architecture

```
+-------------------------------------------------------------------------+
|                       KNOWLEDGE GRAPH LAYER                             |
+-------------------------------------------------------------------------+
|                                                                          |
|  +-------------------------------------------------------------------+   |
|  |                        ENTITIES (Nodes)                          |   |
|  |  -----------------------------------------------------------------  |   |
|  |  - name: Unique identifier within character context              |   |
|  |  - entity_type: Person | Location | Concept | Item | Project    |   |
|  |  - description: Contextual information about the entity          |   |
|  |  - Isolated per-character (character_id foreign key)             |   |
|  +-------------------------------------------------------------------+   |
|                              |                                          |
|                              | Relationships                            |
|                              v                                          |
|  +-------------------------------------------------------------------+   |
|  |                     RELATIONSHIPS (Edges)                        |   |
|  |  -----------------------------------------------------------------  |   |
|  |  - source_entity -> target_entity                                 |   |
|  |  - relation_type: KNOWS | OWNS | LOCATED_IN | WORKS_AT | etc.   |   |
|  |  - strength: 0.0 - 1.0 confidence score                          |   |
|  |  - Directed edges with semantic meaning                          |   |
|  +-------------------------------------------------------------------+   |
|                                                                          |
+-------------------------------------------------------------------------+
```

#### 3.5.2 Entity Extraction (The Graph Extractor)

The ``GraphExtractor`` class runs in a background thread after each interaction, analyzing conversation content for permanent facts:

```python
# memory/graph_extractor.py
class GraphExtractor:
    def extract(self, character_id: str, user_text: str, assistant_text: str) -> GraphUpdate:
        """
        Uses LLM to extract entities and relationships from conversation.
        Runs asynchronously to avoid blocking TTS pipeline.
        """
        prompt = """
        Analyze this conversation for PERMANENT facts.
        
        CRITICAL RULES:
        1. Extract Entities: People, Locations, Concepts, Items, Projects
        2. Extract Relationships: How entities connect
        3. Resolve Pronouns: "I live in London" -> Entity: "User", Relation: "LIVES_IN", Target: "London"
        4. IGNORE casual chatter, greetings, temporary states
        5. Use consistent naming: "Harry", "Potter" -> "Harry Potter"
        """
```

**Extraction Rules:**
- Minimum text length threshold (10 chars) before extraction
- Low temperature (0.0) for deterministic JSON output
- Fire-and-forget threading to avoid blocking
- Automatic pronoun resolution ("I" -> "User", "you" -> character name)

#### 3.5.3 Database Schema

```sql
-- Entities (Nodes)
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL,
    name TEXT NOT NULL,
    entity_type TEXT,           -- Person|Location|Concept|Item|Project
    description TEXT,
    meta_data TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(character_id, name)  -- One entity per name per character
);

-- Relationships (Edges)
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_id TEXT NOT NULL,
    source_entity TEXT NOT NULL,
    target_entity TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    strength REAL DEFAULT 1.0,  -- Confidence/importance score
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(character_id, source_entity, target_entity, relation_type)
);
```

#### 3.5.4 Graph Operations

The SQLite storage provides these graph operations:

```python
# Add a node (upsert)
storage.add_graph_node(
    character_id="hermione",
    name="User",
    entity_type="Person",
    description="The human interacting with Hermione"
)

# Add an edge
storage.add_graph_edge(
    character_id="hermione",
    source="User",
    target="Tasmania",
    relation_type="LIVES_IN"
)

# Query relationships
edges = storage.get_entity_relationships(
    character_id="hermione",
    entity_name="User"
)
# Returns: [{"source": "User", "target": "Tasmania", "relation": "LIVES_IN", "strength": 1.0}]
```

#### 3.5.5 Visualization

The memory manager can export the knowledge graph for visualization:

```python
# Generate interactive HTML visualization
html_path = manager.generate_memory_graph("hermione")
# Uses networkx + pyvis to create force-directed graph
```

The visualization shows:
- Nodes colored by entity type
- Edges labeled with relationship type
- Interactive pan/zoom/drag

---

## 4. Embedding System

### 4.1 Model Selection: Qwen3-Embedding-0.6B ONNX

The system uses `Svenni551/Qwen3-Embedding-0.6B-ONNX-INT8` as the default embedding model, chosen for superior quality and speed:

| Metric | Qwen3-Embedding-0.6B | E5-small-v2 (legacy) | Improvement |
|--------|----------------------|----------------------|-------------|
| **Embedding Dimension** | 1024 | 384 | +167% |
| **Multilingual Support** | 100+ languages | English-focused | Much better |
| **Quantization** | INT8 ONNX | FP32 | ~2x faster |
| **Model Size** | ~600MB | 134MB | Larger but faster |
| **Inference Backend** | onnxruntime | sentence-transformers | More optimized |

### 4.2 Qwen3 Instruction Format

Qwen3 embedding models use instruction prefixes for optimal retrieval:

```python
def _prepare_text(text: str, is_query: bool = False) -> str:
    """
    Qwen3 uses instruction format for queries:
    - Queries: "Instruct: <task>\nQuery:<text>"
    - Documents: plain text (no prefix)
    """
    if is_query:
        task = "Given a query, retrieve relevant information"
        return f"Instruct: {task}\nQuery:{text}"
    return text
```

**Critical**: Always use `is_query=True` for search queries and `is_query=False` for documents being stored. This improves retrieval quality by 1-5%.

### 4.3 Embedding Generation Pipeline

```
Input Text → Instruction Format → Tokenization → ONNX Forward Pass → Mean Pooling → L2 Normalize → 1024-dim Vector
     │              │                │                 │                   │              │
     │              │                │                 │                   │              └─ Output: List[float]
     │              │                │                 │                   │
     │              │                │                 │                   └─ Average token embeddings
     │              │                │                 │
     │              │                │                 └─ ORTModelForFeatureExtraction
     │              │                │
     │              │                └─ Qwen tokenizer (up to 8192 tokens)
     │              │
     │              └─ "Instruct: ...\nQuery:" for queries, plain for docs
     │
     └─ Raw user/assistant text
```

---

## 5. Weighted Retrieval Algorithm

### 5.1 The Core Formula

The weighted retrieval system implements a state-of-the-art scoring function combining three signals:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│        SCORE = 0.2 × RECENCY + 0.5 × RELEVANCE + 0.3 × IMPORTANCE       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Component Definitions

#### Recency Score (Weight: 0.2)

Exponential decay based on time since last access:

```
RECENCY = 0.995^(hours_since_last_access)
```

| Hours Since Access | Recency Score |
|-------------------|---------------|
| 0 (just accessed) | 1.000 |
| 1 hour | 0.995 |
| 24 hours (1 day) | 0.887 |
| 168 hours (1 week) | 0.431 |
| 720 hours (1 month) | 0.027 |

**Decay Curve Visualization:**
```
1.0 │●
    │ ●
0.8 │  ●
    │   ●
0.6 │    ●
    │     ●●
0.4 │       ●●
    │         ●●●
0.2 │            ●●●●●
    │                 ●●●●●●●●●
0.0 │─────────────────────────────────
    0   24   48   72   96  120  144  168  (hours)
```

#### Relevance Score (Weight: 0.5)

Cosine similarity between query embedding and memory embedding:

```
RELEVANCE = cos(query_vector, memory_vector) = (q · m) / (||q|| × ||m||)
```

Implementation:
```python
def cosine_similarity(query_vec: np.ndarray, mem_vec: np.ndarray) -> float:
    dot_product = np.dot(query_vec, mem_vec)
    norms = np.linalg.norm(query_vec) * np.linalg.norm(mem_vec)
    return dot_product / (norms + 1e-8)  # Epsilon for numerical stability
```

#### Importance Score (Weight: 0.3)

Multi-factor heuristic computed during memory formation (see Section 7 for details).

### 5.3 Implementation

```python
class WeightedRetrievalScorer:
    RECENCY_WEIGHT = 0.2
    RELEVANCE_WEIGHT = 0.5
    IMPORTANCE_WEIGHT = 0.3
    RECENCY_DECAY_BASE = 0.995  # Per-hour decay
    
    @classmethod
    def compute_score(cls, memory: Memory, relevance_score: float, now: datetime = None) -> float:
        now = now or datetime.utcnow()
        
        # Recency: exponential decay
        hours_since_access = (now - memory.last_accessed).total_seconds() / 3600
        recency_score = math.pow(cls.RECENCY_DECAY_BASE, hours_since_access)
        
        # Importance: stored in memory
        importance_score = memory.importance_score
        
        # Weighted combination
        final_score = (
            cls.RECENCY_WEIGHT * recency_score +
            cls.RELEVANCE_WEIGHT * relevance_score +
            cls.IMPORTANCE_WEIGHT * importance_score
        )
        
        return final_score
```

---

## 6. MMR Re-ranking for Diversity

### 6.1 The Redundancy Problem

Pure relevance-based retrieval often returns semantically similar memories, wasting context tokens on redundant information. MMR (Maximal Marginal Relevance) addresses this by penalizing memories similar to already-selected ones.

### 6.2 MMR Formula

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   MMR(m) = λ × Score(m) - (1 - λ) × max[Sim(m, s) for s in Selected]   │
│                                                                          │
│   Where:                                                                 │
│   • λ = 0.7 (default) - trade-off parameter                             │
│   • Score(m) = weighted retrieval score from Section 5                  │
│   • Sim(m, s) = cosine similarity between memory m and selected s       │
│   • Selected = set of already-selected memories                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 λ Parameter Effects

| λ Value | Behavior | Use Case |
|---------|----------|----------|
| 1.0 | Pure relevance (no diversity) | When redundancy is acceptable |
| 0.7 | Balanced (default) | General conversation |
| 0.5 | Strong diversity | Exploratory queries |
| 0.3 | Very diverse | Creative/brainstorming |

### 6.4 MMR Selection Algorithm

```python
def rank_with_mmr(memories_with_relevance, limit=10, lambda_param=0.7):
    """
    Greedy MMR selection algorithm.
    Time complexity: O(k × n) where k=limit, n=candidates
    """
    selected = []
    selected_embeddings = []
    candidates = [...prepare candidates with weighted scores...]
    
    while len(selected) < limit and candidates:
        best_idx = 0
        best_mmr_score = float('-inf')
        
        for i, candidate in enumerate(candidates):
            # Relevance term
            relevance_term = lambda_param * candidate['weighted_score']
            
            # Diversity penalty
            if selected_embeddings:
                max_sim = max(
                    cosine_sim(candidate['embedding'], sel_emb)
                    for sel_emb in selected_embeddings
                )
                diversity_term = (1 - lambda_param) * max_sim
                mmr_score = relevance_term - diversity_term
            else:
                mmr_score = relevance_term
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i
        
        # Add best candidate
        best = candidates.pop(best_idx)
        selected.append(best)
        selected_embeddings.append(best['embedding'])
    
    return selected
```

### 6.5 MMR Application in Context Building

MMR is applied **only to episodic memories** (Layer 1) because:
- Episodic memories are most likely to contain redundant conversational turns
- Semantic memories (facts) are typically unique
- Procedural memories (behaviors) are non-redundant by design

```python
# In build_context():
episodic_results = self.weighted_search(
    character_id=character_id,
    query=current_query,
    limit=5,
    memory_type='episodic',
    use_mmr=True,        # Enable MMR for episodic
    mmr_lambda=0.7       # Balance relevance and diversity
)
```

---

## 7. Importance Scoring Heuristics

### 7.1 Multi-Factor Scoring System

When LLM-based scoring is unavailable, the system uses a sophisticated heuristic:

```python
def _score_importance(content: str, character_id: str) -> float:
    score = 0.4  # Base score
    content_lower = content.lower()
    
    # Factor 1: Emotional Significance (0.0 - 0.25)
    strong_emotions = ['love', 'hate', 'terrified', 'ecstatic', 'devastated', 'furious']
    mild_emotions = ['happy', 'sad', 'angry', 'excited', 'nervous', 'worried']
    
    if any(word in content_lower for word in strong_emotions):
        score += 0.25
    elif any(word in content_lower for word in mild_emotions):
        score += 0.12
    
    # Factor 2: Personal Information Density (0.0 - 0.25)
    personal_phrases = [
        'my name is', 'i am', 'i work', 'i live', 'my family',
        'my job', 'my home', 'i was born', 'my favorite'
    ]
    personal_matches = sum(1 for phrase in personal_phrases if phrase in content_lower)
    score += min(0.25, personal_matches * 0.08)
    
    # Factor 3: Interaction Depth (0.0 - 0.15)
    question_count = content.count('?')
    if question_count >= 3:
        score += 0.15
    elif question_count >= 1:
        score += 0.08
    
    # Factor 4: Relationship Development (0.0 - 0.15)
    relationship_positive = ['thank you', 'appreciate', 'trust', 'friend']
    relationship_negative = ['disappointed', 'hurt', 'angry at', 'lied']
    
    if any(phrase in content_lower for phrase in relationship_positive):
        score += 0.1
    if any(phrase in content_lower for phrase in relationship_negative):
        score += 0.15  # Negative events are more memorable
    
    # Factor 5: Content Complexity (0.0 - 0.1)
    word_count = len(content.split())
    if word_count > 200:
        score += 0.1
    elif word_count > 100:
        score += 0.05
    
    return min(1.0, score)
```

### 7.2 Importance Score Distribution

Based on heuristics, typical score ranges:

| Content Type | Expected Score Range |
|--------------|---------------------|
| Generic greeting | 0.40 - 0.45 |
| Casual conversation | 0.45 - 0.55 |
| Personal revelation | 0.60 - 0.75 |
| Emotional moment | 0.65 - 0.80 |
| Major life event | 0.75 - 0.90 |
| Relationship turning point | 0.80 - 1.00 |

---

## 8. Streaming TTS Pipeline

### 8.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STREAMING TTS PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LLM Response Stream                                                     │
│        │                                                                 │
│        ▼                                                                 │
│  ┌─────────────────┐                                                    │
│  │ SentenceBuffer  │ ──► Detects sentence boundaries                    │
│  │                 │     using regex + heuristics                       │
│  └────────┬────────┘                                                    │
│           │ Complete sentences                                           │
│           ▼                                                              │
│  ┌─────────────────┐     ┌─────────────────┐                           │
│  │ PhraseCache     │────►│ Cache Hit?      │                           │
│  │ Lookup          │     │ (common phrases)│                           │
│  └─────────────────┘     └────────┬────────┘                           │
│                                   │                                      │
│                    ┌──────────────┼──────────────┐                      │
│                    │ Yes          │              │ No                   │
│                    ▼              │              ▼                      │
│           ┌───────────────┐       │     ┌───────────────┐              │
│           │ Load Cached   │       │     │ IndexTTS2     │              │
│           │ Audio         │       │     │ Generation    │              │
│           └───────┬───────┘       │     └───────┬───────┘              │
│                   │               │             │                       │
│                   └───────────────┼─────────────┘                       │
│                                   ▼                                      │
│                          ┌───────────────┐                              │
│                          │ AudioQueue    │                              │
│                          │ (gapless)     │                              │
│                          └───────┬───────┘                              │
│                                  │                                       │
│                                  ▼                                       │
│                          Browser Playback                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Sentence Boundary Detection

```python
# Primary sentence endings
SENTENCE_ENDINGS = re.compile(r'[.!?]+(?:\s|$)|[。！？]+')

# Soft breaks for long sentences
SOFT_BREAKS = re.compile(r'[,;:]+\s|[，；：]+')

# Abbreviations that don't end sentences
ABBREVIATIONS = {
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr',
    'vs', 'etc', 'e.g', 'i.e', 'fig', 'inc', 'ltd'
}
```

**Edge Cases Handled:**
1. Abbreviations (Mr., Dr., etc.)
2. Unclosed quotation marks
3. Minimum sentence length threshold
4. Maximum wait time before forcing flush

### 8.3 Phrase Cache System

Common phrases are pre-generated and cached for instant playback:

```python
COMMON_PHRASES = [
    "I understand.",
    "That's interesting.",
    "Let me think about that.",
    "Good question.",
    "I see what you mean.",
    # ... ~50 common phrases
]

# Cache structure: sessions/audio_cache/{character_id}/{phrase_hash}.wav
```

**Cache Hit Rate:** Typically 15-25% for natural conversation.

### 8.4 Latency Characteristics

| Stage | Typical Latency |
|-------|-----------------|
| Sentence detection | <1ms |
| Cache lookup | <5ms |
| Cache hit playback | ~50ms |
| IndexTTS2 generation | 200-800ms per sentence |
| Audio encoding | ~20ms |

**Perceived Latency:** First audio typically plays within 300-500ms of LLM response start.

---

## 9. Voice Activity Detection

### 9.1 Backend Comparison

| Backend | Quality | CPU Usage | Dependencies | Best For |
|---------|---------|-----------|--------------|----------|
| **Silero** | ★★★★★ | Medium | torch | Production |
| **WebRTC** | ★★★★☆ | Low | webrtcvad | Low-resource |
| **Energy** | ★★★☆☆ | Very Low | None | Fallback |

### 9.2 Silero VAD Implementation

Silero VAD uses a neural network trained on diverse speech data:

```python
class SileroVAD:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model, _ = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad'
        )
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        # Expects 512 samples at 16kHz (32ms)
        tensor = torch.from_numpy(audio_chunk.astype(np.float32))
        speech_prob = self.model(tensor, 16000).item()
        return speech_prob > self.threshold
```

### 9.3 VAD State Machine

```
                    ┌─────────────────────────────┐
                    │                             │
                    ▼                             │
            ┌───────────────┐                     │
   ─────────►   LISTENING   │                     │
            └───────┬───────┘                     │
                    │ speech_detected             │
                    ▼                             │
            ┌───────────────┐                     │
            │   RECORDING   │                     │
            └───────┬───────┘                     │
                    │ silence > threshold         │
                    ▼                             │
            ┌───────────────┐                     │
            │  PROCESSING   │─────────────────────┘
            └───────────────┘     save & reset
```

### 9.4 Configuration Parameters

```python
@dataclass
class VADConfig:
    backend: str = "silero"           # 'silero', 'webrtc', 'energy' (silero recommended)
    silence_threshold: float = 0.8    # Seconds of silence before stopping
    min_speech_duration: float = 0.3  # Minimum valid speech length
    max_recording_duration: float = 60.0

    # Backend-specific
    energy_threshold: float = 0.022   # RMS energy for speech (raised from 0.015)
    webrtc_aggressiveness: int = 2    # 0-3 (higher = more aggressive)
    silero_threshold: float = 0.6     # Neural network confidence (raised from 0.5)
    consecutive_frames_required: int = 5  # Require 150ms of speech before recording
```

---

## 10. MCP and Agent Tools

### 10.1 Model Context Protocol Overview

IndexTTS2 implements the **Model Context Protocol (MCP)**, an open standard for connecting AI agents to external tools and services. This replaces the legacy tool registry pattern with a standardized, extensible architecture.

```
+-------------------------------------------------------------------------+
|                         MCP ARCHITECTURE                                 |
+-------------------------------------------------------------------------+
|                                                                          |
|  +-------------------------------------------------------------------+   |
|  |                      MCP CLIENT (mcp_client.py)                  |   |
|  |  -----------------------------------------------------------------  |   |
|  |  - MCPManager class manages all server connections               |   |
|  |  - Reads configuration from mcp_config.json                      |   |
|  |  - Maintains tool schema cache for LLM calls                     |   |
|  |  - Routes tool executions to correct server                      |   |
|  +-------------------------------------------------------------------+   |
|                              |                                          |
|                    stdio transport (stdin/stdout)                       |
|                              |                                          |
|         +--------------------+--------------------+                    |
|         v                    v                    v                    |
|  +-------------+     +-------------+     +-------------+             |
|  | Filesystem  |     |  (Future)   |     |  (Future)   |             |
|  |   Server    |     | Web Search  |     |  Database   |             |
|  +-------------+     +-------------+     +-------------+             |
|                                                                          |
+-------------------------------------------------------------------------+
```

### 10.2 MCP Client Implementation

The ``MCPManager`` class handles all MCP operations:

```python
# mcp_client.py
class MCPManager:
    def __init__(self, config_path: str = "mcp_config.json"):
        self.sessions: Dict[str, ClientSession] = {}
        self.tools_cache: List[Dict] = []
        self._tool_map: Dict[str, str] = {}  # tool_name -> server_name
    
    async def initialize(self):
        """Start all servers defined in config"""
        # Read config, connect to each server via stdio
    
    async def refresh_tools(self):
        """Query all servers for available tools"""
        # Cache OpenAI-compatible function schemas
    
    async def call_tool(self, name: str, arguments: Dict) -> str:
        """Execute tool on appropriate server"""
        # Route by tool name, return result text
```

### 10.3 Server Configuration

Servers are defined in ``mcp_config.json``:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["tools/mcp_server_local.py"],
      "env": {}
    },
    "web-search": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-key"
      }
    }
  }
}
```

### 10.4 Built-in Filesystem Server

The included ``tools/mcp_server_local.py`` provides sandboxed file operations:

| Tool | Description | Arguments |
|------|-------------|-----------|
| ``read_file`` | Read file contents | ``path: str`` |
| ``write_file`` | Create/overwrite file | ``path: str, content: str`` |
| ``list_files`` | List directory | ``path: str`` |

**Security:** All file operations are sandboxed to ``sessions/files/`` by default.

### 10.5 Per-Character Tool Access

Tools are enabled per-character in YAML definitions:

```yaml
# characters/assistant.yaml
allowed_tools:
  - read_file
  - write_file
  - list_files

# characters/hermione.yaml
allowed_tools: []  # No tools - pure roleplay immersion
```

### 10.6 Tool Execution Flow

```
User Message -> LLM decides tool needed -> MCPManager.call_tool()
                                               |
                                               v
                                       Route to correct server
                                               |
                                               v
                                       Execute via stdio
                                               |
                                               v
                                       Return result to LLM
                                               |
                                               v
                                       LLM generates final response
```

### 10.7 Adding New MCP Servers

To add external MCP servers (e.g., Brave Search):

1. Install the server package
2. Add configuration to ``mcp_config.json``
3. Restart the application
4. Verify connection: ``[MCP] Connected to 'server-name' checkmark``

```bash
# Example: Install Brave Search MCP server
npm install -g @anthropic/mcp-server-brave-search
```

### 10.8 Future: MCP Tasks (Async Operations)

The MCP 2025 specification introduces "Tasks" for long-running operations:

```python
# Future implementation
result = await mcp_client.call_tool_async(
    name="web_search",
    arguments={"query": "latest AI news"}
)
# Returns taskId immediately, polls for completion
```

This enables non-blocking tool execution for voice agents.

---
## 11. Character System

### 11.1 Character Definition Schema

```yaml
id: hermione                        # Unique identifier
name: Hermione Granger              # Internal name
display_name: "🧙‍♀️ Hermione Granger" # UI display name
default_voice: emmawatson.wav       # Voice reference file

system_prompt: |
  You are Hermione Granger, several years after the events of Hogwarts...
  [Detailed roleplay instructions]

personality_traits:
  - Intellectually curious and well-read
  - Slightly nervous but warm
  - Values knowledge and preparation

initial_memories:
  - Hermione graduated top of her class at Hogwarts
  - She now works at the Ministry of Magic
  - She maintains close friendships with Harry and Ron

speech_patterns:
  - Uses precise vocabulary
  - Occasionally references books and research
  - Speaks with British English patterns

allowed_tools: []                   # No tools for roleplay immersion

metadata:
  setting: "Post-Hogwarts, Ministry of Magic office"
  mood: "focused but friendly"
  background: "Brightest witch of her age"

tags:
  - fantasy
  - roleplay
  - harry-potter
```

### 11.2 Character Isolation

Each character maintains completely separate:
- Memory storage (episodic, semantic, procedural)
- Character state (mood, relationship, interaction count)
- Session context (conversation history)
- Voice reference (TTS voice)

```python
# Memory isolation is enforced by character_id in all queries
memories = storage.get_memories_by_character(
    character_id="hermione",  # Isolation boundary
    memory_type='episodic'
)
```

### 11.3 Character State Persistence

State is saved after every interaction to prevent data loss:

```python
def add_interaction(self, character_id, user_message, assistant_response):
    # ... process interaction ...
    
    # Save state immediately (was every 5, caused data loss)
    self.storage.save_character_state(state)
```

---

## 12. LLM Integration

### 12.1 Supported Providers

| Provider | Type | Models | Vision | Tools |
|----------|------|--------|--------|-------|
| **OpenRouter** | Cloud | 100+ | ✅ | ✅ |
| **LM Studio** | Local | Any GGUF | ❌ | ✅ |
| **Direct API** | Cloud | Varies | Varies | Varies |

### 12.2 Model Selection UI

Vision-capable models are marked with 👁️ in the dropdown:
- `openai/gpt-4o` 👁️
- `anthropic/claude-3.5-sonnet` 👁️
- `google/gemini-2.0-flash-exp` 👁️
- `x-ai/grok-vision-beta` 👁️
- `meta-llama/llama-3.3-70b-instruct`

### 12.3 LM Studio Integration

For Windows+WSL setups, LM Studio must enable "Serve on Local Network":

```python
def get_lm_studio_host():
    """Auto-detect LM Studio host from WSL"""
    if is_wsl():
        # Read Windows host IP from WSL
        with open('/etc/resolv.conf') as f:
            for line in f:
                if 'nameserver' in line:
                    return line.split()[1]
    return 'localhost'
```

---

## 13. Data Flow & Processing Pipeline

### 13.1 Complete Request-Response Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. AUDIO INPUT                                                            │
├──────────────────────────────────────────────────────────────────────────┤
│   User speaks → PTT/VAD captures → WAV file saved                        │
│                                    recordings/rec_TIMESTAMP.wav          │
└──────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. SPEECH-TO-TEXT                                                         │
├──────────────────────────────────────────────────────────────────────────┤
│   faster-whisper (CPU) → Transcription with timestamps                   │
│   Model: base/small, Beam size: 5, VAD filter: enabled                   │
└──────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. MEMORY RETRIEVAL                                                       │
├──────────────────────────────────────────────────────────────────────────┤
│   Query embedding (E5-small-v2, is_query=True)                           │
│                      │                                                    │
│   ┌──────────────────┼──────────────────┐                                │
│   ▼                  ▼                  ▼                                │
│   Semantic       Episodic           Procedural                           │
│   Search         Search + MMR       Search                               │
│                      │                                                    │
│                      ▼                                                    │
│   Weighted Scoring: 0.2×R + 0.5×V + 0.3×I                               │
└──────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 4. CONTEXT ASSEMBLY                                                       │
├──────────────────────────────────────────────────────────────────────────┤
│   System Prompt (500 tokens)                                             │
│   + Character State (1,000 tokens)                                       │
│   + Episode Summaries (4,000 tokens)                                     │
│   + Semantic Memories (3,000 tokens)                                     │
│   + Episodic Memories (2,000 tokens)                                     │
│   + Current Message (500 tokens)                                         │
│   ─────────────────────────────────                                      │
│   Total: ~11,000 tokens input                                            │
└──────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 5. LLM INFERENCE                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│   OpenRouter / LM Studio                                                 │
│   ┌─────────────────────────────────────────────────────────┐           │
│   │ If tools enabled:                                        │           │
│   │   → Tool call? → Execute → Feed result → Loop (max 5)   │           │
│   │ Else:                                                    │           │
│   │   → Stream response directly                             │           │
│   └─────────────────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 6. STREAMING TTS                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│   Response stream → SentenceBuffer → Sentence detected                   │
│                                           │                              │
│                          ┌────────────────┼────────────────┐             │
│                          ▼                                 ▼             │
│                     Cache hit?                        Cache miss         │
│                          │                                 │             │
│                     Load cached                      IndexTTS2           │
│                          │                           Generate            │
│                          └────────────────┬────────────────┘             │
│                                           ▼                              │
│                                      AudioQueue                          │
│                                           │                              │
│                                           ▼                              │
│                                    Browser Playback                      │
└──────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 7. MEMORY UPDATE                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│   Store episodic memory (user + assistant exchange)                      │
│   Update character state (mood, relationship, interaction_count)         │
│   Check summarization threshold (every 10 interactions)                  │
│   Persist state to storage                                               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 14. Token Budget Management

### 14.1 Context Window Allocation

Based on 16K context window target:

| Component | Token Budget | Purpose |
|-----------|-------------|---------|
| System Prompt | 500 | Character instructions |
| Character State | 1,000 | Mood, relationship, stats |
| Episode Summaries | 4,000 | Compressed past sessions |
| Semantic Memories | 3,000 | Key facts |
| Episodic Memories | 2,000 | Recent relevant interactions |
| Current Message | 500 | User's current input |
| **Input Total** | **~11,000** | |
| Response Budget | 5,000 | LLM's response space |
| **Total** | **~16,000** | |

### 14.2 Dynamic Budget Adjustment

```python
def build_context(self, character_id, current_query, max_tokens=16000):
    # Calculate available budget
    system_budget = 500
    state_budget = 1000
    response_reserve = 5000
    
    available = max_tokens - system_budget - state_budget - response_reserve
    
    # Allocate remaining budget (prioritize recent/relevant)
    summary_budget = min(4000, available * 0.4)
    semantic_budget = min(3000, available * 0.3)
    episodic_budget = min(2000, available * 0.2)
    message_budget = available - summary_budget - semantic_budget - episodic_budget
```

---

## 15. Performance Characteristics

### 15.1 Latency Breakdown

| Operation | Typical Latency | Notes |
|-----------|-----------------|-------|
| Audio capture | Real-time | PTT/VAD dependent |
| Whisper STT | 0.5-2s | Depends on audio length |
| Embedding generation | 10-15ms | E5-small-v2 |
| Memory retrieval | 20-50ms | Local storage |
| LLM first token | 200-800ms | Model/provider dependent |
| TTS per sentence | 200-800ms | IndexTTS2 |
| **Total perceived** | **1-3s** | First audio playback |

### 15.2 Memory Usage

| Component | VRAM | RAM |
|-----------|------|-----|
| IndexTTS2 | 4-6GB | 2GB |
| Whisper (base) | 0 (CPU) | 1GB |
| E5-small-v2 | 0 (CPU) | 500MB |
| Gradio UI | 0 | 200MB |
| **Total** | **4-6GB** | **~4GB** |

### 15.3 Storage Requirements

| Data Type | Growth Rate | Cleanup |
|-----------|-------------|---------|
| Memories | ~1KB/interaction | Manual/decay |
| Summaries | ~500B/10 interactions | None |
| Audio cache | ~50KB/phrase | LRU eviction |
| Conversation logs | ~5KB/session | Manual |

---

## 16. Configuration Reference

### 16.1 Environment Variables

```bash
# config.env
OPENROUTER_API_KEY=sk-or-v1-...     # Required for cloud LLMs
SUPABASE_URL=https://...            # Optional: Cloud memory storage
SUPABASE_KEY=eyJ...                 # Optional: Cloud memory storage
```

### 16.2 Memory Manager Configuration

```python
# memory_manager.py
SUMMARY_INTERVAL = 10       # Summarize every N interactions
MAX_CONTEXT_TOKENS = 16000  # Target context size
DECAY_RATE = 0.95           # Daily decay multiplier

# Weighted retrieval weights
RECENCY_WEIGHT = 0.2
RELEVANCE_WEIGHT = 0.5
IMPORTANCE_WEIGHT = 0.3
RECENCY_DECAY_BASE = 0.995  # Per-hour decay
```

### 16.3 VAD Configuration

```python
# audio/vad_recorder.py
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 30
DEFAULT_SILENCE_THRESHOLD = 0.8
DEFAULT_MIN_SPEECH_DURATION = 0.3
DEFAULT_MAX_RECORDING_DURATION = 60.0
```

### 16.4 Streaming Configuration

```python
# streaming.py
@dataclass
class StreamingConfig:
    min_sentence_length: int = 10
    max_sentence_wait: float = 2.0
    enable_phrase_cache: bool = True
    max_cache_size_mb: int = 100
    gap_between_sentences_ms: int = 100
    prefetch_sentences: int = 2
```

---

## 17. API Reference

### 17.1 Memory Manager API

```python
from memory.memory_manager import create_memory_manager

# Initialize
manager = create_memory_manager(use_local=True)

# Activate character (loads state, applies decay)
state = manager.activate_character("hermione")

# Add interaction (stores memory, updates state)
manager.add_interaction(
    character_id="hermione",
    user_message="Hello Hermione!",
    assistant_response="Oh, hello! *adjusts books* How can I help you?"
)

# Weighted search with MMR
results = manager.weighted_search(
    character_id="hermione",
    query="What do you remember about our first meeting?",
    limit=5,
    memory_type='episodic',
    use_mmr=True,
    mmr_lambda=0.7
)
# Returns: List[(Memory, weighted_score, relevance_score)]

# Build context for LLM
context = manager.build_context(
    character_id="hermione",
    current_query="Tell me about your work at the Ministry"
)

# Format context as string
context_str = manager.format_context_for_prompt(context)

# Export all memories
data = manager.storage.export_all_memories("hermione")

# Generate visualization
graph_path = manager.generate_memory_graph("hermione")
```

### 17.2 Character Manager API

```python
from memory.characters import create_character_manager

manager = create_character_manager()

# Get character
char = manager.get_character("hermione")

# List all
ids = manager.list_characters()  # ['hermione', 'assistant', 'lisbeth']

# Add custom character
manager.add_character(Character(
    id="custom",
    name="Custom Character",
    display_name="🎭 Custom Character",
    system_prompt="You are a helpful assistant..."
))

# Export/Import
data = manager.export_character("custom", include_state=True)
success, msg = manager.import_character(data)
```

### 17.3 Tool Registry API

```python
from tools import REGISTRY, init_tools

# Initialize with configuration
init_tools(
    sandbox_path="./sessions/files",
    enable_full_file_access=False,
    enable_code_execution=False
)

# Get schemas for specific tools
schemas = REGISTRY.list_tools(["web_search", "calculate", "get_current_time"])

# Execute tool call
result = REGISTRY.execute({
    "function": {
        "name": "web_search",
        "arguments": '{"query": "latest AI news"}'
    }
})

# Get usage stats
stats = REGISTRY.get_stats()
```

---

## 18. Future Research Directions

This section outlines potential improvements for researchers or AI systems analyzing this codebase:

### 18.1 Memory System Enhancements

1. **Adaptive Weight Learning**
   - Current: Fixed weights (0.2, 0.5, 0.3)
   - Improvement: Learn optimal weights per-character or per-query-type
   - Approach: Reinforcement learning from user feedback signals

2. **Hierarchical Memory Networks**
   - Current: Flat three-layer architecture
   - Improvement: Graph-structured memory with explicit relationships
   - Reference: Memory Networks (Weston et al., 2014)

3. **Episodic Memory Compression**
   - Current: Summarization every N interactions
   - Improvement: Continuous online compression with attention-based pooling
   - Reference: Compressive Transformers (Rae et al., 2019)

4. **Cross-Character Memory Transfer**
   - Current: Complete isolation
   - Improvement: Selective knowledge transfer (user facts, not personality)
   - Challenge: Maintaining character consistency

### 18.2 Retrieval Improvements

1. **Hybrid Retrieval**
   - Current: Dense retrieval only (embeddings)
   - Improvement: Combine with BM25 sparse retrieval
   - Benefit: Better keyword matching for specific terms

2. **Query Expansion**
   - Current: Single query embedding
   - Improvement: Generate multiple query variations
   - Reference: HyDE (Gao et al., 2022)

3. **Contextual Re-ranking**
   - Current: MMR for diversity
   - Improvement: LLM-based re-ranking of top candidates
   - Reference: RankGPT, Cohere Rerank

### 18.3 Voice System Improvements

1. **Emotion-Conditioned TTS**
   - Current: Neutral voice cloning
   - Improvement: Detect emotion in text, condition TTS accordingly
   - Reference: EmotiVoice, StyleTTS2

2. **Speaker Diarization**
   - Current: Single speaker assumption
   - Improvement: Handle multi-speaker audio input
   - Benefit: Support for group conversations

3. **Real-Time Voice Conversion**
   - Current: Generate from text
   - Improvement: Voice-to-voice conversion for lower latency
   - Reference: So-VITS-SVC, RVC

### 18.4 Scalability Enhancements

1. **Vector Database Migration**
   - Current: In-memory numpy similarity
   - Improvement: Dedicated vector DB (Qdrant, Pinecone, Weaviate)
   - Benefit: Millions of memories, faster retrieval

2. **Embedding Caching**
   - Current: Generate on demand
   - Improvement: Pre-compute and cache embeddings
   - Benefit: Faster memory storage

3. **Async Processing Pipeline**
   - Current: Sequential processing
   - Improvement: Parallel STT, memory retrieval, TTS
   - Benefit: Lower overall latency

---

## Appendix A: Mathematical Notation Summary

| Symbol | Meaning |
|--------|---------|
| R | Recency score ∈ [0, 1] |
| V | Relevance score (cosine similarity) ∈ [0, 1] |
| I | Importance score ∈ [0, 1] |
| λ | MMR trade-off parameter ∈ [0, 1] |
| t | Time since last access (hours) |
| q | Query embedding vector ∈ ℝ^384 |
| m | Memory embedding vector ∈ ℝ^384 |

---

## Appendix B: File Structure Reference

```
voice_chat/
├── voice_chat_app.py           # Main application (Entry point)
├── mcp_client.py               # MCP Client Implementation
├── mcp_manager_ui.py           # UI for MCP management
├── group_manager.py            # Group chat logic
├── streaming.py                # Streaming TTS module
├── utils.py                    # Shared utilities
│
├── memory/
│   ├── __init__.py
│   ├── memory_manager.py       # Core memory system (Multi-layer)
│   ├── graph_rag.py            # GraphRAG processor
│   ├── embeddings_onnx.py      # ONNX embedding generator
│   ├── characters.py           # Character definitions
│   └── sqlite_storage.py       # SQLite Graph backend
│
├── audio/
│   ├── vad_recorder.py         # VAD module
│   ├── emotion_detector.py     # SER module
│   ├── tts_kokoro.py           # Kokoro TTS backend
│   ├── ptt_windows.py          # Windows PTT
│   └── ptt_linux.py            # Linux PTT
│
├── tools/
│   ├── __init__.py             # Tool registry
│   └── mcp_server_local.py     # Built-in filesystem MCP server
│
├── docs/
│   ├── archive/                # Archived reports (PHASE1_REPORT.md)
│   ├── ARCHITECTURE.md         # Visual architecture diagrams
│   ├── TECHNICAL_REFERENCE.md  # Deep technical details
│   ├── INSTALL.md              # Installation guide
│   ├── USER_MANUAL.md          # User manual
│   ├── MIGRATION.md            # PC Migration guide
│   └── MCP_SERVERS.md          # MCP guide
│
├── scripts/
│   ├── ptt_hidden.vbs          # Helper for background PTT
│   └── vad_hidden.vbs          # Helper for background VAD
│
├── sessions/                   # Runtime data (Databases, conversations)
├── characters/                 # YAML character definitions
├── voice_reference/            # Voice samples
├── recordings/                 # Audio recordings
├── VoiceChat.bat               # Windows launcher
└── voicechat.sh                # Linux launcher
```

---

## Appendix C: Changelog

### v2.2.7 (January 2026)
- **LM Studio Vision & Tools Fix (Pending Verification):**
  - Implemented recursive image wrapping fix using `copy.deepcopy`.
  - Enabled native tool calling support for LM Studio (v0.2.x+).
  - Prevented base64 image blobs from bloating session history files.
  - Unified safe payload construction across OpenRouter and LM Studio.

### v2.2.6 (January 2026)
- **LM Studio Hybrid Connectivity (FINAL):**
  - Implemented reliable Windows Host IP detection via WSL Gateway (`ip route show default`).
  - Removed hardcoded IP requirement in `config.env`.
  - Verified connection on Port 1235.

### v2.2.4 (January 2026)
- **Documentation Overhaul:**
  - Updated README and INSTALL for Hybrid Architecture.
  - Refined SKILL.md and CLAUDE.md for agentic clarity.
  - Consolidated redundant docs.

### v2.2.3 (January 2026)
- **Port Conflict Fix:**
  - Moved LM Studio default port to 1235 (avoiding Windows `svchost` conflict).
  - Added firewall troubleshooting guides.

### v2.2.2 (December 2025)
- **Cyberpunk UI Launcher:**
  - Redesigned `VoiceChat.bat` with ANSI colors and ASCII art.
  - Improved startup diagnostics and cleanup.

### v2.2.1 (December 2025)
- **Progressive Disclosure:**
  - 90% context reduction for casual chat.
  - Intelligent tool schema loading (triggers only on demand).

### v2.2.0 (December 2025)
- **Knowledge Graph Memory (Layer 4)**
  - SQLite-based entity and relationship storage
  - Background graph extraction via LLM
  - Interactive visualization with networkx/pyvis
- **MCP Integration**
  - Model Context Protocol client implementation
  - Stdio server transport
  - Tool schema caching and routing
  - Built-in filesystem server

### v2.1.0 (December 2025)
- SQLite v2 schema with graph tables
- Graph extractor background processing

### v2.0.0 (December 2025)
- Knowledge Graph foundation
- Four-layer memory architecture

### v1.2.0 (December 2025)
- Docker containerization support
- TTS performance optimization (removed blocking empty_cache)
- Audio stability fixes (silence padding, player persistence)
- Improved text cleaning for TTS
- Voice refresh functionality

### v1.1.0 (December 2025)
- Enhanced Character Manager UI
- Voice upload with auto-emoji
- Cross-platform improvements

### v1.0.0 (December 2025)
- Initial release
- Three-layer memory architecture
- E5-small-v2 embeddings
- Weighted retrieval with MMR
- Streaming TTS
- Multi-turn tools
- Group chat

---

<div align="center">

**IndexTTS2 Voice Agent**

*Built with ❤️ for AI companions that remember*

</div>
