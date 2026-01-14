"""
SQLite Storage Backend for IndexTTS2 Voice Chat Memory System (Graph Edition)

Version 2.3 - Added sqlite-vec for O(log n) vector search.
High-performance local storage with:
- Full-text search (FTS5)
- Vector similarity (sqlite-vec native KNN, fallback to numpy cosine)
- Knowledge Graph (Nodes & Edges) for "Deep Thought" visualization
"""

import sqlite3
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import asdict
import hashlib
import numpy as np

# Try to import sqlite-vec for native vector search
try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    print("[SQLite] Warning: sqlite-vec not installed. Using Python cosine sim (O(n)).")
    print("[SQLite] Install with: pip install sqlite-vec")

# Import data classes from main memory manager
from .memory_manager import Memory, EpisodicSummary, CharacterState

class SQLiteStorage:
    VERSION = "2.3"
    EMBEDDING_DIM = 1024  # Qwen3 embedding dimension
    MAX_SEARCH_ROWS = 5000  # Fallback: Hard cap on rows fetched for vector search

    def __init__(self, db_path: str = None):
        """Initialize SQLite storage"""
        if db_path:
            self.db_path = Path(db_path)
        else:
            root_dir = Path(__file__).parent.parent
            self.db_path = root_dir / 'sessions' / 'memory.db'
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_database()
        
        # Run migration for v2 if needed
        self._migrate_v2()
        
        print(f"[SQLite] Database: {self.db_path}")
    
    @property
    def conn(self) -> sqlite3.Connection:
        """Get thread-local database connection with sqlite-vec loaded"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            
            # Load sqlite-vec extension if available
            if SQLITE_VEC_AVAILABLE:
                try:
                    self._local.conn.enable_load_extension(True)
                    sqlite_vec.load(self._local.conn)
                    self._local.conn.enable_load_extension(False)
                except Exception as e:
                    print(f"[SQLite] Warning: Failed to load sqlite-vec: {e}")
        return self._local.conn
    
    def _init_database(self):
        """Initialize database schema"""
        conn = self.conn
        
        # 1. Main memories table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                importance_score REAL DEFAULT 0.5,
                decay_factor REAL DEFAULT 1.0,
                tags TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                superseded_by TEXT,
                is_active INTEGER DEFAULT 1
            )
        """)
        
        # Indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_character ON memories(character_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
        
        # 2. FTS5 Virtual Table (Full Text Search)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id,
                character_id,
                content,
                content='memories',
                content_rowid='rowid'
            )
        """)
        
        # 3. Graph: Entities Table (Nodes)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                name TEXT NOT NULL,
                entity_type TEXT,
                description TEXT,
                meta_data TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(character_id, name)
            )
        """)
        
        # 4. Graph: Relationships Table (Edges)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id TEXT NOT NULL,
                source_entity TEXT NOT NULL,
                target_entity TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(character_id, source_entity, target_entity, relation_type)
            )
        """)

        # 5. Summaries & State
        conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                summary_short TEXT NOT NULL,
                summary_medium TEXT NOT NULL,
                summary_long TEXT NOT NULL,
                key_entities TEXT DEFAULT '[]',
                emotional_arc TEXT DEFAULT '',
                plot_points TEXT DEFAULT '[]',
                relationship_delta TEXT DEFAULT '{}',
                interaction_count INTEGER DEFAULT 0,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                embedding BLOB
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS character_states (
                character_id TEXT PRIMARY KEY,
                current_mood TEXT DEFAULT 'neutral',
                emotional_state TEXT DEFAULT '{}',
                relationship_with_user TEXT DEFAULT '{}',
                interaction_count INTEGER DEFAULT 0,
                topics_discussed TEXT DEFAULT '{}',
                last_interaction TEXT,
                updated_at TEXT NOT NULL
            )
        """)

        # 6. Conversations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                title TEXT,
                preview TEXT,
                history TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                is_archived INTEGER DEFAULT 0
            )
        """)

        # 7. Graph Communities (for GraphRAG global queries)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_communities (
                id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                level INTEGER DEFAULT 0,
                entity_ids TEXT DEFAULT '[]',
                summary TEXT NOT NULL,
                keywords TEXT DEFAULT '[]',
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_communities_character ON graph_communities(character_id)")

        # 8. Vector search table (sqlite-vec) - only if extension available
        if SQLITE_VEC_AVAILABLE:
            try:
                conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                        embedding float[{self.EMBEDDING_DIM}]
                    )
                """)
                print(f"[SQLite] vec_memories table ready (dim={self.EMBEDDING_DIM})")
            except Exception as e:
                print(f"[SQLite] Could not create vec_memories: {e}")

        conn.commit()

    def _migrate_v2(self):
        """Ensure v2 tables exist for users migrating from v1"""
        try:
            self.conn.execute("SELECT 1 FROM entities LIMIT 1")
        except sqlite3.OperationalError:
            print("[SQLite] Migrating to v2.0 (Graph Support)...")
            self._init_database()  # Re-running init will create missing tables

        # v2.2 migration: Add version column for CRUD tracking
        self._migrate_v2_2()

    def _migrate_v2_2(self):
        """Add version tracking columns for memory CRUD operations"""
        try:
            # Check if version column exists
            cursor = self.conn.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'version' not in columns:
                print("[SQLite] Migrating to v2.2 (Memory CRUD Support)...")
                self.conn.execute("ALTER TABLE memories ADD COLUMN version INTEGER DEFAULT 1")
                self.conn.commit()
                print("[SQLite] Added 'version' column to memories table")

        except Exception as e:
            print(f"[SQLite] v2.2 migration note: {e}")

        # v2.3 migration: Populate vec_memories from existing embeddings
        self._migrate_v2_3_vec()

    def _migrate_v2_3_vec(self):
        """Migrate existing embeddings to vec_memories table"""
        if not SQLITE_VEC_AVAILABLE:
            return
        
        try:
            # Check if vec_memories has any rows
            count = self.conn.execute("SELECT COUNT(*) FROM vec_memories").fetchone()[0]
            if count > 0:
                return  # Already migrated
            
            # Get count of memories with embeddings
            mem_count = self.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL AND is_active = 1"
            ).fetchone()[0]
            
            if mem_count == 0:
                return  # Nothing to migrate
            
            print(f"[SQLite] Migrating {mem_count} embeddings to vec_memories...")
            
            # Migrate in batches
            cursor = self.conn.execute(
                "SELECT rowid, embedding FROM memories WHERE embedding IS NOT NULL AND is_active = 1"
            )
            
            migrated = 0
            for row in cursor:
                try:
                    embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                    if embedding.shape[0] == self.EMBEDDING_DIM:
                        # Insert as JSON array for sqlite-vec
                        embedding_json = json.dumps(embedding.tolist())
                        self.conn.execute(
                            "INSERT OR IGNORE INTO vec_memories(rowid, embedding) VALUES (?, ?)",
                            (row['rowid'], embedding_json)
                        )
                        migrated += 1
                except Exception as e:
                    continue
            
            self.conn.commit()
            print(f"[SQLite] Migrated {migrated} embeddings to vec_memories")
            
        except Exception as e:
            print(f"[SQLite] v2.3 vec migration note: {e}")

    # ==================== Memory Operations ====================
    
    def store_memory(self, memory: Memory) -> bool:
        try:
            embedding_blob = None
            embedding_json = None
            if memory.embedding:
                embedding_blob = np.array(memory.embedding, dtype=np.float32).tobytes()
                embedding_json = json.dumps(memory.embedding)  # For sqlite-vec
            
            # Insert into main memories table
            self.conn.execute("""
                INSERT OR REPLACE INTO memories 
                (id, character_id, memory_type, content, embedding, importance_score,
                 decay_factor, tags, created_at, updated_at, last_accessed, access_count, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                memory.id, memory.character_id, memory.memory_type, memory.content, embedding_blob,
                memory.importance_score, memory.decay_factor,
                json.dumps(memory.tags) if isinstance(memory.tags, dict) else memory.tags,
                memory.created_at.isoformat() if isinstance(memory.created_at, datetime) else memory.created_at,
                memory.updated_at.isoformat() if isinstance(memory.updated_at, datetime) else memory.updated_at,
                memory.last_accessed.isoformat() if isinstance(memory.last_accessed, datetime) else memory.last_accessed,
                memory.access_count
            ))
            
            # Also insert into vec_memories if available
            if SQLITE_VEC_AVAILABLE and embedding_json:
                try:
                    # Get the rowid of the just-inserted memory
                    rowid = self.conn.execute(
                        "SELECT rowid FROM memories WHERE id = ?", (memory.id,)
                    ).fetchone()[0]
                    
                    self.conn.execute(
                        "INSERT OR REPLACE INTO vec_memories(rowid, embedding) VALUES (?, ?)",
                        (rowid, embedding_json)
                    )
                except Exception as e:
                    # Non-fatal: fall back to Python cosine
                    pass
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[SQLite] Error storing memory: {e}")
            return False
            
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID"""
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row:
            return self._row_to_memory(row)
        return None

    def get_memories_by_character(self, character_id: str, memory_type: str = None, limit: int = 50) -> List[Memory]:
        query = "SELECT * FROM memories WHERE character_id = ? AND is_active = 1"
        params = [character_id]
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.execute(query, params)
        return [self._row_to_memory(row) for row in cursor]
    
    def update_memory_access(self, memory_id: str):
        """Update last_accessed timestamp and access_count for a memory"""
        try:
            with self.conn:
                self.conn.execute("""
                    UPDATE memories 
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE id = ?
                """, (datetime.utcnow().isoformat(), memory_id))
        except Exception as e:
            print(f"[SQLite] Error updating memory access: {e}")
    
    def apply_decay(self, character_id: str, decay_rate: float):
        """
        Apply time-based decay to episodic memory importance scores.
        """
        try:
            with self.conn:
                self.conn.execute("""
                    UPDATE memories 
                    SET importance_score = MAX(0.1, importance_score * (1.0 - ?))
                    WHERE character_id = ? 
                    AND memory_type = 'episodic'
                    AND importance_score > 0.1
                """, (decay_rate, character_id))
        except Exception as e:
            print(f"[SQLite] Error applying decay: {e}")

    def delete_memory(self, memory_id: str, hard: bool = False) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: The memory ID to delete
            hard: If True, permanently removes the row. If False (default), soft-delete via is_active=0
        """
        try:
            with self.conn:
                if hard:
                    self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                else:
                    self.conn.execute("UPDATE memories SET is_active = 0 WHERE id = ?", (memory_id,))
            return True
        except Exception as e:
            print(f"[SQLite] Error deleting memory: {e}")
            return False

    def update_memory(
        self,
        memory_id: str,
        content: str = None,
        embedding: list = None,
        importance_score: float = None,
        tags: dict = None,
        superseded_by: str = None
    ) -> bool:
        """
        Update a memory's content and/or metadata.

        Args:
            memory_id: The memory ID to update
            content: New content (if provided)
            embedding: New embedding vector (if provided)
            importance_score: New importance score (if provided)
            tags: New tags dict (if provided)
            superseded_by: ID of the memory that replaces this one (for version tracking)

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            # Build dynamic UPDATE query
            updates = []
            params = []

            if content is not None:
                updates.append("content = ?")
                params.append(content)

            if embedding is not None:
                updates.append("embedding = ?")
                params.append(np.array(embedding, dtype=np.float32).tobytes())

            if importance_score is not None:
                updates.append("importance_score = ?")
                params.append(importance_score)

            if tags is not None:
                updates.append("tags = ?")
                params.append(json.dumps(tags))

            if superseded_by is not None:
                updates.append("superseded_by = ?")
                params.append(superseded_by)

            if not updates:
                return True  # Nothing to update

            # Increment version if content changed
            if content is not None:
                updates.append("version = version + 1")

            # Always update the updated_at timestamp
            updates.append("updated_at = ?")
            params.append(datetime.utcnow().isoformat())

            params.append(memory_id)

            query = f"UPDATE memories SET {', '.join(updates)} WHERE id = ?"

            with self.conn:
                cursor = self.conn.execute(query, params)
                if cursor.rowcount == 0:
                    print(f"[SQLite] No memory found with id: {memory_id}")
                    return False

            return True
        except Exception as e:
            print(f"[SQLite] Error updating memory: {e}")
            return False

    def find_similar_memories(
        self,
        character_id: str,
        query_embedding: list,
        memory_type: str = 'semantic',
        threshold: float = 0.85,
        limit: int = 5
    ) -> List[Tuple['Memory', float]]:
        """
        Find memories highly similar to a query embedding.
        Used for contradiction detection - finds facts that might conflict.

        Args:
            character_id: Character to search within
            query_embedding: Embedding vector of the new fact
            memory_type: Type of memory to search (default: semantic)
            threshold: Minimum similarity score (default: 0.85 for near-duplicates)
            limit: Max results to return

        Returns:
            List of (Memory, similarity_score) tuples
        """
        return self.semantic_search(
            character_id=character_id,
            query_embedding=query_embedding,
            limit=limit,
            memory_type=memory_type,
            min_similarity=threshold
        )

    def semantic_search(self, character_id: str, query_embedding: List[float], limit: int = 10,
                        memory_type: str = None, min_similarity: float = 0.5) -> List[Tuple[Memory, float]]:
        """
        Vector search for similar memories.
        
        Uses sqlite-vec native KNN if available (O(log n)), 
        otherwise falls back to Python cosine similarity (O(n)).
        """
        # Try sqlite-vec native search first
        if SQLITE_VEC_AVAILABLE:
            try:
                return self._semantic_search_vec(character_id, query_embedding, limit, memory_type, min_similarity)
            except Exception as e:
                print(f"[SQLite] vec search failed, falling back to numpy: {e}")
        
        # Fallback to Python cosine similarity
        return self._semantic_search_numpy(character_id, query_embedding, limit, memory_type, min_similarity)
    
    def _semantic_search_vec(self, character_id: str, query_embedding: List[float], limit: int = 10,
                             memory_type: str = None, min_similarity: float = 0.5) -> List[Tuple[Memory, float]]:
        """Native sqlite-vec KNN search (O(log n))"""
        query_json = json.dumps(query_embedding)
        
        # Build query with character_id and memory_type filters
        # Note: vec0 KNN requires a subquery pattern for filtering
        if memory_type:
            sql = """
                SELECT m.*, vec_distance_cosine(v.embedding, ?) as distance
                FROM vec_memories v
                JOIN memories m ON m.rowid = v.rowid
                WHERE m.character_id = ?
                  AND m.memory_type = ?
                  AND m.is_active = 1
                ORDER BY distance ASC
                LIMIT ?
            """
            params = (query_json, character_id, memory_type, limit * 2)  # Fetch extra for min_similarity filter
        else:
            sql = """
                SELECT m.*, vec_distance_cosine(v.embedding, ?) as distance
                FROM vec_memories v
                JOIN memories m ON m.rowid = v.rowid
                WHERE m.character_id = ?
                  AND m.is_active = 1
                ORDER BY distance ASC
                LIMIT ?
            """
            params = (query_json, character_id, limit * 2)
        
        cursor = self.conn.execute(sql, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            # Convert cosine distance to similarity (distance = 1 - similarity for normalized vectors)
            distance = row['distance']
            similarity = 1.0 - distance
            
            if similarity >= min_similarity:
                results.append((self._row_to_memory(row), similarity))
                if len(results) >= limit:
                    break
        
        if results:
            print(f"[SQLite] vec search found {len(results)} memories (sqlite-vec)")
        
        return results
    
    def _semantic_search_numpy(self, character_id: str, query_embedding: List[float], limit: int = 10,
                               memory_type: str = None, min_similarity: float = 0.5) -> List[Tuple[Memory, float]]:
        """Fallback: Python cosine similarity search (O(n))"""
        query = "SELECT * FROM memories WHERE character_id = ? AND embedding IS NOT NULL AND is_active = 1"
        params = [character_id]
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        # Order by recency and cap at MAX_SEARCH_ROWS to prevent OOM on large datasets
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(self.MAX_SEARCH_ROWS)

        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()
        if not rows: return []

        # Bulk compute
        mem_vecs = []
        mem_rows = []
        expected_dim = len(query_embedding)

        for row in rows:
            vec = np.frombuffer(row['embedding'], dtype=np.float32)
            if vec.shape[0] == expected_dim:
                mem_vecs.append(vec)
                mem_rows.append(row)
            else:
                print(f"[Memory] Warning: Skipping memory {row['id'][:8]} due to dimension mismatch (Expected {expected_dim}, got {vec.shape[0]})")
            
        if not mem_vecs:
            return []
            
        matrix = np.array(mem_vecs)
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        # Cosine Sim: (A . B) / (|A|*|B|)
        norm_matrix = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / (norm_matrix + 1e-9)
        norm_query = np.linalg.norm(query_vec)
        query_vec = query_vec / (norm_query + 1e-9)
        
        scores = np.dot(matrix, query_vec)
        
        # Top K
        top_indices = np.argsort(scores)[::-1][:limit]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= min_similarity:
                results.append((self._row_to_memory(mem_rows[idx]), score))
        
        if results:
            print(f"[SQLite] vec search found {len(results)} memories (numpy fallback)")
                
        return results

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        embedding = None
        if row['embedding']:
            embedding = np.frombuffer(row['embedding'], dtype=np.float32).tolist()
        return Memory(
            id=row['id'], character_id=row['character_id'], memory_type=row['memory_type'],
            content=row['content'], embedding=embedding, importance_score=row['importance_score'],
            decay_factor=row['decay_factor'], tags=json.loads(row['tags']) if row['tags'] else {},
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            last_accessed=datetime.fromisoformat(row['last_accessed']),
            access_count=row['access_count']
        )

    # ==================== Graph Operations (New for v2.0) ====================

    def add_graph_node(self, character_id: str, name: str, node_type: str, description: str = ""):
        """Create or update a node in the knowledge graph"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO entities (id, character_id, name, entity_type, description)
                VALUES (?, ?, ?, ?, ?)
            """, (
                f"{character_id}_{name.lower().replace(' ', '_')}", 
                character_id, name, node_type, description
            ))
            self.conn.commit()
        except Exception as e:
            print(f"[Graph] Error adding node: {e}")

    def add_graph_edge(self, character_id: str, source: str, target: str, relation: str):
        """Create a relationship edge"""
        try:
            self.conn.execute("""
                INSERT OR IGNORE INTO relationships (character_id, source_entity, target_entity, relation_type)
                VALUES (?, ?, ?, ?)
            """, (character_id, source, target, relation))
            self.conn.commit()
        except Exception as e:
            print(f"[Graph] Error adding edge: {e}")

    def get_knowledge_graph(self, character_id: str, limit: int = 50) -> Dict[str, Any]:
        """Retrieve graph data for visualization"""
        nodes = self.conn.execute("SELECT * FROM entities WHERE character_id = ? LIMIT ?", (character_id, limit)).fetchall()
        edges = self.conn.execute("SELECT * FROM relationships WHERE character_id = ? LIMIT ?", (character_id, limit)).fetchall()

        return {
            "nodes": [{"id": n['name'], "label": n['name'], "group": n['entity_type']} for n in nodes],
            "edges": [{"from": e['source_entity'], "to": e['target_entity'], "label": e['relation_type']} for e in edges]
        }

    def get_all_entities(self, character_id: str) -> List[Dict[str, Any]]:
        """Get all entities for a character (for community detection)"""
        rows = self.conn.execute(
            "SELECT * FROM entities WHERE character_id = ?",
            (character_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def get_all_relationships(self, character_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a character (for community detection)"""
        rows = self.conn.execute(
            "SELECT * FROM relationships WHERE character_id = ?",
            (character_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    # ==================== Graph Community Operations (GraphRAG) ====================

    def store_community(
        self,
        community_id: str,
        character_id: str,
        level: int,
        entity_ids: List[str],
        summary: str,
        keywords: List[str],
        embedding: List[float] = None
    ) -> bool:
        """Store a graph community with its summary"""
        try:
            embedding_blob = None
            if embedding:
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

            self.conn.execute("""
                INSERT OR REPLACE INTO graph_communities
                (id, character_id, level, entity_ids, summary, keywords, embedding, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                community_id,
                character_id,
                level,
                json.dumps(entity_ids),
                summary,
                json.dumps(keywords),
                embedding_blob,
                datetime.utcnow().isoformat()
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[SQLite] Error storing community: {e}")
            return False

    def get_communities(self, character_id: str, level: int = None) -> List[Dict[str, Any]]:
        """Get communities for a character, optionally filtered by level"""
        if level is not None:
            rows = self.conn.execute(
                "SELECT * FROM graph_communities WHERE character_id = ? AND level = ? ORDER BY created_at DESC",
                (character_id, level)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM graph_communities WHERE character_id = ? ORDER BY level, created_at DESC",
                (character_id,)
            ).fetchall()

        communities = []
        for row in rows:
            embedding = None
            if row['embedding']:
                embedding = np.frombuffer(row['embedding'], dtype=np.float32).tolist()
            communities.append({
                'id': row['id'],
                'character_id': row['character_id'],
                'level': row['level'],
                'entity_ids': json.loads(row['entity_ids']),
                'summary': row['summary'],
                'keywords': json.loads(row['keywords']),
                'embedding': embedding,
                'created_at': row['created_at'],
                'updated_at': row['updated_at']
            })
        return communities

    def search_communities(
        self,
        character_id: str,
        query_embedding: List[float],
        limit: int = 5,
        min_similarity: float = 0.3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search communities by embedding similarity"""
        communities = self.get_communities(character_id)
        if not communities:
            return []

        # Filter communities with embeddings
        with_embeddings = [c for c in communities if c['embedding']]
        if not with_embeddings:
            return []

        # Compute similarities
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-9:
            return []
        query_vec = query_vec / query_norm

        results = []
        for community in with_embeddings:
            comm_vec = np.array(community['embedding'], dtype=np.float32)
            comm_norm = np.linalg.norm(comm_vec)
            if comm_norm < 1e-9:
                continue
            comm_vec = comm_vec / comm_norm

            similarity = float(np.dot(query_vec, comm_vec))
            if similarity >= min_similarity:
                results.append((community, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def clear_communities(self, character_id: str) -> bool:
        """Clear all communities for a character (for regeneration)"""
        try:
            self.conn.execute(
                "DELETE FROM graph_communities WHERE character_id = ?",
                (character_id,)
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[SQLite] Error clearing communities: {e}")
            return False

    # ==================== State, Summary, Conversation (Standard) ====================

    def save_character_state(self, state: CharacterState) -> bool:
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO character_states
                (character_id, current_mood, emotional_state, relationship_with_user,
                 interaction_count, last_interaction, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                state.character_id, state.current_mood, json.dumps(state.emotional_state),
                json.dumps(state.relationship_with_user), state.interaction_count,
                datetime.utcnow().isoformat(), datetime.utcnow().isoformat()
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[SQLite] Error saving state: {e}")
            return False

    def load_character_state(self, character_id: str) -> Optional[CharacterState]:
        row = self.conn.execute("SELECT * FROM character_states WHERE character_id = ?", (character_id,)).fetchone()
        if row:
            return CharacterState(
                character_id=row['character_id'], current_mood=row['current_mood'],
                emotional_state=json.loads(row['emotional_state']),
                relationship_with_user=json.loads(row['relationship_with_user']),
                interaction_count=row['interaction_count']
            )
        return None

    def store_summary(self, summary: EpisodicSummary) -> bool:
        try:
            emb_blob = np.array(summary.embedding, dtype=np.float32).tobytes() if summary.embedding else None
            self.conn.execute("""
                INSERT OR REPLACE INTO summaries
                (id, character_id, summary_short, summary_medium, summary_long,
                 key_entities, emotional_arc, plot_points, relationship_delta,
                 interaction_count, start_time, end_time, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.id, summary.character_id, summary.summary_short, summary.summary_medium,
                summary.summary_long, json.dumps(summary.key_entities), summary.emotional_arc,
                json.dumps(summary.plot_points), json.dumps(summary.relationship_delta),
                summary.interaction_count, 
                summary.start_time.isoformat() if isinstance(summary.start_time, datetime) else summary.start_time,
                summary.end_time.isoformat() if isinstance(summary.end_time, datetime) else summary.end_time,
                emb_blob
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[SQLite] Error storing summary: {e}")
            return False

    def get_recent_summaries(self, character_id: str, limit: int = 5) -> List[EpisodicSummary]:
        rows = self.conn.execute("SELECT * FROM summaries WHERE character_id = ? ORDER BY end_time DESC LIMIT ?", (character_id, limit)).fetchall()
        summaries = []
        for row in rows:
            emb = np.frombuffer(row['embedding'], dtype=np.float32).tolist() if row['embedding'] else None
            summaries.append(EpisodicSummary(
                id=row['id'], character_id=row['character_id'], summary_short=row['summary_short'],
                summary_medium=row['summary_medium'], summary_long=row['summary_long'],
                key_entities=json.loads(row['key_entities']), emotional_arc=row['emotional_arc'],
                plot_points=json.loads(row['plot_points']), relationship_delta=json.loads(row['relationship_delta']),
                interaction_count=row['interaction_count'], 
                start_time=datetime.fromisoformat(row['start_time']),
                end_time=datetime.fromisoformat(row['end_time']), embedding=emb
            ))
        return summaries

    def save_conversation(self, conversation_id, character_id, history, title=None):
        try:
            self.conn.execute("INSERT OR REPLACE INTO conversations (id, character_id, history, updated_at, created_at) VALUES (?, ?, ?, ?, ?)",
                             (conversation_id, character_id, json.dumps(history), datetime.utcnow().isoformat(), datetime.utcnow().isoformat()))
            self.conn.commit()
            return True
        except (sqlite3.Error, json.JSONDecodeError) as e:
            print(f"[Memory] Failed to save conversation {conversation_id}: {e}")
            return False

    def list_conversations(self, character_id: str, limit: int = 50):
        rows = self.conn.execute("SELECT * FROM conversations WHERE character_id = ? ORDER BY updated_at DESC LIMIT ?", (character_id, limit)).fetchall()
        return [dict(row) for row in rows]
    
    def load_conversation(self, conversation_id):
        row = self.conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if row: return dict(row)
        return None