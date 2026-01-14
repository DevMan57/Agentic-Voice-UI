import sys
import os
import sqlite3
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from memory.embeddings_onnx import ONNXEmbeddingManager

DB_PATH = "sessions/memory.db"

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    print("Initializing embedding model...")
    # This will load the new 1024-dim model
    embedder = ONNXEmbeddingManager()
    
    # Trigger load to get dimension
    try:
        embedder.embed("warmup")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    new_dim = embedder.dimension
    print(f"Target dimension: {new_dim}")
    
    print(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all memories with embeddings
    cursor.execute("SELECT id, content, embedding FROM memories WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    
    print(f"Found {len(rows)} memories. Checking dimensions...")
    
    updated_count = 0
    
    for row in rows:
        mem_id = row['id']
        content = row['content']
        blob = row['embedding']
        
        vec = np.frombuffer(blob, dtype=np.float32)
        
        if vec.shape[0] != new_dim:
            print(f"Fixing memory {mem_id[:8]}... (Old: {vec.shape[0]}, New: {new_dim}, Len: {len(content)})")
            
            try:
                # Truncate to avoid OOM on huge memories (e.g. 60k chars)
                safe_content = content[:8000]
                if len(content) > 8000:
                    print(f"  Truncating {len(content)} -> 8000 chars")

                # Re-embed
                # is_query=False for stored memories (passage)
                new_embedding = embedder.embed(safe_content, is_query=False)
                new_blob = np.array(new_embedding, dtype=np.float32).tobytes()
                
                conn.execute("UPDATE memories SET embedding = ? WHERE id = ?", (new_blob, mem_id))
                updated_count += 1
                conn.commit()
                
            except Exception as e:
                print(f"Failed to update memory {mem_id}: {e}")
                
    conn.commit()
    conn.close()
    print(f"Migration complete. Updated {updated_count} memories.")

if __name__ == "__main__":
    migrate()
