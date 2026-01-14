"""
ONNX Embedding Manager for Qwen3-Embedding-0.6B

Uses optimum[onnxruntime] for fast CPU/GPU inference with INT8 quantization.
Significantly faster than sentence-transformers for production use.

Model: Svenni551/Qwen3-Embedding-0.6B-ONNX-INT8
- INT8 quantized (~600MB vs 1.2GB fp16)
- 1024 embedding dimensions
- AVX2 optimized for CPU
"""

import os
import threading
from collections import OrderedDict
from typing import List, Optional, Tuple
import numpy as np

# Lazy imports to avoid startup delay
_ort_model = None
_tokenizer = None
_lock = threading.Lock()

# LRU cache for embeddings (avoids redundant computation)
_embedding_cache: OrderedDict[Tuple[str, bool], List[float]] = OrderedDict()
_cache_lock = threading.Lock()
CACHE_MAX_SIZE = 1000  # Max cached embeddings

# Model configuration
DEFAULT_MODEL_ID = "Svenni551/Qwen3-Embedding-0.6B-ONNX-INT8"
EMBEDDING_DIM = 1024
MAX_LENGTH = 8192  # Qwen3 supports up to 32k, but 8k is practical


def _cache_get(text: str, is_query: bool) -> Optional[List[float]]:
    """Get embedding from cache if exists, updating LRU order."""
    key = (text, is_query)
    with _cache_lock:
        if key in _embedding_cache:
            # Move to end (most recently used)
            _embedding_cache.move_to_end(key)
            return _embedding_cache[key]
    return None


def _cache_set(text: str, is_query: bool, embedding: List[float]) -> None:
    """Store embedding in cache, evicting oldest if full."""
    key = (text, is_query)
    with _cache_lock:
        if key in _embedding_cache:
            _embedding_cache.move_to_end(key)
        else:
            if len(_embedding_cache) >= CACHE_MAX_SIZE:
                # Remove oldest (first item)
                _embedding_cache.popitem(last=False)
            _embedding_cache[key] = embedding


def clear_embedding_cache() -> int:
    """Clear embedding cache. Returns number of entries cleared."""
    with _cache_lock:
        count = len(_embedding_cache)
        _embedding_cache.clear()
        return count


def _ensure_loaded(model_id: str = None):
    """Lazy load model and tokenizer on first use."""
    global _ort_model, _tokenizer

    if _ort_model is not None:
        return

    with _lock:
        if _ort_model is not None:
            return

        model_id = model_id or DEFAULT_MODEL_ID
        print(f"[Embeddings-ONNX] Loading model: {model_id}")

        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Force CPU to avoid competing with TTS for GPU memory
            # INT8 quantized model runs efficiently on CPU
            _ort_model = ORTModelForFeatureExtraction.from_pretrained(
                model_id,
                provider="CPUExecutionProvider"
            )

            print(f"[Embeddings-ONNX] Model loaded successfully (dim={EMBEDDING_DIM}, CPU)")

        except ImportError as e:
            raise ImportError(
                "ONNX embeddings require: pip install optimum[onnxruntime] transformers\n"
                f"Error: {e}"
            )


def _mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Apply mean pooling to token embeddings."""
    # Expand attention mask to match embedding dimensions
    input_mask_expanded = np.expand_dims(attention_mask, -1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape).astype(np.float32)

    # Sum embeddings weighted by attention mask
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)

    return sum_embeddings / sum_mask


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-9, a_max=None)


def _prepare_text(text: str, is_query: bool = False) -> str:
    """
    Prepare text for Qwen3 embedding model.

    Qwen3 uses instruction format for queries:
    - Queries: "Instruct: <task>\nQuery:<text>"
    - Documents: plain text (no prefix)
    """
    if is_query:
        task = "Given a query, retrieve relevant information"
        return f"Instruct: {task}\nQuery:{text}"
    return text


def _add_position_ids(inputs):
    """Add position_ids to inputs if missing (required by some ONNX models)."""
    if 'position_ids' in inputs:
        return inputs

    input_ids = inputs['input_ids']
    batch_size, seq_length = input_ids.shape

    # Create position_ids: [0, 1, 2, ..., seq_len-1] repeated for each batch
    position_ids = np.arange(seq_length, dtype=np.int64).reshape(1, -1)
    inputs['position_ids'] = np.repeat(position_ids, batch_size, axis=0)

    return inputs


class ONNXEmbeddingManager:
    """
    ONNX-based embedding manager for Qwen3-Embedding-0.6B.

    Drop-in replacement for EmbeddingManager with faster inference.
    Uses INT8 quantization for ~2x speedup on CPU.
    """

    def __init__(self, model_id: str = None):
        self.model_id = model_id or DEFAULT_MODEL_ID
        self._loaded = False

    def _load(self):
        """Ensure model is loaded."""
        if not self._loaded:
            _ensure_loaded(self.model_id)
            self._loaded = True

    def embed(self, text: str, is_query: bool = False) -> List[float]:
        """
        Generate embedding for a single text (with LRU caching).

        Args:
            text: Text to embed
            is_query: If True, uses query instruction format (for search queries)
                     If False, uses passage format (for documents/memories)

        Returns:
            List of floats (1024 dimensions)
        """
        # Check cache first
        cached = _cache_get(text, is_query)
        if cached is not None:
            return cached

        self._load()

        prepared = _prepare_text(text, is_query)

        inputs = _tokenizer(
            prepared,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np"
        )

        inputs = _add_position_ids(inputs)
        outputs = _ort_model(**{k: v for k, v in inputs.items()})

        # Get token embeddings from last hidden state
        token_embeddings = outputs.last_hidden_state
        if hasattr(token_embeddings, 'numpy'):
            token_embeddings = token_embeddings.numpy()

        attention_mask = inputs['attention_mask']
        if hasattr(attention_mask, 'numpy'):
            attention_mask = attention_mask.numpy()

        # Mean pooling
        embeddings = _mean_pooling(token_embeddings, attention_mask)

        # L2 normalize
        embeddings = _normalize(embeddings)

        result = embeddings[0].tolist()
        _cache_set(text, is_query, result)
        return result

    def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            is_query: If True, uses query instruction format

        Returns:
            List of embedding vectors (each 1024 dimensions)
        """
        if not texts:
            return []

        self._load()

        prepared = [_prepare_text(t, is_query) for t in texts]

        inputs = _tokenizer(
            prepared,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np"
        )

        inputs = _add_position_ids(inputs)
        outputs = _ort_model(**{k: v for k, v in inputs.items()})

        token_embeddings = outputs.last_hidden_state
        if hasattr(token_embeddings, 'numpy'):
            token_embeddings = token_embeddings.numpy()

        attention_mask = inputs['attention_mask']
        if hasattr(attention_mask, 'numpy'):
            attention_mask = attention_mask.numpy()

        embeddings = _mean_pooling(token_embeddings, attention_mask)
        embeddings = _normalize(embeddings)

        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        """Get embedding dimension (1024 for Qwen3-Embedding-0.6B)."""
        return EMBEDDING_DIM

    @property
    def model_name(self) -> str:
        """Get model name for logging."""
        return self.model_id


# Convenience functions for direct use
def embed(text: str, is_query: bool = False) -> List[float]:
    """Generate embedding for text using default ONNX model (with LRU caching)."""
    # Check cache first
    cached = _cache_get(text, is_query)
    if cached is not None:
        return cached

    _ensure_loaded()

    prepared = _prepare_text(text, is_query)

    inputs = _tokenizer(
        prepared,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np"
    )

    inputs = _add_position_ids(inputs)
    outputs = _ort_model(**{k: v for k, v in inputs.items()})

    token_embeddings = outputs.last_hidden_state
    if hasattr(token_embeddings, 'numpy'):
        token_embeddings = token_embeddings.numpy()

    attention_mask = inputs['attention_mask']
    if hasattr(attention_mask, 'numpy'):
        attention_mask = attention_mask.numpy()

    embeddings = _mean_pooling(token_embeddings, attention_mask)
    embeddings = _normalize(embeddings)

    result = embeddings[0].tolist()
    _cache_set(text, is_query, result)
    return result


def embed_batch(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """Generate embeddings for multiple texts using default ONNX model."""
    if not texts:
        return []

    _ensure_loaded()

    prepared = [_prepare_text(t, is_query) for t in texts]

    inputs = _tokenizer(
        prepared,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np"
    )

    inputs = _add_position_ids(inputs)
    outputs = _ort_model(**{k: v for k, v in inputs.items()})

    token_embeddings = outputs.last_hidden_state
    if hasattr(token_embeddings, 'numpy'):
        token_embeddings = token_embeddings.numpy()

    attention_mask = inputs['attention_mask']
    if hasattr(attention_mask, 'numpy'):
        attention_mask = attention_mask.numpy()

    embeddings = _mean_pooling(token_embeddings, attention_mask)
    embeddings = _normalize(embeddings)

    return [e.tolist() for e in embeddings]
