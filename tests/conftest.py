"""
Shared pytest fixtures for IndexTTS2 Voice Agent tests

This module provides reusable fixtures for mocking dependencies
and creating test data objects.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
from pathlib import Path


# ============================================================================
# Mock Dependencies for MemoryManager Tests
# ============================================================================

@pytest.fixture
def mock_storage():
    """
    Mock SQLiteStorage for MemoryManager tests.

    Provides default return values for common operations:
    - load_character_state() → None (new character)
    - save_character_state() → None
    - add_memory() → "mem_id_123"
    - search_memories() → []
    """
    storage = Mock()
    storage.load_character_state.return_value = None
    storage.save_character_state.return_value = None
    storage.add_memory.return_value = "mem_id_123"
    storage.search_memories.return_value = []
    storage.get_memories_by_character.return_value = []
    storage.apply_decay.return_value = None
    return storage


@pytest.fixture
def mock_embeddings():
    """
    Mock EmbeddingManager for MemoryManager tests.

    Provides:
    - model_name: "mock-model"
    - dimension: 1024
    - embed() → [0.1] * 1024
    - embed_batch() → [[0.1] * 1024]
    """
    embeddings = Mock()
    embeddings.model_name = "mock-model"
    embeddings.dimension = 1024
    embeddings.backend = "mock"
    embeddings.embed.return_value = [0.1] * 1024
    embeddings.embed_batch.return_value = [[0.1] * 1024]
    return embeddings


@pytest.fixture
def mock_llm_client():
    """
    Mock LLM client for importance scoring and summarization.

    Default return value for generate():
    {"importance": 0.7}
    """
    client = Mock()
    client.generate.return_value = {"importance": 0.7}
    return client


@pytest.fixture
def mock_graph_extractor():
    """
    Mock GraphExtractor for background graph extraction.

    Provides:
    - use_local: True
    - extract_from_text() → None (background operation)
    """
    extractor = Mock()
    extractor.use_local = True
    extractor.extract_from_text.return_value = None
    return extractor


# ============================================================================
# Sample Data Objects
# ============================================================================

@pytest.fixture
def sample_memory():
    """
    Create a sample Memory object for testing.

    Returns a semantic memory with:
    - character_id: "hermione"
    - content: "Hermione loves studying magic"
    - importance_score: 0.8
    - decay_factor: 1.0 (no decay)
    """
    # Import here to avoid circular dependencies
    from memory.memory_manager import Memory

    return Memory(
        id="mem_123",
        character_id="hermione",
        memory_type="semantic",
        content="Hermione loves studying magic",
        embedding=[0.1] * 1024,
        importance_score=0.8,
        decay_factor=1.0,
        tags={},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        last_accessed=datetime.utcnow(),
        access_count=0
    )


@pytest.fixture
def sample_character_state():
    """
    Create a sample CharacterState for testing.

    Returns a CharacterState with:
    - character_id: "hermione"
    - current_mood: "neutral"
    - interaction_count: 5
    """
    from memory.memory_manager import CharacterState

    return CharacterState(
        character_id="hermione",
        current_mood="neutral",
        interaction_count=5
    )


# ============================================================================
# Environment Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """
    Auto-cleanup fixture that runs before each test.

    Clears environment variables to ensure test isolation.
    This prevents config tests from interfering with each other.
    """
    # Store original env vars
    import os
    original_env = os.environ.copy()

    # Clear all test-related env vars
    test_env_vars = [
        'VAD_ENERGY_THRESHOLD', 'SILERO_THRESHOLD', 'VAD_CONSECUTIVE_FRAMES',
        'AUDIO_SAMPLE_RATE', 'TTS_BACKEND', 'KOKORO_SPEED',
        'MEMORY_RECENCY_WEIGHT', 'MEMORY_RELEVANCE_WEIGHT', 'MEMORY_IMPORTANCE_WEIGHT',
        'SERVER_PORT', 'LLM_PROVIDER', 'EMOTION_ENABLED'
    ]

    for var in test_env_vars:
        monkeypatch.delenv(var, raising=False)

    yield

    # Restore original env (pytest handles this automatically with monkeypatch)


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_test_dir(tmp_path):
    """
    Provide a temporary directory for test files.

    Args:
        tmp_path: pytest's built-in tmp_path fixture

    Returns:
        Path: Temporary directory path
    """
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    return test_dir
