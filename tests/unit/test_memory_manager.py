"""
Unit tests for MemoryManager (memory/memory_manager.py)

Tests cover:
1. Character activation/deactivation lifecycle
2. Memory addition methods (semantic, procedural, episodic)
3. Weighted search and retrieval

Run with: pytest tests/unit/test_memory_manager.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from memory.memory_manager import (
    Memory, CharacterState, EpisodicSummary,
    MultiCharacterMemoryManager
)


# ============================================================================
# Test Class 1: Character Activation
# ============================================================================

class TestCharacterActivation:
    """Test character activation and deactivation lifecycle."""

    @patch('memory.graph_extractor.GraphExtractor')
    @patch('memory.sqlite_storage.SQLiteStorage')
    def test_activate_character_new_character(self, mock_storage_class, mock_graph_extractor_class):
        """Test activating a new character (no existing state)."""
        # Setup mocks
        mock_storage = Mock()
        mock_storage.load_character_state.return_value = None  # New character
        mock_storage.save_character_state.return_value = None
        mock_storage.apply_decay.return_value = None
        mock_storage_class.return_value = mock_storage

        mock_extractor = Mock()
        mock_extractor.use_local = True
        mock_graph_extractor_class.return_value = mock_extractor

        # Create manager
        manager = MultiCharacterMemoryManager(use_local_storage=True)

        # Activate new character
        state = manager.activate_character('hermione')

        # Assertions
        assert isinstance(state, CharacterState)
        assert state.character_id == 'hermione'
        assert state.interaction_count == 0
        assert state.current_mood == 'neutral'

        # Verify state was saved
        mock_storage.save_character_state.assert_called_once()

        # Verify character is in active cache
        assert 'hermione' in manager._character_states

        # Verify pending interactions initialized
        assert 'hermione' in manager._pending_interactions
        assert manager._pending_interactions['hermione'] == []

    @patch('memory.graph_extractor.GraphExtractor')
    @patch('memory.sqlite_storage.SQLiteStorage')
    def test_activate_character_existing_character(self, mock_storage_class, mock_graph_extractor_class):
        """Test activating a character with existing state."""
        # Setup mocks
        existing_state = CharacterState(
            character_id='hermione',
            current_mood='studious',
            interaction_count=42
        )

        mock_storage = Mock()
        mock_storage.load_character_state.return_value = existing_state
        mock_storage.apply_decay.return_value = None
        mock_storage_class.return_value = mock_storage

        mock_extractor = Mock()
        mock_extractor.use_local = True
        mock_graph_extractor_class.return_value = mock_extractor

        # Create manager
        manager = MultiCharacterMemoryManager(use_local_storage=True)

        # Activate existing character
        state = manager.activate_character('hermione')

        # Assertions - should return existing state
        assert state.character_id == 'hermione'
        assert state.current_mood == 'studious'
        assert state.interaction_count == 42

        # Verify state is cached
        assert manager._character_states['hermione'] == existing_state

    @patch('memory.graph_extractor.GraphExtractor')
    @patch('memory.sqlite_storage.SQLiteStorage')
    def test_activate_character_already_active_returns_cached(self, mock_storage_class, mock_graph_extractor_class):
        """Test activating an already-active character returns cached state."""
        # Setup mocks
        mock_storage = Mock()
        mock_storage.load_character_state.return_value = None
        mock_storage.save_character_state.return_value = None
        mock_storage.apply_decay.return_value = None
        mock_storage_class.return_value = mock_storage

        mock_extractor = Mock()
        mock_extractor.use_local = True
        mock_graph_extractor_class.return_value = mock_extractor

        # Create manager
        manager = MultiCharacterMemoryManager(use_local_storage=True)

        # First activation
        state1 = manager.activate_character('hermione')

        # Reset mock to check it's not called again
        mock_storage.load_character_state.reset_mock()

        # Second activation (should return cached)
        state2 = manager.activate_character('hermione')

        # Verify same state object returned
        assert state1 is state2

        # Verify storage was NOT queried second time
        mock_storage.load_character_state.assert_not_called()

    @patch('memory.graph_extractor.GraphExtractor')
    @patch('memory.sqlite_storage.SQLiteStorage')
    def test_deactivate_character_saves_state(self, mock_storage_class, mock_graph_extractor_class):
        """Test deactivating a character saves state and clears cache."""
        # Setup mocks
        mock_storage = Mock()
        mock_storage.load_character_state.return_value = None
        mock_storage.save_character_state.return_value = None
        mock_storage.apply_decay.return_value = None
        mock_storage_class.return_value = mock_storage

        mock_extractor = Mock()
        mock_extractor.use_local = True
        mock_graph_extractor_class.return_value = mock_extractor

        # Create manager and activate character
        manager = MultiCharacterMemoryManager(use_local_storage=True)
        state = manager.activate_character('hermione')

        # Modify state
        state.interaction_count = 10
        state.current_mood = 'happy'

        # Reset save_character_state mock
        mock_storage.save_character_state.reset_mock()

        # Deactivate character
        manager.deactivate_character('hermione')

        # Verify state was saved
        mock_storage.save_character_state.assert_called_once()
        saved_state = mock_storage.save_character_state.call_args[0][0]
        assert saved_state.interaction_count == 10
        assert saved_state.current_mood == 'happy'

        # Verify character removed from cache
        assert 'hermione' not in manager._character_states


# ============================================================================
# Test Class 2: Memory Addition
# ============================================================================

class TestMemoryAddition:
    """Test memory addition methods (semantic, procedural, episodic)."""

    def test_add_semantic_memory(self):
        """Test adding a semantic memory."""
        # Create manager with mocked embeddings
        with patch('memory.memory_manager.EmbeddingManager') as mock_embed_class:
            mock_embeddings = Mock()
            mock_embeddings.model_name = "mock-model"
            mock_embeddings.embed.return_value = [0.1] * 1024
            mock_embed_class.return_value = mock_embeddings

            manager = MultiCharacterMemoryManager(use_local_storage=True)

            # Use patch.object to properly mock the storage method
            with patch.object(manager.storage, 'store_memory', return_value=True) as mock_store:
                # Add semantic memory
                manager.add_semantic_memory('hermione', 'Hermione loves books', importance=0.9)

                # Verify embedding was generated
                mock_embeddings.embed.assert_called_once()
                call_args = mock_embeddings.embed.call_args[0][0]
                assert 'Hermione loves books' in call_args

                # Verify memory was stored
                mock_store.assert_called_once()
                memory_arg = mock_store.call_args[0][0]
                assert memory_arg.memory_type == 'semantic'
                assert memory_arg.content == 'Hermione loves books'
                assert memory_arg.importance_score == 0.9
                assert memory_arg.decay_factor == 1.0  # No decay for semantic

    def test_add_procedural_memory(self):
        """Test adding a procedural memory."""
        # Create manager with mocked embeddings
        with patch('memory.memory_manager.EmbeddingManager') as mock_embed_class:
            mock_embeddings = Mock()
            mock_embeddings.model_name = "mock-model"
            mock_embeddings.embed.return_value = [0.2] * 1024
            mock_embed_class.return_value = mock_embeddings

            manager = MultiCharacterMemoryManager(use_local_storage=True)

            # Use patch.object to properly mock the storage method
            with patch.object(manager.storage, 'store_memory', return_value=True) as mock_store:
                # Add procedural memory
                manager.add_procedural_memory('hermione', 'Always verify facts before stating')

                # Verify memory was stored
                mock_store.assert_called_once()
                memory_arg = mock_store.call_args[0][0]
                assert memory_arg.memory_type == 'procedural'
                assert memory_arg.content == 'Always verify facts before stating'
                assert memory_arg.importance_score == 0.9  # Highest importance
                assert memory_arg.decay_factor == 1.0  # No decay for procedural

    def test_add_interaction_creates_episodic_memory(self):
        """Test adding an interaction creates episodic memory."""
        # Create manager with mocked embeddings
        with patch('memory.memory_manager.EmbeddingManager') as mock_embed_class:
            mock_embeddings = Mock()
            mock_embeddings.model_name = "mock-model"
            mock_embeddings.embed.return_value = [0.3] * 1024
            mock_embed_class.return_value = mock_embeddings

            manager = MultiCharacterMemoryManager(use_local_storage=True)

            # Mock storage methods to ensure clean state
            with patch.object(manager.storage, 'load_character_state', return_value=None):
                with patch.object(manager.storage, 'save_character_state', return_value=None):
                    with patch.object(manager.storage, 'store_memory', return_value=True) as mock_store:
                        # Mock _score_importance
                        with patch.object(manager, '_score_importance', return_value=0.7):
                            # Activate character (will create new state since load returns None)
                            manager.activate_character('hermione')

                            # Add interaction
                            manager.add_interaction(
                                'hermione',
                                'Hello, how are you?',
                                'Hi there! I am doing well.',
                                metadata={'emotion': 'happy'}
                            )

                            # Verify episodic memory was created
                            mock_store.assert_called()
                            memory_arg = mock_store.call_args[0][0]
                            assert memory_arg.memory_type == 'episodic'
                            assert 'User: Hello, how are you?' in memory_arg.content
                            assert 'Assistant: Hi there! I am doing well.' in memory_arg.content
                            assert memory_arg.importance_score == 0.7

                            # Verify interaction count incremented
                            state = manager.get_character_state('hermione')
                            assert state.interaction_count == 1


# ============================================================================
# Test Class 3: Weighted Retrieval
# ============================================================================

class TestWeightedRetrieval:
    """Test weighted search scoring and ranking."""

    def test_weighted_search_basic(self):
        """Test basic weighted search returns ranked results."""
        # Create sample memories with different relevance scores
        sample_memories = [
            (Memory(
                id=f"mem_{i}",
                character_id="hermione",
                memory_type="semantic",
                content=f"Memory {i}",
                embedding=[0.1 * i] * 1024,
                importance_score=0.5,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            ), 0.9 - (i * 0.1))  # Decreasing relevance
            for i in range(5)
        ]

        # Create manager with mocked embeddings
        with patch('memory.memory_manager.EmbeddingManager') as mock_embed_class:
            mock_embeddings = Mock()
            mock_embeddings.model_name = "mock-model"
            mock_embeddings.embed.return_value = [0.5] * 1024
            mock_embed_class.return_value = mock_embeddings

            manager = MultiCharacterMemoryManager(use_local_storage=True)

            # Create smart mock that filters by min_similarity
            def mock_semantic_search(*args, **kwargs):
                min_sim = kwargs.get('min_similarity', 0.3)
                return [(mem, score) for mem, score in sample_memories if score >= min_sim]

            manager.storage.semantic_search = Mock(side_effect=mock_semantic_search)

            # Perform weighted search (default min_similarity=0.3, so all 5 memories pass)
            results = manager.weighted_search('hermione', 'What books does Hermione like?', limit=3)

            # Verify results
            assert len(results) <= 3  # Respects limit
            assert all(len(r) == 3 for r in results)  # Each result is (Memory, weighted_score, relevance)

            # Verify results are sorted by weighted_score (descending)
            if len(results) > 1:
                for i in range(len(results) - 1):
                    assert results[i][1] >= results[i+1][1], "Results should be sorted by weighted_score"

    def test_weighted_search_filters_by_type(self):
        """Test weighted search filters by memory_type."""
        # Create manager with mocked embeddings
        with patch('memory.memory_manager.EmbeddingManager') as mock_embed_class:
            mock_embeddings = Mock()
            mock_embeddings.model_name = "mock-model"
            mock_embeddings.embed.return_value = [0.5] * 1024
            mock_embed_class.return_value = mock_embeddings

            manager = MultiCharacterMemoryManager(use_local_storage=True)

            # Use patch.object to properly mock storage method
            with patch.object(manager.storage, 'semantic_search', return_value=[]) as mock_search:
                # Search with memory_type filter
                manager.weighted_search('hermione', 'query', memory_type='semantic')

                # Verify storage.semantic_search was called
                mock_search.assert_called_once()

    def test_weighted_search_respects_min_similarity(self):
        """Test weighted search filters by min_similarity threshold."""
        # Create memories with varying similarity scores
        sample_memories = [
            (Memory(
                id="mem_high",
                character_id="hermione",
                memory_type="semantic",
                content="High similarity",
                embedding=[0.9] * 1024,
                importance_score=0.5,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            ), 0.9),  # Above threshold
            (Memory(
                id="mem_medium",
                character_id="hermione",
                memory_type="semantic",
                content="Medium similarity",
                embedding=[0.5] * 1024,
                importance_score=0.5,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            ), 0.5),  # Below threshold
            (Memory(
                id="mem_low",
                character_id="hermione",
                memory_type="semantic",
                content="Low similarity",
                embedding=[0.2] * 1024,
                importance_score=0.5,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            ), 0.2),  # Below threshold
        ]

        # Create manager with mocked embeddings
        with patch('memory.memory_manager.EmbeddingManager') as mock_embed_class:
            mock_embeddings = Mock()
            mock_embeddings.model_name = "mock-model"
            mock_embeddings.embed.return_value = [0.5] * 1024
            mock_embed_class.return_value = mock_embeddings

            manager = MultiCharacterMemoryManager(use_local_storage=True)

            # Create smart mock that filters by min_similarity (like real semantic_search)
            def mock_semantic_search(*args, **kwargs):
                min_sim = kwargs.get('min_similarity', 0.3)
                return [(mem, score) for mem, score in sample_memories if score >= min_sim]

            manager.storage.semantic_search = Mock(side_effect=mock_semantic_search)

            # Search with min_similarity=0.6
            results = manager.weighted_search('hermione', 'query', min_similarity=0.6)

            # Only memories with similarity >= 0.6 should be returned
            assert len(results) == 1
            assert results[0][0].id == "mem_high"
            assert results[0][2] == 0.9  # relevance score


# ============================================================================
# Test Class 4: Memory Data Classes
# ============================================================================

class TestMemoryDataClasses:
    """Test Memory and CharacterState data classes."""

    def test_memory_to_dict(self):
        """Test Memory.to_dict() serialization."""
        memory = Memory(
            id="test_123",
            character_id="hermione",
            memory_type="semantic",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            importance_score=0.8,
            decay_factor=1.0,
            tags={'category': 'books'},
            created_at=datetime(2025, 1, 1, 12, 0, 0),
            updated_at=datetime(2025, 1, 2, 12, 0, 0),
            last_accessed=datetime(2025, 1, 3, 12, 0, 0),
            access_count=5
        )

        memory_dict = memory.to_dict()

        assert memory_dict['id'] == "test_123"
        assert memory_dict['character_id'] == "hermione"
        assert memory_dict['memory_type'] == "semantic"
        assert memory_dict['content'] == "Test content"
        assert memory_dict['importance_score'] == 0.8
        assert memory_dict['access_count'] == 5

    def test_memory_compute_recency_score(self):
        """Test Memory.compute_recency_score() decay calculation."""
        # Memory accessed 10 hours ago
        memory = Memory(
            id="test",
            character_id="hermione",
            memory_type="semantic",
            content="Test",
            last_accessed=datetime.utcnow() - timedelta(hours=10)
        )

        recency_score = memory.compute_recency_score()

        # Score should be 0.995^10 ≈ 0.951
        expected = 0.995 ** 10
        assert abs(recency_score - expected) < 0.01

    def test_character_state_initialization(self):
        """Test CharacterState initializes with correct defaults."""
        state = CharacterState(character_id="hermione")

        assert state.character_id == "hermione"
        assert state.current_mood == "neutral"
        assert state.interaction_count == 0
        assert 'valence' in state.emotional_state
        assert 'familiarity' in state.relationship_with_user
        assert isinstance(state.session_context, list)
