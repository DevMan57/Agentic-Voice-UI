# IndexTTS2 Voice Agent Test Suite

## Quick Start

### Running Tests in WSL

The tests must be run in WSL where the Python virtual environment is configured:

```bash
# Navigate to project directory
cd /mnt/c/AI/index-tts/voice_chat

# Activate virtual environment
source ~/indextts2/.venv/bin/activate

# Run all tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_config_manager.py -v
pytest tests/unit/test_memory_manager.py -v

# Run specific test class
pytest tests/unit/test_config_manager.py::TestEnvironmentVariableParsing -v

# Run specific test
pytest tests/unit/test_config_manager.py::TestEnvironmentVariableParsing::test_env_float_valid_values -v
```

## Test Coverage

### ConfigManager Tests (20 tests)
- **TestEnvironmentVariableParsing** (7 tests): Env var parsing functions
- **TestConfigDataclasses** (7 tests): Config initialization and defaults
- **TestPerCharacterOverrides** (4 tests): Character-specific overrides
- **TestConfigIntegration** (2 tests): Config container and exports

### MemoryManager Tests (13 tests)
- **TestCharacterActivation** (4 tests): Character lifecycle
- **TestMemoryAddition** (3 tests): Semantic, procedural, episodic memories
- **TestWeightedRetrieval** (3 tests): Weighted search and filtering
- **TestMemoryDataClasses** (3 tests): Data class behavior

**Total: 33 unit tests**

## Test Structure

```
tests/
├── __init__.py              # Test package marker
├── conftest.py              # Shared fixtures (mocks, sample data)
├── unit/                    # Unit tests (isolated components)
│   ├── test_config_manager.py
│   └── test_memory_manager.py
├── integration/             # Future: Integration tests
└── fixtures/                # Future: Test data files
```

## Fixtures Available

Defined in `conftest.py`:

- `mock_storage` - Mock SQLiteStorage
- `mock_embeddings` - Mock EmbeddingManager
- `mock_llm_client` - Mock LLM client
- `mock_graph_extractor` - Mock GraphExtractor
- `sample_memory` - Sample Memory object
- `sample_character_state` - Sample CharacterState object
- `clean_environment` - Auto-cleanup of env vars (autouse=True)
- `temp_test_dir` - Temporary directory for test files

## Writing New Tests

### Example: Adding a new test

```python
import pytest
from unittest.mock import patch

def test_my_new_feature(mock_storage, mock_embeddings):
    """Test description."""
    # Setup
    manager = create_manager(mock_storage, mock_embeddings)

    # Execute
    result = manager.my_new_method()

    # Assert
    assert result == expected_value
    mock_storage.some_method.assert_called_once()
```

### Test Markers

Use markers to categorize tests:

```python
@pytest.mark.unit
def test_isolated_component():
    pass

@pytest.mark.slow
def test_long_running_operation():
    pass
```

Run specific markers:
```bash
pytest -m unit          # Run only unit tests
pytest -m "not slow"    # Skip slow tests
```

## Troubleshooting

### Import Errors

If you see import errors, ensure:
1. You're in the WSL environment
2. Virtual environment is activated
3. You're in the project root (`/mnt/c/AI/index-tts/voice_chat`)

### pytest Not Found

```bash
# Install pytest in WSL venv
source ~/indextts2/.venv/bin/activate
pip install pytest>=7.0.0
```

### Test Failures

1. **Environment variable pollution**: Tests use `clean_environment` fixture (autouse=True)
2. **Mock not configured**: Check `conftest.py` for fixture setup
3. **Import paths**: Verify imports match actual module structure

## Next Steps

### Future Enhancements

1. **Coverage Reporting**
   ```bash
   pip install pytest-cov
   pytest --cov=config --cov=memory tests/unit/ -v
   ```

2. **Integration Tests**
   - Test ConfigManager + MemoryManager integration
   - Test with real (temporary) SQLite database
   - End-to-end memory retrieval

3. **Performance Tests**
   - Benchmark weighted_search with 1000+ memories
   - Verify O(log n) sqlite-vec performance

4. **Property-Based Testing**
   ```bash
   pip install hypothesis
   ```
   - Generate random config values
   - Fuzz test memory importance scoring

## References

- **pytest docs**: https://docs.pytest.org/
- **unittest.mock**: https://docs.python.org/3/library/unittest.mock.html
- **Implementation plan**: `C:\Users\Henri Smith\.claude-membership\plans\lucky-hatching-papert.md`
