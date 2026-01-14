"""
NuExtract Local Model for Graph Extraction

A lightweight, specialized model for extracting entities and relationships
from conversation text. Runs locally via llama-cpp-python.

Model: NuExtract-2.0-2B-GGUF (Q4_K_M quantization, ~986MB)
"""

import os
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any

import paths

# Lazy imports to avoid startup delay
_llama_cpp = None
_hf_hub = None

NUEXTRACT_AVAILABLE = False
_model_instance = None

# Model configuration
MODEL_REPO = "numind/NuExtract-2.0-2B-GGUF"
MODEL_FILE = "NuExtract-2.0-2B-Q4_K_M.gguf"
MODEL_DIR = paths.NUEXTRACT_DIR

# Context window management - uses ConfigManager if available
def _get_context_config():
    """Get context window settings from ConfigManager or use defaults."""
    try:
        from config import config
        return {
            'context_window': config.memory.nuextract_context_window,
            'max_input_tokens': config.memory.nuextract_max_input_tokens,
        }
    except ImportError:
        return {
            'context_window': 4096,
            'max_input_tokens': 3234,  # 4096 - 350 (prompt) - 512 (output)
        }

# Default values (used for initial load, then ConfigManager takes over)
CONTEXT_WINDOW = 4096  # Model's context window
PROMPT_TEMPLATE_TOKENS = 350  # Reserved for prompt structure
OUTPUT_TOKENS = 512  # Reserved for model output
MAX_INPUT_TOKENS = CONTEXT_WINDOW - PROMPT_TEMPLATE_TOKENS - OUTPUT_TOKENS  # ~3234 tokens
CHARS_PER_TOKEN = 4  # Rough estimate for English text


def _truncate_for_context(user_text: str, assistant_text: str) -> tuple:
    """
    Truncate user and assistant text to fit within context window.
    
    Distributes available tokens proportionally between user and assistant text,
    favoring user text slightly (60/40 split) since user statements often contain
    more extractable facts.
    
    Returns:
        (truncated_user_text, truncated_assistant_text, was_truncated)
    """
    max_chars = MAX_INPUT_TOKENS * CHARS_PER_TOKEN  # ~12936 chars
    total_chars = len(user_text) + len(assistant_text)
    
    if total_chars <= max_chars:
        return user_text, assistant_text, False
    
    # Distribute available chars: 60% user, 40% assistant
    user_max = int(max_chars * 0.6)
    assistant_max = int(max_chars * 0.4)
    
    # If one text is shorter, give extra space to the other
    if len(user_text) < user_max:
        assistant_max = max_chars - len(user_text)
    elif len(assistant_text) < assistant_max:
        user_max = max_chars - len(assistant_text)
    
    truncated_user = user_text[:user_max]
    truncated_assistant = assistant_text[:assistant_max]
    
    # Add truncation indicator if shortened
    if len(user_text) > user_max:
        truncated_user = truncated_user[:-20] + "... [truncated]"
    if len(assistant_text) > assistant_max:
        truncated_assistant = truncated_assistant[:-20] + "... [truncated]"
    
    return truncated_user, truncated_assistant, True


def _lazy_import():
    """Lazy import heavy dependencies."""
    global _llama_cpp, _hf_hub, NUEXTRACT_AVAILABLE

    if _llama_cpp is not None:
        return True

    try:
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
        _llama_cpp = Llama
        _hf_hub = hf_hub_download
        NUEXTRACT_AVAILABLE = True
        return True
    except ImportError as e:
        print(f"[NuExtract] Dependencies not installed: {e}")
        print("[NuExtract] Run: pip install llama-cpp-python huggingface-hub")
        NUEXTRACT_AVAILABLE = False
        return False


def _download_model() -> Optional[Path]:
    """Download the NuExtract GGUF model if not present."""
    if not _lazy_import():
        return None

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_FILE

    if model_path.exists():
        return model_path

    print(f"[NuExtract] Downloading {MODEL_FILE} (~986MB)...")
    print("[NuExtract] This is a one-time download for local graph extraction.")

    try:
        downloaded_path = _hf_hub(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"[NuExtract] Model downloaded to {downloaded_path}")
        return Path(downloaded_path)
    except Exception as e:
        print(f"[NuExtract] Download failed: {e}")
        return None


def _get_model():
    """Get or initialize the model instance (singleton)."""
    global _model_instance

    if _model_instance is not None:
        return _model_instance

    if not _lazy_import():
        return None

    model_path = _download_model()
    if model_path is None:
        return None

    print(f"[NuExtract] Loading model from {model_path}...")

    try:
        _model_instance = _llama_cpp(
            model_path=str(model_path),
            n_ctx=4096,           # Context window
            n_threads=4,          # CPU threads
            n_gpu_layers=0,       # CPU-only by default (change to -1 for full GPU)
            verbose=False
        )
        print("[NuExtract] Model loaded successfully")
        return _model_instance
    except Exception as e:
        print(f"[NuExtract] Failed to load model: {e}")
        return None


def extract_entities_and_relationships(
    user_text: str,
    assistant_text: str,
    character_id: str = "assistant"
) -> Dict[str, Any]:
    """
    Extract entities and relationships from a conversation turn.

    Args:
        user_text: What the user said
        assistant_text: What the assistant replied
        character_id: Character context

    Returns:
        Dict with 'entities' and 'relationships' lists
    """
    model = _get_model()

    if model is None:
        return {"entities": [], "relationships": []}

    # Skip very short exchanges
    if len(user_text) < 10 and len(assistant_text) < 10:
        return {"entities": [], "relationships": []}
    
    # Truncate to fit context window (prevents "Requested tokens exceed context" errors)
    user_text, assistant_text, was_truncated = _truncate_for_context(user_text, assistant_text)
    if was_truncated:
        print(f"[NuExtract] Truncated input to fit {CONTEXT_WINDOW} token context window")

    # NuExtract prompt format
    prompt = f"""<|input|>
Extract entities and relationships from this conversation.

Conversation:
User: {user_text}
Assistant ({character_id}): {assistant_text}

Extract:
1. Entities: People, Locations, Concepts, Items, Projects mentioned
2. Relationships: How entities connect (KNOWS, OWNS, LIVES_IN, WORKS_ON, LIKES, etc.)

Rules:
- Resolve pronouns: "I live in London" -> User LIVES_IN London
- Use consistent names: "Harry", "Potter" -> "Harry Potter"
- Skip casual greetings and temporary states
<|output|>
{{
    "entities": [
        {{"name": "Entity Name", "type": "Person|Location|Concept|Item|Project", "description": "Brief context"}}
    ],
    "relationships": [
        {{"source": "Entity1", "target": "Entity2", "relation": "RELATION_TYPE", "strength": 0.8}}
    ]
}}"""

    try:
        response = model(
            prompt,
            max_tokens=512,
            temperature=0.0,
            stop=["<|input|>", "<|end|>"]
        )

        output_text = response["choices"][0]["text"].strip()

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', output_text)
        if not json_match:
            return {"entities": [], "relationships": []}

        data = json.loads(json_match.group())

        return {
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", [])
        }

    except json.JSONDecodeError as e:
        print(f"[NuExtract] JSON parse error: {e}")
        return {"entities": [], "relationships": []}
    except Exception as e:
        print(f"[NuExtract] Extraction error: {e}")
        return {"entities": [], "relationships": []}


def nuextract_llm_client(prompt: str, temperature: float = 0.0) -> str:
    """
    LLM client interface compatible with GraphExtractor.

    This allows NuExtract to be used as a drop-in replacement
    for the main LLM in graph extraction.
    """
    model = _get_model()

    if model is None:
        return ""

    try:
        response = model(
            prompt,
            max_tokens=512,
            temperature=temperature,
            stop=["<|input|>", "<|end|>", "```"]
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[NuExtract] Generation error: {e}")
        return ""


def is_available() -> bool:
    """Check if NuExtract is available (dependencies installed)."""
    return _lazy_import()


def preload_model():
    """Preload the model (call during startup if desired)."""
    return _get_model() is not None


# For testing
if __name__ == "__main__":
    print("Testing NuExtract...")

    result = extract_entities_and_relationships(
        user_text="I'm working on a Python project called VoiceChat. I live in San Francisco.",
        assistant_text="That sounds interesting! VoiceChat sounds like a cool project. San Francisco is a great city for tech work.",
        character_id="assistant"
    )

    print(f"Entities: {result['entities']}")
    print(f"Relationships: {result['relationships']}")
