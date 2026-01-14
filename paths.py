"""
Centralized path configuration for tts2-voice-agent.

Import this module FIRST in any file that needs project paths.
It automatically adds lib/ to sys.path for vendored packages.
"""
from pathlib import Path
import sys
import os

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()

# Add lib/ to Python path for vendored packages (indextts)
LIB_DIR = PROJECT_ROOT / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
INDEXTTS_MODELS_DIR = MODELS_DIR / "indextts2"
KOKORO_MODELS_DIR = MODELS_DIR / "kokoro"
SUPERTONIC_MODELS_DIR = MODELS_DIR / "supertonic"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
NUEXTRACT_DIR = MODELS_DIR / "nuextract"
HF_CACHE_DIR = MODELS_DIR / "hf_cache"

# Set HuggingFace cache environment variables
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))

# Runtime directories
SESSIONS_DIR = PROJECT_ROOT / "sessions"
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
SKILLS_DIR = PROJECT_ROOT / "skills"
VOICES_DIR = PROJECT_ROOT / "voice_reference"
