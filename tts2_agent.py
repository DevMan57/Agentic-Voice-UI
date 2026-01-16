#!/usr/bin/env python3

# ============================================================================
# Warning Suppression (MUST be before any other imports)
# ============================================================================
# Set PYTHONWARNINGS env var first - this is checked by Python before warnings module loads
import os
import sys

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::DeprecationWarning,ignore::ResourceWarning,ignore::UserWarning"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence TensorFlow
os.environ["PYTHONTRACEMALLOC"] = "0"     # No tracemalloc hints
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python" # Fix for protobuf conflict in IndexTTS2

import warnings
import sys
import threading

# Additional filterwarnings for this process
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weight_norm.*")
warnings.filterwarnings("ignore", message=".*unclosed.*")
warnings.filterwarnings("ignore", message=".*np.bool8.*")
warnings.filterwarnings("ignore", message=".*WebSocketServerProtocol.*")

# Silence noisy loggers
import logging

# Force all output to match the Lapis Lazuli theme (RGB 0, 191, 255)
LAPIS = "\033[38;2;0;191;255m"
RESET = "\033[0m"

# Wrap stdout/stderr to always output lapis lazuli
class LapisStream:
    def __init__(self, stream):
        self._stream = stream
        self._at_line_start = True

    def write(self, text):
        if text:
            # Add lapis lazuli color code at the start of each line
            if self._at_line_start and text.strip():
                self._stream.write(LAPIS)
            try:
                self._stream.write(text)
            except UnicodeEncodeError:
                # Windows cp1252 can't handle some Unicode - replace with ASCII equivalents
                safe_text = text.replace('✓', '[OK]').replace('✗', '[X]').replace('⏳', '[...]')
                try:
                    self._stream.write(safe_text)
                except UnicodeEncodeError:
                    self._stream.write(safe_text.encode('ascii', 'replace').decode('ascii'))
            self._at_line_start = text.endswith('\n')

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)

# Apply lapis wrapper to stdout/stderr
sys.stdout = LapisStream(sys.__stdout__)
sys.stderr = LapisStream(sys.__stderr__)

class LapisFormatter(logging.Formatter):
    def format(self, record):
        return f"{LAPIS}{super().format(record)}"

# Apply lapis formatter to all existing and future handlers
def apply_lapis_formatter():
    formatter = LapisFormatter("%(message)s")
    root_logger = logging.getLogger()

    # Apply to root logger
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
        handler.stream = sys.stdout
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # Apply to specific loggers that Gradio/uvicorn use
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "gradio", "fastapi"]:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            handler.setFormatter(formatter)
            handler.stream = sys.stdout

apply_lapis_formatter()

# Silence noisy loggers (but keep them lapis when they do speak)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("uvicorn").setLevel(logging.WARNING)  # Allow startup messages
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)  # Silence access logs
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

# ============================================================================
# Standard Imports
# ============================================================================

import re
import asyncio
import base64
import copy
import platform
from pathlib import Path
from datetime import datetime
import time
import socket
import json
import scipy.io.wavfile as wav
from threading import Lock, Thread
from typing import Optional, List, Tuple, Dict, Any
import tempfile

from tools import init_tools, REGISTRY, set_graph_tool_memory_manager
from mcp_client import MCPManager, MCP_AVAILABLE
from config.manager import config

# Service Layer (Phase 1: Architecture Cleanup)
from services import ServiceContainer, ChatService, ChatResult, ToolCallInfo
from ui.formatters import (
    format_memory_for_ui as ui_format_memory,
    format_tool_call_html as ui_format_tool,
    format_message_with_extras as ui_format_message,
    format_memory_recall_html as ui_format_recall,
    filter_for_speech as ui_filter_speech,
    format_chat_result_for_ui as ui_format_chat_result
)


import gradio as gr
import requests
import numpy as np
import torch

# ============================================================================
# Platform Detection
# ============================================================================

PLATFORM = platform.system().lower()  # 'windows', 'linux', 'darwin'
IS_WINDOWS = PLATFORM == 'windows'
IS_LINUX = PLATFORM == 'linux'
IS_MAC = PLATFORM == 'darwin'
IS_WSL = IS_LINUX and 'microsoft' in platform.uname().release.lower()

# ============================================================================
# IndexTTS2 Import with Graceful Fallback
# ============================================================================

TTS_AVAILABLE = False
IndexTTS2 = None

try:
    # Import paths module to configure sys.path for vendored packages
    import paths
    from indextts.infer_v2 import IndexTTS2 as _IndexTTS2
    IndexTTS2 = _IndexTTS2
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"[TTS] IndexTTS2 import failed: {e}")
    TTS_AVAILABLE = False
except Exception as e:
    print(f"[TTS] IndexTTS2 import error: {e}")
    TTS_AVAILABLE = False

# Kokoro TTS
try:
    from audio.tts_kokoro import KokoroTTS
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

# Import memory system
from memory.characters import CharacterManager, Character, create_character_manager
from memory.memory_manager import MultiCharacterMemoryManager, create_memory_manager
from group_manager import GROUP_MANAGER, GroupChatManager

# Import shared utilities
from utils import create_dark_theme

# PIL for image handling
try:
    from PIL import Image, ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# PDF support
try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Screenshot support - DISABLED in WSL (X11 issues)
# pyautogui requires X11 display which isn't available in WSL
SCREEN_AVAILABLE = False
print("[Info] Screenshot support disabled (WSL environment)")

# Speech Emotion Recognition
try:
    from audio.emotion_detector import EmotionDetector, EmotionResult
    from audio.emotion_bridge import should_run_external_emotion
    from concurrent.futures import ThreadPoolExecutor, Future
    EMOTION_AVAILABLE = True
    _emotion_detector = None
    _current_emotion = None  # Stores last detected emotion for LLM context
    _emotion_future: Optional[Future] = None  # For async emotion detection
    _emotion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="emotion")
except ImportError:
    EMOTION_AVAILABLE = False
    _emotion_detector = None
    _current_emotion = None
    _emotion_future = None
    _emotion_executor = None
    # Provide fallback for emotion bridge function
    def should_run_external_emotion(stt_backend: str) -> bool:
        return stt_backend != "sensevoice"
    print("[Info] Emotion detection not available. Install transformers and torch for SER.")

# GraphRAG availability
try:
    from memory.graph_rag import GraphRAGProcessor
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False


def _detect_emotion_sync(audio_data) -> Optional['EmotionResult']:
    """
    Internal synchronous emotion detection (runs in thread pool).
    """
    global _emotion_detector, _current_emotion

    try:
        sample_rate, audio_np = audio_data

        # Initialize detector lazily
        if _emotion_detector is None:
            _emotion_detector = EmotionDetector()

        # Analyze the audio
        result = _emotion_detector.analyze(
            audio_array=audio_np.astype(np.float32) / 32768.0 if audio_np.dtype == np.int16 else audio_np,
            sample_rate=sample_rate
        )

        if result:
            _current_emotion = result
            print(f"[SER] Detected emotion: {result.label} (V={result.valence:.2f}, A={result.arousal:.2f})")

        return result

    except Exception as e:
        print(f"[SER] Emotion detection failed: {e}")
        return None


def detect_emotion_from_audio_async(audio_data) -> None:
    """
    Start async emotion detection (non-blocking).

    Runs in parallel with memory retrieval to hide CPU latency.
    Call get_current_emotion_context() later to get the result.
    """
    global _emotion_future

    if not EMOTION_AVAILABLE:
        return

    # Check if emotion detection is enabled in settings
    if not SETTINGS.get("emotion_detection_enabled", True):
        return

    if audio_data is None:
        return

    # Submit to thread pool (non-blocking)
    _emotion_future = _emotion_executor.submit(_detect_emotion_sync, audio_data)
    print("[SER] Started async emotion detection")


def detect_emotion_from_audio(audio_data) -> Optional['EmotionResult']:
    """
    Detect emotion from audio data (blocking, legacy compatibility).

    For new code, prefer detect_emotion_from_audio_async() + get_current_emotion_context().
    """
    global _current_emotion

    if not EMOTION_AVAILABLE:
        return None

    # Check if emotion detection is enabled in settings
    if not SETTINGS.get("emotion_detection_enabled", True):
        return None

    if audio_data is None:
        return None

    return _detect_emotion_sync(audio_data)


def get_current_emotion_context() -> str:
    """
    Get the current emotion context for LLM prompt injection.

    If async detection is in progress, waits for it to complete (with timeout).
    """
    global _current_emotion, _emotion_future

    # If async detection is running, wait for result
    if _emotion_future is not None:
        try:
            # Wait up to 2 seconds for emotion detection to complete
            _emotion_future.result(timeout=2.0)
        except Exception as e:
            print(f"[SER] Async emotion detection timed out or failed: {e}")
        finally:
            _emotion_future = None

    if _current_emotion:
        return _current_emotion.to_prompt_context()
    return ""


def clear_current_emotion():
    """Clear the current emotion after use"""
    global _current_emotion, _emotion_future
    _current_emotion = None
    _emotion_future = None

# ============================================================================
# Helpers
# ============================================================================

def take_screenshot():
    """Capture screen and return PIL Image"""
    if not SCREEN_AVAILABLE:
        return None
    try:
        # Minimal delay to allow UI to minimize if needed (optional)
        return pyautogui.screenshot()
    except Exception as e:
        print(f"[Error] Screenshot: {e}")
        return None

def extract_text_from_pdf(filepath: str) -> str:
    """Extract text from PDF file"""
    if not PDF_AVAILABLE:
        return ""
    try:
        reader = pypdf.PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"[Error] PDF Read: {e}")
        return ""

# ============================================================================
# Hands-free Mode (VAD) Manager
# ============================================================================

vad_manager_instance = None

class VADManager:
    """
    Manages hands-free voice recording mode across all platforms.
    
    Platform behavior:
    - Native Windows: Uses vad_recorder.py directly (in-process)
    - Native Linux: Uses vad_recorder.py directly (in-process)  
    - WSL: CANNOT record audio - user must run vad_windows.py on Windows
    
    All modes write to recordings/ directory using latest.txt as trigger.
    The UI polling picks up recordings identically regardless of source.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.recorder = None
        self.is_active = False
        # Cache platform detection
        self._is_wsl = IS_WSL
        self._platform = PLATFORM
        
    def start(self, threshold: float = 0.8) -> str:
        """Start VAD. Returns status message."""
        if self.is_active:
            return "Already active"
        
        # WSL: Audio recording is IMPOSSIBLE - no microphone access
        # User must run vad_windows.py on Windows side
        if self._is_wsl:
            self.is_active = True
            print("[VAD] WSL detected - audio must be captured on Windows")
            print("[VAD] Run: python vad_windows.py  (from Windows CMD)")
            return "wsl_remote"
        
        # Native Windows or Linux: Use vad_recorder.py directly
        try:
            from audio.vad_recorder import ContinuousVADRecorder, VADConfig
            
            # Convert threshold (0.0-1.0 from UI) to VAD config
            # Higher UI threshold = more silence needed = higher silence_threshold
            config = VADConfig(
                backend="silero",  # Neural network VAD - much better at ignoring coughs/bumps
                silence_threshold=threshold,
                energy_threshold=0.022,  # Raised from 0.015 to reduce false triggers
                min_speech_duration=0.3,
                max_recording_duration=30.0
            )
            
            self.recorder = ContinuousVADRecorder(
                config=config, 
                output_dir=self.output_dir
            )
            
            if self.recorder.start():
                self.is_active = True
                print(f"[VAD] Hands-free mode started ({self._platform})")
                return "started"
            else:
                print("[VAD] Failed to start recorder")
                return "error: Failed to start audio capture"
                
        except ImportError as e:
            print(f"[VAD] Missing dependency: {e}")
            print("[VAD] Install with: pip install pyaudio numpy")
            return f"error: {e}"
        except Exception as e:
            print(f"[VAD] Failed to start: {e}")
            self.is_active = False
            return f"error: {e}"
            
    def stop(self) -> str:
        """Stop VAD. Returns status message."""
        if not self.is_active:
            return "Not active"
        
        # WSL: Just update state - Windows script runs independently
        if self._is_wsl:
            self.is_active = False
            print("[VAD] WSL mode disabled - stop vad_windows.py on Windows")
            return "stopped_wsl"
        
        # Native: Stop the recorder
        if self.recorder:
            try:
                self.recorder.stop()
                print("[VAD] Hands-free mode stopped")
            except Exception as e:
                print(f"[VAD] Error stopping: {e}")
        
        self.recorder = None
        self.is_active = False
        return "stopped"

    def pause(self, duration: float = None):
        """
        Pause VAD for a duration (or indefinitely) to prevent self-hearing during TTS.
        """
        if self.recorder:
            self.recorder.pause(duration)
        elif self._is_wsl:
            # For WSL, we communicate via file. 
            # We must maintain 'enabled' state based on self.is_active, but set 'tts_playing' to True.
            update_vad_control(enabled=self.is_active, tts_playing=True)
            
            if duration:
                # Spawn a thread to reset it after audio finishes
                def _reset():
                    time.sleep(duration)
                    # CRITICAL FIX: Only set enabled=True if VAD is actually supposed to be active!
                    # Previous bug: this blinded set enabled=True, overriding the checkbox.
                    update_vad_control(enabled=self.is_active, tts_playing=False)
                    
                threading.Thread(target=_reset, daemon=True).start()

# ============================================================================
# Configuration - Cross-Platform Paths
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "config.env"
VOICE_REF_DIR = SCRIPT_DIR / "voice_reference"
SESSIONS_DIR = SCRIPT_DIR / "sessions"
CONVERSATIONS_DIR = SESSIONS_DIR / "conversations"  # Consolidated under sessions/
SETTINGS_FILE = SCRIPT_DIR / "settings.json"
RECORDINGS_DIR = SCRIPT_DIR / "recordings"
PTT_STATUS_FILE = RECORDINGS_DIR / "ptt_status.txt"
VAD_CONTROL_FILE = RECORDINGS_DIR / "vad_control.txt"  # Controls VAD: enabled|tts_playing

# Cross-platform temp directory for TTS output
if IS_WINDOWS:
    OUTPUT_DIR = Path(tempfile.gettempdir()) / "indextts2_audio"
else:
    OUTPUT_DIR = Path("/tmp/indextts2_audio")

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
CONVERSATIONS_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR.mkdir(exist_ok=True)

# Clean up leftover PTT trigger file from previous session
PTT_TRIGGER_FILE = RECORDINGS_DIR / "latest.txt"
if PTT_TRIGGER_FILE.exists():
    try:
        PTT_TRIGGER_FILE.unlink()
        print("[Startup] Cleaned up leftover PTT trigger file")
    except Exception as e:
        print(f"[Startup] Warning: Could not delete trigger file: {e}")

# Threading
processing_lock = Lock()
cleanup_counter = [0]

# ============================================================================
# LM Studio Configuration - Cross-Platform
# ============================================================================

def get_lm_studio_host() -> str:
    """Get LM Studio host IP based on platform"""
    # 1. Check for manual override in config.env (supports both HOST and full URL)
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    # Full URL override (highest priority)
                    if line.startswith("LM_STUDIO_URL="):
                        val = line.split("=", 1)[1].strip()
                        if val and not val.startswith("#"):
                            print(f"[LM Studio] Using manual URL from config: {val}")
                            # Return special marker; caller will use full URL
                            return f"URL:{val}"
                    # Host-only override
                    if line.startswith("LM_STUDIO_HOST="):
                        val = line.split("=", 1)[1].strip()
                        if val and not val.startswith("#"):
                            print(f"[LM Studio] Using manual host from config: {val}")
                            return val
    except Exception as e:
        print(f"[LM Studio] Error reading config: {e}")

    # 2. Auto-detect for WSL
    if IS_WSL:
        # Strategy A: Default Gateway (Most reliable for connectivity)
        try:
            import subprocess
            result = subprocess.run(['ip', 'route', 'show', 'default'], capture_output=True, text=True)
            if result.returncode == 0:
                # Output format: "default via 172.29.64.1 dev eth0 ..."
                parts = result.stdout.split()
                if 'via' in parts:
                    gateway_ip = parts[parts.index('via') + 1]
                    print(f"[LM Studio] WSL detected, Gateway IP: {gateway_ip}")
                    return gateway_ip
        except Exception as e:
            print(f"[LM Studio] Error getting gateway IP: {e}")

        # Strategy B: resolv.conf (Fallback)
        try:
            with open('/etc/resolv.conf', 'r') as f:
                for line in f:
                    if line.startswith('nameserver'):
                        host_ip = line.split()[1].strip()
                        print(f"[LM Studio] WSL detected, Nameserver IP: {host_ip}")
                        return host_ip
        except Exception as e:
            print(f"[LM Studio] Could not get Nameserver IP: {e}")
    
    # 3. Default: localhost
    return "localhost"

LM_STUDIO_HOST = get_lm_studio_host()
# Support full URL override via config.env (LM_STUDIO_URL=http://...)
if LM_STUDIO_HOST.startswith("URL:"):
    LM_STUDIO_BASE_URL = LM_STUDIO_HOST[4:]  # Strip "URL:" prefix
    LM_STUDIO_HOST = "custom"  # Mark as custom
else:
    LM_STUDIO_BASE_URL = f"http://{LM_STUDIO_HOST}:1235/v1"
LM_STUDIO_TIMEOUT = 60

# LM Studio state
LM_STUDIO_AVAILABLE = False
LM_STUDIO_MODEL = None

def check_lm_studio_available(force_refresh: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Check if LM Studio is running and what model is loaded.
    Implements smart fallback for WSL networking.
    Returns (is_available, model_name)
    """
    global LM_STUDIO_AVAILABLE, LM_STUDIO_MODEL, LM_STUDIO_BASE_URL
    
    # Candidate URLs to try
    urls_to_try = [LM_STUDIO_BASE_URL]
    
    # If in WSL, add the auto-detected nameserver as a fallback candidate
    if IS_WSL and "localhost" not in LM_STUDIO_BASE_URL:
        try:
            with open('/etc/resolv.conf', 'r') as f:
                for line in f:
                    if line.startswith('nameserver'):
                        wsl_host = line.split()[1].strip()
                        fallback_url = f"http://{wsl_host}:1235/v1"
                        if fallback_url != LM_STUDIO_BASE_URL:
                            urls_to_try.append(fallback_url)
        except (IOError, IndexError, OSError):
            pass  # resolv.conf unavailable or malformed
            
    for url in urls_to_try:
        try:
            response = requests.get(
                f"{url}/models",
                timeout=LM_STUDIO_TIMEOUT,
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            
            # If we get here, connection worked! Update global URL if it changed
            if url != LM_STUDIO_BASE_URL:
                print(f"[LM Studio] ⚠ Primary IP failed, but fallback worked: {url}")
                LM_STUDIO_BASE_URL = url
            
            models = data.get("data", [])
            if models:
                model_id = models[0].get("id", "unknown-model")
                print(f"[LM Studio] ✓ Connected! Model: {model_id}")
                LM_STUDIO_AVAILABLE = True
                LM_STUDIO_MODEL = model_id
                return True, model_id
            else:
                print("[LM Studio] ⚠ Server running but no model loaded")
                LM_STUDIO_AVAILABLE = True
                LM_STUDIO_MODEL = None
                return True, None
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            continue # Try next candidate
        except Exception as e:
            print(f"[LM Studio] ✗ Error connecting to {url}: {e}")
            continue

    # If all candidates fail
    print(f"[LM Studio] Could not connect. Tried: {', '.join(urls_to_try)}")
    print(f"[LM Studio] Troubleshooting:")
    print(f"  1. Ensure LM Studio is running on Windows")
    print(f"  2. In LM Studio: Settings > check 'Serve on Local Network' (binds to 0.0.0.0)")
    print(f"  3. Check Windows Firewall allows port 1235")
    print(f"  4. Or add LM_STUDIO_URL=http://YOUR_IP:1235/v1 to config.env")
    LM_STUDIO_AVAILABLE = False
    LM_STUDIO_MODEL = None
    return False, None

def get_lm_studio_status_display() -> str:
    """Get a user-friendly status display for LM Studio"""
    if LM_STUDIO_AVAILABLE and LM_STUDIO_MODEL:
        model_display = LM_STUDIO_MODEL
        if len(model_display) > 40:
            model_display = model_display[:37] + "..."
        return f"✓ {model_display}"
    elif LM_STUDIO_AVAILABLE:
        return "⚠️ No model loaded"
    else:
        return "✗ Not connected"

# ============================================================================
# Text Cleaning for TTS (Optimized for IndexTTS2)
# ============================================================================

# Track if we've warned about long text this session
_long_text_warning_shown = False

def clean_text_for_tts(text: str) -> Tuple[str, bool]:
    """
    Clean and optimize text for IndexTTS2 voice synthesis.
    
    Returns: (cleaned_text, was_long)
    - was_long: True if text was over recommended length (for UI warning)
    
    IndexTTS2 Best Practices:
    - Keep text natural and conversational
    - Under 1000 chars for best quality
    - Remove all formatting/markup
    - Remove action descriptions (*actions*) completely - they should NOT be spoken
    - Ensure proper punctuation for natural pacing
    """
    global _long_text_warning_shown
    
    if not text:
        return "", False
    
    original_length = len(text)
    
    # 1. CRITICAL: Remove action descriptions in asterisks COMPLETELY
    # These are roleplay actions like *sighs*, *thinks*, *looks around*
    # They should NOT be spoken - remove them entirely, not just the asterisks
    text = re.sub(r'\*[^*]+\*', '', text)  # Remove *action descriptions* completely
    
    # 2. Remove markdown formatting (keep the text content)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **Bold**
    text = re.sub(r'__([^_]+)__', r'\1', text)      # __Bold alt__
    text = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', text)  # _Italic alt_
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)  # Code blocks
    text = re.sub(r'`([^`]+)`', r'\1', text)        # `Inline code`
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)  # Headers
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [Links](url)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)    # Images
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # Bullet lists
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Numbered lists
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)  # Blockquotes
    text = re.sub(r'---+|===+|\*\*\*+', '', text)  # Horizontal rules
    
    # 3. Remove remaining problematic characters
    problematic_chars = ['*', '`', '=', '#', '|', '~', '^', '<', '>', '{', '}', '[', ']', '\\']
    for char in problematic_chars:
        text = text.replace(char, '')
    
    # 4. Convert symbols to spoken words for natural speech
    text = text.replace('&', ' and ')
    text = text.replace('%', ' percent')
    text = text.replace('+', ' plus ')
    text = text.replace('@', ' at ')
    text = text.replace('$', ' dollars ')
    text = text.replace('€', ' euros ')
    text = text.replace('£', ' pounds ')
    
    # 5. Handle ellipsis - normalize to three dots with space after
    text = re.sub(r'\.{2,}', '... ', text)
    
    # 6. Handle quotes - remove fancy quotes
    text = re.sub(r'["""]', '', text)
    text = re.sub(r"[''']", "'", text)
    
    # 7. Fix common TTS stumbling blocks
    # Em-dashes and en-dashes to regular hyphens (IndexTTS2 handles hyphens well)
    text = text.replace('—', ' - ')  # Em-dash
    text = text.replace('–', ' - ')  # En-dash
    text = text.replace('−', '-')    # Minus sign
    text = text.replace('‐', '-')    # Unicode hyphen
    text = text.replace('‑', '-')    # Non-breaking hyphen
    text = text.replace('⁃', '-')    # Hyphen bullet
    # Remove parenthetical asides or convert to natural phrasing
    text = re.sub(r'\s*\([^)]*\)\s*', ' ', text)
    
    # 8. Clean up whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*,\s*,\s*', ', ', text)  # Fix double commas
    text = text.strip()
    
    # 9. Ensure proper sentence ending for natural completion
    if text and text[-1] not in '.!?':
        text += '.'
    
    # 10. Final cleanup - remove any leading/trailing punctuation oddities
    text = re.sub(r'^[\s,;:]+', '', text)
    text = re.sub(r'[\s,;:]+$', '.', text)
    
    # 11. Check if text is long (warn but don't truncate)
    # IndexTTS2 handles up to ~1000 chars well
    was_long = len(text) > 800
    if was_long and not _long_text_warning_shown:
        print(f"[TTS] Long response ({len(text)} chars) - synthesis may take longer")
        _long_text_warning_shown = True
    
    return text, was_long

# ============================================================================
# Settings Persistence
# ============================================================================

DEFAULT_SETTINGS = {
    "model": "x-ai/grok-4.1-fast",
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "last_character": "hermione",
    "last_voice": "reference.wav",
    "auto_play": True,
    "tts_enabled": True,
    "tts_backend": "indextts",
    "current_conversation_id": None,
    "llm_provider": "openrouter",
    "emotion_detection_enabled": True,  # Can disable to free resources when using LM Studio
    "stt_backend": "faster_whisper",  # Options: faster_whisper, sensevoice, funasr
    "stt_device": "cuda:0",  # Device for SenseVoice/FunASR
    "use_sensevoice_emotion": True,  # Use built-in emotion when available
    "use_sensevoice_vad": True,  # Use built-in VAD when available
}

def load_settings() -> dict:
    """Load persistent settings"""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                return {**DEFAULT_SETTINGS, **saved}
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"[Settings] Failed to load settings, using defaults: {e}")
    return DEFAULT_SETTINGS.copy()

def save_settings(settings: dict):
    """Save settings to disk"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"[Settings] Error saving: {e}")

SETTINGS = load_settings()

# ============================================================================
# Conversation History Management
# ============================================================================

def get_conversations_dir(character_id: str) -> Path:
    """Get the conversations directory for a character"""
    char_dir = CONVERSATIONS_DIR / character_id
    char_dir.mkdir(exist_ok=True)
    return char_dir

def generate_conversation_id() -> str:
    """Generate a unique conversation ID"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def list_conversations(character_id: str) -> List[Dict[str, Any]]:
    """List all conversations for a character, sorted by date (newest first)"""
    char_dir = get_conversations_dir(character_id)
    conversations = []
    
    for f in char_dir.glob("*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                conversations.append({
                    'id': f.stem,
                    'title': data.get('title', 'Untitled'),
                    'preview': data.get('preview', ''),
                    'message_count': len(data.get('history', [])),
                    'updated_at': data.get('updated_at', f.stat().st_mtime),
                    'created_at': data.get('created_at', f.stat().st_ctime)
                })
        except Exception as e:
            print(f"[Conversations] Error reading {f}: {e}")
    
    conversations.sort(key=lambda x: x['updated_at'], reverse=True)
    return conversations

def save_conversation(character_id: str, conversation_id: str, history: list, title: str = None):
    """Save a conversation to disk"""
    char_dir = get_conversations_dir(character_id)
    filepath = char_dir / f"{conversation_id}.json"
    
    if not title and history:
        # Handle messages format: [{"role": "user", "content": "..."}, ...]
        first_item = history[0]
        if isinstance(first_item, dict):
            first_msg = first_item.get('content', '')
        else:
            # Legacy tuple format
            first_msg = first_item[0] if first_item[0] else first_item[1]
        title = first_msg[:50] + "..." if len(first_msg) > 50 else first_msg
    
    preview = ""
    if history:
        # Find last user message for preview
        last_item = history[-1]
        if isinstance(last_item, dict):
            # Look for last user message
            for msg in reversed(history):
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    preview = msg.get('content', '')[:100]
                    break
        else:
            # Legacy tuple format
            last_user = last_item[0] or ""
            preview = last_user[:100]
    
    created_at = datetime.now().isoformat()
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                created_at = existing.get('created_at', created_at)
        except (json.JSONDecodeError, IOError, KeyError):
            pass  # Existing conversation file corrupted or unreadable
    
    data = {
        'character_id': character_id,
        'conversation_id': conversation_id,
        'title': title or "Untitled",
        'preview': preview,
        'history': history,
        'created_at': created_at,
        'updated_at': datetime.now().isoformat()
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Conversations] Error saving: {e}")

def load_session(character_id: str) -> list:
    """Load chat session from file, converting legacy tuples to messages format"""
    # 1. Look for character-specific session
    filename = f"{character_id}_session.json"
    filepath = SESSIONS_DIR / filename
    
    # 2. If not found, look for default session
    if not filepath.exists():
        filepath = SESSIONS_DIR / "session.json"
    
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not data:
                    return []
                
                # Handle dict wrapper format (new save_session format)
                if isinstance(data, dict) and 'history' in data:
                    history = data['history']
                    if not history:
                        return []
                    data = history
                
                # Ensure data is a list
                if not isinstance(data, list):
                    print(f"[Session] Invalid data type: {type(data)}, returning empty")
                    return []
                
                if len(data) == 0:
                    return []
                
                # Check format and convert if necessary
                first_item = data[0]
                
                # Legacy: [[user, bot], ...]
                if isinstance(first_item, (list, tuple)):
                    print(f"[Session] Converting legacy session for {character_id}")
                    new_history = []
                    for turn in data:
                        if len(turn) >= 1 and turn[0]:
                            new_history.append({"role": "user", "content": str(turn[0])})
                        if len(turn) >= 2 and turn[1]:
                            new_history.append({"role": "assistant", "content": str(turn[1])})
                    print(f"[Session] Converted {len(new_history)} messages")
                    return new_history
                
                # Messages format - validate structure
                if isinstance(first_item, dict):
                    validated = []
                    for msg in data:
                        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                            # Ensure content is a string
                            validated.append({
                                "role": str(msg['role']),
                                "content": str(msg['content']) if msg['content'] else ""
                            })
                        else:
                            print(f"[Session] Skipping invalid message: {msg}")
                    print(f"[Session] Loaded {len(validated)} messages for {character_id}")
                    return validated
                
                # Unknown format - return empty
                print(f"[Session] Unknown format, first item type: {type(first_item)}")
                return []
                    
        except Exception as e:
            print(f"[Session] Error converting/loading: {e}")
            import traceback
            traceback.print_exc()
            return []
            
    return []

def load_conversation(character_id: str, conversation_id: str) -> Tuple[list, str]:
    """Load a conversation from disk. Returns (history, title)"""
    char_dir = get_conversations_dir(character_id)
    filepath = char_dir / f"{conversation_id}.json"
    
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                history = data.get('history', [])
                title = data.get('title', 'Untitled')
                
                if not history:
                    return [], title
                
                # Convert and validate history format
                first_item = history[0]
                
                # Legacy tuple format: [[user, bot], ...]
                if isinstance(first_item, (list, tuple)):
                    print(f"[Conversations] Converting legacy format for {conversation_id}")
                    new_history = []
                    for turn in history:
                        if len(turn) >= 1 and turn[0]:
                            new_history.append({"role": "user", "content": str(turn[0])})
                        if len(turn) >= 2 and turn[1]:
                            new_history.append({"role": "assistant", "content": str(turn[1])})
                    return new_history, title
                
                # Messages format - validate structure
                if isinstance(first_item, dict):
                    validated = []
                    for msg in history:
                        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                            validated.append({
                                "role": str(msg['role']),
                                "content": str(msg['content']) if msg['content'] else ""
                            })
                    return validated, title
                
                return [], title
        except Exception as e:
            print(f"[Conversations] Error loading: {e}")
    return [], "Untitled"

def delete_conversation(character_id: str, conversation_id: str) -> bool:
    """Delete a conversation"""
    char_dir = get_conversations_dir(character_id)
    filepath = char_dir / f"{conversation_id}.json"
    
    if filepath.exists():
        try:
            filepath.unlink()
            return True
        except Exception as e:
            print(f"[Conversations] Error deleting: {e}")
    return False

def export_conversation(character_id: str, conversation_id: str, format: str = "txt") -> Tuple[Optional[str], Optional[str]]:
    """
    Export a conversation to a file.
    Returns (file_content, filename) or (None, None) on error.
    """
    history, title = load_conversation(character_id, conversation_id)
    if not history:
        return None, None
    
    character = CHARACTER_MANAGER.get_character(character_id)
    char_name = character.name if character else character_id
    
    safe_title = re.sub(r'[^\w\s-]', '', title)[:30].strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert messages format to paired format for export
    # Messages format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    paired_history = []
    i = 0
    while i < len(history):
        msg = history[i]
        if isinstance(msg, dict):
            if msg.get('role') == 'user':
                user_content = msg.get('content', '')
                assistant_content = ''
                # Look for next assistant message
                if i + 1 < len(history):
                    next_msg = history[i + 1]
                    if isinstance(next_msg, dict) and next_msg.get('role') == 'assistant':
                        assistant_content = next_msg.get('content', '')
                        i += 1
                paired_history.append((user_content, assistant_content))
            i += 1
        else:
            # Legacy tuple format
            paired_history.append((msg[0] if len(msg) > 0 else '', msg[1] if len(msg) > 1 else ''))
            i += 1
    
    if format == "json":
        content = json.dumps({
            'character': char_name,
            'character_id': character_id,
            'title': title,
            'exported_at': datetime.now().isoformat(),
            'messages': [
                {'role': 'user', 'content': h[0], 'response': h[1]}
                for h in paired_history
            ]
        }, indent=2, ensure_ascii=False)
        filename = f"{safe_title}_{timestamp}.json"
    else:  # txt format
        lines = [
            f"Conversation with {char_name}",
            f"Title: {title}",
            f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            ""
        ]
        for i, (user_msg, assistant_msg) in enumerate(paired_history, 1):
            lines.append(f"[{i}] You: {user_msg}")
            lines.append(f"    {char_name}: {assistant_msg}")
            lines.append("")
        content = "\n".join(lines)
        filename = f"{safe_title}_{timestamp}.txt"
    
    return content, filename

def get_conversation_choices(character_id: str) -> list:
    """Get conversation choices for dropdown"""
    conversations = list_conversations(character_id)
    choices = [("➕ New Conversation", "new")]
    
    for conv in conversations:
        date_str = conv['updated_at'][:10] if isinstance(conv['updated_at'], str) else ""
        label = f"{conv['title'][:30]} ({conv['message_count']} msgs) - {date_str}"
        choices.append((label, conv['id']))
    
    return choices

# ============================================================================
# OpenRouter Model Fetching
# ============================================================================

CACHED_MODELS = []
POPULAR_MODELS = [
    "x-ai/grok-4.1-fast",
    "x-ai/grok-4.1",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large-2411",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct",
]

VISION_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.0-flash-exp:free",
    "x-ai/grok-4.1",
    "x-ai/grok-4.1-fast",
]

def fetch_openrouter_models(api_key: str) -> list:
    """Fetch available models from OpenRouter"""
    global CACHED_MODELS
    
    if CACHED_MODELS:
        return CACHED_MODELS
    
    try:
        print("[Models] Fetching from OpenRouter...")
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        models = []
        for m in data.get("data", []):
            model_id = m.get("id", "")
            if model_id and not any(x in model_id.lower() for x in ["embedding", "image", "dall"]):
                pricing = m.get("pricing", {})
                prompt_price = float(pricing.get("prompt", 0)) * 1000000
                name = m.get("name", model_id)
                context = m.get("context_length", 0)
                
                vision_marker = " 👁️" if model_id in VISION_MODELS else ""
                
                if context > 0:
                    display = f"{name}{vision_marker} ({context//1000}k) - ${prompt_price:.2f}/1M"
                else:
                    display = f"{name}{vision_marker} - ${prompt_price:.2f}/1M"
                
                models.append((display, model_id))
        
        def sort_key(m):
            model_id = m[1]
            for i, pop in enumerate(POPULAR_MODELS):
                if model_id == pop:
                    return (0, i)
            return (1, m[0])
        
        models.sort(key=sort_key)
        CACHED_MODELS = models
        print(f"[Models] Loaded {len(models)} models")
        return models
        
    except Exception as e:
        print(f"[Models] Error fetching: {e}")
        return [(m, m) for m in POPULAR_MODELS]

def refresh_models(api_key: str):
    """Force refresh model list"""
    global CACHED_MODELS
    CACHED_MODELS = []
    return gr.update(choices=fetch_openrouter_models(api_key))

# ============================================================================
# Voice Reference Management
# ============================================================================

# Voice emoji keyword mapping (shared with character_manager_ui.py)
VOICE_EMOJI_MAP = {
    "emma": "🧙‍♀️", "watson": "🧙‍♀️", "hermione": "🧙‍♀️",
    "gandalf": "🧙", "wizard": "🧙",
    "rooney": "🖤", "mara": "🖤", "lisbeth": "🖤",
    "samantha": "💝", "her": "💝",
    "male": "🎤", "female": "🎵", "deep": "🔊", "soft": "🌸",
    "reference": "🎭", "custom": "🎤", "me": "🎤", "maya": "🌸",
}

def get_voice_metadata_path() -> Path:
    """Get path to voice metadata file"""
    return VOICE_REF_DIR / "voice_metadata.json"

def load_voice_metadata() -> dict:
    """Load voice metadata (custom emojis, etc.)"""
    meta_path = get_voice_metadata_path()
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass  # Metadata file corrupted or unreadable
    return {}

def get_emoji_for_voice(filename: str) -> str:
    """Get emoji for voice based on name keywords or metadata"""
    if not filename:
        return "🔊"
    
    # First check metadata for custom emoji
    metadata = load_voice_metadata()
    if filename in metadata and "emoji" in metadata[filename]:
        return metadata[filename]["emoji"]
    
    # Auto-assign based on keywords
    name_lower = Path(filename).stem.lower()
    for keyword, emoji in VOICE_EMOJI_MAP.items():
        if keyword in name_lower:
            return emoji
    
    # Fallback
    return "🔊"

def get_available_voices() -> list:
    """Get all available voice reference files"""
    VOICE_REF_DIR.mkdir(exist_ok=True)
    voices = []
    for f in VOICE_REF_DIR.glob("*.wav"):
        voices.append(f.name)
    voices.sort()
    return voices

def get_voice_path(voice_name: str) -> str:
    """Get full path to a voice file"""
    return str(VOICE_REF_DIR / voice_name)

def get_voice_display_name(voice_file: str) -> str:
    """Convert filename to display name with emoji"""
    name = Path(voice_file).stem
    emoji = get_emoji_for_voice(voice_file)
    display = name.title().replace('_', ' ')
    return f"{emoji} {display}"

def refresh_voice_choices() -> list:
    """Refresh voice choices list (call when voices are added/removed)"""
    global AVAILABLE_VOICES, VOICE_CHOICES
    AVAILABLE_VOICES = get_available_voices()
    VOICE_CHOICES = [(get_voice_display_name(v), v) for v in AVAILABLE_VOICES]
    return VOICE_CHOICES

# Supertonic preset voices (no .wav files needed)
SUPERTONIC_VOICES = [
    ("🎤 Male 1", "male_1"),
    ("🎤 Male 2", "male_2"),
    ("🎤 Male 3", "male_3"),
    ("🎤 Female 1", "female_1"),
    ("🎤 Female 2", "female_2"),
    ("🎤 Female 3", "female_3"),
]

SOPRANO_VOICES = [
    ("🎵 Default", "default"),
]

def get_voice_choices_for_backend(backend: str = None) -> tuple:
    """Get voice choices appropriate for the TTS backend.

    Args:
        backend: TTS backend name (indextts, kokoro, supertonic)

    Returns:
        Tuple of (choices_list, default_value)
    """
    if backend is None:
        backend = SETTINGS.get("tts_backend", "indextts")

    if backend == "supertonic":
        # Supertonic uses preset voices, not .wav files
        return SUPERTONIC_VOICES, "female_1"

    elif backend == "soprano":
        # Soprano has a single default voice (no cloning yet)
        return SOPRANO_VOICES, "default"

    elif backend == "kokoro":
        # Kokoro only works well with af_, am_, bf_, bm_ prefixed voices
        kokoro_voices = [
            (get_voice_display_name(v), v) for v in AVAILABLE_VOICES
            if v.startswith(('af_', 'am_', 'bf_', 'bm_'))
        ]
        if kokoro_voices:
            # Validate last_voice is in kokoro_voices
            last = SETTINGS.get("last_voice", kokoro_voices[0][1])
            valid_values = [v[1] for v in kokoro_voices]
            if last not in valid_values:
                last = kokoro_voices[0][1]
            return kokoro_voices, last
        return VOICE_CHOICES, SETTINGS.get("last_voice", "reference.wav")

    else:  # indextts - supports all voices for cloning
        # Validate last_voice is in VOICE_CHOICES
        last = SETTINGS.get("last_voice", "reference.wav")
        valid_values = [v[1] for v in VOICE_CHOICES]
        if last not in valid_values:
            last = "reference.wav"
        return VOICE_CHOICES, last

AVAILABLE_VOICES = get_available_voices()
VOICE_CHOICES = [(get_voice_display_name(v), v) for v in AVAILABLE_VOICES]

# Show filtered voices for default TTS backend
_default_backend = SETTINGS.get("tts_backend", "indextts")
_filtered_voices, _ = get_voice_choices_for_backend(_default_backend)
print(f"[Voices] {_default_backend}: {len(_filtered_voices)} voices available")

# ============================================================================
# GPU Memory Management
# ============================================================================

def get_gpu_memory_info():
    if not torch.cuda.is_available():
        return None
    allocated = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return {
        'allocated': allocated,
        'total': total,
        'percentage': (allocated / total) * 100
    }

def clear_cuda_memory(aggressive=False):
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if aggressive:
        import gc
        gc.collect()
        for _ in range(3):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def cleanup_old_recordings(max_age_hours: int = 1, max_files: int = 5):
    """Clean up old recording files to prevent disk bloat

    Deletes recordings that are:
    - Older than max_age_hours, OR
    - Beyond the max_files limit (keeps newest)
    Also cleans up orphan trigger files.
    """
    try:
        recordings = list(RECORDINGS_DIR.glob("rec_*.wav"))

        deleted = 0
        cutoff_time = time.time() - (max_age_hours * 3600)

        if recordings:
            # Sort by modification time (newest first)
            recordings.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            for i, rec_file in enumerate(recordings):
                try:
                    file_time = rec_file.stat().st_mtime
                    # Delete if: beyond max_files limit OR older than cutoff
                    if i >= max_files or file_time < cutoff_time:
                        rec_file.unlink()
                        deleted += 1
                except Exception:
                    pass  # File may have been deleted by another process

        # Also clean up trigger file if it's orphaned (no recent recordings)
        trigger_file = RECORDINGS_DIR / "latest.txt"
        if trigger_file.exists():
            try:
                # If there are no recent recordings, the trigger is orphaned
                if not recordings or trigger_file.stat().st_mtime < cutoff_time:
                    trigger_file.unlink()
                    deleted += 1
            except Exception:
                pass

        if deleted > 0:
            print(f"[Cleanup] Deleted {deleted} old files")
        return deleted
    except Exception as e:
        print(f"[Cleanup] Error: {e}")
        return 0

# ============================================================================
# VAD Control File - Communication with vad_windows.py
# ============================================================================

def update_vad_control(enabled: bool, tts_playing: bool = False):
    """
    Update the VAD control file to communicate state to vad_windows.py.
    
    Format: enabled|tts_playing
    - enabled: 1 if hands-free mode is on, 0 if off
    - tts_playing: 1 if TTS is currently playing, 0 if not
    
    VAD should only record when: enabled=1 AND tts_playing=0
    """
    try:
        RECORDINGS_DIR.mkdir(exist_ok=True)
        content = f"{1 if enabled else 0}|{1 if tts_playing else 0}"
        VAD_CONTROL_FILE.write_text(content)
    except Exception as e:
        print(f"[VAD] Error writing control file: {e}")

# ============================================================================
# Configuration Loading
# ============================================================================

def load_config():
    config = {
        "OPENROUTER_API_KEY": "",
        "OPENROUTER_MODEL": "x-ai/grok-4.1-fast",
        "MAX_TOKENS": 2000,
        "TEMPERATURE": 0.7,
        "USE_FP16": True,
        "SERVER_PORT": 7861,
        "ENABLE_FULL_FILE_ACCESS": False,
    }
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key, value = key.strip(), value.strip()
                    if key in config:
                        if isinstance(config[key], bool):
                            config[key] = value.lower() in ("true", "1", "yes")
                        elif isinstance(config[key], int):
                            config[key] = int(value)
                        elif isinstance(config[key], float):
                            config[key] = float(value)
                        else:
                            config[key] = value
    return config

def find_free_port(start_port: int, max_attempts: int = 100) -> int:
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise Exception("No free ports found")

CONFIG = load_config()
SERVER_PORT = find_free_port(CONFIG['SERVER_PORT'])

# Mobile/Remote access mode - enables Gradio share for HTTPS access
SHARE_MODE = os.environ.get("SHARE_MODE", "").lower() in ("1", "true", "yes")
if SHARE_MODE:
    print("[Config] Share Mode: ENABLED (remote/mobile access)")

print(f"[Config] API Key: {'✓' if CONFIG['OPENROUTER_API_KEY'] else '✗'}")
print(f"[Config] Server Port: {SERVER_PORT}")

# Check LM Studio status at startup
LM_STUDIO_AVAILABLE, LM_STUDIO_MODEL = check_lm_studio_available()


# ============================================================================
# Initialize Systems
# ============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
TTS_MODEL = None

def init_tts():
    """Initialize TTS model with graceful fallback"""
    global TTS_MODEL

    # === SUPERTONIC BACKEND (Ultra-fast ONNX) ===
    if SETTINGS.get("tts_backend") == "supertonic":
        try:
            from audio.backends.supertonic import SupertonicBackend
            # Clear any existing model
            if TTS_MODEL is not None and not isinstance(TTS_MODEL, SupertonicBackend):
                print("[TTS] Switching to Supertonic...")
                clear_cuda_memory()

            if TTS_MODEL is None or not isinstance(TTS_MODEL, SupertonicBackend):
                TTS_MODEL = SupertonicBackend(device="cpu")
                print("[TTS] Supertonic loaded on CPU (167x realtime)")
            return TTS_MODEL

        except ImportError:
            print("[TTS] Supertonic not available, falling back to IndexTTS2")
            SETTINGS["tts_backend"] = "indextts"
            # Fall through to IndexTTS2

    # === SOPRANO BACKEND (Ultra-fast GPU) ===
    if SETTINGS.get("tts_backend") == "soprano":
        try:
            from audio.backends.soprano import SopranoBackend
            # Clear any existing model
            if TTS_MODEL is not None and not isinstance(TTS_MODEL, SopranoBackend):
                print("[TTS] Switching to Soprano...")
                clear_cuda_memory()

            if TTS_MODEL is None or not isinstance(TTS_MODEL, SopranoBackend):
                TTS_MODEL = SopranoBackend(device="cuda")
                print("[TTS] Soprano loaded on CUDA (2000x realtime)")
            return TTS_MODEL

        except ImportError:
            print("[TTS] Soprano not available, falling back to IndexTTS2")
            SETTINGS["tts_backend"] = "indextts"
            # Fall through to IndexTTS2

    # 1. Kokoro Backend
    if SETTINGS.get("tts_backend", "indextts") == "kokoro":
        if not KOKORO_AVAILABLE:
            print("[TTS] Kokoro not available (install kokoro-onnx soundfile)")
            return None
            
        # Re-init if switching backends
        if TTS_MODEL is not None and not isinstance(TTS_MODEL, KokoroTTS):
            print("[TTS] Switching to Kokoro...")
            TTS_MODEL = None
            clear_cuda_memory()
            
        if TTS_MODEL is None:
            print("[TTS] Initializing Kokoro...")
            model_dir = paths.MODELS_DIR
            try:
                TTS_MODEL = KokoroTTS(checkpoints_dir=str(model_dir))
                print("[TTS] Kokoro initialized")
            except Exception as e:
                print(f"[TTS] Failed to initialize Kokoro: {e}")
                return None
        return TTS_MODEL

    # 2. IndexTTS2 Backend (Default)
    else:
        if not TTS_AVAILABLE:
            print("[TTS] IndexTTS2 not available - voice synthesis disabled")
            return None
        
        # Re-init if switching backends
        is_kokoro = False
        try:
             is_kokoro = isinstance(TTS_MODEL, KokoroTTS)
        except (NameError, TypeError):
            pass  # KokoroTTS may not be imported or TTS_MODEL is None
        
        if TTS_MODEL is not None and is_kokoro:
            print("[TTS] Switching to IndexTTS2...")
            TTS_MODEL = None
            clear_cuda_memory()
        
        if TTS_MODEL is None:
            print("[TTS] Initializing IndexTTS2...")
            clear_cuda_memory(aggressive=True)
            model_dir = paths.INDEXTTS_MODELS_DIR

            if not model_dir.exists():
                print(f"[TTS] ERROR: Checkpoints not found at {model_dir}")
                print("[TTS] Please download IndexTTS2 model files")
                return None

            try:
                TTS_MODEL = IndexTTS2(
                    cfg_path=str(model_dir / "config.yaml"),
                    model_dir=str(model_dir),
                    use_fp16=CONFIG["USE_FP16"],
                    use_cuda_kernel=True,
                    use_deepspeed=False,
                    use_torch_compile=False,  # Disabled - causes audio corruption on some systems
                )
                print("[TTS] IndexTTS2 initialized")
            except Exception as e:
                print(f"[TTS] Failed to initialize: {e}")
                return None
        return TTS_MODEL

# STT Model
WHISPER_MODEL = None
WHISPER_MODEL_SIZE = "base"
STT_MODEL = None
STT_BACKEND_NAME = None

def init_stt():
    """
    Initialize STT backend based on settings.

    Supports:
    - faster_whisper: CPU-based, default fallback
    - sensevoice: CUDA, unified STT+Emotion+VAD
    - funasr: CUDA, high-accuracy STT
    """
    global STT_MODEL, STT_BACKEND_NAME, WHISPER_MODEL

    backend = SETTINGS.get("stt_backend", "faster_whisper")

    # Return cached if same backend
    if STT_MODEL is not None and STT_BACKEND_NAME == backend:
        return STT_MODEL

    # Clear old model
    if STT_MODEL is not None:
        STT_MODEL = None
        STT_BACKEND_NAME = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        if backend == "sensevoice":
            from audio.backends.sensevoice import SenseVoiceBackend
            STT_MODEL = SenseVoiceBackend(device="cuda:0", use_vad=True)
            STT_BACKEND_NAME = "sensevoice"
            print("[STT] SenseVoice loaded on CUDA (unified STT+Emotion+VAD)")
            return STT_MODEL

        elif backend == "funasr":
            from audio.backends.funasr_backend import FunASRBackend
            STT_MODEL = FunASRBackend(device="cuda:0", model_variant="zh")
            STT_BACKEND_NAME = "funasr"
            print("[STT] FunASR Paraformer loaded on CUDA (high-accuracy)")
            return STT_MODEL

        else:  # faster_whisper (default)
            # Use existing init_whisper logic
            model = init_whisper()
            if model:
                STT_MODEL = model
                STT_BACKEND_NAME = "faster_whisper"
            return STT_MODEL

    except ImportError as e:
        print(f"[STT] {backend} not available: {e}")
        print("[STT] Falling back to faster-whisper...")
        SETTINGS["stt_backend"] = "faster_whisper"
        model = init_whisper()
        if model:
            STT_MODEL = model
            STT_BACKEND_NAME = "faster_whisper"
        return STT_MODEL

def init_whisper():
    """Initialize Whisper model (lazy loading) - runs on CPU to avoid VRAM conflicts with TTS"""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
            print(f"[STT] Loading Whisper model ({WHISPER_MODEL_SIZE})...")
            
            # Keep on CPU to avoid VRAM conflicts with IndexTTS2
            # IndexTTS2 needs most of the GPU memory for quality inference
            device = "cpu"
            compute_type = "int8"  # INT8 is fast and efficient on CPU
            
            WHISPER_MODEL = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=device,
                compute_type=compute_type
            )
            print(f"[STT] Whisper loaded on {device}")
        except ImportError:
            print("[STT] faster-whisper not installed, trying openai-whisper...")
            try:
                import whisper
                WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_SIZE)
                print(f"[STT] OpenAI Whisper loaded ({WHISPER_MODEL_SIZE})")
            except ImportError:
                print("[STT] ERROR: No whisper package installed!")
                print("[STT] Install with: pip install faster-whisper")
                return None
    return WHISPER_MODEL

def get_stt_status() -> str:
    """Get current STT status for UI display"""
    if WHISPER_MODEL is not None:
        return f"✓ Whisper ({WHISPER_MODEL_SIZE}) CPU"
    elif init_whisper() is not None:
        return f"✓ Whisper ({WHISPER_MODEL_SIZE}) CPU"
    else:
        return "✗ Whisper not loaded"

def get_tts_status() -> str:
    """Get current TTS status for UI display"""
    # Supertonic backend
    if SETTINGS.get("tts_backend") == "supertonic":
        try:
            from audio.backends.supertonic import SupertonicBackend
            if TTS_MODEL is not None and isinstance(TTS_MODEL, SupertonicBackend):
                return "✓ Supertonic (CPU/ONNX 167x)"
            return "⏳ Supertonic loading..."
        except ImportError:
            return "✗ Supertonic not installed"

    # Kokoro backend
    if SETTINGS.get("tts_backend") == "kokoro":
        if not KOKORO_AVAILABLE: return "✗ Kokoro not installed"
        return "✓ Kokoro (CPU/ONNX)" if TTS_MODEL else "⏳ Kokoro loading..."

    # IndexTTS2 backend (default)
    if not TTS_AVAILABLE:
        return "✗ IndexTTS2 not installed"
    elif TTS_MODEL is not None:
        return "✓ IndexTTS2 (GPU)"
    else:
        return "⏳ IndexTTS2 not initialized"

# Memory System
print("[Memory] Initializing memory system...")

def llm_client_for_memory(prompt: str, temperature: float = 0.7) -> str:
    """Simple LLM wrapper for memory system graph extraction.

    Uses the currently selected provider (OpenRouter or LM Studio).
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        provider = SETTINGS.get("llm_provider", "openrouter")

        if provider == "lmstudio" and LM_STUDIO_AVAILABLE:
            response = chat_with_lm_studio(
                messages=messages,
                temperature=temperature,
                max_tokens=1000,
                top_p=1.0,
                freq_penalty=0.0,
                pres_penalty=0.0
            )
        else:
            model = CONFIG.get("OPENROUTER_MODEL", "x-ai/grok-4.1-fast")
            response = chat_with_openrouter(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=1000,
                top_p=1.0,
                freq_penalty=0.0,
                pres_penalty=0.0
            )
        return response.get("content", "")
    except Exception as e:
        print(f"[Memory LLM] Error: {e}")
        return ""

MEMORY_MANAGER = create_memory_manager(use_local=True, llm_client=llm_client_for_memory)
CHARACTER_MANAGER = create_character_manager()

# Initialize Tools
print("[Tools] Initializing tool registry...")
init_tools(
    sandbox_path=str(SESSIONS_DIR / "files"),
    enable_full_file_access=CONFIG.get("ENABLE_FULL_FILE_ACCESS", False)
)
# Connect graph tool to memory manager
set_graph_tool_memory_manager(MEMORY_MANAGER)

# Initialize MCP Manager
MCP_MANAGER = None

# ============================================================================
# Service Container Initialization (Phase 1: Architecture Cleanup)
# ============================================================================
print("[ServiceContainer] Initializing dependency injection container...")
service_container = ServiceContainer.get_instance()
service_container.initialize(
    memory_manager=MEMORY_MANAGER,
    character_manager=CHARACTER_MANAGER,
    tts_model=TTS_MODEL,
    settings=SETTINGS,
    whisper_model=WHISPER_MODEL,
    vad_manager=vad_manager_instance
)
print("[ServiceContainer] ✓ Initialized with all services")
MCP_TOOLS_CACHE = []
MCP_LOOP = None
MCP_THREAD = None

def _run_mcp_loop(loop):
    """Run the asyncio event loop forever in a background thread"""
    asyncio.set_event_loop(loop)
    loop.run_forever()

def init_mcp():
    """Initialize MCP manager and connect to configured servers in a background thread"""
    global MCP_MANAGER, MCP_TOOLS_CACHE, MCP_LOOP, MCP_THREAD

    if not MCP_AVAILABLE:
        print("[MCP] Package not installed - MCP tools disabled")
        return False

    try:
        MCP_MANAGER = MCPManager()
        
        # Create a persistent loop and thread for MCP operations
        MCP_LOOP = asyncio.new_event_loop()
        MCP_THREAD = Thread(target=_run_mcp_loop, args=(MCP_LOOP,), daemon=True)
        MCP_THREAD.start()

        # Run async initialization in the background loop
        init_future = asyncio.run_coroutine_threadsafe(MCP_MANAGER.initialize(), MCP_LOOP)
        init_future.result(timeout=30) # Wait for init to complete

        # Cache tools
        list_future = asyncio.run_coroutine_threadsafe(MCP_MANAGER.list_tools(), MCP_LOOP)
        MCP_TOOLS_CACHE = list_future.result(timeout=10)

        if MCP_TOOLS_CACHE:
            print(f"[MCP] Connected - {len(MCP_TOOLS_CACHE)} tools available")
            return True
        else:
            print("[MCP] Connected but no tools found")
            return True
    except Exception as e:
        print(f"[MCP] Initialization failed: {e}")
        return False


def execute_tool_with_mcp_fallback(tool_call: dict) -> str:
    """Execute tool, falling back to MCP if not in local registry"""
    global MCP_MANAGER, MCP_LOOP

    func_name = tool_call.get('function', {}).get('name')

    # Try local registry first
    local_tool = REGISTRY.get_tool(func_name)
    if local_tool:
        return REGISTRY.execute(tool_call)

    # Fallback to MCP
    if MCP_MANAGER and MCP_AVAILABLE and MCP_LOOP:
        try:
            args_str = tool_call.get('function', {}).get('arguments', '{}')
            if isinstance(args_str, str):
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = args_str

            # Execute tool in the persistent MCP loop
            future = asyncio.run_coroutine_threadsafe(
                MCP_MANAGER.call_tool(func_name, args), 
                MCP_LOOP
            )
            return future.result(timeout=20)
        except Exception as e:
            return f"MCP tool error: {e}"

    return f"Unknown tool: {func_name}"


def get_combined_tools_schema(allowed_tools: list = None, include_mcp: bool = True) -> list:
    """Get combined tool schemas from local registry and MCP

    Args:
        allowed_tools: List of allowed local tool names (None means allow all)
        include_mcp: Whether to include MCP tools (for progressive disclosure)

    Returns:
        List of tool schemas
    """
    schemas = []

    # Local tools
    local_schemas = REGISTRY.list_tools(allowed_tools)
    schemas.extend(local_schemas)

    # MCP tools - include if requested
    # Characters with allowed_tools=None (no restrictions) also get MCP tools
    # MCP tools are configured separately via mcp_config.json
    if MCP_TOOLS_CACHE and include_mcp:
        for mcp_tool in MCP_TOOLS_CACHE:
            # Avoid duplicates (MCP tool might override a local tool name)
            if not any(s.get('function', {}).get('name') == mcp_tool.get('function', {}).get('name') for s in schemas):
                schemas.append(mcp_tool)

    return schemas


# ============================================================================
# Skill Context Detection
# ============================================================================

SKILLS_DIR = SCRIPT_DIR / "skills"
SKILLS_CACHE = []

def load_skills_cache():
    """Load all available skills for context matching"""
    global SKILLS_CACHE
    SKILLS_CACHE = []

    if not SKILLS_DIR.exists():
        return

    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue

        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue

        try:
            content = skill_file.read_text(encoding='utf-8')

            # Parse frontmatter
            import re
            pattern = r'^---\s*\n(.*?)\n---\s*\n'
            match = re.match(pattern, content, re.DOTALL)

            if match:
                try:
                    import yaml
                    frontmatter = yaml.safe_load(match.group(1)) or {}
                except (yaml.YAMLError, AttributeError, TypeError):
                    frontmatter = {}
            else:
                frontmatter = {}

            # Extract keywords from description
            description = frontmatter.get('description', '')
            keywords = set()

            # Add words from description
            for word in re.findall(r'\b\w{4,}\b', description.lower()):
                keywords.add(word)

            # Add skill name
            keywords.add(skill_dir.name.lower())

            SKILLS_CACHE.append({
                'id': skill_dir.name,
                'name': frontmatter.get('name', skill_dir.name),
                'description': description,
                'keywords': keywords,
                'content': content,
                'allowed_tools': frontmatter.get('allowed_tools', [])
            })

        except Exception as e:
            print(f"[Skills] Error loading {skill_dir.name}: {e}")

    if SKILLS_CACHE:
        print(f"[Skills] Loaded {len(SKILLS_CACHE)} skills for context detection")


def detect_relevant_skills(user_message: str, top_k: int = 2) -> list:
    """
    Detect which skills are relevant to the user's message.
    Returns list of (skill, score) tuples.
    """
    if not SKILLS_CACHE:
        return []

    message_lower = user_message.lower()
    message_words = set(re.findall(r'\b\w{4,}\b', message_lower))

    scored_skills = []
    for skill in SKILLS_CACHE:
        # Count keyword matches
        matches = message_words & skill['keywords']
        score = len(matches)

        # Boost for direct name mention
        if skill['id'].lower() in message_lower or skill['name'].lower() in message_lower:
            score += 5

        # Boost for description phrase match
        if any(phrase in message_lower for phrase in skill['description'].lower().split(',')):
            score += 2

        if score > 0:
            scored_skills.append((skill, score))

    # Sort by score descending
    scored_skills.sort(key=lambda x: x[1], reverse=True)
    return scored_skills[:top_k]


def get_skill_context(user_message: str) -> str:
    """
    Get additional context from relevant skills to append to system prompt.
    """
    relevant = detect_relevant_skills(user_message)

    if not relevant:
        return ""

    context_parts = []
    for skill, score in relevant:
        if score >= 2:  # Only include if reasonably confident
            # Extract the main skill content (after frontmatter)
            content = skill['content']
            # Remove frontmatter
            import re
            content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
            # Truncate if too long
            if len(content) > 1500:
                content = content[:1500] + "..."
            context_parts.append(f"[Skill: {skill['name']}]\n{content}")

    if context_parts:
        return "\n\n---\nRELEVANT SKILLS:\n" + "\n\n".join(context_parts)

    return ""


# Load skills on module init
load_skills_cache()


# ============================================================================
# Session Persistence
# ============================================================================

def get_session_file(character_id: str) -> Path:
    return SESSIONS_DIR / f"{character_id}_session.json"

# load_session is defined earlier in the file (lines 445-517) with proper format validation

def save_session(character_id: str, history: list):
    session_file = get_session_file(character_id)
    try:
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump({
                'character_id': character_id,
                'history': history,
                'updated_at': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Session] Error saving: {e}")

# ============================================================================
# Image Handling for Vision Models
# ============================================================================

def encode_image_to_base64(image) -> Tuple[Optional[str], Optional[str]]:
    """
    Encode image to base64 for vision models.
    Returns (base64_data, mime_type) or (None, None) on failure.
    """
    # Handle None, False, empty values from Gradio
    if image is None or image is False or not PIL_AVAILABLE:
        return None, None
    
    try:
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            
            pil_image = Image.fromarray(image)
        elif hasattr(image, 'save'):
            pil_image = image
        else:
            print(f"[Image] Unknown image type: {type(image)}")
            return None, None
        
        import io
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        b64_data = base64.b64encode(buffer.read()).decode('utf-8')
        return b64_data, "image/jpeg"
        
    except Exception as e:
        print(f"[Image] Error encoding: {e}")
        return None, None

def is_vision_model(model: str) -> bool:
    """Check if model supports vision/images"""
    for vm in VISION_MODELS:
        if vm in model or model in vm:
            return True
    vision_patterns = ['vision', 'gpt-4o', 'claude-3', 'gemini', 'grok-4']
    return any(p in model.lower() for p in vision_patterns)

# ============================================================================
# Audio Functions
# ============================================================================

# Track last processed audio to prevent duplicate transcription
_last_audio_hash = None
_last_transcript = None

def _compute_audio_hash(audio_np) -> str:
    """Compute a simple hash of audio data to detect duplicates"""
    import hashlib
    # Use first and last 1000 samples + length for fast hash
    sample = audio_np[:1000].tobytes() + audio_np[-1000:].tobytes() + str(len(audio_np)).encode()
    return hashlib.md5(sample).hexdigest()

def transcribe_audio(audio_data) -> tuple:
    """
    Transcribe audio using selected STT backend.

    Returns:
        Tuple of (text, emotion_result)
        - SenseVoice: returns emotion directly from unified model
        - Others: returns (text, None), emotion handled separately
    """
    global _last_audio_hash, _last_transcript

    if audio_data is None:
        return "", None

    model = init_stt()
    if model is None:
        return "[STT not available - check backend settings]", None

    try:
        sample_rate, audio_np = audio_data
        if audio_np.size == 0:
            return "[Empty audio]", None

        # Check for duplicate audio to prevent reprocessing stale data
        audio_hash = _compute_audio_hash(audio_np)
        if audio_hash == _last_audio_hash:
            print(f"[STT] Skipping duplicate audio (hash match)")
            return "", None  # Return empty to avoid reprocessing
        _last_audio_hash = audio_hash

        if audio_np.dtype == np.float32 or audio_np.dtype == np.float64:
            audio_np = (audio_np * 32767).astype(np.int16)
        elif audio_np.dtype == np.int32:
            audio_np = (audio_np / 65536).astype(np.int16)

        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, sample_rate, audio_np)
            temp_path = f.name

        try:
            text = ""
            emotion_result = None

            # Handle each backend appropriately
            if STT_BACKEND_NAME == "sensevoice":
                # SenseVoice: unified STT + emotion in one pass
                transcription, emotion_result = model.transcribe_with_emotion(temp_path)
                text = transcription.text if hasattr(transcription, 'text') else str(transcription)

            elif STT_BACKEND_NAME == "funasr":
                # FunASR: STT only, emotion handled separately
                result = model.transcribe(temp_path)
                text = result.text if hasattr(result, 'text') else str(result)

            elif STT_BACKEND_NAME == "faster_whisper":
                # faster-whisper: original logic
                if hasattr(model, 'transcribe'):
                    # Explicitly set language and disable VAD filter
                    segments, info = model.transcribe(temp_path, beam_size=5, language="en", vad_filter=False)
                    all_segments = list(segments)
                    if all_segments:
                        text = " ".join([s.text for s in all_segments]).strip()
                    else:
                        text = ""
                else:
                    result = model.transcribe(temp_path)
                    text = result["text"].strip()
            else:
                # Unknown backend - try generic transcribe
                if hasattr(model, 'transcribe'):
                    result = model.transcribe(temp_path)
                    text = result.text if hasattr(result, 'text') else str(result)
                else:
                    return "[Unknown STT backend]", None

            if not text:
                return "[No speech detected]", None

            # Filter out empty/silence transcriptions
            # These are typically just punctuation, ellipses, or whitespace
            cleaned = text.strip()
            # Remove all punctuation and whitespace to check if there's actual content
            content_only = re.sub(r'[\s\.\,\!\?\-\…]+', '', cleaned)
            if not content_only or len(content_only) < 2:
                print(f"[STT] Skipping empty/silence transcription: '{text[:30]}'")
                # Still return emotion if detected (for non-verbal audio like sighs)
                return "[No speech detected]", emotion_result

            print(f"[STT] Transcribed ({STT_BACKEND_NAME}): {text[:50]}...")
            return text, emotion_result

        finally:
            Path(temp_path).unlink(missing_ok=True)

    except Exception as e:
        print(f"[STT] {STT_BACKEND_NAME or 'Unknown'} error: {e}")
        import traceback
        traceback.print_exc()
        return f"[STT Error: {str(e)[:50]}]", None


def _map_voice_to_supertonic(voice_name: str) -> str:
    """
    Map a voice file name to a Supertonic voice ID.

    Supertonic supports: male_1, male_2, male_3, female_1, female_2, female_3

    Args:
        voice_name: The stem of the voice file (e.g., 'alice', 'bob', 'narrator')

    Returns:
        One of the Supertonic voice IDs
    """
    voice_lower = voice_name.lower()

    # Direct mappings for common names
    female_indicators = ['female', 'woman', 'girl', 'she', 'alice', 'emma', 'sarah',
                         'luna', 'aria', 'nova', 'stella', 'aurora', 'ivy', 'rose',
                         'lily', 'sophie', 'mia', 'chloe', 'zoe', 'ava']
    male_indicators = ['male', 'man', 'boy', 'he', 'bob', 'john', 'james',
                       'narrator', 'alex', 'max', 'leo', 'sam', 'jake', 'ryan',
                       'david', 'mike', 'chris', 'nick', 'tom', 'dan']

    # Check for female voices
    for indicator in female_indicators:
        if indicator in voice_lower:
            # Distribute across female voices based on hash
            voice_hash = hash(voice_lower) % 3
            return f"female_{voice_hash + 1}"

    # Check for male voices
    for indicator in male_indicators:
        if indicator in voice_lower:
            # Distribute across male voices based on hash
            voice_hash = hash(voice_lower) % 3
            return f"male_{voice_hash + 1}"

    # Default: use hash to pick a voice, defaulting to female_1 for unknown
    voice_hash = hash(voice_lower) % 6
    if voice_hash < 3:
        return f"female_{voice_hash + 1}"
    else:
        return f"male_{voice_hash - 2}"


def generate_speech(text: str, voice_file: str, emotion: str = None) -> Tuple[Optional[Tuple], bool]:
    """
    Generate speech using specified voice reference.
    Returns (audio_data, was_long_text) or (None, was_long_text)
    
    Audio format returned: (sample_rate, numpy_array_float32)
    - Sample rate: typically 22050 or 24000 Hz from IndexTTS2
    - Audio: float32 numpy array normalized to [-1.0, 1.0]
    
    Args:
        text: Text to synthesize
        voice_file: Voice reference file
        emotion: Optional detected emotion label (e.g., 'happy', 'calm', 'frustrated')
                 Used to modulate TTS parameters for more expressive output.
    """
    tts = init_tts()
    if tts is None:
        return None, False
    
    clean_text, was_long = clean_text_for_tts(text)
    
    # Skip if text is empty after cleaning
    if not clean_text or len(clean_text.strip()) < 2:
        print("[TTS] Text empty after cleaning - skipping")
        return None, was_long
    
    # Get emotion-aware TTS parameters
    emo_speed = 1.0
    emo_text = None
    if emotion:
        try:
            from audio.emotion_tts import get_tts_params_for_emotion, get_indextts_emotion_params
            # Kokoro uses speed adjustment
            kokoro_params = get_tts_params_for_emotion(emotion)
            emo_speed = kokoro_params.get('speed', 1.0)
            # IndexTTS2 uses emo_text parameter
            indextts_params = get_indextts_emotion_params(emotion)
            emo_text = indextts_params.get('emo_text')
            print(f"[TTS] Emotion-aware: {emotion} → speed={emo_speed:.2f}, emo_text='{emo_text}'")
        except ImportError:
            pass  # emotion_tts not available

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUT_DIR / f"response_{timestamp}.wav"

    try:
        # SUPERTONIC BACKEND (Ultra-fast ONNX)
        try:
            from audio.backends.supertonic import SupertonicBackend
            is_supertonic = isinstance(tts, SupertonicBackend)
        except ImportError:
            is_supertonic = False

        # SOPRANO BACKEND (Ultra-fast GPU)
        try:
            from audio.backends.soprano import SopranoBackend
            is_soprano = isinstance(tts, SopranoBackend)
        except ImportError:
            is_soprano = False

        soprano_success = False
        if is_soprano:
            print(f"[TTS] Generating with Soprano: {clean_text[:50]}...")
            try:
                tts.synthesize(clean_text, "default", emotion=emotion, output_path=str(output_path))
                if output_path.exists():
                    soprano_success = True
                else:
                    print("[TTS] Soprano failed - no output, falling back to Kokoro")
            except Exception as e:
                print(f"[TTS] Soprano error: {e}")
                print("[TTS] Falling back to Kokoro...")

        supertonic_success = False
        if is_supertonic:
            # Map voice file to Supertonic voice ID
            voice_name = Path(voice_file).stem.lower()
            # If already a valid Supertonic preset, use directly
            valid_supertonic = ['male_1', 'male_2', 'male_3', 'female_1', 'female_2', 'female_3']
            if voice_name in valid_supertonic:
                supertonic_voice = voice_name
            else:
                supertonic_voice = _map_voice_to_supertonic(voice_name)
            print(f"[TTS] Generating with Supertonic (Voice: {supertonic_voice}): {clean_text[:50]}...")

            try:
                tts.synthesize(clean_text, supertonic_voice, emotion=emotion, output_path=str(output_path))
                if output_path.exists():
                    supertonic_success = True
                else:
                    print("[TTS] Supertonic failed - no output, falling back to Kokoro")
            except Exception as e:
                print(f"[TTS] Supertonic error: {e}")
                print("[TTS] Falling back to Kokoro...")

        # KOKORO BACKEND (also fallback from Supertonic/Soprano)
        kokoro_fallback = ((is_supertonic and not supertonic_success) or (is_soprano and not soprano_success)) and KOKORO_AVAILABLE
        if kokoro_fallback:
            # Load Kokoro for fallback (KokoroTTS already imported globally at line 162)
            tts = KokoroTTS(checkpoints_dir=str(SCRIPT_DIR.parent / "checkpoints"))
            print("[TTS] Loaded Kokoro as fallback")

        # Skip to validation if Supertonic or Soprano already succeeded
        if supertonic_success or soprano_success:
            pass  # Audio already generated by fast backend
        elif (not is_supertonic and not is_soprano and KOKORO_AVAILABLE and isinstance(tts, KokoroTTS)) or kokoro_fallback:
             voice_name = Path(voice_file).stem

             # Map Supertonic voice names to Kokoro voices when falling back
             if kokoro_fallback and voice_name in ('male_1', 'male_2', 'male_3', 'female_1', 'female_2', 'female_3'):
                 supertonic_to_kokoro = {
                     'female_1': 'af_sarah',
                     'female_2': 'af_bella',
                     'female_3': 'af_nicole',
                     'male_1': 'am_adam',
                     'male_2': 'am_michael',
                     'male_3': 'am_eric',
                 }
                 voice_name = supertonic_to_kokoro.get(voice_name, 'af_sarah')
                 print(f"[TTS] Mapped Supertonic voice to Kokoro: {voice_name}")

             print(f"[TTS] Generating with Kokoro (Voice: {voice_name}): {clean_text[:50]}...")

             # Pass emotion-aware speed parameter
             success = tts.infer(voice_name, clean_text, str(output_path), speed=emo_speed)
             if not success or not output_path.exists():
                 print("[TTS] Kokoro generation failed")
                 return None, was_long

        # INDEXTTS2 BACKEND
        else:
            voice_path = get_voice_path(voice_file)
            if not Path(voice_path).exists():
                print(f"[TTS] Voice file not found: {voice_path}")
                return None, False

            # Clear GPU memory before IndexTTS2 to prevent fragmentation
            # This helps maintain consistent RTF across multiple generations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            print(f"[TTS] Generating with IndexTTS2: {clean_text[:50]}..." +
                  (" (long text)" if was_long else "") +
                  (f" [emotion: {emo_text}]" if emo_text else ""))

            # IndexTTS2 emotion-aware generation
            # API: infer(spk_audio_prompt, text, output_path, emo_text=..., emo_alpha=..., use_emo_text=True)
            if emo_text:
                try:
                    # Get full emotion params including alpha
                    from audio.emotion_tts import get_indextts_emotion_params
                    emo_params = get_indextts_emotion_params(emotion)
                    tts.infer(
                        voice_path, 
                        clean_text, 
                        str(output_path),
                        emo_text=emo_params.get('emo_text'),
                        emo_alpha=emo_params.get('emo_alpha', 0.6),
                        use_emo_text=True,
                        use_random=False
                    )
                    print(f"[TTS] Using emotion: '{emo_params.get('emo_text')}' (alpha={emo_params.get('emo_alpha', 0.6):.2f})")
                except TypeError as e:
                    # Fallback if emo parameters not supported in this version
                    print(f"[TTS] Emotion params not supported, using default: {e}")
                    tts.infer(voice_path, clean_text, str(output_path))
            else:
                tts.infer(voice_path, clean_text, str(output_path))
        
        if not output_path.exists():
            print("[TTS] Output file not created")
            return None, was_long
        
        # Read the generated audio
        sample_rate, audio_data = wav.read(str(output_path))
        
        # Validate audio data
        if audio_data is None or len(audio_data) == 0:
            print("[TTS] Empty audio data generated")
            output_path.unlink(missing_ok=True)
            return None, was_long
        
        # Convert to float32 for Gradio audio component
        if audio_data.dtype == np.int16:
            audio_np = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_np = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio_np = audio_data.astype(np.float32)
        
        # Normalize if needed
        max_val = np.abs(audio_np).max()
        if max_val < 0.001:
            # Very quiet audio - amplify
            audio_np = audio_np * 10.0
        elif max_val > 1.0:
            # Clip to prevent distortion
            audio_np = np.clip(audio_np, -1.0, 1.0)
        
        # Ensure mono
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1)
        audio_np = audio_np.flatten()
        
        # Add brief silence at start for smooth playback
        # This helps prevent audio cutting off at the beginning
        silence_samples = int(sample_rate * 0.1)  # 100ms silence
        silence = np.zeros(silence_samples, dtype=np.float32)
        audio_np = np.concatenate([silence, audio_np, silence])  # Add silence at start and end
        
        # Final validation
        audio_duration = len(audio_np) / sample_rate
        if audio_duration < 0.5:
            print(f"[TTS] Warning: Very short audio ({audio_duration:.2f}s)")
        
        # Cleanup temp file
        output_path.unlink(missing_ok=True)
        torch.cuda.empty_cache()
        
        print(f"[TTS] Generated {audio_duration:.2f}s of audio")
        return (sample_rate, audio_np), was_long
        
    except Exception as e:
        print(f"[TTS] Error: {e}")
        import traceback
        traceback.print_exc()
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return None, False


# ============================================================================
# LLM Functions (OpenRouter + LM Studio)
# ============================================================================

def chat_with_openrouter(messages: list, model: str, temperature: float, max_tokens: int,
                          top_p: float, freq_penalty: float, pres_penalty: float,
                          image_data: Tuple[str, str] = None, tools: list = None) -> Any:
    """Send messages to OpenRouter API, with optional image and VALID tools"""
    headers = {
        "Authorization": f"Bearer {CONFIG['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "IndexTTS2 Voice Agent"
    }
    
    # Use deep copy to avoid modifying the global session history
    messages_payload = copy.deepcopy(messages)
    
    # 1. Image Handling
    if image_data and is_vision_model(model):
        b64_data, mime_type = image_data
        if b64_data and messages_payload:
            for i in range(len(messages_payload) - 1, -1, -1):
                if messages_payload[i]["role"] == "user":
                    text_content = messages_payload[i]["content"]
                    
                    if isinstance(text_content, list):
                         has_image = any(item.get("type") == "image_url" for item in text_content)
                         if not has_image:
                             messages_payload[i]["content"].insert(0, {
                                "type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}
                             })
                    else:
                        messages_payload[i]["content"] = [
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}},
                            {"type": "text", "text": text_content}
                        ]
                    break
    
    # 2. Payload Construction
    payload = {
        "model": model,
        "messages": messages_payload,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": freq_penalty,
        "presence_penalty": pres_penalty,
    }

    # 3. Tool Handling - ONLY send if model supports it and tools exist
    # Most local models or small models fail if 'tools' is sent.
    # Grok, GPT-4, Claude support tools.
    supports_tools = any(x in model.lower() for x in ['gpt-', 'claude', 'grok', 'mistral', 'llama-3'])
    if tools and supports_tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=payload, timeout=120
        )
        response.raise_for_status()

        # Return full message object with usage stats attached
        data = response.json()
        message = data["choices"][0]["message"]
        # Attach usage stats to message for HUD display
        if "usage" in data:
            message["_usage"] = data["usage"]
        return message

    except Exception as e:
        print(f"[LLM] OpenRouter error: {e}")
        # Return a fake error message object
        return {"content": f"Error: {str(e)}", "role": "assistant"}

def chat_with_lm_studio(messages: list, temperature: float, max_tokens: int,
                        top_p: float, freq_penalty: float, pres_penalty: float,
                        image_data: Tuple[str, str] = None, tools: list = None) -> Any:
    """Send messages to LM Studio local server."""
    global LM_STUDIO_AVAILABLE, LM_STUDIO_MODEL
    
    if not LM_STUDIO_AVAILABLE:
        LM_STUDIO_AVAILABLE, LM_STUDIO_MODEL = check_lm_studio_available()
        if not LM_STUDIO_AVAILABLE:
            return {"role": "assistant", "content": "Error: LM Studio is not running. Please start LM Studio and enable the server."}
    
    if not LM_STUDIO_MODEL:
        return {"role": "assistant", "content": "Error: LM Studio is running but no model is loaded. Please load a model."}
    
    # Deep copy messages to prevent polluting the session history with base64 images
    # or causing recursive nesting bugs on retries.
    messages_payload = copy.deepcopy(messages)

    if image_data:
        b64_data, mime_type = image_data
        if b64_data and messages_payload:
            for i in range(len(messages_payload) - 1, -1, -1):
                if messages_payload[i]["role"] == "user":
                    text_content = messages_payload[i]["content"]
                    
                    # Ensure we handle existing list content (though deepcopy resets us to clean state usually)
                    # But if the history passed in WAS corrupted, this safety check helps.
                    if isinstance(text_content, list):
                         # If already a list, just prepend image if not present
                         has_image = any(item.get("type") == "image_url" for item in text_content)
                         if not has_image:
                             messages_payload[i]["content"].insert(0, {
                                "type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}
                             })
                    else:
                        # Standard string - wrap it
                        messages_payload[i]["content"] = [
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}},
                            {"type": "text", "text": text_content}
                        ]
                    break
    
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": messages_payload,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": freq_penalty,
        "presence_penalty": pres_penalty,
        "stream": False,
    }

    # Add tools if provided
    if tools:
        print(f"[LLM] Sending {len(tools)} tools to LM Studio")
        payload["tools"] = tools
        # payload["tool_choice"] = "auto"  # Let server decide default (safer for local models)
    
    try:
        print(f"[LLM] LM Studio model: {LM_STUDIO_MODEL}")
        response = requests.post(
            f"{LM_STUDIO_BASE_URL}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload, timeout=120
        )
        response.raise_for_status()
        # Return full message object with usage stats attached
        data = response.json()
        message = data["choices"][0]["message"]
        # Attach usage stats to message for HUD display
        if "usage" in data:
            message["_usage"] = data["usage"]
        return message

    except requests.exceptions.ConnectionError:
        LM_STUDIO_AVAILABLE = False
        LM_STUDIO_MODEL = None
        return {"role": "assistant", "content": "Error: Lost connection to LM Studio."}
    except requests.exceptions.Timeout:
        return {"role": "assistant", "content": "Error: LM Studio request timed out."}
    except Exception as e:
        print(f"[LLM] LM Studio error: {e}")
        return {"role": "assistant", "content": f"Error: {str(e)}"}

def chat_with_llm(messages: list, model: str, temperature: float, max_tokens: int,
                  top_p: float, freq_penalty: float, pres_penalty: float,
                  provider: str = "openrouter", image_data: Tuple[str, str] = None,
                  tools: list = None) -> Any:
    """Send messages to LLM based on selected provider."""
    if provider == "lmstudio":
        # LM Studio tool support enabled (requires LM Studio 0.2.x+ and compatible model)
        return chat_with_lm_studio(messages, temperature, max_tokens, top_p, 
                                   freq_penalty, pres_penalty, image_data, tools)
    else:
        return chat_with_openrouter(messages, model, temperature, max_tokens, 
                                     top_p, freq_penalty, pres_penalty, image_data, tools)

# ============================================================================
# Memory Context Formatting
# ============================================================================

def format_memory_for_ui(context: Dict[str, Any]) -> str:
    """Format memory context for display in the UI"""
    parts = []

    if context.get('character_state'):
        state = context['character_state']
        parts.append(f"**Session:** {state['total_interactions']} interactions | Mood: {state['mood']}")

    semantic = context.get('semantic_memories', [])
    if semantic:
        parts.append("\n**Character Knowledge:**")
        for mem in semantic[:3]:
            parts.append(f"• {mem[:80]}...")

    episodic = context.get('episodic_memories', [])
    if episodic:
        parts.append("\n**Relevant Past:**")
        for mem in episodic[:2]:
            lines = mem.split('\n')
            if lines:
                parts.append(f"• {lines[0][:60]}...")

    return '\n'.join(parts) if parts else "*No memories yet*"

def format_tool_call_html(tool_name: str, arguments: Dict[str, Any], result: str = None,
                         step: int = None, total: int = None, status: str = "complete") -> str:
    """Format a tool call as HTML for CYBERDECK System Protocol display"""
    args_str = ", ".join([f"{k}: {v}" for k, v in arguments.items()])

    # Status indicator - Military style
    status_icons = {"pending": "[ EXEC ]", "complete": "[ OK ]", "failed": "[ FAIL ]"}
    status_icon = status_icons.get(status, "[ OK ]")

    # Protocol indicator for chains
    step_html = ""
    chain_class = ""
    if step is not None and total is not None and total > 1:
        step_html = f'<span class="tool-step-badge">PROTOCOL {step:02d}</span>'
        if step < total:
            chain_class = " tool-chain-continues"
        if step > 1:
            chain_class += " tool-chain-continued"

    html = f'<div class="tool-call-block{chain_class}">'
    html += f'<div class="tool-call-header">{step_html}{status_icon} {tool_name}</div>'
    html += f'<div class="tool-call-args">{args_str}</div>'

    if result:
        # Truncate result if too long
        display_result = result[:200] + "..." if len(result) > 200 else result
        # Escape HTML in result
        display_result = display_result.replace('<', '&lt;').replace('>', '&gt;')
        html += f'<div class="tool-call-result">{display_result}</div>'

    html += '</div>'
    return html

def format_message_with_extras(content: str, add_timestamp: bool = True, expandable_threshold: int = 800) -> str:
    """Wrap message content with expandable container (if long) and timestamp"""
    from datetime import datetime

    result = content

    # Wrap in expandable container if content is long
    if len(content) > expandable_threshold:
        result = f'<div class="message-expandable">{content}</div>'
        result += '<div class="expand-btn">▼ Show more</div>'

    # Add timestamp
    if add_timestamp:
        timestamp = datetime.now().strftime("%H:%M")
        result += f'<div class="message-timestamp">{timestamp}</div>'

    return result

def format_memory_recall_html(context: Dict[str, Any]) -> str:
    """Format memory recall as HTML for Claude Desktop-style display"""
    if not context:
        return ""

    items = []

    # Extract memories for display
    semantic = context.get('semantic_memories', [])
    episodic = context.get('episodic_memories', [])

    for mem in semantic[:2]:
        items.append(f'<div class="memory-item">{mem[:100]}...</div>')

    for mem in episodic[:2]:
        lines = mem.split('\n')
        if lines:
            items.append(f'<div class="memory-item">{lines[0][:80]}...</div>')

    if not items:
        return ""

    html = '<div class="memory-recall-block">'
    html += f'<div class="memory-recall-header">🧠 Recalled {len(items)} memories:</div>'
    html += '\n'.join(items)
    html += '</div>'

    return html

def should_load_tools(message: str) -> bool:
    """
    Progressive Disclosure: Determine if we should load tool schemas for this message.

    This dramatically reduces context bloat by skipping 10-15k tokens of tool schemas
    for casual conversational messages.

    Args:
        message: The user's message

    Returns:
        True if tools should be loaded, False for casual conversation
    """
    message_lower = message.lower().strip()

    # Casual conversation patterns that DON'T need tools
    casual_patterns = [
        'hi', 'hello', 'hey', 'sup', 'yo',
        'thanks', 'thank you', 'thx', 'ty',
        'okay', 'ok', 'sure', 'alright', 'k',
        'bye', 'goodbye', 'see you', 'later',
        'yes', 'no', 'yeah', 'nah', 'yep', 'nope',
        'cool', 'nice', 'great', 'awesome', 'lol', 'haha'
    ]

    # If message is ONLY a casual response, skip tools
    if message_lower in casual_patterns:
        return False

    # If message is very short (3 words or less) and doesn't ask a question,
    # check if it starts with casual pattern
    if len(message.split()) <= 3 and '?' not in message:
        for pattern in casual_patterns:
            if message_lower.startswith(pattern):
                return False

    # Everything else (questions, requests, complex messages) - load tools
    # This includes:
    # - Questions: "What time is it?", "Can you search for X?"
    # - Requests: "Search for...", "Read the file...", "Calculate..."
    # - Complex conversation: "I was thinking about..."
    return True


def should_load_mcp_tools(message: str) -> bool:
    """
    Progressive Disclosure for MCP Tools: Decide whether to load MCP tools.

    MCP tools (filesystem, sqlite, etc.) are specialized and add tokens to context.
    Now defaults to True if MCP is available - tools should be accessible.

    Args:
        message: The user's message

    Returns:
        True if MCP tools should be loaded, False otherwise
    """
    # If MCP is available, default to including MCP tools
    # This ensures the agent can use all available tools
    if not MCP_TOOLS_CACHE:
        return False

    message_lower = message.lower()

    # Only EXCLUDE MCP tools for clearly casual messages (greetings, small talk)
    casual_patterns = [
        'hello', 'hi there', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', "what's up", 'thanks', 'thank you', 'bye', 'goodbye',
    ]

    # If message is very short and casual, skip MCP tools
    if len(message) < 30:
        for pattern in casual_patterns:
            if message_lower.strip() == pattern or message_lower.startswith(pattern + ' '):
                return False

    # For all other messages, include MCP tools
    return True


def filter_for_speech(full_message: str, has_tools: bool = False) -> str:
    """
    Extract only conversational narration for TTS.
    Removes HTML blocks (tool calls, memory recalls) and keeps only spoken text.

    Args:
        full_message: Full assistant message with HTML blocks
        has_tools: Whether this message involved tool calls (deprecated - now auto-detects)

    Returns:
        Clean text suitable for TTS
    """
    # Auto-detect if message contains ANY HTML blocks (check for both quote styles)
    has_html_blocks = ('<div class="' in full_message) or ('<div class=\'' in full_message)

    # If no HTML blocks present, return as-is
    if not has_html_blocks:
        return full_message

    print(f"[TTS Filter] Detected HTML blocks, filtering...")
    print(f"[TTS Filter] Original length: {len(full_message)} chars")

    # AGGRESSIVE APPROACH: Remove ALL <div...>...</div> tags and keep only plain text
    # This handles nested divs properly
    text = full_message

    # Remove all div blocks (handles nesting by removing from innermost to outermost)
    while '<div' in text and '</div>' in text:
        text = re.sub(r'<div[^>]*>.*?</div>', '', text, flags=re.DOTALL, count=1)

    # Also remove any remaining orphaned tags
    text = re.sub(r'<[^>]+>', '', text)

    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = text.strip()

    print(f"[TTS Filter] Filtered length: {len(text)} chars")
    print(f"[TTS Filter] Will speak: {text[:100]}...")

    # If filtering removed everything, provide a fallback
    if not text or len(text) < 10:
        print("[TTS Filter] WARNING: Filtering removed all text, using fallback")
        return "Working on that now."

    return text

# ============================================================================
# Core Message Processing
# ============================================================================

def process_message_with_memory_v2(user_message: str, chat_history: list, character_id: str, 
                                 voice_file: str, model: str, temperature: float, 
                                 max_tokens: int, top_p: float, freq_penalty: float, 
                                 pres_penalty: float, conversation_id: str, tts_enabled: bool = True,
                                 llm_provider: str = "openrouter", image = None,
                                 incognito: bool = False):
    """
    Process a message using the new service layer (Phase 1: Architecture Cleanup).
    
    This is the refactored version that delegates to ChatService.
    The old implementation is kept below as process_message_with_memory_legacy().
    """
    has_image = image is not None
    has_text = user_message and user_message.strip()
    
    if not has_text and not has_image:
        return chat_history, None, "", gr.update(), conversation_id, None, ""
    
    # Prepare image data if present
    image_data = None
    if has_image:
        image_data = encode_image_to_base64(image)
        if image_data[0]:
            print(f"[Image] Encoded image for vision model")
    
    # Get emotion context (if available from async detection)
    emotion_context = get_current_emotion_context()
    emotion_label = None  # For TTS emotion modulation
    if emotion_context:
        # Extract the label before clearing (for TTS modulation)
        if _current_emotion:
            emotion_label = _current_emotion.label
        clear_current_emotion()
    
    # Get skill context
    skill_context = get_skill_context(user_message)
    
    # Get ChatService from container
    chat_service = service_container.get_chat_service()
    
    # Process message through service layer
    result = chat_service.process_message(
        user_message=user_message,
        character_id=character_id,
        voice_file=voice_file,
        model=model,
        conversation_id=conversation_id,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        freq_penalty=freq_penalty,
        pres_penalty=pres_penalty,
        llm_provider=llm_provider,
        image_data=image_data,
        incognito=incognito,
        chat_history=chat_history,
        emotion_context=emotion_context,
        emotion_label=emotion_label,  # For TTS emotion modulation
        skill_context=skill_context,
        # Pass function references (dependency injection)
        chat_with_llm_func=chat_with_llm,
        execute_tool_func=execute_tool_with_mcp_fallback,
        get_tools_schema_func=get_combined_tools_schema,
        should_load_tools_func=should_load_tools,
        should_load_mcp_tools_func=should_load_mcp_tools,
        generate_speech_func=generate_speech,
        save_conversation_func=save_conversation,
        generate_conversation_id_func=generate_conversation_id,
    )
    
    # Handle TTS for tool call responses (regenerate with filtered text)
    if result.tool_calls and result.audio_data:
        # Need to filter the response text for TTS (remove tool HTML blocks)
        tts_text = ui_filter_speech(result.response_text, has_tools=True)
        # Regenerate audio with filtered text
        if tts_enabled and TTS_AVAILABLE:
            # Prevent VAD from triggering during generation (prevent "thinking" noise interruptions)
            if vad_manager_instance:
                 vad_manager_instance.pause(5.0)  # Temporary block while generating
                 
            audio_result, was_long = generate_speech(tts_text, voice_file, emotion=emotion_label)
            result.audio_data = audio_result
            if was_long:
                result.tts_warning = "⚠️ Long response - TTS may take longer"
    
    # CRITICAL FIX: Pause VAD for ALL TTS audio playback (not just tool calls)
    # This prevents coughs, breathing, or ambient noise from interrupting playback.
    # Previously this was ONLY inside the tool_calls block, causing regular responses
    # to play without VAD protection - allowing noise to trigger new recordings.
    if result.audio_data and vad_manager_instance:
        sr, audio = result.audio_data
        duration = len(audio) / sr
        # Use 2.5x multiplier + buffer to account for:
        # - Browser autoplay delay (1-3 seconds)
        # - Network latency from WSL to browser
        # - Variable playback speed
        # - User breathing/making small sounds
        # - Walking, footsteps, ambient noise in room
        vad_manager_instance.pause(duration * 2.5 + 8.0)
    
    # Format result for UI using formatters
    formatted_history = ui_format_chat_result(result, chat_history)
    memory_ui_text = ui_format_memory(result.memory_context)
    
    # Save conversation (unless incognito)
    if not incognito:
        save_session(character_id, formatted_history)
        save_conversation(character_id, result.conversation_id, formatted_history)
    
    # Periodic GPU cleanup
    cleanup_counter[0] += 1
    if cleanup_counter[0] % 10 == 0:
        clear_cuda_memory(aggressive=True)

    # Estimate TTS speed based on backend (hardcoded estimates)
    tts_backend = SETTINGS.get("tts_backend", "indextts")
    tts_speed_estimates = {
        "indextts": 2.0,   # ~2x realtime
        "kokoro": 10.0,     # ~10x realtime
        "supertonic": 167.0, # ~167x realtime
        "soprano": 2000.0   # ~2000x realtime
    }
    estimated_tts_speed = tts_speed_estimates.get(tts_backend, 2.0)

    # Map emotion to value (0-1 scale for meter width)
    emotion_values = {
        "happy": 0.9, "neutral": 0.5, "calm": 0.6,
        "sad": 0.3, "angry": 0.8, "fear": 0.7, "frustrated": 0.75
    }
    emotion_val = emotion_values.get(result.emotion.lower() if result.emotion else "neutral", 0.5)

    # Generate HUD update script (executed by browser when rendered)
    hud_script = f"""<script>
if (window.HUD) {{
    window.HUD.update({{
        latency: {result.latency_ms},
        ttsSpeed: {estimated_tts_speed:.0f},
        tokensIn: {result.tokens_in},
        tokensOut: {result.tokens_out},
        memoryNodes: {result.memory_nodes},
        emotion: '{result.emotion or "neutral"}',
        emotionValue: {emotion_val}
    }});
}}
</script>"""

    # Return order: [chatbot, audio, msg_input, memory_display, conv_id, image_input, tts_warning, hud_update]
    return (formatted_history, result.audio_data, "", memory_ui_text,
            result.conversation_id, gr.update(), result.tts_warning, hud_script)


# Direct assignment to the service-based implementation (Phase 1: Architecture Cleanup)
# Legacy implementation removed - v2 is now the only implementation
process_message_with_memory = process_message_with_memory_v2


def process_voice_input(audio_data, chat_history, character_id, voice_file, *args):
    """Process voice input with thread safety - handles variable settings args"""
    # args contains: model, temp, tokens, top_p, freq, pres, conv_id, tts, provider, [group_args...], image
    if not processing_lock.acquire(blocking=False):
        return chat_history, None, "⏳ Processing...", gr.update(), args[6] if len(args) > 6 else "new", gr.update(), "", ""

    try:
        if audio_data is None:
             # args[6] is conversation_id usually, but let's be safe
            conv_id = args[6] if len(args) > 6 else "new"
            return chat_history, None, "", gr.update(), conv_id, gr.update(), "", ""

        # Transcribe with potential emotion (SenseVoice provides both)
        user_message, stt_emotion = transcribe_audio(audio_data)
        if not user_message or user_message.startswith("["):
            conv_id = args[6] if len(args) > 6 else "new"
            return chat_history, None, user_message or "", gr.update(), conv_id, gr.update(), "", ""

        # Handle emotion routing based on STT backend
        global _current_emotion
        if stt_emotion is not None:
            # SenseVoice provided emotion - use it directly
            _current_emotion = stt_emotion
            print(f"[Emotion] Using SenseVoice: {stt_emotion.label}")
        elif should_run_external_emotion(SETTINGS.get("stt_backend", "faster_whisper")):
            # Run wav2vec2 async for non-SenseVoice backends
            if EMOTION_AVAILABLE and SETTINGS.get("emotion_detection_enabled", True):
                detect_emotion_from_audio_async(audio_data)
        
        # Use the wrapper now to support Group Chat via Voice
        # Map args from UI wiring: [model, temp, tokens, top_p, freq, pres, conv_id, tts, provider, group_enabled, members, turns, image]
        
        model = args[0]
        temperature = args[1]
        max_tokens = args[2]
        top_p = args[3]
        freq_penalty = args[4]
        pres_penalty = args[5]
        conversation_id = args[6]
        tts_enabled = args[7]
        llm_provider = args[8]
        
        group_enabled = args[9] if len(args) > 9 else False
        group_members = args[10] if len(args) > 10 else []
        group_turns = args[11] if len(args) > 11 else 2
        incognito = args[12] if len(args) > 12 else False
        
        # Image is typically the last one added in inputs list: + [image_input]
        image = args[-1]
        
        # Iterate through generator to get final state
        final_result = None
        for res in process_group_chat_wrapper(user_message, chat_history, character_id, voice_file,
                                            model, temperature, max_tokens, top_p, freq_penalty, 
                                            pres_penalty, conversation_id, tts_enabled, llm_provider,
                                            group_enabled, group_members, group_turns, incognito, image):
            final_result = res
            
        # Clear audio hash after successful processing to allow fresh recordings
        global _last_audio_hash
        _last_audio_hash = None

        # Include HUD update (element 7)
        hud_update = final_result[7] if len(final_result) > 7 else ""
        return final_result[0], final_result[1], user_message, final_result[3], final_result[4], final_result[5], final_result[6], hud_update
    finally:
        processing_lock.release()


# ============================================================================
# Character & Conversation Switching
# ============================================================================

def switch_character(new_character_id: str, current_character_id: str, current_history: list,
                     current_conversation_id: str):
    """Switch to a different character"""
    if current_character_id and current_history:
        save_session(current_character_id, current_history)
        if current_conversation_id and current_conversation_id != "new":
            save_conversation(current_character_id, current_conversation_id, current_history)
        MEMORY_MANAGER.deactivate_character(current_character_id)
    
    character = CHARACTER_MANAGER.get_character(new_character_id)
    if not character:
        return current_history, current_character_id, "reference.wav", "Character not found!", "", "", gr.update(), "new"
    
    MEMORY_MANAGER.activate_character(new_character_id)
    
    state = MEMORY_MANAGER.get_character_state(new_character_id)
    if state and state.interaction_count == 0:
        for mem in character.initial_memories:
            MEMORY_MANAGER.add_semantic_memory(new_character_id, mem, importance=0.9)
        for trait in character.personality_traits:
            MEMORY_MANAGER.add_procedural_memory(new_character_id, f"Personality: {trait}")
    
    new_history = load_session(new_character_id)
    stats = MEMORY_MANAGER.get_stats(new_character_id)
    
    SETTINGS["last_character"] = new_character_id
    SETTINGS["last_voice"] = character.default_voice
    save_settings(SETTINGS)
    
    status = f"✓ {character.display_name}"
    
    stats_text = f"""**Interactions:** {stats.get('total_interactions', 0)}
**Episodic:** {stats.get('episodic_count', 0)}
**Semantic:** {stats.get('semantic_count', 0)}"""
    
    conv_choices = get_conversation_choices(new_character_id)
    
    print(f"[Switch] → {new_character_id} (voice: {character.default_voice})")
    
    return (new_history, new_character_id, character.default_voice, status, 
            stats_text, "*Ready for new memories*", gr.update(choices=conv_choices, value="new"), "new")

def switch_conversation(conversation_id: str, character_id: str, current_history: list,
                        current_conversation_id: str):
    """Switch to a different conversation or create new one"""
    if current_conversation_id and current_conversation_id != "new" and current_history:
        save_conversation(character_id, current_conversation_id, current_history)
    
    if conversation_id == "new":
        new_id = generate_conversation_id()
        SETTINGS["current_conversation_id"] = new_id
        save_settings(SETTINGS)
        return [], new_id, f"✓ New conversation started"
    else:
        history, title = load_conversation(character_id, conversation_id)
        SETTINGS["current_conversation_id"] = conversation_id
        save_settings(SETTINGS)
        return history, conversation_id, f"✓ Loaded: {title[:30]}"

def clear_chat(character_id: str, conversation_id: str):
    """
    Start a fresh conversation while KEEPING all memories.
    
    This is like starting a new conversation with someone you know -
    they still remember you, but you're starting a new topic.
    """
    clear_cuda_memory(aggressive=True)
    
    # Clear the current session file (temporary state)
    session_file = get_session_file(character_id)
    if session_file.exists():
        session_file.unlink()
    
    # Don't delete the conversation - just archive it
    # (User can delete manually from conversation dropdown if they want)
    
    # Generate new conversation ID
    new_id = generate_conversation_id()
    conv_choices = get_conversation_choices(character_id)
    
    # Memory stays intact - character still remembers you!
    stats = MEMORY_MANAGER.get_stats(character_id)
    print(f"[Chat] New conversation started (memories intact: {stats.get('episodic_count', 0)} episodic)")
    
    return [], gr.update(value=None), "✓ New conversation (memories kept)", gr.update(choices=conv_choices, value="new"), new_id

def clear_all_memory(character_id: str, conversation_id: str):
    """
    NUCLEAR OPTION: Clear ALL memory and conversations for this character.
    
    This completely resets the character as if you've never met.
    Use sparingly - this cannot be undone!
    """
    clear_cuda_memory(aggressive=True)
    
    # Clear session
    session_file = get_session_file(character_id)
    if session_file.exists():
        session_file.unlink()
    
    # Delete ALL conversations for this character
    char_dir = get_conversations_dir(character_id)
    for f in char_dir.glob("*.json"):
        f.unlink()
    
    # Clear ALL memories for this character
    MEMORY_MANAGER.clear_character_memory(character_id)
    
    # Re-initialize with base memories from character definition
    character = CHARACTER_MANAGER.get_character(character_id)
    if character:
        MEMORY_MANAGER.activate_character(character_id)
        # Re-add initial semantic memories (personality, background)
        for mem in character.initial_memories:
            MEMORY_MANAGER.add_semantic_memory(character_id, mem, importance=0.9)
        # Re-add personality traits as procedural memories
        for trait in character.personality_traits:
            MEMORY_MANAGER.add_procedural_memory(character_id, f"Personality: {trait}")
    
    new_id = generate_conversation_id()
    conv_choices = get_conversation_choices(character_id)
    
    print(f"[Memory] WIPED all memory for {character_id}")
    
    return [], gr.update(value=None), "⚠️ Memory wiped - character reset", gr.update(choices=conv_choices, value="new"), new_id

# ============================================================================
# PTT Support - Cross-Platform
# ============================================================================

def get_ptt_status() -> Tuple[str, str, float, str]:
    """Read PTT status from status file."""
    try:
        if PTT_STATUS_FILE.exists():
            content = PTT_STATUS_FILE.read_text().strip()
            parts = content.split("|")
            status = parts[0] if len(parts) > 0 else "offline"
            duration = float(parts[1]) if len(parts) > 1 else 0.0
            extra = parts[2] if len(parts) > 2 else ""
            
            status_map = {
                "ready": ("🎤 PTT Ready (Right Shift)", "#00BFFF", "lapis"),
                "recording": (f"🔴 RECORDING ({duration:.1f}s)", "#00BFFF", "lapis"),
                "processing": (f"⏳ Processing...", "#00BFFF", "lapis"),
                "sent": ("✅ Sent!", "#00BFFF", "lapis"),
                "error": (f"❌ Error: {extra}", "#00BFFF", "lapis"),
                "offline": ("⚫ Offline", "#006699", "lapis-dim"),
            }

            display, color, _ = status_map.get(status, ("❓ Unknown", "#006699", "lapis-dim"))
            if extra and status == "ready":
                display = f"🎤 {extra}"
            
            return status, display, duration, extra
        else:
            return "offline", "⚫ Offline", 0.0, ""
    except Exception as e:
        return "error", f"❌ Error: {str(e)[:30]}", 0.0, str(e)

def get_ptt_recordings_path() -> Path:
    """Get the appropriate recordings path based on platform"""
    # For all platforms, use the recordings directory relative to this script
    # This works for Windows native, Linux native, and WSL (when run from /mnt/...)
    return RECORDINGS_DIR

def load_ptt_audio():
    """Load audio from PTT trigger file - cross-platform"""
    recordings_path = get_ptt_recordings_path()
    trigger_file = recordings_path / "latest.txt"

    if not trigger_file.exists():
        return None

    try:
        # Small delay to ensure file is fully written
        time.sleep(0.1)

        # Try to read trigger file - may fail if another process got it first
        try:
            audio_path_str = trigger_file.read_text().strip()
        except (FileNotFoundError, OSError, IOError):
            # File was deleted between exists() check and read - normal race condition
            return None

        # DELETE TRIGGER FILE IMMEDIATELY to prevent race condition
        # This atomically "claims" this recording for processing
        try:
            trigger_file.unlink()
        except (FileNotFoundError, OSError, IOError):
            # Another process already claimed it, skip
            return None

        # Handle WSL path conversion if needed
        if IS_WSL:
            # Handle Windows paths (C:\ or c:\) -> WSL paths (/mnt/c/)
            # Pattern: c:\path\to\file.wav -> /mnt/c/path/to/file.wav
            if len(audio_path_str) >= 2 and audio_path_str[1] == ':':
                drive = audio_path_str[0].lower()
                path_part = audio_path_str[3:].replace('\\', '/')
                audio_path_str = f"/mnt/{drive}/{path_part}"

        audio_path = Path(audio_path_str)

        if audio_path.exists():
            # Check file size - skip if too small (< 5KB = ~0.15s of audio)
            try:
                file_size = audio_path.stat().st_size
            except (FileNotFoundError, OSError):
                return None

            if file_size < 5000:
                print(f"[PTT] Skipping tiny recording ({file_size} bytes)")
                try:
                    audio_path.unlink()
                except (OSError, PermissionError):
                    pass  # File already deleted or locked
                return None

            sample_rate, audio_data = wav.read(str(audio_path))
            try:
                audio_path.unlink()
            except (OSError, PermissionError):
                pass  # File already deleted or locked

            # Additional check - if audio is less than 0.5 seconds, skip
            duration = len(audio_data) / sample_rate
            if duration < 0.5:
                print(f"[PTT] Skipping short recording ({duration:.2f}s)")
                return None

            return (sample_rate, audio_data)
        else:
            # Audio file doesn't exist (shouldn't happen)
            return None
    except (FileNotFoundError, OSError, IOError):
        # Expected file access errors - silently ignore
        return None
    except Exception as e:
        # Log only truly unexpected errors (not file access issues)
        error_str = str(e).lower()
        if 'no such file' not in error_str and 'errno 2' not in error_str:
            print(f"[PTT] Error: {e}")
        # Trigger file already deleted, just clean up audio if it exists
        try:
            if 'audio_path' in locals() and audio_path.exists():
                audio_path.unlink()
        except (OSError, PermissionError):
            pass  # Cleanup failed, not critical
    return None

def check_ptt_recording(*args):
    """Check for PTT recording and process if found"""
    audio_data = load_ptt_audio()
    if audio_data is None:
        # Periodically cleanup orphan recordings (every ~10 checks when idle)
        import random
        if random.random() < 0.1:  # 10% chance each check
            cleanup_old_recordings(max_age_hours=1, max_files=5)
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    
    print("[PTT] Recording detected")
    result = process_voice_input(audio_data, *args)
    
    # Always cleanup after processing a recording
    cleanup_old_recordings(max_age_hours=1, max_files=5)
    
    return result

# ============================================================================
# Export Handlers
# ============================================================================

def handle_export_txt(character_id: str, conversation_id: str):
    """Export current conversation as TXT"""
    if not conversation_id or conversation_id == "new":
        return None, "No conversation to export"
    
    content, filename = export_conversation(character_id, conversation_id, "txt")
    if content:
        # Save to temp file for download
        export_path = OUTPUT_DIR / filename
        export_path.write_text(content, encoding='utf-8')
        return str(export_path), f"✓ Exported: {filename}"
    return None, "Export failed"

def handle_export_json(character_id: str, conversation_id: str):
    """Export current conversation as JSON"""
    if not conversation_id or conversation_id == "new":
        return None, "No conversation to export"
    
    content, filename = export_conversation(character_id, conversation_id, "json")
    if content:
        export_path = OUTPUT_DIR / filename
        export_path.write_text(content, encoding='utf-8')
        return str(export_path), f"✓ Exported: {filename}"
    return None, "Export failed"


# ============================================================================
# Gradio UI
# ============================================================================

def create_ui():
    
    model_choices = fetch_openrouter_models(CONFIG['OPENROUTER_API_KEY'])
    
    initial_char = SETTINGS.get("last_character", "hermione")
    initial_conv_choices = get_conversation_choices(initial_char)
    initial_provider = SETTINGS.get("llm_provider", "openrouter")
    
    # Platform-specific tips
    if IS_WINDOWS:
        ptt_tip = "Run ptt_windows.py for Push-to-Talk"
    elif IS_WSL:
        ptt_tip = "Run VoiceChat.bat from Windows for PTT"
    else:
        ptt_tip = "Run ptt_linux.py for Push-to-Talk (requires root)"
    
    custom_css = """
        /* ============================================
           SWITCHABLE COLOR THEME SYSTEM
           Lapis Lazuli | Lightsaber Purple | Orange | Green
           ============================================ */

        /* CSS Variables - Theme Colors (defaults to Lapis Lazuli) */
        :root {
            /* THEME COLOR VARIABLES - Changed dynamically by JavaScript */
            --theme-primary: #00BFFF;      /* Main accent color */
            --theme-dim: #006699;          /* Darker variant */
            --theme-bright: #33CCFF;       /* Lighter/hover variant */
            --theme-medium: #0088AA;       /* Medium variant */
            --theme-glow: rgba(0, 191, 255, 0.4);      /* Glow effect */
            --theme-glow-soft: rgba(0, 191, 255, 0.1); /* Soft glow */
            --theme-glow-text: rgba(0, 191, 255, 0.5); /* Text shadow */
            --theme-glow-text-strong: rgba(0, 191, 255, 0.6); /* Strong text shadow */

            /* FORCE GRADIO INTERNAL VARIABLES */
            --color-accent: var(--theme-primary) !important;
            --color-accent-soft: var(--theme-glow-soft) !important;
            --slider-color: var(--theme-primary) !important;
            --loader-color: var(--theme-primary) !important;
            --checkbox-background-color-selected: var(--theme-primary) !important;
            --checkbox-border-color-selected: var(--theme-primary) !important;
            --checkbox-label-background-fill-hover: transparent !important;
            --button-primary-background-fill: var(--theme-primary) !important;
            --button-primary-background-fill-hover: var(--theme-bright) !important;
            --input-border-color-focus: var(--theme-primary) !important;
            --ring-color: var(--theme-glow) !important;

            /* Structure */
            --primary-50: var(--theme-glow-soft);
            --primary-100: var(--theme-dim);
            --primary-200: var(--theme-medium);
            --primary-300: var(--theme-primary);
            --primary-400: var(--theme-primary);
            --primary-500: var(--theme-primary);
            --primary-600: var(--theme-bright);
            --primary-700: var(--theme-bright);
            --plasma-primary: var(--theme-primary);
            --plasma-glow: var(--theme-glow);

            /* Text */
            --body-text-color: var(--theme-primary);
            --block-label-text-color: var(--theme-primary);
            --plasma-text: var(--theme-primary);

            /* Backgrounds - Pure Black */
            --plasma-bg: #000000;
            --panel-bg: #000000;
        }

        /* ============================================
           TEXTURES & SURFACES - Cyberdeck Materials
           ============================================ */

        /* 1. Woven Carbon Fiber - Structural areas (Sidebar, Headers) */
        .carbon-fiber {
            background:
                radial-gradient(black 15%, transparent 16%) 0 0,
                radial-gradient(black 15%, transparent 16%) 8px 8px,
                radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
                radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
            background-color: #1a1a1a !important;
            background-size: 16px 16px;
        }

        /* 2. Matte Dark Grey - Input/Control areas */
        .matte-plate {
            background-color: #111111 !important;
            background-image: linear-gradient(to bottom, #1a1a1a, #0d0d0d);
            border: 1px solid var(--theme-dim) !important;
        }

        /* 3. Settings Panel - Clean Black */
        #settings-panel {
            background-color: #000000 !important;
            border-left: 2px solid var(--theme-primary) !important;
            padding: 10px !important;
        }

        /* Import monospace fonts */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Fira+Code:wght@400;500;700&display=swap');

        /* SOLID BLACK BASE */
        html, body {
            background-color: #000000 !important;
            min-height: 100vh;
        }

        /* Global Font - Full Monospace Terminal */
        * {
            font-family: 'Fira Code', 'JetBrains Mono', 'Consolas', 'Lucida Console', monospace !important;
            border-radius: 0px !important;
        }

        /* Text glow effect */
        body, button, input, textarea {
            text-shadow: 0 0 4px var(--theme-glow-text);
        }

        /* Main container - Black with theme border */
        .gradio-container {
            background-color: #000000 !important;
            border: 2px solid var(--theme-primary) !important;
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }

        /* Scrollbars - Theme Color */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #000000;
            border: 1px solid var(--theme-dim);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--theme-primary);
            border: none;
            filter: drop-shadow(0 0 5px var(--theme-primary));
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--theme-bright);
        }

        /* Force slider accent color */
        input[type="range"] {
            accent-color: var(--theme-primary) !important;
            filter: drop-shadow(0 0 5px var(--theme-primary));
        }

        /* ============================================
           HUD HEADER - Latency/Token/Memory Counters
           Matte gradient surface like a dashboard module
           ============================================ */
        #hud-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 16px;
            background: linear-gradient(180deg, #1a1a1a 0%, #0a0a0a 100%);
            border: 2px solid var(--theme-primary);
            border-bottom: 3px solid var(--theme-primary);
            margin-bottom: 10px;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 0 15px var(--theme-glow), inset 0 1px 0 rgba(255,255,255,0.05);
        }
        .hud-item {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--theme-primary);
        }
        .hud-label {
            color: var(--theme-dim);
        }
        .hud-value {
            color: var(--theme-primary);
            font-weight: bold;
            text-shadow: 0 0 5px var(--theme-primary);
        }
        .hud-value.warning { color: var(--theme-primary); }
        .hud-value.danger { color: var(--theme-primary); }

        /* ============================================
           EMOTION METER - HUD Bar Gauge
           ============================================ */
        #emotion-meter {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 6px 12px;
            background: #000000;
            border: 1px solid var(--theme-primary);
        }
        #emotion-meter .meter-label {
            color: var(--theme-dim);
            font-size: 0.75em;
            text-transform: uppercase;
            min-width: 60px;
        }
        #emotion-meter .meter-bar {
            flex: 1;
            height: 12px;
            background: #000000;
            border: 1px solid var(--theme-dim);
            position: relative;
            overflow: hidden;
        }
        #emotion-meter .meter-fill {
            height: 100%;
            transition: width 0.3s, background 0.3s;
        }
        #emotion-meter .meter-fill.happy { background: var(--theme-primary); }
        #emotion-meter .meter-fill.neutral { background: var(--theme-primary); }
        #emotion-meter .meter-fill.sad { background: var(--theme-primary); }
        #emotion-meter .meter-fill.angry { background: var(--theme-primary); }
        #emotion-meter .meter-fill.fear { background: var(--theme-primary); }
        #emotion-meter .meter-value {
            color: var(--theme-primary);
            font-size: 0.8em;
            min-width: 40px;
            text-align: right;
        }

        /* ============================================
           PTT STATUS BOX - Active State Indicators
           ============================================ */
        #ptt-status-box {
            padding: 12px;
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
            transition: all 0.2s ease;
            background: #000000;
            border: 2px solid var(--theme-primary);
            color: var(--theme-primary);
            text-transform: uppercase;
            letter-spacing: 2px;
            text-shadow: 0 0 10px var(--theme-primary);
        }

        /* ============================================
           MOBILE PTT BUTTON - Touch-friendly
           ============================================ */
        #mobile-ptt-btn {
            padding: 20px 40px;
            font-size: 1.4em;
            font-weight: bold;
            background: #000000 !important;
            border: 3px solid var(--theme-primary);
            color: var(--theme-primary);
            text-transform: uppercase;
            letter-spacing: 2px;
            text-shadow: 0 0 10px var(--theme-primary);
            box-shadow: none;
            cursor: pointer;
            user-select: none;
            -webkit-user-select: none;
            touch-action: none;
            transition: all 0.2s ease;
        }
        #mobile-ptt-btn:hover {
            background: var(--theme-primary) !important;
            color: #000000 !important;
            text-shadow: none !important;
            box-shadow: 0 0 30px var(--theme-glow);
        }
        #mobile-ptt-btn.recording {
            background: #000000 !important;
            border-color: #FF4444;
            color: #FF4444;
            text-shadow: 0 0 15px #FF4444;
            box-shadow: 0 0 40px rgba(255, 68, 68, 0.6);
            animation: mobile-ptt-pulse 0.5s ease-in-out infinite;
        }
        @keyframes mobile-ptt-pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }

        /* Mobile responsive */
        @media (max-width: 768px) {
            #mobile-ptt-btn {
                padding: 30px 50px;
                font-size: 1.6em;
                width: 100%;
            }
            .hide-on-mobile {
                display: none !important;
            }
        }

        /* RECORDING STATE - Theme pulse */
        @keyframes pulse-recording {
            0%, 100% {
                box-shadow: 0 0 10px var(--theme-primary), inset 0 0 20px var(--theme-glow-soft);
                border-color: var(--theme-primary);
                color: var(--theme-primary);
            }
            50% {
                box-shadow: 0 0 30px var(--theme-bright), inset 0 0 40px var(--theme-glow-soft);
                border-color: var(--theme-bright);
                color: var(--theme-primary);
            }
        }
        .recording {
            animation: pulse-recording 0.8s ease-in-out infinite;
        }

        /* PROCESSING STATE - Theme pulse */
        @keyframes pulse-processing {
            0%, 100% {
                box-shadow: 0 0 10px var(--theme-primary);
                border-color: var(--theme-primary);
            }
            50% {
                box-shadow: 0 0 25px var(--theme-bright);
                border-color: var(--theme-bright);
            }
        }
        .processing {
            animation: pulse-processing 1s ease-in-out infinite;
            border-color: var(--theme-primary) !important;
        }

        /* SPEAKING STATE - Full theme glow */
        @keyframes pulse-burn {
            from { box-shadow: 0 0 10px var(--theme-primary); }
            to { box-shadow: 0 0 25px var(--theme-bright); }
        }
        .speaking {
            animation: pulse-burn 0.5s infinite alternate;
            border-color: var(--theme-primary) !important;
            color: var(--theme-primary) !important;
            text-shadow: 0 0 15px var(--theme-primary) !important;
        }

        #tts-warning {
            color: var(--theme-primary);
            font-size: 0.9em;
            text-transform: uppercase;
        }
        #audio-response {
            min-height: 60px;
            border: 1px solid var(--theme-primary);
            background: #000000;
        }
        #audio-response audio {
            width: 100%;
        }

        /* ============================================
           SYSTEM PROTOCOLS - Tool Chain Styling
           ============================================ */
        .tool-chain-container {
            position: relative;
            padding-left: 12px;
            margin: 8px 0;
        }
        .tool-chain-container::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 3px;
            background: linear-gradient(to bottom, var(--theme-primary), var(--theme-bright));
        }

        /* Protocol Block - Black text on theme header */
        .tool-call-block,
        #main-chatbot .tool-call-block,
        #main-chatbot [role="assistant"] .tool-call-block {
            background: #000000 !important;
            border: 2px solid var(--theme-primary) !important;
            padding: 0 !important;
            margin: 8px 0 !important;
            font-size: 0.9em !important;
            position: relative !important;
            z-index: 10 !important;
            box-shadow: 0 0 10px var(--theme-glow) !important;
            isolation: isolate;
        }
        .tool-call-block.tool-chain-continues::after {
            content: '▼ CHAIN';
            position: absolute;
            bottom: -18px;
            left: 50%;
            transform: translateX(-50%);
            color: var(--theme-primary);
            font-size: 10px;
            z-index: 1;
            background: #000000;
            padding: 2px 8px;
            border: 1px solid var(--theme-primary);
        }

        /* Protocol Step Badge - Theme style */
        .tool-step-badge {
            background: var(--theme-primary) !important;
            color: #000000 !important;
            padding: 4px 10px !important;
            font-size: 0.75em !important;
            margin-right: 8px !important;
            font-weight: bold !important;
            border: none !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Protocol Header - Theme bar */
        .tool-call-header,
        #main-chatbot .tool-call-header {
            background: var(--theme-primary) !important;
            color: #000000 !important;
            font-weight: bold !important;
            margin: 0 !important;
            padding: 8px 12px !important;
            display: flex !important;
            align-items: center !important;
            text-shadow: none !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Protocol Args - Theme text on black */
        .tool-call-args,
        #main-chatbot .tool-call-args {
            color: var(--theme-primary) !important;
            margin: 0 !important;
            padding: 8px 12px !important;
            font-size: 0.85em !important;
            word-break: break-word !important;
            text-shadow: none !important;
            font-weight: normal !important;
            background: #000000 !important;
            border-top: 1px solid var(--theme-dim) !important;
        }

        /* Protocol Result - Dim theme output */
        .tool-call-result,
        #main-chatbot .tool-call-result,
        #main-chatbot [role="assistant"] .tool-call-result {
            background: #000000 !important;
            border-top: 2px solid var(--theme-medium) !important;
            border-left: none !important;
            color: var(--theme-medium) !important;
            margin: 0 !important;
            padding: 8px 12px !important;
            font-size: 0.85em !important;
            word-break: break-word !important;
            position: relative !important;
            z-index: 10 !important;
            text-shadow: none !important;
            font-weight: normal !important;
        }

        /* Copy button - Theme style */
        .message-copy-btn {
            position: absolute;
            top: 8px;
            right: 8px;
            background: #000000;
            border: 1px solid var(--theme-primary);
            color: var(--theme-primary);
            cursor: pointer;
            padding: 4px 8px;
            font-size: 12px;
            opacity: 0;
            transition: all 0.2s;
        }
        .message-wrapper:hover .message-copy-btn {
            opacity: 1;
        }
        .message-copy-btn:hover {
            background: var(--theme-primary);
            color: #000000;
        }
        .message-copy-btn.copied {
            background: var(--theme-primary);
            color: #000000;
        }

        /* Expandable messages */
        .message-expandable {
            max-height: 300px;
            overflow: hidden;
            position: relative;
        }
        .message-expandable.expanded {
            max-height: none;
        }
        .message-expandable:not(.expanded)::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: linear-gradient(transparent, #000000);
            pointer-events: none;
        }
        .expand-btn {
            display: block;
            text-align: center;
            padding: 8px;
            color: var(--theme-primary);
            cursor: pointer;
            font-size: 0.9em;
            background: #000000;
            margin-top: 4px;
            transition: background 0.2s;
            border: 1px solid var(--theme-primary);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .expand-btn:hover {
            background: var(--theme-primary);
            color: #000000;
        }

        /* Message timestamps */
        .message-timestamp {
            font-size: 0.7em;
            color: var(--theme-dim);
            margin-top: 4px;
            text-align: right;
            opacity: 0.7;
        }

        /* Thinking indicator - Heating coil bar */
        .thinking-indicator {
            color: var(--theme-primary);
            font-style: normal;
            font-size: 0.9em;
            margin: 4px 0;
            display: flex;
            align-items: center;
            gap: 12px;
            text-transform: uppercase;
        }
        .thinking-bar-container {
            width: 150px;
            height: 6px;
            background: #000000;
            border: 1px solid var(--theme-primary);
            overflow: hidden;
            position: relative;
        }
        .thinking-bar {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            width: 40%;
            background: linear-gradient(90deg, transparent, var(--theme-primary), var(--theme-bright), transparent);
            animation: thinking-slide 1.2s infinite linear;
        }
        @keyframes thinking-slide {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(250%); }
        }

        /* Token usage - HUD style */
        .token-usage {
            font-size: 0.75em;
            color: var(--theme-primary);
            padding: 6px 12px;
            background: #000000;
            display: inline-flex;
            gap: 16px;
            border: 1px solid var(--theme-primary);
            text-transform: uppercase;
        }
        .token-usage span {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .token-usage .token-in { color: var(--theme-primary); }
        .token-usage .token-out { color: var(--theme-primary); }
        .token-usage .token-total { color: var(--theme-medium); }

        /* Mood indicator - HUD style */
        .mood-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: #000000;
            font-size: 0.8em;
            border: 1px solid var(--theme-primary);
            text-transform: uppercase;
        }
        .mood-indicator .mood-text { color: var(--theme-primary); }

        /* ============================================
           PANELS - Black with Theme Borders
           ============================================ */

        /* All panels - Solid black */
        .block, .panel,
        .gradio-accordion,
        .gradio-group {
            background-color: #000000 !important;
            border: 1px solid var(--theme-primary) !important;
        }

        /* Input areas - Black */
        .gradio-dropdown,
        .gradio-textbox {
            background-color: #000000 !important;
        }

        /* Chatbot container - Grey gradient background */
        .gradio-chatbot {
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 50%, #0d0d0d 100%) !important;
            border: 1px solid var(--theme-primary) !important;
        }

        /* ============================================
           CHATBOT MESSAGE STYLING - Visual Separation
           User: Black | Assistant: Pure black void
           ============================================ */

        /* User messages - Black background */
        #main-chatbot [role="user"],
        #main-chatbot .role-user,
        #main-chatbot .user {
            background: #000000 !important;
            border: 1px solid var(--theme-medium) !important;
            border-left: 3px solid var(--theme-dim) !important;
        }
        #main-chatbot [role="user"] > p,
        #main-chatbot [role="user"] > span,
        #main-chatbot [role="user"] > div:not(.tool-call-block):not(.tool-chain-container) {
            color: var(--theme-primary) !important;
        }

        /* Assistant messages - Pure black void (AI emanates from darkness) */
        #main-chatbot [role="assistant"],
        #main-chatbot .role-assistant,
        #main-chatbot .bot {
            background: #000000 !important;
            border: 1px solid var(--theme-primary) !important;
            border-left: 3px solid var(--theme-primary) !important;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
        }
        #main-chatbot [role="assistant"] > p,
        #main-chatbot [role="assistant"] > span,
        #main-chatbot [role="assistant"] > div:not(.tool-call-block):not(.tool-chain-container):not(.message-expandable) {
            color: var(--theme-primary) !important;
        }
        #main-chatbot [role="assistant"] .message-expandable {
            color: var(--theme-primary) !important;
        }

        /* Remove inner box styling */
        #main-chatbot .message-bubble-border,
        #main-chatbot .message-content,
        #main-chatbot .message-row,
        #main-chatbot .message-wrap,
        #main-chatbot .wrap {
            background: transparent !important;
            border: none !important;
        }

        /* ============================================
           INPUT AREA - Recessed Terminal Entry Slot
           ============================================ */
        #msg-input,
        #msg-input textarea {
            background-color: #000000 !important;
            border: 2px solid var(--theme-primary) !important;
            color: var(--theme-primary) !important;
        }
        #msg-input:focus-within,
        #msg-input textarea:focus {
            box-shadow: 0 0 15px var(--theme-glow) !important;
            background-color: #000000 !important;
        }

        /* ============================================
           TAB AND ACCORDION BORDERS - Double border style
           ============================================ */

        .tabs {
            border: 1px solid var(--theme-primary) !important;
            background: #000000 !important;
            padding: 4px !important;
        }
        .tab-nav {
            border: 1px solid var(--theme-primary) !important;
            background: #000000 !important;
        }
        .tab-nav button {
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 50%, #0d0d0d 100%) !important;
            border: 1px solid var(--theme-primary) !important;
            color: var(--theme-primary) !important;
            transition: all 0.2s ease !important;
        }
        .tab-nav button:hover {
            background: var(--theme-primary) !important;
            color: #000000 !important;
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }
        .tab-nav button.selected {
            background: var(--theme-primary) !important;
            color: #000000 !important;
        }
        .tabitem {
            border-color: var(--theme-primary) !important;
        }

        .gradio-accordion {
            border: 1px solid var(--theme-primary) !important;
        }
        /* Accordion headers - Grey gradient for dimensionality */
        .gradio-accordion > .label-wrap {
            border-bottom: 1px solid var(--theme-primary) !important;
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 50%, #0d0d0d 100%) !important;
            color: var(--theme-primary) !important;
            transition: all 0.2s ease !important;
        }
        .gradio-accordion > .label-wrap:hover {
            background: var(--theme-primary) !important;
            color: #000000 !important;
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }
        .gradio-accordion > .label-wrap:hover * {
            color: #000000 !important;
        }

        .gradio-group {
            border: 1px solid var(--theme-primary) !important;
            background: #000000 !important;
            padding: 4px !important;
        }

        /* Audio component - Double border style */
        .gradio-audio,
        [data-testid="audio"] {
            border: 1px solid var(--theme-primary) !important;
            background: #000000 !important;
            padding: 4px !important;
        }
        .gradio-audio audio,
        [data-testid="audio"] audio {
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 50%, #0d0d0d 100%) !important;
            border: 1px solid var(--theme-primary) !important;
        }
        /* Audio component buttons and dropdowns - grey gradient */
        .gradio-audio button,
        .gradio-audio select,
        .gradio-audio .dropdown,
        [data-testid="audio"] button,
        [data-testid="audio"] select {
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 50%, #0d0d0d 100%) !important;
            border: 1px solid var(--theme-primary) !important;
            color: var(--theme-primary) !important;
        }
        .gradio-audio button:hover,
        .gradio-audio select:hover,
        [data-testid="audio"] button:hover,
        [data-testid="audio"] select:hover {
            background: var(--theme-primary) !important;
            color: #000000 !important;
        }

        /* Image component - Double border style */
        .gradio-image,
        [data-testid="image"] {
            border: 1px solid var(--theme-primary) !important;
            background: #000000 !important;
            padding: 4px !important;
        }

        /* File component - Double border style */
        .gradio-file,
        [data-testid="file"] {
            border: 1px solid var(--theme-primary) !important;
            background: #000000 !important;
            padding: 4px !important;
        }

        /* All inputs - Double border style */
        .gradio-textbox {
            border: 1px solid var(--theme-primary) !important;
            background: #000000 !important;
            padding: 4px !important;
            transition: all 0.2s ease !important;
        }
        .gradio-textbox textarea,
        .gradio-textbox input {
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 50%, #0d0d0d 100%) !important;
            border: 1px solid var(--theme-primary) !important;
        }
        .gradio-textbox:focus-within {
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }
        .gradio-textbox:hover:not(:focus-within) {
            box-shadow: 0 0 10px var(--theme-glow) !important;
        }

        /* ============================================
           DROPDOWNS - Matching Accordion Style Exactly
           Outer cyan border > black padding > grey gradient inner
           ============================================ */
        .gradio-dropdown {
            border: 1px solid var(--theme-primary) !important;
            background: #000000 !important;
            padding: 4px !important;
            transition: all 0.2s ease !important;
        }
        .gradio-dropdown:focus-within {
            box-shadow: 0 0 15px var(--theme-glow) !important;
        }
        /* Dropdown clickable area - Grey gradient with inner cyan border */
        .gradio-dropdown > .wrap,
        .gradio-dropdown > div:first-child,
        .gradio-dropdown [data-testid="dropdown"] {
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 50%, #0d0d0d 100%) !important;
            border: 1px solid var(--theme-primary) !important;
            transition: all 0.2s ease !important;
        }
        /* Dropdown hover - Solid lapis blue with black text */
        .gradio-dropdown:hover > .wrap,
        .gradio-dropdown:hover > div:first-child,
        .gradio-dropdown:hover [data-testid="dropdown"] {
            background: var(--theme-primary) !important;
            color: #000000 !important;
        }
        .gradio-dropdown:hover {
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }
        .gradio-dropdown:hover input,
        .gradio-dropdown:hover span {
            color: #000000 !important;
        }
        /* Dropdown input field */
        .gradio-dropdown input,
        .gradio-dropdown [data-testid="textbox"] {
            border: none !important;
            background: transparent !important;
        }
        /* Dropdown arrow - theme color */
        .gradio-dropdown svg,
        .gradio-dropdown .wrap svg,
        .gradio-dropdown button svg {
            color: var(--theme-primary) !important;
            fill: var(--theme-primary) !important;
            transition: all 0.2s ease !important;
        }
        .gradio-dropdown:hover svg,
        .gradio-dropdown:hover .wrap svg,
        .gradio-dropdown:hover button svg {
            color: #000000 !important;
            fill: #000000 !important;
        }
        /* Dropdown menu when open - cyan border with grey gradient */
        .gradio-dropdown ul,
        .gradio-dropdown [role="listbox"],
        .gradio-dropdown .options,
        ul[role="listbox"],
        div[role="listbox"] {
            border: 2px solid var(--theme-primary) !important;
            background: linear-gradient(180deg, #1a1a1a 0%, #0d0d0d 100%) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.8), 0 0 15px var(--theme-glow) !important;
        }
        /* Dropdown items - transparent by default with theme text */
        .gradio-dropdown li,
        .gradio-dropdown [role="option"],
        ul[role="listbox"] li,
        div[role="listbox"] [role="option"] {
            background: transparent !important;
            color: var(--theme-primary) !important;
            transition: all 0.15s ease !important;
        }
        /* Dropdown items inner text - theme color */
        .gradio-dropdown li *,
        .gradio-dropdown [role="option"] *,
        ul[role="listbox"] li *,
        div[role="listbox"] [role="option"] * {
            color: var(--theme-primary) !important;
        }
        /* Dropdown items - solid lapis on hover with black text */
        .gradio-dropdown li:hover,
        .gradio-dropdown [role="option"]:hover,
        ul[role="listbox"] li:hover,
        div[role="listbox"] [role="option"]:hover {
            background: var(--theme-primary) !important;
            color: #000000 !important;
        }
        .gradio-dropdown li:hover *,
        .gradio-dropdown [role="option"]:hover *,
        ul[role="listbox"] li:hover *,
        div[role="listbox"] [role="option"]:hover * {
            color: #000000 !important;
        }
        /* Selected dropdown item */
        .gradio-dropdown li.selected,
        .gradio-dropdown [role="option"][aria-selected="true"],
        ul[role="listbox"] li.selected,
        div[role="listbox"] [role="option"][aria-selected="true"] {
            background: linear-gradient(180deg, #333333 0%, #222222 100%) !important;
            border-left: 3px solid var(--theme-primary) !important;
        }
        /* Selected dropdown item HOVER - must also change background */
        .gradio-dropdown li.selected:hover,
        .gradio-dropdown [role="option"][aria-selected="true"]:hover,
        ul[role="listbox"] li.selected:hover,
        div[role="listbox"] [role="option"][aria-selected="true"]:hover {
            background: var(--theme-primary) !important;
            color: #000000 !important;
        }
        .gradio-dropdown li.selected:hover *,
        .gradio-dropdown [role="option"][aria-selected="true"]:hover *,
        ul[role="listbox"] li.selected:hover *,
        div[role="listbox"] [role="option"][aria-selected="true"]:hover * {
            color: #000000 !important;
        }

        /* Labels - Theme */
        label, .label-wrap, .svelte-1gfkn6j {
            color: var(--theme-primary) !important;
        }

        /* Info/helper text below components - Theme color */
        .wrap > span:not(.token),
        .wrap > p,
        .info-text,
        .block-info,
        .gradio-container span[data-testid],
        .gradio-textbox .wrap > div:last-child,
        .gradio-dropdown .wrap > div:last-child,
        .gradio-slider .wrap > div:last-child,
        .gradio-checkbox .wrap > span,
        .gradio-radio .wrap > span:not(.selected),
        form span,
        .form span,
        .block span:not(.token):not([role]),
        [class*="info"],
        [class*="hint"],
        [class*="desc"] {
            color: var(--theme-primary) !important;
        }

        /* VAD Indicator - Theme style */
        .vad-indicator {
            background: repeating-linear-gradient(
                45deg,
                #000000,
                #000000 10px,
                var(--theme-bg-stripe, #003344) 10px,
                var(--theme-bg-stripe, #003344) 20px
            );
            border: 1px solid var(--theme-dim);
            color: var(--theme-dim);
        }
        .vad-indicator.active {
            background: var(--theme-primary);
            box-shadow: 0 0 30px var(--theme-primary);
            color: #000;
            font-weight: bold;
        }

        /* ============================================
           PHYSICAL SWITCH CHECKBOXES
           Black unchecked, grey hover, theme filled
           FORCE override any Gradio error/validation states
           ============================================ */
        input[type="checkbox"],
        input[type="checkbox"]:invalid,
        input[type="checkbox"]:required,
        input[type="checkbox"].error,
        .checkbox-container input[type="checkbox"],
        [data-testid="checkbox"] input {
            width: 20px;
            height: 20px;
            border: 2px solid var(--theme-primary) !important;
            border-radius: 0 !important;
            appearance: none;
            -webkit-appearance: none;
            cursor: pointer;
            background: #000000 !important;
            background-color: #000000 !important;
            transition: all 0.15s ease;
            outline: none !important;
            box-shadow: none !important;
        }
        input[type="checkbox"]:checked,
        input[type="checkbox"]:checked:invalid,
        [data-testid="checkbox"] input:checked {
            background: var(--theme-primary) !important;
            background-color: var(--theme-primary) !important;
            border-color: var(--theme-primary) !important;
            box-shadow: 0 0 10px var(--theme-primary) !important;
        }
        input[type="checkbox"]:hover:not(:checked) {
            background: var(--theme-primary) !important;
            background-color: var(--theme-primary) !important;
            border-color: var(--theme-primary) !important;
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }
        /* Kill any red/error borders Gradio might add */
        input[type="checkbox"]:focus,
        input[type="checkbox"]:focus-visible {
            border-color: var(--theme-primary) !important;
            outline: none !important;
            box-shadow: 0 0 8px var(--theme-glow) !important;
        }

        /* Force Radio Buttons to Theme */
        input[type="radio"] {
            background: #000000 !important;
            background-color: #000000 !important;
        }
        input[type="radio"]:checked {
            background: var(--theme-primary) !important;
            background-color: var(--theme-primary) !important;
            border-color: var(--theme-primary) !important;
            box-shadow: 0 0 10px var(--theme-primary);
        }
        input[type="radio"]:hover:not(:checked) {
            background: #333333 !important;
            background-color: #333333 !important;
            border-color: var(--theme-primary);
            box-shadow: 0 0 5px var(--theme-glow-text);
        }

        /* Force Gradio Slider Track and Thumb to Theme Color */
        input[type="range"] {
            transition: all 0.2s ease !important;
        }
        input[type="range"]::-webkit-slider-runnable-track {
            background: linear-gradient(to right, var(--theme-primary), var(--theme-primary)) !important;
            border: 1px solid var(--theme-primary) !important;
        }
        input[type="range"]::-webkit-slider-thumb {
            background: var(--theme-primary) !important;
            box-shadow: 0 0 8px var(--theme-primary);
            transition: all 0.2s ease !important;
        }
        input[type="range"]:hover::-webkit-slider-thumb {
            box-shadow: 0 0 20px var(--theme-glow) !important;
            transform: scale(1.2);
        }
        input[type="range"]::-moz-range-track {
            background: var(--theme-primary) !important;
            border: 1px solid var(--theme-primary) !important;
        }
        input[type="range"]::-moz-range-thumb {
            background: var(--theme-primary) !important;
            box-shadow: 0 0 8px var(--theme-primary);
            transition: all 0.2s ease !important;
        }
        input[type="range"]:hover::-moz-range-thumb {
            box-shadow: 0 0 20px var(--theme-glow) !important;
            transform: scale(1.2);
        }
        /* Slider container - double border style, NO hover background */
        .gradio-slider {
            border: 1px solid var(--theme-primary) !important;
            background: #000000 !important;
            padding: 4px !important;
        }
        .gradio-slider:hover,
        .gradio-slider:hover > *,
        .gradio-slider .label-wrap,
        .gradio-slider .label-wrap:hover,
        .gradio-slider:focus-within {
            background: #000000 !important;
            box-shadow: none !important;
        }
        /* Slider focus state - use theme color, not blue */
        input[type="range"]:focus {
            outline: none !important;
            box-shadow: 0 0 0 2px var(--theme-glow) !important;
        }
        input[type="range"]:focus::-webkit-slider-thumb {
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }
        input[type="range"]:active::-webkit-slider-thumb {
            background: var(--theme-bright) !important;
            transform: scale(1.3);
        }
        input[type="range"]:focus::-moz-range-thumb {
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }
        input[type="range"]:active::-moz-range-thumb {
            background: var(--theme-bright) !important;
            transform: scale(1.3);
        }

        /* Checkbox containers - NO hover effects on container/label, ONLY on input */
        .gradio-checkbox,
        .gradio-checkboxgroup,
        [data-testid="checkbox"] {
            background: #000000 !important;
            border: none !important;
        }
        .gradio-checkbox:hover,
        .gradio-checkboxgroup:hover,
        [data-testid="checkbox"]:hover,
        .gradio-checkbox .label-wrap:hover,
        .gradio-checkboxgroup .label-wrap:hover,
        .gradio-checkbox label:hover,
        .gradio-checkboxgroup > .wrap:hover,
        .gradio-checkbox > .wrap:hover {
            background: #000000 !important;
            box-shadow: none !important;
        }
        /* Checkbox labels - KEEP theme color on hover, no background change */
        .gradio-checkbox label,
        .gradio-checkbox span:not(input),
        .gradio-checkboxgroup label span,
        [data-testid="checkbox"] label,
        [data-testid="checkbox"] span {
            color: var(--theme-primary) !important;
            background: transparent !important;
        }
        .gradio-checkbox:hover label,
        .gradio-checkbox:hover span:not(input),
        [data-testid="checkbox"]:hover label,
        [data-testid="checkbox"]:hover span {
            color: var(--theme-primary) !important;
            background: transparent !important;
        }

        /* Override Gradio's internal toggle/switch styling */
        .gr-check-radio input:checked,
        .gr-input-label input:checked,
        [data-testid="checkbox"] input:checked {
            background-color: var(--theme-primary) !important;
            border-color: var(--theme-primary) !important;
        }

        /* ============================================
           RADIO BUTTON GROUP FIX - Selected State
           Black text on theme background when selected
           ============================================ */

        /* Radio button labels - theme text by default */
        .gr-radio label,
        [data-testid="radio"] label,
        .wrap label span,
        label.svelte-1gfkn6j span {
            color: var(--theme-primary) !important;
        }

        /* Radio button group - selected item styling */
        .gr-radio label.selected,
        .gr-radio label[data-selected="true"],
        [data-testid="radio"] label.selected,
        [data-testid="radio"] label[data-selected="true"],
        .wrap label.selected,
        .wrap label[data-selected="true"],
        input[type="radio"]:checked + label,
        input[type="radio"]:checked ~ label,
        input[type="radio"]:checked + span,
        .gr-radio .selected,
        label.selected span,
        label[data-selected="true"] span {
            background: var(--theme-primary) !important;
            color: #000000 !important;
        }

        /* Force selected radio button text to be black */
        .gr-radio label.selected span,
        .gr-radio label[data-selected="true"] span,
        [data-testid="radio"] label.selected span,
        [data-testid="radio"] label[data-selected="true"] span {
            color: #000000 !important;
        }

        /* Gradio 4.x Radio specific - uses button-like styling */
        .gradio-radio .wrap .selected,
        .gradio-radio label.selected,
        .gradio-radio [data-testid="radio"] label.selected,
        button[role="radio"][aria-checked="true"],
        [role="radiogroup"] button[aria-checked="true"],
        [role="radiogroup"] label.selected {
            background: var(--theme-primary) !important;
            color: #000000 !important;
        }

        /* Handle the inner span text for selected radio */
        .gradio-radio .wrap .selected span,
        .gradio-radio label.selected span,
        button[role="radio"][aria-checked="true"] span,
        [role="radiogroup"] button[aria-checked="true"] span {
            color: #000000 !important;
        }

        /* Unselected radio buttons - black bg with theme border */
        .gr-radio label:not(.selected),
        [data-testid="radio"] label:not(.selected),
        button[role="radio"][aria-checked="false"],
        [role="radiogroup"] button[aria-checked="false"],
        [role="radiogroup"] label:not(.selected) {
            background: #000000 !important;
            border: 1px solid var(--theme-primary) !important;
            color: var(--theme-primary) !important;
            transition: all 0.2s ease !important;
        }
        /* Unselected radio buttons - solid lapis on hover */
        .gr-radio label:not(.selected):hover,
        [data-testid="radio"] label:not(.selected):hover,
        button[role="radio"][aria-checked="false"]:hover,
        [role="radiogroup"] button[aria-checked="false"]:hover,
        [role="radiogroup"] label:not(.selected):hover,
        .gradio-radio label:hover,
        .gradio-radio button:hover {
            background: var(--theme-primary) !important;
            color: #000000 !important;
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }
        .gr-radio label:not(.selected):hover span,
        [data-testid="radio"] label:not(.selected):hover span,
        button[role="radio"][aria-checked="false"]:hover span,
        [role="radiogroup"] button[aria-checked="false"]:hover span,
        .gradio-radio label:hover span,
        .gradio-radio button:hover span {
            color: #000000 !important;
        }

        /* ============================================
           HOVER TEXT FIX - Radio and CheckboxGroup only
           Regular checkboxes keep cyan text (black background)
           ============================================ */
        .gradio-radio label:hover,
        .gradio-radio label:hover *,
        .gradio-radio button:hover,
        .gradio-radio button:hover *,
        [role="radiogroup"] label:hover,
        [role="radiogroup"] label:hover *,
        [role="radiogroup"] button:hover,
        [role="radiogroup"] button:hover * {
            color: #000000 !important;
            fill: #000000 !important;
        }
        /* CheckboxGroup (multi-select) labels only */
        .gradio-checkboxgroup > div > label:hover,
        .gradio-checkboxgroup > div > label:hover * {
            color: #000000 !important;
        }

        /* Radio button container - double border style, NO container-level hover */
        .gradio-radio,
        [data-testid="radio"],
        [role="radiogroup"] {
            border: 1px solid var(--theme-primary) !important;
            background: #000000 !important;
            padding: 4px !important;
        }
        .gradio-radio:hover,
        [data-testid="radio"]:hover,
        [role="radiogroup"]:hover {
            background: #000000 !important;
            box-shadow: none !important;
        }
        /* Radio container inner wrapper - no hover effect */
        .gradio-radio > .wrap,
        .gradio-radio > div,
        [role="radiogroup"] > div {
            background: transparent !important;
        }
        .gradio-radio > .wrap:hover,
        .gradio-radio > div:hover,
        [role="radiogroup"] > div:hover {
            background: transparent !important;
            box-shadow: none !important;
        }

        /* ============================================
           BUTTON FIXES - Grey bg, Lapis hover with black text
           ============================================ */

        /* Primary buttons - Grey Gradient, Solid Lapis on Hover */
        button.primary {
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 50%, #0d0d0d 100%) !important;
            color: var(--theme-primary) !important;
            border: 2px solid var(--theme-primary) !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: bold;
            text-shadow: none !important;
            transition: all 0.2s ease !important;
        }
        button.primary:hover {
            background: var(--theme-primary) !important;
            color: #000000 !important;
            box-shadow: 0 0 25px var(--theme-glow) !important;
            text-shadow: none !important;
        }

        /* Secondary buttons - Grey Gradient, Solid Lapis on Hover */
        button.secondary {
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 50%, #0d0d0d 100%) !important;
            color: var(--theme-primary) !important;
            border: 1px solid var(--theme-primary) !important;
            transition: all 0.2s ease !important;
        }
        button.secondary:hover {
            background: var(--theme-primary) !important;
            color: #000000 !important;
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }

        /* All other buttons - Grey Gradient, Solid Lapis on Hover */
        button:not(.primary):not(.secondary):not(#mobile-ptt-btn) {
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 50%, #0d0d0d 100%) !important;
            color: var(--theme-primary) !important;
            border: 1px solid var(--theme-primary) !important;
            transition: all 0.2s ease !important;
        }
        button:not(.primary):not(.secondary):not(#mobile-ptt-btn):hover {
            background: var(--theme-primary) !important;
            color: #000000 !important;
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }

        /* ============================================
           TEXT COLOR FIX - Theme Color
           ============================================ */

        /* Override body text to theme color */
        body, p, span, div {
            color: var(--theme-primary) !important;
        }

        /* Labels - theme color */
        label, .label-wrap, .svelte-1gfkn6j, h1, h2, h3, h4, h5, h6 {
            color: var(--theme-primary) !important;
        }

        /* Input text - theme color */
        input, textarea {
            color: var(--theme-primary) !important;
        }

        /* Native select elements - Theme styling */
        select {
            color: var(--theme-primary) !important;
            background: #000000 !important;
            border: 2px solid var(--theme-primary) !important;
            transition: all 0.2s ease !important;
        }
        select:hover {
            background: var(--theme-primary) !important;
            color: #000000 !important;
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }
        select:focus {
            border-color: var(--theme-bright) !important;
            box-shadow: 0 0 15px var(--theme-glow) !important;
        }

        /* Placeholder text - dimmer theme */
        input::placeholder, textarea::placeholder {
            color: var(--theme-medium) !important;
        }

        /* Chatbot messages */
        #main-chatbot [role="user"] > p,
        #main-chatbot [role="user"] > span,
        #main-chatbot [role="user"] > div:not(.tool-call-block):not(.tool-chain-container) {
            color: var(--theme-primary) !important;
        }
        #main-chatbot [role="assistant"] > p,
        #main-chatbot [role="assistant"] > span,
        #main-chatbot [role="assistant"] > div:not(.tool-call-block):not(.tool-chain-container):not(.message-expandable) {
            color: var(--theme-primary) !important;
        }
        #main-chatbot [role="assistant"] .message-expandable {
            color: var(--theme-primary) !important;
        }

        /* ============================================
           GRADIO FOOTER - Double border style with hover
           ============================================ */
        footer {
            border: 1px solid var(--theme-primary) !important;
            background: #000000 !important;
            padding: 4px !important;
        }
        footer a, footer span, .footer, .built-with {
            color: var(--theme-primary) !important;
        }
        /* Footer links - grey gradient with inner border */
        footer a {
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 50%, #0d0d0d 100%) !important;
            border: 1px solid var(--theme-primary) !important;
            padding: 4px 8px !important;
            text-decoration: none !important;
            transition: all 0.2s ease !important;
        }
        footer a:hover {
            background: var(--theme-primary) !important;
            color: #000000 !important;
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }

        /* ============================================
           ALL BUTTONS - Grey Gradient, Solid Lapis Hover
           Override Gradio's default button styling completely
           ============================================ */
        button, .gr-button, [class*="button"] {
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 50%, #0d0d0d 100%) !important;
            background-color: #1a1a1a !important;
            color: var(--theme-primary) !important;
            border: 1px solid var(--theme-primary) !important;
            transition: all 0.2s ease !important;
        }
        button:hover, .gr-button:hover, [class*="button"]:hover {
            background: var(--theme-primary) !important;
            background-color: var(--theme-primary) !important;
            color: #000000 !important;
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }
        button:active, .gr-button:active, [class*="button"]:active {
            background: var(--theme-bright) !important;
            color: #000000 !important;
        }

        /* ============================================
           UNIVERSAL HOVER TEXT COLOR FIX
           Force ALL text/icons to black on hover
           ============================================ */

        /* Buttons - all text and icons black on hover (fill AND stroke for SVGs) */
        button:hover *,
        .gr-button:hover *,
        [class*="button"]:hover * {
            color: #000000 !important;
            fill: #000000 !important;
            stroke: #000000 !important;
        }

        /* Accordion headers - all content black on hover */
        .gradio-accordion > .label-wrap:hover,
        .gradio-accordion > .label-wrap:hover *,
        .gradio-accordion > .label-wrap:hover span,
        .gradio-accordion > .label-wrap:hover svg {
            color: #000000 !important;
            fill: #000000 !important;
            stroke: #000000 !important;
        }

        /* Dropdowns - all content black on hover */
        .gradio-dropdown:hover *,
        .gradio-dropdown:hover span,
        .gradio-dropdown:hover input,
        .gradio-dropdown:hover svg {
            color: #000000 !important;
            fill: #000000 !important;
            stroke: #000000 !important;
        }

        /* Checkbox labels - KEEP theme color on hover (no background change on label) */
        .gr-checkbox:hover label,
        .gr-checkbox:hover span,
        [data-testid="checkbox"]:hover label,
        [data-testid="checkbox"]:hover span,
        .gradio-checkbox:hover label,
        .gradio-checkbox:hover span {
            color: var(--theme-primary) !important;
            background: transparent !important;
        }

        /* CheckboxGroup item labels - BLACK text on hover */
        .gradio-checkboxgroup label:hover,
        .gradio-checkboxgroup label:hover span,
        [data-testid="checkbox-group"] label:hover,
        [data-testid="checkbox-group"] label:hover span {
            background: var(--theme-primary) !important;
            color: #000000 !important;
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }

        /* Radio buttons - black text when selected/hovered on the button itself */
        .gradio-radio button.selected,
        .gradio-radio button:hover,
        [role="radiogroup"] button.selected,
        [role="radiogroup"] button:hover {
            color: #000000 !important;
            background: var(--theme-primary) !important;
        }

        /* File upload areas - styling and hover (glow effect, not solid background) */
        .gradio-file,
        .gradio-image,
        [data-testid="file"],
        [data-testid="image"] {
            border: 1px solid var(--theme-primary) !important;
            background: #000000 !important;
            transition: all 0.2s ease !important;
        }
        .gradio-file:hover,
        .gradio-image:hover,
        [data-testid="file"]:hover,
        [data-testid="image"]:hover {
            border-color: var(--theme-bright) !important;
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }

        /* Tab buttons - black on hover and selected */
        .tab-nav button:hover *,
        .tab-nav button.selected * {
            color: #000000 !important;
        }

        /* Footer links - black on hover */
        footer a:hover,
        footer a:hover * {
            color: #000000 !important;
        }

        /* Select elements - black on hover */
        select:hover {
            color: #000000 !important;
        }

        /* ============================================
           KILL ALL RED/ERROR STATES
           Force theme colors everywhere, no validation red
           ============================================ */
        *:invalid,
        *:required,
        *.error,
        *[aria-invalid="true"],
        .has-error *,
        .error-border,
        [class*="error"],
        [class*="invalid"] {
            border-color: var(--theme-primary) !important;
            outline-color: var(--theme-primary) !important;
        }

        /* Override Gradio's specific error classes */
        .gr-box.error,
        .gr-input.error,
        .gr-check-radio.error,
        .gradio-container [class*="error"]:not(.tool-call-block),
        .gradio-container [class*="invalid"] {
            border-color: var(--theme-primary) !important;
            box-shadow: none !important;
        }

        /* SVG icons inside checkboxes - force theme color */
        input[type="checkbox"] + svg,
        input[type="checkbox"] ~ svg,
        [data-testid="checkbox"] svg {
            color: var(--theme-primary) !important;
            fill: var(--theme-primary) !important;
            stroke: var(--theme-primary) !important;
        }

        /* ============================================
           STOP/DELETE BUTTONS (variant="stop")
           Blue glow background, black icon on hover
           ============================================ */
        button.stop,
        button[variant="stop"],
        .gradio-button.stop,
        .gr-button-stop {
            background: #000000 !important;
            border: 1px solid var(--theme-primary) !important;
            color: var(--theme-primary) !important;
            transition: all 0.2s ease !important;
        }
        button.stop:hover,
        button[variant="stop"]:hover,
        .gradio-button.stop:hover,
        .gr-button-stop:hover {
            background: var(--theme-primary) !important;
            color: #000000 !important;
            box-shadow: 0 0 20px var(--theme-glow) !important;
        }
        button.stop:hover *,
        button[variant="stop"]:hover *,
        .gradio-button.stop:hover *,
        .gr-button-stop:hover * {
            color: #000000 !important;
            fill: #000000 !important;
        }

        /* ============================================
           ACCORDION COLLAPSE BUTTON - Black on hover
           ============================================ */
        .gradio-accordion .icon,
        .gradio-accordion svg.icon,
        .gradio-accordion > .label-wrap svg {
            color: var(--theme-primary) !important;
            fill: var(--theme-primary) !important;
            transition: all 0.2s ease !important;
        }
        .gradio-accordion > .label-wrap:hover svg,
        .gradio-accordion > .label-wrap:hover .icon {
            color: #000000 !important;
            fill: #000000 !important;
        }

        /* ============================================
           CHATBOT ACTION BUTTONS (download, delete, etc)
           Precise hover - only the icon button, not the container
           ============================================ */
        #main-chatbot button,
        .chatbot button,
        [data-testid="chatbot"] button {
            background: transparent !important;
            border: 1px solid var(--theme-primary) !important;
            color: var(--theme-primary) !important;
            transition: all 0.2s ease !important;
            /* Tight padding for precise hover targeting */
            padding: 4px 6px !important;
            min-width: auto !important;
            width: auto !important;
        }
        #main-chatbot button:hover,
        .chatbot button:hover,
        [data-testid="chatbot"] button:hover {
            background: var(--theme-primary) !important;
            color: #000000 !important;
            box-shadow: 0 0 15px var(--theme-glow) !important;
        }
        #main-chatbot button:hover svg,
        .chatbot button:hover svg,
        [data-testid="chatbot"] button:hover svg {
            color: #000000 !important;
            fill: #000000 !important;
            stroke: #000000 !important;
        }
        #main-chatbot button svg,
        .chatbot button svg,
        [data-testid="chatbot"] button svg {
            color: var(--theme-primary) !important;
            fill: var(--theme-primary) !important;
            stroke: var(--theme-primary) !important;
        }
        /* Chatbot message action container - no hover effect */
        #main-chatbot .message-actions,
        #main-chatbot .actions,
        .chatbot .message-actions {
            background: transparent !important;
        }
        #main-chatbot .message-actions:hover,
        #main-chatbot .actions:hover,
        .chatbot .message-actions:hover {
            background: transparent !important;
        }

        /* ============================================
           CONVERSATION LIST ITEMS - Hover with black text
           ============================================ */
        .gradio-dataframe tr:hover,
        .gradio-dataframe tbody tr:hover,
        .table-wrap tr:hover,
        .conversation-item:hover,
        [data-testid="dataframe"] tr:hover {
            background: var(--theme-primary) !important;
        }
        .gradio-dataframe tr:hover td,
        .gradio-dataframe tbody tr:hover td,
        .table-wrap tr:hover td,
        .conversation-item:hover *,
        [data-testid="dataframe"] tr:hover td {
            color: #000000 !important;
        }

        /* ============================================
           AUDIO PLAYER CONTROLS - Hover with black text
           ============================================ */
        audio::-webkit-media-controls-panel {
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 50%, #0d0d0d 100%) !important;
        }
        .gradio-audio button:hover,
        .gradio-audio select:hover,
        [data-testid="audio"] button:hover {
            color: #000000 !important;
        }
        .gradio-audio button:hover *,
        [data-testid="audio"] button:hover * {
            color: #000000 !important;
            fill: #000000 !important;
        }

        /* ============================================
           UPLOAD/DROP ZONES - Glow effect, not solid color
           Keep background visible, just add glow on hover
           ============================================ */
        .upload-container,
        .drop-target,
        [data-testid="dropzone"],
        .gradio-file .upload-button,
        .gradio-image .upload-button {
            background: linear-gradient(180deg, #1a1a1a 0%, #0d0d0d 100%) !important;
            border: 2px dashed var(--theme-primary) !important;
            color: var(--theme-primary) !important;
            transition: all 0.2s ease !important;
        }
        .upload-container:hover,
        .drop-target:hover,
        [data-testid="dropzone"]:hover,
        .gradio-file .upload-button:hover,
        .gradio-image .upload-button:hover {
            background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 100%) !important;
            border-color: var(--theme-bright) !important;
            box-shadow: 0 0 25px var(--theme-glow) !important;
            color: var(--theme-primary) !important;
        }
        /* Upload zone text stays theme color on hover */
        .upload-container:hover *,
        .drop-target:hover *,
        [data-testid="dropzone"]:hover *,
        .gradio-file .upload-button:hover *,
        .gradio-image .upload-button:hover * {
            color: var(--theme-primary) !important;
        }

        /* ============================================
           INPUT FIELDS FOCUS - Theme glow, not blue ring
           ============================================ */
        input:focus,
        textarea:focus,
        select:focus {
            outline: none !important;
            border-color: var(--theme-primary) !important;
            box-shadow: 0 0 0 2px var(--theme-glow) !important;
        }
        .gradio-textbox:focus-within input,
        .gradio-textbox:focus-within textarea {
            border-color: var(--theme-primary) !important;
            box-shadow: 0 0 15px var(--theme-glow) !important;
        }
    """
    
    # JavaScript for keyboard shortcuts, copy buttons, HUD updates, and expandable messages
    keyboard_js = """
    function() {
        // ==================== THEME SWITCHER SYSTEM ====================
        window.ThemeSwitcher = {
            themes: {
                'lapis': {
                    name: 'Lapis Lazuli',
                    primary: '#00BFFF',
                    dim: '#006699',
                    bright: '#33CCFF',
                    medium: '#0088AA',
                    glow: 'rgba(0, 191, 255, 0.4)',
                    glowSoft: 'rgba(0, 191, 255, 0.1)',
                    glowText: 'rgba(0, 191, 255, 0.5)',
                    glowTextStrong: 'rgba(0, 191, 255, 0.6)',
                    bgDark: '#001a33',
                    bgMid: '#003366',
                    bgStripe: '#003344'
                },
                'purple': {
                    name: 'Lightsaber Purple',
                    primary: '#9B30FF',
                    dim: '#6B1FA3',
                    bright: '#B041FF',
                    medium: '#8A2BE2',
                    glow: 'rgba(155, 48, 255, 0.4)',
                    glowSoft: 'rgba(155, 48, 255, 0.1)',
                    glowText: 'rgba(155, 48, 255, 0.5)',
                    glowTextStrong: 'rgba(155, 48, 255, 0.6)',
                    bgDark: '#1a0033',
                    bgMid: '#330066',
                    bgStripe: '#2a0044'
                },
                'orange': {
                    name: 'Lightsaber Orange',
                    primary: '#FF6600',
                    dim: '#CC5200',
                    bright: '#FF8533',
                    medium: '#E65C00',
                    glow: 'rgba(255, 102, 0, 0.4)',
                    glowSoft: 'rgba(255, 102, 0, 0.1)',
                    glowText: 'rgba(255, 102, 0, 0.5)',
                    glowTextStrong: 'rgba(255, 102, 0, 0.6)',
                    bgDark: '#331a00',
                    bgMid: '#663300',
                    bgStripe: '#442200'
                },
                'green': {
                    name: 'Lightsaber Green',
                    primary: '#39FF14',
                    dim: '#2ACC10',
                    bright: '#66FF44',
                    medium: '#30DD12',
                    glow: 'rgba(57, 255, 20, 0.4)',
                    glowSoft: 'rgba(57, 255, 20, 0.1)',
                    glowText: 'rgba(57, 255, 20, 0.5)',
                    glowTextStrong: 'rgba(57, 255, 20, 0.6)',
                    bgDark: '#0a3300',
                    bgMid: '#146600',
                    bgStripe: '#0d4400'
                }
            },

            currentTheme: 'lapis',

            apply: function(themeName) {
                const theme = this.themes[themeName];
                if (!theme) return;

                this.currentTheme = themeName;
                const root = document.documentElement;

                root.style.setProperty('--theme-primary', theme.primary);
                root.style.setProperty('--theme-dim', theme.dim);
                root.style.setProperty('--theme-bright', theme.bright);
                root.style.setProperty('--theme-medium', theme.medium);
                root.style.setProperty('--theme-glow', theme.glow);
                root.style.setProperty('--theme-glow-soft', theme.glowSoft);
                root.style.setProperty('--theme-glow-text', theme.glowText);
                root.style.setProperty('--theme-glow-text-strong', theme.glowTextStrong);
                root.style.setProperty('--theme-bg-dark', theme.bgDark);
                root.style.setProperty('--theme-bg-mid', theme.bgMid);
                root.style.setProperty('--theme-bg-stripe', theme.bgStripe);

                // Update banner dynamically
                const banner = document.querySelector('#app-banner');
                if (banner) {
                    banner.style.borderColor = theme.primary;
                    banner.style.boxShadow = '0 0 25px ' + theme.glow + ', inset 0 0 30px ' + theme.glowSoft;
                }
                const bannerPre = document.querySelector('#app-banner pre');
                if (bannerPre) {
                    bannerPre.style.color = theme.primary;
                    bannerPre.style.textShadow = '0 0 10px ' + theme.glowTextStrong;
                }
                const bannerDiv = document.querySelector('#app-banner > div:last-child');
                if (bannerDiv) {
                    bannerDiv.style.color = theme.primary;
                    const span = bannerDiv.querySelector('span');
                    if (span) span.style.color = theme.primary;
                }

                // Save preference
                try {
                    localStorage.setItem('tts2_theme', themeName);
                } catch(e) {}

                console.log('[Theme] Applied: ' + theme.name);
            },

            init: function() {
                // Load saved theme on page load
                try {
                    const saved = localStorage.getItem('tts2_theme');
                    if (saved && this.themes[saved]) {
                        this.apply(saved);
                    }
                } catch(e) {}

                // Also check the dropdown value after Gradio loads (handles server-side saved theme)
                setTimeout(() => {
                    const containers = document.querySelectorAll('.gradio-dropdown');
                    containers.forEach(container => {
                        const label = container.querySelector('label');
                        if (label && label.textContent.includes('Color Theme')) {
                            const input = container.querySelector('input');
                            if (input && input.value && this.themes[input.value]) {
                                this.apply(input.value);
                            }
                        }
                    });
                }, 500);

                // Watch for theme dropdown changes
                this.watchDropdown();
            },

            watchDropdown: function() {
                const self = this;
                // Poll for the dropdown since Gradio loads dynamically
                const setupWatcher = () => {
                    // Look for dropdown with "Color Theme" label or containing theme values
                    const dropdowns = document.querySelectorAll('select, input[type="text"][data-testid]');
                    const containers = document.querySelectorAll('.gradio-dropdown');

                    containers.forEach(container => {
                        const label = container.querySelector('label');
                        if (label && label.textContent.includes('Color Theme')) {
                            // Found theme dropdown container - watch for input changes
                            const input = container.querySelector('input');
                            if (input && !input._themeWatcherAttached) {
                                input._themeWatcherAttached = true;

                                // Watch for value changes
                                const observer = new MutationObserver(() => {
                                    const value = input.value;
                                    if (value && self.themes[value] && value !== self.currentTheme) {
                                        self.apply(value);
                                    }
                                });
                                observer.observe(input, { attributes: true, attributeFilter: ['value'] });

                                // Also listen for input events
                                input.addEventListener('input', () => {
                                    setTimeout(() => {
                                        const value = input.value;
                                        if (value && self.themes[value]) {
                                            self.apply(value);
                                        }
                                    }, 100);
                                });

                                console.log('[Theme] Dropdown watcher attached');
                            }
                        }
                    });

                    // Also watch listbox options being clicked
                    document.addEventListener('click', (e) => {
                        const listItem = e.target.closest('[role="option"], .dropdown-item, li');
                        if (listItem) {
                            const text = listItem.textContent.trim().toLowerCase();
                            // Map display text to theme key
                            const themeMap = {
                                'lapis lazuli (blue)': 'lapis',
                                'lightsaber purple': 'purple',
                                'lightsaber orange': 'orange',
                                'lightsaber green': 'green'
                            };
                            const themeKey = themeMap[text];
                            if (themeKey && self.themes[themeKey]) {
                                setTimeout(() => self.apply(themeKey), 50);
                            }
                        }
                    }, true);
                };

                // Run setup after a short delay for Gradio to load
                setTimeout(setupWatcher, 1000);
                setTimeout(setupWatcher, 3000);  // Retry in case of slow load
            }
        };

        // Initialize theme on load
        window.ThemeSwitcher.init();

        // ==================== HOVER TEXT FIX ====================
        // CSS-only solution - no JavaScript to avoid side effects
        // Targets ONLY radio buttons and checkbox groups with cyan hover backgrounds

        // ==================== CYBERDECK HUD SYSTEM ====================
        window.HUD = {
            latency: 0,
            ttsSpeed: 0,
            tokensIn: 0,
            tokensOut: 0,
            memoryNodes: 0,
            emotion: 'neutral',
            emotionValue: 0.5,

            // Update HUD display
            update: function(data) {
                if (data.latency !== undefined) {
                    this.latency = data.latency;
                    const el = document.getElementById('hud-latency');
                    if (el) {
                        el.textContent = data.latency + 'ms';
                        el.className = 'hud-value' + (data.latency > 2000 ? ' danger' : data.latency > 1000 ? ' warning' : '');
                    }
                }
                if (data.ttsSpeed !== undefined) {
                    this.ttsSpeed = data.ttsSpeed;
                    const el = document.getElementById('hud-tts-speed');
                    if (el) el.textContent = data.ttsSpeed + 'x';
                }
                if (data.tokensIn !== undefined) {
                    this.tokensIn = data.tokensIn;
                    const el = document.getElementById('hud-tokens-in');
                    if (el) el.textContent = data.tokensIn.toLocaleString();
                }
                if (data.tokensOut !== undefined) {
                    this.tokensOut = data.tokensOut;
                    const el = document.getElementById('hud-tokens-out');
                    if (el) el.textContent = data.tokensOut.toLocaleString();
                }
                if (data.memoryNodes !== undefined) {
                    this.memoryNodes = data.memoryNodes;
                    const el = document.getElementById('hud-memory-nodes');
                    if (el) el.textContent = data.memoryNodes;
                }
                if (data.emotion !== undefined) {
                    this.emotion = data.emotion;
                    this.emotionValue = data.emotionValue || 0.5;
                    const fill = document.getElementById('emotion-fill');
                    const value = document.getElementById('emotion-value');
                    if (fill && value) {
                        fill.className = 'meter-fill ' + data.emotion.toLowerCase();
                        fill.style.width = (this.emotionValue * 100) + '%';
                        value.textContent = data.emotion.toUpperCase();
                    }
                }
            },

            // Set state indicator on PTT box
            setState: function(state) {
                const pttBox = document.getElementById('ptt-status-box');
                if (!pttBox) return;
                pttBox.classList.remove('recording', 'processing', 'speaking');
                if (state) pttBox.classList.add(state);
            }
        };

        // Wait for DOM to be ready
        setTimeout(() => {
            const msgInput = document.querySelector('#msg-input textarea');
            if (msgInput) {
                msgInput.addEventListener('keydown', (e) => {
                    // Ctrl+Enter or Cmd+Enter to send
                    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                        e.preventDefault();
                        const sendBtn = document.querySelector('#send-btn');
                        if (sendBtn) sendBtn.click();
                    }
                    // Escape to clear input
                    if (e.key === 'Escape') {
                        e.preventDefault();
                        msgInput.value = '';
                        msgInput.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                });
            }

            // Copy button functionality using event delegation
            document.addEventListener('click', (e) => {
                // Handle copy button clicks
                if (e.target.classList.contains('message-copy-btn')) {
                    const msgContent = e.target.closest('.message-bubble');
                    if (msgContent) {
                        // Get text content, stripping HTML
                        const text = msgContent.innerText || msgContent.textContent;
                        navigator.clipboard.writeText(text).then(() => {
                            e.target.classList.add('copied');
                            e.target.textContent = '[ OK ]';
                            setTimeout(() => {
                                e.target.classList.remove('copied');
                                e.target.textContent = 'COPY';
                            }, 2000);
                        });
                    }
                }

                // Handle expand button clicks
                if (e.target.classList.contains('expand-btn')) {
                    const expandable = e.target.previousElementSibling;
                    if (expandable && expandable.classList.contains('message-expandable')) {
                        expandable.classList.toggle('expanded');
                        e.target.textContent = expandable.classList.contains('expanded')
                            ? '[ COLLAPSE ]' : '[ EXPAND ]';
                    }
                }
            });

            // Add copy buttons to messages using MutationObserver
            const addCopyButtons = () => {
                document.querySelectorAll('.message-bubble').forEach(bubble => {
                    if (!bubble.querySelector('.message-copy-btn') && !bubble.dataset.copyAdded) {
                        bubble.dataset.copyAdded = 'true';
                        bubble.style.position = 'relative';
                        const btn = document.createElement('button');
                        btn.className = 'message-copy-btn';
                        btn.textContent = 'COPY';
                        btn.title = 'Copy message';
                        bubble.appendChild(btn);
                    }
                });
            };

            // Observe chatbot for new messages
            const chatbot = document.querySelector('[data-testid="chatbot"]');
            if (chatbot) {
                const observer = new MutationObserver(addCopyButtons);
                observer.observe(chatbot, { childList: true, subtree: true });
                addCopyButtons(); // Initial run
            }

            // ============================================
            // MOBILE PTT - Touch-based recording
            // ============================================
            const mobilePttBtn = document.getElementById('mobile-ptt-btn');
            if (mobilePttBtn) {
                let mediaRecorder = null;
                let audioChunks = [];
                let isRecording = false;
                let recordingStartTime = null;

                const startRecording = async () => {
                    if (isRecording) return;
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({
                            audio: {
                                echoCancellation: true,
                                noiseSuppression: true,
                                sampleRate: 16000
                            }
                        });
                        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                        audioChunks = [];

                        mediaRecorder.ondataavailable = (e) => {
                            if (e.data.size > 0) audioChunks.push(e.data);
                        };

                        mediaRecorder.onstop = async () => {
                            stream.getTracks().forEach(t => t.stop());
                            if (audioChunks.length > 0) {
                                const blob = new Blob(audioChunks, { type: 'audio/webm' });
                                // Send to hidden audio input for processing
                                const audioInput = document.querySelector('#component-23 input[type="file"], .audio-input input[type="file"]');
                                if (audioInput) {
                                    const file = new File([blob], 'mobile_recording.webm', { type: 'audio/webm' });
                                    const dt = new DataTransfer();
                                    dt.items.add(file);
                                    audioInput.files = dt.files;
                                    audioInput.dispatchEvent(new Event('change', { bubbles: true }));
                                    // Auto-click send after recording
                                    setTimeout(() => {
                                        const sendBtn = document.querySelector('button[id*="voice_btn"], .voice-send-btn');
                                        if (sendBtn) sendBtn.click();
                                    }, 500);
                                }
                            }
                        };

                        mediaRecorder.start(100);
                        isRecording = true;
                        recordingStartTime = Date.now();
                        mobilePttBtn.classList.add('recording');
                        mobilePttBtn.textContent = '🔴 RECORDING...';
                    } catch (err) {
                        console.error('Mic access denied:', err);
                        mobilePttBtn.textContent = '⚠️ MIC ACCESS DENIED';
                        setTimeout(() => { mobilePttBtn.textContent = '🎤 HOLD TO TALK'; }, 2000);
                    }
                };

                const stopRecording = () => {
                    if (!isRecording || !mediaRecorder) return;
                    const duration = Date.now() - recordingStartTime;
                    if (duration < 500) {
                        // Too short, cancel
                        mediaRecorder.stop();
                        audioChunks = [];
                        mobilePttBtn.textContent = '⚠️ TOO SHORT';
                    } else {
                        mediaRecorder.stop();
                        mobilePttBtn.textContent = '✓ PROCESSING...';
                    }
                    isRecording = false;
                    mobilePttBtn.classList.remove('recording');
                    setTimeout(() => { mobilePttBtn.textContent = '🎤 HOLD TO TALK'; }, 1500);
                };

                // Touch events for mobile
                mobilePttBtn.addEventListener('touchstart', (e) => {
                    e.preventDefault();
                    startRecording();
                }, { passive: false });
                mobilePttBtn.addEventListener('touchend', (e) => {
                    e.preventDefault();
                    stopRecording();
                }, { passive: false });
                mobilePttBtn.addEventListener('touchcancel', stopRecording);

                // Mouse events for desktop testing
                mobilePttBtn.addEventListener('mousedown', startRecording);
                mobilePttBtn.addEventListener('mouseup', stopRecording);
                mobilePttBtn.addEventListener('mouseleave', () => { if (isRecording) stopRecording(); });
            }
        }, 1000);
        return [];
    }
    """
    
    # PWA manifest for webapp installation
    pwa_js = """
    () => {
        // Inject PWA meta tags for mobile webapp support
        const addMeta = (name, content) => {
            if (!document.querySelector(`meta[name="${name}"]`)) {
                const meta = document.createElement('meta');
                meta.name = name;
                meta.content = content;
                document.head.appendChild(meta);
            }
        };

        // Mobile webapp meta tags
        addMeta('mobile-web-app-capable', 'yes');
        addMeta('apple-mobile-web-app-capable', 'yes');
        addMeta('apple-mobile-web-app-status-bar-style', 'black-translucent');
        addMeta('apple-mobile-web-app-title', 'TTS2 Voice');
        addMeta('theme-color', '#000000');
        addMeta('viewport', 'width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no');

        // Create and inject manifest dynamically
        const manifest = {
            name: 'TTS2 Voice Agent',
            short_name: 'TTS2 Voice',
            description: 'Multi-character voice AI assistant',
            start_url: window.location.origin,
            display: 'standalone',
            background_color: '#000000',
            theme_color: '#00BFFF',
            icons: [
                { src: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect fill="%23000" width="100" height="100"/><text y="70" x="50" text-anchor="middle" font-size="60" fill="%2300BFFF">🎤</text></svg>', sizes: '192x192', type: 'image/svg+xml' },
                { src: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect fill="%23000" width="100" height="100"/><text y="70" x="50" text-anchor="middle" font-size="60" fill="%2300BFFF">🎤</text></svg>', sizes: '512x512', type: 'image/svg+xml' }
            ]
        };
        const manifestBlob = new Blob([JSON.stringify(manifest)], { type: 'application/json' });
        const manifestUrl = URL.createObjectURL(manifestBlob);
        if (!document.querySelector('link[rel="manifest"]')) {
            const link = document.createElement('link');
            link.rel = 'manifest';
            link.href = manifestUrl;
            document.head.appendChild(link);
        }

        // iOS splash screen color
        const appleLink = document.createElement('link');
        appleLink.rel = 'apple-touch-icon';
        appleLink.href = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect fill="%23000" width="100" height="100"/><text y="70" x="50" text-anchor="middle" font-size="60" fill="%2300BFFF">🎤</text></svg>';
        document.head.appendChild(appleLink);

        return [];
    }
    """

    # Combine JS - merge keyboard_js and pwa_js into single function
    # Remove keyboard_js closing (return + brace), remove pwa_js opening (arrow function)
    combined_js = keyboard_js.replace("return [];\n    }", "") + pwa_js.replace("() => {\n", "")

    with gr.Blocks(title="IndexTTS2 Voice Agent", theme=create_dark_theme(), css=custom_css, js=combined_js) as app:

        # State variables
        current_character = gr.State(value=initial_char)
        current_voice = gr.State(value=SETTINGS.get("last_voice", "reference.wav"))
        current_conversation_id = gr.State(value="new")
        current_provider = gr.State(value=initial_provider)

        gr.HTML(f"""
        <div id="app-banner" style="border: 3px solid var(--theme-primary, #00BFFF); background: linear-gradient(180deg, #141414 0%, #000000 100%); padding: 15px 20px; margin-bottom: 15px; box-shadow: 0 0 25px var(--theme-glow, rgba(0, 191, 255, 0.4)), inset 0 0 30px var(--theme-glow-soft, rgba(0, 191, 255, 0.08)), inset 0 1px 0 rgba(255,255,255,0.05);">
            <pre style="color: var(--theme-primary, #00BFFF); font-family: 'Fira Code', 'JetBrains Mono', 'Consolas', monospace; font-size: 16px; line-height: 1.15; margin: 0; text-align: center; letter-spacing: 0px; text-shadow: 0 0 10px var(--theme-glow-text-strong, rgba(0, 191, 255, 0.6));">
╔════════════════════════════════════════════════════════════════════════════════════╗
║  ████████╗████████╗ ███████╗ ██████╗    ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗      ║
║  ╚══██╔══╝╚══██╔══╝ ██╔════╝ ╚════██╗   ██║   ██║██╔═══██╗██║██╔════╝██╔════╝      ║
║     ██║      ██║    ███████╗  █████╔╝   ██║   ██║██║   ██║██║██║     █████╗        ║
║     ██║      ██║    ╚════██║ ██╔═══╝    ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝        ║
║     ██║      ██║    ███████║ ███████╗    ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗      ║
║     ╚═╝      ╚═╝    ╚══════╝ ╚══════╝     ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝      ║
╚════════════════════════════════════════════════════════════════════════════════════╝
            </pre>
            <div style="text-align: center; color: var(--theme-primary, #00BFFF); font-size: 0.9em; margin-top: 10px; text-transform: uppercase; letter-spacing: 2px;">
                Multi-Character • Memory • Tools • Vision • MCP &nbsp;|&nbsp; <span style="color: var(--theme-primary, #00BFFF);">Running on {PLATFORM.title()}{" (WSL)" if IS_WSL else ""}</span>
            </div>
        </div>
        """)

        # ==================== HUD BAR - System Status ====================
        gr.HTML("""
        <div id="hud-bar">
            <div class="hud-item">
                <span class="hud-label">LATENCY:</span>
                <span class="hud-value" id="hud-latency">--ms</span>
            </div>
            <div class="hud-item">
                <span class="hud-label">TTS:</span>
                <span class="hud-value" id="hud-tts-speed">--x</span>
            </div>
            <div class="hud-item">
                <span class="hud-label">TOKENS:</span>
                <span class="hud-value" id="hud-tokens-in">--</span>
                <span style="color: #666;">IN</span>
                <span class="hud-value" id="hud-tokens-out">--</span>
                <span style="color: #666;">OUT</span>
            </div>
            <div class="hud-item">
                <span class="hud-label">MEMORY:</span>
                <span class="hud-value" id="hud-memory-nodes">0</span>
                <span style="color: #666;">NODES</span>
            </div>
            <div id="emotion-meter">
                <span class="meter-label">EMOTION:</span>
                <div class="meter-bar">
                    <div class="meter-fill neutral" id="emotion-fill" style="width: 50%;"></div>
                </div>
                <span class="meter-value" id="emotion-value">NEUTRAL</span>
            </div>
        </div>
        """)

        # Hidden component to trigger HUD updates via JavaScript
        hud_update_trigger = gr.HTML(
            value="",
            visible=False,
            elem_id="hud-update-trigger"
        )

        # PTT Status
        _, initial_ptt_display, _, _ = get_ptt_status()
        
        with gr.Row():
            # ==================== LEFT COLUMN - Chat ====================
            with gr.Column(scale=2):
                
                ptt_status_display = gr.Textbox(
                    value=initial_ptt_display,
                    show_label=False,  # Hide label for cleaner look
                    placeholder="🎙️ Voice Status",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                    elem_id="ptt-status-box"
                )

                # Mobile PTT Button - Touch-friendly for phones/tablets
                mobile_ptt_btn = gr.Button(
                    "🎤 HOLD TO TALK",
                    variant="secondary",
                    elem_id="mobile-ptt-btn"
                )
                mobile_ptt_status = gr.Textbox(visible=False, elem_id="mobile-ptt-status")

                with gr.Row():
                    character_dropdown = gr.Dropdown(
                        choices=[(c.display_name, cid) for cid, c in CHARACTER_MANAGER.characters.items()],
                        value=initial_char,
                        label="🎭 Character",
                        interactive=True,
                        scale=2
                    )
                    # Get initial voice choices based on current TTS backend
                    initial_voice_choices, initial_voice = get_voice_choices_for_backend()
                    voice_dropdown = gr.Dropdown(
                        choices=initial_voice_choices,
                        value=initial_voice,
                        label="🎤 Voice",
                        interactive=True,
                        scale=2
                    )
                    refresh_voice_btn = gr.Button("🔄", scale=0, size="sm", min_width=40)

                # LLM Provider selection
                with gr.Row():
                    provider_radio = gr.Radio(
                        choices=[("☁️ OpenRouter", "openrouter"), ("💻 LM Studio (Local)", "lmstudio")],
                        value=initial_provider,
                        label="LLM Provider",
                        interactive=True,
                        scale=2
                    )
                    lm_studio_status = gr.Textbox(
                        value=get_lm_studio_status_display() if initial_provider == "lmstudio" else "",
                        label="Local Model",
                        interactive=False,
                        scale=2,
                        visible=(initial_provider == "lmstudio")
                    )
                    refresh_lm_btn = gr.Button("🔄", scale=1, size="sm", visible=(initial_provider == "lmstudio"))
                
                # Model selector (OpenRouter only)
                with gr.Row(visible=(initial_provider == "openrouter")) as openrouter_row:
                    model_dropdown = gr.Dropdown(
                        choices=model_choices,
                        value=SETTINGS.get("model", "x-ai/grok-4.1-fast"),
                        label="🤖 OpenRouter Model (👁️ = Vision)",
                        interactive=True,
                        scale=4
                    )
                    refresh_models_btn = gr.Button("🔄", scale=1, size="sm")
                
                # Chat display
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600,
                    value=load_session(initial_char),
                    elem_id="main-chatbot",
                    type="messages"
                )
                
                # TTS warning display
                tts_warning_display = gr.Textbox(
                    value="",
                    show_label=False,  # Hide label
                    interactive=False,
                    visible=False,
                    elem_id="tts-warning"
                )
                
                # Text input
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message",
                        placeholder="Type here and press Enter (or Ctrl+Enter), Escape to clear...",
                        scale=5,
                        lines=1,
                        elem_id="msg-input"
                    )
                    send_btn = gr.Button("📤 Send", variant="primary", scale=1, elem_id="send-btn")
                
                # Transcription display
                transcription_display = gr.Textbox(
                    label="Last Transcription",
                    interactive=False,
                    lines=1,
                    visible=True
                )
                
                # Input Options
                with gr.Accordion("📷 Vision & Screen", open=False):
                    gr.Markdown("*Upload an image or capture your screen.*")
                    with gr.Row():
                        image_input = gr.Image(
                            label="Image Input",
                            type="numpy",
                            height=250,
                            scale=4
                        )
                        with gr.Column(scale=1):
                            screen_btn = gr.Button("📸 Capture Screen", variant="secondary")
                            clear_image_btn = gr.Button("🗑️ Clear", size="sm")

                with gr.Accordion("📄 Document Analysis", open=False):
                    doc_upload = gr.File(
                        label="Upload Document",
                        file_types=[".pdf", ".txt", ".md", ".json", ".csv", ".py", ".js", ".ts", ".html", ".css", ".yaml", ".yml", ".xml", ".log", ".docx"],
                        type="filepath"
                    )
                    doc_status = gr.Textbox(label="Status", interactive=False, lines=1)
                    gr.Markdown("*Supports: PDF, TXT, MD, JSON, CSV, DOCX, code files (py/js/ts/html/css/yaml/xml)*")
                
                with gr.Accordion("🎤 Manual Voice Recording", open=False):
                    gr.Markdown("*Backup option - Push-to-Talk (Right Shift key) is recommended*")
                    with gr.Row():
                        audio_input = gr.Audio(
                            label=None,
                            sources=["microphone"],
                            type="numpy",
                            scale=4
                        )
                        voice_btn = gr.Button("🎤 Send", variant="secondary", scale=1)
                
                # Audio output
                with gr.Row(visible=True) as audio_perm_row:
                    audio_perm_btn = gr.Button(
                        "🔊 Click to Enable Audio Playback",
                        variant="primary",
                        size="lg"
                    )
                
                audio_output = gr.Audio(
                    label="🔊 Response",
                    type="numpy",
                    autoplay=True,
                    interactive=False,
                    streaming=False,  # Disable streaming to prevent player reset
                    elem_id="audio-response"
                )
            
            # ==================== RIGHT COLUMN - Settings ====================
            with gr.Column(scale=1, elem_id="settings-panel"):
                
                with gr.Accordion("👥 Group Chat (Multi-Character)", open=False):
                    group_enabled = gr.Checkbox(
                        label="Enable Group Mode", 
                        value=False,
                        info="Characters respond based on who you address"
                    )
                    group_members = gr.CheckboxGroup(
                        choices=CHARACTER_MANAGER.list_characters(),
                        label="Characters in Group",
                        info="Address by name or say 'both' for multiple responses"
                    )
                    group_turns = gr.Slider(
                        minimum=1, maximum=3, step=1, value=2, 
                        label="Max Responses per Message",
                        info="How many characters can respond to one message"
                    )
                
                with gr.Accordion("⚙️ LLM Settings", open=True):
                    temperature_slider = gr.Slider(
                        minimum=0.0, maximum=2.0, step=0.1,
                        value=SETTINGS.get("temperature", 0.7),
                        label="Temperature"
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=100, maximum=8000, step=100,
                        value=SETTINGS.get("max_tokens", 2000),
                        label="Max Tokens"
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.05,
                        value=SETTINGS.get("top_p", 1.0),
                        label="Top P"
                    )
                    freq_penalty_slider = gr.Slider(
                        minimum=0.0, maximum=2.0, step=0.1,
                        value=SETTINGS.get("frequency_penalty", 0.0),
                        label="Frequency Penalty"
                    )
                    pres_penalty_slider = gr.Slider(
                        minimum=0.0, maximum=2.0, step=0.1,
                        value=SETTINGS.get("presence_penalty", 0.0),
                        label="Presence Penalty"
                    )

                with gr.Accordion("🔊 Audio Settings", open=True):
                    # Platform-specific hands-free info
                    # Unified models (SenseVoice/FunASR) have built-in VAD for filtering
                    # but still need Silero/mic capture for recording
                    current_stt = SETTINGS.get("stt_backend", "faster_whisper")
                    has_unified_stt = current_stt in ("sensevoice", "funasr")
                    if has_unified_stt:
                        hf_info = f"Silero capture + {current_stt} filtering" if IS_WSL else f"Uses {current_stt} built-in VAD"
                    else:
                        hf_info = "Run vad_windows.py" if IS_WSL else "Auto-detects speech"
                    hands_free_toggle = gr.Checkbox(
                        label="🗣️ Voice Activated (VAD)",
                        value=SETTINGS.get("vad_enabled", False),
                        info=hf_info,
                        interactive=True  # Always interactive - unified models use mic capture
                    )
                    tts_toggle = gr.Checkbox(
                        label="Enable TTS",
                        value=SETTINGS.get("tts_enabled", True)
                    )
                    tts_backend = gr.Radio(
                        choices=["indextts", "kokoro", "supertonic", "soprano"],
                        value=SETTINGS.get("tts_backend", "indextts"),
                        label="TTS Backend",
                        info="IndexTTS2 (GPU/Clone) | Kokoro (CPU) | Supertonic (CPU) | Soprano (GPU/Fastest)"
                    )
                    stt_backend = gr.Radio(
                        choices=["faster_whisper", "sensevoice", "funasr"],
                        value=SETTINGS.get("stt_backend", "faster_whisper"),
                        label="STT Backend",
                        info="Whisper (CPU) | SenseVoice (GPU+Emotion) | FunASR (GPU+Accuracy)"
                    )
                    # Check if SenseVoice is active to determine initial state
                    sensevoice_active = SETTINGS.get("stt_backend") == "sensevoice"
                    emotion_toggle = gr.Checkbox(
                        label="🎭 Emotion Detection (wav2vec2)",
                        value=SETTINGS.get("emotion_detection_enabled", True) and not sensevoice_active,
                        info="Not used with SenseVoice (built-in)" if sensevoice_active else "Detect user emotion (~300ms CPU)",
                        interactive=not sensevoice_active
                    )

                with gr.Accordion("🎨 Appearance", open=False):
                    theme_dropdown = gr.Dropdown(
                        choices=[
                            ("Lapis Lazuli (Blue)", "lapis"),
                            ("Lightsaber Purple", "purple"),
                            ("Lightsaber Orange", "orange"),
                            ("Lightsaber Green", "green")
                        ],
                        value=SETTINGS.get("theme", "lapis"),
                        label="Color Theme",
                        info="Switch UI colors instantly"
                    )

                with gr.Accordion("🔧 Tools", open=False):
                    # Show available tools dynamically
                    tool_list = REGISTRY.list_tools()
                    tool_names = [t["function"]["name"] for t in tool_list]
                    mcp_count = len(MCP_TOOLS_CACHE) if MCP_TOOLS_CACHE else 0

                    # Build tool descriptions
                    tool_descriptions = []
                    for t in tool_list:
                        name = t["function"]["name"]
                        desc = t["function"].get("description", "")[:50]
                        tool_descriptions.append(f"- `{name}` - {desc}...")

                    tools_md = f"""**Local Tools:** {len(tool_names)}
{chr(10).join(tool_descriptions[:8])}
{"*...and " + str(len(tool_names) - 8) + " more*" if len(tool_names) > 8 else ""}

**MCP Tools:** {mcp_count} (from connected servers)

**Full File Access:** {"✓ Enabled" if CONFIG.get("ENABLE_FULL_FILE_ACCESS") else "✗ Disabled (sandbox only)"}"""

                    tools_available = gr.Markdown(tools_md)
                    
                    full_file_access = gr.Checkbox(
                        label="Enable Full File Access",
                        value=CONFIG.get("ENABLE_FULL_FILE_ACCESS", False),
                        info="⚠️ Allows AI to read/write ANY file on your system"
                    )
                    tools_status = gr.Textbox(label="Status", interactive=False, visible=False)
                
                with gr.Accordion("📜 Conversation History", open=True):
                    conversation_dropdown = gr.Dropdown(
                        choices=initial_conv_choices,
                        value="new",
                        label="Select Conversation",
                        interactive=True
                    )
                    incognito_chk = gr.Checkbox(
                        label="🕵️ Incognito Mode",
                        value=False,
                        info="Chat without saving to memory or history (fresh temporary session)"
                    )
                    conv_status = gr.Textbox(
                        value="Ready",
                        show_label=False,
                        interactive=False,
                        max_lines=1
                    )
                    with gr.Row():
                        delete_conv_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                    with gr.Row():
                        export_txt_btn = gr.Button("📄 Export TXT", size="sm")
                        export_json_btn = gr.Button("📋 Export JSON", size="sm")
                    export_file = gr.File(label="Download", visible=False)
                
                with gr.Accordion("🧠 Active Memory", open=False):
                    memory_display = gr.Markdown("*Memories will appear here during conversation*")
                
                with gr.Accordion("📊 Memory Stats", open=False):
                    memory_stats = gr.Markdown("*Loading...*")
                    with gr.Row():
                        refresh_stats_btn = gr.Button("🔄 Refresh", size="sm")
                        reset_memory_btn = gr.Button("🧹 Reset Memory", size="sm", variant="stop")
                    reset_memory_status = gr.Textbox(visible=False, show_label=False)

                with gr.Accordion("✏️ Memory Editor", open=False):
                    memory_type_filter = gr.Dropdown(
                        choices=["semantic", "episodic", "procedural"],
                        value="semantic",
                        label="Memory Type",
                        interactive=True
                    )
                    browse_memories_btn = gr.Button("🔍 Browse Memories", size="sm")
                    memory_list_display = gr.Markdown("*Click 'Browse Memories' to view*")

                    gr.Markdown("---")
                    gr.Markdown("**Edit/Delete Memory**")
                    memory_id_input = gr.Textbox(
                        label="Memory ID",
                        placeholder="Paste memory ID from list above",
                        max_lines=1
                    )
                    memory_content_input = gr.Textbox(
                        label="New Content (for edit)",
                        placeholder="Leave empty to keep current content",
                        lines=3
                    )
                    with gr.Row():
                        update_memory_btn = gr.Button("💾 Update", size="sm", variant="primary")
                        delete_memory_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                    memory_editor_status = gr.Textbox(visible=False, show_label=False)
                
                with gr.Accordion("🖥️ System Info", open=False):
                    def get_system_info_text():
                        """Generate current system info text"""
                        stt_status = get_stt_status()
                        tts_status = get_tts_status()
                        lm_status_str = get_lm_studio_status_display()

                        # Count tools
                        local_tools = len(REGISTRY.list_tools())
                        mcp_tools = len(MCP_TOOLS_CACHE) if MCP_TOOLS_CACHE else 0

                        # Check feature availability
                        everything_status = "✓" if REGISTRY.get_tool("everything_search") else "✗ (install Everything)"
                        emotion_status = "✓" if EMOTION_AVAILABLE else "✗"
                        graphrag_status = "✓" if GRAPHRAG_AVAILABLE else "✗"
                        
                        # Check NuExtract availability
                        nuextract_status = "✗"
                        try:
                            from memory.nuextract import is_available as nuextract_is_available
                            if nuextract_is_available():
                                nuextract_status = "✓ NuExtract-2.0-2B"
                        except (ImportError, RuntimeError):
                            pass  # NuExtract not available

                        # Get current settings
                        tts_backend = SETTINGS.get("tts_backend", "indextts")
                        tts_backend_display = "Kokoro (ONNX)" if tts_backend == "kokoro" else "IndexTTS2 (GPU)"
                        llm_provider = SETTINGS.get("llm_provider", "openrouter")
                        llm_display = "LM Studio (Local)" if llm_provider == "lmstudio" else "OpenRouter (Cloud)"

                        # Get embedding model from memory manager
                        embedding_model = "Qwen3-Embedding-0.6B"
                        try:
                            if MEMORY_MANAGER and hasattr(MEMORY_MANAGER, 'embeddings'):
                                embedding_model = getattr(MEMORY_MANAGER.embeddings, 'model_name', embedding_model)
                                # Shorten if too long
                                if len(embedding_model) > 30:
                                    embedding_model = embedding_model.split('/')[-1][:25]
                        except (AttributeError, TypeError):
                            pass  # Embedding model info not available

                        return f"""**Platform:** {PLATFORM.title()}{" (WSL)" if IS_WSL else ""}
**Port:** {SERVER_PORT}
**GPU:** {'✓ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '✗ CPU only'}

**Voice:**
- STT: {stt_status}
- TTS: {tts_status} ({tts_backend_display})
- Emotion: {emotion_status}

**AI:**
- Provider: {llm_display}
- API Key: {'✓' if CONFIG['OPENROUTER_API_KEY'] else '✗'}
- LM Studio: {lm_status_str}

**Tools:** {local_tools} local + {mcp_tools} MCP
- Everything Search: {everything_status}

**Memory:**
- Embeddings: {embedding_model}
- GraphRAG: {graphrag_status}
- Extractor: {nuextract_status}"""

                    system_info = gr.Markdown(get_system_info_text())
                    refresh_stt_btn = gr.Button("🔄 Refresh Status", size="sm")
                
                gr.Markdown("---")
                
                with gr.Row():
                    clear_btn = gr.Button("➕ New Chat", size="sm")
                    nuke_btn = gr.Button("☢️ Wipe Memory", variant="stop", size="sm")
                
                # Platform-specific tips
                if IS_WSL:
                    hf_tip = "**Hands-free:** Run vad_windows.py on Windows"
                else:
                    hf_tip = "**Hands-free:** Toggle checkbox for auto speech detection"
                
                gr.Markdown(f"""
### 💡 Tips
- **Push-to-Talk:** Hold **Right Shift** to record
- {hf_tip}
- **New Chat:** Starts fresh topic, character still remembers you
- **Wipe Memory:** Complete reset, character forgets everything
- **Group Chat:** Enable to have multiple characters in conversation
- **Local LLMs:** Use LM Studio for free, private chat
                """)
        
        # ========== Event Handlers ==========
        
        audio_perm_btn.click(fn=lambda: gr.update(visible=False), outputs=[audio_perm_row])
        clear_image_btn.click(fn=lambda: None, outputs=[image_input])
        
        # Provider switching
        def on_provider_change(provider):
            global LM_STUDIO_AVAILABLE, LM_STUDIO_MODEL
            SETTINGS["llm_provider"] = provider
            save_settings(SETTINGS)
            
            if provider == "lmstudio":
                LM_STUDIO_AVAILABLE, LM_STUDIO_MODEL = check_lm_studio_available()
                status = get_lm_studio_status_display()
                return (
                    provider,
                    gr.update(visible=False),
                    gr.update(visible=True, value=status),
                    gr.update(visible=True),
                )
            else:
                return (
                    provider,
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
        
        provider_radio.change(
            fn=on_provider_change,
            inputs=[provider_radio],
            outputs=[current_provider, openrouter_row, lm_studio_status, refresh_lm_btn]
        )
        
        def refresh_lm_studio():
            global LM_STUDIO_AVAILABLE, LM_STUDIO_MODEL
            LM_STUDIO_AVAILABLE, LM_STUDIO_MODEL = check_lm_studio_available(force_refresh=True)
            return get_lm_studio_status_display()
        
        refresh_lm_btn.click(fn=refresh_lm_studio, outputs=[lm_studio_status])
        
        # Character switching
        def get_character_tools_display(char_id):
            """Get tools display for a character"""
            char = CHARACTER_MANAGER.get_character(char_id)
            if not char:
                return "No character selected"
            tools = char.allowed_tools if char.allowed_tools else []
            full_enabled = CONFIG.get("ENABLE_FULL_FILE_ACCESS", False)
            return f"""**Available Tools:** {len(REGISTRY.list_tools())}
- `get_current_time` - Current date/time
- `web_search` - DuckDuckGo (no API key needed)
- `read_file` / `write_file` - Sandbox only
- `read_file_full` / `write_file_full` - {"✓ Enabled" if full_enabled else "✗ Disabled"}

**{char.name}'s Tools:** {', '.join(f'`{t}`' for t in tools) if tools else '*None (pure roleplay)*'}"""
        
        character_dropdown.change(
            fn=switch_character,
            inputs=[character_dropdown, current_character, chatbot, current_conversation_id],
            outputs=[chatbot, current_character, voice_dropdown, conv_status, 
                     memory_stats, memory_display, conversation_dropdown, current_conversation_id]
        ).then(
            fn=get_character_tools_display,
            inputs=[character_dropdown],
            outputs=[tools_available]
        )
        
        # Conversation switching
        conversation_dropdown.change(
            fn=switch_conversation,
            inputs=[conversation_dropdown, current_character, chatbot, current_conversation_id],
            outputs=[chatbot, current_conversation_id, conv_status]
        )
        
        # Delete conversation
        def delete_current_conversation(character_id, conversation_id):
            if conversation_id and conversation_id != "new":
                delete_conversation(character_id, conversation_id)
                conv_choices = get_conversation_choices(character_id)
                return [], "new", "Conversation deleted", gr.update(choices=conv_choices, value="new")
            return gr.update(), gr.update(), "No conversation to delete", gr.update()
        
        delete_conv_btn.click(
            fn=delete_current_conversation,
            inputs=[current_character, current_conversation_id],
            outputs=[chatbot, current_conversation_id, conv_status, conversation_dropdown]
        )
        
        # Export handlers
        def do_export_txt(char_id, conv_id):
            path, status = handle_export_txt(char_id, conv_id)
            if path:
                return gr.update(value=path, visible=True), status
            return gr.update(visible=False), status
        
        def do_export_json(char_id, conv_id):
            path, status = handle_export_json(char_id, conv_id)
            if path:
                return gr.update(value=path, visible=True), status
            return gr.update(visible=False), status
        
        export_txt_btn.click(
            fn=do_export_txt,
            inputs=[current_character, current_conversation_id],
            outputs=[export_file, conv_status]
        )
        
        export_json_btn.click(
            fn=do_export_json,
            inputs=[current_character, current_conversation_id],
            outputs=[export_file, conv_status]
        )
        
        # Voice switching
        def on_voice_change(v):
            SETTINGS["last_voice"] = v
            save_settings(SETTINGS)
            return v
        voice_dropdown.change(fn=on_voice_change, inputs=[voice_dropdown], outputs=[current_voice])
        
        # Voice refresh button - reload voices from disk
        def refresh_voices():
            """Refresh voice list from disk, filtered for current TTS backend"""
            # First refresh the underlying voice list from disk
            refresh_voice_choices()
            # Then get choices appropriate for current backend
            backend = SETTINGS.get("tts_backend", "indextts")
            choices, default = get_voice_choices_for_backend(backend)
            print(f"[Voices] Refreshed for {backend}: {len(choices)} voices found")
            return gr.update(choices=choices, value=default)
        
        refresh_voice_btn.click(fn=refresh_voices, outputs=[voice_dropdown])
        
        # Model switching
        def on_model_change(m):
            SETTINGS["model"] = m
            save_settings(SETTINGS)
            return m
        model_dropdown.change(fn=on_model_change, inputs=[model_dropdown], outputs=[])
        
        refresh_models_btn.click(
            fn=lambda: refresh_models(CONFIG['OPENROUTER_API_KEY']),
            outputs=[model_dropdown]
        )
        
        # Refresh system status
        def refresh_system_status():
            """Refresh system status info with current settings"""
            global LM_STUDIO_AVAILABLE, LM_STUDIO_MODEL
            init_whisper()
            LM_STUDIO_AVAILABLE, LM_STUDIO_MODEL = check_lm_studio_available()
            return get_system_info_text()
        refresh_stt_btn.click(fn=refresh_system_status, outputs=[system_info])
        
        # Settings changes
        def save_setting(key):
            def saver(value):
                SETTINGS[key] = value
                save_settings(SETTINGS)
                return value
            return saver
        
        temperature_slider.change(fn=save_setting("temperature"), inputs=[temperature_slider])
        max_tokens_slider.change(fn=save_setting("max_tokens"), inputs=[max_tokens_slider])
        top_p_slider.change(fn=save_setting("top_p"), inputs=[top_p_slider])
        freq_penalty_slider.change(fn=save_setting("frequency_penalty"), inputs=[freq_penalty_slider])
        pres_penalty_slider.change(fn=save_setting("presence_penalty"), inputs=[pres_penalty_slider])
        tts_toggle.change(fn=save_setting("tts_enabled"), inputs=[tts_toggle])
        
        def on_tts_backend_change(value):
            SETTINGS["tts_backend"] = value
            save_settings(SETTINGS)
            print(f"\n[Settings] TTS Backend switched to: {value}")

            # Get appropriate voices for this backend
            choices, default = get_voice_choices_for_backend(value)

            if value == "kokoro":
                print(f"  → {len(choices)} Kokoro voices available (af_/am_/bf_/bm_)")
                print("  → Running on CPU (ONNX)")
            elif value == "supertonic":
                print(f"  → {len(choices)} preset voices available")
                print("  → Ultra-fast streaming TTS")
                print("  → Running on CPU (ONNX)")
            else:  # indextts
                print(f"  → {len(choices)} voices available for cloning")
                print("  → Voice cloning enabled")
                print("  → Running on CUDA:0")

            return gr.update(choices=choices, value=default)

        tts_backend.change(fn=on_tts_backend_change, inputs=[tts_backend], outputs=[voice_dropdown])

        def on_stt_backend_change(value):
            """Handle STT backend switch with appropriate cleanup and feedback."""
            global STT_MODEL, STT_BACKEND_NAME

            SETTINGS["stt_backend"] = value
            save_settings(SETTINGS)

            # Clear cached model to force reload on next transcription
            STT_MODEL = None
            STT_BACKEND_NAME = None

            # Clear CUDA memory if switching away from GPU backend
            if value == "faster_whisper" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"\n[Settings] STT Backend switched to: {value}")
            if value == "sensevoice":
                print("  → Built-in emotion detection (wav2vec2 disabled)")
                print("  → Built-in VAD (Silero disabled)")
                print("  → Running on CUDA:0 (shared with IndexTTS2)")
            elif value == "funasr":
                print("  → External emotion detection (wav2vec2)")
                print("  → Built-in VAD (Silero disabled)")
                print("  → High-accuracy mode (800M params)")
                print("  → Running on CUDA:0 (shared with IndexTTS2)")
            else:
                print("  → External emotion detection (wav2vec2)")
                print("  → External VAD (Silero)")
                print("  → Running on CPU")

            # Update emotion toggle based on backend
            if value == "sensevoice":
                # SenseVoice has built-in emotion - disable external wav2vec2
                emotion_update = gr.update(
                    value=False,
                    interactive=False,
                    info="Not used with SenseVoice (built-in)"
                )
            else:
                # Other backends use external wav2vec2 emotion detection
                emotion_update = gr.update(
                    value=SETTINGS.get("emotion_detection_enabled", True),
                    interactive=True,
                    info="Detect user emotion (~300ms CPU)"
                )

            # Update VAD toggle based on backend
            # All backends support hands-free mode via Silero capture
            # Unified models (SenseVoice/FunASR) use their built-in VAD for filtering
            if value in ("sensevoice", "funasr"):
                vad_update = gr.update(
                    interactive=True,
                    info=f"Silero capture + {value} filtering" if IS_WSL else f"Uses {value} built-in VAD"
                )
            else:
                # Whisper uses external Silero VAD
                vad_update = gr.update(
                    interactive=True,
                    info="Run vad_windows.py" if IS_WSL else "Auto-detects speech"
                )

            return emotion_update, vad_update

        stt_backend.change(fn=on_stt_backend_change, inputs=[stt_backend], outputs=[emotion_toggle, hands_free_toggle])

        # Emotion detection toggle
        def on_emotion_toggle(enabled):
            SETTINGS["emotion_detection_enabled"] = enabled
            save_settings(SETTINGS)
            status = "enabled" if enabled else "disabled (saves CPU when using LM Studio)"
            print(f"[Settings] Emotion detection {status}")
            return enabled
        
        emotion_toggle.change(fn=on_emotion_toggle, inputs=[emotion_toggle])

        # Theme switcher - uses JavaScript to update CSS variables instantly
        def on_theme_change(value):
            SETTINGS["theme"] = value
            save_settings(SETTINGS)
            print(f"[Settings] Theme switched to: {value}")
            return value

        theme_dropdown.change(
            fn=on_theme_change,
            inputs=[theme_dropdown],
            outputs=[theme_dropdown],
            js="(value) => { if (window.ThemeSwitcher) { window.ThemeSwitcher.apply(value); } return value; }"
        )

        # Full file access toggle
        def toggle_full_file_access(enabled):
            global REGISTRY
            CONFIG["ENABLE_FULL_FILE_ACCESS"] = enabled
            # Re-initialize tools with new setting
            from tools import ToolRegistry, TimeTool, WebSearchTool, FileReadTool, FileWriteTool, FileReadFullTool, FileWriteFullTool
            REGISTRY = ToolRegistry()
            REGISTRY.register(TimeTool())
            REGISTRY.register(WebSearchTool())
            REGISTRY.register(FileReadTool(SESSIONS_DIR / "files"))
            REGISTRY.register(FileWriteTool(SESSIONS_DIR / "files"))
            if enabled:
                REGISTRY.register(FileReadFullTool())
                REGISTRY.register(FileWriteFullTool())
            status = "✓ Full file access ENABLED" if enabled else "✓ Full file access disabled"
            print(f"[Tools] {status}")
            # Update display
            tool_display = f"""**Available Tools:** {len(REGISTRY.list_tools())}
- `get_current_time` - Current date/time
- `web_search` - DuckDuckGo (no API key needed)
- `read_file` / `write_file` - Sandbox only
- `read_file_full` / `write_file_full` - {"✓ Enabled" if enabled else "✗ Disabled"}

**Note:** Restart required for some changes to take full effect."""
            return tool_display, gr.update(value=status, visible=True)
        
        full_file_access.change(
            fn=toggle_full_file_access,
            inputs=[full_file_access],
            outputs=[tools_available, tools_status]
        )
        
        
        # Message sending (with TTS warning)
        all_settings = [model_dropdown, temperature_slider, max_tokens_slider, 
                        top_p_slider, freq_penalty_slider, pres_penalty_slider,
                        current_conversation_id, tts_toggle, current_provider,
                        group_enabled, group_members, group_turns, incognito_chk, hands_free_toggle]
        
        def send_with_warning(*args):
            """
            Process message with TTS warning handling.

            Important: For audio stability, we track the latest valid audio
            and only update the component when we have new audio data.
            This prevents the audio player from disappearing during streaming updates.
            """
            # args match inputs: msg, history, char, voice, model...
            # process_group_chat_wrapper is a generator
            latest_audio = None

            for result in process_group_chat_wrapper(*args):
                # Show TTS warning if present
                warning = result[6] if len(result) > 6 else ""
                warning_visible = bool(warning)

                # Get HUD update script (8th element)
                hud_update = result[7] if len(result) > 7 else ""

                # Track the latest valid audio data
                current_audio = result[1]
                if current_audio is not None:
                    latest_audio = current_audio

                # Only update audio if we have valid audio data
                # Use gr.update() to preserve existing audio when no new audio
                audio_update = latest_audio if latest_audio is not None else gr.update()

                yield result[0], audio_update, result[2], result[3], result[4], result[5], gr.update(value=warning, visible=warning_visible), hud_update
        
        msg_input.submit(
            fn=send_with_warning,
            inputs=[msg_input, chatbot, current_character, voice_dropdown] + all_settings[:-1] + [image_input],
            outputs=[chatbot, audio_output, msg_input, memory_display, current_conversation_id, image_input, tts_warning_display, hud_update_trigger]
        )

        send_btn.click(
            fn=send_with_warning,
            inputs=[msg_input, chatbot, current_character, voice_dropdown] + all_settings[:-1] + [image_input],
            outputs=[chatbot, audio_output, msg_input, memory_display, current_conversation_id, image_input, tts_warning_display, hud_update_trigger]
        )
        
        # Voice input
        def voice_with_warning(*args):
            result = process_voice_input(*args)
            warning = result[6] if len(result) > 6 else ""
            warning_visible = bool(warning)
            hud_update = result[7] if len(result) > 7 else ""
            return result[0], result[1], result[2], result[3], result[4], result[5], gr.update(value=warning, visible=warning_visible), hud_update

        audio_input.stop_recording(
            fn=voice_with_warning,
            inputs=[audio_input, chatbot, current_character, voice_dropdown] + all_settings + [image_input],
            outputs=[chatbot, audio_output, transcription_display, memory_display, current_conversation_id, image_input, tts_warning_display, hud_update_trigger]
        ).then(fn=lambda: None, outputs=[audio_input])

        voice_btn.click(
            fn=voice_with_warning,
            inputs=[audio_input, chatbot, current_character, voice_dropdown] + all_settings + [image_input],
            outputs=[chatbot, audio_output, transcription_display, memory_display, current_conversation_id, image_input, tts_warning_display, hud_update_trigger]
        ).then(fn=lambda: None, outputs=[audio_input])
        
        # Clear buttons
        clear_btn.click(
            fn=clear_chat, 
            inputs=[current_character, current_conversation_id], 
            outputs=[chatbot, audio_output, conv_status, conversation_dropdown, current_conversation_id]
        )
        nuke_btn.click(
            fn=clear_all_memory, 
            inputs=[current_character, current_conversation_id], 
            outputs=[chatbot, audio_output, conv_status, conversation_dropdown, current_conversation_id]
        )
        
        # Refresh stats
        def refresh_stats(char_id):
            stats = MEMORY_MANAGER.get_stats(char_id)
            gpu = get_gpu_memory_info()
            gpu_text = f"\n**GPU:** {gpu['allocated']:.1f}/{gpu['total']:.1f} GB" if gpu else ""
            return f"""**Interactions:** {stats.get('total_interactions', 0)}
**Episodic:** {stats.get('episodic_count', 0)}
**Semantic:** {stats.get('semantic_count', 0)}
**Model:** {stats.get('embedding_model', 'unknown')}{gpu_text}"""
        
        refresh_stats_btn.click(fn=refresh_stats, inputs=[current_character], outputs=[memory_stats])
        
        # Reset Memory button
        def reset_character_memory(character_id):
            """Reset just the memory (not conversations) for current character"""
            MEMORY_MANAGER.clear_character_memory(character_id)
            
            # Re-add default memories
            character = CHARACTER_MANAGER.get_character(character_id)
            if character:
                MEMORY_MANAGER.activate_character(character_id)
                for mem in character.initial_memories:
                    MEMORY_MANAGER.add_semantic_memory(character_id, mem, importance=0.9)
                for trait in character.personality_traits:
                    MEMORY_MANAGER.add_procedural_memory(character_id, f"Personality: {trait}")
            
            stats = MEMORY_MANAGER.get_stats(character_id)
            stats_text = f"""**Interactions:** {stats.get('total_interactions', 0)}
**Episodic:** {stats.get('episodic_count', 0)}
**Semantic:** {stats.get('semantic_count', 0)}"""
            
            return stats_text, "*Memory reset - ready for fresh start*", gr.update(value="✓ Memory cleared!", visible=True)
        
        reset_memory_btn.click(
            fn=reset_character_memory,
            inputs=[current_character],
            outputs=[memory_stats, memory_display, reset_memory_status]
        )

        # Memory Editor handlers
        def browse_character_memories(character_id, memory_type):
            """Browse memories for the current character"""
            try:
                memories = MEMORY_MANAGER.storage.get_memories_by_character(
                    character_id,
                    memory_type=memory_type,
                    limit=20
                )
                if not memories:
                    return "*No memories found for this type*"

                lines = [f"**{memory_type.title()} Memories** ({len(memories)} shown)\n"]
                for mem in memories:
                    # Truncate content for display
                    content_preview = mem.content[:100].replace('\n', ' ')
                    if len(mem.content) > 100:
                        content_preview += "..."
                    importance = f"{mem.importance_score:.2f}"
                    lines.append(f"**ID:** `{mem.id}`")
                    lines.append(f"- {content_preview}")
                    lines.append(f"- *Importance: {importance} | Created: {mem.created_at.strftime('%Y-%m-%d %H:%M')}*\n")

                return "\n".join(lines)
            except Exception as e:
                return f"*Error browsing memories: {e}*"

        browse_memories_btn.click(
            fn=browse_character_memories,
            inputs=[current_character, memory_type_filter],
            outputs=[memory_list_display]
        )

        def update_character_memory(character_id, memory_id, new_content):
            """Update a memory's content"""
            if not memory_id or not memory_id.strip():
                return gr.update(value="Please enter a memory ID", visible=True)

            memory_id = memory_id.strip()

            # Check if memory exists
            existing = MEMORY_MANAGER.get_memory(memory_id)
            if not existing:
                return gr.update(value=f"Memory not found: {memory_id}", visible=True)

            # Update content if provided
            content = new_content.strip() if new_content else None
            if not content:
                return gr.update(value="Please enter new content to update", visible=True)

            success = MEMORY_MANAGER.update_memory(
                memory_id=memory_id,
                content=content
            )

            if success:
                return gr.update(value=f"Updated memory {memory_id[:16]}...", visible=True)
            else:
                return gr.update(value="Failed to update memory", visible=True)

        update_memory_btn.click(
            fn=update_character_memory,
            inputs=[current_character, memory_id_input, memory_content_input],
            outputs=[memory_editor_status]
        )

        def delete_character_memory(character_id, memory_id):
            """Delete a memory"""
            if not memory_id or not memory_id.strip():
                return gr.update(value="Please enter a memory ID", visible=True)

            memory_id = memory_id.strip()

            # Check if memory exists
            existing = MEMORY_MANAGER.get_memory(memory_id)
            if not existing:
                return gr.update(value=f"Memory not found: {memory_id}", visible=True)

            success = MEMORY_MANAGER.delete_memory(memory_id, hard=True)

            if success:
                return gr.update(value=f"Deleted memory {memory_id[:16]}...", visible=True)
            else:
                return gr.update(value="Failed to delete memory", visible=True)

        delete_memory_btn.click(
            fn=delete_character_memory,
            inputs=[current_character, memory_id_input],
            outputs=[memory_editor_status]
        )

        # PTT Status Display Timer (shows recording status in UI)
        def update_ptt_status():
            status, display, _, _ = get_ptt_status()
            return display
        
        try:
            ptt_status_timer = gr.Timer(value=0.1)
            ptt_status_timer.tick(
                fn=update_ptt_status,
                outputs=[ptt_status_display]
            )
        except (AttributeError, TypeError):
            print("[Warning] gr.Timer not available for PTT status")
        
        # Init
        def init_app():
            # Cleanup any orphaned recordings from previous sessions
            deleted = cleanup_old_recordings(max_age_hours=1, max_files=5)
            if deleted > 0:
                print(f"[Init] Cleaned up {deleted} orphaned recording files")
            
            char_id = SETTINGS.get("last_character", "hermione")
            MEMORY_MANAGER.activate_character(char_id)
            char = CHARACTER_MANAGER.get_character(char_id)
            
            if char:
                state = MEMORY_MANAGER.get_character_state(char_id)
                if state and state.interaction_count == 0:
                    for mem in char.initial_memories:
                        MEMORY_MANAGER.add_semantic_memory(char_id, mem, importance=0.9)
            
            stats = MEMORY_MANAGER.get_stats(char_id)
            return f"""**Interactions:** {stats.get('total_interactions', 0)}
**Episodic:** {stats.get('episodic_count', 0)}
**Semantic:** {stats.get('semantic_count', 0)}
**Model:** {stats.get('embedding_model', 'unknown')}"""
        
        def on_hands_free_change(enabled):
            global vad_manager_instance
            if vad_manager_instance is None:
                vad_manager_instance = VADManager(RECORDINGS_DIR)
            
            if enabled:
                result = vad_manager_instance.start(threshold=SETTINGS.get("vad_threshold", 0.8))
                SETTINGS["vad_enabled"] = True
                save_settings(SETTINGS)
                
                # Update VAD control file for vad_windows.py
                update_vad_control(enabled=True, tts_playing=False)
                
                # Platform-specific messages
                if result == "wsl_remote":
                    return "🗣️ Voice Active (Remote)"
                elif result == "started":
                    return "🟢 Voice Active (Listening...)"
                elif result.startswith("error"):
                    error_msg = result[7:] if len(result) > 7 else result
                    SETTINGS["vad_enabled"] = False
                    save_settings(SETTINGS)
                    update_vad_control(enabled=False, tts_playing=False)
                    return f"❌ {error_msg[:40]}"
                else:
                    return "🟢 Voice Active"
            else:
                result = vad_manager_instance.stop()
                SETTINGS["vad_enabled"] = False
                save_settings(SETTINGS)
                
                # Update VAD control file to disable VAD
                update_vad_control(enabled=False, tts_playing=False)
                
                if result == "stopped_wsl":
                    return "🎤 PTT Ready (Right Shift)"
                else:
                    return "🎤 PTT Ready (Right Shift)"

        hands_free_toggle.change(
            fn=on_hands_free_change,
            inputs=[hands_free_toggle],
            outputs=[ptt_status_display]
        )

        app.load(fn=init_app, outputs=[memory_stats])

        # --- Vision & Documents ---
        
        def handle_screenshot():
            print("[Screen] Capturing...")
            img = take_screenshot()
            if img:
                return np.array(img) # Convert PIL to numpy for Gradio
            return None
            
        screen_btn.click(fn=handle_screenshot, outputs=[image_input])
        clear_image_btn.click(lambda: None, outputs=[image_input])
        
        def handle_doc_upload(filepath):
            if not filepath:
                return "", "No file"

            print(f"[Doc] Processing {filepath}...")
            ext = os.path.splitext(filepath)[1].lower()
            text = ""

            try:
                if ext == ".pdf":
                    text = extract_text_from_pdf(filepath)
                elif ext == ".docx":
                    # Try python-docx for Word documents
                    try:
                        from docx import Document
                        doc = Document(filepath)
                        text = "\n".join([para.text for para in doc.paragraphs])
                    except ImportError:
                        text = "Error: Install python-docx to read .docx files (pip install python-docx)"
                elif ext == ".csv":
                    import csv
                    with open(filepath, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        rows = list(reader)[:100]  # Limit rows
                        text = "\n".join([", ".join(row) for row in rows])
                        if len(rows) == 100:
                            text += "\n... (showing first 100 rows)"
                elif ext == ".json":
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        text = json.dumps(data, indent=2)
                elif ext in [".yaml", ".yml"]:
                    try:
                        import yaml
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f)
                            text = yaml.dump(data, default_flow_style=False)
                    except ImportError:
                        with open(filepath, "r", encoding="utf-8") as f:
                            text = f.read()
                else:
                    # Plain text files (txt, md, py, js, ts, html, css, xml, log, etc.)
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
            except Exception as e:
                text = f"Error reading file: {e}"

            # Truncate long content
            max_chars = 8000
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n... (truncated, showing first {max_chars} chars)"

            summary = f"✓ Loaded {os.path.basename(filepath)} ({len(text)} chars)"

            # Format context message based on file type
            file_type_desc = {
                ".pdf": "PDF document",
                ".docx": "Word document",
                ".csv": "CSV data",
                ".json": "JSON data",
                ".md": "Markdown document",
                ".py": "Python code",
                ".js": "JavaScript code",
                ".ts": "TypeScript code",
                ".html": "HTML document",
                ".css": "CSS stylesheet",
                ".yaml": "YAML configuration",
                ".yml": "YAML configuration",
                ".xml": "XML document",
                ".log": "Log file",
                ".txt": "text file"
            }.get(ext, "file")

            context_msg = f"I uploaded a {file_type_desc}. Here is the content:\n\n```\n{text}\n```\n\nPlease analyze this."
            return context_msg, summary

        doc_upload.upload(
            fn=handle_doc_upload,
            inputs=[doc_upload],
            outputs=[msg_input, doc_status]
        )
        
        # 1. PTT Status Update Timer (Updates the "Voice Status" box)
        def update_ptt_display_loop():
            _, display, _, _ = get_ptt_status()
            return display
            
        ptt_display_timer = gr.Timer(value=0.2) # Fast update for responsiveness
        ptt_display_timer.tick(fn=update_ptt_display_loop, outputs=[ptt_status_display])

        # 2. PTT Processing Timer (Checks for recording files)
        ptt_process_timer = gr.Timer(value=0.5)
        ptt_process_timer.tick(
            fn=check_ptt_recording,
            inputs=[
                chatbot, current_character, current_voice,
                model_dropdown, temperature_slider, max_tokens_slider, top_p_slider,
                freq_penalty_slider, pres_penalty_slider, current_conversation_id,
                tts_toggle, current_provider, group_enabled, group_members, group_turns,
                incognito_chk, image_input
            ],
            outputs=[chatbot, audio_output, transcription_display, memory_display, current_conversation_id, image_input, tts_warning_display]
        )
    
    return app


# ============================================================================
# Main
# ============================================================================

def process_group_chat_wrapper(user_message, chat_history, character_id, voice_file,
                              model, temperature, max_tokens, top_p, freq_penalty,
                              pres_penalty, conversation_id, tts_enabled, llm_provider,
                              group_enabled, group_members, group_turns, incognito, image):
    """
    Natural multi-character group chat handler.

    Behavior:
    - Detects who the user is addressing (by name or context)
    - Addressed character(s) respond with their own voice
    - If no one addressed, primary character responds
    - "Both" or "everyone" triggers multiple responses

    Yields updates for streaming UI updates.
    """

    # Thinking indicator HTML
    thinking_html = '''<div class="thinking-indicator">
        <div class="thinking-bar-container">
            <div class="thinking-bar"></div>
        </div>
        <span>Thinking...</span>
    </div>'''

    # 1. Single Chat Mode (group disabled or only one character)
    if not group_enabled or len(group_members) < 2:
        # Yield thinking state first
        user_formatted = format_message_with_extras(user_message)
        thinking_history = chat_history + [
            {"role": "user", "content": user_formatted},
            {"role": "assistant", "content": thinking_html}
        ]
        yield (thinking_history, None, "", gr.update(), conversation_id, None, "", "")

        result = process_message_with_memory(
            user_message, chat_history, character_id, voice_file,
            model, temperature, max_tokens, top_p, freq_penalty,
            pres_penalty, conversation_id, tts_enabled, llm_provider, image,
            incognito=incognito
        )
        yield result
        return

    # 2. Group Chat Mode - Natural Conversation
    from group_manager import GROUP_MANAGER
    
    # Set up group with character names for detection
    GROUP_MANAGER.set_active_characters(group_members, CHARACTER_MANAGER)
    GROUP_MANAGER.enable(True)
    
    current_history = chat_history
    
    # Ensure primary character is in members list
    active_members = list(group_members)
    if character_id not in active_members:
        active_members.insert(0, character_id)
    
    # Detect who should respond based on user's message
    speakers = GROUP_MANAGER.get_speakers_for_message(user_message, character_id, int(group_turns))
    
    print(f"[Group] User message: {user_message[:50]}...")
    print(f"[Group] Speakers to respond: {[s[0] for s in speakers]}")
    
    last_audio = None
    last_conv_id = conversation_id
    responses_collected = []
    
    for i, (speaker_id, context_hint) in enumerate(speakers):
        speaker_char = CHARACTER_MANAGER.get_character(speaker_id)
        if not speaker_char:
            continue
            
        speaker_voice = speaker_char.default_voice
        
        # For subsequent speakers, modify the prompt to include context
        if i == 0:
            # First speaker responds to user directly
            prompt_for_speaker = user_message
        else:
            # Subsequent speakers see what others said
            prev_responses = "\n".join([f"{r['name']}: {r['text'][:100]}..." for r in responses_collected])
            prompt_for_speaker = f"[Group conversation - {responses_collected[-1]['name']} just responded to the user]\n\nUser said: {user_message}\n\n{prev_responses}\n\nNow respond as {speaker_char.name}. Don't repeat what was already said."
        
        print(f"[Group] Speaker {i+1}: {speaker_id} ({speaker_char.name})")
        
        # Process message for this speaker
        result = process_message_with_memory(
            prompt_for_speaker, current_history, speaker_id, speaker_voice,
            model, temperature, max_tokens, top_p, freq_penalty,
            pres_penalty, last_conv_id, tts_enabled, llm_provider, 
            image if i == 0 else None,  # Only first speaker sees image
            incognito=incognito
        )
        
        # Extract response details
        updated_history = result[0]
        audio = result[1]
        last_conv_id = result[4]
        
        # Get the response text from history
        response_text = ""
        if updated_history:
            last_item = updated_history[-1]
            if isinstance(last_item, dict):
                response_text = last_item.get("content", "")
            elif isinstance(last_item, (list, tuple)) and len(last_item) > 1:
                response_text = last_item[1]
        
        # Track response for context
        responses_collected.append({
            'id': speaker_id,
            'name': speaker_char.name,
            'text': response_text
        })
        
        # Update history for next speaker
        current_history = updated_history
        last_audio = audio
        
        # Yield result after each speaker (for streaming UI)
        yield result
    
    # Save final conversation state
    if last_conv_id and last_conv_id != "new":
        save_conversation(character_id, last_conv_id, current_history)


if __name__ == "__main__":
    # Force lapis lazuli color at very start of execution
    sys.stdout.write("\033[38;2;0;191;255m")
    sys.stderr.write("\033[38;2;0;191;255m")
    sys.stdout.flush()
    sys.stderr.flush()

    print("\n" + "="*60)
    print("   IndexTTS2 Voice Agent")
    print("   Multi-Character • Memory • Tools • Vision • MCP")
    print(f"   Platform: {PLATFORM.title()}" + (" (WSL)" if IS_WSL else ""))
    print("="*60)

    # Ensure lapis stays active
    sys.stdout.write("\033[38;2;0;191;255m")

    # Move all status logs AFTER the banner
    print(f"\n[Platform] {PLATFORM.title()}" + (" (WSL)" if IS_WSL else ""))
    print(f"[Voices] Available: {', '.join(AVAILABLE_VOICES)}")
    
    if TTS_AVAILABLE:
        print("[TTS] IndexTTS2 module found")
        print("[Startup] Loading TTS model...")
        try:
            init_tts()
        except Exception as e:
            print(f"[Error] TTS: {e}")
    else:
        print("[Startup] TTS not available - voice synthesis disabled")
    
    print("[Startup] Checking STT availability...")
    try:
        if init_whisper():
            print(f"[STT] Whisper ready ({WHISPER_MODEL_SIZE})")
    except Exception as e:
        print(f"[Error] STT: {e}")
    
    LM_STUDIO_AVAILABLE, LM_STUDIO_MODEL = check_lm_studio_available()
    if LM_STUDIO_AVAILABLE and LM_STUDIO_MODEL:
        print(f"[LM Studio] ✓ Connected - Model: {LM_STUDIO_MODEL}")
    elif LM_STUDIO_AVAILABLE:
        print("[LM Studio] ⚠ Connected but no model loaded")

    # Initialize MCP servers
    print("[Startup] Connecting to MCP servers...")
    mcp_ok = init_mcp()

    # Feature summary instead of scattered warnings
    features = []
    if TTS_AVAILABLE: features.append("Voice Synthesis")
    if init_whisper(): features.append("Speech-to-Text")
    if PIL_AVAILABLE: features.append("Vision")

    # Tool count - both local registry and MCP
    local_tool_count = len(REGISTRY.list_tools())
    mcp_tool_count = len(MCP_TOOLS_CACHE) if MCP_TOOLS_CACHE else 0
    total_tools = local_tool_count + mcp_tool_count
    features.append(f"Tools ({total_tools}: {local_tool_count} local + {mcp_tool_count} MCP)")

    if SCREEN_AVAILABLE: features.append("Screen Capture")
    if PDF_AVAILABLE: features.append("Docs (PDF/TXT/MD/DOCX/CSV/JSON/Code)")
    
    # Re-apply lapis color before features
    sys.stdout.write("\033[38;2;0;191;255m")
    print(f"[Features] Enabled: {', '.join(features)}")

    print(f"\n✓ Starting on http://127.0.0.1:{SERVER_PORT}")
    print(f"✓ Press Ctrl+C to stop\n")
    
    app = create_ui()
    
    # Auto-start VAD if enabled in settings
    if SETTINGS.get("vad_enabled"):
        if vad_manager_instance is None:
            vad_manager_instance = VADManager(RECORDINGS_DIR)
        vad_manager_instance.start(threshold=SETTINGS.get("vad_threshold", 0.8))
        update_vad_control(enabled=True, tts_playing=False)
    else:
        # Ensure control file shows disabled if VAD is off
        update_vad_control(enabled=False, tts_playing=False)

    # Final color enforcement before uvicorn/gradio takes over
    sys.stdout.write("\033[38;2;0;191;255m")
    sys.stdout.flush()

    # Use 0.0.0.0 for share mode to allow external connections
    server_host = "0.0.0.0" if SHARE_MODE else "127.0.0.1"

    if SHARE_MODE:
        print(f"\n{'='*60}")
        print("📱 MOBILE/REMOTE ACCESS MODE")
        print("="*60)
        print("A public HTTPS URL will be generated below.")
        print("Use this URL on your phone or any device.")
        print("="*60 + "\n")

    app.launch(
        server_port=SERVER_PORT,
        server_name=server_host,
        inbrowser=False,
        share=SHARE_MODE,  # Enables Gradio's public HTTPS URL
        show_error=True
    )
