#!/usr/bin/env python3
"""
Response Streaming Module for IndexTTS2 Voice Chat

Enables streaming LLM responses with sentence-boundary TTS generation.
This significantly improves perceived latency by:
1. Starting TTS generation as soon as complete sentences are available
2. Playing audio while the rest of the response is still generating
3. Pre-caching common phrases for instant playback

Architecture:
- StreamingLLM: Handles streaming API calls with chunk assembly
- SentenceBuffer: Detects sentence boundaries in streaming text
- StreamingTTS: Generates audio for sentences as they arrive
- AudioQueue: Manages playback queue for smooth audio output
"""

import os
import re
import time
import queue
import threading
import hashlib
from pathlib import Path
from typing import Optional, Callable, Generator, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

import requests
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class StreamingConfig:
    """Configuration for streaming response system"""
    # Sentence detection
    min_sentence_length: int = 10  # Minimum chars before considering sentence complete
    max_sentence_wait: float = 2.0  # Max seconds to wait for sentence to complete
    
    # Audio caching
    enable_phrase_cache: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("./sessions/audio_cache"))
    max_cache_size_mb: int = 100
    
    # Playback
    gap_between_sentences_ms: int = 100  # Small gap between sentence audio
    prefetch_sentences: int = 2  # Number of sentences to pre-generate


# Sentence ending patterns
SENTENCE_ENDINGS = re.compile(r'[.!?]+(?:\s|$)|[。！？]+')
SOFT_BREAKS = re.compile(r'[,;:]+\s|[，；：]+')


# ============================================================================
# Sentence Buffer
# ============================================================================

class SentenceBuffer:
    """
    Buffers streaming text and emits complete sentences.
    Handles edge cases like abbreviations, quotes, etc.
    """
    
    # Common abbreviations that don't end sentences
    ABBREVIATIONS = {
        'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 
        'vs', 'etc', 'e.g', 'i.e', 'fig', 'inc', 'ltd',
        'st', 'ave', 'blvd', 'no', 'vol', 'rev'
    }
    
    def __init__(self, min_length: int = 10):
        self.buffer = ""
        self.min_length = min_length
        self.sentences_emitted = 0
    
    def add(self, text: str) -> List[str]:
        """Add text to buffer, return list of complete sentences"""
        self.buffer += text
        return self._extract_sentences()
    
    def flush(self) -> Optional[str]:
        """Flush remaining text as final sentence"""
        if self.buffer.strip():
            sentence = self.buffer.strip()
            self.buffer = ""
            self.sentences_emitted += 1
            return sentence
        return None
    
    def _extract_sentences(self) -> List[str]:
        """Extract complete sentences from buffer"""
        sentences = []
        
        while True:
            # Find potential sentence end
            match = SENTENCE_ENDINGS.search(self.buffer)
            if not match:
                break
            
            end_pos = match.end()
            potential_sentence = self.buffer[:end_pos].strip()
            
            # Check minimum length
            if len(potential_sentence) < self.min_length:
                break
            
            # Check for abbreviations
            words = potential_sentence.lower().split()
            if words and words[-1].rstrip('.!?') in self.ABBREVIATIONS:
                break
            
            # Check for unclosed quotes
            quote_count = potential_sentence.count('"') + potential_sentence.count("'")
            if quote_count % 2 != 0:
                # Look for closing quote after sentence end
                next_quote = self.buffer.find('"', end_pos)
                if next_quote == -1:
                    next_quote = self.buffer.find("'", end_pos)
                if next_quote != -1 and next_quote < end_pos + 50:
                    end_pos = next_quote + 1
                    potential_sentence = self.buffer[:end_pos].strip()
            
            # Valid sentence found
            sentences.append(potential_sentence)
            self.buffer = self.buffer[end_pos:].lstrip()
            self.sentences_emitted += 1
        
        return sentences
    
    def reset(self):
        """Reset buffer state"""
        self.buffer = ""
        self.sentences_emitted = 0


# ============================================================================
# Streaming LLM Client
# ============================================================================

class StreamingLLMClient:
    """
    Handles streaming API calls to LLM providers.
    Supports OpenRouter, OpenAI, and LM Studio.
    """
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "x-ai/grok-4.1-fast",
        timeout: int = 60
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: List[Dict] = None
    ) -> Generator[str, None, None]:
        """
        Stream response from LLM.
        Yields text chunks as they arrive.
        """
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        if tools:
            payload["tools"] = tools
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                line_str = line.decode('utf-8')
                if not line_str.startswith('data: '):
                    continue
                
                data_str = line_str[6:]  # Remove 'data: ' prefix
                if data_str == '[DONE]':
                    break
                
                try:
                    data = json.loads(data_str)
                    delta = data.get('choices', [{}])[0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            print(f"[Streaming] Error: {e}")
            yield f"[Error: {e}]"


# ============================================================================
# Audio Cache
# ============================================================================

class PhraseCache:
    """
    Cache for pre-generated common phrases.
    Speeds up responses by avoiding TTS for frequent phrases.
    """
    
    COMMON_PHRASES = [
        # Acknowledgments
        "I understand.",
        "I see.",
        "That's interesting.",
        "Good question.",
        "Let me think about that.",
        "Hmm, let me consider this.",
        
        # Transitions
        "Well,",
        "So,",
        "Actually,",
        "You know,",
        "To be honest,",
        
        # Responses
        "Yes, exactly.",
        "That's right.",
        "I agree.",
        "Interesting point.",
        "Good observation.",
        
        # Character-specific (Hermione)
        "Oh, that's fascinating!",
        "I've read about that somewhere.",
        "Brilliant!",
    ]
    
    def __init__(self, cache_dir: Path, character_id: str = "default"):
        self.cache_dir = Path(cache_dir) / character_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, str]:
        """Load cache index from file"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_index(self):
        """Save cache index to file"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _get_hash(self, text: str) -> str:
        """Get hash for text phrase"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()[:12]
    
    def get(self, text: str) -> Optional[str]:
        """Get cached audio path for text, if available"""
        text_hash = self._get_hash(text)
        if text_hash in self.index:
            path = self.cache_dir / self.index[text_hash]
            if path.exists():
                return str(path)
        return None
    
    def put(self, text: str, audio_path: str):
        """Cache audio for text phrase"""
        text_hash = self._get_hash(text)
        filename = f"{text_hash}.wav"
        cache_path = self.cache_dir / filename
        
        try:
            import shutil
            shutil.copy(audio_path, cache_path)
            self.index[text_hash] = filename
            self._save_index()
        except Exception as e:
            print(f"[Cache] Failed to cache audio: {e}")
    
    def pregenerate(self, tts_function: Callable[[str], str]):
        """Pre-generate audio for common phrases"""
        for phrase in self.COMMON_PHRASES:
            if not self.get(phrase):
                try:
                    audio_path = tts_function(phrase)
                    if audio_path:
                        self.put(phrase, audio_path)
                        print(f"[Cache] Pre-generated: {phrase[:30]}...")
                except Exception as e:
                    print(f"[Cache] Failed to pre-generate '{phrase}': {e}")
    
    def get_size_mb(self) -> float:
        """Get total cache size in MB"""
        total = 0
        for f in self.cache_dir.glob("*.wav"):
            total += f.stat().st_size
        return total / (1024 * 1024)
    
    def cleanup(self, max_size_mb: int):
        """Remove oldest files if cache exceeds max size"""
        current_size = self.get_size_mb()
        if current_size <= max_size_mb:
            return
        
        files = sorted(
            self.cache_dir.glob("*.wav"),
            key=lambda f: f.stat().st_mtime
        )
        
        for f in files:
            if current_size <= max_size_mb * 0.8:  # Clean to 80% capacity
                break
            try:
                size = f.stat().st_size / (1024 * 1024)
                f.unlink()
                current_size -= size
                
                # Update index
                for k, v in list(self.index.items()):
                    if v == f.name:
                        del self.index[k]
                        break
            except:
                pass
        
        self._save_index()


# ============================================================================
# Audio Queue
# ============================================================================

class AudioQueue:
    """
    Manages audio playback queue for smooth streaming output.
    Handles gaps between sentences and preloading.
    """
    
    def __init__(self, gap_ms: int = 100):
        self.queue = queue.Queue()
        self.gap_ms = gap_ms
        self.is_playing = False
        self.playback_thread = None
        self.on_playback_start: Optional[Callable] = None
        self.on_playback_end: Optional[Callable] = None
        self.stop_flag = threading.Event()
    
    def add(self, audio_path: str, text: str = ""):
        """Add audio file to playback queue"""
        self.queue.put((audio_path, text))
    
    def start_playback(self):
        """Start background playback thread"""
        if self.is_playing:
            return
        
        self.stop_flag.clear()
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()
    
    def stop_playback(self):
        """Stop playback and clear queue"""
        self.stop_flag.set()
        self.is_playing = False
        
        # Clear queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                pass
    
    def _playback_loop(self):
        """Background playback loop"""
        self.is_playing = True
        
        while not self.stop_flag.is_set():
            try:
                audio_path, text = self.queue.get(timeout=0.5)
                
                if self.on_playback_start:
                    self.on_playback_start(text)
                
                self._play_audio(audio_path)
                
                if self.on_playback_end:
                    self.on_playback_end(text)
                
                # Gap between sentences
                if self.gap_ms > 0:
                    time.sleep(self.gap_ms / 1000.0)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AudioQueue] Playback error: {e}")
        
        self.is_playing = False
    
    def _play_audio(self, audio_path: str):
        """Play audio file"""
        try:
            # Try pygame first (best quality)
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    if self.stop_flag.is_set():
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.1)
                return
            except:
                pass
            
            # Try sounddevice + soundfile
            try:
                import sounddevice as sd
                import soundfile as sf
                data, samplerate = sf.read(audio_path)
                sd.play(data, samplerate)
                sd.wait()
                return
            except:
                pass
            
            # Fallback: system command
            import subprocess
            import platform
            
            if platform.system() == 'Windows':
                subprocess.run(
                    ['powershell', '-c', f'(New-Object Media.SoundPlayer "{audio_path}").PlaySync()'],
                    capture_output=True
                )
            elif platform.system() == 'Darwin':
                subprocess.run(['afplay', audio_path], capture_output=True)
            else:
                subprocess.run(['aplay', audio_path], capture_output=True)
                
        except Exception as e:
            print(f"[AudioQueue] Failed to play audio: {e}")


# ============================================================================
# Streaming Response Handler
# ============================================================================

class StreamingResponseHandler:
    """
    Orchestrates streaming response with sentence-by-sentence TTS.
    
    Usage:
        handler = StreamingResponseHandler(
            llm_client=StreamingLLMClient(...),
            tts_function=my_tts_function,
        )
        
        for event in handler.stream(messages):
            if event['type'] == 'text':
                print(event['text'], end='')
            elif event['type'] == 'audio':
                print(f"[Audio: {event['path']}]")
    """
    
    def __init__(
        self,
        llm_client: StreamingLLMClient,
        tts_function: Callable[[str], str],
        config: StreamingConfig = None,
        phrase_cache: PhraseCache = None
    ):
        self.llm = llm_client
        self.tts = tts_function
        self.config = config or StreamingConfig()
        self.cache = phrase_cache
        
        # State
        self.sentence_buffer = SentenceBuffer(min_length=self.config.min_sentence_length)
        self.audio_queue = AudioQueue(gap_ms=self.config.gap_between_sentences_ms)
        self.full_response = ""
        
        # TTS thread pool for parallel generation
        self.tts_queue = queue.Queue()
        self.tts_results = queue.Queue()
        self.tts_workers = []
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream response with text and audio events.
        
        Events:
            {'type': 'text', 'text': '...', 'full': '...'}
            {'type': 'sentence', 'text': '...'}
            {'type': 'audio', 'path': '...', 'text': '...'}
            {'type': 'done', 'full_text': '...'}
            {'type': 'error', 'message': '...'}
        """
        self.sentence_buffer.reset()
        self.full_response = ""
        
        # Start TTS worker
        self._start_tts_worker()
        
        try:
            for chunk in self.llm.stream(messages, temperature, max_tokens):
                self.full_response += chunk
                
                # Emit text event
                yield {
                    'type': 'text',
                    'text': chunk,
                    'full': self.full_response
                }
                
                # Check for complete sentences
                sentences = self.sentence_buffer.add(chunk)
                for sentence in sentences:
                    yield {'type': 'sentence', 'text': sentence}
                    
                    # Queue for TTS
                    self.tts_queue.put(sentence)
                
                # Check for TTS results
                while True:
                    try:
                        result = self.tts_results.get_nowait()
                        yield {
                            'type': 'audio',
                            'path': result['path'],
                            'text': result['text']
                        }
                    except queue.Empty:
                        break
            
            # Flush remaining text
            final_sentence = self.sentence_buffer.flush()
            if final_sentence:
                yield {'type': 'sentence', 'text': final_sentence}
                self.tts_queue.put(final_sentence)
            
            # Wait for TTS to complete
            self.tts_queue.put(None)  # Signal end
            
            # Drain remaining results
            while True:
                try:
                    result = self.tts_results.get(timeout=10)
                    if result is None:
                        break
                    yield {
                        'type': 'audio',
                        'path': result['path'],
                        'text': result['text']
                    }
                except queue.Empty:
                    break
            
            yield {
                'type': 'done',
                'full_text': self.full_response
            }
            
        except Exception as e:
            yield {
                'type': 'error',
                'message': str(e)
            }
        finally:
            self._stop_tts_worker()
    
    def _start_tts_worker(self):
        """Start background TTS generation worker"""
        def worker():
            while True:
                try:
                    text = self.tts_queue.get(timeout=1)
                    if text is None:
                        self.tts_results.put(None)
                        break
                    
                    # Check cache first
                    if self.cache:
                        cached = self.cache.get(text)
                        if cached:
                            self.tts_results.put({'path': cached, 'text': text})
                            continue
                    
                    # Generate TTS
                    try:
                        audio_path = self.tts(text)
                        if audio_path:
                            # Cache if enabled
                            if self.cache:
                                self.cache.put(text, audio_path)
                            self.tts_results.put({'path': audio_path, 'text': text})
                    except Exception as e:
                        print(f"[StreamingTTS] Error: {e}")
                        
                except queue.Empty:
                    continue
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.tts_workers.append(thread)
    
    def _stop_tts_worker(self):
        """Stop TTS workers"""
        for _ in self.tts_workers:
            self.tts_queue.put(None)
        self.tts_workers = []


# ============================================================================
# Convenience Functions
# ============================================================================

def stream_with_tts(
    messages: List[Dict[str, str]],
    api_key: str,
    tts_function: Callable[[str], str],
    model: str = "x-ai/grok-4.1-fast",
    on_text: Callable[[str], None] = None,
    on_audio: Callable[[str], None] = None
) -> str:
    """
    Convenience function for streaming response with TTS.
    Returns full response text.
    """
    llm = StreamingLLMClient(api_key=api_key, model=model)
    handler = StreamingResponseHandler(llm, tts_function)
    
    full_text = ""
    for event in handler.stream(messages):
        if event['type'] == 'text':
            full_text = event['full']
            if on_text:
                on_text(event['text'])
        elif event['type'] == 'audio':
            if on_audio:
                on_audio(event['path'])
    
    return full_text


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Response Streaming Test")
    print("=" * 40)
    
    # Test sentence buffer
    buffer = SentenceBuffer()
    
    test_text = "Hello there! How are you doing today? I hope you're having a great time. This is a test."
    
    print("\nTesting sentence buffer with streaming simulation:")
    for i in range(0, len(test_text), 5):
        chunk = test_text[i:i+5]
        print(f"Chunk: '{chunk}'")
        sentences = buffer.add(chunk)
        for s in sentences:
            print(f"  -> Sentence: {s}")
    
    final = buffer.flush()
    if final:
        print(f"  -> Final: {final}")
    
    print("\nDone!")
