# Voice Chat TTS/STT Backend Fixes Plan

**Date:** 2026-01-11
**Status:** In Progress

---

## Completed Fixes

### 1. Kokoro UnboundLocalError (FIXED)
**File:** `voice_chat_app.py:2289`

Removed local import that shadowed global `KokoroTTS` import. This was blocking ALL TTS backends.

### 2. Voice Dropdown Crash (FIXED)
**File:** `voice_chat_app.py:1306-1324`

Added validation that `last_voice` is in the choices list before returning. Prevents crash when switching TTS backends.

### 3. Supertonic Voice Selection (FIXED)
**File:** `voice_chat_app.py:2279-2287`

Added check for valid Supertonic preset names. Now uses selected voice directly instead of hashing it.

---

## Remaining Work

### 4. Audio Playback Debug
Supertonic generates audio (`[TTS] Generated 6.61s of audio`) but it doesn't play. Need to trace return path to Gradio.

### 5. Replace Supertonic with Soprano-80M
- Install: `pip install soprano-tts`
- Create `audio/backends/soprano.py`
- Update `voice_chat_app.py` init/generate paths

### 6. Documentation Update
- Update `.claude/CLAUDE.md`
- Update `README.md`
- Create `docs/BACKENDS.md`

---

## Quick Reference

### TTS Backends
| Backend | Device | Speed | Voice Cloning |
|---------|--------|-------|---------------|
| IndexTTS2 | CUDA:0 | 2x | Yes |
| Kokoro | CPU | 10x | No (presets) |
| Supertonic | CPU | 167x | No (presets) |
| Soprano (NEW) | CUDA | 2000x | No (single) |

### STT Backends
| Backend | Device | VAD | Emotion |
|---------|--------|-----|---------|
| faster_whisper | CPU | External | External |
| SenseVoice | CUDA | Built-in | Built-in |
| FunASR | CUDA | Built-in | External |

---

## Files Modified

- `voice_chat_app.py` - Bug fixes for TTS
- `docs/plans/2026-01-11-tts-stt-fixes.md` - This file

## Files To Create

- `audio/backends/soprano.py` - Soprano backend
- `docs/BACKENDS.md` - Backend reference
