"""
Emotion Bridge - Routes emotion detection based on STT backend.

SenseVoice mode:
- Uses built-in emoji-based emotion (happy/sad/angry/neutral)
- Skips wav2vec2 EmotionDetector (saves ~300ms latency)
- Emotion returned from transcribe_with_emotion()

FunASR/Faster-Whisper mode:
- Uses external wav2vec2 EmotionDetector
- Runs async in parallel with LLM (existing pattern)

Usage:
    from audio.emotion_bridge import should_run_external_emotion

    if not should_run_external_emotion(stt_backend):
        # SenseVoice handles emotion internally
        pass
"""

from typing import Optional, Tuple
import numpy as np
from audio.interface import EmotionResult


def get_emotion_for_tts(
    sensevoice_emotion: Optional[EmotionResult],
    stt_backend: str = "faster_whisper"
) -> Optional[EmotionResult]:
    """
    Get emotion result for TTS synthesis.

    Priority:
    1. SenseVoice built-in emotion (if available)
    2. Return None for other backends (wav2vec2 handles externally)

    Args:
        sensevoice_emotion: Emotion from SenseVoice transcribe_with_emotion()
        stt_backend: Current STT backend name

    Returns:
        EmotionResult compatible with get_indextts_emotion_params()
    """
    if sensevoice_emotion is not None:
        return sensevoice_emotion

    if stt_backend == "sensevoice":
        # SenseVoice didn't detect emotion - return neutral
        return EmotionResult(
            valence=0.5,
            arousal=0.5,
            dominance=0.5,
            label="neutral",
            confidence=0.5
        )

    # For other backends, wav2vec2 handles this externally
    return None


def should_run_external_emotion(stt_backend: str) -> bool:
    """
    Check if we need async wav2vec2 emotion detection.

    Returns False if SenseVoice is active (it provides emotion).
    Returns True for FunASR/Faster-Whisper.
    """
    return stt_backend != "sensevoice"


def should_run_external_vad(stt_backend: str, use_sensevoice_vad: bool = True) -> bool:
    """
    Check if we need Silero VAD.

    Returns False if SenseVoice is active with built-in VAD.
    Returns True for FunASR/Faster-Whisper.
    """
    return not (stt_backend == "sensevoice" and use_sensevoice_vad)


__all__ = [
    'get_emotion_for_tts',
    'should_run_external_emotion',
    'should_run_external_vad',
]
