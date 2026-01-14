"""
Emotion to TTS Mapping Utility

Converts detected emotion (from EmotionDetector) to TTS parameters
for both Kokoro (speed) and IndexTTS2 (8-emotion vectors).

IndexTTS2 Eight-Emotion Vector (correct order):
    happy, angry, sad, afraid, disgusted, melancholic, surprised, calm

Usage:
    from audio.emotion_tts import get_tts_params_for_emotion, get_indextts_emotion_params
    
    # For Kokoro
    params = get_tts_params_for_emotion('happy')
    # Returns: {'speed': 1.1, 'pitch': 0.1}
    
    # For IndexTTS2
    emo_params = get_indextts_emotion_params('frustrated')
    # Returns: {'emo_text': 'frustrated and annoyed', 'emo_vector': {'angry': 0.6, 'sad': 0.3, ...}}
"""

from typing import Dict, Optional, Union

# Type hint for EmotionResult without importing to avoid circular deps
try:
    from audio.emotion_detector import EmotionResult
except ImportError:
    EmotionResult = None


# =============================================================================
# IndexTTS2 Eight-Emotion Mapping
# =============================================================================

# CORRECT IndexTTS2 emotion vector order (per official docs):
# [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
#
# Note: emo_text goes in a SEPARATE parameter, NOT in the text itself!
# Wrong: tts.infer(text="[happy] Hello!")  <-- reads "[happy]" literally
# Right: tts.infer(text="Hello!", emo_text="cheerful", use_emo_text=True)

# Mapping from detected emotion labels to IndexTTS2's 8-emotion vector
# Order: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
INDEXTTS_EMOTION_VECTORS = {
    'happy': {
        'happy': 0.8, 'angry': 0.0, 'sad': 0.0, 'afraid': 0.0,
        'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 0.0, 'calm': 0.2
    },
    'excited': {
        'happy': 0.6, 'angry': 0.0, 'sad': 0.0, 'afraid': 0.0,
        'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 0.4, 'calm': 0.0
    },
    'calm': {
        'happy': 0.1, 'angry': 0.0, 'sad': 0.0, 'afraid': 0.0,
        'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 0.0, 'calm': 0.9
    },
    'neutral': {
        'happy': 0.0, 'angry': 0.0, 'sad': 0.0, 'afraid': 0.0,
        'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 0.0, 'calm': 0.7
    },
    'sad': {
        'happy': 0.0, 'angry': 0.0, 'sad': 0.7, 'afraid': 0.0,
        'disgusted': 0.0, 'melancholic': 0.3, 'surprised': 0.0, 'calm': 0.0
    },
    'tired': {
        'happy': 0.0, 'angry': 0.0, 'sad': 0.2, 'afraid': 0.0,
        'disgusted': 0.0, 'melancholic': 0.4, 'surprised': 0.0, 'calm': 0.4
    },
    'angry': {
        'happy': 0.0, 'angry': 0.9, 'sad': 0.0, 'afraid': 0.0,
        'disgusted': 0.1, 'melancholic': 0.0, 'surprised': 0.0, 'calm': 0.0
    },
    'frustrated': {
        'happy': 0.0, 'angry': 0.6, 'sad': 0.1, 'afraid': 0.0,
        'disgusted': 0.2, 'melancholic': 0.1, 'surprised': 0.0, 'calm': 0.0
    },
    'annoyed': {
        'happy': 0.0, 'angry': 0.4, 'sad': 0.0, 'afraid': 0.0,
        'disgusted': 0.3, 'melancholic': 0.0, 'surprised': 0.0, 'calm': 0.3
    },
    'anxious': {
        'happy': 0.0, 'angry': 0.0, 'sad': 0.1, 'afraid': 0.6,
        'disgusted': 0.0, 'melancholic': 0.1, 'surprised': 0.2, 'calm': 0.0
    },
    'fearful': {
        'happy': 0.0, 'angry': 0.0, 'sad': 0.0, 'afraid': 0.8,
        'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 0.2, 'calm': 0.0
    },
    'surprised': {
        'happy': 0.2, 'angry': 0.0, 'sad': 0.0, 'afraid': 0.1,
        'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 0.7, 'calm': 0.0
    },
    'content': {
        'happy': 0.4, 'angry': 0.0, 'sad': 0.0, 'afraid': 0.0,
        'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 0.0, 'calm': 0.6
    },
    'relaxed': {
        'happy': 0.2, 'angry': 0.0, 'sad': 0.0, 'afraid': 0.0,
        'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 0.0, 'calm': 0.8
    },
    # Additional labels from wav2vec2 emotion detector
    'pleased': {
        'happy': 0.5, 'angry': 0.0, 'sad': 0.0, 'afraid': 0.0,
        'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 0.0, 'calm': 0.5
    },
    'upset': {
        'happy': 0.0, 'angry': 0.3, 'sad': 0.4, 'afraid': 0.0,
        'disgusted': 0.1, 'melancholic': 0.2, 'surprised': 0.0, 'calm': 0.0
    },
}

# Natural language descriptions for emo_text parameter
INDEXTTS_EMO_TEXT = {
    'happy': 'cheerful and warm',
    'excited': 'enthusiastic and energetic',
    'calm': 'relaxed and gentle',
    'neutral': 'conversational',
    'sad': 'melancholic and subdued',
    'tired': 'weary and slow',
    'angry': 'intense and forceful',
    'frustrated': 'irritated and tense',
    'annoyed': 'slightly impatient',
    'anxious': 'nervous and uncertain',
    'fearful': 'worried and breathy',
    'surprised': 'amazed and expressive',
    'content': 'pleased and comfortable',
    'relaxed': 'calm and peaceful',
    # Additional labels from wav2vec2 emotion detector
    'pleased': 'satisfied and pleasant',
    'upset': 'distressed and troubled',
}


def get_indextts_emotion_params(
    emotion: Union[str, "EmotionResult", None],
    use_vector: bool = False
) -> Dict[str, any]:
    """
    Get IndexTTS2 emotion parameters for the detected emotion.
    
    Args:
        emotion: Emotion label string or EmotionResult object
        use_vector: If True, include emo_vector list. If False, use emo_text only.
    
    Returns:
        Dict with:
        - 'emo_text': Natural language description (always included)
        - 'emo_vector': 8-value list [happy, angry, sad, fear, hate, love, surprise, neutral]
        - 'emo_alpha': Intensity (0.6 recommended for natural speech)
        - 'use_emo_text': True if emo_text should be enabled
    """
    if emotion is None:
        return {'emo_text': None, 'emo_vector': None, 'emo_alpha': 0.0, 'use_emo_text': False}
    
    # Extract label
    if hasattr(emotion, 'label'):
        label = emotion.label.lower()
        # Use confidence to modulate alpha (capped at 0.6 for natural speech)
        alpha = min(0.6, getattr(emotion, 'confidence', 0.6) * 0.75)
    else:
        label = str(emotion).lower()
        alpha = 0.6  # Recommended default
    
    result = {
        'emo_text': INDEXTTS_EMO_TEXT.get(label, 'conversational'),
        'emo_alpha': alpha,
        'use_emo_text': True,
    }
    
    if use_vector:
        # Convert dict to list in IndexTTS2's expected order:
        # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        vec_dict = INDEXTTS_EMOTION_VECTORS.get(label, INDEXTTS_EMOTION_VECTORS['neutral'])
        result['emo_vector'] = [
            vec_dict.get('happy', 0.0),
            vec_dict.get('angry', 0.0),
            vec_dict.get('sad', 0.0),
            vec_dict.get('afraid', 0.0),
            vec_dict.get('disgusted', 0.0),
            vec_dict.get('melancholic', 0.0),
            vec_dict.get('surprised', 0.0),
            vec_dict.get('calm', 0.0),
        ]
    
    return result


# =============================================================================
# Kokoro TTS Mapping (Speed-based)
# =============================================================================

def get_tts_params_for_emotion(
    emotion: Union[str, "EmotionResult", None],
    character_id: str = None
) -> Dict[str, float]:
    """
    Get Kokoro TTS parameters (speed, pitch) based on detected emotion.
    
    Args:
        emotion: Either an emotion label string ('happy', 'sad', etc.)
                 or an EmotionResult object from emotion_detector
        character_id: Optional character ID for per-character overrides
    
    Returns:
        Dict with 'speed' and 'pitch' adjustments:
        - speed: Multiplier (1.0 = normal, 0.9 = slower, 1.1 = faster)
        - pitch: Offset (-0.2 = lower, 0.0 = normal, 0.2 = higher)
    """
    # Default neutral values
    defaults = {'speed': 1.0, 'pitch': 0.0}
    
    if emotion is None:
        return defaults
    
    # Extract label if EmotionResult object
    if hasattr(emotion, 'label'):
        label = emotion.label.lower()
    else:
        label = str(emotion).lower()
    
    # Try to get config values
    try:
        from config import config
        speed_map = config.emotion.emotion_speed_map
        pitch_map = config.emotion.emotion_pitch_map
        
        # Check for per-character overrides
        if character_id:
            char_speed = config.get_for_character(character_id, f'emotion.speed.{label}')
            char_pitch = config.get_for_character(character_id, f'emotion.pitch.{label}')
            if char_speed is not None:
                speed_map = {**speed_map, label: char_speed}
            if char_pitch is not None:
                pitch_map = {**pitch_map, label: char_pitch}
                
    except ImportError:
        # Fallback hardcoded values if config not available
        speed_map = {
            'calm': 1.0,
            'happy': 1.1,
            'excited': 1.15,
            'sad': 0.9,
            'tired': 0.85,
            'angry': 1.15,
            'frustrated': 1.1,
            'fearful': 1.05,
            'anxious': 1.1,
            'surprised': 1.1,
            'neutral': 1.0,
            'annoyed': 1.05,
            'content': 0.95,
            'relaxed': 0.95,
            # Additional labels from wav2vec2 emotion detector
            'pleased': 1.0,
            'upset': 1.05,
        }
        pitch_map = {
            'calm': 0.0,
            'happy': 0.1,
            'excited': 0.15,
            'sad': -0.15,
            'tired': -0.1,
            'angry': 0.2,
            'frustrated': 0.1,
            'fearful': 0.1,
            'anxious': 0.1,
            'surprised': 0.15,
            'neutral': 0.0,
            'annoyed': 0.05,
            'content': 0.0,
            'relaxed': -0.05,
            # Additional labels from wav2vec2 emotion detector
            'pleased': 0.05,
            'upset': 0.05,
        }
    
    speed = speed_map.get(label, 1.0)
    pitch = pitch_map.get(label, 0.0)
    
    return {
        'speed': speed,
        'pitch': pitch
    }


def apply_emotion_to_audio(
    audio_array,
    sample_rate: int,
    emotion: Union[str, "EmotionResult", None],
    character_id: str = None
):
    """
    Apply emotion-based modifications to audio array.
    
    Currently supports:
    - Speed adjustment (via resampling)
    
    Note: Pitch adjustment requires sophisticated DSP (rubberband, etc.)
          and is not implemented here. IndexTTS2 may support native pitch.
    
    Args:
        audio_array: NumPy array of audio samples
        sample_rate: Original sample rate
        emotion: Emotion label or EmotionResult
        character_id: Optional for per-character config
    
    Returns:
        (modified_audio_array, new_sample_rate)
    """
    import numpy as np
    
    params = get_tts_params_for_emotion(emotion, character_id)
    speed = params['speed']
    
    # If speed is 1.0 (or very close), no modification needed
    if abs(speed - 1.0) < 0.01:
        return audio_array, sample_rate
    
    # Apply speed adjustment via resampling
    # Note: This also changes pitch slightly (like a tape speed change)
    # For true speed-without-pitch change, need rubberband or similar
    try:
        from scipy import signal
        
        # Calculate new length
        new_length = int(len(audio_array) / speed)
        
        # Resample
        modified = signal.resample(audio_array, new_length)
        
        return modified.astype(np.float32), sample_rate
        
    except Exception as e:
        print(f"[EmotionTTS] Speed adjustment failed: {e}")
        return audio_array, sample_rate


# For testing
if __name__ == "__main__":
    print("=" * 60)
    print("Emotion -> TTS Parameter Mapping")
    print("=" * 60)
    
    # Test Kokoro mappings
    print("\n[Kokoro - Speed/Pitch]")
    print("-" * 40)
    test_emotions = ['happy', 'sad', 'angry', 'calm', 'neutral', 'excited', 'frustrated']
    for e in test_emotions:
        params = get_tts_params_for_emotion(e)
        print(f"  {e:12}: speed={params['speed']:.2f}, pitch={params['pitch']:+.2f}")
    
    # Test IndexTTS2 mappings
    print("\n[IndexTTS2 - Emotion Vector]")
    print("-" * 40)
    print("  Vector order: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]")
    print()
    for e in test_emotions:
        params = get_indextts_emotion_params(e, use_vector=True)
        print(f"  {e:12}: emo_text='{params['emo_text']}' (alpha={params['emo_alpha']:.2f})")
        vec = params['emo_vector']
        # Show non-zero values with their indices
        active = [(i, v) for i, v in enumerate(vec) if v > 0.1]
        labels = ['happy', 'angry', 'sad', 'afraid', 'disgusted', 'melancholic', 'surprised', 'calm']
        active_named = {labels[i]: v for i, v in active}
        print(f"               vector={vec}")
        print(f"               active={active_named}")

