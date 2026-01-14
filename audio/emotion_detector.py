"""
Speech Emotion Recognition (SER) using wav2vec2

Detects emotional state from audio using dimensional model (Valence, Arousal, Dominance).

Model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
- Pre-trained on diverse, noisy speech data
- Pruned to 12 layers for fast inference
- Outputs continuous VAD scores (0-1)

Usage:
    detector = EmotionDetector()
    result = detector.analyze("audio.wav")
    print(result)  # {'valence': 0.45, 'arousal': 0.78, 'dominance': 0.52, 'label': 'frustrated'}

NOTE: This model requires custom EmotionModel class - AutoModelForAudioClassification
does NOT work correctly and will output near-zero values.
"""

import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np

# Lazy imports for heavy dependencies
_model = None
_processor = None
_model_lock = threading.Lock()
_EmotionModel = None  # Custom model class


@dataclass
class EmotionResult:
    """Result from emotion analysis"""
    valence: float  # 0-1, pleasantness (low=negative, high=positive)
    arousal: float  # 0-1, intensity (low=calm, high=excited)
    dominance: float  # 0-1, control (low=submissive, high=dominant)
    label: str  # Human-readable emotion label
    confidence: float  # Overall confidence in the prediction

    def to_dict(self) -> Dict[str, Any]:
        return {
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance,
            'label': self.label,
            'confidence': self.confidence
        }

    def to_prompt_context(self) -> str:
        """Format for injection into LLM prompt"""
        return f"[User emotional state: {self.label} (arousal={self.arousal:.2f}, valence={self.valence:.2f})]"


class EmotionDetector:
    """
    Speech Emotion Recognition using wav2vec2.

    Uses the audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim model
    which outputs dimensional emotion scores (Valence, Arousal, Dominance).
    """

    MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    SAMPLE_RATE = 16000

    # Emotion label mapping based on VAD values
    # Based on Russell's circumplex model of affect (with wider neutral zone)
    EMOTION_MAP = {
        # (valence_threshold, arousal_threshold, dominance_threshold) -> label
        'excited': (0.6, 0.7, 0.5),      # High V, High A
        'happy': (0.65, 0.5, 0.5),       # High V, Medium A
        'pleased': (0.55, 0.5, 0.5),     # Medium-High V, Medium A
        'content': (0.55, 0.3, 0.5),     # High V, Low A
        'relaxed': (0.45, 0.2, 0.5),     # Medium V, Very Low A
        'calm': (0.4, 0.35, 0.5),        # Medium-Low V, Low A (NOT sad)
        'neutral': (0.5, 0.45, 0.5),     # Medium everything
        'frustrated': (0.35, 0.7, 0.4),  # Low V, High A, Low D
        'angry': (0.3, 0.7, 0.7),        # Low V, High A, High D
        'anxious': (0.4, 0.7, 0.35),     # Medium V, High A, Low D
        'upset': (0.3, 0.5, 0.4),        # Low V, Medium A
        'sad': (0.25, 0.2, 0.35),        # Very Low V, Very Low A (strict)
    }

    def __init__(self, device: str = None):
        """
        Initialize the emotion detector.

        Args:
            device: 'cuda', 'cpu', or None for auto-detection
        """
        self.device = device
        self._model = None
        self._processor = None
        self._loaded = False

    def _load_model(self):
        """Lazy load the model on first use"""
        global _model, _processor, _EmotionModel

        if self._loaded:
            return

        with _model_lock:
            if _model is not None:
                self._model = _model
                self._processor = _processor
                self._loaded = True
                return

            try:
                import torch
                import torch.nn as nn
                from transformers import Wav2Vec2Processor
                from transformers.models.wav2vec2.modeling_wav2vec2 import (
                    Wav2Vec2Model,
                    Wav2Vec2PreTrainedModel,
                )

                print(f"[SER] Loading emotion model: {self.MODEL_ID}")

                # Force CPU to reserve GPU memory for IndexTTS2
                # wav2vec2 runs efficiently on CPU with Ryzen 7800X3D
                if self.device is None:
                    self.device = "cpu"

                # Define custom model classes required by audeering model
                # These are necessary because AutoModelForAudioClassification doesn't work
                class RegressionHead(nn.Module):
                    """Regression head for dimensional emotion prediction."""
                    def __init__(self, config):
                        super().__init__()
                        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                        self.dropout = nn.Dropout(config.final_dropout)
                        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

                    def forward(self, features, **kwargs):
                        x = features
                        x = self.dropout(x)
                        x = self.dense(x)
                        x = torch.tanh(x)
                        x = self.dropout(x)
                        x = self.out_proj(x)
                        return x

                class EmotionModel(Wav2Vec2PreTrainedModel):
                    """Speech emotion model outputting arousal, dominance, valence."""
                    def __init__(self, config):
                        super().__init__(config)
                        self.config = config
                        self.wav2vec2 = Wav2Vec2Model(config)
                        self.classifier = RegressionHead(config)
                        self.init_weights()

                    def forward(self, input_values):
                        outputs = self.wav2vec2(input_values)
                        hidden_states = outputs[0]
                        hidden_states = torch.mean(hidden_states, dim=1)
                        logits = self.classifier(hidden_states)
                        return hidden_states, logits

                # Store for use in analyze()
                _EmotionModel = EmotionModel

                # Load processor and model with correct classes
                self._processor = Wav2Vec2Processor.from_pretrained(self.MODEL_ID)
                self._model = EmotionModel.from_pretrained(self.MODEL_ID)
                self._model.to(self.device)
                self._model.eval()

                # Cache globally
                _model = self._model
                _processor = self._processor

                self._loaded = True
                print(f"[SER] Model loaded on {self.device}")

            except Exception as e:
                print(f"[SER] Failed to load model: {e}")
                print("[SER] Install with: pip install transformers torch torchaudio")
                self._loaded = False
                raise

    def analyze(
        self,
        audio_path: str = None,
        audio_array: np.ndarray = None,
        sample_rate: int = None
    ) -> Optional[EmotionResult]:
        """
        Analyze emotion from audio.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            audio_array: Numpy array of audio samples (alternative to path)
            sample_rate: Sample rate of audio_array (required if using array)

        Returns:
            EmotionResult with VAD scores and label, or None on failure
        """
        try:
            self._load_model()
        except Exception:
            return None

        try:
            import torch
            # torchaudio removed to avoid Windows FFmpeg dependency hell
            import scipy.io.wavfile as wav
            from scipy import signal

            # Load audio
            if audio_path:
                try:
                    import time
                    # Force absolute path immediately
                    path_str = os.path.abspath(str(audio_path).strip().strip("'").strip('"'))
                    
                    # Retry logic for file access (race condition handling)
                    for i in range(5):
                        if os.path.exists(path_str) and os.path.getsize(path_str) > 0:
                            break
                        time.sleep(1.5)
                        
                    if not os.path.exists(path_str):
                        print(f"[SER] File not found: {path_str}")
                        return None
                        
                    sr, audio_data = wav.read(path_str)
                    
                    # Convert to float32 [-1, 1]
                    if audio_data.dtype == np.int16:
                        audio = audio_data.astype(np.float32) / 32768.0
                    elif audio_data.dtype == np.int32:
                        audio = audio_data.astype(np.float32) / 2147483648.0
                    elif audio_data.dtype == np.uint8:
                        audio = (audio_data.astype(np.float32) - 128) / 128.0
                    else:
                        audio = audio_data.astype(np.float32)
                        
                    # Convert to mono if stereo
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                        
                    # Resample if needed
                    if sr != self.SAMPLE_RATE:
                        audio = signal.resample(
                            audio,
                            int(len(audio) * self.SAMPLE_RATE / sr)
                        )
                except Exception as e:
                    print(f"[SER] Failed to load audio file: {e}")
                    return None

            elif audio_array is not None:
                audio = audio_array
                if sample_rate and sample_rate != self.SAMPLE_RATE:
                    # Resample using scipy
                    from scipy import signal
                    audio = signal.resample(
                        audio,
                        int(len(audio) * self.SAMPLE_RATE / sample_rate)
                    )
            else:
                print("[SER] No audio provided")
                return None

            # Preprocess - processor returns dict with 'input_values'
            processed = self._processor(audio, sampling_rate=self.SAMPLE_RATE)
            input_values = processed['input_values'][0]
            input_values = np.array(input_values).reshape(1, -1)
            input_tensor = torch.from_numpy(input_values).to(self.device)

            # Inference - custom EmotionModel returns (hidden_states, logits)
            with torch.no_grad():
                _, logits = self._model(input_tensor)

            # Extract VAD scores
            # The model outputs [arousal, dominance, valence] in that order
            vad = logits.cpu().numpy()[0]

            # Normalize to 0-1 range (model outputs can be outside this range)
            arousal = float(np.clip(vad[0], 0, 1))
            dominance = float(np.clip(vad[1], 0, 1))
            valence = float(np.clip(vad[2], 0, 1))

            # Determine emotion label
            label = self._vad_to_label(valence, arousal, dominance)

            # Compute confidence (based on how extreme the values are)
            confidence = self._compute_confidence(valence, arousal, dominance)

            return EmotionResult(
                valence=valence,
                arousal=arousal,
                dominance=dominance,
                label=label,
                confidence=confidence
            )

        except Exception as e:
            print(f"[SER] Analysis failed: {e}")
            return None

    def _vad_to_label(self, v: float, a: float, d: float) -> str:
        """Map VAD values to emotion label (Personalized)"""
        # Calibrated: 2026-01-04 21:48
        # Your baselines:
        #   NEUTRAL:    V=0.35, A=0.35
        #   HAPPY:      V=0.51, A=0.47
        #   FRUSTRATED: V=0.27, A=0.46
        #   TIRED:      V=0.19, A=0.14

        # 1. Low arousal zone (calm/tired)
        if a < 0.30:
            if v < 0.25:
                return 'tired'
            return 'calm'

        # 2. High arousal zone (excited/frustrated)
        if a > 0.40:
            if v < 0.40:
                return 'frustrated'
            if v > 0.55:
                return 'excited'
            return 'anxious'

        # 3. Mid arousal zone (happy/annoyed/neutral)
        if v > 0.55:
            return 'happy'
        if v < 0.25:
            return 'annoyed'

        return 'neutral'

    def _compute_confidence(self, v: float, a: float, d: float) -> float:
        """Compute confidence based on how far from neutral"""
        # Distance from neutral (0.5, 0.5, 0.5)
        dist = np.sqrt((v - 0.5) ** 2 + (a - 0.5) ** 2 + (d - 0.5) ** 2)
        # Normalize to 0-1 (max distance is sqrt(0.75) ≈ 0.866)
        confidence = min(1.0, dist / 0.866)
        return float(confidence)

    def analyze_async(
        self,
        audio_path: str,
        callback: callable = None
    ) -> threading.Thread:
        """
        Analyze emotion asynchronously.

        Args:
            audio_path: Path to audio file
            callback: Function to call with EmotionResult when done

        Returns:
            Thread object (already started)
        """
        def _run():
            result = self.analyze(audio_path=audio_path)
            if callback:
                callback(result)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread


# Global singleton for easy access
_detector_instance = None


def get_detector() -> EmotionDetector:
    """Get or create the global EmotionDetector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = EmotionDetector()
    return _detector_instance


def analyze_emotion(audio_path: str) -> Optional[EmotionResult]:
    """Convenience function to analyze emotion from audio file"""
    return get_detector().analyze(audio_path=audio_path)
