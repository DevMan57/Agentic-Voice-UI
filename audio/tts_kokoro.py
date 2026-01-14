import os
import json
import urllib.request
import soundfile as sf
import numpy as np
import warnings
from pathlib import Path

# Suppress Kokoro's "words count mismatch" warnings
warnings.filterwarnings("ignore", message=".*words count mismatch.*")

class KokoroTTS:
    def __init__(self, checkpoints_dir: str):
        """
        Initialize Kokoro TTS wrapper.
        
        Args:
            checkpoints_dir: Path to the checkpoints directory (where models will be stored)
        """
        self.kokoro_dir = Path(checkpoints_dir) / "kokoro"
        self.kokoro_dir.mkdir(exist_ok=True, parents=True)
        
        self.model_path = self.kokoro_dir / "kokoro-v0_19.onnx"
        self.voices_path = self.kokoro_dir / "voices.json"
        
        # Check and download model files if missing
        self._ensure_model_exists()
        
        try:
            from kokoro_onnx import Kokoro
            print(f"[Kokoro] Loading model from {self.model_path}")
            self.model = Kokoro(str(self.model_path), str(self.voices_path))
            print("[Kokoro] Model loaded successfully")
        except ImportError:
            print("[Kokoro] Error: kokoro-onnx package not found. Please install dependencies.")
            self.model = None
        except Exception as e:
            print(f"[Kokoro] Error loading model: {e}")
            self.model = None

    def _ensure_model_exists(self):
        """Download model files if they don't exist"""
        
        # Model URL (v0.19 ONNX)
        model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
        voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"
        
        if not self.model_path.exists():
            print(f"[Kokoro] Downloading model to {self.model_path}...")
            try:
                urllib.request.urlretrieve(model_url, self.model_path)
                print("[Kokoro] Model download complete")
            except Exception as e:
                print(f"[Kokoro] Failed to download model: {e}")
                
        if not self.voices_path.exists():
            print(f"[Kokoro] Downloading voices.json to {self.voices_path}...")
            try:
                urllib.request.urlretrieve(voices_url, self.voices_path)
                print("[Kokoro] Voices download complete")
            except Exception as e:
                print(f"[Kokoro] Failed to download voices.json: {e}")

    def get_available_voices(self):
        """Return list of available voices from voices.json"""
        if not self.voices_path.exists():
            return []
        try:
            with open(self.voices_path, 'r') as f:
                data = json.load(f)
                return sorted(list(data.keys()))
        except:
            return []

    def infer(self, voice_name: str, text: str, output_path: str, speed: float = 1.0) -> bool:
        """
        Generate audio from text.
        
        Args:
            voice_name: Name of the voice to use (e.g. 'af_sarah')
            text: Text to synthesize
            output_path: Path to save the wav file
            speed: Speech speed multiplier (default 1.0, range 0.5-2.0)
                   - Lower values = slower speech
                   - Higher values = faster speech
            
        Returns:
            bool: True if successful
        """
        if not self.model:
            print("[Kokoro] Model not initialized")
            return False
            
        try:
            # Map common names to Kokoro voices if needed, or use default
            # Default fallback
            target_voice = "af_sarah"
            
            # If voice_name is a direct match, use it
            available = self.get_available_voices()
            if voice_name in available:
                target_voice = voice_name
            else:
                # Simple heuristic mapping for existing characters
                voice_lower = voice_name.lower()
                if "hermione" in voice_lower: target_voice = "bf_emma"
                elif "gandalf" in voice_lower: target_voice = "am_adam" # closest deep male?
                elif "male" in voice_lower: target_voice = "am_michael"
                elif "female" in voice_lower: target_voice = "af_sarah"
            
            # Clamp speed to valid range
            speed = max(0.5, min(2.0, speed))
            
            # Generate
            # Kokoro create returns samples, sample_rate
            # Version compatibility: Older versions don't have 'lang'
            try:
                samples, sample_rate = self.model.create(
                    text, 
                    voice=target_voice, 
                    speed=speed, 
                    lang="en-us"
                )
            except TypeError:
                # Fallback for older kokoro-onnx versions (v0.2.4)
                samples, sample_rate = self.model.create(
                    text, 
                    voice=target_voice, 
                    speed=speed
                )
            
            # Save to file
            sf.write(output_path, samples, sample_rate)
            return True
            
        except Exception as e:
            print(f"[Kokoro] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return False
