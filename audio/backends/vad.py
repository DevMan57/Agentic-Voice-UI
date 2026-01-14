"""
VAD Backend Re-exports

Re-exports existing VAD implementations from vad_recorder.py as VADBackend-conformant classes.
The existing implementations already conform to the interface pattern.

Usage:
    from audio.backends.vad import SileroVAD, WebRTCVAD, EnergyVAD

    vad = SileroVAD(threshold=0.6)
    is_speech = vad.is_speech(audio_chunk)
"""

from audio.vad_recorder import SileroVAD, WebRTCVAD, EnergyVAD, VADConfig

# The existing VAD classes already conform to the VADBackend interface:
# - is_speech(audio_chunk: np.ndarray) -> bool
# - reset() -> None
#
# They just need the 'name' property added for full conformance.
# Rather than modifying the original classes, we create thin wrappers.


class SileroVADBackend(SileroVAD):
    """Silero VAD with VADBackend interface"""

    @property
    def name(self) -> str:
        return "silero"


class WebRTCVADBackend(WebRTCVAD):
    """WebRTC VAD with VADBackend interface"""

    @property
    def name(self) -> str:
        return "webrtc"


class EnergyVADBackend(EnergyVAD):
    """Energy-based VAD with VADBackend interface"""

    @property
    def name(self) -> str:
        return "energy"


# Export both original classes and wrappers
# The originals are still useful for direct use in vad_recorder.py
__all__ = [
    # Original classes (for compatibility)
    'SileroVAD',
    'WebRTCVAD',
    'EnergyVAD',
    # Interface-conformant wrappers
    'SileroVADBackend',
    'WebRTCVADBackend',
    'EnergyVADBackend',
    # Config
    'VADConfig',
]
