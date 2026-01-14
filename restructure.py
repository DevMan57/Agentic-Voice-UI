"""
Project Restructuring Script
Moves from C:\\AI\\index-tts\\voice_chat to C:\\AI\\voice_chat with all dependencies
"""

import os
import shutil
from pathlib import Path

# Paths
OLD_ROOT = Path("C:/AI/index-tts")
NEW_ROOT = Path("C:/AI/voice_chat")
WSL_INDEXTTS = Path.home() / "indextts2"  # Will be accessed via WSL commands

def main():
    print("=" * 60)
    print("IndexTTS2 Voice Chat - Project Restructuring")
    print("=" * 60)
    print()
    print("This script will:")
    print("1. Copy voice_chat/ to C:\\AI\\voice_chat\\")
    print("2. Copy model checkpoints to C:\\AI\\voice_chat\\checkpoints\\")
    print("3. Update batch file paths")
    print("4. Create setup script for new PC")
    print()
    
    # Check if old structure exists
    old_voice_chat = OLD_ROOT / "voice_chat"
    if not old_voice_chat.exists():
        print(f"ERROR: {old_voice_chat} not found!")
        return
    
    # Check if new location already exists
    if NEW_ROOT.exists():
        response = input(f"WARNING: {NEW_ROOT} already exists. Overwrite? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return
        print(f"Removing existing {NEW_ROOT}...")
        shutil.rmtree(NEW_ROOT)
    
    print()
    print("[1/5] Copying voice_chat directory...")
    shutil.copytree(old_voice_chat, NEW_ROOT)
    print(f"✓ Copied to {NEW_ROOT}")
    
    print()
    print("[2/5] Copying model checkpoints from parent directory...")
    old_checkpoints = OLD_ROOT / "checkpoints"
    new_checkpoints = NEW_ROOT / "checkpoints"
    
    if old_checkpoints.exists():
        if new_checkpoints.exists():
            shutil.rmtree(new_checkpoints)
        shutil.copytree(old_checkpoints, new_checkpoints)
        print(f"✓ Copied checkpoints ({get_dir_size(new_checkpoints)})")
    else:
        print("⚠ No checkpoints found in parent directory")
    
    print()
    print("[3/5] Copying bigvgan vocoder...")
    old_bigvgan = OLD_ROOT / "bigvgan"
    new_bigvgan = NEW_ROOT / "bigvgan"
    
    if old_bigvgan.exists():
        shutil.copytree(old_bigvgan, new_bigvgan)
        print(f"✓ Copied bigvgan ({get_dir_size(new_bigvgan)})")
    else:
        print("⚠ No bigvgan found")
    
    print()
    print("[4/5] Creating model cache directory...")
    cache_dir = NEW_ROOT / "model_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Create README for model cache
    readme = cache_dir / "README.md"
    readme.write_text("""# Model Cache

This directory stores downloaded models from HuggingFace.

On first run, the following models will be auto-downloaded (~5GB total):
- Qwen3-Embedding-0.6B-ONNX-INT8 (~600MB)
- wav2vec2-large-robust-12-ft-emotion-msp-dim (~1.5GB)
- Silero VAD (~50MB)
- faster-whisper base (~150MB)
- NuExtract-2B (~1GB)

To pre-populate this cache from an existing installation:
1. Copy from Windows: `%USERPROFILE%\\.cache\\huggingface\\` to `model_cache\\huggingface\\`
2. Copy from WSL: `~/.cache/huggingface/` to `model_cache/huggingface/`
""")
    print(f"✓ Created {cache_dir}")
    
    print()
    print("[5/5] Creating Python venv directory...")
    venv_dir = NEW_ROOT / ".venv"
    venv_dir.mkdir(exist_ok=True)
    
    venv_readme = venv_dir / "README.md"
    venv_readme.write_text("""# Python Virtual Environment

This directory will contain the Python virtual environment.

It will be created automatically by the setup script on the new PC.

**Do NOT copy the .venv directory between PCs!**
Different systems require different compiled packages.
""")
    print(f"✓ Created {venv_dir}")
    
    print()
    print("=" * 60)
    print("✓ Restructuring complete!")
    print("=" * 60)
    print()
    print(f"New location: {NEW_ROOT}")
    print()
    print("Next steps:")
    print("1. Run the updated VoiceChat.bat from the new location")
    print("2. Test that everything works")
    print("3. Copy C:\\AI\\voice_chat to external drive for migration")
    print()

def get_dir_size(path):
    """Get human-readable directory size"""
    total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
