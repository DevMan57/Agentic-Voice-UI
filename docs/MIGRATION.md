# PC Migration Guide - Complete Guide

**Goal:** Copy `C:\AI\voice_chat` to a new PC and get it running quickly.

---

## What You're Copying

**Single folder:** `C:\AI\voice_chat\` (~16GB total)

Contains:
- Voice chat application code
- Model checkpoints (IndexTTS2, Kokoro)
- Character memories and skills
- Configuration files
- Python virtual environment (will be recreated on new PC)

---

## Pre-Migration: Prepare the Folder

### Step 1: Restructure (One-Time, on Current PC)

If you're currently using `C:\AI\index-tts\voice_chat\`, run the restructuring script:

```batch
cd C:\AI\index-tts\voice_chat
python restructure.py
```

This moves everything to `C:\AI\voice_chat\` with all dependencies included.

**IMPORTANT: After restructuring, update the batch file generator:**

1. Edit `generate_bat.py` lines ~128, ~145, ~160
2. Change from: `source ~/indextts2/.venv/bin/activate`
3. Change to: `source .venv/bin/activate`
4. Run: `python generate_bat.py`

This updates the batch file to use the new venv location in the restructured folder.

### Step 2: Copy Model Weights from WSL

The model checkpoints are in WSL. Copy them to the Windows folder:

```batch
:: Copy checkpoints from WSL to Windows
wsl -d Ubuntu -e bash -c "cp -r ~/indextts2/checkpoints /mnt/c/AI/voice_chat/"

:: Copy bigvgan if it exists
wsl -d Ubuntu -e bash -c "cp -r ~/indextts2/bigvgan /mnt/c/AI/voice_chat/ 2>/dev/null"
```

### Step 3: (Optional) Copy Model Cache

Save ~5GB of downloads by copying the model cache:

```batch
:: Copy HuggingFace cache
robocopy "%USERPROFILE%\.cache\huggingface" "C:\AI\voice_chat\model_cache\huggingface" /E /Z
```

Or from WSL:
```batch
wsl -d Ubuntu -e bash -c "cp -r ~/.cache/huggingface /mnt/c/AI/voice_chat/model_cache/"
```

### Step 4: Copy to External Drive

```batch
:: Copy entire folder to external drive (E: example)
robocopy "C:\AI\voice_chat" "E:\voice_chat_backup" /E /Z /R:3 /W:5
```

**Total size:** ~16GB (or ~21GB with model cache)

---

## New PC Setup

### Step 1: Copy Folder

Copy `voice_chat` from external drive to `C:\AI\`:

```batch
robocopy "E:\voice_chat_backup" "C:\AI\voice_chat" /E /Z
```

### Step 2: Install Prerequisites

**Required:**
1. **Windows 11** (22H2 or later)
2. **NVIDIA Driver** (latest from nvidia.com)
3. **Python 3.10+** (from python.org) - Add to PATH!
4. **Node.js 20+** (from nodejs.org) - For MCP servers

**Optional but Recommended:**
5. **LM Studio** (from lmstudio.ai) - For local AI
6. **Everything Search** (from voidtools.com) - For file search
7. **ES.exe CLI** (from voidtools.com/downloads) - Command line tool

### Step 3: Run Auto-Setup

```batch
cd C:\AI\voice_chat
python setup_new_pc.py
```

This script will:
- ✓ Check Windows prerequisites
- ✓ Install WSL2 Ubuntu
- ✓ Install CUDA 12.8 in WSL
- ✓ Create Python virtual environment
- ✓ Install all dependencies (~10-15 minutes)

### Step 4: Restore Model Cache (Optional)

If you copied the model cache:

```batch
:: Restore to Windows
robocopy "C:\AI\voice_chat\model_cache\huggingface" "%USERPROFILE%\.cache\huggingface" /E /Z

:: Or restore to WSL
wsl -d Ubuntu -e bash -c "mkdir -p ~/.cache && cp -r /mnt/c/AI/voice_chat/model_cache/huggingface ~/.cache/"
```

### Step 5: Configure LM Studio (If Using)

1. Open LM Studio
2. Load a model (e.g., Qwen2.5-7B)
3. Go to Server tab:
   - Port: **1235**
   - Enable "Serve on Local Network"
   - Start server

### Step 6: Launch

```batch
cd C:\AI\voice_chat
VoiceChat.bat
```

Select **[1] Voice Agent** → Opens at http://localhost:7861

---

## Troubleshooting

### "Python not found"
- Install Python from python.org
- Check "Add Python to PATH" during installation

### "WSL not installed"
- Run in PowerShell (Admin): `wsl --install`
- Restart computer
- Run `setup_new_pc.py` again

### "NVIDIA driver not found"
- Download from nvidia.com
- Select your GPU model
- Install and restart

### "LM Studio connection refused"
- Ensure "Serve on Local Network" is enabled
- Check port is 1235
- Run LM Studio as Administrator

### "Everything search not working"
- Install Everything from voidtools.com
- Download ES.exe separately
- Add ES.exe location to Windows PATH

---

## What Gets Recreated vs Preserved

**Preserved (copied):**
- ✓ Character memories (`sessions/*.db`)
- ✓ Skills (`skills/`)
- ✓ Voice samples (`voices/`)
- ✓ Configuration (`config.env`, `mcp_config.json`)
- ✓ Model checkpoints (`checkpoints/`)
- ✓ Application code

**Recreated (by setup script):**
- Python virtual environment (`.venv/`)
- WSL Ubuntu installation
- CUDA toolkit
- Python packages

**Auto-downloaded (on first run):**
- HuggingFace models (~5GB) - unless you copied model_cache
- Whisper STT model
- Emotion detection model
- Embedding models

---

## File Structure

```
C:\AI\voice_chat\
├── voice_chat_app.py       # Main application
├── VoiceChat.bat           # Launcher
├── setup_new_pc.py         # Auto-setup script
├── restructure.py          # One-time restructuring
├── config.env              # API keys (IMPORTANT!)
├── mcp_config.json         # MCP server config
├── checkpoints/            # Model weights (~8GB)
│   ├── indextts2/
│   └── kokoro/
├── bigvgan/                # Vocoder (~100MB)
├── sessions/               # Character data
│   ├── memory.db           # SQLite database
│   ├── conversations/
│   └── files/
├── skills/                 # Agent skills
├── voices/                 # Voice samples
├── model_cache/            # Optional: HuggingFace cache
├── .venv/                  # Python venv (recreated)
├── services/               # Business logic
├── ui/                     # UI formatters
├── memory/                 # Memory system
├── audio/                  # Audio processing
└── tools/                  # Agent tools
```

---

## Quick Reference

**Backup command:**
```batch
robocopy "C:\AI\voice_chat" "E:\backup" /E /Z
```

**Restore command:**
```batch
robocopy "E:\backup" "C:\AI\voice_chat" /E /Z
```

**Setup on new PC:**
```batch
cd C:\AI\voice_chat
python setup_new_pc.py
```

**Launch:**
```batch
VoiceChat.bat
```

---

