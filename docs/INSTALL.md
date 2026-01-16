# Installation Guide

Complete setup guide for TTS2 Voice Agent.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start (Recommended)](#quick-start-recommended)
3. [Manual Installation](#manual-installation)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA RTX 3060 (8GB VRAM) | RTX 3090/4090 (24GB VRAM) |
| **RAM** | 16GB | 32GB+ |
| **Storage** | 20GB free | 40GB free |

### Software

| Requirement | Details |
|-------------|---------|
| **OS** | Windows 11 (Windows 10 may work) |
| **WSL2** | Windows Subsystem for Linux 2 |
| **Ubuntu** | Ubuntu distribution in WSL2 |
| **NVIDIA Driver** | Version 535+ (Game Ready driver) |

---

## Quick Start (Recommended)

The `VoiceChat.bat` launcher handles everything automatically.

### Step 1: Clone the Repository

```cmd
cd C:\AI
git clone https://github.com/Alchemyst0x/tts2-voice-agent.git
cd tts2-voice-agent
```

### Step 2: Run the Launcher

```cmd
VoiceChat.bat
```

### Step 3: First-Time Setup

Select **[5] Installer** from the main menu, then:

1. **[0] First-Time Setup (WSL2 + Ubuntu)**
   - Run this if you don't have WSL2 or Ubuntu installed
   - Requires running as Administrator
   - May require a reboot
   - After reboot, Ubuntu will prompt you to create a username/password

2. **[1] Full Install (Recommended)**
   - Checks all prerequisites (GPU, WSL2, Ubuntu, CUDA)
   - Installs all Python dependencies
   - Downloads all AI models (~10GB)
   - Takes 15-30 minutes depending on internet speed

### Step 4: Install LM Studio

1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Install and run as Administrator
3. Load a model (e.g., Qwen2.5-7B, Llama-3)
4. Go to **Developer Tab** → Set port to `1235`
5. Enable **"Serve on Local Network"**

### Step 5: Start Voice Agent

Return to `VoiceChat.bat` and select **[1] Voice Agent**.

---

## Manual Installation

If you prefer manual setup or the automated installer fails:

### 1. Install WSL2 and Ubuntu

Open PowerShell as Administrator:

```powershell
# Install WSL2
wsl --install

# Restart your computer, then install Ubuntu
wsl --install -d Ubuntu
```

After restart, Ubuntu will open. Create your Linux username and password.

### 2. Configure WSL Memory

Create `C:\Users\%USERNAME%\.wslconfig`:

```ini
[wsl2]
memory=12GB
swap=4GB
localhostForwarding=true
```

Restart WSL: `wsl --shutdown`

### 3. Install NVIDIA Drivers

Install the latest "Game Ready" drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers).

**Important:** Do NOT install NVIDIA drivers inside WSL. Windows drivers pass through automatically.

### 4. Setup Python Environment (in WSL)

```bash
cd /mnt/c/AI/tts2-voice-agent

# Install system dependencies
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential ffmpeg git-lfs

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Install FunASR (STT)
pip install funasr modelscope
```

### 5. Download Models

```bash
source .venv/bin/activate

# Create models directory
mkdir -p models/{indextts2,supertonic,embeddings,nuextract,kokoro,hf_cache}

# IndexTTS2 (~4.4GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS2', local_dir='models/indextts2', local_dir_use_symlinks=False)"

# Supertonic (~500MB)
git lfs install
git clone https://huggingface.co/neongeckocom/tts-vits-cv-en models/supertonic

# NuExtract (~940MB)
python -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('models/nuextract', exist_ok=True); hf_hub_download('numind/NuExtract-2.0-2B-GGUF', filename='NuExtract-2.0-2B-Q4_K_M.gguf', local_dir='models/nuextract')"

# Embeddings (~1.2GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='models/embeddings/qwen0.6b', local_dir_use_symlinks=False)"

# SenseVoice STT
python -c "from funasr import AutoModel; AutoModel(model='FunAudioLLM/SenseVoiceSmall', device='cpu', hub='hf')"
```

### 6. Windows Audio Dependencies (Optional)

For PTT (Push-to-Talk) keyboard support, install in Windows Command Prompt:

```cmd
pip install keyboard pyaudio numpy
```

**Note:** If you don't have Python installed on Windows, PTT will use a fallback mode. The Voice Agent will still work - you can use the web UI's record button instead.

---

## Configuration

### LM Studio Setup (Critical)

For WSL to communicate with LM Studio on Windows:

1. **Run as Administrator** (required for network binding)
2. Load a model (Qwen2.5-7B recommended)
3. **Developer Tab:**
   - Port: `1235`
   - Enable "Serve on Local Network"
   - Status should show **Green** (listening on 0.0.0.0)

### Environment Variables (Optional)

Create `config.env` in the project root:

```bash
# OpenRouter API (for cloud LLMs)
OPENROUTER_API_KEY=sk-or-v1-your-key...

# TTS Backend: indextts, kokoro, supertonic
TTS_BACKEND=indextts

# STT Backend: faster_whisper, sensevoice, funasr
STT_BACKEND=sensevoice
```

---

## Troubleshooting

### WSL/Ubuntu Issues

**"Ubuntu not found"**
```powershell
wsl --install -d Ubuntu
```

**"CUDA not accessible from WSL"**
- Ensure NVIDIA driver is 535+ on Windows
- Run `nvidia-smi` in WSL - should show your GPU
- If not, update Windows NVIDIA drivers

### LM Studio Issues

**"Connection Refused"**
1. Run LM Studio as Administrator
2. Change port to `1235`
3. Allow through firewall:
   ```powershell
   New-NetFirewallRule -DisplayName "LM Studio 1235" -Direction Inbound -LocalPort 1235 -Protocol TCP -Action Allow
   ```

### PTT Not Working

- Launch via `VoiceChat.bat` (not direct Python)
- Use **Right Shift** key (Left Shift is for typing)
- Check that `ptt_windows.py` process is running

### Slow Performance

- Verify GPU access: `nvidia-smi` in WSL
- Check VRAM usage isn't maxed out
- In LM Studio, increase "GPU Offload" slider

---

## Architecture Overview

```
Windows Host                    WSL2 (Ubuntu)
============                    =============
VoiceChat.bat (launcher)   -->  tts2_agent.py (main app)
LM Studio (LLM on :1235)   <--  HTTP requests
vad_windows.py (mic)       -->  recordings/*.wav
ptt_windows.py (keyboard)  -->  recordings/ptt_status.txt
```

The Windows host handles hardware (microphone, keyboard) while WSL runs the AI models with native CUDA support.

---

**Need Help?** Check `docs/TECHNICAL_REFERENCE.md` for detailed architecture documentation.
