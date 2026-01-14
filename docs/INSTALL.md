# 🛠️ Installation Guide

Complete setup guide for IndexTTS2 Voice Agent. This project supports three primary architectures:

1. **[Hybrid (Recommended)](#hybrid-windows--wsl)** - Best for NVIDIA GPU users on Windows (WSL for Core, Windows for Tools).
2. **[Windows Native](#windows-native)** - Easiest setup, but some features (IndexTTS2 training) may be limited.
3. **[Linux Native](#linux-native)** - Best for pure Linux environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Hybrid (Windows + WSL)](#hybrid-windows--wsl)
3. [Windows Native](#windows-native)
4. [Linux Native](#linux-native)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA RTX 3060 (12GB VRAM) | RTX 3090/4090 (24GB VRAM) |
| **RAM** | 16GB | 32GB+ |
| **Storage** | 20GB free | 40GB free |
| **OS** | Windows 10/11 or Ubuntu 22.04+ | Windows 11 + WSL2 |

---

## Hybrid (Windows + WSL)

**The "Power User" Setup.**
This architecture runs the heavy AI core (IndexTTS2, Whisper) inside **WSL (Ubuntu)** to leverage Linux-native GPU libraries, while running hardware-dependent tools (LM Studio, PTT, Audio I/O) on the **Windows Host**.

**Architecture:**
*   **WSL (Ubuntu):** Runs `voice_chat_app.py`, IndexTTS2, Whisper, Vector DB.
*   **Windows Host:** Runs `VoiceChat.bat` (Launcher), LM Studio (LLM), `vad_windows.py` (Microphone).
*   **Bridge:** They communicate via the **WSL Gateway IP** (Auto-detected).

### 1. Enable WSL2

Open PowerShell as Administrator:
```powershell
wsl --install
# Restart computer
```

### 2. Configure WSL Memory (Critical)

Create `C:\Users\%USERNAME%\.wslconfig`:
```ini
[wsl2]
memory=12GB
swap=4GB
localhostForwarding=true
```
*Prevents "CUDA out of memory" crashes.*

### 3. Install Drivers & Tools (Windows)

1.  **NVIDIA Drivers:** Install latest "Game Ready" drivers on Windows. **Do NOT install drivers inside WSL.**
2.  **Git:** Install [Git for Windows](https://git-scm.com/download/win).
3.  **Python:** Install [Python 3.10+](https://www.python.org/downloads/windows/) on Windows.
4.  **LM Studio:** Install [LM Studio](https://lmstudio.ai/).

### 4. Clone Project (Windows)

```cmd
cd C:\
mkdir AI
cd AI
git clone https://github.com/index-labs/index-tts.git index-tts
cd index-tts\voice_chat
```

### 5. Run the Launcher

The magic `VoiceChat.bat` script handles the hybrid installation for you.

```cmd
VoiceChat.bat
```

1.  Select **[3] Install Dependencies**.
    *   It will set up the Python venv inside WSL.
    *   It will install audio libraries on Windows.
2.  Follow the on-screen prompts.

---

## Windows Native

*Simple setup, everything runs on Windows.*

1.  **Install Python 3.10+**
2.  **Install CUDA Toolkit 12.1** (must match PyTorch version).
3.  Clone the repo:
    ```cmd
git clone https://github.com/index-labs/index-tts.git
    ```
4.  Run `VoiceChat.bat` -> **[3] Install Dependencies**.

---

## Linux Native

*For Ubuntu 22.04+.*

1.  **Install System Deps:**
    ```bash
sudo apt update && sudo apt install -y build-essential git ffmpeg python3-venv portaudio19-dev
    ```
2.  **Install CUDA:** Follow NVIDIA's Linux installation guide.
3.  **Clone & Install:**
    ```bash
git clone https://github.com/index-labs/index-tts.git
cd index-tts/voice_chat
./voicechat.sh
    ```

---

## Configuration

### 1. OpenRouter (Cloud LLMs)

Copy `config.env.example` to `config.env`:
```bash
OPENROUTER_API_KEY=sk-or-v1-your-key...
```

### 2. LM Studio (Local LLMs) - **Critical Setup**

For the Hybrid setup to work, LM Studio must listen on the network.

1.  **Run as Administrator** (Right-click shortcut).
2.  Load a model (e.g., Qwen2.5-7B, Llama-3).
3.  Go to **Developer Tab (Server)**.
4.  **Port:** Set to `1235` (Default 1234 conflicts with Windows services).
5.  **Cross-Origin (CORS):** Enable "Serve on Local Network".
6.  **Status:** Ensure the dropdown is **Green** (Listening on `0.0.0.0`).

**Note:** The app automatically finds your Windows IP. Do **not** set `LM_STUDIO_HOST` in `config.env` unless auto-detection fails.

---

## Troubleshooting

### 🔴 LM Studio "Connection Refused" / Orange Status

*   **Cause:** Windows permission block or Port conflict.
*   **Fix 1:** Restart LM Studio as **Administrator**.
*   **Fix 2:** Change port to `1235`.
*   **Fix 3:** Open PowerShell (Admin) and allow the port:
    ```powershell
New-NetFirewallRule -DisplayName "LM Studio 1235" -Direction Inbound -LocalPort 1235 -Protocol TCP -Action Allow
    ```

### 🔇 PTT Not Working

*   **Cause:** Windows Python script not running.
*   **Fix:** Ensure you launch via `VoiceChat.bat` (not direct `python`). Look for the `ptt_windows.py` window or process.
*   **Key:** Use **Right Shift** (Left Shift is for typing).

### 🐢 Slow Performance

*   **Cause:** CPU offloading.
*   **Fix:** Ensure `device="cuda"` is set. Check `nvidia-smi` in WSL to confirm GPU access.
*   **LM Studio:** Increase "GPU Offload" bar to Max.

---

**Need Help?** Check `docs/DOCUMENTATION.md` for deep technical details.