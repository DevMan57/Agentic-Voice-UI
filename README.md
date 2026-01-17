# TTS2 Voice Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Hybrid%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)](https://github.com)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com)

**Multi-character AI voice agent with Knowledge Graph memory.**

Self-contained project with all models and dependencies bundled. Designed for **Hybrid** environments: Run the heavy AI core (IndexTTS2) in **WSL** for Linux performance, while keeping your tools (LM Studio, Microphone) on **Windows**.


---

## Key Features

*   **Multi-Character:** Isolated memory graphs and voices (Hermione, Lisbeth, Assistant).
*   **Graph Memory:** SQLite-based Knowledge Graph with GraphRAG community detection.
*   **Agent Skills:** MCP tools, Everything search, web search, file access, and more.
*   **Dual TTS:** IndexTTS2 (voice cloning) or Kokoro (fast ONNX, ~80ms latency).
*   **Emotion Detection:** wav2vec2-based speech emotion recognition.
*   **Hybrid Architecture:** Seamlessly bridges WSL (AI) and Windows (Audio/Tools).
*   **Flexible AI:** Supports OpenRouter (Cloud) and LM Studio (Local).

---

## Quick Start (Hybrid / Windows)

> **New User?** Read the **[Installation Guide](docs/INSTALL.md)** for detailed setup.

### 1. Run the Launcher
**Right-click `VoiceChat.bat` → Run as administrator**

> **Note:** Administrator privileges are required for WSL2 installation and keyboard hooks (PTT).

### 2. Install
Select **Option [5] Install Dependencies**.
*   Sets up Python venv in WSL (for AI).
*   Sets up Audio tools in Windows.

### 3. Configure
Edit `config.env`:
```bash
OPENROUTER_API_KEY=sk-or-v1-your-key...
# LM_STUDIO_HOST=  <-- Leave commented for Auto-Detection!
```

### 4. Launch
Select **Option [1] Start Voice Chat**.

---

## PC Migration

**Moving to a new PC?** This project is now fully self-contained!

**Quick summary:**
1. Copy `C:\AI\tts2-voice-agent` to external drive (~9GB)
2. On new PC: Copy folder and run `python setup_new_pc.py`

All models and dependencies are bundled in the `models/` and `lib/` directories.

---

## Architecture

```
                        WINDOWS HOST
    ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
    │  PTT / VAD   │   │  LM Studio   │   │ Microphone  │
    │   (Python)   │   │ (Local AI)   │   │  (Input)    │
    └──────┬───────┘   └──────▲───────┘   └──────┬──────┘
           │ Audio            │ HTTP             │ Audio
           ▼                  ▼                  ▼
    ═══════╪══════════════════╪══════════════════╪═══════════
           │                  │                  │
           │             WSL (UBUNTU)            │
    ┌──────▼───────┐   ┌──────┴───────┐   ┌──────▼──────┐
    │  App Server  │◄─►│  LLM Client  │   │   Whisper   │
    │ (Gradio UI)  │   │   (Logic)    │   │    STT      │
    └──────┬───────┘   └──────┬───────┘   └─────────────┘
           │                  │
           ▼                  ▼
    ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
    │  Services    │   │ Graph Memory │   │  Emotion    │
    │   Layer ◆    │   │ + GraphRAG   │   │  Detection  │
    └──────┬───────┘   └──────────────┘   └─────────────┘
           │
           ▼
    ┌──────────────┐   ┌──────────────┐
    │  IndexTTS2   │   │   Kokoro     │
    │  (High Fid)  │   │   (Fast)     │
    └──────────────┘   └──────────────┘

◆ Phase 1 Complete: Service layer extracted for better maintainability
```

---

## Mobile Access

Access the voice agent from your phone or any device on your network.

### Quick Start
1. Right-click `VoiceChat.bat` → **Run as administrator**
2. Select **Option [2] Mobile/Remote Access**
3. A public HTTPS URL will be generated (e.g., `https://xxxxx.gradio.live`)
4. Open the URL on your phone

### Features
- **Mobile PTT Button** - Large touch-friendly "HOLD TO TALK" button
- **PWA Support** - Install as a webapp on your phone (Add to Home Screen)
- **HTTPS Required** - Microphone access requires secure connection (handled automatically)

### Tips
- The Gradio share URL is temporary (72 hours max)
- For permanent access, consider [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
- PWA install: On mobile browser, tap Share > "Add to Home Screen"

---

## Controls

| Action | Key / Mode |
|--------|------------|
| **Talk (PTT)** | Hold **Right Shift** (Release to send) |
| **Mobile PTT** | Hold the "HOLD TO TALK" button |
| **Hands-Free** | Toggle "Hands-Free Mode" in UI |
| **Stop Audio** | Click "Stop" in UI |

---

## TTS Options

| Backend | Speed | Quality | VRAM | Voice Clone |
|---------|-------|---------|------|-------------|
| **IndexTTS2** | ~800ms | Excellent | 4-6GB | Yes (5s sample) |
| **Kokoro** | ~80ms | Good | ~500MB | Preset voices |

Switch in Settings > Audio Settings > TTS Backend.

---

## Tools

| Tool | Description |
|------|-------------|
| `everything_search` | PC-wide file search (requires Everything) |
| `web_search` | DuckDuckGo search (no API key) |
| `wikipedia` | Wikipedia lookups |
| `read_file` / `write_file` | Sandboxed file access |
| `create_skill` | Agent creates new skills |
| MCP Tools | Via `mcp_config.json` |

---

## Documentation

*   **[Installation Guide](docs/INSTALL.md)** - Step-by-step setup.
*   **[User Manual](docs/USER_MANUAL.md)** - Complete user guide.
*   **[Architecture Visuals](docs/ARCHITECTURE.md)** - Diagrams of Hybrid System & Memory.
*   **[Technical Reference](docs/TECHNICAL_REFERENCE.md)** - Deep dive into Graph Memory & MCP.
*   **[Migration Guide](docs/MIGRATION.md)** - Moving to a new PC.
*   **[Disclaimer](docs/DISCLAIMER.md)** - Legal & Ethical guidelines.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **LM Studio Orange/Red** | Run as **Admin**, Port **1235**. See [Docs](docs/INSTALL.md). |
| **PTT Not Working** | Use **Right Shift**. Ensure `VoiceChat.bat` is running. |
| **Connection Refused** | Check `config.env`. Comment out `LM_STUDIO_HOST`. |
| **Emotion always "sad"** | Likely audio issue - check mic levels |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
