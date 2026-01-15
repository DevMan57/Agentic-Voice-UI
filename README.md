# Agentic Voice UI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Hybrid%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)](https://github.com)
[![Version](https://img.shields.io/badge/version-2.3.1-blue.svg)](https://github.com)

**Multi-character AI voice agent with Knowledge Graph memory.**

Designed for **Hybrid** environments: Run the heavy AI core (IndexTTS2) in **WSL** for Linux performance, while keeping your tools (LM Studio, Microphone) on **Windows**.


---

## Features

- **Multi-Character System** - Each character has isolated memory, personality, and voice
- **Persistent Memory** - Knowledge graph with semantic search (sqlite-vec)
- **Multiple TTS Backends** - IndexTTS2 (voice cloning), Kokoro (fast), Soprano (ultra-fast GPU)
- **Multiple STT Backends** - Faster-Whisper, SenseVoice (with emotion), FunASR
- **Emotion Detection** - Speech emotion recognition affects TTS output
- **MCP Tool Calling** - Web search, file access, and custom tools
- **Vision Support** - Analyze images and screenshots
- **Document Parsing** - PDF, TXT, MD, DOCX, CSV, JSON, Code files
- **Push-to-Talk or Hands-Free** - Voice Activity Detection (VAD) modes

---

## Requirements

- **Windows 11** with WSL2 enabled
- **Ubuntu** WSL distribution
- **NVIDIA GPU** with CUDA drivers (for TTS/STT acceleration)
- **~10GB disk space** for models

---

## One-Click Installation

### 1. Clone the Repository

```cmd
git clone https://github.com/DevMan57/Agentic-Voice-UI.git
cd Agentic-Voice-UI
```

### 2. Run the Launcher

```cmd
VoiceChat.bat
```

### 3. Install Dependencies

Select **Option [5] Install Dependencies** from the menu.

This automatically:
- Installs WSL packages (python3-venv, git-lfs)
- Creates Python virtual environment
- Installs PyTorch with CUDA support
- Downloads required models
- Sets up Node.js via NVM
- Installs Windows audio tools

### 4. Configure API Key

Create `config.env` from the example:

```cmd
copy config.env.example config.env
```

Edit `config.env` and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Get a free API key at [openrouter.ai](https://openrouter.ai)

### 5. Launch Voice Agent

Select **Option [1] Voice Agent** from the menu.

Open your browser to: **http://localhost:7861**

---

## Mobile Access

Access the voice agent from your phone or any device on your network.

### Quick Start
1. Run `VoiceChat.bat`
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

| Action | Control |
|--------|---------|
| **Push-to-Talk** | Hold **Right Shift**, release to send |
| **Mobile PTT** | Hold the "HOLD TO TALK" button |
| **Hands-Free** | Enable VAD toggle in Settings |
| **Stop Audio** | Click Stop button or press Escape |

---

## TTS Backends

| Backend | Speed | Quality | VRAM | Voice Cloning |
|---------|-------|---------|------|---------------|
| **IndexTTS2** | ~2x realtime | Excellent | 4-6GB | Yes (5s sample) |
| **Kokoro** | ~10x realtime | Good | ~500MB | No (presets) |
| **Supertonic** | ~167x realtime | Good | CPU | No (6 voices) |
| **Soprano** | ~2000x realtime | Good | GPU | No |

## STT Backends

| Backend | Speed | Emotion | Built-in VAD |
|---------|-------|---------|--------------|
| **Faster-Whisper** | Fast | External | No |
| **SenseVoice** | Fast | Yes | Yes |
| **FunASR** | Fast | External | Yes |

---

## Project Structure

```
Agentic-Voice-UI/
├── VoiceChat.bat          # One-click launcher
├── tts2_agent.py          # Main application
├── generate_bat.py        # Installer generator
├── config.env.example     # Configuration template
├── requirements.txt       # Python dependencies
├── audio/
│   ├── backends/          # TTS/STT implementations
│   └── interface.py       # Abstract base classes
├── memory/                # Knowledge graph system
├── skills/                # Character definitions
├── voice_reference/       # Voice sample .wav files
├── lib/indextts/          # Vendored IndexTTS2
└── docs/                  # Documentation
```

---

## Adding Custom Characters

1. Create a new folder in `skills/` (e.g., `skills/my-character/`)
2. Add a `SKILL.md` file with character definition
3. Add a voice reference `.wav` file to `voice_reference/`
4. Restart the application

See `skills/assistant-utility/SKILL.md` for an example.

---

## Adding Custom Voices

1. Record a 5-10 second `.wav` file of the target voice
2. Place it in `voice_reference/`
3. Select it from the Voice dropdown in the UI

Supported formats: WAV, 16kHz or higher, mono or stereo.

---

## Emotion Detection

The app includes Speech Emotion Recognition (SER) that detects your emotional state from your voice and adjusts the AI's TTS output accordingly. When you sound excited, the AI responds with more energy; when you're calm, the response is more relaxed.

### Enabling Emotion Detection

1. In the Voice Agent settings, enable the **Emotion Detection** toggle
2. Use SenseVoice STT backend for best results (has built-in emotion detection)

### Calibrating to Your Voice (Optional)

The SER model works out-of-the-box, but you can calibrate it to YOUR voice for improved accuracy:

1. Run `VoiceChat.bat`
2. Select **Option [6] Calibrate Emotion Detection**
3. Follow the prompts to record samples:
   - **Neutral**: Speak normally (like reading news)
   - **Happy**: Say something with genuine excitement
   - **Frustrated**: Say something with frustration
   - **Tired/Sad**: Say something with low energy
4. The tool automatically updates the emotion thresholds

Each recording is 3 seconds. The calibration only needs to be done once.

**Note**: Calibration is optional. If you skip it, the default thresholds work for most voices. You can also leave Emotion Detection disabled if you prefer consistent TTS output.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **WSL not found** | Enable WSL2: `wsl --install` |
| **CUDA not working** | Update NVIDIA drivers, ensure CUDA is installed |
| **PTT not working** | Use **Right Shift** (not Left Shift) |
| **No audio output** | Check TTS backend, ensure speakers are configured |
| **API errors** | Verify `OPENROUTER_API_KEY` in `config.env` |
| **Models not loading** | Run Option [4] Install Dependencies again |

---

## Local LLM Support

Instead of OpenRouter, you can use a local LLM:

1. Install [LM Studio](https://lmstudio.ai/)
2. Load a model and start the server on port 1235
3. In the app, switch to "LM Studio" provider in Settings

---

## Documentation

- **[Installation Guide](docs/INSTALL.md)** - Detailed setup instructions
- **[User Manual](docs/USER_MANUAL.md)** - Complete feature guide
- **[Technical Reference](docs/TECHNICAL_REFERENCE.md)** - Architecture deep dive
- **[MCP Servers](docs/MCP_SERVERS.md)** - Tool configuration

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [IndexTTS2](https://github.com/index-labs/IndexTTS) - Voice cloning TTS
- [Kokoro](https://github.com/hexgrad/kokoro) - Fast ONNX TTS
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - Optimized STT
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - Emotion-aware STT
- [Gradio](https://gradio.app/) - Web UI framework
