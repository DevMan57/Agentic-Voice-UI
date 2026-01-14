# 🐳 Docker Setup Guide

Run IndexTTS2 Voice Agent in a Docker container with full GPU support.

## Prerequisites

### Hardware
- NVIDIA GPU with **12GB+ VRAM** (RTX 3060 minimum)
- 32GB RAM recommended
- 40GB disk space (for models + Docker image)

### Software
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- NVIDIA Container Toolkit (for GPU access)

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/voice_chat.git
cd voice_chat

# 2. Download model checkpoints (~10GB)
# Linux/Mac:
chmod +x docker/download_models.sh
./docker/download_models.sh

# Windows:
docker\download_models.bat

# 3. Configure API key
cp config.env.example config.env
# Edit config.env and add your OpenRouter API key

# 4. Build and run
docker-compose up --build
```

Open http://localhost:7861 in your browser.

---

## Detailed Setup

### Step 1: Install NVIDIA Container Toolkit

**Linux (Ubuntu/Debian):**
```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

**Windows (Docker Desktop):**
1. Install latest NVIDIA drivers (535+)
2. Enable WSL2 backend in Docker Desktop
3. GPU support is automatic with Docker Desktop 4.x+

**Verify GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Step 2: Download Model Checkpoints

The model files are too large to include in the Docker image (~10GB). Download them separately:

**Option A: Using the provided script**
```bash
# Linux/Mac
./docker/download_models.sh

# Windows
docker\download_models.bat
```

**Option B: Manual download**
1. Go to https://huggingface.co/IndexTeam/Index-TTS
2. Download all files
3. Place them in the `checkpoints/` directory

**Expected structure:**
```
checkpoints/
├── config.yaml
├── bigvgan_discriminator_optimizer.pth
├── bigvgan_generator.pth
├── bpe.model
├── gpt2_xl.safetensors
├── s2mel.pth
├── unigram_12000.model
└── ... (other model files)
```

### Step 3: Configure

```bash
cp config.env.example config.env
```

Edit `config.env`:
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Step 4: Build and Run

```bash
# First time (builds the image)
docker-compose up --build

# Subsequent runs
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HOST MACHINE                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Docker Container (GPU)                   │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │  │
│  │  │ IndexTTS2  │  │ Whisper    │  │ Voice Chat     │  │  │
│  │  │ (GPU)      │  │ (CPU)      │  │ (Gradio)       │  │  │
│  │  └────────────┘  └────────────┘  └────────────────┘  │  │
│  │                      │                                │  │
│  │                      ▼                                │  │
│  │              Ports 7861, 7863                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────┼──────────────────────────────┐  │
│  │ Mounted Volumes       │                               │  │
│  │  • checkpoints/  (read-only, ~10GB)                  │  │
│  │  • voice_reference/  (voice samples)                 │  │
│  │  • characters/  (YAML definitions)                   │  │
│  │  • sessions/  (memory, graphs)                       │  │
│  │  • conversations/  (chat history)                    │  │
│  │  • config.env  (API keys)                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────┐  (Optional - for voice input)        │
│  │ PTT/VAD Listener │  Run on HOST for microphone access   │
│  │ (Host Python)    │  → Writes to recordings/             │
│  └──────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Voice Input in Docker

**The Challenge:** Docker containers can't easily access the host's microphone.

**Solutions:**

### Option 1: Text-Only Mode (Simplest)
Just type in the chat interface. TTS output works perfectly.

### Option 2: Host-Side Audio Capture (Recommended)
Run the PTT/VAD listener on your host machine:

```bash
# On your HOST (not in Docker), in a separate terminal:

# Windows
python ptt_windows.py

# Linux
sudo python ptt_linux.py
```

The listener writes to `recordings/` which is mounted into the container.

### Option 3: Audio Passthrough (Advanced)
For Linux hosts with PulseAudio:

```yaml
# Add to docker-compose.yml under voice-chat service:
volumes:
  - /run/user/1000/pulse:/run/user/1000/pulse
environment:
  - PULSE_SERVER=unix:/run/user/1000/pulse/native
```

---

## Customization

### Using Local LLMs (LM Studio)

1. Run LM Studio on your host
2. Enable "Serve on Local Network" in LM Studio
3. Add to `docker-compose.yml`:

```yaml
environment:
  - LM_STUDIO_HOST=host.docker.internal
```

### Changing Ports

```yaml
ports:
  - "8080:7861"   # Voice Chat on port 8080
  - "8081:7863"   # Character Manager on port 8081
```

### Resource Limits

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
    limits:
      memory: 16G
```

---

## Troubleshooting

### "GPU not available"

```bash
# Verify nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# If this fails, reinstall nvidia-container-toolkit
```

### "CUDA out of memory"

1. Close other GPU applications
2. Reduce `shm_size` in docker-compose.yml
3. Ensure no other containers are using the GPU

### "Checkpoints not found"

Verify the checkpoint mount:
```bash
docker-compose exec voice-chat ls -la /app/index-tts/checkpoints/
```

### Container crashes on startup

Check logs:
```bash
docker-compose logs voice-chat
```

Common issues:
- Missing checkpoints
- Invalid config.env
- Insufficient GPU memory

### Slow first startup

First run compiles CUDA kernels (~2-5 minutes). Subsequent starts are faster.

---

## Development

### Rebuild after code changes

```bash
docker-compose build --no-cache
docker-compose up
```

### Shell access

```bash
docker-compose exec voice-chat bash
```

### Run specific commands

```bash
# Character Manager only
docker-compose exec voice-chat python character_manager_ui.py

# Memory test
docker-compose exec voice-chat python -m pytest tests/
```

---

## Image Details

| Component | Version |
|-----------|---------|
| Base Image | nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 |
| Python | 3.10 |
| PyTorch | 2.1.0+cu121 |
| Image Size | ~15GB (without checkpoints) |
| Total with Models | ~25GB |

---

## Uninstall

```bash
# Stop and remove containers
docker-compose down

# Remove image
docker rmi indextts2-voice-chat

# Remove volumes (WARNING: deletes all data!)
docker volume prune

# Keep checkpoints for reuse, delete everything else
rm -rf sessions/ conversations/ recordings/*.wav
```
