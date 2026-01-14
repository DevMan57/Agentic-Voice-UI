# ============================================================================
# IndexTTS2 Voice Chat - Docker Image
# ============================================================================
# Build:   docker build -t indextts2-voice-chat .
# Run:     docker-compose up
#
# Requirements:
#   - NVIDIA GPU with 12GB+ VRAM
#   - nvidia-container-toolkit installed on host
#   - Model checkpoints downloaded separately (mounted as volume)
# ============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# ============================================================================
# System Dependencies
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# ============================================================================
# Python Dependencies - PyTorch with CUDA
# ============================================================================
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ============================================================================
# Clone and Install IndexTTS2
# ============================================================================
RUN git clone https://github.com/index-labs/index-tts.git /app/index-tts && \
    cd /app/index-tts && \
    pip install --no-cache-dir -e .

# ============================================================================
# Voice Chat Dependencies
# ============================================================================
COPY requirements.txt /app/voice_chat/requirements.txt
RUN pip install --no-cache-dir -r /app/voice_chat/requirements.txt

# Install additional dependencies for Docker environment
RUN pip install --no-cache-dir \
    pyaudio \
    webrtcvad

# ============================================================================
# Copy Voice Chat Application
# ============================================================================
COPY . /app/voice_chat/

# Create necessary directories
RUN mkdir -p /app/voice_chat/recordings \
    /app/voice_chat/sessions/files \
    /app/voice_chat/sessions/memory_storage \
    /app/voice_chat/sessions/graphs \
    /app/voice_chat/conversations \
    /app/voice_chat/generated_audio

# ============================================================================
# Environment Variables
# ============================================================================
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Gradio settings for Docker
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7861

# ============================================================================
# Ports
# ============================================================================
# 7861 - Voice Chat UI
# 7863 - Character Manager UI
EXPOSE 7861 7863

# ============================================================================
# Health Check
# ============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7861/ || exit 1

# ============================================================================
# Entry Point
# ============================================================================
WORKDIR /app/voice_chat

# Default command runs the voice chat app
CMD ["python", "voice_chat_app.py"]
