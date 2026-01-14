#!/bin/bash
# ============================================================================
# Download IndexTTS2 Model Checkpoints
# ============================================================================
# This script downloads the required model files from HuggingFace.
# Run this BEFORE starting Docker containers.
#
# Usage:
#   chmod +x download_models.sh
#   ./download_models.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${SCRIPT_DIR}/../checkpoints"

echo "============================================"
echo "IndexTTS2 Model Downloader"
echo "============================================"
echo ""

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install --quiet huggingface_hub
fi

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"
cd "$CHECKPOINT_DIR"

echo "Downloading IndexTTS2 checkpoints to: $CHECKPOINT_DIR"
echo "This may take a while (~10GB)..."
echo ""

# Download from HuggingFace
huggingface-cli download IndexTeam/Index-TTS \
    --local-dir . \
    --local-dir-use-symlinks False

echo ""
echo "============================================"
echo "Download Complete!"
echo "============================================"
echo ""
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "You can now start the Docker container:"
echo "  cd ${SCRIPT_DIR}/.."
echo "  docker-compose up --build"
echo ""
