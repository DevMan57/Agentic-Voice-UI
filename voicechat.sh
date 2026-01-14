#!/bin/bash
# ============================================================================
# IndexTTS2 Voice Chat - Linux Launcher
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "  ╔═══════════════════════════════════════════════════════════╗"
echo "  ║                                                           ║"
echo "  ║   ##### #   # ####  ##### #   # ##### ##### ####          ║"
echo "  ║     #   ##  # #   # #      # #    #     #   #             ║"
echo "  ║     #   # # # #   # ###     #     #     #   ####          ║"
echo "  ║     #   #  ## #   # #      # #    #     #      #          ║"
echo "  ║   ##### #   # ####  ##### #   #   #     #   ####          ║"
echo "  ║                                                           ║"
echo "  ║            Voice Chat with Multi-Character Memory         ║"
echo "  ╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

show_menu() {
    echo -e "${GREEN}"
    echo "  [1]  Start Voice Chat"
    echo "       - PTT: Hold Shift to record (requires sudo or input group)"
    echo "       - Hands-Free: Toggle in UI (auto-detects speech)"
    echo "       - http://127.0.0.1:7861"
    echo ""
    echo "  [2]  Character & Memory Manager"
    echo "       - Create/edit characters"
    echo "       - Manage memories and conversations"
    echo "       - http://127.0.0.1:7863"
    echo ""
    echo "  [3]  Start PTT Listener Only (requires sudo)"
    echo "  [4]  Install Dependencies"
    echo "  [5]  Exit"
    echo -e "${NC}"
}

# Activate virtual environment if it exists (prioritize local .venv)
activate_venv() {
    if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
    elif [ -f "$PARENT_DIR/.venv/bin/activate" ]; then
        source "$PARENT_DIR/.venv/bin/activate"
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
    else
        echo -e "${YELLOW}⚠ No virtual environment found, using system Python${NC}"
    fi
}

start_voice_chat() {
    echo -e "\n${BLUE}Starting Voice Chat...${NC}\n"
    
    mkdir -p "$SCRIPT_DIR/recordings"
    mkdir -p "$SCRIPT_DIR/generated_audio"
    
    echo "ready|0|Starting..." > "$SCRIPT_DIR/recordings/ptt_status.txt"
    
    activate_venv
    
    echo -e "${GREEN}"
    echo "  ┌──────────────────────────────────────────────────────────┐"
    echo "  │  URL:  http://127.0.0.1:7861                             │"
    echo "  │                                                          │"
    echo "  │  Input Modes (toggle in UI):                             │"
    echo "  │    PTT:        Run 'sudo python audio/ptt_linux.py' in   │"
    echo "  │                another terminal, then hold Shift         │"
    echo "  │    Hands-Free: Check the box in UI - auto-detects speech │"
    echo "  │                                                          │"
    echo "  │  Stop: Ctrl+C                                            │"
    echo "  └──────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
    
    cd "$SCRIPT_DIR"
    
    # Suppress noisy warnings from dependencies (torch, transformers, gradio)
    export PYTHONWARNINGS="ignore::ResourceWarning,ignore::DeprecationWarning,ignore::FutureWarning"
    export PYTHONTRACEMALLOC=0
    
    python tts2_agent.py
}

start_manager() {
    echo -e "\n${BLUE}Starting Character Manager...${NC}\n"
    
    activate_venv
    
    echo -e "${GREEN}"
    echo "  ┌──────────────────────────────────────────────────────────┐"
    echo "  │  URL:  http://127.0.0.1:7863                             │"
    echo "  │  Stop: Ctrl+C                                            │"
    echo "  └──────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
    
    cd "$SCRIPT_DIR"
    python character_manager_ui.py
}

start_ptt() {
    echo -e "\n${BLUE}Starting PTT Listener...${NC}"
    echo -e "${YELLOW}Note: Requires sudo for keyboard access${NC}\n"
    
    activate_venv
    
    cd "$SCRIPT_DIR"
    if [ "$EUID" -ne 0 ]; then
        echo -e "${YELLOW}Running with sudo...${NC}"
        sudo python audio/ptt_linux.py
    else
        python audio/ptt_linux.py
    fi
}

install_deps() {
    echo -e "\n${BLUE}Installing Dependencies...${NC}\n"
    
    activate_venv
    
    echo "Installing Python packages (this may take a few minutes)..."
    pip install -r requirements.txt
    
    # Install optional dependencies for enhanced features
    echo ""
    echo "Installing optional audio dependencies..."
    pip install sounddevice soundfile pyaudio 2>/dev/null || echo "Optional: some audio packages not installed"
    
    echo ""
    echo -e "${GREEN}✓ Dependencies installed!${NC}"
    echo ""
    echo "Note: For PTT on Linux, you need to run with sudo or add yourself to the 'input' group:"
    echo "  sudo usermod -a -G input \$USER"
    echo "  (then log out and back in)"
    echo ""
    read -p "Press Enter to continue..."
}

# Main loop
while true; do
    show_menu
    read -p "  Select option [1-5]: " choice
    
    case $choice in
        1) start_voice_chat ;;
        2) start_manager ;;
        3) start_ptt ;;
        4) install_deps ;;
        5) exit 0 ;;
        *) echo -e "${RED}Invalid choice!${NC}"; sleep 1 ;;
    esac
done
