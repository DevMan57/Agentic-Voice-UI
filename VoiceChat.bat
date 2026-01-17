@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
title IndexTTS2 Voice Agent
mode con: cols=92 lines=42
powershell -command "&{$W=(get-host).ui.rawui;$B=$W.buffersize;$B.height=1000;$W.buffersize=$B;}"
set "WSL_DISTRO=Ubuntu"
set "WIN_SCRIPT_DIR=%~dp0"
set "WIN_SCRIPT_DIR=%WIN_SCRIPT_DIR:~0,-1%"
set "WSL_WIN_PATH=%WIN_SCRIPT_DIR:\=/%"
set "WSL_WIN_PATH=%WSL_WIN_PATH:C:=/mnt/c%"
set "WSL_WIN_PATH=%WSL_WIN_PATH:D:=/mnt/d%"
set "WSL_WIN_PATH=%WSL_WIN_PATH:E:=/mnt/e%"
cd /d "%WIN_SCRIPT_DIR%"
:MENU
cls
echo 
echo ========================================================================================
echo :
echo :   ╔════════════════════════════════════════════════════════════════════════════════╗
echo :   ║  ████████╗████████╗ ███████╗ ██████╗    ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗  ║
echo :   ║  ╚══██╔══╝╚══██╔══╝ ██╔════╝ ╚════██╗   ██║   ██║██╔═══██╗██║██╔════╝██╔════╝  ║
echo :   ║     ██║      ██║    ███████╗  █████╔╝   ██║   ██║██║   ██║██║██║     █████╗    ║
echo :   ║     ██║      ██║    ╚════██║ ██╔═══╝    ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝    ║
echo :   ║     ██║      ██║    ███████║ ███████╗    ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗  ║
echo :   ║     ╚═╝      ╚═╝    ╚══════╝ ╚══════╝     ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝  ║
echo :   ╚════════════════════════════════════════════════════════════════════════════════╝
echo :
echo :                           Multi-Character Memory System
echo :
echo ========================================================================================
echo :
echo :  [1] Voice Agent                                                       Port: 7861
echo :       +-- PTT/VAD + Tools + Memory + Vision
echo :       +-- Docs: PDF,TXT,MD,DOCX,CSV,JSON,Code
echo :
echo :  [2] Mobile/Remote Access                                              HTTPS Share
echo :       +-- Access from phone or any device
echo :       +-- Generates public HTTPS URL
echo :
echo :  [3] Character and Memory Manager                                      Port: 7863
echo :       +-- Create/Edit Personalities
echo :       +-- Manage Knowledge Graph
echo :
echo :  [4] MCP Server Manager                                                Port: 7864
echo :       +-- Configure Agent Tools
echo :       +-- Install MCP Servers
echo :
echo :  [5] Install Dependencies
echo :       +-- Python + Node.js (v20)
echo :
echo :  [6] Calibrate Emotion Detection
echo :       +-- Personalize SER to your voice
echo :
echo :  [7] Exit
echo :
echo ========================================================================================
echo.
set /p "choice=Select option [1-7]: "
if "%choice%"=="1" goto VOICECHAT
if "%choice%"=="2" goto MOBILE
if "%choice%"=="3" goto MANAGER
if "%choice%"=="4" goto MCPMANAGER
if "%choice%"=="5" goto INSTALL
if "%choice%"=="6" goto CALIBRATE
if "%choice%"=="7" exit /b 0
goto MENU
:VOICECHAT
cls
echo 
echo ========================================================================================
echo :
echo :   ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗
echo :   ██║   ██║██╔═══██╗██║██╔════╝██╔════╝    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
echo :   ██║   ██║██║   ██║██║██║     █████╗      ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
echo :   ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝      ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
echo :    ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
echo :     ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   
echo :
echo :                              PTT + VAD + Memory + Tools + Vision
echo :
echo ========================================================================================
echo.
echo   URL: http://localhost:7861
echo.
echo ----------------------------------------------------------------------------------------
echo.
if not exist "recordings" mkdir recordings
echo ready^|0^|Starting... > recordings\ptt_status.txt
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ -d .venv ] && echo VENV_OK || echo VENV_MISSING" > "%TEMP%\venv_check.txt"
set /p VENV_STATUS=<"%TEMP%\venv_check.txt"
if "%VENV_STATUS%"=="VENV_MISSING" (
    echo ERROR: Virtual environment not found!
    echo Please run option [5] Installer first.
    echo.
    pause
    goto MENU
)
start "" /min wscript.exe scripts\ptt_hidden.vbs
start "" /min pythonw audio\vad_windows.py
echo 
wsl -d %WSL_DISTRO% -e bash -c "export NVM_DIR=$HOME/.nvm; [ -s $NVM_DIR/nvm.sh ] && . $NVM_DIR/nvm.sh; cd %WSL_WIN_PATH% && source .venv/bin/activate && python -W ignore tts2_agent.py"
taskkill /f /im pythonw.exe 2>nul
echo.
echo Process exited. Press any key to return to menu...
pause >nul
goto MENU
:MOBILE
cls
echo 
echo ========================================================================================
echo :
echo :   ███╗   ███╗ ██████╗ ██████╗ ██╗██╗     ███████╗
echo :   ████╗ ████║██╔═══██╗██╔══██╗██║██║     ██╔════╝
echo :   ██╔████╔██║██║   ██║██████╔╝██║██║     █████╗  
echo :   ██║╚██╔╝██║██║   ██║██╔══██╗██║██║     ██╔══╝  
echo :   ██║ ╚═╝ ██║╚██████╔╝██████╔╝██║███████╗███████╗
echo :   ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚═╝╚══════╝╚══════╝
echo :
echo :              Phone + Tablet + Any Device Access
echo :
echo ========================================================================================
echo.
echo   This mode generates a public HTTPS URL for your phone.
echo   The URL will appear below when the server starts.
echo.
echo   Features:
echo     - Touch-friendly HOLD TO TALK button
echo     - Install as webapp (Add to Home Screen)
echo     - Works on iOS and Android
echo.
echo ----------------------------------------------------------------------------------------
echo.
if not exist "recordings" mkdir recordings
echo ready^|0^|Starting... > recordings\ptt_status.txt
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ -d .venv ] && echo VENV_OK || echo VENV_MISSING" > "%TEMP%\venv_check.txt"
set /p VENV_STATUS=<"%TEMP%\venv_check.txt"
if "%VENV_STATUS%"=="VENV_MISSING" (
    echo ERROR: Virtual environment not found!
    echo Please run option [5] Install Dependencies first.
    echo.
    pause
    goto MENU
)
start "" /min wscript.exe scripts\ptt_hidden.vbs
start "" /min pythonw audio\vad_windows.py
echo 
wsl -d %WSL_DISTRO% -e bash -c "export NVM_DIR=$HOME/.nvm; [ -s $NVM_DIR/nvm.sh ] && . $NVM_DIR/nvm.sh; cd %WSL_WIN_PATH% && source .venv/bin/activate && SHARE_MODE=1 python -W ignore tts2_agent.py"
taskkill /f /im pythonw.exe 2>nul
echo.
echo Process exited. Press any key to return to menu...
pause >nul
goto MENU
:MANAGER
cls
color 06
chcp 65001 >nul
echo 
echo ========================================================================================
echo :
echo :    ██████╗██╗  ██╗ █████╗ ██████╗  █████╗  ██████╗████████╗███████╗██████╗ ███████╗
echo :   ██╔════╝██║  ██║██╔══██╗██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
echo :   ██║     ███████║███████║██████╔╝███████║██║        ██║   █████╗  ██████╔╝███████╗
echo :   ██║     ██╔══██║██╔══██║██╔══██╗██╔══██║██║        ██║   ██╔══╝  ██╔══██╗╚════██║
echo :   ╚██████╗██║  ██║██║  ██║██║  ██║██║  ██║╚██████╗   ██║   ███████╗██║  ██║███████║
echo :    ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝
echo :
echo :                        Personalities + Memory + Knowledge Graph
echo :
echo ========================================================================================
echo.
echo   URL: http://localhost:7863
echo.
echo ----------------------------------------------------------------------------------------
echo.
echo 
wsl -d %WSL_DISTRO% -e bash -c "export NVM_DIR=$HOME/.nvm; [ -s $NVM_DIR/nvm.sh ] && . $NVM_DIR/nvm.sh; cd %WSL_WIN_PATH% && source .venv/bin/activate 2>/dev/null; python character_manager_ui.py"
goto MENU
:MCPMANAGER
cls
color 04
chcp 65001 >nul
echo 
echo ========================================================================================
echo :
echo :   ███╗   ███╗ ██████╗██████╗     ███╗   ███╗ █████╗ ███╗   ██╗ █████╗  ██████╗ ███████╗██████╗ 
echo :   ████╗ ████║██╔════╝██╔══██╗    ████╗ ████║██╔══██╗████╗  ██║██╔══██╗██╔════╝ ██╔════╝██╔══██╗
echo :   ██╔████╔██║██║     ██████╔╝    ██╔████╔██║███████║██╔██╗ ██║███████║██║  ███╗█████╗  ██████╔╝
echo :   ██║╚██╔╝██║██║     ██╔═══╝     ██║╚██╔╝██║██╔══██║██║╚██╗██║██╔══██║██║   ██║██╔══╝  ██╔══██╗
echo :   ██║ ╚═╝ ██║╚██████╗██║         ██║ ╚═╝ ██║██║  ██║██║ ╚████║██║  ██║╚██████╔╝███████╗██║  ██║
echo :   ╚═╝     ╚═╝ ╚═════╝╚═╝         ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
echo :
echo                         Configure Agent Tools + Install Servers
echo :
echo ========================================================================================
echo.
echo   URL: http://localhost:7864
echo.
echo ----------------------------------------------------------------------------------------
echo.
echo 
wsl -d %WSL_DISTRO% -e bash -c "export NVM_DIR=$HOME/.nvm; [ -s $NVM_DIR/nvm.sh ] && . $NVM_DIR/nvm.sh; cd %WSL_WIN_PATH% && source .venv/bin/activate 2>/dev/null; python mcp_manager_ui.py"
goto MENU
:INSTALL
cls
color 08
chcp 65001 >nul
echo 
echo ========================================================================================
echo :
echo :   ██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗     ███████╗██████╗ 
echo :   ██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║     ██╔════╝██╔══██╗
echo :   ██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║     █████╗  ██████╔╝
echo :   ██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║     ██╔══╝  ██╔══██╗
echo :   ██║██║ ╚████║███████║   ██║   ██║  ██║███████╗███████╗███████╗██║  ██║
echo :   ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝
echo :
echo :                     Complete TTS2 Voice Agent Installation
echo :
echo ========================================================================================
echo :
echo :  [0] First-Time Setup (WSL2 + Ubuntu)                               Requires reboot
echo :       +-- Run this FIRST if you don't have WSL2/Ubuntu
echo :
echo :  [1] Full Install (Recommended)                                    ~15-30 min
echo :       +-- Check prerequisites, install deps, download all models
echo :
echo :  [2] Check Prerequisites Only
echo :       +-- Verify NVIDIA GPU, WSL2, Ubuntu, CUDA
echo :
echo :  [3] Install Dependencies Only                                     ~10-20 min
echo :       +-- Python venv, PyTorch, pip packages
echo :
echo :  [4] Download Models                                               ~5-15 min
echo :       +-- TTS models, STT models, embeddings (~10GB total)
echo :
echo :  [5] Back to Main Menu
echo :
echo ========================================================================================
echo.
set /p "install_choice=Select option [0-5]: "
if "%install_choice%"=="0" goto INSTALL_WSL
if "%install_choice%"=="1" goto INSTALL_FULL
if "%install_choice%"=="2" goto INSTALL_PREREQ
if "%install_choice%"=="3" goto INSTALL_DEPS
if "%install_choice%"=="4" goto INSTALL_MODELS_MENU
if "%install_choice%"=="5" goto MENU
goto INSTALL
:INSTALL_WSL
cls
echo 
echo ========================================================================================
echo                             First-Time Setup: WSL2 + Ubuntu
echo ========================================================================================
echo.
echo   This will install:
echo     - WSL2 (Windows Subsystem for Linux)
echo     - Ubuntu distribution
echo.
echo   IMPORTANT:
echo     - Requires administrator privileges
echo     - May require a system reboot
echo     - After reboot, Ubuntu will open for initial setup
echo     - You'll create a Linux username and password
echo.
echo ----------------------------------------------------------------------------------------
echo.
echo   Checking current status...
echo.
set WSL_INSTALLED=0
set UBUNTU_INSTALLED=0
wsl --status >nul 2>&1 && set WSL_INSTALLED=1
wsl -d Ubuntu -e echo OK >nul 2>&1 && set UBUNTU_INSTALLED=1
if "%WSL_INSTALLED%"=="1" (
    echo   [OK] WSL2 is already installed
) else (
    echo   [ ] WSL2 not installed - will install
)
if "%UBUNTU_INSTALLED%"=="1" (
    echo   [OK] Ubuntu is already installed
) else (
    echo   [ ] Ubuntu not installed - will install
)
echo.
if "%WSL_INSTALLED%"=="1" if "%UBUNTU_INSTALLED%"=="1" (
    echo   WSL2 and Ubuntu are already installed!
    echo   You can proceed with option [1] Full Install.
    echo.
    pause
    goto INSTALL
)
echo ----------------------------------------------------------------------------------------
echo.
set /p "wsl_confirm=Proceed with WSL2/Ubuntu installation? (Y/N): "
if /i not "%wsl_confirm%"=="Y" goto INSTALL
echo.
echo   Checking administrator privileges...
net session >nul 2>&1
if errorlevel 1 (
    echo   ERROR: This requires administrator privileges!
    echo.
    echo   Please:
    echo     1. Close this window
    echo     2. Right-click VoiceChat.bat
    echo     3. Select "Run as administrator"
    echo.
    pause
    goto INSTALL
)
echo   [OK] Running as administrator
echo.
if "%WSL_INSTALLED%"=="0" (
    echo   Installing WSL2...
    wsl --install --no-distribution
    echo   WSL2 installed.
)
echo.
if "%UBUNTU_INSTALLED%"=="0" (
    echo   Installing Ubuntu...
    wsl --install -d Ubuntu
)
echo.
echo ========================================================================================
echo   WSL2 + Ubuntu installation initiated!
echo ========================================================================================
echo.
echo   Next steps:
echo     1. If prompted, restart your computer
echo     2. After restart, Ubuntu will open automatically
echo     3. Create your Linux username and password
echo     4. Run VoiceChat.bat again and select [1] Full Install
echo.
pause
goto INSTALL
:INSTALL_PREREQ
cls
echo 
echo ========================================================================================
echo                              Checking System Prerequisites
echo ========================================================================================
echo.
set PREREQ_OK=1
echo   [1/4] Checking NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo         [X] NVIDIA GPU not detected or drivers not installed
    echo             Install drivers from: https://www.nvidia.com/drivers
    set PREREQ_OK=0
) else (
    echo         [OK] NVIDIA GPU detected
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>nul
)
echo.
echo   [2/4] Checking WSL2...
wsl --status >nul 2>&1
if errorlevel 1 (
    echo         [X] WSL2 not installed
    echo             Run in PowerShell (Admin): wsl --install
    set PREREQ_OK=0
) else (
    echo         [OK] WSL2 installed
)
echo.
echo   [3/4] Checking Ubuntu distribution...
wsl -d %WSL_DISTRO% -e echo "Ubuntu OK" >nul 2>&1
if errorlevel 1 (
    echo         [X] Ubuntu not found in WSL
    echo             Run: wsl --install -d Ubuntu
    set PREREQ_OK=0
) else (
    echo         [OK] Ubuntu distribution found
)
echo.
echo   [4/4] Checking CUDA access in WSL...
wsl -d %WSL_DISTRO% -e bash -c "nvidia-smi > /dev/null 2>&1 && echo CUDA_OK || echo CUDA_FAIL" > "%TEMP%\cuda_check.txt"
set /p CUDA_STATUS=<"%TEMP%\cuda_check.txt"
if "%CUDA_STATUS%"=="CUDA_FAIL" (
    echo         [X] CUDA not accessible from WSL
    echo             Ensure you have a recent NVIDIA driver (535+)
    set PREREQ_OK=0
) else (
    echo         [OK] CUDA accessible from WSL
)
echo.
echo ========================================================================================
if "%PREREQ_OK%"=="1" (
    echo   All prerequisites met! Ready to install.
) else (
    echo   Some prerequisites are missing. Please fix before installing.
)
echo ========================================================================================
echo.
pause
goto INSTALL
:INSTALL_FULL
cls
echo 
echo ========================================================================================
echo                                    Full Installation
echo ========================================================================================
echo.
echo   [Step 1/3] Verifying prerequisites...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   ERROR: NVIDIA GPU/driver not detected. Run option [2] for details.
    pause
    goto INSTALL
)
wsl -d %WSL_DISTRO% -e echo "OK" >nul 2>&1
if errorlevel 1 (
    echo   ERROR: WSL Ubuntu not found. Run option [2] for details.
    pause
    goto INSTALL
)
echo         [OK] Prerequisites verified
echo.
echo   [Step 2/3] Installing dependencies...
goto INSTALL_DEPS_RUN
:INSTALL_DEPS
cls
echo 
echo ========================================================================================
echo                                 Installing Dependencies
echo ========================================================================================
echo.
set FROM_FULL=0
goto INSTALL_DEPS_RUN
:INSTALL_DEPS_RUN
echo   [1/7] Installing Node.js 20 (via NVM)...
wsl -d %WSL_DISTRO% -e bash -c "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh 2>/dev/null | bash && export NVM_DIR=\"$HOME/.nvm\" && [ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\" && nvm install 20 && nvm alias default 20 && nvm use default && node -v"
echo.
echo   [2/7] Installing system dependencies (ffmpeg, build tools)...
wsl -d %WSL_DISTRO% -e bash -c "sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-dev build-essential ffmpeg git-lfs"
echo.
echo   [3/7] Creating Python virtual environment...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ ! -d .venv ] && python3 -m venv .venv && echo Created .venv || echo .venv already exists"
echo.
echo   [4/7] Installing PyTorch with CUDA 12.1...
echo         This may take several minutes...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && pip install --upgrade pip -q && pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo.
echo   [5/7] Installing Python packages from requirements.txt...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && pip install -q -r requirements.txt"
echo.
echo   [6/7] Installing FunASR + SenseVoice (STT backends)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && pip install -q funasr modelscope"
echo.
echo   [7/7] Installing Windows audio dependencies...
where python >nul 2>&1
if errorlevel 1 (
    echo         [SKIP] Windows Python not found - PTT will use fallback
    echo                 Install Python from python.org if you want PTT support
) else (
    pip install keyboard pyaudio numpy --quiet 2>nul
    echo         [OK] Windows audio deps installed
)
echo.
echo   Dependencies installed!
echo.
if "%install_choice%"=="1" goto INSTALL_MODELS_ALL
pause
goto INSTALL
:INSTALL_MODELS_MENU
cls
echo 
echo ========================================================================================
echo                                     Download Models
echo ========================================================================================
echo :
echo :  [1] Download All Models (Recommended)                            ~9 GB
echo :       +-- IndexTTS2, Kokoro, Supertonic, STT, Embeddings
echo :
echo :  [2] IndexTTS2 (High-quality TTS)                                  ~4.4 GB
echo :       +-- Voice cloning, emotion control, GPU required
echo :
echo :  [3] Supertonic (Fast CPU TTS)                                     ~500 MB
echo :       +-- 6 preset voices, 167x realtime, CPU only
echo :
echo :  [4] STT Models (SenseVoice + FunASR)                              ~1.5 GB
echo :       +-- Speech recognition with emotion detection
echo :
echo :  [5] Embeddings (Qwen3 for Memory)                                 ~1.2 GB
echo :       +-- Required for memory/knowledge graph
echo :
echo :  [6] NuExtract (Entity Extraction)                                 ~940 MB
echo :       +-- Extracts entities for knowledge graph
echo :
echo :  [7] Back
echo :
echo ========================================================================================
echo.
set /p "model_choice=Select option [1-7]: "
if "%model_choice%"=="1" goto INSTALL_MODELS_ALL
if "%model_choice%"=="2" goto INSTALL_MODEL_INDEXTTS
if "%model_choice%"=="3" goto INSTALL_MODEL_SUPERTONIC
if "%model_choice%"=="4" goto INSTALL_MODEL_STT
if "%model_choice%"=="5" goto INSTALL_MODEL_EMBEDDINGS
if "%model_choice%"=="6" goto INSTALL_MODEL_NUEXTRACT
if "%model_choice%"=="7" goto INSTALL
goto INSTALL_MODELS_MENU
:INSTALL_MODELS_ALL
cls
echo 
echo ========================================================================================
echo                                  Downloading All Models
echo ========================================================================================
echo.
echo   Total download size: ~9 GB. This may take 10-30 minutes.
echo.
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && mkdir -p models/{indextts2,supertonic,embeddings,nuextract,kokoro,hf_cache}"
echo   [1/5] Downloading IndexTTS2 models (~4.4 GB)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \"from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS2', local_dir='models/indextts2', local_dir_use_symlinks=False)\""
echo.
echo   [2/5] Downloading Supertonic models (~500 MB)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && git lfs install && if [ -d models/supertonic/.git ]; then echo Supertonic already exists; elif [ -d models/supertonic ]; then rmdir models/supertonic 2>/dev/null; git clone https://huggingface.co/neongeckocom/tts-vits-cv-en models/supertonic; else git clone https://huggingface.co/neongeckocom/tts-vits-cv-en models/supertonic; fi"
echo.
echo   [3/5] Downloading STT models (SenseVoice + FunASR)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \"from funasr import AutoModel; print('Downloading SenseVoice...'); AutoModel(model='FunAudioLLM/SenseVoiceSmall', device='cpu', hub='hf'); print('Done!')\""
echo.
echo   [4/5] Downloading embedding model (Qwen3)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='models/embeddings/qwen0.6b', local_dir_use_symlinks=False)\""
echo.
echo   [5/5] Downloading NuExtract model (~940 MB)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \"from huggingface_hub import hf_hub_download; import os; os.makedirs('models/nuextract', exist_ok=True); hf_hub_download('numind/NuExtract-2.0-2B-GGUF', filename='NuExtract-2.0-2B-Q4_K_M.gguf', local_dir='models/nuextract')\""
echo.
echo ========================================================================================
echo   All models downloaded!
echo ========================================================================================
echo.
if "%install_choice%"=="1" goto INSTALL_COMPLETE
pause
goto INSTALL
:INSTALL_MODEL_INDEXTTS
cls
echo   Downloading IndexTTS2 models (~4.4 GB)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && mkdir -p models/indextts2 && . .venv/bin/activate && python -c \"from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS2', local_dir='models/indextts2', local_dir_use_symlinks=False)\""
echo   Done!
pause
goto INSTALL_MODELS_MENU
:INSTALL_MODEL_SUPERTONIC
cls
echo   Downloading Supertonic models (~500 MB)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && mkdir -p models && git lfs install && if [ -d models/supertonic/.git ]; then echo Supertonic already exists; elif [ -d models/supertonic ]; then rmdir models/supertonic 2>/dev/null; git clone https://huggingface.co/neongeckocom/tts-vits-cv-en models/supertonic; else git clone https://huggingface.co/neongeckocom/tts-vits-cv-en models/supertonic; fi"
echo   Done!
pause
goto INSTALL_MODELS_MENU
:INSTALL_MODEL_STT
cls
echo   Downloading STT models (SenseVoice + FunASR)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \"from funasr import AutoModel; print('Downloading SenseVoice...'); AutoModel(model='FunAudioLLM/SenseVoiceSmall', device='cpu', hub='hf'); print('Done!')\""
echo   Done!
pause
goto INSTALL_MODELS_MENU
:INSTALL_MODEL_EMBEDDINGS
cls
echo   Downloading embedding model (Qwen3, ~1.2 GB)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && mkdir -p models/embeddings && . .venv/bin/activate && python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='models/embeddings/qwen0.6b', local_dir_use_symlinks=False)\""
echo   Done!
pause
goto INSTALL_MODELS_MENU
:INSTALL_MODEL_NUEXTRACT
cls
echo   Downloading NuExtract model (~940 MB)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \"from huggingface_hub import hf_hub_download; import os; os.makedirs('models/nuextract', exist_ok=True); hf_hub_download('numind/NuExtract-2.0-2B-GGUF', filename='NuExtract-2.0-2B-Q4_K_M.gguf', local_dir='models/nuextract')\""
echo   Done!
pause
goto INSTALL_MODELS_MENU
:INSTALL_COMPLETE
cls
echo 
echo ========================================================================================
echo :
echo :   ██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗         ██████╗ ██╗  ██╗██╗
echo :   ██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║        ██╔═══██╗██║ ██╔╝██║
echo :   ██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║        ██║   ██║█████╔╝ ██║
echo :   ██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║        ██║   ██║██╔═██╗ ╚═╝
echo :   ██║██║ ╚████║███████║   ██║   ██║  ██║███████╗███████╗   ╚██████╔╝██║  ██╗██╗
echo :   ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝    ╚═════╝ ╚═╝  ╚═╝╚═╝
echo :
echo ========================================================================================
echo.
echo   Installation Complete!
echo.
echo   Installed Components:
echo     - PyTorch with CUDA 12.1
echo     - TTS: IndexTTS2, Kokoro, Supertonic
echo     - STT: Faster-Whisper, SenseVoice, FunASR
echo     - Memory: sqlite-vec, Qwen3 embeddings
echo     - Node.js 20 (for MCP servers)
echo.
echo   Next Steps:
echo     1. Install LM Studio from lmstudio.ai (for LLM)
echo     2. Run option [1] Voice Agent to start
echo.
echo ========================================================================================
echo.
pause
goto MENU
:CALIBRATE
cls
color 08
chcp 65001 >nul
echo 
echo ========================================================================================
echo :
echo :   ███████╗███╗   ███╗ ██████╗ ████████╗██╗ ██████╗ ███╗   ██╗
echo :   ██╔════╝████╗ ████║██╔═══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
echo :   █████╗  ██╔████╔██║██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║
echo :   ██╔══╝  ██║╚██╔╝██║██║   ██║   ██║   ██║██║   ██║██║╚██╗██║
echo :   ███████╗██║ ╚═╝ ██║╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║
echo :   ╚══════╝╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
echo :
echo :    ██████╗ █████╗ ██╗     ██╗██████╗ ██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
echo :   ██╔════╝██╔══██╗██║     ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
echo :   ██║     ███████║██║     ██║██████╔╝██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║
echo :   ██║     ██╔══██║██║     ██║██╔══██╗██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
echo :   ╚██████╗██║  ██║███████╗██║██████╔╝██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
echo :    ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
echo :
echo                   Personalize Speech Emotion Recognition to YOUR Voice
echo :
echo ========================================================================================
echo.
echo   This tool calibrates emotion detection to YOUR voice.
echo   You will be asked to speak with different emotions.
echo   Each recording lasts 3 seconds.
echo.
echo   Make sure your microphone is connected and working.
echo.
echo ----------------------------------------------------------------------------------------
echo.
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ -d .venv ] && echo VENV_OK || echo VENV_MISSING" > "%TEMP%\venv_check.txt"
set /p VENV_STATUS=<"%TEMP%\venv_check.txt"
if "%VENV_STATUS%"=="VENV_MISSING" (
    echo ERROR: Virtual environment not found!
    echo Please run option [5] Installer first.
    echo.
    pause
    goto MENU
)
echo   Step 1: Recording samples on Windows...
echo.
python tools/calibrate_record_windows.py
echo.
echo   Step 2: Analyzing with SER model in WSL...
echo.
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && source .venv/bin/activate && python tools/calibrate_emotion_standalone.py --from-files"
echo.
pause
goto MENU