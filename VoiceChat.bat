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
echo [38;2;0;191;255m
echo [38;2;0;191;255m========================================================================================[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m╔════════════════════════════════════════════════════════════════════════════════╗[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m║  ████████╗████████╗ ███████╗ ██████╗    ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗  ║[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m║  ╚══██╔══╝╚══██╔══╝ ██╔════╝ ╚════██╗   ██║   ██║██╔═══██╗██║██╔════╝██╔════╝  ║[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m║     ██║      ██║    ███████╗  █████╔╝   ██║   ██║██║   ██║██║██║     █████╗    ║[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m║     ██║      ██║    ╚════██║ ██╔═══╝    ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝    ║[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m║     ██║      ██║    ███████║ ███████╗    ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗  ║[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m║     ╚═╝      ╚═╝    ╚══════╝ ╚══════╝     ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝  ║[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m╚════════════════════════════════════════════════════════════════════════════════╝[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m                           [38;2;0;191;255mMulti-Character Memory System[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m========================================================================================[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m  [38;2;0;191;255m[1] Voice Agent                                                       [90mPort: 7861[0m
echo [38;2;0;191;255m:[0m       [90m+-- PTT/VAD + Tools + Memory + Vision[0m
echo [38;2;0;191;255m:[0m       [90m+-- Docs: PDF,TXT,MD,DOCX,CSV,JSON,Code[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m  [38;2;0;191;255m[2] Mobile/Remote Access                                              [90mHTTPS Share[0m
echo [38;2;0;191;255m:[0m       [90m+-- Access from phone or any device[0m
echo [38;2;0;191;255m:[0m       [90m+-- Generates public HTTPS URL[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m  [38;2;0;191;255m[3] Character and Memory Manager                                      [90mPort: 7863[0m
echo [38;2;0;191;255m:[0m       [90m+-- Create/Edit Personalities[0m
echo [38;2;0;191;255m:[0m       [90m+-- Manage Knowledge Graph[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m  [38;2;0;136;170m[4] MCP Server Manager                                                [90mPort: 7864[0m
echo [38;2;0;191;255m:[0m       [90m+-- Configure Agent Tools[0m
echo [38;2;0;191;255m:[0m       [90m+-- Install MCP Servers[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m  [90m[5] Install Dependencies[0m
echo [38;2;0;191;255m:[0m       [90m+-- Python + Node.js (v20)[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m  [90m[6] Calibrate Emotion Detection[0m
echo [38;2;0;191;255m:[0m       [90m+-- Personalize SER to your voice[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m  [38;2;0;191;255m[7] Exit[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m========================================================================================[0m
echo.
set /p "choice=[38;2;0;191;255mSelect option [1-7]: [0m"
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
echo [38;2;0;191;255m
echo [38;2;0;191;255m========================================================================================[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██╗   ██╗ ██████╗ ██╗ ██████╗███████╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██║   ██║██╔═══██╗██║██╔════╝██╔════╝    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██║   ██║██║   ██║██║██║     █████╗      ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   [0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝      ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   [0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   [0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m  ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   [0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m                              [90mPTT + VAD + Memory + Tools + Vision[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m========================================================================================[0m
echo.
echo   [38;2;0;191;255mURL:[0m [38;2;0;191;255mhttp://localhost:7861[0m
echo.
echo [90m----------------------------------------------------------------------------------------[0m
echo.
if not exist "recordings" mkdir recordings
echo ready^|0^|Starting... > recordings\ptt_status.txt
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ -d .venv ] && echo VENV_OK || echo VENV_MISSING" > "%TEMP%\venv_check.txt"
set /p VENV_STATUS=<"%TEMP%\venv_check.txt"
if "%VENV_STATUS%"=="VENV_MISSING" (
    echo [38;2;0;191;255mERROR: Virtual environment not found![0m
    echo [38;2;0;191;255mPlease run option [4] Install Dependencies first.[0m
    echo.
    pause
    goto MENU
)
start "" /min wscript.exe scripts\ptt_hidden.vbs
start "" /min pythonw audio\vad_windows.py
echo [38;2;0;191;255m
wsl -d %WSL_DISTRO% -e bash -c "export NVM_DIR=$HOME/.nvm; [ -s $NVM_DIR/nvm.sh ] && . $NVM_DIR/nvm.sh; cd %WSL_WIN_PATH% && source .venv/bin/activate && python -W ignore tts2_agent.py"
taskkill /f /im pythonw.exe 2>nul
echo.
echo [38;2;0;191;255mProcess exited. Press any key to return to menu...[0m
pause >nul
goto MENU
:MOBILE
cls
echo [38;2;0;191;255m
echo [38;2;0;191;255m========================================================================================[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m███╗   ███╗ ██████╗ ██████╗ ██╗██╗     ███████╗[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m████╗ ████║██╔═══██╗██╔══██╗██║██║     ██╔════╝[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██╔████╔██║██║   ██║██████╔╝██║██║     █████╗  [0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██║╚██╔╝██║██║   ██║██╔══██╗██║██║     ██╔══╝  [0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██║ ╚═╝ ██║╚██████╔╝██████╔╝██║███████╗███████╗[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚═╝╚══════╝╚══════╝[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m              [90mPhone + Tablet + Any Device Access[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m========================================================================================[0m
echo.
echo   [38;2;0;191;255mThis mode generates a public HTTPS URL for your phone.[0m
echo   [38;2;0;191;255mThe URL will appear below when the server starts.[0m
echo.
echo   [90mFeatures:[0m
echo   [90m  - Touch-friendly HOLD TO TALK button[0m
echo   [90m  - Install as webapp (Add to Home Screen)[0m
echo   [90m  - Works on iOS and Android[0m
echo.
echo [90m----------------------------------------------------------------------------------------[0m
echo.
if not exist "recordings" mkdir recordings
echo ready^|0^|Starting... > recordings\ptt_status.txt
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ -d .venv ] && echo VENV_OK || echo VENV_MISSING" > "%TEMP%\venv_check.txt"
set /p VENV_STATUS=<"%TEMP%\venv_check.txt"
if "%VENV_STATUS%"=="VENV_MISSING" (
    echo [38;2;0;191;255mERROR: Virtual environment not found![0m
    echo [38;2;0;191;255mPlease run option [5] Install Dependencies first.[0m
    echo.
    pause
    goto MENU
)
start "" /min wscript.exe scripts\ptt_hidden.vbs
start "" /min pythonw audio\vad_windows.py
echo [38;2;0;191;255m
wsl -d %WSL_DISTRO% -e bash -c "export NVM_DIR=$HOME/.nvm; [ -s $NVM_DIR/nvm.sh ] && . $NVM_DIR/nvm.sh; cd %WSL_WIN_PATH% && source .venv/bin/activate && SHARE_MODE=1 python -W ignore tts2_agent.py"
taskkill /f /im pythonw.exe 2>nul
echo.
echo [38;2;0;191;255mProcess exited. Press any key to return to menu...[0m
pause >nul
goto MENU
:MANAGER
cls
color 06
chcp 65001 >nul
echo [38;2;0;191;255m
echo [38;2;0;191;255m========================================================================================[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m ██████╗██╗  ██╗ █████╗ ██████╗  █████╗  ██████╗████████╗███████╗██████╗ ███████╗[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██╔════╝██║  ██║██╔══██╗██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██║     ███████║███████║██████╔╝███████║██║        ██║   █████╗  ██████╔╝███████╗[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██║     ██╔══██║██╔══██║██╔══██╗██╔══██║██║        ██║   ██╔══╝  ██╔══██╗╚════██║[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m╚██████╗██║  ██║██║  ██║██║  ██║██║  ██║╚██████╗   ██║   ███████╗██║  ██║███████║[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m                        [90mPersonalities + Memory + Knowledge Graph[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m========================================================================================[0m
echo.
echo   [38;2;0;191;255mURL:[0m [38;2;0;191;255mhttp://localhost:7863[0m
echo.
echo [90m----------------------------------------------------------------------------------------[0m
echo.
echo [38;2;0;191;255m
wsl -d %WSL_DISTRO% -e bash -c "export NVM_DIR=$HOME/.nvm; [ -s $NVM_DIR/nvm.sh ] && . $NVM_DIR/nvm.sh; cd %WSL_WIN_PATH% && source .venv/bin/activate 2>/dev/null; python character_manager_ui.py"
goto MENU
:MCPMANAGER
cls
color 04
chcp 65001 >nul
echo [38;2;0;191;255m
echo [38;2;0;191;255m========================================================================================[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m███╗   ███╗ ██████╗██████╗     ███╗   ███╗ █████╗ ███╗   ██╗ █████╗  ██████╗ ███████╗██████╗ [0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m████╗ ████║██╔════╝██╔══██╗    ████╗ ████║██╔══██╗████╗  ██║██╔══██╗██╔════╝ ██╔════╝██╔══██╗[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██╔████╔██║██║     ██████╔╝    ██╔████╔██║███████║██╔██╗ ██║███████║██║  ███╗█████╗  ██████╔╝[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██║╚██╔╝██║██║     ██╔═══╝     ██║╚██╔╝██║██╔══██║██║╚██╗██║██╔══██║██║   ██║██╔══╝  ██╔══██╗[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██║ ╚═╝ ██║╚██████╗██║         ██║ ╚═╝ ██║██║  ██║██║ ╚████║██║  ██║╚██████╔╝███████╗██║  ██║[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m╚═╝     ╚═╝ ╚═════╝╚═╝         ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m                            [90mConfigure Agent Tools + Install Servers[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m========================================================================================[0m
echo.
echo   [38;2;0;191;255mURL:[0m [38;2;0;191;255mhttp://localhost:7864[0m
echo.
echo [90m----------------------------------------------------------------------------------------[0m
echo.
echo [38;2;0;191;255m
wsl -d %WSL_DISTRO% -e bash -c "export NVM_DIR=$HOME/.nvm; [ -s $NVM_DIR/nvm.sh ] && . $NVM_DIR/nvm.sh; cd %WSL_WIN_PATH% && source .venv/bin/activate 2>/dev/null; python mcp_manager_ui.py"
goto MENU
:INSTALL
cls
color 08
chcp 65001 >nul
echo [38;2;0;136;170m
echo [38;2;0;136;170m========================================================================================[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m   [38;2;0;136;170m██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗     ███████╗██████╗ [0m
echo [38;2;0;136;170m:[0m   [38;2;0;136;170m██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║     ██╔════╝██╔══██╗[0m
echo [38;2;0;136;170m:[0m   [38;2;0;136;170m██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║     █████╗  ██████╔╝[0m
echo [38;2;0;136;170m:[0m   [38;2;0;136;170m██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║     ██╔══╝  ██╔══██╗[0m
echo [38;2;0;136;170m:[0m   [38;2;0;136;170m██║██║ ╚████║███████║   ██║   ██║  ██║███████╗███████╗███████╗██║  ██║[0m
echo [38;2;0;136;170m:[0m   [38;2;0;136;170m╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m                     [90mComplete TTS2 Voice Agent Installation[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m========================================================================================[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m  [38;2;0;191;255m[0] First-Time Setup (WSL2 + Ubuntu)[0m                               [90mRequires reboot[0m
echo [38;2;0;136;170m:[0m       [90m+-- Run this FIRST if you don't have WSL2/Ubuntu[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m  [38;2;0;191;255m[1] Full Install (Recommended)[0m                                    [90m~15-30 min[0m
echo [38;2;0;136;170m:[0m       [90m+-- Check prerequisites, install deps, download all models[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m  [38;2;0;191;255m[2] Check Prerequisites Only[0m
echo [38;2;0;136;170m:[0m       [90m+-- Verify NVIDIA GPU, WSL2, Ubuntu, CUDA[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m  [38;2;0;191;255m[3] Install Dependencies Only[0m                                     [90m~10-20 min[0m
echo [38;2;0;136;170m:[0m       [90m+-- Python venv, PyTorch, pip packages[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m  [38;2;0;191;255m[4] Download Models[0m                                               [90m~5-15 min[0m
echo [38;2;0;136;170m:[0m       [90m+-- TTS models, STT models, embeddings (~10GB total)[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m  [90m[5] Back to Main Menu[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m========================================================================================[0m
echo.
set /p "install_choice=[38;2;0;191;255mSelect option [0-5]: [0m"
if "%install_choice%"=="0" goto INSTALL_WSL
if "%install_choice%"=="1" goto INSTALL_FULL
if "%install_choice%"=="2" goto INSTALL_PREREQ
if "%install_choice%"=="3" goto INSTALL_DEPS
if "%install_choice%"=="4" goto INSTALL_MODELS_MENU
if "%install_choice%"=="5" goto MENU
goto INSTALL
:INSTALL_WSL
cls
echo [38;2;0;191;255m
echo [38;2;0;191;255m========================================================================================[0m
echo [38;2;0;191;255m:[0m                     [38;2;0;191;255mFirst-Time Setup: WSL2 + Ubuntu[0m
echo [38;2;0;191;255m========================================================================================[0m
echo.
echo   [38;2;0;191;255mThis will install:[0m
echo   [90m  - WSL2 (Windows Subsystem for Linux)[0m
echo   [90m  - Ubuntu distribution[0m
echo.
echo   [38;2;0;191;255mIMPORTANT:[0m
echo   [90m  - Requires administrator privileges[0m
echo   [90m  - May require a system reboot[0m
echo   [90m  - After reboot, Ubuntu will open for initial setup[0m
echo   [90m  - You'll create a Linux username and password[0m
echo.
echo [90m----------------------------------------------------------------------------------------[0m
echo.
echo   [38;2;0;191;255mChecking current status...[0m
echo.
set WSL_INSTALLED=0
set UBUNTU_INSTALLED=0
wsl --status >nul 2>&1 && set WSL_INSTALLED=1
wsl -d Ubuntu -e echo OK >nul 2>&1 && set UBUNTU_INSTALLED=1
if "%WSL_INSTALLED%"=="1" (
    echo   [38;2;0;191;255m[OK] WSL2 is already installed[0m
) else (
    echo   [90m[ ] WSL2 not installed - will install[0m
)
if "%UBUNTU_INSTALLED%"=="1" (
    echo   [38;2;0;191;255m[OK] Ubuntu is already installed[0m
) else (
    echo   [90m[ ] Ubuntu not installed - will install[0m
)
echo.
if "%WSL_INSTALLED%"=="1" if "%UBUNTU_INSTALLED%"=="1" (
    echo   [38;2;0;191;255mWSL2 and Ubuntu are already installed![0m
    echo   [90mYou can proceed with option [1] Full Install.[0m
    echo.
    pause
    goto INSTALL
)
echo [90m----------------------------------------------------------------------------------------[0m
echo.
set /p "wsl_confirm=[38;2;0;191;255mProceed with WSL2/Ubuntu installation? (Y/N): [0m"
if /i not "%wsl_confirm%"=="Y" goto INSTALL
echo.
echo   [38;2;0;191;255mChecking administrator privileges...[0m
net session >nul 2>&1
if errorlevel 1 (
    echo   [38;2;0;191;255mERROR: This requires administrator privileges![0m
    echo.
    echo   [90mPlease:[0m
    echo   [90m  1. Close this window[0m
    echo   [90m  2. Right-click VoiceChat.bat[0m
    echo   [90m  3. Select "Run as administrator"[0m
    echo.
    pause
    goto INSTALL
)
echo   [38;2;0;191;255m[OK] Running as administrator[0m
echo.
if "%WSL_INSTALLED%"=="0" (
    echo   [38;2;0;191;255mInstalling WSL2...[0m
    wsl --install --no-distribution
    echo   [38;2;0;191;255mWSL2 installed.[0m
)
echo.
if "%UBUNTU_INSTALLED%"=="0" (
    echo   [38;2;0;191;255mInstalling Ubuntu...[0m
    wsl --install -d Ubuntu
)
echo.
echo [38;2;0;191;255m========================================================================================[0m
echo   [38;2;0;191;255mWSL2 + Ubuntu installation initiated![0m
echo [38;2;0;191;255m========================================================================================[0m
echo.
echo   [38;2;0;191;255mNext steps:[0m
echo   [90m  1. If prompted, restart your computer[0m
echo   [90m  2. After restart, Ubuntu will open automatically[0m
echo   [90m  3. Create your Linux username and password[0m
echo   [90m  4. Run VoiceChat.bat again and select [1] Full Install[0m
echo.
pause
goto INSTALL
:INSTALL_PREREQ
cls
echo [38;2;0;136;170m
echo [38;2;0;136;170m========================================================================================[0m
echo [38;2;0;136;170m:[0m                        [38;2;0;191;255mChecking System Prerequisites[0m
echo [38;2;0;136;170m========================================================================================[0m
echo.
set PREREQ_OK=1
echo   [38;2;0;191;255m[1/4][0m Checking NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo         [38;2;0;191;255m[X] NVIDIA GPU not detected or drivers not installed[0m
    echo         [90m    Install drivers from: https://www.nvidia.com/drivers[0m
    set PREREQ_OK=0
) else (
    echo         [38;2;0;191;255m[OK] NVIDIA GPU detected[0m
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>nul
)
echo.
echo   [38;2;0;191;255m[2/4][0m Checking WSL2...
wsl --status >nul 2>&1
if errorlevel 1 (
    echo         [38;2;0;191;255m[X] WSL2 not installed[0m
    echo         [90m    Run in PowerShell (Admin): wsl --install[0m
    set PREREQ_OK=0
) else (
    echo         [38;2;0;191;255m[OK] WSL2 installed[0m
)
echo.
echo   [38;2;0;191;255m[3/4][0m Checking Ubuntu distribution...
wsl -d %WSL_DISTRO% -e echo "Ubuntu OK" >nul 2>&1
if errorlevel 1 (
    echo         [38;2;0;191;255m[X] Ubuntu not found in WSL[0m
    echo         [90m    Run: wsl --install -d Ubuntu[0m
    set PREREQ_OK=0
) else (
    echo         [38;2;0;191;255m[OK] Ubuntu distribution found[0m
)
echo.
echo   [38;2;0;191;255m[4/4][0m Checking CUDA access in WSL...
wsl -d %WSL_DISTRO% -e bash -c "nvidia-smi > /dev/null 2>&1 && echo CUDA_OK || echo CUDA_FAIL" > "%TEMP%\cuda_check.txt"
set /p CUDA_STATUS=<"%TEMP%\cuda_check.txt"
if "%CUDA_STATUS%"=="CUDA_FAIL" (
    echo         [38;2;0;191;255m[X] CUDA not accessible from WSL[0m
    echo         [90m    Ensure you have a recent NVIDIA driver (535+)[0m
    set PREREQ_OK=0
) else (
    echo         [38;2;0;191;255m[OK] CUDA accessible from WSL[0m
)
echo.
echo [38;2;0;136;170m========================================================================================[0m
if "%PREREQ_OK%"=="1" (
    echo   [38;2;0;191;255mAll prerequisites met! Ready to install.[0m
) else (
    echo   [38;2;0;191;255mSome prerequisites are missing. Please fix before installing.[0m
)
echo [38;2;0;136;170m========================================================================================[0m
echo.
pause
goto INSTALL
:INSTALL_FULL
cls
echo [38;2;0;136;170m
echo [38;2;0;136;170m========================================================================================[0m
echo [38;2;0;136;170m:[0m                          [38;2;0;191;255mFull Installation[0m
echo [38;2;0;136;170m========================================================================================[0m
echo.
echo   [38;2;0;191;255m[Step 1/3][0m Verifying prerequisites...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   [38;2;0;191;255mERROR: NVIDIA GPU/driver not detected. Run option [2] for details.[0m
    pause
    goto INSTALL
)
wsl -d %WSL_DISTRO% -e echo "OK" >nul 2>&1
if errorlevel 1 (
    echo   [38;2;0;191;255mERROR: WSL Ubuntu not found. Run option [2] for details.[0m
    pause
    goto INSTALL
)
echo         [38;2;0;191;255m[OK] Prerequisites verified[0m
echo.
echo   [38;2;0;191;255m[Step 2/3][0m Installing dependencies...
goto INSTALL_DEPS_RUN
:INSTALL_DEPS
cls
echo [38;2;0;136;170m
echo [38;2;0;136;170m========================================================================================[0m
echo [38;2;0;136;170m:[0m                       [38;2;0;191;255mInstalling Dependencies[0m
echo [38;2;0;136;170m========================================================================================[0m
echo.
set FROM_FULL=0
goto INSTALL_DEPS_RUN
:INSTALL_DEPS_RUN
echo   [38;2;0;191;255m[1/7][0m Installing Node.js 20 (via NVM)...
wsl -d %WSL_DISTRO% -e bash -c "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh 2>/dev/null | bash && export NVM_DIR=\"$HOME/.nvm\" && [ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\" && nvm install 20 && nvm alias default 20 && nvm use default && node -v"
echo.
echo   [38;2;0;191;255m[2/7][0m Installing system dependencies (ffmpeg, build tools)...
wsl -d %WSL_DISTRO% -e bash -c "sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-dev build-essential ffmpeg git-lfs"
echo.
echo   [38;2;0;191;255m[3/7][0m Creating Python virtual environment...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ ! -d .venv ] && python3 -m venv .venv && echo Created .venv || echo .venv already exists"
echo.
echo   [38;2;0;191;255m[4/7][0m Installing PyTorch with CUDA 12.1...
echo         [90mThis may take several minutes...[0m
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && pip install --upgrade pip -q && pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo.
echo   [38;2;0;191;255m[5/7][0m Installing Python packages from requirements.txt...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && pip install -q -r requirements.txt"
echo.
echo   [38;2;0;191;255m[6/7][0m Installing FunASR + SenseVoice (STT backends)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && pip install -q funasr modelscope"
echo.
echo   [38;2;0;191;255m[7/7][0m Installing Windows audio dependencies...
pip install keyboard pyaudio numpy --quiet 2>nul
echo.
echo   [38;2;0;191;255mDependencies installed![0m
echo.
if "%install_choice%"=="1" goto INSTALL_MODELS_ALL
pause
goto INSTALL
:INSTALL_MODELS_MENU
cls
echo [38;2;0;136;170m
echo [38;2;0;136;170m========================================================================================[0m
echo [38;2;0;136;170m:[0m                          [38;2;0;191;255mDownload Models[0m
echo [38;2;0;136;170m========================================================================================[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m  [38;2;0;191;255m[1] Download All Models (Recommended)[0m                            [90m~9 GB[0m
echo [38;2;0;136;170m:[0m       [90m+-- IndexTTS2, Kokoro, Supertonic, STT, Embeddings[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m  [38;2;0;191;255m[2] IndexTTS2 (High-quality TTS)[0m                                  [90m~4.4 GB[0m
echo [38;2;0;136;170m:[0m       [90m+-- Voice cloning, emotion control, GPU required[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m  [38;2;0;191;255m[3] Supertonic (Fast CPU TTS)[0m                                     [90m~500 MB[0m
echo [38;2;0;136;170m:[0m       [90m+-- 6 preset voices, 167x realtime, CPU only[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m  [38;2;0;191;255m[4] STT Models (SenseVoice + FunASR)[0m                              [90m~1.5 GB[0m
echo [38;2;0;136;170m:[0m       [90m+-- Speech recognition with emotion detection[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m  [38;2;0;191;255m[5] Embeddings (Qwen3 for Memory)[0m                                 [90m~1.2 GB[0m
echo [38;2;0;136;170m:[0m       [90m+-- Required for memory/knowledge graph[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m  [38;2;0;191;255m[6] NuExtract (Entity Extraction)[0m                                 [90m~940 MB[0m
echo [38;2;0;136;170m:[0m       [90m+-- Extracts entities for knowledge graph[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m:[0m  [90m[7] Back[0m
echo [38;2;0;136;170m:[0m
echo [38;2;0;136;170m========================================================================================[0m
echo.
set /p "model_choice=[38;2;0;191;255mSelect option [1-7]: [0m"
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
echo [38;2;0;136;170m
echo [38;2;0;136;170m========================================================================================[0m
echo [38;2;0;136;170m:[0m                       [38;2;0;191;255mDownloading All Models[0m
echo [38;2;0;136;170m========================================================================================[0m
echo.
echo   [90mTotal download size: ~9 GB. This may take 10-30 minutes.[0m
echo.
echo   [38;2;0;191;255m[1/5][0m Downloading IndexTTS2 models (~4.4 GB)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \"from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS2', local_dir='models/indextts2', local_dir_use_symlinks=False)\""
echo.
echo   [38;2;0;191;255m[2/5][0m Downloading Supertonic models (~500 MB)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && git lfs install && [ ! -d models/supertonic/.git ] && git clone https://huggingface.co/neongeckocom/tts-vits-cv-en models/supertonic || echo Supertonic already exists"
echo.
echo   [38;2;0;191;255m[3/5][0m Downloading STT models (SenseVoice + FunASR)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \"from funasr import AutoModel; print('Downloading SenseVoice...'); AutoModel(model='FunAudioLLM/SenseVoiceSmall', device='cpu', hub='hf'); print('Done!')\""
echo.
echo   [38;2;0;191;255m[4/5][0m Downloading embedding model (Qwen3)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='models/embeddings/qwen0.6b', local_dir_use_symlinks=False)\""
echo.
echo   [38;2;0;191;255m[5/5][0m Downloading NuExtract model (~940 MB)...
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \"from huggingface_hub import hf_hub_download; import os; os.makedirs('models/nuextract', exist_ok=True); hf_hub_download('numind/NuExtract-2.0-2B-GGUF', filename='NuExtract-2.0-2B-Q4_K_M.gguf', local_dir='models/nuextract')\""
echo.
echo [38;2;0;191;255m========================================================================================[0m
echo   [38;2;0;191;255mAll models downloaded![0m
echo [38;2;0;191;255m========================================================================================[0m
echo.
if "%install_choice%"=="1" goto INSTALL_COMPLETE
pause
goto INSTALL
:INSTALL_MODEL_INDEXTTS
cls
echo   [38;2;0;191;255mDownloading IndexTTS2 models (~4.4 GB)...[0m
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \"from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS2', local_dir='models/indextts2', local_dir_use_symlinks=False)\""
echo   [38;2;0;191;255mDone![0m
pause
goto INSTALL_MODELS_MENU
:INSTALL_MODEL_SUPERTONIC
cls
echo   [38;2;0;191;255mDownloading Supertonic models (~500 MB)...[0m
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && git lfs install && [ ! -d models/supertonic/.git ] && git clone https://huggingface.co/neongeckocom/tts-vits-cv-en models/supertonic || echo Already exists"
echo   [38;2;0;191;255mDone![0m
pause
goto INSTALL_MODELS_MENU
:INSTALL_MODEL_STT
cls
echo   [38;2;0;191;255mDownloading STT models (SenseVoice + FunASR)...[0m
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \"from funasr import AutoModel; print('Downloading SenseVoice...'); AutoModel(model='FunAudioLLM/SenseVoiceSmall', device='cpu', hub='hf'); print('Done!')\""
echo   [38;2;0;191;255mDone![0m
pause
goto INSTALL_MODELS_MENU
:INSTALL_MODEL_EMBEDDINGS
cls
echo   [38;2;0;191;255mDownloading embedding model (Qwen3, ~1.2 GB)...[0m
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='models/embeddings/qwen0.6b', local_dir_use_symlinks=False)\""
echo   [38;2;0;191;255mDone![0m
pause
goto INSTALL_MODELS_MENU
:INSTALL_MODEL_NUEXTRACT
cls
echo   [38;2;0;191;255mDownloading NuExtract model (~940 MB)...[0m
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \"from huggingface_hub import hf_hub_download; import os; os.makedirs('models/nuextract', exist_ok=True); hf_hub_download('numind/NuExtract-2.0-2B-GGUF', filename='NuExtract-2.0-2B-Q4_K_M.gguf', local_dir='models/nuextract')\""
echo   [38;2;0;191;255mDone![0m
pause
goto INSTALL_MODELS_MENU
:INSTALL_COMPLETE
cls
echo [38;2;0;191;255m
echo [38;2;0;191;255m========================================================================================[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗         ██████╗ ██╗  ██╗██╗[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║        ██╔═══██╗██║ ██╔╝██║[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║        ██║   ██║█████╔╝ ██║[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║        ██║   ██║██╔═██╗ ╚═╝[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m██║██║ ╚████║███████║   ██║   ██║  ██║███████╗███████╗   ╚██████╔╝██║  ██╗██╗[0m
echo [38;2;0;191;255m:[0m   [38;2;0;191;255m╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝    ╚═════╝ ╚═╝  ╚═╝╚═╝[0m
echo [38;2;0;191;255m:[0m
echo [38;2;0;191;255m========================================================================================[0m
echo.
echo   [38;2;0;191;255mInstallation Complete![0m
echo.
echo   [90mInstalled Components:[0m
echo   [90m  - PyTorch with CUDA 12.1[0m
echo   [90m  - TTS: IndexTTS2, Kokoro, Supertonic[0m
echo   [90m  - STT: Faster-Whisper, SenseVoice, FunASR[0m
echo   [90m  - Memory: sqlite-vec, Qwen3 embeddings[0m
echo   [90m  - Node.js 20 (for MCP servers)[0m
echo.
echo   [38;2;0;191;255mNext Steps:[0m
echo   [90m  1. Install LM Studio from lmstudio.ai (for LLM)[0m
echo   [90m  2. Run option [1] Voice Agent to start[0m
echo.
echo [38;2;0;191;255m========================================================================================[0m
echo.
pause
goto MENU
:CALIBRATE
cls
color 08
chcp 65001 >nul
echo [90m
echo [90m========================================================================================[0m
echo [90m:[0m
echo [90m:[0m   [90m███████╗███╗   ███╗ ██████╗ ████████╗██╗ ██████╗ ███╗   ██╗[0m
echo [90m:[0m   [90m██╔════╝████╗ ████║██╔═══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║[0m
echo [90m:[0m   [90m█████╗  ██╔████╔██║██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║[0m
echo [90m:[0m   [90m██╔══╝  ██║╚██╔╝██║██║   ██║   ██║   ██║██║   ██║██║╚██╗██║[0m
echo [90m:[0m   [90m███████╗██║ ╚═╝ ██║╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║[0m
echo [90m:[0m   [90m╚══════╝╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝[0m
echo [90m:[0m
echo [90m:[0m   [90m ██████╗ █████╗ ██╗     ██╗██████╗ ██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗[0m
echo [90m:[0m   [90m██╔════╝██╔══██╗██║     ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║[0m
echo [90m:[0m   [90m██║     ███████║██║     ██║██████╔╝██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║[0m
echo [90m:[0m   [90m██║     ██╔══██║██║     ██║██╔══██╗██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║[0m
echo [90m:[0m   [90m╚██████╗██║  ██║███████╗██║██████╔╝██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║[0m
echo [90m:[0m   [90m ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝[0m
echo [90m:[0m
echo [90m:[0m                     [38;2;0;191;255mPersonalize Speech Emotion Recognition to YOUR Voice[0m
echo [90m:[0m
echo [90m========================================================================================[0m
echo.
echo   [38;2;0;191;255mThis tool calibrates emotion detection to YOUR voice.[0m
echo   [38;2;0;191;255mYou will be asked to speak with different emotions.[0m
echo   [38;2;0;191;255mEach recording lasts 3 seconds.[0m
echo.
echo   [90mMake sure your microphone is connected and working.[0m
echo.
echo [90m----------------------------------------------------------------------------------------[0m
echo.
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ -d .venv ] && echo VENV_OK || echo VENV_MISSING" > "%TEMP%\venv_check.txt"
set /p VENV_STATUS=<"%TEMP%\venv_check.txt"
if "%VENV_STATUS%"=="VENV_MISSING" (
    echo [38;2;0;191;255mERROR: Virtual environment not found![0m
    echo [38;2;0;191;255mPlease run option [4] Install Dependencies first.[0m
    echo.
    pause
    goto MENU
)
echo   [38;2;0;191;255mStep 1: Recording samples on Windows...[0m
echo.
python tools/calibrate_record_windows.py
echo.
echo   [38;2;0;191;255mStep 2: Analyzing with SER model in WSL...[0m
echo.
wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && source .venv/bin/activate && python tools/calibrate_emotion_standalone.py --from-files"
echo.
pause
goto MENU