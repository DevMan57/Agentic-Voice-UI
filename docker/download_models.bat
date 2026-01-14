@echo off
REM ============================================================================
REM Download IndexTTS2 Model Checkpoints (Windows)
REM ============================================================================
REM This script downloads the required model files from HuggingFace.
REM Run this BEFORE starting Docker containers.
REM ============================================================================

echo ============================================
echo IndexTTS2 Model Downloader
echo ============================================
echo.

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "CHECKPOINT_DIR=%SCRIPT_DIR%..\checkpoints"

REM Check for Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Install huggingface_hub if needed
pip show huggingface_hub >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing huggingface_hub...
    pip install huggingface_hub
)

REM Create checkpoint directory
if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"

echo Downloading IndexTTS2 checkpoints to: %CHECKPOINT_DIR%
echo This may take a while (~10GB)...
echo.

REM Download from HuggingFace
cd /d "%CHECKPOINT_DIR%"
huggingface-cli download IndexTeam/Index-TTS --local-dir . --local-dir-use-symlinks False

echo.
echo ============================================
echo Download Complete!
echo ============================================
echo.
echo Checkpoints saved to: %CHECKPOINT_DIR%
echo.
echo You can now start the Docker container:
echo   cd %SCRIPT_DIR%..
echo   docker-compose up --build
echo.
pause
