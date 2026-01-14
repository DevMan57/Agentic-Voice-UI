import os

# --- Configuration ---
TARGET_FILE = "VoiceChat.bat"
DELETE_FILE = "VoiceChat_new.bat"

# --- ANSI Colors (Synthwave Purple Theme - TrueColor) ---
ESC = "\x1b"
C_RESET   = f"{ESC}[0m"
C_GREY    = f"{ESC}[90m"
C_WHITE   = f"{ESC}[38;2;230;217;255m"  # Pale Lavender #e6d9ff
C_CYAN    = f"{ESC}[38;2;0;255;255m"    # Neon Cyan #00ffff (synthwave accent)
C_RED     = f"{ESC}[91m"
C_HOTPINK = f"{ESC}[38;2;255;20;147m"   # Hot Pink #ff1493 (synthwave accent)
C_VIOLET  = f"{ESC}[38;2;191;0;255m"    # Electric Purple #bf00ff (primary)
C_PURPLE  = f"{ESC}[38;2;191;0;255m"    # Electric Purple #bf00ff (primary)
C_TEAL    = f"{ESC}[38;2;0;200;255m"    # Bright Teal #00c8ff (MCP)

# --- ASCII Art (TTS2 VOICE - big 8-line chunky style for home screen) ---
HEADER_ART = [
    r"╔════════════════════════════════════════════════════════════════════════════════╗",
    r"║  ████████╗████████╗ ███████╗ ██████╗    ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗  ║",
    r"║  ╚══██╔══╝╚══██╔══╝ ██╔════╝ ╚════██╗   ██║   ██║██╔═══██╗██║██╔════╝██╔════╝  ║",
    r"║     ██║      ██║    ███████╗  █████╔╝   ██║   ██║██║   ██║██║██║     █████╗    ║",
    r"║     ██║      ██║    ╚════██║ ██╔═══╝    ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝    ║",
    r"║     ██║      ██║    ███████║ ███████╗    ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗  ║",
    r"║     ╚═╝      ╚═╝    ╚══════╝ ╚══════╝     ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝  ║",
    r"╚════════════════════════════════════════════════════════════════════════════════╝"
]

# Terminal width for consistent bars (93 + prefix = ~96 for MCP MANAGER)
TERM_WIDTH = 93

def make_bar(char="=", length=TERM_WIDTH, color=C_CYAN, end_color=C_RESET):
    solid = char * length
    return f"{color}{solid}{end_color}"

def make_item(key, label, status="", color_main=C_WHITE):
    label_part = f"[{key}] {label}"
    if status:
        padding = 55 - len(label_part)
        return f"{C_PURPLE}:{C_RESET}  {color_main}{label_part}{' ' * padding}{C_GREY}{status}{C_RESET}"
    else:
        return f"{C_PURPLE}:{C_RESET}  {color_main}{label_part}{C_RESET}"

def make_sub(label, info=""):
    return f"{C_PURPLE}:{C_RESET}       {C_GREY}+-- {label}{C_RESET}"

def get_batch_content():
    lines = []

    # --- HEADER ---
    lines.append("@echo off")
    lines.append("chcp 65001 >nul")
    lines.append("setlocal EnableDelayedExpansion")
    lines.append("title IndexTTS2 Voice Agent")

    # --- RESIZE WINDOW & BUFFER ---
    lines.append('mode con: cols=96 lines=36')
    lines.append('powershell -command "&{$W=(get-host).ui.rawui;$B=$W.buffersize;$B.height=1000;$W.buffersize=$B;}"')

    # --- PATH SETUP ---
    lines.append('set "WSL_DISTRO=Ubuntu"')
    lines.append('REM WSL_PROJECT is no longer needed - using local .venv')
    lines.append('set "WIN_SCRIPT_DIR=%~dp0"')
    lines.append('set "WIN_SCRIPT_DIR=%WIN_SCRIPT_DIR:~0,-1%"')
    lines.append('set "WSL_WIN_PATH=%WIN_SCRIPT_DIR:\\=/%"')
    lines.append('set "WSL_WIN_PATH=%WSL_WIN_PATH:C:=/mnt/c%"')
    lines.append('set "WSL_WIN_PATH=%WSL_WIN_PATH:D:=/mnt/d%"')
    lines.append('set "WSL_WIN_PATH=%WSL_WIN_PATH:E:=/mnt/e%"')
    lines.append('cd /d "%WIN_SCRIPT_DIR%"')

    # --- NVM LOADER STRING ---
    nvm_load = 'export NVM_DIR=$HOME/.nvm; [ -s $NVM_DIR/nvm.sh ] && . $NVM_DIR/nvm.sh;'

    # --- MENU LOOP ---
    lines.append(":MENU")
    lines.append("cls")
    lines.append(f"echo {C_PURPLE}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_PURPLE)}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}")
    for line in HEADER_ART:
        safe_line = line.replace("|", "^|").replace("%", "%%")
        lines.append(f"echo {C_PURPLE}:{C_RESET} {C_PURPLE}{safe_line}{C_RESET}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}                       {C_WHITE}Multi-Character Memory System{C_RESET}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_PURPLE)}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}")

    # Menu Items
    lines.append(f"echo {make_item('1', 'Voice Agent', 'Port: 7861', C_PURPLE)}")
    lines.append(f"echo {make_sub('PTT/VAD + Tools + Memory + Vision')}")
    lines.append(f"echo {make_sub('Docs: PDF,TXT,MD,DOCX,CSV,JSON,Code')}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}")

    lines.append(f"echo {make_item('2', 'Character and Memory Manager', 'Port: 7863', C_HOTPINK)}")
    lines.append(f"echo {make_sub('Create/Edit Personalities')}")
    lines.append(f"echo {make_sub('Manage Knowledge Graph')}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}")

    lines.append(f"echo {make_item('3', 'MCP Server Manager', 'Port: 7864', C_TEAL)}")
    lines.append(f"echo {make_sub('Configure Agent Tools')}")
    lines.append(f"echo {make_sub('Install MCP Servers')}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}")

    lines.append(f"echo {make_item('4', 'Install Dependencies', '', C_VIOLET)}")
    lines.append(f"echo {make_sub('Python + Node.js (v20)')}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}")

    lines.append(f"echo {make_item('5', 'Calibrate Emotion Detection', '', C_CYAN)}")
    lines.append(f"echo {make_sub('Personalize SER to your voice')}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}")

    lines.append(f"echo {make_item('6', 'Exit', '', C_RED)}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}")

    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_PURPLE)}")
    lines.append("echo.")
    lines.append(f'set /p "choice={C_WHITE}Select option [1-6]: {C_RESET}"')

    lines.append('if "%choice%"=="1" goto VOICECHAT')
    lines.append('if "%choice%"=="2" goto MANAGER')
    lines.append('if "%choice%"=="3" goto MCPMANAGER')
    lines.append('if "%choice%"=="4" goto INSTALL')
    lines.append('if "%choice%"=="5" goto CALIBRATE')
    lines.append('if "%choice%"=="6" exit /b 0')
    lines.append("goto MENU")

    # --- VOICE CHAT ---
    lines.append(":VOICECHAT")
    lines.append("cls")
    lines.append(f"echo {C_PURPLE}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_PURPLE)}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}   {C_PURPLE}██╗   ██╗ ██████╗ ██╗ ██████╗███████╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗{C_RESET}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}   {C_PURPLE}██║   ██║██╔═══██╗██║██╔════╝██╔════╝    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝{C_RESET}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}   {C_PURPLE}██║   ██║██║   ██║██║██║     █████╗      ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   {C_RESET}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}   {C_PURPLE}╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝      ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   {C_RESET}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}   {C_PURPLE} ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   {C_RESET}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}   {C_PURPLE}  ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   {C_RESET}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}                              {C_GREY}PTT + VAD + Memory + Tools + Vision{C_RESET}")
    lines.append(f"echo {C_PURPLE}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_PURPLE)}")
    lines.append(f"echo.")
    lines.append(f"echo   {C_WHITE}URL:{C_RESET} {C_CYAN}http://localhost:7861{C_RESET}")
    lines.append(f"echo.")
    lines.append(f"echo {make_bar('-', TERM_WIDTH, C_GREY)}")
    lines.append(f"echo.")
    lines.append('if not exist "recordings" mkdir recordings')
    lines.append('echo ready^|0^|Starting... > recordings\\ptt_status.txt')
    # Check if .venv exists before trying to start
    lines.append(f'wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ -d .venv ] && echo VENV_OK || echo VENV_MISSING" > "%TEMP%\\venv_check.txt"')
    lines.append('set /p VENV_STATUS=<"%TEMP%\\venv_check.txt"')
    lines.append('if "%VENV_STATUS%"=="VENV_MISSING" (')
    lines.append(f'    echo {C_RED}ERROR: Virtual environment not found!{C_RESET}')
    lines.append(f'    echo {C_WHITE}Please run option [4] Install Dependencies first.{C_RESET}')
    lines.append('    echo.')
    lines.append('    pause')
    lines.append('    goto MENU')
    lines.append(')')
    lines.append('start "" /min wscript.exe scripts\\ptt_hidden.vbs')
    lines.append('start "" /min pythonw audio\\vad_windows.py')
    lines.append(f'echo {C_PURPLE}')
    lines.append(f'wsl -d %WSL_DISTRO% -e bash -c "{nvm_load} cd %WSL_WIN_PATH% && source .venv/bin/activate && python -W ignore tts2_agent.py"')
    lines.append("taskkill /f /im pythonw.exe 2>nul")
    lines.append("echo.")
    lines.append(f"echo {C_RED}Process exited. Press any key to return to menu...{C_RESET}")
    lines.append("pause >nul")
    lines.append("goto MENU")

    # --- MANAGER ---
    lines.append(":MANAGER")
    lines.append("cls")
    lines.append("color 06")
    lines.append("chcp 65001 >nul")
    lines.append(f"echo {C_HOTPINK}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_HOTPINK)}")
    lines.append(f"echo {C_HOTPINK}:{C_RESET}")
    lines.append(f"echo {C_HOTPINK}:{C_RESET}   {C_HOTPINK} ██████╗██╗  ██╗ █████╗ ██████╗  █████╗  ██████╗████████╗███████╗██████╗ ███████╗{C_RESET}")
    lines.append(f"echo {C_HOTPINK}:{C_RESET}   {C_HOTPINK}██╔════╝██║  ██║██╔══██╗██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝{C_RESET}")
    lines.append(f"echo {C_HOTPINK}:{C_RESET}   {C_HOTPINK}██║     ███████║███████║██████╔╝███████║██║        ██║   █████╗  ██████╔╝███████╗{C_RESET}")
    lines.append(f"echo {C_HOTPINK}:{C_RESET}   {C_HOTPINK}██║     ██╔══██║██╔══██║██╔══██╗██╔══██║██║        ██║   ██╔══╝  ██╔══██╗╚════██║{C_RESET}")
    lines.append(f"echo {C_HOTPINK}:{C_RESET}   {C_HOTPINK}╚██████╗██║  ██║██║  ██║██║  ██║██║  ██║╚██████╗   ██║   ███████╗██║  ██║███████║{C_RESET}")
    lines.append(f"echo {C_HOTPINK}:{C_RESET}   {C_HOTPINK} ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝{C_RESET}")
    lines.append(f"echo {C_HOTPINK}:{C_RESET}")
    lines.append(f"echo {C_HOTPINK}:{C_RESET}                        {C_GREY}Personalities + Memory + Knowledge Graph{C_RESET}")
    lines.append(f"echo {C_HOTPINK}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_HOTPINK)}")
    lines.append(f"echo.")
    lines.append(f"echo   {C_WHITE}URL:{C_RESET} {C_CYAN}http://localhost:7863{C_RESET}")
    lines.append(f"echo.")
    lines.append(f"echo {make_bar('-', TERM_WIDTH, C_GREY)}")
    lines.append(f"echo.")
    lines.append(f'echo {C_HOTPINK}')
    lines.append(f'wsl -d %WSL_DISTRO% -e bash -c "{nvm_load} cd %WSL_WIN_PATH% && source .venv/bin/activate 2>/dev/null; python character_manager_ui.py"')
    lines.append("goto MENU")

    # --- MCP MANAGER ---
    lines.append(":MCPMANAGER")
    lines.append("cls")
    lines.append("color 0B")
    lines.append("chcp 65001 >nul")
    lines.append(f"echo {C_TEAL}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_TEAL)}")
    lines.append(f"echo {C_TEAL}:{C_RESET}")
    lines.append(f"echo {C_TEAL}:{C_RESET}   {C_TEAL}███╗   ███╗ ██████╗██████╗     ███╗   ███╗ █████╗ ███╗   ██╗ █████╗  ██████╗ ███████╗██████╗ {C_RESET}")
    lines.append(f"echo {C_TEAL}:{C_RESET}   {C_TEAL}████╗ ████║██╔════╝██╔══██╗    ████╗ ████║██╔══██╗████╗  ██║██╔══██╗██╔════╝ ██╔════╝██╔══██╗{C_RESET}")
    lines.append(f"echo {C_TEAL}:{C_RESET}   {C_TEAL}██╔████╔██║██║     ██████╔╝    ██╔████╔██║███████║██╔██╗ ██║███████║██║  ███╗█████╗  ██████╔╝{C_RESET}")
    lines.append(f"echo {C_TEAL}:{C_RESET}   {C_TEAL}██║╚██╔╝██║██║     ██╔═══╝     ██║╚██╔╝██║██╔══██║██║╚██╗██║██╔══██║██║   ██║██╔══╝  ██╔══██╗{C_RESET}")
    lines.append(f"echo {C_TEAL}:{C_RESET}   {C_TEAL}██║ ╚═╝ ██║╚██████╗██║         ██║ ╚═╝ ██║██║  ██║██║ ╚████║██║  ██║╚██████╔╝███████╗██║  ██║{C_RESET}")
    lines.append(f"echo {C_TEAL}:{C_RESET}   {C_TEAL}╚═╝     ╚═╝ ╚═════╝╚═╝         ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝{C_RESET}")
    lines.append(f"echo {C_TEAL}:{C_RESET}")
    lines.append(f"echo {C_TEAL}:{C_RESET}                            {C_GREY}Configure Agent Tools + Install Servers{C_RESET}")
    lines.append(f"echo {C_TEAL}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_TEAL)}")
    lines.append(f"echo.")
    lines.append(f"echo   {C_WHITE}URL:{C_RESET} {C_CYAN}http://localhost:7864{C_RESET}")
    lines.append(f"echo.")
    lines.append(f"echo {make_bar('-', TERM_WIDTH, C_GREY)}")
    lines.append(f"echo.")
    lines.append(f'echo {C_TEAL}')
    lines.append(f'wsl -d %WSL_DISTRO% -e bash -c "{nvm_load} cd %WSL_WIN_PATH% && source .venv/bin/activate 2>/dev/null; python mcp_manager_ui.py"')
    lines.append("goto MENU")

    # --- INSTALL ---
    lines.append(":INSTALL")
    lines.append("cls")
    lines.append("color 0D")
    lines.append("chcp 65001 >nul")
    lines.append(f"echo {C_VIOLET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_VIOLET)}")
    lines.append(f"echo {C_VIOLET}:{C_RESET}")
    lines.append(f"echo {C_VIOLET}:{C_RESET}   {C_VIOLET}██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗         ██████╗ ███████╗██████╗ ███████╗{C_RESET}")
    lines.append(f"echo {C_VIOLET}:{C_RESET}   {C_VIOLET}██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║         ██╔══██╗██╔════╝██╔══██╗██╔════╝{C_RESET}")
    lines.append(f"echo {C_VIOLET}:{C_RESET}   {C_VIOLET}██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║         ██║  ██║█████╗  ██████╔╝███████╗{C_RESET}")
    lines.append(f"echo {C_VIOLET}:{C_RESET}   {C_VIOLET}██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║         ██║  ██║██╔══╝  ██╔═══╝ ╚════██║{C_RESET}")
    lines.append(f"echo {C_VIOLET}:{C_RESET}   {C_VIOLET}██║██║ ╚████║███████║   ██║   ██║  ██║███████╗███████╗    ██████╔╝███████╗██║     ███████║{C_RESET}")
    lines.append(f"echo {C_VIOLET}:{C_RESET}   {C_VIOLET}╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝    ╚═════╝ ╚══════╝╚═╝     ╚══════╝{C_RESET}")
    lines.append(f"echo {C_VIOLET}:{C_RESET}")
    lines.append(f"echo {C_VIOLET}:{C_RESET}                              {C_GREY}Python + Node.js Dependencies{C_RESET}")
    lines.append(f"echo {C_VIOLET}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_VIOLET)}")
    lines.append(f"echo.")
    lines.append(f"echo {make_bar('-', TERM_WIDTH, C_GREY)}")
    lines.append("echo.")

    # 1. Install Node.js
    lines.append(f"echo   [*] Checking/Installing Node.js 20 (via NVM)...{C_VIOLET}")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && export NVM_DIR=\\"$HOME/.nvm\\" && [ -s \\"$NVM_DIR/nvm.sh\\" ] && . \\"$NVM_DIR/nvm.sh\\" && nvm install 20 && nvm alias default 20 && nvm use default && node -v"')
    lines.append("echo.")

    # 2. Python Deps - ensure python3-venv is installed, create venv, then install packages
    lines.append(f"echo   [*] Checking Python Environment...{C_VIOLET}")
    # Step 1: Ensure python3-venv package is installed (required for venv creation)
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "dpkg -s python3-venv >/dev/null 2>&1 || (echo Installing python3-venv... && sudo apt-get update && sudo apt-get install -y python3-venv)"')
    # Step 2: Create venv if it doesn't exist
    lines.append(f"echo   [*] Creating virtual environment...{C_VIOLET}")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ ! -d .venv ] && python3 -m venv .venv && echo Created .venv || echo .venv already exists"')
    # Step 3: Activate and install packages
    lines.append(f"echo   [*] Installing Python packages...{C_VIOLET}")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"')
    lines.append("echo.")

    # 3. Windows Audio
    lines.append(f"echo   [*] Checking Windows audio deps...{C_VIOLET}")
    lines.append("pip install keyboard pyaudio numpy --quiet 2>nul")

    lines.append("echo.")
    lines.append(f"echo {C_VIOLET}Done!{C_RESET}")
    lines.append("pause")
    lines.append("goto MENU")

    # --- CALIBRATE EMOTION ---
    lines.append(":CALIBRATE")
    lines.append("cls")
    lines.append("color 0B")
    lines.append("chcp 65001 >nul")
    lines.append(f"echo {C_CYAN}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_CYAN)}")
    lines.append(f"echo {C_CYAN}:{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}   {C_CYAN}███████╗███╗   ███╗ ██████╗ ████████╗██╗ ██████╗ ███╗   ██╗{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}   {C_CYAN}██╔════╝████╗ ████║██╔═══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}   {C_CYAN}█████╗  ██╔████╔██║██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}   {C_CYAN}██╔══╝  ██║╚██╔╝██║██║   ██║   ██║   ██║██║   ██║██║╚██╗██║{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}   {C_CYAN}███████╗██║ ╚═╝ ██║╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}   {C_CYAN}╚══════╝╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}   {C_CYAN} ██████╗ █████╗ ██╗     ██╗██████╗ ██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}   {C_CYAN}██╔════╝██╔══██╗██║     ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}   {C_CYAN}██║     ███████║██║     ██║██████╔╝██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}   {C_CYAN}██║     ██╔══██║██║     ██║██╔══██╗██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}   {C_CYAN}╚██████╗██║  ██║███████╗██║██████╔╝██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}   {C_CYAN} ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}                     {C_GREY}Personalize Speech Emotion Recognition to YOUR Voice{C_RESET}")
    lines.append(f"echo {C_CYAN}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_CYAN)}")
    lines.append(f"echo.")
    lines.append(f"echo   {C_WHITE}This tool calibrates emotion detection to YOUR voice.{C_RESET}")
    lines.append(f"echo   {C_WHITE}You will be asked to speak with different emotions.{C_RESET}")
    lines.append(f"echo   {C_WHITE}Each recording lasts 3 seconds.{C_RESET}")
    lines.append(f"echo.")
    lines.append(f"echo   {C_GREY}Make sure your microphone is connected and working.{C_RESET}")
    lines.append(f"echo.")
    lines.append(f"echo {make_bar('-', TERM_WIDTH, C_GREY)}")
    lines.append("echo.")
    # Check if .venv exists
    lines.append(f'wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ -d .venv ] && echo VENV_OK || echo VENV_MISSING" > "%TEMP%\\venv_check.txt"')
    lines.append('set /p VENV_STATUS=<"%TEMP%\\venv_check.txt"')
    lines.append('if "%VENV_STATUS%"=="VENV_MISSING" (')
    lines.append(f'    echo {C_RED}ERROR: Virtual environment not found!{C_RESET}')
    lines.append(f'    echo {C_WHITE}Please run option [4] Install Dependencies first.{C_RESET}')
    lines.append('    echo.')
    lines.append('    pause')
    lines.append('    goto MENU')
    lines.append(')')
    # Step 1: Record on Windows (where microphone is accessible)
    lines.append(f"echo   {C_WHITE}Step 1: Recording samples on Windows...{C_RESET}")
    lines.append(f"echo.")
    lines.append(f'python tools/calibrate_record_windows.py')
    lines.append(f"echo.")
    # Step 2: Analyze in WSL (where the SER model runs)
    lines.append(f"echo   {C_WHITE}Step 2: Analyzing with SER model in WSL...{C_RESET}")
    lines.append(f"echo.")
    lines.append(f'wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && source .venv/bin/activate && python tools/calibrate_emotion_standalone.py --from-files"')
    lines.append("echo.")
    lines.append("pause")
    lines.append("goto MENU")

    return lines

if __name__ == "__main__":
    try:
        with open(TARGET_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(get_batch_content()))
        print(f"[OK] Successfully created {TARGET_FILE}")
        if os.path.exists(DELETE_FILE):
            try:
                os.remove(DELETE_FILE)
            except OSError:
                pass
    except Exception as e:
        print(f"[ERROR] {e}")
