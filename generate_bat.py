import os

# --- Configuration ---
TARGET_FILE = "VoiceChat.bat"
DELETE_FILE = "VoiceChat_new.bat"

# --- ANSI Colors (Lapis Lazuli Palette - TrueColor) ---
ESC = "\x1b"
C_RESET   = f"{ESC}[0m"
C_GREY    = f"{ESC}[90m"

# The Palette - Lapis Lazuli Glowing Blue (ALL lapis lazuli now)
C_ORANGE  = f"{ESC}[38;2;0;191;255m"   # Lapis Lazuli #00BFFF (Borders)
C_AMBER   = f"{ESC}[38;2;0;191;255m"   # Lapis Lazuli #00BFFF (Main Text)
C_GOLD    = f"{ESC}[38;2;0;191;255m"   # Lapis Lazuli #00BFFF (Highlights)
C_RED     = f"{ESC}[38;2;0;191;255m"   # Lapis Lazuli #00BFFF (Alerts - now blue)
C_DARK    = f"{ESC}[38;2;0;136;170m"   # Dim Lapis #0088AA (Subtitles)

# --- ASCII Art (TTS2 VOICE) ---
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

TERM_WIDTH = 88

def make_bar(char="=", length=TERM_WIDTH, color=C_ORANGE, end_color=C_RESET):
    solid = char * length
    return f"{color}{solid}{end_color}"

def make_item(key, label, status="", color_main=C_AMBER):
    label_part = f"[{key}] {label}"
    if status:
        padding = 55 - len(label_part)
        return f"{C_ORANGE}:{C_RESET}  {color_main}{label_part}{' ' * padding}{C_GREY}{status}{C_RESET}"
    else:
        return f"{C_ORANGE}:{C_RESET}  {color_main}{label_part}{C_RESET}"

def make_sub(label, info=""):
    return f"{C_ORANGE}:{C_RESET}       {C_GREY}+-- {label}{C_RESET}"

def get_batch_content():
    lines = []

    # --- HEADER ---
    lines.append("@echo off")
    lines.append("chcp 65001 >nul")
    lines.append("setlocal EnableDelayedExpansion")
    lines.append("title IndexTTS2 Voice Agent")

    # --- RESIZE WINDOW ---
    lines.append('mode con: cols=92 lines=42')
    lines.append('powershell -command "&{$W=(get-host).ui.rawui;$B=$W.buffersize;$B.height=1000;$W.buffersize=$B;}"')

    # --- PATH SETUP ---
    lines.append('set "WSL_DISTRO=Ubuntu"')
    lines.append('set "WIN_SCRIPT_DIR=%~dp0"')
    lines.append('set "WIN_SCRIPT_DIR=%WIN_SCRIPT_DIR:~0,-1%"')
    lines.append('set "WSL_WIN_PATH=%WIN_SCRIPT_DIR:\\=/%"')
    lines.append('set "WSL_WIN_PATH=%WSL_WIN_PATH:C:=/mnt/c%"')
    lines.append('set "WSL_WIN_PATH=%WSL_WIN_PATH:D:=/mnt/d%"')
    lines.append('set "WSL_WIN_PATH=%WSL_WIN_PATH:E:=/mnt/e%"')
    lines.append('cd /d "%WIN_SCRIPT_DIR%"')
    nvm_load = 'export NVM_DIR=$HOME/.nvm; [ -s $NVM_DIR/nvm.sh ] && . $NVM_DIR/nvm.sh;'

    # --- MENU LOOP ---
    lines.append(":MENU")
    lines.append("cls")
    lines.append(f"echo {C_ORANGE}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_ORANGE)}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")
    for line in HEADER_ART:
        safe_line = line.replace("|", "^|").replace("%", "%%")
        lines.append(f"echo {C_ORANGE}:{C_RESET} {C_ORANGE}{safe_line}{C_RESET}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}                       {C_AMBER}Multi-Character Memory System{C_RESET}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_ORANGE)}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")

    # Menu Items
    lines.append(f"echo {make_item('1', 'Voice Agent', 'Port: 7861', C_ORANGE)}")
    lines.append(f"echo {make_sub('PTT/VAD + Tools + Memory + Vision')}")
    lines.append(f"echo {make_sub('Docs: PDF,TXT,MD,DOCX,CSV,JSON,Code')}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")

    lines.append(f"echo {make_item('2', 'Mobile/Remote Access', 'HTTPS Share', C_GOLD)}")
    lines.append(f"echo {make_sub('Access from phone or any device')}")
    lines.append(f"echo {make_sub('Generates public HTTPS URL')}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")

    lines.append(f"echo {make_item('3', 'Character and Memory Manager', 'Port: 7863', C_AMBER)}")
    lines.append(f"echo {make_sub('Create/Edit Personalities')}")
    lines.append(f"echo {make_sub('Manage Knowledge Graph')}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")

    lines.append(f"echo {make_item('4', 'MCP Server Manager', 'Port: 7864', C_DARK)}")
    lines.append(f"echo {make_sub('Configure Agent Tools')}")
    lines.append(f"echo {make_sub('Install MCP Servers')}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")

    lines.append(f"echo {make_item('5', 'Install Dependencies', '', C_GREY)}")
    lines.append(f"echo {make_sub('Python + Node.js (v20)')}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")

    lines.append(f"echo {make_item('6', 'Calibrate Emotion Detection', '', C_GREY)}")
    lines.append(f"echo {make_sub('Personalize SER to your voice')}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")

    lines.append(f"echo {make_item('7', 'Exit', '', C_RED)}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")

    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_ORANGE)}")
    lines.append("echo.")
    lines.append(f'set /p "choice={C_AMBER}Select option [1-7]: {C_RESET}"')

    lines.append('if "%choice%"=="1" goto VOICECHAT')
    lines.append('if "%choice%"=="2" goto MOBILE')
    lines.append('if "%choice%"=="3" goto MANAGER')
    lines.append('if "%choice%"=="4" goto MCPMANAGER')
    lines.append('if "%choice%"=="5" goto INSTALL')
    lines.append('if "%choice%"=="6" goto CALIBRATE')
    lines.append('if "%choice%"=="7" exit /b 0')
    lines.append("goto MENU")

    # --- VOICE CHAT ---
    lines.append(":VOICECHAT")
    lines.append("cls")
    lines.append(f"echo {C_ORANGE}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_ORANGE)}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}   {C_ORANGE}██╗   ██╗ ██████╗ ██╗ ██████╗███████╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗{C_RESET}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}   {C_ORANGE}██║   ██║██╔═══██╗██║██╔════╝██╔════╝    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝{C_RESET}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}   {C_ORANGE}██║   ██║██║   ██║██║██║     █████╗      ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   {C_RESET}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}   {C_ORANGE}╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝      ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   {C_RESET}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}   {C_ORANGE} ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   {C_RESET}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}   {C_ORANGE}  ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   {C_RESET}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}                              {C_GREY}PTT + VAD + Memory + Tools + Vision{C_RESET}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_ORANGE)}")
    lines.append(f"echo.")
    lines.append(f"echo   {C_AMBER}URL:{C_RESET} {C_GOLD}http://localhost:7861{C_RESET}")
    lines.append(f"echo.")
    lines.append(f"echo {make_bar('-', TERM_WIDTH, C_GREY)}")
    lines.append(f"echo.")
    lines.append('if not exist "recordings" mkdir recordings')
    lines.append('echo ready^|0^|Starting... > recordings\\ptt_status.txt')
    lines.append(f'wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ -d .venv ] && echo VENV_OK || echo VENV_MISSING" > "%TEMP%\\venv_check.txt"')
    lines.append('set /p VENV_STATUS=<"%TEMP%\\venv_check.txt"')
    lines.append('if "%VENV_STATUS%"=="VENV_MISSING" (')
    lines.append(f'    echo {C_RED}ERROR: Virtual environment not found!{C_RESET}')
    lines.append(f'    echo {C_AMBER}Please run option [4] Install Dependencies first.{C_RESET}')
    lines.append('    echo.')
    lines.append('    pause')
    lines.append('    goto MENU')
    lines.append(')')
    lines.append('start "" /min wscript.exe scripts\\ptt_hidden.vbs')
    lines.append('start "" /min pythonw audio\\vad_windows.py')
    lines.append(f'echo {C_ORANGE}')
    lines.append(f'wsl -d %WSL_DISTRO% -e bash -c "{nvm_load} cd %WSL_WIN_PATH% && source .venv/bin/activate && python -W ignore tts2_agent.py"')
    lines.append("taskkill /f /im pythonw.exe 2>nul")
    lines.append("echo.")
    lines.append(f"echo {C_RED}Process exited. Press any key to return to menu...{C_RESET}")
    lines.append("pause >nul")
    lines.append("goto MENU")

    # --- MOBILE ACCESS MODE ---
    lines.append(":MOBILE")
    lines.append("cls")
    lines.append(f"echo {C_GOLD}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_GOLD)}")
    lines.append(f"echo {C_GOLD}:{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}   {C_GOLD}███╗   ███╗ ██████╗ ██████╗ ██╗██╗     ███████╗{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}   {C_GOLD}████╗ ████║██╔═══██╗██╔══██╗██║██║     ██╔════╝{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}   {C_GOLD}██╔████╔██║██║   ██║██████╔╝██║██║     █████╗  {C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}   {C_GOLD}██║╚██╔╝██║██║   ██║██╔══██╗██║██║     ██╔══╝  {C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}   {C_GOLD}██║ ╚═╝ ██║╚██████╔╝██████╔╝██║███████╗███████╗{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}   {C_GOLD}╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚═╝╚══════╝╚══════╝{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}              {C_GREY}Phone + Tablet + Any Device Access{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_GOLD)}")
    lines.append(f"echo.")
    lines.append(f"echo   {C_AMBER}This mode generates a public HTTPS URL for your phone.{C_RESET}")
    lines.append(f"echo   {C_AMBER}The URL will appear below when the server starts.{C_RESET}")
    lines.append(f"echo.")
    lines.append(f"echo   {C_GREY}Features:{C_RESET}")
    lines.append(f"echo   {C_GREY}  - Touch-friendly HOLD TO TALK button{C_RESET}")
    lines.append(f"echo   {C_GREY}  - Install as webapp (Add to Home Screen){C_RESET}")
    lines.append(f"echo   {C_GREY}  - Works on iOS and Android{C_RESET}")
    lines.append(f"echo.")
    lines.append(f"echo {make_bar('-', TERM_WIDTH, C_GREY)}")
    lines.append(f"echo.")
    lines.append('if not exist "recordings" mkdir recordings')
    lines.append('echo ready^|0^|Starting... > recordings\\ptt_status.txt')
    lines.append(f'wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ -d .venv ] && echo VENV_OK || echo VENV_MISSING" > "%TEMP%\\venv_check.txt"')
    lines.append('set /p VENV_STATUS=<"%TEMP%\\venv_check.txt"')
    lines.append('if "%VENV_STATUS%"=="VENV_MISSING" (')
    lines.append(f'    echo {C_RED}ERROR: Virtual environment not found!{C_RESET}')
    lines.append(f'    echo {C_AMBER}Please run option [5] Install Dependencies first.{C_RESET}')
    lines.append('    echo.')
    lines.append('    pause')
    lines.append('    goto MENU')
    lines.append(')')
    lines.append('start "" /min wscript.exe scripts\\ptt_hidden.vbs')
    lines.append('start "" /min pythonw audio\\vad_windows.py')
    lines.append(f'echo {C_GOLD}')
    # Run with SHARE_MODE=1 environment variable for mobile access
    lines.append(f'wsl -d %WSL_DISTRO% -e bash -c "{nvm_load} cd %WSL_WIN_PATH% && source .venv/bin/activate && SHARE_MODE=1 python -W ignore tts2_agent.py"')
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
    lines.append(f"echo {C_GOLD}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_GOLD)}")
    lines.append(f"echo {C_GOLD}:{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}   {C_GOLD} ██████╗██╗  ██╗ █████╗ ██████╗  █████╗  ██████╗████████╗███████╗██████╗ ███████╗{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}   {C_GOLD}██╔════╝██║  ██║██╔══██╗██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}   {C_GOLD}██║     ███████║███████║██████╔╝███████║██║        ██║   █████╗  ██████╔╝███████╗{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}   {C_GOLD}██║     ██╔══██║██╔══██║██╔══██╗██╔══██║██║        ██║   ██╔══╝  ██╔══██╗╚════██║{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}   {C_GOLD}╚██████╗██║  ██║██║  ██║██║  ██║██║  ██║╚██████╗   ██║   ███████╗██║  ██║███████║{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}   {C_GOLD} ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}                        {C_GREY}Personalities + Memory + Knowledge Graph{C_RESET}")
    lines.append(f"echo {C_GOLD}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_GOLD)}")
    lines.append(f"echo.")
    lines.append(f"echo   {C_AMBER}URL:{C_RESET} {C_GOLD}http://localhost:7863{C_RESET}")
    lines.append(f"echo.")
    lines.append(f"echo {make_bar('-', TERM_WIDTH, C_GREY)}")
    lines.append(f"echo.")
    lines.append(f'echo {C_GOLD}')
    lines.append(f'wsl -d %WSL_DISTRO% -e bash -c "{nvm_load} cd %WSL_WIN_PATH% && source .venv/bin/activate 2>/dev/null; python character_manager_ui.py"')
    lines.append("goto MENU")

    # --- MCP MANAGER ---
    lines.append(":MCPMANAGER")
    lines.append("cls")
    lines.append("color 04")
    lines.append("chcp 65001 >nul")
    lines.append(f"echo {C_AMBER}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_AMBER)}")
    lines.append(f"echo {C_AMBER}:{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}   {C_AMBER}███╗   ███╗ ██████╗██████╗     ███╗   ███╗ █████╗ ███╗   ██╗ █████╗  ██████╗ ███████╗██████╗ {C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}   {C_AMBER}████╗ ████║██╔════╝██╔══██╗    ████╗ ████║██╔══██╗████╗  ██║██╔══██╗██╔════╝ ██╔════╝██╔══██╗{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}   {C_AMBER}██╔████╔██║██║     ██████╔╝    ██╔████╔██║███████║██╔██╗ ██║███████║██║  ███╗█████╗  ██████╔╝{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}   {C_AMBER}██║╚██╔╝██║██║     ██╔═══╝     ██║╚██╔╝██║██╔══██║██║╚██╗██║██╔══██║██║   ██║██╔══╝  ██╔══██╗{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}   {C_AMBER}██║ ╚═╝ ██║╚██████╗██║         ██║ ╚═╝ ██║██║  ██║██║ ╚████║██║  ██║╚██████╔╝███████╗██║  ██║{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}   {C_AMBER}╚═╝     ╚═╝ ╚═════╝╚═╝         ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}                            {C_GREY}Configure Agent Tools + Install Servers{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_AMBER)}")
    lines.append(f"echo.")
    lines.append(f"echo   {C_AMBER}URL:{C_RESET} {C_GOLD}http://localhost:7864{C_RESET}")
    lines.append(f"echo.")
    lines.append(f"echo {make_bar('-', TERM_WIDTH, C_GREY)}")
    lines.append(f"echo.")
    lines.append(f'echo {C_AMBER}')
    lines.append(f'wsl -d %WSL_DISTRO% -e bash -c "{nvm_load} cd %WSL_WIN_PATH% && source .venv/bin/activate 2>/dev/null; python mcp_manager_ui.py"')
    lines.append("goto MENU")

    # --- INSTALL ---
    lines.append(":INSTALL")
    lines.append("cls")
    lines.append("color 08")
    lines.append("chcp 65001 >nul")
    lines.append(f"echo {C_DARK}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}   {C_DARK}██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗     ███████╗██████╗ {C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}   {C_DARK}██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║     ██╔════╝██╔══██╗{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}   {C_DARK}██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║     █████╗  ██████╔╝{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}   {C_DARK}██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║     ██╔══╝  ██╔══██╗{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}   {C_DARK}██║██║ ╚████║███████║   ██║   ██║  ██║███████╗███████╗███████╗██║  ██║{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}   {C_DARK}╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}                     {C_GREY}Complete TTS2 Voice Agent Installation{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append(f"echo.")
    lines.append(f"echo {make_bar('-', TERM_WIDTH, C_GREY)}")
    lines.append("echo.")

    # Node.js installation
    lines.append(f"echo   {C_AMBER}[1/8]{C_RESET} Checking/Installing Node.js 20 (via NVM)...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && export NVM_DIR=\\"$HOME/.nvm\\" && [ -s \\"$NVM_DIR/nvm.sh\\" ] && . \\"$NVM_DIR/nvm.sh\\" && nvm install 20 && nvm alias default 20 && nvm use default && node -v"')
    lines.append("echo.")

    # Python venv + system dependencies (ffmpeg for audio processing)
    lines.append(f"echo   {C_AMBER}[2/8]{C_RESET} Checking Python Environment + System Dependencies...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "sudo apt-get update && sudo apt-get install -y python3-venv python3-dev build-essential ffmpeg"')
    lines.append(f"echo   {C_AMBER}[3/8]{C_RESET} Creating virtual environment...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ ! -d .venv ] && python3 -m venv .venv && echo Created .venv || echo .venv already exists"')
    lines.append("echo.")

    # PyTorch with CUDA (must be first before other ML packages)
    lines.append(f"echo   {C_AMBER}[4/8]{C_RESET} Installing PyTorch with CUDA 12.1 support...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && pip install --upgrade pip --progress-bar off && pip install --progress-bar off torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"')
    lines.append("echo.")

    # Main requirements
    lines.append(f"echo   {C_AMBER}[5/8]{C_RESET} Installing Python packages from requirements.txt...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && pip install --progress-bar off -r requirements.txt"')
    lines.append("echo.")

    # FunASR + SenseVoice (STT backends)
    lines.append(f"echo   {C_AMBER}[6/8]{C_RESET} Installing FunASR + SenseVoice (STT backends)...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && pip install --progress-bar off funasr modelscope"')
    lines.append("echo.")

    # Pre-download STT models
    lines.append(f"echo   {C_AMBER}[7/8]{C_RESET} Pre-downloading STT models (SenseVoice + FunASR)...")
    lines.append(f"echo         {C_GREY}This may take a few minutes on first run...{C_RESET}")
    # Download SenseVoiceSmall and paraformer models
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \\"from funasr import AutoModel; print(\'Downloading SenseVoiceSmall...\'); m = AutoModel(model=\'FunAudioLLM/SenseVoiceSmall\', device=\'cpu\', hub=\'hf\'); print(\'SenseVoice ready!\')\\""')
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \\"from funasr import AutoModel; print(\'Downloading Paraformer-zh...\'); m = AutoModel(model=\'paraformer-zh\', device=\'cpu\'); print(\'FunASR ready!\')\\""')
    lines.append("echo.")

    # Windows audio dependencies
    lines.append(f"echo   {C_AMBER}[8/8]{C_RESET} Installing Windows audio dependencies...")
    lines.append("pip install keyboard pyaudio numpy --quiet 2>nul")

    lines.append("echo.")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_AMBER)}")
    lines.append(f"echo.")
    lines.append(f"echo   {C_AMBER}Installation Complete!{C_RESET}")
    lines.append(f"echo.")
    lines.append(f"echo   {C_GREY}Installed Components:{C_RESET}")
    lines.append(f"echo   {C_GREY}  - PyTorch with CUDA 12.1{C_RESET}")
    lines.append(f"echo   {C_GREY}  - TTS: IndexTTS2, Kokoro, Supertonic, Soprano{C_RESET}")
    lines.append(f"echo   {C_GREY}  - STT: Faster-Whisper, SenseVoice, FunASR{C_RESET}")
    lines.append(f"echo   {C_GREY}  - Memory: sqlite-vec, Qwen3 embeddings{C_RESET}")
    lines.append(f"echo   {C_GREY}  - Node.js 20 (for MCP servers){C_RESET}")
    lines.append(f"echo.")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_AMBER)}")
    lines.append("echo.")
    lines.append("pause")
    lines.append("goto MENU")

    # --- CALIBRATE ---
    lines.append(":CALIBRATE")
    lines.append("cls")
    lines.append("color 08")
    lines.append("chcp 65001 >nul")
    lines.append(f"echo {C_GREY}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_GREY)}")
    lines.append(f"echo {C_GREY}:{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}   {C_GREY}███████╗███╗   ███╗ ██████╗ ████████╗██╗ ██████╗ ███╗   ██╗{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}   {C_GREY}██╔════╝████╗ ████║██╔═══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}   {C_GREY}█████╗  ██╔████╔██║██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}   {C_GREY}██╔══╝  ██║╚██╔╝██║██║   ██║   ██║   ██║██║   ██║██║╚██╗██║{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}   {C_GREY}███████╗██║ ╚═╝ ██║╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}   {C_GREY}╚══════╝╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}   {C_GREY} ██████╗ █████╗ ██╗     ██╗██████╗ ██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}   {C_GREY}██╔════╝██╔══██╗██║     ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}   {C_GREY}██║     ███████║██║     ██║██████╔╝██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}   {C_GREY}██║     ██╔══██║██║     ██║██╔══██╗██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}   {C_GREY}╚██████╗██║  ██║███████╗██║██████╔╝██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}   {C_GREY} ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}                     {C_AMBER}Personalize Speech Emotion Recognition to YOUR Voice{C_RESET}")
    lines.append(f"echo {C_GREY}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_GREY)}")
    lines.append(f"echo.")
    lines.append(f"echo   {C_AMBER}This tool calibrates emotion detection to YOUR voice.{C_RESET}")
    lines.append(f"echo   {C_AMBER}You will be asked to speak with different emotions.{C_RESET}")
    lines.append(f"echo   {C_AMBER}Each recording lasts 3 seconds.{C_RESET}")
    lines.append(f"echo.")
    lines.append(f"echo   {C_GREY}Make sure your microphone is connected and working.{C_RESET}")
    lines.append(f"echo.")
    lines.append(f"echo {make_bar('-', TERM_WIDTH, C_GREY)}")
    lines.append("echo.")
    lines.append(f'wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ -d .venv ] && echo VENV_OK || echo VENV_MISSING" > "%TEMP%\\venv_check.txt"')
    lines.append('set /p VENV_STATUS=<"%TEMP%\\venv_check.txt"')
    lines.append('if "%VENV_STATUS%"=="VENV_MISSING" (')
    lines.append(f'    echo {C_RED}ERROR: Virtual environment not found!{C_RESET}')
    lines.append(f'    echo {C_AMBER}Please run option [4] Install Dependencies first.{C_RESET}')
    lines.append('    echo.')
    lines.append('    pause')
    lines.append('    goto MENU')
    lines.append(')')
    lines.append(f"echo   {C_AMBER}Step 1: Recording samples on Windows...{C_RESET}")
    lines.append(f"echo.")
    lines.append(f'python tools/calibrate_record_windows.py')
    lines.append(f"echo.")
    lines.append(f"echo   {C_AMBER}Step 2: Analyzing with SER model in WSL...{C_RESET}")
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