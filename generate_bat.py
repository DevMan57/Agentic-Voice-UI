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
        padding = 70 - len(label_part)
        return f"{C_ORANGE}:{C_RESET}  {color_main}{label_part}{' ' * padding}{C_GREY}{status}{C_RESET}"
    else:
        return f"{C_ORANGE}:{C_RESET}  {color_main}{label_part}{C_RESET}"

def make_sub(label, info=""):
    return f"{C_ORANGE}:{C_RESET}       {C_GREY}+-- {label}{C_RESET}"

def make_title(text, color=C_AMBER, width=TERM_WIDTH):
    """Center a title within the terminal width"""
    # Account for ANSI codes in padding calculation
    text_len = len(text)
    total_padding = width - text_len
    left_pad = total_padding // 2
    return f"{' ' * left_pad}{color}{text}{C_RESET}"

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
        lines.append(f"echo {C_ORANGE}:{C_RESET}   {C_ORANGE}{safe_line}{C_RESET}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}")
    lines.append(f"echo {C_ORANGE}:{C_RESET}                           {C_AMBER}Multi-Character Memory System{C_RESET}")
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
    lines.append(f'    echo {C_AMBER}Please run option [5] Installer first.{C_RESET}')
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
    lines.append(f"echo {make_title('Configure Agent Tools + Install Servers', C_GREY)}")
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

    # --- INSTALL MENU ---
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
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}  {C_RED}[0] First-Time Setup (WSL2 + Ubuntu){C_RESET}                               {C_GREY}Requires reboot{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}       {C_GREY}+-- Run this FIRST if you don't have WSL2/Ubuntu{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}  {C_AMBER}[1] Full Install (Recommended){C_RESET}                                    {C_GREY}~15-30 min{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}       {C_GREY}+-- Check prerequisites, install deps, download all models{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}  {C_AMBER}[2] Check Prerequisites Only{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}       {C_GREY}+-- Verify NVIDIA GPU, WSL2, Ubuntu, CUDA{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}  {C_AMBER}[3] Install Dependencies Only{C_RESET}                                     {C_GREY}~10-20 min{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}       {C_GREY}+-- Python venv, PyTorch, pip packages{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}  {C_AMBER}[4] Download Models{C_RESET}                                               {C_GREY}~5-15 min{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}       {C_GREY}+-- TTS models, STT models, embeddings (~10GB total){C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}  {C_GREY}[5] Back to Main Menu{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append("echo.")
    lines.append(f'set /p "install_choice={C_AMBER}Select option [0-5]: {C_RESET}"')
    lines.append('if "%install_choice%"=="0" goto INSTALL_WSL')
    lines.append('if "%install_choice%"=="1" goto INSTALL_FULL')
    lines.append('if "%install_choice%"=="2" goto INSTALL_PREREQ')
    lines.append('if "%install_choice%"=="3" goto INSTALL_DEPS')
    lines.append('if "%install_choice%"=="4" goto INSTALL_MODELS_MENU')
    lines.append('if "%install_choice%"=="5" goto MENU')
    lines.append("goto INSTALL")

    # --- WSL2 + UBUNTU SETUP ---
    lines.append(":INSTALL_WSL")
    lines.append("cls")
    lines.append(f"echo {C_RED}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_RED)}")
    lines.append(f"echo {make_title('First-Time Setup: WSL2 + Ubuntu', C_AMBER)}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_RED)}")
    lines.append("echo.")
    lines.append(f"echo   {C_AMBER}This will install:{C_RESET}")
    lines.append(f"echo   {C_GREY}  - WSL2 (Windows Subsystem for Linux){C_RESET}")
    lines.append(f"echo   {C_GREY}  - Ubuntu distribution{C_RESET}")
    lines.append("echo.")
    lines.append(f"echo   {C_RED}IMPORTANT:{C_RESET}")
    lines.append(f"echo   {C_GREY}  - Requires administrator privileges{C_RESET}")
    lines.append(f"echo   {C_GREY}  - May require a system reboot{C_RESET}")
    lines.append(f"echo   {C_GREY}  - After reboot, Ubuntu will open for initial setup{C_RESET}")
    lines.append(f"echo   {C_GREY}  - You'll create a Linux username and password{C_RESET}")
    lines.append("echo.")
    lines.append(f"echo {make_bar('-', TERM_WIDTH, C_GREY)}")
    lines.append("echo.")

    # Check if already have WSL and Ubuntu
    lines.append(f"echo   {C_AMBER}Checking current status...{C_RESET}")
    lines.append("echo.")
    lines.append("set WSL_INSTALLED=0")
    lines.append("set UBUNTU_INSTALLED=0")
    lines.append('wsl --status >nul 2>&1 && set WSL_INSTALLED=1')
    lines.append('wsl -d Ubuntu -e echo OK >nul 2>&1 && set UBUNTU_INSTALLED=1')

    lines.append('if "%WSL_INSTALLED%"=="1" (')
    lines.append(f'    echo   {C_AMBER}[OK] WSL2 is already installed{C_RESET}')
    lines.append(') else (')
    lines.append(f'    echo   {C_GREY}[ ] WSL2 not installed - will install{C_RESET}')
    lines.append(')')

    lines.append('if "%UBUNTU_INSTALLED%"=="1" (')
    lines.append(f'    echo   {C_AMBER}[OK] Ubuntu is already installed{C_RESET}')
    lines.append(') else (')
    lines.append(f'    echo   {C_GREY}[ ] Ubuntu not installed - will install{C_RESET}')
    lines.append(')')
    lines.append("echo.")

    # If both installed, skip
    lines.append('if "%WSL_INSTALLED%"=="1" if "%UBUNTU_INSTALLED%"=="1" (')
    lines.append(f'    echo   {C_AMBER}WSL2 and Ubuntu are already installed!{C_RESET}')
    lines.append(f'    echo   {C_GREY}You can proceed with option [1] Full Install.{C_RESET}')
    lines.append('    echo.')
    lines.append('    pause')
    lines.append('    goto INSTALL')
    lines.append(')')

    lines.append(f"echo {make_bar('-', TERM_WIDTH, C_GREY)}")
    lines.append("echo.")
    lines.append(f'set /p "wsl_confirm={C_AMBER}Proceed with WSL2/Ubuntu installation? (Y/N): {C_RESET}"')
    lines.append('if /i not "%wsl_confirm%"=="Y" goto INSTALL')
    lines.append("echo.")

    # Check admin privileges
    lines.append(f"echo   {C_AMBER}Checking administrator privileges...{C_RESET}")
    lines.append('net session >nul 2>&1')
    lines.append('if errorlevel 1 (')
    lines.append(f'    echo   {C_RED}ERROR: This requires administrator privileges!{C_RESET}')
    lines.append("    echo.")
    lines.append(f'    echo   {C_GREY}Please:{C_RESET}')
    lines.append(f'    echo   {C_GREY}  1. Close this window{C_RESET}')
    lines.append(f'    echo   {C_GREY}  2. Right-click VoiceChat.bat{C_RESET}')
    lines.append(f'    echo   {C_GREY}  3. Select \"Run as administrator\"{C_RESET}')
    lines.append("    echo.")
    lines.append('    pause')
    lines.append('    goto INSTALL')
    lines.append(')')
    lines.append(f"echo   {C_AMBER}[OK] Running as administrator{C_RESET}")
    lines.append("echo.")

    # Install WSL if needed
    lines.append('if "%WSL_INSTALLED%"=="0" (')
    lines.append(f'    echo   {C_AMBER}Installing WSL2...{C_RESET}')
    lines.append('    wsl --install --no-distribution')
    lines.append(f'    echo   {C_AMBER}WSL2 installed.{C_RESET}')
    lines.append(')')
    lines.append("echo.")

    # Install Ubuntu
    lines.append('if "%UBUNTU_INSTALLED%"=="0" (')
    lines.append(f'    echo   {C_AMBER}Installing Ubuntu...{C_RESET}')
    lines.append('    wsl --install -d Ubuntu')
    lines.append(')')
    lines.append("echo.")

    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_AMBER)}")
    lines.append(f"echo   {C_AMBER}WSL2 + Ubuntu installation initiated!{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_AMBER)}")
    lines.append("echo.")
    lines.append(f"echo   {C_AMBER}Next steps:{C_RESET}")
    lines.append(f"echo   {C_GREY}  1. If prompted, restart your computer{C_RESET}")
    lines.append(f"echo   {C_GREY}  2. After restart, Ubuntu will open automatically{C_RESET}")
    lines.append(f"echo   {C_GREY}  3. Create your Linux username and password{C_RESET}")
    lines.append(f"echo   {C_GREY}  4. Run VoiceChat.bat again and select [1] Full Install{C_RESET}")
    lines.append("echo.")
    lines.append("pause")
    lines.append("goto INSTALL")

    # --- PREREQUISITES CHECK ---
    lines.append(":INSTALL_PREREQ")
    lines.append("cls")
    lines.append(f"echo {C_DARK}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append(f"echo {make_title('Checking System Prerequisites', C_AMBER)}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append("echo.")
    lines.append("set PREREQ_OK=1")

    # Check NVIDIA GPU
    lines.append(f"echo   {C_AMBER}[1/4]{C_RESET} Checking NVIDIA GPU...")
    lines.append('nvidia-smi >nul 2>&1')
    lines.append('if errorlevel 1 (')
    lines.append(f'    echo         {C_RED}[X] NVIDIA GPU not detected or drivers not installed{C_RESET}')
    lines.append(f'    echo         {C_GREY}    Install drivers from: https://www.nvidia.com/drivers{C_RESET}')
    lines.append('    set PREREQ_OK=0')
    lines.append(') else (')
    lines.append(f'    echo         {C_AMBER}[OK] NVIDIA GPU detected{C_RESET}')
    lines.append('    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>nul')
    lines.append(')')
    lines.append("echo.")

    # Check WSL2
    lines.append(f"echo   {C_AMBER}[2/4]{C_RESET} Checking WSL2...")
    lines.append('wsl --status >nul 2>&1')
    lines.append('if errorlevel 1 (')
    lines.append(f'    echo         {C_RED}[X] WSL2 not installed{C_RESET}')
    lines.append(f'    echo         {C_GREY}    Run in PowerShell (Admin): wsl --install{C_RESET}')
    lines.append('    set PREREQ_OK=0')
    lines.append(') else (')
    lines.append(f'    echo         {C_AMBER}[OK] WSL2 installed{C_RESET}')
    lines.append(')')
    lines.append("echo.")

    # Check Ubuntu distro
    lines.append(f"echo   {C_AMBER}[3/4]{C_RESET} Checking Ubuntu distribution...")
    lines.append('wsl -d %WSL_DISTRO% -e echo "Ubuntu OK" >nul 2>&1')
    lines.append('if errorlevel 1 (')
    lines.append(f'    echo         {C_RED}[X] Ubuntu not found in WSL{C_RESET}')
    lines.append(f'    echo         {C_GREY}    Run: wsl --install -d Ubuntu{C_RESET}')
    lines.append('    set PREREQ_OK=0')
    lines.append(') else (')
    lines.append(f'    echo         {C_AMBER}[OK] Ubuntu distribution found{C_RESET}')
    lines.append(')')
    lines.append("echo.")

    # Check CUDA in WSL
    lines.append(f"echo   {C_AMBER}[4/4]{C_RESET} Checking CUDA access in WSL...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "nvidia-smi > /dev/null 2>&1 && echo CUDA_OK || echo CUDA_FAIL" > "%TEMP%\\cuda_check.txt"')
    lines.append('set /p CUDA_STATUS=<"%TEMP%\\cuda_check.txt"')
    lines.append('if "%CUDA_STATUS%"=="CUDA_FAIL" (')
    lines.append(f'    echo         {C_RED}[X] CUDA not accessible from WSL{C_RESET}')
    lines.append(f'    echo         {C_GREY}    Ensure you have a recent NVIDIA driver (535+){C_RESET}')
    lines.append('    set PREREQ_OK=0')
    lines.append(') else (')
    lines.append(f'    echo         {C_AMBER}[OK] CUDA accessible from WSL{C_RESET}')
    lines.append(')')
    lines.append("echo.")

    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append('if "%PREREQ_OK%"=="1" (')
    lines.append(f'    echo   {C_AMBER}All prerequisites met! Ready to install.{C_RESET}')
    lines.append(') else (')
    lines.append(f'    echo   {C_RED}Some prerequisites are missing. Please fix before installing.{C_RESET}')
    lines.append(')')
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append("echo.")
    lines.append("pause")
    lines.append("goto INSTALL")

    # --- FULL INSTALL ---
    lines.append(":INSTALL_FULL")
    lines.append("cls")
    lines.append(f"echo {C_DARK}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append(f"echo {make_title('Full Installation', C_AMBER)}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append("echo.")

    # Quick prereq check
    lines.append(f"echo   {C_AMBER}[Step 1/3]{C_RESET} Verifying prerequisites...")
    lines.append('nvidia-smi >nul 2>&1')
    lines.append('if errorlevel 1 (')
    lines.append(f'    echo   {C_RED}ERROR: NVIDIA GPU/driver not detected. Run option [2] for details.{C_RESET}')
    lines.append('    pause')
    lines.append('    goto INSTALL')
    lines.append(')')
    lines.append('wsl -d %WSL_DISTRO% -e echo "OK" >nul 2>&1')
    lines.append('if errorlevel 1 (')
    lines.append(f'    echo   {C_RED}ERROR: WSL Ubuntu not found. Run option [2] for details.{C_RESET}')
    lines.append('    pause')
    lines.append('    goto INSTALL')
    lines.append(')')
    lines.append(f"echo         {C_AMBER}[OK] Prerequisites verified{C_RESET}")
    lines.append("echo.")

    lines.append(f"echo   {C_AMBER}[Step 2/3]{C_RESET} Installing dependencies...")
    lines.append("goto INSTALL_DEPS_RUN")

    # --- INSTALL DEPENDENCIES ---
    lines.append(":INSTALL_DEPS")
    lines.append("cls")
    lines.append(f"echo {C_DARK}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append(f"echo {make_title('Installing Dependencies', C_AMBER)}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append("echo.")
    lines.append("set FROM_FULL=0")
    lines.append("goto INSTALL_DEPS_RUN")

    lines.append(":INSTALL_DEPS_RUN")

    # Node.js installation
    lines.append(f"echo   {C_AMBER}[1/7]{C_RESET} Installing Node.js 20 (via NVM)...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh 2>/dev/null | bash && export NVM_DIR=\\"$HOME/.nvm\\" && [ -s \\"$NVM_DIR/nvm.sh\\" ] && . \\"$NVM_DIR/nvm.sh\\" && nvm install 20 && nvm alias default 20 && nvm use default && node -v"')
    lines.append("echo.")

    # Python venv + system dependencies
    lines.append(f"echo   {C_AMBER}[2/7]{C_RESET} Installing system dependencies (ffmpeg, build tools)...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-dev build-essential ffmpeg git-lfs"')
    lines.append("echo.")

    lines.append(f"echo   {C_AMBER}[3/7]{C_RESET} Creating Python virtual environment...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && [ ! -d .venv ] && python3 -m venv .venv && echo Created .venv || echo .venv already exists"')
    lines.append("echo.")

    # PyTorch with CUDA
    lines.append(f"echo   {C_AMBER}[4/7]{C_RESET} Installing PyTorch with CUDA 12.1...")
    lines.append(f"echo         {C_GREY}This may take several minutes...{C_RESET}")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && pip install --upgrade pip -q && pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"')
    lines.append("echo.")

    # Main requirements
    lines.append(f"echo   {C_AMBER}[5/7]{C_RESET} Installing Python packages from requirements.txt...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && pip install -q -r requirements.txt"')
    lines.append("echo.")

    # FunASR + SenseVoice
    lines.append(f"echo   {C_AMBER}[6/7]{C_RESET} Installing FunASR + SenseVoice (STT backends)...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && pip install -q funasr modelscope"')
    lines.append("echo.")

    # Windows audio dependencies
    lines.append(f"echo   {C_AMBER}[7/7]{C_RESET} Installing Windows audio dependencies...")
    lines.append('where python >nul 2>&1')
    lines.append('if errorlevel 1 (')
    lines.append(f'    echo         {C_GREY}[SKIP] Windows Python not found - PTT will use fallback{C_RESET}')
    lines.append(f'    echo         {C_GREY}        Install Python from python.org if you want PTT support{C_RESET}')
    lines.append(') else (')
    lines.append('    pip install keyboard pyaudio numpy --quiet 2>nul')
    lines.append(f'    echo         {C_AMBER}[OK] Windows audio deps installed{C_RESET}')
    lines.append(')')
    lines.append("echo.")

    lines.append(f"echo   {C_AMBER}Dependencies installed!{C_RESET}")
    lines.append("echo.")

    # Check if coming from full install
    lines.append('if "%install_choice%"=="1" goto INSTALL_MODELS_ALL')
    lines.append("pause")
    lines.append("goto INSTALL")

    # --- MODELS MENU ---
    lines.append(":INSTALL_MODELS_MENU")
    lines.append("cls")
    lines.append(f"echo {C_DARK}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append(f"echo {make_title('Download Models', C_AMBER)}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}  {C_AMBER}[1] Download All Models (Recommended){C_RESET}                            {C_GREY}~9 GB{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}       {C_GREY}+-- IndexTTS2, Kokoro, Supertonic, STT, Embeddings{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}  {C_AMBER}[2] IndexTTS2 (High-quality TTS){C_RESET}                                  {C_GREY}~4.4 GB{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}       {C_GREY}+-- Voice cloning, emotion control, GPU required{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}  {C_AMBER}[3] Supertonic (Fast CPU TTS){C_RESET}                                     {C_GREY}~500 MB{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}       {C_GREY}+-- 6 preset voices, 167x realtime, CPU only{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}  {C_AMBER}[4] STT Models (SenseVoice + FunASR){C_RESET}                              {C_GREY}~1.5 GB{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}       {C_GREY}+-- Speech recognition with emotion detection{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}  {C_AMBER}[5] Embeddings (Qwen3 for Memory){C_RESET}                                 {C_GREY}~1.2 GB{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}       {C_GREY}+-- Required for memory/knowledge graph{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}  {C_AMBER}[6] NuExtract (Entity Extraction){C_RESET}                                 {C_GREY}~940 MB{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}       {C_GREY}+-- Extracts entities for knowledge graph{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}  {C_GREY}[7] Back{C_RESET}")
    lines.append(f"echo {C_DARK}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append("echo.")
    lines.append(f'set /p "model_choice={C_AMBER}Select option [1-7]: {C_RESET}"')
    lines.append('if "%model_choice%"=="1" goto INSTALL_MODELS_ALL')
    lines.append('if "%model_choice%"=="2" goto INSTALL_MODEL_INDEXTTS')
    lines.append('if "%model_choice%"=="3" goto INSTALL_MODEL_SUPERTONIC')
    lines.append('if "%model_choice%"=="4" goto INSTALL_MODEL_STT')
    lines.append('if "%model_choice%"=="5" goto INSTALL_MODEL_EMBEDDINGS')
    lines.append('if "%model_choice%"=="6" goto INSTALL_MODEL_NUEXTRACT')
    lines.append('if "%model_choice%"=="7" goto INSTALL')
    lines.append("goto INSTALL_MODELS_MENU")

    # --- DOWNLOAD ALL MODELS ---
    lines.append(":INSTALL_MODELS_ALL")
    lines.append("cls")
    lines.append(f"echo {C_DARK}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append(f"echo {make_title('Downloading All Models', C_AMBER)}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_DARK)}")
    lines.append("echo.")
    lines.append(f"echo   {C_GREY}Total download size: ~9 GB. This may take 10-30 minutes.{C_RESET}")
    lines.append("echo.")

    # Create models directory structure
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && mkdir -p models/{indextts2,supertonic,embeddings,nuextract,kokoro,hf_cache}"')

    # IndexTTS2
    lines.append(f"echo   {C_AMBER}[1/5]{C_RESET} Downloading IndexTTS2 models (~4.4 GB)...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \\"from huggingface_hub import snapshot_download; snapshot_download(\'IndexTeam/IndexTTS2\', local_dir=\'models/indextts2\', local_dir_use_symlinks=False)\\""')
    lines.append("echo.")

    # Supertonic
    lines.append(f"echo   {C_AMBER}[2/5]{C_RESET} Downloading Supertonic models (~500 MB)...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && git lfs install && if [ -d models/supertonic/.git ]; then echo Supertonic already exists; elif [ -d models/supertonic ]; then rmdir models/supertonic 2>/dev/null; git clone https://huggingface.co/neongeckocom/tts-vits-cv-en models/supertonic; else git clone https://huggingface.co/neongeckocom/tts-vits-cv-en models/supertonic; fi"')
    lines.append("echo.")

    # STT Models
    lines.append(f"echo   {C_AMBER}[3/5]{C_RESET} Downloading STT models (SenseVoice + FunASR)...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \\"from funasr import AutoModel; print(\'Downloading SenseVoice...\'); AutoModel(model=\'FunAudioLLM/SenseVoiceSmall\', device=\'cpu\', hub=\'hf\'); print(\'Done!\')\\""')
    lines.append("echo.")

    # Embeddings
    lines.append(f"echo   {C_AMBER}[4/5]{C_RESET} Downloading embedding model (Qwen3)...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \\"from huggingface_hub import snapshot_download; snapshot_download(\'Qwen/Qwen3-Embedding-0.6B\', local_dir=\'models/embeddings/qwen0.6b\', local_dir_use_symlinks=False)\\""')
    lines.append("echo.")

    # NuExtract
    lines.append(f"echo   {C_AMBER}[5/5]{C_RESET} Downloading NuExtract model (~940 MB)...")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \\"from huggingface_hub import hf_hub_download; import os; os.makedirs(\'models/nuextract\', exist_ok=True); hf_hub_download(\'numind/NuExtract-2.0-2B-GGUF\', filename=\'NuExtract-2.0-2B-Q4_K_M.gguf\', local_dir=\'models/nuextract\')\\""')
    lines.append("echo.")

    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_AMBER)}")
    lines.append(f"echo   {C_AMBER}All models downloaded!{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_AMBER)}")
    lines.append("echo.")

    # Check if from full install
    lines.append('if "%install_choice%"=="1" goto INSTALL_COMPLETE')
    lines.append("pause")
    lines.append("goto INSTALL")

    # --- INDIVIDUAL MODEL DOWNLOADS ---
    lines.append(":INSTALL_MODEL_INDEXTTS")
    lines.append("cls")
    lines.append(f"echo   {C_AMBER}Downloading IndexTTS2 models (~4.4 GB)...{C_RESET}")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && mkdir -p models/indextts2 && . .venv/bin/activate && python -c \\"from huggingface_hub import snapshot_download; snapshot_download(\'IndexTeam/IndexTTS2\', local_dir=\'models/indextts2\', local_dir_use_symlinks=False)\\""')
    lines.append(f"echo   {C_AMBER}Done!{C_RESET}")
    lines.append("pause")
    lines.append("goto INSTALL_MODELS_MENU")

    lines.append(":INSTALL_MODEL_SUPERTONIC")
    lines.append("cls")
    lines.append(f"echo   {C_AMBER}Downloading Supertonic models (~500 MB)...{C_RESET}")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && mkdir -p models && git lfs install && if [ -d models/supertonic/.git ]; then echo Supertonic already exists; elif [ -d models/supertonic ]; then rmdir models/supertonic 2>/dev/null; git clone https://huggingface.co/neongeckocom/tts-vits-cv-en models/supertonic; else git clone https://huggingface.co/neongeckocom/tts-vits-cv-en models/supertonic; fi"')
    lines.append(f"echo   {C_AMBER}Done!{C_RESET}")
    lines.append("pause")
    lines.append("goto INSTALL_MODELS_MENU")

    lines.append(":INSTALL_MODEL_STT")
    lines.append("cls")
    lines.append(f"echo   {C_AMBER}Downloading STT models (SenseVoice + FunASR)...{C_RESET}")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \\"from funasr import AutoModel; print(\'Downloading SenseVoice...\'); AutoModel(model=\'FunAudioLLM/SenseVoiceSmall\', device=\'cpu\', hub=\'hf\'); print(\'Done!\')\\""')
    lines.append(f"echo   {C_AMBER}Done!{C_RESET}")
    lines.append("pause")
    lines.append("goto INSTALL_MODELS_MENU")

    lines.append(":INSTALL_MODEL_EMBEDDINGS")
    lines.append("cls")
    lines.append(f"echo   {C_AMBER}Downloading embedding model (Qwen3, ~1.2 GB)...{C_RESET}")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && mkdir -p models/embeddings && . .venv/bin/activate && python -c \\"from huggingface_hub import snapshot_download; snapshot_download(\'Qwen/Qwen3-Embedding-0.6B\', local_dir=\'models/embeddings/qwen0.6b\', local_dir_use_symlinks=False)\\""')
    lines.append(f"echo   {C_AMBER}Done!{C_RESET}")
    lines.append("pause")
    lines.append("goto INSTALL_MODELS_MENU")

    lines.append(":INSTALL_MODEL_NUEXTRACT")
    lines.append("cls")
    lines.append(f"echo   {C_AMBER}Downloading NuExtract model (~940 MB)...{C_RESET}")
    lines.append('wsl -d %WSL_DISTRO% -e bash -c "cd %WSL_WIN_PATH% && . .venv/bin/activate && python -c \\"from huggingface_hub import hf_hub_download; import os; os.makedirs(\'models/nuextract\', exist_ok=True); hf_hub_download(\'numind/NuExtract-2.0-2B-GGUF\', filename=\'NuExtract-2.0-2B-Q4_K_M.gguf\', local_dir=\'models/nuextract\')\\""')
    lines.append(f"echo   {C_AMBER}Done!{C_RESET}")
    lines.append("pause")
    lines.append("goto INSTALL_MODELS_MENU")

    # --- INSTALL COMPLETE ---
    lines.append(":INSTALL_COMPLETE")
    lines.append("cls")
    lines.append(f"echo {C_AMBER}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_AMBER)}")
    lines.append(f"echo {C_AMBER}:{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}   {C_AMBER}██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗         ██████╗ ██╗  ██╗██╗{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}   {C_AMBER}██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║        ██╔═══██╗██║ ██╔╝██║{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}   {C_AMBER}██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║        ██║   ██║█████╔╝ ██║{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}   {C_AMBER}██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║        ██║   ██║██╔═██╗ ╚═╝{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}   {C_AMBER}██║██║ ╚████║███████║   ██║   ██║  ██║███████╗███████╗   ╚██████╔╝██║  ██╗██╗{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}   {C_AMBER}╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝    ╚═════╝ ╚═╝  ╚═╝╚═╝{C_RESET}")
    lines.append(f"echo {C_AMBER}:{C_RESET}")
    lines.append(f"echo {make_bar('=', TERM_WIDTH, C_AMBER)}")
    lines.append("echo.")
    lines.append(f"echo   {C_AMBER}Installation Complete!{C_RESET}")
    lines.append("echo.")
    lines.append(f"echo   {C_GREY}Installed Components:{C_RESET}")
    lines.append(f"echo   {C_GREY}  - PyTorch with CUDA 12.1{C_RESET}")
    lines.append(f"echo   {C_GREY}  - TTS: IndexTTS2, Kokoro, Supertonic{C_RESET}")
    lines.append(f"echo   {C_GREY}  - STT: Faster-Whisper, SenseVoice, FunASR{C_RESET}")
    lines.append(f"echo   {C_GREY}  - Memory: sqlite-vec, Qwen3 embeddings{C_RESET}")
    lines.append(f"echo   {C_GREY}  - Node.js 20 (for MCP servers){C_RESET}")
    lines.append("echo.")
    lines.append(f"echo   {C_AMBER}Next Steps:{C_RESET}")
    lines.append(f"echo   {C_GREY}  1. Install LM Studio from lmstudio.ai (for LLM){C_RESET}")
    lines.append(f"echo   {C_GREY}  2. Run option [1] Voice Agent to start{C_RESET}")
    lines.append("echo.")
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
    lines.append(f"echo {make_title('Personalize Speech Emotion Recognition to YOUR Voice', C_AMBER)}")
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
    lines.append(f'    echo {C_AMBER}Please run option [5] Installer first.{C_RESET}')
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