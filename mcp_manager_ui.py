#!/usr/bin/env python3
"""
MCP Server Manager for IndexTTS2 Voice Chat

Manage Model Context Protocol (MCP) servers:
- View configured servers and their tools
- Add/remove server configurations
- Install popular MCP servers from npm
- Test server connections

Runs on port 7864
"""

import os
import sys
import json
import subprocess
import shutil
import asyncio
import platform
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import threading

import gradio as gr

# Import shared utilities
from utils import create_dark_theme

# Check for MCP
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "mcp_config.json"
TOOLS_DIR = SCRIPT_DIR / "tools"

PLATFORM = platform.system().lower()
IS_WINDOWS = PLATFORM == 'windows'
IS_WSL = PLATFORM == 'linux' and 'microsoft' in platform.uname().release.lower()

# Popular MCP servers that can be installed
POPULAR_SERVERS = {
    "filesystem": {
        "name": "Filesystem",
        "description": "Read/write files, list directories, search files",
        "package": "@modelcontextprotocol/server-filesystem",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
        "category": "Files"
    },
    "brave-search": {
        "name": "Brave Search",
        "description": "Web search using Brave Search API",
        "package": "@modelcontextprotocol/server-brave-search",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {"BRAVE_API_KEY": ""},
        "category": "Search"
    },
    "github": {
        "name": "GitHub",
        "description": "Interact with GitHub repositories, issues, PRs",
        "package": "@modelcontextprotocol/server-github",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_TOKEN": ""},
        "category": "Dev Tools"
    },
    "sqlite": {
        "name": "SQLite",
        "description": "Query and modify SQLite databases",
        "package": "mcp-server-sqlite",
        "command": "npx",
        "args": ["-y", "mcp-server-sqlite", "database.db"],
        "category": "Database"
    },
    "memory": {
        "name": "Memory",
        "description": "Persistent key-value memory storage",
        "package": "@modelcontextprotocol/server-memory",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "category": "Storage"
    },
    "puppeteer": {
        "name": "Puppeteer",
        "description": "Browser automation - navigate, screenshot, interact",
        "package": "@modelcontextprotocol/server-puppeteer",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
        "category": "Browser"
    },
    "fetch": {
        "name": "Fetch",
        "description": "Fetch and extract content from URLs",
        "package": "@modelcontextprotocol/server-fetch",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-fetch"],
        "category": "Web"
    },
    "time": {
        "name": "Time",
        "description": "Get current time in various timezones",
        "package": "@modelcontextprotocol/server-time",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-time"],
        "category": "Utility"
    },
    "sequential-thinking": {
        "name": "Sequential Thinking",
        "description": "Step-by-step reasoning and problem solving",
        "package": "@modelcontextprotocol/server-sequential-thinking",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        "category": "Reasoning"
    }
}


# ============================================================================
# Config Management
# ============================================================================

def load_config() -> Dict[str, Any]:
    """Load MCP configuration from file"""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[MCP] Error loading config: {e}")
    return {"mcpServers": {}}


def save_config(config: Dict[str, Any]):
    """Save MCP configuration to file"""
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"[MCP] Error saving config: {e}")
        return False


def get_configured_servers() -> List[Dict[str, Any]]:
    """Get list of configured servers with details"""
    config = load_config()
    servers = []

    for name, server_config in config.get("mcpServers", {}).items():
        servers.append({
            "name": name,
            "command": server_config.get("command", ""),
            "args": server_config.get("args", []),
            "env": server_config.get("env", {}),
            "status": "configured"
        })

    return servers


def add_server_to_config(name: str, command: str, args: List[str], env: Dict[str, str] = None) -> Tuple[bool, str]:
    """Add a new server to configuration"""
    if not name or not name.strip():
        return False, "Server name is required"

    name = name.strip().lower().replace(" ", "-")

    config = load_config()

    if name in config.get("mcpServers", {}):
        return False, f"Server '{name}' already exists"

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    server_entry = {
        "command": command,
        "args": args
    }

    if env:
        # Filter out empty values
        env = {k: v for k, v in env.items() if v}
        if env:
            server_entry["env"] = env

    config["mcpServers"][name] = server_entry

    if save_config(config):
        return True, f"✓ Server '{name}' added successfully"
    else:
        return False, "Failed to save configuration"


def remove_server_from_config(name: str) -> Tuple[bool, str]:
    """Remove a server from configuration"""
    config = load_config()

    if name not in config.get("mcpServers", {}):
        return False, f"Server '{name}' not found"

    del config["mcpServers"][name]

    if save_config(config):
        return True, f"✓ Server '{name}' removed"
    else:
        return False, "Failed to save configuration"


def update_server_in_config(name: str, command: str, args: List[str], env: Dict[str, str] = None) -> Tuple[bool, str]:
    """Update an existing server configuration"""
    config = load_config()

    if name not in config.get("mcpServers", {}):
        return False, f"Server '{name}' not found"

    server_entry = {
        "command": command,
        "args": args
    }

    if env:
        env = {k: v for k, v in env.items() if v}
        if env:
            server_entry["env"] = env

    config["mcpServers"][name] = server_entry

    if save_config(config):
        return True, f"✓ Server '{name}' updated"
    else:
        return False, "Failed to save configuration"


# ============================================================================
# Server Testing
# ============================================================================

async def test_server_connection(name: str) -> Tuple[bool, str, List[Dict]]:
    """Test connection to a server and list its tools"""
    if not MCP_AVAILABLE:
        return False, "MCP package not installed", []

    config = load_config()
    server_config = config.get("mcpServers", {}).get(name)

    if not server_config:
        return False, f"Server '{name}' not found in config", []

    command = server_config.get("command")
    args = server_config.get("args", [])
    env = server_config.get("env", {})

    # Merge environment
    full_env = os.environ.copy()
    full_env.update(env)

    try:
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=full_env
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List tools
                result = await session.list_tools()
                tools = []
                for tool in result.tools:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description or "No description"
                    })

                return True, f"Connected! Found {len(tools)} tools", tools

    except FileNotFoundError:
        return False, f"Command not found: {command}. Is Node.js installed?", []
    except asyncio.TimeoutError:
        return False, "Connection timed out. Server may be unresponsive.", []
    except Exception as e:
        # Show more of the error message for better debugging
        error_msg = str(e)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        return False, f"Connection failed: {error_msg}", []


def test_server_sync(name: str) -> Tuple[bool, str, str]:
    """Synchronous wrapper for testing server connection with timeout"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Add timeout to prevent UI freeze (30 seconds max)
        async def test_with_timeout():
            return await asyncio.wait_for(test_server_connection(name), timeout=30.0)

        success, message, tools = loop.run_until_complete(test_with_timeout())
        loop.close()

        if tools:
            tools_text = "\n".join([f"- **{t['name']}**: {t['description'][:80]}..." for t in tools])
        else:
            tools_text = "*No tools available*"

        return success, message, tools_text
    except asyncio.TimeoutError:
        return False, "Server test timed out after 30 seconds", ""
    except Exception as e:
        return False, f"Error: {e}", ""


# ============================================================================
# npm/npx Operations
# ============================================================================

def check_node_installed() -> Tuple[bool, str]:
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, f"Node.js {result.stdout.strip()}"
        return False, "Node.js not working properly"
    except FileNotFoundError:
        return False, "Node.js not installed"
    except Exception as e:
        return False, f"Error checking Node.js: {e}"


def check_npx_available() -> bool:
    """Check if npx is available"""
    try:
        result = subprocess.run(
            ["npx", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        print(f"[MCP] npx check failed: {e}")
        return False


def install_npm_package(package: str) -> Tuple[bool, str]:
    """Install an npm package globally"""
    try:
        result = subprocess.run(
            ["npm", "install", "-g", package],
            capture_output=True,
            text=True,
            timeout=180  # Increased timeout for slow connections
        )
        if result.returncode == 0:
            return True, f"Installed {package}"
        # Show full error message for better debugging
        error_msg = result.stderr.strip() if result.stderr else "Unknown error"
        if len(error_msg) > 800:
            error_msg = error_msg[:800] + "..."
        return False, f"Installation failed:\n{error_msg}"
    except FileNotFoundError:
        return False, "npm not found. Install Node.js first."
    except subprocess.TimeoutExpired:
        return False, "Installation timed out. Check your network connection."
    except Exception as e:
        return False, f"Error: {e}"


# ============================================================================
# UI Functions
# ============================================================================

def get_servers_table() -> str:
    """Get configured servers as markdown table"""
    servers = get_configured_servers()

    if not servers:
        return """| Name | Command | Status |
|------|---------|--------|
| *No servers configured* | | |

**Get started:** Add a server from the catalog or configure one manually."""

    header = "| Name | Command | Args | Status |\n|------|---------|------|--------|\n"
    rows = []

    for s in servers:
        args_str = " ".join(s["args"][:3])
        if len(s["args"]) > 3:
            args_str += "..."
        rows.append(f"| `{s['name']}` | {s['command']} | {args_str} | ⚪ {s['status']} |")

    return header + "\n".join(rows)


def get_server_dropdown_choices() -> List[str]:
    """Get server names for dropdown"""
    servers = get_configured_servers()
    return [s["name"] for s in servers]


def get_catalog_table() -> str:
    """Get MCP server catalog as markdown table"""
    header = "| ID | Name | Description | Category |\n|----|------|-------------|----------|\n"
    rows = []

    for server_id, info in POPULAR_SERVERS.items():
        rows.append(f"| `{server_id}` | {info['name']} | {info['description']} | {info['category']} |")

    return header + "\n".join(rows)


def get_catalog_dropdown_choices() -> List[Tuple[str, str]]:
    """Get catalog servers for dropdown"""
    return [(f"{info['name']} - {info['description'][:40]}...", server_id)
            for server_id, info in POPULAR_SERVERS.items()]


def load_server_details(name: str) -> Tuple[str, str, str]:
    """Load server details for editing"""
    if not name:
        return "", "", ""

    config = load_config()
    server = config.get("mcpServers", {}).get(name, {})

    command = server.get("command", "")
    args = " ".join(server.get("args", []))
    env = json.dumps(server.get("env", {}), indent=2) if server.get("env") else ""

    return command, args, env


def add_from_catalog(server_id: str, custom_args: str = "", env_vars: str = "") -> str:
    """Add a server from the catalog"""
    if not server_id:
        return "❌ Select a server from the catalog"

    if server_id not in POPULAR_SERVERS:
        return f"❌ Unknown server: {server_id}"

    info = POPULAR_SERVERS[server_id]

    # Parse custom args if provided
    if custom_args and custom_args.strip():
        args = custom_args.strip().split()
    else:
        args = info.get("args", [])

    # Parse env vars
    env = info.get("env", {}).copy()
    if env_vars and env_vars.strip():
        try:
            custom_env = json.loads(env_vars)
            env.update(custom_env)
        except json.JSONDecodeError:
            # Try key=value format
            for line in env_vars.strip().split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    env[key.strip()] = value.strip()

    success, message = add_server_to_config(
        name=server_id,
        command=info.get("command", "npx"),
        args=args,
        env=env
    )

    if success:
        return f"✓ Added '{info['name']}' to configuration\n\n**Next:** Restart Voice Chat to use new MCP tools"
    return f"❌ {message}"


def add_custom_server(name: str, command: str, args: str, env_json: str) -> str:
    """Add a custom server configuration"""
    if not name or not command:
        return "❌ Name and command are required"

    args_list = args.strip().split() if args else []

    env = {}
    if env_json and env_json.strip():
        try:
            env = json.loads(env_json)
        except json.JSONDecodeError:
            return "❌ Invalid JSON in environment variables"

    success, message = add_server_to_config(name, command, args_list, env)
    return message


def update_server(name: str, command: str, args: str, env_json: str) -> str:
    """Update server configuration"""
    if not name:
        return "❌ No server selected"

    args_list = args.strip().split() if args else []

    env = {}
    if env_json and env_json.strip():
        try:
            env = json.loads(env_json)
        except json.JSONDecodeError:
            return "❌ Invalid JSON in environment variables"

    success, message = update_server_in_config(name, command, args_list, env)
    return message


def remove_server(name: str) -> Tuple[str, Any]:
    """Remove server and return updated UI"""
    if not name:
        return "❌ No server selected", gr.update()

    success, message = remove_server_from_config(name)
    return message, gr.update(choices=get_server_dropdown_choices())


def test_server(name: str) -> Tuple[str, str]:
    """Test server connection"""
    if not name:
        return "❌ No server selected", ""

    success, message, tools = test_server_sync(name)
    return message, tools


def get_config_json() -> str:
    """Get raw config JSON for viewing"""
    config = load_config()
    return json.dumps(config, indent=2)


def save_config_json(config_text: str) -> str:
    """Save raw config JSON"""
    try:
        config = json.loads(config_text)
        if save_config(config):
            return "✓ Configuration saved"
        return "❌ Failed to save"
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON: {e}"


# ============================================================================
# Gradio UI
# ============================================================================

def create_mcp_manager_ui():

    # Check prerequisites
    node_ok, node_msg = check_node_installed()
    npx_ok = check_npx_available()

    with gr.Blocks(title="MCP Server Manager", theme=create_dark_theme()) as app:

        gr.Markdown(f"""
        # 🔌 MCP Server Manager
        ### Configure Model Context Protocol Servers for Agent Tools

        {"✓ Node.js: " + node_msg if node_ok else "⚠️ Node.js not installed - npx servers won't work"}
        {"✓ npx available" if npx_ok else "⚠️ npx not available"}
        {"✓ MCP package installed" if MCP_AVAILABLE else "⚠️ MCP package not installed - run Install Dependencies"}
        """)

        with gr.Tabs():

            # ==================== CONFIGURED SERVERS ====================
            with gr.Tab("📋 My Servers"):

                servers_table = gr.Markdown(get_servers_table())
                refresh_btn = gr.Button("🔄 Refresh", size="sm")

                gr.Markdown("---")

                with gr.Row():
                    # Left - Server selector and actions
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔧 Manage Server")

                        server_select = gr.Dropdown(
                            choices=get_server_dropdown_choices(),
                            label="Select Server",
                            interactive=True
                        )

                        with gr.Row():
                            test_btn = gr.Button("🧪 Test Connection", variant="primary")
                            remove_btn = gr.Button("🗑️ Remove", variant="stop")

                        test_status = gr.Textbox(label="Status", interactive=False)
                        tools_display = gr.Markdown("*Select and test a server to see its tools*")

                    # Right - Edit server
                    with gr.Column(scale=2):
                        gr.Markdown("### ✏️ Edit Configuration")

                        edit_command = gr.Textbox(label="Command", placeholder="npx, python, node, etc.")
                        edit_args = gr.Textbox(label="Arguments (space-separated)", placeholder="-y @modelcontextprotocol/server-filesystem .")
                        edit_env = gr.Textbox(
                            label="Environment Variables (JSON)",
                            placeholder='{"API_KEY": "your-key"}',
                            lines=2
                        )

                        update_btn = gr.Button("💾 Update Server", variant="primary")
                        update_status = gr.Textbox(label="Status", interactive=False)

            # ==================== ADD FROM CATALOG ====================
            with gr.Tab("📦 Server Catalog"):

                gr.Markdown("""
                ### Popular MCP Servers
                *One-click installation of official Model Context Protocol servers*
                """)

                catalog_table = gr.Markdown(get_catalog_table())

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column():
                        catalog_select = gr.Dropdown(
                            choices=get_catalog_dropdown_choices(),
                            label="Select Server to Add",
                            interactive=True
                        )

                        catalog_args = gr.Textbox(
                            label="Custom Arguments (optional)",
                            placeholder="Leave empty for defaults",
                            info="Override default arguments"
                        )

                        catalog_env = gr.Textbox(
                            label="Environment Variables",
                            placeholder='{"API_KEY": "your-key"}',
                            info="Required for some servers (e.g., BRAVE_API_KEY for Brave Search)",
                            lines=2
                        )

                        add_catalog_btn = gr.Button("➕ Add to Configuration", variant="primary", size="lg")
                        catalog_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column():
                        gr.Markdown("""
                        ### 📝 Notes

                        **Prerequisites:**
                        - Node.js 18+ installed
                        - Some servers require API keys

                        **After adding:**
                        1. Server is added to `mcp_config.json`
                        2. Restart Voice Chat to connect
                        3. New tools appear in character tool options

                        **API Keys needed for:**
                        - Brave Search: `BRAVE_API_KEY`
                        - GitHub: `GITHUB_TOKEN`
                        """)

            # ==================== CUSTOM SERVER ====================
            with gr.Tab("➕ Add Custom"):

                gr.Markdown("""
                ### Add Custom MCP Server
                *Configure any MCP-compatible server*
                """)

                with gr.Row():
                    with gr.Column():
                        custom_name = gr.Textbox(
                            label="Server Name",
                            placeholder="my-server",
                            info="Unique identifier (lowercase, no spaces)"
                        )

                        custom_command = gr.Textbox(
                            label="Command",
                            placeholder="npx, python, node, uvx...",
                            info="The executable to run"
                        )

                        custom_args = gr.Textbox(
                            label="Arguments",
                            placeholder="-y @scope/package-name ./path",
                            info="Space-separated arguments"
                        )

                        custom_env = gr.Textbox(
                            label="Environment Variables (JSON)",
                            placeholder='{"API_KEY": "value"}',
                            lines=3
                        )

                        add_custom_btn = gr.Button("➕ Add Server", variant="primary", size="lg")
                        custom_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column():
                        gr.Markdown("""
                        ### 📖 Examples

                        **Python MCP Server:**
                        ```
                        Name: my-python-server
                        Command: python
                        Args: tools/my_mcp_server.py
                        ```

                        **npm Package:**
                        ```
                        Name: custom-tool
                        Command: npx
                        Args: -y @company/mcp-server-custom
                        ```

                        **Local Script:**
                        ```
                        Name: local-tools
                        Command: node
                        Args: ./my-server/index.js
                        ```
                        """)

            # ==================== RAW CONFIG ====================
            with gr.Tab("📝 Raw Config"):

                gr.Markdown("### Edit mcp_config.json directly")

                config_editor = gr.Code(
                    value=get_config_json(),
                    language="json",
                    label="mcp_config.json",
                    lines=20
                )

                with gr.Row():
                    refresh_config_btn = gr.Button("🔄 Reload")
                    save_config_btn = gr.Button("💾 Save", variant="primary")

                config_status = gr.Textbox(label="Status", interactive=False)

        # ==================== EVENT HANDLERS ====================

        # Refresh servers
        refresh_btn.click(
            fn=get_servers_table,
            outputs=[servers_table]
        ).then(
            fn=lambda: gr.update(choices=get_server_dropdown_choices()),
            outputs=[server_select]
        )

        # Load server details on select
        server_select.change(
            fn=load_server_details,
            inputs=[server_select],
            outputs=[edit_command, edit_args, edit_env]
        )

        # Test server
        test_btn.click(
            fn=test_server,
            inputs=[server_select],
            outputs=[test_status, tools_display]
        )

        # Remove server
        remove_btn.click(
            fn=remove_server,
            inputs=[server_select],
            outputs=[test_status, server_select]
        ).then(
            fn=get_servers_table,
            outputs=[servers_table]
        )

        # Update server
        update_btn.click(
            fn=update_server,
            inputs=[server_select, edit_command, edit_args, edit_env],
            outputs=[update_status]
        ).then(
            fn=get_servers_table,
            outputs=[servers_table]
        )

        # Add from catalog
        add_catalog_btn.click(
            fn=add_from_catalog,
            inputs=[catalog_select, catalog_args, catalog_env],
            outputs=[catalog_status]
        ).then(
            fn=get_servers_table,
            outputs=[servers_table]
        ).then(
            fn=lambda: gr.update(choices=get_server_dropdown_choices()),
            outputs=[server_select]
        )

        # Add custom server
        add_custom_btn.click(
            fn=add_custom_server,
            inputs=[custom_name, custom_command, custom_args, custom_env],
            outputs=[custom_status]
        ).then(
            fn=get_servers_table,
            outputs=[servers_table]
        ).then(
            fn=lambda: gr.update(choices=get_server_dropdown_choices()),
            outputs=[server_select]
        )

        # Raw config
        refresh_config_btn.click(
            fn=get_config_json,
            outputs=[config_editor]
        )

        save_config_btn.click(
            fn=save_config_json,
            inputs=[config_editor],
            outputs=[config_status]
        ).then(
            fn=get_servers_table,
            outputs=[servers_table]
        ).then(
            fn=lambda: gr.update(choices=get_server_dropdown_choices()),
            outputs=[server_select]
        )

    return app


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   [+] MCP Server Manager")
    print("   Configure Model Context Protocol Servers")
    print(f"   Platform: {PLATFORM.title()}" + (" (WSL)" if IS_WSL else ""))
    print("="*60)

    node_ok, node_msg = check_node_installed()
    print(f"\n{'✓' if node_ok else '✗'} Node.js: {node_msg}")
    print(f"{'✓' if MCP_AVAILABLE else '✗'} MCP Package: {'Installed' if MCP_AVAILABLE else 'Not installed'}")

    config = load_config()
    server_count = len(config.get("mcpServers", {}))
    print(f"✓ Configured servers: {server_count}")

    print(f"\n✓ Starting on http://127.0.0.1:7864")
    print("✓ Press Ctrl+C to stop\n")

    app = create_mcp_manager_ui()
    app.launch(server_port=7864, server_name="127.0.0.1", inbrowser=True, share=False)
