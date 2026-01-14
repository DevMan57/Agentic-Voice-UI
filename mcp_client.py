"""
Model Context Protocol (MCP) Client for IndexTTS2
Enables "Agent Skills" and connection to external tools/servers.
"""

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import AsyncExitStack

# Get the directory where this script lives (for config file resolution)
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "mcp_config.json"

# Check for MCP installation
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import CallToolResult, TextContent, ImageContent, EmbeddedResource
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("[MCP] Warning: 'mcp' package not found. Run Option 3 in VoiceChat.bat")

class MCPManager:
    """
    Manages connections to multiple MCP servers defined in mcp_config.json.
    Mimics the behavior of Claude Desktop's tool handling.
    """

    def __init__(self, config_path: Path = None):
        # Use absolute path to ensure config is found regardless of cwd
        self.config_path = config_path if config_path else DEFAULT_CONFIG_PATH
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.tools_cache: List[Dict] = []
        self._tool_map: Dict[str, str] = {}  # tool_name -> server_name
        
    async def initialize(self):
        """Start all servers defined in config"""
        if not MCP_AVAILABLE:
            return

        if not self.config_path.exists():
            self._create_default_config()
            
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"[MCP] Config error: {e}")
            return

        mcp_servers = config.get("mcpServers", {})
        print(f"[MCP] Found {len(mcp_servers)} servers in config.")

        for name, server_config in mcp_servers.items():
            try:
                await self._connect_server(name, server_config)
            except Exception as e:
                print(f"[MCP] Failed to connect to '{name}': {e}")

        # Refresh tool cache
        await self.refresh_tools()

    async def _connect_server(self, name: str, config: Dict):
        """Connect to a single MCP server via Stdio"""
        command = config.get("command")
        args = config.get("args", [])
        env = config.get("env", {})
        
        # Merge current env with config env
        full_env = os.environ.copy()
        full_env.update(env)

        # Resolve command (e.g., 'npx', 'uvx', 'python')
        # If it's a python script in our tools folder, resolve path
        if command == "python" and args and args[0].endswith(".py"):
            script_path = Path(args[0])
            if not script_path.is_absolute():
                args[0] = str(Path.cwd() / args[0])

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=full_env
        )

        print(f"[MCP] Connecting to '{name}'...")
        
        # We use the exit stack to keep connections open until close() is called
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        
        session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        
        await session.initialize()
        self.sessions[name] = session
        print(f"[MCP] Connected to '{name}' ✓")

    async def refresh_tools(self):
        """Query all connected servers for their tools"""
        self.tools_cache = []
        self._tool_map = {}
        
        for server_name, session in self.sessions.items():
            try:
                result = await session.list_tools()
                for tool in result.tools:
                    # Store schema for LLM
                    tool_schema = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    }
                    self.tools_cache.append(tool_schema)
                    self._tool_map[tool.name] = server_name
                    
            except Exception as e:
                print(f"[MCP] Error listing tools for '{server_name}': {e}")
        
        if self.tools_cache:
            print(f"[MCP] Total Tools Available: {len(self.tools_cache)}")

    async def list_tools(self) -> List[Dict]:
        """Return cached tool schemas for LLM"""
        return self.tools_cache

    async def call_tool(self, name: str, arguments: Dict) -> str:
        """Execute a tool on the appropriate server"""
        server_name = self._tool_map.get(name)
        if not server_name:
            return f"Error: Tool '{name}' not found in any connected MCP server."
            
        session = self.sessions[server_name]
        try:
            result = await session.call_tool(name, arguments)
            
            # Format result text
            output = []
            if result.content:
                for item in result.content:
                    if isinstance(item, TextContent):
                        output.append(item.text)
                    elif isinstance(item, ImageContent):
                        output.append(f"[Image returned: {item.mimeType}]")
                    elif isinstance(item, EmbeddedResource):
                        output.append(f"[Resource: {item.resource.uri}]")
            
            final_text = "\n".join(output)
            if result.isError:
                return f"Tool Error: {final_text}"
            return final_text
            
        except Exception as e:
            return f"Execution Error: {str(e)}"

    async def cleanup(self):
        """Close all connections"""
        await self.exit_stack.aclose()

    def _create_default_config(self):
        """Create a sample config file if none exists"""
        default_config = {
            "mcpServers": {
                "filesystem": {
                    "command": "python",
                    "args": ["tools/mcp_server_local.py"],
                    "env": {}
                }
            }
        }
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"[MCP] Created default config at {self.config_path}")

# Global instance
MCP_CLIENT = MCPManager()