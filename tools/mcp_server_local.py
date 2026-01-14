"""
Local Filesystem MCP Server
Exposes the 'sessions/files' directory to the agent securely.
"""
from mcp.server.fastmcp import FastMCP
from pathlib import Path

# Initialize server
mcp = FastMCP("Local Filesystem")

# Define root directory for safety
ROOT_DIR = Path(__file__).parent.parent / "sessions" / "files"
ROOT_DIR.mkdir(parents=True, exist_ok=True)

@mcp.tool()
def read_file(filename: str) -> str:
    """Read a file from the session directory"""
    safe_path = (ROOT_DIR / filename).resolve()
    if not str(safe_path).startswith(str(ROOT_DIR.resolve())):
        return "Error: Access denied (outside sandbox)"
    
    if not safe_path.exists():
        return "Error: File not found"
        
    return safe_path.read_text(encoding='utf-8')

@mcp.tool()
def write_file(filename: str, content: str) -> str:
    """Write content to a file in the session directory"""
    safe_path = (ROOT_DIR / filename).resolve()
    if not str(safe_path).startswith(str(ROOT_DIR.resolve())):
        return "Error: Access denied (outside sandbox)"
    
    safe_path.write_text(content, encoding='utf-8')
    return f"Successfully wrote {len(content)} chars to {filename}"

@mcp.tool()
def list_files() -> str:
    """List all files in the session directory"""
    files = [f.name for f in ROOT_DIR.glob("*") if f.is_file()]
    return "\n".join(files) if files else "(No files found)"

if __name__ == "__main__":
    mcp.run()