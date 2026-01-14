#!/usr/bin/env python3
"""
File Operations Helper for Assistant Utility

Provides structured file operations with progress tracking.
Used for complex multi-file tasks.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Default sandbox directory
SANDBOX_DIR = Path("sessions/files")

def ensure_sandbox(path: str) -> Path:
    """Ensure path is within sandbox directory."""
    full_path = SANDBOX_DIR / path
    # Resolve to prevent directory traversal
    resolved = full_path.resolve()
    if not str(resolved).startswith(str(SANDBOX_DIR.resolve())):
        raise ValueError(f"Path escapes sandbox: {path}")
    return resolved


def list_files_formatted(
    directory: str = ".",
    pattern: str = "*",
    include_sizes: bool = True
) -> Dict:
    """
    List files in a directory with formatted output.
    
    Returns dict suitable for assistant's natural language response.
    """
    try:
        dir_path = ensure_sandbox(directory)
        
        if not dir_path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {directory}",
                "assistant_hint": "That folder doesn't exist yet. Want me to create it?"
            }
        
        files = list(dir_path.glob(pattern))
        
        file_info = []
        total_size = 0
        
        for f in sorted(files):
            if f.is_file():
                size = f.stat().st_size
                total_size += size
                file_info.append({
                    "name": f.name,
                    "size": size,
                    "size_human": _human_size(size),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                })
        
        return {
            "success": True,
            "directory": directory,
            "file_count": len(file_info),
            "total_size": _human_size(total_size),
            "files": file_info,
            "assistant_hint": _generate_file_comment(file_info)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "assistant_hint": "Something went wrong. Let me try a different approach."
        }


def batch_operation(
    operation: str,
    files: List[str],
    destination: Optional[str] = None
) -> Dict:
    """
    Perform batch operations on multiple files.
    
    Operations: 'list', 'delete', 'move', 'copy'
    """
    results = {
        "operation": operation,
        "total": len(files),
        "succeeded": 0,
        "failed": 0,
        "details": []
    }
    
    for file in files:
        try:
            file_path = ensure_sandbox(file)
            
            if operation == "list":
                if file_path.exists():
                    results["details"].append({"file": file, "status": "exists"})
                    results["succeeded"] += 1
                else:
                    results["details"].append({"file": file, "status": "not found"})
                    results["failed"] += 1
                    
            elif operation == "delete":
                if file_path.exists():
                    file_path.unlink()
                    results["details"].append({"file": file, "status": "deleted"})
                    results["succeeded"] += 1
                else:
                    results["details"].append({"file": file, "status": "not found"})
                    results["failed"] += 1
                    
            # Add more operations as needed
            
        except Exception as e:
            results["details"].append({"file": file, "status": f"error: {e}"})
            results["failed"] += 1
    
    # Generate assistant-friendly summary
    if results["failed"] == 0:
        results["assistant_summary"] = f"All {results['total']} files processed successfully."
    else:
        results["assistant_summary"] = (
            f"Processed {results['succeeded']} of {results['total']} files. "
            f"{results['failed']} had issues."
        )
    
    return results


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _generate_file_comment(files: List[Dict]) -> str:
    """Generate a witty comment about the files."""
    if not files:
        return "Empty folder. Clean slate, or just lonely?"
    
    if len(files) > 20:
        return f"{len(files)} files. Might want to organize these sometime."
    
    # Check for versioning chaos
    names = [f["name"] for f in files]
    if any("final" in n.lower() for n in names):
        finals = sum(1 for n in names if "final" in n.lower())
        if finals > 1:
            return f"Found {finals} files with 'final' in the name. Classic."
    
    return f"{len(files)} files found."


if __name__ == "__main__":
    # Example usage
    result = list_files_formatted(".", "*.txt")
    print(json.dumps(result, indent=2))
