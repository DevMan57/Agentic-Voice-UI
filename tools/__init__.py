"""
Enhanced Tools System for IndexTTS2 Voice Chat

Modular, extensible tools that characters can use:
- Time/Date queries
- Web search (DuckDuckGo, no API key required)
- Wikipedia lookups
- Calculator (safe math evaluation)
- File I/O (sandboxed and full access)
- Weather (via wttr.in, no API key)
- URL fetching
- Code execution (sandboxed Python)
- Reminder system

Tools are enabled per-character in the character definition.
"""

import os
import time
import json
import re
import ast
import operator
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import threading
import hashlib

# ============================================================================
# Base Tool Classes
# ============================================================================

class Tool(ABC):
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str, version: str = "1.0"):
        self.name = name
        self.description = description
        self.version = version
        self.call_count = 0
        self.last_called = None
    
    @abstractmethod
    def run(self, **kwargs) -> str:
        """Execute the tool and return result as string"""
        pass
    
    @property
    def schema(self) -> Dict[str, Any]:
        """Get OpenAI/OpenRouter compatible function schema"""
        raise NotImplementedError
    
    def _track_call(self):
        """Track tool usage"""
        self.call_count += 1
        self.last_called = datetime.now()


class ToolRegistry:
    """Registry for available tools with filtering and execution"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(self, tool: Tool, category: str = "general"):
        """Register a tool with optional category"""
        self._tools[tool.name] = tool
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(tool.name)
        print(f"[Tools] Registered: {tool.name} ({category})")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def list_tools(self, names: List[str] = None) -> List[Dict[str, Any]]:
        """List tool schemas, optionally filtered by names"""
        tools = self._tools.values()
        if names:
            tools = [t for t in tools if t.name in names]
        return [t.schema for t in tools]
    
    def list_by_category(self, category: str) -> List[str]:
        """List tool names in a category"""
        return self._categories.get(category, [])
    
    def execute(self, call_data: Dict[str, Any]) -> str:
        """Execute a tool call from LLM response"""
        try:
            # Handle OpenRouter/OpenAI format
            if isinstance(call_data, dict):
                func_name = call_data.get('function', {}).get('name')
                args_str = call_data.get('function', {}).get('arguments', '{}')
            else:
                return f"Error: Invalid tool call format: {call_data}"
            
            tool = self.get_tool(func_name)
            if not tool:
                return f"Error: Tool '{func_name}' not found. Available: {list(self._tools.keys())}"
            
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                return f"Error: Invalid JSON arguments for {func_name}: {args_str}"
            
            print(f"[Tools] Executing {func_name} with {args}")
            tool._track_call()
            
            result = tool.run(**args)
            
            # Truncate very long results
            if len(result) > 4000:
                result = result[:3900] + "\n\n[Result truncated - too long]"
            
            return result
            
        except Exception as e:
            import traceback
            return f"Error executing tool: {e}\n{traceback.format_exc()}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tools"""
        return {
            name: {
                'call_count': tool.call_count,
                'last_called': tool.last_called.isoformat() if tool.last_called else None
            }
            for name, tool in self._tools.items()
        }


# ============================================================================
# Time Tools
# ============================================================================

class TimeTool(Tool):
    """Get current date and time"""
    
    def __init__(self):
        super().__init__(
            "get_current_time",
            "Get the current date, time, and day of the week in the local timezone."
        )
    
    def run(self, format: str = None, **kwargs) -> str:
        now = datetime.now()
        if format:
            try:
                return now.strftime(format)
            except:
                pass
        return now.strftime("%Y-%m-%d %H:%M:%S %A")
    
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "description": "Optional strftime format string"
                        }
                    },
                    "required": []
                }
            }
        }


class TimerTool(Tool):
    """Set reminders/timers"""
    
    def __init__(self, callback: Callable[[str], None] = None):
        super().__init__(
            "set_reminder",
            "Set a reminder for a specified time in the future."
        )
        self.callback = callback or self._default_callback
        self._reminders: Dict[str, Dict] = {}
        self._timer_thread = None
    
    def _default_callback(self, message: str):
        print(f"[Reminder] {message}")
    
    def run(self, message: str, minutes: int = None, time_str: str = None, **kwargs) -> str:
        reminder_id = hashlib.md5(f"{message}{datetime.now()}".encode()).hexdigest()[:8]
        
        if minutes:
            trigger_time = datetime.now() + timedelta(minutes=minutes)
        elif time_str:
            try:
                # Parse time like "14:30" or "2:30 PM"
                for fmt in ["%H:%M", "%I:%M %p", "%I:%M%p"]:
                    try:
                        parsed = datetime.strptime(time_str.strip(), fmt)
                        trigger_time = datetime.now().replace(
                            hour=parsed.hour, minute=parsed.minute, second=0
                        )
                        if trigger_time < datetime.now():
                            trigger_time += timedelta(days=1)
                        break
                    except:
                        continue
                else:
                    return f"Error: Could not parse time '{time_str}'. Use format like '14:30' or '2:30 PM'"
            except Exception as e:
                return f"Error parsing time: {e}"
        else:
            return "Error: Must specify either 'minutes' or 'time_str'"
        
        self._reminders[reminder_id] = {
            'message': message,
            'trigger_time': trigger_time,
            'created': datetime.now()
        }
        
        # Start timer thread
        def _timer():
            import time as time_module
            while True:
                time_module.sleep(30)  # Check every 30 seconds
                now = datetime.now()
                triggered = []
                for rid, reminder in self._reminders.items():
                    if now >= reminder['trigger_time']:
                        self.callback(f"⏰ Reminder: {reminder['message']}")
                        triggered.append(rid)
                for rid in triggered:
                    del self._reminders[rid]
        
        if not self._timer_thread or not self._timer_thread.is_alive():
            self._timer_thread = threading.Thread(target=_timer, daemon=True)
            self._timer_thread.start()
        
        return f"✓ Reminder set for {trigger_time.strftime('%Y-%m-%d %H:%M')} (ID: {reminder_id}): {message}"
    
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The reminder message"
                        },
                        "minutes": {
                            "type": "integer",
                            "description": "Minutes from now to trigger the reminder"
                        },
                        "time_str": {
                            "type": "string",
                            "description": "Specific time to trigger (e.g., '14:30' or '2:30 PM')"
                        }
                    },
                    "required": ["message"]
                }
            }
        }


# ============================================================================
# Search Tools
# ============================================================================

class WebSearchTool(Tool):
    """Search the web using DuckDuckGo (no API key required)"""
    
    def __init__(self):
        super().__init__(
            "web_search",
            "Search the web using DuckDuckGo. Returns titles, URLs, and snippets. Use for current events, facts, or any information lookup."
        )
        self.available = False
        try:
            from ddgs import DDGS
            self.ddgs = DDGS()
            self.available = True
        except ImportError:
            try:
                from duckduckgo_search import DDGS
                self.ddgs = DDGS()
                self.available = True
            except ImportError:
                # Silently fail, main app will handle feature list
                self.available = False
    
    def run(self, query: str, max_results: int = 5, **kwargs) -> str:
        if not self.available:
            return "Error: Web search is not available. Install: pip install ddgs"
        
        try:
            results = list(self.ddgs.text(query, max_results=min(max_results, 10)))
            if not results:
                return f"No results found for: {query}"
            
            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"{i}. **{r.get('title', 'No title')}**\n"
                    f"   URL: {r.get('href', r.get('link', 'N/A'))}\n"
                    f"   {r.get('body', r.get('snippet', 'No description'))}"
                )
            
            return f"Search results for '{query}':\n\n" + "\n\n".join(formatted)
            
        except Exception as e:
            return f"Search error: {e}"
    
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (1-10)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }


class WikipediaTool(Tool):
    """Search and retrieve Wikipedia articles"""
    
    def __init__(self):
        super().__init__(
            "wikipedia",
            "Search Wikipedia and get article summaries. Great for factual information, biographies, concepts, and encyclopedic knowledge."
        )
        self.available = False
        try:
            import wikipedia
            self.wikipedia = wikipedia
            self.available = True
        except ImportError:
            # Silently fail - we check availability before registering
            pass
    
    def run(self, query: str, sentences: int = 5, **kwargs) -> str:
        if not self.available:
            return "Error: Wikipedia is not available. Install: pip install wikipedia"
        
        try:
            # Search for matching articles
            search_results = self.wikipedia.search(query, results=3)
            
            if not search_results:
                return f"No Wikipedia articles found for: {query}"
            
            # Try to get summary for the first result
            try:
                summary = self.wikipedia.summary(search_results[0], sentences=sentences)
                page = self.wikipedia.page(search_results[0])
                
                return (
                    f"**{page.title}**\n\n"
                    f"{summary}\n\n"
                    f"URL: {page.url}\n\n"
                    f"Related topics: {', '.join(search_results[1:]) if len(search_results) > 1 else 'None'}"
                )
            except self.wikipedia.DisambiguationError as e:
                # Multiple possible articles
                options = e.options[:5]
                return f"Multiple articles found. Please be more specific:\n- " + "\n- ".join(options)
            except self.wikipedia.PageError:
                return f"No Wikipedia article found for: {query}"
                
        except Exception as e:
            return f"Wikipedia error: {e}"
    
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Topic to search for"
                        },
                        "sentences": {
                            "type": "integer",
                            "description": "Number of sentences in the summary (default 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }


class UrlFetchTool(Tool):
    """Fetch and extract text from a URL"""
    
    def __init__(self):
        super().__init__(
            "fetch_url",
            "Fetch a webpage and extract its text content. Useful for reading articles, documentation, or web pages."
        )
    
    def run(self, url: str, max_length: int = 3000, **kwargs) -> str:
        try:
            import requests
            from html import unescape
            import re
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            html = response.text
            
            # Try to use BeautifulSoup if available
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()
                
                # Get text
                text = soup.get_text(separator='\n', strip=True)
                
            except ImportError:
                # Fallback: basic HTML stripping
                text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
                text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = unescape(text)
            
            # Clean up whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r' +', ' ', text)
            text = text.strip()
            
            if len(text) > max_length:
                text = text[:max_length] + "\n\n[Content truncated...]"
            
            return f"Content from {url}:\n\n{text}"
            
        except Exception as e:
            return f"Error fetching URL: {e}"
    
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch"
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum characters to return (default 3000)",
                            "default": 3000
                        }
                    },
                    "required": ["url"]
                }
            }
        }


# ============================================================================
# Weather Tool
# ============================================================================

class WeatherTool(Tool):
    """Get weather using wttr.in (no API key required)"""
    
    def __init__(self):
        super().__init__(
            "get_weather",
            "Get current weather and forecast for a location. No API key required."
        )
    
    def run(self, location: str, format: str = "short", **kwargs) -> str:
        try:
            import requests
            import time

            # Format options
            if format == "full":
                url = f"https://wttr.in/{location}?format=4"
            elif format == "detailed":
                url = f"https://wttr.in/{location}?format=%l:+%c+%t+%h+%w+%p"
            else:  # short
                url = f"https://wttr.in/{location}?format=%l:+%c+%t"

            print(f"[Weather] Fetching: {url}")

            # Retry up to 2 times on connection errors
            for attempt in range(2):
                try:
                    response = requests.get(
                        url,
                        timeout=10,
                        headers={'User-Agent': 'curl/7.68.0'},
                        verify=True  # SSL verification
                    )
                    response.raise_for_status()
                    break
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    if attempt == 0:
                        print(f"[Weather] Connection failed, retrying... ({e})")
                        time.sleep(1)
                    else:
                        raise

            weather = response.text.strip()
            print(f"[Weather] Result: {weather}")

            if "Unknown location" in weather or "not found" in weather.lower():
                return f"Could not find weather for: {location}"

            return weather

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Cannot reach weather service - check your internet connection"
            print(f"[Weather] Connection error: {e}")
            return error_msg
        except requests.exceptions.Timeout:
            error_msg = f"Weather service timed out - try again in a moment"
            print(f"[Weather] Timeout")
            return error_msg
        except Exception as e:
            error_msg = f"Weather service error: {str(e)[:50]}"
            print(f"[Weather] {error_msg}")
            return error_msg
    
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or location (e.g., 'London', 'New York', 'Tokyo')"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["short", "detailed", "full"],
                            "description": "Level of detail (default: short)",
                            "default": "short"
                        }
                    },
                    "required": ["location"]
                }
            }
        }


# ============================================================================
# Calculator Tool
# ============================================================================

class CalculatorTool(Tool):
    """Safe mathematical expression evaluator"""
    
    # Allowed operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Allowed math functions
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'floor': math.floor,
        'ceil': math.ceil,
        'pi': math.pi,
        'e': math.e,
    }
    
    def __init__(self):
        super().__init__(
            "calculate",
            "Evaluate mathematical expressions safely. Supports basic arithmetic, powers, and common math functions (sqrt, sin, cos, log, etc.)."
        )
    
    def _eval(self, node):
        """Safely evaluate an AST node"""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(operand)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in self.FUNCTIONS:
                    args = [self._eval(arg) for arg in node.args]
                    return self.FUNCTIONS[func_name](*args)
            raise ValueError(f"Unsupported function")
        elif isinstance(node, ast.Name):
            if node.id in self.FUNCTIONS:
                return self.FUNCTIONS[node.id]
            raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.List):
            return [self._eval(elem) for elem in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._eval(elem) for elem in node.elts)
        else:
            raise ValueError(f"Unsupported expression: {type(node).__name__}")
    
    def run(self, expression: str, **kwargs) -> str:
        try:
            # Parse and evaluate safely
            tree = ast.parse(expression, mode='eval')
            result = self._eval(tree.body)
            
            # Format result nicely
            if isinstance(result, float):
                if result.is_integer():
                    return str(int(result))
                return f"{result:.10g}"  # Remove trailing zeros
            return str(result)
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: {e}"
    
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }


# ============================================================================
# File Tools
# ============================================================================

class FileReadTool(Tool):
    """Read files from sandboxed directory"""
    
    def __init__(self, sandbox_dir: Path):
        super().__init__(
            "read_file",
            "Read the contents of a file from your personal storage area."
        )
        self.sandbox = Path(sandbox_dir).resolve()
        self.sandbox.mkdir(parents=True, exist_ok=True)
    
    def run(self, filename: str, **kwargs) -> str:
        # Security: Prevent directory traversal
        safe_path = (self.sandbox / filename).resolve()
        if not str(safe_path).startswith(str(self.sandbox)):
            return "Error: Access denied. You can only read files in your sandbox."
        
        if not safe_path.exists():
            # List available files
            files = list(self.sandbox.glob("*"))
            file_list = ", ".join(f.name for f in files[:10]) if files else "none"
            return f"Error: File '{filename}' not found. Available files: {file_list}"
        
        try:
            content = safe_path.read_text(encoding='utf-8')
            if len(content) > 10000:
                content = content[:10000] + "\n\n[File truncated - too long]"
            return content
        except Exception as e:
            return f"Error reading file: {e}"
    
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to read"
                        }
                    },
                    "required": ["filename"]
                }
            }
        }


class FileWriteTool(Tool):
    """Write files to sandboxed directory"""
    
    def __init__(self, sandbox_dir: Path):
        super().__init__(
            "write_file",
            "Create or overwrite a file in your personal storage area."
        )
        self.sandbox = Path(sandbox_dir).resolve()
        self.sandbox.mkdir(parents=True, exist_ok=True)
    
    def run(self, filename: str, content: str, **kwargs) -> str:
        # Security: Prevent directory traversal
        safe_path = (self.sandbox / filename).resolve()
        if not str(safe_path).startswith(str(self.sandbox)):
            return "Error: Access denied. You can only write files in your sandbox."
        
        try:
            safe_path.write_text(content, encoding='utf-8')
            return f"✓ Successfully wrote {len(content)} characters to '{filename}'."
        except Exception as e:
            return f"Error writing file: {e}"
    
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to create/overwrite"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        }
                    },
                    "required": ["filename", "content"]
                }
            }
        }


class FileListTool(Tool):
    """List files in sandboxed directory"""
    
    def __init__(self, sandbox_dir: Path):
        super().__init__(
            "list_files",
            "List all files in your personal storage area."
        )
        self.sandbox = Path(sandbox_dir).resolve()
        self.sandbox.mkdir(parents=True, exist_ok=True)
    
    def run(self, **kwargs) -> str:
        try:
            files = list(self.sandbox.iterdir())
            if not files:
                return "Your storage area is empty."
            
            file_info = []
            for f in sorted(files):
                if f.is_file():
                    size = f.stat().st_size
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size/1024:.1f} KB"
                    else:
                        size_str = f"{size/1024/1024:.1f} MB"
                    file_info.append(f"  📄 {f.name} ({size_str})")
                else:
                    file_info.append(f"  📁 {f.name}/")
            
            return f"Files in your storage:\n" + "\n".join(file_info)
        except Exception as e:
            return f"Error listing files: {e}"
    
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }


class FileReadFullTool(Tool):
    """Read any file from filesystem (unrestricted)"""
    
    def __init__(self):
        super().__init__(
            "read_file_full",
            "Read any file from the filesystem using a full path. Use with caution."
        )
    
    def run(self, filepath: str, **kwargs) -> str:
        try:
            path = Path(filepath).expanduser().resolve()
            
            if not path.exists():
                return f"Error: File '{filepath}' not found."
            
            if path.is_dir():
                # List directory contents
                items = list(path.iterdir())[:50]
                return "Directory contents:\n" + "\n".join(f"  {f.name}" for f in items)
            
            content = path.read_text(encoding='utf-8')
            if len(content) > 10000:
                content = content[:10000] + "\n\n[File truncated]"
            return content
            
        except Exception as e:
            return f"Error reading file: {e}"
    
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Full path to the file (e.g., /home/user/doc.txt or C:/Users/doc.txt)"
                        }
                    },
                    "required": ["filepath"]
                }
            }
        }


class FileWriteFullTool(Tool):
    """Write to any file on filesystem (unrestricted)"""
    
    def __init__(self):
        super().__init__(
            "write_file_full",
            "Write to any file on the filesystem using a full path. Use with caution."
        )
    
    def run(self, filepath: str, content: str, **kwargs) -> str:
        try:
            path = Path(filepath).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            return f"✓ Successfully wrote to '{filepath}'."
        except Exception as e:
            return f"Error writing file: {e}"
    
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Full path to the file"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write"
                        }
                    },
                    "required": ["filepath", "content"]
                }
            }
        }


# ============================================================================
# Code Execution Tool
# ============================================================================

class PythonExecuteTool(Tool):
    """Execute Python code in a sandboxed environment"""
    
    def __init__(self, timeout: int = 10):
        super().__init__(
            "run_python",
            "Execute Python code and return the output. Code runs in a restricted sandbox."
        )
        self.timeout = timeout
    
    def run(self, code: str, **kwargs) -> str:
        import subprocess
        import tempfile
        
        # Create temp file with the code
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            # Run with timeout
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            
            if not output.strip():
                output = "(No output)"
            
            return output[:5000]  # Limit output size
            
        except subprocess.TimeoutExpired:
            return f"Error: Code execution timed out after {self.timeout} seconds"
        except Exception as e:
            return f"Error executing code: {e}"
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute"
                        }
                    },
                    "required": ["code"]
                }
            }
        }


class BashCommandTool(Tool):
    """Execute bash or PowerShell commands for file operations and system tasks"""
    
    # Commands that are explicitly blocked for safety
    BLOCKED_COMMANDS = [
        'rm -rf /',
        'mkfs',
        'dd if=',
        ':(){:|:&};:',  # Fork bomb
        '> /dev/sda',
        'chmod -R 777 /',
        'chown -R',
        'format c:',
        'del /f /s /q c:',
    ]
    
    # Dangerous patterns to warn about but allow
    DANGEROUS_PATTERNS = [
        'rm -rf',
        'sudo',
        'apt',
        'pip install',
        'npm install -g',
        'Remove-Item -Recurse',
    ]
    
    def __init__(self, timeout: int = 30, working_dir: Path = None):
        super().__init__(
            "run_command",
            "Execute a shell command. Use for file operations like mkdir, ls, mv, cp, cat, find, grep. "
            "Supports both bash (default, runs in WSL) and powershell (for Windows-native commands). "
            "Windows paths are accessible via /mnt/c/ in bash. "
            "Commands run in the project directory by default."
        )
        self.timeout = timeout
        self.working_dir = working_dir
    
    def run(self, command: str, working_directory: str = None, shell: str = "bash", **kwargs) -> str:
        import subprocess
        
        shell = shell.lower().strip()
        if shell not in ("bash", "powershell", "pwsh"):
            return f"Error: Invalid shell '{shell}'. Use 'bash' or 'powershell'."
        
        # Safety check - block dangerous commands
        cmd_lower = command.lower().strip()
        for blocked in self.BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return f"Error: Command blocked for safety reasons. '{blocked}' is not allowed."
        
        # Warning for potentially dangerous commands (but allow them)
        warnings = []
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.lower() in cmd_lower:
                warnings.append(f"⚠️ Using '{pattern}' - be careful!")
        
        # Determine working directory
        cwd = None
        if working_directory:
            cwd = Path(working_directory).expanduser()
            if not cwd.exists():
                return f"Error: Working directory '{working_directory}' does not exist."
        elif self.working_dir:
            cwd = self.working_dir
        
        try:
            # Build command based on shell type
            if shell == "bash":
                cmd_args = ['bash', '-c', command]
            else:
                # PowerShell - run via powershell.exe from WSL
                cmd_args = ['powershell.exe', '-NoProfile', '-Command', command]
            
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(cwd) if cwd else None
            )
            
            output_parts = []
            
            # Add warnings if any
            if warnings:
                output_parts.append("\n".join(warnings))
            
            # Add stdout
            if result.stdout.strip():
                output_parts.append(result.stdout.strip())
            
            # Add stderr if present
            if result.stderr.strip():
                output_parts.append(f"[stderr]: {result.stderr.strip()}")
            
            # Add return code info if non-zero
            if result.returncode != 0:
                output_parts.append(f"[exit code: {result.returncode}]")
            
            output = "\n".join(output_parts) if output_parts else "(Command completed with no output)"
            
            # Limit output size
            if len(output) > 8000:
                output = output[:8000] + "\n\n[Output truncated - too long]"
            
            return output
            
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {self.timeout} seconds"
        except FileNotFoundError as e:
            if shell == "bash":
                return "Error: bash not found. Are you running in WSL/Linux?"
            else:
                return "Error: powershell.exe not found."
        except Exception as e:
            return f"Error executing command: {e}"
    
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute (e.g., 'ls -la', 'mkdir new_folder', 'cat file.txt')"
                        },
                        "working_directory": {
                            "type": "string",
                            "description": "Optional directory to run the command in. Defaults to project directory."
                        },
                        "shell": {
                            "type": "string",
                            "enum": ["bash", "powershell"],
                            "description": "Shell to use: 'bash' (default, WSL) or 'powershell' (Windows). Windows paths in bash use /mnt/c/ format."
                        }
                    },
                    "required": ["command"]
                }
            }
        }


# ============================================================================
# Skill Creation Tool
# ============================================================================

class SkillCreatorTool(Tool):
    """Create new skills that extend agent capabilities"""

    def __init__(self, skills_dir: Path = None):
        super().__init__(
            "create_skill",
            "Create a new skill package with SKILL.md, references, and scripts directories. "
            "Skills define specialized capabilities that can be loaded for specific tasks."
        )
        self.skills_dir = skills_dir or Path(__file__).parent.parent / "skills"

    def run(self, skill_id: str, display_name: str, description: str,
            system_prompt: str, allowed_tools: str = "", **kwargs) -> str:
        import re

        if not skill_id or not skill_id.strip():
            return "Error: skill_id is required"

        # Sanitize skill_id
        skill_id = re.sub(r'[^\w-]', '-', skill_id.strip().lower())

        skill_dir = self.skills_dir / skill_id
        if skill_dir.exists():
            return f"Error: Skill '{skill_id}' already exists at {skill_dir}"

        try:
            # Create directory structure
            skill_dir.mkdir(parents=True)
            (skill_dir / "references").mkdir()
            (skill_dir / "scripts").mkdir()

            # Parse allowed_tools (comma-separated string)
            tools_list = [t.strip() for t in allowed_tools.split(",") if t.strip()]
            tools_yaml = ""
            if tools_list:
                tools_yaml = "\n  - " + "\n  - ".join(tools_list)

            # Create SKILL.md
            skill_content = f"""---
name: {skill_id}
description: {description or 'Custom agent skill'}
---

# {display_name or skill_id.replace('-', ' ').title()}

{system_prompt or 'Add skill instructions here...'}

## Configuration

allowed_tools:{tools_yaml if tools_yaml else ' []'}

## Knowledge

Add reference materials to `references/` directory.

## Scripts

Add Python helper scripts to `scripts/` directory.
"""

            (skill_dir / "SKILL.md").write_text(skill_content, encoding='utf-8')

            return f"Success: Skill '{skill_id}' created at skills/{skill_id}/\n" \
                   f"- SKILL.md created\n" \
                   f"- references/ directory created\n" \
                   f"- scripts/ directory created\n" \
                   f"You can now add reference materials and scripts to extend the skill."

        except Exception as e:
            return f"Error creating skill: {e}"

    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_id": {
                            "type": "string",
                            "description": "Unique identifier for the skill (lowercase, hyphens allowed)"
                        },
                        "display_name": {
                            "type": "string",
                            "description": "Human-readable name with optional emoji (e.g., '🔧 Voice Tinkerer')"
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of what the skill does"
                        },
                        "system_prompt": {
                            "type": "string",
                            "description": "The main instructions/personality for this skill"
                        },
                        "allowed_tools": {
                            "type": "string",
                            "description": "Comma-separated list of tools this skill can use (e.g., 'web_search,calculator,read_file')"
                        }
                    },
                    "required": ["skill_id", "display_name", "description", "system_prompt"]
                }
            }
        }


# ============================================================================
# Everything Search Tool (Windows file search via Everything)
# ============================================================================

class EverythingSearchTool(Tool):
    """Search files across the entire PC using Everything (Windows)"""

    def __init__(self):
        super().__init__(
            "everything_search",
            "Search for files anywhere on the PC using Everything search. "
            "Finds files by name, extension, or path patterns. Very fast indexed search."
        )
        self.available = self._check_available()

    def _check_available(self) -> bool:
        """Check if Everything CLI (es.exe) is available"""
        import subprocess
        try:
            # Try to run es.exe via powershell from WSL
            result = subprocess.run(
                ["powershell.exe", "-Command", "es.exe -version"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def run(self, query: str, max_results: int = 20, **kwargs) -> str:
        if not self.available:
            return "Error: Everything search not available. Install Everything (voidtools.com) and add es.exe to PATH."

        import subprocess
        try:
            # Run Everything CLI from WSL via powershell
            cmd = f'es.exe -n {max_results} "{query}"'
            result = subprocess.run(
                ["powershell.exe", "-Command", cmd],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                return f"Search error: {result.stderr}"

            output = result.stdout.strip()
            if not output:
                return f"No files found matching: {query}"

            files = output.split('\n')
            return f"Found {len(files)} files:\n" + "\n".join(files[:max_results])

        except subprocess.TimeoutExpired:
            return "Search timed out. Try a more specific query."
        except Exception as e:
            return f"Search error: {e}"

    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query - filename, extension (*.pdf), or path pattern"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results to return (default 20)",
                            "default": 20
                        }
                    },
                    "required": ["query"]
                }
            }
        }


# ============================================================================
# Knowledge Graph Tool
# ============================================================================

class GraphManagementTool(Tool):
    """Tool for managing the character's knowledge graph"""

    def __init__(self, memory_manager=None):
        super().__init__(
            "manage_knowledge_graph",
            "View and manage your knowledge graph - see entities, relationships, and refresh communities."
        )
        self._memory_manager = memory_manager

    def set_memory_manager(self, manager):
        """Set memory manager reference (called after init)"""
        self._memory_manager = manager

    def run(self, action: str, character_id: str = "assistant", entity_name: str = None, **kwargs) -> str:
        if not self._memory_manager:
            return "Error: Memory manager not initialized"

        try:
            # Use the storage property directly (not _get_storage method)
            storage = self._memory_manager.storage
            if not storage:
                return f"Error: No storage found"

            if action == "stats":
                # Get graph statistics using correct method names
                entities = storage.get_all_entities(character_id) or []
                relationships = storage.get_all_relationships(character_id) or []
                communities = storage.get_communities(character_id) or []

                # Count by type
                entity_types = {}
                for e in entities:
                    t = e.get('entity_type', 'unknown')
                    entity_types[t] = entity_types.get(t, 0) + 1

                rel_types = {}
                for r in relationships:
                    t = r.get('relation_type', 'unknown')
                    rel_types[t] = rel_types.get(t, 0) + 1

                result = f"**Knowledge Graph Stats for {character_id}**\n\n"
                result += f"**Entities:** {len(entities)}\n"
                for t, c in sorted(entity_types.items()):
                    result += f"  - {t}: {c}\n"
                result += f"\n**Relationships:** {len(relationships)}\n"
                for t, c in sorted(rel_types.items()):
                    result += f"  - {t}: {c}\n"
                result += f"\n**Communities:** {len(communities)}\n"

                return result

            elif action == "list_entities":
                entities = storage.get_all_entities(character_id) or []
                if not entities:
                    return "No entities found in knowledge graph."

                # Limit to 50 for display
                display_entities = entities[:50]
                result = f"**Entities** ({len(display_entities)} of {len(entities)} shown):\n\n"
                for e in display_entities:
                    name = e.get('name', 'Unknown')
                    etype = e.get('entity_type', 'unknown')
                    desc = (e.get('description', '') or '')[:60]
                    result += f"- **{name}** ({etype})"
                    if desc:
                        result += f": {desc}..."
                    result += "\n"
                return result

            elif action == "list_relationships":
                relationships = storage.get_all_relationships(character_id) or []
                if not relationships:
                    return "No relationships found in knowledge graph."

                # Limit to 50 for display
                display_rels = relationships[:50]
                result = f"**Relationships** ({len(display_rels)} of {len(relationships)} shown):\n\n"
                for r in display_rels:
                    src = r.get('source_entity', '?')
                    tgt = r.get('target_entity', '?')
                    rel = r.get('relation_type', 'related_to')
                    result += f"- {src} --[{rel}]--> {tgt}\n"
                return result

            elif action == "refresh_communities":
                if hasattr(self._memory_manager, 'graph_rag') and self._memory_manager.graph_rag:
                    # Trigger community update
                    self._memory_manager.graph_rag.update_communities(character_id)
                    return "Communities refreshed. New thematic clusters have been detected."
                else:
                    return "GraphRAG processor not available."

            elif action == "find_entity":
                if not entity_name:
                    return "Error: Please provide entity_name to search for."

                entities = storage.get_all_entities(character_id) or []
                matches = [e for e in entities if entity_name.lower() in e.get('name', '').lower()]

                if not matches:
                    return f"No entities found matching '{entity_name}'"

                result = f"**Entities matching '{entity_name}':**\n\n"
                for e in matches[:10]:
                    name = e.get('name', 'Unknown')
                    etype = e.get('entity_type', 'unknown')
                    desc = e.get('description', 'No description')
                    result += f"**{name}** ({etype})\n{desc}\n\n"
                return result

            else:
                return f"Unknown action '{action}'. Available: stats, list_entities, list_relationships, refresh_communities, find_entity"

        except Exception as e:
            return f"Error: {str(e)}"

    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["stats", "list_entities", "list_relationships", "refresh_communities", "find_entity"],
                            "description": "Action to perform: stats (overview), list_entities, list_relationships, refresh_communities, find_entity"
                        },
                        "character_id": {
                            "type": "string",
                            "description": "Character ID (default: assistant)"
                        },
                        "entity_name": {
                            "type": "string",
                            "description": "Entity name to search for (only for find_entity action)"
                        }
                    },
                    "required": ["action"]
                }
            }
        }


# ============================================================================
# Global Registry & Initialization
# ============================================================================

REGISTRY = ToolRegistry()
GRAPH_TOOL = None  # Will be set after memory manager is available


def init_tools(
    sandbox_path: str = "./sessions/files",
    enable_full_file_access: bool = False,
    enable_code_execution: bool = False,
    reminder_callback: Callable[[str], None] = None
) -> ToolRegistry:
    """
    Initialize the tool registry with available tools.
    
    Args:
        sandbox_path: Directory for sandboxed file tools
        enable_full_file_access: If True, register unrestricted file tools
        enable_code_execution: If True, register Python execution tool
        reminder_callback: Function to call when reminders trigger
    
    Returns:
        Configured ToolRegistry
    """
    sandbox = Path(sandbox_path)
    
    # Time tools
    REGISTRY.register(TimeTool(), "time")
    REGISTRY.register(TimerTool(reminder_callback), "time")
    
    # Search tools
    REGISTRY.register(WebSearchTool(), "search")
    
    # Only register Wikipedia if the package is installed
    wiki_tool = WikipediaTool()
    if wiki_tool.available:
        REGISTRY.register(wiki_tool, "search")
    else:
        print("[Tools] Wikipedia tool disabled (pip install wikipedia to enable)")
    
    REGISTRY.register(UrlFetchTool(), "search")
    
    # Utility tools
    REGISTRY.register(WeatherTool(), "utility")
    REGISTRY.register(CalculatorTool(), "utility")

    # Skill creation tool
    REGISTRY.register(SkillCreatorTool(), "skills")

    # Everything search (Windows PC-wide file search)
    everything_tool = EverythingSearchTool()
    if everything_tool.available:
        REGISTRY.register(everything_tool, "search")
        print("[Tools] Everything search enabled (PC-wide file search)")
    else:
        print("[Tools] Everything search disabled (install Everything + es.exe)")

    # Sandboxed file tools
    REGISTRY.register(FileReadTool(sandbox), "files")
    REGISTRY.register(FileWriteTool(sandbox), "files")
    REGISTRY.register(FileListTool(sandbox), "files")
    
    # Optional: Full file access
    if enable_full_file_access:
        REGISTRY.register(FileReadFullTool(), "files_full")
        REGISTRY.register(FileWriteFullTool(), "files_full")
        print("[Tools] WARNING: Full filesystem access enabled!")
    
    # Optional: Code execution
    if enable_code_execution:
        REGISTRY.register(PythonExecuteTool(), "code")
        print("[Tools] WARNING: Python code execution enabled!")

    # Shell command tool (always available, has safety blocks)
    REGISTRY.register(BashCommandTool(working_dir=sandbox.parent.parent), "shell")
    print("[Tools] Shell command tool registered (bash + powershell)")

    # Knowledge Graph tool (memory manager set later via set_graph_tool_memory_manager)
    global GRAPH_TOOL
    GRAPH_TOOL = GraphManagementTool()
    REGISTRY.register(GRAPH_TOOL, "memory")

    print(f"[Tools] Initialized {len(REGISTRY._tools)} tools")
    return REGISTRY


def set_graph_tool_memory_manager(memory_manager):
    """Set the memory manager for the graph tool after initialization."""
    global GRAPH_TOOL
    if GRAPH_TOOL:
        GRAPH_TOOL.set_memory_manager(memory_manager)
        print("[Tools] Graph tool connected to memory manager")


def get_tools_for_character(character_id: str, allowed_tools: List[str]) -> List[Dict[str, Any]]:
    """Get tool schemas for a specific character based on their allowed_tools list"""
    if not allowed_tools:
        return []
    return REGISTRY.list_tools(allowed_tools)
