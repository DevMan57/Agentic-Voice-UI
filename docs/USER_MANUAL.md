# IndexTTS2 Voice Agent - Complete Manual

> **Version:** 2.3.1 | **Platform:** Windows + WSL | **Ports:** 7861, 7863, 7864

A multi-character AI assistant with voice, persistent memory, tools, and full PC access capabilities.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Using as a PC Assistant](#using-as-a-pc-assistant)
4. [Memory System](#memory-system)
5. [Tools & Capabilities](#tools--capabilities)
6. [MCP Servers](#mcp-servers)
7. [Skills System](#skills-system)
8. [Characters](#characters)
9. [Voice Configuration](#voice-configuration)
10. [Configuration Reference](#configuration-reference)

---

## Quick Start

### Launching

Double-click `VoiceChat.bat`:

| Option | Description | Port |
|--------|-------------|------|
| [1] Voice Chat | Main conversation interface | 7861 |
| [2] Character Manager | Create/edit characters, memories, skills | 7863 |
| [3] MCP Server Manager | Configure external tool servers | 7864 |
| [4] Install Dependencies | Setup Python/Node packages | - |

### Basic Usage

1. Select a character from the dropdown
2. **Voice Input**: Hold **Right Shift** to talk (Push-to-Talk)
3. **Text Input**: Type in the message box and press Enter
4. Agent responds with voice + text

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      VoiceChat.bat                               │
│  [1] Voice Chat  [2] Character Manager  [3] MCP Manager         │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ voice_chat_   │    │ character_    │    │ mcp_manager_  │
│ app.py        │    │ manager_ui.py │    │ ui.py         │
│ Port: 7861    │    │ Port: 7863    │    │ Port: 7864    │
└───────┬───────┘    └───────────────┘    └───────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Core Systems                             │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Memory Manager │   Tool Registry │   MCP Client                │
│  - 4-layer      │   - 15+ tools   │   - External servers        │
│  - GraphRAG     │   - Sandboxed   │   - Dynamic discovery       │
│  - Embeddings   │   - Full access │   - Tool routing            │
└─────────────────┴─────────────────┴─────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Audio Pipeline                              │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  PTT/VAD Input  │   Whisper STT   │   TTS Output                │
│  (Right Shift)  │   (faster-      │   IndexTTS2 (clone) or      │
│  Silero VAD     │    whisper)     │   Kokoro (fast ONNX)        │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### File Structure

```
voice_chat/
├── voice_chat_app.py       # Main application (7861)
├── character_manager_ui.py # Character/Memory/Skills UI (7863)
├── mcp_manager_ui.py       # MCP server config UI (7864)
├── mcp_client.py           # MCP connection handler
├── streaming.py            # TTS sentence buffering
├── memory/
│   ├── memory_manager.py   # 4-layer memory + GraphRAG
│   ├── sqlite_storage.py   # SQLite backend + graph tables
│   ├── graph_rag.py        # Community detection, global queries
│   ├── graph_extractor.py  # Background entity extraction
│   └── characters.py       # Character loading/management
├── audio/
│   ├── ptt_windows.py      # Push-to-talk (keyboard lib)
│   ├── vad_recorder.py     # Voice activity detection
│   └── emotion_detector.py # wav2vec2 emotion analysis
├── tools/
│   └── __init__.py         # All built-in tools
├── skills/                 # Agent skill packages
│   ├── assistant/
│   ├── hermione/
│   └── lisbeth/
├── sessions/
│   ├── files/              # Sandboxed file access area
│   ├── audio_cache/        # TTS audio cache
│   ├── conversations/      # Per-character conversation history
│   └── memory.db           # SQLite Graph Database (Shared)
├── voices/                 # Voice reference WAV files
└── mcp_config.json         # MCP server configuration
```

---

## Using as a PC Assistant

### Enable Full PC Access

To let the agent read/write files anywhere on your PC:

1. Open Voice Chat (option 1)
2. In the sidebar, find **"Enable Full File Access"** checkbox
3. Check it - the agent now has `read_file_full` and `write_file_full` tools

### What the Agent Can Do

With tools enabled, the agent can:

| Capability | Tool | Example |
|------------|------|---------|
| **Read any file** | `read_file_full` | "Read my config at C:\Users\Me\.bashrc" |
| **Write any file** | `write_file_full` | "Create a script at C:\scripts\backup.py" |
| **Search the web** | `web_search` | "Search for Python async best practices" |
| **Fetch URLs** | `fetch_url` | "Get the content from docs.python.org" |
| **Wikipedia** | `wikipedia` | "Look up quantum computing on Wikipedia" |
| **Calculations** | `calculator` | "Calculate compound interest on $10,000 at 5% for 10 years" |
| **Weather** | `weather` | "What's the weather in Tokyo?" |
| **List files** | `list_files` | "List all Python files in sessions/files" |
| **Run Python** | `run_python` | "Run this Python code to process my data" |
| **Create skills** | `create_skill` | "Create a new skill for code review" |

### MCP Servers for Extended Capabilities

Add MCP servers for more powerful tools:

| Server | Capabilities |
|--------|-------------|
| `@anthropic/mcp-server-filesystem` | Full filesystem ops (read, write, search, move) |
| `@anthropic/mcp-server-github` | GitHub repos, issues, PRs |
| `@anthropic/mcp-server-puppeteer` | Browser automation, screenshots |
| `@anthropic/mcp-server-sqlite` | Database queries |
| `@anthropic/mcp-server-brave-search` | Web search (needs API key) |

To add: Open MCP Manager (option 3) → Server Catalog → Select and Add

### Example Workflows

**Research Task:**
```
"Search for the latest React 19 features, summarize them,
and save the summary to C:\Notes\react19.md"
```

**File Organization:**
```
"List all PDF files in my Downloads folder and tell me
what each one is about based on the filename"
```

**Code Generation:**
```
"Create a Python script that monitors a folder for new files
and moves them to categorized subfolders based on extension.
Save it to C:\scripts\file_organizer.py"
```

**System Administration:**
```
"Read my hosts file at C:\Windows\System32\drivers\etc\hosts
and explain what entries are there"
```

---

## Memory System

### 4-Layer Architecture

| Layer | Purpose | Decay | Example |
|-------|---------|-------|---------|
| **Episodic** | Conversation history | Yes | "Yesterday we discussed your project" |
| **Semantic** | Facts about user | No | "User is allergic to peanuts" |
| **Procedural** | How-to knowledge | No | "User prefers tabs over spaces" |
| **Knowledge Graph** | Entity relationships | No | "User → works_at → Acme Corp" |

### Automatic Features

1. **Fact Extraction**: Personal facts from conversations are auto-extracted and stored
2. **Contradiction Detection**: New facts are checked against existing memories
3. **Graph Extraction**: Entities and relationships are extracted in background
4. **Weighted Retrieval**: Memories scored by recency (0.2), relevance (0.5), importance (0.3)

### Memory Management UI

Access via Character Manager (option 2) → Memories tab:

- Browse by type (episodic/semantic/procedural)
- Edit memory content and importance
- Delete outdated memories
- Import/export memory backups

---

## Tools & Capabilities

### Built-in Tools

| Tool | Description | Access Level |
|------|-------------|--------------|
| `get_time` | Current time in any timezone | Always |
| `set_timer` | Set reminders | Always |
| `web_search` | DuckDuckGo search | Always |
| `wikipedia` | Wikipedia lookups | Always |
| `fetch_url` | Fetch webpage content | Always |
| `weather` | Weather via wttr.in | Always |
| `calculator` | Safe math evaluation | Always |
| `read_file` | Read files (sandboxed) | Always |
| `write_file` | Write files (sandboxed) | Always |
| `list_files` | List directory contents | Always |
| `read_file_full` | Read any file | Full Access |
| `write_file_full` | Write any file | Full Access |
| `run_python` | Execute Python code | Code Execution |
| `create_skill` | Create new agent skills | Always |

### Sandboxed vs Full Access

**Sandboxed** (default): Files limited to `sessions/files/` directory
- Safe for experimentation
- Agent can create/read files without risk

**Full Access** (opt-in): Agent can access entire filesystem
- Enable via checkbox in sidebar
- Required for PC assistant workflows
- Use with trusted characters/prompts

### Tool Usage in Conversation

Tools are automatically invoked based on need:

```
User: "What time is it in Tokyo?"
Agent: [Uses get_time tool] → "It's currently 3:45 PM in Tokyo"

User: "Search for Python async tutorials"
Agent: [Uses web_search tool] → "Here are the top results..."

User: "Save this code to my_script.py"
Agent: [Uses write_file tool] → "Created my_script.py with 45 lines"
```

---

## MCP Servers

### What is MCP?

Model Context Protocol (MCP) allows connecting to external tool servers. This extends the agent's capabilities beyond built-in tools.

### Configuration

MCP servers are configured in `mcp_config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-filesystem", "C:\\"],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token-here"
      }
    }
  }
}
```

### Managing MCP Servers

Use MCP Manager (option 3):

1. **My Servers** - View configured servers, test connections
2. **Server Catalog** - One-click add popular servers
3. **Add Custom** - Configure any MCP-compatible server
4. **Raw Config** - Edit JSON directly

### Auto-Connect

MCP servers automatically connect when Voice Chat starts. Tools from MCP servers are merged with built-in tools and available to the agent.

---

## Skills System

### What are Skills?

Skills are specialized capability packages that can be:
- Loaded automatically based on conversation context
- Created by users or by the agent itself
- Shared between characters

### Skill Structure

```
skills/
└── voice-tinkerer/
    ├── SKILL.md           # Main skill definition
    ├── references/        # Knowledge documents
    │   └── knowledge.md
    └── scripts/           # Helper Python scripts
        └── optimizer.py
```

### SKILL.md Format

```markdown
---
name: voice-tinkerer
description: TTS and speech emotion recognition optimizer
---

# Voice Tinkerer

You are a TTS optimization specialist...

## Configuration

allowed_tools:
  - web_search
  - read_file
  - write_file

## Core Knowledge

- IndexTTS2 architecture details
- RTF optimization techniques
- Emotion detection calibration
```

### Skill Context Detection

Skills are automatically loaded when keywords match:

```
User: "Help me optimize the TTS latency"
→ Detects "TTS", "optimize"
→ Loads voice-tinkerer skill context
→ Agent responds with specialized knowledge
```

### Creating Skills

**Via UI**: Character Manager → Skills tab → Create New Skill

**Via Agent**: Ask the agent to create one:
```
"Create a skill called 'code-reviewer' that specializes in
Python code review with access to read_file and web_search tools"
```

### Managing Skills

Character Manager → Skills tab:
- View installed skills
- Edit SKILL.md content
- Delete skills
- Import agent-created skills from `sessions/files/`

---

## Characters

### Pre-built Characters

| Character | Style | Best For |
|-----------|-------|----------|
| `assistant` | Witty, helpful | General tasks, coding, research |
| `hermione` | Studious, detailed | Learning, explanations |
| `lisbeth` | Terse, technical | Hacking, system admin |

### Character Components

Each character has:
- **System Prompt**: Personality and instructions
- **Voice**: Reference WAV for TTS
- **Memory**: Isolated memory database
- **Tools**: Allowed tool list
- **Skill**: Associated skill package

### Creating Characters

Character Manager → Characters tab:

1. Use **AI Wizard** for quick generation (describe personality)
2. Or manually fill:
   - ID (lowercase, no spaces)
   - Display Name
   - System Prompt
   - Voice selection
   - Tool permissions

### Character Memory Isolation

Each character has separate:
- Episodic memories (conversations)
- Semantic memories (facts)
- Knowledge graph (entities)
- Conversation history

Switching characters = switching memory context.

---

## Voice Configuration

### Push-to-Talk (Recommended)

- **Hold Right Shift** to record
- Release to send to agent
- Status shown in UI

### Voice Activity Detection (VAD)

Alternative to PTT - auto-detects speech:

| Setting | Default | Description |
|---------|---------|-------------|
| Backend | `silero` | Neural network VAD |
| Threshold | `0.6` | Confidence for speech detection |
| Consecutive Frames | `5` | 150ms of speech before recording |

### Emotion Detection

Agent detects emotion from your voice:
- Uses wav2vec2 model
- Shows in conversation context
- Agent can adapt responses

### Voice Output

- TTS via IndexTTS2
- Per-character voice references in `voices/`
- Sentence-by-sentence streaming

---

## Configuration Reference

### Main Config (in code)

```python
CONFIG = {
    "ENABLE_FULL_FILE_ACCESS": False,  # Toggle for filesystem access
    "ENABLE_CODE_EXECUTION": False,    # Toggle for Python execution
    "LM_STUDIO_ENDPOINT": "...",       # LLM endpoint
}
```

### LLM Providers

Supported in sidebar dropdown:
- **LM Studio** (local) - Auto-detected via WSL gateway
- **OpenRouter** - Cloud models (needs API key)
- **OpenAI** - GPT models (needs API key)

### Environment Variables

Create `.env` file or set:
```
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
BRAVE_API_KEY=...          # For Brave Search MCP
GITHUB_TOKEN=ghp_...       # For GitHub MCP
```

### Ports

| Service | Port | Purpose |
|---------|------|---------|
| Voice Chat | 7861 | Main conversation UI |
| Character Manager | 7863 | Character/Memory/Skills |
| MCP Manager | 7864 | MCP server config |
| LM Studio | 1235 | Local LLM server |

---

## Troubleshooting

### "Full file access not working"

1. Check the checkbox is enabled in sidebar
2. Verify tools list shows `read_file_full` / `write_file_full`
3. Check character has these tools allowed

### "Agent doesn't use tools"

1. Be explicit: "Use the web_search tool to find..."
2. Check character has tools enabled
3. Verify LLM supports function calling

### "MCP tools not appearing"

1. Check MCP Manager shows servers connected
2. Verify `mcp_config.json` is valid
3. Restart Voice Chat after config changes

### "Memory not persisting"

1. Don't use Incognito mode
2. Check `sessions/memory.db` exists
3. Verify character ID matches

---

## Security Considerations

### Full File Access

When enabled, the agent can:
- Read sensitive files (passwords, keys, configs)
- Overwrite system files
- Execute code that modifies your system

**Recommendations:**
- Only enable for trusted use cases
- Review agent actions in the UI
- Use sandboxed mode for experimentation

### MCP Servers

External MCP servers may:
- Have network access
- Execute code
- Access credentials in env vars

**Recommendations:**
- Only install from trusted sources
- Review server permissions
- Use dedicated API keys with limited scope

---

## Version History

- **2.3.0** - MCP auto-connect, contradiction checking, skill context detection
- **2.2.8** - Skills system, MCP Manager UI, memory CRUD
- **2.2.0** - GraphRAG, 4-layer memory, E5 embeddings
- **2.0.0** - Major architecture overhaul, agent tools
