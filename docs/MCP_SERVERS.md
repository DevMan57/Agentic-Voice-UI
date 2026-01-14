# MCP Server Guide

> Curated MCP servers for IndexTTS2 Voice Agent with installation instructions.

## Quick Links

| Resource | URL |
|----------|-----|
| Official MCP Servers | https://github.com/modelcontextprotocol/servers |
| Awesome MCP Servers | https://github.com/punkpeye/awesome-mcp-servers |
| MCP Server Directory | https://mcpservers.org/ |
| MCP Registry | https://mcp-awesome.com/ |

---

## Your Setup

- **Platform:** Windows 11 + WSL2 (Ubuntu)
- **Voice Agent runs in:** WSL
- **MCP config:** `voice_chat/mcp_config.json`
- **Node.js required:** Yes (for npx servers)

---

## Prerequisites

### 1. Install Node.js on WSL

```bash
# In WSL terminal
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version  # Should be v20+
```

### 2. Verify npx works

```bash
npx --version
```

---

## Recommended Servers

### Tier 1: Essential (Install These First)

#### Brave Search
AI-friendly web search with clean results.

```json
"brave-search": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-brave-search"],
  "env": {
    "BRAVE_API_KEY": "YOUR_API_KEY"
  }
}
```

**Get API Key:** https://brave.com/search/api/ (Free tier: 2000 queries/month)

**Tools provided:**
- `brave_web_search` - Search the web
- `brave_local_search` - Local business search

---

#### Fetch (URL to Markdown)
Converts any URL to clean markdown for LLM consumption.

```json
"fetch": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-fetch"]
}
```

**Tools provided:**
- `fetch` - Fetch URL and convert to markdown

---

#### Git
Read and search git repositories.

```json
"git": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-git"]
}
```

**Tools provided:**
- `git_status`, `git_log`, `git_diff`
- `git_commit`, `git_branch`
- `git_search` - Search code in repos

---

### Tier 2: Productivity

#### GitHub
Full GitHub integration - repos, issues, PRs.

```json
"github": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-github"],
  "env": {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxxxxxxxxxx"
  }
}
```

**Get Token:** GitHub → Settings → Developer settings → Personal access tokens

**Tools provided:**
- `create_issue`, `list_issues`, `update_issue`
- `create_pull_request`, `list_commits`
- `search_repositories`, `get_file_contents`

---

#### Memory (Knowledge Graph)
Persistent memory using a local knowledge graph.

```json
"memory": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-memory"]
}
```

**Tools provided:**
- `create_entities`, `create_relations`
- `search_nodes`, `open_nodes`
- `delete_entities`, `delete_relations`

**Note:** This is separate from your built-in memory system. Useful for cross-session persistence.

---

#### Sequential Thinking
Step-by-step reasoning for complex problems.

```json
"sequential-thinking": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
}
```

**Tools provided:**
- `sequentialthinking` - Break down complex problems

---

### Tier 3: Browser Automation

#### Playwright
Full browser automation - click, type, screenshot, scrape.

```json
"playwright": {
  "command": "npx",
  "args": ["-y", "@anthropic-ai/mcp-server-playwright"]
}
```

**Tools provided:**
- `browser_navigate`, `browser_click`
- `browser_type`, `browser_screenshot`
- `browser_evaluate` - Run JavaScript

**Note:** Requires display. May need X server on WSL or run on Windows side.

---

#### Puppeteer
Alternative to Playwright, Chrome-focused.

```json
"puppeteer": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
}
```

---

### Tier 4: Advanced Search & Research

#### Exa (AI Search Engine)
Search engine built specifically for AI - cleaner, more relevant results.

```json
"exa": {
  "command": "npx",
  "args": ["-y", "exa-mcp-server"],
  "env": {
    "EXA_API_KEY": "YOUR_API_KEY"
  }
}
```

**Get API Key:** https://exa.ai/

---

#### Tavily (Research Search)
Deep research-focused search with content extraction.

```json
"tavily": {
  "command": "npx",
  "args": ["-y", "tavily-mcp"],
  "env": {
    "TAVILY_API_KEY": "YOUR_API_KEY"
  }
}
```

**Get API Key:** https://tavily.com/

---

#### Firecrawl (Deep Scraping)
Convert entire websites to markdown, handle JS-heavy sites.

```json
"firecrawl": {
  "command": "npx",
  "args": ["-y", "firecrawl-mcp"],
  "env": {
    "FIRECRAWL_API_KEY": "YOUR_API_KEY"
  }
}
```

**Get API Key:** https://firecrawl.dev/ (or self-host)

---

### Tier 5: Databases

#### SQLite
Query local SQLite databases.

```json
"sqlite": {
  "command": "npx",
  "args": ["-y", "mcp-server-sqlite", "/path/to/database.db"]
}
```

---

#### PostgreSQL
Connect to PostgreSQL databases.

```json
"postgres": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-postgres"],
  "env": {
    "POSTGRES_CONNECTION_STRING": "postgresql://user:pass@host:5432/db"
  }
}
```

---

### Tier 6: Code Execution

#### E2B (Sandboxed Code)
Run Python/JS code in secure cloud sandboxes.

```json
"e2b": {
  "command": "npx",
  "args": ["-y", "@e2b/mcp-server"],
  "env": {
    "E2B_API_KEY": "YOUR_API_KEY"
  }
}
```

**Get API Key:** https://e2b.dev/

---

## Full Example Config

Here's a complete `mcp_config.json` with multiple servers:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["tools/mcp_server_local.py"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "BSA..."
      }
    },
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    },
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      }
    }
  }
}
```

---

## Installation Steps

### Step 1: Edit mcp_config.json

```bash
# Open in your editor
nano voice_chat/mcp_config.json
# Or from Windows
notepad C:\AI\index-tts\voice_chat\mcp_config.json
```

### Step 2: Add Server Entry

Copy the JSON block for the server you want from above.

### Step 3: Add API Keys (if required)

Get API keys from the links provided and add to the `env` section.

### Step 4: Restart Voice Agent

The servers connect on startup. Check the terminal for:
```
[MCP] Connecting to 'brave-search'...
[MCP] Connected to 'brave-search' ✓
```

### Step 5: Grant Tool Access to Characters

Edit `skills/assistant-utility/SKILL.md`:
```yaml
allowed_tools:
  - brave_web_search
  - fetch
  - git_status
  # ... add new tools
```

---

## Troubleshooting

### "npx: command not found"
```bash
# Install Node.js in WSL
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### "Server failed to connect"
- Check the command path is correct
- Verify API key is set
- Check terminal for error messages

### "Tool not available"
- Add tool name to character's `allowed_tools`
- Restart voice chat

### First run is slow
npx downloads packages on first use. Subsequent runs are faster.

---

## Free vs Paid

| Server | Free Tier | Notes |
|--------|-----------|-------|
| Fetch | Unlimited | No API key needed |
| Git | Unlimited | No API key needed |
| Memory | Unlimited | No API key needed |
| Brave Search | 2000/month | Free tier sufficient |
| GitHub | Unlimited | Just needs PAT |
| Exa | Limited | Trial available |
| Tavily | 1000/month | Free tier available |
| E2B | Limited | Free credits |

---

## Resources

- **MCP Specification:** https://spec.modelcontextprotocol.io/
- **MCP TypeScript SDK:** https://github.com/modelcontextprotocol/typescript-sdk
- **MCP Python SDK:** https://github.com/modelcontextprotocol/python-sdk
- **Server Registry:** https://mcpservers.org/
- **Awesome List:** https://github.com/punkpeye/awesome-mcp-servers
