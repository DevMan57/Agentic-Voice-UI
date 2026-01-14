# Agent Skills

Character skill packages for IndexTTS2 Voice Agent. Each skill provides personality, knowledge references, and optional scripts.

## Available Skills

| Skill | Character | Description |
|-------|-----------|-------------|
| `hermione-companion` | 🧙‍♀️ Hermione Granger | Roleplay + academic research |
| `lisbeth-companion` | 🖤 Lisbeth Salander | Minimalist + security expertise |
| `assistant-utility` | 😏 Witty Assistant | File operations + general productivity |

## Structure

```
skills/
├── hermione-companion/
│   ├── SKILL.md              # Character definition + personality
│   ├── references/           # Lore, knowledge bases
│   └── scripts/              # Python helpers (optional)
├── lisbeth-companion/
│   └── ...
└── assistant-utility/
    └── ...
```

## Progressive Disclosure

Skills use tiered loading to reduce context:

1. **Metadata** (~100 tokens) - Always loaded (name + description)
2. **SKILL.md body** - Loaded when character is active
3. **Scripts/References** - Loaded only when needed

This reduces context usage by 50-90% for casual conversation.

## Creating a New Skill

### Manual Creation

1. Create directory: `skills/<skill-name>/`
2. Create `SKILL.md` with YAML frontmatter:
   ```yaml
   ---
   name: my-skill
   description: Brief description for context
   ---
   # Full instructions here...
   ```
3. Add `references/` for knowledge, `scripts/` for logic
4. Register in character system or load dynamically

### Agent-Created Skills

Characters with the `create_skill` tool permission can create skills via voice:

> "Create a skill called 'code-reviewer' that helps review Python code"

The skill is created directly in `skills/<name>/` with:
- `SKILL.md` - Main skill definition
- `references/` - Directory for knowledge files
- `scripts/` - Directory for Python helpers

**After creation, you may need to:**
- Restart the app or reload via Character Manager
- Add a custom voice file (`.wav`) to `voice_reference/`
- Add an avatar image

## Tool Access

Skills can specify which tools they have access to in their `allowed_tools` list.
The assistant has access to all registered tools including MCP server tools.

## Best Practices

- Keep SKILL.md under 500 lines
- Move detailed reference material to `references/`
- Scripts should return conversational text for TTS
- Match the character's voice and tone

