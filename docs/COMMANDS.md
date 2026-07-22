# MCP CLI Command System

MCP CLI features a unified command system that works consistently across all operational modes: Chat mode, Interactive mode, CLI mode, and Command mode. This document describes the command architecture, patterns, and usage.

## Command Architecture

### Unified Command System

The command system is built on a unified architecture defined in `src/mcp_cli/commands/base.py`:

```
┌─────────────────────────────────────────────────────┐
│           UnifiedCommand (Base Class)               │
│  - Single implementation works in all modes         │
│  - Consistent behavior and validation               │
│  - Mode-specific formatting support                 │
└─────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
│  Chat Mode   │ │ Interactive│ │  CLI Mode  │
│  /command    │ │   command  │ │ mcp-cli cmd│
└──────────────┘ └────────────┘ └────────────┘
```

### Command Components

Every command is built from these components:

1. **Command Definition** (`src/mcp_cli/commands/definitions/`) - Defines command metadata
2. **Command Action** (`src/mcp_cli/commands/actions/`) - Implements command logic
3. **Command Models** (`src/mcp_cli/commands/models/`) - Pydantic models for parameters and responses
4. **Command Registry** (`src/mcp_cli/commands/registry.py`) - Central command registration

### Command Modes

Commands can support one or more operational modes:

```python
class CommandMode(Flag):
    CHAT = auto()         # Chat mode (/command)
    CLI = auto()          # CLI mode (mcp-cli command)
    INTERACTIVE = auto()  # Interactive mode (command)
    ALL = CHAT | CLI | INTERACTIVE
```

## Command Patterns

### Singular vs Plural Pattern

MCP CLI follows a consistent singular/plural pattern for related commands:

| Command Type | Purpose | Example | Result |
|-------------|---------|---------|--------|
| **Singular** | Show current status | `/model` | Shows current model with tips |
| **Plural** | List all available | `/models` | Lists all models in a table |
| **Singular + Name** | Direct action | `/model gpt-4o` | Switches to gpt-4o |
| **Singular + Subcommand** | Specific action | `/model refresh` | Refreshes model discovery |

**Examples:**

```bash
# Models
/model                    # Show current model status
/models                   # List all available models
/model gpt-4o             # Switch to gpt-4o directly
/model refresh            # Refresh model discovery

# Providers
/provider                 # Show current provider status
/providers                # List all available providers
/provider ollama          # Switch to Ollama provider
/provider set openai api_key KEY  # Configure provider

# Themes
/theme                    # Show current theme with preview
/themes                   # List all available themes
/theme dark               # Switch to dark theme

# Servers
/server echo              # Show details about echo server
/servers                  # List all connected MCP servers
/servers ping             # Ping all servers
```


### Command Groups

Some commands have subcommands organized into groups:

```bash
# Tools command group
/tools                    # List available tools (default: list)
/tools list               # List tools explicitly
/tools call               # Interactive tool execution
/tools --all              # Detailed tool information

# Server command group
/server                   # List servers (default)
/server list              # List servers explicitly
/server add <name>        # Add a new server
/server remove <name>     # Remove a server
/server ping <name>       # Ping specific server

# Token command group
/token                    # List tokens (default)
/token list               # List tokens explicitly
/token set <name>         # Store a token
/token get <name>         # Get token details
/token delete <name>      # Delete a token
```

## Core Commands

### Provider & Model Management

#### Provider Commands

**Show Current Provider:**
```bash
/provider                           # Show current configuration (default: ollama)
```

**List All Providers:**
```bash
/providers                          # List all providers
```

**Switch Provider:**
```bash
/provider ollama                    # Switch to Ollama
/provider openai                    # Switch to OpenAI (requires API key)
/provider anthropic                 # Switch to Anthropic (requires API key)
/provider openai gpt-4o             # Switch to OpenAI with specific model
```

**Provider Configuration:**
```bash
/provider set openai api_key sk-...   # Configure API key
/provider set ollama api_base http://localhost:11434  # Configure endpoint
/provider config                      # Show detailed configuration
/provider diagnostic                  # Test provider connectivity
```

**Custom Provider Management:**
```bash
/provider custom                      # List custom providers
/provider add localai http://localhost:8080/v1 gpt-4  # Add custom provider
/provider remove localai              # Remove custom provider
```

#### Model Commands

**Show Current Model:**
```bash
/model                              # Show current model (default: gpt-oss)
```

**List Available Models:**
```bash
/models                             # List models for current provider
/models openai                      # List models for specific provider
```

**Switch Model:**
```bash
/model llama3.3                     # Switch to different Ollama model
/model gpt-4o                       # Switch to GPT-4 (if using OpenAI)
/model claude-3-5-sonnet-20241022   # Switch to Claude 3.5 (if using Anthropic)
```

**Model Discovery:**
```bash
/model refresh                      # Refresh available models
```

### Tool Management

**List Tools:**
```bash
/tools                              # List available tools
/tools --all                        # Show detailed tool information
/tools --raw                        # Show raw JSON definitions
```

**Execute Tools:**
```bash
/tools call                         # Interactive tool execution
```

**Tool History:**
```bash
/toolhistory                        # Show tool execution history
/th -n 5                           # Last 5 tool calls
/th 3                              # Details for call #3
/th --json                         # Full history as JSON
```

### Server Management

**List Servers:**
```bash
/server                             # List all configured servers
/servers                            # List servers (alias)
/servers list                       # List servers explicitly
/servers list all                   # Include disabled servers
```

**Add Servers:**

Runtime server additions persist in `~/.mcp-cli/preferences.json`:

```bash
# STDIO servers
/server add sqlite stdio uvx mcp-server-sqlite --db-path test.db
/server add playwright stdio npx @playwright/mcp@latest
/server add time stdio uvx mcp-server-time
/server add fs stdio npx @modelcontextprotocol/server-filesystem /path/to/dir

# HTTP/SSE servers with authentication
/server add github --transport http --header "Authorization: Bearer ghp_token" -- https://api.github.com/mcp
/server add myapi --transport http --env API_KEY=secret -- https://api.example.com/mcp
/server add events --transport sse -- https://events.example.com/sse
```

**Manage Servers:**
```bash
/server enable <name>               # Enable a disabled server
/server disable <name>              # Disable without removing
/server remove <name>               # Remove user-added server
/server ping <name>                 # Test server connectivity
/server <name>                      # Show server configuration details
```

**Server Details:**
```bash
/server echo                        # Show details about echo server
/servers ping                       # Ping all servers
```

### Token Management

**List Tokens:**
```bash
/token                              # List all tokens (default)
/token list                         # List all tokens explicitly
/token list --oauth                 # Only OAuth tokens
/token list --api-keys              # Only API keys
/token list --bearer                # Only bearer tokens
```

**Store Tokens:**
```bash
/token set my-server --type bearer  # Interactive prompt (secure)
/token set openai --type api-key --provider openai  # API key
/token set my-api --type bearer --value "token123"  # With value (less secure)
```

**Get Token Info:**
```bash
/token get my-server                # Get metadata
/token get openai --namespace api-key  # Get from specific namespace
/token get notion --oauth           # Get OAuth token
```

**Delete Tokens:**
```bash
/token delete my-server             # Delete bearer token
/token delete openai --namespace api-key  # Delete from namespace
/token delete notion --oauth        # Delete OAuth token
/token clear                        # Clear all tokens (with confirmation)
/token clear --force                # Clear without confirmation
```

**Storage Backends:**
```bash
/token backends                     # List available backends
```

See [TOKEN_MANAGEMENT.md](./TOKEN_MANAGEMENT.md) for comprehensive token documentation.

### Conversation Management

**View History:**
```bash
/conversation                       # Show conversation history
/ch -n 10                          # Last 10 messages
/ch 5                              # Details for message #5
/ch --json                         # Full history as JSON
```

**Save & Clear:**
```bash
/save                              # Save session (auto-generates filename)
/compact                           # Summarize conversation
/clear                             # Clear conversation history
/cls                               # Clear screen only
```

### Virtual Memory (Experimental)

Inspect the AI virtual memory subsystem during conversations (requires `--vm` flag):

```bash
/memory                            # Summary dashboard: mode, turn, pages, utilization, metrics
/vm                                # Alias for /memory
/mem                               # Alias for /memory
/memory pages                      # Table of all memory pages (ID, type, tier, tokens, pinned)
/memory page <id>                  # Detailed view of a specific page with content preview
/memory stats                      # Full debug dump of all VM subsystem stats (JSON)
```

The dashboard shows:
- **Working set utilization**: L0/L1 page counts, tokens used/available, visual utilization bar
- **Page table**: Total pages, dirty pages, distribution by storage tier (L0-L4)
- **Metrics**: Page faults, evictions, TLB hit rate
- **Configuration**: VM mode, current turn, token budget

Without `--vm`, the command shows: "VM not enabled. Start with --vm flag."

**VM CLI Flags:**
```bash
# Enable VM with defaults (passive mode, 128K budget)
mcp-cli --server sqlite --vm

# Set a tight budget to force eviction pressure
mcp-cli --server sqlite --vm --vm-budget 500

# Use relaxed mode (VM-aware but conversational)
mcp-cli --server sqlite --vm --vm-mode relaxed
```

### Model-Driven Planning

Enable the LLM to autonomously create and execute multi-step plans during conversation:

```bash
# Enable plan tools — the model decides WHEN to plan
mcp-cli --server sqlite --plan-tools

# Or with the chat subcommand
mcp-cli chat --server sqlite --plan-tools
```

When enabled, three internal tools are added to the LLM's tool list:
- `plan_create_and_execute` — generate and execute a plan in one call (most common)
- `plan_create` — generate a plan without executing it
- `plan_execute` — execute a previously created plan by ID

The model calls these when it determines a task requires multi-step coordination (e.g., "read a file, find importers, run tests"). For simple tasks, it calls tools directly.

See [PLANNING.md](./PLANNING.md) for full documentation.

### Direct LLM Chat (No Tools)

Skip MCP server connections entirely and chat directly with the LLM:

```bash
# No tool calling — MCP servers are not connected
mcp-cli --no-tools
mcp-cli --no-tools --provider openai --model gpt-4o
mcp-cli chat --no-tools --provider anthropic --model claude-sonnet-4-6
```

When `--no-tools` is set:
- No MCP servers are initialized (no network connections attempted)
- The LLM receives no tool definitions
- The model responds conversationally only

This is useful when you want a fast, lightweight chat session without the overhead of server connections, or when using a model that handles tools poorly.

### Token Usage

Track API token consumption across your conversation:

```bash
/usage                             # Show token usage summary
/tokens                            # Alias for /usage
/cost                              # Alias for /usage
```

Displays per-turn and cumulative input/output token counts. When provider usage data is unavailable, tokens are estimated using a chars/4 heuristic (marked as "estimated").

### Session Persistence

Save and restore conversation sessions:

```bash
/sessions list                     # List all saved sessions
/sessions save                     # Save current session
/sessions load <id>                # Load a saved session
/sessions delete <id>              # Delete a saved session
```

Sessions are stored as JSON in `~/.mcp-cli/sessions/`. Auto-save triggers every 10 turns by default.

### Multi-Modal Attachments

Stage files to include in your next message. Supports images, text/code files, and audio.

**Stage Files:**
```bash
/attach photo.png                  # Stage an image
/attach code.py                    # Stage a text/code file
/attach clip.mp3                   # Stage an audio file
/file data.csv                     # Alias for /attach
/image screenshot.heic             # Alias for /attach
```

**Manage Staging:**
```bash
/attach list                       # Show currently staged files
/attach clear                      # Clear all staged files
```

**Inline References (in any message):**
```bash
@file:screenshot.png describe what you see in this image
@file:report.txt @file:data.csv compare these two files
```

Image URLs in messages are automatically detected and sent as vision content.

**Supported File Types:**
- **Images:** `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.heic`
- **Audio:** `.mp3`, `.wav`
- **Text/Code:** `.txt`, `.md`, `.csv`, `.json`, `.html`, `.xml`, `.yaml`, `.yml`, `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.sh`, `.bash`, `.rs`, `.go`, `.java`, `.c`, `.cpp`, `.h`, `.hpp`, `.rb`, `.swift`, `.kt`, `.sql`, `.toml`, `.ini`, `.cfg`, `.env`, `.log`

**Size Limits:**
- Maximum file size: 20 MB per file
- Maximum attachments per message: 10

**CLI Flag:**
```bash
# Attach files to the first message (repeatable)
mcp-cli --server sqlite --attach image.png --attach data.csv
```

See [ATTACHMENTS.md](./ATTACHMENTS.md) for comprehensive attachments documentation.

### Conversation Export

Export conversations in structured formats:

```bash
/export markdown                   # Export as formatted Markdown
/export markdown chat.md           # Export to specific file
/export json                       # Export as structured JSON
/export json chat.json             # Export to specific file
```

Markdown exports include tool calls as code blocks. JSON exports include metadata and token usage.

### UI Customization

**Theme Management:**
```bash
/theme                             # Interactive theme selector with preview
/theme dark                        # Switch to dark theme
/theme monokai                     # Switch to monokai theme
/themes                            # List all available themes
```

**Available Themes:**
- default
- dark
- light
- minimal
- terminal
- monokai
- dracula
- solarized

Themes are persisted across sessions in user preferences.

### Session Control

**Display Settings:**
```bash
/verbose                           # Toggle verbose/compact display (Default: Enabled)
/confirm                           # Toggle tool call confirmation (Default: Enabled)
```

**System Commands:**
```bash
/help                             # Show all commands
/help tools                       # Help for specific command
/interrupt                        # Stop running operations
/exit                             # Exit chat mode
```

## Command Implementation

### Creating a New Command

To create a new command, implement the `UnifiedCommand` interface:

```python
# src/mcp_cli/commands/definitions/mycommand.py
from mcp_cli.commands.base import UnifiedCommand, CommandMode, CommandResult

class MyCommand(UnifiedCommand):
    @property
    def name(self) -> str:
        return "mycommand"

    @property
    def aliases(self) -> List[str]:
        return ["mc"]  # Optional aliases

    @property
    def description(self) -> str:
        return "My custom command"

    @property
    def modes(self) -> CommandMode:
        return CommandMode.ALL  # Support all modes

    async def execute(self, **kwargs) -> CommandResult:
        # Implementation
        return CommandResult(success=True, output="Done!")
```

### Command Parameters

Define parameters for type-safe argument handling:

```python
from mcp_cli.commands.base import CommandParameter

@property
def parameters(self) -> List[CommandParameter]:
    return [
        CommandParameter(
            name="server_name",
            type=str,
            required=True,
            help="Name of the server"
        ),
        CommandParameter(
            name="verbose",
            type=bool,
            default=False,
            is_flag=True,
            help="Enable verbose output"
        )
    ]
```

### Command Models

Use Pydantic models for structured parameters:

```python
# src/mcp_cli/commands/models/mycommand.py
from pydantic import BaseModel

class MyCommandParams(BaseModel):
    server_name: str
    verbose: bool = False

class MyCommandResponse(BaseModel):
    success: bool
    message: str
```

### Command Actions

Implement the action logic separately for better organization:

```python
# src/mcp_cli/commands/actions/mycommand.py
from mcp_cli.commands.models.mycommand import MyCommandParams, MyCommandResponse

async def my_command_action(params: MyCommandParams) -> MyCommandResponse:
    # Implementation logic
    return MyCommandResponse(success=True, message="Complete")
```

### Registering Commands

Register commands in the central registry:

```python
# src/mcp_cli/commands/__init__.py
from mcp_cli.commands.registry import registry
from mcp_cli.commands.definitions.mycommand import MyCommand

def register_commands():
    registry.register(MyCommand())
```

## Command Usage Across Modes

### Chat Mode

Commands in chat mode use the `/` prefix:

```bash
# Start chat mode
mcp-cli --server sqlite

# Use commands
> /provider openai
> /model gpt-4o
> /tools list
> /server add myserver stdio node server.js
```

### Interactive Mode

Commands in interactive mode don't require the `/` prefix:

```bash
# Start interactive mode
mcp-cli interactive --server sqlite

# Use commands
> provider openai
> model gpt-4o
> tools list
> servers
```

### CLI Mode

Commands in CLI mode use the standard CLI syntax:

```bash
# Direct CLI commands
mcp-cli provider list
mcp-cli model set gpt-4o
mcp-cli tools --server sqlite
mcp-cli servers list

# With global options
mcp-cli --provider openai --server sqlite tools
```

### Command Mode

Unix-friendly automation mode:

```bash
# Tool execution
mcp-cli cmd --server sqlite --tool list_tables --raw

# Text processing
echo "data" | mcp-cli cmd --server sqlite --input - --output result.txt

# Pipeline usage
mcp-cli cmd --tool read_query --tool-args '{"query": "SELECT * FROM users"}' | jq .
```

## Design Principles

### 1. Predictability
Users can guess commands based on consistent patterns:
- Singular shows current state
- Plural lists all options
- Direct actions without unnecessary subcommands

### 2. Discoverability
Tips and indicators guide users to related commands:
```
💡 Tip: Use: /model <name> to switch  |  /models to list all  |  /model refresh to discover
```

### 3. Efficiency
Quick actions available without deep menu navigation:
```bash
/theme dark              # Direct action
/provider openai         # Direct switch
/model gpt-4o            # Direct selection
```

### 4. Consistency
Same patterns across all command groups:
- All listing commands use tables
- All status commands use panels
- All commands have help text

### 5. Theme-Aware
Uses theme colors, not hardcoded values:
```python
from chuk_term.ui import output

output.success("Operation complete")  # Uses theme's success color
output.error("Failed")                # Uses theme's error color
output.info("Information")            # Uses theme's info color
```

## Visual Indicators

### Command Help Menu

Typing just `/` shows a menu of all available commands:

```
Available Commands:
  clear ▸      - Clear conversation or screen
  conversation - Show conversation history
  exit         - Exit the application
  export ▸     - Export conversation (markdown/json)
  help         - Show help information
  memory ▸     - View AI virtual memory state (aliases: /vm, /mem)
  model ▸      - Show current model or switch models
  models       - List all available models
  provider ▸   - Show current provider or switch providers
  providers    - List all available providers
  server ▸     - Manage MCP servers
  servers      - List all connected servers
  sessions ▸   - Save/load/list sessions
  theme ▸      - Show or change UI theme
  themes       - List all available themes
  token ▸      - Manage authentication tokens
  tools ▸      - List and manage tools
  usage        - Show token usage stats
```

**Indicators:**
- `▸` - Command has subcommands available
- No indicator - Direct command with no subcommands

### Status Panels

Status commands use panels for clear presentation:

```
╭─────────────────────────────────────────╮
│ ℹ Current provider: openai              │
│ ℹ Current model   : gpt-4o-mini         │
│ ℹ Status          : ✅ Ready            │
│ ℹ Features        : 📡 🔧 👁️            │
╰─────────────────────────────────────────╯
```

### Tables for Lists

List commands use formatted tables:

```
┌──────────┬───────────┬──────────────┐
│ Provider │ Status    │ Models       │
├──────────┼───────────┼──────────────┤
│ openai   │ ✅ Ready  │ 15 models    │
│ ollama   │ ✅ Ready  │ 51 models    │
└──────────┴───────────┴──────────────┘
```

## Advanced Command Features

### Parameter Validation

Commands automatically validate parameters before execution:

```python
def validate_parameters(self, **kwargs) -> Optional[str]:
    for param in self.parameters:
        if param.required and param.name not in kwargs:
            return f"Missing required parameter: {param.name}"

        if param.name in kwargs and param.choices:
            if kwargs[param.name] not in param.choices:
                return f"Invalid choice for {param.name}"

    return None
```

### Mode-Specific Formatting

Commands can format output differently for each mode:

```python
def format_output(self, result: CommandResult, mode: CommandMode) -> str:
    if mode == CommandMode.CLI:
        # Machine-readable format
        return json.dumps(result.data)
    else:
        # Human-readable format
        return result.output
```

### Context Requirements

Commands can specify if they need active context:

```python
@property
def requires_context(self) -> bool:
    return True  # Needs tool_manager, llm_client, etc.
```

### Hidden Commands

Commands can be hidden from help menus:

```python
@property
def hidden(self) -> bool:
    return True  # Won't appear in /help
```

## Error Handling

Commands use consistent error handling patterns:

```python
async def execute(self, **kwargs) -> CommandResult:
    try:
        # Execute action
        result = await my_action(params)
        return CommandResult(success=True, data=result)
    except ValueError as e:
        return CommandResult(success=False, error=f"Invalid parameter: {e}")
    except Exception as e:
        return CommandResult(success=False, error=f"Command failed: {e}")
```

Common error patterns:
- **Missing parameters**: Validated before execution
- **Invalid choices**: Checked against allowed values
- **Context errors**: Graceful handling of missing context
- **Action failures**: Proper error messages with hints

## Best Practices

### Command Design

1. **Keep it Simple**: Commands should do one thing well
2. **Use Defaults**: Sensible defaults for optional parameters
3. **Provide Help**: Clear descriptions and examples
4. **Be Consistent**: Follow established patterns
5. **Think User-First**: Design for how users will actually use it

### Implementation

1. **Separate Concerns**: Definition → Model → Action
2. **Use Type Hints**: Leverage Pydantic for validation
3. **Handle Errors**: Graceful error messages
4. **Test All Modes**: Ensure command works in all supported modes
5. **Document**: Update command help and this guide

### User Experience

1. **Fast Defaults**: Most common action is the default
2. **Clear Feedback**: Users know what happened
3. **Helpful Tips**: Guide users to related commands
4. **Visual Clarity**: Use tables, panels, and colors appropriately
5. **Keyboard Friendly**: Tab completion and shortcuts

## Future Enhancements

Planned command system improvements:

- [ ] Command aliases configuration (user-defined)
- [ ] Command history and replay
- [ ] Command composition and piping
- [ ] Custom command plugins
- [ ] Command macros and scripts
- [ ] Tab completion improvements
- [ ] Command fuzzy search
- [ ] Interactive command builder
- [ ] Command export/import
- [ ] Command templates

## See Also

- [COMMAND_PATTERNS.md](./COMMAND_PATTERNS.md) - Detailed pattern documentation
- [TOKEN_MANAGEMENT.md](./TOKEN_MANAGEMENT.md) - Token command reference
- [OAUTH.md](./OAUTH.md) - OAuth authentication
- [README.md](../README.md) - Main documentation
