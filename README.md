# MCP CLI - Model Context Protocol Command Line Interface

[![CI](https://github.com/chrishayuk/mcp-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/chrishayuk/mcp-cli/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/mcp-cli.svg)](https://pypi.org/project/mcp-cli/)

A powerful, feature-rich command-line interface for interacting with Model Context Protocol servers. This client enables seamless communication with LLMs through integration with the [CHUK Tool Processor](https://github.com/chrishayuk/chuk-tool-processor) and [CHUK-LLM](https://github.com/chrishayuk/chuk-llm), providing tool usage, conversation management, and multiple operational modes.

**Default Configuration**: MCP CLI defaults to using Ollama with the `gpt-oss` reasoning model for local, privacy-focused operation without requiring API keys.

## 🆕 Recent Updates (v0.16)

### AI Virtual Memory (Experimental)
- **`--vm` flag**: Enable OS-style virtual memory for conversation context management, powered by `chuk-ai-session-manager`
- **`--vm-budget`**: Control token budget for conversation events (system prompt is uncapped on top), forcing earlier eviction and page creation
- **`--vm-mode`**: Choose VM mode — `passive` (runtime-managed, default), `relaxed` (VM-aware conversation), or `strict` (model-driven paging with tools)
- **`/memory` command**: Visualize VM state during conversations — page table, working set utilization, eviction metrics, TLB stats (aliases: `/vm`, `/mem`)
- **Multimodal page_fault**: Image pages return multi-block content (text + image_url) so multimodal models can re-analyze recalled images
- **`/memory page <id> --download`**: Export page content to local files with modality-aware extensions (.txt, .json, .png)

### Execution Plans (Tier 6)
- **`/plan` command**: Create, inspect, and execute reproducible tool call graphs — `create`, `list`, `show`, `run`, `delete`, `resume`
- **Model-driven planning (`--plan-tools`)**: The LLM autonomously creates and executes plans during conversation — no `/plan` command needed. It calls `plan_create_and_execute` when multi-step orchestration is required, and uses regular tools for simple tasks. Each step renders with real-time progress in the terminal
- **Parallel batch execution**: Independent plan steps run concurrently via topological batching (Kahn's BFS), with configurable `max_concurrency`
- **Variable resolution**: `${var}`, `${var.field}` nested access, and template strings like `"https://${api.host}/users"` — type-preserving for single refs
- **Dry-run mode**: Trace planned tool calls without executing — safe for production inspection
- **Checkpointing & resume**: Execution state persisted after each batch; resume interrupted plans with `/plan resume <id>`
- **Guard integration**: Plans respect existing budget, per-tool limits, and runaway detection guards
- **DAG visualization**: ASCII rendering with status indicators (○/◉/●/✗) and parallel markers (∥)
- **Re-planning**: Optional LLM-based re-planning on step failure (`enable_replan=True`)
- **Powered by**: [chuk-ai-planner](https://github.com/chrishayuk/chuk-ai-planner) graph-based plan DSL

### MCP Apps (SEP-1865)
- **Interactive HTML UIs**: MCP servers can serve interactive HTML applications (charts, tables, maps, markdown viewers) that render in your browser
- **Sandboxed iframes**: Apps run in secure sandboxed iframes with CSP protection
- **WebSocket bridge**: Real-time bidirectional communication between browser apps and MCP servers
- **Automatic launch**: Tools with `_meta.ui` annotations automatically open in the browser when called
- **Session reliability**: Message queuing, reconnection with exponential backoff, deferred tool result delivery

### Production Hardening
- **Secret Redaction**: All log output (console and file) is automatically redacted for Bearer tokens, API keys, OAuth tokens, and Authorization headers
- **Structured File Logging**: Optional `--log-file` flag enables rotating JSON log files (10MB, 3 backups) at DEBUG level
- **Per-Server Timeouts**: Server configs support `tool_timeout` and `init_timeout` overrides, resolved per-server → global → default
- **Thread-Safe OAuth**: Concurrent OAuth flows serialized with `asyncio.Lock` and copy-on-write header mutation
- **Server Health Monitoring**: `/health` command, health-check-on-failure diagnostics, optional `--health-interval` background polling

### Performance & Polish
- **O(1) Tool Lookups**: Indexed tool lookup replacing O(n) linear scans
- **Cached LLM Tool Metadata**: Per-provider caching with automatic invalidation
- **Startup Progress**: Real-time progress messages during initialization
- **Token Usage Tracking**: Per-turn and cumulative tracking with `/usage` command (aliases: `/tokens`, `/cost`)
- **Session Persistence**: Save/load/list conversation sessions with auto-save every 10 turns (`/sessions`)
- **Conversation Export**: Export conversations as Markdown or JSON with metadata (`/export`)

### Dashboard (Real-Time Browser UI)
- **`--dashboard` flag**: Launch a real-time browser dashboard alongside chat mode
- **Agent Terminal**: Live conversation view with message bubbles, streaming tokens, and attachment rendering
- **Activity Stream**: Tool call/result pairs, reasoning steps, and user attachment events
- **Plan Viewer**: Visual execution plan progress with DAG rendering
- **Tool Registry**: Browse discovered tools, trigger execution from the browser
- **Config Panel**: View and switch providers, models, and system prompt
- **File Attachments**: "+" button for browser file upload, drag-and-drop, and clipboard paste

### Multi-Modal Attachments
- **`/attach` command**: Stage files for the next message — images, text/code, and audio (aliases: `/file`, `/image`)
- **`--attach` CLI flag**: Attach files to the first message (repeatable: `--attach img.png --attach code.py`)
- **Inline `@file:` references**: Mention `@file:path/to/file` anywhere in a message to attach it
- **Image URL detection**: HTTP/HTTPS image URLs in messages are automatically sent as vision content
- **Supported formats**: PNG, JPEG, GIF, WebP, HEIC (images), MP3, WAV (audio), plus 25+ text/code extensions
- **Dashboard rendering**: Image thumbnails, expandable text previews, audio players, file badges
- **Browser upload**: "+" button in dashboard chat input with drag-and-drop and clipboard paste support

### Code Quality
- **Core/UI Separation**: Core modules use `logging` only — no UI imports
- **4,300+ tests**: Comprehensive test suite with branch coverage, integration tests, and 60% minimum threshold
- **15 Architecture Principles**: Documented and enforced (see [architecture.md](architecture.md))
- **Full [Roadmap](roadmap.md)**: Tiers 1-6 complete, Tiers 7-12 planned (traces, memory scopes, skills, scheduling, multi-agent)

## 🔄 Architecture Overview

The MCP CLI is built on a modular architecture with clean separation of concerns:

- **[CHUK Tool Processor](https://github.com/chrishayuk/chuk-tool-processor)**: Production-grade async tool execution with middleware (retry, circuit breaker, rate limiting), multiple execution strategies, and observability
- **[CHUK-LLM](https://github.com/chrishayuk/chuk-llm)**: Unified LLM provider with dynamic model discovery, capability-based selection, and llama.cpp integration (1.53x faster than Ollama with automatic model reuse)
- **[CHUK-Term](https://github.com/chrishayuk/chuk-term)**: Enhanced terminal UI with themes, cross-platform terminal management, and rich formatting
- **MCP CLI**: Command orchestration and integration layer (this project)

## 🌟 Features

### Multiple Operational Modes
- **Chat Mode**: Conversational interface with streaming responses and automated tool usage (default: Ollama/gpt-oss)
- **Interactive Mode**: Command-driven shell interface for direct server operations
- **Command Mode**: Unix-friendly mode for scriptable automation and pipelines
- **Direct Commands**: Run individual commands without entering interactive mode

### Advanced Chat Interface
- **Streaming Responses**: Real-time response generation with live UI updates
- **Reasoning Visibility**: See AI's thinking process with reasoning models (gpt-oss, GPT-5, Claude 4.5)
- **Concurrent Tool Execution**: Execute multiple tools simultaneously while preserving conversation order
- **Smart Interruption**: Interrupt streaming responses or tool execution with Ctrl+C
- **Performance Metrics**: Response timing, words/second, and execution statistics
- **Rich Formatting**: Markdown rendering, syntax highlighting, and progress indicators
- **Token Usage Tracking**: Per-turn and cumulative API token usage with `/usage` command
- **Multi-Modal Attachments**: Attach images, text files, and audio to messages via `/attach`, `--attach`, `@file:` refs, or browser upload
- **Session Persistence**: Auto-save and manual save/load of conversation sessions
- **Conversation Export**: Export to Markdown or JSON with metadata and token usage

### Comprehensive Provider Support

MCP CLI supports all providers and models from CHUK-LLM, including cutting-edge reasoning models:

| Provider | Key Models | Special Features |
|----------|------------|------------------|
| **Ollama** (Default) | 🧠 gpt-oss, llama3.3, llama3.2, qwen3, qwen2.5-coder, deepseek-coder, granite3.3, mistral, gemma3, phi3, codellama | Local reasoning models, privacy-focused, no API key required |
| **OpenAI** | 🚀 GPT-5 family (gpt-5, gpt-5-mini, gpt-5-nano), GPT-4o family, O3 series (o3, o3-mini) | Advanced reasoning, function calling, vision |
| **Anthropic** | 🧠 Claude 4.5 family (claude-4-5-opus, claude-4-5-sonnet), Claude 3.5 Sonnet | Enhanced reasoning, long context |
| **Azure OpenAI** 🏢 | Enterprise GPT-5, GPT-4 models | Private endpoints, compliance, audit logs |
| **Google Gemini** | Gemini 2.0 Flash, Gemini 1.5 Pro | Multimodal, fast inference |
| **Groq** ⚡ | Llama 3.1 models, Mixtral | Ultra-fast inference (500+ tokens/sec) |
| **Perplexity** 🌐 | Sonar models | Real-time web search with citations |
| **IBM watsonx** 🏢 | Granite, Llama models | Enterprise compliance |
| **Mistral AI** 🇪🇺 | Mistral Large, Medium | European, efficient models |

### Robust Tool System (Powered by CHUK Tool Processor v0.22+)
- **Automatic Discovery**: Server-provided tools are automatically detected and catalogued
- **Provider Adaptation**: Tool names are automatically sanitized for provider compatibility
- **Production-Grade Execution**: Middleware layers with timeouts, retries, exponential backoff, caching, and circuit breakers
- **Multiple Execution Strategies**: In-process (fast), isolated subprocess (safe), or remote via MCP
- **Concurrent Execution**: Multiple tools can run simultaneously with proper coordination
- **Rich Progress Display**: Real-time progress indicators and execution timing
- **Tool History**: Complete audit trail of all tool executions
- **Middleware**: Retry with exponential backoff, circuit breakers, and rate limiting via CTP
- **Streaming Tool Calls**: Support for tools that return streaming data

### MCP Apps (Interactive UIs)
- **Browser-based UIs**: MCP servers can serve interactive HTML applications that render in your browser
- **Automatic Detection**: Tools with `_meta.ui` annotations automatically launch browser apps on tool call
- **Sandboxed Execution**: Apps run in secure sandboxed iframes with Content Security Policy protection
- **WebSocket Bridge**: Real-time JSON-RPC bridge between browser apps and MCP tool servers
- **Session Persistence**: Message queuing during disconnects, automatic reconnection, deferred tool result delivery
- **structuredContent Support**: Full MCP spec compliance including structured content extraction and forwarding

### Execution Plans (Powered by chuk-ai-planner)
- **Plan Creation**: Generate execution plans from natural language descriptions using LLM-based plan agents
- **Model-Driven Planning**: With `--plan-tools`, the LLM autonomously decides when to plan — calls `plan_create_and_execute` for complex multi-step tasks, uses regular tools for simple ones
- **DAG Execution**: Plans are directed acyclic graphs — independent steps run in parallel batches, dependent steps wait
- **Variable Resolution**: Step outputs bind to variables (`result_variable`), referenced by later steps as `${var}` or `${var.field}`
- **Dry-Run Mode**: Trace what a plan would do without executing any tools — safe for production
- **Checkpointing**: Execution state saved after each batch; resume interrupted plans without re-running completed steps
- **Guard Integration**: Plans share budget and per-tool limits with the conversation — no bypass
- **Re-planning**: On step failure, optionally invoke the LLM to generate a revised plan for remaining work
- **DAG Visualization**: ASCII rendering shows dependency structure, batch grouping, and parallel markers
- **Persistence**: Plans stored as JSON at `~/.mcp-cli/plans/`

### Advanced Configuration Management
- **Environment Integration**: API keys and settings via environment variables
- **File-based Config**: YAML and JSON configuration files
- **User Preferences**: Persistent settings for active providers and models
- **Validation & Diagnostics**: Built-in provider health checks and configuration validation

### Enhanced User Experience
- **Cross-Platform Support**: Windows, macOS, and Linux with platform-specific optimizations via chuk-term
- **Rich Console Output**: Powered by chuk-term with 8 built-in themes (default, dark, light, minimal, terminal, monokai, dracula, solarized)
- **Advanced Terminal Management**: Cross-platform terminal operations including clearing, resizing, color detection, and cursor control
- **Interactive UI Components**: User input handling through chuk-term's prompt system (ask, confirm, select_from_list, select_multiple)
- **Command Completion**: Context-aware tab completion for all interfaces
- **Comprehensive Help**: Detailed help system with examples and usage patterns
- **Graceful Error Handling**: User-friendly error messages with troubleshooting hints

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

### Project
- **[Architecture](architecture.md)** - 15 design principles, module layout, and coding conventions
- **[Roadmap](roadmap.md)** - Vision, completed tiers (1-5), and planned tiers (6-12: plans, traces, skills, scheduling, multi-agent, remote sessions)

### Core Documentation
- **[Commands System](docs/COMMANDS.md)** - Complete guide to the unified command system, patterns, and usage across all modes
- **[Token Management](docs/TOKEN_MANAGEMENT.md)** - Comprehensive token management for providers and servers including OAuth, bearer tokens, and API keys

### Specialized Documentation
- **[Execution Plans](docs/PLANNING.md)** - Plan creation, parallel execution, variable resolution, checkpointing, guards, and re-planning
- **[Dashboard](docs/DASHBOARD.md)** - Real-time browser UI with agent terminal, activity stream, and file uploads
- **[Attachments](docs/ATTACHMENTS.md)** - Multi-modal file attachments: images, text, audio, and browser upload
- **[MCP Apps](docs/MCP_APPS.md)** - Interactive browser UIs served by MCP servers (SEP-1865)
- **[OAuth Authentication](docs/OAUTH.md)** - OAuth flows, storage backends, and MCP server integration
- **[Streaming Integration](docs/STREAMING.md)** - Real-time response streaming architecture
- **[Package Management](docs/PACKAGE_MANAGEMENT.md)** - Dependency organization and feature groups

### UI Documentation
- **[Themes](docs/ui/themes.md)** - Theme system and customization
- **[Output System](docs/ui/output.md)** - Rich console output and formatting
- **[Terminal Management](docs/ui/terminal.md)** - Cross-platform terminal operations

### Testing Documentation
- **[Unit Testing](docs/testing/UNIT_TESTING.md)** - Test structure and patterns
- **[Test Coverage](docs/testing/TEST_COVERAGE.md)** - Coverage requirements and reporting

## 📋 Prerequisites

- **Python 3.11 or higher**
- **For Local Operation (Default)**:
  - Ollama: Install from [ollama.ai](https://ollama.ai)
  - Pull the default reasoning model: `ollama pull gpt-oss`
- **For Cloud Providers** (Optional):
  - OpenAI: `OPENAI_API_KEY` environment variable (for GPT-5, GPT-4, O3 models)
  - Anthropic: `ANTHROPIC_API_KEY` environment variable (for Claude 4.5, Claude 3.5)
  - Azure: `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` (for enterprise GPT-5)
  - Google: `GEMINI_API_KEY` (for Gemini models)
  - Groq: `GROQ_API_KEY` (for fast Llama models)
  - Custom providers: Provider-specific configuration
- **MCP Servers**: Server configuration file (default: `server_config.json`)

## 🚀 Installation

### Quick Start with Ollama (Default)

1. **Install Ollama** (if not already installed):
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or visit https://ollama.ai for other installation methods
```

2. **Pull the default reasoning model**:
```bash
ollama pull gpt-oss  # Open-source reasoning model with thinking visibility
```

3. **Install and run MCP CLI**:
```bash
# Using uvx (recommended)
uvx mcp-cli --help

# Or install from source
git clone https://github.com/chrishayuk/mcp-cli
cd mcp-cli
pip install -e "."
mcp-cli --help

# Optional: Enable MCP Apps (interactive browser UIs)
pip install -e ".[apps]"
```

### Using Different Models

```bash
# === LOCAL MODELS (No API Key Required) ===

# Use default reasoning model (gpt-oss)
mcp-cli --server sqlite

# Use other Ollama models
mcp-cli --model llama3.3              # Latest Llama
mcp-cli --model qwen2.5-coder         # Coding-focused
mcp-cli --model deepseek-coder        # Another coding model
mcp-cli --model granite3.3            # IBM Granite

# === CLOUD PROVIDERS (API Keys Required) ===

# GPT-5 Family (requires OpenAI API key)
mcp-cli --provider openai --model gpt-5          # Full GPT-5 with reasoning
mcp-cli --provider openai --model gpt-5-mini     # Efficient GPT-5 variant
mcp-cli --provider openai --model gpt-5-nano     # Ultra-lightweight GPT-5

# GPT-4 Family
mcp-cli --provider openai --model gpt-4o         # GPT-4 Optimized
mcp-cli --provider openai --model gpt-4o-mini    # Smaller GPT-4

# O3 Reasoning Models
mcp-cli --provider openai --model o3             # O3 reasoning
mcp-cli --provider openai --model o3-mini        # Efficient O3

# Claude 4.5 Family (requires Anthropic API key)
mcp-cli --provider anthropic --model claude-4-5-opus    # Most advanced Claude
mcp-cli --provider anthropic --model claude-4-5-sonnet  # Balanced Claude 4.5
mcp-cli --provider anthropic --model claude-3-5-sonnet  # Claude 3.5

# Enterprise Azure (requires Azure configuration)
mcp-cli --provider azure_openai --model gpt-5    # Enterprise GPT-5

# Other Providers
mcp-cli --provider gemini --model gemini-2.0-flash      # Google Gemini
mcp-cli --provider groq --model llama-3.1-70b          # Fast Llama via Groq
```

## 🧰 Global Configuration

### Default Configuration

MCP CLI defaults to:
- **Provider**: `ollama` (local, no API key required)
- **Model**: `gpt-oss` (open-source reasoning model with thinking visibility)

### Command-line Arguments

Global options available for all modes and commands:

- `--server`: Specify server(s) to connect to (comma-separated)
- `--config-file`: Path to server configuration file (default: `server_config.json`)
- `--provider`: LLM provider (default: `ollama`)
- `--model`: Specific model to use (default: `gpt-oss` for Ollama)
- `--disable-filesystem`: Disable filesystem access (default: enabled)
- `--api-base`: Override API endpoint URL
- `--api-key`: Override API key (not needed for Ollama)
- `--token-backend`: Override token storage backend (`auto`, `keychain`, `windows`, `secretservice`, `encrypted`, `vault`)
- `--verbose`: Enable detailed logging
- `--quiet`: Suppress non-essential output
- `--log-file`: Write debug logs to a rotating file (secrets auto-redacted)
- `--vm`: [Experimental] Enable AI virtual memory for context management
- `--vm-budget`: Token budget for conversation events in VM mode (default: 128000, on top of system prompt)
- `--vm-mode`: VM mode — `passive` (default), `relaxed`, or `strict`
- `--dashboard`: Launch a real-time browser dashboard UI alongside chat mode
- `--attach`: Attach files to the first message (repeatable: `--attach img.png --attach code.py`)
- `--plan-tools`: Enable model-driven planning — the LLM autonomously creates and executes multi-step plans
- `--no-tools`: Disable MCP tool calling entirely — chat directly with the LLM without connecting to any MCP servers

### Environment Variables

```bash
# Override defaults
export LLM_PROVIDER=ollama              # Default provider (already the default)
export LLM_MODEL=gpt-oss                # Default model (already the default)

# For cloud providers (optional)
export OPENAI_API_KEY=sk-...           # For GPT-5, GPT-4, O3 models
export ANTHROPIC_API_KEY=sk-ant-...    # For Claude 4.5, Claude 3.5
export AZURE_OPENAI_API_KEY=sk-...     # For enterprise GPT-5
export AZURE_OPENAI_ENDPOINT=https://...
export GEMINI_API_KEY=...              # For Gemini models
export GROQ_API_KEY=...                # For Groq fast inference

# Tool configuration
export MCP_TOOL_TIMEOUT=120            # Tool execution timeout (seconds)
```

## 🌐 Available Modes

### 1. Chat Mode (Default)

Provides a natural language interface with streaming responses and automatic tool usage:

```bash
# Default mode with Ollama/gpt-oss reasoning model (no API key needed)
mcp-cli --server sqlite

# See the AI's thinking process with reasoning models
mcp-cli --server sqlite --model gpt-oss     # Open-source reasoning
mcp-cli --server sqlite --provider openai --model gpt-5  # GPT-5 reasoning
mcp-cli --server sqlite --provider anthropic --model claude-4-5-opus  # Claude 4.5 reasoning

# Use different local models
mcp-cli --server sqlite --model llama3.3
mcp-cli --server sqlite --model qwen2.5-coder

# Switch to cloud providers (requires API keys)
mcp-cli chat --server sqlite --provider openai --model gpt-5
mcp-cli chat --server sqlite --provider anthropic --model claude-4-5-sonnet

# Launch with real-time browser dashboard
mcp-cli --server sqlite --dashboard

# Attach files to the first message
mcp-cli --server sqlite --attach image.png --attach data.csv
```

### 2. Interactive Mode

Command-driven shell interface for direct server operations:

```bash
mcp-cli interactive --server sqlite

# With specific models
mcp-cli interactive --server sqlite --model gpt-oss       # Local reasoning
mcp-cli interactive --server sqlite --provider openai --model gpt-5  # Cloud GPT-5
```

### 3. Command Mode

Unix-friendly interface for automation and scripting:

```bash
# Process text with reasoning models
mcp-cli cmd --server sqlite --model gpt-oss --prompt "Think through this step by step" --input data.txt

# Use GPT-5 for complex reasoning
mcp-cli cmd --server sqlite --provider openai --model gpt-5 --prompt "Analyze this data" --input data.txt

# Execute tools directly
mcp-cli cmd --server sqlite --tool list_tables --output tables.json

# Pipeline-friendly processing
echo "SELECT * FROM users LIMIT 5" | mcp-cli cmd --server sqlite --tool read_query --input -
```

### 4. Direct Commands

Execute individual commands without entering interactive mode:

```bash
# List available tools
mcp-cli tools --server sqlite

# Show provider configuration
mcp-cli provider list

# Show available models for current provider
mcp-cli models

# Show models for specific provider
mcp-cli models openai    # Shows GPT-5, GPT-4, O3 models
mcp-cli models anthropic # Shows Claude 4.5, Claude 3.5 models
mcp-cli models ollama    # Shows gpt-oss, llama3.3, etc.

# Ping servers
mcp-cli ping --server sqlite

# List resources
mcp-cli resources --server sqlite

# UI Theme Management
mcp-cli theme                     # Show current theme and list available
mcp-cli theme dark                # Switch to dark theme
mcp-cli theme --select            # Interactive theme selector
mcp-cli theme --list              # List all available themes

# Token Storage Management
mcp-cli token backends            # Show available storage backends
mcp-cli --token-backend encrypted token list  # Use specific backend
```

## 🌐 MCP Apps (Interactive Browser UIs)

MCP Apps allow tool servers to provide interactive HTML UIs that render in your browser. When a tool has a `_meta.ui` annotation pointing to a UI resource, mcp-cli automatically launches a local web server and opens the app in your browser.

### Prerequisites

```bash
# Install the apps extra (adds websockets dependency)
pip install "mcp-cli[apps]"
```

### How It Works

1. Connect to an MCP server that provides app-enabled tools
2. Call a tool that has `_meta.ui` metadata (e.g., `show_chart`, `show_table`)
3. mcp-cli automatically fetches the UI resource, starts a local server, and opens your browser
4. The app receives tool results in real-time via WebSocket

### Example

```bash
# Connect to a server with app-enabled tools
mcp-cli --server view_demo

# In chat, ask for something visual:
> Show me the sales data as a chart
# Browser opens automatically with an interactive chart

# The /tools command shows which tools have app UIs (APP column)
> /tools
```

### Architecture

- **Host page** serves a sandboxed iframe with the app HTML
- **WebSocket bridge** proxies JSON-RPC between the browser and MCP servers
- **Security**: Iframe sandbox, CSP protection, XSS prevention, URL scheme validation
- **Reliability**: Message queuing during disconnects, exponential backoff reconnection, deferred tool result delivery

See [MCP Apps Documentation](docs/MCP_APPS.md) for the full guide.

## 🤖 Using Chat Mode

Chat mode provides the most advanced interface with streaming responses and intelligent tool usage.

### Starting Chat Mode

```bash
# Simple startup with default reasoning model (gpt-oss)
mcp-cli --server sqlite

# Multiple servers
mcp-cli --server sqlite,filesystem

# With advanced reasoning models
mcp-cli --server sqlite --provider openai --model gpt-5
mcp-cli --server sqlite --provider anthropic --model claude-4-5-opus
```

### Chat Commands (Slash Commands)

#### Provider & Model Management
```bash
/provider                           # Show current configuration (default: ollama)
/provider list                      # List all providers
/provider config                    # Show detailed configuration
/provider diagnostic               # Test provider connectivity
/provider set ollama api_base http://localhost:11434  # Configure Ollama endpoint
/provider openai                   # Switch to OpenAI (requires API key)
/provider anthropic                # Switch to Anthropic (requires API key)
/provider openai gpt-5             # Switch to OpenAI GPT-5

# Custom Provider Management
/provider custom                   # List custom providers
/provider add localai http://localhost:8080/v1 gpt-4  # Add custom provider
/provider remove localai           # Remove custom provider

/model                             # Show current model (default: gpt-oss)
/model llama3.3                    # Switch to different Ollama model
/model gpt-5                       # Switch to GPT-5 (if using OpenAI)
/model claude-4-5-opus             # Switch to Claude 4.5 (if using Anthropic)
/models                            # List available models for current provider
```

#### Tool Management
```bash
/tools                             # List available tools
/tools --all                       # Show detailed tool information
/tools --raw                       # Show raw JSON definitions
/tools call                        # Interactive tool execution

/toolhistory                       # Show tool execution history
/th -n 5                          # Last 5 tool calls
/th 3                             # Details for call #3
/th --json                        # Full history as JSON
```

#### Server Management (Runtime Configuration)
```bash
/server                            # List all configured servers
/server list                       # List servers (alias)
/server list all                   # Include disabled servers

# Add servers at runtime (persists in ~/.mcp-cli/preferences.json)
/server add <name> stdio <command> [args...]
/server add sqlite stdio uvx mcp-server-sqlite --db-path test.db
/server add playwright stdio npx @playwright/mcp@latest
/server add time stdio uvx mcp-server-time
/server add fs stdio npx @modelcontextprotocol/server-filesystem /path/to/dir

# HTTP/SSE server examples with authentication
/server add github --transport http --header "Authorization: Bearer ghp_token" -- https://api.github.com/mcp
/server add myapi --transport http --env API_KEY=secret -- https://api.example.com/mcp
/server add events --transport sse -- https://events.example.com/sse

# Manage server state
/server enable <name>              # Enable a disabled server
/server disable <name>             # Disable without removing
/server remove <name>              # Remove user-added server
/server ping <name>                # Test server connectivity

# Server details
/server <name>                     # Show server configuration details
```

**Note**: Servers added via `/server add` are stored in `~/.mcp-cli/preferences.json` and persist across sessions. Project servers remain in `server_config.json`.

#### Multi-Modal Attachments
```bash
/attach image.png                  # Stage an image for the next message
/attach code.py                    # Stage a text file
/attach list                       # Show currently staged files
/attach clear                      # Clear staged files
/file data.csv                     # Alias for /attach
/image screenshot.heic             # Alias for /attach

# Inline file references (in any message)
@file:screenshot.png describe what you see
@file:data.csv summarize this data

# Image URLs are auto-detected
https://example.com/photo.jpg what is in this image?
```

#### Conversation Management
```bash
/conversation                      # Show conversation history
/ch -n 10                         # Last 10 messages
/ch 5                             # Details for message #5
/ch --json                        # Full history as JSON

/save conversation.json            # Save conversation to file
/compact                          # Summarize conversation
/clear                            # Clear conversation history
/cls                              # Clear screen only
```

#### UI Customization
```bash
/theme                            # Interactive theme selector with preview
/theme dark                       # Switch to dark theme
/theme monokai                    # Switch to monokai theme

# Available themes: default, dark, light, minimal, terminal, monokai, dracula, solarized
# Themes are persisted across sessions
```

#### Token Management
```bash
/token                            # List all stored tokens
/token list                       # List all tokens explicitly
/token set <name>                 # Store a bearer token
/token get <name>                 # Get token details
/token delete <name>              # Delete a token
/token clear                      # Clear all tokens (with confirmation)
/token backends                   # Show available storage backends

# Examples
/token set my-api                 # Prompts for token value (secure)
/token get notion --oauth         # Get OAuth token for Notion server
/token list --api-keys            # List only provider API keys
```

**Token Storage Backends:**
MCP CLI supports multiple secure token storage backends:
- **Keychain** (macOS) - Uses macOS Keychain (default on macOS)
- **Windows Credential Manager** - Native Windows storage (default on Windows)
- **Secret Service** - Linux desktop keyring (GNOME/KDE)
- **Encrypted File** - AES-256 encrypted local files (cross-platform fallback)
- **HashiCorp Vault** - Enterprise secret management

Override the default backend with `--token-backend`:
```bash
# Use encrypted file storage instead of keychain
mcp-cli --token-backend encrypted token list

# Use vault for enterprise environments
mcp-cli --token-backend vault token list
```

See [Token Management Guide](docs/TOKEN_MANAGEMENT.md) for comprehensive documentation.

#### Session Control
```bash
/verbose                          # Toggle verbose/compact display (Default: Enabled)
/confirm                          # Toggle tool call confirmation (Default: Enabled)
/interrupt                        # Stop running operations
/server                           # Manage MCP servers (see Server Management above)
/help                            # Show all commands
/help tools                       # Help for specific command
/exit                            # Exit chat mode
```

**For complete command documentation**, see [Commands System Guide](docs/COMMANDS.md).

### Chat Features

#### Streaming Responses with Reasoning Visibility
- **🧠 Reasoning Models**: See the AI's thinking process with gpt-oss, GPT-5, Claude 4
- **Real-time Generation**: Watch text appear token by token
- **Performance Metrics**: Words/second, response time
- **Graceful Interruption**: Ctrl+C to stop streaming
- **Progressive Rendering**: Markdown formatted as it streams

#### Tool Execution
- Automatic tool discovery and usage
- Concurrent execution with progress indicators
- Verbose and compact display modes
- Complete execution history and timing

#### Multi-Modal Attachments
- Attach images, text files, and audio to any message
- `/attach` command with staging, list, and clear (aliases: `/file`, `/image`)
- Inline `@file:path` references in any message
- `--attach` CLI flag for first-message attachments
- Browser "+" button with drag-and-drop and clipboard paste (with `--dashboard`)
- Dashboard renders thumbnails, text previews, and audio players

#### Provider Integration
- Seamless switching between providers
- Model-specific optimizations
- API key and endpoint management
- Health monitoring and diagnostics

## 🖥️ Using Interactive Mode

Interactive mode provides a command shell for direct server interaction.

### Starting Interactive Mode

```bash
mcp-cli interactive --server sqlite
```

### Interactive Commands

```bash
help                              # Show available commands
exit                              # Exit interactive mode
clear                             # Clear terminal

# Provider management
provider                          # Show current provider
provider list                     # List providers
provider anthropic                # Switch provider
provider openai gpt-5             # Switch to GPT-5

# Model management
model                             # Show current model
model gpt-oss                     # Switch to reasoning model
model claude-4-5-opus             # Switch to Claude 4.5
models                            # List available models

# Tool operations
tools                             # List tools
tools --all                       # Detailed tool info
tools call                        # Interactive tool execution

# Server operations
servers                           # List servers
ping                              # Ping all servers
resources                         # List resources
prompts                           # List prompts
```

## 📄 Using Command Mode

Command mode provides Unix-friendly automation capabilities.

### Command Mode Options

```bash
--input FILE                      # Input file (- for stdin)
--output FILE                     # Output file (- for stdout)
--prompt TEXT                     # Prompt template
--tool TOOL                       # Execute specific tool
--tool-args JSON                  # Tool arguments as JSON
--system-prompt TEXT              # Custom system prompt
--raw                             # Raw output without formatting
--single-turn                     # Disable multi-turn conversation
--max-turns N                     # Maximum conversation turns
```

### Examples

```bash
# Text processing with reasoning models
echo "Analyze this data" | mcp-cli cmd --server sqlite --model gpt-oss --input - --output analysis.txt

# Use GPT-5 for complex analysis
mcp-cli cmd --server sqlite --provider openai --model gpt-5 --prompt "Provide strategic analysis" --input report.txt

# Tool execution
mcp-cli cmd --server sqlite --tool list_tables --raw

# Complex queries
mcp-cli cmd --server sqlite --tool read_query --tool-args '{"query": "SELECT COUNT(*) FROM users"}'

# Batch processing with GNU Parallel
ls *.txt | parallel mcp-cli cmd --server sqlite --input {} --output {}.summary --prompt "Summarize: {{input}}"
```

## 🔧 Provider Configuration

### Ollama Configuration (Default)

Ollama runs locally by default on `http://localhost:11434`. MCP CLI v0.11.1+ with CHUK-LLM v0.16+ includes **llama.cpp integration** that automatically discovers and reuses Ollama's downloaded models for 1.53x faster inference (311 vs 204 tokens/sec) without re-downloading.

To use reasoning and other models:

```bash
# Pull reasoning and other models for Ollama
ollama pull gpt-oss          # Default reasoning model
ollama pull llama3.3         # Latest Llama
ollama pull llama3.2         # Llama 3.2
ollama pull qwen3            # Qwen 3
ollama pull qwen2.5-coder    # Coding-focused
ollama pull deepseek-coder   # DeepSeek coder
ollama pull granite3.3       # IBM Granite
ollama pull mistral          # Mistral
ollama pull gemma3           # Google Gemma
ollama pull phi3             # Microsoft Phi
ollama pull codellama        # Code Llama

# List available Ollama models
ollama list

# Configure remote Ollama server
mcp-cli provider set ollama api_base http://remote-server:11434
```

### Cloud Provider Configuration

To use cloud providers with advanced models, configure API keys:

```bash
# Configure OpenAI (for GPT-5, GPT-4, O3 models)
mcp-cli provider set openai api_key sk-your-key-here

# Configure Anthropic (for Claude 4.5, Claude 3.5)
mcp-cli provider set anthropic api_key sk-ant-your-key-here

# Configure Azure OpenAI (for enterprise GPT-5)
mcp-cli provider set azure_openai api_key sk-your-key-here
mcp-cli provider set azure_openai api_base https://your-resource.openai.azure.com

# Configure other providers
mcp-cli provider set gemini api_key your-gemini-key
mcp-cli provider set groq api_key your-groq-key

# Test configuration
mcp-cli provider diagnostic openai
mcp-cli provider diagnostic anthropic
```

### Custom OpenAI-Compatible Providers

MCP CLI supports adding custom OpenAI-compatible providers (LocalAI, custom proxies, etc.):

```bash
# Add a custom provider (persisted across sessions)
mcp-cli provider add localai http://localhost:8080/v1 gpt-4 gpt-3.5-turbo
mcp-cli provider add myproxy https://proxy.example.com/v1 custom-model-1 custom-model-2

# Set API key via environment variable (never stored in config)
export LOCALAI_API_KEY=your-api-key
export MYPROXY_API_KEY=your-api-key

# List custom providers
mcp-cli provider custom

# Use custom provider
mcp-cli --provider localai --server sqlite
mcp-cli --provider myproxy --model custom-model-1 --server sqlite

# Remove custom provider
mcp-cli provider remove localai

# Runtime provider (session-only, not persisted)
mcp-cli --provider temp-ai --api-base https://api.temp.com/v1 --api-key test-key --server sqlite
```

**Security Note**: API keys can be stored securely in OS-native keychains (macOS Keychain, Windows Credential Manager, Linux Secret Service) or HashiCorp Vault using the token management system. Alternatively, use environment variables following the pattern `{PROVIDER_NAME}_API_KEY` or pass via `--api-key` for session-only use. See [Token Management](docs/TOKEN_MANAGEMENT.md) for details.

### Manual Configuration

The `chuk_llm` library configuration in `~/.chuk_llm/config.yaml`:

```yaml
ollama:
  api_base: http://localhost:11434
  default_model: gpt-oss

openai:
  api_base: https://api.openai.com/v1
  default_model: gpt-5

anthropic:
  api_base: https://api.anthropic.com
  default_model: claude-4-5-opus

azure_openai:
  api_base: https://your-resource.openai.azure.com
  default_model: gpt-5

gemini:
  api_base: https://generativelanguage.googleapis.com
  default_model: gemini-2.0-flash

groq:
  api_base: https://api.groq.com
  default_model: llama-3.1-70b
```

API keys can be provided via:
1. **Secure token storage** (recommended) - Stored in OS keychain/Vault, see [Token Management](docs/TOKEN_MANAGEMENT.md)
2. **Environment variables** - Export in your shell or add to `~/.chuk_llm/.env`:

```bash
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
AZURE_OPENAI_API_KEY=sk-your-azure-key-here
GEMINI_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key
```

3. **Command-line** - Pass `--api-key` for session-only use (not persisted)

## 📂 Server Configuration

MCP CLI supports two types of server configurations:

1. **Project Servers** (`server_config.json`): Shared project-level configurations
2. **User Servers** (`~/.mcp-cli/preferences.json`): Personal runtime-added servers that persist across sessions

### Configuration File Discovery

MCP CLI searches for `server_config.json` in the following priority order:

1. **Explicit path** via `--config-file` option:
   ```bash
   mcp-cli --config-file /path/to/custom-config.json
   ```

2. **Current directory** - Automatically detected when running from a project directory:
   ```bash
   cd /path/to/my-project
   mcp-cli --server sqlite    # Uses ./server_config.json if it exists
   ```

3. **Bundled default** - When running via `uvx` or from any directory without a local config:
   ```bash
   uvx mcp-cli --server cloudflare_workers    # Uses packaged server_config.json
   ```

This means you can:
- **Override per-project**: Place a `server_config.json` in your project directory with project-specific server configurations
- **Use defaults globally**: Run `uvx mcp-cli` from anywhere and get the bundled default servers
- **Customize explicitly**: Use `--config-file` to specify any configuration file location

### Bundled Default Servers

MCP CLI v0.11.1+ comes with an expanded set of pre-configured servers in the bundled `server_config.json`:

| Server | Type | Description | Configuration |
|--------|------|-------------|---------------|
| **sqlite** | STDIO | SQLite database operations | `uvx mcp-server-sqlite --db-path test.db` |
| **echo** | STDIO | Echo server for testing | `uvx chuk-mcp-echo stdio` |
| **math** | STDIO | Mathematical computations | `uvx chuk-mcp-math-server` |
| **playwright** | STDIO | Browser automation | `npx @playwright/mcp@latest` |
| **brave_search** | STDIO | Web search via Brave API | Requires `BRAVE_API_KEY` token |
| **notion** | HTTP | Notion workspace integration | `https://mcp.notion.com/mcp` (OAuth) |
| **cloudflare_workers** | HTTP | Cloudflare Workers bindings | `https://bindings.mcp.cloudflare.com/mcp` (OAuth) |
| **monday** | HTTP | Monday.com integration | `https://mcp.monday.com/mcp` (OAuth) |
| **linkedin** | HTTP | LinkedIn integration | `https://linkedin.chukai.io/mcp` |
| **weather** | HTTP | Weather data service | `https://weather.chukai.io/mcp` |

**Note**: HTTP servers and API-based servers require authentication. Use the [Token Management](docs/TOKEN_MANAGEMENT.md) system to configure access tokens.

To use these servers:
```bash
# Use bundled servers from anywhere
uvx mcp-cli --server sqlite
uvx mcp-cli --server echo
uvx mcp-cli --server math
uvx mcp-cli --server playwright

# API-based servers require tokens
mcp-cli token set brave_search --type bearer
uvx mcp-cli --server brave_search

# HTTP/OAuth servers require OAuth authentication
uvx mcp-cli token set notion --oauth
uvx mcp-cli --server notion

# Use multiple servers simultaneously
uvx mcp-cli --server sqlite,math,playwright
```

### Project Configuration

Create a `server_config.json` file with your MCP server configurations:

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "python",
      "args": ["-m", "mcp_server.sqlite_server"],
      "env": {
        "DATABASE_PATH": "database.db"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"],
      "env": {}
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@brave/brave-search-mcp-server"],
      "env": {
        "BRAVE_API_KEY": "${TOKEN:bearer:brave_search}"
      }
    },
    "notion": {
      "url": "https://mcp.notion.com/mcp",
      "headers": {
        "Authorization": "Bearer ${TOKEN:bearer:notion}"
      }
    }
  }
}
```

### Secure Token Replacement

MCP CLI supports automatic token replacement from secure storage using the `${TOKEN:namespace:name}` syntax:

**Syntax**: `${TOKEN:<namespace>:<token-name>}`

**Examples**:
```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@brave/brave-search-mcp-server"],
      "env": {
        "BRAVE_API_KEY": "${TOKEN:bearer:brave_search}"
      }
    },
    "api-server": {
      "url": "https://api.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${TOKEN:bearer:my_api}",
        "X-API-Key": "${TOKEN:api-key:my_service}"
      }
    }
  }
}
```

**Token Storage**:
```bash
# Store tokens securely (never in config files!)
mcp-cli token set brave_search --type bearer
# Enter token value when prompted (hidden input)

mcp-cli token set my_api --type bearer --value "your-token-here"

# Tokens are stored in OS-native secure storage:
# - macOS: Keychain
# - Windows: Credential Manager
# - Linux: Secret Service (GNOME Keyring/KWallet)
```

**Supported Locations**:
- `env`: Environment variables for STDIO servers
- `headers`: HTTP headers for HTTP/SSE servers

**Namespaces**:
- `bearer`: Bearer tokens (default for `--type bearer`)
- `api-key`: API keys (default for `--type api-key`)
- `oauth`: OAuth tokens (automatic)
- `generic`: Custom tokens

**Benefits**:
- ✅ Never store API keys in config files
- ✅ Share `server_config.json` safely (no secrets)
- ✅ Tokens encrypted in OS-native secure storage
- ✅ Works across all transport types (STDIO, HTTP, SSE)

See [Token Management Guide](docs/TOKEN_MANAGEMENT.md) for complete documentation.

### Runtime Server Management

Add servers dynamically during runtime without editing configuration files:

```bash
# Add STDIO servers (most common)
mcp-cli
> /server add sqlite stdio uvx mcp-server-sqlite --db-path mydata.db
> /server add playwright stdio npx @playwright/mcp@latest
> /server add time stdio uvx mcp-server-time

# Add HTTP servers with authentication
> /server add github --transport http --header "Authorization: Bearer ghp_token" -- https://api.github.com/mcp
> /server add myapi --transport http --env API_KEY=secret -- https://api.example.com/mcp

# Add SSE (Server-Sent Events) servers
> /server add events --transport sse -- https://events.example.com/sse

# Manage servers
> /server list                     # Show all servers
> /server disable sqlite           # Temporarily disable
> /server enable sqlite            # Re-enable
> /server remove myapi             # Remove user-added server
```

**Key Points:**
- User-added servers persist in `~/.mcp-cli/preferences.json`
- Survive application restarts
- Can be enabled/disabled without removal
- Support STDIO, HTTP, and SSE transports
- Environment variables and headers for authentication

## 📈 Advanced Usage Examples

### Reasoning Model Comparison

```bash
# Compare reasoning across different models
> /provider ollama
> /model gpt-oss
> Think through this problem step by step: If a train leaves New York at 3 PM...
[See the complete thinking process with gpt-oss]

> /provider openai
> /model gpt-5
> Think through this problem step by step: If a train leaves New York at 3 PM...
[See GPT-5's reasoning approach]

> /provider anthropic
> /model claude-4-5-opus
> Think through this problem step by step: If a train leaves New York at 3 PM...
[See Claude 4.5's analytical process]
```

### Local-First Workflow with Reasoning

```bash
# Start with default Ollama/gpt-oss (no API key needed)
mcp-cli chat --server sqlite

# Use reasoning model for complex problems
> Think through this database optimization problem step by step
[gpt-oss shows its complete thinking process before answering]

# Try different local models for different tasks
> /model llama3.3              # General purpose
> /model qwen2.5-coder         # For coding tasks
> /model deepseek-coder        # Alternative coding model
> /model granite3.3            # IBM's model
> /model gpt-oss               # Back to reasoning model

# Switch to cloud when needed (requires API keys)
> /provider openai
> /model gpt-5
> Complex enterprise architecture design...

> /provider anthropic
> /model claude-4-5-opus
> Detailed strategic analysis...

> /provider ollama
> /model gpt-oss
> Continue with local processing...
```

### Multi-Provider Workflow

```bash
# Start with local reasoning (default, no API key)
mcp-cli chat --server sqlite

# Compare responses across providers
> /provider ollama
> What's the best way to optimize this SQL query?

> /provider openai gpt-5        # Requires API key
> What's the best way to optimize this SQL query?

> /provider anthropic claude-4-5-sonnet  # Requires API key
> What's the best way to optimize this SQL query?

# Use each provider's strengths
> /provider ollama gpt-oss      # Local reasoning, privacy
> /provider openai gpt-5        # Advanced reasoning
> /provider anthropic claude-4-5-opus  # Deep analysis
> /provider groq llama-3.1-70b  # Ultra-fast responses
```

### Complex Tool Workflows with Reasoning

```bash
# Use reasoning model for complex database tasks
> /model gpt-oss
> I need to analyze our database performance. Think through what we should check first.
[gpt-oss shows thinking: "First, I should check the table structure, then indexes, then query patterns..."]
[Tool: list_tables] → products, customers, orders

> Now analyze the indexes and suggest optimizations
[gpt-oss thinks through index analysis]
[Tool: describe_table] → Shows current indexes
[Tool: read_query] → Analyzes query patterns

> Create an optimization plan based on your analysis
[Complete reasoning process followed by specific recommendations]
```

### Automation and Scripting

```bash
# Batch processing with different models
for file in data/*.csv; do
  # Use reasoning model for analysis
  mcp-cli cmd --server sqlite \
    --model gpt-oss \
    --prompt "Analyze this data and think through patterns" \
    --input "$file" \
    --output "analysis/$(basename "$file" .csv)_reasoning.txt"
  
  # Use coding model for generating scripts
  mcp-cli cmd --server sqlite \
    --model qwen2.5-coder \
    --prompt "Generate Python code to process this data" \
    --input "$file" \
    --output "scripts/$(basename "$file" .csv)_script.py"
done

# Pipeline with reasoning
cat complex_problem.txt | \
  mcp-cli cmd --model gpt-oss --prompt "Think through this step by step" --input - | \
  mcp-cli cmd --model llama3.3 --prompt "Summarize the key points" --input - > solution.txt
```

### Performance Monitoring

```bash
# Check provider and model performance
> /provider diagnostic
Provider Diagnostics
Provider      | Status      | Response Time | Features      | Models
ollama        | ✅ Ready    | 56ms         | 📡🔧         | gpt-oss, llama3.3, qwen3, ...
openai        | ✅ Ready    | 234ms        | 📡🔧👁️      | gpt-5, gpt-4o, o3, ...
anthropic     | ✅ Ready    | 187ms        | 📡🔧         | claude-4-5-opus, claude-4-5-sonnet, ...
azure_openai  | ✅ Ready    | 198ms        | 📡🔧👁️      | gpt-5, gpt-4o, ...
gemini        | ✅ Ready    | 156ms        | 📡🔧👁️      | gemini-2.0-flash, ...
groq          | ✅ Ready    | 45ms         | 📡🔧         | llama-3.1-70b, ...

# Check available models
> /models
Models for ollama (Current Provider)
Model                | Status
gpt-oss             | Current & Default (Reasoning)
llama3.3            | Available
llama3.2            | Available
qwen2.5-coder       | Available
deepseek-coder      | Available
granite3.3          | Available
... and 6 more

# Monitor tool execution with reasoning
> /verbose
> /model gpt-oss
> Analyze the database and optimize the slowest queries
[Shows complete thinking process]
[Tool execution with timing]
```

## 🔍 Troubleshooting

### Common Issues

1. **Ollama not running** (default provider):
   ```bash
   # Start Ollama service
   ollama serve
   
   # Or check if it's running
   curl http://localhost:11434/api/tags
   ```

2. **Model not found**:
   ```bash
   # For Ollama (default), pull the model first
   ollama pull gpt-oss      # Reasoning model
   ollama pull llama3.3     # Latest Llama
   ollama pull qwen2.5-coder # Coding model
   
   # List available models
   ollama list
   
   # For cloud providers, check supported models
   mcp-cli models openai     # Shows GPT-5, GPT-4, O3 models
   mcp-cli models anthropic  # Shows Claude 4.5, Claude 3.5 models
   ```

3. **Provider not found or API key missing**:
   ```bash
   # Check available providers
   mcp-cli provider list
   
   # For cloud providers, set API keys
   mcp-cli provider set openai api_key sk-your-key
   mcp-cli provider set anthropic api_key sk-ant-your-key
   
   # Test connection
   mcp-cli provider diagnostic openai
   ```

4. **Connection issues with Ollama**:
   ```bash
   # Check Ollama is running
   ollama list
   
   # Test connection
   mcp-cli provider diagnostic ollama
   
   # Configure custom endpoint if needed
   mcp-cli provider set ollama api_base http://localhost:11434
   ```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
mcp-cli --verbose chat --server sqlite
mcp-cli --log-level DEBUG interactive --server sqlite

# Write debug logs to a rotating file (secrets are automatically redacted)
mcp-cli --log-file ~/.mcp-cli/logs/debug.log --server sqlite
```

## 🔒 Security Considerations

### Privacy & Local-First
- **Local by Default**: Ollama with gpt-oss runs locally, keeping your data private
- **No Cloud Required**: Full functionality without external API dependencies

### Token & Authentication Security
- **Secure Token Storage**: Tokens stored in OS-native credential stores (macOS Keychain, Windows Credential Manager, Linux Secret Service) under the "mcp-cli" service identifier
- **Multiple Storage Backends**: Choose between keychain, encrypted files, or HashiCorp Vault based on security requirements
- **API Keys**: Only needed for cloud providers (OpenAI, Anthropic, etc.), stored securely using token management system
- **OAuth 2.0 Support**: Secure authentication for MCP servers using PKCE and resource indicators (RFC 7636, RFC 8707)

### Log Security
- **Secret Redaction**: All log output (console and file) is automatically redacted for Bearer tokens, API keys (sk-*), OAuth access tokens, and Authorization headers
- **Rotating File Logs**: Optional `--log-file` with JSON format, 10MB rotation, and 3 backup files

### Execution Security
- **Tool Validation**: All tool calls are validated before execution
- **Timeout Protection**: Configurable timeouts prevent hanging operations (v0.13+)
- **Circuit Breakers**: Automatic failure detection and recovery to prevent cascading failures (v0.13+)
- **Server Isolation**: Each server runs in its own process
- **File Access**: Filesystem access can be disabled with `--disable-filesystem`
- **Transport Monitoring**: Automatic detection of connection failures with warnings (v0.11+)

### MCP Apps Security
- **Iframe Sandbox**: Apps run in sandboxed iframes with restricted permissions
- **Content Security Policy**: Server-supplied CSP domains are validated and sanitized
- **XSS Prevention**: Tool names and user-supplied content are HTML-escaped before template injection
- **URL Scheme Validation**: `ui/open-link` only allows `http://` and `https://` schemes
- **Tool Name Validation**: Bridge rejects tool names not matching the MCP spec character set

## 🚀 Performance Features

### LLM Provider Performance (v0.16+)
- **52x Faster Imports**: Reduced from 735ms to 14ms through lazy loading
- **112x Faster Client Creation**: Automatic thread-safe caching
- **llama.cpp Integration**: 1.53x faster inference (311 vs 204 tokens/sec) with automatic Ollama model reuse
- **Dynamic Model Discovery**: Zero overhead capability-based model selection

### Tool Execution Performance (v0.13+)
- **Production Middleware**: Timeouts, retries with exponential backoff, circuit breakers, and result caching
- **Concurrent Tool Execution**: Multiple tools can run simultaneously with proper coordination
- **Connection Health Monitoring**: Automatic detection and recovery from transport failures
- **Optimized Tool Manager**: Reduced from 2000+ to ~800 lines while maintaining all functionality

### Runtime Performance
- **Local Processing**: Default Ollama provider minimizes latency
- **Reasoning Visibility**: See AI thinking process with gpt-oss, GPT-5, Claude 4
- **Streaming Responses**: Real-time response generation
- **Connection Pooling**: Efficient reuse of client connections
- **Caching**: Tool metadata and provider configurations are cached
- **Async Architecture**: Non-blocking operations throughout

## 📦 Dependencies

Core dependencies are organized into feature groups:

- **cli**: Terminal UI and command framework (Rich, Typer, chuk-term)
- **dev**: Development tools, testing utilities, linting
- **chuk-tool-processor v0.22+**: Production-grade tool execution with middleware, multiple execution strategies, and observability
- **chuk-llm v0.17+**: Unified LLM provider with dynamic model discovery, capability-based selection, and llama.cpp integration
- **chuk-term**: Enhanced terminal UI with themes, prompts, and cross-platform support

Install with specific features:
```bash
pip install "mcp-cli[cli]"        # Basic CLI features
pip install "mcp-cli[cli,dev]"    # CLI with development tools
pip install "mcp-cli[apps]"       # MCP Apps (interactive browser UIs)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/chrishayuk/mcp-cli
cd mcp-cli
pip install -e ".[cli,dev]"
pre-commit install
```

### Demo Scripts

Explore the capabilities of MCP CLI:

```bash
# Command Mode Demos

# General cmd mode features (bash)
bash examples/cmd_mode_demo.sh

# LLM integration with cmd mode (bash)
bash examples/cmd_mode_llm_demo.sh

# Python integration example
uv run examples/cmd_mode_python_demo.py

# Custom Provider Management Demos

# Interactive walkthrough demo (educational)
uv run examples/custom_provider_demo.py

# Working demo with actual inference (requires OPENAI_API_KEY)
uv run examples/custom_provider_working_demo.py

# Simple shell script demo (requires OPENAI_API_KEY)
bash examples/custom_provider_simple_demo.sh

# Terminal management features (chuk-term)
uv run examples/ui_terminal_demo.py

# Output system with themes (chuk-term)
uv run examples/ui_output_demo.py

# Streaming UI capabilities (chuk-term)
uv run examples/ui_streaming_demo.py
```

### Running Tests

```bash
pytest
pytest --cov=mcp_cli --cov-report=html
```

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[CHUK Tool Processor](https://github.com/chrishayuk/chuk-tool-processor)** - Production-grade async tool execution with middleware and observability
- **[CHUK-LLM](https://github.com/chrishayuk/chuk-llm)** - Unified LLM provider with dynamic model discovery, llama.cpp integration, and GPT-5/Claude 4.5 support (v0.17+)
- **[CHUK-Term](https://github.com/chrishayuk/chuk-term)** - Enhanced terminal UI with themes and cross-platform support
- **[Rich](https://github.com/Textualize/rich)** - Beautiful terminal formatting
- **[Typer](https://typer.tiangolo.com/)** - CLI framework
- **[Prompt Toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit)** - Interactive input

## 🔗 Related Projects

- **[Model Context Protocol](https://modelcontextprotocol.io/)** - Core protocol specification
- **[MCP Servers](https://github.com/modelcontextprotocol/servers)** - Official MCP server implementations
- **[CHUK Tool Processor](https://github.com/chrishayuk/chuk-tool-processor)** - Production-grade tool execution with middleware and observability
- **[CHUK-LLM](https://github.com/chrishayuk/chuk-llm)** - LLM provider abstraction with dynamic model discovery, GPT-5, Claude 4.5, O3 series support, and llama.cpp integration
- **[CHUK-Term](https://github.com/chrishayuk/chuk-term)** - Terminal UI library with themes and cross-platform support