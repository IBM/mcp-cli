# Dashboard — Real-Time Browser UI

The dashboard provides a real-time browser interface that runs alongside chat mode, giving you visual feedback on conversations, tool execution, plans, and more.

## Quick Start

```bash
# Launch chat with the dashboard
mcp-cli --server sqlite --dashboard

# With a specific model
mcp-cli --server sqlite --provider openai --model gpt-5 --dashboard
```

A browser tab opens automatically at `http://localhost:<port>` with the dashboard UI.

## Views

The dashboard is a tabbed interface with five views:

### Agent Terminal

The primary conversation view — a chat interface mirroring the CLI terminal.

- **Message bubbles**: User messages (right-aligned) and assistant messages (left-aligned) with markdown rendering and syntax highlighting
- **Streaming tokens**: Watch assistant responses appear token-by-token in real time
- **Attachment rendering**: Image thumbnails, expandable text previews, audio players, and file badges (see below)
- **Chat input**: Type messages and press Enter to send — same as the CLI
- **"+" attach button**: Click to attach files from the browser (see [Attachments](./ATTACHMENTS.md))
- **Drag-and-drop**: Drag files onto the chat area to attach them
- **Clipboard paste**: Paste screenshots or images directly into the chat input
- **Search**: Ctrl/Cmd+F to search through conversation history

### Activity Stream

A chronological log of all chat-engine events:

- **Tool calls**: Shows tool name, arguments, server, and timing
- **Tool results**: Shows success/failure status, result previews, and errors
- **Reasoning steps**: Displays the model's thinking/reasoning content when using reasoning models
- **User attachments**: Shows file badges with names and sizes when attachments are included
- **State transitions**: Agent state changes (idle, thinking, tool_calling, streaming)

### Plan Viewer

Visual rendering of execution plans (requires `--plan-tools`):

- **DAG visualization**: Shows plan steps with dependency arrows and parallel markers
- **Real-time progress**: Step status updates as the plan executes (pending, running, completed, failed)
- **Step details**: Click a step to see its tool call, arguments, and result

### Tool Registry

Browse all discovered MCP tools:

- **Tool list**: Name, description, server, and parameter schema
- **Execute from browser**: Trigger tool execution directly from the dashboard
- **Dynamic discovery**: New tools from `_meta.ui` annotations appear automatically

### Config Panel

View and modify runtime configuration:

- **Provider/model**: See current provider and model, switch to a different one
- **System prompt**: View and edit the system prompt
- **Server list**: Connected MCP servers and their status

## Architecture

```
Browser                          Python Backend
┌─────────────────────┐         ┌──────────────────────┐
│  shell.html         │◄──WS──►│  server.py (HTTP+WS) │
│  ├─ agent-terminal  │         │  └─ router.py        │
│  ├─ activity-stream │         │     └─ bridge.py     │
│  ├─ plan-viewer     │         │        └─ ChatContext │
│  ├─ tool-registry   │         └──────────────────────┘
│  └─ config-panel    │
└─────────────────────┘
```

- **`server.py`** — HTTP server for static files + WebSocket endpoint on a single port
- **`shell.html`** — Host page that manages view iframes, tab switching, and WebSocket connection
- **`router.py`** — Routes messages between views and agent bridges (supports multi-agent)
- **`bridge.py`** — Integration layer between the chat engine and browser clients

### Message Flow

1. User types in agent-terminal → `postMessage` to shell.html
2. Shell.html forwards via WebSocket → server.py → router.py → bridge.py
3. Bridge puts message into `_input_queue` → chat loop picks it up
4. Chat engine processes and broadcasts events back through the bridge
5. Bridge sends WebSocket envelopes → shell.html → views update

### WebSocket Protocol

All messages use a standard envelope:

```json
{
  "protocol": "mcp-dashboard",
  "version": 2,
  "type": "MESSAGE_TYPE",
  "payload": { ... }
}
```

Key message types:
- `CONVERSATION_MESSAGE` — Chat messages (role, content, attachments)
- `AGENT_STATE` — State transitions (idle, thinking, tool_calling, streaming)
- `TOOL_RESULT` — Tool execution results
- `STREAMING_TOKEN` — Individual streaming tokens
- `CONVERSATION_HISTORY` — Full conversation replay on connect
- `ACTIVITY_HISTORY` — Activity stream replay on connect

## Session Replay

When a browser connects (or reconnects), the dashboard automatically replays:

1. **Conversation history** — All messages with their attachments, so the chat view is fully populated
2. **Activity history** — Tool calls, results, reasoning steps, and user attachment events

This means you can refresh the browser at any time without losing context.

## File Attachments in the Dashboard

The dashboard supports attaching files directly from the browser:

- **"+" button**: Click to open a file picker
- **Drag-and-drop**: Drag files onto the chat area
- **Clipboard paste**: Paste images from the clipboard

Staged files appear as removable badges above the chat input. When sent, files are base64-encoded and transmitted over WebSocket to the bridge, which stages them on `ChatContext.attachment_staging`. The existing chat loop drains staging and processes attachments — the same pipeline used by the CLI `/attach` command.

Attachments render in message bubbles as:
- **Images <100KB**: Inline thumbnail previews (clickable)
- **Images >100KB**: Metadata badge (name + size)
- **Text files**: Expandable code preview (first 2000 chars)
- **Audio**: HTML5 audio player

See [ATTACHMENTS.md](./ATTACHMENTS.md) for full attachment documentation.

## Multi-Agent Support

The dashboard architecture supports multiple agents via the router. Each agent gets its own bridge instance and can have a dedicated agent terminal view. See `src/mcp_cli/dashboard/MULTI_AGENT_SPEC.md` for the multi-agent specification.
