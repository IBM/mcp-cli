# MCP Apps (SEP-1865)

MCP Apps are interactive HTML UIs served by MCP servers and rendered in the user's browser via sandboxed iframes. When a tool has a `_meta.ui` annotation, mcp-cli launches a local web server that bridges the browser and the MCP backend.

## Prerequisites

```bash
# Install the apps extra (adds websockets dependency)
pip install "mcp-cli[apps]"
```

## Quick Start

```bash
# Connect to a server that provides app-enabled tools
mcp-cli --server view_demo

# Ask for something visual
> Show me the sales data as a chart

# The browser opens automatically with an interactive chart app
# Tool results are pushed to the app in real-time via WebSocket
```

Use `/tools` in chat mode to see which tools have app UIs (shown in the APP column).

## How It Works

### Tool Detection

MCP servers annotate tools with `_meta.ui` metadata indicating they have an associated UI:

```json
{
  "name": "show_chart",
  "description": "Display data as an interactive chart",
  "_meta": {
    "ui": {
      "resourceUri": "ui://view-demo/chart",
      "mediaType": "text/html"
    }
  }
}
```

When mcp-cli detects `_meta.ui` on a tool result, it automatically:

1. Fetches the HTML UI resource from the MCP server (via `resources/read` or HTTP)
2. Starts a local HTTP + WebSocket server on an available port (starting from 9470)
3. Opens the user's default browser
4. Pushes the tool result to the app once the app signals it's ready

### Architecture

```
Browser                    Python Backend                MCP Server
+-----------------+       +------------------+       +--------------+
|  Host Page (JS)  |--WS--|  AppBridge        |--MCP--|  Tool Server |
|  +-------------+ |      |  (bridge.py)      |       |              |
|  | App iframe  | |      +------------------+       +--------------+
|  | (sandboxed) | |              |
|  +-------------+ |      +------------------+
|   postMessage    |      |  AppHostServer   |
+-----------------+       |  (host.py)        |
                          +------------------+
```

**Components:**

- **`host.py` (AppHostServer)** — Manages app lifecycle: port allocation, HTTP serving (host page + app HTML), WebSocket server, browser launch
- **`host_page.py`** — JavaScript host page template; bridges iframe `postMessage` and WebSocket, handles `ui/initialize`, display modes, reconnection with exponential backoff
- **`bridge.py` (AppBridge)** — JSON-RPC protocol handler: proxies `tools/call` and `resources/read` to MCP servers, manages message queue, formats tool results per MCP spec
- **`models.py`** — Pydantic models: `AppInfo`, `AppState` (PENDING -> INITIALIZING -> READY -> CLOSED), `HostContext`

### App Lifecycle

```
PENDING -> INITIALIZING -> READY -> CLOSED
```

1. **PENDING**: App info created, port allocated
2. **INITIALIZING**: WebSocket connected, host page loaded, waiting for app to initialize
3. **READY**: App sent `ui/notifications/initialized`, tool results can be pushed
4. **CLOSED**: App teardown or browser tab closed

### Protocol Messages

**Browser -> Python (inbound):**

| Method | Description |
|--------|-------------|
| `tools/call` | Proxy a tool call to the MCP server |
| `resources/read` | Proxy a resource read to the MCP server |
| `ui/message` | App sends a message to the conversation |
| `ui/update-model-context` | App updates its model context |
| `ui/notifications/initialized` | App signals it's ready to receive data |
| `ui/notifications/teardown` | App is shutting down |

**Python -> Browser (outbound):**

| Method | Description |
|--------|-------------|
| `ui/notifications/tool-result` | Push tool result data to the app |
| `ui/notifications/tool-input` | Push tool input arguments to the app |

## Security Model

### Iframe Sandbox

Apps run in a sandboxed iframe with the following permissions:

```
allow-scripts allow-forms allow-same-origin allow-popups allow-popups-to-escape-sandbox
```

### Content Security Policy

Server-supplied CSP domains are validated against a strict regex (`^[a-zA-Z0-9\-.:/*]+$`) before being included in CSP directives. This prevents CSP injection attacks.

### XSS Prevention

- Tool names are `html.escape()`d before template injection into the host page
- All user-supplied content is sanitized at template boundaries

### URL Scheme Validation

The `ui/open-link` handler only allows `http://` and `https://` schemes, blocking `javascript:` and other dangerous schemes.

### Tool Name Validation

The bridge rejects tool names not matching `^[a-zA-Z0-9_\-./]+$` per the MCP spec.

### Safe JSON Serialization

`_safe_json_dumps()` falls back to `_to_serializable()` on `TypeError`/`ValueError`, with circular reference protection via a visited-object set.

### WebSocket Origin Validation (v0.20.1+)

The local app-host server validates the `Origin` header on every WebSocket upgrade request against the `http://localhost:<port>` (or `127.0.0.1`) host page origin, via `mcp_cli.utils.loopback_origin.is_allowed_origin()`. Browsers attach an `Origin` header to WebSocket handshakes automatically but don't enforce same-origin policy on the connection itself — enforcement has to happen server-side. Handshakes with a mismatched or missing Origin are rejected with HTTP 403 before the upgrade completes.

### Tool Permission Enforcement (v0.20.1+)

`AppInfo.permissions` (from the resource's `_meta.ui.permissions`) is enforced in `AppBridge._handle_tool_call()`: if a resource declares a `tools` allow-list, only tools on that list can be invoked via the bridge, in addition to the existing tool-name syntax check. A resource that declares no permissions keeps the previous unrestricted behavior.

### SSRF-Safe Resource Fetch (v0.20.1+)

Direct HTTP(S) fetches (for `resource_uri` or a tool result's `viewUrl`) are validated with `mcp_cli.utils.url_safety.is_safe_fetch_url()`, which resolves the hostname and rejects private, loopback, link-local, and other non-public address ranges. Redirects are followed manually so each hop is re-validated rather than trusted after the first check.

## Session Reliability

### Deferred Tool Result Delivery

Initial tool results are stored on the bridge and pushed only after the app sends `ui/notifications/initialized`. This prevents race conditions where `postMessage` is silently dropped before the app sets up its message listener.

### Message Queue

When the WebSocket is disconnected, notifications are queued in a `deque(maxlen=50)`. On reconnect, `drain_pending()` flushes queued messages.

### Reconnection

The host page uses exponential backoff for WebSocket reconnection (1s, 2s, 4s, ... capped at 30s). On successful reconnect, the backoff resets and the bridge state returns to INITIALIZING.

### Duplicate Prevention

Calling `launch_app()` for an already-running tool closes the previous instance before launching a new one. If an app is already running, new tool results are pushed to the existing bridge.

### Initialization Timeout

A configurable JavaScript timeout (default 30s) shows "App initialization timed out" in the host page status bar if the app never sends `ui/notifications/initialized`.

## Spec Compliance

- `ui/initialize` response includes protocol version, host capabilities (with sandbox details), host info, and host context
- `ui/resource-teardown` sent to iframe on `beforeunload`
- `ui/notifications/host-context-changed` sent after display mode changes
- `structuredContent` recovered from JSON text blocks when CTP transport normalization discards it

## Configuration

Default values in `src/mcp_cli/config/defaults.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_APP_HOST_PORT_START` | 9470 | Starting port for local app servers |
| `DEFAULT_APP_AUTO_OPEN_BROWSER` | True | Auto-open browser on app launch |
| `DEFAULT_APP_MAX_CONCURRENT` | 10 | Maximum concurrent MCP Apps |
| `DEFAULT_APP_TOOL_TIMEOUT` | 120.0s | Tool call timeout from an app |
| `DEFAULT_APP_INIT_TIMEOUT` | 30s | Initialization timeout (JS-side) |

## Known Limitations

- Map and video rendering quality depends on the server-side app JavaScript, not mcp-cli
- `ui/notifications/tool-input-partial` (streaming argument assembly) is not yet implemented
- HTTPS/TLS for remote deployment is not yet implemented
- CTP transport's `_normalize_mcp_response` discards `structuredContent` — recovered via text block extraction

## Examples

See `examples/apps/` for working demos:

```bash
# Full end-to-end demo (requires: pip install mcp-cli[apps])
python examples/apps/apps_demo.py

# _meta.ui pipeline — shows how metadata survives the tool pipeline
python examples/apps/meta_pipeline_demo.py

# Bridge protocol — demonstrates JSON-RPC message routing
python examples/apps/bridge_protocol_demo.py
```
