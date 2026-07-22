# mcp_cli/apps/host_page.py
"""Embedded host page HTML/JS for MCP Apps.

This module contains the HTML template that serves as the MCP Apps host.
It creates a sandboxed iframe for the app and bridges communication
between the iframe (postMessage) and the Python backend (WebSocket).
"""

from __future__ import annotations

# The template uses {var} for Python format() substitution and
# {{ / }} for literal braces in the JavaScript code.
HOST_PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MCP App — {tool_name}</title>
  <style>
    :root {{
      --bg: #1a1a2e;
      --header-bg: #16213e;
      --text: #e0e0e0;
      --accent: #0f3460;
      --status-ok: #4ecca3;
      --status-err: #e74c3c;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    html, body {{ height: 100%; }}
    body {{
      background: var(--bg); color: var(--text);
      font-family: system-ui, -apple-system, sans-serif;
      display: flex; flex-direction: column;
    }}
    #header {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 16px; background: var(--header-bg);
      border-bottom: 1px solid var(--accent); font-size: 14px;
      flex-shrink: 0;
    }}
    #header .title {{ font-weight: 600; }}
    #header .status {{ font-size: 12px; opacity: 0.7; }}
    #header .status.connected {{ color: var(--status-ok); opacity: 1; }}
    #header .status.error {{ color: var(--status-err); opacity: 1; }}
    #app-container {{
      width: 100%; flex: 1; overflow: hidden; min-height: 0;
    }}
    #app-iframe {{
      width: 100%; height: 100%; border: none; background: #fff;
    }}
  </style>
</head>
<body>
  <div id="header">
    <span class="title">{tool_name}</span>
    <span id="status" class="status">Connecting&hellip;</span>
  </div>
  <div id="app-container">
    <iframe
      id="app-iframe"
      sandbox="allow-scripts allow-forms allow-same-origin allow-popups allow-popups-to-escape-sandbox"
      src="/app"
      {csp_attr}
    ></iframe>
  </div>

<script>
(function() {{
  "use strict";

  var WS_URL   = "ws://localhost:{port}/ws";
  var statusEl = document.getElementById("status");
  var iframe   = document.getElementById("app-iframe");

  // ---- State ----
  var ws            = null;
  var initialized   = false;

  // ---- Helpers ----
  function setStatus(text, cls) {{
    statusEl.textContent = text;
    statusEl.className = "status " + (cls || "");
  }}

  function postToApp(msg) {{
    if (iframe.contentWindow) {{
      iframe.contentWindow.postMessage(msg, "*");
    }}
  }}

  // ---- WebSocket ----
  var reconnectDelay = 1000;
  var MAX_RECONNECT_DELAY = 30000;

  function connectWs() {{
    initialized = false;  // Reset on each connection attempt
    ws = new WebSocket(WS_URL);

    ws.onopen = function() {{
      reconnectDelay = 1000;  // Reset backoff on success
      setStatus("Connected", "connected");
      startInitTimer();  // Restart initialization timeout
      // Notify app of reconnection so it can re-initialize if needed
      postToApp({{ jsonrpc: "2.0", method: "ui/notifications/reconnected", params: {{}} }});
    }};

    ws.onmessage = function(ev) {{
      var msg;
      try {{ msg = JSON.parse(ev.data); }} catch(e) {{ return; }}

      // Notification from Python -> forward to app
      if (msg.method) {{
        postToApp(msg);
        return;
      }}

      // Response to a request we forwarded from the app
      if (msg.id != null) {{
        postToApp(msg);
      }}
    }};

    ws.onerror = function() {{
      setStatus("Connection error", "error");
    }};

    ws.onclose = function() {{
      setStatus("Disconnected \u2014 reconnecting\u2026", "error");
      setTimeout(function() {{
        reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
        connectWs();
      }}, reconnectDelay);
    }};
  }}

  function sendToBackend(msg) {{
    if (ws && ws.readyState === WebSocket.OPEN) {{
      ws.send(JSON.stringify(msg));
    }}
  }}

  // ---- postMessage from iframe (app) ----
  window.addEventListener("message", function(ev) {{
    // Only accept messages from the iframe
    if (ev.source !== iframe.contentWindow) return;

    var msg = ev.data;
    if (!msg || msg.jsonrpc !== "2.0") return;

    // Notification from app (no id — use == null to allow id: 0)
    if (msg.id == null && msg.method) {{
      handleAppNotification(msg);
      return;
    }}

    // Request from app (has id, including id: 0)
    if (msg.id != null && msg.method) {{
      handleAppRequest(msg);
      return;
    }}

    // Response from app (to our request)
    // e.g. response to ui/resource-teardown
  }});

  function handleAppNotification(msg) {{
    switch (msg.method) {{
      case "ui/notifications/initialized":
        initialized = true;
        clearTimeout(initTimer);
        setStatus("App ready", "connected");
        // Tell backend that app is initialized
        sendToBackend({{ jsonrpc: "2.0", method: "ui/notifications/initialized", params: {{}} }});
        break;

      case "ui/notifications/size-changed":
        // Could resize iframe; for now just forward
        sendToBackend(msg);
        break;

      case "notifications/message":
        // Log message from app
        sendToBackend(msg);
        break;

      default:
        sendToBackend(msg);
    }}
  }}

  function handleAppRequest(msg) {{
    switch (msg.method) {{
      case "ui/initialize":
        handleInitialize(msg);
        break;

      case "tools/call":
      case "resources/read":
        // Forward to Python backend
        sendToBackend(msg);
        break;

      case "ui/open-link":
        var linkUrl = (msg.params && msg.params.url) || "";
        if (!/^https?:\/\//i.test(linkUrl)) {{
          postToApp({{
            jsonrpc: "2.0", id: msg.id,
            error: {{ code: -32602, message: "Only http/https URLs are allowed" }}
          }});
          break;
        }}
        window.open(linkUrl, "_blank");
        postToApp({{ jsonrpc: "2.0", id: msg.id, result: {{}} }});
        break;

      case "ui/message":
        sendToBackend(msg);
        break;

      case "ui/update-model-context":
        sendToBackend(msg);
        postToApp({{ jsonrpc: "2.0", id: msg.id, result: {{}} }});
        break;

      case "ui/request-display-mode":
        // We support inline and fullscreen
        var mode = msg.params.mode || "inline";
        if (mode === "fullscreen") {{
          document.getElementById("header").style.display = "none";
        }} else {{
          document.getElementById("header").style.display = "flex";
        }}
        postToApp({{ jsonrpc: "2.0", id: msg.id, result: {{ mode: mode }} }});
        // Notify app of context change per MCP spec
        postToApp({{ jsonrpc: "2.0", method: "ui/notifications/host-context-changed", params: {{ displayMode: mode }} }});
        break;

      default:
        postToApp({{
          jsonrpc: "2.0", id: msg.id,
          error: {{ code: -32601, message: "Method not found: " + msg.method }}
        }});
    }}
  }}

  function handleInitialize(msg) {{
    var result = {{
      protocolVersion: "2026-01-26",
      hostCapabilities: {{
        openLinks: {{}},
        serverTools: {{ listChanged: false }},
        serverResources: {{ listChanged: false }},
        logging: {{}},
        sandbox: {{
          allowScripts: true,
          allowForms: true,
          allowSameOrigin: true,
          allowPopups: true
        }}
      }},
      hostInfo: {{ name: "mcp-cli", version: "{mcp_cli_version}" }},
      hostContext: {{
        theme: "dark",
        displayMode: "inline",
        availableDisplayModes: ["inline", "fullscreen"],
        platform: "desktop",
        locale: navigator.language || "en",
        timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        userAgent: navigator.userAgent
      }}
    }};

    postToApp({{ jsonrpc: "2.0", id: msg.id, result: result }});
  }}

  // ---- Teardown ----
  window.addEventListener("beforeunload", function() {{
    // Notify the app of resource teardown per MCP spec
    postToApp({{ jsonrpc: "2.0", method: "ui/resource-teardown", params: {{}} }});
    // Notify the backend
    sendToBackend({{ jsonrpc: "2.0", method: "ui/notifications/teardown", params: {{}} }});
  }});

  // ---- Initialization timeout ----
  var INIT_TIMEOUT = {init_timeout} * 1000;
  var initTimer = null;

  function startInitTimer() {{
    clearTimeout(initTimer);
    initTimer = setTimeout(function() {{
      if (!initialized) {{
        setStatus("App initialization timed out", "error");
      }}
    }}, INIT_TIMEOUT);
  }}

  // ---- Embedded mode (hide host header when inside dashboard panel) ----
  var isEmbedded = new URLSearchParams(window.location.search).has("embedded");
  if (isEmbedded) {{
    document.getElementById("header").style.display = "none";
    document.body.style.background = "transparent";

    // When embedded in a dashboard, the outer iframe starts display:none
    // (0×0 viewport).  Canvas-based apps (Chart.js, maps) that initialise
    // at 0×0 never recover because their ResizeObserver sees no further
    // change.  Fix: strip the src so the app doesn't load until we have
    // real dimensions, then restore it.
    if (document.documentElement.clientWidth === 0 && typeof ResizeObserver !== "undefined") {{
      iframe.removeAttribute("src");
      var _ro = new ResizeObserver(function() {{
        if (document.documentElement.clientWidth > 0) {{
          _ro.disconnect();
          iframe.src = "/app";
        }}
      }});
      _ro.observe(document.documentElement);
    }}
  }}

  // ---- Boot ----
  connectWs();
  startInitTimer();

}})();
</script>
</body>
</html>"""
