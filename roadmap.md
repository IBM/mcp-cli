# MCP-CLI Roadmap

> **Vision:** Transform mcp-cli from an interactive MCP runtime into a programmable AI operations environment — the neutral runtime where agent capabilities live.

### The Stack

```
Model
  ↓
Skill    (what to do — portable behaviour)
  ↓
Plan     (how to do it — execution graph)
  ↓
Tools    (do it — MCP servers)
  ↓
World
```

Today most systems jump straight from model → tools. Skills become the **stable abstraction layer**. Plans make execution **repeatable**. MCP-CLI is the runtime that binds them.

### The Ecosystem Gap

| Layer               | Status                                |
|---------------------|---------------------------------------|
| Models              | Interchangeable                       |
| Tools (MCP servers) | Standardized                          |
| Agents              | Ad-hoc                                |
| **Skills**          | **Fragmented (Claude/Codex proprietary)** |

The missing piece: a **portable capability layer between prompts and tools**. If mcp-cli becomes the neutral runtime for that layer, it becomes the distribution platform for agent behaviour — Docker for AI capabilities.

---

## Tier 1: Foundation (Fix What Breaks Today) ✅ COMPLETE

### 1.1 Truncate Large Tool Results

**Problem:** Tool results stored in full in conversation history with no size limit. A single maritime API result can be 100K+ chars, causing 1.87M token overflow.

**Files:** `src/mcp_cli/config/defaults.py`, `src/mcp_cli/chat/tool_processor.py`

- Add `DEFAULT_MAX_TOOL_RESULT_CHARS = 100_000` (~25K tokens)
- Add `_truncate_tool_result(content, max_chars)` to `ToolProcessor`
  - Head + truncation notice + tail (preserves value binding `**RESULT: $vN = ...**`)
  - `max_chars=0` disables
- Call in `_add_tool_result_to_history()` before creating the Message
- Safe: value binding extraction uses raw result object, not formatted string

### 1.2 Strip Old Reasoning Content

**Problem:** Thinking models (kimi-k2.5, DeepSeek) produce 100K+ chars of reasoning per turn, all sent back on every subsequent API call.

**File:** `src/mcp_cli/chat/conversation.py`

- Add `_prepare_messages_for_api()` — replaces inline message serialization
- Add `_strip_old_reasoning_content(messages)` — keep only most recent reasoning
- Update 3 call sites in `_handle_streaming_completion` and `_handle_regular_completion`

### 1.3 Conversation History Sliding Window

**Problem:** `conversation_history` reads ALL events without limit. No eviction or compression.

**File:** `src/mcp_cli/chat/chat_context.py`

- Add `max_history_messages` parameter (default ~200)
- Keep system prompt + last N messages
- Warn when approaching threshold
- Summarize evicted messages into compact "earlier context" block

### 1.4 Enable Infinite Context Mode

**Problem:** SessionManager has `infinite_context`, context packing, and auto-summarization — all disabled. ChatContext hardcodes `infinite_context=False`.

**Files:** `src/mcp_cli/chat/chat_context.py`, `src/mcp_cli/chat/chat_handler.py`

- Make `infinite_context` configurable (CLI flag or per-provider default)
- Configure `token_threshold` and `max_turns_per_segment` per model
- Leverage SessionManager's built-in context packing

### 1.5 Streaming Buffer Caps

**Problem:** `StreamingState.accumulated_content` and `reasoning_content` are unbounded with O(n^2) concat.

**File:** `src/mcp_cli/display/models.py`

- Add `max_accumulated_size` cap (1MB) to StreamingState
- Chunk count limit for stalled stream detection
- Use list-join pattern instead of string concatenation

---

## Tier 2: Efficiency & Resilience ✅ COMPLETE

### 2.1 Eliminate Triple Tool Result Storage

**Problem:** Tool results stored 3x: SessionManager events, `tool_history` list, procedural memory.

**Files:** `src/mcp_cli/chat/tool_processor.py`, `src/mcp_cli/chat/chat_context.py`

- Remove `self.context.tool_history: list[ToolExecutionRecord]` (unbounded in-memory list)
- Read from `tool_memory.memory.tool_log[-limit:]` on demand
- Single source of truth: SessionManager for flow, procedural memory for learning

### 2.2 Enforce Procedural Memory Limits

**Problem:** `max_patterns_per_tool` exists but is never enforced. Patterns grow unbounded.

- Enforce with LRU eviction
- Store only first 100 chars of result, not full payloads
- Add per-session memory cap

### 2.3 System Prompt Optimization

**Problem:** With 50+ tools, system prompt includes all tool names. Rebuilds on every model change.

**Files:** `src/mcp_cli/chat/system_prompt.py`, `src/mcp_cli/chat/chat_context.py`

- For 20+ tool servers, show categories + "... and X more"
- Cache system prompt with dirty flag (only regenerate when tools change)

### 2.4 Stale Connection Recovery

**Problem:** No health checks for long-lived MCP connections. Server dies = permanent failure.

**File:** `src/mcp_cli/tools/manager.py`

- Health check on tool execution failure
- Automatic reconnection with backoff
- `--reconnect-on-failure` CLI flag

### 2.5 Tool Batch Timeout

**Problem:** If one tool hangs in a parallel batch, everything blocks. No global timeout envelope.

**File:** `src/mcp_cli/tools/execution.py`

- Wrap task gathering in `asyncio.wait_for` with configurable timeout
- Cancel remaining tasks on timeout

### 2.6 Narrower Exception Handlers

**Problem:** Broad `except Exception` masks critical issues. Inconsistent `CancelledError` handling.

**Files:** `src/mcp_cli/chat/conversation.py`, `src/mcp_cli/chat/streaming_handler.py`

- Catch specific exceptions (APIError, TimeoutError, ValueError)
- Consistent `asyncio.CancelledError` handling across all async loops

### 2.7 Provider Validation

**Problem:** Invalid API keys accepted silently. Chat fails only after user starts typing.

**Files:** `src/mcp_cli/model_management/model_manager.py`, `src/mcp_cli/chat/chat_handler.py`

- Quick auth validation in `ChatContext.initialize()`
- Optional connection test on `add_runtime_provider()`

### 2.8 LLM-Visible Context Management Notices

**Problem:** Tier 1 truncation, sliding window eviction, and reasoning stripping happen silently. The LLM sees truncated results but doesn't know data was removed, so it can't adjust its strategy (e.g., request smaller date ranges, fewer fields, or paginated results).

**Files:** `src/mcp_cli/chat/tool_processor.py`, `src/mcp_cli/chat/conversation.py`, `src/mcp_cli/chat/chat_context.py`

- **Tool result truncation notice:** When `_truncate_tool_result` fires, inject a system-level hint into the conversation: `"The previous tool result was truncated from {N} to {M} chars. Consider requesting less data (smaller date range, fewer fields, pagination)."`
- **Sliding window eviction notice:** When messages are evicted, inject: `"Context window: {N} older messages were evicted. Key context may need to be re-established."`
- **Reasoning stripping notice:** When old reasoning is stripped, inject a compact note so the model knows it lost its earlier chain of thought
- **Context compaction notice:** When SessionManager compacts/summarizes, surface the summary to the model so it knows what was compressed
- Design as injectable system messages placed just before the next API call, not permanently stored in history
- Configurable: `--context-notices / --no-context-notices` flag (default on)

---

## MCP Apps (SEP-1865) ✅ COMPLETE

Interactive HTML UIs served by MCP servers, rendered in sandboxed browser iframes.

### Implementation

- **Host/Bridge/HostPage architecture:** Local websockets server per app, iframe sandbox with postMessage ↔ WebSocket bridge
- **Tool meta propagation:** `_meta.ui` on tool definitions triggers automatic app launch on tool call
- **structuredContent recovery:** Extracts `structuredContent` from JSON text blocks when CTP transport discards it
- **Security hardening:** XSS prevention (html.escape), CSP domain sanitization, tool name validation, URL scheme validation
- **Session reliability:** Message queue with drain-on-reconnect, exponential backoff, state reset, duplicate prevention, push-to-existing-app
- **Spec compliance:** `ui/initialize`, `ui/resource-teardown`, `ui/notifications/host-context-changed`, sandbox capabilities
- **Robustness:** Tool execution timeout, safe JSON serialization, circular reference protection, initialization timeout
- **Test coverage:** 96 tests across bridge, host, security, session, models, and meta propagation

### Known Limitations

- Map and video views depend on server-side app JavaScript (not an mcp-cli issue)
- `ui/notifications/tool-input-partial` (streaming argument assembly) deferred to future work
- HTTPS/TLS for remote deployment not yet implemented
- CTP transport `_normalize_mcp_response` discards `structuredContent` — recovered via text block extraction

---

## Tier 3: Performance & Polish ✅ COMPLETE

### 3.1 Tool Lookup Index

**Problem:** `get_tool_by_name()` is O(n) per call — iterates all tools.

**File:** `src/mcp_cli/tools/manager.py`

- `_tool_index: dict[str, ToolInfo]` with lazy build on first access → O(1) lookups
- Dual-key indexing: fully qualified name + simple name
- Invalidated via `_invalidate_caches()` when tools change

### 3.2 Cache Tool Metadata Per Provider

**Problem:** `get_tools_for_llm()` re-filters and re-validates every call.

**File:** `src/mcp_cli/tools/manager.py`

- `_llm_tools_cache: dict[str, list[dict]]` keyed by provider name
- Returns cached tools on hit, rebuilds on miss
- Invalidated alongside tool index on tool state changes
- Bypassed when `MCP_CLI_DYNAMIC_TOOLS=1`

### 3.3 Startup Progress

**Problem:** No feedback during server init. Blank screen if >2s.

**Files:** `src/mcp_cli/tools/manager.py`, `src/mcp_cli/chat/chat_handler.py`, `src/mcp_cli/chat/chat_context.py`

- `on_progress` callback passed through initialization chain
- Reports: "Loading server configuration...", "Connecting to N server(s)...", "Discovering tools...", "Adapting N tools for {provider}..."
- Chat handler wires callback to `output.info()` for real-time display

### 3.4 Token & Cost Tracking

**Problem:** Zero visibility into token usage or API costs.

**Files:** `src/mcp_cli/chat/token_tracker.py`, `src/mcp_cli/commands/usage/usage.py`

- `TokenTracker` with `TurnUsage` Pydantic models
- Per-turn input/output tracking with chars/4 estimation fallback
- `/usage` command (aliases: `/tokens`, `/cost`) for cumulative display
- Integrated with conversation export

### 3.5 Session Persistence & Resume

**Problem:** Conversations lost on exit. No way to resume.

**Files:** `src/mcp_cli/chat/session_store.py`, `src/mcp_cli/commands/sessions/sessions.py`

- File-based persistence at `~/.mcp-cli/sessions/`
- `/sessions list`, `/sessions save`, `/sessions load <id>`, `/sessions delete <id>`
- Auto-save every 10 turns via `auto_save_check()` in ChatContext

### 3.6 Conversation Export

**Files:** `src/mcp_cli/commands/export/export.py`, `src/mcp_cli/chat/exporters.py`

- `/export markdown [filename]` and `/export json [filename]`
- Includes tool calls with arguments, tool results (truncated), token usage metadata
- Markdown: formatted sections by role; JSON: structured with version and timestamps

---

## Tier 4: Code Quality ✅ COMPLETE

### 4.1 Remove Vestigial Global ToolManager

**Files:** `src/mcp_cli/tools/manager.py`, `src/mcp_cli/run_command.py`

- Deleted `_GLOBAL_TOOL_MANAGER`, `get_tool_manager()`, `set_tool_manager()` from manager.py
- Removed `set_tool_manager` import and both call sites from run_command.py
- ToolManager is now injected via constructors everywhere (zero external call sites for the global)

### 4.2 Rename Local Message Class

**Files:** `src/mcp_cli/chat/models.py`, `src/mcp_cli/chat/chat_context.py`, `src/mcp_cli/chat/response_models.py`

- Renamed `class Message` → `class HistoryMessage` in models.py with updated docstring
- Added backward-compat alias `Message = HistoryMessage` (existing imports still work)
- Updated chat_context.py to import and use `HistoryMessage` in all type annotations
- Updated `ToolProcessorContext` protocol to use `HistoryMessage`
- Added clarifying comments to response_models.py distinguishing the two Message classes

### 4.3 Standardize Logging in Core

**Files:** `src/mcp_cli/chat/conversation.py`, `src/mcp_cli/chat/tool_processor.py`, `src/mcp_cli/chat/chat_context.py`

- Replaced ~30 `output.*` calls with `log.*` across all three core modules
- Removed `from chuk_term.ui import output` imports from all three files
- Core modules now use `logging` only; UI modules continue to use `output` for user-facing messages

### 4.4 Integration Tests

**Files:** `tests/integration/conftest.py`, `tests/integration/test_echo_roundtrip.py`

- Created integration test framework with `@pytest.mark.integration` marker
- `tool_manager_sqlite` fixture: real ToolManager with SQLite MCP server
- Tests: tool lifecycle (discover tools, get server info), tool execution (list_tables, read_query), LLM tool adaptation (OpenAI format validation)
- Graceful skip when server unavailable

### 4.5 Coverage Reporting

**File:** `pyproject.toml`

- Added `[tool.coverage.run]` and `[tool.coverage.report]` sections
- Branch coverage enabled, `fail_under = 60` (conservative start, ratchet up)
- Standard exclusions: `pragma: no cover`, `TYPE_CHECKING`, `__main__`, `@overload`

---

## Tier 5: Production Hardening ✅ COMPLETE

### 5.1 Structured File Logging + Secret Redaction

**Files:** `src/mcp_cli/config/logging.py`, `src/mcp_cli/config/defaults.py`, `src/mcp_cli/main.py`

- Added `SecretRedactingFilter` with 5 regex patterns: Bearer tokens, sk-* API keys, api_key values, OAuth access_tokens, Authorization headers
- Filter always active on console handler (not just file logging)
- Added optional `RotatingFileHandler` via `--log-file` CLI option
- File handler: JSON format, DEBUG level, 10MB rotation with 3 backups, secret redaction
- Added `DEFAULT_LOG_DIR`, `DEFAULT_LOG_MAX_BYTES`, `DEFAULT_LOG_BACKUP_COUNT` to defaults.py
- 16 tests in `tests/config/test_logging_redaction.py`

### 5.2 Server Health Monitoring ✅ COMPLETE

**Files:** `src/mcp_cli/tools/manager.py`, `src/mcp_cli/commands/servers/health.py`, `src/mcp_cli/chat/conversation.py`, `src/mcp_cli/main.py`

- **Health-check-on-failure**: `ToolManager.execute_tool()` detects connection errors via `_is_connection_error()`, runs `_diagnose_server()` to enrich error messages with server status
- **`/health` command**: New `HealthCommand` checks one or all servers via `tool_manager.check_server_health()`, shows status (healthy/unhealthy/timeout/error) and latency
- **Background health polling**: `ConversationProcessor._health_poll_loop()` runs at `--health-interval` seconds, logs status transitions (e.g. healthy → unhealthy)
- **`--health-interval` CLI flag**: Enables background polling (0 = disabled, default)
- **Note**: Server *reconnect* still requires upstream `StreamManager` hooks; health *monitoring* is complete

### 5.3 Per-Server Configuration

**Files:** `src/mcp_cli/config/server_models.py`, `src/mcp_cli/tools/config_loader.py`, `src/mcp_cli/tools/manager.py`

- Added `tool_timeout` and `init_timeout` fields to HTTPServerConfig, STDIOServerConfig, UnifiedServerConfig, ServerConfigInput
- Updated `detect_server_types()` to read timeout fields from config
- Added `_get_server_timeout()` helper to ToolManager: per-server → global → default resolution
- Updated `execute_tool()` to use per-server timeout when available

### 5.4 Thread-Safe OAuth

**Files:** `src/mcp_cli/tools/manager.py`, `tests/tools/test_oauth_safety.py`

- Added `self._oauth_lock = asyncio.Lock()` in ToolManager `__init__`
- Wrapped `_handle_oauth_flow()` body in `async with self._oauth_lock:`
- Replaced direct dict mutation with copy-on-write for `transport.configured_headers`
- Tests: 3 concurrent OAuth flows verify lock serialization, per-server timeout resolution

---

## AI Virtual Memory Integration (Experimental) ✅ COMPLETE

OS-style virtual memory for conversation context management, powered by `chuk-ai-session-manager`.

### Implementation

- **`--vm` CLI flag**: Enables VM subsystem in SessionManager; system prompt replaced with VM-packed `developer_message` containing rules, manifest (page index), and working set content
- **`--vm-budget`**: Token budget for conversation events (system prompt uncapped on top); forces earlier page creation and eviction at low values
- **`--vm-mode`**: `passive` (runtime-managed, default), `relaxed` (VM-aware conversation), `strict` (model-driven paging with page_fault/search_pages tools)
- **Budget-aware context filtering**: `_vm_filter_events()` groups conversation events into logical turns, includes newest-first within budget, guarantees minimum 3 recent turns; evicted content preserved as VM pages in developer_message
- **`/memory` slash command** (aliases: `/vm`, `/mem`): Dashboard showing mode, turn, budget, working set utilization, page table, fault/eviction/TLB metrics; subcommands for page listing, page detail, and full stats dump
- **VM tool wiring (strict/relaxed)**: `page_fault` and `search_pages` tools injected into `openai_tools` for non-passive modes; intercepted in `tool_processor.py` before MCP guard checks and executed locally via `MemoryManager`; short-content annotation guides model to fault adjacent `[assistant]` response pages; `[user]`/`[assistant]` hint prefixes in manifest
- **E2E demo**: 8 recall scenarios (simple facts, creative content, tool results, negative case, deep detail, multi-fault, structured data, image description) with distractor tools; validates correct tool selection and content recall

### Multimodal Content Re-analysis ✅ COMPLETE

**Files:** `src/mcp_cli/chat/tool_processor.py`, `src/mcp_cli/chat/models.py`, `src/mcp_cli/commands/memory/memory.py`

- **Multi-block tool results**: `_build_page_content_blocks()` detects page modality and returns `list[dict]` (text + image_url blocks) for image pages with URLs/data URIs, or JSON string with modality/compression metadata for text/structured pages
- **HistoryMessage content type**: Extended from `str | None` to `str | list[dict[str, Any]] | None` to support OpenAI multi-block content format
- **`_add_tool_result_to_history()`**: Accepts multi-block content, skips truncation for list content
- **Compression-aware notes**: Compressed pages (ABSTRACT/REFERENCE) include a note guiding the model to re-fault at target_level=0 for full content; short pages suggest checking for the adjacent assistant response page
- **`/memory page <id> --download`**: Exports page content to `~/.mcp-cli/downloads/` with modality-aware extensions (.txt, .json, .png) and base64 data URI decoding
- **Modality metadata display**: `/memory page <id>` shows MIME type, dimensions, duration, and caption when available

### Files

| File | Change |
|------|--------|
| `src/mcp_cli/config/defaults.py` | `DEFAULT_ENABLE_VM`, `DEFAULT_VM_MODE`, `DEFAULT_VM_BUDGET` |
| `src/mcp_cli/chat/chat_context.py` | VM params in init/create, `_vm_filter_events()`, VM context in `conversation_history`, `_health_interval` |
| `src/mcp_cli/chat/chat_handler.py` | Thread `enable_vm`, `vm_mode`, `vm_budget`, `health_interval` to ChatContext |
| `src/mcp_cli/chat/conversation.py` | Background health polling (`_health_poll_loop`, `_start_health_polling`, `_stop_health_polling`) |
| `src/mcp_cli/chat/tool_processor.py` | `_build_page_content_blocks()`, multi-block `_add_tool_result_to_history()` |
| `src/mcp_cli/chat/models.py` | `HistoryMessage.content` extended to `str \| list[dict] \| None` |
| `src/mcp_cli/main.py` | `--vm`, `--vm-mode`, `--vm-budget`, `--health-interval` CLI options |
| `src/mcp_cli/tools/manager.py` | `check_server_health()`, `_diagnose_server()`, `_is_connection_error()` |
| `src/mcp_cli/commands/memory/` | `MemoryCommand` with summary/pages/page/stats/download subcommands |
| `src/mcp_cli/commands/servers/health.py` | `HealthCommand` — `/health` slash command |

---

## Tier 6: Execution Graphs & Plans ✅ COMPLETE

> **Shift:** conversation → reasoning → tools **becomes** intent → plan → execution → memory → replay
>
> **Spec:** `specs/6.0-planner-integration.md`
> **Integration:** `chuk-ai-planner>=0.2` — graph-based plan DSL, executor, LLM plan generation

### 6.0 Planner Foundation Wiring ✅

Bridge `chuk-ai-planner` to mcp-cli's MCP tool execution layer.

**Files:**
- `src/mcp_cli/planning/backends.py` — `McpToolBackend` (implements `ToolExecutionBackend` protocol, wraps `ToolManager.execute_tool()`)
- `src/mcp_cli/planning/context.py` — `PlanningContext` (state container: graph store, tool manager, plan registry, tool catalog)
- `src/mcp_cli/planning/executor.py` — `PlanRunner` (orchestrates plan execution with guard integration, dry-run, checkpointing)
- `src/mcp_cli/planning/__init__.py` — Public API

**Key integration:** chuk-ai-planner's `ToolProcessorBackend` calls `CTP.process()` for registered Python functions. `McpToolBackend` instead calls `ToolManager.execute_tool()` for real MCP server tools — same protocol interface, different execution path.

### 6.1 Plan Commands ✅

```
mcp plan create "add auth to this API"
mcp plan list
mcp plan show <id>
mcp plan run <id>
mcp plan run <id> --dry-run
mcp plan delete <id>
mcp plan resume <id>
```

**Files:**
- `src/mcp_cli/commands/plan/plan.py` — `PlanCommand` (unified command, supports CHAT + CLI + INTERACTIVE)
- `src/mcp_cli/config/enums.py` — `PlanAction` enum

**Chat mode:** `/plan create "description"`, `/plan list`, `/plan run <id>`

- Plan = persistent, inspectable execution graph (DAG of tool calls + decisions)
- Plans are serialized as JSON at `~/.mcp-cli/plans/`
- `--dry-run` shows what would execute without side effects
- Plans can be parameterized: `mcp plan run <id> --var date=2026-03-01`

### 6.2 Plan Execution with Guards ✅

Plan execution respects mcp-cli's existing guard infrastructure:

- Pre-execution: `ToolStateManager.check_all_guards()` — budget, runaway, per-tool limits
- Post-execution: `ToolStateManager.record_tool_call()` — tracking + value binding
- Step error handling: retry (via `PlanStep.max_retries`), fallback, or pause for user input
- Budget shared with conversation — plan execution counts against same limits
- 55 tests covering guard integration, PlanRegistry round-trips, DAG visualization

### 6.3 Execution Checkpointing & Resume ✅

- After each step: persist state to `~/.mcp-cli/plans/{id}_state.json`
- `mcp plan resume <id>` — loads checkpoint, skips completed steps, continues
- Tracks: completed steps, variable bindings, failed steps, timing

### 6.4 Simulation / Dry-Run Mode ✅

Critical for trust. Show planned tool calls without executing them.

```
mcp plan run <id> --dry-run
```

- Walks plan in topological order
- Resolves `${var}` references
- Displays each step: tool name, resolved arguments, dependencies
- Reports estimated tool call count
- No side effects — safe to run in production

### 6.5 Parallel Step Execution ✅

Independent plan steps execute concurrently via topological batch ordering:

- `_compute_batches()` uses Kahn's BFS topological sort to group steps into parallel batches
- Steps within a batch run concurrently via `asyncio.gather()` with semaphore-controlled concurrency
- Batches execute sequentially to respect dependency ordering
- `max_concurrency` parameter (default: 4) limits concurrent tool calls
- Diamond DAG (1 → 2,3,4 → 5) executes with 3 batches: [1], [2,3,4], [5]
- Variable resolution: `${var}`, `${var.field.subfield}`, template strings — type-preserving for single refs

### 6.6 DAG Visualization ✅

Terminal visualization of plan execution:

- Terminal: ASCII DAG rendering with step status indicators (○ pending, ◉ running, ● completed, ✗ failed)
- `render_plan_dag()` function for terminal display
- Parallel step indicator (∥) marks steps that run concurrently within a batch
- Browser: MCP App panel with D3 force-directed graph, live WebSocket updates (Future)

### 6.7 Re-planning ✅

Adaptive re-planning when execution hits problems (opt-in via `enable_replan=True`):

- On step failure: injects failure context (completed steps, error, remaining steps, variables) into PlanAgent
- PlanAgent generates a revised plan for the remaining work
- Revised plan executes with the current variable context (no recursive re-planning)
- Results merged: completed steps from original + steps from revised plan
- `max_replans` parameter (default: 2) limits re-planning attempts
- `PlanExecutionResult.replanned` flag indicates whether re-planning occurred
- Disabled by default — failure just fails without LLM involvement

### 6.8 Model-Driven Planning (Plan as a Tool) ✅

The model can autonomously create and execute plans during conversation — no `/plan` command required.

When the model determines a task needs multi-step orchestration, it calls an internal `plan` tool to decompose the task into a structured execution graph, then executes it — all within the normal chat flow.

**Internal tools (intercepted before MCP routing, like VM tools):**

| Tool | Purpose |
|------|---------|
| `plan_create` | Model describes a goal → PlanAgent generates a plan DAG → returns plan ID + step summary |
| `plan_execute` | Model passes plan ID → PlanRunner executes → returns results + variables |
| `plan_create_and_execute` | Combined: generate + execute in one call (common case) |

**How it works:**

```
User: "What's the weather like for sailing in Raglan tomorrow?"

Model (internally): This needs geocoding then weather lookup.
  → calls plan_create_and_execute(goal="Get weather forecast for Raglan, NZ")
  → PlanAgent generates: [geocode Raglan] → [get weather for coords]
  → PlanRunner executes both steps via MCP servers
  → Results flow back to model as tool result

Model: "Tomorrow in Raglan: 18°C, light winds from the SW at 12 km/h,
        partly cloudy. Good conditions for sailing."
```

**Key design decisions:**

- **Intercepted like VM tools:** `plan_create`, `plan_execute`, `plan_create_and_execute` are caught in `tool_processor.py` before MCP guard routing, executed locally via PlanRunner
- **Model decides when to plan:** The system prompt describes the planning tools; the model calls them when it determines multi-step orchestration is more effective than sequential tool calls
- **Plans are ephemeral by default:** Created during conversation, not persisted unless the model or user explicitly saves them. Reduces clutter vs `/plan create`
- **Shares guard budget:** Plan tool calls count against the same budget as regular tool calls
- **Display integration:** Plan execution renders with the same `StreamingDisplayManager` callbacks as regular tool calls — the user sees each step executing in real time
- **Variable flow:** Plan results are returned as the tool result, so the model can reference them naturally in its response
- **Opt-in via system prompt:** The planning tools only appear when `--enable-plan-tools` is set (or equivalent config), so the model doesn't attempt planning on simple tasks

**Files:**
- `src/mcp_cli/chat/tool_processor.py` — Intercept `plan_create` / `plan_execute` / `plan_create_and_execute` before MCP routing
- `src/mcp_cli/planning/tools.py` — Tool definitions (OpenAI function format) and execution handlers
- `src/mcp_cli/chat/system_prompt.py` — Inject planning tool descriptions when enabled
- `src/mcp_cli/config/defaults.py` — `DEFAULT_ENABLE_PLAN_TOOLS = False`

**Why this matters:**

Today: User types `/plan create "get weather for Raglan"` → plan generated → user types `/plan run <id>` → result shown. Three interactions.

With 6.8: User asks a question → model decides it needs a plan → creates and executes it → answers. One interaction. The model becomes a self-orchestrating agent when the task demands it, and a simple chatbot when it doesn't.

---

## Dashboard & Multi-Modal ✅ COMPLETE

> **Goal:** Give users a real-time browser UI for conversations and enable multi-modal input (images, text files, audio) across CLI and browser.

### D.1 Dashboard Infrastructure ✅

Real-time browser dashboard alongside chat mode via `--dashboard` flag.

**Files:** `src/mcp_cli/dashboard/` — `server.py`, `bridge.py`, `launcher.py`, `config.py`, `router.py`

- HTTP + WebSocket server on a single port (`server.py`)
- Bridge integrates chat engine events → browser clients (`bridge.py`)
- Router supports multi-agent coordination (`router.py`)
- Shell host page manages view iframes and WebSocket connection (`shell.html`)
- Session replay on connect: `CONVERSATION_HISTORY` + `ACTIVITY_HISTORY`

### D.2 Dashboard Views ✅

Five tabbed views in the browser UI.

**Files:** `src/mcp_cli/dashboard/static/views/` — `agent-terminal.html`, `activity-stream.html`, `plan-viewer.html`, `tool-registry.html`, `config-panel.html`

- **Agent Terminal**: Chat bubbles, streaming tokens, markdown rendering, syntax highlighting, search
- **Activity Stream**: Tool call/result pairs, reasoning steps, state transitions
- **Plan Viewer**: DAG visualization with real-time step progress
- **Tool Registry**: Browse tools, trigger execution from browser
- **Config Panel**: View/switch providers, models, system prompt

### D.3 Multi-Modal Attachments ✅

Attach images, text files, and audio to messages via CLI and browser.

**Files:** `src/mcp_cli/chat/attachments.py`, `src/mcp_cli/chat/chat_handler.py`, `src/mcp_cli/chat/chat_context.py`, `src/mcp_cli/commands/attach/attach.py`, `src/mcp_cli/main.py`

- `/attach` command with staging, list, clear (aliases: `/file`, `/image`)
- `--attach` CLI flag (repeatable) for first-message attachments
- Inline `@file:path` references parsed from message text
- Image URL auto-detection (HTTP/HTTPS `.png`, `.jpg`, `.gif`, `.webp`)
- `AttachmentStaging` on `ChatContext` — drain-on-send lifecycle
- `build_multimodal_content()` assembles content block lists
- Supported: PNG, JPEG, GIF, WebP, HEIC, MP3, WAV, 25+ text/code extensions
- 20 MB max per file, 10 attachments per message

### D.4 Dashboard Attachment Visualization ✅

Render attachments in the browser UI.

**Files:** `src/mcp_cli/dashboard/bridge.py`, `src/mcp_cli/dashboard/static/views/agent-terminal.html`, `src/mcp_cli/dashboard/static/views/activity-stream.html`

- Lightweight attachment descriptors over WebSocket (no large base64 payloads)
- Image thumbnails for files <100KB, metadata badges for larger files
- Expandable text previews (first 2000 chars)
- Audio players (HTML5 `<audio>`)
- Activity stream shows attachment events with paperclip badges

### D.5 Browser File Upload ✅

Attach files directly from the dashboard browser UI.

**Files:** `src/mcp_cli/dashboard/static/views/agent-terminal.html`, `src/mcp_cli/dashboard/static/shell.html`, `src/mcp_cli/dashboard/bridge.py`, `src/mcp_cli/dashboard/server.py`

- "+" attach button in chat input area
- Hidden file input with supported extension filter
- Staging strip with removable badges and image thumbnails
- Drag-and-drop overlay
- Clipboard paste support (images)
- `process_browser_file()` constructs `Attachment` from browser base64 data
- Bridge stages files on `ChatContext.attachment_staging`
- WebSocket `max_size` increased to 25 MB

---

## Dashboard v2: Intelligence Layer

> **Goal:** Evolve the dashboard from a passive conversation viewer into an active operations console — memory visualization, token economics, tool analytics, session management, and multi-agent oversight.

### Original Spec Compliance

The v0.1.0 Dashboard Shell Specification is nearly fully implemented:

| Spec Section | Status | Notes |
|-------------|--------|-------|
| §2 Launch (`--dashboard`, port, browser) | ✅ | Port auto-select from 9120, `--no-browser` flag |
| §3 Architecture (server, bridge, shell, iframes) | ✅ | Exact architecture as specified |
| §4 Shell page (CSS Grid, panel chrome, toolbar) | ✅ | Pop-out, minimize, close, drag-swap, resize handles |
| §5 View protocol (postMessage, INIT, READY, TOOL_RESULT) | ✅ | Full protocol with 5s READY timeout |
| §6.1 Agent terminal (markdown, streaming, /commands) | ✅ | Plus attachments, search, drag-drop |
| §6.2 Activity stream (events, filters, virtual scroll) | ✅ | Plus agent badges, plan updates |
| §7 View discovery (_meta.ui → VIEW_REGISTRY) | ✅ | Dynamic discovery + "+ Add Panel" |
| §8 Themes (8 themes, CSS variables, THEME message) | ✅ | Full theme sync |
| §9 Layout presets (Minimal, Standard, Full, custom) | ✅ | Save/load/delete in localStorage + JSON file |
| §10 Module structure | ✅ | Exact structure as specified |
| §11 Bridge protocol | ✅ | All message types + extras (sessions, config) |
| §14 Design principles (dumb shell, no build, sandbox) | ✅ | All 8 principles followed |
| Panel min-size enforcement during drag | ⚠️ | CSS min 200×200px but no runtime clamp in resize handlers |
| Dashboard-only mode (`mcp-cli dashboard --config`) | ❌ | Spec §2 "future, Phase 5" — not started |

**Beyond spec:** The implementation added features not in the original spec: multi-modal attachments ("+" button, drag-drop, paste), plan-viewer view, tool-registry/browser view, config-panel view, agent-overview view, multi-agent router, session management (new/switch/delete/rename), and the full multi-agent spec (MULTI_AGENT_SPEC.md).

### D2.1 Memory Panel

Visual representation of the AI Virtual Memory subsystem (mirrors `/memory` command). The CLI already exposes working set stats, page table, per-page content, and full subsystem stats — the panel makes this visual and live.

**New view:** `memory-panel.html`

- **Summary gauges**: Working set utilization bar (tokens used / budget), L0/L1 page counts, page fault and eviction counters
- **Page table**: Sortable table of all pages — ID, type (text/image/tool), tier (L0–L4), token count, pinned status, age in turns
- **Page inspector**: Click a page to see content preview, creation turn, access history, eviction score
- **Live metrics**: Page faults, evictions, TLB hit rate — updating in real time as conversation progresses
- **Tier distribution**: Visual breakdown of pages across storage tiers (stacked bar or treemap)
- **Budget pressure indicator**: Warning state when utilization >80%, critical at >95%
- **Page lifecycle animation**: Visual indication when pages are faulted in, evicted, or migrated between tiers

**Bridge changes:** New `on_memory_event()` hook called from VM subsystem on page fault, eviction, and tier migration. New `MEMORY_STATE` WebSocket message type for full state broadcast on connect. `MEMORY_EVENT` for incremental updates (fault, evict, migrate).

**Shell integration:** New tab in shell.html, only visible when `--vm` flag is active (bridge advertises `vm_enabled` in CONFIG_STATE).

**Data source:** `vm.working_set.get_stats()`, `vm.page_table.get_stats()`, `vm.page_table.entries` — same data the `/memory` command already reads.

### D2.2 Token Usage Dashboard

Live token economics — per-turn and cumulative cost tracking.

**New view:** `token-usage.html`

- **Per-turn bar chart**: Input/output tokens per turn, stacked bars
- **Cumulative line**: Running total with estimated cost (provider-specific pricing)
- **Rate limit gauge**: Visual indicator showing proximity to provider rate limits
- **Model comparison**: When model is switched mid-session, show cost delta at the switch point
- **Context window utilization**: How much of the model's context window is in use
- **Export**: Download token usage report as CSV

**Bridge changes:** Extend `CONVERSATION_MESSAGE` payload to include `usage` (input_tokens, output_tokens) when available. New `TOKEN_USAGE_HISTORY` aggregate message on connect for replay.

### D2.3 Tool Execution Timeline

Visual Gantt-style view of tool calls with timing and concurrency.

**New view:** `tool-timeline.html`

- **Gantt chart**: Horizontal bars showing tool call start → end, color-coded by server
- **Concurrent calls**: Overlapping bars visible when tools run in parallel (plan batch execution)
- **Drill-down**: Click a bar to see arguments, result preview, error details
- **Timing stats**: Min/max/avg/p95 execution time per tool
- **Server health**: Aggregate success rate and latency per server

**Bridge changes:** Add `started_at` timestamp to tool call initiation (new `on_tool_call()` hook alongside existing `on_tool_result()`). Activity history pairs start + end for timeline rendering.

### D2.4 Session Management Panel

Manage conversation sessions entirely from the browser.

**Backend status:** Bridge already handles `REQUEST_SESSIONS`, `LOAD_SESSION`, `SAVE_SESSION`, `DELETE_SESSION`, `RENAME_SESSION`, `NEW_SESSION`, `SWITCH_SESSION` — all wired to ChatContext. What's missing is a dedicated view UI.

**New view or config-panel extension:**

- **Session list**: Browse saved sessions with preview (first message, turn count, date, model used)
- **Load session**: Click to load — replays conversation and activity history in all views
- **Save session**: Manual save button with optional name
- **Delete/rename session**: Manage old sessions
- **Session comparison**: Side-by-side diff of two session transcripts
- **Auto-save indicator**: Show when auto-save triggers, link to saved file

### D2.5 Tool Approval UI

Interactive tool approval from the browser when confirmation is required.

**Backend status:** Bridge already has `request_tool_approval()` and `TOOL_APPROVAL_RESPONSE` handler with pending futures. What's missing is the frontend modal.

- **Approval modal**: Shows tool name, arguments (syntax-highlighted JSON), server — approve/deny buttons
- **Approval queue**: Multiple pending approvals shown as stacked cards with countdown timer
- **Auto-approve toggle**: Per-tool or global toggle for trusted tools
- **Audit trail**: Log of approved/denied tool calls with timestamps
- **CLI fallback**: If no browser clients connected, falls back to CLI confirmation (already implemented)

### D2.6 Inline Tool Execution

Execute tools directly from the tool browser view.

**Backend status:** `REQUEST_TOOL` message type already handled by bridge. Needs frontend form UI.

- **Run button**: Each tool card gets a "Run" button
- **Argument form**: Auto-generated from JSON schema (text inputs, number spinners, dropdowns for enums, textarea for objects, checkbox for booleans)
- **Validation**: Client-side validation against schema before sending
- **Result display**: Inline result rendering below the tool card (syntax-highlighted JSON)
- **History**: Recent executions per tool with timing and success/failure indicators

### D2.7 Export from Browser

Download conversations and data directly from the dashboard.

- **Markdown export**: Download formatted conversation as `.md` (reuse existing export logic)
- **JSON export**: Download structured conversation with metadata as `.json`
- **Activity log export**: Download tool call history as CSV
- **Screenshot**: Capture current view as PNG (via browser Canvas API)

**Bridge changes:** New `REQUEST_EXPORT` message type. Bridge calls existing export logic (`/export` command internals) and returns file content as download.

### D2.8 Dashboard-Only Mode

Run the dashboard without a CLI terminal — browser-first experience.

```bash
# Future: standalone dashboard mode
mcp-cli dashboard --server sqlite --config workspace.yaml
```

- Dashboard opens as the primary interface (no terminal chat loop)
- Agent terminal view is the sole conversation input
- All `/commands` work through the browser input
- Workspace configs define layout + servers + default views
- Useful for: demos, shared screens, non-technical users, remote operation

**Requires:** Decoupling the chat loop from terminal stdin — the input queue already supports this (browser messages go through `_input_queue`), but startup assumes terminal mode.

### D2.9 MCP Apps as Dashboard Panels

Embed MCP App UIs (tool `_meta.ui.resourceUri` web apps) as panels within the dashboard, with the ability to maximize into a full browser window.

**Current state:** MCP Apps run on separate `AppHostServer` instances (ports 9470+) using JSON-RPC protocol. Dashboard runs on `DashboardServer` (port 9120+) with mcp-dashboard envelope protocol v2. The two systems are fully independent — apps open in standalone browser tabs.

**Integration approach:** Keep WebSocket servers separate; embed app iframes inside dashboard panels pointing to their existing `localhost:947X` endpoints.

- **Apps panel**: New dashboard view listing all running MCP apps with name, description, status, and "Open" button
- **Inline embedding**: Clicking "Open" adds the app as a resizable iframe panel in the current layout (same panel system as existing views)
- **Maximize / pop-out**: Each embedded app panel gets a maximize button (full dashboard area) and a pop-out button (⤢) to open in a standalone browser window — reuses the shell's existing pop-out mechanism
- **Suppress standalone launch**: When `--dashboard` is active, tools with `_meta.ui.resourceUri` route to an embedded dashboard panel instead of opening a new browser tab
- **Auto-placement**: New apps triggered by tool execution appear as panels in the current layout automatically, with focus
- **Tool result routing**: Tool calls that return `_meta.ui.resourceUri` in their result metadata display the app inline in the activity stream with an "Open in panel" action
- **Lifecycle management**: Apps panel shows running/stopped status; closing a panel doesn't kill the app server (can re-open)
- **Multi-app support**: Multiple app panels can be open simultaneously in the dashboard layout

**Bridge changes:** New `APP_OPENED` / `APP_CLOSED` / `APP_LIST` message types. Dashboard bridge tracks active `AppHostServer` instances and their ports. Shell.html manages app iframe lifecycle.

**Requires:** D2.8 (Dashboard-Only Mode) benefits from this — apps become first-class dashboard citizens.

### D2.10 Dashboard Polish

Quality-of-life improvements.

- **Runtime panel min-size enforcement**: Clamp resize drag handlers to 200×200px minimum (spec gap — CSS minimums exist but no JS enforcement)
- **Theme sync**: Dashboard matches CLI `/theme` selection live (THEME message exists, needs CLI→bridge hook)
- **Keyboard shortcuts**: `Ctrl+1/2/3` for tab switching, `Ctrl+N` to focus chat input, `Ctrl+Shift+F` for global search
- **Background notifications**: Browser Notification API when agent completes while tab is in background
- **Mobile-responsive**: Single-column layout for narrow screens (<768px), collapsible sidebar
- **Message queue during disconnect**: Buffer outbound messages while WebSocket reconnects (reconnection backoff already implemented)

---

## Tier 7: Observability & Traces

> **Goal:** Turn mcp-cli into a debugger for AI behavior. No other agent CLI has a proper observability layer yet.

### 7.1 Tool Reasoning Traces

Operators need to understand *why* the AI chose specific tools and how data flowed.

```
mcp trace last
mcp trace step 4
mcp trace graph
```

Graph output:
```
User Intent
   |
Find location
   |
Weather API ---+
               +--- Decision -> unsafe
Tide API ------+
```

- Every tool call gets a trace ID linking intent → reasoning → tool → result → decision
- Traces are persistent (stored alongside session)
- Exportable for audit: `mcp trace export --format json`

### 7.2 Structured Outputs as First-Class

Allow output schemas for automation pipelines.

```
mcp run "analyse stocks" --schema portfolio.json
```

- Validate LLM output against JSON Schema
- Retry with schema hint on validation failure
- Composable with plans: plan steps can have typed inputs/outputs

---

## Tier 8: Memory Scopes

> **Shift:** Agents stop re-discovering facts every run. Unlocks long-running assistants.

### 8.1 Scoped Memory System

| Scope       | Purpose                  | Lifetime          |
|-------------|--------------------------|-------------------|
| `session`   | Current conversation     | Until exit        |
| `workspace` | Ongoing project context  | Until cleared     |
| `global`    | Personal knowledge base  | Persistent        |
| `plan`      | Workflow-specific memory | Tied to plan      |
| `skill`     | Skill-scoped knowledge   | Tied to skill     |

```
mcp memory show workspace
mcp memory edit global
mcp memory diff
```

- Memory injected into system prompt based on scope
- Workspace memory scoped to current directory / project
- Global memory shared across all sessions
- Plan and skill memory are portable — travel with the artifact

### 8.2 Memory-Aware Context Building

- When building messages for API, inject relevant memory by scope
- Workspace memory overrides global on conflict
- Plan/skill memory available during execution
- Memory compaction: summarize old entries, keep recent ones verbatim

---

## Tier 9: Skills & Capability Layer

> **The strategic keystone.** Skills make behaviour reusable the way MCP made tools reusable. This is the missing standard in the agent stack — the portable capability layer between prompts and tools.

### The Problem

Today's agent ecosystem:

| Ecosystem   | Problem                           |
|-------------|-----------------------------------|
| Claude Code | Locked skills (proprietary)       |
| OpenAI/Codex| Tool calls but no portability     |
| LangChain   | Code-level abstraction only       |
| Marketplaces| Shareable socially, not at runtime|

GitHub repos can be shared between ecosystems, but can't be *executed* across runtimes without rewriting glue. That's manual porting, not interoperability. The incompatibility isn't in prompts — it's in the **hidden runtime contract**: tool naming, approval models, execution semantics, memory models, error handling.

### The Solution: Declarative Capability Binding

Skills don't assume specific tools exist. They declare **capability intent**:

```yaml
capabilities:
  - web_search
  - file_edit
  - structured_reasoning
```

The runtime resolves capabilities to concrete tools:

```
web_search         → tavily server (or serpapi, or browser.search)
file_edit          → fs.mutate (or code.patch)
structured_reasoning → reasoning model
```

Same skill runs everywhere. Different infrastructure, same behaviour.

### Plugins vs Skills

|                      | Plugin (MCP Server) | Skill              |
|----------------------|---------------------|--------------------|
| Adds capability      | Yes                 | No                 |
| Defines behaviour    | No                  | Yes                |
| Implements functions | Yes                 | No                 |
| Reusable workflow    | No                  | Yes                |

MCP server = plugin. Skill = behaviour. mcp-cli hosts both.

### 9.1 Skill Format

A skill is a directory:

```
skills/
  travel-plan/
    skill.yaml         # manifest: capabilities, policy, metadata
    instructions.md     # reasoning instructions (natural language)
    schema.json         # structured output contract (optional)
```

Minimal `skill.yaml`:

```yaml
name: travel-plan
version: 1.0.0
description: Plan a day trip using weather and transit data

inputs:
  - name: query
    type: string
    description: What to plan

capabilities:
  required:
    - weather_forecast
    - route_planning
  optional:
    - tide_data

output:
  schema: schema.json

policy:
  allow_web: false
  allow_code_exec: false
  deterministic: true
  max_tool_calls: 20
```

The spec is intentionally tiny — small enough people actually adopt it.

### 9.2 Skill Runtime

```
mcp skill run travel-plan "day trip to Raglan tomorrow"
mcp skill inspect travel-plan
mcp skill validate travel-plan
```

**Runtime resolution flow:**

1. Load `skill.yaml` → parse capabilities
2. Resolve capabilities to available MCP tools:
   ```
   Skill needs: weather_forecast
   You have: 3 weather servers
   → mcp-cli asks: Use local weather, NOAA, or MetOffice?
   ```
3. Apply policy constraints (tool allowlist, deterministic mode, etc.)
4. Inject `instructions.md` as system prompt context
5. Execute with plan/trace support (Tiers 6-7)
6. Validate output against `schema.json`

Skills are portable across infrastructures because they bind to capabilities, not tools.

### 9.3 Skill Discovery & Installation

```
mcp skill search "finance"
mcp skill install mortgage-analyzer
mcp skill list
mcp skill recommend "analyse my expenses"
```

Registry is just Git repos:

```
mcp://skills/travel-plan → github.com/user/travel-plan-skill
```

Downloaded to `~/.mcp-cli/skills/`. No new marketplace needed — Git is the registry, `skill.yaml` is the manifest.

### 9.4 Skill Packaging & Publishing

```
mcp skill pack travel-plan        # validate + bundle
mcp skill publish travel-plan     # push to registry
```

- Validates `skill.yaml` schema
- Checks capability names against known capability vocabulary
- Bundles instructions, schema, and metadata
- Publishes to configured registry (Git, npm-style, or local)

### 9.5 Cross-Ecosystem Adapters

Don't copy proprietary formats — translate them:

```
Claude skill    → MCP skill adapter
Codex agent     → MCP skill adapter
LangChain chain → MCP skill adapter
```

mcp-cli becomes the **interoperability layer**. The Linux of agent tooling.

### 9.6 Capability Vocabulary

A small, shared vocabulary of abstract capabilities that skills declare and runtimes resolve:

| Capability             | Example tools that satisfy it         |
|------------------------|---------------------------------------|
| `web_search`           | tavily, serpapi, browser.search       |
| `fetch_url`            | fetch, browser.navigate               |
| `file_read`            | fs.read, code.view                    |
| `file_edit`            | fs.write, code.patch                  |
| `weather_forecast`     | openweather, noaa, metoffice          |
| `structured_reasoning` | reasoning model, chain-of-thought     |
| `code_execution`       | python.exec, sandbox.run              |
| `database_query`       | sql.query, sqlite.exec               |

Vocabulary grows organically. Skills can declare custom capabilities; the runtime warns if unresolvable.

### 9.7 Skill + Plan Integration

Skills can contain optional workflows (plans):

```yaml
# skill.yaml
workflow: workflow.yaml   # optional execution graph
```

When present, the skill uses the plan engine (Tier 6) for structured execution instead of free-form conversation. This bridges "chat-style" skills and "deterministic pipeline" skills.

### 9.8 Skill Memory

Each skill gets its own memory scope (Tier 8):

- Persists across runs of the same skill
- Stores learned patterns (which tools worked, common errors)
- Portable: travels with the skill package
- Separate from session/workspace/global memory

---

## Tier 10: Scheduling & Background Agents

> **Shift:** mcp-cli becomes cron + AI + tools.

### 10.1 Agent Definitions

```
mcp agent create surf-check --prompt "check surf conditions at Raglan"
mcp agent create daily-report --skill daily-summary
mcp agent list
mcp agent logs surf-check
```

- Agent = named prompt OR skill + server config + schedule (optional)
- Stored in `~/.mcp-cli/agents/`
- Each agent has its own memory scope
- Agents can reference skills for portable, reusable behaviour

### 10.2 Scheduling

```
mcp agent schedule surf-check "6:00 daily"
mcp agent schedule report "0 9 * * MON"   # cron syntax
mcp agent unschedule surf-check
```

- Lightweight daemon or OS-native scheduling (launchd/systemd/cron)
- Results stored in agent log, accessible via `mcp agent logs`
- Notifications on failure or interesting results (webhook, email, or local)

### 10.3 Background Execution

```
mcp agent run surf-check --background
mcp agent status surf-check
mcp agent stop surf-check
```

- Detached execution with log tailing
- Graceful interruption

---

## Tier 11: Multi-Agent Coordination

> **Shift:** Single conversation loop becomes agent pipelines. First CLI to naturally express this.

### 11.1 Workflow Definitions

```
mcp workflow run travel.yaml
mcp workflow inspect travel.yaml
mcp workflow validate travel.yaml
```

Example `travel.yaml`:
```yaml
agents:
  planner:
    model: gpt-5
    skill: travel-plan
  researcher:
    model: local-llama
    servers: [search]
  verifier:
    model: reasoning-model
    skill: safety-check

flow:
  - researcher: "find options for {destination}"
  - planner: "create itinerary from research"
  - verifier: "validate safety and feasibility"
  - user: "review and approve"
```

- Agents pass structured outputs between steps (via skill schemas)
- Each agent has its own tool set, model, and optional skill
- Workflow memory shared across agents (plan-scoped)
- Conditional branching: `if verifier.unsafe then planner.revise`

### 11.2 Tool Capability Discovery & Ranking

Instead of listing tools, rank them by relevance.

```
mcp tools suggest "calculate mortgage"
```

Output:
```
finance.mortgage_calculator  (confidence 0.91)
spreadsheet.compute          (confidence 0.52)
python.exec                  (confidence 0.12)
```

- Semantic search across tool descriptions
- Boosted by procedural memory (tools that worked before rank higher)
- Used by skill runtime for automatic capability resolution

---

## Tier 12: Remote Sessions

> **Shift:** ssh for AI systems. Run agents remotely, control locally.

### 12.1 Remote Agent Control

```
mcp connect prod-agent.company.net
mcp remote status
mcp remote logs --follow
```

- Local CLI connects to remote mcp-cli instance
- Authentication via SSH keys or OAuth
- Stream reasoning and tool traces in real-time
- Execute commands against remote tool servers

### 12.2 Shared Sessions

- Multiple operators can observe the same agent session
- Read-only mode for auditors
- Collaborative mode for pair-debugging agents

## Code Review Fixes (Post-Audit)

> **Goal:** Address findings from the comprehensive codebase review. Fixes organized by priority — high items are correctness/reliability, medium are consistency/maintainability, low are cleanup.

### R.1 Add Logging to Remaining Silent Exception Blocks ✅

**Problem:** 18 `except Exception: pass` blocks in commands/ and UI code lose error context entirely. The Tier 4 architecture audit fixed 6 in core modules; these are the remaining locations.

**Files & Locations:**

| File | Line | Context |
|------|------|---------|
| `commands/servers/ping.py` | 87-88 | Silent pass in ping check |
| `commands/servers/health.py` | 68-69 | Silent pass in health check |
| `commands/tokens/token.py` | 51-52 | Silent fallback to AUTO backend |
| `commands/providers/providers.py` | 196-197 | Silent pass in provider status |
| `commands/providers/providers.py` | 263-266 | Hardcoded error, missing logs |
| `commands/providers/models.py` | 244-245 | Silent pass in Ollama discovery |
| `commands/providers/models.py` | 286-287 | Silent pass in provider fetch |
| `commands/providers/models.py` | 322-323 | Silent pass in API model fetch |
| `commands/core/clear.py` | 91-99 | Nested silent passes |
| `chat/tool_processor.py` | 634, 651, 686, 774 | Silent pass for UI errors |
| `chat/tool_processor.py` | 914 | Silent pass for JSON parsing |
| `chat/chat_handler.py` | 139-140 | Swallows tool count error |
| `tools/manager.py` | 342-343 | Silent "non-critical" pass |
| `config/discovery.py` | 212-213 | Returns False, loses error |

**Action:** Add `logger.debug("context: %s", e)` to each block. Same pattern used in the 6 core fixes from Tier 4.

### R.2 Delete Dead Code: `chat/__main__.py` ✅

**Problem:** 196-line file marked dead in pyproject.toml coverage omit. Imports non-existent modules. Never executed.

**File:** `src/mcp_cli/chat/__main__.py`

**Action:** Delete the file. Remove from coverage omit in pyproject.toml.

### R.3 Standardize Logger Variable Naming ✅

**Problem:** 5 modules use `log = getLogger(__name__)` while the rest use `logger`. Inconsistent grep-ability.

**Files:**
- `apps/bridge.py` — `log`
- `apps/host.py` — `log`
- `chat/conversation.py` — `log`
- `chat/tool_processor.py` — `log`
- `commands/memory/memory.py` — `log`

**Action:** Rename `log` → `logger` in these 5 files. Update all references.

### R.4 Consolidate `constants/` Into `config/` ✅

**Problem:** Two locations for project constants: `constants/__init__.py` (118 lines) and `config/defaults.py` + `config/enums.py`. Splits the single source of truth.

**Action:** Move status values and enums from `constants/` to `config/enums.py` or `config/defaults.py`. Update imports. Delete `constants/` package.

### R.5 Add Unit Tests for `core/model_resolver.py` ✅

**Problem:** 178-line user-facing module with zero test coverage. Handles error display and model resolution fallback logic.

**File:** `src/mcp_cli/core/model_resolver.py`

**Action:** Create `tests/core/test_model_resolver.py` with tests for resolution paths, error handling, and fallback behavior.

### R.6 Add Unit Tests for High-Risk Command Modules

**Problem:** 48 command modules lack direct unit tests. Existing tests are end-to-end command usage tests that don't cover internal logic. Highest risk in large modules.

**Priority files:**
- `commands/tokens/token.py` (942 lines)
- `commands/tools/execute_tool.py` (565 lines)
- `commands/memory/memory.py` (538 lines)

**Action:** Add targeted unit tests for complex internal logic in each module.

---

## Priority Summary

| Tier | Focus | What Changes | Status |
|------|-------|-------------|--------|
| **1** | Memory & context | Stops crashing on large payloads | ✅ Complete |
| **2** | Efficiency & resilience | Reliable under real workloads | ✅ Complete |
| **Apps** | MCP Apps (SEP-1865) | Interactive browser UIs from MCP servers | ✅ Complete |
| **3** | Performance & polish | Feels fast, saves work | ✅ Complete |
| **4** | Code quality | Maintainable, testable | ✅ Complete |
| **5** | Production hardening | Observable, auditable | ✅ Complete |
| **VM** | AI Virtual Memory | OS-style context management | ✅ Complete (Experimental) |
| **Review** | Code review fixes | Silent exceptions, dead code, test gaps | ✅ Complete |
| **6** | Plans & execution graphs | Reproducible workflows | ✅ Complete (6.0–6.8) |
| **Dashboard** | Dashboard & multi-modal | Real-time browser UI, file attachments | ✅ Complete |
| **Dashboard v2** | Dashboard intelligence | Memory panel, token usage, tool timeline, session mgmt, approvals, MCP Apps panels, dashboard-only mode | High |
| **7** | Observability & traces | Debugger for AI behavior | High |
| **8** | Memory scopes | Long-running assistants | High |
| **9** | Skills & capabilities | Portable behaviour layer | High |
| **10** | Scheduling & agents | Autonomous operations | High |
| **11** | Multi-agent coordination | Agent pipelines | Very High |
| **12** | Remote sessions | Distributed AI ops | Very High |

### If You Build Only Four Things

These change the category of the tool from **chat interface** to **agent operating system**:

1. **Plans** (Tier 6) — reproducible, inspectable execution; model-driven planning (6.8) makes the model a self-orchestrating agent
2. **Traces** (Tier 7) — explainable AI operations
3. **Skills** (Tier 9) — portable, reusable behaviour (the npm for agents)
4. **Scheduling** (Tier 10) — autonomous background agents

### The Strategic Position

| Ecosystem | Problem                           | mcp-cli Opportunity                  |
|-----------|-----------------------------------|--------------------------------------|
| Claude    | Locked skills                     | Open skill format + adapter          |
| OpenAI    | Tool calls, no portability        | Capability binding layer             |
| LangChain | Code-level only                   | Declarative behaviour packages       |
| All       | Prompts not reusable, tools are   | **Skills make behaviour reusable**   |

### The Meta Insight

Most tools optimize *how the model thinks*. The winning layer optimizes *how thinking is executed, repeated, inspected, and trusted*. mcp-cli is already positioned exactly there.

Skills are the keystone: they turn mcp-cli from a runtime into a **distribution platform for agent behaviour** — the neutral ground where capabilities live, regardless of which model or infrastructure runs them.
