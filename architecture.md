# MCP-CLI Architecture Principles

> These principles govern all code in mcp-cli.
> Every PR should be evaluated against them.

---

## 1. Pydantic Native

Structured data flows through Pydantic models, not raw dicts.

**Rules:**
- Inputs and outputs of public APIs are `BaseModel` instances
- Configuration objects are Pydantic models with `frozen=True` for immutability
- Use `Field()` for defaults, descriptions, and constraints
- Use `field_validator` and `model_validator` for construction-time checks
- Serialization goes through `.model_dump()` / `.model_dump_json()` at boundaries only
- Factory methods (`from_dict()`, `create()`) return model instances

**Why:** Pydantic gives us validation at construction time, clear field documentation, and serialization for free. Raw dicts defer errors to runtime and make refactoring dangerous.

```python
# Yes
class ToolResult(BaseModel):
    name: str
    content: str
    success: bool = True

# No
result = {"name": "foo", "content": "bar", "success": True}
```

---

## 2. Async Native

Every public API that performs I/O is `async def`. No blocking calls in the hot path.

**Rules:**
- All tool execution, MCP communication, and LLM calls use `async`/`await`
- Use `asyncio.Lock` for shared async state, `threading.Lock` only for sync-only code paths (e.g., file-based memory store)
- Synchronous helpers (pure computation, no I/O) are acceptable but must not block the event loop
- Config loading provides both sync and async variants (`load_sync()`, `load_async()`) for startup flexibility
- Background tasks use `asyncio.create_task()` with proper cancellation handling

**Why:** Tool execution is inherently concurrent. A single blocking call in the hot path stalls every tool call sharing that event loop.

```python
# Yes
async def execute_tool(self, name: str, args: dict) -> ToolResult: ...

# No
def execute_tool(self, name: str, args: dict) -> ToolResult:
    return asyncio.run(self._execute(name, args))
```

---

## 3. No Dictionary Goop

Never pass `dict[str, Any]` through public interfaces when a model will do.

**Rules:**
- If a dict has a known shape, define a model or `TypedDict`
- If a function returns `dict[str, Any]`, ask: should this be a model?
- Accessing nested dicts with `.get("key")` chains is a code smell — model it
- Internal dict usage for caches, indexes, and transient lookups is fine
- JSON schemas from external systems (MCP, OpenAI) are exempt at the boundary — but wrap them in models as early as possible

**Why:** `data["tool_calls"][0]["function"]["name"]` is unreadable, unrefactorable, and produces `KeyError` at runtime instead of a validation error at construction.

```python
# Yes
msg = Message.from_dict(raw_data)
process(msg)

# No
process(raw_data)  # passing a dict through the stack
```

---

## 4. No Magic Strings

Use enums, constants, or Pydantic `Literal` types — never bare string comparisons.

**Rules:**
- Status values → `str` Enum (e.g., `ServerStatus`, `AppState`, `ChatStatus`)
- Role values → `str` Enum (e.g., `MessageRole.USER`, `MessageRole.ASSISTANT`)
- Timeout types → `TimeoutType` enum
- Config keys → named constants in `config/defaults.py`
- If you find yourself writing `if x == "some_string"`, define a constant or enum first
- Enum members that need to serialize as strings use `class Foo(str, Enum)`

**Why:** Magic strings are invisible to refactoring tools, produce silent bugs when misspelled, and can't be auto-completed by IDEs.

```python
# Yes
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

# No
if msg["role"] == "assistant": ...
```

---

## 5. Core / UI Separation

Logic that is UI-independent must not import from `display/`, `interactive/`, or `commands/`. Core modules use `logging` only — never `chuk_term.ui.output`.

**Core modules** (use `logging` only):
- `chat/` — conversation, tool processing, context, session management
- `config/` — defaults, configuration loading, server models
- `tools/` — tool management, execution, filtering
- `model_management/` — provider/model resolution
- `memory/` — persistent memory scopes
- `auth/` — token management
- `context/` — application context

**UI modules** (may use `chuk_term.ui.output`):
- `display/` — streaming display, rendering
- `interactive/` — terminal shell, prompt sessions
- `commands/` — CLI command handlers
- `adapters/` — mode-specific command dispatch
- `chat/ui_manager.py` — chat UI (streaming, tool call display)

**Acceptable exception:** `tools/manager.py` OAuth browser-open notifications are user-facing.

**Future goal:** core modules extractable into a standalone `mcp-cli-core` package.

**Why:** Core logic should be testable without a terminal. UI concerns change independently from business logic.

---

## 6. Single Source of Truth

All default values live in `config/defaults.py`. Business logic imports constants from there, never hardcodes values.

**Configuration precedence:** `defaults.py` → environment variables → config file → CLI flags → `RuntimeConfig` → component init.

**Rules:**
- Every timeout, limit, threshold, path, and feature flag has a named constant in `defaults.py`
- Constants are grouped by category with section headers
- Each constant has a docstring explaining its purpose
- Business logic references the constant, never a literal value

**Why:** When a default needs to change, there's exactly one place to look. When reading code, the constant name documents intent.

```python
# Yes
from mcp_cli.config.defaults import DEFAULT_MAX_TOOL_RESULT_CHARS
content = truncate(content, DEFAULT_MAX_TOOL_RESULT_CHARS)

# No
content = truncate(content, 100_000)  # magic number
```

---

## 7. Explicit Dependencies

Constructor injection over global singletons. When a component needs a dependency, accept it as a parameter.

**Rules:**
- Core classes accept dependencies via `__init__` (e.g., `ToolProcessor(context, ui_manager)`)
- No module-level mutable state outside of lazy caches
- Singletons from external libraries (`get_tool_state()`, `get_search_engine()`) are acceptable but documented as known violations
- Lazy imports in function bodies are acceptable for breaking circular dependencies or deferring heavy initialization

**Why:** Explicit dependencies make code testable with simple mocks and make the dependency graph visible.

---

## 8. Fail Loudly at Boundaries, Recover Gracefully Inside

Validate inputs at system boundaries (CLI args, API responses, config files). Inside the core, trust the type system.

**Rules:**
- Pydantic validation catches malformed inputs at construction time
- Config files validated on load with clear error messages
- Custom exception hierarchy (`CommandError`, `InvalidParameterError`, `CommandExecutionError`) carries context
- Errors logged at the point of origin with full context, not at a distant catch site
- Silent `except Exception: pass` is forbidden in production paths — use targeted exception handling
- UI errors in tool display are non-fatal (caught and ignored to prevent tool execution failures)
- Transport recovery: detect failure → attempt recovery → log outcome → return structured error if recovery fails

**Why:** Validation at boundaries prevents garbage from propagating. Structured errors enable programmatic handling.

---

## 9. Protocol-Based Interfaces

Use `Protocol` (structural subtyping) for component boundaries — not ABC inheritance.

**Rules:**
- Core interfaces defined as `@runtime_checkable` Protocols (e.g., `ToolProcessorContext`, `UIManagerProtocol`)
- Protocols specify the minimal surface area needed by consumers
- Concrete classes satisfy protocols implicitly — no explicit `implements` declaration
- Tests use simple dummy classes that satisfy the protocol without subclassing
- Access optional context attributes via `getattr(obj, "attr", default)` with `hasattr` guards rather than expanding the protocol

**Why:** Protocols enable duck typing with type safety. Tests don't need to mock an entire class hierarchy — just the methods actually called. Components stay loosely coupled.

```python
@runtime_checkable
class ToolProcessorContext(Protocol):
    tool_manager: "ToolManager"
    conversation_history: list[HistoryMessage]
    def inject_tool_message(self, message: HistoryMessage) -> None: ...

# Test — no inheritance needed
class DummyContext:
    def __init__(self):
        self.conversation_history = []
        self.tool_manager = DummyToolManager()
    def inject_tool_message(self, message):
        self.conversation_history.append(message)
```

---

## 10. Tool Interception Pattern

Internal tools (VM, memory) are intercepted before guard checks and never routed to MCP servers.

**Rules:**
- Internal tool names defined as `frozenset` constants (e.g., `_VM_TOOL_NAMES`, `_MEMORY_TOOL_NAMES`)
- Interception happens early in `process_tool_calls()`, before confirmation, guard checks, and MCP dispatch
- Each internal tool category has a dedicated handler method (`_handle_vm_tool()`, `_handle_memory_tool()`)
- Tool definitions injected into `openai_tools` in `_load_tools()` so the LLM knows they exist
- Results added to conversation history via the same `_add_tool_result_to_history()` path as MCP tools

**Why:** Internal tools need to bypass the MCP stack entirely. The interception pattern keeps the dispatch logic clean and makes it easy to add new internal tool categories.

```python
# In process_tool_calls(), before guard checks:
if execution_tool_name in _VM_TOOL_NAMES:
    await self._handle_vm_tool(...)
    continue

if execution_tool_name in _MEMORY_TOOL_NAMES:
    await self._handle_memory_tool(...)
    continue

# Only MCP tools reach this point
```

---

## 11. Dirty Flag Regeneration

Expensive computed state (system prompts, tool lists) uses a dirty flag to avoid unnecessary recomputation.

**Rules:**
- `_system_prompt_dirty: bool` starts `True` and is set back to `True` when state changes (memory mutations, tool list changes)
- `_generate_system_prompt()` checks the flag first and returns cached value when clean
- Mutations that affect the prompt (remember, forget, tool discovery) set the flag
- The prompt is regenerated lazily on next access, not eagerly on mutation

**Why:** System prompt generation involves iterating all tools and formatting server groups. Doing this on every turn is wasteful when most turns don't change the tool set.

---

## 12. Unified Command System

All user commands (CLI, chat slash commands, interactive shell) share a single implementation.

**Rules:**
- Every command extends `UnifiedCommand` with `name`, `aliases`, `modes`, `parameters`, and `execute()`
- Commands declare which modes they support via `CommandMode` flags (`CHAT`, `CLI`, `INTERACTIVE`, `ALL`)
- `CommandParameter` defines parameters once; adapters convert to mode-specific formats (Typer options, shell args, chat arguments)
- `CommandResult` is the universal return type with `success`, `output`, `error`, and `data` fields
- Commands are registered in a singleton `UnifiedCommandRegistry` at startup
- Subcommands use `CommandGroup` with dispatch to child commands

**Why:** Write the logic once, use it everywhere. No drift between what `/help` shows in chat mode and what `--help` shows on the CLI.

---

## 13. Linting and Type Checking

All code must pass `make check` (ruff lint + ruff format + mypy + pytest). No exceptions before merging.

**Rules:**
- `ruff check` for linting (unused imports, style violations)
- `ruff format` for consistent formatting
- `mypy` for type checking (strict on new code)
- Fix issues before merging, not after
- `TYPE_CHECKING` imports to avoid circular dependencies at runtime
- Typed annotations on all public function signatures

---

## 14. Test Coverage

New code ships with tests. Minimum **90% coverage per file** for new code.

**Rules:**
- Each `src/.../foo.py` has a corresponding `tests/.../test_foo.py`
- Async tests use `pytest-asyncio` with `@pytest.mark.asyncio` (auto mode enabled)
- Mock external dependencies — never hit real services in unit tests
- Integration tests in `tests/integration/` marked with `@pytest.mark.integration`
- Use standard dummy classes (`DummyContext`, `DummyUIManager`, `DummyToolManager`) for tool processor tests
- Guard state reset via `_fresh_tool_state` fixture with permissive limits
- Verify with `uv run pytest --cov=src/mcp_cli`

**Project minimum:** `fail_under=60` with `branch=true` (conservative baseline; ratchet upward).

---

## 15. Secret Redaction

Secrets must never appear in logs, error messages, or telemetry.

**Rules:**
- `SecretRedactingFilter` in `config/logging.py` is always active on all log handlers
- Patterns redacted: Bearer tokens, `sk-*` API keys, `api_key=` values, OAuth `access_token`, `Authorization` headers
- The filter is non-throwing — redaction failures don't break logging
- OAuth tokens use copy-on-write headers (copy before tool execution, never mutate shared state)
- Optional rotating file handler via `--log-file` (JSON format, DEBUG level)

**Why:** A single leaked API key in a log file is a security incident. Defense in depth means the filter catches what developers miss.

---

## Checklist for PRs

- [ ] All new public APIs are `async def` (or pure computation)
- [ ] New data structures use Pydantic models (not raw dicts)
- [ ] No new magic string comparisons (use enums/constants)
- [ ] Defaults added to `config/defaults.py` with docstrings
- [ ] Core modules use `logging` only — no `chuk_term.ui.output`
- [ ] Interfaces use `Protocol`, not ABC
- [ ] Internal tools use interception pattern (frozenset + handler)
- [ ] New file has corresponding test file with good coverage
- [ ] `make check` passes (ruff + mypy + pytest)
- [ ] No secrets in log messages or error output

---

## Two Message Classes

The codebase has two classes that represent messages, serving different purposes:

- **`chuk_llm.core.models.Message`** (re-exported via `chat/response_models.py`) — canonical LLM message with typed `ToolCall` objects. Used by `tool_processor.py` and `conversation.py`.
- **`mcp_cli.chat.models.HistoryMessage`** (aliased as `Message` for backward compat) — SessionManager-compatible message with `tool_calls: list[dict]`. Used by `chat_context.py`.

The roundtrip: chuk_llm Message → `to_dict()` → SessionEvent → `from_dict()` → HistoryMessage → `to_dict()` → API.

---

## MCP Apps (SEP-1865)

MCP Apps are interactive HTML UIs served by MCP servers and rendered in the user's browser via sandboxed iframes. When a tool has a `_meta.ui` annotation, mcp-cli launches a local web server that bridges the browser and the MCP backend.

### Architecture

```
Browser                    Python Backend                MCP Server
┌─────────────────┐       ┌──────────────────┐       ┌──────────────┐
│  Host Page (JS)  │──WS──│  AppBridge        │──MCP──│  Tool Server │
│  ┌─────────────┐ │      │  (bridge.py)      │       │              │
│  │ App iframe  │ │      └──────────────────┘       └──────────────┘
│  │ (sandboxed) │ │              │
│  └─────────────┘ │      ┌──────────────────┐
│   postMessage ↕  │      │  AppHostServer   │
└─────────────────┘       │  (host.py)        │
                          └──────────────────┘
```

- **`host.py`** — `AppHostServer` manages lifecycle: port allocation, HTTP serving (host page + app HTML), WebSocket server, browser launch
- **`host_page.py`** — JavaScript host page template; bridges iframe postMessage ↔ WebSocket, handles `ui/initialize`, display modes, reconnection
- **`bridge.py`** — `AppBridge` handles JSON-RPC protocol: proxies `tools/call` and `resources/read` to MCP servers, manages message queue for disconnected WS, formats tool results per MCP spec
- **`models.py`** — Pydantic models: `AppInfo`, `AppState` (PENDING → INITIALIZING → READY → CLOSED), `HostContext`

### Security Model

- **Iframe sandbox:** `allow-scripts allow-forms allow-same-origin allow-popups allow-popups-to-escape-sandbox`
- **XSS prevention:** Tool names are `html.escape()`d before template injection
- **CSP domain sanitization:** Server-supplied domains validated against `^[a-zA-Z0-9\-.:/*]+$`
- **Tool name validation:** Bridge rejects tool names not matching `^[a-zA-Z0-9_\-./]+$`
- **URL scheme validation:** `ui/open-link` only allows `http://` and `https://` schemes
- **Safe JSON serialization:** `_safe_json_dumps()` with `_to_serializable()` fallback; circular reference protection

### Session Reliability

- **Message queue:** `_pending_notifications: deque[str]` (maxlen=50) queues notifications when WS is disconnected
- **Drain on reconnect:** `drain_pending()` flushes queued messages when WS reconnects
- **State reset:** `set_ws()` resets state to INITIALIZING, closes old WS
- **Reconnect notification:** Host page sends `ui/notifications/reconnected` to app iframe on WS reconnect
- **Exponential backoff:** WS reconnection uses 1s→30s exponential backoff with reset on success
- **Initialization timeout:** Configurable JS timeout (default 30s) shows "initialization timed out" if app never initializes
- **Deferred tool result delivery:** Initial tool results are stored on the bridge and pushed only after the app sends `ui/notifications/initialized`, preventing race conditions where postMessage is dropped before the app sets up its listener
- **Duplicate prevention:** `launch_app()` closes previous instance before launching new one
- **Push to existing:** `tool_processor.py` pushes new tool results to running apps instead of re-launching

### Spec Compliance

- `ui/initialize` response includes protocol version, host capabilities (with sandbox details), host info, host context
- `ui/resource-teardown` sent to iframe on `beforeunload`
- `ui/notifications/host-context-changed` sent after display mode changes
- `structuredContent` recovered from JSON text blocks when transport loses it (CTP normalization)

---

## Known Violations

Architecture review performed after Tier 2. Tier 4 (Code Quality) resolved the most impactful issues. Remaining items are tracked here.

### Core/UI Separation (#5)

**Resolved in Tier 4.3:** `chat/conversation.py`, `chat/tool_processor.py`, and `chat/chat_context.py` no longer import `chuk_term.ui.output`. All core logging goes through the `logging` module.

**Remaining:**

| File | Issue | Severity |
|------|-------|----------|
| `chat/ui_manager.py` | Imports `prompt_toolkit`, `display/`, `commands/` | HIGH — move to `interactive/` |
| `chat/command_completer.py` | Imports `prompt_toolkit`, `commands/` | HIGH — move to `interactive/` |
| `chat/streaming_handler.py:20` | Imports `StreamingDisplayManager` from `display/` | MEDIUM — use protocol |
| `chat/__main__.py:15` | Imports `register_commands` from `commands/` | MEDIUM — entry point, acceptable |

### Pydantic Native (#1, #3)

| File | Issue | Severity |
|------|-------|----------|
| `chat/chat_context.py` | `openai_tools: list[dict]` instead of typed model | MEDIUM |
| `chat/models.py` | `HistoryMessage.tool_calls: list[dict]` instead of `list[ToolCallData]` | MEDIUM — by design for SessionManager compat |
| `chat/conversation.py` | `_validate_tool_messages()` works on raw dicts | MEDIUM — by design at serialization boundary |

### Explicit Dependencies (#7)

**Resolved in Tier 4.1:** `_GLOBAL_TOOL_MANAGER` singleton removed. ToolManager is constructor-injected everywhere.

**Remaining (deferred — low impact):**

| File | Issue | Severity |
|------|-------|----------|
| `chat/tool_processor.py` | Uses `get_tool_state()`, `get_search_engine()` globals | MEDIUM — external library singletons |
| `chat/conversation.py` | Uses `get_tool_state()` global | MEDIUM — external library singleton |
| `chat/tool_processor.py` | Uses `get_preference_manager()` global | LOW — 15 call sites, marginal payoff |
