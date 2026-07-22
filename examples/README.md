# MCP-CLI Examples

Organized examples demonstrating mcp-cli capabilities. Most examples run self-contained; those requiring API keys note it in their docstring.

## Directory Structure

```
examples/
  getting_started/       Basic examples to get up and running
  streaming/             Streaming response demos
  tools/                 Tool execution and round-trip demos
  commands/              Command system and slash commands
  servers/               Server management and custom providers
  apps/                  MCP Apps (SEP-1865) interactive UI demos
  planning/              Plan execution, parallel batches, guards
  safety/                Context safety mechanisms (Tier 1)
  sample_tools/          Reusable tool classes for demos
```

## Getting Started

```bash
# Run from the project root
cd /path/to/mcp-cli

# Basic LLM call (requires OPENAI_API_KEY)
python examples/getting_started/basic_llm_call.py

# Ollama local model (requires running Ollama)
python examples/getting_started/ollama_llm_call.py

# Simple streaming demo (requires OPENAI_API_KEY)
python examples/getting_started/streaming_simple.py

# UI banner demo (no API key needed)
python examples/getting_started/clear_with_banner_demo.py
```

## Streaming

```bash
# Full streaming showcase with multiple models
python examples/streaming/mcp_streaming_showcase.py

# Real LLM streaming with tool integration
python examples/streaming/real_llm_streaming_demo.py

# Working demo of streaming in mcp-cli context
python examples/streaming/mcp_cli_working_demo.py
```

## Tools

```bash
# LLM <-> local tool round-trip (requires OPENAI_API_KEY)
python examples/tools/tool_round_trip.py

# Tool execution in interactive mode
python examples/tools/demo_tool_execution.py

# Tool argument flow tracing (diagnostic)
python examples/tools/tool_args_flow_demo.py

# Runtime server add + tool execution
python examples/tools/server_add_and_execute_tool.py

# Playwright browser automation with tools
python examples/tools/playwright_server_with_tool_execution.py
```

## Commands

```bash
# Command system demo
python examples/commands/command_system_demo.py

# End-to-end command verification
python examples/commands/command_system_e2e_demo.py

# Slash command autocomplete
python examples/commands/slash_commands_demo.py

# Interactive mode capabilities
python examples/commands/demo_interactive_mode.py

# Command mode from shell
bash examples/commands/cmd_mode_demo.sh
```

## Servers

```bash
# Server lifecycle: list, add, use, disable, enable, remove
python examples/servers/server_management_e2e.py

# Custom provider integration
python examples/servers/custom_provider_working_demo.py
```

## Apps (MCP Apps — SEP-1865)

MCP Apps allow tools to declare interactive HTML UIs that render in the browser.
The host fetches a `ui://` resource, serves it in a sandboxed iframe, and bridges
JSON-RPC between the app and the MCP server.

```bash
# Full end-to-end demo — launches a browser with a working MCP App
# (requires: pip install mcp-cli[apps])
python examples/apps/apps_demo.py

# _meta.ui pipeline — shows how metadata survives the tool pipeline
# (no extra dependencies)
python examples/apps/meta_pipeline_demo.py

# Bridge protocol — demonstrates JSON-RPC message routing
# (no extra dependencies)
python examples/apps/bridge_protocol_demo.py
```

Demonstrates:
1. `_meta.ui` preservation through `ToolDefinitionInput` → `ToolInfo`
2. `AppHostServer` launching a local HTTP + WebSocket server
3. `AppBridge` routing `tools/call`, `ui/message`, `ui/update-model-context`
4. Host page serving a sandboxed iframe with WebSocket communication
5. Full `ui/initialize` handshake and tool call round-trip

## Planning (Tier 6)

Execution plans — reproducible, inspectable, parallelizable tool call graphs.

### Self-Contained Demos (no API key needed)

```bash
# Plan basics: create, inspect, save, load, delete, DAG visualization
uv run python examples/planning/plan_basics_demo.py

# Plan execution: dry-run, live execution, variable resolution, checkpoints, failure handling
uv run python examples/planning/plan_execution_demo.py

# Parallel execution: topological batching, concurrent steps, timing evidence
uv run python examples/planning/plan_parallel_demo.py

# Guard integration: budget limits, per-tool caps, result recording, error handling
uv run python examples/planning/plan_guard_demo.py
```

### LLM-Integrated Demos (requires OPENAI_API_KEY)

```bash
# Full pipeline: LLM generates plan from natural language → validate → visualize → execute
uv run python examples/planning/plan_llm_demo.py

# Use a different model
uv run python examples/planning/plan_llm_demo.py --model gpt-4o

# Custom task description
uv run python examples/planning/plan_llm_demo.py --prompt "fetch weather for 3 cities and compare"

# Plan-as-a-Tool (Tier 6.8): The LLM decides WHEN to plan — uses plan_create_and_execute
# for complex multi-step tasks, calls tools directly for simple ones
uv run python examples/planning/plan_as_tool_demo.py

# Custom task
uv run python examples/planning/plan_as_tool_demo.py --prompt "read the config and run tests"
```

Demonstrates:
1. PlanningContext initialization and PlanRegistry round-trips (save/load/delete)
2. DAG visualization with status indicators and parallel markers
3. Dry-run mode (trace without executing)
4. Parallel batch execution (independent steps run concurrently)
5. Variable resolution (`${var}`, `${var.field}`, template strings)
6. Execution checkpointing and resume support
7. Step failure handling with checkpoint persistence
8. Guard integration (pre-execution blocking, post-execution recording)
9. MCP content block extraction
10. Fan-out, diamond, and wide pipeline DAG patterns with timing evidence
11. LLM plan generation with PlanAgent (auto-retry on validation failure)
12. End-to-end pipeline: natural language → structured plan → parallel execution
13. Model-driven planning: LLM autonomously invokes plan_create_and_execute when tasks need multi-step coordination

## Safety

### Tier 1: Context Safety

```bash
# All 5 context safety mechanisms — no API key needed
python examples/safety/tier1_context_safety_demo.py
```

Demonstrates:
1. Tool result truncation (100K char cap)
2. Reasoning content stripping
3. Conversation history sliding window
4. Infinite context configuration
5. Streaming buffer caps

### Tier 2: Efficiency & Resilience

```bash
# All 8 efficiency features — no API key needed
python examples/safety/tier2_efficiency_demo.py
```

Demonstrates:
1. Single source of truth for tool history (procedural memory)
2. Procedural memory pattern limits
3. System prompt optimization (caching + tool summary)
4. Connection error classification
5. Tool batch timeout
6. Narrower exception handlers
7. Provider validation at startup
8. LLM-visible context management notices

### AI Virtual Memory

```bash
# VM subsystem: budget enforcement, eviction, page lifecycle — no API key needed
python examples/safety/vm_memory_management_demo.py

# E2E recall scenarios: page_fault, search_pages, distractor tools — requires OPENAI_API_KEY
python examples/safety/vm_relaxed_mode_demo.py

# Server health monitoring + VM multimodal content — no API key needed
python examples/safety/health_vm_multimodal_demo.py
```

Demonstrates:
1. Health-check-on-failure and connection error diagnostics
2. Background health polling lifecycle (start, transition detection, stop)
3. `/health` command (all healthy, mixed, missing server)
4. Multimodal page_fault — image pages as multi-block content (text + image_url)
5. Text/structured page_fault with modality and compression metadata
6. search_pages with hint-based matching and modality filtering
7. `/memory page --download` — export text, JSON, and base64 image pages
8. Multi-block content in HistoryMessage serialization
9. Full VM lifecycle: eviction under pressure → search → fault → content blocks
