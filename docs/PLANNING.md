# Execution Plans

Execution plans are reproducible, inspectable, parallelizable tool call graphs. Instead of ad-hoc conversation-driven tool use, plans define a structured DAG of steps with explicit dependencies, variable bindings, and guard integration.

**Powered by:** [chuk-ai-planner](https://github.com/chrishayuk/chuk-ai-planner) — graph-based plan DSL, executor, and LLM plan generation.

## Quick Start

```bash
# In chat mode or interactive mode:
/plan create "read the auth module, find all usages, then refactor"
/plan list
/plan show <id>
/plan run <id>
/plan run <id> --dry-run
/plan resume <id>
/plan delete <id>
```

## Architecture

```
User Intent
    ↓
PlanAgent (LLM)          ← generates plan from description
    ↓
UniversalPlan            ← graph-based plan object
    ↓
PlanRunner               ← orchestrates execution
    ↓
McpToolBackend           ← bridges to ToolManager
    ↓
MCP Servers              ← actual tool execution
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `McpToolBackend` | `planning/backends.py` | Bridges chuk-ai-planner's `ToolExecutionBackend` protocol to `ToolManager.execute_tool()` |
| `PlanningContext` | `planning/context.py` | State container: graph store, plan registry, tool manager, tool catalog |
| `PlanRunner` | `planning/executor.py` | Orchestrates execution: batching, concurrency, checkpointing, dry-run, re-planning |
| `PlanCommand` | `commands/plan/plan.py` | Unified command interface (`/plan` in all modes) |
| `plan tools` | `planning/tools.py` | Tool definitions + handlers for model-driven planning (`--plan-tools`) |

### Module Layout

```
src/mcp_cli/
  planning/
    __init__.py          # Public API exports
    backends.py          # McpToolBackend + guard helpers
    context.py           # PlanningContext (state + registry)
    executor.py          # PlanRunner + batching + variables + DAG viz
    tools.py             # Plan-as-a-Tool: LLM-callable plan tools (--plan-tools)
  commands/
    plan/
      plan.py            # /plan command (create, list, show, run, delete, resume)
```

## Plan Format

Plans are JSON objects stored at `~/.mcp-cli/plans/`:

```json
{
  "id": "refactor-auth-001",
  "title": "Refactor Auth Module",
  "variables": {
    "module_path": "src/auth/handler.py"
  },
  "steps": [
    {
      "index": "1",
      "title": "Read auth module",
      "tool": "read_file",
      "args": {"path": "${module_path}"},
      "depends_on": [],
      "result_variable": "auth_code"
    },
    {
      "index": "2",
      "title": "Find all usages",
      "tool": "search_code",
      "args": {"query": "from auth.handler import"},
      "depends_on": [],
      "result_variable": "usages"
    },
    {
      "index": "3",
      "title": "Write refactored module",
      "tool": "write_file",
      "args": {"path": "${module_path}", "content": "refactored code"},
      "depends_on": ["1", "2"],
      "result_variable": "write_result"
    },
    {
      "index": "4",
      "title": "Run tests",
      "tool": "run_tests",
      "args": {"path": "tests/auth/"},
      "depends_on": ["3"],
      "result_variable": "test_results"
    }
  ]
}
```

### Step Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `index` | string | yes | Unique step identifier (1-based) |
| `title` | string | yes | Human-readable step description |
| `tool` | string | yes | MCP tool name to execute |
| `args` | dict | yes | Tool arguments (may contain `${var}` references) |
| `depends_on` | list[string] | yes | Indices of steps that must complete first |
| `result_variable` | string | no | Variable name to store the step's result |

## Parallel Execution

Steps are grouped into **topological batches** using Kahn's BFS algorithm. Steps within a batch have no dependencies on each other and run concurrently via `asyncio.gather()`.

### Example: Diamond DAG

```
Step 1: Fetch data        (no deps)
Step 2: Validate schema   (depends on 1)
Step 3: Transform format  (depends on 1)
Step 4: Aggregate results (depends on 2, 3)
```

Batches:
- **Batch 1**: Step 1 (serial — single step)
- **Batch 2**: Steps 2, 3 (parallel — both depend only on step 1)
- **Batch 3**: Step 4 (serial — waits for batch 2)

With 200ms per tool call, the diamond executes in ~600ms instead of ~800ms serial.

### Concurrency Control

```python
runner = PlanRunner(context, max_concurrency=4)
```

The `max_concurrency` parameter limits how many steps run simultaneously within a batch using `asyncio.Semaphore`. Default is 4.

## Variable Resolution

Step outputs can be stored as variables and referenced by later steps.

### Patterns

| Pattern | Behavior | Example |
|---------|----------|---------|
| `${var}` | Direct replacement (type-preserving) | `{"data": "${users}"}` → `{"data": [{"id": 1}, ...]}` |
| `${var.field}` | Nested dict access | `{"host": "${api.host}"}` → `{"host": "api.example.com"}` |
| `"text ${var} more"` | Template string interpolation | `"https://${api.host}/${api.version}/users"` → `"https://api.example.com/v2/users"` |

**Type preservation:** A single `${var}` reference returns the original value (dict, list, int, etc.). Template strings with surrounding text always produce strings.

### Plan Variables

Plans can define initial variables in the `variables` field:

```json
{
  "variables": {
    "api": {"host": "api.example.com", "version": "v2"},
    "output_dir": "/tmp/results"
  }
}
```

Variables can also be passed at execution time:

```python
result = await runner.execute_plan(plan, variables={"date": "2026-03-01"})
```

## Dry-Run Mode

Trace what a plan would do without executing any tools:

```
/plan run <id> --dry-run
```

In dry-run mode:
- Each step is logged with its tool name, resolved arguments, and dependencies
- Variables are simulated (bound to `<tool_name result>` placeholders)
- No tools are executed — safe to run in production
- Returns a `PlanExecutionResult` with all step results marked as dry-run

## Checkpointing & Resume

Execution state is persisted after each batch to `~/.mcp-cli/plans/{id}_state.json`:

```json
{
  "plan_id": "refactor-auth-001",
  "status": "running",
  "completed_steps": ["1", "2"],
  "variables": {
    "auth_code": "def handle_auth(request)...",
    "usages": "Found 12 usages across 5 files..."
  }
}
```

If execution is interrupted (crash, Ctrl+C, step failure), resume with:

```
/plan resume <id>
```

This loads the checkpoint, skips completed steps, and continues from where it left off with the saved variable context.

## Guard Integration

Plan execution respects mcp-cli's existing guard infrastructure:

- **Pre-execution checks**: `ToolStateManager.check_all_guards()` runs before each step — budget limits, runaway detection, per-tool caps
- **Post-execution recording**: `ToolStateManager.record_tool_call()` tracks usage and updates value bindings
- **Shared budget**: Plan tool calls count against the same budget as conversation tool calls
- **Guard blocking**: If a guard blocks a step (e.g., budget exhausted), the step fails with a guard error message and the tool is never called

Guards can be disabled for testing:

```python
backend = McpToolBackend(tool_manager, enable_guards=False)
```

## Re-planning

When enabled, the runner can invoke the LLM to generate a revised plan on step failure:

```python
runner = PlanRunner(
    context,
    enable_replan=True,   # Off by default
    max_replans=2,        # Max re-plan attempts
)
```

On failure:
1. The runner collects context: completed steps, failed step error, remaining steps, current variables
2. A `PlanAgent` generates a revised plan for the remaining work
3. The revised plan executes with the current variable context
4. Results are merged: completed steps from the original plan + steps from the revised plan
5. `PlanExecutionResult.replanned = True` indicates re-planning occurred

Re-planning does not recurse — a revised plan that fails simply fails.

## DAG Visualization

Plans render as ASCII DAGs in the terminal:

```
○ 1. Read auth module                    [read_file]
○ 2. Find all usages                     [search_code]  ∥

○ 3. Write refactored module             [write_file]   ← after: 1, 2

○ 4. Run tests                           [run_tests]    ← after: 3
```

Status indicators:
- `○` pending
- `◉` running
- `●` completed
- `✗` failed
- `∥` parallel (runs concurrently with other steps in the same batch)

Use `render_plan_dag(plan_data)` programmatically:

```python
from mcp_cli.planning.executor import render_plan_dag

dag = render_plan_dag(plan_data)
print(dag)
```

## Programmatic API

### PlanRunner

```python
from mcp_cli.planning.context import PlanningContext
from mcp_cli.planning.executor import PlanRunner

# Create context with a ToolManager
ctx = PlanningContext(tool_manager)

# Create runner with options
runner = PlanRunner(
    ctx,
    on_step_start=lambda idx, title, tool: print(f"  [{idx}] {title}"),
    on_step_complete=lambda result: print(f"    -> {'OK' if result.success else 'FAIL'}"),
    enable_guards=True,
    max_concurrency=4,
    enable_replan=False,
)

# Execute a plan
result = await runner.execute_plan(plan_data, dry_run=False, checkpoint=True)

print(f"Success: {result.success}")
print(f"Steps: {len(result.steps)}")
print(f"Duration: {result.total_duration:.2f}s")
print(f"Variables: {list(result.variables.keys())}")
```

### McpToolBackend

```python
from mcp_cli.planning.backends import McpToolBackend
from chuk_ai_planner.execution.models import ToolExecutionRequest

backend = McpToolBackend(tool_manager, enable_guards=True)

request = ToolExecutionRequest(
    tool_name="read_file",
    args={"path": "/tmp/test.txt"},
    step_id="step-1",
)
result = await backend.execute_tool(request)
```

### Batch Computation

```python
from mcp_cli.planning.executor import _compute_batches

steps = [
    {"index": "1", "title": "Fetch", "depends_on": []},
    {"index": "2", "title": "Parse A", "depends_on": ["1"]},
    {"index": "3", "title": "Parse B", "depends_on": ["1"]},
    {"index": "4", "title": "Merge", "depends_on": ["2", "3"]},
]

batches = _compute_batches(steps)
# [[step1], [step2, step3], [step4]]
```

## Model-Driven Planning (Plan as a Tool)

With the `--plan-tools` flag, the LLM can autonomously create and execute plans during conversation. Instead of the user typing `/plan create`, the model itself decides when multi-step orchestration is needed.

### Enabling

```bash
# Enable plan tools in chat mode
mcp-cli --server sqlite --plan-tools

# Or with the chat subcommand
mcp-cli chat --server sqlite --plan-tools
```

### How It Works

Three internal tools are injected into the LLM's tool list:

| Tool | Purpose |
|------|---------|
| `plan_create` | Generate a plan from a goal description, returns plan ID + step summary |
| `plan_execute` | Execute a previously created plan by ID |
| `plan_create_and_execute` | Generate and execute in one call (most common) |

These tools are **intercepted** in `tool_processor.py` before MCP routing — the same pattern used by VM and memory tools. They never reach the MCP server.

### Example Flow

```
User: "Read the auth module, find all files that import it, and run the tests"

Model (internally): This needs 3 coordinated steps.
  → calls plan_create_and_execute(goal="Read auth module, find importers, run tests")
  → PlanAgent generates: [read_file] → [search_code] → [run_tests]
  → PlanRunner executes all 3 steps
  → Results returned as tool result

Model: "The auth module contains handle_auth() and verify_jwt().
        It's imported in 6 files across src/ and tests/.
        All 8 tests passed (2 skipped)."
```

For simple single-tool tasks, the model calls the tool directly — no planning overhead.

### Display Integration

Plan execution renders step-by-step in the terminal using the same `StreamingDisplayManager` as regular tool calls. Each MCP tool call within the plan gets its own spinner and result display:

```
✓ plan_create_and_execute completed in 17.08s
   Result: Plan generated: Weather for Leavenheath (2 steps)
✓ geocode_location completed in 0.58s
   Result keys: results, generationtime_ms
✓ get_weather_forecast completed in 0.43s
   Result keys: latitude, longitude, elevation, ...
```

The `ui_manager` is passed through from `tool_processor.py` → `handle_plan_tool()` → `PlanRunner` callbacks, so the user sees real-time progress rather than a single long-running spinner.

### Programmatic API

```python
from mcp_cli.planning.tools import get_plan_tools_as_dicts, handle_plan_tool
from mcp_cli.planning.context import PlanningContext

# Get OpenAI-format tool definitions
plan_tools = get_plan_tools_as_dicts()  # 3 tool dicts

# Execute a plan tool (ui_manager is optional, for step-by-step display)
ctx = PlanningContext(tool_manager)
result_json = await handle_plan_tool(
    "plan_create_and_execute",
    {"goal": "Read file and run tests"},
    ctx,
    ui_manager=ui_manager,  # optional: enables per-step progress
)
```

## Examples

Self-contained demos in `examples/planning/` (no API key or MCP server needed):

```bash
# Plan CRUD, DAG visualization, persistence
uv run python examples/planning/plan_basics_demo.py

# Dry-run, live execution, variables, checkpoints, failure handling
uv run python examples/planning/plan_execution_demo.py

# Topological batching, concurrent steps, timing evidence
uv run python examples/planning/plan_parallel_demo.py

# Budget limits, per-tool caps, result recording, error handling
uv run python examples/planning/plan_guard_demo.py
```

### Model-Driven Planning Demo (requires OPENAI_API_KEY)

```bash
# LLM decides WHEN to plan — uses plan_create_and_execute for complex tasks,
# calls tools directly for simple ones
uv run python examples/planning/plan_as_tool_demo.py

# Use a different model
uv run python examples/planning/plan_as_tool_demo.py --model gpt-4o

# Custom task description
uv run python examples/planning/plan_as_tool_demo.py --prompt "read the config, search for usages, and run tests"
```

## Tests

200+ tests covering all planning functionality:

```bash
# Run planning tests
uv run pytest tests/planning/ -v

# Test files:
#   tests/planning/test_backends.py   — McpToolBackend, guards, result extraction
#   tests/planning/test_context.py    — PlanningContext, PlanRegistry round-trips
#   tests/planning/test_executor.py   — PlanRunner, batching, variables, DAG, re-planning
#   tests/planning/test_tools.py      — Plan-as-a-Tool definitions, validation, handlers
```
