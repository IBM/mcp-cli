#!/usr/bin/env python3
"""AI Virtual Memory: E2E Recall Scenarios

Replicates the three recall patterns observed in live testing:

  Scenario A — Simple facts:     "What is my name?"
  Scenario B — Creative content:  "What rhymes with 'light' in the poem?"
  Scenario C — Tool result data:  "What was the temperature?"

Each scenario verifies that the model uses page_fault (not other tools)
to retrieve evicted conversation content.  Distractor tools (geocode,
web_search) are included alongside VM tools to match the real MCP
environment where 30+ tools compete for the model's attention.

Requires: OPENAI_API_KEY environment variable
Usage:   uv run python examples/safety/vm_relaxed_mode_demo.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow running from the examples/ directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Load .env for API keys
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────

VM_BUDGET = 500  # Small budget to force evictions
MODEL = os.getenv("VM_DEMO_MODEL", "gpt-5-mini")
PROVIDER = "openai"
MAX_TOOL_ROUNDS = 4  # Max LLM↔tool round-trips per scenario

# ── Conversation with all three content types ────────────────────────────

CONVERSATION = [
    # Turn 1: Simple facts (name, location)
    (
        "user",
        "My name is Chris and I live in Leavenheath, Suffolk. "
        "It's a small village on the Suffolk-Essex border, about "
        "five miles from Colchester. I've lived there for ten years "
        "and really enjoy the countryside and the local community.",
    ),
    (
        "ai",
        "Nice to meet you, Chris! Leavenheath sounds like a wonderful "
        "place to live. Being on the Suffolk-Essex border gives you "
        "easy access to both Colchester and Sudbury. The countryside "
        "around there is beautiful — rolling farmland and quiet lanes. "
        "How can I help you today?",
    ),
    # Turn 2: Filler (Roman history) to push turn 1 out of budget
    (
        "user",
        "Can you tell me about the history of the Roman Empire? "
        "I'm particularly interested in the rise of Augustus and "
        "how the Republic transitioned into an Empire.",
    ),
    (
        "ai",
        "The Roman Republic's transition to the Empire is one of "
        "history's most fascinating political transformations. After "
        "Julius Caesar's assassination in 44 BC, his adopted heir "
        "Octavian formed the Second Triumvirate with Mark Antony and "
        "Lepidus. Following years of civil war, including the decisive "
        "Battle of Actium in 31 BC against Antony and Cleopatra, "
        "Octavian emerged as sole ruler. In 27 BC, the Senate granted "
        "him the title Augustus, marking the official beginning of "
        "the Roman Empire. He cleverly maintained the facade of "
        "Republican institutions while consolidating absolute power.",
    ),
    # Turn 3: More filler
    ("user", "What about the fall of Rome? What were the main causes?"),
    (
        "ai",
        "The fall of the Western Roman Empire in 476 AD resulted from "
        "multiple interconnected factors. Economic troubles including "
        "inflation, over-taxation, and trade disruption weakened the "
        "empire's financial foundations. Military problems included "
        "overreliance on Germanic mercenaries and difficulty defending "
        "the vast borders. Political instability saw dozens of emperors "
        "in the final century, many assassinated. External pressures "
        "from migrating peoples — Visigoths, Vandals, Huns — culminated "
        "in Odoacer deposing the last emperor Romulus Augustulus.",
    ),
    # Turn 4: Creative content (poem — the hardest recall case)
    ("user", "Tell me a poem about the stars tonight."),
    (
        "ai",
        "Beneath the velvet canopy of night, a thousand diamonds catch "
        "the light. Each star a story, old and bright, a lantern hung "
        "at heaven's height. They whisper tales of time and space, of "
        "galaxies in slow embrace, of suns that lived and left no trace "
        "but photons on a journey's face. And we who gaze from earth "
        "below are humbled by the cosmic show — a reminder that our "
        "fleeting days are stardust scattered through the haze.",
    ),
    # Turn 5: Story with a detail buried deep past 120-char hint cutoff.
    # The hint will show "On a street that smelled of fresh croissants..."
    # but the key detail (the olive tree) is ~400 chars in.
    ("user", "Tell me a short story set in Paris."),
    (
        "ai",
        "On a street that smelled of fresh croissants and rain-washed "
        "stone, there was a little bookshop with a green door. The owner, "
        "Madame Lafont, kept a cat named Moustache who slept on the "
        "counter between stacks of unsold novels. Every Tuesday a young "
        "painter came in to browse, leaving charcoal smudges on the spines. "
        "One afternoon he found a dusty notebook wedged behind a shelf. "
        "Inside were sketches of an ancient olive tree that once stood in "
        "the courtyard of the shop — drawn by someone who had clearly loved "
        "it. Madame Lafont smiled when she saw the sketches. 'That was my "
        "grandmother's tree,' she said. 'She planted it the year the war "
        "ended.' The painter asked if he could paint it from the sketches, "
        "and she agreed. He returned each Tuesday with oils and canvas, "
        "slowly bringing the olive tree back to life on the bookshop wall.",
    ),
]

# Simulated tool result — stored as an artifact page to test recall
# of external tool data (weather, time, etc.)
TOOL_RESULT_CONTENT = (
    "Weather for Leavenheath: Temperature 7.3°C, wind 21 km/h from "
    "the west, overcast skies. High today 9.5°C, low 2.3°C. Total "
    "precipitation 0.9mm. Breezy conditions expected through the evening."
)

# Simulated structured data (JSON API response) — Modality.STRUCTURED
STRUCTURED_CONTENT = json.dumps(
    {
        "type": "flight_status",
        "airline": "British Airways",
        "flight": "BA1472",
        "origin": {"code": "LHR", "city": "London Heathrow"},
        "destination": {"code": "EDI", "city": "Edinburgh"},
        "departure": "2026-02-21T14:30:00Z",
        "arrival": "2026-02-21T15:55:00Z",
        "status": "on_time",
        "gate": "B42",
        "terminal": "5",
        "aircraft": "Airbus A320neo",
    },
    indent=2,
)

# Simulated image analysis result — Modality.IMAGE
IMAGE_CONTENT = (
    "Image analysis of uploaded photo (photo_2026-02-20_sunset.jpg): "
    "A landscape photograph taken at sunset from a hilltop. The sky shows "
    "bands of orange and pink fading into deep purple. In the foreground, "
    "a lone oak tree stands silhouetted against the light. A narrow dirt "
    "path leads from the bottom-left corner toward the tree. In the middle "
    "distance, a stone wall runs across the field, and beyond it three "
    "sheep are grazing near a small pond. The pond reflects the sunset "
    "colours. A church spire is visible on the horizon to the right. "
    "EXIF data: Canon EOS R5, 24mm f/8, ISO 100, 1/125s."
)

# ── Recall scenarios ─────────────────────────────────────────────────────


@dataclass
class RecallScenario:
    """A single recall test case."""

    name: str
    question: str
    expected_keywords: list[str]  # Must appear in final answer (lowered)
    description: str
    reject_keywords: list[str] = field(default_factory=list)  # Must NOT appear
    expect_decline: bool = False  # True if model should say "I don't have that"


SCENARIOS = [
    # ── Original three ───────────────────────────────────────────────
    RecallScenario(
        name="Simple facts",
        question="What is my name and where do I live?",
        expected_keywords=["chris", "leavenheath"],
        description="Name and location from turn 1 — should be evicted",
    ),
    RecallScenario(
        name="Creative content",
        question=(
            "In the poem you wrote about stars, what rhymes with 'light' "
            "in the first two lines?"
        ),
        expected_keywords=["night"],
        description="Specific detail from the poem — requires page_fault",
    ),
    RecallScenario(
        name="Tool result data",
        question="What was the temperature in the weather report?",
        expected_keywords=["7.3"],
        description="Weather data stored as an artifact page",
    ),
    # ── New edge cases ───────────────────────────────────────────────
    RecallScenario(
        name="Negative case",
        question="What did I say about my favourite football team?",
        expected_keywords=[],
        description="Never discussed — model should decline, not hallucinate",
        expect_decline=True,
        reject_keywords=[
            "giants",
            "arsenal",
            "chelsea",
            "united",
            "city",
            "liverpool",
            "tottenham",
            "spurs",
            "cowboys",
        ],
    ),
    RecallScenario(
        name="Deep detail",
        question=("In the Paris story, what kind of tree did the grandmother plant?"),
        expected_keywords=["olive"],
        description="Detail buried ~400 chars in — past the 120-char hint cutoff",
    ),
    RecallScenario(
        name="Multi-fault",
        question=(
            "What is my name, and what was the wind speed in the weather report?"
        ),
        expected_keywords=["chris", "21"],
        description="Requires recalling two different evicted pages",
    ),
    # ── Multimodal scenarios ─────────────────────────────────────────
    RecallScenario(
        name="Structured data",
        question="What gate and terminal is my flight departing from?",
        expected_keywords=["b42", "5"],
        description="JSON flight status — Modality.STRUCTURED artifact page",
    ),
    RecallScenario(
        name="Image recall",
        question=(
            "In the photo I uploaded, what animals were in the field "
            "and what were they near?"
        ),
        expected_keywords=["sheep", "pond"],
        description="Image analysis result — Modality.IMAGE artifact page",
    ),
]


# ── Distractor tools ─────────────────────────────────────────────────────
# Fake MCP tools that look like what the model sees in a real session.
# The model should NEVER call these to retrieve conversation history.

DISTRACTOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "geocode_location",
            "description": "Geocode a place name to latitude/longitude coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Place name or address to geocode",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Get weather forecast for a latitude/longitude location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────


def header(n: int, title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {n}. {title}")
    print("=" * 70)


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def info(msg: str) -> None:
    print(f"    {msg}")


@dataclass
class ScenarioResult:
    """Tracks what happened during a single recall scenario."""

    scenario: str
    page_fault_calls: int = 0
    search_pages_calls: int = 0
    distractor_calls: list[str] = field(default_factory=list)
    answer: str = ""
    keywords_found: list[str] = field(default_factory=list)
    keywords_missing: list[str] = field(default_factory=list)
    reject_found: list[str] = field(default_factory=list)  # Bad keywords present

    @property
    def used_correct_tool(self) -> bool:
        """Pass if no distractors, or page_fault was used alongside them.

        The real failure mode is calling a distractor INSTEAD of page_fault
        (e.g., geocode with a page_id). If the model faulted correctly but
        also made a bonus external call, that's acceptable.
        """
        if len(self.distractor_calls) == 0:
            return True
        return self.page_fault_calls > 0  # faulted + extra call = ok

    @property
    def recalled_content(self) -> bool:
        return len(self.keywords_missing) == 0 and len(self.reject_found) == 0


# ── Setup: Session with evictions ────────────────────────────────────────


async def setup_session() -> dict:
    """Create a VM session, play conversation, inject tool result, return context."""
    header(1, "SESSION SETUP")

    from chuk_ai_session_manager.session_manager import SessionManager
    from chuk_ai_session_manager.memory.models import VMMode, PageType
    from chuk_ai_session_manager.memory.working_set import WorkingSetConfig

    sm = SessionManager(
        system_prompt="You are a helpful assistant.",
        enable_vm=True,
        vm_mode=VMMode.STRICT,
        vm_config=WorkingSetConfig(
            max_l0_tokens=VM_BUDGET,
            reserved_tokens=min(100, VM_BUDGET // 4),
        ),
    )
    await sm._ensure_initialized()
    vm = sm.vm

    # Play conversation turns
    for role, content in CONVERSATION:
        vm.new_turn()
        if role == "user":
            await sm.user_says(content)
        else:
            await sm.ai_responds(content, model=MODEL, provider=PROVIDER)

    # Inject simulated tool results as artifact pages
    from chuk_ai_session_manager.memory.models import Modality

    vm.new_turn()
    vm.create_page(
        content=TOOL_RESULT_CONTENT,
        page_type=PageType.ARTIFACT,
        importance=0.4,
        hint=f"get_weather_forecast: {TOOL_RESULT_CONTENT[:100]}",
    )

    vm.new_turn()
    vm.create_page(
        content=STRUCTURED_CONTENT,
        page_type=PageType.ARTIFACT,
        modality=Modality.STRUCTURED,
        importance=0.4,
        hint=f"[structured] flight_status: BA1472 LHR→EDI {STRUCTURED_CONTENT[:80]}",
    )

    vm.new_turn()
    vm.create_page(
        content=IMAGE_CONTENT,
        page_type=PageType.ARTIFACT,
        modality=Modality.IMAGE,
        importance=0.4,
        hint="[image] photo_2026-02-20_sunset.jpg: sunset landscape with tree, path, sheep, pond, church spire",
    )

    # Add more filler to push earlier content out
    vm.new_turn()
    await sm.user_says(
        "That's really interesting about Rome. Can you tell me more "
        "about Byzantine art and architecture? I've always been fascinated "
        "by the Hagia Sophia and its enormous dome. What made it such "
        "an engineering marvel for its time?"
    )
    await sm.ai_responds(
        "The Hagia Sophia is truly one of humanity's greatest architectural "
        "achievements. Built in just five years from 532 to 537 AD under "
        "Emperor Justinian I, it was designed by Anthemius of Tralles and "
        "Isidore of Miletus. The dome spans 31 metres and appears to float "
        "on a ring of windows, creating the famous effect of light cascading "
        "down from heaven. The ingenious use of pendentives — curved triangular "
        "sections that transition from a square base to a circular dome — was "
        "revolutionary for the period.",
        model=MODEL,
        provider=PROVIDER,
    )

    # Check evictions
    stats = vm.get_stats()
    evictions = stats.get("metrics", {}).get("evictions_total", 0)
    total_pages = stats.get("page_table", {}).get("total_pages", 0)
    l0_count = len(vm.working_set.get_l0_page_ids())

    info(f"Total pages: {total_pages}, L0: {l0_count}, Evictions: {evictions}")

    if evictions > 0:
        ok(f"{evictions} evictions (budget={VM_BUDGET} tokens)")
    else:
        fail("No evictions — budget may be too large")
        return {}

    # Build context
    ctx = vm.build_context(system_prompt="You are a helpful assistant.")
    dev_msg = ctx.get("developer_message", "")
    vm_tools = ctx.get("tools", [])

    # Verify manifest has hints
    manifest = ctx["manifest"]
    hints = sum(1 for p in manifest.available_pages if p.hint)
    ok(f"{hints}/{len(manifest.available_pages)} available pages have hints")

    # Show what's evicted vs in-context
    l0_ids = set(vm.working_set.get_l0_page_ids())
    info("Evicted pages (available for recall):")
    for entry in manifest.available_pages:
        tag = "  L0" if entry.page_id in l0_ids else "  **"
        info(f"  {tag} {entry.page_id}: {(entry.hint or '(no hint)')[:60]}")

    return {
        "session_manager": sm,
        "vm": vm,
        "dev_msg": dev_msg,
        "vm_tools": vm_tools,
    }


# ── Infrastructure checks ───────────────────────────────────────────────


def check_developer_message(dev_msg: str) -> None:
    """Quick sanity check on the developer message structure."""
    header(2, "DEVELOPER MESSAGE")

    info(f"Length: {len(dev_msg)} chars")

    for marker, desc in {
        "<VM:RULES>": "VM:RULES block",
        "<VM:MANIFEST_JSON>": "VM:MANIFEST_JSON block",
        "<VM:CONTEXT>": "VM:CONTEXT block",
        "page_fault": "page_fault in rules",
    }.items():
        (ok if marker in dev_msg else fail)(desc)


def check_tools(vm_tools: list) -> list:
    """Combine VM tools with distractor tools, verify shape."""
    header(3, "TOOL SETUP")

    all_tools = vm_tools + DISTRACTOR_TOOLS
    vm_names = [t["function"]["name"] for t in vm_tools]
    distractor_names = [t["function"]["name"] for t in DISTRACTOR_TOOLS]

    ok(f"VM tools: {vm_names}")
    info(f"Distractor tools: {distractor_names}")
    info(f"Total tools available to model: {len(all_tools)}")

    return all_tools


# ── Generic scenario runner ──────────────────────────────────────────────

_VM_TOOL_NAMES = frozenset({"page_fault", "search_pages"})


async def run_scenario(
    scenario: RecallScenario,
    dev_msg: str,
    tools: list,
    vm,
    client,
) -> ScenarioResult:
    """Run a single recall scenario with tool-call loop."""
    result = ScenarioResult(scenario=scenario.name)

    messages = [
        {"role": "system", "content": dev_msg},
        {"role": "user", "content": scenario.question},
    ]

    info(f"Question: {scenario.question}")
    info(f"Expected: {scenario.expected_keywords}")

    # Reset per-turn fault counter so each scenario gets its own allowance
    vm.new_turn()

    for round_num in range(MAX_TOOL_ROUNDS):
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            temperature=1,
        )
        choice = response.choices[0]

        # No tool calls → final answer
        if not choice.message.tool_calls:
            result.answer = choice.message.content or ""
            break

        # Process tool calls
        messages.append(choice.message.model_dump())

        for tc in choice.message.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            info(f"  Round {round_num + 1}: {fn_name}({json.dumps(fn_args)[:80]})")

            if fn_name == "page_fault":
                result.page_fault_calls += 1
                fault = await vm.handle_fault(
                    page_id=fn_args.get("page_id", ""),
                    target_level=fn_args.get("target_level", 2),
                )
                if fault.success and fault.page:
                    content_preview = fault.page.content[:120]
                    info(f"    → content: {content_preview}...")
                    response = {
                        "success": True,
                        "page_id": fault.page.page_id,
                        "content": fault.page.content[:2000],
                        "source_tier": str(fault.source_tier)
                        if fault.source_tier
                        else None,
                    }
                    # Short content hint — likely a user request
                    if len(fault.page.content) < 120:
                        response["note"] = (
                            "Very short content — this may be a user "
                            "request. Check the manifest for the "
                            "[assistant] response page and fault that."
                        )
                    tool_content = json.dumps(response)
                else:
                    info(f"    → failed: {fault.error}")
                    tool_content = json.dumps(
                        {
                            "success": False,
                            "error": fault.error or "Page not found",
                        }
                    )

            elif fn_name == "search_pages":
                result.search_pages_calls += 1
                search = await vm.search_pages(
                    query=fn_args.get("query", ""),
                    modality=fn_args.get("modality"),
                    limit=fn_args.get("limit", 5),
                )
                tool_content = search.to_json()

            else:
                # Distractor tool was called — this is a failure
                result.distractor_calls.append(fn_name)
                tool_content = json.dumps(
                    {
                        "error": f"Tool '{fn_name}' cannot retrieve conversation "
                        f"history. Use page_fault instead.",
                    }
                )

            messages.append(
                {
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": tool_content,
                }
            )
    else:
        # Exhausted rounds without a final answer
        result.answer = "(no final answer — tool loop exhausted)"

    # Check keywords
    lower = result.answer.lower()
    for kw in scenario.expected_keywords:
        if kw.lower() in lower:
            result.keywords_found.append(kw)
        else:
            result.keywords_missing.append(kw)

    # Check reject keywords (things that should NOT appear)
    for kw in scenario.reject_keywords:
        if kw.lower() in lower:
            result.reject_found.append(kw)

    return result


# ── Main ─────────────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 70)
    print("  AI Virtual Memory — E2E Recall Scenarios")
    print("=" * 70)
    print(f"  Budget: {VM_BUDGET} tokens | Model: {MODEL} | Mode: STRICT")
    print(f"  OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")

    # Phase 1: Setup
    ctx = await setup_session()
    if not ctx:
        print("\n✗ Setup failed — cannot continue")
        sys.exit(1)

    dev_msg = ctx["dev_msg"]
    vm = ctx["vm"]
    vm_tools = ctx["vm_tools"]

    # Phase 2: Infrastructure checks
    check_developer_message(dev_msg)
    all_tools = check_tools(vm_tools)

    # Phase 3: Run recall scenarios
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        info("Skipping LLM scenarios — OPENAI_API_KEY not set")
        info("Set OPENAI_API_KEY to run the full demo")
        return

    try:
        from openai import AsyncOpenAI
    except ImportError:
        info("Skipping LLM scenarios — openai package not installed")
        return

    client = AsyncOpenAI(api_key=api_key)
    results: list[ScenarioResult] = []

    for i, scenario in enumerate(SCENARIOS, start=4):
        header(i, f"SCENARIO: {scenario.name.upper()}")
        info(scenario.description)

        sr = await run_scenario(scenario, dev_msg, all_tools, vm, client)
        results.append(sr)

        # Report
        if sr.page_fault_calls > 0:
            ok(f"page_fault called {sr.page_fault_calls} time(s)")
        else:
            info("page_fault not called (answered from hints or context)")

        if sr.search_pages_calls > 0:
            info(f"search_pages called {sr.search_pages_calls} time(s)")

        if sr.distractor_calls and sr.page_fault_calls == 0:
            fail(f"Distractor tools called WITHOUT page_fault: {sr.distractor_calls}")
        elif sr.distractor_calls:
            info(
                f"Distractor tools called (alongside page_fault): {sr.distractor_calls}"
            )
        else:
            ok("No distractor tools called")

        if scenario.expect_decline:
            if sr.reject_found:
                fail(f"Hallucinated content: {sr.reject_found}")
            else:
                ok("Correctly declined (no hallucinated content)")
        elif sr.recalled_content:
            ok(f"Answer contains expected keywords: {sr.keywords_found}")
        else:
            if sr.keywords_missing:
                fail(f"Missing keywords: {sr.keywords_missing}")
            if sr.reject_found:
                fail(f"Hallucinated content: {sr.reject_found}")

        info(f"Answer: {sr.answer[:150]}...")

    # Phase 4: Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    all_passed = True
    for sr in results:
        tool_status = "✓" if sr.used_correct_tool else "✗"
        recall_status = "✓" if sr.recalled_content else "✗"
        line = (
            f"  {tool_status} Tool  {recall_status} Recall  │ "
            f"{sr.scenario:<20} │ "
            f"faults={sr.page_fault_calls} "
            f"distractors={sr.distractor_calls or 'none'}"
        )
        print(line)
        if not sr.used_correct_tool or not sr.recalled_content:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("  ✓ ALL SCENARIOS PASSED")
    else:
        print("  ✗ SOME SCENARIOS FAILED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
