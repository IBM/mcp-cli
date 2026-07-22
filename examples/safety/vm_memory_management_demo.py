#!/usr/bin/env python3
"""AI Virtual Memory: End-to-End Memory Management Demo

Proves that the VM subsystem correctly enforces token budgets, triggers
eviction under pressure, and tracks pages across tiers.

Demonstrates:
  1. Budget enforcement — TokenBudget.total_limit syncs from WorkingSetConfig
  2. Page creation and L0 admission
  3. Eviction under pressure — pages evicted when budget exceeded
  4. Turn advancement — turn counter increments for recency tracking
  5. VM context building — developer_message reflects L0 contents only
  6. Event filtering — _vm_filter_events respects budget for raw events

No API keys or MCP servers required — runs fully self-contained.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running from the examples/ directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def demo_budget_enforcement() -> None:
    """1 — TokenBudget syncs from WorkingSetConfig on construction."""
    print("=" * 70)
    print("1  BUDGET ENFORCEMENT")
    print("=" * 70)

    from chuk_ai_session_manager.memory.working_set import (
        WorkingSetConfig,
        WorkingSetManager,
    )

    # Default budget (128K)
    default_wsm = WorkingSetManager()
    print(f"  Default total_limit:     {default_wsm.budget.total_limit:>10,}")
    print(f"  Default reserved:        {default_wsm.budget.reserved:>10,}")
    print(f"  Default available:       {default_wsm.budget.available:>10,}")
    print()

    # Custom budget (500 tokens) — the fix ensures this flows to TokenBudget
    config = WorkingSetConfig(max_l0_tokens=500, reserved_tokens=125)
    wsm = WorkingSetManager(config=config)
    print(f"  Custom total_limit:      {wsm.budget.total_limit:>10,}")
    print(f"  Custom reserved:         {wsm.budget.reserved:>10,}")
    print(f"  Custom available:        {wsm.budget.available:>10,}")

    assert wsm.budget.total_limit == 500, "Budget total_limit must match config!"
    assert wsm.budget.reserved == 125, "Budget reserved must match config!"
    assert wsm.budget.available == 375, "Available must be total - reserved!"
    print("  Budget enforcement:      PASS")
    print()


def demo_page_lifecycle() -> None:
    """2 — Pages are created, admitted to L0, and fill the budget."""
    print("=" * 70)
    print("2  PAGE LIFECYCLE — CREATE, ADMIT, FILL")
    print("=" * 70)

    from chuk_ai_session_manager.memory.working_set import (
        WorkingSetConfig,
        WorkingSetManager,
    )
    from chuk_ai_session_manager.memory.models import (
        MemoryPage,
        Modality,
        PageType,
    )

    config = WorkingSetConfig(max_l0_tokens=500, reserved_tokens=125)
    wsm = WorkingSetManager(config=config)

    # Add pages until budget is full (375 available, 50 tokens each = 7 fit)
    results = []
    for i in range(10):
        page = MemoryPage(
            page_id=f"msg_{i:03d}",
            content=f"User message number {i}",
            page_type=PageType.TRANSCRIPT,
            modality=Modality.TEXT,
            size_tokens=50,
        )
        ok = wsm.add_to_l0(page)
        results.append((f"msg_{i:03d}", ok, wsm.tokens_used, wsm.tokens_available))
        status = "admitted" if ok else "REJECTED"
        print(
            f"  Page msg_{i:03d}: {status:>8}  "
            f"used={wsm.tokens_used:>4}  avail={wsm.tokens_available:>4}  "
            f"util={wsm.utilization:.0%}"
        )

    admitted = sum(1 for _, ok, _, _ in results if ok)
    rejected = sum(1 for _, ok, _, _ in results if not ok)
    print()
    print(f"  Admitted: {admitted}   Rejected: {rejected}")
    print(f"  L0 pages: {wsm.l0_count}   Tokens used: {wsm.tokens_used}")
    print(f"  Needs eviction: {wsm.needs_eviction()}")

    assert admitted == 7, f"Expected 7 admitted, got {admitted}"
    assert rejected == 3, f"Expected 3 rejected, got {rejected}"
    assert wsm.needs_eviction(), "Should need eviction at 93% utilization"
    print("  Page lifecycle:          PASS")
    print()


async def demo_eviction_under_pressure() -> None:
    """3 — MemoryManager evicts pages when budget is exceeded."""
    print("=" * 70)
    print("3  EVICTION UNDER PRESSURE")
    print("=" * 70)

    from chuk_ai_session_manager.memory.manager import MemoryManager
    from chuk_ai_session_manager.memory.working_set import WorkingSetConfig
    from chuk_ai_session_manager.memory.models import (
        PageType,
        VMMode,
    )

    config = WorkingSetConfig(max_l0_tokens=500, reserved_tokens=50)
    mm = MemoryManager(
        session_id="demo-eviction",
        mode=VMMode.PASSIVE,
        config=config,
    )

    print(
        f"  Budget: {config.max_l0_tokens} tokens "
        f"(reserved={config.reserved_tokens}, usable={config.max_l0_tokens - config.reserved_tokens})"
    )
    print(f"  Eviction threshold: {config.eviction_threshold:.0%}")
    print(f"  Target utilization: {config.target_utilization:.0%}")
    print()

    # Simulate a conversation — longer messages to trigger eviction.
    # At ~4 chars/token, each message needs to be ~200+ chars to use ~50 tokens.
    messages = [
        (
            "user",
            "My name is Chris and I live in Leavenheath, a village in Suffolk, England. "
            "I'd like to learn about various topics today including history, weather, and jokes.",
        ),
        (
            "ai",
            "Nice to meet you, Chris! Leavenheath is a lovely village in the Babergh district of Suffolk. "
            "I'd be happy to help you explore any topics you're interested in today. What shall we start with?",
        ),
        (
            "user",
            "Tell me about the history of the New York Football Giants, especially their founding and early years "
            "in the NFL. I'm particularly interested in the pre-Super Bowl era championships they won.",
        ),
        (
            "ai",
            "The New York Giants were founded in 1925 by Tim Mara, who purchased the franchise for just $500. "
            "They quickly became one of the NFL's premier franchises, winning championships in 1927, 1934, 1938, "
            "and 1956. The 1934 title game is famously known as the 'Sneakers Game' where they switched footwear.",
        ),
        (
            "user",
            "What's the weather like where I am right now? I'm curious about the temperature, wind conditions, "
            "and whether I should expect rain tonight or tomorrow morning in my area.",
        ),
        (
            "ai",
            "In Leavenheath right now it's 8.5 degrees Celsius with partly cloudy skies. "
            "The wind is coming from the west at 26.7 km/h which makes it feel quite breezy. "
            "You can expect some light rain showers overnight with temperatures dropping to around 6-7C. "
            "Tomorrow morning should be mostly dry but overcast with highs reaching about 9.5C.",
        ),
        (
            "user",
            "Tell me a really good joke about cheese. Something that would make people at a dinner party laugh. "
            "I want to impress my friends with my comedy skills this weekend.",
        ),
        (
            "ai",
            "Here are three cheese jokes for your dinner party: What type of cheese is made backwards? Edam! "
            "What did the cheese say when it looked in the mirror? Hallou-mi! "
            "Why did the cheese refuse to be sliced? It had grater plans! "
            "The Edam one always gets the biggest laugh because people take a second to figure it out.",
        ),
        (
            "user",
            "What's my name again? I want to make sure you remember our conversation from the beginning. "
            "Also, where did I say I live? Just checking your memory.",
        ),
        (
            "ai",
            "Your name is Chris, and you told me at the start of our conversation that you live in Leavenheath, "
            "a village in the Babergh district of Suffolk, England. We've been chatting about the New York Giants "
            "history, the current weather in your area, and cheese jokes for your dinner party this weekend!",
        ),
    ]

    for i, (role, content) in enumerate(messages):
        page = mm.create_page(
            content=content,
            page_type=PageType.TRANSCRIPT,
            importance=0.7 if role == "user" else 0.5,
            page_id=f"msg_{i:03d}",
        )
        await mm.add_to_working_set(page)

        ws = mm.working_set
        pt = mm.page_table.get_stats()
        print(
            f"  Turn {i:>2} ({role:>4}): "
            f"L0={ws.l0_count:>2} pages  "
            f"tokens={ws.tokens_used:>4}/{config.max_l0_tokens}  "
            f"evictions={mm.metrics.evictions_total}  "
            f"total_pages={pt.total_pages}"
        )

    evictions_after = mm.metrics.evictions_total
    print()
    print(f"  Total evictions: {evictions_after}")
    print(f"  Pages in L0: {mm.working_set.l0_count}")
    print(f"  Pages in page table: {mm.page_table.get_stats().total_pages}")
    print(f"  L0 tokens: {mm.working_set.tokens_used}")
    print(f"  L0 utilization: {mm.working_set.utilization:.0%}")

    assert evictions_after > 0, (
        "Must have evictions with 10 messages in 500-token budget!"
    )
    assert mm.working_set.tokens_used <= config.max_l0_tokens, (
        "L0 must not exceed budget!"
    )
    print("  Eviction under pressure: PASS")
    print()


def demo_turn_tracking() -> None:
    """4 — Turn counter increments for recency-based eviction."""
    print("=" * 70)
    print("4  TURN TRACKING")
    print("=" * 70)

    from chuk_ai_session_manager.memory.manager import MemoryManager
    from chuk_ai_session_manager.memory.working_set import WorkingSetConfig
    from chuk_ai_session_manager.memory.models import VMMode

    config = WorkingSetConfig(max_l0_tokens=1000)
    mm = MemoryManager(
        session_id="demo-turns",
        mode=VMMode.PASSIVE,
        config=config,
    )

    print(f"  Initial turn: {mm.turn}")
    assert mm.turn == 0

    for i in range(5):
        mm.new_turn()
        print(f"  After new_turn(): {mm.turn}")

    assert mm.turn == 5, f"Expected turn 5, got {mm.turn}"
    print("  Turn tracking:           PASS")
    print()


def demo_context_building() -> None:
    """5 — VM context building produces developer_message with L0 pages."""
    print("=" * 70)
    print("5  CONTEXT BUILDING (developer_message)")
    print("=" * 70)

    from chuk_ai_session_manager.memory.manager import MemoryManager
    from chuk_ai_session_manager.memory.working_set import WorkingSetConfig
    from chuk_ai_session_manager.memory.models import PageType, VMMode

    config = WorkingSetConfig(max_l0_tokens=2000, reserved_tokens=200)
    mm = MemoryManager(
        session_id="demo-context",
        mode=VMMode.PASSIVE,
        config=config,
    )

    # Add a few pages
    for i, content in enumerate(
        [
            "Hello, my name is Chris",
            "I live in Leavenheath, Suffolk",
            "The weather is 8.5C with rain",
        ]
    ):
        page = mm.create_page(
            content=content,
            page_type=PageType.TRANSCRIPT,
            page_id=f"ctx_{i:03d}",
        )
        # Synchronous workaround: directly add to L0
        mm._working_set.add_to_l0(page)
        mm._page_table.update_location(
            page.page_id, tier=mm._page_table.entries[page.page_id].tier
        )

    # Build context
    ctx = mm.build_context(system_prompt="You are a helpful assistant.")
    dev_msg = ctx["developer_message"]
    packed = ctx["packed_context"]
    tools = ctx["tools"]

    print("  VM mode:                 passive")
    print(f"  L0 pages:                {mm.working_set.l0_count}")
    print(f"  Developer msg length:    {len(dev_msg):,} chars")
    print(f"  Packed pages included:   {len(packed.pages_included)}")
    print(f"  Packed pages omitted:    {len(packed.pages_omitted)}")
    print(f"  Packed tokens est:       {packed.tokens_est}")
    print(f"  VM tools (passive=0):    {len(tools)}")
    print()

    # Verify structure
    assert "<VM:CONTEXT>" in dev_msg, "Must contain VM:CONTEXT block"
    assert "You are a helpful assistant" in dev_msg, "Must contain system prompt"
    assert "Chris" in dev_msg, "Must contain L0 page content"

    # Passive mode: no VM:RULES or VM:MANIFEST
    assert "<VM:RULES>" not in dev_msg, "Passive mode must NOT include VM:RULES"
    assert "<VM:MANIFEST_JSON>" not in dev_msg, "Passive mode must NOT include manifest"
    assert len(tools) == 0, "Passive mode must NOT include VM tools"

    print("  Context content preview:")
    for line in dev_msg.split("\n"):
        if line.strip():
            print(f"    {line[:80]}")
    print()
    print("  Context building:        PASS")
    print()


async def demo_event_filtering() -> None:
    """6 — _vm_filter_events keeps recent turns within budget."""
    print("=" * 70)
    print("6  EVENT FILTERING (_vm_filter_events)")
    print("=" * 70)

    from unittest.mock import Mock
    from mcp_cli.chat.chat_context import ChatContext
    from mcp_cli.chat.models import HistoryMessage, MessageRole
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "demo"
    mock_manager.get_active_model.return_value = "demo-model"
    mock_tm = Mock()

    # Small budget to force filtering
    ctx = ChatContext(
        tool_manager=mock_tm,
        model_manager=mock_manager,
        enable_vm=True,
        vm_budget=300,
    )
    ctx._system_prompt = "You are helpful."
    await ctx._initialize_session()

    # Simulate 10 turns of longer messages (~60 tokens each = ~240 chars)
    events = []
    for i in range(10):
        events.append(
            HistoryMessage(
                role=MessageRole.USER,
                content=f"User message {i}: "
                + "This is a detailed question about a complex topic. " * 4,
            )
        )
        events.append(
            HistoryMessage(
                role=MessageRole.ASSISTANT,
                content=f"Assistant reply {i}: "
                + "Here is a comprehensive answer with supporting details. " * 4,
            )
        )

    print(f"  Total events:            {len(events)}")
    print(f"  VM budget:               {ctx._vm_budget} tokens")
    print(f"  Min recent turns:        {ctx._VM_MIN_RECENT_TURNS}")

    filtered = ctx._vm_filter_events(events, "system prompt")

    # Count turns in filtered output
    filtered_turns = sum(1 for m in filtered if m.role == MessageRole.USER)

    print(f"  Events after filter:     {len(filtered)}")
    print(f"  Turns after filter:      {filtered_turns}")
    print()

    # Should keep at least 3 recent turns but not all 10
    assert filtered_turns >= ctx._VM_MIN_RECENT_TURNS, "Must keep minimum recent turns"
    assert filtered_turns < 10, "Must filter some older turns with small budget"

    # Verify most recent messages are preserved
    last_filtered = filtered[-1].content
    assert "9" in last_filtered, (
        f"Last message must be from turn 9, got: {last_filtered}"
    )
    print(f"  Oldest kept:             {filtered[0].content[:50]}")
    print(f"  Newest kept:             {filtered[-1].content[:50]}")
    print("  Event filtering:         PASS")
    print()


async def demo_full_integration() -> None:
    """7 — Full integration: ChatContext with VM produces filtered history."""
    print("=" * 70)
    print("7  FULL INTEGRATION (ChatContext + VM)")
    print("=" * 70)

    from unittest.mock import Mock
    from mcp_cli.chat.chat_context import ChatContext
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "demo"
    mock_manager.get_active_model.return_value = "demo-model"
    mock_tm = Mock()

    ctx = ChatContext(
        tool_manager=mock_tm,
        model_manager=mock_manager,
        enable_vm=True,
        vm_budget=500,
        vm_mode="passive",
    )
    ctx._system_prompt = "You are a helpful assistant."
    await ctx._initialize_session()

    # Verify VM is enabled
    assert ctx.session.vm is not None, "VM must be enabled"
    print("  VM enabled:              True")
    print(f"  VM mode:                 {ctx._vm_mode}")
    print(f"  VM budget:               {ctx._vm_budget}")

    # Check budget is properly configured
    ws = ctx.session.vm.working_set
    print(f"  L0 total_limit:          {ws.budget.total_limit}")
    print(f"  L0 reserved:             {ws.budget.reserved}")
    print(f"  L0 available:            {ws.budget.available}")

    assert ws.budget.total_limit == 500, (
        f"Budget must be 500, got {ws.budget.total_limit}"
    )
    assert ws.budget.reserved == 125, f"Reserved must be 125, got {ws.budget.reserved}"

    # Simulate conversation with realistic-length messages (~40-80 tokens each)
    exchanges = [
        (
            "My name is Chris and I live in Leavenheath, a village in Suffolk, England. "
            "I'd like to learn about various topics today.",
            "Nice to meet you, Chris! Leavenheath is a lovely village in the Babergh district "
            "of Suffolk. I'd be happy to help with any topics you're interested in.",
        ),
        (
            "Tell me about the history of the New York Football Giants, especially their founding "
            "and early years in the NFL and the pre-Super Bowl championships.",
            "The Giants were founded in 1925 by Tim Mara for $500. They won championships in 1927, "
            "1934, 1938, and 1956. The 1934 game is known as the famous Sneakers Game.",
        ),
        (
            "What's the weather like where I am right now? Temperature, wind conditions, and whether "
            "I should expect rain tonight or tomorrow morning.",
            "In Leavenheath it's currently 8.5C with partly cloudy skies and westerly winds at 26.7km/h. "
            "Expect light showers overnight dropping to 6-7C. Tomorrow will be mostly dry.",
        ),
        (
            "Tell me a really good joke about cheese, something for a dinner party this weekend.",
            "What type of cheese is made backwards? Edam! What did the cheese say in the mirror? "
            "Hallou-mi! Why did cheese refuse to be sliced? It had grater plans!",
        ),
        (
            "What's my name again? And where did I say I live? Just checking your memory.",
            "Your name is Chris and you live in Leavenheath, Suffolk. We've discussed Giants history, "
            "weather in your area, and cheese jokes for your dinner party!",
        ),
        (
            "Can you summarize everything we've talked about today in this conversation?",
            "We covered: your introduction as Chris from Leavenheath Suffolk, the New York Giants "
            "history from 1925 founding through Super Bowl wins, current weather of 8.5C with rain, "
            "cheese jokes for your dinner party, and a memory check which I passed.",
        ),
    ]

    for user_msg, ai_msg in exchanges:
        await ctx.add_user_message(user_msg)
        await ctx.add_assistant_message(ai_msg)

    # Get conversation history (this triggers VM filtering)
    history = ctx.conversation_history

    # Check VM stats
    vm = ctx.session.vm
    metrics = vm.metrics
    ws_stats = ws.get_stats()

    print()
    print(f"  Messages added:          {len(exchanges) * 2}")
    print(f"  History returned:        {len(history)}")
    print(f"  L0 pages:                {ws_stats.l0_pages}")
    print(f"  L0 tokens:               {ws_stats.tokens_used}")
    print(f"  L0 utilization:          {ws_stats.utilization:.0%}")
    print(f"  Evictions:               {metrics.evictions_total}")
    print(f"  Pages in table:          {vm.page_table.get_stats().total_pages}")

    # With 500 token budget, 12 messages should trigger evictions
    assert metrics.evictions_total > 0, (
        "Must have evictions with 12 messages in 500-token budget!"
    )
    assert ws_stats.tokens_used <= 500, (
        f"L0 tokens ({ws_stats.tokens_used}) must not exceed budget (500)!"
    )

    # History should be filtered (not all 12 messages)
    # System prompt + filtered events
    assert len(history) < 14, (
        f"History ({len(history)}) should be filtered, not all 14 (1 sys + 12 msgs + 1 notice)"
    )

    # First message should be system/developer message with VM:CONTEXT
    first = history[0]
    assert "<VM:CONTEXT>" in (first.content or ""), (
        "First message must be VM developer_message"
    )

    print()
    print("  System message preview:")
    content = first.content or ""
    for line in content.split("\n")[:8]:
        if line.strip():
            print(f"    {line[:80]}")
    if len(content.split("\n")) > 8:
        print(f"    ... ({len(content):,} chars total)")

    print()
    print("  Full integration:        PASS")
    print()


async def main() -> None:
    """Run all VM memory management demos."""
    print()
    print("  MCP-CLI: AI Virtual Memory — End-to-End Demo")
    print("  Proves budget enforcement, eviction, and context filtering")
    print()

    demo_budget_enforcement()
    demo_page_lifecycle()
    await demo_eviction_under_pressure()
    demo_turn_tracking()
    demo_context_building()
    await demo_event_filtering()
    await demo_full_integration()

    print("=" * 70)
    print("  All VM memory management demos PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
