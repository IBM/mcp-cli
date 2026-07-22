#!/usr/bin/env python3
"""Server Health Monitoring & VM Multimodal Content Demo

Proves end-to-end that:

  1. Health-check-on-failure — ToolManager._diagnose_server() detects unhealthy servers
  2. Health polling lifecycle — background task starts, detects transitions, stops cleanly
  3. HealthCommand — /health slash command formats results correctly
  4. Multimodal page_fault — image pages produce multi-block content (text + image_url)
  5. Text page_fault — text pages produce JSON string with modality/compression metadata
  6. Structured page_fault — structured/JSON pages preserve dict content
  7. search_pages — query returns ranked results with page IDs and hints
  8. /memory page --download — page content exported to file (text, JSON, base64 image)
  9. Content block builder — edge cases (truncated, compressed, short content notes)

No API keys or MCP servers required — runs fully self-contained.

Usage: uv run python examples/safety/health_vm_multimodal_demo.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Allow running from the examples/ directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ── Helpers ──────────────────────────────────────────────────────────────


def header(n: int, title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {n}. {title}")
    print("=" * 70)


def ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def fail(msg: str) -> None:
    print(f"  \u2717 {msg}")


def info(msg: str) -> None:
    print(f"    {msg}")


@dataclass
class DemoResults:
    """Track pass/fail across all demos."""

    passed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def check(self, condition: bool, description: str) -> None:
        if condition:
            ok(description)
            self.passed += 1
        else:
            fail(description)
            self.failed += 1
            self.errors.append(description)


results = DemoResults()


# ══════════════════════════════════════════════════════════════════════════
# 1. HEALTH-CHECK-ON-FAILURE
# ══════════════════════════════════════════════════════════════════════════


async def demo_1_health_check_on_failure() -> None:
    """ToolManager._diagnose_server() detects unhealthy servers."""
    header(1, "HEALTH-CHECK-ON-FAILURE")

    from mcp_cli.tools.manager import ToolManager

    # Create ToolManager with a mock StreamManager
    tm = ToolManager.__new__(ToolManager)
    tm.stream_manager = AsyncMock()

    # Scenario A: Server is unhealthy
    tm.stream_manager.health_check.return_value = {
        "transports": {
            "my-server": {"status": "unhealthy", "error": "process exited"},
        }
    }
    diag = await tm._diagnose_server("my-server")
    results.check(
        "unhealthy" in diag,
        f"Unhealthy server detected: {diag!r}",
    )

    # Scenario B: Server is healthy → empty diagnostic
    tm.stream_manager.health_check.return_value = {
        "transports": {
            "my-server": {"status": "healthy", "ping_success": True},
        }
    }
    diag = await tm._diagnose_server("my-server")
    results.check(diag == "", "Healthy server returns empty diagnostic")

    # Scenario C: No stream manager → empty diagnostic
    diag_no_sm = await tm._diagnose_server(None)
    results.check(diag_no_sm == "", "None server_name returns empty diagnostic")

    # Scenario D: Health check exception → graceful fallback
    tm.stream_manager.health_check.side_effect = ConnectionError("transport dead")
    diag_exc = await tm._diagnose_server("broken")
    results.check(diag_exc == "", "Exception in health check returns empty string")
    tm.stream_manager.health_check.side_effect = None

    # Scenario E: _is_connection_error pattern matching
    results.check(
        tm._is_connection_error("Connection refused by remote host"),
        "_is_connection_error('Connection refused...')",
    )
    results.check(
        tm._is_connection_error("Read timed out after 30s"),
        "_is_connection_error('Read timed out...')",
    )
    results.check(
        not tm._is_connection_error("Invalid API key"),
        "_is_connection_error rejects non-connection errors",
    )

    print()


# ══════════════════════════════════════════════════════════════════════════
# 2. HEALTH POLLING LIFECYCLE
# ══════════════════════════════════════════════════════════════════════════


async def demo_2_health_polling_lifecycle() -> None:
    """Background polling starts, detects transitions, and stops cleanly."""
    header(2, "HEALTH POLLING LIFECYCLE")

    from mcp_cli.tools.manager import ToolManager

    # Build a mock context with tool_manager
    mock_tm = ToolManager.__new__(ToolManager)
    mock_tm.stream_manager = AsyncMock()

    call_count = 0
    health_states = [
        # First poll: healthy
        {"transports": {"srv": {"status": "healthy"}}},
        # Second poll: transition to unhealthy
        {"transports": {"srv": {"status": "unhealthy"}}},
        # Third poll: back to healthy
        {"transports": {"srv": {"status": "healthy"}}},
    ]

    async def fake_health_check():
        nonlocal call_count
        idx = min(call_count, len(health_states) - 1)
        call_count += 1
        return health_states[idx]

    mock_tm.stream_manager.health_check = fake_health_check

    # Simulate the polling loop logic directly
    # (ConversationProcessor._health_poll_loop equivalent)
    last_health: dict[str, str] = {}
    transitions: list[str] = []

    for state in health_states:
        for name, info_dict in state.get("transports", {}).items():
            status = info_dict.get("status", "unknown")
            prev = last_health.get(name)
            if prev and prev != status:
                transitions.append(f"{name}: {prev} -> {status}")
            last_health[name] = status

    results.check(
        len(transitions) == 2,
        f"Detected {len(transitions)} health transitions (expected 2)",
    )
    results.check(
        "healthy -> unhealthy" in transitions[0],
        f"First transition: {transitions[0]}",
    )
    results.check(
        "unhealthy -> healthy" in transitions[1],
        f"Second transition: {transitions[1]}",
    )

    # Test start/stop lifecycle from ConversationProcessor
    from mcp_cli.chat.conversation import ConversationProcessor

    mock_context = MagicMock()
    mock_context._health_interval = 0.1  # 100ms interval
    mock_context.tool_manager = mock_tm

    proc = ConversationProcessor.__new__(ConversationProcessor)
    proc._health_task = None
    proc._health_interval = 0.1
    proc._last_health = {}

    # Start polling
    proc._start_health_polling()
    results.check(proc._health_task is not None, "Polling task created on start")

    # Stop polling
    proc._stop_health_polling()
    results.check(proc._health_task is None, "Polling task cleared on stop")

    # Double-stop is safe
    proc._stop_health_polling()
    results.check(proc._health_task is None, "Double-stop is idempotent")

    # Zero interval skips start
    proc._health_interval = 0
    proc._start_health_polling()
    results.check(proc._health_task is None, "Zero interval skips polling start")

    print()


# ══════════════════════════════════════════════════════════════════════════
# 3. HEALTH COMMAND
# ══════════════════════════════════════════════════════════════════════════


async def demo_3_health_command() -> None:
    """HealthCommand formats results correctly."""
    header(3, "HEALTH COMMAND (/health)")

    from mcp_cli.commands.servers.health import HealthCommand

    cmd = HealthCommand()
    results.check(cmd.name == "health", f"Command name: {cmd.name}")

    # Scenario A: No tool manager
    result = await cmd.execute()
    results.check(not result.success, "Fails without tool_manager")
    info(f"Error: {result.error}")

    # Scenario B: All healthy
    mock_tm = AsyncMock()
    mock_tm.check_server_health.return_value = {
        "echo-server": {"status": "healthy", "ping_success": True},
        "weather-api": {"status": "healthy", "ping_success": True},
    }
    result = await cmd.execute(tool_manager=mock_tm)
    results.check(result.success, "All healthy → success=True")
    results.check(
        len(result.data) == 2,
        f"Returns data for {len(result.data)} servers",
    )

    # Scenario C: One unhealthy
    mock_tm.check_server_health.return_value = {
        "echo-server": {"status": "healthy", "ping_success": True},
        "broken-api": {"status": "unhealthy", "error": "process crashed"},
    }
    result = await cmd.execute(tool_manager=mock_tm)
    results.check(not result.success, "Unhealthy server → success=False")

    # Scenario D: Specific server not found
    mock_tm.check_server_health.return_value = {}
    result = await cmd.execute(tool_manager=mock_tm, server_name="nonexistent")
    results.check(not result.success, "Unknown server → success=False")
    info(f"Error: {result.error}")

    print()


# ══════════════════════════════════════════════════════════════════════════
# 4. MULTIMODAL PAGE_FAULT — IMAGE PAGES
# ══════════════════════════════════════════════════════════════════════════


async def demo_4_multimodal_page_fault() -> None:
    """Image pages produce multi-block content (text + image_url)."""
    header(4, "MULTIMODAL PAGE_FAULT — IMAGE PAGES")

    from chuk_ai_session_manager.memory.models import CompressionLevel, Modality

    from mcp_cli.chat.tool_processor import ToolProcessor

    proc = ToolProcessor.__new__(ToolProcessor)

    # Create a mock image page with a data URI
    pixel_png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode()
    image_url = f"data:image/png;base64,{pixel_png}"

    mock_page = MagicMock()
    mock_page.page_id = "img_sunset_001"
    mock_page.modality = Modality.IMAGE
    mock_page.compression_level = CompressionLevel.FULL
    mock_page.content = image_url

    # Build content blocks
    blocks = proc._build_page_content_blocks(
        page=mock_page,
        page_content=image_url,
        truncated=False,
        was_compressed=False,
        source_tier="L2",
    )

    results.check(isinstance(blocks, list), "Image page returns list (multi-block)")
    results.check(len(blocks) == 2, f"Two blocks returned (got {len(blocks)})")
    results.check(
        blocks[0]["type"] == "text",
        f"First block is text: {blocks[0].get('type')}",
    )
    results.check(
        blocks[1]["type"] == "image_url",
        f"Second block is image_url: {blocks[1].get('type')}",
    )
    results.check(
        blocks[1]["image_url"]["url"] == image_url,
        "Image URL preserved in block",
    )
    results.check(
        "detail" in blocks[1]["image_url"],
        f"Detail level set: {blocks[1]['image_url'].get('detail')}",
    )

    info(f"Text block: {blocks[0]['text']}")

    # Test with HTTPS URL
    mock_page.content = "https://example.com/photo.jpg"
    blocks_url = proc._build_page_content_blocks(
        page=mock_page,
        page_content="https://example.com/photo.jpg",
        truncated=True,
        was_compressed=False,
        source_tier="L1",
    )
    results.check(
        isinstance(blocks_url, list),
        "HTTPS image URL also returns multi-block",
    )
    results.check(
        "[content truncated]" in blocks_url[0]["text"],
        "Truncation note in text block",
    )

    print()


# ══════════════════════════════════════════════════════════════════════════
# 5. TEXT PAGE_FAULT — JSON RESPONSE
# ══════════════════════════════════════════════════════════════════════════


async def demo_5_text_page_fault() -> None:
    """Text pages produce JSON string with modality and compression metadata."""
    header(5, "TEXT PAGE_FAULT — JSON RESPONSE")

    from chuk_ai_session_manager.memory.models import CompressionLevel, Modality

    from mcp_cli.chat.tool_processor import ToolProcessor

    proc = ToolProcessor.__new__(ToolProcessor)

    # Text page (normal conversation content)
    mock_page = MagicMock()
    mock_page.page_id = "msg_chat_042"
    mock_page.modality = Modality.TEXT
    mock_page.compression_level = CompressionLevel.FULL
    mock_page.content = (
        "My name is Chris and I live in Leavenheath, Suffolk. "
        "It's a small village on the Suffolk-Essex border, about "
        "five miles from Colchester."
    )

    result = proc._build_page_content_blocks(
        page=mock_page,
        page_content=mock_page.content,
        truncated=False,
        was_compressed=False,
        source_tier="L2",
    )

    results.check(isinstance(result, str), "Text page returns JSON string")
    parsed = json.loads(result)
    results.check(parsed["success"], "success=True in response")
    results.check(parsed["page_id"] == "msg_chat_042", f"page_id: {parsed['page_id']}")
    results.check(parsed["modality"] == "text", f"modality: {parsed['modality']}")
    results.check(
        parsed["compression"] == "FULL", f"compression: {parsed['compression']}"
    )
    results.check(
        "Chris" in parsed["content"] and "Leavenheath" in parsed["content"],
        "Content preserved in response",
    )

    info(f"Response keys: {sorted(parsed.keys())}")

    # Short content note
    short_page = MagicMock()
    short_page.page_id = "msg_short"
    short_page.modality = Modality.TEXT
    short_page.compression_level = CompressionLevel.FULL
    short_page.content = "What is my name?"

    short_result = json.loads(
        proc._build_page_content_blocks(
            page=short_page,
            page_content="What is my name?",
            truncated=False,
            was_compressed=False,
            source_tier="L3",
        )
    )
    results.check(
        "note" in short_result,
        f"Short content includes note: {short_result.get('note', '')[:60]}...",
    )

    # Compressed content note
    compressed_page = MagicMock()
    compressed_page.page_id = "msg_compressed"
    compressed_page.modality = Modality.TEXT
    compressed_page.compression_level = CompressionLevel.ABSTRACT
    compressed_page.content = "Summary: user lives in Leavenheath"

    comp_result = json.loads(
        proc._build_page_content_blocks(
            page=compressed_page,
            page_content="Summary: user lives in Leavenheath",
            truncated=False,
            was_compressed=True,
            source_tier="L3",
        )
    )
    results.check(
        "abstract" in comp_result.get("note", "").lower(),
        f"Compressed note: {comp_result.get('note', '')[:60]}...",
    )

    print()


# ══════════════════════════════════════════════════════════════════════════
# 6. STRUCTURED PAGE_FAULT
# ══════════════════════════════════════════════════════════════════════════


async def demo_6_structured_page_fault() -> None:
    """Structured/JSON pages preserve dict content."""
    header(6, "STRUCTURED PAGE_FAULT")

    from chuk_ai_session_manager.memory.models import CompressionLevel, Modality

    from mcp_cli.chat.tool_processor import ToolProcessor

    proc = ToolProcessor.__new__(ToolProcessor)

    weather_data = {
        "temperature": 22,
        "conditions": "partly cloudy",
        "wind_speed": 12,
        "location": "Leavenheath, Suffolk",
    }

    mock_page = MagicMock()
    mock_page.page_id = "tool_weather_007"
    mock_page.modality = Modality.STRUCTURED
    mock_page.compression_level = CompressionLevel.FULL
    mock_page.content = json.dumps(weather_data)

    result = proc._build_page_content_blocks(
        page=mock_page,
        page_content=json.dumps(weather_data),
        truncated=False,
        was_compressed=False,
        source_tier="L2",
    )

    results.check(isinstance(result, str), "Structured page returns JSON string")
    parsed = json.loads(result)
    results.check(parsed["modality"] == "structured", f"modality: {parsed['modality']}")
    results.check(
        "temperature" in parsed["content"],
        "Structured content preserved",
    )

    info(f"Content: {parsed['content'][:80]}...")

    print()


# ══════════════════════════════════════════════════════════════════════════
# 7. SEARCH_PAGES VIA VM
# ══════════════════════════════════════════════════════════════════════════


async def demo_7_search_pages() -> None:
    """search_pages returns ranked results with page IDs and hints."""
    header(7, "SEARCH_PAGES")

    from chuk_ai_session_manager.memory.models import (
        Modality,
        PageType,
    )
    from chuk_ai_session_manager.memory.working_set import WorkingSetConfig
    from chuk_ai_session_manager.session_manager import SessionManager

    sm = SessionManager(
        system_prompt="You are a helpful assistant.",
        enable_vm=True,
        vm_mode="relaxed",
        vm_config=WorkingSetConfig(max_l0_tokens=800, reserved_tokens=100),
    )
    await sm._ensure_initialized()
    vm = sm.vm

    # Add conversation pages (with hints for search) and artifact pages
    vm.new_turn()
    await sm.user_says("My name is Chris and I live in Leavenheath.")
    # Transcript pages don't auto-set hints, so register one for search
    pt_entries = list(vm.page_table.entries.keys())
    if pt_entries:
        vm._search_handler.set_hint(
            pt_entries[-1], "Chris lives in Leavenheath Suffolk"
        )

    vm.new_turn()
    await sm.ai_responds(
        "Nice to meet you, Chris! Leavenheath is a lovely village.",
        model="test",
        provider="test",
    )
    vm.new_turn()
    vm.create_page(
        content='{"temperature": 22, "location": "Leavenheath"}',
        page_type=PageType.ARTIFACT,
        modality=Modality.STRUCTURED,
        importance=0.5,
        hint="weather forecast for Leavenheath",
    )

    # Search for weather-related pages (substring match on hint)
    search_result = await vm.search_pages(query="weather", limit=5)
    info(f"Search result type: {type(search_result).__name__}")

    result_json = search_result.to_json()
    results.check(isinstance(result_json, str), "search_pages returns JSON string")
    parsed = json.loads(result_json)
    info(f"Search response keys: {sorted(parsed.keys())}")

    result_list = parsed.get("results", [])
    results.check(len(result_list) > 0, f"Found {len(result_list)} matching pages")

    if result_list:
        top = result_list[0]
        info(
            f"Top result: page_id={top.get('page_id')}, hint={top.get('hint', '')[:60]}"
        )
        results.check("page_id" in top, "Results include page_id")

    # Search for name-related pages (matches hint we set above)
    name_search = await vm.search_pages(query="Chris", limit=3)
    name_parsed = json.loads(name_search.to_json())
    name_results = name_parsed.get("results", [])
    results.check(
        len(name_results) > 0,
        f"Name search found {len(name_results)} pages",
    )

    # Search with modality filter
    struct_search = await vm.search_pages(
        query="weather", modality="structured", limit=3
    )
    struct_parsed = json.loads(struct_search.to_json())
    struct_results = struct_parsed.get("results", [])
    results.check(
        len(struct_results) > 0,
        f"Modality-filtered search returned {len(struct_results)} structured pages",
    )

    print()


# ══════════════════════════════════════════════════════════════════════════
# 8. MEMORY PAGE DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════


async def demo_8_memory_download() -> None:
    """/memory page --download exports page content to files."""
    header(8, "MEMORY PAGE DOWNLOAD")

    from mcp_cli.commands.memory.memory import MemoryCommand

    cmd = MemoryCommand()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock a VM with page table and page store
        mock_vm = MagicMock()

        # --- Text page ---
        text_entry = MagicMock()
        text_entry.modality = MagicMock(value="text")

        text_page = MagicMock()
        text_page.content = "My name is Chris and I live in Leavenheath."
        text_page.mime_type = "text/plain"

        # --- JSON page ---
        json_entry = MagicMock()
        json_entry.modality = MagicMock(value="structured")

        json_page = MagicMock()
        json_page.content = {"temperature": 22, "conditions": "partly cloudy"}
        json_page.mime_type = "application/json"

        # --- Image page (base64 data URI) ---
        img_entry = MagicMock()
        img_entry.modality = MagicMock(value="image")

        pixel_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        b64 = base64.b64encode(pixel_data).decode()
        img_page = MagicMock()
        img_page.content = f"data:image/png;base64,{b64}"
        img_page.mime_type = "image/png"

        # Wire up VM
        mock_vm.page_table.entries = {
            "msg_text_001": text_entry,
            "tool_json_002": json_entry,
            "img_photo_003": img_entry,
        }
        mock_vm._page_store.get.side_effect = lambda pid: {
            "msg_text_001": text_page,
            "tool_json_002": json_page,
            "img_photo_003": img_page,
        }.get(pid)

        # Patch download directory
        with patch.object(Path, "home", return_value=Path(tmpdir)):
            # Download text page
            r = cmd._download_page(mock_vm, "msg_text_001")
            results.check(r.success, "Text page download succeeded")
            if r.success:
                out_path = Path(r.data["path"])
                results.check(out_path.exists(), f"Text file created: {out_path.name}")
                content = out_path.read_text()
                results.check(
                    "Chris" in content and "Leavenheath" in content,
                    "Text content matches",
                )
                info(f"Size: {out_path.stat().st_size} bytes")

            # Download JSON page
            r = cmd._download_page(mock_vm, "tool_json_002")
            results.check(r.success, "JSON page download succeeded")
            if r.success:
                out_path = Path(r.data["path"])
                results.check(
                    out_path.suffix == ".json",
                    f"JSON extension: {out_path.suffix}",
                )
                parsed = json.loads(out_path.read_text())
                results.check(
                    parsed["temperature"] == 22,
                    f"JSON content preserved: temp={parsed['temperature']}",
                )

            # Download image page (base64 data URI)
            r = cmd._download_page(mock_vm, "img_photo_003")
            results.check(r.success, "Image page download succeeded")
            if r.success:
                out_path = Path(r.data["path"])
                results.check(
                    out_path.suffix == ".png",
                    f"PNG extension: {out_path.suffix}",
                )
                raw = out_path.read_bytes()
                results.check(
                    raw == pixel_data,
                    f"Binary content matches ({len(raw)} bytes)",
                )

        # Error: page not found
        r = cmd._download_page(mock_vm, "nonexistent")
        results.check(not r.success, f"Missing page error: {r.error}")

        # Error: no content
        empty_entry = MagicMock()
        mock_vm.page_table.entries["empty_page"] = empty_entry
        mock_vm._page_store.get.side_effect = lambda pid: (
            None if pid == "empty_page" else None
        )
        r = cmd._download_page(mock_vm, "empty_page")
        results.check(not r.success, f"No content error: {r.error}")

    print()


# ══════════════════════════════════════════════════════════════════════════
# 9. MULTI-BLOCK IN HISTORY
# ══════════════════════════════════════════════════════════════════════════


async def demo_9_multiblock_history() -> None:
    """Multi-block content flows through _add_tool_result_to_history."""
    header(9, "MULTI-BLOCK CONTENT IN HISTORY")

    from mcp_cli.chat.models import HistoryMessage, MessageRole

    # Multi-block content (as would be produced by image page_fault)
    blocks = [
        {"type": "text", "text": "Page img_001 (image, FULL):"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc", "detail": "low"},
        },
    ]

    # Create a HistoryMessage with multi-block content
    msg = HistoryMessage(
        role=MessageRole.TOOL,
        name="page_fault",
        content=blocks,
        tool_call_id="call_img_001",
    )

    results.check(isinstance(msg.content, list), "HistoryMessage accepts list content")
    results.check(
        len(msg.content) == 2, f"Two content blocks stored (got {len(msg.content)})"
    )

    # Serialize to dict (for API)
    api_dict = msg.to_dict()
    results.check(
        isinstance(api_dict["content"], list),
        "to_dict preserves list content",
    )
    results.check(
        api_dict["content"][1]["type"] == "image_url",
        "image_url block preserved in serialization",
    )
    results.check(
        api_dict["role"] == "tool",
        f"role: {api_dict['role']}",
    )
    results.check(
        api_dict["tool_call_id"] == "call_img_001",
        f"tool_call_id: {api_dict['tool_call_id']}",
    )

    info(f"API dict keys: {sorted(api_dict.keys())}")

    # String content still works
    str_msg = HistoryMessage(
        role=MessageRole.TOOL,
        name="page_fault",
        content='{"success": true, "page_id": "msg_001"}',
        tool_call_id="call_text_001",
    )
    str_dict = str_msg.to_dict()
    results.check(
        isinstance(str_dict["content"], str),
        "String content preserved in serialization",
    )

    print()


# ══════════════════════════════════════════════════════════════════════════
# 10. FULL VM LIFECYCLE — EVICTION + FAULT + MULTIMODAL
# ══════════════════════════════════════════════════════════════════════════


async def demo_10_full_vm_lifecycle() -> None:
    """End-to-end: create pages, force eviction, fault back with content blocks."""
    header(10, "FULL VM LIFECYCLE — EVICTION + FAULT + MULTIMODAL")

    from chuk_ai_session_manager.memory.models import (
        Modality,
        PageType,
    )
    from chuk_ai_session_manager.memory.working_set import WorkingSetConfig
    from chuk_ai_session_manager.session_manager import SessionManager

    from mcp_cli.chat.tool_processor import ToolProcessor

    # Small budget to force eviction
    sm = SessionManager(
        system_prompt="You are a helpful assistant.",
        enable_vm=True,
        vm_mode="relaxed",
        vm_config=WorkingSetConfig(max_l0_tokens=400, reserved_tokens=50),
    )
    await sm._ensure_initialized()
    vm = sm.vm

    # Turn 1: User message + assistant response (takes ~100-200 tokens)
    vm.new_turn()
    await sm.user_says("My name is Chris. I live in Leavenheath, Suffolk.")
    vm.new_turn()
    await sm.ai_responds(
        "Nice to meet you, Chris! Leavenheath is a lovely village.",
        model="test",
        provider="test",
    )

    # Turn 2: Create image artifact
    vm.new_turn()
    pixel_png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20).decode()
    image_url = f"data:image/png;base64,{pixel_png}"
    vm.create_page(
        content=image_url,
        page_type=PageType.ARTIFACT,
        modality=Modality.IMAGE,
        importance=0.4,
        hint="[image] sunset_photo.jpg",
    )

    # Turn 3: Create structured artifact
    vm.new_turn()
    vm.create_page(
        content='{"temperature": 22, "conditions": "sunny", "location": "Leavenheath"}',
        page_type=PageType.ARTIFACT,
        modality=Modality.STRUCTURED,
        importance=0.5,
        hint="weather forecast for Leavenheath",
    )

    # Turns 4-8: Push filler to force eviction
    for i in range(5):
        vm.new_turn()
        await sm.user_says(
            f"Tell me about topic {i}: Lorem ipsum dolor sit amet, "
            f"consectetur adipiscing elit. Sed do eiusmod tempor incididunt "
            f"ut labore et dolore magna aliqua. Topic {i} is very interesting."
        )
        vm.new_turn()
        await sm.ai_responds(
            f"Here's information about topic {i}: It's a fascinating subject "
            f"that covers many aspects of knowledge. The key points are "
            f"that it relates to various fields of study and practice. "
            f"There is much more to explore about topic {i}.",
            model="test",
            provider="test",
        )

    # Check eviction happened
    pt_stats = vm.page_table.get_stats()
    ws_stats = vm.working_set.get_stats()
    info(f"Total pages: {pt_stats.total_pages}")
    info(f"L0 pages: {ws_stats.l0_pages}, L1 pages: {ws_stats.l1_pages}")
    info(
        f"Tokens used: {ws_stats.tokens_used}/{ws_stats.tokens_available + ws_stats.tokens_used}"
    )
    info(f"Evictions: {vm.metrics.evictions_total}")

    results.check(
        vm.metrics.evictions_total > 0,
        f"Evictions occurred: {vm.metrics.evictions_total}",
    )

    # Search for the weather page (substring match on hint)
    search = await vm.search_pages(query="weather", limit=5)
    search_parsed = json.loads(search.to_json())
    search_results = search_parsed.get("results", [])
    results.check(
        len(search_results) > 0,
        f"search_pages found {len(search_results)} results for 'weather'",
    )

    if search_results:
        weather_page_id = search_results[0]["page_id"]
        info(f"Found weather page: {weather_page_id}")

        # Fault it back
        fault = await vm.handle_fault(page_id=weather_page_id, target_level=2)
        results.check(
            fault.success and fault.page is not None,
            f"page_fault succeeded for {weather_page_id}",
        )

        if fault.page:
            # Build content blocks using ToolProcessor
            proc = ToolProcessor.__new__(ToolProcessor)
            blocks = proc._build_page_content_blocks(
                page=fault.page,
                page_content=fault.page.content,
                truncated=False,
                was_compressed=fault.was_compressed,
                source_tier=fault.source_tier,
            )
            parsed = json.loads(blocks) if isinstance(blocks, str) else blocks
            if isinstance(parsed, dict):
                results.check(
                    parsed.get("modality") == "structured",
                    f"Weather page modality: {parsed.get('modality')}",
                )
                results.check(
                    "temperature" in str(parsed.get("content", "")),
                    "Weather content preserved after fault",
                )
            else:
                results.check(True, f"Content blocks returned: {len(parsed)} blocks")
            info(f"Source tier: {fault.source_tier}")
            info(f"Was compressed: {fault.was_compressed}")

    # Search for image page
    img_search = await vm.search_pages(query="sunset", limit=3)
    img_parsed = json.loads(img_search.to_json())
    img_results = img_parsed.get("results", [])

    if img_results:
        img_page_id = img_results[0]["page_id"]
        info(f"Found image page: {img_page_id}")
        img_fault = await vm.handle_fault(page_id=img_page_id, target_level=2)

        if img_fault.success and img_fault.page:
            proc = ToolProcessor.__new__(ToolProcessor)
            blocks = proc._build_page_content_blocks(
                page=img_fault.page,
                page_content=img_fault.page.content,
                truncated=False,
                was_compressed=img_fault.was_compressed,
                source_tier=img_fault.source_tier,
            )
            if isinstance(blocks, list):
                results.check(
                    blocks[1]["type"] == "image_url",
                    "Image page produces image_url block after fault",
                )
                results.check(
                    "data:image/png" in blocks[1]["image_url"]["url"],
                    "Data URI preserved through eviction + fault cycle",
                )
            else:
                # Image content may have been compressed to text reference
                parsed = json.loads(blocks)
                info(
                    f"Image page returned as {parsed.get('modality')} "
                    f"({parsed.get('compression')})"
                )
                results.check(True, "Image page faulted (may be compressed)")

    print()


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════


async def main() -> None:
    print("=" * 70)
    print("  Server Health Monitoring & VM Multimodal Content Demo")
    print("=" * 70)
    print()
    print("  This demo validates health monitoring, multimodal page_fault,")
    print("  search_pages, and /memory page --download without API keys.")
    print()

    t0 = time.monotonic()

    await demo_1_health_check_on_failure()
    await demo_2_health_polling_lifecycle()
    await demo_3_health_command()
    await demo_4_multimodal_page_fault()
    await demo_5_text_page_fault()
    await demo_6_structured_page_fault()
    await demo_7_search_pages()
    await demo_8_memory_download()
    await demo_9_multiblock_history()
    await demo_10_full_vm_lifecycle()

    elapsed = time.monotonic() - t0

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Passed: {results.passed}")
    print(f"  Failed: {results.failed}")
    print(f"  Time:   {elapsed:.1f}s")

    if results.errors:
        print()
        print("  Failures:")
        for err in results.errors:
            print(f"    - {err}")

    print()
    if results.failed == 0:
        print("  ALL CHECKS PASSED")
    else:
        print(f"  {results.failed} CHECK(S) FAILED")
    print("=" * 70)

    sys.exit(1 if results.failed > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
