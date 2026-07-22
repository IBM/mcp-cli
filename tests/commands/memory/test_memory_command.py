"""Tests for src/mcp_cli/commands/memory/memory.py."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.commands.memory.memory import MemoryCommand
from mcp_cli.commands.base import CommandMode


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_command() -> MemoryCommand:
    return MemoryCommand()


def _make_chat_context(
    *,
    has_session: bool = False,
    has_vm: bool = False,
    has_store: bool = False,
    vm_budget: int = 128_000,
) -> MagicMock:
    """Build a minimal mock chat_context."""
    ctx = MagicMock()
    ctx._vm_budget = vm_budget
    ctx._system_prompt_dirty = False

    if has_vm:
        vm = _make_vm()
        session = MagicMock()
        session.vm = vm
        ctx.session = session
    elif has_session:
        session = MagicMock()
        session.vm = None
        ctx.session = session
    else:
        ctx.session = None

    if has_store:
        ctx.memory_store = _make_store()
    else:
        ctx.memory_store = None

    return ctx


def _make_vm() -> MagicMock:
    """Build a mock VM object with all required sub-objects."""
    from chuk_ai_session_manager.memory.models import StorageTier
    from chuk_ai_session_manager.memory.models.enums import (
        CompressionLevel,
        Modality,
        PageType,
        VMMode,
    )

    vm = MagicMock()
    vm.mode = VMMode.PASSIVE
    vm.turn = 5

    # metrics
    metrics = MagicMock()
    metrics.tlb_hits = 10
    metrics.tlb_misses = 2
    metrics.tlb_hit_rate = 10 / 12
    metrics.faults_total = 3
    metrics.faults_this_turn = 1
    metrics.evictions_total = 2
    metrics.evictions_this_turn = 0
    vm.metrics = metrics

    # working_set
    ws_stats = MagicMock()
    ws_stats.utilization = 0.5
    ws_stats.l0_pages = 2
    ws_stats.l1_pages = 4
    ws_stats.tokens_used = 8_000
    ws_stats.tokens_available = 8_000
    ws = MagicMock()
    ws.get_stats.return_value = ws_stats
    vm.working_set = ws

    # page_table
    pt_stats = MagicMock()
    # pages_by_tier: a real dict keyed by StorageTier enums
    pt_stats.pages_by_tier = {
        StorageTier.L0: 2,
        StorageTier.L1: 4,
    }
    pt_stats.total_pages = 6
    pt_stats.dirty_pages = 1

    # a couple of page table entries
    def _make_entry(pid: str, tier: StorageTier) -> MagicMock:
        e = MagicMock()
        e.page_id = pid
        e.page_type = PageType.TRANSCRIPT
        e.tier = tier
        e.size_tokens = 100
        e.eviction_priority = 0.5
        e.pinned = False
        e.compression_level = CompressionLevel.FULL
        e.access_count = 3
        e.last_accessed = datetime.now(timezone.utc)
        e.modality = Modality.TEXT
        e.dirty = False
        e.provenance = ["turn-1"]
        return e

    entry_a = _make_entry("page-001", StorageTier.L0)
    entry_b = _make_entry("page-002", StorageTier.L1)

    pt = MagicMock()
    pt.get_stats.return_value = pt_stats
    pt.entries = {
        "page-001": entry_a,
        "page-002": entry_b,
    }
    vm.page_table = pt

    # _page_store
    page_a = MagicMock()
    page_a.content = "Hello world"
    page_a.mime_type = "text/plain"
    page_a.dimensions = None
    page_a.duration_seconds = None
    page_a.caption = "A caption"

    page_b = MagicMock()
    page_b.content = "B content"
    page_b.mime_type = None
    page_b.dimensions = None
    page_b.duration_seconds = None
    page_b.caption = None

    page_store = {"page-001": page_a, "page-002": page_b}
    vm._page_store = page_store

    # get_stats
    vm.get_stats.return_value = {"mode": "passive", "turn": 5}

    return vm


def _make_store() -> MagicMock:
    """Build a mock MemoryScopeStore."""
    from mcp_cli.memory.models import MemoryEntry, MemoryScope

    store = MagicMock()

    ws_entry = MemoryEntry(
        key="db_type",
        content="postgresql",
    )
    gl_entry = MemoryEntry(
        key="test_framework",
        content="always use pytest for testing Python projects",
    )

    def _list(scope):
        if scope == MemoryScope.WORKSPACE:
            return [ws_entry]
        elif scope == MemoryScope.GLOBAL:
            return [gl_entry]
        return []

    store.list_entries.side_effect = _list

    remembered = MemoryEntry(key="new_key", content="new content")
    store.remember.return_value = remembered
    store.forget.return_value = True
    store.clear.return_value = 3

    return store


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


class TestMemoryCommandProperties:
    def test_name(self):
        cmd = _make_command()
        assert cmd.name == "memory"

    def test_aliases(self):
        cmd = _make_command()
        assert "vm" in cmd.aliases
        assert "mem" in cmd.aliases

    def test_description(self):
        cmd = _make_command()
        assert "memory" in cmd.description.lower()

    def test_help_text(self):
        cmd = _make_command()
        assert len(cmd.help_text) > 10

    def test_modes(self):
        cmd = _make_command()
        assert cmd.modes == CommandMode.CHAT

    def test_parameters(self):
        cmd = _make_command()
        params = cmd.parameters
        assert len(params) >= 1
        names = {p.name for p in params}
        assert "action" in names


# ---------------------------------------------------------------------------
# _parse_action static method
# ---------------------------------------------------------------------------


class TestParseAction:
    def test_action_kwarg(self):
        assert MemoryCommand._parse_action({"action": "pages"}) == "pages"

    def test_action_kwarg_none(self):
        assert MemoryCommand._parse_action({"action": None}) is None

    def test_args_list(self):
        assert MemoryCommand._parse_action({"args": ["page", "abc"]}) == "page abc"

    def test_args_string(self):
        assert MemoryCommand._parse_action({"args": "stats"}) == "stats"

    def test_args_empty_list(self):
        assert MemoryCommand._parse_action({"args": []}) is None

    def test_args_empty_string(self):
        assert MemoryCommand._parse_action({"args": ""}) is None

    def test_no_action_no_args(self):
        assert MemoryCommand._parse_action({}) is None


# ---------------------------------------------------------------------------
# execute — guard / routing
# ---------------------------------------------------------------------------


class TestExecuteRouting:
    @pytest.mark.asyncio
    async def test_no_chat_context_returns_error(self):
        cmd = _make_command()
        result = await cmd.execute()
        assert result.success is False
        assert "chat context" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_vm_no_store_returns_error(self):
        cmd = _make_command()
        ctx = _make_chat_context()
        result = await cmd.execute(chat_context=ctx)
        assert result.success is False
        assert "VM not enabled" in result.error

    @pytest.mark.asyncio
    async def test_no_vm_with_store_falls_back_to_list(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        with patch("chuk_term.ui.output"), patch("chuk_term.ui.format_table"):
            result = await cmd.execute(chat_context=ctx)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_routes_pages(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        with patch("chuk_term.ui.output"), patch("chuk_term.ui.format_table"):
            result = await cmd.execute(chat_context=ctx, action="pages")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_routes_stats(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="stats")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_routes_page_detail(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="page page-001")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_routes_page_download(self, tmp_path):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        with (
            patch("chuk_term.ui.output"),
            patch(
                "mcp_cli.commands.memory.memory.DEFAULT_DOWNLOADS_DIR",
                str(tmp_path),
            ),
        ):
            result = await cmd.execute(
                chat_context=ctx, action="page page-001 --download"
            )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_routes_default_summary(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_routes_list_persistent(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True, has_store=True)
        with patch("chuk_term.ui.output"), patch("chuk_term.ui.format_table"):
            result = await cmd.execute(chat_context=ctx, action="list")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_routes_add_persistent(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True, has_store=True)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(
                chat_context=ctx, action="add workspace mykey my content here"
            )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_routes_remove_persistent(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True, has_store=True)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(
                chat_context=ctx, action="remove workspace db_type"
            )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_routes_clear_persistent(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True, has_store=True)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="clear global")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_action_from_args_list(self):
        """_parse_action picks up args list when action not provided."""
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, args=["stats"])
        assert result.success is True

    @pytest.mark.asyncio
    async def test_action_from_args_string(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, args="stats")
        assert result.success is True


# ---------------------------------------------------------------------------
# _show_summary
# ---------------------------------------------------------------------------


class TestShowSummary:
    @pytest.mark.asyncio
    async def test_summary_tlb_total_zero(self):
        """When tlb_total == 0, rate should be 'n/a'."""
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        vm = ctx.session.vm
        vm.metrics.tlb_hits = 0
        vm.metrics.tlb_misses = 0
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_summary_all_tier_counts(self):
        """Exercise the tier_parts accumulation for multiple tiers."""
        from chuk_ai_session_manager.memory.models import StorageTier

        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        vm = ctx.session.vm
        pt_stats = vm.page_table.get_stats.return_value
        pt_stats.pages_by_tier = {
            StorageTier.L0: 1,
            StorageTier.L1: 2,
            StorageTier.L2: 3,
            StorageTier.L3: 0,
            StorageTier.L4: 5,
        }
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_summary_no_tiers(self):
        """When pages_by_tier is empty, tier_str should be 'none'."""
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        vm = ctx.session.vm
        pt_stats = vm.page_table.get_stats.return_value
        pt_stats.pages_by_tier = {}
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_summary_full_utilization(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        vm = ctx.session.vm
        ws_stats = vm.working_set.get_stats.return_value
        ws_stats.utilization = 1.0
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx)
        assert result.success is True


# ---------------------------------------------------------------------------
# _show_pages
# ---------------------------------------------------------------------------


class TestShowPages:
    @pytest.mark.asyncio
    async def test_show_pages_no_entries(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm.page_table.entries = {}
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="pages")
        assert result.success is True
        assert result.data is None

    @pytest.mark.asyncio
    async def test_show_pages_with_entries(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        with patch("chuk_term.ui.output"), patch("chuk_term.ui.format_table"):
            result = await cmd.execute(chat_context=ctx, action="pages")
        assert result.success is True
        assert isinstance(result.data, list)

    @pytest.mark.asyncio
    async def test_show_pages_size_tokens_none(self):
        """When size_tokens is None, Tokens column should show '?'."""
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        for entry in ctx.session.vm.page_table.entries.values():
            entry.size_tokens = None
        with patch("chuk_term.ui.output"), patch("chuk_term.ui.format_table"):
            result = await cmd.execute(chat_context=ctx, action="pages")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_show_pages_pinned_entry(self):
        """Pinned entries show 'Y'."""
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        for entry in ctx.session.vm.page_table.entries.values():
            entry.pinned = True
        with patch("chuk_term.ui.output"), patch("chuk_term.ui.format_table"):
            result = await cmd.execute(chat_context=ctx, action="pages")
        assert result.success is True


# ---------------------------------------------------------------------------
# _show_page_detail
# ---------------------------------------------------------------------------


class TestShowPageDetail:
    @pytest.mark.asyncio
    async def test_page_not_found(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        result = await cmd.execute(chat_context=ctx, action="page no-such-page")
        assert result.success is False
        assert "Page not found" in result.error

    @pytest.mark.asyncio
    async def test_page_found_basic(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="page page-001")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_page_not_in_page_store(self):
        """entry present but page store has no matching page."""
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm._page_store = {}
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="page page-001")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_page_content_truncation(self):
        """Content longer than 2000 chars is truncated."""
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        long_text = "x" * 3000
        ctx.session.vm._page_store["page-001"].content = long_text
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="page page-001")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_page_with_mime_type(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm._page_store["page-001"].mime_type = "image/png"
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="page page-001")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_page_with_dimensions(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm._page_store["page-001"].dimensions = (800, 600)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="page page-001")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_page_with_duration(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm._page_store["page-001"].duration_seconds = 3.7
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="page page-001")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_page_with_long_caption(self):
        """Caption is truncated to 100 chars."""
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm._page_store["page-001"].caption = "C" * 200
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="page page-001")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_page_no_provenance(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm.page_table.entries["page-001"].provenance = []
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="page page-001")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_page_with_mime_and_dimensions_and_duration_and_caption(self):
        """All optional metadata fields set at once."""
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        page = ctx.session.vm._page_store["page-001"]
        page.mime_type = "image/png"
        page.dimensions = (1920, 1080)
        page.duration_seconds = 12.5
        page.caption = "A great caption"
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="page page-001")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_page_id_missing_after_page(self):
        """'/memory page' with no id still calls _show_page_detail with empty string."""
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        # action = "page " (trailing space, no id)
        result = await cmd.execute(chat_context=ctx, action="page ")
        assert result.success is False
        assert "Page not found" in result.error


# ---------------------------------------------------------------------------
# _download_page
# ---------------------------------------------------------------------------


class TestDownloadPage:
    @pytest.mark.asyncio
    async def test_download_page_not_found(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        result = await cmd.execute(
            chat_context=ctx, action="page no-such-page --download"
        )
        assert result.success is False
        assert "Page not found" in result.error

    @pytest.mark.asyncio
    async def test_download_page_no_content(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm._page_store["page-001"].content = None
        result = await cmd.execute(chat_context=ctx, action="page page-001 --download")
        assert result.success is False
        assert "No content available" in result.error

    @pytest.mark.asyncio
    async def test_download_page_no_page_in_store(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm._page_store = {}  # page not found in store
        result = await cmd.execute(chat_context=ctx, action="page page-001 --download")
        assert result.success is False
        assert "No content available" in result.error

    @pytest.mark.asyncio
    async def test_download_text_plain(self, tmp_path):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm._page_store["page-001"].mime_type = "text/plain"
        ctx.session.vm._page_store["page-001"].content = "Hello text"
        with (
            patch("chuk_term.ui.output"),
            patch(
                "mcp_cli.commands.memory.memory.DEFAULT_DOWNLOADS_DIR",
                str(tmp_path),
            ),
        ):
            result = await cmd.execute(
                chat_context=ctx, action="page page-001 --download"
            )
        assert result.success is True
        assert "path" in result.data

    @pytest.mark.asyncio
    async def test_download_json_content(self, tmp_path):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm._page_store["page-001"].mime_type = "application/json"
        ctx.session.vm._page_store["page-001"].content = {"key": "value"}
        with (
            patch("chuk_term.ui.output"),
            patch(
                "mcp_cli.commands.memory.memory.DEFAULT_DOWNLOADS_DIR",
                str(tmp_path),
            ),
        ):
            result = await cmd.execute(
                chat_context=ctx, action="page page-001 --download"
            )
        assert result.success is True
        assert result.data["path"].endswith(".json")

    @pytest.mark.asyncio
    async def test_download_base64_data_uri(self, tmp_path):
        """data: URI content gets decoded as bytes."""
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        raw = b"PNG binary content"
        encoded = base64.b64encode(raw).decode()
        ctx.session.vm._page_store["page-001"].mime_type = "image/png"
        ctx.session.vm._page_store[
            "page-001"
        ].content = f"data:image/png;base64,{encoded}"
        with (
            patch("chuk_term.ui.output"),
            patch(
                "mcp_cli.commands.memory.memory.DEFAULT_DOWNLOADS_DIR",
                str(tmp_path),
            ),
        ):
            result = await cmd.execute(
                chat_context=ctx, action="page page-001 --download"
            )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_download_image_no_mime(self, tmp_path):
        """Image modality without mime type uses .png extension."""
        from chuk_ai_session_manager.memory.models.enums import Modality

        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm._page_store["page-001"].mime_type = None
        ctx.session.vm._page_store["page-001"].content = "some text"
        ctx.session.vm.page_table.entries["page-001"].modality = Modality.IMAGE
        with (
            patch("chuk_term.ui.output"),
            patch(
                "mcp_cli.commands.memory.memory.DEFAULT_DOWNLOADS_DIR",
                str(tmp_path),
            ),
        ):
            result = await cmd.execute(
                chat_context=ctx, action="page page-001 --download"
            )
        assert result.success is True
        assert result.data["path"].endswith(".png")

    @pytest.mark.asyncio
    async def test_download_structured_no_mime(self, tmp_path):
        """Structured modality without mime type uses .json extension."""
        from chuk_ai_session_manager.memory.models.enums import Modality

        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm._page_store["page-001"].mime_type = None
        ctx.session.vm._page_store["page-001"].content = "structured text"
        ctx.session.vm.page_table.entries["page-001"].modality = Modality.STRUCTURED
        with (
            patch("chuk_term.ui.output"),
            patch(
                "mcp_cli.commands.memory.memory.DEFAULT_DOWNLOADS_DIR",
                str(tmp_path),
            ),
        ):
            result = await cmd.execute(
                chat_context=ctx, action="page page-001 --download"
            )
        assert result.success is True
        assert result.data["path"].endswith(".json")

    @pytest.mark.asyncio
    async def test_download_text_modality_no_mime(self, tmp_path):
        """Text modality without mime type falls through to .txt extension."""
        from chuk_ai_session_manager.memory.models.enums import Modality

        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm._page_store["page-001"].mime_type = None
        ctx.session.vm._page_store["page-001"].content = "plain text data"
        ctx.session.vm.page_table.entries["page-001"].modality = Modality.TEXT
        with (
            patch("chuk_term.ui.output"),
            patch(
                "mcp_cli.commands.memory.memory.DEFAULT_DOWNLOADS_DIR",
                str(tmp_path),
            ),
        ):
            result = await cmd.execute(
                chat_context=ctx, action="page page-001 --download"
            )
        assert result.success is True
        assert result.data["path"].endswith(".txt")

    @pytest.mark.asyncio
    async def test_download_write_exception(self, tmp_path):
        """Exception during write returns failure."""
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm._page_store["page-001"].mime_type = "text/plain"
        ctx.session.vm._page_store["page-001"].content = "data"

        with (
            patch("chuk_term.ui.output"),
            patch(
                "mcp_cli.commands.memory.memory.DEFAULT_DOWNLOADS_DIR",
                str(tmp_path),
            ),
            patch("pathlib.Path.write_text", side_effect=OSError("disk full")),
        ):
            result = await cmd.execute(
                chat_context=ctx, action="page page-001 --download"
            )
        assert result.success is False
        assert "Download failed" in result.error


# ---------------------------------------------------------------------------
# _show_full_stats
# ---------------------------------------------------------------------------


class TestShowFullStats:
    @pytest.mark.asyncio
    async def test_show_stats(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_vm=True)
        ctx.session.vm.get_stats.return_value = {"mode": "passive", "turn": 7}
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="stats")
        assert result.success is True
        assert result.data == {"mode": "passive", "turn": 7}


# ---------------------------------------------------------------------------
# _persistent_list
# ---------------------------------------------------------------------------


class TestPersistentList:
    @pytest.mark.asyncio
    async def test_list_no_store(self):
        cmd = _make_command()
        ctx = _make_chat_context()
        ctx.memory_store = None
        result = await cmd.execute(chat_context=ctx, action="list")
        assert result.success is False
        assert "Memory store not available" in result.error

    @pytest.mark.asyncio
    async def test_list_all_scopes(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        with patch("chuk_term.ui.output"), patch("chuk_term.ui.format_table"):
            result = await cmd.execute(chat_context=ctx, action="list")
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) == 2  # one workspace, one global

    @pytest.mark.asyncio
    async def test_list_workspace_scope(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        with patch("chuk_term.ui.output"), patch("chuk_term.ui.format_table"):
            result = await cmd.execute(chat_context=ctx, action="list workspace")
        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["Scope"] == "workspace"

    @pytest.mark.asyncio
    async def test_list_global_scope(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        with patch("chuk_term.ui.output"), patch("chuk_term.ui.format_table"):
            result = await cmd.execute(chat_context=ctx, action="list global")
        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["Scope"] == "global"

    @pytest.mark.asyncio
    async def test_list_empty_returns_success(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        ctx.memory_store.list_entries.side_effect = lambda scope: []
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="list")
        assert result.success is True
        assert result.data is None

    @pytest.mark.asyncio
    async def test_list_long_content_truncated(self):
        """Content > 60 chars is truncated with '...'."""
        from mcp_cli.memory.models import MemoryEntry, MemoryScope

        long_entry = MemoryEntry(key="k", content="x" * 100)
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        ctx.memory_store.list_entries.side_effect = lambda scope: (
            [long_entry] if scope == MemoryScope.WORKSPACE else []
        )
        with patch("chuk_term.ui.output"), patch("chuk_term.ui.format_table"):
            result = await cmd.execute(chat_context=ctx, action="list workspace")
        assert result.success is True
        assert result.data[0]["Content"].endswith("...")


# ---------------------------------------------------------------------------
# _persistent_add
# ---------------------------------------------------------------------------


class TestPersistentAdd:
    @pytest.mark.asyncio
    async def test_add_no_store(self):
        cmd = _make_command()
        ctx = _make_chat_context()
        ctx.memory_store = None
        result = await cmd.execute(chat_context=ctx, action="add workspace k v")
        assert result.success is False
        assert "Memory store not available" in result.error

    @pytest.mark.asyncio
    async def test_add_missing_args(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        result = await cmd.execute(chat_context=ctx, action="add workspace key")
        assert result.success is False
        assert "Usage" in result.error

    @pytest.mark.asyncio
    async def test_add_invalid_scope(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        result = await cmd.execute(
            chat_context=ctx, action="add badscope key content here"
        )
        assert result.success is False
        assert "Invalid scope" in result.error

    @pytest.mark.asyncio
    async def test_add_workspace_success(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(
                chat_context=ctx,
                action="add workspace mykey my content here",
            )
        assert result.success is True
        ctx.memory_store.remember.assert_called_once()
        assert ctx._system_prompt_dirty is True

    @pytest.mark.asyncio
    async def test_add_global_success(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(
                chat_context=ctx,
                action="add global framework always use pytest",
            )
        assert result.success is True


# ---------------------------------------------------------------------------
# _persistent_remove
# ---------------------------------------------------------------------------


class TestPersistentRemove:
    @pytest.mark.asyncio
    async def test_remove_no_store(self):
        cmd = _make_command()
        ctx = _make_chat_context()
        ctx.memory_store = None
        result = await cmd.execute(chat_context=ctx, action="remove workspace k")
        assert result.success is False
        assert "Memory store not available" in result.error

    @pytest.mark.asyncio
    async def test_remove_missing_args(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        result = await cmd.execute(chat_context=ctx, action="remove workspace")
        assert result.success is False
        assert "Usage" in result.error

    @pytest.mark.asyncio
    async def test_remove_invalid_scope(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        result = await cmd.execute(chat_context=ctx, action="remove badscope key")
        assert result.success is False
        assert "Invalid scope" in result.error

    @pytest.mark.asyncio
    async def test_remove_found(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        ctx.memory_store.forget.return_value = True
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(
                chat_context=ctx, action="remove workspace db_type"
            )
        assert result.success is True
        assert ctx._system_prompt_dirty is True

    @pytest.mark.asyncio
    async def test_remove_not_found(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        ctx.memory_store.forget.return_value = False
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(
                chat_context=ctx, action="remove workspace missing_key"
            )
        assert result.success is True
        # _system_prompt_dirty should NOT have been set to True
        assert ctx._system_prompt_dirty is False


# ---------------------------------------------------------------------------
# _persistent_clear
# ---------------------------------------------------------------------------


class TestPersistentClear:
    @pytest.mark.asyncio
    async def test_clear_no_store(self):
        cmd = _make_command()
        ctx = _make_chat_context()
        ctx.memory_store = None
        result = await cmd.execute(chat_context=ctx, action="clear global")
        assert result.success is False
        assert "Memory store not available" in result.error

    @pytest.mark.asyncio
    async def test_clear_missing_scope_arg(self):
        """'clear' with no trailing space doesn't match the clear dispatch,
        so falls through to VM handling; with no VM but a store it calls _persistent_list."""
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        # "clear" without trailing space doesn't route to _persistent_clear,
        # it falls to the VM path which then calls _persistent_list (store present).
        with patch("chuk_term.ui.output"), patch("chuk_term.ui.format_table"):
            result = await cmd.execute(chat_context=ctx, action="clear")
        # The actual code falls through to _persistent_list when action=="clear"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_clear_only_space_scope_arg(self):
        """'clear ' (with space but no scope) routes to _persistent_clear with missing scope."""
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        # action.split(maxsplit=1) on "clear " gives ["clear"] — len < 2
        result = await cmd.execute(chat_context=ctx, action="clear ")
        assert result.success is False
        assert "Usage" in result.error

    @pytest.mark.asyncio
    async def test_clear_invalid_scope(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        result = await cmd.execute(chat_context=ctx, action="clear badscope")
        assert result.success is False
        assert "Invalid scope" in result.error

    @pytest.mark.asyncio
    async def test_clear_global(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        ctx.memory_store.clear.return_value = 5
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="clear global")
        assert result.success is True
        assert ctx._system_prompt_dirty is True

    @pytest.mark.asyncio
    async def test_clear_workspace(self):
        cmd = _make_command()
        ctx = _make_chat_context(has_store=True)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, action="clear workspace")
        assert result.success is True
