# src/mcp_cli/commands/memory/memory.py
"""
Unified memory command — visualize AI Virtual Memory state.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from chuk_term.ui import output, format_table
from mcp_cli.config.defaults import DEFAULT_DOWNLOADS_DIR

logger = logging.getLogger(__name__)


class MemoryCommand(UnifiedCommand):
    """View AI virtual memory state and manage persistent memories."""

    @property
    def name(self) -> str:
        return "memory"

    @property
    def aliases(self) -> list[str]:
        return ["vm", "mem"]

    @property
    def description(self) -> str:
        return "View AI virtual memory state and manage persistent memories"

    @property
    def help_text(self) -> str:
        return """
View AI virtual memory state and manage persistent memories.

VM Subcommands (requires --vm flag):
  /memory             - Summary dashboard (mode, pages, utilization, metrics)
  /memory pages       - Table of all memory pages
  /memory page <id>   - Detailed view of a specific page
  /memory page <id> --download  - Download page content to file
  /memory stats       - Full debug dump of all VM subsystem stats

Persistent Memory Subcommands:
  /memory list [workspace|global]  - List persistent memories
  /memory add <scope> <key> <content> - Add a memory
  /memory remove <scope> <key>     - Remove a memory
  /memory clear <scope>            - Clear all memories in a scope

Aliases: /vm, /mem

Examples:
  /vm                 - Quick overview of VM state
  /memory list workspace  - List workspace memories
  /memory add global test_framework "always use pytest"
  /memory remove workspace db_type
  /memory clear global
"""

    @property
    def modes(self) -> CommandMode:
        return CommandMode.CHAT

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="action",
                type=str,
                required=False,
                help="Subcommand: pages, page <id>, stats",
            ),
        ]

    async def execute(self, **kwargs: Any) -> CommandResult:
        """Execute the memory command."""
        chat_context = kwargs.get("chat_context")
        if not chat_context:
            return CommandResult(
                success=False,
                error="Memory command requires chat context.",
            )

        # Parse action from args
        action = self._parse_action(kwargs)

        # Dispatch persistent memory subcommands first
        if action is not None:
            if action.startswith("list"):
                return self._persistent_list(chat_context, action)
            if action.startswith("add "):
                return self._persistent_add(chat_context, action)
            if action.startswith("remove "):
                return self._persistent_remove(chat_context, action)
            if action.startswith("clear "):
                return self._persistent_clear(chat_context, action)

        # VM subcommands — require VM enabled
        session = getattr(chat_context, "session", None)
        vm = getattr(session, "vm", None) if session else None
        if not vm:
            # If no VM and no persistent memory action, show persistent list
            store = getattr(chat_context, "memory_store", None)
            if store:
                return self._persistent_list(chat_context, "list")
            return CommandResult(
                success=False,
                error="VM not enabled. Use /memory list|add|remove|clear for persistent memories.",
            )

        if action == "pages":
            return self._show_pages(vm)
        elif action is not None and action.startswith("page "):
            parts = action[5:].strip().split()
            page_id = parts[0] if parts else ""
            download = "--download" in parts
            if download:
                return self._download_page(vm, page_id)
            return self._show_page_detail(vm, page_id)
        elif action == "stats":
            return self._show_full_stats(vm)
        else:
            return self._show_summary(vm, chat_context)

    # ------------------------------------------------------------------
    # Arg parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_action(kwargs: dict[str, Any]) -> str | None:
        """Extract the action string from kwargs/args."""
        action = kwargs.get("action")
        if action is not None:
            return str(action)

        args_val = kwargs.get("args")
        if isinstance(args_val, list) and args_val:
            return " ".join(str(a) for a in args_val)
        if isinstance(args_val, str) and args_val:
            return args_val
        return None

    # ------------------------------------------------------------------
    # Subcommands
    # ------------------------------------------------------------------

    def _show_summary(self, vm: Any, chat_context: Any) -> CommandResult:
        """Show the summary dashboard."""
        metrics = vm.metrics
        ws_stats = vm.working_set.get_stats()
        pt_stats = vm.page_table.get_stats()

        # Budget from chat_context
        budget = getattr(chat_context, "_vm_budget", "?")

        # Build tier distribution string
        tier_parts = []
        for tier_name in ("L0", "L1", "L2", "L3", "L4"):
            from chuk_ai_session_manager.memory.models import StorageTier

            tier = StorageTier(tier_name)
            count = pt_stats.pages_by_tier.get(tier, 0)
            if count > 0:
                tier_parts.append(f"{tier_name}: {count}")
        tier_str = ", ".join(tier_parts) if tier_parts else "none"

        # TLB hit rate
        tlb_total = metrics.tlb_hits + metrics.tlb_misses
        tlb_rate = f"{metrics.tlb_hit_rate:.0%}" if tlb_total > 0 else "n/a"

        # Utilization bar
        util_pct = ws_stats.utilization
        bar_len = 20
        filled = int(util_pct * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        lines = [
            f"Mode: {vm.mode.value}   Turn: {vm.turn}   Budget: {budget:,} tokens",
            "",
            f"Working Set  [{bar}] {util_pct:.0%}",
            f"  L0 pages: {ws_stats.l0_pages}   L1 pages: {ws_stats.l1_pages}",
            f"  Tokens used: {ws_stats.tokens_used:,}   Available: {ws_stats.tokens_available:,}",
            "",
            f"Page Table   Total: {pt_stats.total_pages}   Dirty: {pt_stats.dirty_pages}",
            f"  By tier: {tier_str}",
            "",
            "Metrics",
            f"  Faults: {metrics.faults_total} total, {metrics.faults_this_turn} this turn",
            f"  Evictions: {metrics.evictions_total} total, {metrics.evictions_this_turn} this turn",
            f"  TLB: {metrics.tlb_hits} hits, {metrics.tlb_misses} misses ({tlb_rate})",
        ]

        output.panel(
            "\n".join(lines),
            title="AI Virtual Memory",
            style="cyan",
        )
        return CommandResult(success=True)

    def _show_pages(self, vm: Any) -> CommandResult:
        """Show table of all pages."""
        entries = vm.page_table.entries
        if not entries:
            output.info("No pages in page table.")
            return CommandResult(success=True)

        # Build table rows sorted by tier then importance
        rows = []
        for page_id, entry in entries.items():
            rows.append(
                {
                    "Page ID": page_id,
                    "Type": entry.page_type.value,
                    "Tier": entry.tier.value,
                    "Tokens": str(entry.size_tokens or "?"),
                    "Importance": f"{entry.eviction_priority:.1f}",
                    "Pinned": "Y" if entry.pinned else "",
                    "Compression": entry.compression_level.name.lower(),
                    "Accesses": str(entry.access_count),
                }
            )

        # Sort: L0 first, then L1, etc., then by eviction_priority ascending
        tier_order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
        rows.sort(key=lambda r: (tier_order.get(r["Tier"], 9), float(r["Importance"])))

        output.rule("[bold]Memory Pages[/bold]", style="primary")
        table = format_table(
            rows,
            title=None,
            columns=[
                "Page ID",
                "Type",
                "Tier",
                "Tokens",
                "Importance",
                "Pinned",
                "Compression",
                "Accesses",
            ],
        )
        output.print_table(table)
        output.print()
        output.tip("Use: /memory page <id> to see full page content")

        return CommandResult(success=True, data=rows)

    def _show_page_detail(self, vm: Any, page_id: str) -> CommandResult:
        """Show detailed view of a single page."""
        # Look up entry
        entry = vm.page_table.entries.get(page_id)
        if not entry:
            return CommandResult(
                success=False,
                error=f"Page not found: {page_id}",
            )

        # Look up content from page store
        page = vm._page_store.get(page_id)
        content = page.content if page else "[not in memory]"

        # Truncate very long content
        max_preview = 2000
        if isinstance(content, str) and len(content) > max_preview:
            content = (
                content[:max_preview]
                + f"\n\n... ({len(content) - max_preview} more chars)"
            )

        # Build detail view
        lines = [
            f"Page ID:     {entry.page_id}",
            f"Type:        {entry.page_type.value}",
            f"Tier:        {entry.tier.value}",
            f"Tokens:      {entry.size_tokens or '?'}",
            f"Compression: {entry.compression_level.name}",
            f"Pinned:      {entry.pinned}",
            f"Dirty:       {entry.dirty}",
            f"Accesses:    {entry.access_count}",
            f"Last access: {entry.last_accessed}",
            f"Modality:    {entry.modality.value}",
        ]

        # Modality-specific metadata
        if page:
            mime = getattr(page, "mime_type", None)
            if mime:
                lines.append(f"MIME type:   {mime}")
            dims = getattr(page, "dimensions", None)
            if dims:
                lines.append(f"Dimensions:  {dims[0]}x{dims[1]}")
            duration = getattr(page, "duration_seconds", None)
            if duration:
                lines.append(f"Duration:    {duration:.1f}s")
            caption = getattr(page, "caption", None)
            if caption:
                lines.append(f"Caption:     {caption[:100]}")

        if entry.provenance:
            lines.append(f"Provenance:  {', '.join(entry.provenance)}")

        lines.append("")
        lines.append("--- Content ---")
        lines.append(str(content))

        output.panel(
            "\n".join(lines),
            title=f"Page: {page_id}",
            style="cyan",
        )
        return CommandResult(success=True)

    def _download_page(self, vm: Any, page_id: str) -> CommandResult:
        """Download page content to a local file."""
        entry = vm.page_table.entries.get(page_id)
        if not entry:
            return CommandResult(success=False, error=f"Page not found: {page_id}")

        page = vm._page_store.get(page_id)
        if not page or page.content is None:
            return CommandResult(
                success=False, error=f"No content available for page: {page_id}"
            )

        content = page.content
        download_dir = Path(DEFAULT_DOWNLOADS_DIR).expanduser()
        download_dir.mkdir(parents=True, exist_ok=True)

        # Determine extension from mime_type or modality
        mime = getattr(page, "mime_type", None) or ""
        modality = entry.modality.value

        ext_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "application/json": ".json",
            "text/plain": ".txt",
        }
        ext = ext_map.get(mime, "")
        if not ext:
            if modality == "image":
                ext = ".png"
            elif modality == "structured":
                ext = ".json"
            else:
                ext = ".txt"

        out_path = download_dir / f"{page_id}{ext}"

        try:
            if isinstance(content, str) and content.startswith("data:"):
                # base64 data URI: data:image/png;base64,iVBOR...
                _, encoded = content.split(",", 1)
                out_path.write_bytes(base64.b64decode(encoded))
            elif isinstance(content, dict):
                out_path.write_text(json.dumps(content, indent=2))
            else:
                out_path.write_text(str(content))

            output.success(f"Downloaded to: {out_path}")
            output.info(f"  Size: {out_path.stat().st_size:,} bytes")
            return CommandResult(success=True, data={"path": str(out_path)})

        except Exception as exc:
            logger.error(f"Download failed for {page_id}: {exc}")
            return CommandResult(success=False, error=f"Download failed: {exc}")

    def _show_full_stats(self, vm: Any) -> CommandResult:
        """Show full debug dump of all subsystem stats."""
        stats = vm.get_stats()

        # Format as indented JSON for readability
        formatted = json.dumps(stats, indent=2, default=str)

        output.panel(
            formatted,
            title="VM Full Stats Dump",
            style="cyan",
        )
        return CommandResult(success=True, data=stats)

    # ------------------------------------------------------------------
    # Persistent memory subcommands
    # ------------------------------------------------------------------

    @staticmethod
    def _get_store(chat_context: Any):
        """Get the MemoryScopeStore from chat context."""
        return getattr(chat_context, "memory_store", None)

    def _persistent_list(self, chat_context: Any, action: str) -> CommandResult:
        """List persistent memories."""
        from mcp_cli.memory.models import MemoryScope

        store = self._get_store(chat_context)
        if not store:
            return CommandResult(success=False, error="Memory store not available.")

        # Parse optional scope filter
        parts = action.split()
        scope_filter = parts[1] if len(parts) > 1 else None

        scopes = (
            [MemoryScope(scope_filter)]
            if scope_filter
            else [MemoryScope.WORKSPACE, MemoryScope.GLOBAL]
        )

        rows = []
        for scope in scopes:
            for entry in store.list_entries(scope):
                rows.append(
                    {
                        "Scope": scope.value,
                        "Key": entry.key,
                        "Content": (
                            entry.content[:60] + "..."
                            if len(entry.content) > 60
                            else entry.content
                        ),
                        "Updated": entry.updated_at.strftime("%Y-%m-%d %H:%M"),
                    }
                )

        if not rows:
            output.info("No persistent memories found.")
            return CommandResult(success=True)

        table = format_table(
            rows,
            title=None,
            columns=["Scope", "Key", "Content", "Updated"],
        )
        output.rule("[bold]Persistent Memories[/bold]", style="primary")
        output.print_table(table)
        return CommandResult(success=True, data=rows)

    def _persistent_add(self, chat_context: Any, action: str) -> CommandResult:
        """Add a persistent memory."""
        from mcp_cli.memory.models import MemoryScope

        store = self._get_store(chat_context)
        if not store:
            return CommandResult(success=False, error="Memory store not available.")

        # Parse: add <scope> <key> <content...>
        parts = action.split(maxsplit=3)
        if len(parts) < 4:
            return CommandResult(
                success=False,
                error="Usage: /memory add <workspace|global> <key> <content>",
            )

        try:
            scope = MemoryScope(parts[1])
        except ValueError:
            return CommandResult(
                success=False,
                error=f"Invalid scope: {parts[1]}. Use 'workspace' or 'global'.",
            )

        key, content = parts[2], parts[3]
        entry = store.remember(scope, key, content)
        chat_context._system_prompt_dirty = True
        output.success(f"Remembered '{entry.key}' in {scope.value} scope.")
        return CommandResult(success=True)

    def _persistent_remove(self, chat_context: Any, action: str) -> CommandResult:
        """Remove a persistent memory."""
        from mcp_cli.memory.models import MemoryScope

        store = self._get_store(chat_context)
        if not store:
            return CommandResult(success=False, error="Memory store not available.")

        # Parse: remove <scope> <key>
        parts = action.split(maxsplit=2)
        if len(parts) < 3:
            return CommandResult(
                success=False,
                error="Usage: /memory remove <workspace|global> <key>",
            )

        try:
            scope = MemoryScope(parts[1])
        except ValueError:
            return CommandResult(
                success=False,
                error=f"Invalid scope: {parts[1]}. Use 'workspace' or 'global'.",
            )

        removed = store.forget(scope, parts[2])
        if removed:
            chat_context._system_prompt_dirty = True
            output.success(f"Forgot '{parts[2]}' from {scope.value} scope.")
        else:
            output.warning(f"No memory with key '{parts[2]}' in {scope.value} scope.")
        return CommandResult(success=True)

    def _persistent_clear(self, chat_context: Any, action: str) -> CommandResult:
        """Clear all persistent memories in a scope."""
        from mcp_cli.memory.models import MemoryScope

        store = self._get_store(chat_context)
        if not store:
            return CommandResult(success=False, error="Memory store not available.")

        # Parse: clear <scope>
        parts = action.split(maxsplit=1)
        if len(parts) < 2:
            return CommandResult(
                success=False,
                error="Usage: /memory clear <workspace|global>",
            )

        try:
            scope = MemoryScope(parts[1])
        except ValueError:
            return CommandResult(
                success=False,
                error=f"Invalid scope: {parts[1]}. Use 'workspace' or 'global'.",
            )

        count = store.clear(scope)
        chat_context._system_prompt_dirty = True
        output.success(f"Cleared {count} memories from {scope.value} scope.")
        return CommandResult(success=True)
