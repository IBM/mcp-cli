# tests/commands/apps/test_apps_command.py
"""Tests for the AppsCommand."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.commands.apps.apps import AppsCommand
from mcp_cli.commands.base import CommandResult

# The module under test imports get_context with:
#   from mcp_cli.context import get_context
# inside execute(), so the live binding in the calling namespace is
# mcp_cli.context.get_context. We patch that module-level symbol.
_GET_CONTEXT = "mcp_cli.context.get_context"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(
    name: str,
    namespace: str = "srv",
    has_app_ui: bool = True,
    app_resource_uri: str | None = "http://localhost:8080",
    description: str = "A tool",
) -> MagicMock:
    t = MagicMock()
    t.name = name
    t.namespace = namespace
    t.has_app_ui = has_app_ui
    t.app_resource_uri = app_resource_uri
    t.description = description
    return t


def _make_running_app(
    tool_name: str = "my_tool",
    url: str = "http://localhost:8080",
    state_value: str = "running",
    server_name: str = "srv",
) -> MagicMock:
    app = MagicMock()
    app.tool_name = tool_name
    app.url = url
    app.state = MagicMock(value=state_value)
    app.server_name = server_name
    return app


def _make_tool_manager(tools=None, app_host=None):
    """Return a mock ToolManager."""
    tm = MagicMock()
    tm.get_all_tools = AsyncMock(return_value=tools or [])
    if app_host is None:
        tm._app_host = None
        # When _app_host is None, the code checks `tool_manager._app_host is None`.
        # We don't need to set app_host on the mock.
    else:
        tm._app_host = app_host
        tm.app_host = app_host
    return tm


def _make_context(tool_manager=None):
    ctx = MagicMock()
    ctx.tool_manager = tool_manager
    return ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cmd():
    return AppsCommand()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestAppsCommandProperties:
    def test_name(self, cmd):
        assert cmd.name == "apps"

    def test_aliases(self, cmd):
        assert cmd.aliases == []

    def test_description(self, cmd):
        desc = cmd.description.lower()
        assert "apps" in desc or "interactive" in desc

    def test_help_text_contains_usage(self, cmd):
        text = cmd.help_text
        assert "/apps" in text
        assert "running" in text
        assert "stop" in text

    def test_parameters_single_subcommand(self, cmd):
        params = cmd.parameters
        assert len(params) == 1
        assert params[0].name == "subcommand"
        assert params[0].required is False


# ---------------------------------------------------------------------------
# execute() — guard checks (no context / no tool_manager)
# ---------------------------------------------------------------------------


class TestAppsCommandGuards:
    async def test_no_context_returns_error(self, cmd):
        with patch(_GET_CONTEXT, return_value=None):
            result = await cmd.execute()
        assert result.success is False
        assert "No tool manager" in result.error

    async def test_context_without_tool_manager(self, cmd):
        ctx = _make_context(tool_manager=None)
        with patch(_GET_CONTEXT, return_value=ctx):
            result = await cmd.execute()
        assert result.success is False
        assert "No tool manager" in result.error

    async def test_exception_bubbles_as_error(self, cmd):
        with patch(_GET_CONTEXT, side_effect=RuntimeError("boom")):
            result = await cmd.execute()
        assert result.success is False
        assert "Failed to execute apps command" in result.error
        assert "boom" in result.error


# ---------------------------------------------------------------------------
# execute() — subcommand routing
# ---------------------------------------------------------------------------


class TestAppsCommandRouting:
    async def test_default_subcommand_calls_list(self, cmd):
        ctx = _make_context(_make_tool_manager())
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch.object(
                AppsCommand,
                "_list_app_tools",
                new_callable=lambda: (
                    lambda *a, **k: AsyncMock(
                        return_value=CommandResult(success=True)
                    )()
                ),
            ) as _:
                # Use a cleaner mock approach
                pass
        # Re-do with proper AsyncMock
        ctx = _make_context(_make_tool_manager())
        mock_list = AsyncMock(return_value=CommandResult(success=True))
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch.object(AppsCommand, "_list_app_tools", mock_list):
                result = await cmd.execute()
        mock_list.assert_called_once()
        assert result.success is True

    async def test_subcommand_running_routes_to_show_running(self, cmd):
        ctx = _make_context(_make_tool_manager())
        mock_fn = AsyncMock(return_value=CommandResult(success=True))
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch.object(AppsCommand, "_show_running", mock_fn):
                await cmd.execute(subcommand="running")
        mock_fn.assert_called_once()

    async def test_subcommand_stop_routes_to_stop_all(self, cmd):
        ctx = _make_context(_make_tool_manager())
        mock_fn = AsyncMock(return_value=CommandResult(success=True))
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch.object(AppsCommand, "_stop_all", mock_fn):
                await cmd.execute(subcommand="stop")
        mock_fn.assert_called_once()

    async def test_args_kwarg_overrides_subcommand(self, cmd):
        """A non-empty args kwarg overrides the subcommand kwarg."""
        ctx = _make_context(_make_tool_manager())
        mock_fn = AsyncMock(return_value=CommandResult(success=True))
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch.object(AppsCommand, "_show_running", mock_fn):
                # subcommand="list" but args="running" should trigger _show_running
                await cmd.execute(subcommand="list", args="running")
        mock_fn.assert_called_once()

    async def test_whitespace_args_does_not_override(self, cmd):
        """Whitespace-only args kwarg keeps the explicit subcommand."""
        ctx = _make_context(_make_tool_manager())
        mock_fn = AsyncMock(return_value=CommandResult(success=True))
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch.object(AppsCommand, "_list_app_tools", mock_fn):
                await cmd.execute(subcommand="list", args="   ")
        mock_fn.assert_called_once()

    async def test_unknown_subcommand_falls_through_to_list(self, cmd):
        ctx = _make_context(_make_tool_manager())
        mock_fn = AsyncMock(return_value=CommandResult(success=True))
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch.object(AppsCommand, "_list_app_tools", mock_fn):
                await cmd.execute(subcommand="unknown_thing")
        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# _list_app_tools  (called through execute with subcommand="list")
# ---------------------------------------------------------------------------


class TestListAppTools:
    async def test_no_app_ui_tools_returns_message(self, cmd):
        tools = [_make_tool("t1", has_app_ui=False)]
        ctx = _make_context(_make_tool_manager(tools=tools))
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    result = await cmd.execute(subcommand="list")
        assert result.success is True
        assert "No tools" in result.output

    async def test_with_app_ui_tools_builds_table(self, cmd):
        tools = [
            _make_tool(
                "tool_a",
                namespace="s1",
                has_app_ui=True,
                app_resource_uri="http://s1:9000",
            ),
            _make_tool(
                "tool_b", namespace="s1", has_app_ui=True, app_resource_uri=None
            ),  # falls back to "unknown"
            _make_tool("tool_c", namespace="s2", has_app_ui=False),
        ]
        ctx = _make_context(_make_tool_manager(tools=tools))
        mock_table = MagicMock()
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch("chuk_term.ui.output") as mock_out:
                with patch("chuk_term.ui.format_table", return_value=mock_table):
                    result = await cmd.execute(subcommand="list")

        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 2  # only has_app_ui=True
        uris = [row["UI Resource"] for row in result.data]
        assert "http://s1:9000" in uris
        assert "unknown" in uris
        mock_out.print_table.assert_called_once_with(mock_table)

    async def test_description_is_truncated_at_60_chars(self, cmd):
        tools = [_make_tool("t1", has_app_ui=True, description="x" * 100)]
        ctx = _make_context(_make_tool_manager(tools=tools))
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table", return_value=MagicMock()):
                    result = await cmd.execute(subcommand="list")
        assert result.success is True
        assert len(result.data[0]["Description"]) <= 60

    async def test_format_table_title_includes_count(self, cmd):
        tools = [_make_tool("t1", has_app_ui=True)]
        ctx = _make_context(_make_tool_manager(tools=tools))
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch("chuk_term.ui.output"):
                with patch(
                    "chuk_term.ui.format_table", return_value=MagicMock()
                ) as mock_fmt:
                    await cmd.execute(subcommand="list")
        call_kwargs = mock_fmt.call_args
        # title kwarg contains "1"
        assert "1" in call_kwargs.kwargs.get("title", "")


# ---------------------------------------------------------------------------
# _show_running  (subcommand="running")
# ---------------------------------------------------------------------------


class TestShowRunning:
    async def test_no_app_host_returns_message(self, cmd):
        tm = _make_tool_manager()  # _app_host is None by default
        ctx = _make_context(tm)
        with patch(_GET_CONTEXT, return_value=ctx):
            result = await cmd.execute(subcommand="running")
        assert result.success is True
        assert "No MCP Apps have been launched" in result.output

    async def test_app_host_with_no_running_apps(self, cmd):
        app_host = MagicMock()
        app_host.get_running_apps.return_value = []
        tm = _make_tool_manager(app_host=app_host)
        ctx = _make_context(tm)
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    result = await cmd.execute(subcommand="running")
        assert result.success is True
        assert "No MCP Apps currently running" in result.output

    async def test_running_apps_builds_table(self, cmd):
        apps = [
            _make_running_app("tool1", state_value="running"),
            _make_running_app("tool2", state_value="starting", server_name="s2"),
        ]
        app_host = MagicMock()
        app_host.get_running_apps.return_value = apps
        tm = _make_tool_manager(app_host=app_host)
        ctx = _make_context(tm)
        mock_table = MagicMock()
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch("chuk_term.ui.output") as mock_out:
                with patch("chuk_term.ui.format_table", return_value=mock_table):
                    result = await cmd.execute(subcommand="running")
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 2
        assert result.data[0]["Tool"] == "tool1"
        assert result.data[0]["State"] == "running"
        assert result.data[1]["Server"] == "s2"
        mock_out.print_table.assert_called_once_with(mock_table)


# ---------------------------------------------------------------------------
# _stop_all  (subcommand="stop")
# ---------------------------------------------------------------------------


class TestStopAll:
    async def test_no_app_host_returns_message(self, cmd):
        tm = _make_tool_manager()  # _app_host is None
        ctx = _make_context(tm)
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch("chuk_term.ui.output"):
                result = await cmd.execute(subcommand="stop")
        assert result.success is True
        assert "No MCP Apps to stop" in result.output

    async def test_stop_with_running_apps(self, cmd):
        apps = [_make_running_app("t1"), _make_running_app("t2")]
        app_host = MagicMock()
        app_host.get_running_apps.return_value = apps
        app_host.close_all = AsyncMock()
        tm = _make_tool_manager(app_host=app_host)
        ctx = _make_context(tm)
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch("chuk_term.ui.output") as mock_out:
                result = await cmd.execute(subcommand="stop")
        assert result.success is True
        assert "Stopped 2" in result.output
        app_host.close_all.assert_called_once()
        mock_out.info.assert_called_once()

    async def test_stop_when_no_apps_were_running(self, cmd):
        app_host = MagicMock()
        app_host.get_running_apps.return_value = []
        app_host.close_all = AsyncMock()
        tm = _make_tool_manager(app_host=app_host)
        ctx = _make_context(tm)
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch("chuk_term.ui.output") as mock_out:
                result = await cmd.execute(subcommand="stop")
        assert result.success is True
        assert "No MCP Apps were running" in result.output
        app_host.close_all.assert_called_once()
        mock_out.info.assert_called_once()
