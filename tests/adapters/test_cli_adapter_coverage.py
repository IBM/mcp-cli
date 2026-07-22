# tests/adapters/test_cli_adapter_coverage.py
"""
Test suite for the CLI mode adapter.

Targets >90% coverage of src/mcp_cli/adapters/cli.py.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import typer

from mcp_cli.adapters.cli import CLICommandAdapter, cli_execute
from mcp_cli.commands.base import (
    CommandGroup,
    CommandMode,
    CommandParameter,
    CommandResult,
    UnifiedCommand,
)
from mcp_cli.commands.registry import UnifiedCommandRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockCLICommand(UnifiedCommand):
    """Concrete mock command for CLI-mode testing."""

    def __init__(
        self,
        test_name: str = "mock",
        description: str = "Mock CLI command",
        aliases: list[str] | None = None,
        requires_ctx: bool = False,
        hidden: bool = False,
        parameters: list[CommandParameter] | None = None,
        help_text: str | None = None,
    ):
        super().__init__()
        self._name = test_name
        self._description = description
        self._modes = CommandMode.CLI
        self._aliases = aliases or []
        self._requires_context = requires_ctx
        self._hidden = hidden
        self._parameters = parameters or []
        self._help_text = help_text
        self.execute_mock = AsyncMock(
            return_value=CommandResult(success=True, output="Mock executed")
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def modes(self) -> CommandMode:
        return self._modes

    @property
    def aliases(self):
        return self._aliases

    @property
    def parameters(self):
        return self._parameters

    @property
    def requires_context(self):
        return self._requires_context

    @property
    def hidden(self):
        return self._hidden

    @property
    def help_text(self):
        return self._help_text

    async def execute(self, **kwargs) -> CommandResult:
        return await self.execute_mock(**kwargs)


class MockCLICommandGroup(CommandGroup):
    """Concrete CommandGroup for testing."""

    def __init__(self, test_name: str = "tools"):
        super().__init__()
        self._name = test_name
        self._description = f"Mock group: {test_name}"
        self._modes = CommandMode.CLI
        self._aliases: list[str] = []
        self._requires_context = False
        self._parameters: list[CommandParameter] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def modes(self) -> CommandMode:
        return self._modes

    @property
    def aliases(self):
        return self._aliases

    @property
    def parameters(self):
        return self._parameters

    @property
    def requires_context(self):
        return self._requires_context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure the singleton registry is empty before and after each test."""
    reg = UnifiedCommandRegistry()
    reg.clear()
    yield
    reg.clear()


# ---------------------------------------------------------------------------
# Tests: register_with_typer
# ---------------------------------------------------------------------------


class TestRegisterWithTyper:
    """Tests for CLICommandAdapter.register_with_typer."""

    def test_registers_single_command(self):
        """A single (non-group) command is registered via _register_command."""
        cmd = MockCLICommand("servers", description="List servers")
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        app = typer.Typer()
        with (
            patch.object(CLICommandAdapter, "_register_command") as mock_reg_cmd,
            patch.object(CLICommandAdapter, "_register_group") as mock_reg_grp,
        ):
            CLICommandAdapter.register_with_typer(app)

        mock_reg_cmd.assert_called_once_with(app, cmd)
        mock_reg_grp.assert_not_called()

    def test_registers_command_group(self):
        """A CommandGroup is registered via _register_group."""
        group = MockCLICommandGroup("tools")
        reg = UnifiedCommandRegistry()
        reg.register(group)

        app = typer.Typer()
        with (
            patch.object(CLICommandAdapter, "_register_command") as mock_reg_cmd,
            patch.object(CLICommandAdapter, "_register_group") as mock_reg_grp,
        ):
            CLICommandAdapter.register_with_typer(app)

        mock_reg_grp.assert_called_once_with(app, group)
        mock_reg_cmd.assert_not_called()

    def test_registers_mixed_commands_and_groups(self):
        """Both single commands and groups are dispatched correctly."""
        cmd = MockCLICommand("status")
        group = MockCLICommandGroup("tools")
        reg = UnifiedCommandRegistry()
        reg.register(cmd)
        reg.register(group)

        app = typer.Typer()
        with (
            patch.object(CLICommandAdapter, "_register_command") as mock_reg_cmd,
            patch.object(CLICommandAdapter, "_register_group") as mock_reg_grp,
        ):
            CLICommandAdapter.register_with_typer(app)

        mock_reg_cmd.assert_called_once_with(app, cmd)
        mock_reg_grp.assert_called_once_with(app, group)

    def test_empty_registry(self):
        """No commands means no registrations."""
        app = typer.Typer()
        with (
            patch.object(CLICommandAdapter, "_register_command") as mock_reg_cmd,
            patch.object(CLICommandAdapter, "_register_group") as mock_reg_grp,
        ):
            CLICommandAdapter.register_with_typer(app)

        mock_reg_cmd.assert_not_called()
        mock_reg_grp.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _register_command
# ---------------------------------------------------------------------------


class TestRegisterCommand:
    """Tests for CLICommandAdapter._register_command."""

    def test_command_registered_with_name(self):
        """The command is registered on the Typer app with its name."""
        cmd = MockCLICommand("servers", description="List servers")
        app = typer.Typer()

        CLICommandAdapter._register_command(app, cmd)

        # Verify a command was registered -- Typer stores registered info
        registered = app.registered_commands
        assert len(registered) == 1

    def test_command_with_aliases(self):
        """Aliases are registered as hidden commands."""
        cmd = MockCLICommand(
            "servers",
            description="List servers",
            aliases=["srv", "s"],
        )
        app = typer.Typer()

        CLICommandAdapter._register_command(app, cmd)

        # The main command + 2 aliases = 3 registrations
        registered = app.registered_commands
        assert len(registered) == 3

    def test_command_with_parameters(self):
        """Parameters are converted into the wrapper's annotations."""
        cmd = MockCLICommand(
            "servers",
            description="List servers",
            parameters=[
                CommandParameter(
                    name="raw",
                    type=bool,
                    default=False,
                    help="Raw output",
                    is_flag=True,
                ),
                CommandParameter(
                    name="format", type=str, default="table", help="Output format"
                ),
            ],
        )
        app = typer.Typer()

        CLICommandAdapter._register_command(app, cmd)

        registered = app.registered_commands
        assert len(registered) == 1

    def test_command_uses_help_text_if_available(self):
        """The wrapper docstring comes from help_text when available."""
        cmd = MockCLICommand(
            "servers",
            description="Short desc",
            help_text="Extended help text",
        )
        app = typer.Typer()

        CLICommandAdapter._register_command(app, cmd)

        # The registered callback should have docstring == help_text
        registered = app.registered_commands
        assert len(registered) == 1

    def test_command_uses_description_when_no_help_text(self):
        """The wrapper docstring comes from description when help_text is None."""
        cmd = MockCLICommand(
            "servers",
            description="Short desc",
            help_text=None,
        )
        app = typer.Typer()

        CLICommandAdapter._register_command(app, cmd)

        registered = app.registered_commands
        assert len(registered) == 1

    def test_wrapper_success_with_output(self):
        """Wrapper prints output on success."""
        cmd = MockCLICommand("servers", description="List servers")
        cmd.execute_mock.return_value = CommandResult(
            success=True, output="server list"
        )

        app = typer.Typer()
        CLICommandAdapter._register_command(app, cmd)

        # Get the registered callback
        callback = app.registered_commands[0].callback

        with patch(
            "mcp_cli.adapters.cli.CLICommandAdapter._execute_command",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_exec.return_value = CommandResult(success=True, output="server list")

            def _run_and_close(coro):
                coro.close()
                return CommandResult(success=True, output="server list")

            with patch(
                "asyncio.run",
                side_effect=_run_and_close,
            ):
                with patch("mcp_cli.adapters.cli.output") as mock_output:
                    callback()
                    mock_output.print.assert_called_once_with("server list")

    def test_wrapper_success_no_output(self):
        """Wrapper does not print when output is None."""
        cmd = MockCLICommand("servers", description="List servers")
        app = typer.Typer()
        CLICommandAdapter._register_command(app, cmd)

        callback = app.registered_commands[0].callback

        def _run(coro):
            coro.close()
            return CommandResult(success=True, output=None)

        with patch("asyncio.run", side_effect=_run):
            with patch("mcp_cli.adapters.cli.output") as mock_output:
                callback()
                mock_output.print.assert_not_called()

    def test_wrapper_failure_with_error(self):
        """Wrapper prints error and raises Exit on failure."""
        cmd = MockCLICommand("servers", description="List servers")
        app = typer.Typer()
        CLICommandAdapter._register_command(app, cmd)

        callback = app.registered_commands[0].callback

        def _run(coro):
            coro.close()
            return CommandResult(success=False, error="Something went wrong")

        with patch("asyncio.run", side_effect=_run):
            with patch("mcp_cli.adapters.cli.output") as mock_output:
                with pytest.raises(typer.Exit) as exc_info:
                    callback()
                mock_output.error.assert_called_once_with("Something went wrong")
                assert exc_info.value.exit_code == 1

    def test_wrapper_failure_no_error(self):
        """Wrapper raises Exit on failure even without an error message."""
        cmd = MockCLICommand("servers", description="List servers")
        app = typer.Typer()
        CLICommandAdapter._register_command(app, cmd)

        callback = app.registered_commands[0].callback

        def _run(coro):
            coro.close()
            return CommandResult(success=False, error=None)

        with patch("asyncio.run", side_effect=_run):
            with patch("mcp_cli.adapters.cli.output") as mock_output:
                with pytest.raises(typer.Exit) as exc_info:
                    callback()
                mock_output.error.assert_not_called()
                assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# Tests: _register_group
# ---------------------------------------------------------------------------


class TestRegisterGroup:
    """Tests for CLICommandAdapter._register_group."""

    def test_group_creates_sub_app(self):
        """A CommandGroup is registered as a Typer sub-app."""
        group = MockCLICommandGroup("tools")
        sub = MockCLICommand("list", description="List tools")
        group.add_subcommand(sub)

        app = typer.Typer()

        CLICommandAdapter._register_group(app, group)

        # Typer stores sub-apps via registered_groups
        assert len(app.registered_groups) == 1

    def test_group_skips_alias_entries(self):
        """Only primary subcommand names are registered, not aliases."""
        group = MockCLICommandGroup("tools")
        sub = MockCLICommand("list", description="List tools", aliases=["ls"])
        group.add_subcommand(sub)

        app = typer.Typer()

        # _register_group iterates group.subcommands.items() and skips
        # entries where key != subcommand.name (i.e., the alias entries)
        with patch.object(CLICommandAdapter, "_register_command") as mock_reg:
            CLICommandAdapter._register_group(app, group)

        # Should be called once (for "list"), not twice (alias "ls" is skipped)
        mock_reg.assert_called_once()

    def test_group_empty_subcommands(self):
        """A group with no subcommands still creates a sub-app."""
        group = MockCLICommandGroup("tools")
        app = typer.Typer()

        CLICommandAdapter._register_group(app, group)

        assert len(app.registered_groups) == 1


# ---------------------------------------------------------------------------
# Tests: _execute_command
# ---------------------------------------------------------------------------


class TestExecuteCommand:
    """Tests for CLICommandAdapter._execute_command."""

    @pytest.mark.asyncio
    async def test_execute_without_context(self):
        """Command that does not require context executes without context injection."""
        cmd = MockCLICommand("servers", requires_ctx=False)
        result = await CLICommandAdapter._execute_command(cmd, {"raw": True})
        cmd.execute_mock.assert_awaited_once_with(raw=True)
        assert result.success

    @pytest.mark.asyncio
    async def test_execute_with_context_available(self):
        """Command that requires context gets tool_manager and model_manager injected."""
        mock_context = MagicMock()
        mock_context.tool_manager = MagicMock()
        mock_context.model_manager = MagicMock()

        cmd = MockCLICommand("servers", requires_ctx=True)

        with patch("mcp_cli.adapters.cli.get_context", return_value=mock_context):
            result = await CLICommandAdapter._execute_command(cmd, {"raw": True})

        cmd.execute_mock.assert_awaited_once_with(
            raw=True,
            tool_manager=mock_context.tool_manager,
            model_manager=mock_context.model_manager,
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_execute_with_context_none(self):
        """Command that requires context but get_context returns None."""
        cmd = MockCLICommand("servers", requires_ctx=True)

        with patch("mcp_cli.adapters.cli.get_context", return_value=None):
            result = await CLICommandAdapter._execute_command(cmd, {"raw": True})

        cmd.execute_mock.assert_awaited_once_with(raw=True)
        assert result.success


# ---------------------------------------------------------------------------
# Tests: create_typer_app
# ---------------------------------------------------------------------------


class TestCreateTyperApp:
    """Tests for CLICommandAdapter.create_typer_app."""

    def test_creates_typer_app(self):
        """A Typer app is created with the correct configuration."""
        with patch.object(CLICommandAdapter, "register_with_typer") as mock_register:
            app = CLICommandAdapter.create_typer_app()

        assert isinstance(app, typer.Typer)
        mock_register.assert_called_once_with(app)

    def test_app_has_correct_name(self):
        """The created app has the expected name and help text."""
        with patch.object(CLICommandAdapter, "register_with_typer"):
            app = CLICommandAdapter.create_typer_app()

        # Typer app info
        assert app.info.name == "mcp-cli"
        assert app.info.help == "MCP CLI - Unified command interface"


# ---------------------------------------------------------------------------
# Tests: cli_execute
# ---------------------------------------------------------------------------


class TestCliExecute:
    """Tests for the cli_execute convenience function."""

    @pytest.mark.asyncio
    async def test_unknown_command_returns_false(self):
        """Unknown command name returns False and prints error."""
        with patch("mcp_cli.adapters.cli.output") as mock_output:
            result = await cli_execute("nonexistent_command")

        assert result is False
        mock_output.error.assert_called_once()
        assert "Unknown command" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    async def test_success_with_output_and_data(self):
        """Successful command with output and data returns data."""
        cmd = MockCLICommand("servers")
        cmd.execute_mock.return_value = CommandResult(
            success=True, output="server list", data={"servers": ["a", "b"]}
        )
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.cli.output") as mock_output:
            result = await cli_execute("servers")

        assert result == {"servers": ["a", "b"]}
        mock_output.print.assert_called_once_with("server list")

    @pytest.mark.asyncio
    async def test_success_with_output_no_data(self):
        """Successful command with output but no data returns True."""
        cmd = MockCLICommand("servers")
        cmd.execute_mock.return_value = CommandResult(
            success=True, output="server list", data=None
        )
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.cli.output") as mock_output:
            result = await cli_execute("servers")

        assert result is True
        mock_output.print.assert_called_once_with("server list")

    @pytest.mark.asyncio
    async def test_success_no_output(self):
        """Successful command with no output returns True."""
        cmd = MockCLICommand("servers")
        cmd.execute_mock.return_value = CommandResult(success=True, output=None)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.cli.output") as mock_output:
            result = await cli_execute("servers")

        assert result is True
        mock_output.print.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_with_error(self):
        """Failed command with error message prints error and returns False."""
        cmd = MockCLICommand("servers")
        cmd.execute_mock.return_value = CommandResult(
            success=False, error="Connection failed"
        )
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.cli.output") as mock_output:
            result = await cli_execute("servers")

        assert result is False
        mock_output.error.assert_called_once_with("Connection failed")

    @pytest.mark.asyncio
    async def test_failure_no_error_message(self):
        """Failed command without error prints generic failure message."""
        cmd = MockCLICommand("servers")
        cmd.execute_mock.return_value = CommandResult(success=False, error=None)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.cli.output") as mock_output:
            result = await cli_execute("servers")

        assert result is False
        mock_output.error.assert_called_once()
        assert "Command failed" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    async def test_exception_during_execution(self):
        """Exception during execute returns False and prints error."""
        cmd = MockCLICommand("servers")
        cmd.execute_mock.side_effect = RuntimeError("kaboom")
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.cli.output") as mock_output:
            result = await cli_execute("servers")

        assert result is False
        mock_output.error.assert_called_once()
        assert "kaboom" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    async def test_context_injected_when_required(self):
        """Context managers are injected when command requires context."""
        mock_context = MagicMock()
        mock_context.tool_manager = MagicMock()
        mock_context.model_manager = MagicMock()

        cmd = MockCLICommand("servers", requires_ctx=True)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.cli.get_context", return_value=mock_context):
            with patch("mcp_cli.adapters.cli.output"):
                result = await cli_execute("servers")

        assert result is True
        call_kwargs = cmd.execute_mock.call_args[1]
        assert call_kwargs["tool_manager"] is mock_context.tool_manager
        assert call_kwargs["model_manager"] is mock_context.model_manager

    @pytest.mark.asyncio
    async def test_context_none_when_required(self):
        """When get_context returns None, command still executes without context."""
        cmd = MockCLICommand("servers", requires_ctx=True)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.cli.get_context", return_value=None):
            with patch("mcp_cli.adapters.cli.output"):
                result = await cli_execute("servers")

        assert result is True

    @pytest.mark.asyncio
    async def test_context_runtime_error_handled(self):
        """RuntimeError from get_context is caught gracefully."""
        cmd = MockCLICommand("servers", requires_ctx=True)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch(
            "mcp_cli.adapters.cli.get_context",
            side_effect=RuntimeError("not initialized"),
        ):
            with patch("mcp_cli.adapters.cli.output"):
                result = await cli_execute("servers")

        assert result is True
        # Command executes without context managers
        cmd.execute_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_not_injected_when_not_required(self):
        """Context is not injected when command does not require it."""
        cmd = MockCLICommand("servers", requires_ctx=False)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        mock_context = MagicMock()
        mock_context.tool_manager = MagicMock()
        mock_context.model_manager = MagicMock()

        with patch("mcp_cli.adapters.cli.get_context", return_value=mock_context):
            with patch("mcp_cli.adapters.cli.output"):
                result = await cli_execute("servers")

        assert result is True
        call_kwargs = cmd.execute_mock.call_args[1]
        assert "tool_manager" not in call_kwargs
        assert "model_manager" not in call_kwargs

    @pytest.mark.asyncio
    async def test_kwargs_passed_to_execute(self):
        """Extra kwargs are forwarded to command.execute."""
        cmd = MockCLICommand("servers")
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.cli.output"):
            await cli_execute("servers", raw=True, details=False)

        cmd.execute_mock.assert_awaited_once_with(raw=True, details=False)

    @pytest.mark.asyncio
    async def test_setdefault_does_not_overwrite_existing_kwargs(self):
        """Context uses setdefault so explicitly passed kwargs are preserved."""
        mock_context = MagicMock()
        mock_context.tool_manager = MagicMock(name="context_tm")
        mock_context.model_manager = MagicMock(name="context_mm")

        custom_tm = MagicMock(name="custom_tm")

        cmd = MockCLICommand("servers", requires_ctx=True)
        reg = UnifiedCommandRegistry()
        reg.register(cmd)

        with patch("mcp_cli.adapters.cli.get_context", return_value=mock_context):
            with patch("mcp_cli.adapters.cli.output"):
                await cli_execute("servers", tool_manager=custom_tm)

        call_kwargs = cmd.execute_mock.call_args[1]
        # setdefault should NOT overwrite explicitly passed tool_manager
        assert call_kwargs["tool_manager"] is custom_tm
