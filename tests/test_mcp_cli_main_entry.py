# tests/test_mcp_cli_main_entry.py
"""Tests for mcp_cli/__main__.py entry-point script."""

import pytest
import runpy
from unittest.mock import patch, MagicMock


class TestMainEntry:
    """Cover the if __name__ == '__main__' block via runpy.run_module."""

    def test_app_called_successfully(self):
        """Normal flow: app() is called and returns cleanly."""
        mock_app = MagicMock()
        with (
            patch.dict("sys.modules", {"mcp_cli.main": MagicMock(app=mock_app)}),
            patch("mcp_cli.config.PLATFORM_WINDOWS", "win32"),
        ):
            runpy.run_module("mcp_cli.__main__", run_name="__main__", alter_sys=False)
        mock_app.assert_called_once()

    def test_keyboard_interrupt_exits_1(self):
        """KeyboardInterrupt should print message and sys.exit(1)."""
        mock_app = MagicMock(side_effect=KeyboardInterrupt)
        with (
            patch.dict("sys.modules", {"mcp_cli.main": MagicMock(app=mock_app)}),
            patch("mcp_cli.config.PLATFORM_WINDOWS", "win32"),
            patch("builtins.print") as mock_print,
            pytest.raises(SystemExit) as exc_info,
        ):
            runpy.run_module("mcp_cli.__main__", run_name="__main__", alter_sys=False)
        assert exc_info.value.code == 1
        mock_print.assert_called_once()
        assert "Interrupted" in mock_print.call_args[0][0]

    def test_generic_exception_exits_1(self):
        """An arbitrary exception should print error and sys.exit(1)."""
        mock_app = MagicMock(side_effect=RuntimeError("boom"))
        with (
            patch.dict("sys.modules", {"mcp_cli.main": MagicMock(app=mock_app)}),
            patch("mcp_cli.config.PLATFORM_WINDOWS", "win32"),
            patch("builtins.print") as mock_print,
            pytest.raises(SystemExit) as exc_info,
        ):
            runpy.run_module("mcp_cli.__main__", run_name="__main__", alter_sys=False)
        assert exc_info.value.code == 1
        mock_print.assert_called_once()
        assert "boom" in mock_print.call_args[0][0]

    def test_windows_event_loop_policy(self):
        """On Windows, WindowsSelectorEventLoopPolicy should be set."""
        import asyncio

        mock_app = MagicMock()
        mock_policy_cls = MagicMock()

        # asyncio.WindowsSelectorEventLoopPolicy is a lazily-resolved deprecated
        # alias on non-Windows platforms (via module __getattr__), and on 3.14
        # that resolver raises NameError instead of AttributeError when accessed
        # off Windows. mock.patch(..., create=True) snapshots the original value
        # via getattr(target, name, DEFAULT), which doesn't catch NameError, so
        # it blows up before the patch is even applied. Set/restore it as a
        # plain attribute instead, sidestepping that getattr call entirely.
        had_attr = "WindowsSelectorEventLoopPolicy" in vars(asyncio)
        original = vars(asyncio).get("WindowsSelectorEventLoopPolicy")
        asyncio.WindowsSelectorEventLoopPolicy = mock_policy_cls
        try:
            with (
                patch.dict("sys.modules", {"mcp_cli.main": MagicMock(app=mock_app)}),
                patch("mcp_cli.config.PLATFORM_WINDOWS", "win32"),
                patch("sys.platform", "win32"),
                patch("asyncio.set_event_loop_policy") as mock_set_policy,
            ):
                runpy.run_module(
                    "mcp_cli.__main__", run_name="__main__", alter_sys=False
                )
            mock_set_policy.assert_called_once()
        finally:
            if had_attr:
                asyncio.WindowsSelectorEventLoopPolicy = original
            else:
                del asyncio.WindowsSelectorEventLoopPolicy

    def test_non_windows_no_policy_change(self):
        """On non-Windows, event loop policy should not be changed."""
        mock_app = MagicMock()
        with (
            patch.dict("sys.modules", {"mcp_cli.main": MagicMock(app=mock_app)}),
            patch("mcp_cli.config.PLATFORM_WINDOWS", "win32"),
            patch("sys.platform", "darwin"),
            patch("asyncio.set_event_loop_policy") as mock_set_policy,
        ):
            runpy.run_module("mcp_cli.__main__", run_name="__main__", alter_sys=False)
        mock_set_policy.assert_not_called()

    def test_not_run_as_main(self):
        """When __name__ != '__main__' the block should not execute."""
        mock_app = MagicMock()
        with patch.dict("sys.modules", {"mcp_cli.main": MagicMock(app=mock_app)}):
            # run_name defaults to the module's real __name__, not "__main__"
            runpy.run_module(
                "mcp_cli.__main__", run_name="mcp_cli.__main__", alter_sys=False
            )
        mock_app.assert_not_called()
