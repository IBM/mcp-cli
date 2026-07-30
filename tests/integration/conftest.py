# tests/integration/conftest.py
"""Fixtures for integration tests.

All tests in this directory require the @pytest.mark.integration marker
and are skipped by default. Run with:

    uv run pytest -m integration tests/integration/
"""

from __future__ import annotations

import pytest

from mcp_cli.tools.manager import ToolManager


@pytest.fixture
async def tool_manager_sqlite(tmp_path):
    """Create a ToolManager connected to a real SQLite MCP server.

    Requires 'sqlite' server defined in server_config.json.
    Yields the initialized ToolManager, then closes it.
    """
    tm = ToolManager(
        config_file="server_config.json",
        servers=["sqlite"],
        initialization_timeout=30.0,
    )
    ok = await tm.initialize()
    # ToolManager.initialize() can report success even when the external server
    # failed to start (e.g. `uvx mcp-server-sqlite` is unavailable or resolves a
    # version incompatible with the installed MCP SDK), leaving zero tools
    # discovered. Skip rather than fail so a broken/missing external server can
    # never turn a hermetic run red.
    tools = await tm.get_unique_tools() if ok else []
    if not ok or not tools:
        await tm.close()
        pytest.skip("sqlite MCP server unavailable (no tools discovered)")
    try:
        yield tm
    finally:
        await tm.close()
