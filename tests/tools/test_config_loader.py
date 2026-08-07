# tests/tools/test_config_loader.py
"""Tests for MCP configuration loading."""

import json
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.tools.config_loader import ConfigLoader
from mcp_cli.config.server_models import HTTPServerConfig, STDIOServerConfig


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file."""
    config = {
        "mcpServers": {
            "http_server": {
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer token"},
            },
            "sse_server": {
                "url": "https://example.com/sse",
                "transport": "sse",
            },
            "stdio_server": {
                "command": "python",
                "args": ["-m", "server"],
                "env": {"DEBUG": "1"},
            },
        }
    }
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(json.dumps(config))
    return str(config_path)


@pytest.fixture
def token_config_file(tmp_path):
    """Create a config file with token placeholders."""
    config = {
        "mcpServers": {
            "oauth_server": {
                "url": "https://api.example.com",
                "headers": {"Authorization": "{{token:github}}"},
            }
        }
    }
    config_path = tmp_path / "token_config.json"
    config_path.write_text(json.dumps(config))
    return str(config_path)


# ----------------------------------------------------------------------------
# ConfigLoader initialization tests
# ----------------------------------------------------------------------------


def test_config_loader_init():
    """Test ConfigLoader initialization."""
    loader = ConfigLoader("config.json", ["server1", "server2"])

    assert loader.config_file == "config.json"
    assert loader.servers == ["server1", "server2"]
    assert loader.http_servers == []
    assert loader.sse_servers == []
    assert loader.stdio_servers == []


# ----------------------------------------------------------------------------
# Config loading tests
# ----------------------------------------------------------------------------


def test_load_config(temp_config_file):
    """Test loading a valid config file."""
    loader = ConfigLoader(temp_config_file, ["http_server"])

    config = loader.load()

    assert "mcpServers" in config
    assert "http_server" in config["mcpServers"]


def test_load_config_caches_result(temp_config_file):
    """Test config is cached after first load."""
    loader = ConfigLoader(temp_config_file, ["http_server"])

    config1 = loader.load()
    config2 = loader.load()

    assert config1 is config2


def test_load_config_file_not_found(caplog):
    """Test loading nonexistent config file returns empty dict."""
    loader = ConfigLoader("/nonexistent/config.json", [])

    with caplog.at_level(logging.INFO, logger="mcp_cli.tools.config_loader"):
        config = loader.load()

    assert config == {}
    assert any("Config file not found" in r.message for r in caplog.records)


def test_load_config_invalid_json(tmp_path):
    """Test loading invalid JSON file."""
    config_path = tmp_path / "invalid.json"
    config_path.write_text("not valid json {")

    loader = ConfigLoader(str(config_path), [])
    config = loader.load()

    assert config == {}


# ----------------------------------------------------------------------------
# Server type detection tests
# ----------------------------------------------------------------------------


def test_detect_server_types_http(temp_config_file):
    """Test detecting HTTP servers."""
    loader = ConfigLoader(temp_config_file, ["http_server"])
    config = loader.load()

    loader.detect_server_types(config)

    assert len(loader.http_servers) == 1
    assert len(loader.sse_servers) == 0
    assert len(loader.stdio_servers) == 0

    http_server = loader.http_servers[0]
    assert isinstance(http_server, HTTPServerConfig)
    assert http_server.name == "http_server"
    assert http_server.url == "https://example.com/mcp"
    assert http_server.headers == {"Authorization": "Bearer token"}


def test_detect_server_types_sse(temp_config_file):
    """Test detecting SSE servers."""
    loader = ConfigLoader(temp_config_file, ["sse_server"])
    config = loader.load()

    loader.detect_server_types(config)

    assert len(loader.http_servers) == 0
    assert len(loader.sse_servers) == 1
    assert len(loader.stdio_servers) == 0

    sse_server = loader.sse_servers[0]
    assert isinstance(sse_server, HTTPServerConfig)
    assert sse_server.name == "sse_server"


def test_detect_server_types_stdio(temp_config_file):
    """Test detecting STDIO servers."""
    loader = ConfigLoader(temp_config_file, ["stdio_server"])
    config = loader.load()

    loader.detect_server_types(config)

    assert len(loader.http_servers) == 0
    assert len(loader.sse_servers) == 0
    assert len(loader.stdio_servers) == 1

    stdio_server = loader.stdio_servers[0]
    assert isinstance(stdio_server, STDIOServerConfig)
    assert stdio_server.name == "stdio_server"
    assert stdio_server.command == "python"
    assert stdio_server.args == ["-m", "server"]
    assert stdio_server.env == {"DEBUG": "1"}


def test_detect_server_types_multiple(temp_config_file):
    """Test detecting multiple server types."""
    loader = ConfigLoader(
        temp_config_file, ["http_server", "sse_server", "stdio_server"]
    )
    config = loader.load()

    loader.detect_server_types(config)

    assert len(loader.http_servers) == 1
    assert len(loader.sse_servers) == 1
    assert len(loader.stdio_servers) == 1


def test_detect_server_types_unknown_server(temp_config_file):
    """Test handling unknown server names."""
    loader = ConfigLoader(temp_config_file, ["unknown_server"])
    config = loader.load()

    loader.detect_server_types(config)

    assert len(loader.http_servers) == 0
    assert len(loader.sse_servers) == 0
    assert len(loader.stdio_servers) == 0


def test_detect_server_types_clears_existing(temp_config_file):
    """Test detect_server_types clears previous results."""
    loader = ConfigLoader(temp_config_file, ["http_server"])
    config = loader.load()

    # First detection
    loader.detect_server_types(config)
    assert len(loader.http_servers) == 1

    # Second detection with different servers
    loader.servers = ["stdio_server"]
    loader.detect_server_types(config)

    assert len(loader.http_servers) == 0
    assert len(loader.stdio_servers) == 1


def test_detect_server_types_disabled_server(tmp_path):
    """Test detecting disabled servers."""
    config = {
        "mcpServers": {
            "disabled_server": {
                "url": "https://example.com",
                "disabled": True,
            }
        }
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    loader = ConfigLoader(str(config_path), ["disabled_server"])
    loaded = loader.load()
    loader.detect_server_types(loaded)

    assert len(loader.http_servers) == 1
    assert loader.http_servers[0].disabled is True


# ----------------------------------------------------------------------------
# Token placeholder resolution tests
# ----------------------------------------------------------------------------


def test_resolve_token_placeholder(token_config_file):
    """Test resolving token placeholders."""
    loader = ConfigLoader(token_config_file, ["oauth_server"])

    # Mock token store to return OAuth token data
    stored_token_json = json.dumps(
        {
            "token_type": "oauth",
            "name": "github",
            "data": {"access_token": "test_access_token"},
        }
    )

    with patch.object(
        loader._token_store, "_retrieve_raw", return_value=stored_token_json
    ):
        config = loader.load()

    assert (
        config["mcpServers"]["oauth_server"]["headers"]["Authorization"]
        == "Bearer test_access_token"
    )


def test_resolve_token_placeholder_no_token(token_config_file):
    """Test handling missing tokens."""
    loader = ConfigLoader(token_config_file, ["oauth_server"])

    # Mock token store to return None (no token found)
    with patch.object(loader._token_store, "_retrieve_raw", return_value=None):
        config = loader.load()

    # Should keep placeholder if no token
    assert (
        config["mcpServers"]["oauth_server"]["headers"]["Authorization"]
        == "{{token:github}}"
    )


def test_resolve_token_placeholder_nested(tmp_path):
    """Test resolving tokens in nested structures."""
    config = {
        "mcpServers": {"server": {"nested": {"deep": {"token": "{{token:provider}}"}}}}
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    loader = ConfigLoader(str(config_path), ["server"])

    # Mock token store to return OAuth token data
    stored_token_json = json.dumps(
        {
            "token_type": "oauth",
            "name": "provider",
            "data": {"access_token": "nested_token"},
        }
    )

    with patch.object(
        loader._token_store, "_retrieve_raw", return_value=stored_token_json
    ):
        loaded = loader.load()

    assert (
        loaded["mcpServers"]["server"]["nested"]["deep"]["token"]
        == "Bearer nested_token"
    )


def test_resolve_token_placeholder_in_list(tmp_path):
    """Test resolving tokens in list values."""
    config = {
        "mcpServers": {
            "server": {"tokens": ["{{token:provider1}}", "{{token:provider2}}"]}
        }
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    loader = ConfigLoader(str(config_path), ["server"])

    # Mock token store to return OAuth token data
    stored_token_json = json.dumps(
        {
            "token_type": "oauth",
            "name": "provider",
            "data": {"access_token": "list_token"},
        }
    )

    with patch.object(
        loader._token_store, "_retrieve_raw", return_value=stored_token_json
    ):
        loaded = loader.load()

    assert loaded["mcpServers"]["server"]["tokens"] == [
        "Bearer list_token",
        "Bearer list_token",
    ]


# ----------------------------------------------------------------------------
# OAuth refresh callback tests
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_oauth_refresh_callback_no_url():
    """Test refresh callback with no URL."""
    loader = ConfigLoader("config.json", [])

    callback = loader.create_oauth_refresh_callback([], [])
    result = await callback(server_url=None)

    assert result is None


@pytest.mark.asyncio
async def test_create_oauth_refresh_callback_unknown_url():
    """Test refresh callback with unknown URL."""
    loader = ConfigLoader("config.json", [])

    http_servers = [HTTPServerConfig(name="server1", url="https://known.com")]
    callback = loader.create_oauth_refresh_callback(http_servers, [])

    result = await callback(server_url="https://unknown.com")

    assert result is None


@pytest.mark.asyncio
async def test_create_oauth_refresh_callback_maps_url():
    """Test refresh callback correctly maps URL to server."""
    loader = ConfigLoader("config.json", [])

    http_servers = [
        HTTPServerConfig(name="test_server", url="https://api.example.com/mcp")
    ]

    callback = loader.create_oauth_refresh_callback(http_servers, [])

    # Mock TokenStoreFactory to return a mock store with no token
    with patch("mcp_cli.tools.config_loader.TokenStoreFactory") as MockFactory:
        mock_store = MagicMock()
        mock_store._retrieve_raw.return_value = None
        MockFactory.create.return_value = mock_store
        result = await callback(server_url="https://api.example.com/mcp")

    assert result is None  # No token found


@pytest.mark.asyncio
async def test_create_oauth_refresh_callback_no_refresh_token():
    """Test callback when no refresh token is available."""
    loader = ConfigLoader("config.json", [])

    http_servers = [HTTPServerConfig(name="test_server", url="https://api.example.com")]

    callback = loader.create_oauth_refresh_callback(http_servers, [])

    # Return token data without refresh_token
    stored_token_json = json.dumps(
        {
            "token_type": "oauth",
            "name": "test_server",
            "data": {"access_token": "old_token"},  # No refresh_token
        }
    )

    with patch("mcp_cli.tools.config_loader.TokenStoreFactory") as MockFactory:
        mock_store = MagicMock()
        mock_store._retrieve_raw.return_value = stored_token_json
        MockFactory.create.return_value = mock_store

        result = await callback(server_url="https://api.example.com")

    assert result is None


# ----------------------------------------------------------------------------
# Additional coverage tests
# ----------------------------------------------------------------------------


def test_load_config_general_exception(tmp_path):
    """Test handling general exception during config load."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"mcpServers": {}}')

    loader = ConfigLoader(str(config_path), [])

    # Mock json.load to raise a generic exception
    with patch("builtins.open", side_effect=PermissionError("access denied")):
        config = loader.load()

    assert config == {}


def test_resolve_token_placeholder_exception(tmp_path):
    """Test handling exception during token resolution."""
    config = {
        "mcpServers": {
            "server": {"headers": {"Authorization": "{{token:failing_provider}}"}}
        }
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    loader = ConfigLoader(str(config_path), ["server"])

    # Make _retrieve_raw raise an exception
    with patch.object(
        loader._token_store, "_retrieve_raw", side_effect=RuntimeError("token error")
    ):
        loaded = loader.load()

    # Should keep the placeholder on error
    assert (
        loaded["mcpServers"]["server"]["headers"]["Authorization"]
        == "{{token:failing_provider}}"
    )


@pytest.mark.asyncio
async def test_create_oauth_refresh_callback_success():
    """Test successful OAuth token refresh."""
    loader = ConfigLoader("config.json", [])

    http_servers = [HTTPServerConfig(name="test_server", url="https://api.example.com")]

    callback = loader.create_oauth_refresh_callback(http_servers, [])

    # Mock the full refresh flow
    stored_token_json = json.dumps(
        {
            "token_type": "oauth",
            "name": "test_server",
            "data": {
                "access_token": "old_token",
                "refresh_token": "refresh_token_value",
            },
        }
    )

    with (
        patch("mcp_cli.tools.config_loader.TokenStoreFactory") as MockFactory,
        patch("mcp_cli.tools.config_loader.OAuthHandler") as MockOAuth,
    ):
        mock_store = MagicMock()
        mock_store._retrieve_raw.return_value = stored_token_json
        MockFactory.create.return_value = mock_store

        mock_oauth = MockOAuth.return_value
        mock_oauth.refresh_access_token = AsyncMock(
            return_value={
                "access_token": "new_access_token",
                "refresh_token": "new_refresh_token",
            }
        )

        result = await callback(server_url="https://api.example.com")

    assert result == {"Authorization": "Bearer new_access_token"}
    mock_store._store_raw.assert_called_once()


@pytest.mark.asyncio
async def test_create_oauth_refresh_callback_refresh_fails():
    """Test OAuth token refresh when refresh returns empty."""
    loader = ConfigLoader("config.json", [])

    http_servers = [HTTPServerConfig(name="test_server", url="https://api.example.com")]

    callback = loader.create_oauth_refresh_callback(http_servers, [])

    stored_token_json = json.dumps(
        {
            "token_type": "oauth",
            "name": "test_server",
            "data": {
                "access_token": "old_token",
                "refresh_token": "refresh_token_value",
            },
        }
    )

    with (
        patch("mcp_cli.tools.config_loader.TokenStoreFactory") as MockFactory,
        patch("mcp_cli.tools.config_loader.OAuthHandler") as MockOAuth,
    ):
        mock_store = MagicMock()
        mock_store._retrieve_raw.return_value = stored_token_json
        MockFactory.create.return_value = mock_store

        mock_oauth = MockOAuth.return_value
        mock_oauth.refresh_access_token = AsyncMock(return_value=None)

        result = await callback(server_url="https://api.example.com")

    assert result is None


@pytest.mark.asyncio
async def test_create_oauth_refresh_callback_exception():
    """Test OAuth token refresh when exception is raised."""
    loader = ConfigLoader("config.json", [])

    http_servers = [HTTPServerConfig(name="test_server", url="https://api.example.com")]

    callback = loader.create_oauth_refresh_callback(http_servers, [])

    stored_token_json = json.dumps(
        {
            "token_type": "oauth",
            "name": "test_server",
            "data": {
                "access_token": "old_token",
                "refresh_token": "refresh_token_value",
            },
        }
    )

    with (
        patch("mcp_cli.tools.config_loader.TokenStoreFactory") as MockFactory,
        patch("mcp_cli.tools.config_loader.OAuthHandler") as MockOAuth,
    ):
        mock_store = MagicMock()
        mock_store._retrieve_raw.return_value = stored_token_json
        MockFactory.create.return_value = mock_store

        mock_oauth = MockOAuth.return_value
        mock_oauth.refresh_access_token = AsyncMock(
            side_effect=RuntimeError("network error")
        )

        result = await callback(server_url="https://api.example.com")

    assert result is None


@pytest.mark.asyncio
async def test_create_oauth_refresh_callback_sse_server():
    """Test OAuth refresh callback finds server in SSE list."""
    loader = ConfigLoader("config.json", [])

    sse_servers = [HTTPServerConfig(name="sse_server", url="https://sse.example.com")]

    callback = loader.create_oauth_refresh_callback([], sse_servers)

    with patch("mcp_cli.tools.config_loader.TokenStoreFactory") as MockFactory:
        mock_store = MagicMock()
        mock_store._retrieve_raw.return_value = None
        MockFactory.create.return_value = mock_store

        result = await callback(server_url="https://sse.example.com")

    assert result is None  # No token found, but server was mapped


# ----------------------------------------------------------------------------
# Async loading tests (load_async method)
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_async_success(temp_config_file):
    """Test successful async config loading."""
    loader = ConfigLoader(temp_config_file, ["http_server"])

    config = await loader.load_async()

    assert "mcpServers" in config
    assert "http_server" in config["mcpServers"]


@pytest.mark.asyncio
async def test_load_async_caches_result(temp_config_file):
    """Test async load caches result."""
    loader = ConfigLoader(temp_config_file, ["http_server"])

    config1 = await loader.load_async()
    config2 = await loader.load_async()

    assert config1 is config2


@pytest.mark.asyncio
async def test_load_async_file_not_found(caplog):
    """Test async loading nonexistent config file returns empty dict."""
    loader = ConfigLoader("/nonexistent/config.json", [])

    with caplog.at_level(logging.INFO, logger="mcp_cli.tools.config_loader"):
        config = await loader.load_async()

    assert config == {}
    assert any("Config file not found" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_load_async_invalid_json(tmp_path):
    """Test async loading invalid JSON file."""
    config_path = tmp_path / "invalid.json"
    config_path.write_text("not valid json {")

    loader = ConfigLoader(str(config_path), [])
    config = await loader.load_async()

    assert config == {}


@pytest.mark.asyncio
async def test_load_async_general_exception(tmp_path):
    """Test async handling general exception during config load."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"mcpServers": {}}')

    loader = ConfigLoader(str(config_path), [])

    # Mock asyncio.to_thread to raise a generic exception
    with patch("asyncio.to_thread", side_effect=PermissionError("access denied")):
        config = await loader.load_async()

    assert config == {}


@pytest.mark.asyncio
async def test_load_async_resolves_tokens(token_config_file):
    """Test async loading resolves token placeholders."""
    loader = ConfigLoader(token_config_file, ["oauth_server"])

    # Mock token store to return OAuth token data
    stored_token_json = json.dumps(
        {
            "token_type": "oauth",
            "name": "github",
            "data": {"access_token": "async_token"},
        }
    )

    with patch.object(
        loader._token_store, "_retrieve_raw", return_value=stored_token_json
    ):
        config = await loader.load_async()

    assert (
        config["mcpServers"]["oauth_server"]["headers"]["Authorization"]
        == "Bearer async_token"
    )
