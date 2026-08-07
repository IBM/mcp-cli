# tests/config/test_config_manager.py
"""
Comprehensive tests for config/config_manager.py module.
Target: 90%+ coverage
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from mcp_cli.config.config_manager import (
    ServerConfig,
    MCPConfig,
    ConfigManager,
    get_config,
    initialize_config,
    detect_server_types,
    validate_server_config,
)


class TestServerConfig:
    """Test ServerConfig Pydantic model."""

    def test_server_config_minimal(self):
        """Test ServerConfig with minimal fields."""
        config = ServerConfig(name="test-server")
        assert config.name == "test-server"
        assert config.command is None
        assert config.args == []
        assert config.env == {}
        assert config.url is None
        assert config.oauth is None
        assert config.disabled is False

    def test_server_config_with_command(self):
        """Test ServerConfig with command transport."""
        config = ServerConfig(
            name="stdio-server",
            command="python",
            args=["-m", "server"],
            env={"KEY": "value"},
        )
        assert config.name == "stdio-server"
        assert config.command == "python"
        assert config.args == ["-m", "server"]
        assert config.env == {"KEY": "value"}
        assert config.transport == "stdio"

    def test_server_config_with_url(self):
        """Test ServerConfig with HTTP transport."""
        config = ServerConfig(
            name="http-server",
            url="http://localhost:8000",
        )
        assert config.name == "http-server"
        assert config.url == "http://localhost:8000"
        assert config.transport == "http"

    def test_server_config_transport_unknown(self):
        """Test ServerConfig transport returns unknown when neither command nor url."""
        config = ServerConfig(name="unknown-server")
        assert config.transport == "unknown"

    def test_server_config_from_dict_basic(self):
        """Test ServerConfig.from_dict with basic data."""
        data = {
            "command": "uvx",
            "args": ["mcp-server-time"],
            "env": {"TZ": "UTC"},
        }
        config = ServerConfig.from_dict("time-server", data)
        assert config.name == "time-server"
        assert config.command == "uvx"
        assert config.args == ["mcp-server-time"]
        assert "TZ" in config.env
        assert "PATH" in config.env  # PATH is auto-inherited

    def test_server_config_from_dict_with_oauth(self):
        """Test ServerConfig.from_dict with OAuth configuration."""
        data = {
            "command": "node",
            "args": ["server.js"],
            "oauth": {
                "client_id": "test-client",
                "authorization_url": "https://auth.example.com/authorize",
                "token_url": "https://auth.example.com/token",
                "scopes": ["read", "write"],
            },
        }
        config = ServerConfig.from_dict("oauth-server", data)
        assert config.name == "oauth-server"
        assert config.oauth is not None
        assert config.oauth.client_id == "test-client"
        assert config.oauth.scopes == ["read", "write"]

    def test_server_config_from_dict_disabled(self):
        """Test ServerConfig.from_dict with disabled flag."""
        data = {"command": "cmd", "disabled": True}
        config = ServerConfig.from_dict("disabled-server", data)
        assert config.disabled is True

    def test_server_config_to_server_info(self):
        """Test ServerConfig.to_server_info conversion."""
        config = ServerConfig(
            name="test-server",
            command="python",
            args=["server.py"],
            env={"ENV": "test"},
        )
        server_info = config.to_server_info(server_id=42)

        assert server_info.id == 42
        assert server_info.name == "test-server"
        assert server_info.namespace == "test-server"
        assert server_info.command == "python"
        assert server_info.args == ["server.py"]
        assert server_info.env == {"ENV": "test"}
        assert server_info.enabled is True
        assert server_info.connected is False
        assert server_info.transport == "stdio"

    def test_server_config_to_server_info_disabled(self):
        """Test ServerConfig.to_server_info with disabled server."""
        config = ServerConfig(name="disabled", command="cmd", disabled=True)
        server_info = config.to_server_info()
        assert server_info.enabled is False


class TestMCPConfig:
    """Test MCPConfig Pydantic model."""

    def test_mcp_config_defaults(self):
        """Test MCPConfig default values."""
        config = MCPConfig()
        assert config.servers == {}
        assert config.default_provider == "ollama"
        assert config.default_model == "gpt-oss"
        assert config.theme == "default"
        assert config.verbose is True
        assert config.confirm_tools is True
        assert config.token_store_backend == "auto"

    def test_mcp_config_load_from_file_nonexistent(self, tmp_path):
        """Test loading from non-existent file returns empty config."""
        config = MCPConfig.load_from_file(tmp_path / "nonexistent.json")
        assert config.servers == {}
        assert config.default_provider == "openai"

    def test_mcp_config_load_from_file_with_servers(self, tmp_path):
        """Test loading config file with servers."""
        config_data = {
            "mcpServers": {
                "server1": {"command": "cmd1"},
                "server2": {"command": "cmd2", "disabled": True},
            },
            "defaultProvider": "anthropic",
            "defaultModel": "claude-3",
            "theme": "dark",
            "verbose": False,
            "confirmTools": False,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)

        assert len(config.servers) == 2
        assert "server1" in config.servers
        assert "server2" in config.servers
        assert config.servers["server2"].disabled is True
        assert config.default_provider == "anthropic"
        assert config.default_model == "claude-3"
        assert config.theme == "dark"
        assert config.verbose is False
        assert config.confirm_tools is False

    def test_mcp_config_load_from_file_with_token_storage(self, tmp_path):
        """Test loading config file with token storage configuration."""
        config_data = {
            "mcpServers": {},
            "tokenStorage": {
                "backend": "vault",
                "vaultUrl": "https://vault.example.com",
                "vaultToken": "secret-token",
                "vaultMountPoint": "kv",
                "vaultPathPrefix": "mcp/tokens",
                "vaultNamespace": "prod",
            },
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)

        assert config.token_store_backend == "vault"
        assert config.vault_url == "https://vault.example.com"
        assert config.vault_token == "secret-token"
        assert config.vault_mount_point == "kv"
        assert config.vault_path_prefix == "mcp/tokens"
        assert config.vault_namespace == "prod"

    def test_mcp_config_save_to_file(self, tmp_path):
        """Test saving config to file."""
        config = MCPConfig()
        config.servers["test"] = ServerConfig(name="test", command="cmd")
        config.default_provider = "ollama"
        config.default_model = "llama2"

        config_file = tmp_path / "saved_config.json"
        config.save_to_file(config_file)

        assert config_file.exists()

        # Reload and verify
        with open(config_file) as f:
            data = json.load(f)

        assert "mcpServers" in data
        assert "test" in data["mcpServers"]
        assert data["defaultProvider"] == "ollama"
        assert data["defaultModel"] == "llama2"

    def test_mcp_config_save_with_token_storage(self, tmp_path):
        """Test saving config with token storage configuration."""
        config = MCPConfig()
        config.token_store_backend = "vault"
        config.vault_url = "https://vault.test.com"
        config.vault_token = "token123"

        config_file = tmp_path / "config_with_vault.json"
        config.save_to_file(config_file)

        with open(config_file) as f:
            data = json.load(f)

        assert "tokenStorage" in data
        assert data["tokenStorage"]["backend"] == "vault"
        assert data["tokenStorage"]["vaultUrl"] == "https://vault.test.com"

    def test_mcp_config_get_server(self):
        """Test get_server method."""
        config = MCPConfig()
        config.servers["test"] = ServerConfig(name="test", command="cmd")

        server = config.get_server("test")
        assert server is not None
        assert server.name == "test"

        assert config.get_server("nonexistent") is None

    def test_mcp_config_add_server(self):
        """Test add_server method."""
        config = MCPConfig()
        server = ServerConfig(name="new-server", command="new-cmd")

        config.add_server(server)

        assert "new-server" in config.servers
        assert config.servers["new-server"].command == "new-cmd"

    def test_mcp_config_remove_server(self):
        """Test remove_server method."""
        config = MCPConfig()
        config.servers["test"] = ServerConfig(name="test", command="cmd")

        result = config.remove_server("test")
        assert result is True
        assert "test" not in config.servers

        result = config.remove_server("nonexistent")
        assert result is False

    def test_mcp_config_list_servers(self):
        """Test list_servers method."""
        config = MCPConfig()
        config.servers["server1"] = ServerConfig(name="server1", command="cmd1")
        config.servers["server2"] = ServerConfig(name="server2", command="cmd2")

        servers = config.list_servers()
        assert len(servers) == 2
        assert any(s.name == "server1" for s in servers)
        assert any(s.name == "server2" for s in servers)

    def test_mcp_config_list_enabled_servers(self):
        """Test list_enabled_servers method."""
        config = MCPConfig()
        config.servers["enabled"] = ServerConfig(name="enabled", command="cmd1")
        config.servers["disabled"] = ServerConfig(
            name="disabled", command="cmd2", disabled=True
        )

        enabled_servers = config.list_enabled_servers()
        assert len(enabled_servers) == 1
        assert enabled_servers[0].name == "enabled"


class TestConfigManager:
    """Test ConfigManager singleton."""

    def test_config_manager_singleton(self):
        """Test ConfigManager follows singleton pattern."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2

    def test_config_manager_initialize(self, tmp_path):
        """Test ConfigManager initialize method."""
        config_data = {"mcpServers": {"test": {"command": "cmd"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        manager = ConfigManager()
        manager.reset()  # Reset singleton state

        config = manager.initialize(config_file)
        assert config is not None
        assert "test" in config.servers

    def test_config_manager_get_config_not_initialized(self):
        """Test get_config raises error when not initialized."""
        manager = ConfigManager()
        manager.reset()

        with pytest.raises(RuntimeError, match="Config not initialized"):
            manager.get_config()

    def test_config_manager_get_config_after_init(self, tmp_path):
        """Test get_config returns config after initialization."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"mcpServers": {}}))

        manager = ConfigManager()
        manager.reset()
        manager.initialize(config_file)

        config = manager.get_config()
        assert config is not None

    def test_config_manager_save(self, tmp_path):
        """Test ConfigManager save method."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"mcpServers": {}}))

        manager = ConfigManager()
        manager.reset()
        manager.initialize(config_file)

        config = manager.get_config()
        config.servers["new"] = ServerConfig(name="new", command="new-cmd")

        manager.save()

        # Reload and verify
        with open(config_file) as f:
            data = json.load(f)
        assert "new" in data["mcpServers"]

    def test_config_manager_reload(self, tmp_path):
        """Test ConfigManager reload method."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"mcpServers": {"test": {"command": "cmd"}}}))

        manager = ConfigManager()
        manager.reset()
        manager.initialize(config_file)

        # Modify file externally
        config_file.write_text(
            json.dumps({"mcpServers": {"updated": {"command": "new-cmd"}}})
        )

        config = manager.reload()
        assert "updated" in config.servers
        assert "test" not in config.servers

    def test_config_manager_reload_without_path(self):
        """Test reload raises error when no path set."""
        manager = ConfigManager()
        manager.reset()

        with pytest.raises(RuntimeError, match="No config path set"):
            manager.reload()

    def test_config_manager_reset(self, tmp_path):
        """Test ConfigManager reset method."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"mcpServers": {}}))

        manager = ConfigManager()
        manager.reset()
        manager.initialize(config_file)

        manager.reset()

        # After reset, should raise error
        with pytest.raises(RuntimeError):
            manager.get_config()


class TestConfigHelpers:
    """Test module-level helper functions."""

    def test_get_config_function(self, tmp_path):
        """Test get_config module function."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"mcpServers": {}}))

        # Reset and initialize
        manager = ConfigManager()
        manager.reset()
        initialize_config(config_file)

        config = get_config()
        assert config is not None

    def test_initialize_config_function(self, tmp_path):
        """Test initialize_config module function."""
        config_data = {"mcpServers": {"func-test": {"command": "cmd"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        # Reset first
        manager = ConfigManager()
        manager.reset()

        config = initialize_config(config_file)
        assert config is not None
        assert "func-test" in config.servers

    def test_initialize_config_default_path(self):
        """Test initialize_config with default path."""
        manager = ConfigManager()
        manager.reset()

        # Should not raise, even if file doesn't exist
        config = initialize_config()
        assert config is not None


class TestDetectServerTypes:
    """Test detect_server_types function."""

    def test_detect_http_server(self, tmp_path):
        """Test detecting HTTP server."""
        config_data = {"mcpServers": {"http-server": {"url": "http://localhost:8080"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)
        http_servers, stdio_servers = detect_server_types(config, ["http-server"])

        assert len(http_servers) == 1
        assert http_servers[0]["name"] == "http-server"
        assert http_servers[0]["url"] == "http://localhost:8080"
        assert len(stdio_servers) == 0

    def test_detect_stdio_server(self, tmp_path):
        """Test detecting STDIO server."""
        config_data = {
            "mcpServers": {
                "stdio-server": {"command": "python", "args": ["-m", "server"]}
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)
        http_servers, stdio_servers = detect_server_types(config, ["stdio-server"])

        assert len(http_servers) == 0
        assert len(stdio_servers) == 1
        assert stdio_servers[0] == "stdio-server"

    def test_detect_mixed_servers(self, tmp_path):
        """Test detecting mixed HTTP and STDIO servers."""
        config_data = {
            "mcpServers": {
                "http-server": {"url": "http://localhost:8080"},
                "stdio-server": {"command": "python"},
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)
        http_servers, stdio_servers = detect_server_types(
            config, ["http-server", "stdio-server"]
        )

        assert len(http_servers) == 1
        assert len(stdio_servers) == 1

    def test_detect_server_not_found(self, tmp_path):
        """Test detecting server that doesn't exist in config."""
        config_data = {"mcpServers": {"real-server": {"command": "python"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)
        http_servers, stdio_servers = detect_server_types(config, ["missing-server"])

        # Missing server should be assumed STDIO
        assert len(http_servers) == 0
        assert len(stdio_servers) == 1
        assert stdio_servers[0] == "missing-server"

    def test_detect_server_unclear_config(self, tmp_path):
        """Test detecting server with unclear configuration."""
        config_data = {
            "mcpServers": {
                "unclear-server": {"param": "value"}  # No url or command
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)
        http_servers, stdio_servers = detect_server_types(config, ["unclear-server"])

        # Unclear config should default to STDIO
        assert len(http_servers) == 0
        assert len(stdio_servers) == 1

    def test_detect_empty_config(self):
        """Test detecting with empty config."""
        config = MCPConfig()
        http_servers, stdio_servers = detect_server_types(config, ["any-server"])

        # Should assume all STDIO when no config
        assert len(http_servers) == 0
        assert len(stdio_servers) == 1

    def test_detect_none_config(self):
        """Test detecting with None config."""
        http_servers, stdio_servers = detect_server_types(None, ["any-server"])

        # Should assume all STDIO when None config
        assert len(http_servers) == 0
        assert len(stdio_servers) == 1


class TestValidateServerConfig:
    """Test validate_server_config function."""

    def test_validate_valid_stdio_server(self, tmp_path):
        """Test validating valid STDIO server."""
        config_data = {
            "mcpServers": {
                "stdio-server": {"command": "python", "args": ["-m", "server"]}
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)
        is_valid, errors = validate_server_config(config, ["stdio-server"])

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_valid_http_server(self, tmp_path):
        """Test validating valid HTTP server."""
        config_data = {"mcpServers": {"http-server": {"url": "https://localhost:8080"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)
        is_valid, errors = validate_server_config(config, ["http-server"])

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_server_missing_both(self, tmp_path):
        """Test validating server missing both url and command."""
        config_data = {
            "mcpServers": {
                "invalid-server": {"param": "value"}  # No url or command
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)
        is_valid, errors = validate_server_config(config, ["invalid-server"])

        assert is_valid is False
        assert len(errors) == 1
        assert "missing both" in errors[0].lower()

    def test_validate_server_has_both(self, tmp_path):
        """Test validating server with both url and command."""
        config_data = {
            "mcpServers": {
                "both-server": {"url": "http://localhost:8080", "command": "python"}
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)
        is_valid, errors = validate_server_config(config, ["both-server"])

        assert is_valid is False
        assert len(errors) == 1
        assert "both" in errors[0].lower()

    def test_validate_invalid_url_format(self, tmp_path):
        """Test validating server with invalid URL format."""
        config_data = {
            "mcpServers": {
                "bad-url-server": {"url": "ftp://localhost:8080"}  # Not http/https
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)
        is_valid, errors = validate_server_config(config, ["bad-url-server"])

        assert is_valid is False
        assert len(errors) == 1
        assert "http://" in errors[0] or "https://" in errors[0]

    def test_validate_empty_command(self, tmp_path):
        """Test validating server with empty command."""
        config_data = {
            "mcpServers": {
                "empty-cmd-server": {"command": "   "}  # Empty command
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)
        is_valid, errors = validate_server_config(config, ["empty-cmd-server"])

        assert is_valid is False
        assert len(errors) == 1
        assert "non-empty string" in errors[0].lower()

    def test_validate_server_not_found(self, tmp_path):
        """Test validating server that doesn't exist."""
        config_data = {"mcpServers": {"real-server": {"command": "python"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)
        is_valid, errors = validate_server_config(config, ["missing-server"])

        assert is_valid is False
        assert len(errors) == 1
        assert "not found" in errors[0].lower()

    def test_validate_empty_config(self):
        """Test validating with empty config."""
        config = MCPConfig()
        is_valid, errors = validate_server_config(config, ["any-server"])

        assert is_valid is False
        assert len(errors) == 1
        assert "no servers" in errors[0].lower()

    def test_validate_none_config(self):
        """Test validating with None config."""
        is_valid, errors = validate_server_config(None, ["any-server"])

        assert is_valid is False
        assert len(errors) == 1

    def test_validate_multiple_errors(self, tmp_path):
        """Test validating with multiple errors."""
        config_data = {
            "mcpServers": {
                "bad1": {"param": "value"},  # Missing both
                "bad2": {"url": "http://localhost", "command": "python"},  # Has both
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)
        is_valid, errors = validate_server_config(config, ["bad1", "bad2"])

        assert is_valid is False
        assert len(errors) == 2


class TestMCPConfigLoadFromFileTimeouts:
    """Test loading timeouts from config file."""

    def test_load_with_timeouts(self, tmp_path):
        """Test loading config with timeout configuration."""
        # LegacyMCPConfig parses timeouts section and creates TimeoutConfig
        config_data = {
            "mcpServers": {},
            "timeouts": {
                "streamingChunkTimeout": 60.0,
                "streamingGlobalTimeout": 600.0,
                "streamingFirstChunkTimeout": 90.0,
                "toolExecutionTimeout": 180.0,
                "serverInitTimeout": 60.0,
                "httpRequestTimeout": 45.0,
                "httpConnectTimeout": 15.0,
            },
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)

        # Verify timeouts object exists and has positive values
        assert config.timeouts is not None
        assert config.timeouts.streaming_chunk > 0
        assert config.timeouts.streaming_global > 0

    def test_load_with_tools_config(self, tmp_path):
        """Test loading config with tool configuration."""
        config_data = {
            "mcpServers": {},
            "tools": {
                "includeTools": ["tool1", "tool2"],
                "excludeTools": ["bad_tool"],
                "dynamicToolsEnabled": False,
                "confirmTools": True,
                "maxConcurrency": 5,
            },
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)

        assert config.tools.include_tools == ["tool1", "tool2"]
        assert config.tools.exclude_tools == ["bad_tool"]
        assert config.tools.dynamic_tools_enabled is False
        assert config.tools.confirm_tools is True
        assert config.tools.max_concurrency == 5

    def test_load_with_vault_non_defaults(self, tmp_path):
        """Test saving config with non-default vault settings."""
        config = MCPConfig()
        config.vault_mount_point = "kv"
        config.vault_path_prefix = "custom/path"
        config.vault_namespace = "prod"

        config_file = tmp_path / "vault_config.json"
        config.save_to_file(config_file)

        with open(config_file) as f:
            data = json.load(f)

        assert "tokenStorage" in data
        assert data["tokenStorage"]["vaultMountPoint"] == "kv"
        assert data["tokenStorage"]["vaultPathPrefix"] == "custom/path"
        assert data["tokenStorage"]["vaultNamespace"] == "prod"

    def test_load_with_error_returns_empty_config(self, tmp_path):
        """Test that loading invalid config returns default config."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ invalid json }")

        config = MCPConfig.load_from_file(config_file)
        # Should return default config rather than crashing
        assert config is not None
        assert config.servers == {}


class TestConfigManagerPackageFallback:
    """Test ConfigManager initialization behavior."""

    def test_initialize_without_path_no_cwd_file(self, tmp_path, monkeypatch):
        """Test initialize without path when no server_config.json in cwd."""
        monkeypatch.chdir(tmp_path)

        manager = ConfigManager()
        manager.reset()

        config = manager.initialize()
        # Returns empty config (no bundled fallback)
        assert config is not None
        assert config.servers == {}

    def test_initialize_with_cwd_file(self, tmp_path, monkeypatch):
        """Test that cwd config is loaded."""
        config_data = {"mcpServers": {"local-server": {"command": "local-cmd"}}}
        config_file = tmp_path / "server_config.json"
        config_file.write_text(json.dumps(config_data))

        monkeypatch.chdir(tmp_path)

        manager = ConfigManager()
        manager.reset()

        config = manager.initialize()
        assert "local-server" in config.servers


class TestRuntimeConfigOld:
    """Test legacy RuntimeConfig in config_manager.py."""

    def test_runtime_config_creation(self):
        """Test creating RuntimeConfig."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        mcp_config = MCPConfig()
        runtime = LegacyRuntimeConfig(mcp_config)

        assert runtime.mcp_config == mcp_config
        assert runtime.cli_overrides == {}

    def test_runtime_config_with_cli_overrides(self):
        """Test RuntimeConfig with CLI overrides."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        mcp_config = MCPConfig()
        cli_overrides = {"streaming_chunk_timeout": 60.0}
        runtime = LegacyRuntimeConfig(mcp_config, cli_overrides)

        assert runtime.cli_overrides == cli_overrides

    def test_get_timeout_from_cli(self):
        """Test getting timeout from CLI overrides."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        mcp_config = MCPConfig()
        cli_overrides = {"streaming_chunk_timeout": 75.0}
        runtime = LegacyRuntimeConfig(mcp_config, cli_overrides)

        timeout = runtime.get_timeout("streaming_chunk")
        assert timeout == 75.0

    def test_get_timeout_from_env(self, monkeypatch):
        """Test getting timeout from environment variable."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        monkeypatch.setenv("MCP_STREAMING_CHUNK_TIMEOUT", "88.0")

        mcp_config = MCPConfig()
        runtime = LegacyRuntimeConfig(mcp_config)

        timeout = runtime.get_timeout("streaming_chunk")
        assert timeout == 88.0

    def test_get_timeout_from_tool_timeout_env(self, monkeypatch):
        """Test MCP_TOOL_TIMEOUT applies to multiple timeouts."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        monkeypatch.setenv("MCP_TOOL_TIMEOUT", "150.0")

        mcp_config = MCPConfig()
        runtime = LegacyRuntimeConfig(mcp_config)

        assert runtime.get_timeout("streaming_chunk") == 150.0
        assert runtime.get_timeout("streaming_global") == 150.0
        assert runtime.get_timeout("tool_execution") == 150.0

    def test_get_timeout_invalid_env_value(self, monkeypatch):
        """Test invalid env value falls back to config."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        monkeypatch.setenv("MCP_STREAMING_CHUNK_TIMEOUT", "not_a_number")

        mcp_config = MCPConfig()
        runtime = LegacyRuntimeConfig(mcp_config)

        # Should not crash, should get config value
        timeout = runtime.get_timeout("streaming_chunk")
        assert timeout > 0

    def test_get_timeout_fallback(self):
        """Test timeout fallback when not found anywhere."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        mcp_config = MCPConfig()
        runtime = LegacyRuntimeConfig(mcp_config)

        # Non-existent timeout should fall back to 120.0
        timeout = runtime.get_timeout("nonexistent_timeout")
        assert timeout == 120.0

    def test_get_tool_config_value_from_cli(self):
        """Test getting tool config from CLI overrides."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        mcp_config = MCPConfig()
        cli_overrides = {"include_tools": ["tool1", "tool2"]}
        runtime = LegacyRuntimeConfig(mcp_config, cli_overrides)

        tools = runtime.get_tool_config_value("include_tools")
        assert tools == ["tool1", "tool2"]

    def test_get_tool_config_value_from_env_list(self, monkeypatch):
        """Test getting tool list from environment."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        monkeypatch.setenv("MCP_CLI_INCLUDE_TOOLS", "tool_a,tool_b,tool_c")

        mcp_config = MCPConfig()
        runtime = LegacyRuntimeConfig(mcp_config)

        tools = runtime.get_tool_config_value("include_tools")
        assert tools == ["tool_a", "tool_b", "tool_c"]

    def test_get_tool_config_value_dynamic_tools_enabled(self, monkeypatch):
        """Test getting dynamic_tools_enabled from env."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        monkeypatch.setenv("MCP_CLI_DYNAMIC_TOOLS_ENABLED", "true")

        mcp_config = MCPConfig()
        runtime = LegacyRuntimeConfig(mcp_config)

        enabled = runtime.get_tool_config_value("dynamic_tools_enabled")
        assert enabled is True

    def test_get_tool_config_value_confirm_tools_disabled(self, monkeypatch):
        """Test getting confirm_tools disabled from env."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        monkeypatch.setenv("MCP_CLI_CONFIRM_TOOLS", "false")

        mcp_config = MCPConfig()
        runtime = LegacyRuntimeConfig(mcp_config)

        confirm = runtime.get_tool_config_value("confirm_tools")
        assert confirm is False

    def test_get_tool_config_value_max_concurrency(self, monkeypatch):
        """Test getting max_concurrency from env."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        monkeypatch.setenv("MCP_CLI_MAX_CONCURRENCY", "10")

        mcp_config = MCPConfig()
        runtime = LegacyRuntimeConfig(mcp_config)

        concurrency = runtime.get_tool_config_value("max_concurrency")
        assert concurrency == 10

    def test_get_tool_config_value_invalid_max_concurrency(self, monkeypatch):
        """Test invalid max_concurrency falls back to config."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        monkeypatch.setenv("MCP_CLI_MAX_CONCURRENCY", "not_a_number")

        mcp_config = MCPConfig()
        runtime = LegacyRuntimeConfig(mcp_config)

        concurrency = runtime.get_tool_config_value("max_concurrency")
        # Should get from config, not crash
        assert concurrency is not None

    def test_get_all_timeouts(self):
        """Test getting all timeouts."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        mcp_config = MCPConfig()
        runtime = LegacyRuntimeConfig(mcp_config)

        all_timeouts = runtime.get_all_timeouts()
        assert "streaming_chunk" in all_timeouts
        assert "streaming_global" in all_timeouts
        assert "tool_execution" in all_timeouts
        assert "server_init" in all_timeouts

    def test_update_from_cli(self):
        """Test update_from_cli method."""
        from mcp_cli.config.config_manager import RuntimeConfig as LegacyRuntimeConfig

        mcp_config = MCPConfig()
        runtime = LegacyRuntimeConfig(mcp_config)

        runtime.update_from_cli(streaming_chunk_timeout=99.0, custom_key="value")

        assert runtime.cli_overrides["streaming_chunk_timeout"] == 99.0
        assert runtime.cli_overrides["custom_key"] == "value"


class TestGetRuntimeConfig:
    """Test get_runtime_config function."""

    def test_get_runtime_config_with_config(self):
        """Test get_runtime_config with provided config."""
        from mcp_cli.config.config_manager import (
            get_runtime_config,
            RuntimeConfig as LegacyRuntimeConfig,
        )

        mcp_config = MCPConfig()
        runtime = get_runtime_config(mcp_config)

        assert isinstance(runtime, LegacyRuntimeConfig)

    def test_get_runtime_config_without_config(self, tmp_path, monkeypatch):
        """Test get_runtime_config without config uses ConfigManager."""
        from mcp_cli.config.config_manager import (
            get_runtime_config,
            RuntimeConfig as LegacyRuntimeConfig,
        )

        # Reset ConfigManager
        manager = ConfigManager()
        manager.reset()

        # This will create default config since ConfigManager not initialized
        runtime = get_runtime_config()

        assert isinstance(runtime, LegacyRuntimeConfig)

    def test_get_runtime_config_with_cli_overrides(self):
        """Test get_runtime_config with CLI overrides."""
        from mcp_cli.config.config_manager import get_runtime_config

        mcp_config = MCPConfig()
        cli_overrides = {"timeout": 60.0}
        runtime = get_runtime_config(mcp_config, cli_overrides)

        assert runtime.cli_overrides == cli_overrides


class TestServerConfigOAuth:
    """Test ServerConfig OAuth handling."""

    def test_from_dict_with_oauth_dict(self):
        """Test ServerConfig.from_dict when oauth is provided as dict."""
        data = {
            "command": "python",
            "oauth": {
                "client_id": "test",
                "authorization_url": "https://auth.example.com/authorize",
                "token_url": "https://auth.example.com/token",
            },
        }

        config = ServerConfig.from_dict("test", data)
        assert config.command == "python"
        assert config.oauth is not None
        assert config.oauth.client_id == "test"
