# tests/mcp_cli/test_cli_options.py
import json
import os
import logging
from unittest.mock import patch, MagicMock

import pytest

from mcp_cli.config.cli_options import (
    load_config,
    extract_server_names,
    process_options,
    get_config_summary,
)
from mcp_cli.config import MCPConfig


# Stub for removed function - kept for backwards compatibility with tests
def inject_logging_env_vars(config, quiet=False):
    """Stub for removed inject_logging_env_vars function."""
    pass


@pytest.fixture
def valid_config(tmp_path):
    """Create a temporary config file with valid JSON and proper server configs."""
    config_content = {
        "mcpServers": {
            "ServerA": {
                "command": "server-a-cmd",  # Added required field
                "param": "valueA",
            },
            "ServerB": {
                "command": "server-b-cmd",  # Added required field
                "param": "valueB",
            },
            "ServerC": {
                "url": "http://localhost:8080",  # HTTP server example
                "param": "valueC",
            },
        }
    }
    config_file = tmp_path / "config_valid.json"
    config_file.write_text(json.dumps(config_content))
    return config_file


@pytest.fixture
def invalid_config(tmp_path):
    """Create a temporary config file with invalid JSON."""
    config_file = tmp_path / "config_invalid.json"
    config_file.write_text("this is not json")
    return config_file


def test_load_config_valid(valid_config):
    # When the file exists and contains valid JSON, load_config should return an MCPConfig.
    config = load_config(str(valid_config))
    assert isinstance(config, MCPConfig)
    assert "ServerA" in config.servers
    assert config.servers["ServerA"]["command"] == "server-a-cmd"


def test_load_config_missing(tmp_path):
    # Pass a path that does not exist.
    non_existent = tmp_path / "nonexistent.json"
    config = load_config(str(non_existent))
    # load_config returns None if file not found.
    # The implementation may or may not log warnings depending on the logging configuration
    assert config is None


def test_load_config_invalid(invalid_config):
    # When the file has invalid JSON, load_config should return None.
    config = load_config(str(invalid_config))
    assert config is None
    # The implementation may log errors internally but we only care that it returns None


def test_extract_server_names_all(tmp_path):
    # With a valid config and no specified servers,
    # the function should map all server keys.
    config_content = {
        "mcpServers": {
            "ServerA": {"command": "cmd-a"},
            "ServerB": {"command": "cmd-b"},
        }
    }
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config_content))

    config = load_config(str(config_file))
    server_names = extract_server_names(config)
    # Expecting indices 0 and 1 mapped to the keys from servers.
    assert set(server_names.values()) == {"ServerA", "ServerB"}


def test_extract_server_names_subset(caplog, tmp_path):
    # If specified_servers are provided, only matching ones should be added.
    config_content = {
        "mcpServers": {
            "ServerA": {"command": "cmd-a"},
            "ServerB": {"command": "cmd-b"},
        }
    }
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config_content))

    config = load_config(str(config_file))

    # Provide a mix of matching and non-matching server names.
    specified = ["ServerB", "ServerX"]

    # Clear log and set level to capture warnings
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        server_names = extract_server_names(config, specified)

    assert server_names == {0: "ServerB"}  # Only "ServerB" exists in the config.

    # Check that a warning was logged for the non-existent server
    assert any(
        "ServerX" in record.message and "not found" in record.message.lower()
        for record in caplog.records
    )


def test_extract_server_names_no_config():
    # When config is None or missing "mcpServers", should return empty dict.
    assert extract_server_names(None) == {}
    assert extract_server_names({}) == {}


@pytest.fixture
def dummy_config_file(tmp_path):
    """Create a temporary config file that will be used by process_options."""
    config_content = {
        "mcpServers": {
            "Server1": {
                "command": "server1-cmd",  # Added required field
                "param": "value1",
            },
            "Server2": {
                "command": "server2-cmd",  # Added required field
                "param": "value2",
            },
        }
    }
    config_file = tmp_path / "server_config.json"
    config_file.write_text(json.dumps(config_content))
    return str(config_file)


@patch("mcp_cli.config.cli_options.trigger_discovery_after_setup")
def test_process_options_with_servers(mock_discovery, dummy_config_file, monkeypatch):
    # Mock discovery to avoid actual network calls
    mock_discovery.return_value = 0

    # Prepare inputs.
    # server: a comma-separated string.
    server_input = "Server1, Server2"
    disable_filesystem = False
    provider = "openai"
    model = "custom-model"

    # Clear any preexisting environment variables for a clean test.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("SOURCE_FILESYSTEMS", raising=False)

    servers_list, user_specified, server_names = process_options(
        server=server_input,
        disable_filesystem=disable_filesystem,
        provider=provider,
        model=model,
        config_file=dummy_config_file,
    )

    # Check that server list and user_specified were parsed correctly.
    assert servers_list == ["Server1", "Server2"]
    assert user_specified == ["Server1", "Server2"]

    # In the dummy config, the keys are "Server1" and "Server2". Because the user specified
    # these names, extract_server_names should only include those that match.
    # Mapping is based on order as encountered from the specified servers.
    expected_mapping = {0: "Server1", 1: "Server2"}
    assert server_names == expected_mapping

    # Check environment variables.
    assert os.environ["LLM_PROVIDER"] == provider
    assert os.environ["LLM_MODEL"] == model

    # Since disable_filesystem is False, SOURCE_FILESYSTEMS should be set.
    source_fs = json.loads(os.environ["SOURCE_FILESYSTEMS"])
    # For testing, we expect at least the current working directory.
    assert os.getcwd() in source_fs


@patch("mcp_cli.config.cli_options.trigger_discovery_after_setup")
def test_process_options_without_model_and_files(mock_discovery, monkeypatch, tmp_path):
    # Mock discovery to avoid actual network calls
    mock_discovery.return_value = 0

    # Test defaulting of model and disabling filesystem.
    server_input = "Server1"
    disable_filesystem = (
        True  # With filesystem disabled, SOURCE_FILESYSTEMS should not be set.
    )
    provider = "openai"
    model = None  # Empty/None model - let implementation handle it

    # Create a temporary config with one server with proper configuration.
    config_content = {
        "mcpServers": {
            "Server1": {
                "command": "server1-cmd",  # Added required field
                "param": "value1",
            }
        }
    }
    config_file = tmp_path / "server_config.json"
    config_file.write_text(json.dumps(config_content))

    # Clear environment variables to start fresh.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("SOURCE_FILESYSTEMS", raising=False)

    servers_list, user_specified, server_names = process_options(
        server=server_input,
        disable_filesystem=disable_filesystem,
        provider=provider,
        model=model,
        config_file=str(config_file),
    )

    # Check that provider is set
    assert os.environ["LLM_PROVIDER"] == provider

    # The new implementation only sets LLM_MODEL when a model is explicitly provided
    # When model is None/empty, it may not set the environment variable
    # This is intentional - the model will be determined by downstream components
    # So we don't assert on LLM_MODEL being set

    # SOURCE_FILESYSTEMS should not be set because filesystem is disabled.
    assert "SOURCE_FILESYSTEMS" not in os.environ

    # Check that the servers list and server names are as expected.
    assert servers_list == ["Server1"]
    assert user_specified == ["Server1"]
    assert server_names == {0: "Server1"}


@patch("mcp_cli.config.cli_options.trigger_discovery_after_setup")
def test_process_options_with_explicit_model(mock_discovery, monkeypatch, tmp_path):
    """Test that explicit model is properly set."""
    mock_discovery.return_value = 0

    server_input = "TestServer"
    provider = "ollama"
    model = "gpt-oss"  # Explicit model

    config_content = {"mcpServers": {"TestServer": {"command": "test-cmd", "args": []}}}
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config_content))

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    servers_list, user_specified, server_names = process_options(
        server=server_input,
        disable_filesystem=False,
        provider=provider,
        model=model,
        config_file=str(config_file),
    )

    # When model is explicitly provided, it should be set
    assert os.environ["LLM_PROVIDER"] == "ollama"
    assert os.environ["LLM_MODEL"] == "gpt-oss"
    assert servers_list == ["TestServer"]


@patch("mcp_cli.config.cli_options.trigger_discovery_after_setup")
def test_process_options_http_server(mock_discovery, monkeypatch, tmp_path):
    """Test handling of HTTP server configuration."""
    mock_discovery.return_value = 0

    server_input = "HttpServer"
    provider = "openai"
    model = "gpt-5"

    # Create config with HTTP server (has url instead of command)
    config_content = {
        "mcpServers": {
            "HttpServer": {
                "url": "http://localhost:8080"  # HTTP server config
            }
        }
    }
    config_file = tmp_path / "http_config.json"
    config_file.write_text(json.dumps(config_content))

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    servers_list, user_specified, server_names = process_options(
        server=server_input,
        disable_filesystem=True,
        provider=provider,
        model=model,
        config_file=str(config_file),
    )

    # Verify HTTP server was properly detected and processed
    assert servers_list == ["HttpServer"]
    assert server_names == {0: "HttpServer"}
    assert os.environ["LLM_PROVIDER"] == "openai"
    assert os.environ["LLM_MODEL"] == "gpt-5"


@pytest.mark.skip(reason="Config modification removed - configs are now immutable")
@patch("mcp_cli.config.cli_options.trigger_discovery_after_setup")
def test_process_options_quiet_mode(mock_discovery, monkeypatch, tmp_path, caplog):
    """Test that quiet mode suppresses server noise."""
    # This test is no longer relevant as process_options no longer creates
    # modified configs - the original config is used directly
    pass


@patch("mcp_cli.config.cli_options.trigger_discovery_after_setup")
@patch("mcp_cli.utils.preferences.get_preference_manager")
def test_process_options_disabled_server_blocked(
    mock_pref_manager, mock_discovery, monkeypatch, tmp_path, caplog
):
    """Test that disabled servers are blocked even when explicitly requested."""
    mock_discovery.return_value = 0

    # Mock preference manager to mark server as disabled
    mock_pm = MagicMock()
    mock_pm.is_server_disabled.return_value = True
    mock_pref_manager.return_value = mock_pm

    config_content = {
        "mcpServers": {
            "DisabledServer": {"command": "disabled-cmd", "args": []},
            "EnabledServer": {"command": "enabled-cmd", "args": []},
        }
    }
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config_content))

    with monkeypatch.context() as m:
        m.setenv("LLM_PROVIDER", "test")
        m.setenv("LLM_MODEL", "test-model")

        # Test with explicitly requesting disabled server
        servers_list, specified, server_names = process_options(
            server="DisabledServer",
            disable_filesystem=True,
            provider="openai",
            model="gpt-4",
            config_file=str(config_file),
            quiet=False,
        )

        # Should return empty servers list since server is disabled
        assert servers_list == []
        # specified should still contain what was requested
        assert specified == ["DisabledServer"]

        # Should have logged warning about disabled server
        assert any(
            "disabled" in r.message.lower()
            for r in caplog.records
            if r.levelname == "WARNING"
        )


@patch("mcp_cli.config.cli_options.trigger_discovery_after_setup")
@patch("mcp_cli.utils.preferences.get_preference_manager")
def test_process_options_mixed_enabled_disabled(
    mock_pref_manager, mock_discovery, monkeypatch, tmp_path
):
    """Test requesting both enabled and disabled servers."""
    mock_discovery.return_value = 0

    # Mock preference manager - only Server2 is disabled
    mock_pm = MagicMock()

    def is_disabled(server_name):
        return server_name == "Server2"

    mock_pm.is_server_disabled.side_effect = is_disabled
    mock_pref_manager.return_value = mock_pm

    config_content = {
        "mcpServers": {
            "Server1": {"command": "cmd1", "args": []},
            "Server2": {"command": "cmd2", "args": []},
            "Server3": {"command": "cmd3", "args": []},
        }
    }
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config_content))

    with monkeypatch.context() as m:
        m.setenv("LLM_PROVIDER", "test")
        m.setenv("LLM_MODEL", "test-model")

        # Request multiple servers including disabled one
        servers_list, specified, server_names = process_options(
            server="Server1,Server2,Server3",
            disable_filesystem=True,
            provider="openai",
            model="gpt-4",
            config_file=str(config_file),
            quiet=False,
        )

        # Should only return enabled servers
        assert "Server1" in servers_list
        assert "Server2" not in servers_list  # Disabled
        assert "Server3" in servers_list
        assert len(servers_list) == 2


@patch("mcp_cli.config.cli_options.trigger_discovery_after_setup")
@patch("mcp_cli.utils.preferences.get_preference_manager")
def test_process_options_no_servers_when_all_disabled(
    mock_pref_manager, mock_discovery, monkeypatch, tmp_path, caplog
):
    """Test that no servers are loaded when all are disabled."""
    mock_discovery.return_value = 0

    # Mock preference manager - all servers disabled
    mock_pm = MagicMock()
    mock_pm.is_server_disabled.return_value = True
    mock_pref_manager.return_value = mock_pm

    config_content = {
        "mcpServers": {
            "Server1": {"command": "cmd1", "args": []},
            "Server2": {"command": "cmd2", "args": []},
        }
    }
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config_content))

    with monkeypatch.context() as m:
        m.setenv("LLM_PROVIDER", "test")
        m.setenv("LLM_MODEL", "test-model")

        # Request servers when all are disabled
        servers_list, specified, server_names = process_options(
            server=None,  # No specific servers, should check all
            disable_filesystem=True,
            provider="openai",
            model="gpt-4",
            config_file=str(config_file),
            quiet=False,
        )

        # Should return empty list
        assert servers_list == []
        # specified is empty list when no servers are requested
        assert specified == []

        # Should have logged warning
        assert any(
            "No enabled servers found" in record.message for record in caplog.records
        )


@patch("mcp_cli.config.cli_options.trigger_discovery_after_setup")
@patch("mcp_cli.utils.preferences.get_preference_manager")
def test_extract_server_names_with_disabled(
    mock_pref_manager, mock_discovery, tmp_path
):
    """Test that extract_server_names respects disabled status."""
    mock_discovery.return_value = 0

    # Mock preference manager
    mock_pm = MagicMock()

    def is_disabled(server_name):
        return server_name == "DisabledServer"

    mock_pm.is_server_disabled.side_effect = is_disabled
    mock_pref_manager.return_value = mock_pm

    config_content = {
        "mcpServers": {
            "EnabledServer": {"command": "cmd1", "args": []},
            "DisabledServer": {"command": "cmd2", "args": []},
            "AnotherEnabled": {"command": "cmd3", "args": []},
        }
    }
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config_content))

    config = load_config(str(config_file))

    # Test without specifying servers - should filter disabled
    server_names = extract_server_names(config, None)
    assert "EnabledServer" in server_names.values()
    assert "DisabledServer" not in server_names.values()
    assert "AnotherEnabled" in server_names.values()

    # Test with explicitly specifying servers - includes all specified that exist
    server_names = extract_server_names(config, ["EnabledServer", "DisabledServer"])
    assert "EnabledServer" in server_names.values()
    assert (
        "DisabledServer" in server_names.values()
    )  # Included when explicitly specified


class TestLoadConfigBundled:
    """Test config loading when file is missing (no bundled fallback)."""

    def test_load_config_missing_returns_none(self, tmp_path, monkeypatch):
        """Test that missing config file returns None (no bundled fallback)."""
        monkeypatch.chdir(tmp_path)
        config = load_config("server_config.json")
        assert config is None

    def test_load_config_with_empty_but_valid_json(self, tmp_path):
        """Test loading config with empty but valid JSON."""
        config_file = tmp_path / "empty.json"
        config_file.write_text("{}")

        config = load_config(str(config_file))
        # Should return config with empty servers dict
        assert config is not None
        assert len(config.servers) == 0

    def test_load_config_with_content_but_invalid_json(self, tmp_path):
        """Test file has content but invalid JSON returns None."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text('{"invalid": json syntax}')

        config = load_config(str(config_file))
        assert config is None


@pytest.mark.skip(
    reason="inject_logging_env_vars function was removed during refactoring"
)
class TestInjectLoggingEnvVars:
    """Test inject_logging_env_vars function."""

    def test_inject_logging_quiet_mode(self, tmp_path):
        """Test logging injection in quiet mode."""
        config_data = {
            "mcpServers": {"Server1": {"command": "cmd1", "args": [], "env": {}}}
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)

        # Inject with quiet=True
        inject_logging_env_vars(config, quiet=True)

        # Should set ERROR level
        assert config.servers["Server1"].env["LOG_LEVEL"] == "ERROR"
        assert config.servers["Server1"].env["PYTHONWARNINGS"] == "ignore"

    def test_inject_logging_normal_mode(self, tmp_path):
        """Test logging injection in normal mode."""
        config_data = {
            "mcpServers": {"Server1": {"command": "cmd1", "args": [], "env": {}}}
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)

        # Inject with quiet=False (default)
        inject_logging_env_vars(config, quiet=False)

        # Should set WARNING level
        assert config.servers["Server1"].env["LOG_LEVEL"] == "WARNING"
        assert config.servers["Server1"].env["MCP_LOG_LEVEL"] == "WARNING"

    def test_inject_preserves_existing_env_vars(self, tmp_path):
        """Test that existing env vars are not overwritten."""
        config_data = {
            "mcpServers": {
                "Server1": {
                    "command": "cmd1",
                    "args": [],
                    "env": {"LOG_LEVEL": "DEBUG"},
                }
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)

        inject_logging_env_vars(config, quiet=True)

        # Should preserve existing LOG_LEVEL
        assert config.servers["Server1"].env["LOG_LEVEL"] == "DEBUG"

    def test_inject_skips_http_servers(self, tmp_path):
        """Test that HTTP servers (no command) don't get env vars injected."""
        config_data = {
            "mcpServers": {"HttpServer": {"url": "http://localhost:8080", "env": {}}}
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)

        inject_logging_env_vars(config, quiet=True)

        # HTTP server should not have logging env vars
        assert "LOG_LEVEL" not in config.servers["HttpServer"].env

    def test_inject_with_none_config(self):
        """Test that inject handles None config gracefully."""
        # Should return without error
        inject_logging_env_vars(None, quiet=True)

    def test_inject_with_empty_servers(self, tmp_path):
        """Test injection with config that has no servers."""
        config_data = {"mcpServers": {}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = MCPConfig.load_from_file(config_file)

        # Should return without error
        inject_logging_env_vars(config, quiet=True)


class TestProcessOptionsEdgeCases:
    """Test edge cases and error paths in process_options."""

    @patch("mcp_cli.config.cli_options.setup_chuk_llm_environment")
    @patch("mcp_cli.config.cli_options.trigger_discovery_after_setup")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    def test_process_options_with_discovery_results(
        self, mock_pref, mock_discovery, mock_setup_env, tmp_path, monkeypatch
    ):
        """Test that discovery is triggered and returns count."""
        # Mock discovery to return some functions
        mock_discovery.return_value = 5

        # Mock preference manager to allow servers
        mock_pm = MagicMock()
        mock_pm.is_server_disabled.return_value = False
        mock_pref.return_value = mock_pm

        config_content = {"mcpServers": {"Server1": {"command": "cmd1", "args": []}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_content))

        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        servers_list, _, _ = process_options(
            server="Server1",
            disable_filesystem=True,
            provider="openai",
            model="gpt-4",
            config_file=str(config_file),
        )

        # Verify discovery was called and returned correct count
        mock_discovery.assert_called_once()
        assert servers_list == ["Server1"]

    @patch("mcp_cli.config.cli_options.trigger_discovery_after_setup")
    def test_process_options_with_nonexistent_config(
        self, mock_discovery, tmp_path, monkeypatch, caplog
    ):
        """Test handling of non-existent config file."""
        mock_discovery.return_value = 0

        nonexistent_file = str(tmp_path / "nonexistent.json")

        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        with caplog.at_level(logging.DEBUG, logger="mcp_cli.config.cli_options"):
            servers_list, specified, server_names = process_options(
                server="Server1",
                disable_filesystem=True,
                provider="openai",
                model="gpt-4",
                config_file=nonexistent_file,
            )

        # Should return empty lists/dict
        assert servers_list == []
        assert specified == ["Server1"]
        assert server_names == {}

        assert any(
            "Could not load config file" in record.message for record in caplog.records
        )

    @patch("mcp_cli.config.cli_options.validate_server_config")
    @patch("mcp_cli.config.cli_options.trigger_discovery_after_setup")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    def test_process_options_with_validation_errors(
        self, mock_pref, mock_discovery, mock_validate, tmp_path, monkeypatch, caplog
    ):
        """Test handling of server validation errors."""
        mock_discovery.return_value = 0
        mock_validate.return_value = (
            False,
            ["Server1 missing command", "Server2 has invalid config"],
        )

        # Mock preference manager to allow servers
        mock_pm = MagicMock()
        mock_pm.is_server_disabled.return_value = False
        mock_pref.return_value = mock_pm

        config_content = {
            "mcpServers": {
                "Server1": {"command": "cmd1", "args": []},
                "Server2": {"command": "cmd2", "args": []},
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_content))

        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        with caplog.at_level(logging.ERROR):
            servers_list, _, _ = process_options(
                server="Server1,Server2",
                disable_filesystem=True,
                provider="openai",
                model="gpt-4",
                config_file=str(config_file),
            )

        # Should log validation errors
        assert any(
            "Server configuration validation failed" in record.message
            for record in caplog.records
        )
        assert any(
            "Server1 missing command" in record.message for record in caplog.records
        )

    @pytest.mark.skip(
        reason="MCPConfig is now immutable and doesn't have save_to_file method"
    )
    @patch("mcp_cli.config.cli_options.trigger_discovery_after_setup")
    @patch("mcp_cli.utils.preferences.get_preference_manager")
    def test_process_options_modified_config_save_error(
        self, mock_pref, mock_discovery, tmp_path, monkeypatch, caplog
    ):
        """Test handling of error when saving modified config."""
        # This test is no longer relevant as MCPConfig is immutable
        # and process_options no longer modifies or saves config files
        pass


class TestGetConfigSummary:
    """Test get_config_summary function."""

    def test_get_config_summary_valid(self, tmp_path):
        """Test getting summary of valid config."""
        config_content = {
            "mcpServers": {
                "StdioServer": {"command": "stdio-cmd", "args": []},
                "HttpServer": {"url": "http://localhost:8080"},
                "AnotherStdio": {"command": "another-cmd", "args": []},
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_content))

        summary = get_config_summary(str(config_file))

        assert summary["total_servers"] == 3
        assert summary["http_servers"] == 1
        assert summary["stdio_servers"] == 2
        assert "StdioServer" in summary["server_names"]
        assert "HttpServer" in summary["server_names"]
        assert len(summary["http_server_details"]) == 1
        assert summary["http_server_details"][0]["name"] == "HttpServer"
        assert summary["http_server_details"][0]["url"] == "http://localhost:8080"

    def test_get_config_summary_invalid_config(self, tmp_path):
        """Test getting summary when config cannot be loaded."""
        nonexistent_file = str(tmp_path / "nonexistent.json")

        summary = get_config_summary(nonexistent_file)

        assert "error" in summary
        assert summary["error"] == "Could not load config file"
