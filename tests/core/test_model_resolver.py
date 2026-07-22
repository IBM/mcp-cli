# tests/core/test_model_resolver.py
"""Tests for ModelResolver — provider/model resolution and validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.core.model_resolver import ModelResolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mm():
    """Return a pre-configured mock ModelManager."""
    mm = MagicMock()
    mm.get_active_provider.return_value = "openai"
    mm.get_active_model.return_value = "gpt-4"
    mm.get_default_model.return_value = "gpt-4o-mini"
    mm.validate_provider.return_value = True
    mm.validate_model.return_value = True
    mm.get_available_providers.return_value = ["openai", "anthropic", "ollama"]
    mm.get_available_models.return_value = ["gpt-4", "gpt-4o", "gpt-4o-mini"]
    return mm


@pytest.fixture
def resolver(mock_mm):
    return ModelResolver(model_manager=mock_mm)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestModelResolverInit:
    def test_accepts_injected_manager(self, mock_mm):
        r = ModelResolver(model_manager=mock_mm)
        assert r.model_manager is mock_mm

    @patch("mcp_cli.core.model_resolver.ModelManager")
    def test_creates_default_manager_when_none(self, MockMM):
        r = ModelResolver()
        MockMM.assert_called_once()
        assert r.model_manager is MockMM.return_value


# ---------------------------------------------------------------------------
# resolve()
# ---------------------------------------------------------------------------


class TestResolve:
    def test_both_explicit(self, resolver):
        assert resolver.resolve("anthropic", "claude-3") == ("anthropic", "claude-3")

    def test_provider_only_gets_default_model(self, resolver, mock_mm):
        p, m = resolver.resolve(provider="anthropic", model=None)
        assert p == "anthropic"
        mock_mm.get_default_model.assert_called_once_with("anthropic")
        assert m == "gpt-4o-mini"

    def test_model_only_gets_active_provider(self, resolver, mock_mm):
        p, m = resolver.resolve(provider=None, model="gpt-4o")
        mock_mm.get_active_provider.assert_called_once()
        assert p == "openai"
        assert m == "gpt-4o"

    def test_neither_gets_active_configuration(self, resolver, mock_mm):
        p, m = resolver.resolve()
        mock_mm.get_active_provider.assert_called_once()
        mock_mm.get_active_model.assert_called_once()
        assert (p, m) == ("openai", "gpt-4")


# ---------------------------------------------------------------------------
# validate_provider / validate_model
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_provider_delegates(self, resolver, mock_mm):
        assert resolver.validate_provider("openai") is True
        mock_mm.validate_provider.assert_called_once_with("openai")

    def test_validate_provider_invalid(self, resolver, mock_mm):
        mock_mm.validate_provider.return_value = False
        assert resolver.validate_provider("nonexistent") is False

    def test_validate_model_delegates(self, resolver, mock_mm):
        assert resolver.validate_model("gpt-4", provider="openai") is True
        mock_mm.validate_model.assert_called_once_with("gpt-4", "openai")

    def test_validate_model_no_provider(self, resolver, mock_mm):
        resolver.validate_model("gpt-4")
        mock_mm.validate_model.assert_called_once_with("gpt-4", None)


# ---------------------------------------------------------------------------
# validate_and_print_error
# ---------------------------------------------------------------------------


class TestValidateAndPrintError:
    def test_valid_provider_returns_true(self, resolver):
        assert resolver.validate_and_print_error("openai") is True

    def test_invalid_provider_returns_false_and_prints(self, resolver, mock_mm, capsys):
        mock_mm.validate_provider.return_value = False
        assert resolver.validate_and_print_error("bogus") is False
        captured = capsys.readouterr().out
        assert "bogus" in captured
        assert "openai" in captured  # available providers listed

    def test_suggests_command_for_keywords(self, resolver, mock_mm, capsys):
        mock_mm.validate_provider.return_value = False
        resolver.validate_and_print_error("list")
        captured = capsys.readouterr().out
        assert "Did you mean" in captured
        assert "mcp-cli provider list" in captured


# ---------------------------------------------------------------------------
# switch_to
# ---------------------------------------------------------------------------


class TestSwitchTo:
    def test_switch_provider_and_model(self, resolver, mock_mm):
        resolver.switch_to(provider="anthropic", model="claude-3")
        mock_mm.switch_model.assert_called_once_with("anthropic", "claude-3")

    def test_switch_provider_only(self, resolver, mock_mm):
        resolver.switch_to(provider="anthropic")
        mock_mm.switch_provider.assert_called_once_with("anthropic")

    def test_switch_model_only_uses_current_provider(self, resolver, mock_mm):
        resolver.switch_to(model="gpt-4o")
        mock_mm.get_active_provider.assert_called()
        mock_mm.switch_model.assert_called_once_with("openai", "gpt-4o")

    def test_switch_neither_returns_current(self, resolver, mock_mm):
        p, m = resolver.switch_to()
        # No switch calls
        mock_mm.switch_model.assert_not_called()
        mock_mm.switch_provider.assert_not_called()
        assert (p, m) == ("openai", "gpt-4")


# ---------------------------------------------------------------------------
# configure_provider
# ---------------------------------------------------------------------------


class TestConfigureProvider:
    def test_configure_with_key_and_base(self, resolver, mock_mm):
        resolver.configure_provider("deepseek", api_key="sk-test", api_base="http://x")
        mock_mm.add_runtime_provider.assert_called_once_with(
            name="deepseek", api_key="sk-test", api_base="http://x"
        )

    def test_configure_defaults_api_base_to_empty(self, resolver, mock_mm):
        resolver.configure_provider("deepseek", api_key="sk-test")
        mock_mm.add_runtime_provider.assert_called_once_with(
            name="deepseek", api_key="sk-test", api_base=""
        )


# ---------------------------------------------------------------------------
# get_status / get_available_*
# ---------------------------------------------------------------------------


class TestStatus:
    def test_get_status_shape(self, resolver):
        status = resolver.get_status()
        assert "active_provider" in status
        assert "active_model" in status
        assert "available_providers" in status
        assert "provider_model_counts" in status

    def test_get_available_providers(self, resolver):
        assert resolver.get_available_providers() == ["openai", "anthropic", "ollama"]

    def test_get_available_models(self, resolver, mock_mm):
        models = resolver.get_available_models("openai")
        mock_mm.get_available_models.assert_called_once_with("openai")
        assert models == ["gpt-4", "gpt-4o", "gpt-4o-mini"]
