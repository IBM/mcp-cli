"""Tests for built-in gateway provider registration."""

from __future__ import annotations

from unittest.mock import MagicMock

from mcp_cli.auth.provider_tokens import PROVIDER_ENV_VAR_MAP
from mcp_cli.model_management.gateway_providers import (
    GATEWAY_PROVIDERS,
    ORCAROUTER_API_KEY_ENV,
    ORCAROUTER_BASE_URL,
    ORCAROUTER_DEFAULT_MODEL,
    register_gateway_providers,
)


class TestRegisterGatewayProviders:
    """Tests for register_gateway_providers()."""

    def test_registers_orcarouter_with_openai_compatible_wiring(self) -> None:
        """OrcaRouter is registered as a named OpenAI-compatible provider."""
        mock_config = MagicMock()
        mock_config.get_all_providers.return_value = ["openai", "anthropic"]

        register_gateway_providers(mock_config)

        mock_config.register_provider.assert_called_once_with(
            name="orcarouter",
            api_key_env=ORCAROUTER_API_KEY_ENV,
            api_base=ORCAROUTER_BASE_URL,
            default_model=ORCAROUTER_DEFAULT_MODEL,
            models=["*"],
        )

    def test_skips_provider_already_registered(self) -> None:
        """If chuk_llm already ships orcarouter, leave it untouched."""
        mock_config = MagicMock()
        mock_config.get_all_providers.return_value = ["openai", "orcarouter"]

        register_gateway_providers(mock_config)

        mock_config.register_provider.assert_not_called()

    def test_none_config_is_noop(self) -> None:
        """A missing config manager should not raise."""
        register_gateway_providers(None)

    def test_registration_failure_is_swallowed(self) -> None:
        """A failing registration should not crash startup."""
        mock_config = MagicMock()
        mock_config.get_all_providers.side_effect = RuntimeError("boom")

        register_gateway_providers(mock_config)

    def test_gateway_provider_defaults_are_sane(self) -> None:
        """The OrcaRouter definition is a named OpenAI-compatible gateway."""
        orcarouter = GATEWAY_PROVIDERS["orcarouter"]
        assert orcarouter["api_base"] == "https://api.orcarouter.ai/v1"
        assert orcarouter["api_key_env"] == "ORCAROUTER_API_KEY"
        assert orcarouter["models"] == ["*"]


class TestProviderEnvVarMap:
    """Tests for the provider token env var mapping."""

    def test_orcarouter_maps_to_orcarouter_api_key(self) -> None:
        assert PROVIDER_ENV_VAR_MAP["orcarouter"] == "ORCAROUTER_API_KEY"
