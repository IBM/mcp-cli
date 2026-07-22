"""
ChukLLM discovery and provider management.

This module handles the discovery and validation of ChukLLM providers and models.
Uses singleton pattern instead of module-level globals for cleaner state management.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class DiscoveryManager:
    """Singleton manager for ChukLLM discovery state.

    Replaces module-level globals with a proper class-based singleton.
    Thread-safe via Python's GIL for simple flag operations.
    """

    _instance: "DiscoveryManager | None" = None

    # Instance attributes with type annotations
    _env_setup_complete: bool
    _discovery_triggered: bool

    def __new__(cls) -> "DiscoveryManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._env_setup_complete = False
            cls._instance._discovery_triggered = False
        return cls._instance

    @property
    def env_setup_complete(self) -> bool:
        """Whether environment setup has been completed."""
        return self._env_setup_complete

    @property
    def discovery_triggered(self) -> bool:
        """Whether discovery has been triggered."""
        return self._discovery_triggered

    def setup_environment(self) -> None:
        """Set up environment variables for ChukLLM discovery.

        MUST be called before any chuk_llm imports.
        """
        if self._env_setup_complete:
            return

        # Set environment variables (only if not already set by user)
        env_vars = {
            "CHUK_LLM_DISCOVERY_ENABLED": "true",
            "CHUK_LLM_AUTO_DISCOVER": "true",
            "CHUK_LLM_DISCOVERY_ON_STARTUP": "true",
            "CHUK_LLM_DISCOVERY_TIMEOUT": "10",
            "CHUK_LLM_OLLAMA_DISCOVERY": "true",
            "CHUK_LLM_OPENAI_DISCOVERY": "true",
            "CHUK_LLM_OPENAI_TOOL_COMPATIBILITY": "true",
            "CHUK_LLM_UNIVERSAL_TOOLS": "true",
        }

        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value

        self._env_setup_complete = True
        logger.debug("ChukLLM environment variables set")

    def trigger_discovery(self, provider: str | None = None) -> int:
        """Trigger discovery after environment setup.

        Call this after setup_environment() and before using models.

        Args:
            provider: Specific provider to discover (None for all configured providers)

        Returns:
            Number of new functions discovered
        """
        if self._discovery_triggered and provider is None:
            return 0

        try:
            from chuk_llm.api.providers import refresh_provider_functions

            logger.debug(f"Triggering discovery for provider: {provider or 'all'}...")

            # Trigger discovery for specified provider (or all if None)
            new_functions = refresh_provider_functions(provider)

            if provider is None:
                self._discovery_triggered = True

            if new_functions:
                logger.debug(f"CLI discovery: {len(new_functions)} new functions")
            else:
                logger.debug("CLI discovery: no new functions (may already be cached)")

            return len(new_functions)

        except Exception as e:
            logger.debug(f"CLI discovery failed: {e}")
            return 0

    def force_refresh(self) -> int:
        """Force a fresh discovery (useful for debugging).

        Returns:
            Number of new functions discovered
        """
        from mcp_cli.config.env_vars import EnvVar, set_env

        self._discovery_triggered = False

        # Set force refresh environment variable (using constant)
        set_env(EnvVar.CHUK_LLM_DISCOVERY_FORCE_REFRESH, "true")

        # Trigger discovery again
        return self.trigger_discovery()

    def get_status(self) -> dict[str, Any]:
        """Get discovery status for debugging.

        Returns:
            Dictionary with discovery status information
        """
        from mcp_cli.config.discovery_models import DiscoveryConfig, DiscoveryStatus

        status = DiscoveryStatus(
            env_setup_complete=self._env_setup_complete,
            discovery_triggered=self._discovery_triggered,
            config=DiscoveryConfig.from_env(),
        )
        return status.to_dict()


# Singleton instance - use this for access
_discovery_manager = DiscoveryManager()


def get_discovery_manager() -> DiscoveryManager:
    """Get the singleton DiscoveryManager instance."""
    return _discovery_manager


# ──────────────────────────────────────────────────────────────────────────────
# Backward-compatible module-level functions (delegate to singleton)
# ──────────────────────────────────────────────────────────────────────────────


def setup_chuk_llm_environment() -> None:
    """Set up environment variables for ChukLLM discovery.

    MUST be called before any chuk_llm imports.
    """
    _discovery_manager.setup_environment()


def trigger_discovery_after_setup() -> int:
    """Trigger discovery after environment setup.

    Returns:
        Number of new functions discovered
    """
    return _discovery_manager.trigger_discovery()


def get_available_models_quick(provider: str | None = None) -> list[str]:
    """Quick function to get available models after discovery.

    Args:
        provider: Provider name (uses DEFAULT_PROVIDER if None)

    Returns:
        List of available model names
    """
    from mcp_cli.config.defaults import DEFAULT_PROVIDER

    provider = provider or DEFAULT_PROVIDER
    try:
        from chuk_llm.llm.client import list_available_providers

        providers = list_available_providers()
        provider_info = providers.get(provider, {})
        models = provider_info.get("models", [])
        return list(models)  # Ensure it's a list
    except Exception as e:
        logger.debug(f"Could not get models for {provider}: {e}")
        return []


def validate_provider_exists(provider: str) -> bool:
    """Validate provider exists, potentially after discovery.

    Args:
        provider: Provider name to validate

    Returns:
        True if provider exists, False otherwise
    """
    try:
        from chuk_llm.configuration import get_config

        config = get_config()
        config.get_provider(provider)  # This will raise if not found
        return True
    except Exception as e:
        logger.debug("Provider validation failed for %s: %s", provider, e)
        return False


def get_discovery_status() -> dict[str, Any]:
    """Get discovery status for debugging.

    Returns:
        Dictionary with discovery status information
    """
    return _discovery_manager.get_status()


def force_discovery_refresh() -> int:
    """Force a fresh discovery (useful for debugging).

    Returns:
        Number of new functions discovered
    """
    return _discovery_manager.force_refresh()
