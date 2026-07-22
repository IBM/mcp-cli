# mcp_cli/auth/provider_tokens.py
"""
Provider token management with hierarchical resolution.

This module provides utilities for managing API keys for LLM providers
with a clear hierarchy: environment variables > token storage > config.
"""

from __future__ import annotations

import os
from typing import Any
import logging

from chuk_mcp_client_oauth import TokenManager
from chuk_mcp_client_oauth.token_types import TokenType

logger = logging.getLogger(__name__)


# Standard provider environment variable mappings
PROVIDER_ENV_VAR_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GOOGLE_API_KEY",  # Alias for gemini
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "togetherai": "TOGETHER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "watsonx": "WATSONX_API_KEY",
    "advantage": "ADVANTAGE_API_KEY",
    "vllm": "VLLM_API_KEY",
    "litellm": "LITELLM_API_KEY",
}


def get_provider_env_var_name(provider_name: str) -> str:
    """
    Get the standard environment variable name for a provider.

    Args:
        provider_name: Provider name (e.g., 'openai', 'anthropic')

    Returns:
        Environment variable name (e.g., 'OPENAI_API_KEY')
    """
    # Check known mappings first
    if provider_name.lower() in PROVIDER_ENV_VAR_MAP:
        return PROVIDER_ENV_VAR_MAP[provider_name.lower()]

    # Default pattern: PROVIDERNAME_API_KEY
    return f"{provider_name.upper().replace('-', '_')}_API_KEY"


def get_provider_token_with_hierarchy(
    provider_name: str,
    token_manager: TokenManager | None = None,
) -> tuple[str | None, str]:
    """
    Get provider API key using hierarchical resolution.

    Hierarchy:
    1. Environment variable (highest priority)
    2. Token storage (via TokenManager)
    3. None if not found

    Args:
        provider_name: Provider name
        token_manager: TokenManager instance (optional)

    Returns:
        Tuple of (api_key, source) where source is 'env', 'storage', or 'none'
    """
    env_var = get_provider_env_var_name(provider_name)

    # 1. Check environment variable first
    env_value = os.environ.get(env_var)
    if env_value:
        logger.debug(f"Using {provider_name} API key from environment: {env_var}")
        return env_value, "env"

    # 2. Check token storage if manager provided
    if token_manager:
        try:
            # Try to retrieve from storage using provider namespace
            stored_token_data = token_manager.token_store.retrieve_generic(
                provider_name, namespace="provider"
            )

            if stored_token_data:
                logger.debug(f"Using {provider_name} API key from token storage")
                return stored_token_data, "storage"
        except Exception as e:
            logger.warning(
                f"Failed to retrieve {provider_name} token from storage: {e}"
            )

    # 3. Not found
    logger.debug(f"No API key found for {provider_name}")
    return None, "none"


def check_provider_token_status(
    provider_name: str,
    token_manager: TokenManager | None = None,
) -> dict[str, Any]:
    """
    Check the status of a provider's API key.

    Args:
        provider_name: Provider name
        token_manager: TokenManager instance (optional)

    Returns:
        Dict with keys:
        - has_token: bool
        - source: 'env', 'storage', or 'none'
        - env_var: expected environment variable name
        - in_env: bool
        - in_storage: bool
    """
    env_var = get_provider_env_var_name(provider_name)
    in_env = bool(os.environ.get(env_var))
    in_storage = False

    # Check storage if manager provided
    if token_manager:
        try:
            stored = token_manager.token_store.retrieve_generic(
                provider_name, namespace="provider"
            )
            in_storage = bool(stored)
        except Exception as e:
            logger.debug("Token storage check failed for %s: %s", provider_name, e)

    # Determine source and overall status
    if in_env:
        source = "env"
        has_token = True
    elif in_storage:
        source = "storage"
        has_token = True
    else:
        source = "none"
        has_token = False

    return {
        "has_token": has_token,
        "source": source,
        "env_var": env_var,
        "in_env": in_env,
        "in_storage": in_storage,
    }


def set_provider_token(
    provider_name: str,
    api_key: str,
    token_manager: TokenManager,
) -> bool:
    """
    Store a provider API key in secure storage.

    Note: This stores in token storage. Environment variables take precedence
    and must be set separately by the user.

    Args:
        provider_name: Provider name
        api_key: API key to store
        token_manager: TokenManager instance

    Returns:
        True if successful, False otherwise
    """
    try:
        # Store using store_generic for simple string storage
        token_manager.token_store.store_generic(
            provider_name, api_key, namespace="provider"
        )

        # Register in token registry
        token_manager.registry.register(
            provider_name,
            TokenType.API_KEY,
            "provider",
            metadata={"provider": provider_name},
        )

        logger.info(f"Stored API key for provider: {provider_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to store provider token: {e}")
        return False


def delete_provider_token(
    provider_name: str,
    token_manager: TokenManager,
) -> bool:
    """
    Delete a provider API key from secure storage.

    Note: This only removes from storage. Environment variables are unaffected.

    Args:
        provider_name: Provider name
        token_manager: TokenManager instance

    Returns:
        True if successful, False otherwise
    """
    try:
        # Delete from storage
        result = token_manager.token_store.delete_generic(
            provider_name, namespace="provider"
        )

        # Unregister from registry
        if result:
            token_manager.registry.unregister(provider_name, "provider")

        return bool(result)

    except Exception as e:
        logger.error(f"Failed to delete provider token: {e}")
        return False


def get_provider_token_display_status(
    provider_name: str,
    token_manager: TokenManager | None = None,
) -> str:
    """
    Get a human-readable status indicator for a provider's token.

    Args:
        provider_name: Provider name
        token_manager: TokenManager instance (optional)

    Returns:
        Status string like "✅ OPENAI_API_KEY", "🔐 storage", "❌ OPENAI_API_KEY not set"
    """
    status = check_provider_token_status(provider_name, token_manager)

    if status["in_env"]:
        return f"✅ {status['env_var']}"
    elif status["in_storage"]:
        return "🔐 storage"
    else:
        return f"❌ {status['env_var']} not set"


def list_all_provider_tokens(
    token_manager: TokenManager,
) -> dict[str, dict[str, Any]]:
    """
    List all provider tokens stored in secure storage.

    This function checks all known providers from chuk_llm and shows which ones
    have tokens stored in secure storage (excludes environment variables).

    Args:
        token_manager: TokenManager instance

    Returns:
        Dict mapping provider names to their token status (only storage tokens)
    """
    result = {}

    try:
        # Get all providers from chuk_llm
        from chuk_llm.llm.client import list_available_providers

        all_providers = list_available_providers()

        # Check each provider for token status
        from mcp_cli.config import PROVIDER_OLLAMA

        for provider_name in all_providers.keys():
            # Skip ollama (doesn't need tokens)
            if provider_name.lower() == PROVIDER_OLLAMA:
                continue

            status = check_provider_token_status(provider_name, token_manager)

            # Only include providers that have tokens in storage
            if status["in_storage"]:
                result[provider_name] = status

    except Exception as e:
        logger.error(f"Failed to list provider tokens: {e}")

    return result
