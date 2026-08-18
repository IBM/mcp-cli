"""Built-in gateway providers registered into chuk_llm at startup.

chuk-llm ships named providers (openai, anthropic, openrouter, ...) in its
bundled YAML configuration. Aggregator gateways such as OrcaRouter are
registered here so mcp-cli users can select them by name and by their own
API key without waiting for a chuk-llm release to add them.

Registration is idempotent: if a chuk-llm version already ships the provider
natively, we leave its definition untouched.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: OrcaRouter OpenAI-compatible gateway.
ORCAROUTER_BASE_URL = "https://api.orcarouter.ai/v1"
ORCAROUTER_API_KEY_ENV = "ORCAROUTER_API_KEY"
ORCAROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"

#: Named gateway providers to register into the chuk_llm config manager.
#: The default client class (OpenAI-compatible) is used, mirroring how the
#: bundled chuk_llm.yaml wires openrouter / deepseek / moonshot.
GATEWAY_PROVIDERS: dict[str, dict[str, Any]] = {
    "orcarouter": {
        "api_key_env": ORCAROUTER_API_KEY_ENV,
        "api_base": ORCAROUTER_BASE_URL,
        "default_model": ORCAROUTER_DEFAULT_MODEL,
        "models": ["*"],
    },
}


def register_gateway_providers(config: Any) -> None:
    """Register built-in gateway providers into a chuk_llm config manager.

    Args:
        config: A chuk_llm configuration manager (as returned by
            ``chuk_llm.configuration.get_config()``). ``None`` is a no-op.
    """
    if config is None:
        return

    try:
        existing = set(config.get_all_providers())
        for name, definition in GATEWAY_PROVIDERS.items():
            if name in existing:
                continue
            config.register_provider(name=name, **definition)
            logger.info("Registered built-in gateway provider: %s", name)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to register built-in gateway providers: %s", e)
