# mcp_cli/__init__.py - FIXED VERSION
"""
MCP-CLI package root.

Updated to work with current chuk_llm APIs and ensure proper initialization
of the universal tool compatibility system.

Early-loads environment variables from a .env file so that provider
adapters (OpenAI, Groq, Anthropic, …) can read API keys via `os.getenv`
without the caller having to export them in the shell.

If python-dotenv isn't installed, we just continue silently.

Nothing else should be imported from here to keep side-effects minimal.
"""

from __future__ import annotations

import logging
import os

# Set up logging early
logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")

try:
    from dotenv import load_dotenv

    loaded = load_dotenv()  # returns True if a .env file was found
    if loaded:
        logging.getLogger(__name__).debug(".env loaded successfully")
except ModuleNotFoundError:
    # python-dotenv not installed — .env support disabled
    logging.getLogger(__name__).debug("python-dotenv not installed; skipping .env load")

# This ensures the OpenAI client works correctly with universal tool names
os.environ.setdefault("CHUK_LLM_DISCOVERY_ENABLED", "true")
os.environ.setdefault("CHUK_LLM_AUTO_DISCOVER", "true")
os.environ.setdefault("CHUK_LLM_OPENAI_TOOL_COMPATIBILITY", "true")
os.environ.setdefault("CHUK_LLM_UNIVERSAL_TOOLS", "true")

from importlib.metadata import version as _pkg_version, PackageNotFoundError
try:
    __version__ = _pkg_version("mcp-cli")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
__all__ = ["__version__"]
