# mcp_cli/tools/manager.py
"""
ToolManager - orchestrates chuk-tool-processor with mcp-cli features.

Responsibilities:
1. Parse MCP config files and detect server types (HTTP/SSE/STDIO)
2. Integrate with mcp-cli's OAuth TokenManager
3. Filter and validate tools for LLM compatibility
4. Convert between chuk and mcp-cli data models
5. Configure production middleware (retry, circuit breaker, rate limiting)

For direct tool operations, use the exposed stream_manager property.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from chuk_tool_processor import StreamManager, ToolProcessor
from chuk_tool_processor import ToolCall as CTPToolCall
from chuk_tool_processor import ToolResult as CTPToolResult
from chuk_tool_processor.mcp import MiddlewareConfig

from mcp_cli.auth import (
    OAuthHandler,
    StoredToken,
    TokenManager,
    TokenStoreBackend,
    TokenType,
)
from mcp_cli.config import RuntimeConfig, TimeoutType, load_runtime_config
from mcp_cli.config.defaults import DEFAULT_MIDDLEWARE_ENABLED
from mcp_cli.config import ServerStatus
from mcp_cli.llm.content_models import ContentBlockType
from mcp_cli.tools.config_loader import ConfigLoader
from mcp_cli.tools.dynamic_tools import DynamicToolProvider
from mcp_cli.tools.execution import (
    execute_tools_parallel as _execute_tools_parallel,
    stream_execute_tools as _stream_execute_tools,
)
from mcp_cli.tools.filter import DisabledReason, ToolFilter
from mcp_cli.tools.models import (
    ServerInfo,
    ToolCallResult,
    ToolDefinitionInput,
    ToolInfo,
    TransportType,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# OAuth Error Detection Patterns
# ──────────────────────────────────────────────────────────────────────────────
OAUTH_ERROR_PATTERNS = [
    "requires OAuth authorization",
    "requires oauth authorization",
    "OAuth authorization required",
    "oauth authorization required",
    "authentication required",
    "Authorization required",
    "unauthorized",
    "401",
]


def _is_oauth_error(error_message: str) -> bool:
    """Check if an error message indicates OAuth authorization is needed."""
    if not error_message:
        return False
    error_lower = error_message.lower()
    return any(pattern.lower() in error_lower for pattern in OAUTH_ERROR_PATTERNS)


class ToolManager:
    """
    Facade over chuk-tool-processor with mcp-cli specific features.

    Responsibilities:
    1. Parse MCP config files and detect server types (HTTP/SSE/STDIO)
    2. Integrate with mcp-cli's OAuth TokenManager
    3. Filter and validate tools for LLM compatibility
    4. Convert between chuk and mcp-cli data models

    For direct tool operations, use the exposed stream_manager property.
    """

    def __init__(
        self,
        config_file: str,
        servers: list[str],
        server_names: dict[int, str | None] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: int = 4,
        initialization_timeout: float = 120.0,
        runtime_config: RuntimeConfig | None = None,
        middleware_config: MiddlewareConfig | None = None,
        middleware_enabled: bool = DEFAULT_MIDDLEWARE_ENABLED,
    ):
        self.config_file = config_file
        self.servers = servers
        self.server_names = server_names or {}

        # Use runtime config for timeout management (type-safe!)
        self.runtime_config = runtime_config or load_runtime_config()

        # Tool timeout with priority: param > runtime_config
        if tool_timeout is not None:
            self.tool_timeout = tool_timeout
        else:
            self.tool_timeout = self.runtime_config.get_timeout(
                TimeoutType.TOOL_EXECUTION
            )

        # Initialization timeout with priority: param > runtime_config
        if initialization_timeout != 120.0:  # User provided non-default
            self.initialization_timeout = initialization_timeout
        else:
            self.initialization_timeout = self.runtime_config.get_timeout(
                TimeoutType.SERVER_INIT
            )

        self.max_concurrency = max_concurrency

        # Middleware configuration (retry, circuit breaker, rate limiting)
        self._middleware_enabled = middleware_enabled
        self._middleware_config = middleware_config

        logger.debug(
            f"ToolManager initialized with timeouts: "
            f"tool={self.tool_timeout}s, init={self.initialization_timeout}s, "
            f"middleware_enabled={middleware_enabled}"
        )

        # chuk-tool-processor components (publicly accessible)
        self.stream_manager: StreamManager | None = None
        self.processor: ToolProcessor | None = None
        self._registry = None

        # Config loader handles parsing and OAuth
        self._config_loader = ConfigLoader(config_file, servers)

        # mcp-cli features
        self.tool_filter = ToolFilter()

        # Setup dynamic tool provider for on-demand tool discovery
        self.dynamic_tool_provider = DynamicToolProvider(self)

        # O(1) tool lookup index (lazy-built, invalidated on tool changes)
        self._tool_index: dict[str, ToolInfo] | None = None
        # Per-provider LLM tools cache (invalidated on tool changes)
        self._llm_tools_cache: dict[str, list[dict[str, Any]]] = {}
        # Progress callback (set during initialize)
        self._on_progress: Callable[[str], None] | None = None
        # MCP Apps host server (lazy-initialized)
        self._app_host: Any = None
        # Lock for OAuth flows (prevents concurrent browser-open races)
        self._oauth_lock = asyncio.Lock()

    # ================================================================
    # Initialization
    # ================================================================

    async def initialize(
        self,
        namespace: str = "stdio",
        on_progress: Callable[[str], None] | None = None,
    ) -> bool:
        """Initialize by parsing config, setting up OAuth, and creating StreamManager.

        Args:
            namespace: Default namespace for tools
            on_progress: Optional callback for progress updates during startup
        """
        self._on_progress = on_progress
        try:
            self._report_progress("Loading server configuration...")

            # Load config and detect server types (file I/O → off event loop)
            config = await asyncio.to_thread(self._config_loader.load)
            if not config:
                logger.warning("No config found, initializing with empty toolset")
                return await self._setup_empty_toolset()

            self._config_loader.detect_server_types(config)

            # Initialize StreamManager based on detected type
            success = await self._initialize_stream_manager(namespace)

            if success and self.stream_manager:
                # Get registry/processor from StreamManager if available
                if hasattr(self.stream_manager, "registry"):
                    self._registry = self.stream_manager.registry
                if hasattr(self.stream_manager, "processor"):
                    self.processor = self.stream_manager.processor

                # Build tool index for O(1) lookups
                await self._build_tool_index()

                logger.info("ToolManager initialized successfully")

            return success

        except Exception as e:
            logger.error(f"Failed to initialize ToolManager: {e}", exc_info=True)
            return False

    def _report_progress(self, message: str) -> None:
        """Report progress via callback if set."""
        if self._on_progress:
            self._on_progress(message)
        logger.debug(f"Progress: {message}")

    async def _initialize_stream_manager(self, namespace: str) -> bool:
        """Initialize StreamManager with detected transport type.

        ENHANCED: Initializes different transport types in parallel for faster startup.
        """
        self.stream_manager = StreamManager()

        http_servers = self._config_loader.http_servers
        sse_servers = self._config_loader.sse_servers
        stdio_servers = self._config_loader.stdio_servers

        if not (http_servers or sse_servers or stdio_servers):
            logger.info("No servers detected")
            return True

        try:
            # Build initialization tasks for parallel execution
            init_tasks: list[asyncio.Task[None]] = []
            task_names: list[str] = []

            # Create OAuth callback once (shared by HTTP and SSE)
            oauth_callback = self._config_loader.create_oauth_refresh_callback(
                http_servers, sse_servers
            )

            if http_servers:
                self._report_progress(
                    f"Connecting to {len(http_servers)} HTTP server(s)..."
                )
                logger.info(f"Preparing {len(http_servers)} HTTP servers for init")
                http_dicts = [
                    {"name": s.name, "url": s.url, "headers": s.headers or {}}
                    for s in http_servers
                ]
                task = asyncio.create_task(
                    self.stream_manager.initialize_with_http_streamable(
                        servers=http_dicts,
                        server_names=self.server_names,
                        initialization_timeout=self.initialization_timeout,
                        oauth_refresh_callback=oauth_callback,
                    ),
                    name="init_http",
                )
                init_tasks.append(task)
                task_names.append("HTTP")

            if sse_servers:
                self._report_progress(
                    f"Connecting to {len(sse_servers)} SSE server(s)..."
                )
                logger.info(f"Preparing {len(sse_servers)} SSE servers for init")
                sse_dicts = [
                    {"name": s.name, "url": s.url, "headers": s.headers or {}}
                    for s in sse_servers
                ]
                task = asyncio.create_task(
                    self.stream_manager.initialize_with_sse(
                        servers=sse_dicts,
                        server_names=self.server_names,
                        initialization_timeout=self.initialization_timeout,
                        oauth_refresh_callback=oauth_callback,
                    ),
                    name="init_sse",
                )
                init_tasks.append(task)
                task_names.append("SSE")

            if stdio_servers:
                self._report_progress(
                    f"Connecting to {len(stdio_servers)} STDIO server(s)..."
                )
                logger.info(f"Preparing {len(stdio_servers)} STDIO servers for init")
                stdio_dicts = [
                    {
                        "name": s.name,
                        "command": s.command,
                        "args": s.args,
                        "env": s.env,
                    }
                    for s in stdio_servers
                ]
                task = asyncio.create_task(
                    self.stream_manager.initialize_with_stdio(
                        servers=stdio_dicts,
                        server_names=self.server_names,
                        initialization_timeout=self.initialization_timeout,
                    ),
                    name="init_stdio",
                )
                init_tasks.append(task)
                task_names.append("STDIO")

            # Run all transport initializations in parallel
            if init_tasks:
                logger.info(
                    f"Starting parallel initialization of {len(init_tasks)} transport types: {', '.join(task_names)}"
                )
                results = await asyncio.gather(*init_tasks, return_exceptions=True)

                # Check for errors
                errors = []
                for name, result in zip(task_names, results):
                    if isinstance(result, Exception):
                        errors.append(f"{name}: {result}")
                        logger.error(f"{name} initialization failed: {result}")

                if errors:
                    # Log errors but don't fail if at least one transport succeeded
                    logger.warning(
                        f"Some transports failed to initialize: {'; '.join(errors)}"
                    )

                logger.info("Parallel server initialization complete")

                # Report how many tools were loaded
                try:
                    all_tools = self.stream_manager.get_all_tools()
                    tool_count = len(all_tools) if all_tools else 0
                    server_count = len(task_names) - len(errors)
                    self._report_progress(
                        f"Initialized {tool_count} tools from {server_count} server(s)"
                    )
                except Exception as e:
                    logger.debug("Post-init tool count report failed: %s", e)

            # Enable middleware if configured (retry, circuit breaker, rate limiting)
            if self._middleware_enabled and self.stream_manager:
                self.stream_manager.enable_middleware(self._middleware_config)
                logger.info(
                    "CTP middleware enabled (retry, circuit breaker, rate limiting)"
                )

            return True

        except Exception as e:
            logger.error(f"StreamManager initialization failed: {e}", exc_info=True)
            return False

    async def _setup_empty_toolset(self) -> bool:
        """Setup empty toolset when no servers configured."""
        logger.info("Setting up empty toolset")
        self.stream_manager = None
        self._registry = None
        self.processor = None
        return True

    @property
    def app_host(self) -> Any:
        """Get or create the MCP Apps host server (lazy-initialized)."""
        if self._app_host is None:
            from mcp_cli.apps.host import AppHostServer

            self._app_host = AppHostServer(self)
        return self._app_host

    async def close(self) -> None:
        """Close StreamManager and cleanup."""
        if self._app_host is not None:
            try:
                await self._app_host.close_all()
            except Exception as e:
                logger.warning(f"Error closing app host: {e}")

        if self.stream_manager:
            try:
                await self.stream_manager.close()
            except Exception as e:
                logger.warning(f"Error closing stream_manager: {e}")

    # ================================================================
    # Tool Access (with ToolInfo conversion)
    # ================================================================

    async def get_all_tools(self) -> list[ToolInfo]:
        """Get all tools, converted to mcp-cli ToolInfo model."""
        # Support direct registry access for tests (when stream_manager is None)
        if self._registry is not None and self.stream_manager is None:  # type: ignore[unreachable]
            try:  # type: ignore[unreachable]
                registry_items = await self._registry.list_tools()
                tools = []
                for tool_info in registry_items:
                    ns = tool_info.namespace
                    name = tool_info.name
                    try:
                        metadata = await self._registry.get_metadata(name, ns)
                    except Exception as e:
                        logger.debug(f"Failed to get metadata for {name}: {e}")
                        metadata = None

                    tools.append(
                        ToolInfo(
                            name=name,
                            namespace=ns,
                            description=metadata.description if metadata else "",
                            parameters=metadata.argument_schema if metadata else {},
                            is_async=metadata.is_async if metadata else False,
                            tags=list(metadata.tags) if metadata else [],
                        )
                    )
                return tools
            except Exception as e:
                logger.error(f"Error getting tools from registry: {e}")
                return []

        if not self.stream_manager:
            return []

        try:
            tools_dict = self.stream_manager.get_all_tools()
            # Get tool→server mapping for correct namespace
            tool_to_server = getattr(self.stream_manager, "tool_to_server_map", {})
            tools = []
            for t in tools_dict:
                tool_info = self._convert_to_tool_info(t)
                # Override namespace with server name from tool_to_server_map
                server_name = tool_to_server.get(tool_info.name)
                if server_name:
                    tool_info = ToolInfo(
                        name=tool_info.name,
                        namespace=server_name,
                        description=tool_info.description,
                        parameters=tool_info.parameters,
                        is_async=tool_info.is_async,
                        tags=tool_info.tags,
                        supports_streaming=tool_info.supports_streaming,
                        meta=tool_info.meta,
                    )
                tools.append(tool_info)
            return tools
        except Exception as e:
            logger.error(f"Error getting tools: {e}")
            return []

    async def get_unique_tools(self) -> list[ToolInfo]:
        """Get tools with duplicates removed (by name)."""
        all_tools = await self.get_all_tools()

        # De-duplicate by name, preferring non-default namespaces
        seen = {}
        for tool in all_tools:
            if tool.name not in seen or tool.namespace != "default":
                seen[tool.name] = tool

        return list(seen.values())

    async def get_tool_by_name(
        self, tool_name: str, namespace: str | None = None
    ) -> ToolInfo | None:
        """Get a tool by name, optionally filtering by namespace.

        Uses O(1) index lookup instead of scanning all tools.
        """
        if self._tool_index is None:
            await self._build_tool_index()

        if (
            self._tool_index is None
        ):  # pragma: no cover — unreachable after _build_tool_index
            return None

        if namespace:
            # Prefer fully-qualified lookup
            fqn = f"{namespace}.{tool_name}"
            tool = self._tool_index.get(fqn)
            if tool:
                return tool
            # Fall back to name-only if namespace matches
            tool = self._tool_index.get(tool_name)
            if tool and tool.namespace == namespace:
                return tool
            return None

        return self._tool_index.get(tool_name)

    async def _build_tool_index(self) -> None:
        """Build the O(1) tool lookup index from all tools.

        Index keys:
        - tool.name → tool (prefers non-default namespaces)
        - tool.fully_qualified_name → tool (always, for namespaced lookups)
        """
        all_tools = await self.get_all_tools()
        index: dict[str, ToolInfo] = {}

        for tool in all_tools:
            # Always index by fully qualified name
            index[tool.fully_qualified_name] = tool

            # Index by simple name, preferring non-default namespaces
            if tool.name not in index or tool.namespace != "default":
                index[tool.name] = tool

        self._tool_index = index
        logger.debug(f"Built tool index with {len(index)} entries")

    def _invalidate_caches(self) -> None:
        """Invalidate tool index and LLM tools cache, forcing rebuild on next access."""
        self._tool_index = None
        self._llm_tools_cache.clear()

    @staticmethod
    def format_tool_response(response: Any) -> str:
        """Format a tool response for display."""
        import json

        # Handle list of text records (MCP format)
        if isinstance(response, list):
            from mcp_cli.llm.content_models import TextContent

            try:
                text_blocks = [
                    TextContent.model_validate(item)
                    for item in response
                    if isinstance(item, dict)
                    and item.get("type") == ContentBlockType.TEXT.value
                ]
                if text_blocks:
                    return "\n".join(block.text for block in text_blocks)
            except Exception as e:
                logger.debug("TextContent parse fallback: %s", e)

            if all(
                isinstance(item, dict)
                and item.get("type") == ContentBlockType.TEXT.value
                for item in response
            ):
                return "\n".join(item.get("text", "") for item in response)

            return json.dumps(response, indent=2)

        if isinstance(response, dict):
            return json.dumps(response, indent=2)

        return str(response)

    def _convert_to_tool_info(self, tool_dict: dict[str, Any]) -> ToolInfo:
        """Convert chuk tool dict to mcp-cli ToolInfo."""
        tool_input = ToolDefinitionInput.model_validate(tool_dict)

        return ToolInfo(
            name=tool_input.name,
            namespace=tool_input.namespace,
            description=tool_input.description,
            parameters=tool_input.inputSchema,
            is_async=tool_input.is_async,
            tags=tool_input.tags,
            meta=tool_input.meta,
        )

    # ================================================================
    # Tool Execution
    # ================================================================

    def _get_server_url(self, server_name: str) -> str | None:
        """Get the URL for an HTTP/SSE server by name.

        Args:
            server_name: Name of the server

        Returns:
            Server URL or None if not found or not an HTTP server
        """
        # Check HTTP servers
        for server in self._config_loader.http_servers:
            if server.name == server_name:
                return server.url

        # Check SSE servers
        for server in self._config_loader.sse_servers:
            if server.name == server_name:
                return server.url

        return None

    def _get_server_timeout(self, server_name: str | None) -> float:
        """Resolve tool timeout for a server: per-server → global → default.

        Args:
            server_name: Server namespace/name, or None for global default.

        Returns:
            Timeout in seconds.
        """
        if server_name:
            for server in (
                self._config_loader.http_servers
                + self._config_loader.sse_servers
                + self._config_loader.stdio_servers
            ):
                if server.name == server_name and server.tool_timeout is not None:
                    return server.tool_timeout
        return self.tool_timeout

    async def _handle_oauth_flow(self, server_name: str, server_url: str) -> bool:
        """Handle OAuth authentication flow for a server.

        Uses _oauth_lock to prevent concurrent flows from racing
        (e.g. multiple parallel tool calls all detecting OAuth errors).

        Args:
            server_name: Name of the server requiring OAuth
            server_url: URL of the server

        Returns:
            True if OAuth completed successfully, False otherwise
        """
        async with self._oauth_lock:
            try:
                from chuk_term.ui import output

                output.info(
                    f"🔐 OAuth authorization required for server: {server_name}"
                )
                output.info("Opening browser for authentication...")

                # Create token manager with mcp-cli service name
                token_manager = TokenManager(
                    backend=TokenStoreBackend.AUTO,
                    service_name="mcp-cli",
                )
                oauth_handler = OAuthHandler(token_manager=token_manager)

                # Clear any existing tokens - they're clearly invalid since the server
                # returned an OAuth error. This forces a fresh browser-based auth flow.
                oauth_handler.clear_tokens(server_name)
                logger.debug(
                    f"Cleared existing tokens for {server_name} to force re-auth"
                )

                # Perform MCP OAuth flow (discovers metadata, opens browser, gets tokens)
                tokens = await oauth_handler.ensure_authenticated_mcp(
                    server_name=server_name,
                    server_url=server_url,
                )

                if tokens and tokens.access_token:
                    # Also store in the format expected by the oauth_refresh_callback
                    # The refresh callback looks for "oauth:{server_name}" with StoredToken format
                    import json

                    stored = StoredToken(
                        token_type=TokenType.OAUTH,
                        name=server_name,
                        data={
                            "access_token": tokens.access_token,
                            "refresh_token": tokens.refresh_token,
                            "token_type": tokens.token_type,
                            "expires_in": tokens.expires_in,
                            "issued_at": tokens.issued_at,
                        },
                    )
                    token_manager.token_store._store_raw(
                        f"oauth:{server_name}", json.dumps(stored.model_dump())
                    )
                    logger.debug(
                        f"Stored OAuth token for refresh callback: oauth:{server_name}"
                    )

                    # Update transport headers with copy-on-write to avoid races
                    # with concurrent tool calls reading the headers dict.
                    if self.stream_manager and hasattr(
                        self.stream_manager, "transports"
                    ):
                        transport = self.stream_manager.transports.get(server_name)
                        if transport and hasattr(transport, "configured_headers"):
                            new_headers = dict(transport.configured_headers)
                            new_headers["Authorization"] = (
                                f"Bearer {tokens.access_token}"
                            )
                            transport.configured_headers = new_headers
                            logger.debug(f"Updated transport headers for {server_name}")

                    output.success(f"✅ Successfully authenticated with {server_name}")
                    logger.info(f"OAuth flow completed for {server_name}")
                    return True
                else:
                    output.error(
                        f"❌ OAuth flow did not return valid tokens for {server_name}"
                    )
                    return False

            except Exception as e:
                logger.error(f"OAuth flow failed for {server_name}: {e}", exc_info=True)
                try:
                    from chuk_term.ui import output

                    output.error(f"❌ OAuth authentication failed: {e}")
                except ImportError:
                    logger.debug("chuk_term.ui not available for error display")
                return False

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        namespace: str | None = None,
        timeout: float | None = None,
        _oauth_retry: bool = False,
    ) -> ToolCallResult:
        """Execute tool and return ToolCallResult.

        When middleware is enabled (default), CTP handles:
        - Retry with exponential backoff for transient errors
        - Circuit breaker pattern for failing servers
        - Rate limiting (if configured)

        OAuth handling:
        - If a tool fails with OAuth authorization error, automatically
          triggers the OAuth flow and retries the tool call once.
        """
        # Check if this is a dynamic tool
        if self.dynamic_tool_provider.is_dynamic_tool(tool_name):
            logger.info(f"Executing dynamic tool: {tool_name}")
            try:
                result = await self.dynamic_tool_provider.execute_dynamic_tool(
                    tool_name, arguments
                )
                return ToolCallResult(tool_name=tool_name, success=True, result=result)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Dynamic tool execution failed: {error_msg}")
                return ToolCallResult(
                    tool_name=tool_name, success=False, error=error_msg
                )

        # Regular MCP tool execution (middleware handles retries if enabled)
        if not self.stream_manager:
            return ToolCallResult(
                tool_name=tool_name, success=False, error="ToolManager not initialized"
            )

        try:
            result = await self.stream_manager.call_tool(
                tool_name=tool_name,
                arguments=arguments,
                server_name=namespace,
                timeout=timeout or self._get_server_timeout(namespace),
            )

            # Check if result contains an OAuth error (some servers return errors in content)
            # Only check results that are flagged as errors — scanning successful
            # payloads causes false positives (e.g. the number "401" in data).
            result_is_error = (
                getattr(result, "isError", False)
                or (isinstance(result, dict) and result.get("isError", False))
                or (isinstance(result, dict) and "error" in result)
            )
            result_str = str(result) if result_is_error else ""
            if result_str and _is_oauth_error(result_str) and not _oauth_retry:
                logger.info(f"OAuth error detected in tool result for {tool_name}")
                # Determine server name - use namespace or look up from tool
                server_name = namespace or await self.get_server_for_tool(tool_name)
                if server_name:
                    server_url = self._get_server_url(server_name)
                    if server_url:
                        if await self._handle_oauth_flow(server_name, server_url):
                            # Retry the tool call once after OAuth
                            logger.info(f"Retrying tool {tool_name} after OAuth")
                            return await self.execute_tool(
                                tool_name,
                                arguments,
                                namespace,
                                timeout,
                                _oauth_retry=True,
                            )

            return ToolCallResult(tool_name=tool_name, success=True, result=result)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Tool execution failed: {error_msg}")

            # Classify error for better diagnostics
            if self._is_connection_error(error_msg):
                server_name = namespace or await self.get_server_for_tool(tool_name)
                diag = await self._diagnose_server(server_name)
                logger.warning(
                    f"Connection error for tool {tool_name} "
                    f"(server: {server_name or 'unknown'}). {diag}"
                )
                if diag:
                    error_msg += f" ({diag})"

            # Check if this is an OAuth error and we haven't already retried
            if _is_oauth_error(error_msg) and not _oauth_retry:
                logger.info(
                    f"OAuth error detected for tool {tool_name}, attempting authentication"
                )

                # Determine server name - use namespace or look up from tool
                server_name = namespace or await self.get_server_for_tool(tool_name)
                if server_name:
                    server_url = self._get_server_url(server_name)
                    if server_url:
                        if await self._handle_oauth_flow(server_name, server_url):
                            # Retry the tool call once after OAuth
                            logger.info(f"Retrying tool {tool_name} after OAuth")
                            return await self.execute_tool(
                                tool_name,
                                arguments,
                                namespace,
                                timeout,
                                _oauth_retry=True,
                            )
                        else:
                            return ToolCallResult(
                                tool_name=tool_name,
                                success=False,
                                error=f"OAuth authentication failed for {server_name}. {error_msg}",
                            )
                    else:
                        logger.warning(f"Could not find URL for server {server_name}")
                else:
                    logger.warning(f"Could not determine server for tool {tool_name}")

            return ToolCallResult(tool_name=tool_name, success=False, error=error_msg)

    _CONNECTION_ERROR_PATTERNS = (
        "connection",
        "transport",
        "broken pipe",
        "eof",
        "reset by peer",
        "timed out",
        "refused",
    )

    def _is_connection_error(self, error: str) -> bool:
        """Check if an error message indicates a connection problem."""
        error_lower = error.lower()
        return any(pat in error_lower for pat in self._CONNECTION_ERROR_PATTERNS)

    async def _diagnose_server(self, server_name: str | None) -> str:
        """Run health check on a server and return a diagnostic string."""
        if not server_name or not self.stream_manager:
            return ""
        try:
            health = await self.stream_manager.health_check()
            info = health.get("transports", {}).get(server_name, {})
            status = info.get("status", "unknown")
            if status != "healthy":
                return f"Server {server_name} is {status}"
        except Exception as exc:
            logger.debug(f"Health check failed for {server_name}: {exc}")
        return ""

    async def check_server_health(
        self, server_name: str | None = None
    ) -> dict[str, Any]:
        """Check health of one or all servers.

        Returns a dict with per-server status:
          {"server_name": {"status": "healthy"|"unhealthy"|"timeout"|"error", ...}}
        """
        if not self.stream_manager:
            return {}
        try:
            health = await self.stream_manager.health_check()
            transports = health.get("transports", {})
            if server_name:
                info = transports.get(server_name)
                return {server_name: info} if info else {}
            return dict(transports)
        except Exception as exc:
            logger.error(f"Health check failed: {exc}")
            return {}

    async def stream_execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        namespace: str | None = None,
        timeout: float | None = None,
    ):
        """Stream tool execution results (for tools that support streaming)."""
        result = await self.execute_tool(tool_name, arguments, namespace, timeout)
        yield result

    async def execute_tools_parallel(
        self,
        calls: list[CTPToolCall],
        timeout: float | None = None,
        on_tool_start: Callable[[CTPToolCall], Awaitable[None]] | None = None,
        on_tool_result: Callable[[CTPToolResult], Awaitable[None]] | None = None,
        max_concurrency: int = 4,
    ) -> list[CTPToolResult]:
        """Execute multiple tool calls in parallel with optional callbacks."""
        return await _execute_tools_parallel(
            self, calls, timeout, on_tool_start, on_tool_result, max_concurrency
        )

    async def stream_execute_tools(
        self,
        calls: list[CTPToolCall],
        timeout: float | None = None,
        on_tool_start: Callable[[CTPToolCall], Awaitable[None]] | None = None,
        max_concurrency: int = 4,
    ) -> AsyncIterator[CTPToolResult]:
        """Execute multiple tool calls in parallel, yielding results as they complete."""
        async for result in _stream_execute_tools(
            self, calls, timeout, on_tool_start, max_concurrency
        ):
            yield result

    # ================================================================
    # Middleware Status
    # ================================================================

    def get_middleware_status(self) -> dict[str, Any] | None:
        """Get middleware status for diagnostics.

        Returns Pydantic model dict with retry, circuit breaker, and rate limiting status.
        """
        if not self.stream_manager:
            return None

        try:
            status = self.stream_manager.get_middleware_status()
            return status.model_dump() if status else None
        except Exception as e:
            logger.error(f"Error getting middleware status: {e}")
            return None

    @property
    def middleware_enabled(self) -> bool:
        """Check if middleware is currently enabled."""
        if not self.stream_manager:
            return False
        return bool(self.stream_manager.middleware_enabled)

    # ================================================================
    # LLM Integration (filtering + adaptation)
    # ================================================================

    async def get_tools_for_llm(self, provider: str = "openai") -> list[dict[str, Any]]:
        """Get tools filtered and validated for LLM.

        Results are cached per provider. Cache is invalidated on tool changes
        (disable/enable). Dynamic tools mode bypasses the cache.
        """
        try:
            # Check if dynamic tools mode is enabled (bypasses cache)
            dynamic_mode = os.environ.get("MCP_CLI_DYNAMIC_TOOLS") == "1"

            if dynamic_mode:
                dynamic_tools: list[dict[str, Any]] = (
                    self.dynamic_tool_provider.get_dynamic_tools()
                )
                logger.info(
                    f"Dynamic tools mode: Returning {len(dynamic_tools)} dynamic tools only"
                )
                return dynamic_tools

            # Check cache first
            if provider in self._llm_tools_cache:
                cached = self._llm_tools_cache[provider]
                logger.debug(f"Returning {len(cached)} cached tools for {provider}")
                return cached

            # Static mode: load all tools upfront
            all_tools = await self.get_all_tools()

            raw_tools: list[dict[str, Any]] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "No description provided",
                        "parameters": t.parameters
                        or {"type": "object", "properties": {}},
                    },
                }
                for t in all_tools
            ]

            # Apply include/exclude filtering from environment variables
            include_tools = os.environ.get("MCP_CLI_INCLUDE_TOOLS")
            exclude_tools = os.environ.get("MCP_CLI_EXCLUDE_TOOLS")

            if include_tools:
                include_set = {name.strip() for name in include_tools.split(",")}
                raw_tools = [
                    tool
                    for tool in raw_tools
                    if tool["function"]["name"] in include_set
                ]
                logger.info(
                    f"Filtered to {len(raw_tools)} tools using include list: {include_set}"
                )

            if exclude_tools:
                exclude_set = {name.strip() for name in exclude_tools.split(",")}
                raw_tools = [
                    tool
                    for tool in raw_tools
                    if tool["function"]["name"] not in exclude_set
                ]
                logger.info(
                    f"Filtered to {len(raw_tools)} tools using exclude list: {exclude_set}"
                )

            # Filter and validate for provider
            valid_tools, _ = self.tool_filter.filter_tools(raw_tools, provider=provider)

            logger.info(
                f"Returning {len(valid_tools)} tools for LLM after all filtering"
            )
            if len(valid_tools) <= 10:
                for tool in valid_tools:
                    logger.info(f"  - {tool['function']['name']}")

            # Cache the result for this provider
            self._llm_tools_cache[provider] = valid_tools

            return valid_tools

        except Exception as e:
            logger.error(f"Error getting LLM tools: {e}")
            return []

    async def get_adapted_tools_for_llm(
        self,
        provider: str = "openai",
        name_mapping: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Get tools adapted for LLM with name mapping."""
        tools = await self.get_tools_for_llm(provider)

        if name_mapping is None:
            mapping = {
                tool["function"]["name"]: tool["function"]["name"] for tool in tools
            }
        else:
            mapping = name_mapping

        return tools, mapping

    # ================================================================
    # Tool Filtering API (delegates to ToolFilter)
    # ================================================================

    def disable_tool(
        self, tool_name: str, reason: DisabledReason = DisabledReason.USER
    ) -> None:
        """Disable a tool."""
        self.tool_filter.disable_tool(tool_name, reason)
        self._invalidate_caches()

    def enable_tool(self, tool_name: str) -> None:
        """Enable a previously disabled tool."""
        self.tool_filter.enable_tool(tool_name)
        self._invalidate_caches()

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled."""
        return self.tool_filter.is_tool_enabled(tool_name)

    def get_disabled_tools(self) -> dict[str, str]:
        """Get all disabled tools with their reasons."""
        return self.tool_filter.get_disabled_tools()

    def set_auto_fix_enabled(self, enabled: bool) -> None:
        """Enable/disable auto-fix for tools."""
        self.tool_filter.set_auto_fix_enabled(enabled)

    def is_auto_fix_enabled(self) -> bool:
        """Check if auto-fix is enabled."""
        return self.tool_filter.is_auto_fix_enabled()

    def clear_validation_disabled_tools(self) -> None:
        """Clear all validation-disabled tools."""
        self.tool_filter.clear_validation_disabled()

    def get_validation_summary(self) -> dict[str, Any]:
        """Get validation summary."""
        return self.tool_filter.get_validation_summary()

    async def validate_single_tool(
        self, tool_name: str, provider: str = "openai"
    ) -> tuple[bool, str | None]:
        """Validate a single tool for LLM compatibility."""
        try:
            tools = await self.get_all_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if not tool:
                return False, f"Tool '{tool_name}' not found"

            tool_dict = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
            }

            valid_tools, invalid_tools = self.tool_filter.filter_tools(
                [tool_dict], provider=provider
            )
            if valid_tools:
                return True, None
            elif invalid_tools:
                return False, invalid_tools[0].get("error", "Unknown validation error")
            return False, "Tool validation failed"

        except Exception as e:
            return False, str(e)

    async def revalidate_tools(self, provider: str = "openai") -> dict[str, Any]:
        """Revalidate all tools and return summary."""
        try:
            tools = await self.get_all_tools()
            tool_dicts = [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.parameters,
                }
                for t in tools
            ]

            valid_tools, invalid_tools = self.tool_filter.filter_tools(
                tool_dicts, provider=provider
            )

            return {
                "total": len(tools),
                "valid": len(valid_tools),
                "invalid": len(invalid_tools),
                "invalid_tools": [t.get("name", "unknown") for t in invalid_tools],
            }

        except Exception as e:
            logger.error(f"Error revalidating tools: {e}")
            return {"total": 0, "valid": 0, "invalid": 0, "invalid_tools": []}

    def get_tool_validation_details(self, tool_name: str) -> dict[str, Any] | None:
        """Get validation details for a specific tool."""
        return {"name": tool_name, "status": "unknown"}

    # ================================================================
    # Server Info
    # ================================================================

    async def get_server_info(self) -> list[ServerInfo]:
        """Get information about connected servers."""
        if not self.stream_manager:
            return []

        try:
            servers = []
            server_id = 0

            http_servers = self._config_loader.http_servers
            sse_servers = self._config_loader.sse_servers
            stdio_servers = self._config_loader.stdio_servers
            all_servers = http_servers + sse_servers + stdio_servers

            # Get tool counts per server from StreamManager's tool_to_server_map
            # This is the authoritative source for tool→server mapping
            tool_counts: dict[str, int] = {}
            if hasattr(self.stream_manager, "tool_to_server_map"):
                for server_name in self.stream_manager.tool_to_server_map.values():
                    tool_counts[server_name] = tool_counts.get(server_name, 0) + 1

            for server in all_servers:
                server_name = server.name

                # Determine transport type and get transport-specific fields
                from mcp_cli.config.server_models import (
                    HTTPServerConfig,
                    STDIOServerConfig,
                )

                command: str | None = None
                url: str | None = None
                args: list[str] = []
                env: dict[str, str] = {}

                if server in http_servers:
                    transport = TransportType.HTTP
                    if isinstance(server, HTTPServerConfig):
                        url = server.url
                elif server in sse_servers:
                    transport = TransportType.SSE
                    if isinstance(server, HTTPServerConfig):
                        url = server.url
                else:
                    transport = TransportType.STDIO
                    if isinstance(server, STDIOServerConfig):
                        command = server.command
                        args = list(server.args)
                        env = dict(server.env)

                servers.append(
                    ServerInfo(
                        id=server_id,
                        name=server_name,
                        status=ServerStatus.CONNECTED.value,
                        tool_count=tool_counts.get(server_name, 0),
                        namespace=server_name,
                        enabled=not server.disabled,
                        connected=True,
                        transport=transport,
                        capabilities={},
                        command=command,
                        url=url,
                        args=args,
                        env=env,
                    )
                )
                server_id += 1

            return servers

        except Exception as e:
            logger.error(f"Error getting server info: {e}")
            return []

    async def get_server_for_tool(self, tool_name: str) -> str | None:
        """Get the server name for a given tool."""
        if not self.stream_manager:
            return None

        try:
            tools = await self.get_all_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            return tool.namespace if tool else None

        except Exception as e:
            logger.error(f"Error getting server for tool: {e}")
            return None

    def get_streams(self):
        """Get active streams from StreamManager."""
        if not self.stream_manager:
            return []

        try:
            if hasattr(self.stream_manager, "get_streams"):
                return self.stream_manager.get_streams()
            return []
        except Exception as e:
            logger.error(f"Error getting streams: {e}")
            return []

    def list_resources(self):
        """List available resources from servers."""
        if not self.stream_manager:
            return []

        try:
            if hasattr(self.stream_manager, "list_resources"):
                return self.stream_manager.list_resources()
            return []
        except Exception as e:
            logger.error(f"Error listing resources: {e}")
            return []

    async def read_resource(
        self, uri: str, server_name: str | None = None
    ) -> dict[str, Any]:
        """Read a resource by URI.

        Args:
            uri: Resource URI to read (e.g. ui://tool-demo/chart)
            server_name: Optional server name to target

        Returns:
            Resource content dict from the server.
        """
        if not self.stream_manager:
            logger.debug("read_resource: no stream_manager available")
            return {}

        try:
            result: dict[str, Any] = await self.stream_manager.read_resource(
                uri, server_name
            )
            if not result:
                logger.debug(
                    "read_resource returned empty for %s (server=%s)", uri, server_name
                )
            return result
        except Exception as e:
            logger.error("Error reading resource %s from %s: %s", uri, server_name, e)
            return {}

    def list_prompts(self):
        """List available prompts from servers."""
        if not self.stream_manager:
            return []

        try:
            if hasattr(self.stream_manager, "list_prompts"):
                return self.stream_manager.list_prompts()
            return []
        except Exception as e:
            logger.error(f"Error listing prompts: {e}")
            return []
