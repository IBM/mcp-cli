# src/mcp_cli/commands/definitions/token.py
"""
Unified token command implementation with all sub-actions.
"""

from __future__ import annotations

import json
import logging

from chuk_term.ui import output, format_table
from mcp_cli.auth import TokenManager, TokenStoreBackend, TokenStoreFactory
from mcp_cli.auth import APIKeyToken, BearerToken, TokenType
from mcp_cli.config.config_manager import get_config
from mcp_cli.config.enums import TokenNamespace
from mcp_cli.config import NAMESPACE, OAUTH_NAMESPACE, GENERIC_NAMESPACE
from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from mcp_cli.commands.models import (
    TokenListParams,
    TokenSetParams,
    TokenDeleteParams,
    TokenClearParams,
    TokenProviderParams,
)

logger = logging.getLogger(__name__)


def _get_token_manager() -> TokenManager:
    """Get configured token manager instance with mcp-cli namespace."""
    import os

    # Check for CLI override first
    backend_override = os.environ.get("MCP_CLI_TOKEN_BACKEND")
    if backend_override:
        try:
            backend = TokenStoreBackend(backend_override)
        except (ValueError, KeyError):
            # Invalid backend specified, fall through to config
            backend = None
    else:
        backend = None

    # If no override or invalid override, check config
    if backend is None:
        try:
            config = get_config()
            backend = TokenStoreBackend(config.token_store_backend)
        except Exception as e:
            logger.debug("Failed to read token backend from config: %s", e)
            backend = TokenStoreBackend.AUTO

    return TokenManager(backend=backend, namespace=NAMESPACE, service_name="mcp-cli")


class TokenCommand(UnifiedCommand):
    """Manage OAuth and authentication tokens."""

    @property
    def name(self) -> str:
        return "token"

    @property
    def aliases(self) -> list[str]:
        return ["tokens"]

    @property
    def description(self) -> str:
        return "Manage OAuth and authentication tokens"

    @property
    def help_text(self) -> str:
        return """
Manage OAuth and authentication tokens.

Usage:
  /token              - List all stored tokens (chat/interactive mode)
  /token list         - List all stored tokens
  /token set <name> <value> - Store a bearer token
  /token get <name>   - Get details for a specific token
  /token clear        - Clear all tokens (with confirmation)
  /token clear --force - Clear all tokens without confirmation
  /token delete <name> - Delete a specific token

Examples:
  /token              # Show all tokens
  /token list         # Show all tokens
  /token set my-api secret-token  # Store a bearer token
  /token get my-api   # Show token details
  /token get notion   # Show notion OAuth token details
  /token clear        # Clear all tokens (asks for confirmation)
  /token delete my-api # Delete the token
"""

    @property
    def parameters(self) -> list[CommandParameter]:
        """Define parameters for token command."""
        return [
            CommandParameter(
                name="action",
                type=str,
                required=False,
                help="Action: list, set, get, delete, clear, backends, set-provider, get-provider, delete-provider",
            ),
            CommandParameter(
                name="name",
                type=str,
                required=False,
                help="Token/provider name",
            ),
            CommandParameter(
                name="value",
                type=str,
                required=False,
                help="Token value (for set action)",
            ),
            CommandParameter(
                name="token_type",
                type=str,
                default="bearer",
                help="Token type: bearer, api-key, generic",
            ),
            CommandParameter(
                name="provider",
                type=str,
                required=False,
                help="Provider name (for API keys)",
            ),
            CommandParameter(
                name="namespace",
                type=str,
                required=False,
                help="Storage namespace",
            ),
            CommandParameter(
                name="show_oauth",
                type=bool,
                default=True,
                help="Show OAuth tokens",
                is_flag=True,
            ),
            CommandParameter(
                name="show_bearer",
                type=bool,
                default=True,
                help="Show bearer tokens",
                is_flag=True,
            ),
            CommandParameter(
                name="show_api_keys",
                type=bool,
                default=True,
                help="Show API keys",
                is_flag=True,
            ),
            CommandParameter(
                name="show_providers",
                type=bool,
                default=True,
                help="Show provider tokens",
                is_flag=True,
            ),
            CommandParameter(
                name="is_oauth",
                type=bool,
                default=False,
                help="Delete OAuth token",
                is_flag=True,
            ),
            CommandParameter(
                name="force",
                type=bool,
                default=False,
                help="Skip confirmation (clear)",
                is_flag=True,
            ),
            CommandParameter(
                name="api_key",
                type=str,
                required=False,
                help="API key value (for set-provider)",
            ),
        ]

    @property
    def modes(self) -> CommandMode:
        """Token is available in all modes."""
        return CommandMode.CLI | CommandMode.CHAT | CommandMode.INTERACTIVE

    @property
    def requires_context(self) -> bool:
        """Token needs context for server list."""
        return True

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the token command with all sub-actions."""
        # Get action from kwargs
        action = kwargs.get("action", "list")

        # Handle args array for chat/interactive mode
        args = kwargs.get("args", [])
        if isinstance(args, str):
            args = [args]
        elif args and len(args) > 0:
            # First arg is the action in chat mode
            action = args[0].lower()

        # Get tool_manager for server list
        tool_manager = kwargs.get("tool_manager")
        server_names = tool_manager.servers if tool_manager else []

        # Route to appropriate sub-action
        from mcp_cli.config import TokenAction

        try:
            if not action or action == TokenAction.LIST.value:
                return await self._action_list(kwargs, server_names)
            elif action == TokenAction.SET.value:
                return await self._action_set(kwargs, args)
            elif action == TokenAction.GET.value:
                return await self._action_get(kwargs, args)
            elif action == TokenAction.DELETE.value:
                return await self._action_delete(kwargs, args)
            elif action == TokenAction.CLEAR.value:
                return await self._action_clear(kwargs, args)
            elif action == TokenAction.BACKENDS.value:
                return await self._action_backends()
            elif action == TokenAction.SET_PROVIDER.value:
                return await self._action_set_provider(kwargs)
            elif action == TokenAction.GET_PROVIDER.value:
                return await self._action_get_provider(kwargs)
            elif action == TokenAction.DELETE_PROVIDER.value:
                return await self._action_delete_provider(kwargs)
            else:
                return CommandResult(
                    success=False,
                    error=f"Unknown token action: {action}. Valid actions: list, set, get, delete, clear, backends, set-provider, get-provider, delete-provider",
                )
        except Exception as e:
            return CommandResult(success=False, error=f"Token command error: {e}")

    async def _action_list(
        self, kwargs: dict, server_names: list[str]
    ) -> CommandResult:
        """List all stored tokens."""
        params = TokenListParams(
            namespace=kwargs.get("namespace"),
            show_oauth=kwargs.get("show_oauth", True),
            show_bearer=kwargs.get("show_bearer", True),
            show_api_keys=kwargs.get("show_api_keys", True),
            show_providers=kwargs.get("show_providers", True),
            server_names=server_names,
        )

        try:
            manager = _get_token_manager()

            output.rule("[bold]🔐 Stored Tokens[/bold]", style="primary")

            # Track if we showed any tokens at all
            provider_tokens = {}
            oauth_entries = []

            # Show provider tokens with hierarchical status
            if params.show_providers and (
                params.namespace is None or params.namespace == TokenNamespace.PROVIDER
            ):
                from mcp_cli.auth.provider_tokens import list_all_provider_tokens

                provider_tokens = list_all_provider_tokens(manager)

                if provider_tokens:
                    output.print(
                        "\n[bold]Provider API Keys (Stored in Secure Storage):[/bold]"
                    )
                    provider_table_data = []

                    for provider_name, status_info in provider_tokens.items():
                        env_var = status_info["env_var"]
                        status_display = "🔐 storage"

                        if status_info["in_env"]:
                            note = f"(overridden by {env_var})"
                        else:
                            note = "active"

                        provider_table_data.append(
                            {
                                "Provider": provider_name,
                                "Status": status_display,
                                "Env Var": env_var,
                                "Note": note,
                            }
                        )

                    provider_table = format_table(
                        provider_table_data,
                        title=None,
                        columns=["Provider", "Status", "Env Var", "Note"],
                    )
                    output.print_table(provider_table)
                    output.info(
                        "💡 Environment variables take precedence over stored tokens"
                    )
                    output.print()

            # List OAuth tokens
            if params.show_oauth and params.server_names:
                for server_name in params.server_names:
                    tokens = manager.load_tokens(server_name)
                    if tokens:
                        metadata = {}
                        if tokens.expires_in:
                            import time

                            if tokens.issued_at:
                                metadata["expires_at"] = (
                                    tokens.issued_at + tokens.expires_in
                                )
                            else:
                                metadata["expires_at"] = time.time() + tokens.expires_in

                        oauth_entries.append(
                            {
                                "name": server_name,
                                "type": "oauth",
                                "namespace": OAUTH_NAMESPACE,
                                "registered_at": tokens.issued_at
                                if tokens.issued_at
                                else None,
                                "metadata": metadata,
                            }
                        )

                if oauth_entries:
                    output.print("\n[bold]OAuth Tokens (Server Authentication):[/bold]")
                    oauth_table_data = []

                    for entry in oauth_entries:
                        import time
                        from datetime import datetime

                        token_name = entry.get("name", "unknown")
                        token_type = entry.get("type", "unknown")

                        registered_at = entry.get("registered_at")
                        created = "-"
                        if registered_at and isinstance(registered_at, (int, float)):
                            dt = datetime.fromtimestamp(registered_at)
                            created = dt.strftime("%Y-%m-%d")

                        metadata_raw = entry.get("metadata", {})
                        metadata = (
                            metadata_raw if isinstance(metadata_raw, dict) else {}
                        )
                        expires = metadata.get("expires_at", "-")
                        if expires != "-" and isinstance(expires, (int, float)):
                            exp_dt = datetime.fromtimestamp(expires)
                            if time.time() > expires:
                                expires = f"{exp_dt.strftime('%Y-%m-%d')} ⚠️ Expired"
                            else:
                                expires = exp_dt.strftime("%Y-%m-%d")

                        oauth_table_data.append(
                            {
                                "Server": token_name,
                                "Type": token_type,
                                "Created": created,
                                "Expires": expires,
                            }
                        )

                    oauth_table = format_table(
                        oauth_table_data,
                        title=None,
                        columns=["Server", "Type", "Created", "Expires"],
                    )
                    output.print_table(oauth_table)
                    output.info("💡 Use '/token get <server>' to view token details")
                    output.print()
            elif params.show_oauth and not params.server_names:
                output.info(
                    "No servers configured. OAuth tokens are stored per server."
                )
                output.print()

            # List tokens from registry
            registry = manager.registry
            registered_tokens = registry.list_tokens(namespace=params.namespace)

            table_data = []
            for entry in registered_tokens:
                token_type = entry.get("type", "unknown")
                token_name = entry.get("name", "unknown")
                token_namespace = entry.get("namespace", "unknown")

                if params.show_providers and token_namespace == TokenNamespace.PROVIDER:
                    continue
                if params.show_oauth and token_namespace == OAUTH_NAMESPACE:
                    continue

                if token_type == TokenType.BEARER.value and not params.show_bearer:
                    continue
                if token_type == TokenType.API_KEY.value and not params.show_api_keys:
                    continue

                import time
                from datetime import datetime

                registered_at = entry.get("registered_at")
                created = "-"
                if registered_at and isinstance(registered_at, (int, float)):
                    dt = datetime.fromtimestamp(registered_at)
                    created = dt.strftime("%Y-%m-%d")

                metadata_raw = entry.get("metadata", {})
                metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
                expires = metadata.get("expires_at", "-")
                if expires != "-" and isinstance(expires, (int, float)):
                    exp_dt = datetime.fromtimestamp(expires)
                    if time.time() > expires:
                        expires = f"{exp_dt.strftime('%Y-%m-%d')} ⚠️"
                    else:
                        expires = exp_dt.strftime("%Y-%m-%d")

                details = []
                if metadata.get("provider"):
                    details.append(f"provider={metadata['provider']}")
                if token_namespace != TokenNamespace.GENERIC:
                    details.append(f"ns={token_namespace}")

                table_data.append(
                    {
                        "Type": token_type,
                        "Name": token_name,
                        "Created": created,
                        "Expires": expires,
                        "Details": ", ".join(details) if details else "-",
                    }
                )

            if table_data:
                output.print("\n[bold]Other Tokens:[/bold]")
                table = format_table(
                    table_data,
                    title=None,
                    columns=["Type", "Name", "Created", "Expires", "Details"],
                )
                output.print_table(table)
            elif not provider_tokens and not oauth_entries:
                output.warning("No tokens found.")

            output.print()
            output.tip("💡 Token Management:")
            output.info("  • Store provider key: mcp-cli token set-provider <provider>")
            output.info(
                "  • Store bearer token: mcp-cli token set <name> --type bearer"
            )
            output.info("  • View: mcp-cli token get <name>")
            output.info("  • Delete: mcp-cli token delete <name>")

            return CommandResult(success=True)

        except Exception as e:
            return CommandResult(success=False, error=f"Error listing tokens: {e}")

    async def _action_set(self, kwargs: dict, args: list[str]) -> CommandResult:
        """Store a token."""
        # Get parameters from kwargs or args
        name = kwargs.get("name")
        value = kwargs.get("value")

        # For chat mode, parse from args
        if args and len(args) >= 3:
            name = args[1]
            value = args[2]

        if not name:
            return CommandResult(success=False, error="Token name is required")

        params = TokenSetParams(
            name=name,
            value=value,
            token_type=kwargs.get("token_type", "bearer"),
            provider=kwargs.get("provider"),
            namespace=kwargs.get("namespace") or GENERIC_NAMESPACE,
        )

        try:
            manager = _get_token_manager()
            store = manager.token_store

            # Prompt for value if not provided
            if params.value is None:
                from getpass import getpass

                params.value = getpass(f"Enter token value for '{params.name}': ")

            if not params.value:
                return CommandResult(success=False, error="Token value is required")

            registry = manager.registry

            # Normalize token_type: CLI uses hyphens (api-key) but
            # TokenType enum uses underscores (api_key)
            normalized_type = params.token_type.replace("-", "_")

            if normalized_type == TokenType.BEARER.value:
                bearer = BearerToken(token=params.value)
                stored = bearer.to_stored_token(params.name)
                stored.metadata = {"namespace": params.namespace}
                store._store_raw(
                    f"{params.namespace}:{params.name}", json.dumps(stored.model_dump())
                )

                reg_metadata = {}
                if bearer.expires_at:
                    reg_metadata["expires_at"] = bearer.expires_at

                registry.register(
                    params.name,
                    TokenType.BEARER,
                    params.namespace,
                    metadata=reg_metadata,
                )
                output.success(f"Bearer token '{params.name}' stored successfully")

            elif normalized_type == TokenType.API_KEY.value:
                if not params.provider:
                    return CommandResult(
                        success=False,
                        error="Provider name is required for API keys. Use: --provider <name>",
                    )

                api_key = APIKeyToken(provider=params.provider, key=params.value)
                stored = api_key.to_stored_token(params.name)
                stored.metadata = {"namespace": params.namespace}
                store._store_raw(
                    f"{params.namespace}:{params.name}", json.dumps(stored.model_dump())
                )

                registry.register(
                    params.name,
                    TokenType.API_KEY,
                    params.namespace,
                    metadata={"provider": params.provider},
                )
                output.success(
                    f"API key '{params.name}' for '{params.provider}' stored successfully"
                )

            elif normalized_type == TokenNamespace.GENERIC:
                store.store_generic(params.name, params.value, params.namespace)
                registry.register(
                    params.name, TokenType.BEARER, params.namespace, metadata={}
                )
                output.success(
                    f"Token '{params.name}' stored in namespace '{params.namespace}'"
                )

            else:
                return CommandResult(
                    success=False,
                    error=f"Unknown token type: {params.token_type}. Valid types: bearer, api-key, generic",
                )

            return CommandResult(success=True)

        except Exception as e:
            return CommandResult(success=False, error=f"Error storing token: {e}")

    async def _action_get(self, kwargs: dict, args: list[str]) -> CommandResult:
        """Get information about a stored token."""
        name = kwargs.get("name")

        # For chat mode, parse from args
        if args and len(args) >= 2:
            name = args[1]

        if not name:
            return CommandResult(success=False, error="Token name is required")

        namespace = kwargs.get("namespace", "generic")

        try:
            manager = _get_token_manager()
            store = manager.token_store

            raw_data = store._retrieve_raw(f"{namespace}:{name}")
            if not raw_data:
                # Try OAuth namespace
                raw_data = store._retrieve_raw(f"{OAUTH_NAMESPACE}:{name}")
                if raw_data:
                    namespace = OAUTH_NAMESPACE

            if not raw_data:
                output.warning(f"Token '{name}' not found")
                return CommandResult(success=False)

            try:
                from mcp_cli.auth import StoredToken

                stored = StoredToken.model_validate(json.loads(raw_data))
                info = stored.get_display_info()

                output.rule(f"[bold]Token: {name}[/bold]", style="primary")
                output.info(f"Type: {stored.token_type.value}")
                output.info(f"Namespace: {namespace}")

                for key, value in info.items():
                    if key not in ["name", "type"]:
                        output.info(f"{key}: {value}")

            except Exception as e:
                output.warning(f"Could not parse token data: {e}")

            return CommandResult(success=True)

        except Exception as e:
            return CommandResult(success=False, error=f"Error retrieving token: {e}")

    async def _action_delete(self, kwargs: dict, args: list[str]) -> CommandResult:
        """Delete a stored token."""
        name = kwargs.get("name")

        # For chat mode, parse from args
        if args and len(args) >= 2:
            name = args[1]

        if not name:
            return CommandResult(success=False, error="Token name is required")

        params = TokenDeleteParams(
            name=name,
            namespace=kwargs.get("namespace"),
            oauth=kwargs.get("is_oauth", False),
        )

        try:
            manager = _get_token_manager()
            store = manager.token_store
            registry = manager.registry

            if params.oauth:
                if manager.delete_tokens(params.name):
                    output.success(f"OAuth token for server '{params.name}' deleted")
                else:
                    output.warning(f"OAuth token for server '{params.name}' not found")
                return CommandResult(success=True)

            # Delete generic token
            if params.namespace:
                namespaces = [params.namespace]
            else:
                namespaces = [
                    TokenNamespace.BEARER,
                    TokenNamespace.API_KEY,
                    TokenNamespace.PROVIDER,
                    TokenNamespace.GENERIC,
                ]

            deleted = False
            for ns in namespaces:
                if store.delete_generic(params.name, ns):
                    registry.unregister(params.name, ns)
                    output.success(
                        f"Token '{params.name}' deleted from namespace '{ns}'"
                    )
                    deleted = True
                    break

            if not deleted:
                output.warning(f"Token '{params.name}' not found")
                return CommandResult(success=False)

            return CommandResult(success=True)

        except Exception as e:
            return CommandResult(success=False, error=f"Error deleting token: {e}")

    async def _action_clear(self, kwargs: dict, args: list[str]) -> CommandResult:
        """Clear all stored tokens."""
        force = kwargs.get("force", False) or "--force" in args or "-f" in args

        params = TokenClearParams(
            namespace=kwargs.get("namespace"),
            force=force,
        )

        try:
            manager = _get_token_manager()
            store = manager.token_store
            registry = manager.registry

            # Confirm before clearing
            if not params.force:
                if params.namespace:
                    msg = f"Clear all tokens in namespace '{params.namespace}'?"
                else:
                    msg = "Clear ALL tokens from ALL namespaces?"

                from chuk_term.ui.prompts import confirm

                if not confirm(msg):
                    output.warning("Cancelled")
                    return CommandResult(success=False)

            # Get tokens to clear from registry
            tokens_to_clear = registry.list_tokens(namespace=params.namespace)

            if not tokens_to_clear:
                output.warning("No tokens to clear")
                return CommandResult(success=True)

            # Clear each token from storage
            count = 0
            for entry in tokens_to_clear:
                token_name = entry.get("name")
                token_namespace = entry.get("namespace")
                if (
                    token_name
                    and token_namespace
                    and store.delete_generic(token_name, token_namespace)
                ):
                    count += 1

            # Clear from registry
            if params.namespace:
                registry.clear_namespace(params.namespace)
            else:
                registry.clear_all()

            if count > 0:
                output.success(f"Cleared {count} token(s)")
            else:
                output.warning("No tokens to clear")

            return CommandResult(success=True)

        except Exception as e:
            return CommandResult(success=False, error=f"Error clearing tokens: {e}")

    async def _action_backends(self) -> CommandResult:
        """List available token storage backends."""
        import os

        try:
            available = TokenStoreFactory.get_available_backends()

            backend_override = os.environ.get("MCP_CLI_TOKEN_BACKEND")
            override_succeeded = False
            if backend_override:
                try:
                    detected = TokenStoreBackend(backend_override)
                    override_succeeded = True
                except (ValueError, KeyError):
                    detected = TokenStoreFactory._detect_backend()
                    output.warning(
                        f"Invalid backend '{backend_override}', using auto-detected backend"
                    )
            else:
                detected = TokenStoreFactory._detect_backend()

            output.rule("[bold]🔒 Token Storage Backends[/bold]", style="primary")

            all_backends = [
                ("keychain", "macOS Keychain"),
                ("windows", "Windows Credential Manager"),
                ("secretservice", "Linux Secret Service"),
                ("vault", "HashiCorp Vault"),
                ("encrypted", "Encrypted File Storage"),
            ]

            table_data = []
            for backend_id, backend_name in all_backends:
                backend = TokenStoreBackend(backend_id)
                is_available = backend in available
                is_detected = backend == detected

                status = []
                if is_detected:
                    status.append("🎯 Auto-detected")
                if is_available:
                    status.append("✓")

                table_data.append(
                    {
                        "Backend": backend_name,
                        "Available": "✓" if is_available else "✗",
                        "Status": " ".join(status) if status else "-",
                    }
                )

            table = format_table(
                table_data, title=None, columns=["Backend", "Available", "Status"]
            )
            output.print_table(table)
            output.print()
            if override_succeeded:
                output.info(
                    f"Current backend: {detected.value} (overridden via --token-backend)"
                )
            else:
                output.info(f"Current backend: {detected.value}")

            return CommandResult(success=True)

        except Exception as e:
            return CommandResult(success=False, error=f"Error listing backends: {e}")

    async def _action_set_provider(self, kwargs: dict) -> CommandResult:
        """Store a provider API key."""
        provider = kwargs.get("provider")
        if not provider:
            return CommandResult(success=False, error="Provider name is required")

        params = TokenProviderParams(
            provider=provider,
            api_key=kwargs.get("api_key"),
        )

        try:
            from mcp_cli.auth.provider_tokens import (
                set_provider_token,
                get_provider_env_var_name,
            )
            import os

            manager = _get_token_manager()

            # Prompt for api_key if not provided
            api_key = params.api_key
            if api_key is None:
                from getpass import getpass

                api_key = getpass(f"Enter API key for '{params.provider}': ")

            if not api_key:
                return CommandResult(success=False, error="API key is required")

            # Store the token
            if set_provider_token(params.provider, api_key, manager):
                output.success(f"✅ Stored API key for provider '{params.provider}'")

                # Show hierarchy info
                env_var = get_provider_env_var_name(params.provider)
                output.print()
                output.info("📋 Token Hierarchy:")
                output.info(f"  1. Environment variable: {env_var} (highest priority)")
                output.info("  2. Secure storage: 🔐 (currently set)")

                # Check if env var is also set
                if os.environ.get(env_var):
                    output.warning(
                        f"\n⚠️  Note: {env_var} is set in environment and will take precedence"
                    )
                return CommandResult(success=True)
            else:
                return CommandResult(
                    success=False,
                    error=f"Failed to store API key for provider '{params.provider}'",
                )

        except Exception as e:
            return CommandResult(
                success=False, error=f"Error storing provider token: {e}"
            )

    async def _action_get_provider(self, kwargs: dict) -> CommandResult:
        """Get information about a provider's API key."""
        provider = kwargs.get("provider")
        if not provider:
            return CommandResult(success=False, error="Provider name is required")

        params = TokenProviderParams(provider=provider)

        try:
            from mcp_cli.auth.provider_tokens import check_provider_token_status

            manager = _get_token_manager()
            status = check_provider_token_status(params.provider, manager)

            output.rule(
                f"[bold]Provider Token: {params.provider}[/bold]", style="primary"
            )

            if status["has_token"]:
                output.success("✅ API key is configured")
                output.info(f"   Source: {status['source']}")
            else:
                output.warning("❌ No API key configured")

            output.print()
            output.info("Token Status:")
            output.info(
                f"  • Environment variable ({status['env_var']}): {'✅ set' if status['in_env'] else '❌ not set'}"
            )
            output.info(
                f"  • Secure storage: {'✅ set' if status['in_storage'] else '❌ not set'}"
            )

            output.print()
            output.tip("Hierarchy: Environment variables take precedence over storage")

            if not status["has_token"]:
                output.print()
                output.info("To set API key:")
                output.info(
                    f"  • Via storage: mcp-cli token set-provider {params.provider}"
                )
                output.info(f"  • Via environment: export {status['env_var']}=your-key")

            return CommandResult(success=True)

        except Exception as e:
            return CommandResult(
                success=False, error=f"Error retrieving provider token info: {e}"
            )

    async def _action_delete_provider(self, kwargs: dict) -> CommandResult:
        """Delete a provider API key from secure storage."""
        provider = kwargs.get("provider")
        if not provider:
            return CommandResult(success=False, error="Provider name is required")

        try:
            from mcp_cli.auth.provider_tokens import delete_provider_token

            manager = _get_token_manager()

            if delete_provider_token(provider, manager):
                output.success(f"Deleted API key for provider '{provider}'")
                return CommandResult(success=True)
            else:
                output.warning(f"No API key found for provider '{provider}'")
                return CommandResult(success=False)

        except Exception as e:
            return CommandResult(
                success=False, error=f"Error deleting provider token: {e}"
            )
