# mcp_cli/chat/chat_context.py
"""
Chat context using chuk-ai-session-manager as the native conversation backend.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any, AsyncIterator

from mcp_cli.chat.system_prompt import generate_system_prompt
from mcp_cli.chat.models import HistoryMessage, MessageRole, ChatStatus, ServerToolGroup
from mcp_cli.chat.token_tracker import TokenTracker
from mcp_cli.chat.session_store import SessionStore, SessionData, SessionMetadata
from mcp_cli.tools.manager import ToolManager
from mcp_cli.tools.models import ToolInfo, ServerInfo
from mcp_cli.model_management import ModelManager

# Native session management from chuk-ai-session-manager
from chuk_ai_session_manager import SessionManager
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.procedural_memory import (
    ToolMemoryManager,
    ToolOutcome,
    ProceduralContextFormatter,
    FormatterConfig,
)
from chuk_ai_session_manager.memory.models import VMMode
from chuk_ai_session_manager.memory.working_set import WorkingSetConfig

logger = logging.getLogger(__name__)


class ChatContext:
    """
    Chat context with SessionManager as the native conversation backend.

    SessionManager is required - no fallback to local state.
    All conversation tracking flows through chuk-ai-session-manager.

    Responsibilities:
    - Conversation history management (via SessionManager)
    - Tool discovery and adaptation coordination
    - Procedural memory for tool learning
    - Session state (exit requests, etc.)
    """

    def __init__(
        self,
        tool_manager: ToolManager,
        model_manager: ModelManager,
        session_id: str | None = None,
        max_history_messages: int = 0,
        infinite_context: bool = False,
        token_threshold: int = 4000,
        max_turns_per_segment: int = 20,
        enable_vm: bool = False,
        vm_mode: str = "passive",
        vm_budget: int = 128_000,
        health_interval: int = 0,
        enable_plan_tools: bool = False,
        agent_id: str = "default",
    ):
        """
        Create chat context with required managers.

        Args:
            tool_manager: Tool management interface
            model_manager: Model configuration and LLM client manager
            session_id: Optional session ID for conversation tracking
            max_history_messages: Sliding window size (0 = unlimited)
            infinite_context: Enable infinite context mode in SessionManager
            token_threshold: Token threshold for infinite context segmentation
            max_turns_per_segment: Max turns per segment before context packing
            enable_vm: Enable AI Virtual Memory subsystem (experimental)
            vm_mode: VM mode - strict, relaxed, or passive
            vm_budget: Max tokens for VM L0 working set (context window budget)
            health_interval: Background health check interval in seconds (0 = disabled)
            enable_plan_tools: Enable plan_create/plan_execute as LLM-callable tools
        """
        self.tool_manager = tool_manager
        self.model_manager = model_manager
        self.session_id = session_id or self._generate_session_id()
        self.agent_id = agent_id

        # Context management
        self._max_history_messages = max_history_messages
        self._infinite_context = infinite_context
        self._token_threshold = token_threshold
        self._max_turns_per_segment = max_turns_per_segment
        self._enable_vm = enable_vm
        self._vm_mode = vm_mode
        self._vm_budget = vm_budget
        self._health_interval = health_interval
        self._enable_plan_tools = enable_plan_tools

        # Core session manager - always required
        self.session: SessionManager = SessionManager(session_id=self.session_id)
        self._system_prompt: str = ""

        # Procedural memory for tool learning
        self.tool_memory = ToolMemoryManager.create(session_id=self.session_id)
        self.procedural_formatter = ProceduralContextFormatter(
            config=FormatterConfig(
                max_recent_calls=5,
                max_errors_per_tool=3,
                max_successes_per_tool=2,
                include_fix_suggestions=True,
            )
        )

        # Session state
        self.exit_requested = False

        # Context management notices (ephemeral, drained before each API call)
        self._pending_context_notices: list[str] = []
        self._system_prompt_dirty: bool = True

        # Persistent memory scopes (workspace + global)
        self.memory_store: Any = None

        # Token usage tracking
        self.token_tracker = TokenTracker()

        # Session persistence
        self._session_store = SessionStore(agent_id=self.agent_id)
        self._auto_save_counter = 0

        # ToolProcessor back-reference (set by ToolProcessor.__init__)
        self.tool_processor: Any = None

        # Dashboard bridge (set by chat_handler when --dashboard is active, else None)
        self.dashboard_bridge: Any = None

        # Agent manager (set by chat_handler when multi-agent enabled, else None)
        self.agent_manager: Any = None

        # Attachment staging for multi-modal input (/attach command)
        from mcp_cli.chat.attachments import AttachmentStaging

        self.attachment_staging = AttachmentStaging()

        # Tool state (filled during initialization)
        self.tools: list[ToolInfo] = []
        self.internal_tools: list[ToolInfo] = []
        self.server_info: list[ServerInfo] = []
        self.tool_to_server_map: dict[str, str] = {}
        self.openai_tools: list[dict[str, Any]] = []
        self.tool_name_mapping: dict[str, str] = {}
        self._tool_index: dict[str, ToolInfo] = {}
        self.no_tools: bool = False

        logger.debug(f"ChatContext created with {self.provider}/{self.model}")

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a unique session ID."""
        return f"chat-{uuid.uuid4().hex[:12]}"

    @classmethod
    def create(
        cls,
        tool_manager: ToolManager,
        provider: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        model_manager: ModelManager | None = None,
        session_id: str | None = None,
        max_history_messages: int = 0,
        infinite_context: bool = False,
        token_threshold: int = 4000,
        max_turns_per_segment: int = 20,
        enable_vm: bool = False,
        vm_mode: str = "passive",
        vm_budget: int = 128_000,
        health_interval: int = 0,
        enable_plan_tools: bool = False,
        agent_id: str = "default",
    ) -> "ChatContext":
        """
        Factory method for convenient creation.

        Args:
            tool_manager: Tool management interface
            provider: Provider to switch to (optional)
            model: Model to switch to (optional)
            api_base: API base URL override (optional)
            api_key: API key override (optional)
            model_manager: Pre-configured ModelManager (optional, creates new if None)
            session_id: Session ID for conversation tracking (optional)
            max_history_messages: Sliding window size (0 = unlimited)
            infinite_context: Enable infinite context mode in SessionManager
            token_threshold: Token threshold for infinite context segmentation
            max_turns_per_segment: Max turns per segment before context packing
            enable_vm: Enable AI Virtual Memory subsystem (experimental)
            vm_mode: VM mode - strict, relaxed, or passive
            vm_budget: Max tokens for VM L0 working set (context window budget)
            health_interval: Background health check interval in seconds (0 = disabled)
            enable_plan_tools: Enable plan_create/plan_execute as LLM-callable tools

        Returns:
            Configured ChatContext instance
        """
        if model_manager is None:
            model_manager = ModelManager()

            if provider and (api_base or api_key):
                model_manager.add_runtime_provider(
                    name=provider, api_key=api_key, api_base=api_base or ""
                )

            if provider and model:
                model_manager.switch_model(provider, model)
            elif provider:
                model_manager.switch_provider(provider)
            elif model:
                current_provider = model_manager.get_active_provider()
                model_manager.switch_model(current_provider, model)

        return cls(
            tool_manager,
            model_manager,
            session_id,
            max_history_messages=max_history_messages,
            infinite_context=infinite_context,
            token_threshold=token_threshold,
            max_turns_per_segment=max_turns_per_segment,
            enable_vm=enable_vm,
            vm_mode=vm_mode,
            vm_budget=vm_budget,
            health_interval=health_interval,
            enable_plan_tools=enable_plan_tools,
            agent_id=agent_id,
        )

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def client(self) -> Any:
        """Get current LLM client (cached automatically by ModelManager)."""
        return self.model_manager.get_client()

    @property
    def provider(self) -> str:
        """Current provider name."""
        return self.model_manager.get_active_provider()

    @property
    def model(self) -> str:
        """Current model name."""
        return self.model_manager.get_active_model()

    @property
    def conversation_history(self) -> list[HistoryMessage]:
        """
        Get conversation history as list of HistoryMessage objects.

        Provides backwards compatibility while using SessionManager internally.
        Handles both regular messages and tool-related messages.

        System prompt is always included. If max_history_messages > 0,
        only the most recent N event-based messages are returned (sliding window).

        When VM is enabled:
        - System prompt is replaced with VM-packed developer_message
          (manifest, working set summaries, VM rules).
        - Only recent turn groups that fit within the VM token budget
          are sent as raw events. Older turns are represented by VM pages
          in the developer_message, avoiding double-counting.
        """
        messages = []

        # Determine system prompt content — VM replaces it with packed context
        if self.session.vm:
            vm_ctx = self.session.get_vm_context()
            system_content = (
                vm_ctx["developer_message"] if vm_ctx else self._system_prompt
            )
        else:
            system_content = self._system_prompt

        # System prompt always included (outside the window)
        if system_content:
            messages.append(
                HistoryMessage(role=MessageRole.SYSTEM, content=system_content)
            )

        # Build event-based messages
        event_messages: list[HistoryMessage] = []
        if self.session._session:
            for event in self.session._session.events:
                if event.type == EventType.MESSAGE:
                    if event.source == EventSource.USER:
                        event_messages.append(
                            HistoryMessage(
                                role=MessageRole.USER, content=str(event.message)
                            )
                        )
                    elif event.source in (EventSource.LLM, EventSource.SYSTEM):
                        event_messages.append(
                            HistoryMessage(
                                role=MessageRole.ASSISTANT, content=str(event.message)
                            )
                        )
                elif event.type == EventType.TOOL_CALL:
                    # Tool messages stored as dict - reconstruct HistoryMessage
                    if isinstance(event.message, dict):
                        event_messages.append(HistoryMessage.from_dict(event.message))

        # VM-aware context filtering: fit raw events within token budget
        if self.session.vm and system_content:
            event_messages = self._vm_filter_events(event_messages, system_content)

        # Apply sliding window if configured (non-VM fallback)
        elif (
            self._max_history_messages > 0
            and len(event_messages) > self._max_history_messages
        ):
            evicted = len(event_messages) - self._max_history_messages
            logger.info(
                f"Sliding window: keeping {self._max_history_messages} of "
                f"{len(event_messages)} messages (evicted {evicted})"
            )
            event_messages = event_messages[-self._max_history_messages :]
            self.add_context_notice(
                f"{evicted} older messages were evicted from context. "
                "Key context may need to be re-established."
            )

        messages.extend(event_messages)
        return messages

    # ── VM context filtering ─────────────────────────────────────────────

    # Minimum recent turns always included regardless of VM budget.
    # Ensures the model can always see immediate conversation context.
    _VM_MIN_RECENT_TURNS = 3

    def _vm_filter_events(
        self,
        events: list[HistoryMessage],
        system_content: str,
    ) -> list[HistoryMessage]:
        """Filter events to fit within VM token budget.

        Groups events into logical turns (starting at each user message),
        estimates token cost per group, and includes turn groups from newest
        to oldest until the budget is exhausted. Tool-call pairs are kept
        intact within their turn group.

        The most recent ``_VM_MIN_RECENT_TURNS`` turns are always included
        regardless of budget, so the model always has immediate context.
        Older turns beyond those are included only if budget allows.

        Evicted turns are NOT lost — they're represented by VM pages in the
        developer_message (manifest + working set summaries).

        Args:
            events: All event-based messages from the session.
            system_content: The VM developer_message (not counted against budget).

        Returns:
            Filtered event list that fits within the VM budget.
        """
        from mcp_cli.config.defaults import DEFAULT_CHARS_PER_TOKEN_ESTIMATE

        if not events:
            return events

        cpt = DEFAULT_CHARS_PER_TOKEN_ESTIMATE

        # Budget is for conversation events only — system prompt is on top
        remaining = self._vm_budget

        # Group events into logical turns.
        # A new turn starts at each user message.
        turns: list[list[HistoryMessage]] = []
        current_turn: list[HistoryMessage] = []

        for msg in events:
            if msg.role == MessageRole.USER and current_turn:
                turns.append(current_turn)
                current_turn = []
            current_turn.append(msg)
        if current_turn:
            turns.append(current_turn)

        # Nothing to filter
        if len(turns) <= self._VM_MIN_RECENT_TURNS:
            return events

        # Estimate token cost per turn group
        def _estimate_turn_tokens(turn: list[HistoryMessage]) -> int:
            total_chars = 0
            for msg in turn:
                total_chars += len(msg.content or "")
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        total_chars += len(str(tc))
            return max(total_chars // cpt, 1)

        # Always include the most recent N turns (guaranteed minimum)
        guaranteed = turns[-self._VM_MIN_RECENT_TURNS :]
        older = turns[: -self._VM_MIN_RECENT_TURNS]

        # Deduct guaranteed turns from budget
        for turn in guaranteed:
            remaining -= _estimate_turn_tokens(turn)

        # Include older turns from newest to oldest while budget allows
        budget_included: list[list[HistoryMessage]] = []
        for turn in reversed(older):
            cost = _estimate_turn_tokens(turn)
            if remaining >= cost:
                budget_included.append(turn)
                remaining -= cost
            else:
                break

        # Restore chronological order
        budget_included.reverse()

        evicted_turns = len(older) - len(budget_included)
        if evicted_turns > 0:
            evicted_msgs = sum(len(t) for t in older[:evicted_turns])
            logger.info(
                f"VM context filter: keeping {len(budget_included) + len(guaranteed)}"
                f"/{len(turns)} turns "
                f"({evicted_msgs} messages evicted, budget={self._vm_budget} tokens)"
            )
            self.add_context_notice(
                f"{evicted_turns} older conversation turns were moved to virtual memory. "
                "Their content is available via the VM manifest in the system context."
            )

        # Flatten: budget-included older turns + guaranteed recent turns
        result: list[HistoryMessage] = []
        for turn in budget_included:
            result.extend(turn)
        for turn in guaranteed:
            result.extend(turn)
        return result

    # ── Initialization ────────────────────────────────────────────────────
    async def initialize(
        self,
        on_progress: Callable[[str], None] | None = None,
    ) -> bool:
        """Initialize tools, session, and procedural memory.

        Args:
            on_progress: Optional callback for progress updates during startup
        """
        try:
            await self._initialize_tools(on_progress=on_progress)
            self._generate_system_prompt()
            await self._initialize_session()

            # Quick provider validation (non-blocking)
            try:
                _client = self.client  # noqa: F841 — fails fast if no API key
                logger.info(f"Provider {self.provider} client created successfully")
            except Exception as e:
                logger.warning(f"Provider validation warning: {e}")
                logger.warning("Chat may fail when making API calls.")

            if not self.tools:
                logger.warning("No tools available. Chat functionality may be limited.")

            logger.info(
                f"ChatContext ready: {len(self.tools)} tools, {self.provider}/{self.model}"
            )
            return True

        except Exception:
            logger.exception("Error initializing chat context")
            return False

    async def _initialize_session(self) -> None:
        """Initialize the session with system prompt and context management."""
        vm_config = (
            WorkingSetConfig(
                max_l0_tokens=self._vm_budget,
                reserved_tokens=min(4000, self._vm_budget // 4),
            )
            if self._enable_vm
            else None
        )
        self.session = SessionManager(
            session_id=self.session_id,
            system_prompt=self._system_prompt,
            infinite_context=self._infinite_context,
            token_threshold=self._token_threshold,
            max_turns_per_segment=self._max_turns_per_segment,
            enable_vm=self._enable_vm,
            vm_mode=VMMode(self._vm_mode),
            vm_config=vm_config,
        )
        await self.session._ensure_initialized()

        # Initialize persistent memory scopes
        try:
            from mcp_cli.memory.store import MemoryScopeStore

            self.memory_store = MemoryScopeStore()
            logger.debug("Persistent memory store initialized")
        except Exception as exc:
            logger.warning("Could not initialize memory store: %s", exc)
            self.memory_store = None

        logger.debug(
            f"Session initialized: {self.session_id} "
            f"(infinite_context={self._infinite_context}, "
            f"vm={self._enable_vm}, vm_budget={self._vm_budget})"
        )

    def _generate_system_prompt(self) -> None:
        """Generate system prompt from available tools (cached with dirty flag)."""
        if not self._system_prompt_dirty and self._system_prompt:
            return

        tools_for_prompt = [
            tool.to_llm_format().to_dict() for tool in self.internal_tools
        ]
        server_tool_groups = self._build_server_tool_groups()
        self._system_prompt = generate_system_prompt(
            tools=tools_for_prompt,
            server_tool_groups=server_tool_groups,
        )

        # Append persistent memory context
        if self.memory_store:
            memory_section = self.memory_store.format_for_system_prompt()
            if memory_section:
                self._system_prompt += "\n\n" + memory_section

        self._system_prompt_dirty = False

    def _build_server_tool_groups(self) -> list[ServerToolGroup]:
        """Build server-to-tools grouping for the system prompt."""
        if not self.server_info:
            return []

        # Group tools by server namespace
        server_tools: dict[str, list[str]] = {}
        for tool_name, namespace in self.tool_to_server_map.items():
            server_tools.setdefault(namespace, []).append(tool_name)

        groups = []
        for server in self.server_info:
            tools = server_tools.get(server.namespace, [])
            if tools:
                groups.append(
                    ServerToolGroup(
                        name=server.name,
                        description=server.display_description,
                        tools=sorted(tools),
                    )
                )
        return groups

    async def _initialize_tools(
        self,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize tool discovery and adaptation."""
        if on_progress:
            on_progress("Discovering tools...")

        self.tools = await self.tool_manager.get_unique_tools()
        logger.debug(f"ChatContext: Initialized with {len(self.tools)} tools")

        self.server_info = await self.tool_manager.get_server_info()
        self.tool_to_server_map = {t.name: t.namespace for t in self.tools}

        if on_progress:
            on_progress(f"Adapting {len(self.tools)} tools for {self.provider}...")

        await self._adapt_tools_for_provider()
        self.internal_tools = list(self.tools)
        self._system_prompt_dirty = True

        # Build O(1) tool lookup index
        self._tool_index = {}
        for tool in self.tools:
            self._tool_index[tool.name] = tool
            if tool.namespace:
                self._tool_index[tool.fully_qualified_name] = tool

    def find_tool_by_name(self, name: str) -> ToolInfo | None:
        """Find a tool by its name (handles both simple and namespaced names).

        Uses O(1) dict lookup instead of linear scanning.
        """
        # Direct lookup by name or fully qualified name
        tool = self._tool_index.get(name)
        if tool:
            return tool

        # Fallback: try extracting simple name from dotted notation
        if "." in name:
            simple_name = name.rsplit(".", 1)[-1]
            return self._tool_index.get(simple_name)

        return None

    def find_server_by_name(self, name: str) -> ServerInfo | None:
        """Find a server by its name."""
        for server in self.server_info:
            if server.name == name or server.namespace == name:
                return server
        return None

    async def _adapt_tools_for_provider(self) -> None:
        """Adapt tools for current provider."""
        try:
            if hasattr(self.tool_manager, "get_adapted_tools_for_llm"):
                tools_and_mapping = await self.tool_manager.get_adapted_tools_for_llm(
                    self.provider
                )
                self.openai_tools = tools_and_mapping[0]
                self.tool_name_mapping = tools_and_mapping[1]
                logger.debug(
                    f"Adapted {len(self.openai_tools)} tools for {self.provider}"
                )
            else:
                self.openai_tools = await self.tool_manager.get_tools_for_llm()
                self.tool_name_mapping = {}
        except Exception as exc:
            logger.warning(f"Error adapting tools: {exc}")
            self.openai_tools = await self.tool_manager.get_tools_for_llm()
            self.tool_name_mapping = {}

    # ── Model change handling ─────────────────────────────────────────────
    async def refresh_after_model_change(self) -> None:
        """Refresh context after ModelManager changes the model."""
        await self._adapt_tools_for_provider()
        self._system_prompt_dirty = True
        logger.debug(f"ChatContext refreshed for {self.provider}/{self.model}")

    # ── Tool execution (delegate to ToolManager) ──────────────────────────
    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool."""
        return await self.tool_manager.execute_tool(tool_name, arguments)

    async def stream_execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> AsyncIterator[Any]:
        """Execute a tool with streaming."""
        async for result in self.tool_manager.stream_execute_tool(tool_name, arguments):
            yield result

    async def get_server_for_tool(self, tool_name: str) -> str:
        """Get server for tool."""
        return await self.tool_manager.get_server_for_tool(tool_name) or "Unknown"

    # ── Conversation management ───────────────────────────────────────────
    async def add_user_message(self, content: str | list[dict[str, Any]]) -> None:
        """Add user message to conversation.

        Accepts plain text (str) or multi-modal content blocks (list[dict]).
        Multi-modal content is injected as a raw event since
        SessionManager.user_says() only accepts strings.
        """
        if isinstance(content, str):
            await self.session.user_says(content)
            logger.debug(f"User message added: {content[:50]}...")
        else:
            # Multi-modal: inject as dict event (same pattern as inject_tool_message)
            msg = HistoryMessage(role=MessageRole.USER, content=content)
            event = SessionEvent(
                message=msg.to_dict(),
                source=EventSource.USER,
                type=EventType.TOOL_CALL,
            )
            self.session._session.events.append(event)
            logger.debug(
                f"Multi-modal user message added: {len(content)} content blocks"
            )

    async def add_assistant_message(self, content: str) -> None:
        """Add assistant message to conversation."""
        await self.session.ai_responds(
            content, model=self.model, provider=self.provider
        )
        logger.debug(f"Assistant message added: {content[:50]}...")

    def inject_assistant_message(self, content: str) -> None:
        """
        Inject a synthetic assistant message for conversation flow control.

        Use this for system-generated messages (budget exhaustion, state summaries,
        error recovery) that guide the model but aren't true AI responses.
        """
        event = SessionEvent(
            message=content,
            source=EventSource.SYSTEM,
            type=EventType.MESSAGE,
        )
        self.session._session.events.append(event)
        logger.debug(f"Injected assistant message: {content[:50]}...")

    def inject_tool_message(self, message: HistoryMessage) -> None:
        """
        Inject a tool-related message into conversation history.

        Tool messages (assistant with tool_calls, tool results) have special structure
        that doesn't map to SessionManager events. These are stored as raw events
        for conversation flow but tracked separately in procedural memory.
        """
        # Store as TOOL_CALL event with the full message structure
        event = SessionEvent(
            message=message.to_dict(),
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL,
        )
        self.session._session.events.append(event)
        logger.debug(f"Injected tool message: role={message.role}")

    async def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        success: bool = True,
        error: str | None = None,
        context_goal: str | None = None,
    ) -> None:
        """
        Record a tool call in session and procedural memory.

        Args:
            tool_name: Name of the tool called
            arguments: Arguments passed to the tool
            result: Result returned by the tool
            success: Whether the call succeeded
            error: Error message if failed
            context_goal: What the user was trying to accomplish
        """
        # Record in session
        await self.session.tool_used(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            error=error,
        )

        # Record in procedural memory for learning
        outcome = ToolOutcome.SUCCESS if success else ToolOutcome.FAILURE
        error_type = (
            type(error).__name__ if error and not isinstance(error, str) else None
        )

        await self.tool_memory.record_call(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            outcome=outcome,
            context_goal=context_goal,
            error_type=error_type,
            error_message=str(error) if error else None,
        )

        # Enforce pattern limits (upstream doesn't enforce max_patterns_per_tool)
        self._enforce_memory_limits()

        logger.debug(f"Tool call recorded: {tool_name} (success={success})")

    async def get_messages_for_llm(self) -> list[dict[str, str]]:
        """Get messages formatted for LLM API calls."""
        result: list[dict[str, str]] = await self.session.get_messages_for_llm(
            include_system=True
        )
        return result

    def get_conversation_length(self) -> int:
        """Get conversation length (excluding system prompt)."""
        if self.session._session:
            return sum(
                1 for e in self.session._session.events if e.type == EventType.MESSAGE
            )
        return 0

    async def clear_conversation_history(self, keep_system_prompt: bool = True) -> None:
        """Clear conversation history by creating a new session."""
        self.session_id = self._generate_session_id()
        await self._initialize_session()
        self.tool_memory = ToolMemoryManager.create(session_id=self.session_id)
        logger.debug(f"Conversation cleared, new session: {self.session_id}")

    async def regenerate_system_prompt(self) -> None:
        """Regenerate system prompt with current tools."""
        self._system_prompt_dirty = True
        self._generate_system_prompt()
        await self.session.update_system_prompt(self._system_prompt)

    # ── Procedural memory helpers ─────────────────────────────────────────
    def get_procedural_context_for_tools(
        self, tool_names: list[str], context_goal: str | None = None
    ) -> str:
        """
        Get procedural memory context for tools about to be called.

        Use this to inject relevant tool history before making LLM calls.
        """
        result: str = self.procedural_formatter.format_for_tools(
            self.tool_memory, tool_names, context_goal
        )
        return result

    def get_recent_tool_history(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get recent tool call history from procedural memory."""
        return [
            {
                "tool": entry.tool_name,
                "arguments": entry.arguments,
                "outcome": entry.outcome.value,
                "timestamp": entry.timestamp.isoformat(),
            }
            for entry in self.tool_memory.memory.tool_log[-limit:]
        ]

    # ── Context management notices ────────────────────────────────────────
    def add_context_notice(self, notice: str) -> None:
        """Queue a context management notice for the next API call."""
        self._pending_context_notices.append(notice)
        logger.debug(f"Context notice queued: {notice[:80]}")

    def drain_context_notices(self) -> list[str]:
        """Return and clear pending context notices."""
        notices = self._pending_context_notices[:]
        self._pending_context_notices.clear()
        return notices

    # ── Memory limits enforcement ─────────────────────────────────────────
    def _enforce_memory_limits(self) -> None:
        """Trim procedural memory patterns to stay within configured limits."""
        max_patterns = self.tool_memory.max_patterns_per_tool
        for _tool_name, pattern in self.tool_memory.memory.tool_patterns.items():
            if len(pattern.error_patterns) > max_patterns:
                pattern.error_patterns = pattern.error_patterns[-max_patterns:]
            if len(pattern.success_patterns) > max_patterns:
                pattern.success_patterns = pattern.success_patterns[-max_patterns:]

    # ── Session persistence ──────────────────────────────────────────────
    def save_session(self) -> str | None:
        """Save current session to disk.

        Returns:
            Path to saved file, or None on failure.
        """
        try:
            messages = self.conversation_history
            message_dicts: list[dict[str, Any]] = [m.to_dict() for m in messages]

            token_usage = None
            if self.token_tracker.turn_count > 0:
                token_usage = {
                    "total_input": self.token_tracker.total_input,
                    "total_output": self.token_tracker.total_output,
                    "total_tokens": self.token_tracker.total_tokens,
                    "turn_count": self.token_tracker.turn_count,
                }

            data = SessionData(
                metadata=SessionMetadata(
                    session_id=self.session_id,
                    agent_id=self.agent_id,
                    provider=self.provider,
                    model=self.model,
                    message_count=len(message_dicts),
                ),
                messages=message_dicts,
                token_usage=token_usage,
            )

            path = self._session_store.save(data)
            return str(path)
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return None

    def load_session(self, session_id: str) -> bool:
        """Load a saved session into the current context.

        Args:
            session_id: Session ID to load

        Returns:
            True if loaded successfully
        """
        data = self._session_store.load(session_id)
        if data is None:
            return False

        try:
            # Inject messages into the session manager
            for msg_dict in data.messages:
                role = msg_dict.get("role", "")
                content = msg_dict.get("content", "")

                if role == MessageRole.SYSTEM:
                    continue  # System prompt is regenerated
                elif role == MessageRole.USER:
                    event = SessionEvent(
                        type=EventType.MESSAGE,
                        source=EventSource.USER,
                        message=content,
                    )
                elif role == MessageRole.ASSISTANT:
                    # Assistant messages with tool_calls need full dict
                    if msg_dict.get("tool_calls"):
                        event = SessionEvent(
                            type=EventType.TOOL_CALL,
                            source=EventSource.SYSTEM,
                            message=msg_dict,
                        )
                    else:
                        event = SessionEvent(
                            type=EventType.MESSAGE,
                            source=EventSource.LLM,
                            message=content,
                        )
                elif role == MessageRole.TOOL:
                    # Tool result messages stored as full dict for reconstruction
                    event = SessionEvent(
                        type=EventType.TOOL_CALL,
                        source=EventSource.SYSTEM,
                        message=msg_dict,
                    )
                else:
                    continue

                self.session._session.events.append(event)

            logger.info(
                f"Loaded session {session_id} with {len(data.messages)} messages"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return False

    def auto_save_check(self) -> None:
        """Increment auto-save counter and save when threshold is reached."""
        from mcp_cli.config.defaults import DEFAULT_AUTO_SAVE_INTERVAL

        self._auto_save_counter += 1
        if self._auto_save_counter >= DEFAULT_AUTO_SAVE_INTERVAL:
            self._auto_save_counter = 0
            self.save_session()
            logger.debug("Auto-save triggered")

    # ── Simple getters ────────────────────────────────────────────────────
    def get_tool_count(self) -> int:
        """Get number of available tools."""
        return len(self.tools)

    def get_server_count(self) -> int:
        """Get number of connected servers."""
        return len(self.server_info)

    @staticmethod
    def get_display_name_for_tool(namespaced_tool_name: str) -> str:
        """Get display name for tool."""
        return namespaced_tool_name

    # ── Serialization ─────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        """Export context for command handlers."""
        return {
            "conversation_history": [
                msg.to_dict() for msg in self.conversation_history
            ],
            "tools": self.tools,
            "internal_tools": self.internal_tools,
            "client": self.client,
            "provider": self.provider,
            "model": self.model,
            "model_manager": self.model_manager,
            "server_info": self.server_info,
            "openai_tools": self.openai_tools,
            "tool_name_mapping": self.tool_name_mapping,
            "exit_requested": self.exit_requested,
            "tool_to_server_map": self.tool_to_server_map,
            "tool_manager": self.tool_manager,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
        }

    def update_from_dict(self, context_dict: dict[str, Any]) -> None:
        """Update context from dictionary."""
        if "exit_requested" in context_dict:
            self.exit_requested = context_dict["exit_requested"]

        if "model_manager" in context_dict:
            self.model_manager = context_dict["model_manager"]

        for key in [
            "tools",
            "internal_tools",
            "server_info",
            "tool_to_server_map",
            "openai_tools",
            "tool_name_mapping",
        ]:
            if key in context_dict:
                setattr(self, key, context_dict[key])

    # ── Context manager ───────────────────────────────────────────────────
    async def __aenter__(self):
        """Async context manager entry."""
        if not await self.initialize():
            raise RuntimeError("Failed to initialize ChatContext")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

    # ── Debug info ────────────────────────────────────────────────────────
    def get_status_summary(self) -> ChatStatus:
        """Get status summary for debugging."""
        return ChatStatus(
            provider=self.provider,
            model=self.model,
            tool_count=len(self.tools),
            internal_tool_count=len(self.internal_tools),
            server_count=len(self.server_info),
            message_count=self.get_conversation_length(),
            tool_execution_count=len(self.tool_memory.memory.tool_log),
        )

    async def get_session_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        result: dict[str, Any] = await self.session.get_stats()
        return result

    def __repr__(self) -> str:
        return (
            f"ChatContext(session='{self.session_id}', provider='{self.provider}', "
            f"model='{self.model}', tools={len(self.tools)}, messages={self.get_conversation_length()})"
        )

    def __str__(self) -> str:
        return f"Chat session {self.session_id} with {self.provider}/{self.model} ({len(self.tools)} tools)"
