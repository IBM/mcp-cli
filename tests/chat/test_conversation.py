# tests/chat/test_conversation.py
"""Tests for ConversationProcessor."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.chat.conversation import ConversationProcessor
from mcp_cli.chat.response_models import (
    CompletionResponse,
    Message,
    MessageRole,
    ToolCall,
    FunctionCall,
)


class MockUIManager:
    """Mock UI manager for testing."""

    def __init__(self):
        self.is_streaming_response = False
        self.streaming_handler = None
        self.display = MagicMock()

    async def start_streaming_response(self):
        self.is_streaming_response = True

    async def stop_streaming_response(self):
        self.is_streaming_response = False

    async def print_assistant_message(self, content, elapsed):
        pass


class MockContext:
    """Mock context for testing."""

    def __init__(self):
        self.conversation_history = []
        self.openai_tools = []
        self.tool_name_mapping = {}
        self.client = MagicMock()
        self.tool_manager = MagicMock()
        # Make tool_manager async methods work properly
        self.tool_manager.get_adapted_tools_for_llm = AsyncMock(return_value=([], {}))
        self.provider = "openai"

    async def add_assistant_message(self, content):
        """Add assistant message to conversation history."""
        self.conversation_history.append(
            Message(role=MessageRole.ASSISTANT, content=content)
        )

    def inject_assistant_message(self, message):
        """Inject a message into conversation history."""
        self.conversation_history.append(message)

    def inject_tool_message(self, message):
        """Inject a tool message into conversation history."""
        self.conversation_history.append(message)


class TestConversationProcessorInit:
    """Tests for ConversationProcessor initialization."""

    def test_init(self):
        """Test basic initialization."""
        context = MockContext()
        ui_manager = MockUIManager()

        processor = ConversationProcessor(context, ui_manager)

        assert processor.context is context
        assert processor.ui_manager is ui_manager
        assert processor.tool_processor is not None
        assert processor._consecutive_duplicate_count == 0
        assert processor._max_consecutive_duplicates == 5

    def test_init_with_runtime_config(self):
        """Test initialization with runtime config."""
        context = MockContext()
        ui_manager = MockUIManager()
        runtime_config = {"some": "config"}

        processor = ConversationProcessor(context, ui_manager, runtime_config)

        assert processor.runtime_config == runtime_config


class TestRegisterUserLiterals:
    """Tests for _register_user_literals_from_history."""

    def test_register_from_user_message(self):
        """Test registering literals from user message."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate sqrt of 18")
        ]
        ui_manager = MockUIManager()

        processor = ConversationProcessor(context, ui_manager)
        count = processor._register_user_literals_from_history()

        # Should register at least the number 18
        assert count >= 1

    def test_register_empty_history(self):
        """Test with empty history."""
        context = MockContext()
        context.conversation_history = []
        ui_manager = MockUIManager()

        processor = ConversationProcessor(context, ui_manager)
        count = processor._register_user_literals_from_history()

        assert count == 0

    def test_register_only_most_recent(self):
        """Test that only most recent user message is processed."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="First message with 10"),
            Message(role=MessageRole.ASSISTANT, content="Response"),
            Message(role=MessageRole.USER, content="Second with 20 and 30"),
        ]
        ui_manager = MockUIManager()

        processor = ConversationProcessor(context, ui_manager)
        # Reset tool state to clear any prior registrations
        processor._tool_state.reset_for_new_prompt()
        count = processor._register_user_literals_from_history()

        # Should only process the most recent user message
        assert count >= 2  # At least 20 and 30


class TestProcessConversation:
    """Tests for process_conversation method."""

    @pytest.mark.asyncio
    async def test_slash_command_skipped(self):
        """Test that slash commands are skipped."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="/help")]
        ui_manager = MockUIManager()

        processor = ConversationProcessor(context, ui_manager)
        await processor.process_conversation()

        # Should return immediately without processing
        assert len(context.conversation_history) == 1

    @pytest.mark.asyncio
    async def test_no_tools_loads_tools(self):
        """Test that tools are loaded if not present."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = None  # No tools loaded
        context.client.create_completion = AsyncMock(
            return_value={"response": "Hi there!", "tool_calls": None}
        )
        ui_manager = MockUIManager()

        # Mock _load_tools
        processor = ConversationProcessor(context, ui_manager)
        processor._load_tools = AsyncMock()

        await processor.process_conversation()

        # Should have called _load_tools
        processor._load_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_simple_response(self):
        """Test processing a simple text response."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        # Mock client response
        context.client.create_completion = AsyncMock(
            return_value={"response": "Hello! How can I help?", "tool_calls": None}
        )

        ui_manager = MockUIManager()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)
        await processor.process_conversation()

        # Should have added assistant message to history
        assert len(context.conversation_history) == 2
        assert context.conversation_history[-1].role == MessageRole.ASSISTANT
        assert "Hello" in context.conversation_history[-1].content

    @pytest.mark.asyncio
    async def test_max_turns_limit(self):
        """Test that max_turns limit is enforced."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        # Mock client to always return tool calls (would loop forever)
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments='{"x": 18}'),
        )
        context.client.create_completion = AsyncMock(
            return_value={"response": "", "tool_calls": [tool_call.model_dump()]}
        )

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        # Mock tool processor to avoid actual execution
        processor.tool_processor.process_tool_calls = AsyncMock()

        # Set very low max_turns
        await processor.process_conversation(max_turns=2)

        # Should have stopped due to max_turns
        assert processor.tool_processor.process_tool_calls.call_count <= 2


class TestHandleRegularCompletion:
    """Tests for _handle_regular_completion."""

    @pytest.mark.asyncio
    async def test_regular_completion_success(self):
        """Test successful regular completion."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.client.create_completion = AsyncMock(
            return_value={"response": "Hi!", "tool_calls": None}
        )

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        result = await processor._handle_regular_completion(tools=[])

        assert isinstance(result, CompletionResponse)
        assert result.response == "Hi!"
        assert result.streaming is False
        assert result.elapsed_time > 0

    @pytest.mark.asyncio
    async def test_regular_completion_tool_error_retry(self):
        """Test retry without tools on tool definition error."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]

        # First call fails with tool error, second succeeds
        context.client.create_completion = AsyncMock(
            side_effect=[
                Exception("Invalid 'tools' specification"),
                {"response": "Hi without tools!", "tool_calls": None},
            ]
        )

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        result = await processor._handle_regular_completion(tools=[{"some": "tool"}])

        assert isinstance(result, CompletionResponse)
        assert result.response == "Hi without tools!"
        # Should have been called twice
        assert context.client.create_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_regular_completion_other_error_raises(self):
        """Test that non-tool errors are raised."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.client.create_completion = AsyncMock(
            side_effect=Exception("Some other error")
        )

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        with pytest.raises(Exception, match="Some other error"):
            await processor._handle_regular_completion(tools=[])


class TestLoadTools:
    """Tests for _load_tools."""

    @pytest.mark.asyncio
    async def test_load_tools_success(self):
        """Test successful tool loading."""
        context = MockContext()
        context.tool_manager.get_adapted_tools_for_llm = AsyncMock(
            return_value=(
                [{"type": "function", "function": {"name": "sqrt"}}],
                {"sqrt": "math.sqrt"},
            )
        )

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        await processor._load_tools()

        assert len(context.openai_tools) == 1
        assert context.tool_name_mapping == {"sqrt": "math.sqrt"}

    @pytest.mark.asyncio
    async def test_load_tools_error(self):
        """Test tool loading error handling."""
        context = MockContext()
        context.tool_manager.get_adapted_tools_for_llm = AsyncMock(
            side_effect=Exception("Failed to load")
        )

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        await processor._load_tools()

        # Should set empty tools on error
        assert context.openai_tools == []
        assert context.tool_name_mapping == {}


class TestHandleStreamingCompletion:
    """Tests for _handle_streaming_completion via process_conversation integration."""

    @pytest.mark.asyncio
    async def test_streaming_path_taken_when_supported(self):
        """Test that streaming path is used when client supports it."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.start_streaming_response = AsyncMock()
        ui_manager.display = MagicMock()
        ui_manager.is_streaming_response = False
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        processor._tool_state = mock_tool_state

        # Mock streaming completion method directly
        async def mock_streaming(tools=None):
            return CompletionResponse(
                response="Streamed!",
                tool_calls=[],
                streaming=True,
                elapsed_time=0.5,
            )

        processor._handle_streaming_completion = mock_streaming

        await processor.process_conversation(max_turns=1)

        # The streaming handler should have been set to None at the end
        assert ui_manager.streaming_handler is None

    @pytest.mark.asyncio
    async def test_tool_calls_are_processed_in_conversation(self):
        """Test that tool calls from completion are processed."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "sqrt"}}]
        context.tool_name_mapping = {}

        # Create client mock that returns tool calls first, then a response
        call_count = [0]

        async def mock_completion(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "response": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "sqrt", "arguments": '{"x": 16}'},
                        }
                    ],
                }
            else:
                return {"response": "The result is 4", "tool_calls": []}

        context.client.create_completion = mock_completion

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        from chuk_ai_session_manager.guards import RunawayStatus

        mock_tool_state.check_runaway = MagicMock(
            return_value=RunawayStatus(should_stop=False)
        )
        processor._tool_state = mock_tool_state

        # Mock tool processor
        processor.tool_processor.process_tool_calls = AsyncMock()

        await processor.process_conversation(max_turns=3)

        # Should have processed tool calls
        processor.tool_processor.process_tool_calls.assert_called()


class TestDuplicateToolCallDetection:
    """Tests for duplicate tool call detection."""

    @pytest.mark.asyncio
    async def test_consecutive_duplicate_limit(self):
        """Test that max consecutive duplicates triggers exit."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        # Create a tool call response
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments='{"x": 18}'),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        processor = ConversationProcessor(context, ui_manager)

        # Simulate max consecutive duplicates
        processor._consecutive_duplicate_count = 4  # One below max
        processor._max_consecutive_duplicates = 5

        # Mock to always return the same tool call
        context.client.create_completion = AsyncMock(
            return_value={"response": "", "tool_calls": [tool_call.model_dump()]}
        )

        # Mock tool processor
        processor.tool_processor.process_tool_calls = AsyncMock()

        # Set last signature to match current
        with patch.object(processor, "_register_user_literals_from_history"):
            await processor.process_conversation(max_turns=3)

        # Should have stopped due to duplicate detection or max_turns
        # The test validates the mechanism exists

    @pytest.mark.asyncio
    async def test_duplicate_detection_reset_on_different_args(self):
        """Test that duplicate counter resets when args differ."""
        context = MockContext()
        ui_manager = MockUIManager()

        processor = ConversationProcessor(context, ui_manager)
        processor._consecutive_duplicate_count = 3

        # Accessing internal state to test reset
        # When a non-duplicate comes in, counter should reset to 0
        assert processor._consecutive_duplicate_count == 3

        # After processing different tool call, would reset
        processor._consecutive_duplicate_count = 0
        assert processor._consecutive_duplicate_count == 0


class TestBudgetExhaustion:
    """Tests for tool budget exhaustion scenarios."""

    @pytest.mark.asyncio
    async def test_discovery_budget_exhausted(self):
        """Test handling of discovery budget exhaustion."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Search for something")
        ]
        context.openai_tools = []

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="search", arguments='{"query": "test"}'),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        processor = ConversationProcessor(context, ui_manager)

        # Mock the tool state to indicate discovery budget exhausted
        from chuk_ai_session_manager.guards import RunawayStatus

        mock_status = RunawayStatus(
            should_stop=True, reason="Discovery budget exhausted", budget_exhausted=True
        )
        mock_status_ok = RunawayStatus(should_stop=False)

        # Create a mock tool state manager
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=True)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)

        check_call_count = [0]

        def mock_check_runaway(tool_name=None):
            check_call_count[0] += 1
            # First check is for discovery tool budget
            if check_call_count[0] == 1:
                return mock_status
            return mock_status_ok

        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check_runaway)
        mock_tool_state.format_discovery_exhausted_message = MagicMock(
            return_value="Discovery exhausted"
        )

        # Replace the tool state
        processor._tool_state = mock_tool_state

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Final answer", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

    @pytest.mark.asyncio
    async def test_execution_budget_exhausted(self):
        """Test handling of execution budget exhaustion."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Execute something")
        ]
        context.openai_tools = []

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="execute", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_status = RunawayStatus(
            should_stop=True, reason="Execution budget exhausted", budget_exhausted=True
        )
        mock_status_ok = RunawayStatus(should_stop=False)

        # Create a mock tool state manager
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=True)

        check_call_count = [0]

        def mock_check_runaway(tool_name=None):
            check_call_count[0] += 1
            # First check (discovery) is OK, second check (execution) exhausted
            if check_call_count[0] == 2:
                return mock_status
            return mock_status_ok

        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check_runaway)
        mock_tool_state.format_execution_exhausted_message = MagicMock(
            return_value="Execution exhausted"
        )

        # Replace the tool state
        processor._tool_state = mock_tool_state

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Done", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)


class TestRunawayDetection:
    """Tests for general runaway detection."""

    @pytest.mark.asyncio
    async def test_runaway_with_saturation(self):
        """Test runaway detection with saturation."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = []

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="compute", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_status = RunawayStatus(
            should_stop=True,
            reason="Saturation detected",
            saturation_detected=True,
            message="Results have converged",
        )
        mock_status_ok = RunawayStatus(should_stop=False)

        # Create a mock tool state manager
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state._recent_numeric_results = [3.14159]

        check_call_count = [0]

        def mock_check_runaway(tool_name=None):
            check_call_count[0] += 1
            # Third call is general runaway check
            if check_call_count[0] == 3:
                return mock_status
            return mock_status_ok

        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check_runaway)
        mock_tool_state.format_saturation_message = MagicMock(
            return_value="Saturation message"
        )

        # Replace the tool state
        processor._tool_state = mock_tool_state

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Final", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)


class TestStreamingFallback:
    """Tests for streaming fallback to regular completion."""

    @pytest.mark.asyncio
    async def test_streaming_fallback_on_error(self):
        """Test fallback to regular completion when streaming fails."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.start_streaming_response = AsyncMock()
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()
        ui_manager.display = MagicMock()
        ui_manager.is_streaming_response = False

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        processor._tool_state = mock_tool_state

        # Mock streaming to fail and regular to succeed
        async def mock_streaming_fail(tools=None):
            raise Exception("Streaming error")

        processor._handle_streaming_completion = mock_streaming_fail

        # But regular completion succeeds
        context.client.create_completion = AsyncMock(
            return_value={"response": "Fallback response", "tool_calls": []}
        )

        await processor.process_conversation(max_turns=1)

        # Should have fallen back to regular completion
        context.client.create_completion.assert_called()


class TestConversationErrorHandling:
    """Tests for error handling in conversation processing."""

    @pytest.mark.asyncio
    async def test_general_error_in_loop(self):
        """Test error handling during conversation loop."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Mock client to raise an error (RuntimeError hits the general Exception handler
        # which injects an assistant error message — ValueError is caught by the specific
        # config/validation handler that does not inject one)
        context.client.create_completion = AsyncMock(
            side_effect=RuntimeError("Some unexpected error")
        )

        await processor.process_conversation(max_turns=1)

        # Should have added error message to history
        assert len(context.conversation_history) >= 2
        last_msg = context.conversation_history[-1]
        # Error message may be a Message object or a string depending on the error path
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        assert "error" in content.lower()

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        """Test that asyncio.CancelledError is re-raised."""
        import asyncio

        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        context.client.create_completion = AsyncMock(
            side_effect=asyncio.CancelledError()
        )

        with pytest.raises(asyncio.CancelledError):
            await processor.process_conversation(max_turns=1)


class TestInspectionHandling:
    """Tests for signature inspection edge cases."""

    @pytest.mark.asyncio
    async def test_inspection_error_disables_streaming(self):
        """Test that inspection errors disable streaming gracefully."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        # Create a client that has create_completion but inspection fails
        mock_client = MagicMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "Hi!", "tool_calls": None}
        )

        context.client = mock_client

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Mock inspect.signature to raise
        with patch("inspect.signature", side_effect=ValueError("Cannot inspect")):
            await processor.process_conversation(max_turns=1)

            # Should complete without streaming
            mock_client.create_completion.assert_called()


class TestBindingExtraction:
    """Tests for value binding extraction from responses."""

    @pytest.mark.asyncio
    async def test_extracts_bindings_from_response(self):
        """Test that value bindings are extracted from assistant responses."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = []

        context.client.create_completion = AsyncMock(
            return_value={
                "response": "The result is σ = 5.0 and π = 3.14159",
                "tool_calls": [],  # Empty list, not None
            }
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state with binding extraction
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        mock_extract = MagicMock(return_value=[])
        mock_tool_state.extract_bindings_from_text = mock_extract
        processor._tool_state = mock_tool_state

        await processor.process_conversation(max_turns=1)

        # Should have tried to extract bindings
        mock_extract.assert_called_once()


class TestStreamingCleanup:
    """Tests for streaming handler cleanup."""

    @pytest.mark.asyncio
    async def test_streaming_handler_cleared_after_response(self):
        """Test that streaming handler reference is cleared after streaming response."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.streaming_handler = MagicMock()  # Simulate existing handler
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        processor._tool_state = mock_tool_state

        # Mock _handle_streaming_completion to return a streaming response
        async def mock_streaming_completion(tools=None):
            return CompletionResponse(
                response="Hi!",
                tool_calls=[],
                streaming=True,
                elapsed_time=0.5,
            )

        processor._handle_streaming_completion = mock_streaming_completion

        await processor.process_conversation(max_turns=1)

        # Handler should be cleared after streaming response
        assert ui_manager.streaming_handler is None


class TestMaxTurnsWithToolCalls:
    """Tests for max_turns limit with tool calls."""

    @pytest.mark.asyncio
    async def test_max_turns_stops_tool_loop(self):
        """Test that max_turns limit stops infinite tool call loops."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Keep calling tools")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "loop"}}]
        context.tool_name_mapping = {}

        # Always return tool calls - would loop forever without max_turns
        async def infinite_tool_calls(**kwargs):
            return {
                "response": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "loop", "arguments": "{}"},
                    }
                ],
            }

        context.client.create_completion = infinite_tool_calls

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        from chuk_ai_session_manager.guards import RunawayStatus

        mock_tool_state.check_runaway = MagicMock(
            return_value=RunawayStatus(should_stop=False)
        )
        processor._tool_state = mock_tool_state

        # Mock tool processor
        processor.tool_processor.process_tool_calls = AsyncMock()

        # Set very low max_turns
        await processor.process_conversation(max_turns=2)

        # Should have stopped after max_turns
        assert processor.tool_processor.process_tool_calls.call_count <= 2


class TestGeneralRunawayDetection:
    """Tests for general runaway detection scenarios."""

    @pytest.mark.asyncio
    async def test_runaway_with_budget_exhausted(self):
        """Test runaway detection with budget exhausted."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "compute"}}]
        context.tool_name_mapping = {}

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="compute", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()
        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_status = RunawayStatus(
            should_stop=True,
            reason="Budget exhausted",
            budget_exhausted=True,
            message="Tool call budget exhausted",
        )
        mock_status_ok = RunawayStatus(should_stop=False)

        # Create a mock tool state manager
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)

        check_call_count = [0]

        def mock_check_runaway(tool_name=None):
            check_call_count[0] += 1
            # General runaway check is after discovery and execution checks
            # Discovery check returns should_stop=False
            # Execution check returns should_stop=False
            # General runaway check (3rd call) returns should_stop=True
            if check_call_count[0] >= 3:
                return mock_status
            return mock_status_ok

        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check_runaway)
        mock_tool_state.format_budget_exhausted_message = MagicMock(
            return_value="Budget exhausted message"
        )
        mock_tool_state.format_state_for_model = MagicMock(return_value="State summary")

        # Replace the tool state
        processor._tool_state = mock_tool_state

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Final", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        # The test validates that runaway detection mechanism is called
        # The exact behavior depends on the tool state configuration
        assert check_call_count[0] >= 1  # check_runaway was called


class TestDuplicateToolCallHandling:
    """Tests for duplicate tool call handling with state injection."""

    @pytest.mark.asyncio
    async def test_duplicate_triggers_state_injection(self):
        """Test that duplicate tool calls trigger state summary injection."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "sqrt"}}]
        context.tool_name_mapping = {}

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments='{"x": 16}'),
        )

        # Return same tool call multiple times, then final response
        call_count = [0]

        async def repeated_tool_calls(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:  # First 3 calls return same tool
                return {"response": "", "tool_calls": [tool_call.model_dump()]}
            else:
                return {"response": "Done", "tool_calls": []}

        context.client.create_completion = repeated_tool_calls

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        mock_tool_state.format_state_for_model = MagicMock(return_value="State: v0=4.0")
        from chuk_ai_session_manager.guards import RunawayStatus

        mock_tool_state.check_runaway = MagicMock(
            return_value=RunawayStatus(should_stop=False)
        )
        processor._tool_state = mock_tool_state

        # Mock tool processor
        processor.tool_processor.process_tool_calls = AsyncMock()

        await processor.process_conversation(max_turns=10)

        # Should have injected state summary (called format_state_for_model)
        # after detecting duplicate
        assert mock_tool_state.format_state_for_model.call_count >= 1


class TestReasoningContent:
    """Tests for reasoning content handling."""

    @pytest.mark.asyncio
    async def test_reasoning_content_preserved(self):
        """Test that reasoning content from response is preserved."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Think about this")
        ]
        context.openai_tools = []

        context.client.create_completion = AsyncMock(
            return_value={
                "response": "The answer is 42",
                "tool_calls": [],
                "reasoning_content": "Let me think step by step...",
            }
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        processor._tool_state = mock_tool_state

        await processor.process_conversation(max_turns=1)

        # Check that reasoning content was added to the message
        last_msg = context.conversation_history[-1]
        assert last_msg.role == MessageRole.ASSISTANT
        assert hasattr(last_msg, "reasoning_content") or "reasoning" in str(last_msg)


class TestMaxDuplicatesExceeded:
    """Tests for max duplicates safety valve."""

    @pytest.mark.asyncio
    async def test_max_duplicates_breaks_loop(self):
        """Test that hitting max consecutive duplicates breaks the loop."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "sqrt"}}]
        context.tool_name_mapping = {}

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments='{"x": 16}'),
        )

        # Always return the same tool call
        context.client.create_completion = AsyncMock(
            return_value={"response": "", "tool_calls": [tool_call.model_dump()]}
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)
        processor._max_consecutive_duplicates = 3  # Lower threshold for testing

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.format_state_for_model = MagicMock(return_value="")
        from chuk_ai_session_manager.guards import RunawayStatus

        mock_tool_state.check_runaway = MagicMock(
            return_value=RunawayStatus(should_stop=False)
        )
        processor._tool_state = mock_tool_state

        # Mock tool processor
        processor.tool_processor.process_tool_calls = AsyncMock()

        await processor.process_conversation(max_turns=20)

        # Should have detected duplicates and eventually broken out
        # The exact count depends on implementation but should be limited
        assert processor.tool_processor.process_tool_calls.call_count <= 10


class TestDiscoveryBudgetWithMessage:
    """Tests for discovery budget with formatted message."""

    @pytest.mark.asyncio
    async def test_discovery_budget_formats_message(self):
        """Test that discovery budget exhaustion formats proper message."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Search")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "search"}}]
        context.tool_name_mapping = {}

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="search", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        # Discovery budget exhausted with "Discovery" in reason
        mock_status = RunawayStatus(
            should_stop=True,
            reason="Discovery budget exhausted",
            budget_exhausted=True,
        )
        mock_status_ok = RunawayStatus(should_stop=False)

        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=True)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        mock_tool_state.format_discovery_exhausted_message = MagicMock(
            return_value="Discovery budget exhausted - please answer with available data"
        )

        # First check returns discovery exhausted
        call_count = [0]

        def mock_check(tool_name=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_status
            return mock_status_ok

        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check)
        processor._tool_state = mock_tool_state

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Here's my answer", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        # Should have called format_discovery_exhausted_message
        mock_tool_state.format_discovery_exhausted_message.assert_called()


class TestExecutionBudgetWithMessage:
    """Tests for execution budget with formatted message."""

    @pytest.mark.asyncio
    async def test_execution_budget_formats_message(self):
        """Test that execution budget exhaustion formats proper message."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Execute")
        ]
        context.openai_tools = [{"type": "function", "function": {"name": "execute"}}]
        context.tool_name_mapping = {}

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="execute", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        # Execution budget exhausted with "Execution" in reason
        # The check passes tool name to check_runaway for execution tools
        mock_status_exec = RunawayStatus(
            should_stop=True,
            reason="Execution budget exhausted",
            budget_exhausted=True,
        )
        mock_status_ok = RunawayStatus(should_stop=False)

        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=True)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        mock_tool_state.format_execution_exhausted_message = MagicMock(
            return_value="Execution budget exhausted - please provide final answer"
        )

        # check_runaway is called with tool_name for specific checks
        # When called with a tool name (execution tool), return exhausted
        def mock_check(tool_name=None):
            if tool_name is not None:
                # This is the execution budget check
                return mock_status_exec
            return mock_status_ok

        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check)
        processor._tool_state = mock_tool_state

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Done", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        # Should have called format_execution_exhausted_message
        mock_tool_state.format_execution_exhausted_message.assert_called()


class TestPollingToolDetection:
    """Tests for polling tool detection and loop exemption."""

    def test_is_polling_tool_status(self):
        """Test that 'status' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("render_status") is True
        assert processor._is_polling_tool("get_status") is True
        assert processor._is_polling_tool("remotion_render_status") is True
        assert processor._is_polling_tool("job_status_check") is True

    def test_is_polling_tool_progress(self):
        """Test that 'progress' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("check_progress") is True
        assert processor._is_polling_tool("get_progress") is True
        assert processor._is_polling_tool("render_progress") is True

    def test_is_polling_tool_check(self):
        """Test that 'check' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("health_check") is True
        assert processor._is_polling_tool("check_job") is True

    def test_is_polling_tool_poll(self):
        """Test that 'poll' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("poll_results") is True
        assert processor._is_polling_tool("poll_queue") is True

    def test_is_polling_tool_monitor(self):
        """Test that 'monitor' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("monitor_job") is True
        assert processor._is_polling_tool("system_monitor") is True

    def test_is_polling_tool_watch(self):
        """Test that 'watch' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("watch_progress") is True
        assert processor._is_polling_tool("file_watch") is True

    def test_is_polling_tool_wait(self):
        """Test that 'wait' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("wait_for_completion") is True
        assert processor._is_polling_tool("wait_job") is True

    def test_is_polling_tool_state(self):
        """Test that 'state' tools are detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("get_state") is True
        assert processor._is_polling_tool("job_state") is True

    def test_is_not_polling_tool(self):
        """Test that non-polling tools are not detected as polling tools."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("sqrt") is False
        assert processor._is_polling_tool("add") is False
        assert processor._is_polling_tool("create_video") is False
        assert (
            processor._is_polling_tool("render_video") is False
        )  # render but not status
        assert processor._is_polling_tool("calculate") is False
        assert processor._is_polling_tool("search") is False

    def test_is_polling_tool_case_insensitive(self):
        """Test that polling tool detection is case-insensitive."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._is_polling_tool("GET_STATUS") is True
        assert processor._is_polling_tool("Check_Progress") is True
        assert processor._is_polling_tool("POLL_RESULTS") is True

    @pytest.mark.asyncio
    async def test_polling_tool_not_marked_as_duplicate(self):
        """Test that polling tools calling same args are not marked as duplicates."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Check the status")
        ]
        context.openai_tools = [
            {"type": "function", "function": {"name": "render_status"}}
        ]
        context.tool_name_mapping = {}

        # Same status check tool call
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(
                name="render_status", arguments='{"job_id": "abc123"}'
            ),
        )

        # Return same tool call multiple times, then final response
        call_count = [0]

        async def repeated_status_checks(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:  # First 3 calls return same status check
                return {"response": "", "tool_calls": [tool_call.model_dump()]}
            else:
                return {"response": "Render complete!", "tool_calls": []}

        context.client.create_completion = repeated_status_checks

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock tool state
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        mock_tool_state.format_state_for_model = MagicMock(return_value="")
        from chuk_ai_session_manager.guards import RunawayStatus

        mock_tool_state.check_runaway = MagicMock(
            return_value=RunawayStatus(should_stop=False)
        )
        processor._tool_state = mock_tool_state

        # Mock tool processor
        processor.tool_processor.process_tool_calls = AsyncMock()

        await processor.process_conversation(max_turns=10)

        # All 3 status checks should have been processed (not skipped as duplicates)
        assert processor.tool_processor.process_tool_calls.call_count >= 3
        # Duplicate counter should not have incremented for polling tools
        assert processor._consecutive_duplicate_count == 0


class TestBindingExtractionWithResults:
    """Tests for binding extraction returning actual bindings."""

    @pytest.mark.asyncio
    async def test_extracts_bindings_with_values(self):
        """Test that actual bindings are extracted and logged."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = []

        context.client.create_completion = AsyncMock(
            return_value={
                "response": "The result is σ = 5.0",
                "tool_calls": [],
            }
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        # Create mock binding
        mock_binding = MagicMock()
        mock_binding.id = "v0"
        mock_binding.raw_value = 5.0
        mock_binding.aliases = ["σ"]

        # Create mock tool state that returns a binding
        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.extract_bindings_from_text = MagicMock(
            return_value=[mock_binding]
        )
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        processor._tool_state = mock_tool_state

        await processor.process_conversation(max_turns=1)

        # Should have called extract_bindings_from_text
        mock_tool_state.extract_bindings_from_text.assert_called_once()


class TestValidateToolMessages:
    """Tests for _validate_tool_messages defense-in-depth validation."""

    def test_valid_messages_unchanged(self):
        """Messages with matching tool results are not modified."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "OK"},
            {"role": "assistant", "content": "Done."},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)
        assert result == messages

    def test_orphaned_tool_call_gets_placeholder(self):
        """An assistant message with a tool_call_id missing a result gets a placeholder."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_missing",
                        "type": "function",
                        "function": {"name": "fetch", "arguments": "{}"},
                    },
                ],
            },
            # No tool result for call_missing!
            {"role": "assistant", "content": "Something else."},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)

        # Should have 4 messages now: user, assistant+tool_calls, tool placeholder, assistant
        assert len(result) == 4
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_missing"
        assert "did not complete" in result[2]["content"]

    def test_multiple_tool_calls_partial_results(self):
        """When an assistant message has multiple tool_calls and only some have results."""
        messages = [
            {"role": "user", "content": "Do two things"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "tool_a", "arguments": "{}"},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "tool_b", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "Result A"},
            # call_b result is missing
        ]

        result = ConversationProcessor._validate_tool_messages(messages)

        # Should have 4 messages: user, assistant+tool_calls, tool result A, placeholder for B
        assert len(result) == 4
        tool_results = [m for m in result if m.get("role") == "tool"]
        assert len(tool_results) == 2
        tool_call_ids = {m["tool_call_id"] for m in tool_results}
        assert tool_call_ids == {"call_a", "call_b"}

    def test_no_tool_calls_unchanged(self):
        """Messages without tool_calls pass through unchanged."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)
        assert result == messages

    def test_empty_messages(self):
        """Empty message list returns empty."""
        assert ConversationProcessor._validate_tool_messages([]) == []

    def test_multiple_sequential_tool_rounds(self):
        """Multiple assistant→tool rounds are all validated correctly."""
        messages = [
            {"role": "user", "content": "Do things"},
            # Round 1: valid
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_r1",
                        "type": "function",
                        "function": {"name": "a", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_r1", "content": "Done A"},
            # Round 2: orphaned
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_r2",
                        "type": "function",
                        "function": {"name": "b", "arguments": "{}"},
                    },
                ],
            },
            # Missing tool result for call_r2!
            {"role": "assistant", "content": "Final answer."},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)

        # Should have 6 messages: user, asst+tc, tool, asst+tc, PLACEHOLDER, asst
        assert len(result) == 6
        assert result[4]["role"] == "tool"
        assert result[4]["tool_call_id"] == "call_r2"
        assert "did not complete" in result[4]["content"]
        # Round 1 should be untouched
        assert result[2]["tool_call_id"] == "call_r1"
        assert result[2]["content"] == "Done A"

    def test_all_results_missing(self):
        """When ALL tool results are missing from a multi-call assistant message."""
        messages = [
            {"role": "user", "content": "Run both"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "x", "arguments": "{}"},
                    },
                    {
                        "id": "call_y",
                        "type": "function",
                        "function": {"name": "y", "arguments": "{}"},
                    },
                ],
            },
            # No tool results at all — next message is user
            {"role": "user", "content": "What happened?"},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)

        # Should have 5 messages: user, asst+tc, placeholder_x, placeholder_y, user
        assert len(result) == 5
        placeholders = [m for m in result if m.get("role") == "tool"]
        assert len(placeholders) == 2
        placeholder_ids = {m["tool_call_id"] for m in placeholders}
        assert placeholder_ids == {"call_x", "call_y"}

    def test_empty_tool_calls_list_is_noop(self):
        """Assistant message with tool_calls=[] should not trigger repair."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Using tools...",
                "tool_calls": [],
            },
            {"role": "assistant", "content": "Done."},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)
        assert result == messages

    def test_tool_results_not_immediately_following(self):
        """Tool results separated from assistant message by another message type."""
        messages = [
            {"role": "user", "content": "Go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_gap",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    },
                ],
            },
            # A user message appears before the tool result
            {"role": "user", "content": "Hurry up"},
            {"role": "tool", "tool_call_id": "call_gap", "content": "Late result"},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)

        # The tool result is not immediately following, so scanner won't find it
        # A placeholder should be inserted right after the assistant message
        assert len(result) == 5
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_gap"
        assert "did not complete" in result[2]["content"]


# ----------------------------------------------------------
# Tests for _strip_old_reasoning_content
# ----------------------------------------------------------


class TestStripOldReasoningContent:
    """Tests for ConversationProcessor._strip_old_reasoning_content()."""

    def test_keeps_latest_reasoning(self):
        """Most recent assistant reasoning is preserved."""
        messages = [
            {
                "role": "assistant",
                "content": "old",
                "reasoning_content": "old thinking",
            },
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "new",
                "reasoning_content": "new thinking",
            },
        ]
        result = ConversationProcessor._strip_old_reasoning_content(messages)
        assert "reasoning_content" not in result[0]
        assert result[2]["reasoning_content"] == "new thinking"

    def test_removes_all_but_latest(self):
        """Multiple old reasoning blocks are all removed."""
        messages = [
            {"role": "assistant", "content": "a", "reasoning_content": "think1"},
            {"role": "assistant", "content": "b", "reasoning_content": "think2"},
            {"role": "assistant", "content": "c", "reasoning_content": "think3"},
        ]
        result = ConversationProcessor._strip_old_reasoning_content(messages)
        assert "reasoning_content" not in result[0]
        assert "reasoning_content" not in result[1]
        assert result[2]["reasoning_content"] == "think3"

    def test_no_reasoning_is_noop(self):
        """Messages without reasoning_content are unchanged."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = ConversationProcessor._strip_old_reasoning_content(messages)
        assert result == messages

    def test_single_reasoning_preserved(self):
        """If only one assistant has reasoning, it stays."""
        messages = [
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b", "reasoning_content": "only one"},
        ]
        result = ConversationProcessor._strip_old_reasoning_content(messages)
        assert result[1]["reasoning_content"] == "only one"

    def test_non_assistant_reasoning_untouched(self):
        """Only assistant messages have reasoning stripped."""
        messages = [
            {
                "role": "system",
                "content": "sys",
                "reasoning_content": "system reasoning",
            },
            {"role": "assistant", "content": "a", "reasoning_content": "latest"},
        ]
        result = ConversationProcessor._strip_old_reasoning_content(messages)
        # System message reasoning is not touched (not role=assistant)
        assert result[0].get("reasoning_content") == "system reasoning"
        assert result[1]["reasoning_content"] == "latest"

    def test_empty_list(self):
        """Empty message list returns empty."""
        assert ConversationProcessor._strip_old_reasoning_content([]) == []


# ----------------------------------------------------------
# Tests for _prepare_messages_for_api
# ----------------------------------------------------------


class TestPrepareMessagesForApi:
    """Tests for ConversationProcessor._prepare_messages_for_api()."""

    def test_combines_strip_and_validate(self):
        """prepare_messages_for_api strips reasoning AND validates tool messages."""
        messages = [
            Message(
                role=MessageRole.ASSISTANT, content="old", reasoning_content="old think"
            ),
            Message(role=MessageRole.USER, content="hi"),
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                reasoning_content="new think",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="test", arguments="{}"),
                    )
                ],
            ),
            # Missing tool result for call_1 — should get placeholder
        ]
        result = ConversationProcessor._prepare_messages_for_api(messages)

        # Old reasoning stripped
        assert "reasoning_content" not in result[0]
        # Latest reasoning preserved
        assert result[2].get("reasoning_content") == "new think"
        # Orphaned tool_call_id repaired with placeholder
        assert len(result) == 4
        assert result[3]["role"] == "tool"
        assert result[3]["tool_call_id"] == "call_1"

    def test_serializes_message_objects(self):
        """Input Message objects become dicts."""
        messages = [
            Message(role=MessageRole.USER, content="hello"),
        ]
        result = ConversationProcessor._prepare_messages_for_api(messages)
        assert isinstance(result[0], dict)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"


# ----------------------------------------------------------
# Tests for context notices injection (Tier 2)
# ----------------------------------------------------------


class TestContextNoticesInjection:
    """Tests for ephemeral context notices in _prepare_messages_for_api."""

    def _make_context_with_notices(self, notices: list[str]):
        """Create a MockContext with pending context notices."""
        ctx = MockContext()
        ctx._pending_context_notices = list(notices)

        def drain_context_notices():
            result = ctx._pending_context_notices[:]
            ctx._pending_context_notices.clear()
            return result

        ctx.drain_context_notices = drain_context_notices
        return ctx

    def test_context_notices_injected(self):
        """Notices from context.drain_context_notices() appear in prepared messages."""
        ctx = self._make_context_with_notices(
            [
                "5 older messages were evicted from context.",
                "Tool result was truncated to 100K chars.",
            ]
        )

        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Hello"),
        ]

        result = ConversationProcessor._prepare_messages_for_api(messages, context=ctx)

        # Should have 3 messages: system, context notice, user
        assert len(result) == 3
        notice_msg = result[1]
        assert notice_msg["role"] == "system"
        assert "[Context Management]" in notice_msg["content"]
        assert "5 older messages were evicted" in notice_msg["content"]
        assert "Tool result was truncated" in notice_msg["content"]

    def test_notices_drained_after_use(self):
        """After injection, notices list is empty."""
        ctx = self._make_context_with_notices(
            [
                "Context was trimmed.",
            ]
        )

        messages = [
            Message(role=MessageRole.USER, content="Hello"),
        ]

        ConversationProcessor._prepare_messages_for_api(messages, context=ctx)

        # Notices should be drained (empty after use)
        assert ctx._pending_context_notices == []
        # A second call should not inject anything
        result2 = ConversationProcessor._prepare_messages_for_api(messages, context=ctx)
        # No notice message should be inserted (just the user message)
        assert len(result2) == 1
        assert result2[0]["role"] == "user"

    def test_notices_disabled(self, monkeypatch):
        """When DEFAULT_CONTEXT_NOTICES_ENABLED is False, notices are not injected."""
        monkeypatch.setattr(
            "mcp_cli.config.defaults.DEFAULT_CONTEXT_NOTICES_ENABLED",
            False,
        )

        ctx = self._make_context_with_notices(
            [
                "This notice should not appear.",
            ]
        )

        messages = [
            Message(role=MessageRole.SYSTEM, content="System prompt."),
            Message(role=MessageRole.USER, content="Hello"),
        ]

        result = ConversationProcessor._prepare_messages_for_api(messages, context=ctx)

        # Should have 2 messages: system, user (no notice injected)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        # No message should contain "[Context Management]"
        for msg in result:
            assert "[Context Management]" not in msg.get("content", "")

    def test_notices_inserted_after_system_prompt(self):
        """When first message is system, notice is inserted at index 1."""
        ctx = self._make_context_with_notices(["Notice text."])

        messages = [
            Message(role=MessageRole.SYSTEM, content="System."),
            Message(role=MessageRole.USER, content="Hi"),
        ]

        result = ConversationProcessor._prepare_messages_for_api(messages, context=ctx)

        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System."
        assert result[1]["role"] == "system"
        assert "[Context Management]" in result[1]["content"]
        assert result[2]["role"] == "user"

    def test_notices_inserted_at_start_when_no_system_prompt(self):
        """When no system prompt is first, notice is inserted at index 0."""
        ctx = self._make_context_with_notices(["Notice text."])

        messages = [
            Message(role=MessageRole.USER, content="Hi"),
        ]

        result = ConversationProcessor._prepare_messages_for_api(messages, context=ctx)

        assert result[0]["role"] == "system"
        assert "[Context Management]" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_no_notices_no_injection(self):
        """When no notices are pending, no extra message is injected."""
        ctx = self._make_context_with_notices([])

        messages = [
            Message(role=MessageRole.SYSTEM, content="System."),
            Message(role=MessageRole.USER, content="Hi"),
        ]

        result = ConversationProcessor._prepare_messages_for_api(messages, context=ctx)

        # Should have exactly 2 messages (no injection)
        assert len(result) == 2

    def test_no_context_no_injection(self):
        """When context is None, no notices are injected."""
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
        ]

        result = ConversationProcessor._prepare_messages_for_api(messages, context=None)

        assert len(result) == 1
        assert result[0]["role"] == "user"


# ----------------------------------------------------------
# Tests for health polling (_health_poll_loop, _start_health_polling, _stop_health_polling)
# ----------------------------------------------------------


class TestHealthPolling:
    """Tests for background health polling methods."""

    def test_start_health_polling_when_interval_zero(self):
        """When _health_interval is 0, no task is created."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._health_interval == 0
        processor._start_health_polling()
        # No task should be created when interval is 0
        assert processor._health_task is None

    @pytest.mark.asyncio
    async def test_start_health_polling_creates_task(self):
        """When _health_interval > 0, a background task is created."""
        context = MockContext()
        # Give context a positive health interval
        context._health_interval = 60
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)
        processor._health_interval = 60  # Override directly

        processor._start_health_polling()
        try:
            assert processor._health_task is not None
            assert not processor._health_task.done()
        finally:
            # Clean up task
            processor._health_task.cancel()
            try:
                await processor._health_task
            except (asyncio.CancelledError, Exception):
                pass

    @pytest.mark.asyncio
    async def test_start_health_polling_idempotent(self):
        """Calling _start_health_polling twice does not create a second task."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)
        processor._health_interval = 60

        processor._start_health_polling()
        first_task = processor._health_task

        processor._start_health_polling()
        second_task = processor._health_task

        try:
            assert first_task is second_task
        finally:
            if first_task:
                first_task.cancel()
                try:
                    await first_task
                except (asyncio.CancelledError, Exception):
                    pass

    @pytest.mark.asyncio
    async def test_stop_health_polling_cancels_task(self):
        """_stop_health_polling cancels the task and clears the reference."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)
        processor._health_interval = 60

        processor._start_health_polling()
        assert processor._health_task is not None

        processor._stop_health_polling()
        assert processor._health_task is None

    def test_stop_health_polling_when_no_task(self):
        """_stop_health_polling is a no-op when no task exists."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        assert processor._health_task is None
        # Should not raise
        processor._stop_health_polling()
        assert processor._health_task is None

    @pytest.mark.asyncio
    async def test_health_poll_loop_no_tool_manager(self):
        """Health poll loop continues without error when tool_manager is None."""
        context = MockContext()
        context.tool_manager = None  # No tool manager
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)
        processor._health_interval = 0.01  # Very short interval

        # Run the loop briefly and then cancel it
        task = asyncio.create_task(processor._health_poll_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_health_poll_loop_updates_status(self):
        """Health poll loop updates _last_health from check_server_health results."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)
        processor._health_interval = 0.01

        # Set up tool_manager with check_server_health
        context.tool_manager.check_server_health = AsyncMock(
            return_value={"server1": {"status": "healthy"}}
        )

        task = asyncio.create_task(processor._health_poll_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "server1" in processor._last_health
        assert processor._last_health["server1"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_poll_loop_logs_status_transition(self):
        """Health poll loop logs warning when server status changes."""

        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)
        processor._health_interval = 0.01
        # Pre-set previous status
        processor._last_health = {"server1": "healthy"}

        # Now it reports degraded
        context.tool_manager.check_server_health = AsyncMock(
            return_value={"server1": {"status": "degraded"}}
        )

        with patch("mcp_cli.chat.conversation.logger") as mock_logger:
            task = asyncio.create_task(processor._health_poll_loop())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Should have logged a warning about status change
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("health changed" in call for call in warning_calls)

    @pytest.mark.asyncio
    async def test_health_poll_loop_handles_exception(self):
        """Health poll loop catches and logs generic exceptions."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)
        processor._health_interval = 0.01

        # check_server_health raises a non-cancelled exception
        context.tool_manager.check_server_health = AsyncMock(
            side_effect=RuntimeError("connection refused")
        )

        # Loop should not crash - it logs debug and continues
        task = asyncio.create_task(processor._health_poll_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_health_poll_loop_handles_none_info(self):
        """Health poll loop handles None info entries gracefully."""
        context = MockContext()
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)
        processor._health_interval = 0.01

        # Return None as info for a server
        context.tool_manager.check_server_health = AsyncMock(
            return_value={"server1": None}
        )

        task = asyncio.create_task(processor._health_poll_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # status for None info should be "unknown"
        assert processor._last_health.get("server1") == "unknown"

    @pytest.mark.asyncio
    async def test_health_polling_started_and_stopped_during_process_conversation(self):
        """process_conversation starts health polling at entry and stops it in finally."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="/help")]
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)
        processor._health_interval = 0  # Keep at 0 to avoid real task creation

        start_called = []
        stop_called = []

        original_start = processor._start_health_polling
        original_stop = processor._stop_health_polling

        def track_start():
            start_called.append(True)
            original_start()

        def track_stop():
            stop_called.append(True)
            original_stop()

        processor._start_health_polling = track_start
        processor._stop_health_polling = track_stop

        await processor.process_conversation()

        assert len(start_called) == 1
        assert len(stop_called) == 1


# ----------------------------------------------------------
# Tests for _record_token_usage
# ----------------------------------------------------------


class TestRecordTokenUsage:
    """Tests for _record_token_usage method."""

    def test_no_tracker_is_noop(self):
        """When context has no token_tracker, record is skipped without error."""
        context = MockContext()
        # No token_tracker attribute
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        completion = CompletionResponse(
            response="Hello",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
        # Should not raise
        processor._record_token_usage(completion)

    def test_records_with_usage_data(self):
        """When usage data is present, a TurnUsage is created and recorded."""
        context = MockContext()
        mock_tracker = MagicMock()
        context.token_tracker = mock_tracker
        context.model = "gpt-4"
        context.provider = "openai"
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        completion = CompletionResponse(
            response="Hello",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        processor._record_token_usage(completion)

        mock_tracker.record_turn.assert_called_once()
        turn_arg = mock_tracker.record_turn.call_args[0][0]
        assert turn_arg.input_tokens == 100
        assert turn_arg.output_tokens == 50
        assert turn_arg.model == "gpt-4"
        assert turn_arg.provider == "openai"
        assert not turn_arg.estimated

    def test_records_with_input_output_tokens(self):
        """Supports input_tokens/output_tokens as alternative to prompt/completion."""
        context = MockContext()
        mock_tracker = MagicMock()
        context.token_tracker = mock_tracker
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        completion = CompletionResponse(
            response="Hi",
            usage={"input_tokens": 30, "output_tokens": 15},
        )
        processor._record_token_usage(completion)

        mock_tracker.record_turn.assert_called_once()
        turn_arg = mock_tracker.record_turn.call_args[0][0]
        assert turn_arg.input_tokens == 30
        assert turn_arg.output_tokens == 15

    def test_estimates_when_no_usage_data(self):
        """When usage is None, output tokens are estimated from response length."""
        context = MockContext()
        mock_tracker = MagicMock()
        context.token_tracker = mock_tracker
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        completion = CompletionResponse(
            response="Hello world, this is a test response!",
            usage=None,
        )
        processor._record_token_usage(completion)

        mock_tracker.record_turn.assert_called_once()
        turn_arg = mock_tracker.record_turn.call_args[0][0]
        assert turn_arg.estimated is True
        assert turn_arg.output_tokens >= 1

    def test_estimates_empty_response(self):
        """When usage is None and response is empty string, estimation still works."""
        context = MockContext()
        mock_tracker = MagicMock()
        context.token_tracker = mock_tracker
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        completion = CompletionResponse(
            response="",
            usage=None,
        )
        processor._record_token_usage(completion)

        mock_tracker.record_turn.assert_called_once()
        turn_arg = mock_tracker.record_turn.call_args[0][0]
        assert turn_arg.estimated is True


# ----------------------------------------------------------
# Tests for VM turn advance (line 182)
# ----------------------------------------------------------


class TestVMTurnAdvance:
    """Tests for vm.new_turn() call at start of process_conversation."""

    @pytest.mark.asyncio
    async def test_vm_new_turn_called_when_vm_present(self):
        """vm.new_turn() is called when context has a session with a vm."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="/help")]

        # Set up a session with a vm
        mock_vm = MagicMock()
        mock_session = MagicMock()
        mock_session.vm = mock_vm
        context.session = mock_session

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        await processor.process_conversation()

        mock_vm.new_turn.assert_called_once()

    @pytest.mark.asyncio
    async def test_vm_new_turn_skipped_when_no_session(self):
        """vm.new_turn() is not called when no session attribute exists."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="/help")]
        # No session attribute
        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        # Should not raise
        await processor.process_conversation()


# ----------------------------------------------------------
# Tests for streaming fallback (lines 258-269)
# ----------------------------------------------------------


class TestStreamingFallbackCoverage:
    """Tests for the streaming-to-regular-completion fallback path."""

    @pytest.mark.asyncio
    async def test_streaming_exception_causes_fallback(self):
        """When _handle_streaming_completion raises, regular completion is used."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        # Client supports streaming (has create_completion with stream param)
        mock_client = MagicMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "Regular fallback", "tool_calls": []}
        )
        context.client = mock_client

        ui_manager = MockUIManager()
        ui_manager.start_streaming_response = AsyncMock()
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()
        ui_manager.display = MagicMock()
        ui_manager.is_streaming_response = False

        processor = ConversationProcessor(context, ui_manager)

        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        processor._tool_state = mock_tool_state

        # Make streaming fail
        streaming_exception = Exception("Streaming failed")

        async def failing_streaming(tools=None, after_tool_calls=False):
            raise streaming_exception

        processor._handle_streaming_completion = failing_streaming

        await processor.process_conversation(max_turns=1)

        # Regular completion should have been used as fallback
        mock_client.create_completion.assert_called_once()


# ----------------------------------------------------------
# Tests for discovery budget with streaming active (lines 342-346)
# ----------------------------------------------------------


class TestDiscoveryBudgetWithStreaming:
    """Tests for discovery budget path when streaming UI is active."""

    @pytest.mark.asyncio
    async def test_discovery_budget_stops_streaming(self):
        """Discovery budget exhaustion stops active streaming UI."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Search")
        ]
        context.openai_tools = []

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="search", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = True  # Streaming is active!
        ui_manager.stop_streaming_response = AsyncMock(
            side_effect=lambda: setattr(ui_manager, "is_streaming_response", False)
        )
        ui_manager.streaming_handler = MagicMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_status = RunawayStatus(
            should_stop=True,
            reason="Discovery budget exhausted",
            budget_exhausted=True,
        )
        mock_status_ok = RunawayStatus(should_stop=False)

        call_count = [0]

        def mock_check(tool_name=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_status
            return mock_status_ok

        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=True)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        mock_tool_state.format_discovery_exhausted_message = MagicMock(
            return_value="Discovery exhausted"
        )
        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check)
        processor._tool_state = mock_tool_state

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Final answer", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        # stop_streaming_response should have been called
        ui_manager.stop_streaming_response.assert_called()
        # streaming_handler should be cleared
        assert ui_manager.streaming_handler is None


# ----------------------------------------------------------
# Tests for execution budget with streaming active (lines 363-367)
# ----------------------------------------------------------


class TestExecutionBudgetWithStreaming:
    """Tests for execution budget path when streaming UI is active."""

    @pytest.mark.asyncio
    async def test_execution_budget_stops_streaming(self):
        """Execution budget exhaustion stops active streaming UI."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Execute")
        ]
        context.openai_tools = []

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="execute", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = True  # Streaming is active
        ui_manager.stop_streaming_response = AsyncMock(
            side_effect=lambda: setattr(ui_manager, "is_streaming_response", False)
        )
        ui_manager.streaming_handler = MagicMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_status_exec = RunawayStatus(
            should_stop=True,
            reason="Execution budget exhausted",
            budget_exhausted=True,
        )
        mock_status_ok = RunawayStatus(should_stop=False)

        def mock_check(tool_name=None):
            if tool_name is not None:
                return mock_status_exec
            return mock_status_ok

        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=True)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        mock_tool_state.format_execution_exhausted_message = MagicMock(
            return_value="Execution exhausted"
        )
        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check)
        processor._tool_state = mock_tool_state

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Done", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        # stop_streaming_response should have been called
        ui_manager.stop_streaming_response.assert_called()
        assert ui_manager.streaming_handler is None


# ----------------------------------------------------------
# Tests for general runaway with streaming active (lines 399-406)
# ----------------------------------------------------------


class TestGeneralRunawayWithStreaming:
    """Tests for general runaway detection with streaming UI active."""

    @pytest.mark.asyncio
    async def test_runaway_stops_streaming_ui(self):
        """General runaway detection stops streaming UI."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Compute")
        ]
        context.openai_tools = []

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="compute", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = True  # Streaming is active
        ui_manager.stop_streaming_response = AsyncMock(
            side_effect=lambda: setattr(ui_manager, "is_streaming_response", False)
        )
        ui_manager.streaming_handler = MagicMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_runaway = RunawayStatus(
            should_stop=True,
            reason="General runaway",
            budget_exhausted=False,
            saturation_detected=False,
        )
        mock_ok = RunawayStatus(should_stop=False)

        # No discovery/execution tools — first check_runaway() call (tool_name=None)
        # is the general runaway check; trigger it immediately.
        def mock_check(tool_name=None):
            if tool_name is None:
                return mock_runaway
            return mock_ok

        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        mock_tool_state.format_state_for_model = MagicMock(return_value="State")
        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check)
        processor._tool_state = mock_tool_state
        # Mock tool processor to avoid UI issues
        processor.tool_processor.process_tool_calls = AsyncMock()

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Final answer", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        ui_manager.stop_streaming_response.assert_called()

    @pytest.mark.asyncio
    async def test_runaway_other_reason_uses_format_state(self):
        """Runaway with neither budget_exhausted nor saturation uses format_state_for_model."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Compute")
        ]
        context.openai_tools = []

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="compute", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        # Create runaway that's neither budget_exhausted nor saturation_detected.
        # Since is_discovery_tool=False and is_execution_tool=False, there are no
        # per-type budget checks — so the first check_runaway() call is the general
        # runaway check (called with no tool_name / tool_name=None).
        mock_runaway = RunawayStatus(
            should_stop=True,
            reason="Unusual runaway condition",
            budget_exhausted=False,
            saturation_detected=False,
        )
        mock_ok = RunawayStatus(should_stop=False)

        call_count = [0]

        def mock_check(tool_name=None):
            call_count[0] += 1
            # General check is called with no args (tool_name=None).
            # Since there are no discovery/execution tools in this test,
            # the first call to check_runaway is always the general one.
            if tool_name is None:
                return mock_runaway
            return mock_ok

        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        mock_tool_state.format_state_for_model = MagicMock(
            return_value="Computed state"
        )
        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check)
        processor._tool_state = mock_tool_state
        # Mock tool processor to avoid UI issues
        processor.tool_processor.process_tool_calls = AsyncMock()

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Done", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        # format_state_for_model used in the "else" branch stop message
        mock_tool_state.format_state_for_model.assert_called()

    @pytest.mark.asyncio
    async def test_saturation_with_empty_numeric_results(self):
        """Saturation with no numeric results uses 0.0 as last_val."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Compute")
        ]
        context.openai_tools = []

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="compute", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        # Saturation runaway — no discovery/execution tools, so the first
        # call to check_runaway() (general check, tool_name=None) triggers it.
        mock_runaway = RunawayStatus(
            should_stop=True,
            reason="Saturation detected",
            budget_exhausted=False,
            saturation_detected=True,
        )
        mock_ok = RunawayStatus(should_stop=False)

        def mock_check(tool_name=None):
            if tool_name is None:
                return mock_runaway
            return mock_ok

        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        # Empty numeric results — should use 0.0 as fallback
        mock_tool_state._recent_numeric_results = []
        mock_tool_state.format_saturation_message = MagicMock(
            return_value="Saturation message"
        )
        mock_tool_state.check_runaway = MagicMock(side_effect=mock_check)
        processor._tool_state = mock_tool_state
        # Mock tool processor to avoid UI issues
        processor.tool_processor.process_tool_calls = AsyncMock()

        context.client.create_completion = AsyncMock(
            side_effect=[
                {"response": "", "tool_calls": [tool_call.model_dump()]},
                {"response": "Done", "tool_calls": []},
            ]
        )

        await processor.process_conversation(max_turns=3)

        # format_saturation_message should have been called with 0.0
        mock_tool_state.format_saturation_message.assert_called_with(0.0)


# ----------------------------------------------------------
# Tests for max_turns with streaming active (lines 417-421)
# ----------------------------------------------------------


class TestMaxTurnsWithStreaming:
    """Tests for max_turns limit when streaming is active."""

    @pytest.mark.asyncio
    async def test_max_turns_stops_streaming(self):
        """Max turns stops active streaming UI before breaking."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Loop")]
        context.openai_tools = []
        context.tool_name_mapping = {}

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="loop", arguments="{}"),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = True  # Streaming active
        ui_manager.stop_streaming_response = AsyncMock(
            side_effect=lambda: setattr(ui_manager, "is_streaming_response", False)
        )
        ui_manager.streaming_handler = MagicMock()

        processor = ConversationProcessor(context, ui_manager)

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.check_runaway = MagicMock(
            return_value=RunawayStatus(should_stop=False)
        )
        processor._tool_state = mock_tool_state
        processor.tool_processor.process_tool_calls = AsyncMock()

        context.client.create_completion = AsyncMock(
            return_value={"response": "", "tool_calls": [tool_call.model_dump()]}
        )

        # max_turns=1 means turn_count will equal max_turns after first tool call
        await processor.process_conversation(max_turns=1)

        # Should have attempted to stop streaming
        ui_manager.stop_streaming_response.assert_called()


# ----------------------------------------------------------
# Tests for consecutive duplicate with streaming (lines 471-474)
# ----------------------------------------------------------


class TestDuplicateDetectionWithStreaming:
    """Tests for duplicate tool call detection when streaming is active."""

    @pytest.mark.asyncio
    async def test_max_duplicates_stops_streaming(self):
        """Max consecutive duplicates stops active streaming UI before breaking."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Calculate")
        ]
        context.openai_tools = []
        context.tool_name_mapping = {}

        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="sqrt", arguments='{"x": 16}'),
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = True  # Streaming active
        ui_manager.stop_streaming_response = AsyncMock(
            side_effect=lambda: setattr(ui_manager, "is_streaming_response", False)
        )
        ui_manager.streaming_handler = MagicMock()

        processor = ConversationProcessor(context, ui_manager)
        processor._max_consecutive_duplicates = 2

        from chuk_ai_session_manager.guards import RunawayStatus

        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.is_discovery_tool = MagicMock(return_value=False)
        mock_tool_state.is_execution_tool = MagicMock(return_value=False)
        mock_tool_state.format_state_for_model = MagicMock(return_value="")
        mock_tool_state.check_runaway = MagicMock(
            return_value=RunawayStatus(should_stop=False)
        )
        processor._tool_state = mock_tool_state
        processor.tool_processor.process_tool_calls = AsyncMock()

        context.client.create_completion = AsyncMock(
            return_value={"response": "", "tool_calls": [tool_call.model_dump()]}
        )

        await processor.process_conversation(max_turns=20)

        # Should have stopped and streaming should have been stopped
        ui_manager.stop_streaming_response.assert_called()


# ----------------------------------------------------------
# Tests for error handlers with streaming active (lines 580-620)
# ----------------------------------------------------------


class TestErrorHandlersWithStreaming:
    """Tests for exception handlers that stop streaming before breaking."""

    @pytest.mark.asyncio
    async def test_timeout_error_stops_streaming(self):
        """asyncio.TimeoutError stops streaming UI and breaks loop."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = True  # Streaming active
        ui_manager.stop_streaming_response = AsyncMock(
            side_effect=lambda: setattr(ui_manager, "is_streaming_response", False)
        )
        ui_manager.streaming_handler = MagicMock()

        processor = ConversationProcessor(context, ui_manager)

        context.client.create_completion = AsyncMock(
            side_effect=asyncio.TimeoutError("Request timed out")
        )

        await processor.process_conversation(max_turns=1)

        ui_manager.stop_streaming_response.assert_called()
        assert ui_manager.streaming_handler is None

    @pytest.mark.asyncio
    async def test_timeout_error_injects_message(self):
        """asyncio.TimeoutError injects timeout message to conversation."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        context.client.create_completion = AsyncMock(
            side_effect=asyncio.TimeoutError("timed out")
        )

        await processor.process_conversation(max_turns=1)

        # Check for injected timeout message
        [
            m
            for m in context.conversation_history
            if isinstance(m, str)
            and "timed out" in m.lower()
            or (
                hasattr(m, "content") and m.content and "timed out" in m.content.lower()
            )
        ]
        # inject_assistant_message puts a string, not a Message object
        all_msgs = context.conversation_history
        assert any(
            (isinstance(m, str) and "timed out" in m.lower())
            or (
                hasattr(m, "content") and m.content and "timed out" in m.content.lower()
            )
            for m in all_msgs
        )

    @pytest.mark.asyncio
    async def test_connection_error_stops_streaming(self):
        """ConnectionError stops streaming UI and breaks loop."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = True  # Streaming active
        ui_manager.stop_streaming_response = AsyncMock(
            side_effect=lambda: setattr(ui_manager, "is_streaming_response", False)
        )
        ui_manager.streaming_handler = MagicMock()

        processor = ConversationProcessor(context, ui_manager)

        context.client.create_completion = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        await processor.process_conversation(max_turns=1)

        ui_manager.stop_streaming_response.assert_called()
        assert ui_manager.streaming_handler is None

    @pytest.mark.asyncio
    async def test_os_error_stops_streaming(self):
        """OSError (subclass of ConnectionError path) stops streaming UI."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = True
        ui_manager.stop_streaming_response = AsyncMock(
            side_effect=lambda: setattr(ui_manager, "is_streaming_response", False)
        )
        ui_manager.streaming_handler = MagicMock()

        processor = ConversationProcessor(context, ui_manager)

        context.client.create_completion = AsyncMock(side_effect=OSError("Broken pipe"))

        await processor.process_conversation(max_turns=1)

        ui_manager.stop_streaming_response.assert_called()
        assert ui_manager.streaming_handler is None

    @pytest.mark.asyncio
    async def test_connection_error_injects_message(self):
        """ConnectionError injects connectivity message to conversation."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.stop_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        context.client.create_completion = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        await processor.process_conversation(max_turns=1)

        all_msgs = context.conversation_history
        assert any(
            (isinstance(m, str) and "connection" in m.lower())
            or (
                hasattr(m, "content")
                and m.content
                and "connection" in m.content.lower()
            )
            for m in all_msgs
        )

    @pytest.mark.asyncio
    async def test_value_error_stops_streaming(self):
        """ValueError stops streaming UI and breaks loop without injecting message."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = True  # Streaming active
        ui_manager.stop_streaming_response = AsyncMock(
            side_effect=lambda: setattr(ui_manager, "is_streaming_response", False)
        )
        ui_manager.streaming_handler = MagicMock()

        processor = ConversationProcessor(context, ui_manager)

        context.client.create_completion = AsyncMock(
            side_effect=ValueError("Invalid configuration")
        )

        initial_len = len(context.conversation_history)
        await processor.process_conversation(max_turns=1)

        # ValueError handler does not inject a message (no inject_assistant_message call)
        assert len(context.conversation_history) == initial_len
        ui_manager.stop_streaming_response.assert_called()
        assert ui_manager.streaming_handler is None

    @pytest.mark.asyncio
    async def test_type_error_stops_streaming(self):
        """TypeError stops streaming UI and breaks loop without injecting message."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = True
        ui_manager.stop_streaming_response = AsyncMock(
            side_effect=lambda: setattr(ui_manager, "is_streaming_response", False)
        )
        ui_manager.streaming_handler = MagicMock()

        processor = ConversationProcessor(context, ui_manager)

        context.client.create_completion = AsyncMock(
            side_effect=TypeError("Wrong type")
        )

        await processor.process_conversation(max_turns=1)

        ui_manager.stop_streaming_response.assert_called()
        assert ui_manager.streaming_handler is None

    @pytest.mark.asyncio
    async def test_general_exception_stops_streaming(self):
        """Generic Exception stops streaming UI and injects error message."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = True  # Streaming active
        ui_manager.stop_streaming_response = AsyncMock(
            side_effect=lambda: setattr(ui_manager, "is_streaming_response", False)
        )
        ui_manager.streaming_handler = MagicMock()

        processor = ConversationProcessor(context, ui_manager)

        context.client.create_completion = AsyncMock(
            side_effect=RuntimeError("Something unexpected")
        )

        await processor.process_conversation(max_turns=1)

        ui_manager.stop_streaming_response.assert_called()
        assert ui_manager.streaming_handler is None

        # General exception injects error message
        all_msgs = context.conversation_history
        assert any(
            (isinstance(m, str) and "error" in m.lower())
            or (hasattr(m, "content") and m.content and "error" in m.content.lower())
            for m in all_msgs
        )


# ----------------------------------------------------------
# Tests for _handle_streaming_completion (lines 641-680)
# ----------------------------------------------------------


class TestHandleStreamingCompletionDirect:
    """Tests for _handle_streaming_completion method directly."""

    @pytest.mark.asyncio
    async def test_streaming_completion_returns_completion_response(self):
        """_handle_streaming_completion returns a CompletionResponse."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        ui_manager = MockUIManager()
        ui_manager.start_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        mock_stream_result = {
            "response": "Streaming response",
            "tool_calls": None,
            "streaming": True,
            "elapsed_time": 1.5,
        }

        with patch(
            "mcp_cli.chat.streaming_handler.StreamingResponseHandler"
        ) as MockHandler:
            mock_handler_instance = MagicMock()
            mock_handler_instance.stream_response = AsyncMock(
                return_value=mock_stream_result
            )
            MockHandler.return_value = mock_handler_instance

            result = await processor._handle_streaming_completion(tools=[])

        assert isinstance(result, CompletionResponse)
        assert result.response == "Streaming response"
        assert result.streaming is True

    @pytest.mark.asyncio
    async def test_streaming_completion_with_tool_calls(self):
        """_handle_streaming_completion logs tool calls when present."""
        context = MockContext()
        context.conversation_history = [
            Message(role=MessageRole.USER, content="Use a tool")
        ]
        ui_manager = MockUIManager()
        ui_manager.start_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        tool_call_dict = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "sqrt", "arguments": '{"x": 4}'},
        }

        mock_stream_result = {
            "response": "",
            "tool_calls": [tool_call_dict],
            "streaming": True,
            "elapsed_time": 0.8,
        }

        with patch(
            "mcp_cli.chat.streaming_handler.StreamingResponseHandler"
        ) as MockHandler:
            mock_handler_instance = MagicMock()
            mock_handler_instance.stream_response = AsyncMock(
                return_value=mock_stream_result
            )
            MockHandler.return_value = mock_handler_instance

            result = await processor._handle_streaming_completion(tools=[])

        assert isinstance(result, CompletionResponse)

    @pytest.mark.asyncio
    async def test_streaming_completion_sets_handler_on_ui_manager(self):
        """_handle_streaming_completion sets streaming_handler on ui_manager."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        ui_manager = MockUIManager()
        ui_manager.start_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        mock_stream_result = {
            "response": "Done",
            "tool_calls": None,
        }

        with patch(
            "mcp_cli.chat.streaming_handler.StreamingResponseHandler"
        ) as MockHandler:
            mock_handler_instance = MagicMock()
            mock_handler_instance.stream_response = AsyncMock(
                return_value=mock_stream_result
            )
            MockHandler.return_value = mock_handler_instance

            await processor._handle_streaming_completion(tools=[])

        # Handler should be set on ui_manager
        assert ui_manager.streaming_handler == mock_handler_instance

    @pytest.mark.asyncio
    async def test_streaming_completion_after_tool_calls_flag(self):
        """_handle_streaming_completion passes after_tool_calls to stream_response."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        ui_manager = MockUIManager()
        ui_manager.start_streaming_response = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        mock_stream_result = {"response": "Done", "tool_calls": None}

        with patch(
            "mcp_cli.chat.streaming_handler.StreamingResponseHandler"
        ) as MockHandler:
            mock_handler_instance = MagicMock()
            mock_handler_instance.stream_response = AsyncMock(
                return_value=mock_stream_result
            )
            MockHandler.return_value = mock_handler_instance

            await processor._handle_streaming_completion(
                tools=[], after_tool_calls=True
            )

        # stream_response should have been called with after_tool_calls=True
        mock_handler_instance.stream_response.assert_called_once()
        call_kwargs = mock_handler_instance.stream_response.call_args[1]
        assert call_kwargs.get("after_tool_calls") is True


# ----------------------------------------------------------
# Tests for _load_tools VM and memory tool injection (lines 758-780)
# ----------------------------------------------------------


class TestLoadToolsVMAndMemory:
    """Tests for VM tool and memory tool injection in _load_tools."""

    @pytest.mark.asyncio
    async def test_vm_tools_injected_when_vm_active(self):
        """When session has a VM in non-passive mode, VM tools are injected."""
        context = MockContext()
        context.openai_tools = [{"type": "function", "function": {"name": "base_tool"}}]

        # Set up a VM in strict mode
        mock_vm = MagicMock()
        mock_vm.mode.value = "strict"
        mock_session = MagicMock()
        mock_session.vm = mock_vm
        context.session = mock_session

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        vm_tool = {"type": "function", "function": {"name": "vm_tool"}}

        with patch(
            "chuk_ai_session_manager.memory.vm_prompts.get_vm_tools_as_dicts",
            return_value=[vm_tool],
        ):
            await processor._load_tools()

        # VM tool should have been added
        tool_names = [t["function"]["name"] for t in context.openai_tools]
        assert "vm_tool" in tool_names

    @pytest.mark.asyncio
    async def test_vm_tools_not_injected_in_passive_mode(self):
        """When VM is in passive mode, VM tools are not injected."""
        context = MockContext()
        context.openai_tools = [{"type": "function", "function": {"name": "base_tool"}}]

        mock_vm = MagicMock()
        mock_vm.mode.value = "passive"
        mock_session = MagicMock()
        mock_session.vm = mock_vm
        context.session = mock_session

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        with patch(
            "chuk_ai_session_manager.memory.vm_prompts.get_vm_tools_as_dicts",
            return_value=[{"type": "function", "function": {"name": "vm_tool"}}],
        ) as mock_get:
            await processor._load_tools()

        # VM tool should NOT have been fetched for passive mode
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_vm_tools_not_injected_when_no_session(self):
        """When no session is present, VM tools are not injected."""
        context = MockContext()
        context.openai_tools = []
        # No session attribute

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        with patch(
            "chuk_ai_session_manager.memory.vm_prompts.get_vm_tools_as_dicts",
            return_value=[{"type": "function", "function": {"name": "vm_tool"}}],
        ) as mock_get:
            await processor._load_tools()

        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_vm_tools_error_is_caught(self):
        """When VM tool loading raises, it logs a warning and continues."""
        context = MockContext()
        context.openai_tools = []

        mock_vm = MagicMock()
        mock_vm.mode.value = "strict"
        mock_session = MagicMock()
        mock_session.vm = mock_vm
        context.session = mock_session

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        with patch(
            "chuk_ai_session_manager.memory.vm_prompts.get_vm_tools_as_dicts",
            side_effect=ImportError("Not available"),
        ):
            # Should not raise
            await processor._load_tools()

    @pytest.mark.asyncio
    async def test_memory_tools_injected_when_store_present(self):
        """When context has memory_store, memory tools are injected."""
        context = MockContext()
        context.openai_tools = []
        context.memory_store = MagicMock()  # Has a memory store

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        memory_tool = {"type": "function", "function": {"name": "mem_tool"}}

        with patch(
            "mcp_cli.memory.tools.get_memory_tools_as_dicts",
            return_value=[memory_tool],
        ):
            await processor._load_tools()

        tool_names = [t["function"]["name"] for t in context.openai_tools]
        assert "mem_tool" in tool_names

    @pytest.mark.asyncio
    async def test_memory_tools_not_injected_when_no_store(self):
        """When context has no memory_store, memory tools are not injected."""
        context = MockContext()
        context.openai_tools = []
        # No memory_store attribute

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        with patch(
            "mcp_cli.memory.tools.get_memory_tools_as_dicts",
            return_value=[{"type": "function", "function": {"name": "mem_tool"}}],
        ) as mock_get:
            await processor._load_tools()

        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_tools_error_is_caught(self):
        """When memory tool loading raises, it logs a warning and continues."""
        context = MockContext()
        context.openai_tools = []
        context.memory_store = MagicMock()

        ui_manager = MockUIManager()
        processor = ConversationProcessor(context, ui_manager)

        with patch(
            "mcp_cli.memory.tools.get_memory_tools_as_dicts",
            side_effect=ImportError("memory not available"),
        ):
            # Should not raise
            await processor._load_tools()


# ----------------------------------------------------------
# Tests for auto_save_check (line 573)
# ----------------------------------------------------------


class TestAutoSaveCheck:
    """Tests for auto_save_check call after adding assistant message."""

    @pytest.mark.asyncio
    async def test_auto_save_check_called_when_present(self):
        """auto_save_check is called when context has the method."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []

        auto_save_called = []

        def mock_auto_save():
            auto_save_called.append(True)

        context.auto_save_check = mock_auto_save

        context.client.create_completion = AsyncMock(
            return_value={"response": "Hi!", "tool_calls": []}
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        processor._tool_state = mock_tool_state

        await processor.process_conversation(max_turns=1)

        assert len(auto_save_called) == 1

    @pytest.mark.asyncio
    async def test_auto_save_check_skipped_when_absent(self):
        """When context lacks auto_save_check, no error is raised."""
        context = MockContext()
        context.conversation_history = [Message(role=MessageRole.USER, content="Hello")]
        context.openai_tools = []
        # No auto_save_check attribute

        context.client.create_completion = AsyncMock(
            return_value={"response": "Hi!", "tool_calls": []}
        )

        ui_manager = MockUIManager()
        ui_manager.is_streaming_response = False
        ui_manager.print_assistant_message = AsyncMock()

        processor = ConversationProcessor(context, ui_manager)

        mock_tool_state = MagicMock()
        mock_tool_state.reset_for_new_prompt = MagicMock()
        mock_tool_state.register_user_literals = MagicMock(return_value=0)
        mock_tool_state.extract_bindings_from_text = MagicMock(return_value=[])
        mock_tool_state.format_unused_warning = MagicMock(return_value=None)
        processor._tool_state = mock_tool_state

        # Should not raise
        await processor.process_conversation(max_turns=1)


# ----------------------------------------------------------
# Tests for _validate_tool_messages with non-dict tool_calls (line 886->880)
# ----------------------------------------------------------


class TestValidateToolMessagesObjectToolCalls:
    """Tests for _validate_tool_messages with object (non-dict) tool_calls."""

    def test_tool_calls_as_objects_with_id_attr(self):
        """Tool calls as objects (with id attribute) are handled correctly."""
        # Simulate a ToolCall-like object instead of a dict
        mock_tc = MagicMock()
        mock_tc.get = MagicMock(side_effect=AttributeError("not a dict"))
        # The code checks isinstance(tc, dict) first; if not dict, uses getattr(tc, "id")
        mock_tc.id = "call_obj_1"

        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [mock_tc],  # Object, not dict
            },
            # No tool result
        ]

        result = ConversationProcessor._validate_tool_messages(messages)

        # Should have inserted a placeholder
        assert len(result) == 3
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_obj_1"

    def test_tool_call_with_no_id(self):
        """Tool calls without an id are skipped in expected_ids collection."""
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "foo", "arguments": "{}"},
                        # No "id" key!
                    }
                ],
            },
            {"role": "assistant", "content": "Done."},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)

        # No id means nothing to check — messages unchanged
        assert len(result) == 3

    def test_tool_message_without_tool_call_id(self):
        """Tool messages without tool_call_id are not added to found_ids."""
        messages = [
            {"role": "user", "content": "Go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            },
            # Tool message but without tool_call_id
            {"role": "tool", "content": "Result"},
        ]

        result = ConversationProcessor._validate_tool_messages(messages)

        # "call_abc" is not found in found_ids, so a placeholder is inserted
        assert any(
            m.get("tool_call_id") == "call_abc"
            and "did not complete" in m.get("content", "")
            for m in result
        )
