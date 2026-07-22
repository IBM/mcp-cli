"""Refactored streaming response handler using unified display system.

This is a clean, async-native implementation using:
- Pydantic models for all state (zero dictionaries!)
- StreamingDisplayManager for UI (chuk-term only)
- No fallback display paths
- Type-safe throughout
"""

from __future__ import annotations

import asyncio
import json
import time
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp_cli.dashboard.bridge import DashboardBridge

from pydantic import BaseModel, Field

from mcp_cli.display import StreamingDisplayManager
from mcp_cli.chat.models import ToolCallData
from mcp_cli.config.logging import get_logger
from mcp_cli.config import RuntimeConfig, TimeoutType, load_runtime_config

logger = get_logger("streaming")


class StreamingResponseField(str, Enum):
    """Field names for streaming response serialization."""

    RESPONSE = "response"
    TOOL_CALLS = "tool_calls"
    CHUNKS_RECEIVED = "chunks_received"
    ELAPSED_TIME = "elapsed_time"
    STREAMING = "streaming"
    INTERRUPTED = "interrupted"
    REASONING_CONTENT = "reasoning_content"
    USAGE = "usage"


class StreamingResponse(BaseModel):
    """Container for streaming response data.

    Pydantic model for type-safe streaming responses.
    """

    content: str = Field(description="Response content")
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list, description="Tool calls from the model"
    )
    chunks_received: int = Field(default=0, description="Number of chunks processed")
    elapsed_time: float = Field(description="Time taken for the response")
    interrupted: bool = Field(
        default=False, description="Whether streaming was interrupted"
    )
    reasoning_content: str | None = Field(
        default=None, description="Reasoning content (for DeepSeek reasoner)"
    )
    streaming: bool = Field(
        default=True, description="Whether this was a streaming response"
    )
    usage: dict[str, int] | None = Field(
        default=None,
        description="Token usage from provider (input_tokens, output_tokens)",
    )

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for backwards compatibility using enums."""
        return {
            StreamingResponseField.RESPONSE: self.content,
            StreamingResponseField.TOOL_CALLS: self.tool_calls,
            StreamingResponseField.CHUNKS_RECEIVED: self.chunks_received,
            StreamingResponseField.ELAPSED_TIME: self.elapsed_time,
            StreamingResponseField.STREAMING: self.streaming,
            StreamingResponseField.INTERRUPTED: self.interrupted,
            StreamingResponseField.REASONING_CONTENT: self.reasoning_content,
            StreamingResponseField.USAGE: self.usage,
        }


class ToolCallAccumulator:
    """Manages accumulation of tool calls across streaming chunks.

    Uses Pydantic models for type-safe state management.
    Handles fragmented JSON and validates tool call structure.
    """

    def __init__(self) -> None:
        self._accumulated: list[ToolCallData] = []

    def process_chunk_tool_calls(self, chunk_tool_calls: list[dict[str, Any]]) -> None:
        """Process tool calls from a chunk and accumulate them.

        Args:
            chunk_tool_calls: Tool calls extracted from current chunk
        """
        for tc_dict in chunk_tool_calls:
            # Convert dict to Pydantic model
            tc = ToolCallData.from_dict(tc_dict)

            # Find or create accumulator for this tool call
            existing = self._find_accumulated_call(tc.id, tc.index)

            if existing:
                # Merge chunk data using intelligent JSON merging
                self._merge_tool_call(existing, tc)
            else:
                # New tool call - add to accumulated list
                self._accumulated.append(tc)

    def _find_accumulated_call(self, tc_id: str, tc_index: int) -> ToolCallData | None:
        """Find existing accumulated tool call by ID or index."""
        for tc in self._accumulated:
            if tc.id == tc_id or tc.index == tc_index:
                return tc
        return None

    def _merge_tool_call(self, existing: ToolCallData, new: ToolCallData) -> None:
        """Merge new tool call data into existing accumulator."""
        # Update function name if provided
        if new.function.name:
            existing.function.name = new.function.name

        # Accumulate arguments using intelligent JSON merging
        if new.function.arguments:
            merged_args = self._merge_json_strings(
                existing.function.arguments, new.function.arguments
            )
            existing.function.arguments = merged_args

    def _merge_json_strings(self, current: str, new: str) -> str:
        """Merge two JSON strings intelligently.

        Tries multiple strategies:
        1. Parse both and merge dicts
        2. Concatenate and validate
        3. Fix common issues
        """
        if not current:
            return new
        if not new:
            return current

        # Strategy 1: Both valid JSON objects - merge
        try:
            current_obj = json.loads(current)
            new_obj = json.loads(new)
            if isinstance(current_obj, dict) and isinstance(new_obj, dict):
                current_obj.update(new_obj)
                return json.dumps(current_obj)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Concatenate
        combined = current + new

        # Try validating
        try:
            json.loads(combined)
            return combined
        except json.JSONDecodeError:
            pass

        # Strategy 3: Fix common issues
        # Try adding missing braces
        if not combined.startswith("{"):
            combined = "{" + combined
        if not combined.endswith("}"):
            combined = combined + "}"

        # Fix duplicated braces: }{  ->  },{
        combined = combined.replace("}{", "},{")

        try:
            json.loads(combined)
            return combined
        except json.JSONDecodeError:
            # Give up, return concatenated
            logger.warning(f"Could not merge JSON: {current[:50]}... + {new[:50]}...")
            return current + new

    def finalize(self) -> list[dict[str, Any]]:
        """Finalize and validate all accumulated tool calls.

        Returns only valid, complete tool calls as dicts for API compatibility.
        """
        finalized = []

        for tc in self._accumulated:
            # Must have name
            if not tc.function.name:
                logger.debug(f"Skipping tool call with no name: {tc.id}")
                continue

            # Validate arguments JSON
            args = tc.function.arguments
            if not args or args.strip() == "{}":
                args = "{}"
            else:
                try:
                    json.loads(args)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON in tool call arguments, skipping: {args[:100]}"
                    )
                    continue

            # Convert Pydantic model to dict for API
            cleaned = tc.to_dict()
            finalized.append(cleaned)

            logger.debug(f"Finalized tool call: {tc.function.name}")

        return finalized


class StreamingResponseHandler:
    """Clean, async-native streaming handler.

    Uses unified display system - no fallbacks, no dual paths.
    """

    def __init__(
        self,
        display: StreamingDisplayManager,
        runtime_config: RuntimeConfig | None = None,
        dashboard_bridge: "DashboardBridge | None" = None,
    ):
        """Initialize handler.

        Args:
            display: The unified display manager (required, no fallback)
            runtime_config: Runtime configuration (optional, will load defaults if not provided)
            dashboard_bridge: Optional DashboardBridge for live token streaming to browser.
        """
        self.display = display
        self.tool_accumulator = ToolCallAccumulator()
        self._interrupted = False
        self._usage: dict[str, int] | None = None
        self.runtime_config = runtime_config or load_runtime_config()
        self._dashboard_bridge = dashboard_bridge

    async def stream_response(
        self,
        client,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        after_tool_calls: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Stream response from LLM client.

        Args:
            client: LLM client with streaming support
            messages: Conversation messages
            tools: Available tools for function calling
            after_tool_calls: True when this is a follow-up after tool execution
                (thinking models need extra time to process tool results)
            **kwargs: Additional arguments for client

        Returns:
            Response dictionary (for backwards compatibility)
        """
        # Reset state
        self._interrupted = False
        self._usage = None
        self.tool_accumulator = ToolCallAccumulator()

        # Start display
        await self.display.start_streaming()
        start_time = time.time()

        try:
            # Check client capabilities
            if not hasattr(client, "create_completion"):
                logger.warning("Client doesn't support create_completion")
                return await self._handle_non_streaming(
                    client, messages, tools, **kwargs
                )

            # Stream with chunk timeout protection
            await self._stream_with_timeout(
                client, messages, tools, after_tool_calls=after_tool_calls, **kwargs
            )

            # Finalize tool calls
            tool_calls = self.tool_accumulator.finalize()

            # Capture state values BEFORE stop_streaming clears them
            chunks_received = (
                self.display.streaming_state.chunks_received
                if self.display.streaming_state
                else 0
            )
            reasoning_content = (
                self.display.streaming_state.reasoning_content
                if self.display.streaming_state
                else None
            )

            # Stop display and get final content (this clears streaming_state!)
            final_content = await self.display.stop_streaming(
                interrupted=self._interrupted
            )

            # Build response using captured values
            elapsed = time.time() - start_time
            response = StreamingResponse(
                content=final_content,
                tool_calls=tool_calls,
                chunks_received=chunks_received,
                elapsed_time=elapsed,
                interrupted=self._interrupted,
                reasoning_content=reasoning_content,
                usage=self._usage,
            )

            logger.info(
                f"Streaming complete: {len(final_content)} chars, "
                f"{len(tool_calls)} tools, {elapsed:.2f}s"
            )

            # Signal stream end to dashboard
            if self._dashboard_bridge is not None:
                try:
                    await self._dashboard_bridge.on_token("", done=True)
                except Exception as _e:
                    logger.debug("Dashboard on_token(done) error: %s", _e)

            return response.to_dict()

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            await self.display.stop_streaming(interrupted=True)
            raise

    async def _stream_with_timeout(
        self,
        client,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        after_tool_calls: bool = False,
        **kwargs,
    ) -> None:
        """Stream with per-chunk timeout protection.

        Timeouts are configurable via (in priority order):
        1. CLI arguments (--tool-timeout)
        2. Environment variables (MCP_STREAMING_CHUNK_TIMEOUT, etc.)
        3. Config file (server_config.json -> timeouts.streamingChunkTimeout)
        4. Defaults (45s chunk, 60s first chunk, 180s first chunk after tools, 300s global)

        The first chunk uses a longer timeout because the model needs time to
        begin generating. After tool calls, thinking models (e.g. kimi-k2.5,
        DeepSeek-R1) may need significantly more time to process tool results
        before emitting their first token.
        """
        from mcp_cli.config.defaults import (
            DEFAULT_STREAMING_FIRST_CHUNK_AFTER_TOOLS_TIMEOUT,
        )

        # Get timeouts from runtime config (type-safe with enums!)
        chunk_timeout = self.runtime_config.get_timeout(TimeoutType.STREAMING_CHUNK)
        first_chunk_timeout = self.runtime_config.get_timeout(
            TimeoutType.STREAMING_FIRST_CHUNK
        )
        global_timeout = self.runtime_config.get_timeout(TimeoutType.STREAMING_GLOBAL)

        # After tool calls, thinking models need extended time for the first chunk
        if after_tool_calls:
            first_chunk_timeout = max(
                first_chunk_timeout,
                DEFAULT_STREAMING_FIRST_CHUNK_AFTER_TOOLS_TIMEOUT,
            )

        logger.debug(
            f"Streaming timeouts: chunk={chunk_timeout}s, "
            f"first_chunk={first_chunk_timeout}s "
            f"(after_tools={after_tool_calls}), "
            f"global={global_timeout}s"
        )

        async def stream_chunks():
            """Inner streaming function."""
            stream = client.create_completion(
                messages=messages,
                tools=tools,
                stream=True,
                **kwargs,
            )

            stream_iter = stream.__aiter__()
            first_chunk_received = False

            while True:
                try:
                    # Use longer timeout for first chunk
                    effective_timeout = (
                        chunk_timeout if first_chunk_received else first_chunk_timeout
                    )

                    # Wait for next chunk with timeout
                    chunk = await asyncio.wait_for(
                        stream_iter.__anext__(),
                        timeout=effective_timeout,
                    )
                    first_chunk_received = True

                    # Check for interrupt
                    if self._interrupted:
                        logger.debug("Stream interrupted by user")
                        try:
                            await stream_iter.aclose()
                        except Exception as e:
                            logger.debug(f"Error closing stream: {e}")
                        break

                    # Process chunk
                    await self._process_chunk(chunk)

                    # Small yield for smooth UI
                    await asyncio.sleep(0.0005)

                except StopAsyncIteration:
                    logger.debug("Stream completed normally")
                    break
                except asyncio.TimeoutError:
                    effective_timeout = (
                        chunk_timeout if first_chunk_received else first_chunk_timeout
                    )
                    logger.warning(
                        f"Chunk timeout after {effective_timeout}s "
                        f"(first_chunk={not first_chunk_received}, "
                        f"after_tools={after_tool_calls})"
                    )
                    if not first_chunk_received and after_tool_calls:
                        logger.warning(
                            "Streaming timeout after %.0fs waiting for first response after tool calls. "
                            "Increase with MCP_STREAMING_FIRST_CHUNK_TIMEOUT=%.0f",
                            effective_timeout,
                            effective_timeout * 2,
                        )
                    else:
                        logger.warning(
                            "Streaming timeout after %.0fs waiting for response. "
                            "Increase with --tool-timeout %.0f",
                            effective_timeout,
                            effective_timeout * 2,
                        )
                    break

        # Run with global timeout
        try:
            await asyncio.wait_for(stream_chunks(), timeout=global_timeout)
        except asyncio.TimeoutError:
            logger.error(
                "Global streaming timeout after %.0fs. "
                "Increase with --tool-timeout %.0f or MCP_STREAMING_GLOBAL_TIMEOUT=%.0f",
                global_timeout,
                global_timeout * 2,
                global_timeout * 2,
            )
            self._interrupted = True

    async def _process_chunk(self, raw_chunk: dict[str, Any]) -> None:
        """Process a single streaming chunk.

        Args:
            raw_chunk: Raw chunk from LLM provider
        """
        # Use display to process chunk (normalizes format)
        await self.display.add_chunk(raw_chunk)

        # Broadcast token to dashboard if connected
        if self._dashboard_bridge is not None:
            token: str | None = None
            if "response" in raw_chunk:
                token = raw_chunk["response"]
            elif "content" in raw_chunk:
                token = raw_chunk["content"]
            elif "text" in raw_chunk:
                token = raw_chunk["text"]
            elif "delta" in raw_chunk and isinstance(raw_chunk["delta"], dict):
                token = raw_chunk["delta"].get("content")
            elif "choices" in raw_chunk and raw_chunk["choices"]:
                delta = raw_chunk["choices"][0].get("delta", {})
                if isinstance(delta, dict):
                    token = delta.get("content")
            if token:
                try:
                    await self._dashboard_bridge.on_token(token)
                except Exception as _e:
                    logger.debug("Dashboard on_token error: %s", _e)

        # Extract tool calls if present
        if "tool_calls" in raw_chunk and raw_chunk["tool_calls"]:
            self.tool_accumulator.process_chunk_tool_calls(raw_chunk["tool_calls"])

        # Capture usage data (providers send it with the final chunk)
        if "usage" in raw_chunk and raw_chunk["usage"]:
            self._usage = raw_chunk["usage"]

    async def _handle_non_streaming(
        self,
        client,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        **kwargs,
    ) -> dict[str, Any]:
        """Fallback for non-streaming clients."""
        start_time = time.time()

        logger.debug("Non-streaming fallback: generating response...")
        if hasattr(client, "complete"):
            result = await client.complete(messages=messages, tools=tools, **kwargs)
        else:
            raise RuntimeError("Client has no streaming or completion method")

        elapsed = time.time() - start_time

        return {
            "response": result.get("response", ""),
            "tool_calls": result.get("tool_calls", []),
            "chunks_received": 1,
            "elapsed_time": elapsed,
            "streaming": False,
            "interrupted": False,
        }

    def interrupt_streaming(self) -> None:
        """Interrupt current streaming operation."""
        self._interrupted = True
        logger.debug("Streaming interrupted by user")
