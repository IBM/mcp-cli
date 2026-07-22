# tests/planning/test_agentic_loop_integration.py
"""Integration test proving the agentic loop works end-to-end.

Simulates a real plan execution where:
1. Geocode returns empty results → LLM retries with simpler name
2. Weather tool gets wrong types → LLM corrects and retries
3. Both steps succeed through the agentic loop
"""

from __future__ import annotations

import json
from typing import Any
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_cli.planning.executor import PlanRunner, ModelManagerProtocol
from mcp_cli.planning.context import PlanningContext


# ── Fake MCP Tool Backend ──────────────────────────────────────────────────


@dataclass
class FakeToolCallResult:
    tool_name: str
    success: bool = True
    result: Any = None
    error: str | None = None


class SimulatedToolManager:
    """Simulates real MCP tool behavior including validation errors."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def get_all_tools(self):
        return []

    async def get_adapted_tools_for_llm(self, provider: str = "openai"):
        """Return tool catalog matching real MCP servers."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "geocode_location",
                    "description": "Geocode a location name to coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Location name to geocode",
                            }
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather_forecast",
                    "description": "Get weather forecast for coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "latitude": {
                                "type": "number",
                                "description": "Latitude",
                            },
                            "longitude": {
                                "type": "number",
                                "description": "Longitude",
                            },
                        },
                        "required": ["latitude", "longitude"],
                    },
                },
            },
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        namespace: str | None = None,
        timeout: float | None = None,
    ) -> FakeToolCallResult:
        self.calls.append((tool_name, arguments))

        if tool_name == "geocode_location":
            name = arguments.get("name", "")
            # "Leavenheath, Suffolk" returns null (too specific)
            if "suffolk" in name.lower() or "," in name:
                return FakeToolCallResult(
                    tool_name=tool_name,
                    result={"results": None, "generationtime_ms": 0.9},
                )
            # Simpler "Leavenheath" works
            return FakeToolCallResult(
                tool_name=tool_name,
                result={
                    "results": [
                        {
                            "name": "Leavenheath",
                            "latitude": 52.0,
                            "longitude": 0.85,
                            "timezone": "Europe/London",
                        }
                    ]
                },
            )

        if tool_name == "get_weather_forecast":
            lat = arguments.get("latitude")
            lon = arguments.get("longitude")
            # Simulate MCP server type validation
            if isinstance(lat, str) or isinstance(lon, str):
                return FakeToolCallResult(
                    tool_name=tool_name,
                    success=False,
                    error=(
                        "ParameterValidationError: Invalid parameter "
                        "'latitude': expected number, got str"
                    ),
                )
            return FakeToolCallResult(
                tool_name=tool_name,
                result={
                    "current_weather": {
                        "temperature": 12.3,
                        "windspeed": 15.2,
                        "weathercode": 3,
                    }
                },
            )

        return FakeToolCallResult(
            tool_name=tool_name,
            success=False,
            error=f"Unknown tool: {tool_name}",
        )


# ── LLM Response Builders ──────────────────────────────────────────────────


def _tool_call_response(name: str, args: dict) -> dict:
    """Build a mock LLM response containing a tool call.

    Uses the chuk_llm native format (top-level tool_calls key),
    which is what the real client returns with stream=False.
    """
    return {
        "response": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
                },
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 15},
    }


def _text_response(text: str = "Step complete.") -> dict:
    """Build a mock LLM response with text only (step done).

    Uses the chuk_llm native format (top-level response key).
    """
    return {
        "response": text,
        "tool_calls": None,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


# ── Integration Tests ──────────────────────────────────────────────────────


class TestAgenticLoopIntegration:
    """Prove the agentic loop handles real-world scenarios."""

    @pytest.mark.asyncio
    async def test_geocode_retry_on_empty_result(self, tmp_path):
        """Geocode returns null results → LLM retries with simpler name."""
        tm = SimulatedToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")

        client = AsyncMock()
        client.create_completion = AsyncMock(
            side_effect=[
                # Turn 1: LLM tries "Leavenheath, Suffolk" (returns null)
                _tool_call_response(
                    "geocode_location", {"name": "Leavenheath, Suffolk"}
                ),
                # Turn 2: LLM sees null results, tries simpler "Leavenheath"
                _tool_call_response("geocode_location", {"name": "Leavenheath"}),
                # Turn 3: LLM sees valid results, signals done
                _text_response("Geocoded: lat=52.0, lon=0.85"),
            ]
        )

        mm = MagicMock(spec=ModelManagerProtocol)
        mm.get_client.return_value = client

        runner = PlanRunner(context, model_manager=mm, enable_guards=False)

        plan = {
            "id": "geo-test",
            "title": "Geocode Test",
            "steps": [
                {
                    "index": "1",
                    "title": "Geocode Leavenheath",
                    "tool": "geocode_location",
                    "args": {"name": "Leavenheath, Suffolk"},
                    "result_variable": "geocode_result",
                },
            ],
        }

        result = await runner.execute_plan(plan, checkpoint=False)

        assert result.success, f"Expected success, got error: {result.error}"
        assert result.steps[0].success

        # Should have the good result (second call)
        geo = result.variables["geocode_result"]
        assert geo["results"] is not None
        assert geo["results"][0]["latitude"] == 52.0

        # Verify two geocode calls were made
        geocode_calls = [c for c in tm.calls if c[0] == "geocode_location"]
        assert len(geocode_calls) == 2
        assert geocode_calls[0][1]["name"] == "Leavenheath, Suffolk"
        assert geocode_calls[1][1]["name"] == "Leavenheath"

        # Verify LLM saw 3 turns (2 tool calls + 1 text response)
        assert client.create_completion.call_count == 3

    @pytest.mark.asyncio
    async def test_weather_retry_on_type_error(self, tmp_path):
        """Weather tool rejects string types → LLM retries with numbers."""
        tm = SimulatedToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")

        client = AsyncMock()
        client.create_completion = AsyncMock(
            side_effect=[
                # Turn 1: LLM passes lat/lon as strings (error)
                _tool_call_response(
                    "get_weather_forecast",
                    {"latitude": "52.0", "longitude": "0.85"},
                ),
                # Turn 2: LLM sees error, retries with numbers
                _tool_call_response(
                    "get_weather_forecast",
                    {"latitude": 52.0, "longitude": 0.85},
                ),
                # Turn 3: LLM sees weather data, signals done
                _text_response("Current temp: 12.3°C"),
            ]
        )

        mm = MagicMock(spec=ModelManagerProtocol)
        mm.get_client.return_value = client

        runner = PlanRunner(context, model_manager=mm, enable_guards=False)

        plan = {
            "id": "weather-test",
            "title": "Weather Test",
            "steps": [
                {
                    "index": "1",
                    "title": "Get weather",
                    "tool": "get_weather_forecast",
                    "args": {"latitude": 52.0, "longitude": 0.85},
                    "result_variable": "weather",
                },
            ],
        }

        result = await runner.execute_plan(plan, checkpoint=False)

        assert result.success, f"Expected success, got error: {result.error}"
        weather = result.variables["weather"]
        assert weather["current_weather"]["temperature"] == 12.3

        # First call with strings failed, second with numbers succeeded
        weather_calls = [c for c in tm.calls if c[0] == "get_weather_forecast"]
        assert len(weather_calls) == 2
        assert isinstance(weather_calls[0][1]["latitude"], str)  # First: string
        assert isinstance(weather_calls[1][1]["latitude"], float)  # Second: number

    @pytest.mark.asyncio
    async def test_full_two_step_plan_with_agentic_retries(self, tmp_path):
        """Full plan: geocode (retry empty) → weather (retry types)."""
        tm = SimulatedToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")

        client = AsyncMock()
        client.create_completion = AsyncMock(
            side_effect=[
                # Step 1, Turn 1: geocode with full name (null)
                _tool_call_response(
                    "geocode_location", {"name": "Leavenheath, Suffolk"}
                ),
                # Step 1, Turn 2: geocode with simpler name (success)
                _tool_call_response("geocode_location", {"name": "Leavenheath"}),
                # Step 1, Turn 3: LLM signals done
                _text_response("Found: lat=52.0, lon=0.85"),
                # Step 2, Turn 1: weather with numbers (success first try)
                _tool_call_response(
                    "get_weather_forecast",
                    {"latitude": 52.0, "longitude": 0.85},
                ),
                # Step 2, Turn 2: LLM sees result, signals done
                _text_response("Weather: 12.3°C"),
            ]
        )

        mm = MagicMock(spec=ModelManagerProtocol)
        mm.get_client.return_value = client

        step_starts = []
        step_completions = []
        tool_starts = []
        tool_completions = []

        runner = PlanRunner(
            context,
            model_manager=mm,
            on_step_start=lambda i, t, tn: step_starts.append((i, t, tn)),
            on_step_complete=lambda sr: step_completions.append(
                (sr.step_index, sr.success)
            ),
            on_tool_start=lambda name, args: tool_starts.append((name, args)),
            on_tool_complete=lambda name, result, ok, elapsed: tool_completions.append(
                (name, ok)
            ),
            enable_guards=False,
        )

        plan = {
            "id": "full-test",
            "title": "Weather for Leavenheath",
            "steps": [
                {
                    "index": "1",
                    "title": "Geocode Leavenheath",
                    "tool": "geocode_location",
                    "args": {"name": "Leavenheath, Suffolk"},
                    "depends_on": [],
                    "result_variable": "geocode_result",
                },
                {
                    "index": "2",
                    "title": "Get current weather",
                    "tool": "get_weather_forecast",
                    "args": {
                        "latitude": "${geocode_result.results.0.latitude}",
                        "longitude": "${geocode_result.results.0.longitude}",
                    },
                    "depends_on": ["1"],
                    "result_variable": "weather",
                },
            ],
        }

        result = await runner.execute_plan(plan, checkpoint=False)

        # Both steps should succeed
        assert result.success, f"Plan failed: {result.error}"
        assert len(result.steps) == 2
        assert result.steps[0].success
        assert result.steps[1].success

        # Variables should contain the results
        assert result.variables["geocode_result"]["results"][0]["latitude"] == 52.0
        assert result.variables["weather"]["current_weather"]["temperature"] == 12.3

        # Callbacks should have fired
        assert len(step_starts) == 2
        assert len(step_completions) == 2
        assert step_completions[0] == ("1", True)
        assert step_completions[1] == ("2", True)

        # Total tool calls: 2 geocode + 1 weather = 3
        assert len(tm.calls) == 3

        # Tool-level callbacks should have fired for each tool call
        assert len(tool_starts) == 3
        assert tool_starts[0][0] == "geocode_location"
        assert tool_starts[1][0] == "geocode_location"
        assert tool_starts[2][0] == "get_weather_forecast"

        assert len(tool_completions) == 3
        assert tool_completions[0] == ("geocode_location", True)
        assert tool_completions[1] == ("geocode_location", True)
        assert tool_completions[2] == ("get_weather_forecast", True)

    @pytest.mark.asyncio
    async def test_static_fallback_without_model_manager(self, tmp_path):
        """Without model_manager, static args are used (no agentic loop)."""
        tm = SimulatedToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")

        runner = PlanRunner(context, enable_guards=False)  # No model_manager

        plan = {
            "id": "static-test",
            "title": "Static Test",
            "steps": [
                {
                    "index": "1",
                    "title": "Geocode",
                    "tool": "geocode_location",
                    "args": {"name": "Leavenheath"},
                    "result_variable": "geo",
                },
            ],
        }

        result = await runner.execute_plan(plan, checkpoint=False)

        # Should succeed (Leavenheath without comma works)
        assert result.success
        assert result.variables["geo"]["results"][0]["latitude"] == 52.0

        # Only one tool call (no retry without LLM)
        assert len(tm.calls) == 1
