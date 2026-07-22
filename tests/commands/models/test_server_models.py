"""Tests for server command models."""

import pytest
from pydantic import ValidationError

from mcp_cli.commands.models.server import (
    ServerActionParams,
    ServerCapabilities,
    ServerPerformanceInfo,
    ServerStatusInfo,
)


class TestServerActionParams:
    """Test ServerActionParams model."""

    def test_default_params(self):
        """Test default parameter values."""
        params = ServerActionParams()

        assert params.args == []
        assert not params.detailed
        assert not params.show_capabilities
        assert not params.show_transport
        assert params.output_format == "table"
        assert not params.ping_servers

    def test_custom_params(self):
        """Test custom parameter values."""
        params = ServerActionParams(
            args=["list", "servers"],
            detailed=True,
            show_capabilities=True,
            show_transport=True,
            output_format="json",
            ping_servers=True,
        )

        assert params.args == ["list", "servers"]
        assert params.detailed
        assert params.show_capabilities
        assert params.show_transport
        assert params.output_format == "json"
        assert params.ping_servers

    def test_invalid_output_format(self):
        """Test validation of invalid output format."""
        with pytest.raises(ValidationError) as exc_info:
            ServerActionParams(output_format="invalid")

        error = exc_info.value.errors()[0]
        assert "output_format" in error["loc"]

    def test_valid_output_formats(self):
        """Test valid output formats."""
        for fmt in ["table", "json"]:
            params = ServerActionParams(output_format=fmt)
            assert params.output_format == fmt


class TestServerStatusInfo:
    """Test ServerStatusInfo model."""

    def test_creation(self):
        """Test creating status info."""
        status = ServerStatusInfo(
            icon="✅", status="Connected", reason="Server is online"
        )

        assert status.icon == "✅"
        assert status.status == "Connected"
        assert status.reason == "Server is online"

    def test_required_fields(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError):
            ServerStatusInfo()

    def test_min_length_validation(self):
        """Test minimum length validation."""
        with pytest.raises(ValidationError) as exc_info:
            ServerStatusInfo(icon="", status="Connected", reason="Online")

        # Should fail on empty icon
        error = exc_info.value.errors()[0]
        assert "icon" in error["loc"]


class TestServerPerformanceInfo:
    """Test ServerPerformanceInfo model."""

    def test_creation(self):
        """Test creating performance info."""
        perf = ServerPerformanceInfo(icon="🚀", latency="25.5ms", ping_ms=25.5)

        assert perf.icon == "🚀"
        assert perf.latency == "25.5ms"
        assert perf.ping_ms == 25.5

    def test_optional_ping_ms(self):
        """Test that ping_ms is optional."""
        perf = ServerPerformanceInfo(icon="🚀", latency="N/A")

        assert perf.icon == "🚀"
        assert perf.latency == "N/A"
        assert perf.ping_ms is None

    def test_negative_ping_validation(self):
        """Test validation of negative ping time."""
        with pytest.raises(ValidationError) as exc_info:
            ServerPerformanceInfo(icon="🚀", latency="25.5ms", ping_ms=-10.0)

        error = exc_info.value.errors()[0]
        assert "ping_ms" in error["loc"]

    def test_zero_ping(self):
        """Test that zero ping is valid."""
        perf = ServerPerformanceInfo(icon="🚀", latency="0ms", ping_ms=0.0)

        assert perf.ping_ms == 0.0

    def test_required_fields(self):
        """Test that icon and latency are required."""
        with pytest.raises(ValidationError):
            ServerPerformanceInfo(ping_ms=25.5)


class TestServerCapabilities:
    """Test ServerCapabilities model."""

    def test_default_capabilities(self):
        """Test default values."""
        caps = ServerCapabilities()

        assert caps.tools is False
        assert caps.prompts is False
        assert caps.resources is False
        assert caps.experimental == {}

    def test_custom_capabilities(self):
        """Test with custom values."""
        caps = ServerCapabilities(
            tools=True,
            prompts=True,
            resources=True,
            experimental={"events": True, "streaming": True},
        )

        assert caps.tools is True
        assert caps.prompts is True
        assert caps.resources is True
        assert caps.experimental["events"] is True

    def test_has_events_property(self):
        """Test has_events property."""
        caps_no_events = ServerCapabilities()
        assert caps_no_events.has_events is False

        caps_with_events = ServerCapabilities(experimental={"events": True})
        assert caps_with_events.has_events is True

        caps_events_false = ServerCapabilities(experimental={"events": False})
        assert caps_events_false.has_events is False

    def test_has_streaming_property(self):
        """Test has_streaming property."""
        caps_no_streaming = ServerCapabilities()
        assert caps_no_streaming.has_streaming is False

        caps_with_streaming = ServerCapabilities(experimental={"streaming": True})
        assert caps_with_streaming.has_streaming is True

    def test_to_display_string_empty(self):
        """Test to_display_string with no capabilities."""
        caps = ServerCapabilities()
        assert caps.to_display_string() == "None"

    def test_to_display_string_basic(self):
        """Test to_display_string with basic capabilities."""
        caps = ServerCapabilities(tools=True)
        assert caps.to_display_string() == "Tools"

        caps2 = ServerCapabilities(tools=True, prompts=True)
        assert caps2.to_display_string() == "Tools, Prompts"

        caps3 = ServerCapabilities(tools=True, prompts=True, resources=True)
        assert caps3.to_display_string() == "Tools, Prompts, Resources"

    def test_to_display_string_with_experimental(self):
        """Test to_display_string with experimental capabilities."""
        caps = ServerCapabilities(
            tools=True, experimental={"events": True, "streaming": True}
        )
        display = caps.to_display_string()
        assert "Tools" in display
        assert "Events*" in display
        assert "Streaming*" in display


class TestServerActionParamsValidatorBranch:
    """Cover the validator raise path (line 46)."""

    def test_output_format_validator_rejects_xml(self):
        with pytest.raises(ValidationError):
            ServerActionParams(output_format="xml")


class TestServerPerformanceInfoValidatorBranch:
    """Cover the validator raise path (line 90)."""

    def test_ping_ms_validator_rejects_negative(self):
        with pytest.raises(ValidationError):
            ServerPerformanceInfo(icon="X", latency="bad", ping_ms=-1.0)
