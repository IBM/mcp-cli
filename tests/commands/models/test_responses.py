# tests/commands/models/test_responses.py
"""Tests for commands/models/responses.py."""

import pytest
from pydantic import ValidationError

from mcp_cli.commands.models.responses import (
    PromptInfoResponse,
    ResourceInfoResponse,
    ServerInfoResponse,
    ToolInfoResponse,
)


class TestServerInfoResponse:
    """Test ServerInfoResponse model."""

    def test_creation_basic(self):
        """Test creating a ServerInfoResponse with required fields."""
        server = ServerInfoResponse(
            name="my-server",
            transport="stdio",
            status="connected",
        )

        assert server.name == "my-server"
        assert server.transport == "stdio"
        assert server.status == "connected"
        assert server.capabilities == {}
        assert server.tool_count == 0
        assert server.ping_ms is None

    def test_creation_full(self):
        """Test creating a ServerInfoResponse with all fields."""
        server = ServerInfoResponse(
            name="my-server",
            transport="http",
            capabilities={"tools": True, "prompts": False},
            tool_count=5,
            status="connected",
            ping_ms=25.5,
        )

        assert server.name == "my-server"
        assert server.transport == "http"
        assert server.capabilities == {"tools": True, "prompts": False}
        assert server.tool_count == 5
        assert server.status == "connected"
        assert server.ping_ms == 25.5

    def test_transport_stdio(self):
        """Test that 'stdio' transport is accepted."""
        server = ServerInfoResponse(name="s", transport="stdio", status="ok")
        assert server.transport == "stdio"

    def test_transport_http(self):
        """Test that 'http' transport is accepted."""
        server = ServerInfoResponse(name="s", transport="http", status="ok")
        assert server.transport == "http"

    def test_transport_sse(self):
        """Test that 'sse' transport is accepted."""
        server = ServerInfoResponse(name="s", transport="sse", status="ok")
        assert server.transport == "sse"

    def test_invalid_transport_raises(self):
        """Test that an invalid transport raises ValidationError (covers lines 42-44)."""
        with pytest.raises(ValidationError) as exc_info:
            ServerInfoResponse(name="s", transport="websocket", status="ok")

        errors = exc_info.value.errors()
        assert len(errors) >= 1
        # The field_validator raises ValueError with a descriptive message
        error_messages = " ".join(str(e) for e in errors)
        assert "transport" in error_messages.lower() or any(
            "transport" in str(e.get("loc", "")).lower() for e in errors
        )

    def test_invalid_transport_grpc(self):
        """Test that 'grpc' transport raises ValidationError (covers branch in lines 42-44)."""
        with pytest.raises(ValidationError):
            ServerInfoResponse(name="s", transport="grpc", status="ok")

    def test_invalid_transport_empty_string(self):
        """Test that an empty transport string raises ValidationError."""
        with pytest.raises(ValidationError):
            ServerInfoResponse(name="s", transport="", status="ok")

    def test_name_min_length(self):
        """Test that empty name raises ValidationError."""
        with pytest.raises(ValidationError):
            ServerInfoResponse(name="", transport="stdio", status="ok")

    def test_status_min_length(self):
        """Test that empty status raises ValidationError."""
        with pytest.raises(ValidationError):
            ServerInfoResponse(name="s", transport="stdio", status="")

    def test_tool_count_non_negative(self):
        """Test that negative tool_count raises ValidationError."""
        with pytest.raises(ValidationError):
            ServerInfoResponse(name="s", transport="stdio", status="ok", tool_count=-1)

    def test_ping_ms_non_negative(self):
        """Test that negative ping_ms raises ValidationError."""
        with pytest.raises(ValidationError):
            ServerInfoResponse(name="s", transport="stdio", status="ok", ping_ms=-1.0)

    def test_ping_ms_zero_valid(self):
        """Test that zero ping_ms is valid."""
        server = ServerInfoResponse(
            name="s", transport="stdio", status="ok", ping_ms=0.0
        )
        assert server.ping_ms == 0.0

    def test_validate_transport_raises_directly(self):
        """Test validate_transport raises ValueError when called directly with invalid value.

        The field pattern constraint prevents invalid values reaching the validator
        via normal instantiation, so we call the classmethod directly to cover
        lines 42-43 (the raise branch).
        """
        with pytest.raises(ValueError, match="Invalid transport"):
            ServerInfoResponse.validate_transport("websocket")

    def test_validate_transport_returns_valid_directly(self):
        """Test validate_transport returns value when called directly with valid value."""
        assert ServerInfoResponse.validate_transport("stdio") == "stdio"
        assert ServerInfoResponse.validate_transport("http") == "http"
        assert ServerInfoResponse.validate_transport("sse") == "sse"


class TestResourceInfoResponse:
    """Test ResourceInfoResponse model."""

    def test_creation_minimal(self):
        """Test creating with only required fields."""
        resource = ResourceInfoResponse(
            uri="file:///path/to/file.txt",
            server="my-server",
        )

        assert resource.uri == "file:///path/to/file.txt"
        assert resource.server == "my-server"
        assert resource.name is None
        assert resource.description is None
        assert resource.mime_type is None

    def test_creation_full(self):
        """Test creating with all fields."""
        resource = ResourceInfoResponse(
            uri="file:///path/to/file.txt",
            name="file.txt",
            description="A text file",
            mime_type="text/plain",
            server="my-server",
        )

        assert resource.uri == "file:///path/to/file.txt"
        assert resource.name == "file.txt"
        assert resource.description == "A text file"
        assert resource.mime_type == "text/plain"
        assert resource.server == "my-server"

    def test_uri_min_length(self):
        """Test that empty uri raises ValidationError."""
        with pytest.raises(ValidationError):
            ResourceInfoResponse(uri="", server="my-server")

    def test_server_min_length(self):
        """Test that empty server raises ValidationError."""
        with pytest.raises(ValidationError):
            ResourceInfoResponse(uri="file:///x", server="")


class TestPromptInfoResponse:
    """Test PromptInfoResponse model."""

    def test_creation_minimal(self):
        """Test creating with only required fields."""
        prompt = PromptInfoResponse(
            name="generate-code",
            server="my-server",
        )

        assert prompt.name == "generate-code"
        assert prompt.server == "my-server"
        assert prompt.description is None
        assert prompt.arguments == []

    def test_creation_full(self):
        """Test creating with all fields."""
        prompt = PromptInfoResponse(
            name="generate-code",
            description="Generate code from description",
            arguments=[{"name": "language", "required": True}],
            server="my-server",
        )

        assert prompt.name == "generate-code"
        assert prompt.description == "Generate code from description"
        assert len(prompt.arguments) == 1
        assert prompt.arguments[0]["name"] == "language"
        assert prompt.server == "my-server"

    def test_name_min_length(self):
        """Test that empty name raises ValidationError."""
        with pytest.raises(ValidationError):
            PromptInfoResponse(name="", server="my-server")

    def test_server_min_length(self):
        """Test that empty server raises ValidationError."""
        with pytest.raises(ValidationError):
            PromptInfoResponse(name="prompt", server="")


class TestToolInfoResponse:
    """Test ToolInfoResponse model."""

    def test_creation_minimal(self):
        """Test creating with only required fields."""
        tool = ToolInfoResponse(
            name="search",
            namespace="my-server",
        )

        assert tool.name == "search"
        assert tool.namespace == "my-server"
        assert tool.description is None
        assert tool.parameters == {}

    def test_creation_full(self):
        """Test creating with all fields."""
        tool = ToolInfoResponse(
            name="search",
            namespace="my-server",
            description="Search for files",
            parameters={"query": {"type": "string"}},
        )

        assert tool.name == "search"
        assert tool.namespace == "my-server"
        assert tool.description == "Search for files"
        assert tool.parameters == {"query": {"type": "string"}}

    def test_name_min_length(self):
        """Test that empty name raises ValidationError."""
        with pytest.raises(ValidationError):
            ToolInfoResponse(name="", namespace="my-server")

    def test_namespace_min_length(self):
        """Test that empty namespace raises ValidationError."""
        with pytest.raises(ValidationError):
            ToolInfoResponse(name="tool", namespace="")
