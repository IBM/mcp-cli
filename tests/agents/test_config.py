# tests/agents/test_config.py
"""Unit tests for AgentConfig."""

from __future__ import annotations

import pytest

from mcp_cli.agents.config import AgentConfig


class TestAgentConfigDefaults:
    def test_minimal(self):
        cfg = AgentConfig(agent_id="a1")
        assert cfg.agent_id == "a1"
        assert cfg.name == ""
        assert cfg.role == ""
        assert cfg.model is None
        assert cfg.provider is None
        assert cfg.system_prompt is None
        assert cfg.allowed_tools is None
        assert cfg.denied_tools is None
        assert cfg.allowed_servers is None
        assert cfg.tool_timeout_override is None
        assert cfg.auto_approve_tools is None
        assert cfg.parent_agent_id is None
        assert cfg.initial_prompt == ""

    def test_full(self):
        cfg = AgentConfig(
            agent_id="agent-research",
            name="Researcher",
            role="research",
            model="gpt-4o",
            provider="openai",
            system_prompt="You are a research assistant.",
            allowed_tools=["web_search", "read_file"],
            denied_tools=["write_file"],
            allowed_servers=["server-1"],
            tool_timeout_override=30.0,
            auto_approve_tools=["web_search"],
            parent_agent_id="agent-main",
            initial_prompt="Find the MCP spec.",
        )
        assert cfg.name == "Researcher"
        assert cfg.role == "research"
        assert cfg.model == "gpt-4o"
        assert cfg.provider == "openai"
        assert cfg.allowed_tools == ["web_search", "read_file"]
        assert cfg.denied_tools == ["write_file"]
        assert cfg.parent_agent_id == "agent-main"
        assert cfg.initial_prompt == "Find the MCP spec."


class TestAgentConfigSerialization:
    def test_dict_roundtrip(self):
        cfg = AgentConfig(agent_id="x", name="X", role="worker", model="gpt-4")
        d = cfg.model_dump()
        assert d["agent_id"] == "x"
        assert d["role"] == "worker"
        cfg2 = AgentConfig(**d)
        assert cfg2 == cfg

    def test_json_roundtrip(self):
        cfg = AgentConfig(
            agent_id="a",
            name="A",
            allowed_tools=["t1", "t2"],
        )
        json_str = cfg.model_dump_json()
        cfg2 = AgentConfig.model_validate_json(json_str)
        assert cfg2 == cfg


class TestAgentConfigImport:
    def test_importable_from_package(self):
        from mcp_cli.agents import AgentConfig as AC

        assert AC is AgentConfig


class TestAgentConfigValidation:
    def test_agent_id_required(self):
        with pytest.raises(Exception):
            AgentConfig()  # type: ignore[call-arg]

    def test_extra_fields_ignored(self):
        """Extra fields are silently ignored by Pydantic v2 default."""
        cfg = AgentConfig(agent_id="x", unknown_field="ignored")  # type: ignore[call-arg]
        assert cfg.agent_id == "x"
        assert not hasattr(cfg, "unknown_field")
