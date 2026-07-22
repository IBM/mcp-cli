# mcp_cli/agents/group_store.py
"""Save and restore multi-agent groups.

A *group* is a snapshot of all agents' configurations and their sessions,
stored on disk so the entire multi-agent setup can be resumed later.

Layout::

    ~/.mcp-cli/groups/{group_id}/
        group.json                 # AgentDescriptors + relationships
        {agent_id}/session.json    # Each agent's session state
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp_cli.agents.manager import AgentManager

logger = logging.getLogger(__name__)

_DEFAULT_GROUPS_DIR = Path.home() / ".mcp-cli" / "groups"


async def save_group(
    agent_manager: AgentManager,
    description: str = "",
    group_dir: Path | None = None,
) -> Path:
    """Save all agents' configs and sessions to disk.

    Parameters
    ----------
    agent_manager:
        The AgentManager whose agents to save.
    description:
        Human-readable description for the group snapshot.
    group_dir:
        Override directory. Default: ``~/.mcp-cli/groups/{group_id}/``

    Returns
    -------
    Path
        The directory where the group was saved.
    """
    group_id = f"group-{int(time.time())}"
    base = group_dir or (_DEFAULT_GROUPS_DIR / group_id)
    base.mkdir(parents=True, exist_ok=True)

    agents_data: list[dict[str, Any]] = []

    for status in agent_manager.list_agents():
        agent_id = status["agent_id"]
        snapshot = agent_manager.get_agent_snapshot(agent_id)
        if snapshot is None:
            continue

        # Save agent config
        config_dict = snapshot["config"].model_dump()
        agents_data.append(
            {
                **config_dict,
                "status": status.get("status", "unknown"),
            }
        )

        # Save session if available
        ctx = snapshot["context"]
        agent_dir = base / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)
        try:
            history = getattr(ctx, "conversation_history", [])
            session_data = {
                "agent_id": agent_id,
                "session_id": getattr(ctx, "session_id", ""),
                "messages": [
                    m.to_dict() if hasattr(m, "to_dict") else dict(m) for m in history
                ],
            }
            (agent_dir / "session.json").write_text(
                json.dumps(session_data, indent=2, default=str)
            )
        except Exception as exc:
            logger.warning("Failed to save session for %s: %s", agent_id, exc)

    # Write group manifest
    group_manifest = {
        "group_id": group_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": description,
        "agents": agents_data,
    }
    (base / "group.json").write_text(json.dumps(group_manifest, indent=2, default=str))

    logger.info("Group saved to %s (%d agents)", base, len(agents_data))
    return base


def load_group_manifest(group_path: Path) -> dict[str, Any]:
    """Load a group manifest from disk.

    Parameters
    ----------
    group_path:
        Path to the group directory containing ``group.json``.

    Returns
    -------
    dict
        The parsed group manifest.
    """
    manifest_file = group_path / "group.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"No group.json in {group_path}")
    result: dict[str, Any] = json.loads(manifest_file.read_text())
    return result


def list_groups(groups_dir: Path | None = None) -> list[dict[str, Any]]:
    """List all saved groups.

    Returns
    -------
    list[dict]
        List of ``{group_id, description, created_at, agent_count, path}``
    """
    base = groups_dir or _DEFAULT_GROUPS_DIR
    if not base.exists():
        return []

    results = []
    for d in sorted(base.iterdir()):
        manifest_file = d / "group.json"
        if not manifest_file.exists():
            continue
        try:
            manifest = json.loads(manifest_file.read_text())
            results.append(
                {
                    "group_id": manifest.get("group_id", d.name),
                    "description": manifest.get("description", ""),
                    "created_at": manifest.get("created_at", ""),
                    "agent_count": len(manifest.get("agents", [])),
                    "path": str(d),
                }
            )
        except Exception as exc:
            logger.debug("Skipping invalid group %s: %s", d, exc)

    return results
