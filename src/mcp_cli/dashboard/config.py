# mcp_cli/dashboard/config.py
"""Dashboard layout configuration: presets and user-saved layouts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mcp_cli.config.defaults import DEFAULT_DASHBOARD_LAYOUTS_FILE

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Built-in layout presets                                            #
# ------------------------------------------------------------------ #

LAYOUT_PRESETS: dict[str, dict[str, Any]] = {
    "Mobile": {
        "name": "Mobile",
        "layout": {
            "rows": [
                {
                    "height": "100%",
                    "columns": [
                        {"width": "100%", "view": "builtin:agent-terminal"},
                    ],
                }
            ]
        },
    },
    "Minimal": {
        "name": "Minimal",
        "layout": {
            "rows": [
                {
                    "height": "100%",
                    "columns": [
                        {"width": "100%", "view": "builtin:agent-terminal"},
                    ],
                }
            ]
        },
    },
    "Standard": {
        "name": "Standard",
        "layout": {
            "rows": [
                {
                    "height": "100%",
                    "columns": [
                        {"width": "70%", "view": "builtin:agent-terminal"},
                        {"width": "30%", "view": "builtin:activity-stream"},
                    ],
                }
            ]
        },
    },
    "Full": {
        "name": "Full",
        "layout": {
            "rows": [
                {
                    "height": "60%",
                    "columns": [
                        {"width": "65%", "view": "builtin:agent-terminal"},
                        {"width": "35%", "view": "builtin:activity-stream"},
                    ],
                },
                {
                    "height": "40%",
                    "columns": [
                        {"width": "100%", "view": "auto"},
                    ],
                },
            ]
        },
    },
}

DEFAULT_PRESET_NAME = "Standard"

# ------------------------------------------------------------------ #
#  Built-in view registry (always available)                          #
# ------------------------------------------------------------------ #

BUILTIN_VIEWS: list[dict[str, Any]] = [
    {
        "id": "builtin:agent-terminal",
        "name": "Agent Terminal",
        "source": "builtin",
        "icon": "terminal",
        "type": "conversation",
        "url": "/views/agent-terminal.html",
    },
    {
        "id": "builtin:activity-stream",
        "name": "Activity Stream",
        "source": "builtin",
        "icon": "activity",
        "type": "stream",
        "url": "/views/activity-stream.html",
    },
    {
        "id": "builtin:agent-overview",
        "name": "Agent Overview",
        "source": "builtin",
        "icon": "agents",
        "type": "agents",
        "url": "/views/agent-overview.html",
    },
]

# ------------------------------------------------------------------ #
#  User layout persistence                                            #
# ------------------------------------------------------------------ #


def _layouts_path() -> Path:
    return Path(DEFAULT_DASHBOARD_LAYOUTS_FILE).expanduser()


def load_user_layouts() -> list[dict[str, Any]]:
    """Load saved user layouts from disk. Returns empty list on any error."""
    path = _layouts_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        logger.warning("Dashboard layouts file has unexpected format: %s", path)
        return []
    except Exception as exc:
        logger.warning("Could not load dashboard layouts from %s: %s", path, exc)
        return []


def save_user_layout(name: str, layout: dict[str, Any]) -> None:
    """Save a named layout to disk, replacing any existing layout with the same name."""
    layouts = load_user_layouts()
    layouts = [lyt for lyt in layouts if lyt.get("name") != name]
    layouts.append({"name": name, "layout": layout})
    _write_layouts(layouts)


def delete_user_layout(name: str) -> bool:
    """Delete a named user layout. Returns True if it existed."""
    layouts = load_user_layouts()
    filtered = [lyt for lyt in layouts if lyt.get("name") != name]
    if len(filtered) == len(layouts):
        return False
    _write_layouts(filtered)
    return True


def _write_layouts(layouts: list[dict[str, Any]]) -> None:
    path = _layouts_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(layouts, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not save dashboard layouts to %s: %s", path, exc)
