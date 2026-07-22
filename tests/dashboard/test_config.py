# tests/dashboard/test_config.py
"""Unit tests for mcp_cli.dashboard.config."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Constants / presets
# ---------------------------------------------------------------------------


class TestConstants:
    def test_layout_presets_contains_all_expected(self):
        from mcp_cli.dashboard.config import LAYOUT_PRESETS

        assert set(LAYOUT_PRESETS) == {"Mobile", "Minimal", "Standard", "Full"}

    def test_each_preset_has_layout_key(self):
        from mcp_cli.dashboard.config import LAYOUT_PRESETS

        for name, preset in LAYOUT_PRESETS.items():
            assert "layout" in preset, f"Preset {name!r} missing 'layout'"
            assert "rows" in preset["layout"], f"Preset {name!r} layout missing 'rows'"

    def test_builtin_views_ids(self):
        from mcp_cli.dashboard.config import BUILTIN_VIEWS

        ids = {v["id"] for v in BUILTIN_VIEWS}
        assert "builtin:agent-terminal" in ids
        assert "builtin:activity-stream" in ids

    def test_default_preset_name_is_standard(self):
        from mcp_cli.dashboard.config import DEFAULT_PRESET_NAME

        assert DEFAULT_PRESET_NAME == "Standard"


# ---------------------------------------------------------------------------
# load_user_layouts
# ---------------------------------------------------------------------------


class TestLoadUserLayouts:
    def test_returns_empty_list_when_file_missing(self, tmp_path):
        from mcp_cli.dashboard.config import load_user_layouts

        with patch(
            "mcp_cli.dashboard.config._layouts_path",
            return_value=tmp_path / "nope.json",
        ):
            result = load_user_layouts()
        assert result == []

    def test_returns_list_from_valid_file(self, tmp_path):
        from mcp_cli.dashboard.config import load_user_layouts

        f = tmp_path / "layouts.json"
        f.write_text(json.dumps([{"name": "A", "layout": {}}]))
        with patch("mcp_cli.dashboard.config._layouts_path", return_value=f):
            result = load_user_layouts()
        assert result == [{"name": "A", "layout": {}}]

    def test_returns_empty_when_file_is_not_a_list(self, tmp_path):
        from mcp_cli.dashboard.config import load_user_layouts

        f = tmp_path / "layouts.json"
        f.write_text(json.dumps({"key": "value"}))
        with patch("mcp_cli.dashboard.config._layouts_path", return_value=f):
            result = load_user_layouts()
        assert result == []

    def test_returns_empty_on_invalid_json(self, tmp_path):
        from mcp_cli.dashboard.config import load_user_layouts

        f = tmp_path / "layouts.json"
        f.write_text("not json {{{{")
        with patch("mcp_cli.dashboard.config._layouts_path", return_value=f):
            result = load_user_layouts()
        assert result == []

    def test_returns_empty_on_read_error(self, tmp_path):
        from mcp_cli.dashboard.config import load_user_layouts

        f = tmp_path / "layouts.json"
        f.write_text("[]")  # file exists so path.exists() is True
        with patch("mcp_cli.dashboard.config._layouts_path", return_value=f):
            with patch.object(Path, "read_text", side_effect=OSError("perm")):
                result = load_user_layouts()
        assert result == []


# ---------------------------------------------------------------------------
# save_user_layout
# ---------------------------------------------------------------------------


class TestSaveUserLayout:
    def test_saves_new_layout(self, tmp_path):
        from mcp_cli.dashboard.config import load_user_layouts, save_user_layout

        f = tmp_path / "layouts.json"
        with patch("mcp_cli.dashboard.config._layouts_path", return_value=f):
            save_user_layout("MyLayout", {"rows": []})
            result = load_user_layouts()

        assert len(result) == 1
        assert result[0]["name"] == "MyLayout"

    def test_replaces_existing_layout_by_name(self, tmp_path):
        from mcp_cli.dashboard.config import load_user_layouts, save_user_layout

        f = tmp_path / "layouts.json"
        with patch("mcp_cli.dashboard.config._layouts_path", return_value=f):
            save_user_layout("MyLayout", {"rows": [{"id": "old"}]})
            save_user_layout("MyLayout", {"rows": [{"id": "new"}]})
            result = load_user_layouts()

        assert len(result) == 1
        assert result[0]["layout"]["rows"][0]["id"] == "new"

    def test_multiple_layouts_coexist(self, tmp_path):
        from mcp_cli.dashboard.config import load_user_layouts, save_user_layout

        f = tmp_path / "layouts.json"
        with patch("mcp_cli.dashboard.config._layouts_path", return_value=f):
            save_user_layout("Alpha", {})
            save_user_layout("Beta", {})
            result = load_user_layouts()

        names = {r["name"] for r in result}
        assert names == {"Alpha", "Beta"}

    def test_write_error_does_not_raise(self, tmp_path):
        from mcp_cli.dashboard.config import save_user_layout

        f = tmp_path / "layouts.json"
        with patch("mcp_cli.dashboard.config._layouts_path", return_value=f):
            with patch.object(Path, "write_text", side_effect=OSError("disk full")):
                save_user_layout("MyLayout", {})  # should not raise


# ---------------------------------------------------------------------------
# delete_user_layout
# ---------------------------------------------------------------------------


class TestDeleteUserLayout:
    def test_delete_existing_returns_true(self, tmp_path):
        from mcp_cli.dashboard.config import delete_user_layout, save_user_layout

        f = tmp_path / "layouts.json"
        with patch("mcp_cli.dashboard.config._layouts_path", return_value=f):
            save_user_layout("ToDelete", {})
            result = delete_user_layout("ToDelete")

        assert result is True

    def test_delete_actually_removes_layout(self, tmp_path):
        from mcp_cli.dashboard.config import (
            delete_user_layout,
            load_user_layouts,
            save_user_layout,
        )

        f = tmp_path / "layouts.json"
        with patch("mcp_cli.dashboard.config._layouts_path", return_value=f):
            save_user_layout("ToDelete", {})
            save_user_layout("Keep", {})
            delete_user_layout("ToDelete")
            result = load_user_layouts()

        assert len(result) == 1
        assert result[0]["name"] == "Keep"

    def test_delete_nonexistent_returns_false(self, tmp_path):
        from mcp_cli.dashboard.config import delete_user_layout

        f = tmp_path / "nope.json"
        with patch("mcp_cli.dashboard.config._layouts_path", return_value=f):
            result = delete_user_layout("Ghost")

        assert result is False
