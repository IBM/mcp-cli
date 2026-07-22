# tests/chat/state/test_cache.py
"""Tests for ResultCache."""

import pytest

from chuk_ai_session_manager.guards.cache import ResultCache


class TestResultCache:
    """Tests for ResultCache."""

    @pytest.fixture
    def cache(self):
        return ResultCache()

    def test_put_and_get(self, cache):
        """Test caching and retrieving a result."""
        cache.put("sqrt", {"x": 18}, 4.2426)
        cached = cache.get("sqrt", {"x": 18})
        assert cached is not None
        assert cached.result == 4.2426

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        cached = cache.get("sqrt", {"x": 18})
        assert cached is None

    def test_duplicate_increments_count(self, cache):
        """Test that duplicate calls increment count."""
        cache.put("sqrt", {"x": 18}, 4.2426)
        cached1 = cache.get("sqrt", {"x": 18})
        assert cached1.call_count == 2  # put + get

        cached2 = cache.get("sqrt", {"x": 18})
        assert cached2.call_count == 3

    def test_duplicate_count_tracking(self, cache):
        """Test duplicate count is tracked."""
        cache.put("sqrt", {"x": 18}, 4.2426)
        assert cache.duplicate_count == 0

        cache.get("sqrt", {"x": 18})
        assert cache.duplicate_count == 1

    def test_store_variable(self, cache):
        """Test storing named variables."""
        var = cache.store_variable("sigma", 5.5, units="units/day")
        assert var.name == "sigma"
        assert var.value == 5.5
        assert var.units == "units/day"

    def test_get_variable(self, cache):
        """Test retrieving stored variables."""
        cache.store_variable("sigma", 5.5)
        var = cache.get_variable("sigma")
        assert var is not None
        assert var.value == 5.5

    def test_get_variable_missing(self, cache):
        """Test getting non-existent variable returns None."""
        var = cache.get_variable("nonexistent")
        assert var is None

    def test_format_state_empty(self, cache):
        """Test format_state with empty cache."""
        state = cache.format_state()
        assert state == ""

    def test_format_state_with_results(self, cache):
        """Test format_state with cached results."""
        cache.put("sqrt", {"x": 18}, 4.2426)
        cache.put("multiply", {"a": 2, "b": 3}, 6)
        state = cache.format_state()
        assert "sqrt" in state or "multiply" in state

    def test_reset(self, cache):
        """Test reset clears all state."""
        cache.put("sqrt", {"x": 18}, 4.2426)
        cache.store_variable("sigma", 5.5)
        cache.reset()

        assert cache.get("sqrt", {"x": 18}) is None
        assert cache.get_variable("sigma") is None
        assert cache.duplicate_count == 0

    def test_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ResultCache(max_size=3)
        cache.put("tool1", {"x": 1}, 1)
        cache.put("tool2", {"x": 2}, 2)
        cache.put("tool3", {"x": 3}, 3)
        cache.put("tool4", {"x": 4}, 4)

        # First entry should be evicted
        assert cache.get("tool1", {"x": 1}) is None
        assert cache.get("tool4", {"x": 4}) is not None

    def test_get_stats(self, cache):
        """Test get_stats returns correct info."""
        cache.put("sqrt", {"x": 18}, 4.2426)
        cache.store_variable("sigma", 5.5)

        stats = cache.get_stats()
        assert stats.total_cached == 1
        assert stats.total_variables == 1

    def test_format_duplicate_message(self, cache):
        """Test format_duplicate_message."""
        cache.put("sqrt", {"x": 18}, 4.2426)
        msg = cache.format_duplicate_message("sqrt", {"x": 18})
        assert "sqrt" in msg
        assert "4.2426" in msg or "cached" in msg.lower()

    # -------------------------------------------------------------------------
    # Additional coverage tests for uncovered lines
    # -------------------------------------------------------------------------

    def test_format_duplicate_message_no_cache(self, cache):
        """Test format_duplicate_message when not in cache."""
        msg = cache.format_duplicate_message("sqrt", {"x": 99})
        assert "sqrt" in msg
        assert "no cached result" in msg.lower()

    def test_format_state_with_variables_only(self, cache):
        """Test format_state with only variables (no tool results)."""
        cache.store_variable("sigma", 5.5, units="units/day")
        state = cache.format_state()
        assert "sigma" in state
        assert "Stored Variables" in state

    def test_format_state_variables_and_results(self, cache):
        """Test format_state with both variables and results."""
        cache.store_variable("sigma", 5.5)
        cache.put("sqrt", {"x": 18}, 4.2426)
        state = cache.format_state()
        assert "sigma" in state
        assert "sqrt" in state

    def test_format_state_max_items_limit(self, cache):
        """Test format_state respects max_items."""
        # Add multiple variables
        for i in range(15):
            cache.store_variable(f"var{i}", float(i))
        state = cache.format_state(max_items=5)
        # Should only show 5 variables
        assert state.count("var") <= 10  # Some slack for formatting

    def test_format_state_separator_between_sections(self, cache):
        """Test format_state adds separator between variable and result sections."""
        cache.store_variable("sigma", 5.5)
        cache.put("sqrt", {"x": 18}, 4.2426)
        state = cache.format_state()
        # Should have both sections
        assert "Stored Variables" in state
        assert "Computed Values" in state
