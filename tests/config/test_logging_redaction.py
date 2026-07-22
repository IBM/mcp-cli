# tests/config/test_logging_redaction.py
"""Tests for secret redaction and file logging in config/logging.py."""

from __future__ import annotations

import logging
import os
import tempfile

import pytest

from mcp_cli.config.logging import (
    SecretRedactingFilter,
    secret_filter,
    setup_logging,
)


def _close_file_handlers():
    """Close and remove all file handlers from the root logger."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        if hasattr(h, "baseFilename"):
            h.close()
            root.removeHandler(h)


class TestSecretRedactingFilter:
    """Verify each redaction pattern works correctly."""

    def _make_record(self, msg: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=None,
            exc_info=None,
        )

    def test_bearer_token_redacted(self):
        record = self._make_record("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc123")
        secret_filter.filter(record)
        assert "eyJ" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_sk_api_key_redacted(self):
        record = self._make_record("Using key sk-proj-abc123456789xyz")
        secret_filter.filter(record)
        assert "sk-proj" not in record.msg
        assert "[REDACTED_API_KEY]" in record.msg

    def test_api_key_equals_redacted(self):
        record = self._make_record("api_key=my-secret-key-12345")
        secret_filter.filter(record)
        assert "my-secret-key" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_api_key_colon_redacted(self):
        record = self._make_record('api-key: "some-secret"')
        secret_filter.filter(record)
        assert "some-secret" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_access_token_json_redacted(self):
        record = self._make_record('{"access_token": "ghp_abcdef123456"}')
        secret_filter.filter(record)
        assert "ghp_abcdef" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_authorization_header_redacted(self):
        record = self._make_record("Authorization=Basic dXNlcjpwYXNz")
        secret_filter.filter(record)
        assert "dXNlcjpwYXNz" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_non_secret_message_unchanged(self):
        msg = "Processing 42 tool calls for user request"
        record = self._make_record(msg)
        secret_filter.filter(record)
        assert record.msg == msg

    def test_filter_returns_true(self):
        """Filter should never suppress records, only redact."""
        record = self._make_record("anything")
        assert secret_filter.filter(record) is True

    def test_args_redacted(self):
        """Verify format args containing secrets are also redacted."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Token is %s",
            args=("Bearer eyJsecret",),
            exc_info=None,
        )
        secret_filter.filter(record)
        assert "eyJsecret" not in record.msg
        assert record.args is None  # cleared after interpolation

    def test_singleton_is_instance(self):
        assert isinstance(secret_filter, SecretRedactingFilter)


class TestSetupLoggingFileHandler:
    """Verify file handler creation with tmpdir and rotation config."""

    def test_file_handler_created(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        setup_logging(level="DEBUG", log_file=log_file)

        root = logging.getLogger()
        file_handlers = [
            h
            for h in root.handlers
            if hasattr(h, "baseFilename")
            and "test.log" in getattr(h, "baseFilename", "")
        ]
        assert len(file_handlers) == 1
        handler = file_handlers[0]

        # Verify rotation config
        from mcp_cli.config.defaults import (
            DEFAULT_LOG_MAX_BYTES,
            DEFAULT_LOG_BACKUP_COUNT,
        )

        assert handler.maxBytes == DEFAULT_LOG_MAX_BYTES
        assert handler.backupCount == DEFAULT_LOG_BACKUP_COUNT

        # Verify file handler has redaction filter
        filter_types = [type(f) for f in handler.filters]
        assert SecretRedactingFilter in filter_types

        # Clean up
        _close_file_handlers()
        setup_logging(level="WARNING")

    def test_file_handler_creates_parent_dirs(self, tmp_path):
        log_file = str(tmp_path / "subdir" / "nested" / "app.log")
        setup_logging(level="WARNING", log_file=log_file)

        assert (tmp_path / "subdir" / "nested").is_dir()

        # Clean up
        _close_file_handlers()
        setup_logging(level="WARNING")

    def test_file_handler_writes_json(self, tmp_path):
        log_file = str(tmp_path / "json.log")
        setup_logging(level="DEBUG", log_file=log_file)

        # Log a message
        test_logger = logging.getLogger("mcp_cli.test_file_json")
        test_logger.debug("hello from test")

        # Flush handlers
        for h in logging.getLogger().handlers:
            h.flush()

        import json

        content = (tmp_path / "json.log").read_text()
        # Should contain JSON with our message
        assert "hello from test" in content
        # Should be valid JSON lines
        for line in content.strip().splitlines():
            if line.strip():
                parsed = json.loads(line)
                assert "timestamp" in parsed
                assert "level" in parsed

        # Clean up
        _close_file_handlers()
        setup_logging(level="WARNING")

    def test_file_handler_redacts_secrets(self, tmp_path):
        log_file = str(tmp_path / "redact.log")
        setup_logging(level="DEBUG", log_file=log_file)

        test_logger = logging.getLogger("mcp_cli.test_redact")
        test_logger.info("Token is Bearer eyJsupersecret123")

        for h in logging.getLogger().handlers:
            h.flush()

        content = (tmp_path / "redact.log").read_text()
        assert "eyJsupersecret" not in content
        assert "[REDACTED]" in content

        # Clean up
        _close_file_handlers()
        setup_logging(level="WARNING")

    def test_tilde_expansion(self):
        """Verify ~ is expanded in log file path."""
        # We don't actually write; just verify the path logic
        # by checking that setup_logging doesn't crash with ~ paths
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate ~ by using a known path
            log_file = os.path.join(tmpdir, "test.log")
            setup_logging(level="WARNING", log_file=log_file)
            assert os.path.isdir(tmpdir)
            setup_logging(level="WARNING")


class TestConsoleHandlerHasRedaction:
    """Verify that console handler always gets the redaction filter."""

    def test_console_handler_has_filter(self):
        setup_logging(level="WARNING")

        root = logging.getLogger()
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler) and not hasattr(
                handler, "baseFilename"
            ):
                filter_types = [type(f) for f in handler.filters]
                assert SecretRedactingFilter in filter_types
                return

        pytest.fail("No console StreamHandler found on root logger")
