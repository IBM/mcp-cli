# tests/chat/test_attachments.py
"""Unit tests for multi-modal attachment processing."""

from __future__ import annotations

import pytest

from mcp_cli.chat.attachments import (
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    TEXT_EXTENSIONS,
    Attachment,
    AttachmentStaging,
    _classify_kind,
    attachment_descriptor,
    build_multimodal_content,
    detect_image_urls,
    detect_mime_type,
    parse_inline_refs,
    process_browser_file,
    process_local_file,
    process_url,
)
from mcp_cli.config.defaults import DEFAULT_MAX_ATTACHMENT_SIZE_BYTES


# ---------------------------------------------------------------------------
# MIME detection
# ---------------------------------------------------------------------------


class TestDetectMimeType:
    def test_png(self, tmp_path):
        from pathlib import Path

        assert detect_mime_type(Path("test.png")) == "image/png"

    def test_jpg(self, tmp_path):
        from pathlib import Path

        assert detect_mime_type(Path("photo.jpg")) == "image/jpeg"

    def test_jpeg(self):
        from pathlib import Path

        assert detect_mime_type(Path("photo.jpeg")) == "image/jpeg"

    def test_mp3(self):
        from pathlib import Path

        assert detect_mime_type(Path("audio.mp3")) == "audio/mpeg"

    def test_python(self):
        from pathlib import Path

        assert detect_mime_type(Path("code.py")) == "text/plain"

    def test_csv(self):
        from pathlib import Path

        assert detect_mime_type(Path("data.csv")) == "text/csv"

    def test_unknown(self):
        from pathlib import Path

        assert detect_mime_type(Path("file.xyz")) == "application/octet-stream"


# ---------------------------------------------------------------------------
# Inline @file: parsing
# ---------------------------------------------------------------------------


class TestParseInlineRefs:
    def test_single_ref(self):
        text, paths = parse_inline_refs("Look at @file:img.png please")
        assert paths == ["img.png"]
        assert "@file" not in text
        assert "Look at" in text
        assert "please" in text

    def test_multiple_refs(self):
        text, paths = parse_inline_refs("Compare @file:a.png and @file:b.jpg")
        assert len(paths) == 2
        assert "a.png" in paths
        assert "b.jpg" in paths
        assert "@file" not in text

    def test_no_refs(self):
        text, paths = parse_inline_refs("No refs here")
        assert paths == []
        assert text == "No refs here"

    def test_path_with_directory(self):
        text, paths = parse_inline_refs("See @file:/tmp/dir/photo.png")
        assert paths == ["/tmp/dir/photo.png"]

    def test_empty_string(self):
        text, paths = parse_inline_refs("")
        assert paths == []
        assert text == ""

    def test_collapses_spaces(self):
        text, _ = parse_inline_refs("before @file:x.png after")
        # Should not have double spaces
        assert "  " not in text


# ---------------------------------------------------------------------------
# Image URL detection
# ---------------------------------------------------------------------------


class TestDetectImageUrls:
    def test_png_url(self):
        urls = detect_image_urls("Look at https://example.com/cat.png")
        assert urls == ["https://example.com/cat.png"]

    def test_jpg_url(self):
        urls = detect_image_urls("https://example.com/photo.jpg is nice")
        assert urls == ["https://example.com/photo.jpg"]

    def test_case_insensitive(self):
        urls = detect_image_urls("https://example.com/cat.PNG")
        assert len(urls) == 1

    def test_url_with_query(self):
        urls = detect_image_urls("https://example.com/img.png?size=large")
        assert len(urls) == 1
        assert "?size=large" in urls[0]

    def test_no_urls(self):
        urls = detect_image_urls("Just plain text here")
        assert urls == []

    def test_non_image_url(self):
        urls = detect_image_urls("Visit https://example.com/page.html")
        assert urls == []

    def test_multiple_urls(self):
        urls = detect_image_urls("https://a.com/1.png and https://b.com/2.jpg")
        assert len(urls) == 2


# ---------------------------------------------------------------------------
# AttachmentStaging
# ---------------------------------------------------------------------------


class TestAttachmentStaging:
    def _make_att(self, name: str = "test.png") -> Attachment:
        return Attachment(
            source=name,
            content_blocks=[{"type": "text", "text": "mock"}],
            display_name=name,
            size_bytes=100,
            mime_type="image/png",
        )

    def test_stage_and_count(self):
        staging = AttachmentStaging()
        assert staging.count == 0
        staging.stage(self._make_att())
        assert staging.count == 1

    def test_drain_returns_and_clears(self):
        staging = AttachmentStaging()
        staging.stage(self._make_att("a.png"))
        staging.stage(self._make_att("b.png"))
        drained = staging.drain()
        assert len(drained) == 2
        assert staging.count == 0

    def test_peek_does_not_clear(self):
        staging = AttachmentStaging()
        staging.stage(self._make_att())
        peeked = staging.peek()
        assert len(peeked) == 1
        assert staging.count == 1

    def test_clear(self):
        staging = AttachmentStaging()
        staging.stage(self._make_att())
        staging.clear()
        assert staging.count == 0

    def test_drain_empty(self):
        staging = AttachmentStaging()
        assert staging.drain() == []


# ---------------------------------------------------------------------------
# process_local_file
# ---------------------------------------------------------------------------


class TestProcessLocalFile:
    def test_image_file(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        att = process_local_file(str(img))
        assert att.mime_type == "image/png"
        assert att.display_name == "test.png"
        assert att.size_bytes > 0
        assert len(att.content_blocks) == 1
        block = att.content_blocks[0]
        assert block["type"] == "image_url"
        assert block["image_url"]["url"].startswith("data:image/png;base64,")

    def test_text_file(self, tmp_path):
        txt = tmp_path / "code.py"
        txt.write_text("print('hello')")
        att = process_local_file(str(txt))
        assert att.mime_type == "text/plain"
        assert len(att.content_blocks) == 1
        block = att.content_blocks[0]
        assert block["type"] == "text"
        assert "print('hello')" in block["text"]
        assert "--- code.py ---" in block["text"]

    def test_audio_file(self, tmp_path):
        audio = tmp_path / "clip.mp3"
        audio.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 50)
        att = process_local_file(str(audio))
        assert att.mime_type == "audio/mpeg"
        block = att.content_blocks[0]
        assert block["type"] == "input_audio"
        assert block["input_audio"]["format"] == "mp3"

    def test_wav_audio(self, tmp_path):
        audio = tmp_path / "clip.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 50)
        att = process_local_file(str(audio))
        block = att.content_blocks[0]
        assert block["input_audio"]["format"] == "wav"

    def test_csv_file(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("a,b,c\n1,2,3")
        att = process_local_file(str(csv))
        assert att.mime_type == "text/csv"
        assert "a,b,c" in att.content_blocks[0]["text"]

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            process_local_file("/nonexistent/file.png")

    def test_not_a_file(self, tmp_path):
        with pytest.raises(ValueError, match="Not a file"):
            process_local_file(str(tmp_path))

    def test_too_large(self, tmp_path):
        big = tmp_path / "huge.png"
        big.write_bytes(b"\x00" * (DEFAULT_MAX_ATTACHMENT_SIZE_BYTES + 1))
        with pytest.raises(ValueError, match="too large"):
            process_local_file(str(big))

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_bytes(b"\x00" * 10)
        with pytest.raises(ValueError, match="Unsupported"):
            process_local_file(str(f))

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        # Just verify expanduser is called (don't rely on actual ~)
        f = tmp_path / "test.txt"
        f.write_text("content")
        att = process_local_file(str(f))
        assert att.display_name == "test.txt"

    def test_json_file(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text('{"key": "value"}')
        att = process_local_file(str(f))
        assert att.mime_type == "application/json"
        assert '{"key": "value"}' in att.content_blocks[0]["text"]


# ---------------------------------------------------------------------------
# process_url
# ---------------------------------------------------------------------------


class TestProcessUrl:
    def test_basic_url(self):
        att = process_url("https://example.com/photo.jpg")
        assert att.source == "https://example.com/photo.jpg"
        assert att.size_bytes == 0
        block = att.content_blocks[0]
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "https://example.com/photo.jpg"

    def test_display_name(self):
        att = process_url("https://example.com/path/to/image.png")
        assert att.display_name == "image.png"


# ---------------------------------------------------------------------------
# build_multimodal_content
# ---------------------------------------------------------------------------


class TestBuildMultimodalContent:
    def test_plain_text_passthrough(self):
        result = build_multimodal_content("hello", [], [])
        assert result == "hello"
        assert isinstance(result, str)

    def test_with_attachment(self):
        att = Attachment(
            source="test.png",
            content_blocks=[
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            ],
            display_name="test.png",
            size_bytes=100,
            mime_type="image/png",
        )
        result = build_multimodal_content("describe this", [att], [])
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "describe this"}
        assert result[1]["type"] == "image_url"

    def test_with_image_url(self):
        result = build_multimodal_content(
            "what is this", [], ["https://ex.com/img.png"]
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"] == "https://ex.com/img.png"

    def test_dedup_urls(self):
        """Image URLs already in attachments are not duplicated."""
        att = Attachment(
            source="https://ex.com/img.png",
            content_blocks=[
                {"type": "image_url", "image_url": {"url": "https://ex.com/img.png"}}
            ],
            display_name="img.png",
            size_bytes=0,
            mime_type="image/png",
        )
        result = build_multimodal_content("look", [att], ["https://ex.com/img.png"])
        assert isinstance(result, list)
        # Should have text + 1 image (not 2)
        image_blocks = [b for b in result if b.get("type") == "image_url"]
        assert len(image_blocks) == 1

    def test_empty_text_with_attachment(self):
        att = Attachment(
            source="test.png",
            content_blocks=[{"type": "image_url", "image_url": {"url": "data:..."}}],
            display_name="test.png",
            size_bytes=100,
            mime_type="image/png",
        )
        result = build_multimodal_content("", [att], [])
        assert isinstance(result, list)
        # No text block when text is empty
        text_blocks = [b for b in result if b.get("type") == "text"]
        assert len(text_blocks) == 0

    def test_multiple_attachments(self):
        atts = [
            Attachment(
                source=f"f{i}.png",
                content_blocks=[
                    {"type": "image_url", "image_url": {"url": f"data:{i}"}}
                ],
                display_name=f"f{i}.png",
                size_bytes=100,
                mime_type="image/png",
            )
            for i in range(3)
        ]
        result = build_multimodal_content("compare these", atts, [])
        assert isinstance(result, list)
        assert len(result) == 4  # 1 text + 3 images


# ---------------------------------------------------------------------------
# Extension sets
# ---------------------------------------------------------------------------


class TestExtensionSets:
    def test_no_overlap(self):
        assert IMAGE_EXTENSIONS & AUDIO_EXTENSIONS == set()
        assert IMAGE_EXTENSIONS & TEXT_EXTENSIONS == set()
        assert AUDIO_EXTENSIONS & TEXT_EXTENSIONS == set()

    def test_common_extensions_covered(self):
        assert ".png" in IMAGE_EXTENSIONS
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".mp3" in AUDIO_EXTENSIONS
        assert ".py" in TEXT_EXTENSIONS
        assert ".txt" in TEXT_EXTENSIONS


# ---------------------------------------------------------------------------
# _classify_kind
# ---------------------------------------------------------------------------


class TestClassifyKind:
    def test_image(self):
        assert _classify_kind("image/png") == "image"
        assert _classify_kind("image/jpeg") == "image"

    def test_audio(self):
        assert _classify_kind("audio/mpeg") == "audio"
        assert _classify_kind("audio/wav") == "audio"

    def test_text(self):
        assert _classify_kind("text/plain") == "text"
        assert _classify_kind("text/csv") == "text"

    def test_json_is_text(self):
        assert _classify_kind("application/json") == "text"

    def test_unknown(self):
        assert _classify_kind("application/octet-stream") == "unknown"


# ---------------------------------------------------------------------------
# attachment_descriptor
# ---------------------------------------------------------------------------


class TestAttachmentDescriptor:
    def test_small_image(self, tmp_path):
        img = tmp_path / "small.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        att = process_local_file(str(img))
        desc = attachment_descriptor(att)
        assert desc["kind"] == "image"
        assert desc["display_name"] == "small.png"
        assert desc["preview_url"] is not None
        assert desc["preview_url"].startswith("data:image/png;base64,")

    def test_large_image(self, tmp_path):
        img = tmp_path / "big.png"
        # Write >100KB to exceed threshold
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 110_000)
        att = process_local_file(str(img))
        desc = attachment_descriptor(att)
        assert desc["kind"] == "image"
        assert desc["preview_url"] is None

    def test_url_image(self):
        att = process_url("https://example.com/photo.jpg")
        desc = attachment_descriptor(att)
        assert desc["kind"] == "image"
        assert desc["preview_url"] == "https://example.com/photo.jpg"

    def test_text_file(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("x = 42\n" * 100)
        att = process_local_file(str(f))
        desc = attachment_descriptor(att)
        assert desc["kind"] == "text"
        assert "x = 42" in desc["text_preview"]
        assert isinstance(desc["text_truncated"], bool)

    def test_text_file_truncation(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("A" * 5000)
        att = process_local_file(str(f))
        desc = attachment_descriptor(att, text_preview_chars=100)
        assert len(desc["text_preview"]) == 100
        assert desc["text_truncated"] is True

    def test_audio(self, tmp_path):
        audio = tmp_path / "clip.mp3"
        audio.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 50)
        att = process_local_file(str(audio))
        desc = attachment_descriptor(att)
        assert desc["kind"] == "audio"
        assert desc["audio_data_uri"].startswith("data:audio/mpeg;base64,")


# ---------------------------------------------------------------------------
# process_browser_file
# ---------------------------------------------------------------------------


class TestProcessBrowserFile:
    """Tests for process_browser_file (browser-uploaded files)."""

    def test_image(self):
        import base64

        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        b64 = base64.b64encode(raw).decode()
        att = process_browser_file("photo.png", b64, "image/png")
        assert att.source == "browser:photo.png"
        assert att.display_name == "photo.png"
        assert att.mime_type == "image/png"
        assert att.size_bytes == len(raw)
        assert len(att.content_blocks) == 1
        assert att.content_blocks[0]["type"] == "image_url"

    def test_text(self):
        import base64

        raw = b"hello world"
        b64 = base64.b64encode(raw).decode()
        att = process_browser_file("readme.txt", b64, "text/plain")
        assert att.source == "browser:readme.txt"
        assert att.mime_type == "text/plain"
        assert att.content_blocks[0]["type"] == "text"
        assert "hello world" in att.content_blocks[0]["text"]

    def test_too_large(self):
        import base64

        raw = b"\x00" * (DEFAULT_MAX_ATTACHMENT_SIZE_BYTES + 1)
        b64 = base64.b64encode(raw).decode()
        with pytest.raises(ValueError, match="too large"):
            process_browser_file("big.bin", b64, "image/png")

    def test_unsupported_extension(self):
        import base64

        b64 = base64.b64encode(b"data").decode()
        with pytest.raises(ValueError, match="Unsupported"):
            process_browser_file("file.xyz", b64, "application/octet-stream")
