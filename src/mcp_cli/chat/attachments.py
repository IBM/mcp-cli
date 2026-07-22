# mcp_cli/chat/attachments.py
"""Multi-modal attachment processing.

Handles file detection, MIME typing, base64 encoding, content block
construction, and staging for the ``/attach`` command.  This is a
**core module** — logging only, no ``chuk_term`` imports.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_cli.config.defaults import (
    DEFAULT_DASHBOARD_INLINE_IMAGE_THRESHOLD,
    DEFAULT_DASHBOARD_TEXT_PREVIEW_CHARS,
    DEFAULT_IMAGE_DETAIL_LEVEL,
    DEFAULT_MAX_ATTACHMENT_SIZE_BYTES,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  MIME / extension maps                                               #
# ------------------------------------------------------------------ #

MIME_MAP: dict[str, str] = {
    # Images
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".heic": "image/heic",
    # Audio
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    # Text / code
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".html": "text/html",
    ".xml": "text/xml",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    # Programming languages
    ".py": "text/plain",
    ".js": "text/plain",
    ".ts": "text/plain",
    ".jsx": "text/plain",
    ".tsx": "text/plain",
    ".sh": "text/plain",
    ".bash": "text/plain",
    ".rs": "text/plain",
    ".go": "text/plain",
    ".java": "text/plain",
    ".c": "text/plain",
    ".cpp": "text/plain",
    ".h": "text/plain",
    ".hpp": "text/plain",
    ".rb": "text/plain",
    ".swift": "text/plain",
    ".kt": "text/plain",
    ".sql": "text/plain",
    ".toml": "text/plain",
    ".ini": "text/plain",
    ".cfg": "text/plain",
    ".env": "text/plain",
    ".log": "text/plain",
}

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic"})
AUDIO_EXTENSIONS = frozenset({".mp3", ".wav"})
TEXT_EXTENSIONS = frozenset(MIME_MAP.keys()) - IMAGE_EXTENSIONS - AUDIO_EXTENSIONS

# Audio format mapping (extension → OpenAI input_audio format value)
_AUDIO_FORMAT: dict[str, str] = {".mp3": "mp3", ".wav": "wav"}


# ------------------------------------------------------------------ #
#  Data types                                                          #
# ------------------------------------------------------------------ #


@dataclass
class Attachment:
    """A single processed attachment ready for content-block injection."""

    source: str
    content_blocks: list[dict[str, Any]]
    display_name: str
    size_bytes: int
    mime_type: str


class AttachmentStaging:
    """Staging area for ``/attach`` command.  Lives on ChatContext."""

    def __init__(self) -> None:
        self._pending: list[Attachment] = []

    def stage(self, attachment: Attachment) -> None:
        """Add an attachment to the staging area."""
        self._pending.append(attachment)

    def drain(self) -> list[Attachment]:
        """Return all pending attachments and clear the staging area."""
        items = list(self._pending)
        self._pending.clear()
        return items

    def peek(self) -> list[Attachment]:
        """Return pending attachments without clearing."""
        return list(self._pending)

    def clear(self) -> None:
        """Clear the staging area."""
        self._pending.clear()

    @property
    def count(self) -> int:
        return len(self._pending)


# ------------------------------------------------------------------ #
#  MIME detection                                                      #
# ------------------------------------------------------------------ #


def detect_mime_type(path: Path) -> str:
    """Detect MIME type from file extension."""
    return MIME_MAP.get(path.suffix.lower(), "application/octet-stream")


# ------------------------------------------------------------------ #
#  File processing                                                     #
# ------------------------------------------------------------------ #


def process_local_file(path: str | Path) -> Attachment:
    """Read a local file and build content blocks.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is too large or has an unsupported extension.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise ValueError(f"Not a file: {path}")

    size = p.stat().st_size
    if size > DEFAULT_MAX_ATTACHMENT_SIZE_BYTES:
        raise ValueError(
            f"File too large: {size:,} bytes "
            f"(max {DEFAULT_MAX_ATTACHMENT_SIZE_BYTES:,})"
        )

    ext = p.suffix.lower()
    mime = detect_mime_type(p)

    if ext in IMAGE_EXTENSIONS:
        blocks = _build_image_blocks(p, mime)
    elif ext in AUDIO_EXTENSIONS:
        blocks = _build_audio_blocks(p, ext)
    elif ext in TEXT_EXTENSIONS:
        blocks = _build_text_blocks(p)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: images ({', '.join(sorted(IMAGE_EXTENSIONS))}), "
            f"audio ({', '.join(sorted(AUDIO_EXTENSIONS))}), "
            f"text ({', '.join(sorted(TEXT_EXTENSIONS))})"
        )

    logger.debug("Processed attachment: %s (%s, %d bytes)", p.name, mime, size)
    return Attachment(
        source=str(path),
        content_blocks=blocks,
        display_name=p.name,
        size_bytes=size,
        mime_type=mime,
    )


def _image_blocks_from_b64(b64: str, mime: str) -> list[dict[str, Any]]:
    """Build image content blocks from base64-encoded data."""
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}",
                "detail": DEFAULT_IMAGE_DETAIL_LEVEL,
            },
        }
    ]


def _audio_blocks_from_b64(b64: str, ext: str) -> list[dict[str, Any]]:
    """Build audio content blocks from base64-encoded data."""
    fmt = _AUDIO_FORMAT.get(ext, "mp3")
    return [{"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}]


def _text_blocks_from_string(text: str, label: str) -> list[dict[str, Any]]:
    """Build text content blocks from a string."""
    return [
        {
            "type": "text",
            "text": f"--- {label} ---\n{text}\n--- end {label} ---",
        }
    ]


def _build_image_blocks(path: Path, mime: str) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return _image_blocks_from_b64(b64, mime)


def _build_audio_blocks(path: Path, ext: str) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return _audio_blocks_from_b64(b64, ext)


def _build_text_blocks(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    return _text_blocks_from_string(text, path.name)


# ------------------------------------------------------------------ #
#  Browser file processing                                             #
# ------------------------------------------------------------------ #


def process_browser_file(
    filename: str,
    data_b64: str,
    mime_type: str,
) -> Attachment:
    """Build an ``Attachment`` from browser-uploaded file data.

    Parameters
    ----------
    filename:
        Original filename from the browser ``File`` object.
    data_b64:
        Base64-encoded file contents.
    mime_type:
        MIME type reported by the browser (e.g. ``image/png``).

    Raises
    ------
    ValueError
        If the file is too large or has an unsupported extension.
    """
    raw = base64.b64decode(data_b64)
    size = len(raw)
    if size > DEFAULT_MAX_ATTACHMENT_SIZE_BYTES:
        raise ValueError(
            f"File too large: {size:,} bytes "
            f"(max {DEFAULT_MAX_ATTACHMENT_SIZE_BYTES:,})"
        )

    ext = Path(filename).suffix.lower()
    # Fall back to provided mime_type if extension not in our map
    mime = MIME_MAP.get(ext, mime_type)

    if ext in IMAGE_EXTENSIONS:
        b64 = base64.b64encode(raw).decode("ascii")
        blocks = _image_blocks_from_b64(b64, mime)
    elif ext in AUDIO_EXTENSIONS:
        b64 = base64.b64encode(raw).decode("ascii")
        blocks = _audio_blocks_from_b64(b64, ext)
    elif ext in TEXT_EXTENSIONS:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        blocks = _text_blocks_from_string(text, filename)
    else:
        raise ValueError(
            f"Unsupported file type: {ext or '(none)'}. "
            f"Supported: images ({', '.join(sorted(IMAGE_EXTENSIONS))}), "
            f"audio ({', '.join(sorted(AUDIO_EXTENSIONS))}), "
            f"text ({', '.join(sorted(TEXT_EXTENSIONS))})"
        )

    logger.debug(
        "Processed browser attachment: %s (%s, %d bytes)", filename, mime, size
    )
    return Attachment(
        source=f"browser:{filename}",
        content_blocks=blocks,
        display_name=filename,
        size_bytes=size,
        mime_type=mime,
    )


# ------------------------------------------------------------------ #
#  URL processing                                                      #
# ------------------------------------------------------------------ #


def process_url(url: str) -> Attachment:
    """Build an image_url content block from a URL (no download)."""
    return Attachment(
        source=url,
        content_blocks=[
            {
                "type": "image_url",
                "image_url": {"url": url, "detail": DEFAULT_IMAGE_DETAIL_LEVEL},
            }
        ],
        display_name=url.rsplit("/", 1)[-1][:60],
        size_bytes=0,
        mime_type="image/unknown",
    )


# ------------------------------------------------------------------ #
#  Inline @file:path parsing                                           #
# ------------------------------------------------------------------ #

_INLINE_REF_RE = re.compile(r"@file:(\S+)")


def parse_inline_refs(text: str) -> tuple[str, list[str]]:
    """Extract ``@file:path`` references from message text.

    Returns
    -------
    (cleaned_text, list_of_paths)
        *cleaned_text* has the ``@file:...`` tokens removed.
    """
    paths = _INLINE_REF_RE.findall(text)
    if not paths:
        return text, []
    cleaned = _INLINE_REF_RE.sub("", text).strip()
    # Collapse multiple spaces left by removal
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned, paths


# ------------------------------------------------------------------ #
#  Image URL detection                                                 #
# ------------------------------------------------------------------ #

_IMAGE_URL_RE = re.compile(
    r"(https?://\S+\.(?:png|jpg|jpeg|gif|webp)(?:\?\S*)?)",
    re.IGNORECASE,
)


def detect_image_urls(text: str) -> list[str]:
    """Find image URLs in message text."""
    return _IMAGE_URL_RE.findall(text)


# ------------------------------------------------------------------ #
#  Multi-modal content builder                                         #
# ------------------------------------------------------------------ #


def build_multimodal_content(
    user_text: str,
    attachments: list[Attachment],
    image_urls: list[str],
) -> str | list[dict[str, Any]]:
    """Build the ``content`` field for a user message.

    Returns the plain string when there are no attachments or image URLs
    (backward compatible).  Otherwise returns a list of content blocks.
    """
    if not attachments and not image_urls:
        return user_text

    blocks: list[dict[str, Any]] = []

    # Text always first
    if user_text:
        blocks.append({"type": "text", "text": user_text})

    # Staged / inline attachments
    for att in attachments:
        blocks.extend(att.content_blocks)

    # Auto-detected image URLs (deduplicate against attachment URLs)
    seen_urls: set[str] = set()
    for att in attachments:
        for blk in att.content_blocks:
            if blk.get("type") == "image_url":
                seen_urls.add(blk.get("image_url", {}).get("url", ""))
    for url in image_urls:
        if url not in seen_urls:
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": DEFAULT_IMAGE_DETAIL_LEVEL},
                }
            )

    return blocks


# ------------------------------------------------------------------ #
#  Dashboard attachment descriptors                                    #
# ------------------------------------------------------------------ #


def _classify_kind(mime_type: str) -> str:
    """Map MIME type to a UI kind: image, text, audio, or unknown."""
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("audio/"):
        return "audio"
    if mime_type.startswith("text/") or mime_type in ("application/json",):
        return "text"
    return "unknown"


def attachment_descriptor(
    att: Attachment,
    inline_threshold: int = DEFAULT_DASHBOARD_INLINE_IMAGE_THRESHOLD,
    text_preview_chars: int = DEFAULT_DASHBOARD_TEXT_PREVIEW_CHARS,
) -> dict[str, Any]:
    """Build a dashboard-safe descriptor for an Attachment.

    Keeps payloads small by applying thresholds:
    - Images < *inline_threshold* bytes: include data URI preview
    - URL images: pass through the URL
    - Text files: include first *text_preview_chars* chars
    - Audio < *inline_threshold* bytes: include data URI
    """
    desc: dict[str, Any] = {
        "display_name": att.display_name,
        "size_bytes": att.size_bytes,
        "mime_type": att.mime_type,
        "kind": _classify_kind(att.mime_type),
    }

    for block in att.content_blocks:
        btype = block.get("type")

        if btype == "image_url":
            url = block["image_url"]["url"]
            if url.startswith("http"):
                desc["preview_url"] = url
            elif att.size_bytes <= inline_threshold:
                desc["preview_url"] = url
            else:
                desc["preview_url"] = None

        elif btype == "text":
            text = block.get("text", "")
            desc["text_preview"] = text[:text_preview_chars]
            desc["text_truncated"] = len(text) > text_preview_chars

        elif btype == "input_audio":
            if att.size_bytes <= inline_threshold:
                audio_data = block["input_audio"]["data"]
                audio_fmt = block["input_audio"]["format"]
                mime = "audio/mpeg" if audio_fmt == "mp3" else f"audio/{audio_fmt}"
                desc["audio_data_uri"] = f"data:{mime};base64,{audio_data}"
            else:
                desc["audio_data_uri"] = None

    return desc
