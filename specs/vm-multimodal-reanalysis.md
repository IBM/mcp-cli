# Spec: VM Multimodal Content Re-analysis

## Problem

When the VM system faults a multimodal page (image, audio, video, structured data), `_handle_vm_tool` serializes the content as a plain JSON string. Multimodal models receive a text description instead of the actual media, so they can't re-analyze images, re-process structured data, or inspect code with language awareness.

The upstream session manager (`fault_handler._format_content_for_modality()`) already returns typed content objects (`ImageContent`, `AudioContent`, `VideoContent`, `StructuredContent`) — but mcp-cli throws away the structure during serialization.

## Current State

**Session manager (already built):**

| Component | What It Does |
|-----------|-------------|
| `_format_content_for_modality()` | Returns modality-specific content objects |
| `ImageContent` model | Fields: `caption`, `url`, `base64`, `embedding` |
| `AudioContent` model | Fields: `transcript`, `timestamps`, `duration_seconds` |
| `VideoContent` model | Fields: `scenes`, `transcript`, `duration_seconds` |
| `StructuredContent` model | Fields: `data` (dict), `schema_name` |
| `MemoryPage` modality fields | `modality`, `mime_type`, `caption`, `dimensions`, `duration_seconds` |
| Compression pipeline | Per-modality: FULL → REDUCED → ABSTRACT → REFERENCE |

**mcp-cli (gap):**

| Gap | Impact |
|-----|--------|
| `_handle_vm_tool` returns string-only JSON | Multimodal models can't re-analyze images |
| `HistoryMessage.content` is `str | None` | Can't represent multi-block tool results |
| `/memory page` shows text only | No way to inspect/download multimodal content |
| No compression level in tool result | Model doesn't know if it's seeing summary or original |

## Design

### Multi-block Tool Results

OpenAI API supports multi-part content in tool results:

```json
{
  "role": "tool",
  "tool_call_id": "call_abc",
  "content": [
    {"type": "text", "text": "Page img_001 (image, FULL):"},
    {"type": "image_url", "image_url": {"url": "https://...", "detail": "low"}}
  ]
}
```

### Changes

#### 1. Extend `HistoryMessage.content` type

**File:** `src/mcp_cli/chat/models.py`

Change:
```python
content: str | None = None
```
To:
```python
content: str | list[dict[str, Any]] | None = None
```

Update `to_dict()` to preserve list content as-is (no stringification).

#### 2. Content routing in `_handle_vm_tool`

**File:** `src/mcp_cli/chat/tool_processor.py`

Add `_build_page_content_blocks()` method that checks `page.modality`:

| Modality | Behaviour |
|----------|-----------|
| **TEXT** | Current behaviour — return `{"type": "text", "text": content}` |
| **IMAGE** | If `content` starts with `http` → `[text_block, image_url_block]`. If starts with `data:` → `[text_block, image_url_block]`. Otherwise → text-only with caption. |
| **AUDIO** | Text block with transcript + duration metadata |
| **VIDEO** | Text block with transcript + scene summaries |
| **STRUCTURED** | Text block with JSON-formatted data + schema hint |

For IMAGE with URL/data URI, the tool result becomes a list:
```python
[
    {"type": "text", "text": f"Page {page_id} ({modality}, {compression}):"},
    {"type": "image_url", "image_url": {"url": url, "detail": "low"}},
]
```

For all other cases (including IMAGE with caption-only), return a single JSON string as today — no multi-block needed.

**Important:** Only use multi-block format when actual media data is available (URL or base64). If the page has been compressed to ABSTRACT/REFERENCE level (caption only), use text format.

#### 3. Compression level annotation

Include compression level in the text block so the model knows what it's seeing:

```
Page img_001 (image, ABSTRACT — caption only):
  "Sunset landscape with tree, path, sheep, pond, church spire"

  Note: This is a compressed summary. Use page_fault("img_001", target_level=0) for full content.
```

vs.

```
Page img_001 (image, FULL):
  [image attached below]
```

#### 4. `/memory page <id> --download`

**File:** `src/mcp_cli/commands/memory/memory.py`

Add `--download` flag to `_show_page_detail()`:

- Parse `--download` from the action string (e.g., `/memory page img_001 --download`)
- For pages with URL content: download to `~/.mcp-cli/downloads/<page_id>.<ext>`
- For pages with base64 data URI: decode and save to file
- For text/structured: save as `.txt` or `.json`
- Report the saved file path

Extension detection from `mime_type` field on the page, with fallback to `.bin`.

#### 5. Enhanced modality display in `/memory page`

Show modality-specific metadata in the detail view:

```
Page ID:     img_001
Modality:    image
MIME Type:   image/png
Dimensions:  1920x1080
Compression: ABSTRACT
Content:     [caption] Sunset landscape with tree...
URL:         https://example.com/sunset.png
```

## Files Modified

| File | Change |
|------|--------|
| `src/mcp_cli/chat/models.py` | `content: str | list[dict] | None` in HistoryMessage |
| `src/mcp_cli/chat/tool_processor.py` | `_build_page_content_blocks()`, update `_handle_vm_tool` |
| `src/mcp_cli/commands/memory/memory.py` | `--download` flag, modality-aware display |

## What We're NOT Doing

- **No binary storage in VM** — pages continue to store URLs/data URIs/captions, not raw bytes
- **No image preprocessing pipeline** — if the original page only has a caption, we can't reconstruct the image
- **No provider detection** — we don't check if the model supports vision; multi-block content is valid OpenAI format regardless
- **No audio/video playback** — download only, no inline rendering
- **No upstream changes** — session manager already returns the right types

## Scope

This is a focused change: use the structured content the session manager already provides instead of flattening it to a string. The heavy lifting (modality detection, compression, content objects) is already done upstream.

## Test Plan

- Unit test: `_build_page_content_blocks()` for each modality
- Unit test: `HistoryMessage` with list content serializes correctly
- Unit test: `--download` command with mock page store
- Update e2e demo: verify image scenario returns multi-block content
