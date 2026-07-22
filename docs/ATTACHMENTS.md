# Multi-Modal Attachments

MCP CLI supports attaching images, text files, and audio to messages. Attachments are converted to content blocks that multimodal LLMs can process (vision, text analysis, audio understanding).

## Quick Start

```bash
# Attach files to the first message via CLI flag
mcp-cli --server sqlite --attach photo.png --attach data.csv

# In chat, use the /attach command
/attach screenshot.png
/attach code.py
Tell me what you see and review the code

# Or use inline @file: references
@file:image.png describe what's in this image
```

## Three Ways to Attach

### 1. `/attach` Command (Chat Mode)

Stage files before sending a message:

```bash
/attach photo.png              # Stage an image
/attach src/main.py            # Stage a code file
/attach recording.mp3          # Stage an audio file
```

Aliases: `/file`, `/image`

**Manage staging:**
```bash
/attach list                   # Show currently staged files
/attach clear                  # Clear all staged files
```

Staged files are sent with your next message and automatically cleared.

### 2. `--attach` CLI Flag

Attach files to the first message when starting chat:

```bash
mcp-cli --server sqlite --attach image.png
mcp-cli --server sqlite --attach img.png --attach data.csv --attach code.py
```

The flag is repeatable — use it multiple times for multiple files.

### 3. Inline `@file:` References

Reference files directly in any message:

```bash
@file:screenshot.png describe what you see
@file:report.txt @file:data.csv compare these two files
Look at @file:diagram.png and explain the architecture
```

The `@file:` prefix is removed from the message text before sending.

### Image URL Detection

HTTP/HTTPS image URLs in messages are automatically detected and sent as vision content:

```
https://example.com/chart.png what does this chart show?
```

Supported URL patterns: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`

## Supported File Types

### Images
| Extension | MIME Type |
|-----------|-----------|
| `.png` | `image/png` |
| `.jpg`, `.jpeg` | `image/jpeg` |
| `.gif` | `image/gif` |
| `.webp` | `image/webp` |
| `.heic` | `image/heic` |

Images are base64-encoded and sent as `image_url` content blocks with configurable detail level.

### Audio
| Extension | MIME Type |
|-----------|-----------|
| `.mp3` | `audio/mpeg` |
| `.wav` | `audio/wav` |

Audio is base64-encoded and sent as `input_audio` content blocks.

### Text & Code
| Extension | MIME Type |
|-----------|-----------|
| `.txt` | `text/plain` |
| `.md` | `text/markdown` |
| `.csv` | `text/csv` |
| `.json` | `application/json` |
| `.html` | `text/html` |
| `.xml` | `text/xml` |
| `.yaml`, `.yml` | `text/yaml` |
| `.py` | `text/plain` |
| `.js`, `.jsx` | `text/plain` |
| `.ts`, `.tsx` | `text/plain` |
| `.sh`, `.bash` | `text/plain` |
| `.rs` | `text/plain` |
| `.go` | `text/plain` |
| `.java` | `text/plain` |
| `.c`, `.cpp` | `text/plain` |
| `.h`, `.hpp` | `text/plain` |
| `.rb` | `text/plain` |
| `.swift` | `text/plain` |
| `.kt` | `text/plain` |
| `.sql` | `text/plain` |
| `.toml` | `text/plain` |
| `.ini`, `.cfg` | `text/plain` |
| `.env` | `text/plain` |
| `.log` | `text/plain` |

Text files are read as UTF-8 (with Latin-1 fallback) and wrapped in labeled text blocks.

## Size Limits

| Limit | Value |
|-------|-------|
| Maximum file size | 20 MB |
| Maximum attachments per message | 10 |

These defaults are configured in `src/mcp_cli/config/defaults.py`.

## Browser Upload (Dashboard)

When using `--dashboard`, the agent terminal provides browser-based file attachment:

### "+" Button
Click the "+" button next to the chat input to open a file picker. Select one or more files to stage them.

### Drag and Drop
Drag files from your file manager onto the chat area. A drop overlay appears to confirm the target.

### Clipboard Paste
Paste images directly into the chat input (Ctrl/Cmd+V). Screenshots and copied images are automatically staged.

### Staging Strip
Staged files appear as removable badges above the chat input:
- Image files show a small thumbnail preview
- All files show the filename with a "x" remove button
- Click "x" to remove a file before sending

Files are sent when you press Enter or click Send. The staging strip clears automatically.

## Dashboard Rendering

When messages with attachments appear in the dashboard, they render as:

- **Small images (<100KB)**: Inline thumbnail previews (max 200x150px)
- **Large images (>100KB)**: Metadata badge showing filename and size
- **URL images**: Thumbnail loaded from the URL
- **Text files**: Expandable preview showing the first 2000 characters
- **Audio files**: HTML5 `<audio>` player with playback controls

The activity stream shows attachment events as badge cards with a paperclip icon, filenames, and total size.

## How It Works

### Content Block Construction

Each file type produces specific OpenAI-compatible content blocks:

- **Images** → `{"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "auto"}}`
- **Audio** → `{"type": "input_audio", "input_audio": {"data": "...", "format": "mp3"}}`
- **Text** → `{"type": "text", "text": "--- filename ---\n...\n--- end filename ---"}`

When attachments are present, the user message `content` field becomes a list of content blocks (multimodal format) instead of a plain string.

### Attachment Staging

The `AttachmentStaging` class on `ChatContext` manages the staging lifecycle:

1. Files are staged via `/attach`, `--attach`, or browser upload
2. The chat loop calls `drain()` to collect and clear staged files
3. Combined with inline `@file:` refs and detected image URLs
4. `build_multimodal_content()` assembles the final content block list
5. If no attachments exist, the message stays as a plain string (backward compatible)

### Dashboard Descriptors

To avoid sending large base64 payloads over WebSocket, the dashboard uses lightweight **attachment descriptors**:

- `display_name`, `size_bytes`, `mime_type`, `kind` (image/text/audio/unknown)
- `preview_url`: data URI for small images, HTTP URL for URL images, `None` for large files
- `text_preview`: first 2000 characters for text files
- `audio_data_uri`: data URI for small audio files

These thresholds are configured in `src/mcp_cli/config/defaults.py`:
- `DEFAULT_DASHBOARD_INLINE_IMAGE_THRESHOLD`: 100 KB
- `DEFAULT_DASHBOARD_TEXT_PREVIEW_CHARS`: 2000
