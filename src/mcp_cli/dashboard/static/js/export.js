// ================================================================
//  js/export.js — Export conversation
// ================================================================
'use strict';

import { viewPool } from './state.js';
import { showToast } from './utils.js';

export function exportConversation(format) {
  // Collect messages from the agent-terminal iframe
  const msgs = collectConversationMessages();
  if (!msgs.length) { showToast('warning', 'No messages to export'); return; }

  let content, ext, mime;
  if (format === 'json') {
    content = JSON.stringify({ exported: new Date().toISOString(), messages: msgs }, null, 2);
    ext = 'json'; mime = 'application/json';
  } else {
    const lines = [`# Conversation Export\n_${new Date().toISOString()}_\n`];
    for (const m of msgs) {
      const label = m.role === 'user' ? '**You**' : m.role === 'assistant' ? '**Agent**' : `**${m.role}**`;
      lines.push(`### ${label}\n${m.content}\n`);
    }
    content = lines.join('\n');
    ext = 'md'; mime = 'text/markdown';
  }

  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `conversation-${Date.now()}.${ext}`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('success', `Exported as ${ext.toUpperCase()}`);
}

export function collectConversationMessages() {
  // Find the agent-terminal iframe and read its messages
  for (const [viewId, view] of viewPool) {
    if (viewId !== 'builtin:agent-terminal' || !view.iframe) continue;
    try {
      const doc = view.iframe.contentDocument || view.iframe.contentWindow.document;
      const msgEls = doc.querySelectorAll('.msg');
      const messages = [];
      for (const el of msgEls) {
        const role = el.classList.contains('user') ? 'user' :
                     el.classList.contains('tool-call') ? 'tool' : 'assistant';
        const contentEl = el.querySelector('.msg-content');
        const toolNameEl = el.querySelector('.tool-name');
        let content = '';
        if (contentEl) content = contentEl.textContent || '';
        else if (toolNameEl) content = `[Tool: ${toolNameEl.textContent}]`;
        if (content) messages.push({ role, content });
      }
      return messages;
    } catch (e) {
      // Cross-origin iframe — cannot read messages directly
      showToast('warning', 'Cannot export: cross-origin view');
      return [];
    }
  }
  return [];
}
