// ================================================================
//  js/sessions.js — Session list management
// ================================================================
'use strict';

import {
  _sessionList, _currentSessionId,
  setSessionList, setCurrentSessionId,
} from './state.js';
import { sendToBridge } from './websocket.js';
import { showToast } from './utils.js';

export function handleSessionList(payload) {
  setSessionList(payload.sessions || []);
  const currentId = payload.current_session_id || _currentSessionId;
  renderSessionList(_sessionList, currentId);
}

export function handleSessionState(payload) {
  setCurrentSessionId(payload.session_id || null);
  // Could update a session indicator in toolbar later
}

export function renderSessionList(sessions, currentId) {
  const ul = document.getElementById('session-list');
  if (!sessions.length) {
    ul.innerHTML = '<li class="session-empty">No saved sessions</li>';
    return;
  }
  ul.innerHTML = '';
  for (const s of sessions) {
    const li = document.createElement('li');
    li.className = 'session-item' + (s.session_id === currentId ? ' active' : '');

    const info = document.createElement('div');
    info.className = 'session-info';

    const idEl = document.createElement('div');
    idEl.className = 'session-id';
    idEl.textContent = s.description || s.session_id.slice(0, 12);
    info.appendChild(idEl);

    const meta = document.createElement('div');
    meta.className = 'session-meta';
    const msgs = document.createElement('span');
    msgs.textContent = s.message_count + ' msgs';
    meta.appendChild(msgs);
    if (s.model) {
      const model = document.createElement('span');
      model.textContent = s.model;
      meta.appendChild(model);
    }
    const time = document.createElement('span');
    time.textContent = formatSessionTime(s.updated_at);
    meta.appendChild(time);
    info.appendChild(meta);

    li.appendChild(info);

    const actions = document.createElement('div');
    actions.className = 'session-actions';

    if (s.session_id !== currentId) {
      const loadBtn = document.createElement('button');
      loadBtn.textContent = 'Load';
      loadBtn.title = 'Switch to this session';
      loadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        sendToBridge({ type: 'SWITCH_SESSION', session_id: s.session_id });
        showToast('info', 'Switching session...');
      });
      actions.appendChild(loadBtn);

      const delBtn = document.createElement('button');
      delBtn.textContent = 'Del';
      delBtn.className = 'delete';
      delBtn.title = 'Delete this session';
      delBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (confirm('Delete this session permanently?')) {
          sendToBridge({ type: 'DELETE_SESSION', session_id: s.session_id });
        }
      });
      actions.appendChild(delBtn);
    } else {
      const cur = document.createElement('span');
      cur.textContent = 'current';
      cur.style.cssText = 'font-size:10px;color:var(--dash-accent)';
      actions.appendChild(cur);
    }

    li.appendChild(actions);
    ul.appendChild(li);
  }
}

export function formatSessionTime(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    const now = new Date();
    const diff = now - d;
    if (diff < 60000) return 'just now';
    if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
    if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
    return d.toLocaleDateString();
  } catch { return ''; }
}
