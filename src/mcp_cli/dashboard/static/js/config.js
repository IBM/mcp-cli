// ================================================================
//  js/config.js — Config state (model, provider, servers, prompt)
// ================================================================
'use strict';

import {
  _configState, _availableProviders,
  setConfigState, setAvailableProviders,
} from './state.js';
import { sendToBridge } from './websocket.js';
import { esc, showToast } from './utils.js';

export function handleConfigState(payload) {
  setConfigState(payload);
  setAvailableProviders(payload.available_providers || []);
  updateProviderSelect(payload.provider);
  updateModelSelect(payload.provider, payload.model);
  updateServerList(payload.servers || []);
  updateSystemPromptEditor(payload.system_prompt || '');
  // Show model group once we have config
  document.getElementById('model-group').style.display = 'flex';
}

// ── Provider / Model selectors ────────────────────────────────────
export function updateProviderSelect(activeProvider) {
  const sel = document.getElementById('provider-select');
  const prev = sel.value;
  sel.innerHTML = '';
  for (const p of _availableProviders) {
    const opt = document.createElement('option');
    opt.value = p.name;
    opt.textContent = p.name;
    sel.appendChild(opt);
  }
  sel.value = activeProvider || prev;
}

export function updateModelSelect(provider, activeModel) {
  const sel = document.getElementById('model-select');
  sel.innerHTML = '';
  const pInfo = _availableProviders.find(p => p.name === provider);
  const models = pInfo ? pInfo.models : [];
  if (models.length === 0) {
    const opt = document.createElement('option');
    opt.value = activeModel || '';
    opt.textContent = activeModel || '(unknown)';
    sel.appendChild(opt);
  } else {
    for (const m of models) {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      sel.appendChild(opt);
    }
  }
  sel.value = activeModel || '';
  // If active model not in list, add it
  if (sel.value !== activeModel && activeModel) {
    const opt = document.createElement('option');
    opt.value = activeModel;
    opt.textContent = activeModel;
    sel.insertBefore(opt, sel.firstChild);
    sel.value = activeModel;
  }
}

export function updateServerList(servers) {
  const list = document.getElementById('server-list');
  list.innerHTML = '';
  if (!servers.length) {
    list.innerHTML = '<li style="font-size:12px;color:var(--dash-fg-muted)">No servers connected</li>';
    return;
  }
  for (const s of servers) {
    const li = document.createElement('li');
    li.className = 'server-item';
    li.innerHTML = `
      <span class="server-dot ${s.connected ? 'on' : 'off'}"></span>
      <span class="server-name" title="${esc(s.namespace || s.name)}">${esc(s.name)}</span>
      <span class="server-tools">${esc(String(s.tool_count))} tools</span>
    `;
    list.appendChild(li);
  }
}

// ── System prompt editor ──────────────────────────────────────────
export function updateSystemPromptEditor(prompt) {
  const editor = document.getElementById('system-prompt-editor');
  // Only update if not focused (avoid overwriting user edits)
  if (document.activeElement !== editor) {
    editor.value = prompt;
  }
}

// ── Wire config event listeners ───────────────────────────────────
export function wireConfigEvents() {
  document.getElementById('provider-select').addEventListener('change', (e) => {
    const provider = e.target.value;
    const pInfo = _availableProviders.find(p => p.name === provider);
    const models = pInfo ? pInfo.models : [];
    const model = models[0] || '';
    updateModelSelect(provider, model);
    if (model) sendToBridge({ type: 'SWITCH_MODEL', provider, model });
  });

  document.getElementById('model-select').addEventListener('change', (e) => {
    const model = e.target.value;
    const provider = document.getElementById('provider-select').value;
    if (provider && model) sendToBridge({ type: 'SWITCH_MODEL', provider, model });
  });

  document.getElementById('apply-prompt-btn').addEventListener('click', () => {
    const text = document.getElementById('system-prompt-editor').value;
    sendToBridge({ type: 'UPDATE_SYSTEM_PROMPT', system_prompt: text });
    showToast('info', 'System prompt updated');
  });

  document.getElementById('reset-prompt-btn').addEventListener('click', () => {
    sendToBridge({ type: 'UPDATE_SYSTEM_PROMPT', system_prompt: '' });
    showToast('info', 'System prompt reset to default');
  });
}
