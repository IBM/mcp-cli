// ================================================================
//  js/toolbar.js — Toolbar button wiring + overflow menu
// ================================================================
'use strict';

import { layoutConfig, _configState } from './state.js';
import { sendToBridge } from './websocket.js';
import { addPanelWithView, rebuildAddPanelMenu } from './layout.js';
import { openDrawer, closeDrawer, closeAllDrawers, showToast } from './utils.js';
import { exportConversation } from './export.js';

export function buildOverflowMenu() {
  const menu = document.getElementById('overflow-menu');
  menu.innerHTML = '';
  const isMobile = window.innerWidth < 768;
  if (!isMobile) {
    menu.classList.remove('open');
    document.getElementById('overflow-btn').style.display = 'none';
    return;
  }

  const items = [];
  // Items hidden at sm (480-767)
  items.push({ label: '+ Add Panel', action: () => {
    rebuildAddPanelMenu();
    const m = document.getElementById('add-panel-menu');
    m.style.position = 'fixed'; m.style.right = '8px'; m.style.top = '48px'; m.style.left = 'auto';
    m.classList.toggle('open');
  }});
  items.push({ label: 'Export as Markdown', action: () => exportConversation('markdown') });
  items.push({ label: 'Export as JSON', action: () => exportConversation('json') });

  if (_configState) {
    const modelLabel = _configState.model || '(none)';
    items.push({ label: 'Model: ' + modelLabel, action: () => {
      openDrawer(document.getElementById('settings-panel'));
    }});
  }

  // Items also hidden at xs (<480)
  if (window.innerWidth < 480) {
    items.unshift({ label: '+ New Session', action: () => document.getElementById('new-session-btn').click() });
    items.unshift({ label: 'Sessions', action: () => {
      openDrawer(document.getElementById('session-drawer'));
      sendToBridge({ type: 'REQUEST_SESSIONS' });
    }});
  }

  for (const item of items) {
    const el = document.createElement('div');
    el.className = 'dropdown-item';
    el.textContent = item.label;
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      menu.classList.remove('open');
      item.action();
    });
    menu.appendChild(el);
  }
}

export function wireToolbarEvents() {
  document.getElementById('layout-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    const menu = document.getElementById('layout-menu');
    menu.classList.toggle('open');
  });

  document.getElementById('add-panel-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    rebuildAddPanelMenu();
    const menu = document.getElementById('add-panel-menu');
    const btn  = document.getElementById('add-panel-btn');
    const rect = btn.getBoundingClientRect();
    menu.style.left = rect.left + 'px';
    menu.style.top  = (rect.bottom + 4) + 'px';
    menu.style.position = 'fixed';
    menu.classList.toggle('open');
  });

  document.getElementById('settings-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    const panel = document.getElementById('settings-panel');
    if (panel.classList.contains('open')) { closeDrawer(panel); } else { openDrawer(panel); }
  });

  document.getElementById('save-layout-btn').addEventListener('click', () => {
    try {
      localStorage.setItem('dash-layout', JSON.stringify(layoutConfig));
      showToast('success', 'Layout saved');
    } catch (e) {
      showToast('error', 'Could not save layout');
    }
  });

  document.getElementById('clear-history-btn').addEventListener('click', () => {
    if (!confirm('Clear all chat history and start a new session?')) return;
    sendToBridge({ type: 'CLEAR_HISTORY' });
    showToast('info', 'Chat history cleared');
  });

  document.getElementById('new-session-btn').addEventListener('click', () => {
    if (!confirm('Save current session and start a new one?')) return;
    sendToBridge({ type: 'NEW_SESSION' });
    showToast('success', 'New session started');
    // Close session drawer if open
    document.getElementById('session-drawer').classList.remove('open');
  });

  document.getElementById('sessions-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    const drawer = document.getElementById('session-drawer');
    if (drawer.classList.contains('open')) {
      closeDrawer(drawer);
    } else {
      openDrawer(drawer);
      sendToBridge({ type: 'REQUEST_SESSIONS' });
    }
  });

  document.getElementById('refresh-sessions-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    sendToBridge({ type: 'REQUEST_SESSIONS' });
  });

  // Export button + menu
  document.getElementById('export-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    document.getElementById('export-menu').classList.toggle('open');
  });

  document.getElementById('export-menu').addEventListener('click', (e) => {
    const item = e.target.closest('[data-format]');
    if (!item) return;
    e.stopPropagation();
    document.getElementById('export-menu').classList.remove('open');
    exportConversation(item.dataset.format);
  });

  // Close dropdowns/menus on outside click (but NOT settings panel when clicking inside it)
  document.addEventListener('click', (e) => {
    document.getElementById('layout-menu').classList.remove('open');
    document.getElementById('add-panel-menu').classList.remove('open');
    document.getElementById('export-menu').classList.remove('open');
    document.getElementById('overflow-menu').classList.remove('open');
    document.querySelectorAll('.panel-view-menu.open').forEach(m => m.classList.remove('open'));
    // Only close settings panel if click was outside both the panel and the settings button
    const settingsPanel = document.getElementById('settings-panel');
    const settingsBtn   = document.getElementById('settings-btn');
    if (!settingsPanel.contains(e.target) && e.target !== settingsBtn) {
      closeDrawer(settingsPanel);
    }
    // Only close session drawer if click was outside both the drawer and the sessions button
    const sessionDrawer = document.getElementById('session-drawer');
    const sessionsBtn   = document.getElementById('sessions-btn');
    if (!sessionDrawer.contains(e.target) && e.target !== sessionsBtn) {
      closeDrawer(sessionDrawer);
    }
  });

  // Overflow menu button
  document.getElementById('overflow-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    buildOverflowMenu();
    const menu = document.getElementById('overflow-menu');
    const rect = e.target.getBoundingClientRect();
    menu.style.top = (rect.bottom + 4) + 'px';
    menu.style.right = '8px';
    menu.style.left = 'auto';
    menu.classList.toggle('open');
  });

  // Drawer backdrop
  document.getElementById('drawer-backdrop').addEventListener('click', closeAllDrawers);
}
