// ================================================================
//  js/views.js — View pool, iframe management, message routing
// ================================================================
'use strict';

import {
  viewRegistry, viewPool, panels, popoutWindows,
  PROTOCOL, VERSION, READY_TIMEOUT_MS,
  focusedAgentId, themes, activeTheme,
  _cachedToolRegistry, _cachedPlanUpdate, _configState,
  setViewRegistry, isSidebarView, _sidebarOpen,
} from './state.js';
import { showToast } from './utils.js';
import { sendToBridge } from './websocket.js';
import { themeToCSS } from './theme.js';

// ── Late-binding imports to avoid circular deps with layout.js/sidebar.js ──
let _syncViewPositions = null;
let _showPanelError = null;
let _notifyResize = null;
let _findPopoutViewIdByWindow = null;
let _handlePopoutReady = null;
let _postToPopout = null;
let _notifySidebarUpdate = null;

export function setViewDeps(deps) {
  _syncViewPositions = deps.syncViewPositions;
  _showPanelError = deps.showPanelError;
  _notifyResize = deps.notifyResize;
  _findPopoutViewIdByWindow = deps.findPopoutViewIdByWindow;
  _handlePopoutReady = deps.handlePopoutReady;
  _postToPopout = deps.postToPopout;
  _notifySidebarUpdate = deps.notifySidebarUpdate;
}

// ── View registry ─────────────────────────────────────────────────
export function mergeViewRegistry(dynamicViews) {
  // Keep builtins, add/update dynamic views
  const builtinIds = new Set(viewRegistry.filter(v => v.source === 'builtin').map(v => v.id));
  const merged = viewRegistry.filter(v => v.source === 'builtin');
  for (const v of dynamicViews) {
    if (!builtinIds.has(v.id)) merged.push(v);
  }
  setViewRegistry(merged);
}

export function srcForView(viewId) {
  const vInfo = viewRegistry.find(v => v.id === viewId);
  if (vInfo && vInfo.url) return vInfo.url;
  if (viewId === 'builtin:agent-terminal') return '/views/agent-terminal.html';
  if (viewId === 'builtin:activity-stream') return '/views/activity-stream.html';
  if (viewId === 'builtin:tool-browser')    return '/views/tool-browser.html';
  if (viewId === 'builtin:plan-viewer')     return '/views/plan-viewer.html';
  return '';
}

export function iconForView(viewId) {
  if (!viewId) return '□';
  if (viewId === 'builtin:agent-terminal') return '⌨';
  if (viewId === 'builtin:activity-stream') return '◈';
  if (viewId === 'builtin:tool-browser')    return '🔧';
  if (viewId === 'builtin:plan-viewer')     return '📋';
  if (viewId === 'builtin:agent-overview')  return '👥';
  const v = viewRegistry.find(v => v.id === viewId);
  return v?.icon || '◻';
}

export function labelForView(viewId) {
  if (!viewId) return 'Empty';
  if (viewId === 'builtin:agent-terminal') return 'Agent Terminal';
  if (viewId === 'builtin:activity-stream') return 'Activity Stream';
  if (viewId === 'builtin:tool-browser')    return 'Tool Browser';
  if (viewId === 'builtin:plan-viewer')     return 'Plan Viewer';
  if (viewId === 'builtin:agent-overview')  return 'Agent Overview';
  const v = viewRegistry.find(v => v.id === viewId);
  return v ? v.name : viewId;
}

// ── View Pool — iframes persist across layout changes ─────────────
export function getOrCreateView(viewId) {
  if (viewPool.has(viewId)) return viewPool.get(viewId);
  const view = { iframe: null, ready: false, accepts: [], _readyTimeout: null };
  const iframe = document.createElement('iframe');
  // App iframes load cross-origin host pages — need permissive sandbox
  if (viewId.startsWith('app:')) {
    iframe.sandbox = 'allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox';
  } else {
    iframe.sandbox = 'allow-scripts allow-same-origin allow-forms';
  }
  iframe.allow = '';
  // App views: append ?embedded=1 to hide the host page header (dashboard panel
  // already provides its own header).  srcForView returns the raw URL so pop-out
  // still shows the standalone host header.
  const viewUrl = srcForView(viewId);
  iframe.src = viewId.startsWith('app:') && viewUrl
    ? viewUrl + (viewUrl.includes('?') ? '&' : '?') + 'embedded=1'
    : viewUrl;
  iframe.dataset.viewId = viewId;
  view.iframe = iframe;
  viewPool.set(viewId, view);
  // Append to view-overlay — iframes NEVER move between parents (reparenting
  // destroys browsing context in all modern browsers).  They stay here
  // permanently and are positioned over panel body slots via syncViewPositions().
  document.getElementById('view-overlay').appendChild(iframe);
  return view;
}

export function attachViewToSlot(viewId) {
  if (!viewId) return;
  const view = getOrCreateView(viewId);
  // No reparenting — iframe stays in #view-overlay.
  // syncViewPositions() will position it over the panel body.
  requestAnimationFrame(() => { if (_syncViewPositions) _syncViewPositions(); });
  // App views use JSON-RPC, not mcp-dashboard READY — skip timeout
  if (viewId.startsWith('app:')) { view.ready = true; return; }
  // Start ready timeout only for first load
  if (!view.ready && !view._readyTimeout) {
    view._readyTimeout = setTimeout(() => {
      if (!view.ready) {
        const p = findPanelHostingView(viewId);
        const body = p?.el.querySelector('.panel-body');
        if (body && _showPanelError) _showPanelError(body, 'View did not respond in time.');
      }
    }, READY_TIMEOUT_MS);
  }
}

export function findViewIdByWindow(win) {
  for (const [viewId, view] of viewPool) {
    if (view.iframe && view.iframe.contentWindow === win) return viewId;
  }
  return null;
}

export function findPanelHostingView(viewId) {
  for (const panel of Object.values(panels)) {
    if (panel.viewId === viewId) return panel;
  }
  return null;
}

export function sendInitToView(viewId, panel) {
  const view = viewPool.get(viewId);
  if (!view || !view.iframe) return;
  const themeObj = themes[activeTheme] || themes['dark'] || {};
  const bodyEl = panel.el.querySelector('.panel-body');
  const dims = bodyEl
    ? { width: bodyEl.clientWidth, height: bodyEl.clientHeight }
    : { width: 800, height: 450 };
  postToIframe(view.iframe, 'INIT', {
    view_id: viewId,
    panel_id: panel.panelId,
    agent_id: focusedAgentId,
    theme: themeToCSS(themeObj),
    dimensions: dims,
  });
}

export function sendInitToSidebarView(viewId) {
  const view = viewPool.get(viewId);
  if (!view || !view.iframe) return;
  const themeObj = themes[activeTheme] || themes['dark'] || {};
  const sectionBody = document.querySelector(`.sidebar-section-body[data-view-id="${viewId}"]`);
  const dims = sectionBody
    ? { width: sectionBody.clientWidth, height: sectionBody.clientHeight }
    : { width: 360, height: 300 };
  postToIframe(view.iframe, 'INIT', {
    view_id: viewId,
    panel_id: 'sidebar',
    agent_id: focusedAgentId,
    theme: themeToCSS(themeObj),
    dimensions: dims,
  });
}

export function updatePanelHeader(panel) {
  const iconEl = panel.el.querySelector('.panel-icon');
  const nameEl = panel.el.querySelector('.panel-view-toggle');
  if (iconEl) iconEl.textContent = iconForView(panel.viewId);
  if (nameEl) nameEl.textContent = labelForView(panel.viewId) + ' ▾';
}

export function switchPanelView(panelId, newViewId) {
  const panel = panels[panelId];
  if (!panel || panel.viewId === newViewId) return;
  // Just update the mapping — no iframe DOM manipulation needed.
  // The old view's iframe hides automatically via syncViewPositions()
  // (it will no longer be hosted by any panel).
  panel.viewId = newViewId;
  getOrCreateView(newViewId); // ensure iframe exists
  updatePanelHeader(panel);
  requestAnimationFrame(() => { if (_syncViewPositions) _syncViewPositions(); });
  // If already ready, send INIT with current dimensions
  const view = viewPool.get(newViewId);
  if (view?.ready) sendInitToView(newViewId, panel);
}

// ── Message routing ───────────────────────────────────────────────
export function routeToViews(type, payload) {
  const serverName = payload.server_name;
  const metaView   = payload.meta_ui?.view;
  const sent = new Set();
  for (const panel of Object.values(panels)) {
    const view = viewPool.get(panel.viewId);
    if (!view?.ready || !view.iframe) continue;
    const vid = panel.viewId;
    if (vid === 'builtin:activity-stream') { postToIframe(view.iframe, type, payload); sent.add(vid); continue; }
    if (metaView && vid === metaView)       { postToIframe(view.iframe, type, payload); sent.add(vid); continue; }
    const vInfo = viewRegistry.find(v => v.id === vid);
    if (vInfo && vInfo.source === serverName) { postToIframe(view.iframe, type, payload); sent.add(vid); }
  }
  // Also route to sidebar views not in panels
  for (const [viewId, view] of viewPool) {
    if (sent.has(viewId) || !view?.ready || !view.iframe || !isSidebarView(viewId)) continue;
    if (viewId === 'builtin:activity-stream' ||
        (metaView && viewId === metaView) ||
        viewRegistry.find(v => v.id === viewId && v.source === serverName)) {
      postToIframe(view.iframe, type, payload);
    }
  }
  for (const [viewId, entry] of popoutWindows) {
    if (entry.win.closed) continue;
    if (viewId === 'builtin:activity-stream' ||
        (metaView && viewId === metaView) ||
        viewRegistry.find(v => v.id === viewId && v.source === serverName)) {
      if (_postToPopout) _postToPopout(entry.win, type, payload);
    }
  }
}

export function broadcastToViews(type, payload) {
  for (const [viewId, view] of viewPool) {
    if (view?.ready && view.iframe) postToIframe(view.iframe, type, payload);
  }
  for (const [, entry] of popoutWindows) {
    if (!entry.win.closed && _postToPopout) _postToPopout(entry.win, type, payload);
  }
}

export function sendToActivityStream(type, payload) {
  // Activity stream lives in the sidebar — send directly to its viewPool iframe
  const view = viewPool.get('builtin:activity-stream');
  if (view?.ready && view.iframe) postToIframe(view.iframe, type, payload);
  // Notify sidebar toggle on mobile when sidebar is closed
  if (_notifySidebarUpdate) _notifySidebarUpdate();
  // Also send to popout window if open
  for (const [viewId, entry] of popoutWindows) {
    if (viewId === 'builtin:activity-stream' && !entry.win.closed) {
      if (_postToPopout) _postToPopout(entry.win, type, payload);
    }
  }
}

export function broadcastToViewType(viewType, type, payload) {
  for (const [vid, view] of viewPool) {
    if (!view?.ready || !view.iframe) continue;
    const vInfo = viewRegistry.find(v => v.id === vid);
    if (vInfo && vInfo.type === viewType) {
      postToIframe(view.iframe, type, payload);
    } else if (!vInfo && vid === 'builtin:agent-terminal' && viewType === 'conversation') {
      // Fallback before viewRegistry is populated
      postToIframe(view.iframe, type, payload);
    }
  }
  for (const [viewId, entry] of popoutWindows) {
    if (entry.win.closed) continue;
    const vInfo = viewRegistry.find(v => v.id === viewId);
    const matches = (vInfo && vInfo.type === viewType) ||
                    (!vInfo && viewId === 'builtin:agent-terminal' && viewType === 'conversation');
    if (matches && _postToPopout) _postToPopout(entry.win, type, payload);
  }
}

export function postToIframe(iframe, type, payload) {
  try {
    iframe.contentWindow.postMessage({ protocol: PROTOCOL, version: VERSION, type, payload }, '*');
  } catch (e) { /* ignore — iframe may not be ready */ }
}

export function populateViewMenu(menu, panelId) {
  menu.innerHTML = '';
  const currentViewId = panels[panelId]?.viewId;
  for (const v of viewRegistry) {
    const item = document.createElement('div');
    item.className = 'dropdown-item' + (v.id === currentViewId ? ' active' : '');
    item.textContent = (v.icon ? v.icon + ' ' : '') + v.name;
    item.addEventListener('click', (e) => {
      e.stopPropagation();
      menu.classList.remove('open');
      if (v.id !== currentViewId) switchPanelView(panelId, v.id);
    });
    menu.appendChild(item);
  }
  if (!menu.children.length) {
    const item = document.createElement('div');
    item.className = 'dropdown-item';
    item.style.color = 'var(--dash-fg-muted)';
    item.textContent = 'No views available';
    menu.appendChild(item);
  }
}

// ── View → Shell postMessage handler ──────────────────────────────
export function handleViewReady(viewId, payload) {
  const view = viewPool.get(viewId);
  if (!view) return;
  const wasReady = view.ready;
  view.ready = true;
  view.accepts = payload.accepts || [];
  view.agent_scope = payload.agent_scope || null; // "focused" | "all" | specific agent_id
  if (view._readyTimeout) { clearTimeout(view._readyTimeout); view._readyTimeout = null; }
  const panel = findPanelHostingView(viewId);
  if (panel) {
    updatePanelHeader(panel);
    if (!wasReady) {
      sendInitToView(viewId, panel);
      replayCachedState(viewId, view);
    } else {
      if (_notifyResize) _notifyResize(panel.panelId);
    }
  } else if (isSidebarView(viewId)) {
    // Sidebar view (no panel) — send INIT directly
    if (!wasReady) {
      sendInitToSidebarView(viewId);
      replayCachedState(viewId, view);
    }
  }
}

export function replayCachedState(viewId, view) {
  const vInfo = viewRegistry.find(v => v.id === viewId);
  const viewType = vInfo?.type;

  // Replay TOOL_REGISTRY to tools-type views
  if (viewType === 'tools' && _cachedToolRegistry) {
    postToIframe(view.iframe, 'TOOL_REGISTRY', _cachedToolRegistry);
  }
  // Replay PLAN_UPDATE to plan-type views and activity stream
  if (_cachedPlanUpdate && (viewType === 'plan' || viewId === 'builtin:activity-stream')) {
    postToIframe(view.iframe, 'PLAN_UPDATE', _cachedPlanUpdate);
  }
  // Replay CONFIG_STATE to all views
  if (_configState) {
    postToIframe(view.iframe, 'CONFIG_STATE', _configState);
  }
  // Replay CONVERSATION_HISTORY to conversation-type views
  if (viewType === 'conversation') {
    // Request fresh history from bridge
    sendToBridge({ type: 'REQUEST_CONFIG' });
  }
}

// ── Window message listener for iframe communication ──────────────
export function setupViewMessageListener() {
  window.addEventListener('message', (evt) => {
    const msg = evt.data;
    if (!msg || msg.protocol !== PROTOCOL) return;
    const viewId = findViewIdByWindow(evt.source);
    const panel  = viewId ? findPanelHostingView(viewId) : null;

    switch (msg.type) {
      case 'READY': {
        if (viewId) {
          handleViewReady(viewId, msg.payload);
        } else {
          const popId = _findPopoutViewIdByWindow ? _findPopoutViewIdByWindow(evt.source) : null;
          if (popId && _handlePopoutReady) _handlePopoutReady(popId, msg.payload, evt.source);
        }
        break;
      }
      case 'USER_ACTION':
        sendToBridge({ type: 'USER_ACTION', view_id: viewId, ...msg.payload });
        break;
      case 'REQUEST_TOOL':
        sendToBridge({ type: 'REQUEST_TOOL', view_id: viewId, ...msg.payload });
        break;
      case 'USER_MESSAGE':
        sendToBridge({ type: 'USER_MESSAGE', content: msg.payload.content, files: msg.payload.files || undefined });
        break;
      case 'USER_COMMAND':
        sendToBridge({ type: 'USER_COMMAND', command: msg.payload.command });
        break;
      case 'NOTIFY':
        showToast(msg.payload.level || 'info', msg.payload.message, msg.payload.duration_ms);
        break;
      default:
        break;
    }
  });
}
