// ================================================================
//  js/layout.js — Layout rendering, panel ops, resize handles
// ================================================================
'use strict';

import {
  panels, layoutConfig, viewRegistry, viewPool, popoutWindows,
  PROTOCOL, VERSION, focusedAgentId, themes, activeTheme,
  setPanels, incPanelCounter, setLayoutConfig,
  isSidebarView, _sidebarOpen,
} from './state.js';
import { makeDraggable, showToast } from './utils.js';
import {
  getOrCreateView, attachViewToSlot, iconForView, labelForView, srcForView,
  updatePanelHeader, switchPanelView, populateViewMenu, postToIframe,
  broadcastToViews, findPanelHostingView,
} from './views.js';
import { themeToCSS } from './theme.js';

// ── Cached overlay element ───────────────────────────────────────
let _cachedOverlay = null;
function getOverlay() {
  if (!_cachedOverlay) _cachedOverlay = document.getElementById('view-overlay');
  return _cachedOverlay;
}

// ── Late-binding for sidebar deps to avoid circular imports ───────
let _buildSidebarSections = null;

export function setLayoutDeps(deps) {
  _buildSidebarSections = deps.buildSidebarSections;
}

// ── Layout rendering ──────────────────────────────────────────────
export function defaultLayout() {
  // Terminal fills the left panel; activity stream + apps live in the sidebar
  return {
    rows: [
      {
        height: '100%',
        columns: [
          { width: '100%', view: 'builtin:agent-terminal' },
        ],
      },
    ],
  };
}

export function renderLayout(layout) {
  const root = document.getElementById('grid-root');
  // Iframes stay in #view-overlay — we only rebuild the panel grid.
  // No iframe reparenting occurs, so browsing context is preserved.
  root.innerHTML = '';
  setPanels({});

  const rows = layout.rows || [];
  rows.forEach((row, ri) => {
    if (ri > 0) root.appendChild(makeRowHandle(root, ri));
    const rowEl = document.createElement('div');
    rowEl.className = 'grid-row';
    rowEl.style.flex = parseFlex(row.height || '100%');
    root.appendChild(rowEl);
    (row.columns || []).forEach((col, ci) => {
      if (ci > 0) rowEl.appendChild(makeColHandle(rowEl, ci));
      const panelEl = createPanelSlot(col.view || 'auto', rowEl);
      panelEl.style.flex = parseFlex(col.width || '100%');
      rowEl.appendChild(panelEl);
    });
  });

  // Position iframes over their panel body slots after browser layout
  requestAnimationFrame(() => syncViewPositions());
}

export function parseFlex(pct) {
  const n = parseFloat(pct);
  return isNaN(n) ? '1' : String(n);
}

export function createPanelSlot(viewId, rowEl) {
  const panelId = 'panel_' + incPanelCounter();
  const resolvedViewId = resolveAutoView(viewId);

  const panelEl = document.createElement('div');
  panelEl.className = 'panel';
  panelEl.dataset.panelId = panelId;

  // Header
  const header = document.createElement('div');
  header.className = 'panel-header';
  header.draggable = true;
  header.innerHTML = `
    <span class="panel-icon">${iconForView(resolvedViewId)}</span>
    <button class="panel-view-toggle" title="Switch view">${labelForView(resolvedViewId)} ▾</button>
    <div class="panel-view-menu dropdown-menu"></div>
    <button class="panel-btn" title="Pop out" data-action="popout">⤢</button>
    <button class="panel-btn" title="Minimize" data-action="minimize">−</button>
    <button class="panel-btn" title="Close" data-action="close">×</button>
  `;
  panelEl.appendChild(header);

  const body = document.createElement('div');
  body.className = 'panel-body';
  panelEl.appendChild(body);

  const panel = { panelId, el: panelEl, viewId: resolvedViewId, rowEl };
  panels[panelId] = panel;

  if (resolvedViewId) {
    attachViewToSlot(resolvedViewId);
  } else {
    const ph = document.createElement('div');
    ph.className = 'panel-placeholder';
    ph.textContent = 'Connect an MCP server with views to populate this panel.';
    body.appendChild(ph);
  }

  // Button + view-picker click handler
  header.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (btn) {
      e.stopPropagation();
      const action = btn.dataset.action;
      if (action === 'minimize') { panelEl.classList.toggle('minimized'); requestAnimationFrame(() => syncViewPositions()); }
      else if (action === 'close') closePanel(panelId);
      else if (action === 'popout') popoutPanel(panel);
      return;
    }
    if (e.target.classList.contains('panel-view-toggle')) {
      e.stopPropagation();
      const menu = header.querySelector('.panel-view-menu');
      if (!menu.classList.contains('open')) populateViewMenu(menu, panelId);
      menu.classList.toggle('open');
    }
  });

  // Drag-to-swap
  header.addEventListener('dragstart', (e) => e.dataTransfer.setData('text/plain', panelId));
  panelEl.addEventListener('dragover', (e) => e.preventDefault());
  panelEl.addEventListener('drop', (e) => {
    e.preventDefault();
    const srcId = e.dataTransfer.getData('text/plain');
    if (srcId && srcId !== panelId) swapPanels(srcId, panelId);
  });

  return panelEl;
}

export function resolveAutoView(viewId) {
  if (viewId !== 'auto') return viewId;
  const placed = new Set(Object.values(panels).map(p => p.viewId));
  const builtins = new Set(['builtin:agent-terminal', 'builtin:activity-stream']);
  for (const v of viewRegistry) {
    if (!builtins.has(v.id) && !placed.has(v.id)) return v.id;
  }
  return null;
}

// ── Panel operations ──────────────────────────────────────────────
export function closePanel(panelId) {
  const panel = panels[panelId];
  if (!panel) return;
  // Don't touch the iframe — it stays alive in #view-overlay.
  // syncViewPositions() will hide it since no panel hosts it.
  const rowEl = panel.rowEl;
  panel.el.remove();
  delete panels[panelId];
  if (rowEl && rowEl.querySelectorAll('.panel').length === 0) rowEl.remove();
  rebuildAddPanelMenu();
  // Rebuild sidebar sections in case panel had a sidebar view
  if (_buildSidebarSections) _buildSidebarSections();
  requestAnimationFrame(() => syncViewPositions());
}

function popoutPanel(panel) {
  popoutSidebarView(panel.viewId);
}

export function popoutSidebarView(viewId) {
  const url = srcForView(viewId);
  if (!url) return;
  // Reuse existing popup if still open
  const existing = popoutWindows.get(viewId);
  if (existing && !existing.win.closed) { existing.win.focus(); return; }
  const win = window.open(url, `dash-pop-${viewId}`, 'width=900,height=600,menubar=no,toolbar=no,location=no');
  if (!win) return;
  const intervalId = setInterval(() => {
    if (win.closed) { popoutWindows.delete(viewId); clearInterval(intervalId); }
  }, 1000);
  popoutWindows.set(viewId, { win, intervalId });
}

export function findPopoutViewIdByWindow(win) {
  for (const [viewId, entry] of popoutWindows) {
    if (entry.win === win) return viewId;
  }
  return null;
}

export function handlePopoutReady(viewId, payload, win) {
  const themeObj = themes[activeTheme] || themes['dark'] || {};
  try {
    win.postMessage({
      protocol: PROTOCOL, version: VERSION, type: 'INIT',
      payload: {
        view_id: viewId, panel_id: null, agent_id: focusedAgentId,
        theme: themeToCSS(themeObj),
        dimensions: { width: win.innerWidth || 900, height: win.innerHeight || 600 },
      },
    }, '*');
  } catch (e) { /* ignore */ }
}

export function postToPopout(win, type, payload) {
  try { win.postMessage({ protocol: PROTOCOL, version: VERSION, type, payload }, '*'); } catch (e) { /* ignore */ }
}

export function swapPanels(aId, bId) {
  const a = panels[aId], b = panels[bId];
  if (!a || !b || a.viewId === b.viewId) return;
  // Just swap the viewId mappings — no iframe DOM manipulation.
  // syncViewPositions() repositions iframes over the correct panel bodies.
  const tmp = a.viewId;
  a.viewId = b.viewId;
  b.viewId = tmp;
  updatePanelHeader(a);
  updatePanelHeader(b);
  requestAnimationFrame(() => syncViewPositions());
}

export function showPanelError(bodyEl, msg) {
  const ph = document.createElement('div');
  ph.className = 'panel-placeholder';
  ph.style.color = 'var(--dash-error)';
  ph.textContent = msg;
  bodyEl.innerHTML = '';
  bodyEl.appendChild(ph);
}

// ── Resize handles (mouse + touch via makeDraggable) ──────────────
export function makeColHandle(rowEl, idx) {
  const handle = document.createElement('div');
  handle.className = 'resize-handle-col';
  handle.dataset.handleType = 'col';
  const overlay = getOverlay();

  makeDraggable(handle, {
    onStart(x, _y) {
      handle.classList.add('dragging');
      if (overlay) overlay.style.pointerEvents = 'none';
      const handleIdx = Array.from(rowEl.children).indexOf(handle);
      const panelsBefore = Array.from(rowEl.children).slice(0, handleIdx).filter(c => c.classList.contains('panel'));
      const panelsAfter  = Array.from(rowEl.children).slice(handleIdx + 1).filter(c => c.classList.contains('panel'));
      const pBefore = panelsBefore[panelsBefore.length - 1];
      const pAfter  = panelsAfter[0];
      if (!pBefore || !pAfter) return null;
      return {
        startX: x,
        pBefore, pAfter,
        startBefore: pBefore.getBoundingClientRect().width,
        startAfter: pAfter.getBoundingClientRect().width,
      };
    },
    onMove(state, x, _y) {
      const dx = x - state.startX;
      const nb = Math.max(200, state.startBefore + dx);
      const na = Math.max(200, state.startAfter  - dx);
      state.pBefore.style.flex = `0 0 ${nb}px`;
      state.pAfter.style.flex  = `0 0 ${na}px`;
      syncViewPositions();
      notifyResize(state.pBefore.dataset.panelId);
      notifyResize(state.pAfter.dataset.panelId);
    },
    onEnd(_state) {
      handle.classList.remove('dragging');
      if (overlay) overlay.style.pointerEvents = '';
    },
  });

  handle.addEventListener('dblclick', () => {
    Array.from(rowEl.children).filter(c => c.classList.contains('panel')).forEach(p => {
      p.style.flex = '';
    });
    requestAnimationFrame(() => syncViewPositions());
  });

  return handle;
}

export function makeRowHandle(root, idx) {
  const handle = document.createElement('div');
  handle.className = 'resize-handle-row';
  const overlay = getOverlay();

  makeDraggable(handle, {
    onStart(_x, y) {
      handle.classList.add('dragging');
      if (overlay) overlay.style.pointerEvents = 'none';
      const handleIdx = Array.from(root.children).indexOf(handle);
      const rowsBefore = Array.from(root.children).slice(0, handleIdx).filter(c => c.classList.contains('grid-row'));
      const rowsAfter  = Array.from(root.children).slice(handleIdx + 1).filter(c => c.classList.contains('grid-row'));
      const rBefore = rowsBefore[rowsBefore.length - 1];
      const rAfter  = rowsAfter[0];
      if (!rBefore || !rAfter) return null;
      return {
        startY: y,
        rBefore, rAfter,
        startBefore: rBefore.getBoundingClientRect().height,
        startAfter: rAfter.getBoundingClientRect().height,
      };
    },
    onMove(state, _x, y) {
      const dy = y - state.startY;
      const nb = Math.max(150, state.startBefore + dy);
      const na = Math.max(150, state.startAfter  - dy);
      state.rBefore.style.flex = `0 0 ${nb}px`;
      state.rAfter.style.flex  = `0 0 ${na}px`;
      syncViewPositions();
    },
    onEnd(_state) {
      handle.classList.remove('dragging');
      if (overlay) overlay.style.pointerEvents = '';
    },
  });

  handle.addEventListener('dblclick', () => {
    Array.from(root.children).filter(c => c.classList.contains('grid-row')).forEach(r => {
      r.style.flex = '';
    });
    requestAnimationFrame(() => syncViewPositions());
  });

  return handle;
}

export function notifyResize(panelId) {
  const panel = panels[panelId];
  if (!panel?.viewId) return;
  const view = viewPool.get(panel.viewId);
  if (!view?.ready || !view.iframe) return;
  const body = panel.el.querySelector('.panel-body');
  if (!body) return;
  postToIframe(view.iframe, 'RESIZE', { width: body.clientWidth, height: body.clientHeight });
}

// ── View overlay positioning — positions iframes over panel body slots ──
export function syncViewPositions() {
  const overlay = getOverlay();
  if (!overlay) return;
  const overlayRect = overlay.getBoundingClientRect();
  const inMobileSidebar = document.body.classList.contains('mobile-sidebar');

  for (const [viewId, view] of viewPool) {
    if (!view.iframe) continue;

    // Sidebar views: position over their .sidebar-section-body in the sidebar panel
    if (isSidebarView(viewId)) {
      // On mobile, sidebar must be open; on desktop, sidebar is always in flow
      const sidebarVisible = inMobileSidebar ? _sidebarOpen : true;
      if (sidebarVisible) {
        const sectionBody = document.querySelector(`.sidebar-section-body[data-view-id="${viewId}"]`);
        if (sectionBody) {
          const rect = sectionBody.getBoundingClientRect();
          if (rect.width > 0 && rect.height > 0) {
            view.iframe.style.display = 'block';
            view.iframe.style.left   = (rect.left   - overlayRect.left) + 'px';
            view.iframe.style.top    = (rect.top    - overlayRect.top)  + 'px';
            view.iframe.style.width  = rect.width  + 'px';
            view.iframe.style.height = rect.height + 'px';
            // On mobile, sidebar overlays content — iframes need high z-index
            view.iframe.style.zIndex = inMobileSidebar ? '220' : '';
            continue;
          }
        }
      }
      view.iframe.style.display = 'none';
      view.iframe.style.zIndex = '';
      continue;
    }

    // Non-sidebar views: position over their panel body slot
    const panel = findPanelHostingView(viewId);
    if (panel && !panel.el.classList.contains('minimized')) {
      const body = panel.el.querySelector('.panel-body');
      if (body) {
        const rect = body.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
          view.iframe.style.display = 'block';
          view.iframe.style.left   = (rect.left   - overlayRect.left) + 'px';
          view.iframe.style.top    = (rect.top    - overlayRect.top)  + 'px';
          view.iframe.style.width  = rect.width  + 'px';
          view.iframe.style.height = rect.height + 'px';
          view.iframe.style.zIndex = '';
          continue;
        }
      }
    }
    // Not placed, minimized, or zero-size — hide
    view.iframe.style.display = 'none';
    view.iframe.style.zIndex = '';
  }
}

// ── Layout dropdown ───────────────────────────────────────────────
const BUILTIN_PRESETS = ['Minimal', 'Standard', 'Full'];

export function buildLayoutMenu() {
  const menu = document.getElementById('layout-menu');
  menu.innerHTML = '';

  for (const name of BUILTIN_PRESETS) {
    const item = document.createElement('div');
    item.className = 'dropdown-item';
    item.textContent = name;
    item.addEventListener('click', () => {
      applyPreset(name);
      menu.classList.remove('open');
    });
    menu.appendChild(item);
  }
}

export function applyPreset(name) {
  const presets = {
    'Minimal': { rows: [{ height: '100%', columns: [{ width: '100%', view: 'builtin:agent-terminal' }] }] },
    'Standard': { rows: [{ height: '100%', columns: [
      { width: '70%', view: 'builtin:agent-terminal' },
      { width: '30%', view: 'builtin:tool-browser' },
    ] }] },
    'Full': { rows: [
      { height: '70%', columns: [
        { width: '70%', view: 'builtin:agent-terminal' },
        { width: '30%', view: 'builtin:plan-viewer' },
      ] },
      { height: '30%', columns: [
        { width: '50%', view: 'builtin:tool-browser' },
        { width: '50%', view: 'builtin:activity-stream' },
      ] },
    ] },
  };
  const layout = presets[name];
  if (layout) { setLayoutConfig(layout); renderLayout(layout); }
}

// + Add Panel menu
export function rebuildAddPanelMenu() {
  const menu = document.getElementById('add-panel-menu');
  menu.innerHTML = '';
  const placed = new Set(Object.values(panels).map(p => p.viewId));

  for (const v of viewRegistry) {
    if (placed.has(v.id)) continue;
    const item = document.createElement('div');
    item.className = 'dropdown-item';
    item.textContent = v.name;
    item.addEventListener('click', () => {
      addPanelWithView(v.id);
      menu.classList.remove('open');
    });
    menu.appendChild(item);
  }

  if (!menu.children.length) {
    const item = document.createElement('div');
    item.className = 'dropdown-item';
    item.style.color = 'var(--dash-fg-muted)';
    item.textContent = 'No additional views';
    menu.appendChild(item);
  }
}

export function addPanelWithView(viewId) {
  const root = document.getElementById('grid-root');
  let lastRow = root.querySelector('.grid-row:last-child');
  if (!lastRow) {
    lastRow = document.createElement('div');
    lastRow.className = 'grid-row';
    lastRow.style.flex = '1';
    root.appendChild(lastRow);
  }
  const existingPanels = lastRow.querySelectorAll('.panel');
  if (existingPanels.length > 0) lastRow.appendChild(makeColHandle(lastRow, existingPanels.length));
  const panelEl = createPanelSlot(viewId, lastRow);
  panelEl.style.flex = '1';
  lastRow.appendChild(panelEl);
  rebuildAddPanelMenu();
  requestAnimationFrame(() => syncViewPositions());
}
