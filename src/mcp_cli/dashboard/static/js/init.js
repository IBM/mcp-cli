// ================================================================
//  js/init.js — Entry point: wires everything together and boots
// ================================================================
'use strict';

import {
  layoutConfig, setLayoutConfig, setViewRegistry,
} from './state.js';
import { setCloseSidebarFn } from './utils.js';
import { loadThemes } from './theme.js';
import { connectWS, setMessageHandler } from './websocket.js';
import { handleBridgeMessage } from './dispatcher.js';
import {
  renderLayout, buildLayoutMenu, defaultLayout,
  rebuildAddPanelMenu, syncViewPositions,
  showPanelError, notifyResize,
  findPopoutViewIdByWindow, handlePopoutReady, postToPopout,
  setLayoutDeps,
} from './layout.js';
import {
  updateSidebarMode, setupSidebarResize,
  wireSidebarEvents, closeSidebar,
  buildSidebarSections, notifySidebarUpdate,
} from './sidebar.js';
import { buildOverflowMenu, wireToolbarEvents } from './toolbar.js';
import { wireConfigEvents } from './config.js';
import { wireApprovalEvents } from './approval.js';
import { setViewDeps, setupViewMessageListener } from './views.js';
import { setAppDeps } from './apps.js';

// ── Wire up late-binding deps to break circular imports ───────────

// views.js needs layout.js + sidebar.js functions
setViewDeps({
  syncViewPositions,
  showPanelError,
  notifyResize,
  findPopoutViewIdByWindow,
  handlePopoutReady,
  postToPopout,
  notifySidebarUpdate,
});

// layout.js needs sidebar.js functions
setLayoutDeps({
  buildSidebarSections,
});

// apps.js needs sidebar.js functions
setAppDeps({
  buildSidebarSections,
});

// utils.js closeAllDrawers needs closeSidebar from sidebar.js
setCloseSidebarFn(closeSidebar);

// websocket.js needs dispatcher.js handleBridgeMessage
setMessageHandler(handleBridgeMessage);

// ── Container-aware layout (for IDE embedding) ────────────────────
function setupContainerObserver() {
  if (typeof ResizeObserver === 'undefined') return;
  const gridWrapper = document.getElementById('grid-wrapper');
  const ro = new ResizeObserver(entries => {
    for (const entry of entries) {
      const w = entry.contentRect.width;
      document.body.classList.toggle('container-narrow', w < 600);
      document.body.classList.toggle('container-xs', w < 400);
    }
    requestAnimationFrame(() => syncViewPositions());
  });
  ro.observe(gridWrapper);
}

// ── Responsive resize handler ─────────────────────────────────────
window.addEventListener('resize', () => {
  updateSidebarMode();
  buildOverflowMenu();
});

// ── Initialisation ────────────────────────────────────────────────
async function init() {
  await loadThemes();
  buildLayoutMenu();

  // Bootstrap view registry with builtins
  setViewRegistry([
    { id: 'builtin:agent-terminal', name: 'Agent Terminal', source: 'builtin', icon: '⌨', type: 'conversation', url: '/views/agent-terminal.html' },
    { id: 'builtin:activity-stream', name: 'Activity Stream', source: 'builtin', icon: '◈', type: 'stream', url: '/views/activity-stream.html' },
    { id: 'builtin:tool-browser',    name: 'Tool Browser',    source: 'builtin', icon: '🔧', type: 'tools',        url: '/views/tool-browser.html' },
    { id: 'builtin:plan-viewer',     name: 'Plan Viewer',     source: 'builtin', icon: '📋', type: 'plan',         url: '/views/plan-viewer.html' },
    { id: 'builtin:agent-overview',  name: 'Agent Overview',  source: 'builtin', icon: '👥', type: 'agents',       url: '/views/agent-overview.html' },
  ]);
  rebuildAddPanelMenu();

  // Render layout — auto-select based on screen size if no stored layout
  let config = null;
  try {
    const stored = localStorage.getItem('dash-layout');
    config = stored ? JSON.parse(stored) : null;
  } catch (e) {
    console.warn('Could not parse stored layout, using default:', e);
    config = null;
  }
  if (!config) {
    config = defaultLayout();
  }
  setLayoutConfig(config);
  renderLayout(config);

  // Wire up all event listeners
  setupViewMessageListener();
  wireSidebarEvents();
  wireToolbarEvents();
  wireConfigEvents();
  wireApprovalEvents();

  // Initialise responsive state
  updateSidebarMode();
  buildOverflowMenu();
  setupContainerObserver();
  setupSidebarResize();

  // Connect WebSocket
  connectWS();
}

init();
