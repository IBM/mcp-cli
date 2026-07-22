// ================================================================
//  js/sidebar.js — Right sidebar: collapsible stacked sections
// ================================================================
'use strict';

import {
  _sidebarOpen, setSidebarOpen,
  viewRegistry, isSidebarView, panels,
} from './state.js';
import { makeDraggable } from './utils.js';
import { getOrCreateView, sendInitToSidebarView, labelForView } from './views.js';
import { syncViewPositions, popoutSidebarView, rebuildAddPanelMenu } from './layout.js';

export function updateSidebarMode() {
  const isMobile = window.innerWidth < 768;
  document.body.classList.toggle('mobile-sidebar', isMobile);

  if (isMobile) {
    // Mark panels whose views belong in the sidebar
    for (const panel of Object.values(panels)) {
      panel.el.classList.toggle('sidebar-hosted', isSidebarView(panel.viewId));
    }
  } else {
    // Exiting mobile — clear sidebar-hosted, close sidebar overlay
    for (const panel of Object.values(panels)) {
      panel.el.classList.remove('sidebar-hosted');
    }
    closeSidebar();
  }
  // Always rebuild sidebar sections (works on both desktop and mobile)
  buildSidebarSections();
  requestAnimationFrame(() => syncViewPositions());
}

export function getSidebarViewIds() {
  // Collect viewIds that should appear in the sidebar, in order.
  // Always include ALL SIDEBAR_VIEW_IDS (activity stream, tool browser,
  // plan viewer, agent overview) plus any registered app: views.
  const views = [];
  // Activity stream always first
  views.push('builtin:activity-stream');
  // Then other builtin sidebar views in a stable order
  for (const vid of ['builtin:tool-browser', 'builtin:plan-viewer', 'builtin:agent-overview']) {
    views.push(vid);
  }
  // Then any app: views from viewRegistry (not viewPool — may not be created yet)
  for (const entry of viewRegistry) {
    if (entry.id.startsWith('app:') && !views.includes(entry.id)) {
      views.push(entry.id);
    }
  }
  return views;
}

export function buildSidebarSections() {
  // Build collapsible accordion sections inside #sidebar-view-slot.
  // Each section has a clickable header + a body that's the iframe positioning target.
  const slot = document.getElementById('sidebar-view-slot');
  // Remember which sections were expanded
  const expandedSet = new Set();
  slot.querySelectorAll('.sidebar-section.expanded').forEach(el => {
    expandedSet.add(el.dataset.viewId);
  });

  slot.innerHTML = '';
  const views = getSidebarViewIds();
  const isFirstBuild = expandedSet.size === 0;

  for (let i = 0; i < views.length; i++) {
    const viewId = views[i];
    const section = document.createElement('div');
    section.className = 'sidebar-section';
    section.dataset.viewId = viewId;

    // Expand: restore previous state, or expand first section on first build
    if (isFirstBuild ? i === 0 : expandedSet.has(viewId)) {
      section.classList.add('expanded');
    }

    // Header (click to toggle, with action buttons)
    const header = document.createElement('div');
    header.className = 'sidebar-section-header';
    const chevron = document.createElement('span');
    chevron.className = 'sidebar-section-chevron';
    chevron.textContent = '▸';
    const name = document.createElement('span');
    name.className = 'sidebar-section-name';
    name.textContent = labelForView(viewId);

    // Action buttons (maximize, popout)
    const btns = document.createElement('span');
    btns.className = 'sidebar-section-btns';
    const maxBtn = document.createElement('button');
    maxBtn.className = 'sidebar-section-btn';
    maxBtn.title = 'Maximize';
    maxBtn.textContent = '⤢';
    maxBtn.dataset.action = 'maximize';
    const popBtn = document.createElement('button');
    popBtn.className = 'sidebar-section-btn';
    popBtn.title = 'Pop out';
    popBtn.textContent = '↗';
    popBtn.dataset.action = 'popout';
    btns.appendChild(maxBtn);
    btns.appendChild(popBtn);

    header.appendChild(chevron);
    header.appendChild(name);
    header.appendChild(btns);

    header.addEventListener('click', (e) => {
      const actionBtn = e.target.closest('[data-action]');
      if (actionBtn) {
        e.stopPropagation();
        const action = actionBtn.dataset.action;
        if (action === 'maximize') {
          const wasMaximized = section.classList.contains('maximized');
          // Clear all maximized states first
          slot.querySelectorAll('.sidebar-section.maximized').forEach(s => s.classList.remove('maximized'));
          if (!wasMaximized) {
            section.classList.add('expanded');
            section.classList.add('maximized');
          }
          requestAnimationFrame(() => syncViewPositions());
        } else if (action === 'popout') {
          popoutSidebarView(viewId);
        }
        return;
      }
      // Clicking header toggles expand/collapse
      section.classList.toggle('expanded');
      // Clear maximized if collapsing
      if (!section.classList.contains('expanded')) {
        section.classList.remove('maximized');
      }
      requestAnimationFrame(() => syncViewPositions());
    });

    // Body (iframe positioning target)
    const body = document.createElement('div');
    body.className = 'sidebar-section-body';
    body.dataset.viewId = viewId;

    section.appendChild(header);
    section.appendChild(body);
    slot.appendChild(section);

    // Ensure the iframe exists
    getOrCreateView(viewId);
  }
}

export function openSidebar() {
  setSidebarOpen(true);
  buildSidebarSections();
  document.getElementById('sidebar-panel').classList.add('open');
  document.getElementById('drawer-backdrop').classList.add('visible');
  document.getElementById('sidebar-toggle').classList.remove('has-update');
  requestAnimationFrame(() => syncViewPositions());
}

export function closeSidebar() {
  if (!_sidebarOpen) return;
  setSidebarOpen(false);
  document.getElementById('sidebar-panel').classList.remove('open');
  // Only hide backdrop if no drawers are also open
  const settingsOpen = document.getElementById('settings-panel').classList.contains('open');
  const sessionsOpen = document.getElementById('session-drawer').classList.contains('open');
  if (!settingsOpen && !sessionsOpen) {
    document.getElementById('drawer-backdrop').classList.remove('visible');
  }
}

export function notifySidebarUpdate() {
  // Highlight the sidebar toggle when new activity arrives while sidebar is closed on mobile
  if (!_sidebarOpen && document.body.classList.contains('mobile-sidebar')) {
    document.getElementById('sidebar-toggle').classList.add('has-update');
  }
}

// Sidebar toggle + close event handlers
export function wireSidebarEvents() {
  document.getElementById('sidebar-toggle').addEventListener('click', () => {
    if (_sidebarOpen) closeSidebar(); else openSidebar();
  });
  document.getElementById('sidebar-close').addEventListener('click', closeSidebar);
}

// ── Sidebar resize handle ─────────────────────────────────────────
export function setupSidebarResize() {
  const handle = document.getElementById('sidebar-resize-handle');
  if (!handle) return;
  const overlay = document.getElementById('view-overlay');

  makeDraggable(handle, {
    onStart(x, _y) {
      handle.classList.add('dragging');
      // Block pointer events on iframes during drag so they don't steal the mouse
      if (overlay) overlay.style.pointerEvents = 'none';
      const sidebar = document.getElementById('sidebar-panel');
      const gridWrapper = document.getElementById('grid-wrapper');
      const mainContent = document.getElementById('main-content');
      if (!sidebar || !gridWrapper || !mainContent) return null;
      const totalWidth = mainContent.getBoundingClientRect().width - handle.getBoundingClientRect().width;
      return {
        startX: x,
        startSidebarWidth: sidebar.getBoundingClientRect().width,
        sidebar,
        gridWrapper,
        totalWidth,
      };
    },
    onMove(state, x, _y) {
      const dx = x - state.startX;
      // Dragging right → sidebar narrower; dragging left → sidebar wider
      const newSidebar = Math.max(200, Math.min(600, state.startSidebarWidth - dx));
      state.sidebar.style.width = newSidebar + 'px';
      // Grid takes remaining space via flex: 1 (no fixed sizing needed)
      syncViewPositions();
    },
    onEnd(_state) {
      handle.classList.remove('dragging');
      // Restore pointer events on overlay
      if (overlay) overlay.style.pointerEvents = '';
    },
  });

  // Double-click resets sidebar to default 30%
  handle.addEventListener('dblclick', () => {
    const sidebar = document.getElementById('sidebar-panel');
    if (sidebar) sidebar.style.width = '';
    requestAnimationFrame(() => syncViewPositions());
  });
}
