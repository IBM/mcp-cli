// ================================================================
//  js/apps.js — MCP App panel handlers
// ================================================================
'use strict';

import {
  viewRegistry, viewPool, popoutWindows,
  setViewRegistry,
} from './state.js';
import { syncViewPositions, rebuildAddPanelMenu } from './layout.js';
import { showToast } from './utils.js';

// Late-binding for sidebar deps to avoid direct import from sidebar.js
let _buildSidebarSections = null;
export function setAppDeps(deps) {
  _buildSidebarSections = deps.buildSidebarSections;
}

export function handleAppLaunched(payload) {
  const toolName = payload.tool_name;
  const appUrl = payload.url;
  const resourceUri = payload.resource_uri || null;
  const viewId = 'app:' + toolName;
  const prettyName = toolName.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

  // Already registered by the same tool name — update URL and refresh iframe
  const existing = viewRegistry.find(v => v.id === viewId);
  if (existing) {
    existing.url = appUrl;
    existing.resourceUri = resourceUri;
    const view = viewPool.get(viewId);
    if (view && view.iframe) {
      view.iframe.src = appUrl + (appUrl.includes('?') ? '&' : '?') + 'embedded=1';
    }
    showToast('info', 'App updated: ' + prettyName);
    return;
  }

  // Different tool name but same resourceUri — reuse the existing view slot
  if (resourceUri) {
    const sameResource = viewRegistry.find(
      v => v.type === 'app' && v.resourceUri === resourceUri
    );
    if (sameResource) {
      const oldId = sameResource.id;
      // Update registry entry in place (keep same slot in sidebar)
      sameResource.id = viewId;
      sameResource.name = prettyName;
      sameResource.url = appUrl;
      sameResource.resourceUri = resourceUri;
      sameResource.source = payload.server_name || 'app';

      // Migrate viewPool entry to new id
      const oldView = viewPool.get(oldId);
      if (oldView) {
        viewPool.delete(oldId);
        viewPool.set(viewId, oldView);
        // Refresh iframe with new URL
        if (oldView.iframe) {
          oldView.iframe.src = appUrl + (appUrl.includes('?') ? '&' : '?') + 'embedded=1';
        }
      }

      // Migrate popout window if any
      const oldPopout = popoutWindows.get(oldId);
      if (oldPopout) {
        popoutWindows.delete(oldId);
        popoutWindows.set(viewId, oldPopout);
      }

      // Rebuild sidebar sections so the data-view-id attributes update
      if (_buildSidebarSections) _buildSidebarSections();
      rebuildAddPanelMenu();
      requestAnimationFrame(() => syncViewPositions());
      showToast('info', 'View updated: ' + prettyName);
      return;
    }
  }

  // Register as new view
  viewRegistry.push({
    id: viewId,
    name: prettyName,
    source: payload.server_name || 'app',
    icon: '\u{1F5A5}',
    type: 'app',
    url: appUrl,
    resourceUri: resourceUri,
  });
  rebuildAddPanelMenu();

  // Add to sidebar as collapsible section (not a grid panel)
  if (_buildSidebarSections) _buildSidebarSections();

  // Auto-expand the new app section
  const newSection = document.querySelector(`.sidebar-section[data-view-id="${viewId}"]`);
  if (newSection && !newSection.classList.contains('expanded')) {
    newSection.classList.add('expanded');
  }

  // Mark auto-ready (app uses JSON-RPC, not mcp-dashboard READY)
  const view = viewPool.get(viewId);
  if (view) {
    view.ready = true;
    if (view._readyTimeout) { clearTimeout(view._readyTimeout); view._readyTimeout = null; }
  }

  requestAnimationFrame(() => syncViewPositions());
  showToast('success', 'App launched: ' + prettyName);
}

export function handleAppClosed(payload) {
  const viewId = 'app:' + payload.tool_name;

  setViewRegistry(viewRegistry.filter(v => v.id !== viewId));

  // Clean up iframe and popout
  const view = viewPool.get(viewId);
  if (view && view.iframe) view.iframe.remove();
  viewPool.delete(viewId);

  const popout = popoutWindows.get(viewId);
  if (popout && !popout.win.closed) { popout.win.close(); clearInterval(popout.intervalId); }
  popoutWindows.delete(viewId);

  // Rebuild sidebar sections (app section disappears)
  if (_buildSidebarSections) _buildSidebarSections();
  rebuildAddPanelMenu();
  requestAnimationFrame(() => syncViewPositions());
  showToast('info', 'App closed: ' + payload.tool_name);
}
