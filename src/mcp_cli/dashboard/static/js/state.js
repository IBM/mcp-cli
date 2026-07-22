// ================================================================
//  js/state.js — All shared state variables as module-level exports
// ================================================================
'use strict';

// ── Constants ─────────────────────────────────────────────────────
export const PROTOCOL = 'mcp-dashboard';
export const VERSION  = 1;
export const WS_URL   = `ws://${location.host}/ws`;
export const READY_TIMEOUT_MS = 5000;
export const AGENT_COLORS  = ['#7aa2f7','#9ece6a','#e0af68','#f7768e','#7dcfff','#bb9af7','#ff9e64','#73daca'];
export const _WS_BACKOFF_MAX = 30000; // cap at 30 s

// ── Sidebar view constants (shared by layout.js and sidebar.js) ───
export const SIDEBAR_VIEW_IDS = new Set([
  'builtin:activity-stream', 'builtin:tool-browser',
  'builtin:plan-viewer', 'builtin:agent-overview',
]);

export function isSidebarView(viewId) {
  return SIDEBAR_VIEW_IDS.has(viewId) || (viewId && viewId.startsWith('app:'));
}

// ── Mutable state ─────────────────────────────────────────────────
export let ws           = null;
export let connected    = false;
export let themes       = {};
export let activeTheme  = localStorage.getItem('dash-theme') || 'dark';
export let viewRegistry = [];
export let layoutConfig = null;
export let panels       = {};   // panelId → { panelId, el, viewId, rowEl }
export let panelCounter = 0;
export const viewPool      = new Map(); // viewId → { iframe, ready, accepts, _readyTimeout }
export const popoutWindows = new Map(); // viewId → { win, intervalId }

// Multi-agent state
export let agentList       = [];        // array of agent descriptors from AGENT_LIST
export let focusedAgentId  = null;      // currently focused agent id
export const agentColorMap = new Map(); // agent_id → stable color

// WebSocket backoff
export let _wsBackoff = 1000;         // current backoff delay (ms)
export let _wsReconnectTimer = null;

// Cached payloads for replaying to late-loading views
export let _cachedToolRegistry = null;
export let _cachedPlanUpdate = null;

// Config state
export let _configState = null; // cached CONFIG_STATE payload
export let _availableProviders = []; // [{name, models: [...]}]

// Session state
export let _sessionList = [];
export let _currentSessionId = null;

// Tool approval
export let _pendingApprovalCallId = null;

// Sidebar
export let _sidebarOpen = false;

// ── Setter functions ──────────────────────────────────────────────
export function setWs(val) { ws = val; }
export function setConnected(val) { connected = val; }
export function setThemes(val) { themes = val; }
export function setActiveTheme(val) { activeTheme = val; }
export function setViewRegistry(val) { viewRegistry = val; }
export function setLayoutConfig(val) { layoutConfig = val; }
export function setPanels(val) { panels = val; }
export function incPanelCounter() { return ++panelCounter; }
export function setAgentList(val) { agentList = val; }
export function setFocusedAgentId(val) { focusedAgentId = val; }
export function setWsBackoff(val) { _wsBackoff = val; }
export function setWsReconnectTimer(val) { _wsReconnectTimer = val; }
export function setCachedToolRegistry(val) { _cachedToolRegistry = val; }
export function setCachedPlanUpdate(val) { _cachedPlanUpdate = val; }
export function setConfigState(val) { _configState = val; }
export function setAvailableProviders(val) { _availableProviders = val; }
export function setSessionList(val) { _sessionList = val; }
export function setCurrentSessionId(val) { _currentSessionId = val; }
export function setPendingApprovalCallId(val) { _pendingApprovalCallId = val; }
export function setSidebarOpen(val) { _sidebarOpen = val; }
