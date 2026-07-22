// ================================================================
//  js/websocket.js — WebSocket connection + multi-agent helpers
// ================================================================
'use strict';

import {
  ws, connected, WS_URL, _wsBackoff, _WS_BACKOFF_MAX, _wsReconnectTimer,
  agentList, focusedAgentId, AGENT_COLORS, agentColorMap,
  setWs, setConnected, setWsBackoff, setWsReconnectTimer,
  setAgentList, setFocusedAgentId,
} from './state.js';
import { showToast } from './utils.js';

// ── Late-binding message handler (set by init.js) ─────────────────
let _messageHandler = null;

export function setMessageHandler(fn) {
  _messageHandler = fn;
}

// ── WebSocket (exponential backoff reconnect) ─────────────────────
export function connectWS() {
  if (_wsReconnectTimer) { clearTimeout(_wsReconnectTimer); setWsReconnectTimer(null); }
  // Close existing connection before creating a new one
  if (ws) { try { ws.onclose = null; ws.close(); } catch(e) {} setWs(null); }
  const socket = new WebSocket(WS_URL);
  setWs(socket);

  socket.onopen = () => {
    setConnected(true);
    setWsBackoff(1000); // reset on success
    const dot = document.getElementById('conn-dot');
    if (dot) dot.classList.add('connected');
    showToast('success', 'Connected to dashboard server', 2000);
    // Request agent list (router will send AGENT_LIST)
    sendToBridge({ type: 'REQUEST_AGENT_LIST' });
    // Request current config (model, servers, system prompt)
    sendToBridge({ type: 'REQUEST_CONFIG' });
    // Request tool registry
    sendToBridge({ type: 'REQUEST_TOOLS' });
  };

  socket.onclose = () => {
    const wasConnected = connected;
    setConnected(false);
    const dot = document.getElementById('conn-dot');
    if (dot) dot.classList.remove('connected');
    if (wasConnected) showToast('warning', `Disconnected — reconnecting in ${Math.round(_wsBackoff/1000)}s…`, _wsBackoff);
    setWsReconnectTimer(setTimeout(() => {
      setWsBackoff(Math.min(_wsBackoff * 2, _WS_BACKOFF_MAX));
      connectWS();
    }, _wsBackoff));
  };

  socket.onerror = () => {
    socket.close();
  };

  socket.onmessage = (evt) => {
    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }
    if (_messageHandler) _messageHandler(msg);
  };
}

export function sendToBridge(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

// ── Multi-agent helpers ───────────────────────────────────────────
export function agentColor(agentId) {
  if (!agentColorMap.has(agentId)) {
    agentColorMap.set(agentId, AGENT_COLORS[agentColorMap.size % AGENT_COLORS.length]);
  }
  return agentColorMap.get(agentId);
}

export function renderAgentTabs() {
  const bar = document.getElementById('agent-tabs');
  bar.innerHTML = '';
  if (agentList.length <= 1) { bar.classList.remove('visible'); return; }
  bar.classList.add('visible');
  for (const agent of agentList) {
    const btn = document.createElement('button');
    btn.className = 'agent-tab' + (agent.agent_id === focusedAgentId ? ' focused' : '');
    const dot = document.createElement('span');
    dot.className = 'agent-tab-indicator ' + (agent.status || 'active');
    btn.appendChild(dot);
    const label = document.createElement('span');
    label.textContent = agent.name || agent.agent_id;
    btn.appendChild(label);
    btn.addEventListener('click', () => focusAgent(agent.agent_id));
    bar.appendChild(btn);
  }
}

export function focusAgent(agentId) {
  if (agentId === focusedAgentId) return;
  setFocusedAgentId(agentId);
  renderAgentTabs();
  sendToBridge({ type: 'FOCUS_AGENT', agent_id: agentId });
  // Update subscription to include focused agent + global
  sendToBridge({ type: 'SUBSCRIBE', agents: [agentId], global: true });
}

export function handleAgentList(payload) {
  setAgentList(payload.agents || []);
  if (!focusedAgentId && agentList.length > 0) {
    setFocusedAgentId(agentList[0].agent_id);
  }
  renderAgentTabs();
}

export function handleAgentRegistered(payload) {
  const existing = agentList.find(a => a.agent_id === payload.agent_id);
  if (!existing) agentList.push(payload);
  if (!focusedAgentId && agentList.length > 0) {
    setFocusedAgentId(agentList[0].agent_id);
  }
  renderAgentTabs();
}

export function handleAgentUnregistered(payload) {
  setAgentList(agentList.filter(a => a.agent_id !== payload.agent_id));
  if (focusedAgentId === payload.agent_id) {
    setFocusedAgentId(agentList.length > 0 ? agentList[0].agent_id : null);
  }
  renderAgentTabs();
}

export function handleAgentStatus(payload) {
  const agent = agentList.find(a => a.agent_id === payload.agent_id);
  if (agent) agent.status = payload.status;
  renderAgentTabs();
}

export function isMultiAgent() { return agentList.length > 1; }
export function isFocusedAgent(msg) {
  if (!isMultiAgent()) return true;
  const aid = (msg.payload && msg.payload.agent_id) || msg.agent_id;
  return !aid || aid === focusedAgentId;
}
