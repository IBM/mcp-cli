// ================================================================
//  js/dispatcher.js — Bridge message routing (big switch statement)
// ================================================================
'use strict';

import { setCachedToolRegistry, setCachedPlanUpdate } from './state.js';
import {
  mergeViewRegistry, routeToViews, broadcastToViews,
  broadcastToViewType, sendToActivityStream,
} from './views.js';
import { rebuildAddPanelMenu } from './layout.js';
import {
  handleAgentList, handleAgentRegistered,
  handleAgentUnregistered, handleAgentStatus,
  isFocusedAgent,
} from './websocket.js';
import { handleAppLaunched, handleAppClosed } from './apps.js';
import { handleConfigState } from './config.js';
import { handleSessionList, handleSessionState } from './sessions.js';
import { handleToolApprovalRequest } from './approval.js';
import { applyTheme } from './theme.js';

export function handleBridgeMessage(msg) {
  switch (msg.type) {
    case 'VIEW_REGISTRY':
      // Merge dynamic views from bridge without clobbering builtins
      // Envelope format: msg.payload.views; legacy format: msg.views
      mergeViewRegistry((msg.payload && msg.payload.views) || msg.views || []);
      rebuildAddPanelMenu();
      break;

    // ── MCP App lifecycle ──────────────────────────────────────
    case 'APP_LAUNCHED':
      handleAppLaunched(msg.payload);
      break;
    case 'APP_CLOSED':
      handleAppClosed(msg.payload);
      break;

    // ── Agent lifecycle messages ───────────────────────────────
    case 'AGENT_LIST':
      handleAgentList(msg.payload);
      broadcastToViewType('agents', 'AGENT_LIST', msg.payload);
      break;

    case 'AGENT_REGISTERED':
      handleAgentRegistered(msg.payload);
      broadcastToViewType('agents', 'AGENT_REGISTERED', msg.payload);
      break;

    case 'AGENT_UNREGISTERED':
      handleAgentUnregistered(msg.payload);
      broadcastToViewType('agents', 'AGENT_UNREGISTERED', msg.payload);
      break;

    case 'AGENT_STATUS':
      handleAgentStatus(msg.payload);
      broadcastToViewType('agents', 'AGENT_STATUS', msg.payload);
      break;

    // ── Agent-scoped messages ────────────────────────────────
    case 'TOOL_RESULT':
      sendToActivityStream('TOOL_RESULT', msg.payload);
      if (isFocusedAgent(msg)) routeToViews('TOOL_RESULT', msg.payload);
      break;

    case 'AGENT_STATE':
      if (isFocusedAgent(msg)) broadcastToViews('AGENT_STATE', msg.payload);
      broadcastToViewType('agents', 'AGENT_STATE', msg.payload);
      break;

    case 'CONVERSATION_MESSAGE':
      sendToActivityStream('CONVERSATION_MESSAGE', msg.payload);
      if (isFocusedAgent(msg)) broadcastToViewType('conversation', 'CONVERSATION_MESSAGE', msg.payload);
      break;

    case 'CONVERSATION_TOKEN':
      if (isFocusedAgent(msg)) broadcastToViewType('conversation', 'CONVERSATION_TOKEN', msg.payload);
      break;

    case 'CONVERSATION_HISTORY':
      if (isFocusedAgent(msg)) broadcastToViewType('conversation', 'CONVERSATION_HISTORY', msg.payload);
      break;

    case 'ACTIVITY_HISTORY':
      sendToActivityStream('ACTIVITY_HISTORY', msg.payload);
      break;

    case 'CONFIG_STATE':
      handleConfigState(msg.payload);
      broadcastToViews('CONFIG_STATE', msg.payload);
      break;

    case 'TOOL_REGISTRY':
      setCachedToolRegistry(msg.payload);
      broadcastToViewType('tools', 'TOOL_REGISTRY', msg.payload);
      break;

    case 'TOOL_APPROVAL_REQUEST':
      handleToolApprovalRequest(msg.payload);
      break;

    case 'PLAN_UPDATE':
      setCachedPlanUpdate(msg.payload);
      broadcastToViewType('plan', 'PLAN_UPDATE', msg.payload);
      sendToActivityStream('PLAN_UPDATE', msg.payload);
      break;

    case 'SESSION_STATE':
      handleSessionState(msg.payload);
      break;

    case 'SESSION_LIST':
      handleSessionList(msg.payload);
      break;

    case 'THEME':
      applyTheme(msg.payload);
      break;

    default:
      break;
  }
}
