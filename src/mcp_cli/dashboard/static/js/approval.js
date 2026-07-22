// ================================================================
//  js/approval.js — Tool approval dialog
// ================================================================
'use strict';

import {
  _pendingApprovalCallId,
  setPendingApprovalCallId,
} from './state.js';
import { sendToBridge } from './websocket.js';
import { showToast } from './utils.js';

export function handleToolApprovalRequest(payload) {
  // If another approval is already showing, deny it first (auto-deny stale)
  if (_pendingApprovalCallId) {
    sendToBridge({ type: 'TOOL_APPROVAL_RESPONSE', call_id: _pendingApprovalCallId, approved: false });
  }
  setPendingApprovalCallId(payload.call_id || '');
  document.getElementById('approval-tool-name').textContent = payload.tool_name || 'unknown';
  try {
    document.getElementById('approval-args').textContent =
      JSON.stringify(payload.arguments, null, 2);
  } catch {
    document.getElementById('approval-args').textContent = String(payload.arguments || '{}');
  }
  document.getElementById('approval-overlay').classList.add('open');
  showToast('warning', `Approval needed: ${payload.tool_name}`);
}

export function wireApprovalEvents() {
  document.getElementById('approval-approve').addEventListener('click', () => {
    if (_pendingApprovalCallId) {
      sendToBridge({ type: 'TOOL_APPROVAL_RESPONSE', call_id: _pendingApprovalCallId, approved: true });
    }
    document.getElementById('approval-overlay').classList.remove('open');
    setPendingApprovalCallId(null);
  });

  document.getElementById('approval-deny').addEventListener('click', () => {
    if (_pendingApprovalCallId) {
      sendToBridge({ type: 'TOOL_APPROVAL_RESPONSE', call_id: _pendingApprovalCallId, approved: false });
    }
    document.getElementById('approval-overlay').classList.remove('open');
    setPendingApprovalCallId(null);
  });
}
