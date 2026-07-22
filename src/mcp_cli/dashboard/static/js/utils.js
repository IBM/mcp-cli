// ================================================================
//  js/utils.js — HTML escape, toast, drag utility, drawer helpers
// ================================================================
'use strict';

export function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

export function showToast(level, message, duration) {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast ${level}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), duration || 5000);
}

export function makeDraggable(el, callbacks) {
  // Mouse
  el.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const state = callbacks.onStart(e.clientX, e.clientY);
    if (!state) return;
    function onMove(e) { callbacks.onMove(state, e.clientX, e.clientY); }
    function onUp() {
      callbacks.onEnd(state);
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
  // Touch
  el.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const state = callbacks.onStart(touch.clientX, touch.clientY);
    if (!state) return;
    function onMove(e) {
      const t = e.touches[0];
      callbacks.onMove(state, t.clientX, t.clientY);
    }
    function onEnd() {
      callbacks.onEnd(state);
      document.removeEventListener('touchmove', onMove);
      document.removeEventListener('touchend', onEnd);
    }
    document.addEventListener('touchmove', onMove, { passive: false });
    document.addEventListener('touchend', onEnd);
  }, { passive: false });
}

export function openDrawer(el) {
  el.classList.add('open');
  if (window.innerWidth < 768) {
    document.getElementById('drawer-backdrop').classList.add('visible');
  }
}

export function closeDrawer(el) {
  el.classList.remove('open');
  document.getElementById('drawer-backdrop').classList.remove('visible');
}

// Late-binding for closeSidebar to avoid circular dependency with sidebar.js
let _closeSidebarFn = null;
export function setCloseSidebarFn(fn) {
  _closeSidebarFn = fn;
}

export function closeAllDrawers() {
  document.getElementById('settings-panel').classList.remove('open');
  document.getElementById('session-drawer').classList.remove('open');
  if (_closeSidebarFn) _closeSidebarFn();
  document.getElementById('drawer-backdrop').classList.remove('visible');
}
