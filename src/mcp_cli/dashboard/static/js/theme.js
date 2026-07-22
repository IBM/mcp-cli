// ================================================================
//  js/theme.js — Theme loading, applying, and select widget
// ================================================================
'use strict';

import {
  themes, activeTheme,
  setThemes, setActiveTheme,
} from './state.js';
import { broadcastToViews } from './views.js';

export async function loadThemes() {
  try {
    const resp = await fetch('/themes/themes.json');
    setThemes(await resp.json());
  } catch {
    setThemes({});
  }
  buildThemeSelect();
  applyTheme(themes[activeTheme] || themes['dark'] || {});
}

export function themeToCSS(t) {
  return {
    name: t.name,
    bg: t.bg, bg_surface: t.bg_surface, bg_hover: t.bg_hover,
    fg: t.fg, fg_muted: t.fg_muted,
    accent: t.accent, success: t.success, warning: t.warning,
    error: t.error, info: t.info, border: t.border,
    font_mono: t.font_mono, font_ui: t.font_ui,
    font_size: t.font_size, radius: t.radius, spacing: t.spacing,
  };
}

export function applyTheme(t) {
  if (!t || !t.bg) return;
  setActiveTheme(t.name || 'dark');
  localStorage.setItem('dash-theme', activeTheme);

  const root = document.documentElement;
  root.style.setProperty('--dash-bg',         t.bg || '');
  root.style.setProperty('--dash-bg-surface',  t.bg_surface || '');
  root.style.setProperty('--dash-bg-hover',    t.bg_hover || '');
  root.style.setProperty('--dash-fg',          t.fg || '');
  root.style.setProperty('--dash-fg-muted',    t.fg_muted || '');
  root.style.setProperty('--dash-accent',      t.accent || '');
  root.style.setProperty('--dash-success',     t.success || '');
  root.style.setProperty('--dash-warning',     t.warning || '');
  root.style.setProperty('--dash-error',       t.error || '');
  root.style.setProperty('--dash-info',        t.info || '');
  root.style.setProperty('--dash-border',      t.border || '');
  root.style.setProperty('--dash-font-mono',   t.font_mono || '');
  root.style.setProperty('--dash-font-ui',     t.font_ui || '');
  root.style.setProperty('--dash-font-size',   t.font_size || '');
  root.style.setProperty('--dash-radius',      t.radius || '');
  root.style.setProperty('--dash-spacing',     t.spacing || '');

  // Update theme select
  const sel = document.getElementById('theme-select');
  if (sel) sel.value = activeTheme;

  // Propagate THEME to all views
  broadcastToViews('THEME', themeToCSS(t));
}

let _themeChangeHandler = null;
export function buildThemeSelect() {
  const sel = document.getElementById('theme-select');
  if (!sel) return;
  sel.innerHTML = '';
  for (const name of Object.keys(themes)) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name.charAt(0).toUpperCase() + name.slice(1);
    sel.appendChild(opt);
  }
  sel.value = activeTheme;
  // Remove previous listener to avoid stacking
  if (_themeChangeHandler) sel.removeEventListener('change', _themeChangeHandler);
  _themeChangeHandler = () => { applyTheme(themes[sel.value] || {}); };
  sel.addEventListener('change', _themeChangeHandler);
}
