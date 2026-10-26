import { create } from 'zustand';

export type ThemeMode = 'light' | 'dark' | 'system';
export type DataMode = 'demo' | 'live';
const STORAGE_THEME = 'oa-second-theme';
const STORAGE_REDUCE_MOTION = 'oa-second-reduce-motion';
const STORAGE_DATA_MODE = 'oa-second-data-mode';
const STORAGE_DEMO_PROJECT = 'oa-second-demo-project';

function resolveTheme(theme: ThemeMode): 'light' | 'dark' {
  if (theme !== 'system') return theme;
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function readBoolean(key: string, fallback: boolean) {
  try {
    const stored = localStorage.getItem(key);
    return stored === null ? fallback : stored === 'true';
  } catch { return fallback; }
}

type WorkspaceState = {
  collapsed: boolean;
  theme: ThemeMode;
  reduceMotion: boolean;
  dataMode: DataMode;
  selectedDemoProject: string;
  demoProjectCreateRequest: number;
  setCollapsed: (collapsed: boolean) => void;
  toggleCollapsed: () => void;
  setTheme: (theme: ThemeMode) => void;
  setReduceMotion: (reduceMotion: boolean) => void;
  setDataMode: (mode: DataMode) => void;
  setSelectedDemoProject: (name: string) => void;
  requestDemoProjectCreate: () => void;
  consumeDemoProjectCreateRequest: (requestId: number) => void;
};

function readDemoProject() {
  try { return localStorage.getItem(STORAGE_DEMO_PROJECT) || 'Personal Workspace'; }
  catch { return 'Personal Workspace'; }
}

const readDataMode = (): DataMode => {
  // Demo data is an explicit local development choice; deployed builds always start live.
  if (!import.meta.env.DEV) {
    // A same-origin production upgrade may inherit a development Demo flag.
    // Production has no Demo mode, so remove the stale preference where
    // storage is available and remain Live even when storage is denied.
    try { localStorage.removeItem(STORAGE_DATA_MODE); } catch { /* Production mode remains Live in memory. */ }
    return 'live';
  }
  try {
    const stored = localStorage.getItem(STORAGE_DATA_MODE);
    if (stored === 'live') return 'live';
    if (stored !== 'demo') localStorage.setItem(STORAGE_DATA_MODE, 'demo');
    return 'demo';
  } catch { return 'demo'; }
};

const readTheme = (): ThemeMode => {
  try {
    const stored = localStorage.getItem(STORAGE_THEME);
    return stored === 'dark' || stored === 'light' || stored === 'system' ? stored : 'light';
  } catch {
    return 'light';
  }
};

export const useWorkspaceState = create<WorkspaceState>((set) => ({
  collapsed: false,
  theme: readTheme(),
  reduceMotion: readBoolean(STORAGE_REDUCE_MOTION, false),
  dataMode: readDataMode(),
  selectedDemoProject: readDemoProject(),
  demoProjectCreateRequest: 0,
  setCollapsed: (collapsed) => set({ collapsed }),
  toggleCollapsed: () => set((state) => ({ collapsed: !state.collapsed })),
  setTheme: (theme) => {
    try { localStorage.setItem(STORAGE_THEME, theme); } catch { /* Theme still updates for this session. */ }
    const resolved = resolveTheme(theme);
    if (typeof document !== 'undefined') document.documentElement.dataset.theme = resolved;
    set({ theme });
  },
  setReduceMotion: (reduceMotion) => {
    try { localStorage.setItem(STORAGE_REDUCE_MOTION, String(reduceMotion)); } catch { /* Session preference still applies. */ }
    set({ reduceMotion });
  },
  setDataMode: (dataMode) => {
    const resolvedMode = import.meta.env.DEV ? dataMode : 'live';
    try {
      if (resolvedMode === 'demo') localStorage.setItem(STORAGE_DATA_MODE, 'demo');
      else localStorage.removeItem(STORAGE_DATA_MODE);
    } catch { /* The selected mode remains available in memory for this session. */ }
    set({ dataMode: resolvedMode });
    if (typeof window !== 'undefined') window.dispatchEvent(new CustomEvent('openagent-data-mode-changed', { detail: resolvedMode }));
  },
  setSelectedDemoProject: (selectedDemoProject) => {
    const normalized = selectedDemoProject.trim() || 'Personal Workspace';
    try { localStorage.setItem(STORAGE_DEMO_PROJECT, normalized); } catch { /* The selected project remains available in memory for this session. */ }
    set({ selectedDemoProject: normalized });
  },
  requestDemoProjectCreate: () => set(state => ({ demoProjectCreateRequest: state.demoProjectCreateRequest + 1 })),
  consumeDemoProjectCreateRequest: requestId => set(state => state.demoProjectCreateRequest === requestId ? { demoProjectCreateRequest: 0 } : state),
}));

export { resolveTheme };
