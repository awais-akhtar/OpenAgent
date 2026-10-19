export type ThemeMode = 'dark' | 'light' | 'system';
const key = 'openagent.ui.v2.theme';
const eventName = 'openagent:theme-changed';
function resolve(mode: ThemeMode): 'dark'|'light' { return mode === 'system' ? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : mode; }
function sync(mode: ThemeMode) {
  const resolved = resolve(mode);
  document.documentElement.dataset.theme = resolved;
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', resolved === 'dark' ? '#0e0e0e' : '#ffffff');
}
export const theme = {
  read(): ThemeMode {
    try { const value = localStorage.getItem(key); return value === 'dark' || value === 'light' ? value : 'system'; }
    catch { return 'system'; }
  },
  set(mode: ThemeMode) { try { localStorage.setItem(key, mode); } catch { /* Theme still applies for this session. */ } sync(mode); if (typeof window !== 'undefined') window.dispatchEvent(new CustomEvent(eventName, { detail: mode })); },
  init() {
    sync(this.read());
    const media = matchMedia('(prefers-color-scheme: dark)');
    const update = () => { if (this.read() === 'system') sync('system'); };
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  },
  eventName,
};
