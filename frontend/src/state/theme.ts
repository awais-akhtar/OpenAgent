export type ThemeMode = 'dark' | 'light' | 'system';
const key = 'openagent.ui.v2.theme';
function resolve(mode: ThemeMode): 'dark'|'light' { return mode === 'system' ? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : mode; }
function sync(mode: ThemeMode) { const resolved=resolve(mode); document.documentElement.dataset.theme=resolved; document.querySelector('meta[name="theme-color"]')?.setAttribute('content',resolved==='dark'?'#0B1016':'#F6F8FA'); }
export const theme = { read(): ThemeMode { try { const m=localStorage.getItem(key);return m==='dark'||m==='light'?m:'system';}catch{return 'system';} }, set(mode:ThemeMode) { try { localStorage.setItem(key,mode); } catch {} sync(mode); }, init() { const mode=this.read();sync(mode);const media=matchMedia('(prefers-color-scheme: dark)');const update=()=>{if(this.read()==='system')sync('system');};media.addEventListener('change',update);return()=>media.removeEventListener('change',update); } };
