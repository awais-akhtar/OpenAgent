export type MotionPreference = 'system' | 'reduce' | 'allow';

const storageKey = 'openagent.ui.v2.reduced-motion';
const eventName = 'openagent:accessibility-changed';

function systemPrefersReducedMotion() {
  try { return typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches; }
  catch { return false; }
}

function apply(preference: MotionPreference) {
  if (typeof document !== 'undefined') document.documentElement.dataset.reducedMotion = preference;
}

export const accessibility = {
  read(): MotionPreference {
    try {
      const value = localStorage.getItem(storageKey);
      return value === 'reduce' || value === 'allow' ? value : 'system';
    } catch { return 'system'; }
  },
  shouldReduceMotion(preference: MotionPreference = accessibility.read(), systemPreference = systemPrefersReducedMotion()) {
    return preference === 'reduce' || (preference === 'system' && systemPreference);
  },
  set(preference: MotionPreference) {
    try { localStorage.setItem(storageKey, preference); } catch { /* Apply for this session if storage is unavailable. */ }
    apply(preference);
    if (typeof window !== 'undefined') window.dispatchEvent(new CustomEvent(eventName, { detail: preference }));
  },
  init() {
    const current = this.read();
    apply(current);
    if (typeof window === 'undefined') return () => undefined;
    let media: MediaQueryList | undefined;
    try { media = window.matchMedia('(prefers-reduced-motion: reduce)'); } catch { /* CSS and explicit preference still apply. */ }
    const update = () => { if (this.read() === 'system') apply('system'); };
    media?.addEventListener('change', update);
    return () => media?.removeEventListener('change', update);
  },
  eventName,
};
