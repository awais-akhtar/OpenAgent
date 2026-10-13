import { afterEach, describe, expect, it, vi } from 'vitest';
import { theme } from './theme';

afterEach(() => { localStorage.clear(); document.documentElement.removeAttribute('data-theme'); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('theme preference', () => {
  it('persists explicit dark and light modes and updates the root semantic mode', () => {
    theme.set('dark'); expect(theme.read()).toBe('dark'); expect(document.documentElement.dataset.theme).toBe('dark');
    theme.set('light'); expect(theme.read()).toBe('light'); expect(document.documentElement.dataset.theme).toBe('light');
  });

  it('resolves system mode from the operating system and follows changes until overridden', () => {
    let listener: ((event: MediaQueryListEvent) => void) | undefined;
    const media = { matches: true, addEventListener: (_: string, cb: (event: MediaQueryListEvent) => void) => { listener = cb; }, removeEventListener: vi.fn() } as unknown as MediaQueryList;
    vi.stubGlobal('matchMedia', vi.fn(() => media));
    theme.set('system'); expect(document.documentElement.dataset.theme).toBe('dark');
    const cleanup = theme.init();
    Object.defineProperty(media, 'matches', { value: false, configurable: true }); listener?.({} as MediaQueryListEvent);
    expect(document.documentElement.dataset.theme).toBe('light');
    theme.set('dark'); Object.defineProperty(media, 'matches', { value: true, configurable: true }); listener?.({} as MediaQueryListEvent);
    expect(document.documentElement.dataset.theme).toBe('dark');
    cleanup(); expect(media.removeEventListener).toHaveBeenCalled();
  });
});
