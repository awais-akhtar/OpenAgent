import { afterEach, describe, expect, it, vi } from 'vitest';
import { accessibility } from './accessibility';

afterEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute('data-reduced-motion');
  vi.unstubAllGlobals();
});

describe('reduced-motion preference', () => {
  it('persists and applies explicit local preferences without a server request', () => {
    accessibility.set('reduce');
    expect(accessibility.read()).toBe('reduce');
    expect(document.documentElement.dataset.reducedMotion).toBe('reduce');
    expect(accessibility.shouldReduceMotion('reduce', false)).toBe(true);

    accessibility.set('allow');
    expect(accessibility.read()).toBe('allow');
    expect(document.documentElement.dataset.reducedMotion).toBe('allow');
    expect(accessibility.shouldReduceMotion('allow', true)).toBe(false);
  });

  it('follows the operating system only while the preference is system', () => {
    expect(accessibility.shouldReduceMotion('system', true)).toBe(true);
    expect(accessibility.shouldReduceMotion('system', false)).toBe(false);
    expect(accessibility.shouldReduceMotion('reduce', false)).toBe(true);
    expect(accessibility.shouldReduceMotion('allow', true)).toBe(false);
  });

  it('notifies the shell when the local preference changes', () => {
    const listener = vi.fn();
    window.addEventListener(accessibility.eventName, listener);
    accessibility.set('reduce');
    expect(listener).toHaveBeenCalledTimes(1);
    expect((listener.mock.calls[0][0] as CustomEvent).detail).toBe('reduce');
    window.removeEventListener(accessibility.eventName, listener);
  });
});
