import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const get = vi.fn();
vi.mock('../lib/react/api', () => ({
  api: { get, post: vi.fn(), put: vi.fn() },
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
  records: (value: unknown, key?: string) => {
    const rows = Array.isArray(value) ? value : key && value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : [];
    return Array.isArray(rows) ? rows.filter(row => row && typeof row === 'object') : [];
  },
}));

describe('workspace split panes and mobile preview', () => {
  let host: HTMLDivElement;
  let root: Root;
  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    get.mockReset().mockResolvedValue({ items: [] });
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); });
  async function renderWorkspace() {
    const { WorkspacePage } = await import('./WorkspacePage');
    await act(async () => { root.render(createElement(WorkspacePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
  }

  it('resizes files and details panes with bounded keyboard and pointer controls', async () => {
    await renderWorkspace();
    const files = host.querySelector<HTMLElement>('[aria-label="Resize project files pane"]')!;
    expect(files.getAttribute('aria-valuenow')).toBe('218');
    await act(async () => files.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true })));
    expect(files.getAttribute('aria-valuenow')).toBe('360');
    await act(async () => files.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true })));
    expect(files.getAttribute('aria-valuenow')).toBe('360');
    await act(async () => files.dispatchEvent(new KeyboardEvent('keydown', { key: 'Home', bubbles: true })));
    expect(files.getAttribute('aria-valuenow')).toBe('200');

    const details = host.querySelector<HTMLElement>('[aria-label="Resize document details pane"]')!;
    await act(async () => {
      const down = new MouseEvent('pointerdown', { button: 0, clientX: 500, bubbles: true });
      files.dispatchEvent(down);
      const move = new MouseEvent('pointermove', { clientX: 540, bubbles: true });
      window.dispatchEvent(move);
    });
    expect(files.getAttribute('aria-valuenow')).toBe('240');
    await act(async () => details.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true })));
    expect(details.getAttribute('aria-valuenow')).toBe('226');
  });

  it('returns from mobile preview to the editor without losing the draft', async () => {
    await renderWorkspace();
    const editor = host.querySelector<HTMLTextAreaElement>('#oa-file-content')!;
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
      setter?.call(editor, 'Draft survives preview'); editor.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => host.querySelectorAll<HTMLButtonElement>('[role="tab"]')[1]?.click());
    expect(host.querySelector<HTMLButtonElement>('[aria-label="Back to editor"]')).toBeTruthy();
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Back to editor"]')?.click());
    expect(host.querySelector<HTMLTextAreaElement>('#oa-file-content')?.value).toBe('Draft survives preview');
    expect(host.querySelectorAll<HTMLButtonElement>('[role="tab"]')[0]?.getAttribute('aria-selected')).toBe('true');
  });

  it('supports arrow and boundary key navigation across mobile editor tabs', async () => {
    await renderWorkspace();
    const tabs = host.querySelectorAll<HTMLButtonElement>('[role="tab"]');
    await act(async () => tabs[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true })));
    expect(tabs[1].getAttribute('aria-selected')).toBe('true');
    expect(document.activeElement).toBe(tabs[1]);
    await act(async () => tabs[1].dispatchEvent(new KeyboardEvent('keydown', { key: 'Home', bubbles: true, cancelable: true })));
    expect(tabs[0].getAttribute('aria-selected')).toBe('true');
    expect(document.activeElement).toBe(tabs[0]);
    expect(tabs[1].tabIndex).toBe(-1);
  });
});
