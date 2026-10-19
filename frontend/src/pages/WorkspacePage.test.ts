import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const get = vi.fn();
const post = vi.fn();
vi.mock('../lib/react/api', () => ({
  api: { get, post, put: vi.fn() },
  ApiError: class ApiError extends Error { constructor(message: string, readonly status: number) { super(message); } },
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
    post.mockReset().mockResolvedValue({});
    history.replaceState(null, '', '/?view=workspace&project=workspace-test');
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); history.replaceState(null, '', '/'); });
  async function renderWorkspace() {
    const { WorkspacePage } = await import('./WorkspacePage');
    await act(async () => { root.render(createElement(WorkspacePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
  }
  async function createDraft() { await act(async () => host.querySelector<HTMLButtonElement>('.oa-workspace-create button')?.click()); }

  it('resizes the file library pane with bounded keyboard and pointer controls', async () => {
    await renderWorkspace();
    await createDraft();
    const files = host.querySelector<HTMLElement>('[aria-label="Resize project files pane"]')!;
    expect(files.getAttribute('aria-valuenow')).toBe('430');
    await act(async () => files.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true })));
    expect(files.getAttribute('aria-valuenow')).toBe('420');
    await act(async () => files.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true })));
    expect(files.getAttribute('aria-valuenow')).toBe('420');
    await act(async () => files.dispatchEvent(new KeyboardEvent('keydown', { key: 'Home', bubbles: true })));
    expect(files.getAttribute('aria-valuenow')).toBe('220');

    await act(async () => {
      const down = new MouseEvent('pointerdown', { button: 0, clientX: 500, bubbles: true });
      files.dispatchEvent(down);
      const move = new MouseEvent('pointermove', { clientX: 540, bubbles: true });
      window.dispatchEvent(move);
    });
    expect(files.getAttribute('aria-valuenow')).toBe('260');
  });

  it('returns from mobile preview to the editor without losing the draft', async () => {
    await renderWorkspace();
    await createDraft();
    const editor = host.querySelector<HTMLTextAreaElement>('#oa-file-content')!;
    const draft = '# Preview title\n\n[unsafe](javascript:alert(1)) and [reference](https://example.com/source).';
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
      setter?.call(editor, draft); editor.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => host.querySelector<HTMLButtonElement>('[data-pane-tab="preview"]')?.click());
    expect(host.querySelector('.oa-document-preview h3')?.textContent).toBe('Preview title');
    expect(host.querySelector('.oa-document-preview a[href^="javascript:"]')).toBeNull();
    expect(host.querySelector('.oa-document-preview a[href="https://example.com/source"]')?.getAttribute('rel')).toBe('noopener noreferrer');
    await act(async () => host.querySelector<HTMLButtonElement>('[data-pane-tab="edit"]')?.click());
    expect(host.querySelector<HTMLTextAreaElement>('#oa-file-content')?.value).toBe(draft);
    expect(host.querySelector<HTMLButtonElement>('[data-pane-tab="edit"]')?.getAttribute('aria-selected')).toBe('true');
  });

  it('supports arrow and boundary key navigation across mobile editor tabs', async () => {
    await renderWorkspace();
    await createDraft();
    const tabs = host.querySelectorAll<HTMLButtonElement>('[role="tab"]');
    await act(async () => tabs[1].dispatchEvent(new KeyboardEvent('keydown', { key: 'Home', bubbles: true, cancelable: true })));
    expect(tabs[0].getAttribute('aria-selected')).toBe('true');
    expect(document.activeElement).toBe(tabs[0]);
    await act(async () => tabs[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true })));
    expect(tabs[1].getAttribute('aria-selected')).toBe('true');
    expect(document.activeElement).toBe(tabs[1]);
    expect(tabs[2].tabIndex).toBe(-1);
  });

  it('keeps the document title and editor actions in separate toolbar groups', async () => {
    await renderWorkspace();
    await createDraft();
    const toolbar = host.querySelector<HTMLElement>('.oa-file-toolbar');
    const title = toolbar?.querySelector<HTMLInputElement>('.oa-file-title input');
    const actions = toolbar?.querySelector<HTMLElement>('.oa-file-actions');
    expect(toolbar?.children).toHaveLength(2);
    expect(title?.getAttribute('aria-label')).toBe('Document title');
    expect(actions?.querySelector('[aria-label="Document format"]')).toBeTruthy();
    expect(actions?.querySelector('button')).toBeTruthy();
  });

  it('starts file-first with one create action and no empty editor or details panes', async () => {
    await renderWorkspace();
    expect(host.querySelector('.oa-file-editor')).toBeNull();
    expect(host.querySelector('.oa-file-inspector')).toBeNull();
    expect(host.querySelector('.oa-file-splitter-details')).toBeNull();
    expect(host.querySelector('.oa-workspace-empty h2')?.textContent).toBe('No document selected');
    expect(host.querySelector('.oa-workspace-empty p')?.textContent).toContain('Choose a project file');
    expect(host.querySelectorAll('.oa-workspace-create button')).toHaveLength(1);
    expect(host.querySelector('.oa-workspace-create')?.textContent).toContain('Create document');
    await act(async () => host.querySelector<HTMLButtonElement>('.oa-workspace-create button')?.click());
    expect(host.querySelector('.oa-workspace-create')).toBeNull();
    expect(host.querySelector('.oa-file-editor')).toBeTruthy();
    expect(host.querySelector<HTMLButtonElement>('.oa-file-actions .primary')?.textContent).toBe('Create');
  });

  it('opens the first real project document in its safe preview', async () => {
    get.mockImplementation(async (path: string) => {
      if (path === '/api/documents/library?project=workspace-test&q=&limit=100') return { items: [{ doc_id: 'doc-first', title: 'Project brief', format: 'markdown', project: 'workspace-test' }] };
      if (path === '/api/document/doc-first') return { document: { doc_id: 'doc-first', title: 'Project brief', format: 'markdown', content: '# Real project brief', project: 'workspace-test', version: 1 } };
      if (path === '/api/document/doc-first/versions') return { versions: [] };
      return { items: [] };
    });
    await renderWorkspace();
    await act(async () => new Promise(resolve => setTimeout(resolve, 220)));
    expect(host.textContent).toContain('Project brief');
    expect(host.querySelector('.oa-file-content')?.classList.contains('pane-preview')).toBe(true);
    expect(host.querySelector('.oa-document-preview h3')?.textContent).toBe('Real project brief');
    expect(host.querySelector<HTMLButtonElement>('[data-pane-tab="preview"]')?.getAttribute('aria-selected')).toBe('true');
    expect(get).toHaveBeenCalledWith('/api/document/doc-first');
  });

  it('keeps the true loading state visible and explains a permission-denied library read', async () => {
    get.mockImplementation(async (path: string) => path.startsWith('/api/workspace/items') ? Promise.reject(new (await import('../lib/react/api')).ApiError('Forbidden', 403)) : { items: [] });
    await renderWorkspace();
    expect(host.querySelector('.oa-files-rows [role="status"]')?.textContent).toContain('Loading files');
    await act(async () => new Promise(resolve => setTimeout(resolve, 220)));
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('do not have permission');
    expect(host.querySelector('.oa-files-retry')?.textContent).toContain('Retry');
  });

  it('imports a selected file through the project-scoped Python document endpoint', async () => {
    get.mockImplementation(async (path: string) => {
      if (path === '/api/document/imported-1') return { document: { doc_id: 'imported-1', title: 'Notes', content: 'Imported text', version: 1 } };
      if (path === '/api/document/imported-1/versions') return { versions: [] };
      return { items: [] };
    });
    post.mockResolvedValue({ document: { doc_id: 'imported-1', title: 'Notes', project: 'workspace-test', version: 1 }, chars_extracted: 14 });
    await renderWorkspace();
    const input = host.querySelector<HTMLInputElement>('.oa-files-list input[type="file"]')!;
    const file = new File(['Imported text'], 'notes.txt', { type: 'text/plain' });
    Object.defineProperty(file, 'arrayBuffer', { value: async () => new TextEncoder().encode('Imported text').buffer });
    Object.defineProperty(input, 'files', { configurable: true, value: [file] });
    await act(async () => { input.dispatchEvent(new Event('change', { bubbles: true })); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(post).toHaveBeenCalledWith('/api/documents/import', { filename: 'notes.txt', title: 'notes', project: 'workspace-test', content_base64: 'SW1wb3J0ZWQgdGV4dA==' });
    expect(host.querySelector<HTMLTextAreaElement>('#oa-file-content')?.value).toBe('Imported text');
    expect(host.textContent).toContain('imported to this project');
  });

  it('clears an open document when switching project scope and loads only the new project library', async () => {
    get.mockImplementation(async (path: string) => {
      if (path === '/api/documents/library?project=alpha&q=&limit=100') return { items: [{ doc_id: 'doc-alpha', title: 'Alpha private note', project: 'alpha', version: 1 }] };
      if (path === '/api/document/doc-alpha') return { document: { doc_id: 'doc-alpha', title: 'Alpha private note', content: 'private alpha content', project: 'alpha', version: 1 } };
      if (path === '/api/document/doc-alpha/versions') return { versions: [] };
      return { items: [] };
    });
    history.replaceState(null, '', '/?view=workspace&project=alpha');
    await renderWorkspace();
    await act(async () => new Promise(resolve => setTimeout(resolve, 220)));
    const file = [...host.querySelectorAll<HTMLButtonElement>('.oa-files-rows button')].find(button => button.textContent?.includes('Alpha private note'));
    expect(file).toBeTruthy();
    await act(async () => { file?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector<HTMLTextAreaElement>('#oa-file-content')?.value).toBe('private alpha content');

    await act(async () => root.unmount());
    root = createRoot(host);
    history.pushState(null, '', '/?view=workspace&project=beta');
    // App.tsx keys the page subtree by project, so switching scope remounts it.
    await renderWorkspace();
    await act(async () => new Promise(resolve => setTimeout(resolve, 220)));
    expect(host.querySelector<HTMLTextAreaElement>('#oa-file-content')).toBeNull();
    expect(host.querySelector('.oa-workspace-create')).toBeTruthy();
    expect(host.textContent).not.toContain('Alpha private note');
    expect(get).toHaveBeenCalledWith('/api/workspace/items?project=beta&q=&limit=100', expect.any(AbortSignal));
    expect(get).toHaveBeenCalledWith('/api/documents/library?project=beta&q=&limit=100', expect.any(AbortSignal));
  });
});
