import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const get = vi.fn();
const post = vi.fn();
const getEvents = vi.fn();
vi.mock('../lib/react/api', () => ({
  api: { get, post },
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
  records: (value: unknown, key?: string) => {
    const rows = Array.isArray(value) ? value : key && value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : [];
    return Array.isArray(rows) ? rows.filter(row => row && typeof row === 'object') : [];
  },
}));
vi.mock('../lib/react/stream', () => ({ getEvents }));

describe('CodeQ repository-first workspace', () => {
  let host: HTMLDivElement;
  let root: Root;
  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    localStorage.setItem('oa-project', 'codeq-test');
    history.replaceState({}, '', '/?view=codebase&project=codeq-test');
    get.mockReset().mockImplementation(async (path: string) => {
      if (path.startsWith('/api/codebase/projects?')) return { projects: [] };
      if (path.startsWith('/api/codebase/runs?')) return { runs: [] };
      if (path.endsWith('/tree')) return { entries: [{ path: 'src', type: 'directory' }, { path: 'src/main.py', type: 'file' }] };
      if (path.endsWith('/file?path=src%2Fmain.py')) return { file: { content: 'print("before")', sha256: 'version-1' } };
      return {};
    });
    post.mockReset().mockImplementation(async (path: string) => {
      if (path === '/api/codebase/folder-picker') return { path: 'C:\\Projects\\Sample' };
      if (path === '/api/codebase/projects') return { project: { id: 'repo-1', name: 'Sample' } };
      if (path === '/api/codebase/projects/repo-1/file') return { change: { new_sha256: 'version-2' } };
      return {};
    });
    getEvents.mockReset().mockImplementation(async function* () {});
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); localStorage.clear(); history.replaceState({}, '', '/'); });

  it('replaces empty editor/run panels with a repository setup action', async () => {
    const { CodebasePage } = await import('./CodebasePage');
    await act(async () => { root.render(createElement(CodebasePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector('.codeq-onboarding h2')?.textContent).toBe('Open a project');
    expect(host.querySelector('.codeq-editor')).toBeNull();
    expect(host.querySelector('.codeq-task')).toBeNull();
    expect(host.querySelector('.codeq-runs')).toBeNull();
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent === 'Choose local folder')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(post).toHaveBeenCalledWith('/api/codebase/folder-picker', {});
  });

  it('opens a selected repository before showing files and keeps source writes versioned and confirmed', async () => {
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/codebase/projects?')) return { projects: [{ id: 'repo-1', name: 'Sample' }] };
      if (path.startsWith('/api/codebase/runs?')) return { runs: [] };
      if (path.endsWith('/tree')) return { entries: [{ path: 'src', type: 'directory' }, { path: 'src/main.py', type: 'file' }] };
      if (path.endsWith('/file?path=src%2Fmain.py')) return { file: { content: 'print("before")', sha256: 'version-1' } };
      return {};
    });
    const { CodebasePage } = await import('./CodebasePage');
    await act(async () => { root.render(createElement(CodebasePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Sample'))?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector('.codeq-editor')).toBeTruthy();
    expect(host.querySelector('.codeq-task')).toBeTruthy();
    expect(host.querySelector('[role="tablist"][aria-label="CodeQ workspace panels"]')).toBeTruthy();
    expect(host.querySelector('.codeq-workspace.has-project')?.getAttribute('data-active-pane')).toBe('files');
    await act(async () => { [...host.querySelectorAll('[role="tab"]')].find(tab => tab.textContent === 'CodeQ')?.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
    expect(host.querySelector('.codeq-workspace.has-project')?.getAttribute('data-active-pane')).toBe('assistant');
    const file = host.querySelector<HTMLButtonElement>('button[aria-label="Open src/main.py"]');
    await act(async () => { file?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector('.codeq-tree-folder')?.textContent).toContain('src');
    const editor = host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Edit src/main.py"]');
    await act(async () => { if (editor) { const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set; setter?.call(editor, 'print("after")'); editor.dispatchEvent(new Event('input', { bubbles: true })); } });
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent === 'Write to source…')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(confirm).toHaveBeenCalled();
    expect(post).toHaveBeenCalledWith('/api/codebase/projects/repo-1/file', { path: 'src/main.py', content: 'print("after")', expected_sha256: 'version-1' });
    expect(host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Edit src/main.py"]')?.value).toBe('print("after")');
    expect(host.textContent).toContain('Source file saved.');
    expect([...host.querySelectorAll('button')].some(button => button.textContent === 'Write to source…')).toBe(false);
    confirm.mockRestore();
  });

  it('asks before leaving an edited protected-copy file and retains the draft when cancelled', async () => {
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/codebase/projects?')) return { projects: [{ id: 'repo-1', name: 'Sample' }] };
      if (path.startsWith('/api/codebase/runs?')) return { runs: [] };
      if (path.endsWith('/tree')) return { entries: [{ path: 'src/one.py', type: 'file' }, { path: 'src/two.py', type: 'file' }] };
      if (path.endsWith('/file?path=src%2Fone.py')) return { file: { content: 'one', sha256: 'hash-one' } };
      if (path.endsWith('/file?path=src%2Ftwo.py')) return { file: { content: 'two', sha256: 'hash-two' } };
      return {};
    });
    const { CodebasePage } = await import('./CodebasePage');
    await act(async () => { root.render(createElement(CodebasePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Sample'))?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Open src/one.py"]')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    const editor = host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Edit src/one.py"]');
    await act(async () => { if (editor) { const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set; setter?.call(editor, 'draft'); editor.dispatchEvent(new Event('input', { bubbles: true })); } });
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Open src/two.py"]')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('Discard the unsaved editor changes'));
    expect(host.querySelector('textarea[aria-label="Edit src/one.py"]')).toBeTruthy();
    expect(host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Edit src/one.py"]')?.value).toBe('draft');
    expect(get).not.toHaveBeenCalledWith('/api/codebase/projects/repo-1/file?path=src%2Ftwo.py', undefined);
    confirm.mockRestore();
  });

  it('refreshes the real project tree without dropping an open draft', async () => {
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/codebase/projects?')) return { projects: [{ id: 'repo-1', name: 'Sample' }] };
      if (path.startsWith('/api/codebase/runs?')) return { runs: [] };
      if (path === '/api/codebase/projects/repo-1') return { project: { id: 'repo-1', name: 'Sample' } };
      if (path.endsWith('/tree')) return { entries: [{ path: 'src/one.py', type: 'file' }, { path: 'src/two.py', type: 'file' }] };
      if (path.endsWith('/file?path=src%2Fone.py')) return { file: { content: 'one', sha256: 'hash-one' } };
      return {};
    });
    const { CodebasePage } = await import('./CodebasePage');
    await act(async () => { root.render(createElement(CodebasePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Sample'))?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Open src/one.py"]')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    const editor = host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Edit src/one.py"]');
    await act(async () => { if (editor) { const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set; setter?.call(editor, 'unsaved draft'); editor.dispatchEvent(new Event('input', { bubbles: true })); } });
    const confirm = vi.spyOn(window, 'confirm');
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Rescan'))?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(post).toHaveBeenCalledWith('/api/codebase/projects/repo-1/scan', {});
    expect(get).toHaveBeenCalledWith('/api/codebase/projects/repo-1/tree');
    expect(confirm).not.toHaveBeenCalled();
    expect(host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Edit src/one.py"]')?.value).toBe('unsaved draft');
    confirm.mockRestore();
  });

  it('runs preflight before starting a task and keeps external access disabled', async () => {
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/codebase/projects?')) return { projects: [{ id: 'repo-1', name: 'Sample' }] };
      if (path.startsWith('/api/codebase/runs?')) return { runs: [] };
      if (path.endsWith('/tree')) return { entries: [] };
      if (path === '/api/codebase/runs/run-1') return { run: { id: 'run-1', codebase_project_id: 'repo-1', status: 'completed' } };
      if (path.endsWith('/events?after=0')) return { events: [] };
      return {};
    });
    post.mockImplementation(async (path: string) => path === '/api/codebase/preflight' ? { preflight: { summary: 'Workspace and framework detected.', profile: { framework: 'react-vite' }, local_validation: 'ready', host: { system: 'Linux', supported: true }, toolchains: [{ alternatives: ['node', 'npm'], available: true }], external_boundaries: [{ boundary: 'deployment', status: 'approval_required' }] } } : path === '/api/codebase/runs' ? { run: { id: 'run-1', status: 'completed' } } : {});
    const { CodebasePage } = await import('./CodebasePage');
    await act(async () => { root.render(createElement(CodebasePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Sample'))?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    const task = host.querySelector<HTMLTextAreaElement>('.codeq-task textarea');
    await act(async () => { if (task) { const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set; setter?.call(task, 'Add coverage for parser'); task.dispatchEvent(new Event('input', { bubbles: true })); } });
    await act(async () => { task?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', ctrlKey: true, bubbles: true })); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(post).toHaveBeenCalledWith('/api/codebase/preflight', { codebase_project_id: 'repo-1', goal: 'Add coverage for parser', acceptance_criteria: [] });
    expect(host.textContent).toContain('Workspace and framework detected.');
    expect(host.textContent).toContain('react-vite');
    expect(host.textContent).toContain('Linux · Supported');
    expect(host.textContent).toContain('node / npm · Available');
    expect(host.textContent).toContain('deployment · approval_required');
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent === 'Start reviewed run')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(post).toHaveBeenCalledWith('/api/codebase/runs', { codebase_project_id: 'repo-1', prompt: 'Add coverage for parser', provider: 'auto', allow_external: false });
  });

  it('searches protected project text through the read-only grep tool and opens a real match', async () => {
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/codebase/projects?')) return { projects: [{ id: 'repo-1', name: 'Sample' }] };
      if (path.startsWith('/api/codebase/runs?')) return { runs: [] };
      if (path === '/api/codebase/projects/repo-1') return { project: { id: 'repo-1', name: 'Sample' } };
      if (path.endsWith('/tree')) return { entries: [{ path: 'src/main.py', type: 'file' }] };
      if (path.endsWith('/file?path=src%2Fmain.py')) return { file: { content: 'def parse(value):\n    return value', sha256: 'file-hash' } };
      return {};
    });
    post.mockImplementation(async (path: string, body?: Record<string, unknown>) => path === '/api/codebase/projects/repo-1/tools' && body?.tool === 'grep'
      ? { run: { id: 'interactive-1' }, result: { matches: [{ path: 'src/main.py', line: 2, text: 'return value' }] } }
      : {});
    const { CodebasePage } = await import('./CodebasePage');
    await act(async () => { root.render(createElement(CodebasePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Sample'))?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    const search = host.querySelector<HTMLInputElement>('#codeq-content-query');
    await act(async () => { if (search) { const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set; setter?.call(search, 'return value'); search.dispatchEvent(new Event('input', { bubbles: true })); } });
    await act(async () => { host.querySelector<HTMLFormElement>('.codeq-content-search')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(post).toHaveBeenCalledWith('/api/codebase/projects/repo-1/tools', { tool: 'grep', arguments: { query: 'return value', limit: 100 } });
    expect(host.textContent).toContain('src/main.py:2');
    await act(async () => { host.querySelector<HTMLButtonElement>('.codeq-search-match')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(get).toHaveBeenCalledWith('/api/codebase/projects/repo-1/file?path=src%2Fmain.py', expect.anything());
    expect(host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Edit src/main.py"]')?.value).toContain('return value');
  });

  it('shows only backend run task and validation records in the run inspector', async () => {
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/codebase/projects?')) return { projects: [{ id: 'repo-1', name: 'Sample' }] };
      if (path.startsWith('/api/codebase/runs?')) return { runs: [{ id: 'run-1', codebase_project_id: 'repo-1', status: 'completed', provider: 'local' }] };
      if (path === '/api/codebase/projects/repo-1') return { project: { id: 'repo-1', name: 'Sample' } };
      if (path.endsWith('/tree')) return { entries: [] };
      if (path === '/api/codebase/runs/run-1') return { run: { id: 'run-1', status: 'completed', tasks: [{ id: 'task-1', title: 'Run tests', assigned_agent: 'tester', status: 'completed', depends_on: [], result: { summary: 'Tests completed.' } }], validation: [{ id: 'test-1', kind: 'pytest', status: 'passed', command: ['pytest'], exit_code: 0, duration_ms: 83, stdout: '2 passed', stderr: '' }] } };
      if (path.endsWith('/events?after=0')) return { events: [] };
      return {};
    });
    const { CodebasePage } = await import('./CodebasePage');
    await act(async () => { root.render(createElement(CodebasePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Sample'))?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent === 'Inspect run')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.textContent).toContain('Run tests');
    expect(host.textContent).toContain('Tests completed.');
    expect(host.textContent).toContain('pytest · passed');
    expect(host.textContent).toContain('2 passed');
    expect(host.textContent).not.toContain('Preview running');
  });

  it('creates, moves, and explicitly confirms deletion of selected source paths', async () => {
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/codebase/projects?')) return { projects: [{ id: 'repo-1', name: 'Sample' }] };
      if (path.startsWith('/api/codebase/runs?')) return { runs: [] };
      if (path === '/api/codebase/projects/repo-1') return { project: { id: 'repo-1', name: 'Sample' } };
      if (path.endsWith('/tree')) return { entries: [{ path: 'src', type: 'directory' }, { path: 'src/main.py', type: 'file' }] };
      if (path.endsWith('/file?path=src%2Fmain.py') || path.endsWith('/file?path=src%2Frenamed.py')) return { file: { content: 'print("ready")', sha256: 'file-hash' } };
      return {};
    });
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const { CodebasePage } = await import('./CodebasePage');
    await act(async () => { root.render(createElement(CodebasePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Sample'))?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { host.querySelector<HTMLButtonElement>('.codeq-path-toolbar button')?.click(); });
    const createInput = host.querySelector<HTMLInputElement>('#codeq-path-value');
    await act(async () => { if (createInput) { const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set; setter?.call(createInput, 'src/new.py'); createInput.dispatchEvent(new Event('input', { bubbles: true })); } });
    await act(async () => { [...host.querySelectorAll<HTMLButtonElement>('.codeq-path-action button')].find(button => button.textContent === 'Create file')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(post).toHaveBeenCalledWith('/api/codebase/projects/repo-1/paths', { action: 'create_file', path: 'src/new.py', destination: '' });

    await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Open src/main.py"]')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll<HTMLButtonElement>('.codeq-path-toolbar button')].find(button => button.textContent === 'Rename / move')?.click(); });
    const renameInput = host.querySelector<HTMLInputElement>('#codeq-path-value');
    await act(async () => { if (renameInput) { const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set; setter?.call(renameInput, 'src/renamed.py'); renameInput.dispatchEvent(new Event('input', { bubbles: true })); } });
    await act(async () => { [...host.querySelectorAll<HTMLButtonElement>('.codeq-path-action button')].find(button => button.textContent === 'Move file')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(post).toHaveBeenCalledWith('/api/codebase/projects/repo-1/paths', { action: 'rename', path: 'src/main.py', destination: 'src/renamed.py' });
    expect(host.querySelector('.codeq-file-heading h3')?.textContent).toBe('src/renamed.py');

    await act(async () => { [...host.querySelectorAll<HTMLButtonElement>('.codeq-path-toolbar button')].find(button => button.textContent === 'Delete file…')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('Permanently delete src/renamed.py'));
    expect(post).toHaveBeenCalledWith('/api/codebase/projects/repo-1/paths', { action: 'delete', path: 'src/renamed.py', destination: '' });
    expect(host.querySelector('textarea[aria-label="Edit src/renamed.py"]')).toBeNull();
    confirm.mockRestore();
  });

  it.each([
    ['git_status', 'Git status', ' M src/main.py'],
    ['git_diff', 'Git diff', 'diff --git a/src/main.py b/src/main.py'],
    ['git_log', 'Git history', 'abc1234 Add parser coverage'],
  ] as const)('shows the protected workspace %s read-only result', async (tool, buttonText, output) => {
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/codebase/projects?')) return { projects: [{ id: 'repo-1', name: 'Sample' }] };
      if (path.startsWith('/api/codebase/runs?')) return { runs: [] };
      if (path === '/api/codebase/projects/repo-1') return { project: { id: 'repo-1', name: 'Sample' } };
      if (path.endsWith('/tree')) return { entries: [] };
      return {};
    });
    post.mockImplementation(async (path: string, body?: Record<string, unknown>) => path === '/api/codebase/projects/repo-1/tools' && body?.tool === tool
      ? { run: { id: 'interactive-git' }, result: { repository: true, stdout: output, stderr: '', exit_code: 0, duration_ms: 4 } }
      : {});
    const { CodebasePage } = await import('./CodebasePage');
    await act(async () => { root.render(createElement(CodebasePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Sample'))?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll<HTMLButtonElement>('.codeq-path-toolbar button')].find(button => button.textContent === buttonText)?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(post).toHaveBeenCalledWith('/api/codebase/projects/repo-1/tools', { tool, arguments: {} });
    expect(host.querySelector('.codeq-git-evidence')?.textContent).toContain('Read-only result');
    expect(host.querySelector('.codeq-git-evidence pre')?.textContent).toBe(output);
  });

  it('reports a non-Git project from the backend rather than rendering sample status', async () => {
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/codebase/projects?')) return { projects: [{ id: 'repo-1', name: 'Sample' }] };
      if (path.startsWith('/api/codebase/runs?')) return { runs: [] };
      if (path === '/api/codebase/projects/repo-1') return { project: { id: 'repo-1', name: 'Sample' } };
      if (path.endsWith('/tree')) return { entries: [] };
      return {};
    });
    post.mockImplementation(async (path: string) => path === '/api/codebase/projects/repo-1/tools'
      ? { run: { id: 'interactive-git' }, result: { repository: false, stderr: 'selected workspace is not a Git repository', exit_code: 0 } }
      : {});
    const { CodebasePage } = await import('./CodebasePage');
    await act(async () => { root.render(createElement(CodebasePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Sample'))?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll<HTMLButtonElement>('.codeq-path-toolbar button')].find(button => button.textContent === 'Git status')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector('.codeq-git-evidence summary')?.textContent).toContain('Not a Git repository');
    expect(host.querySelector('.codeq-git-evidence pre')?.textContent).toBe('selected workspace is not a Git repository');
  });

  it('surfaces read-only Git API failures without retaining stale evidence', async () => {
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/codebase/projects?')) return { projects: [{ id: 'repo-1', name: 'Sample' }] };
      if (path.startsWith('/api/codebase/runs?')) return { runs: [] };
      if (path === '/api/codebase/projects/repo-1') return { project: { id: 'repo-1', name: 'Sample' } };
      if (path.endsWith('/tree')) return { entries: [] };
      return {};
    });
    post.mockRejectedValue(new Error('Git inspection unavailable'));
    const { CodebasePage } = await import('./CodebasePage');
    await act(async () => { root.render(createElement(CodebasePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Sample'))?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { [...host.querySelectorAll<HTMLButtonElement>('.codeq-path-toolbar button')].find(button => button.textContent === 'Git status')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector('[role="alert"]')?.textContent).toBe('Git inspection unavailable');
    expect(host.querySelector('.codeq-git-evidence')).toBeNull();
  });
});
