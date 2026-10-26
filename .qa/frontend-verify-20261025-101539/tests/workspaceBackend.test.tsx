import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CalendarScreen, CodeQScreen, FilesScreen, KnowledgeScreen, ResearchScreen } from '../src/screens/WorkspaceScreens';
import { useWorkspaceState } from '../src/state';
import { demoProvider } from '../src/demo/provider';

vi.mock('@monaco-editor/react', () => ({ default: ({ value, onChange }: { value: string; onChange: (value: string) => void }) => <textarea aria-label="Code editor" value={value} onChange={event => onChange(event.target.value)} /> }));

afterEach(() => { cleanup(); window.localStorage.removeItem('oa-second-codeq-live-preview:p1'); vi.unstubAllGlobals(); useWorkspaceState.getState().setDataMode('live'); });

function renderScreen(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}
function json(payload: unknown, status = 200) { return new Response(JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json' } }); }

describe('workspace backend integrations', () => {
  it('starts a confirmed Live CodeQ preview, reports backend readiness/output, refreshes and stops it', async () => {
    useWorkspaceState.getState().setDataMode('live');
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET');
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      calls.push({ url, method, body });
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1')) return json({ project: { id: 'p1', name: 'Project One', workspace: { workspace_path: 'C:/sandbox/p1' } } });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [] });
      if (url.endsWith('/api/codebase/projects/p1/tools') && body?.tool === 'start_process') return json({ run: { id: 'ide-run-1' }, result: { id: 'preview-process-1', status: 'running' } }, 201);
      if (url.endsWith('/api/codebase/projects/p1/tools') && body?.tool === 'process_status') return json({ run: { id: 'ide-run-1' }, result: { id: 'preview-process-1', status: 'running', exit_code: null } });
      if (url.endsWith('/api/codebase/projects/p1/tools') && body?.tool === 'inspect_logs') return json({ run: { id: 'ide-run-1' }, result: { id: 'preview-process-1', stdout: 'Local: http://localhost:5173/' } });
      if (url.endsWith('/api/codebase/projects/p1/tools') && body?.tool === 'probe_local_url') return json({ result: { url: 'http://localhost:5173/', reachable: true, http_status: 200, content_type: 'text/html', redirect: false } });
      if (url.endsWith('/api/codebase/projects/p1/tools') && body?.tool === 'stop_process') return json({ run: { id: 'ide-run-1' }, result: { id: 'preview-process-1', status: 'exited', exit_code: 0 } });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));
    renderScreen(<CodeQScreen/>);
    fireEvent.change(await screen.findByRole('textbox', { name: 'Preview command' }), { target: { value: 'npm run dev -- --host 127.0.0.1' } });
    fireEvent.click(screen.getByRole('checkbox', { name: /I reviewed this command/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Start preview' }));
    await screen.findByRole('link', { name: 'http://localhost:5173/' });
    expect(await screen.findByText(/Preview is ready/)).toBeVisible();
    expect(calls.find(call => call.body?.tool === 'probe_local_url')?.body).toMatchObject({ arguments: { url: 'http://localhost:5173/' }, approved: false });
    expect(calls.find(call => call.body?.tool === 'start_process')?.body).toMatchObject({ arguments: { command: ['npm', 'run', 'dev', '--', '--host', '127.0.0.1'] }, approved: true });
    fireEvent.click(screen.getByRole('tab', { name: 'Preview' }));
    expect(await screen.findByRole('link', { name: 'http://localhost:5173/' })).toBeVisible();
    expect(calls.some(call => call.body?.tool === 'stop_process')).toBe(false);
    fireEvent.click(screen.getByRole('button', { name: 'Refresh preview status' }));
    await waitFor(() => expect(calls.filter(call => call.body?.tool === 'process_status').length).toBeGreaterThan(1));
    fireEvent.click(screen.getByRole('button', { name: 'Stop preview' }));
    await waitFor(() => expect(calls.some(call => call.body?.tool === 'stop_process' && call.body.arguments && (call.body.arguments as Record<string, unknown>).process_id === 'preview-process-1')).toBe(true));
    expect(await screen.findByText('Preview process stopped by the backend.')).toBeInTheDocument();
    cleanup();
    useWorkspaceState.getState().setDataMode('demo');
  });
  it('restores a saved Live preview for the same project and clears a stale process id', async () => {
    useWorkspaceState.getState().setDataMode('live');
    window.localStorage.setItem('oa-second-codeq-live-preview:p1', 'preview-process-restored');
    const calls: Array<{ tool: string; project: string }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [] });
      if (url.endsWith('/api/codebase/projects/p1')) return json({ project: { id: 'p1', name: 'Project One' } });
      if (url.endsWith('/api/codebase/projects/p1/tools')) {
        const tool = String(body?.tool ?? ''); calls.push({ tool, project: 'p1' });
        const args = body?.arguments as Record<string, unknown> | undefined;
        if (tool === 'process_status' && args?.process_id === 'preview-process-restored') return json({ result: { id: 'preview-process-restored', status: 'running', exit_code: null } });
        if (tool === 'inspect_logs' && args?.process_id === 'preview-process-restored') return json({ result: { id: 'preview-process-restored', stdout: 'Local: http://127.0.0.1:5173/' } });
        if (tool === 'probe_local_url') return json({ result: { url: 'http://127.0.0.1:5173/', reachable: false, http_status: null, error: 'ConnectionRefusedError: local preview is not responding' } });
        if (tool === 'stop_process') return json({ result: { id: 'preview-process-restored', status: 'exited', exit_code: 0 } });
      }
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderScreen(<CodeQScreen/>);
    expect(await screen.findByRole('link', { name: 'http://127.0.0.1:5173/' })).toBeVisible();
    expect(calls.some(call => call.tool === 'start_process')).toBe(false);
    expect(calls).toEqual(expect.arrayContaining([{ tool: 'process_status', project: 'p1' }, { tool: 'inspect_logs', project: 'p1' }]));
    expect(window.localStorage.getItem('oa-second-codeq-live-preview:p1')).toBe('preview-process-restored');
  });
  it('shows a real not-ready response and recovers after a manual readiness retry', async () => {
    useWorkspaceState.getState().setDataMode('live');
    window.localStorage.setItem('oa-second-codeq-live-preview:p1', 'preview-process-warming');
    let probes = 0;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [] });
      if (url.endsWith('/api/codebase/projects/p1')) return json({ project: { id: 'p1', name: 'Project One' } });
      if (url.endsWith('/api/codebase/projects/p1/tools') && body?.tool === 'process_status') return json({ result: { id: 'preview-process-warming', status: 'running', exit_code: null } });
      if (url.endsWith('/api/codebase/projects/p1/tools') && body?.tool === 'inspect_logs') return json({ result: { id: 'preview-process-warming', stdout: 'Local: http://localhost:5173/' } });
      if (url.endsWith('/api/codebase/projects/p1/tools') && body?.tool === 'probe_local_url') {
        probes += 1;
        return json({ result: probes === 1
          ? { url: 'http://localhost:5173/', reachable: false, http_status: 503, redirect: false }
          : { url: 'http://localhost:5173/', reachable: true, http_status: 200, content_type: 'text/html', redirect: false } });
      }
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderScreen(<CodeQScreen/>);
    expect(await screen.findByText(/Preview responded with HTTP 503/)).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Retry preview readiness check' }));
    expect(await screen.findByText(/Preview is ready/)).toBeVisible();
    expect(probes).toBe(2);
  });
  it('stops and forgets a preview when CodeQ is left, while reload restoration remains covered separately', async () => {
    useWorkspaceState.getState().setDataMode('live');
    vi.stubGlobal('confirm', vi.fn(() => true));
    const calls: Array<{ tool: string; processId: string }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      const args = body?.arguments as Record<string, unknown> | undefined;
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [] });
      if (url.endsWith('/api/codebase/projects/p1')) return json({ project: { id: 'p1', name: 'Project One' } });
      if (url.endsWith('/api/codebase/projects/p1/tools')) {
        const tool = String(body?.tool ?? '');
        const processId = String(args?.process_id ?? '');
        calls.push({ tool, processId });
        if (tool === 'start_process') return json({ result: { id: 'preview-process-leave' } }, 201);
        if (tool === 'process_status') return json({ result: { id: processId, status: 'running', exit_code: null } });
        if (tool === 'inspect_logs') return json({ result: { id: processId, stdout: '' } });
        if (tool === 'stop_process') return json({ result: { id: processId, status: 'exited', exit_code: 0 } });
      }
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderScreen(<CodeQScreen/>);
    fireEvent.click(await screen.findByRole('checkbox', { name: /I reviewed this command/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Start preview' }));
    await waitFor(() => expect(window.localStorage.getItem('oa-second-codeq-live-preview:p1')).toBe('preview-process-leave'));
    cleanup();
    await waitFor(() => expect(calls).toContainEqual({ tool: 'stop_process', processId: 'preview-process-leave' }));
    await waitFor(() => expect(window.localStorage.getItem('oa-second-codeq-live-preview:p1')).toBeNull());
  });
  it('clears a stale restored Live preview id when the backend reports it missing', async () => {
    useWorkspaceState.getState().setDataMode('live');
    window.localStorage.setItem('oa-second-codeq-live-preview:p1', 'preview-process-stale');
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [] });
      if (url.endsWith('/api/codebase/projects/p1')) return json({ project: { id: 'p1', name: 'Project One' } });
      if (url.endsWith('/api/codebase/projects/p1/tools') && body?.tool === 'process_status') return json({ error: 'managed process not found' }, 404);
      if (url.endsWith('/api/codebase/projects/p1/tools') && body?.tool === 'inspect_logs') return json({ error: 'managed process not found' }, 404);
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderScreen(<CodeQScreen/>);
    expect(await screen.findByText('The saved preview process is no longer available. Start a new preview when ready.')).toBeVisible();
    expect(window.localStorage.getItem('oa-second-codeq-live-preview:p1')).toBeNull();
    expect(await screen.findByRole('button', { name: 'Start preview' })).toBeDisabled();
  });
  it('keeps project-only acceptance criteria out of CodeQ until a real project is open', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => String(input).includes('/api/codebase/projects?') ? json({ projects: [] }) : json({ error: 'unavailable' }, 503)));
    renderScreen(<CodeQScreen/>);
    expect(await screen.findByRole('heading', { name: 'Open a project to begin' })).toBeVisible();
    expect(screen.queryByRole('textbox', { name: 'CodeQ acceptance criteria' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Run framework preflight' })).not.toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Ask CodeQ' })).toBeDisabled();
  });
  it('loads project files from CodeQ tree and requests the selected file', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.includes('/tree')) return json({ entries: [{ path: 'src', type: 'directory' }, { path: 'src/main.ts', type: 'file' }] });
      if (url.includes('/file?')) return json({ file: { path: 'src/main.ts', content: 'export const value = 1;' } });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<CodeQScreen/>);
    await screen.findByRole('option', { name: 'Project One' });
    fireEvent.click(await screen.findByRole('treeitem', { name: 'main.ts' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/codebase/projects/p1/file?path=src%2Fmain.ts'), expect.anything()));
  });

  it('saves a CodeQ file with its read hash and keeps the user draft on a conflict', async () => {
    vi.stubGlobal('confirm', vi.fn(() => true));
    let savedBody: Record<string, unknown> | null = null;
    let conflict = true;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [{ path: 'src/main.ts', type: 'file' }] });
      if (url.endsWith('/api/codebase/projects/p1/file?path=src%2Fmain.ts')) return json({ file: { path: 'src/main.ts', content: 'export const value = 1;', sha256: 'original-hash' } });
      if (url.endsWith('/api/codebase/projects/p1/file') && init?.method === 'POST') {
        savedBody = JSON.parse(String(init.body)) as Record<string, unknown>;
        return conflict ? json({ error: 'file changed since it was read' }, 400) : json({ change: { path: 'src/main.ts', new_sha256: 'updated-hash' } });
      }
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<CodeQScreen/>);
    fireEvent.click(await screen.findByRole('treeitem', { name: 'main.ts' }));
    const editor = await screen.findByRole('textbox', { name: 'Code editor' });
    fireEvent.change(editor, { target: { value: 'export const value = 2;' } });
    expect(screen.getByRole('status')).toHaveTextContent('Unsaved changes');
    fireEvent.click(screen.getByRole('button', { name: 'Save file' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('file changed since it was read');
    expect(savedBody).toEqual({ path: 'src/main.ts', content: 'export const value = 2;', expected_sha256: 'original-hash' });
    expect(screen.getByRole('textbox', { name: 'Code editor' })).toHaveValue('export const value = 2;');
    conflict = false;
    fireEvent.click(screen.getByRole('button', { name: 'Reload' }));
    await waitFor(() => expect(fetchMock.mock.calls.filter(([url]) => String(url).includes('/api/codebase/projects/p1/file?')).length).toBeGreaterThan(1));
    fireEvent.change(screen.getByRole('textbox', { name: 'Code editor' }), { target: { value: 'export const value = 3;' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save file' }));
    await waitFor(() => expect(savedBody).toEqual({ path: 'src/main.ts', content: 'export const value = 3;', expected_sha256: 'original-hash' }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Saved'));
  });

  it('runs framework preflight and refreshes the project scan with backend results', async () => {
    const requests: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET');
      let body: Record<string, unknown> = {}; try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* GET */ }
      requests.push({ url, method, body });
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [{ path: 'src/app.ts', type: 'file' }] });
      if (url.endsWith('/api/codebase/preflight')) return json({ preflight: { framework: { id: 'react', label: 'React' }, local_validation: 'ready', missing_toolchains: [], host: { system: 'Windows', supported: true }, generation_contract: { acceptance_criteria: ['Build succeeds'] } } });
      if (url.endsWith('/api/codebase/projects/p1/scan')) return json({ project_map: { file_count: 8, framework: 'react' } });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<CodeQScreen/>);
    fireEvent.change(await screen.findByRole('textbox', { name: 'Ask CodeQ' }), { target: { value: 'Review this React project' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'CodeQ acceptance criteria' }), { target: { value: 'Build succeeds\nNo security regressions' } });
    fireEvent.click(screen.getByRole('button', { name: 'Preflight' }));
    expect(await screen.findByRole('region', { name: 'CodeQ framework preflight' })).toHaveTextContent('React');
    expect(screen.getByRole('region', { name: 'CodeQ framework preflight' })).toHaveTextContent('Build succeeds');
    const preflight = requests.find(request => request.url.endsWith('/api/codebase/preflight'));
    expect(preflight?.body).toMatchObject({ goal: 'Review this React project', codebase_project_id: 'p1', acceptance_criteria: ['Build succeeds', 'No security regressions'] });
    fireEvent.click(screen.getByRole('button', { name: 'Scan project files' }));
    expect(await screen.findByText('Project scan returned 8 files.')).toBeVisible();
    expect(requests.some(request => request.url.endsWith('/api/codebase/projects/p1/scan') && request.method === 'POST')).toBe(true);
  });

  it('confirms path changes and sends explorer operations only to the codebase path API', async () => {
    const calls: Array<{ body: Record<string, unknown> }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [{ path: 'src/app.ts', type: 'file' }] });
      if (url.includes('/api/codebase/projects/p1/file?')) return json({ file: { path: 'src/app.ts', content: 'const app = 1;', sha256: 'h1' } });
      if (url.endsWith('/api/codebase/projects/p1/paths') && init?.method === 'POST') { calls.push({ body: JSON.parse(String(init.body)) }); return json({ change: JSON.parse(String(init.body)) }); }
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    const confirm = vi.fn(() => true);
    const prompt = vi.fn(() => 'src/new-file.ts');
    vi.stubGlobal('confirm', confirm); vi.stubGlobal('prompt', prompt);
    renderScreen(<CodeQScreen/>);
    await screen.findByRole('option', { name: 'Project One' });
    fireEvent.click(screen.getByRole('button', { name: 'Create file' }));
    await waitFor(() => expect(calls).toContainEqual({ body: { action: 'create_file', path: 'src/new-file.ts', destination: '' } }));
    prompt.mockReturnValueOnce('src/renamed.ts');
    fireEvent.click(screen.getByRole('button', { name: 'Rename selected path' }));
    await waitFor(() => expect(calls.some(call => call.body.action === 'rename' && call.body.destination === 'src/renamed.ts')).toBe(true));
    prompt.mockReturnValueOnce('archive/renamed.ts');
    fireEvent.click(screen.getByRole('button', { name: 'Move selected path' }));
    await waitFor(() => expect(calls.some(call => call.body.action === 'move' && call.body.path === 'src/renamed.ts' && call.body.destination === 'archive/renamed.ts')).toBe(true));
    fireEvent.click(screen.getByRole('button', { name: 'Delete selected path' }));
    await waitFor(() => expect(calls.some(call => call.body.action === 'delete')).toBe(true));
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('Rename'));
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('Move'));
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('connected source folder'));
    expect(calls.at(-1)?.body).toMatchObject({ action: 'delete', path: 'archive/renamed.ts' });
  });

  it('searches project sources and requests read-only Git diff/history tools', async () => {
    const toolCalls: Array<{ url: string; body: Record<string, unknown> }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [{ path: 'src/app.ts', type: 'file' }] });
      if (url.endsWith('/api/codebase/projects/p1')) return json({ project: { id: 'p1', workspace: { workspace_path: 'D:/workspace/p1' } } });
      if (url.endsWith('/api/codebase/projects/p1/file?path=src%2Fapp.ts')) return json({ file: { path: 'src/app.ts', content: 'export const match = true;', sha256: 'h1' } });
      if (url.endsWith('/api/codebase/projects/p1/tools') && init?.method === 'POST') {
        const body = JSON.parse(String(init.body)) as Record<string, unknown>; toolCalls.push({ url, body });
        const tool = String(body.tool);
        const result = tool === 'grep' ? { matches: [{ path: 'src/app.ts', line: 1, preview: 'export const match = true;' }] } : { stdout: tool === 'git_diff' ? 'diff --git a/src/app.ts b/src/app.ts' : 'a1b2c3d (HEAD -> main) latest change', exit_code: 0 };
        return json({ run: { id: `tool-${tool}` }, result });
      }
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<CodeQScreen/>);
    fireEvent.click(await screen.findByRole('button', { name: 'Search project files' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Search project source' }), { target: { value: 'match' } });
    fireEvent.click(screen.getByRole('button', { name: 'Search', exact: true }));
    fireEvent.click(await screen.findByRole('button', { name: /src\/app\.ts/ }));
    expect(await screen.findByRole('textbox', { name: 'Code editor' })).toHaveValue('export const match = true;');
    expect(toolCalls[0]?.body).toMatchObject({ tool: 'grep', arguments: { query: 'match', limit: 100 }, approved: false });
    fireEvent.click(screen.getByRole('tab', { name: /Git/ }));
    fireEvent.click(await screen.findByRole('button', { name: 'Diff' }));
    expect(await screen.findByRole('log', { name: 'Git diff output' })).toHaveTextContent('diff --git');
    fireEvent.click(screen.getByRole('button', { name: 'History' }));
    expect(await screen.findByRole('log', { name: 'Git history output' })).toHaveTextContent('latest change');
    expect(toolCalls.map(call => call.body)).toEqual(expect.arrayContaining([
      expect.objectContaining({ tool: 'git_diff', arguments: { staged: false }, approved: false }),
      expect.objectContaining({ tool: 'git_log', arguments: { limit: 30 }, approved: false }),
    ]));
  });

  it('protects unsaved CodeQ drafts when changing files and on browser unload', async () => {
    const fileReads: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [{ path: 'src/a.ts', type: 'file' }, { path: 'src/b.ts', type: 'file' }] });
      if (url.includes('/api/codebase/projects/p1/file?')) { fileReads.push(url); const path = decodeURIComponent(url.split('path=')[1] ?? ''); return json({ file: { path, content: `source for ${path}`, sha256: `hash-${path}` } }); }
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('confirm', vi.fn(() => false));
    renderScreen(<CodeQScreen/>);
    fireEvent.click(await screen.findByRole('treeitem', { name: 'a.ts' }));
    const editor = await screen.findByRole('textbox', { name: 'Code editor' });
    fireEvent.change(editor, { target: { value: 'unsaved user draft' } });
    const countBefore = fileReads.length;
    fireEvent.click(screen.getByRole('treeitem', { name: 'b.ts' }));
    expect(screen.getByRole('textbox', { name: 'Code editor' })).toHaveValue('unsaved user draft');
    expect(fileReads).toHaveLength(countBefore);
    const beforeUnload = new Event('beforeunload', { cancelable: true });
    window.dispatchEvent(beforeUnload);
    expect(beforeUnload.defaultPrevented).toBe(true);
  });

  it('restores only a backend-listed CodeQ run snapshot after explicit confirmation', async () => {
    const restoreCalls: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [] });
      if (url.endsWith('/api/codebase/runs') && init?.method === 'POST') return json({ run: { id: 'r1', status: 'queued' } }, 202);
      if (url.includes('/api/codebase/runs/r1/events/stream')) return new Response('data: {"type":"stream.closed","sequence":1}\n\n', { headers: { 'Content-Type': 'text/event-stream' } });
      if (url.endsWith('/api/codebase/runs/r1')) return json({ run: { id: 'r1', status: 'completed', snapshots: [{ id: 's1', label: 'Before autonomous run', size_bytes: 256, created_at: 'now' }] } });
      if (url.endsWith('/api/codebase/runs/r1/snapshots/s1/restore') && init?.method === 'POST') { restoreCalls.push(url); return json({ restored: true, snapshot_id: 's1' }); }
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    const confirm = vi.fn(() => false); vi.stubGlobal('confirm', confirm);
    renderScreen(<CodeQScreen/>);
    fireEvent.change(await screen.findByRole('textbox', { name: 'Ask CodeQ' }), { target: { value: 'Inspect this project' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send to CodeQ' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Restore' }));
    expect(confirm).toHaveBeenCalled();
    expect(restoreCalls).toHaveLength(0);
    confirm.mockReturnValue(true);
    fireEvent.click(screen.getByRole('button', { name: 'Restore' }));
    await waitFor(() => expect(restoreCalls).toHaveLength(1));
    expect(await screen.findByText('Run snapshot restored to the CodeQ sandbox workspace.')).toHaveAttribute('role', 'status');
  });

  it('presents only backend-reported CodeQ approvals and submits the user decision', async () => {
    let approvalResolved = false;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [] });
      if (url.endsWith('/api/codebase/runs') && init?.method === 'POST') return json({ run: { id: 'r1', status: 'queued' } }, 202);
      if (url.endsWith('/api/codebase/runs/r1/events/stream')) return new Response('data: {"type":"stream.closed","sequence":1}\n\n', { headers: { 'Content-Type': 'text/event-stream' } });
      if (url.endsWith('/api/codebase/runs/r1')) return json({ run: { id: 'r1', status: 'blocked', approvals: approvalResolved ? [] : [{ id: 'a1', tool: 'run_shell', reason: 'The command needs one-time approval.', status: 'pending' }] } });
      if (url.endsWith('/api/codebase/approvals/a1/resolve') && init?.method === 'POST') { approvalResolved = true; return json({ approval: { id: 'a1', status: 'approved' } }); }
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<CodeQScreen/>);
    fireEvent.change(await screen.findByRole('textbox', { name: 'Ask CodeQ' }), { target: { value: 'Run the project checks' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send to CodeQ' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Approve' }));
    await waitFor(() => expect(approvalResolved).toBe(true));
    const decision = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/codebase/approvals/a1/resolve') && init?.method === 'POST');
    expect(JSON.parse(String(decision?.[1]?.body))).toEqual({ approved: true });
  });

  it('runs an allowlisted terminal command without approval in the selected project sandbox', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [] });
      if (url.endsWith('/api/codebase/projects/p1')) return json({ project: { id: 'p1', source_path: 'D:/outside/source', workspace: { workspace_path: 'D:/openagent/data/workspaces/p1' } } });
      if (url.endsWith('/api/terminal/runs') && init?.method === 'POST') return json({ run: { id: 't1', command: ['git', 'status', '--short'], cwd: 'D:/openagent/data/workspaces/p1', status: 'completed', exit_code: 0, events: [{ stream: 'stdout', chunk: ' M file.txt' }] } }, 202);
      if (url.includes('/api/terminal/runs') && !url.includes('/api/terminal/runs/')) return json({ runs: [] });
      if (url.endsWith('/api/terminal/runs/t1')) return json({ run: { id: 't1', command: ['git', 'status', '--short'], cwd: 'D:/openagent/data/workspaces/p1', status: 'completed', exit_code: 0, events: [{ stream: 'stdout', chunk: ' M file.txt' }] } });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<CodeQScreen/>);
    fireEvent.click(await screen.findByRole('tab', { name: /Terminal/ }));
    fireEvent.change(await screen.findByRole('textbox', { name: 'Terminal command' }), { target: { value: 'git status --short' } });
    expect(screen.getByText('Allowlisted command · no approval required.')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Run command' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/terminal/runs'), expect.objectContaining({ method: 'POST', body: expect.stringContaining('"approved":false') })));
    const terminalRequest = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/terminal/runs') && init?.method === 'POST');
    expect(JSON.parse(String(terminalRequest?.[1]?.body))).toMatchObject({ command: ['git', 'status', '--short'], cwd: 'D:/openagent/data/workspaces/p1', approved: false });
    expect(await screen.findByRole('log', { name: 'Terminal run output' })).toHaveTextContent('M file.txt');
  });

  it('requires one-run approval for non-default terminal commands and submits only to the mocked API', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [] });
      if (url.endsWith('/api/codebase/projects/p1')) return json({ project: { id: 'p1', workspace: { workspace_path: 'D:/openagent/data/workspaces/p1' } } });
      if (url.endsWith('/api/terminal/runs') && init?.method === 'POST') return json({ run: { id: 't2', command: ['echo', 'hello'], cwd: 'D:/openagent/data/workspaces/p1', status: 'completed', exit_code: 0, events: [{ stream: 'stdout', chunk: 'hello' }] } }, 202);
      if (url.includes('/api/terminal/runs') && !url.includes('/api/terminal/runs/')) return json({ runs: [] });
      if (url.endsWith('/api/terminal/runs/t2')) return json({ run: { id: 't2', command: ['echo', 'hello'], cwd: 'D:/openagent/data/workspaces/p1', status: 'completed', exit_code: 0, events: [{ stream: 'stdout', chunk: 'hello' }] } });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<CodeQScreen/>);
    fireEvent.click(await screen.findByRole('tab', { name: /Terminal/ }));
    fireEvent.change(await screen.findByRole('textbox', { name: 'Terminal command' }), { target: { value: 'echo hello' } });
    expect(screen.getByRole('button', { name: 'Run command' })).toBeDisabled();
    expect(fetchMock.mock.calls.some(([url, init]) => String(url).endsWith('/api/terminal/runs') && init?.method === 'POST')).toBe(false);
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.click(screen.getByRole('button', { name: 'Run command' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/terminal/runs'), expect.objectContaining({ method: 'POST', body: expect.stringContaining('"approved":true') })));
    expect(await screen.findByText('hello')).toBeVisible();
  });

  it('shows incrementally polled Live terminal output and requests owner-scoped cancellation', async () => {
    useWorkspaceState.getState().setDataMode('live');
    let status = 'running';
    const calls: Array<{ path: string; method: string }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), 'http://localhost');
      const method = String(init?.method ?? 'GET');
      calls.push({ path: `${url.pathname}${url.search}`, method });
      if (url.pathname === '/api/codebase/projects') return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.pathname === '/api/codebase/projects/p1/tree') return json({ entries: [] });
      if (url.pathname === '/api/codebase/projects/p1') return json({ project: { id: 'p1', workspace: { workspace_path: 'D:/openagent/data/workspaces/p1' } } });
      if (url.pathname === '/api/terminal/runs' && method === 'GET') return json({ runs: [] });
      if (url.pathname === '/api/terminal/runs' && method === 'POST') return json({ run: { id: 'terminal-1', command: ['python', '--version'], cwd: 'D:/openagent/data/workspaces/p1', status, exit_code: -1, events: [] } }, 202);
      if (url.pathname === '/api/terminal/runs/terminal-1' && method === 'GET') return json({ run: { id: 'terminal-1', command: ['python', '--version'], cwd: 'D:/openagent/data/workspaces/p1', status, exit_code: status === 'cancelled' ? 130 : -1, events: [{ id: 'event-1', stream: 'stdout', chunk: 'Python 3.14.3\\n' }], cancel_requested: status === 'running' } });
      if (url.pathname === '/api/terminal/runs/terminal-1' && method === 'DELETE') { status = 'cancelled'; return json({ run: { id: 'terminal-1', status, exit_code: 130 }, cancel_requested: true }); }
      return json({ error: `Unexpected request ${method} ${url.pathname}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<CodeQScreen/>);
    fireEvent.click(await screen.findByRole('tab', { name: /Terminal/ }));
    fireEvent.change(await screen.findByRole('textbox', { name: 'Terminal command' }), { target: { value: 'python --version' } });
    fireEvent.click(screen.getByRole('button', { name: 'Run command' }));
    await waitFor(() => expect(calls).toContainEqual({ path: '/api/terminal/runs', method: 'POST' }));
    expect(await screen.findByRole('log', { name: 'Terminal run output' })).toHaveTextContent('Python 3.14.3');
    fireEvent.click(await screen.findByRole('button', { name: 'Stop' }));
    await waitFor(() => expect(calls).toContainEqual({ path: '/api/terminal/runs/terminal-1', method: 'DELETE' }));
    expect(await screen.findByText(/cancelled/)).toBeVisible();
  });

  it('runs Git status only on refresh against the CodeQ sandbox path', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [] });
      if (url.endsWith('/api/codebase/projects/p1')) return json({ project: { id: 'p1', source_path: 'D:/picked/source', workspace: { workspace_path: 'D:/openagent/data/workspaces/p1' } } });
      if (url.endsWith('/api/terminal/stream') && init?.method === 'POST') return json({ run: { id: 'g1', command: ['git', 'status', '--short'], cwd: 'D:/openagent/data/workspaces/p1', status: 'completed', exit_code: 0, events: [{ stream: 'stdout', chunk: ' M file.txt' }] } }, 201);
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<CodeQScreen/>);
    fireEvent.click(await screen.findByRole('tab', { name: /Git/ }));
    expect(fetchMock).not.toHaveBeenCalledWith(expect.stringContaining('/api/terminal/stream'), expect.anything());
    await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh status' })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: 'Refresh status' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/terminal/stream'), expect.objectContaining({ method: 'POST' })));
    const gitRequest = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/terminal/stream') && init?.method === 'POST');
    expect(JSON.parse(String(gitRequest?.[1]?.body))).toMatchObject({ command: ['git', 'status', '--short'], cwd: 'D:/openagent/data/workspaces/p1', approved: false });
    expect(await screen.findByRole('log', { name: 'Git status output' })).toHaveTextContent('M file.txt');
  });

  it('lists uploaded media and submits a selected file as data_base64', async () => {
    let assetAvailable = false;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/media/assets')) return json({ assets: assetAvailable ? [{ id: 'a1', title: 'hello.txt', mime_type: 'text/plain', size_bytes: 2, media_type: 'text' }] : [] });
      if (url.endsWith('/api/media/ingest') && init?.method === 'POST') { assetAvailable = true; return json({ asset: { id: 'a1', title: 'hello.txt' } }, 201); }
      if (url.includes('/api/documents/library')) return json({ items: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<FilesScreen/>);
    fireEvent.click(screen.getByRole('tab', { name: 'Uploaded files' }));
    expect(await screen.findByText('No uploaded files')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Add' }));
    const file = new File(['hi'], 'hello.txt', { type: 'text/plain' });
    Object.defineProperty(file, 'arrayBuffer', { value: async () => new TextEncoder().encode('hi').buffer });
    fireEvent.change(screen.getByLabelText('Choose uploaded file'), { target: { files: [file] } });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/media/ingest'), expect.objectContaining({ method: 'POST' })));
    expect(await screen.findByRole('status')).toHaveTextContent('hello.txt uploaded and stored.');
    await waitFor(() => expect(fetchMock.mock.calls.filter(([url, init]) => String(url).includes('/api/media/assets') && init?.method === 'GET').length).toBeGreaterThan(1));
    expect(await screen.findByText('hello.txt')).toBeVisible();
    const uploadRequest = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/media/ingest') && init?.method === 'POST');
    expect(JSON.parse(String(uploadRequest?.[1]?.body))).toMatchObject({ filename: 'hello.txt', data_base64: 'aGk=', metadata: { mime_type: 'text/plain' } });
  });

  it('loads and selects a document from the document library', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => String(input).includes('/api/documents/library')
      ? json({ items: [{ doc_id: 'd1', title: 'Design notes', format: 'markdown', description: 'A real backend summary', size_bytes: 1280, updated_at: '2026-10-12T12:00:00Z' }], count: 1 })
      : String(input).includes('/api/document/d1') ? json({ document: { doc_id: 'd1', title: 'Design notes', content: 'Actual stored note', version: 2 } })
        : json({ error: 'Unexpected request' }, 404));
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<FilesScreen/>);
    const listRow = await screen.findByRole('button', { name: /Design notes/ });
    expect(listRow).toHaveClass('file-record');
    expect(listRow).toHaveTextContent('MD');
    expect(listRow).toHaveTextContent('1.3 KB');
    expect(listRow).toHaveTextContent('A real backend summary');
    fireEvent.click(listRow);
    expect(await screen.findByText('Actual stored note')).toBeVisible();
    expect(document.querySelector('.document-canvas')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
    expect(screen.getByText('110%')).toBeVisible();
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/document/d1'), expect.anything());
  });

  it('edits documents with an if_version guard and preserves the draft on conflict', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/documents/library')) return json({ items: [{ doc_id: 'd1', title: 'Notes' }] });
      if (url.endsWith('/api/document/d1') && init?.method === 'PUT') return json({ error: 'version conflict: expected 3, got 2' }, 409);
      if (url.endsWith('/api/document/d1')) return json({ document: { doc_id: 'd1', title: 'Notes', content: 'Original', version: 2 } });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<FilesScreen/>);
    fireEvent.click(await screen.findByRole('button', { name: /Notes/ }));
    fireEvent.click(screen.getByRole('button', { name: 'File options' }));
    fireEvent.click(screen.getByRole('button', { name: 'Edit document' }));
    fireEvent.change(await screen.findByRole('textbox', { name: 'Document content' }), { target: { value: 'My unsaved draft' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await screen.findByRole('alert');
    const put = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/document/d1') && init?.method === 'PUT');
    expect(JSON.parse(String(put?.[1]?.body))).toMatchObject({ title: 'Notes', content: 'My unsaved draft', if_version: 2 });
    expect(screen.getByRole('textbox', { name: 'Document content' })).toHaveValue('My unsaved draft');
  });

  it('loads version history and restores an earlier version through the versioned backend route', async () => {
    vi.stubGlobal('confirm', vi.fn(() => true));
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/documents/library')) return json({ items: [{ doc_id: 'd2', title: 'Plan' }] });
      if (url.endsWith('/api/document/d2/versions')) return json({ versions: [{ version: 2, hash: 'h2', created_at: 'today' }, { version: 1, hash: 'h1', created_at: 'yesterday' }] });
      if (url.endsWith('/api/document/d2/restore/1') && init?.method === 'POST') return json({ document: { doc_id: 'd2', title: 'Plan', content: 'Restored body', version: 3 } });
      if (url.endsWith('/api/document/d2')) return json({ document: { doc_id: 'd2', title: 'Plan', content: 'Current body', version: 2 } });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<FilesScreen/>);
    fireEvent.click(await screen.findByRole('button', { name: /Plan/ }));
    fireEvent.click(screen.getByRole('button', { name: 'File options' }));
    fireEvent.click(screen.getByRole('button', { name: 'Version history' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Restore' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/document/d2/restore/1'), expect.objectContaining({ method: 'POST' })));
    expect(await screen.findByText('Restored body')).toBeVisible();
  });

  it('archives and lists documents using the backend status filter', async () => {
    let archived = false;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/documents/library')) return json({ items: [{ doc_id: 'd3', title: 'Reference', status: archived ? 'archived' : 'active' }] });
      if (url.endsWith('/api/document/d3/archive') && init?.method === 'POST') { archived = JSON.parse(String(init.body)).archived; return json({ doc_id: 'd3', status: archived ? 'archived' : 'active' }); }
      if (url.endsWith('/api/document/d3')) return json({ document: { doc_id: 'd3', title: 'Reference', content: '', version: 1, status: archived ? 'archived' : 'active' } });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<FilesScreen/>);
    fireEvent.click(await screen.findByRole('button', { name: /Reference/ }));
    fireEvent.click(screen.getByRole('button', { name: 'File options' }));
    fireEvent.click(screen.getByRole('button', { name: 'Archive document' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/document/d3/archive'), expect.objectContaining({ method: 'POST', body: JSON.stringify({ archived: true }) })));
    fireEvent.click(screen.getByRole('button', { name: 'Document filters and actions' }));
    fireEvent.click(screen.getByRole('button', { name: 'Archived documents' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('status=archived'), expect.anything()));
  });

  it('requires confirmation before deleting a backend document', async () => {
    vi.stubGlobal('confirm', vi.fn(() => true));
    let exists = true;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/documents/library')) return json({ items: exists ? [{ doc_id: 'd4', title: 'Temporary note' }] : [] });
      if (url.endsWith('/api/document/d4') && init?.method === 'DELETE') { exists = false; return json({ deleted: true }); }
      if (url.endsWith('/api/document/d4')) return json({ document: { doc_id: 'd4', title: 'Temporary note', content: 'Stored note', version: 1 } });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<FilesScreen/>);
    fireEvent.click(await screen.findByRole('button', { name: /Temporary note/ }));
    fireEvent.click(screen.getByRole('button', { name: 'File options' }));
    fireEvent.click(screen.getByRole('button', { name: 'Delete document' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/document/d4'), expect.objectContaining({ method: 'DELETE' })));
    expect(await screen.findByText('No documents yet')).toBeVisible();
  });

  it('imports a PDF through the document processing API and opens the returned document', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/documents/library')) return json({ items: [] });
      if (url.endsWith('/api/documents/import-pdf') && init?.method === 'POST') return json({ document: { doc_id: 'pdf1', title: 'brief', content: 'Imported text', version: 1 } }, 201);
      if (url.endsWith('/api/document/pdf1')) return json({ document: { doc_id: 'pdf1', title: 'brief', content: 'Imported text', version: 1 } });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<FilesScreen/>);
    fireEvent.click(screen.getByRole('button', { name: 'Document filters and actions' }));
    fireEvent.click(screen.getByRole('button', { name: 'Import documents' }));
    const file = new File(['pdf'], 'brief.pdf', { type: 'application/pdf' });
    Object.defineProperty(file, 'arrayBuffer', { value: async () => new TextEncoder().encode('pdf').buffer });
    fireEvent.change(screen.getByLabelText('Choose document to import'), { target: { files: [file] } });
    expect(await screen.findByText('Imported text')).toBeVisible();
    const request = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/documents/import-pdf') && init?.method === 'POST');
    expect(JSON.parse(String(request?.[1]?.body))).toMatchObject({ filename: 'brief.pdf', content_base64: 'cGRm', project: 'default' });
  });

  it('rejects legacy Office binaries before uploading them as document text', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/documents/library')) return json({ items: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<FilesScreen/>);
    fireEvent.click(screen.getByRole('button', { name: 'Document filters and actions' }));
    fireEvent.click(screen.getByRole('button', { name: 'Import documents' }));
    fireEvent.change(screen.getByLabelText('Choose document to import'), { target: { files: [new File(['binary'], 'old.doc', { type: 'application/msword' })] } });
    expect(await screen.findByRole('alert')).toHaveTextContent(/Legacy Office files cannot be safely extracted/);
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/api/documents/import'))).toBe(false);
  });

  it('submits actual knowledge search queries and renders returned evidence', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/knowledge/status')) return json({ documents: 1, chunks: 2 });
      if (url.includes('/api/knowledge/documents')) return json({ documents: [{ id: 'k1', title: 'Handbook' }] });
      if (url.includes('/api/knowledge/search')) return json({ results: [{ id: 'chunk1', title: 'Handbook', content: 'Actual search evidence' }] });
      return json({ error: 'Unexpected request' }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<KnowledgeScreen/>);
    fireEvent.change(screen.getByRole('textbox', { name: 'Search knowledge' }), { target: { value: 'retention policy' } });
    fireEvent.click(screen.getByRole('button', { name: 'Search' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/knowledge/search'), expect.anything()));
    expect(await screen.findByText('Actual search evidence')).toBeVisible();
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('q=retention%20policy'), expect.anything());
  });

  it('uses the hybrid knowledge endpoint and renders only backend evidence', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/knowledge/status')) return json({ documents: 0, chunks: 0 });
      if (url.includes('/api/knowledge/documents')) return json({ documents: [] });
      if (url.endsWith('/api/knowledge/tags')) return json({ tags: [] });
      if (url.endsWith('/api/knowledge/timeline')) return json({ timeline: [] });
      if (url.endsWith('/api/knowledge/hybrid-search') && init?.method === 'POST') return json({ results: [{ id: 'chunk-a', title: 'Operations handbook', content: 'Backend returned hybrid passage', score: 0.82, document_id: 'doc-a' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<KnowledgeScreen/>);
    fireEvent.change(screen.getByLabelText('Search knowledge'), { target: { value: 'approval controls' } });
    fireEvent.change(screen.getByLabelText('Knowledge search mode'), { target: { value: 'hybrid' } });
    fireEvent.click(screen.getByRole('button', { name: 'Search' }));
    expect(await screen.findByText('Backend returned hybrid passage')).toBeVisible();
    const request = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/knowledge/hybrid-search') && init?.method === 'POST');
    expect(JSON.parse(String(request?.[1]?.body))).toMatchObject({ query: 'approval controls', project: 'default', limit: 10 });
  });

  it('synthesizes with returned citations and saves the result into ordinary chat history', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/knowledge/status')) return json({ documents: 1, chunks: 1 });
      if (url.includes('/api/knowledge/documents')) return json({ documents: [{ id: 'doc-1', title: 'Policy' }] });
      if (url.endsWith('/api/knowledge/tags')) return json({ tags: [] });
      if (url.endsWith('/api/knowledge/timeline')) return json({ timeline: [] });
      if (url.endsWith('/api/knowledge/search')) return json({ results: [{ id: 'chunk-1', title: 'Policy', content: 'Evidence' }] });
      if (url.endsWith('/api/knowledge/synthesize') && init?.method === 'POST') return json({ query: 'retention rules', summary: 'Backend synthesis summary', sources: [{ chunk_id: 'chunk-1', document_id: 'doc-1', title: 'Policy', score: 0.9 }] });
      if (url.endsWith('/api/chat/research-record') && init?.method === 'POST') return json({ conversation_id: 'chat-1', conversation: { id: 'chat-1' } }, 201);
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<KnowledgeScreen/>);
    fireEvent.change(screen.getByLabelText('Search knowledge'), { target: { value: 'retention rules' } });
    fireEvent.click(screen.getByRole('button', { name: 'Search' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Synthesize with sources' }));
    expect(await screen.findByText('Backend synthesis summary')).toBeVisible();
    expect(screen.getAllByText('Policy').length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole('button', { name: 'Save to chat' }));
    expect(await screen.findByRole('status')).toHaveTextContent('Saved to chat');
    expect(screen.getByRole('link', { name: 'Open conversation' })).toHaveAttribute('href', '/?view=chat&conversation_id=chat-1');
    const saved = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/chat/research-record') && init?.method === 'POST');
    expect(JSON.parse(String(saved?.[1]?.body))).toMatchObject({ query: 'retention rules', report: 'Backend synthesis summary', project: 'default', sources: [{ document_id: 'doc-1', chunk_id: 'chunk-1' }] });
  });

  it('uploads a user-selected source through the backend extraction endpoint', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/knowledge/status')) return json({ documents: 0, chunks: 0 });
      if (url.includes('/api/knowledge/documents')) return json({ documents: [] });
      if (url.endsWith('/api/knowledge/tags')) return json({ tags: [] });
      if (url.endsWith('/api/knowledge/timeline')) return json({ timeline: [] });
      if (url.endsWith('/api/knowledge/upload') && init?.method === 'POST') return json({ filename: 'manual.txt', chars_extracted: 17, document: { id: 'doc-upload', title: 'manual.txt' } }, 201);
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<KnowledgeScreen/>);
    const file = new File(['Actual file text'], 'manual.txt', { type: 'text/plain' });
    fireEvent.change(screen.getByLabelText('Upload knowledge document'), { target: { files: [file] } });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/knowledge/upload'), expect.objectContaining({ method: 'POST', body: expect.any(FormData) })));
    const request = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/api/knowledge/upload'));
    const body = request?.[1]?.body as FormData;
    expect((body.get('file') as File).name).toBe('manual.txt');
    expect(body.get('project')).toBe('default');
  });

  it('opens and deletes an indexed Knowledge source through project-scoped Live routes', async () => {
    vi.stubGlobal('confirm', vi.fn(() => true));
    let deleted = false;
    const calls: Array<{ url: string; method: string }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET'); calls.push({ url, method });
      if (url.includes('/api/knowledge/status')) return json({ documents: deleted ? 0 : 1, chunks: deleted ? 0 : 1 });
      if (url.endsWith('/api/knowledge/documents?project=default&limit=100')) return json({ documents: deleted ? [] : [{ id: 'source-1', title: 'Incident guide', source: 'manual' }] });
      if (url.endsWith('/api/knowledge/documents/source-1?project=default') && method === 'GET') return json({ document: { id: 'source-1', project: 'default', title: 'Incident guide', source: 'manual', created_at: '2026-10-23T10:00:00Z', tags: ['operations'], chunks: [{ id: 'chunk-1' }], content: 'Persisted source body from the backend.' } });
      if (url.endsWith('/api/knowledge/documents/source-1?project=default') && method === 'DELETE') { deleted = true; return json({ deleted: true, document_id: 'source-1', chunks: 1 }); }
      if (url.endsWith('/api/knowledge/tags')) return json({ tags: [] });
      if (url.endsWith('/api/knowledge/timeline')) return json({ timeline: [] });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<KnowledgeScreen/>);
    fireEvent.click(await screen.findByRole('button', { name: 'Open Incident guide' }));
    const dialog = await screen.findByRole('dialog', { name: 'Incident guide' });
    expect(await within(dialog).findByText('Persisted source body from the backend.')).toBeVisible();
    expect(within(dialog).getByText('operations')).toBeVisible();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Delete source' }));
    await waitFor(() => expect(deleted).toBe(true));
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Incident guide' })).not.toBeInTheDocument());
    expect(calls.some(call => call.method === 'GET' && call.url.endsWith('/api/knowledge/documents/source-1?project=default'))).toBe(true);
    expect(calls.some(call => call.method === 'DELETE' && call.url.endsWith('/api/knowledge/documents/source-1?project=default'))).toBe(true);
    expect(await screen.findByRole('status')).toHaveTextContent('Indexed source and its retrieval chunks were deleted');
  });

  it('edits a Live Knowledge source through the scoped update route and keeps its identity', async () => {
    let title = 'Incident guide';
    let content = 'Persisted source body from the backend.';
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET');
      const body = typeof init?.body === 'string' ? JSON.parse(init.body) as Record<string, unknown> : undefined;
      calls.push({ url, method, body });
      if (url.includes('/api/knowledge/status')) return json({ documents: 1, chunks: 1 });
      if (url.endsWith('/api/knowledge/documents?project=default&limit=100')) return json({ documents: [{ id: 'source-1', title, source: 'manual' }] });
      if (url.endsWith('/api/knowledge/documents/source-1?project=default') && method === 'GET') return json({ document: { id: 'source-1', project: 'default', title, source: 'manual', created_at: '2026-10-23T10:00:00Z', tags: ['operations'], chunks: [{ id: 'chunk-1' }], content } });
      if (url.endsWith('/api/knowledge/documents/source-1') && method === 'PUT') { title = String(body?.title); content = String(body?.content); return json({ document: { id: 'source-1', project: 'default', title, source: 'manual', tags: ['revised'], chunks: [{ id: 'chunk-2' }], content } }); }
      if (url.endsWith('/api/knowledge/tags')) return json({ tags: [] });
      if (url.endsWith('/api/knowledge/timeline')) return json({ timeline: [] });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<KnowledgeScreen/>);
    fireEvent.click(await screen.findByRole('button', { name: 'Open Incident guide' }));
    const details = await screen.findByRole('dialog', { name: 'Incident guide' });
    await within(details).findByText('Persisted source body from the backend.');
    expect(within(details).getByRole('button', { name: 'Edit source' })).toBeEnabled();
    fireEvent.click(within(details).getByRole('button', { name: 'Edit source' }));
    const edit = await screen.findByRole('dialog', { name: 'Edit knowledge source' });
    fireEvent.change(within(edit).getByRole('textbox', { name: 'Knowledge source title' }), { target: { value: 'Updated incident guide' } });
    fireEvent.change(within(edit).getByRole('textbox', { name: 'Knowledge source text' }), { target: { value: 'Revised searchable evidence.' } });
    fireEvent.click(within(edit).getByRole('button', { name: 'Save source' }));
    await waitFor(() => expect(calls.some(call => call.method === 'PUT')).toBe(true));
    expect(calls.find(call => call.method === 'PUT')).toMatchObject({
      url: '/api/knowledge/documents/source-1',
      body: { project: 'default', title: 'Updated incident guide', content: 'Revised searchable evidence.' },
    });
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Updated incident guide' })).toBeVisible());
    expect(await screen.findByText('Revised searchable evidence.')).toBeVisible();
    expect(calls.some(call => call.method === 'PUT' && !call.body?.project)).toBe(false);
  });

  it('deletes a selected Live workspace note through its explicit project scope', async () => {
    vi.stubGlobal('confirm', vi.fn(() => true));
    let deleted = false;
    const calls: Array<{ url: string; method: string }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET'); calls.push({ url, method });
      if (url.includes('/api/documents/library')) return json({ documents: [] });
      if (url.endsWith('/api/workspace/items?project=default&kind=note&limit=500')) return json({ items: deleted ? [] : [{ id: 'workspace-note-1', kind: 'note', title: 'Release note', content: 'Local project record', metadata: { workspace_type: 'note' }, updated_at: '2026-10-23T10:00:00Z' }] });
      if (url.endsWith('/api/workspace/items/workspace-note-1?project=default') && method === 'DELETE') { deleted = true; return json({ deleted: true, item_id: 'workspace-note-1' }); }
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<FilesScreen/>);
    fireEvent.click(screen.getByRole('tab', { name: 'Notes', exact: true }));
    const noteRow = await screen.findByRole('button', { name: /Release note/ });
    fireEvent.click(noteRow);
    fireEvent.click(await screen.findByRole('button', { name: 'Delete note' }));
    await waitFor(() => expect(deleted).toBe(true));
    expect(calls.some(call => call.method === 'DELETE' && call.url.endsWith('/api/workspace/items/workspace-note-1?project=default'))).toBe(true);
    await waitFor(() => expect(screen.queryByRole('button', { name: /Release note/ })).not.toBeInTheDocument());
  });

  it('shows backend graph, timeline, tags and RAG index details and requests an index sync', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/knowledge/status')) return json({ documents: 1, chunks: 1 });
      if (url.includes('/api/knowledge/documents')) return json({ documents: [{ id: 'd1', title: 'Runbook' }] });
      if (url.includes('/api/knowledge/tags')) return json({ tags: [{ id: 't1', tag: 'operations', confidence: 0.94 }] });
      if (url.includes('/api/knowledge/timeline')) return json({ timeline: [{ source_id: 'd1', title: 'Runbook added', timestamp: '2026-10-14T10:00:00Z' }] });
      if (url.includes('/api/knowledge/graph')) return json({ entities: [{ id: 'e1', name: 'Incident response', entity_type: 'topic', source_id: 'c1' }], relationships: [{ id: 'r1', subject: 'Incident response', predicate: 'covered by', object: 'Runbook', source_id: 'c1', weight: 1 }] });
      if (url.includes('/api/rag/status')) return json({ status: 'ready', documents: 1, vectors: 1 });
      if (url.endsWith('/api/rag/sync') && init?.method === 'POST') return json({ status: 'completed', indexed: 1 }, 201);
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<KnowledgeScreen/>);
    expect(await screen.findByText('operations')).toBeVisible();
    expect(await screen.findByText('Runbook added')).toBeVisible();
    fireEvent.click(screen.getByRole('tab', { name: 'Retrieval details' }));
    expect(await screen.findByText(/Incident response · topic/)).toBeVisible();
    expect(screen.getByText(/Source chunk c1/)).toBeVisible();
    fireEvent.click(await screen.findByRole('button', { name: 'Sync vector index' }));
    expect(await screen.findByRole('status')).toHaveTextContent('completed');
    const sync = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/rag/sync') && init?.method === 'POST');
    expect(JSON.parse(String(sync?.[1]?.body))).toEqual({ project: 'default', force: false });
  });

  it('creates calendar events from the user supplied title and time range', async () => {
    let createdBody: Record<string, unknown> | null = null;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/calendar/events') && init?.method === 'POST') { createdBody = JSON.parse(String(init.body)) as Record<string, unknown>; return json({ event: { id: 'e1' } }, 201); }
      if (url.includes('/api/calendar/events')) return json({ events: [] });
      return json({ error: 'Unexpected request' }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<CalendarScreen/>);
    fireEvent.click(screen.getByRole('button', { name: /New/ }));
    const dialog = await screen.findByRole('dialog', { name: 'New calendar event' });
    fireEvent.change(within(dialog).getByLabelText('Title'), { target: { value: 'Planning session' } });
    fireEvent.change(within(dialog).getByLabelText('Starts'), { target: { value: '2026-10-14T09:00' } });
    fireEvent.change(within(dialog).getByLabelText('Ends'), { target: { value: '2026-10-14T10:00' } });
    fireEvent.change(within(dialog).getByLabelText('Location'), { target: { value: 'Room 4' } });
    fireEvent.change(within(dialog).getByLabelText('Description'), { target: { value: 'Quarterly planning' } });
    fireEvent.change(within(dialog).getByLabelText('Attendees (comma separated)'), { target: { value: 'alex@example.test, sam@example.test' } });
    fireEvent.click(within(dialog).getByRole('checkbox', { name: '15 minutes before' }));
    fireEvent.click(within(dialog).getByRole('checkbox', { name: '30 minutes before' }));
    fireEvent.click(within(dialog).getByLabelText('All-day event'));
    fireEvent.change(within(dialog).getByLabelText('Repeat'), { target: { value: 'FREQ=WEEKLY' } });
    fireEvent.click(within(dialog).getByRole('button', { name: 'Create event' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/calendar/events'), expect.objectContaining({ method: 'POST', body: expect.stringContaining('Planning session') })));
    expect(createdBody).toMatchObject({ title: 'Planning session', description: 'Quarterly planning', location: 'Room 4', attendees: ['alex@example.test', 'sam@example.test'], reminders: [30], all_day: true, rrule: 'FREQ=WEEKLY', project: 'default' });
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  });

  it('acknowledges Live due reminders and removes them only after backend confirmation', async () => {
    useWorkspaceState.getState().setDataMode('live');
    let due = [{ reminder_id: 'reminder-live-1', event_id: 'event-1', event_title: 'Planning review', trigger_at: '2026-10-24T09:00:00Z' }];
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET');
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      calls.push({ url, method, body });
      if (url.includes('/api/calendar/reminders') && method === 'GET') return json({ reminders: due });
      if (url.endsWith('/api/calendar/reminders/reminder-live-1/acknowledge') && method === 'POST') { due = []; return json({ acknowledged: true, reminder_id: 'reminder-live-1' }); }
      if (url.includes('/api/calendar/events')) return json({ events: [] });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));
    renderScreen(<CalendarScreen/>);
    fireEvent.click(screen.getByText('Calendar tools', { exact: true }));
    fireEvent.click(await screen.findByRole('tab', { name: 'Reminders' }));
    expect(await screen.findByText('Planning review')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Acknowledge' }));
    await waitFor(() => expect(screen.queryByText('Planning review')).not.toBeInTheDocument());
    expect(calls.find(call => call.method === 'POST')).toMatchObject({ url: expect.stringContaining('/api/calendar/reminders/reminder-live-1/acknowledge'), body: { project: 'default' } });
    expect(await screen.findByText('No reminders are due right now.')).toBeVisible();
  });

  it('keeps failed Live reminder acknowledgement visible with an error', async () => {
    useWorkspaceState.getState().setDataMode('live');
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/calendar/reminders') && String(init?.method ?? 'GET') === 'GET') return json({ reminders: [{ reminder_id: 'reminder-fail', event_title: 'Keep this reminder', trigger_at: 'now' }] });
      if (url.endsWith('/api/calendar/reminders/reminder-fail/acknowledge')) return json({ error: 'Reminder service unavailable' }, 503);
      if (url.includes('/api/calendar/events')) return json({ events: [] });
      return json({ error: 'Unexpected request' }, 404);
    }));
    renderScreen(<CalendarScreen/>);
    fireEvent.click(screen.getByText('Calendar tools', { exact: true }));
    fireEvent.click(await screen.findByRole('tab', { name: 'Reminders' }));
    expect(await screen.findByText('Keep this reminder')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Acknowledge' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Reminder service unavailable');
    expect(screen.getByText('Keep this reminder')).toBeVisible();
  });

  it('dismisses Demo reminders locally without making backend requests', async () => {
    useWorkspaceState.getState().setDataMode('demo');
    window.localStorage.removeItem('oa-second-demo-calendar-dismissed-reminders');
    const hash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/demo/reference-fixtures.json')) return json({ sourceSha256: hash, data: { topLevel: {}, advanced: {} } });
      if (url.endsWith('/demo/approved-shell-fixtures.json')) return json({ sourceSha256: hash, data: {} });
      throw new Error(`Unexpected Demo request: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    await demoProvider.reset();
    renderScreen(<CalendarScreen/>);
    fireEvent.click(screen.getByText('Calendar tools', { exact: true }));
    fireEvent.click(await screen.findByRole('tab', { name: 'Reminders' }));
    const reminderSection = screen.getByRole('heading', { name: 'Due reminders' }).closest('section');
    expect(reminderSection).not.toBeNull();
    const dismiss = (await screen.findAllByRole('button', { name: 'Dismiss' }))[0];
    const firstTitle = dismiss.closest('article')?.querySelector('strong')?.textContent;
    expect(firstTitle).toBeTruthy();
    fireEvent.click(dismiss);
    await waitFor(() => expect(within(reminderSection!).queryByText(firstTitle!)).not.toBeInTheDocument());
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/api/'))).toBe(false);
  });

  it('edits and deletes a Live calendar event through the scoped API', async () => {
    const start = new Date(); start.setHours(14, 0, 0, 0); if (start <= new Date()) start.setDate(start.getDate() + 1);
    const end = new Date(start.getTime() + 60 * 60_000);
    let event = { id: 'calendar-edit-1', title: 'Review block', starts_at: start.toISOString(), ends_at: end.toISOString(), project: 'default', attendees: ['Alex'], reminders: [15] };
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET');
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      calls.push({ url, method, body });
      if (url.includes('/api/calendar/events/calendar-edit-1') && method === 'PATCH') { event = { ...event, ...body } as typeof event; return json({ event }); }
      if (url.includes('/api/calendar/events/calendar-edit-1') && method === 'DELETE') return json({ deleted: true, id: event.id });
      if (url.includes('/api/calendar/events')) return json({ events: [event] });
      if (url.includes('/api/calendar/calendars')) return json({ calendars: [] });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('confirm', vi.fn(() => true));
    renderScreen(<CalendarScreen/>);
    fireEvent.click(await waitFor(() => {
      const row = document.querySelector('.calendar-upcoming-row');
      if (!(row instanceof HTMLButtonElement)) throw new Error('Calendar event row not rendered');
      return row;
    }));
    const details = await screen.findByRole('dialog', { name: 'Review block' });
    fireEvent.click(within(details).getByRole('button', { name: 'Edit' }));
    const edit = await screen.findByRole('dialog', { name: 'Edit calendar event' });
    fireEvent.change(within(edit).getByLabelText('Title'), { target: { value: 'Review block revised' } });
    fireEvent.click(within(edit).getByRole('button', { name: 'Save changes' }));
    await waitFor(() => expect(calls.some(call => call.method === 'PATCH' && call.url.includes('/api/calendar/events/calendar-edit-1'))).toBe(true));
    expect(calls.find(call => call.method === 'PATCH')?.body).toMatchObject({ title: 'Review block revised', attendees: ['Alex'], reminders: [15] });
    expect(await screen.findByText('Calendar event updated.')).toBeVisible();
    fireEvent.click(await waitFor(() => {
      const row = Array.from(document.querySelectorAll('.calendar-upcoming-row')).find(item => item.textContent?.includes('Review block revised'));
      if (!(row instanceof HTMLButtonElement)) throw new Error('Updated calendar event row not rendered');
      return row;
    }));
    const revised = await screen.findByRole('dialog', { name: 'Review block revised' });
    fireEvent.click(within(revised).getByRole('button', { name: 'Delete' }));
    await waitFor(() => expect(calls.some(call => call.method === 'DELETE' && call.url.includes('/api/calendar/events/calendar-edit-1?project=default'))).toBe(true));
    expect(vi.mocked(window.confirm)).toHaveBeenCalledWith('Delete “Review block revised”?');
  });

  it('registers a CalDAV provider, confirms network sync, and resolves real backend conflict records', async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.includes('/api/calendar/calendars')) return json({ calendars: [{ id: 'work', name: 'Work' }] });
      if (url.includes('/api/calendar/providers/register') && init?.method === 'POST') return json({ provider_id: 'provider-1', provider: 'caldav_http', status: 'configured' }, 201);
      if (url.includes('/api/calendar/providers')) return json({ providers: [{ provider_id: 'provider-1', provider: 'caldav_http', status: 'connected', base_url: 'https://calendar.example/dav' }] });
      if (url.includes('/api/calendar/conflicts')) return json({ conflicts: [{ conflict_id: 'conflict-1', status: 'open', local_payload: { title: 'Planning', starts_at: '2026-10-14T09:00:00Z' }, remote_payload: { title: 'Planning', starts_at: '2026-10-14T10:00:00Z' } }] });
      if (url.endsWith('/api/calendar/sync') && init?.method === 'POST') return json({ sync_job_id: 'sync-1', state: 'completed', pulled: 1, pushed: 0, conflicts: 1 }, 202);
      if (url.endsWith('/api/calendar/sync/sync-1')) return json({ sync_job_id: 'sync-1', state: 'completed', pulled: 1, pushed: 0, conflicts: 1 });
      if (url.endsWith('/api/calendar/conflicts/conflict-1/resolve') && init?.method === 'POST') return json({ conflict_id: 'conflict-1', status: 'resolved', resolution: 'client_wins' });
      if (url.includes('/api/calendar/events')) return json({ events: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('confirm', vi.fn(() => true));
    renderScreen(<CalendarScreen/>);
    fireEvent.click(screen.getByText('Calendar tools', { exact: true }));
    fireEvent.click(await screen.findByRole('tab', { name: 'Sync' }));
    fireEvent.change(screen.getByLabelText('Provider'), { target: { value: 'caldav_http' } });
    fireEvent.change(screen.getByLabelText('CalDAV collection URL'), { target: { value: 'https://calendar.example/dav' } });
    fireEvent.change(screen.getByLabelText('Username (optional)'), { target: { value: 'alex' } });
    fireEvent.change(screen.getByLabelText('Password (optional)'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Register provider' }));
    await waitFor(() => expect(requests.some(request => request.url.includes('/api/calendar/providers/register'))).toBe(true));
    const registration = requests.find(request => request.url.includes('/api/calendar/providers/register'));
    expect(JSON.parse(String(registration?.init?.body))).toMatchObject({ provider: 'caldav_http', base_url: 'https://calendar.example/dav', config: { username: 'alex', password: 'secret' } });
    const providerRow = await screen.findByText(/connected · https:\/\/calendar\.example\/dav/);
    expect(providerRow).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Sync…' }));
    await waitFor(() => expect(requests.some(request => request.url.endsWith('/api/calendar/sync') && request.init?.method === 'POST')).toBe(true));
    const syncRequest = requests.find(request => request.url.endsWith('/api/calendar/sync') && request.init?.method === 'POST');
    expect(JSON.parse(String(syncRequest?.init?.body))).toMatchObject({ provider_id: 'provider-1', direction: 'pull', allow_external: true });
    expect(vi.mocked(window.confirm)).toHaveBeenCalled();
    expect(await screen.findByText(/1 pulled · 0 pushed · 1 conflicts/)).toBeVisible();
    expect(await screen.findByText('Local: 2026-10-14T09:00:00Z')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Keep local' }));
    await waitFor(() => expect(requests.some(request => request.url.endsWith('/api/calendar/conflicts/conflict-1/resolve'))).toBe(true));
    const resolution = requests.find(request => request.url.endsWith('/api/calendar/conflicts/conflict-1/resolve'));
    expect(JSON.parse(String(resolution?.init?.body))).toEqual({ resolution: 'client_wins' });
  });

  it('requests real available time from the visible calendar period', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/calendar/free-slots?')) return json({ slots: [{ starts_at: '2026-10-14T13:00:00Z', duration_minutes: 90 }] });
      if (url.includes('/api/calendar/events')) return json({ events: [] });
      if (url.includes('/api/calendar/calendars')) return json({ calendars: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<CalendarScreen/>);
    fireEvent.click(screen.getByText('Calendar tools', { exact: true }));
    fireEvent.click(screen.getByRole('button', { name: 'Find slots' }));
    expect(await screen.findByText(/backend suggested/)).toBeVisible();
    const request = fetchMock.mock.calls.find(([url]) => String(url).includes('/api/calendar/free-slots?'));
    const query = new URL(String(request?.[0]), window.location.origin).searchParams;
    expect(query.get('project')).toBe('default');
    expect(query.get('duration')).toBe('60');
    expect(query.get('from')).toBeTruthy();
    expect(query.get('to')).toBeTruthy();
  });

  it('imports and exports iCalendar through backend contracts', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/api/calendar/ical-import') && init?.method === 'POST') return json({ imported: 2 });
      if (url.endsWith('/api/calendar/ical-export') && init?.method === 'POST') return json({ ical: 'BEGIN:VCALENDAR\r\nEND:VCALENDAR' });
      if (url.includes('/api/calendar/events')) return json({ events: [] });
      if (url.includes('/api/calendar/calendars')) return json({ calendars: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<CalendarScreen/>);
    fireEvent.click(screen.getByText('Calendar tools', { exact: true }));
    fireEvent.click(screen.getByRole('tab', { name: 'Manage' }));
    const source = 'BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nSUMMARY:Planning\r\nEND:VEVENT\r\nEND:VCALENDAR';
    fireEvent.change(screen.getByRole('textbox', { name: 'iCalendar data' }), { target: { value: source } });
    fireEvent.click(screen.getByRole('button', { name: 'Import' }));
    expect(await screen.findByText(/2 event\(s\) processed by the backend/)).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Export' }));
    expect(await screen.findByRole('textbox', { name: 'iCalendar data' })).toHaveValue('BEGIN:VCALENDAR\nEND:VCALENDAR');
    expect(screen.getByRole('button', { name: 'Download .ics' })).toBeEnabled();
  });

  it('starts real research sessions and fetches the returned report', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/research/sessions') && init?.method === 'POST') return json({ session: { id: 's1', query: 'Compare battery chemistries', status: 'queued' }, job: { id: 'j1' } }, 202);
      if (url.includes('/api/research/sessions/s1/stream')) return new Response('data: {"type":"complete","cursor":1}\n\n', { headers: { 'Content-Type': 'text/event-stream' } });
      if (url.includes('/api/research/sessions/s1/report')) return json({ session_id: 's1', query: 'Compare battery chemistries', status: 'completed', report: 'Actual backend report', sources: [] });
      if (url.includes('/api/research/sessions')) return json({ sessions: [{ id: 's1', query: 'Compare battery chemistries', status: 'queued', created_at: '2026-10-14' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<ResearchScreen/>);
    fireEvent.change(screen.getByRole('textbox', { name: 'Research question' }), { target: { value: 'Compare battery chemistries' } });
    fireEvent.click(screen.getByRole('button', { name: 'Run research' }));
    expect(await screen.findByText('Actual backend report')).toBeVisible();
    const startRequest = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/research/sessions') && init?.method === 'POST');
    expect(JSON.parse(String(startRequest?.[1]?.body))).toMatchObject({ query: 'Compare battery chemistries', allow_external: false });
  });

  it('sends external research consent only after the user opts in', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/api/research/sessions') && init?.method === 'POST') return json({ session: { id: 's1', query: 'Recent battery papers', status: 'queued' } }, 202);
      if (url.includes('/api/research/sessions?')) return json({ sessions: [{ id: 's1', query: 'Recent battery papers', status: 'queued' }] });
      if (url.endsWith('/api/research/sessions/s1/report')) return json({ report: '', status: 'queued', sources: [] });
      if (url.includes('/api/research/sessions/s1/stream')) return new Response('data: {"type":"complete","cursor":1}\n\n', { headers: { 'Content-Type': 'text/event-stream' } });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<ResearchScreen/>);
    fireEvent.click(screen.getByText('New research', { exact: true }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Research question' }), { target: { value: 'Recent battery papers' } });
    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow external research sources' }));
    fireEvent.click(screen.getByRole('button', { name: 'Run research' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/research/sessions'), expect.objectContaining({ method: 'POST' })));
    const request = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/research/sessions') && init?.method === 'POST');
    expect(JSON.parse(String(request?.[1]?.body))).toMatchObject({ allow_external: true });
    expect(screen.getByText(/When enabled, the backend may contact external sources/)).toBeVisible();
  });

  it('loads durable research checkpoints and source graph, and saves a completed report to chat', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/research/sessions?')) return json({ sessions: [{ id: 's1', project: 'default', query: 'Battery safety', status: 'completed', created_at: '2026-10-14' }] });
      if (url.endsWith('/api/research/sessions/s1/report')) return json({ session_id: 's1', query: 'Battery safety', status: 'completed', report: 'Verified report body', sources: [{ id: 'src1', title: 'Safety paper', url: 'https://example.test/paper' }] });
      if (url.endsWith('/api/research/sessions/s1/source-graph')) return json({ session_id: 's1', nodes: [{ id: 's1', kind: 'research', label: 'Battery safety' }, { id: 'src1', kind: 'paper', label: 'Safety paper' }], edges: [{ from: 's1', to: 'src1', relation: 'uses' }] });
      if (url.endsWith('/api/research/sessions/s1/checkpoints')) return json({ checkpoints: [{ id: 1, stage: 'completed', message: 'Research report completed.', progress: 100 }] });
      if (url.endsWith('/api/chat/research-record') && init?.method === 'POST') return json({ conversation: { id: 'c1' }, conversation_id: 'c1' }, 201);
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<ResearchScreen/>);
    expect(await screen.findByText('Verified report body')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'More research panels' }));
    fireEvent.click(screen.getByRole('tab', { name: 'Graph' }));
    expect(await screen.findByText('1 source relationship(s) reported by the backend.')).toBeVisible();
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/research/sessions/s1/source-graph'), expect.anything());
    fireEvent.click(screen.getByRole('button', { name: 'More research panels' }));
    fireEvent.click(screen.getByRole('tab', { name: 'Checkpoints' }));
    expect(await screen.findByRole('log', { name: 'Research checkpoints' })).toHaveTextContent('Research report completed.');
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/research/sessions/s1/checkpoints'), expect.anything());
    fireEvent.click(screen.getByRole('button', { name: 'Save to chat' }));
    expect(await screen.findByRole('link', { name: 'Open conversation' })).toHaveAttribute('href', '/?view=chat&conversation_id=c1');
    const saveRequest = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/chat/research-record') && init?.method === 'POST');
    expect(JSON.parse(String(saveRequest?.[1]?.body))).toMatchObject({ query: 'Battery safety', report: 'Verified report body', project: 'default', research_session_id: 's1', sources: [{ id: 'src1' }] });
  });

  it('renders the persisted Research cancellation checkpoint in Activity immediately', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/research/sessions?')) return json({ sessions: [{ id: 's-cancel', project: 'default', query: 'Cancel this research', status: 'running' }] });
      if (url.endsWith('/api/research/sessions/s-cancel/report')) return json({ session_id: 's-cancel', status: 'running', report: '', sources: [] });
      if (url.endsWith('/api/research/sessions/s-cancel/stream')) return new Response('data: {"type":"checkpoint","checkpoint":{"id":1,"stage":"planning","progress":10,"message":"Planning"},"cursor":1}\n\n', { headers: { 'Content-Type': 'text/event-stream' } });
      if (url.endsWith('/api/research/sessions/s-cancel/cancel') && init?.method === 'POST') return json({ session: { id: 's-cancel', status: 'running', cancel_requested: true } }, 202);
      if (url.endsWith('/api/research/sessions/s-cancel/checkpoints')) return json({ checkpoints: [{ id: 2, stage: 'cancel-requested', progress: 0, message: 'Cancellation requested.' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<ResearchScreen/>);
    await screen.findByText('Cancel this research');
    fireEvent.click(screen.getByRole('button', { name: 'More research panels' }));
    fireEvent.click(screen.getByRole('tab', { name: 'Activity' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Cancel run' }));
    expect(await screen.findByRole('log')).toHaveTextContent('cancel-requested');
    expect(await screen.findByRole('log')).toHaveTextContent('Cancellation requested.');
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/research/sessions/s-cancel/checkpoints'), expect.anything());
  });

  it('loads durable Research activity for a completed session when its event stream was missed', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/research/sessions?')) return json({ sessions: [{ id: 's-finished', project: 'default', query: 'Completed before opening Activity', status: 'completed' }] });
      if (url.endsWith('/api/research/sessions/s-finished/report')) return json({ session_id: 's-finished', status: 'completed', report: 'Persisted final report.', sources: [] });
      if (url.endsWith('/api/research/sessions/s-finished/checkpoints')) return json({ checkpoints: [
        { id: 4, stage: 'planning', progress: 10, message: 'Planning research strategy.' },
        { id: 5, stage: 'completed', progress: 100, message: 'Research report completed.' },
      ] });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<ResearchScreen/>);
    await screen.findByText('Persisted final report.');
    fireEvent.click(screen.getByRole('button', { name: 'More research panels' }));
    fireEvent.click(screen.getByRole('tab', { name: 'Activity' }));
    const activity = await screen.findByRole('log');
    await waitFor(() => expect(activity).toHaveTextContent('Planning research strategy.'));
    expect(activity).toHaveTextContent('Research report completed.');
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/research/sessions/s-finished/checkpoints'), expect.anything());
    expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith('/api/research/sessions/s-finished/stream'))).toBe(false);
  });

  it('filters backend research sources and saves selected-source notes to the session API', async () => {
    const savedNotes: Array<Record<string, unknown>> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/research/sessions?')) return json({ sessions: [{ id: 's1', query: 'Battery materials', status: 'completed' }] });
      if (url.endsWith('/api/research/sessions/s1/report')) return json({ report: 'Report from backend', sources: [{ id: 'a', title: 'Cathode paper', url: 'https://example.test/a' }, { id: 'b', title: 'Anode paper', url: 'https://example.test/b' }] });
      if (url.endsWith('/api/research/sessions/s1/source-notes?project=default')) return json({ notes: savedNotes });
      if (url.endsWith('/api/research/sessions/s1/source-notes/b') && init?.method === 'POST') {
        const body = JSON.parse(String(init.body)) as Record<string, unknown>;
        const note = { session_id: 's1', source_key: 'b', content: String(body.content ?? ''), updated_at: '2026-10-24T12:00:00Z' };
        if (note.content) savedNotes.splice(0, savedNotes.length, note); else savedNotes.length = 0;
        return json({ note });
      }
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<ResearchScreen/>);
    await screen.findByText('Report from backend');
    fireEvent.change(screen.getByRole('textbox', { name: 'Find a source' }), { target: { value: 'anode' } });
    expect(screen.getByRole('button', { name: /Anode paper/ })).toBeVisible();
    expect(screen.queryByRole('button', { name: /Cathode paper/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Anode paper/ }));
    const note = screen.getByRole('textbox', { name: 'Local source note' });
    fireEvent.change(note, { target: { value: 'Review this citation' } });
    expect(note).toHaveValue('Review this citation');
    expect(screen.getByText(/Notes are saved with this research session/)).toBeVisible();
    expect(fetchMock.mock.calls.some(([url, init]) => String(url).includes('/source-notes/') && init?.method === 'POST')).toBe(false);
    const saveButton = screen.getByRole('button', { name: 'Save source note' });
    await waitFor(() => expect(saveButton).toBeEnabled());
    fireEvent.click(saveButton);
    expect(await screen.findAllByText('Source note saved to this Research session.')).toHaveLength(2);
    const save = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/source-notes/b') && init?.method === 'POST');
    expect(JSON.parse(String(save?.[1]?.body))).toEqual({ project: 'default', content: 'Review this citation' });
    expect(savedNotes).toEqual([{ session_id: 's1', source_key: 'b', content: 'Review this citation', updated_at: '2026-10-24T12:00:00Z' }]);
  });

  it('clears source selection and stream state when deleting the selected Research session', async () => {
    useWorkspaceState.getState().setDataMode('live');
    vi.stubGlobal('confirm', vi.fn(() => true));
    let sessions = [
      { id: 's1', project: 'default', query: 'First run', status: 'completed' },
      { id: 's2', project: 'default', query: 'Second run', status: 'completed' },
    ];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/research/sessions?')) return json({ sessions });
      if (url.endsWith('/api/research/sessions/s1/report')) return json({ report: 'First report', sources: [{ id: 'src-first', title: 'First source' }] });
      if (url.endsWith('/api/research/sessions/s2/report')) return json({ report: 'Second report', sources: [{ id: 'src-second', title: 'Second source' }] });
      if (url.endsWith('/api/research/sessions/s1/delete') && init?.method === 'POST') {
        sessions = sessions.filter(session => session.id !== 's1');
        return json({ deleted: true });
      }
      if (url.includes('/source-notes?')) return json({ notes: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderScreen(<ResearchScreen/>);
    expect(await screen.findByText('First report')).toBeVisible();
    fireEvent.click(screen.getByText('New research', { exact: true }));
    fireEvent.click(screen.getAllByRole('tab', { name: 'Sources' })[0]);
    fireEvent.click((await screen.findAllByRole('button', { name: /First source/ }))[0]);
    const saveNote = screen.getByRole('button', { name: 'Save source note' });
    await waitFor(() => expect(saveNote).toBeEnabled());
    fireEvent.change(screen.getByLabelText('Session'), { target: { value: 's2' } });
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith('/api/research/sessions/s2/source-notes?project=default'))).toBe(true));
    await waitFor(() => expect(screen.queryByText('Loading notes for this research session…')).not.toBeInTheDocument());
    expect(saveNote).toBeDisabled();
    fireEvent.change(screen.getByLabelText('Session'), { target: { value: 's1' } });
    await waitFor(() => expect(screen.getByLabelText('Session')).toHaveValue('s1'));
    fireEvent.click((await screen.findAllByRole('button', { name: /First source/ }))[0]);
    await waitFor(() => expect(saveNote).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    await screen.findAllByText('Research session deleted.');
    await waitFor(() => expect(screen.getByLabelText('Session')).toHaveValue('s2'));
    expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith('/api/research/sessions/s2/report'))).toBe(true);
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith('/api/research/sessions/s2/source-notes?project=default'))).toBe(true));
    await waitFor(() => expect(screen.queryByText('Loading notes for this research session…')).not.toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Save source note' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Save source note' }));
    expect(fetchMock.mock.calls.some(([url, init]) => String(url).includes('/source-notes/src-first') && init?.method === 'POST')).toBe(false);
    expect(fetchMock.mock.calls.some(([url, init]) => String(url).endsWith('/api/research/sessions/s1/delete') && init?.method === 'POST')).toBe(true);
  });

  it('uses backend retry, branch, archive, restore and delete actions for research sessions', async () => {
    const sessions = [
      { id: 's1', project: 'default', query: 'Failed question', status: 'failed', archived: false },
      { id: 's2', project: 'default', query: 'Branch of failed question', status: 'queued', archived: false },
    ];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/research/sessions?')) {
        const includeArchived = url.includes('include_archived=true');
        return json({ sessions: sessions.filter(session => includeArchived || !session.archived) });
      }
      if (url.endsWith('/api/research/sessions/s1/retry') && init?.method === 'POST') { sessions[0].status = 'queued'; return json({ session: sessions[0] }, 202); }
      if (url.endsWith('/api/research/sessions/s1/branch') && init?.method === 'POST') return json({ session: sessions[1] }, 202);
      if (url.endsWith('/api/research/sessions/s2/archive') && init?.method === 'POST') {
        sessions[1].archived = Boolean((JSON.parse(String(init.body)) as Record<string, unknown>).archived);
        return json({ session: sessions[1] }, 202);
      }
      if (url.endsWith('/api/research/sessions/s2/delete') && init?.method === 'POST') { sessions.splice(1, 1); return json({ deleted: true, session_id: 's2' }); }
      if (/\/api\/research\/sessions\/s[12]\/stream/.test(url)) return new Response('data: {"type":"complete","cursor":1}\n\n', { headers: { 'Content-Type': 'text/event-stream' } });
      if (/\/api\/research\/sessions\/s[12]\/report/.test(url)) return json({ report: '', status: sessions.find(session => url.includes(session.id))?.status ?? 'queued' });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    renderScreen(<ResearchScreen/>);
    fireEvent.click(screen.getByText('New research', { exact: true }));
    await screen.findByRole('option', { name: 'Failed question · failed' });
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/research/sessions/s1/retry'), expect.objectContaining({ method: 'POST' })));
    fireEvent.click(await screen.findByRole('button', { name: 'Branch' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/research/sessions/s1/branch'), expect.objectContaining({ method: 'POST', body: expect.stringContaining('Research branch') })));
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'Session' })).toHaveValue('s2'));
    fireEvent.click(screen.getByRole('button', { name: 'Archive' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/research/sessions/s2/archive'), expect.objectContaining({ method: 'POST', body: JSON.stringify({ archived: true }) })));
    fireEvent.click(screen.getByLabelText(/Include archived/));
    expect(await screen.findByRole('button', { name: 'Restore' })).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Restore' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/research/sessions/s2/archive'), expect.objectContaining({ method: 'POST', body: JSON.stringify({ archived: false }) })));
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/research/sessions/s2/delete'), expect.objectContaining({ method: 'POST' })));
    expect(window.confirm).toHaveBeenCalledWith('Delete this research session and its sources?');
  });
});

