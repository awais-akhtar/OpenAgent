import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CodeQScreen } from '../src/screens/WorkspaceScreens';
import { useWorkspaceState } from '../src/state';

vi.mock('@monaco-editor/react', () => ({ default: () => <div aria-label="Code editor"/> }));

afterEach(() => {
  cleanup();
  window.localStorage.removeItem('oa-second-codeq-live-preview:p1');
  useWorkspaceState.getState().setDataMode('live');
  vi.unstubAllGlobals();
});

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

describe('CodeQ preview recovery', () => {
  it('releases an exited process pointer and lets the user explicitly start a replacement', async () => {
    useWorkspaceState.getState().setDataMode('live');
    window.localStorage.setItem('oa-second-codeq-live-preview:p1', 'preview-process-exited');
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      if (url.includes('/api/codebase/projects?')) return json({ projects: [{ id: 'p1', name: 'Project One' }] });
      if (url.endsWith('/api/codebase/projects/p1/tree')) return json({ entries: [] });
      if (url.endsWith('/api/codebase/projects/p1')) return json({ project: { id: 'p1', name: 'Project One' } });
      if (url.endsWith('/api/codebase/projects/p1/tools') && body?.tool === 'process_status') return json({ result: { id: 'preview-process-exited', status: 'exited', exit_code: 1 } });
      if (url.endsWith('/api/codebase/projects/p1/tools') && body?.tool === 'inspect_logs') return json({ result: { id: 'preview-process-exited', stdout: 'Build failed.' } });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } } });
    render(<QueryClientProvider client={client}><CodeQScreen/></QueryClientProvider>);

    expect(await screen.findByText('Preview process exited with code 1.')).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Stop preview' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Start another preview' }));
    expect(await screen.findByRole('textbox', { name: 'Preview command' })).toBeVisible();
    expect(window.localStorage.getItem('oa-second-codeq-live-preview:p1')).toBeNull();
    expect(fetchMock.mock.calls.some(([input, init]) => String(input).includes('/api/codebase/projects/p1/tools') && JSON.parse(String(init?.body)).tool === 'start_process')).toBe(false);
  });
});
