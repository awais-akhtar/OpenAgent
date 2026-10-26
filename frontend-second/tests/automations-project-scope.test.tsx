import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useWorkspaceState } from '../src/state';
import { AutomationsScreen } from '../src/screens/OperationsScreens';

const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), {
  status,
  headers: { 'Content-Type': 'application/json' },
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  localStorage.removeItem('oa-project');
  useWorkspaceState.getState().setDataMode('demo');
  window.history.replaceState({}, '', '/');
});

describe('Live automation project scope', () => {
  it('loads and creates workflows under the active project instead of default', async () => {
    useWorkspaceState.getState().setDataMode('live');
    window.history.replaceState({}, '', '/?project=project-42');
    const requests: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = String(init?.method ?? 'GET');
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : {};
      requests.push({ url, method, body });
      if (url.includes('/api/workflows/builder')) return json({ actions: { add_event: { label: 'Add Memory' } } });
      if (url.includes('/api/workflows?')) return json({ workflows: [] });
      if (url.includes('/api/workflow-runs?')) return json({ runs: [] });
      if (url.endsWith('/api/workflows') && method === 'POST') return json({ workflow: { id: 'project-workflow', ...body } }, 201);
      return json({ error: `Unexpected ${method} ${url}` }, 404);
    }));

    const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    render(<QueryClientProvider client={client}><AutomationsScreen /></QueryClientProvider>);

    await screen.findByRole('heading', { name: 'No automations' });
    await waitFor(() => {
      expect(requests.some(request => request.url.includes('/api/workflows?project=project-42&limit=100'))).toBe(true);
      expect(requests.some(request => request.url.includes('/api/workflow-runs?project=project-42&limit=100'))).toBe(true);
    });

    fireEvent.click(screen.getByRole('button', { name: /New/ }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Name' }), { target: { value: 'Scoped workflow' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'New step 1 action' }), { target: { value: 'add_event' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'New step 1 Title' }), { target: { value: 'Project note' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'New step 1 Content' }), { target: { value: 'Saved in the selected project' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save automation' }));

    await waitFor(() => expect(requests.some(request => request.method === 'POST' && request.url.endsWith('/api/workflows'))).toBe(true));
    expect(requests.find(request => request.method === 'POST' && request.url.endsWith('/api/workflows'))?.body).toMatchObject({
      project: 'project-42',
      name: 'Scoped workflow',
    });
  });
});
