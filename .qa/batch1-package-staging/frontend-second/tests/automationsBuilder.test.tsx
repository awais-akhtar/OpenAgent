import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { AutomationsScreen } from '../src/screens/OperationsScreens';

function renderScreen() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><AutomationsScreen /></QueryClientProvider>);
}
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
const actions = {
  add_event: { label: 'Add Memory', category: 'memory' },
  file_write: { label: 'Write File', category: 'file' },
  browser_fetch: { label: 'Fetch Web Page', category: 'browser', requires_network: true },
};

afterEach(() => { cleanup(); vi.unstubAllGlobals(); sessionStorage.clear(); window.history.replaceState({}, '', '/'); });

describe('automation builder workflows', () => {
  it('creates a real workflow with constrained backend-supported step data', async () => {
    const requests: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET');
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : {};
      requests.push({ url, method, body });
      if (url.includes('/api/workflows/builder')) return json({ actions });
      if (url.includes('/api/workflows?')) return json({ workflows: [] });
      if (url.includes('/api/workflow-runs')) return json({ runs: [] });
      if (url.endsWith('/api/workflows') && method === 'POST') return json({ workflow: { id: 'new-1', ...body } }, 201);
      return json({ error: `Unexpected ${method} ${url}` }, 404);
    }));

    renderScreen();
    fireEvent.click(await screen.findByRole('button', { name: 'New' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Name' }), { target: { value: 'Record summary' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'New step 1 action' }), { target: { value: 'add_event' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'New step 1 Title' }), { target: { value: 'Daily summary' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'New step 1 Content' }), { target: { value: 'Reviewed today' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save automation' }));

    await waitFor(() => expect(requests.some(request => request.method === 'POST' && request.url.endsWith('/api/workflows'))).toBe(true));
    const created = requests.find(request => request.method === 'POST' && request.url.endsWith('/api/workflows'));
    expect(created?.body).toMatchObject({
      name: 'Record summary', project: 'default', trigger_type: 'manual',
      steps: [{ action: 'add_event', title: 'Daily summary', content: 'Reviewed today' }],
    });
  });

  it('previews first and sends network disabled unless the user explicitly opts in for live execution', async () => {
    const requests: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    const workflow = { id: 'wf-1', name: 'Fetch reference', description: 'Review a remote page', project: 'default', enabled: true, trigger_type: 'manual', steps: [{ action: 'browser_fetch', url: 'https://example.test' }] };
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET');
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : {};
      requests.push({ url, method, body });
      if (url.includes('/api/workflows/builder')) return json({ actions });
      if (url.includes('/api/workflows?')) return json({ workflows: [workflow] });
      if (url.includes('/api/workflow-runs')) return json({ runs: [{ id: 'old-1', workflow_id: 'wf-1', status: 'completed', result: { steps: [{ action: 'browser_fetch', status: 'completed' }] } }] });
      if (url.endsWith('/wf-1/run')) return json({ run: { id: `run-${requests.length}`, workflow_id: 'wf-1', status: 'completed', result: { dry_run: body.dry_run, steps: [{ action: 'browser_fetch', status: body.dry_run ? 'planned' : 'completed' }] } } }, 201);
      return json({ error: `Unexpected ${method} ${url}` }, 404);
    }));

    renderScreen();
    fireEvent.click(await screen.findByRole('button', { name: /Fetch reference/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Preview run' }));
    await screen.findByText('Preview result');
    expect(requests.find(request => request.url.endsWith('/wf-1/run'))?.body).toMatchObject({ dry_run: true, allow_network: false });
    expect(await screen.findByText('planned')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow network access for this live run' }));
    fireEvent.click(screen.getByRole('button', { name: 'Confirm live run' }));
    await waitFor(() => expect(requests.filter(request => request.url.endsWith('/wf-1/run'))).toHaveLength(2));
    expect(requests.filter(request => request.url.endsWith('/wf-1/run'))[1].body).toMatchObject({ dry_run: false, allow_network: true });

    fireEvent.click(screen.getByRole('tab', { name: 'Runs' }));
    fireEvent.click(screen.getByRole('button', { name: /wf-1/ }));
    expect(await screen.findAllByText('browser_fetch')).toHaveLength(2);
  });
});
