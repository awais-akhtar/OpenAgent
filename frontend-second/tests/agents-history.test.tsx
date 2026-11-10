import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, cleanup } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { AgentsScreen } from '../src/screens/OperationsScreens';
import { useWorkspaceState } from '../src/state';

const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
function renderAgents() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><AgentsScreen /></QueryClientProvider>);
}
afterEach(() => { cleanup(); vi.unstubAllGlobals(); useWorkspaceState.getState().setDataMode('demo'); window.history.replaceState({}, '', '/'); });

describe('agent history authorization boundary', () => {
  it('requests only the selected project history in Live and renders the secured response', async () => {
    window.history.replaceState({}, '', '/?project=research-room');
    useWorkspaceState.getState().setDataMode('live');
    const requests: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input); requests.push(url);
      if (url.endsWith('/api/agents')) return json({ agents: [{ id: 'research', name: 'Researcher' }] });
      if (url === '/api/agents/jobs?project=research-room&limit=50') return json({ jobs: [] });
      if (url === '/api/agent-runs?project=research-room&limit=50') return json({ runs: [{ id: 'run-1', goal: 'Owned research task', status: 'completed' }] });
      return json({ error: `Unexpected ${url}` }, 404);
    }));
    renderAgents();
    await screen.findByText('1 matching profiles');
    expect(await screen.findByText('Run history · 1')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Owned research task/ })).toBeInTheDocument();
    expect(requests.some(url => url.includes('/api/agents'))).toBe(true);
    expect(requests).toContain('/api/agent-runs?project=research-room&limit=50');
  });

  it('keeps a selected job and a historical run from mixing goals or results', async () => {
    useWorkspaceState.getState().setDataMode('live');
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/agents')) return json({ agents: [{ id: 'research', name: 'Researcher' }] });
      if (url.includes('/api/agents/jobs?')) return json({ jobs: [
        { id: 'job-running', agent: 'research', goal: 'Running current task', status: 'running', progress: 30, result: {} },
        { id: 'job-complete', agent: 'research', goal: 'Finished prior task', status: 'completed', progress: 100, result: { run: { id: 'run-complete', goal: 'Finished prior task', agent: 'research', response: 'Verified prior output' } } },
      ] });
      if (url.includes('/api/agents/jobs/') && url.includes('/events?')) return json({ events: [], next: '' });
      if (url.includes('/api/agent-runs?')) return json({ runs: [{ id: 'run-old', goal: 'Unrelated historical task', agent: 'research', response: 'Old output', status: 'completed' }] });
      return json({ error: `Unexpected ${url}` }, 404);
    }));
    renderAgents();
    expect(await screen.findByRole('heading', { name: 'Running current task' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Unrelated historical task/ }));
    expect(await screen.findByRole('heading', { name: 'Unrelated historical task' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Finished prior task/ }));
    expect(await screen.findByRole('heading', { name: 'Finished prior task' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Running current task/ }));
    expect(await screen.findByRole('heading', { name: 'Running current task' })).toBeInTheDocument();
    expect(screen.queryByText('Old output')).not.toBeInTheDocument();
  });
});
