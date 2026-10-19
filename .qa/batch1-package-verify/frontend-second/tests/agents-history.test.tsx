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
});
