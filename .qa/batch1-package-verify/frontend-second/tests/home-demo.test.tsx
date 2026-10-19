import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { HomeScreen } from '../src/screens/HomeChat';
import { demoProvider } from '../src/demo/provider';
import { useWorkspaceState } from '../src/state';

afterEach(cleanup);

const sourceSha256 = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const referenceFixtures = {
  sourceSha256,
  data: { topLevel: {}, advanced: {} },
};
const approvedShell = {
  sourceSha256,
  data: {
    projects: [{ name: 'E-commerce Platform', tag: 'Next.js · React · TypeScript', color: 'violet', updated: '2 minutes ago' }],
    chats: [{ title: 'Review React project structure', time: '2 minutes ago' }],
    agents: [{ name: 'Researcher', role: 'Finds and checks information and sources', status: 'Available' }],
    activity: [{ title: 'Completed a research summary', detail: 'AI Research Assistant · Research', time: '28m ago' }],
    homeTasks: [
      { id: 'home-task-1', title: 'Review product requirements', completed: true },
      { id: 'home-task-2', title: 'Check research findings', completed: true },
      { id: 'home-task-3', title: 'Prepare launch recommendations', completed: false },
      { id: 'home-task-4', title: 'Update documentation', completed: false },
    ],
    runs: [], automations: [],
  },
};

describe('Home demo fixtures', () => {
  it('renders approved shell records and completes a task in local demo storage', async () => {
    localStorage.removeItem('oa-second-isolated-demo-v1');
    useWorkspaceState.getState().setDataMode('demo');
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async input => {
      const url = String(input);
      if (url.endsWith('/demo/reference-fixtures.json')) return new Response(JSON.stringify(referenceFixtures), { status: 200 });
      if (url.endsWith('/demo/approved-shell-fixtures.json')) return new Response(JSON.stringify(approvedShell), { status: 200 });
      throw new Error(`Unexpected network request in demo mode: ${url}`);
    });
    try {
      await demoProvider.reset();
      const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
      render(<QueryClientProvider client={client}><HomeScreen /></QueryClientProvider>);

      expect(await screen.findByRole('heading', { name: 'E-commerce Platform' })).toBeVisible();
      expect(screen.getByText('Next.js · React · TypeScript')).toBeVisible();
      expect(screen.getByText('Completed a research summary')).toBeVisible();
      expect(screen.getByText('AI Research Assistant · Research')).toBeVisible();
      expect(screen.getByText('Finds and checks information and sources')).toBeVisible();

      const taskCheckbox = await screen.findByRole('checkbox', { name: 'Mark Prepare launch recommendations complete' });
      fireEvent.click(taskCheckbox);
      expect(await screen.findByRole('checkbox', { name: 'Mark Prepare launch recommendations incomplete' })).toBeChecked();
      const stored = JSON.parse(localStorage.getItem('oa-second-isolated-demo-v1') ?? '{}') as { data?: { shell?: { homeTasks?: { completed?: boolean }[] } } };
      expect(stored.data?.shell?.homeTasks?.[2]?.completed).toBe(true);
      await waitFor(() => expect(fetchMock.mock.calls.some(([request]) => String(request).startsWith('/api/'))).toBe(false));
    } finally {
      await demoProvider.reset();
      useWorkspaceState.getState().setDataMode('live');
      fetchMock.mockRestore();
    }
  });
});
