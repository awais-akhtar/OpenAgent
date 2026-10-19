import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import AdvancedScreen from '../src/screens/AdvancedScreen';

function renderAssistant(tab?: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const view = render(<QueryClientProvider client={client}><AdvancedScreen route="assistant" /></QueryClientProvider>);
  if (tab && tab !== 'Today') fireEvent.click(screen.getByRole('tab', { name: tab }));
  return view;
}

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); sessionStorage.clear(); window.history.replaceState({}, '', '/'); });

describe('Assistant workspace backend workflows', () => {
  it('creates a task with backend-supported due, priority and goal fields', async () => {
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    let tasks = [{ id: 'task-open', title: 'Prepare report', status: 'open', priority: 'high', due_at: '', goal_id: '', project: 'default' }];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = init?.method ?? 'GET';
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      calls.push({ url, method, body });
      if (url.endsWith('/api/assistant/tasks') && method === 'POST') {
        const task = { id: 'task-new', ...body, status: 'open' }; tasks = [task, ...tasks]; return json({ task }, 201);
      }
      if (url.endsWith('/api/assistant/tasks')) return json({ tasks });
      if (url.endsWith('/api/assistant/goals')) return json({ goals: [{ id: 'goal-1', title: 'Launch project', status: 'active' }] });
      if (url.endsWith('/api/assistant/daily-brief')) return json({ project: 'default', open_tasks: tasks, active_goals: [{ id: 'goal-1', title: 'Launch project', status: 'active' }], priority_count: 1, recommended_focus: tasks.slice(0, 1) });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));

    renderAssistant('Tasks');
    fireEvent.change(await screen.findByRole('textbox', { name: 'Task' }), { target: { value: 'Review release' } });
    fireEvent.change(screen.getByLabelText('Priority'), { target: { value: 'urgent' } });
    fireEvent.change(screen.getByLabelText('Due date'), { target: { value: '2026-10-15T14:30' } });
    expect(await screen.findByRole('option', { name: 'Launch project' })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Related goal'), { target: { value: 'goal-1' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add task' }));
    expect(await screen.findByText(/Task saved by the backend/)).toBeVisible();
    const request = calls.find((call) => call.url.endsWith('/api/assistant/tasks') && call.method === 'POST');
    expect(request?.body).toMatchObject({ project: 'default', title: 'Review release', priority: 'urgent', goal_id: 'goal-1' });
    expect(new Date(String(request?.body?.due_at)).toISOString()).toBe(new Date('2026-10-15T14:30').toISOString());
  });

  it('completes a task only through the backend complete contract', async () => {
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    let tasks = [{ id: 'task-1', title: 'Review release', status: 'open', priority: 'normal', goal_id: '', project: 'default' }];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = init?.method ?? 'GET'; const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      calls.push({ url, method, body });
      if (url.endsWith('/api/assistant/tasks/complete')) { tasks = tasks.map((task) => task.id === body?.task_id ? { ...task, status: 'completed' } : task); return json({ task: tasks[0] }); }
      if (url.endsWith('/api/assistant/tasks')) return json({ tasks });
      if (url.endsWith('/api/assistant/goals')) return json({ goals: [] });
      if (url.endsWith('/api/assistant/daily-brief')) return json({ project: 'default', open_tasks: tasks.filter((task) => task.status === 'open'), active_goals: [], priority_count: 0, recommended_focus: [] });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));

    renderAssistant('Tasks');
    fireEvent.click(await screen.findByRole('button', { name: 'Mark complete' }));
    expect(await screen.findByText(/Backend task status: completed/)).toBeVisible();
    expect(calls.find((call) => call.url.endsWith('/api/assistant/tasks/complete'))?.body).toEqual({ project: 'default', task_id: 'task-1' });
  });

  it('renders returned daily brief records as-is and avoids fabricated calendar items', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/assistant/tasks')) return json({ tasks: [] });
      if (url.endsWith('/api/assistant/daily-brief')) return json({ project: 'default', open_tasks: [{ id: 'task-brief', title: 'Send proposal', status: 'open', priority: 'high', due_at: '2026-10-15T14:30:00Z' }], active_goals: [{ id: 'goal-brief', title: 'Close first customer', status: 'active', horizon: 'short' }], priority_count: 1, recommended_focus: [{ id: 'task-brief', title: 'Send proposal', status: 'open', priority: 'high' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderAssistant('Daily brief');
    expect((await screen.findAllByText('Send proposal')).length).toBeGreaterThan(0);
    expect(screen.getByText('Close first customer')).toBeVisible();
    expect(screen.getByText(/1 high or urgent open task/)).toBeVisible();
    expect(screen.queryByText(/upcoming meetings/i)).not.toBeInTheDocument();
  });

  it('shows an honest empty brief when the backend returns no content', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/assistant/tasks')) return json({ tasks: [] });
      if (url.endsWith('/api/assistant/daily-brief')) return json({ project: 'default', open_tasks: [], active_goals: [], priority_count: 0, recommended_focus: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    const view = renderAssistant('Daily brief');
    expect((await screen.findAllByText('No tasks returned')).length).toBeGreaterThan(0);
    expect(view.container.textContent).not.toMatch(/calendar event|meeting at|scheduled for today/i);
  });
});
