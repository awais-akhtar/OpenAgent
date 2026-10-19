import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor, cleanup } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SettingsScreen } from '../src/screens/OperationsScreens';

const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
function renderSettings() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><SettingsScreen /></QueryClientProvider>);
}
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('settings backend workflows', () => {
  it('loads backend notifications and marks an unread record through its scoped endpoint', async () => {
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    let notifications = [{ id: 'notice-1', title: 'Task completed', body: 'Daily review finished.', level: 'info', created_at: '2026-10-14T10:00:00Z', read_at: null }];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = init?.method ?? 'GET';
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      calls.push({ url, method, body });
      if (url.includes('/api/tasks/notifications') && method === 'GET') return json({ notifications });
      if (url.includes('/api/tasks/notifications/notice-1/read') && method === 'POST') {
        notifications = notifications.map(item => ({ ...item, read_at: '2026-10-14T10:02:00Z' }));
        return json({ notification: notifications[0] });
      }
      return json({ error: `Unexpected ${method} ${url}` }, 404);
    }));
    renderSettings();
    fireEvent.click(screen.getByRole('tab', { name: 'Notifications' }));
    const markRead = await screen.findByRole('button', { name: 'Mark read' });
    expect(markRead.closest('article')).toHaveTextContent('Daily review finished.');
    fireEvent.click(markRead);
    await waitFor(() => expect(screen.getByText('Notification marked as read.')).toBeVisible());
    expect(screen.getByText(/Read$/)).toBeInTheDocument();
    expect(calls.find(call => call.method === 'POST')?.body).toEqual({ project: 'default' });
    expect(calls.find(call => call.method === 'POST')?.url).toContain('/api/tasks/notifications/notice-1/read');
  });

  it('shows actual provider and privacy routing facts from their read-only backend APIs', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/models')) return json({ providers: [{ id: 'local', label: 'Local Gateway', local: true, available: true, default_model: 'small-chat' }] });
      if (url.endsWith('/api/model-policy')) return json({ routing_policy: 'local-first', fallback_policy: 'none' });
      if (url.endsWith('/api/security/status')) return json({ encryption: 'enabled', audit_chain: true });
      if (url.endsWith('/api/security/egress')) return json({ requests: [{ id: 'e-1', status: 'pending' }] });
      return json({ error: `Unexpected ${url}` }, 404);
    }));
    renderSettings();
    fireEvent.click(screen.getByRole('tab', { name: 'Models & providers' }));
    expect(await screen.findByText('Local Gateway')).toBeVisible();
    expect(screen.getByText('Available · small-chat')).toBeVisible();
    fireEvent.click(screen.getByRole('tab', { name: 'Privacy' }));
    expect(await screen.findByText('local-first')).toBeVisible();
    expect(screen.getByText('enabled')).toBeVisible();
    expect(screen.getByText('1 · 1 pending')).toBeVisible();
  });
});
