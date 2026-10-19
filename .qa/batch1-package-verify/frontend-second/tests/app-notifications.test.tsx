import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, cleanup, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { App } from '../src/App';

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('backend notifications in the app shell', () => {
  it('loads real scheduler notifications when the bell is opened', async () => {
    const requests: string[] = [];
    let read = false;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input); requests.push(url);
      if (url.includes('/api/tasks/notifications/n-1/read')) { read = true; return new Response(JSON.stringify({ notification: { id: 'n-1', title: 'Review due', read_at: '2026-10-14T10:01:00Z' } }), { status: 200 }); }
      if (url.includes('/api/health')) return new Response(JSON.stringify({ status: 'ok' }), { status: 200 });
      if (url.includes('/api/conversations')) return new Response(JSON.stringify({ conversations: [] }), { status: 200 });
      if (url.includes('/api/tasks/notifications')) return new Response(JSON.stringify({ notifications: [{ id: 'n-1', level: 'warning', title: 'Review due', body: 'A scheduled task needs attention.', created_at: '2026-10-14T10:00:00Z', read_at: read ? '2026-10-14T10:01:00Z' : '' }] }), { status: 200 });
      return new Response(JSON.stringify({ error: `Unexpected request ${url}` }), { status: 404 });
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><MemoryRouter initialEntries={['/?view=home']}><App /></MemoryRouter></QueryClientProvider>);

    fireEvent.click(screen.getByRole('button', { name: 'Open notifications' }));
    expect(await screen.findByRole('heading', { name: 'Notifications' })).toBeInTheDocument();
    expect(await screen.findByText('Review due')).toBeInTheDocument();
    expect(screen.getByText('A scheduled task needs attention.')).toBeInTheDocument();
    expect(requests.some(url => url.includes('/api/tasks/notifications?project=default'))).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: 'Mark read' }));
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Mark read' })).not.toBeInTheDocument());
    expect(requests.some(url => url.includes('/api/tasks/notifications/n-1/read'))).toBe(true);
  });
});


