import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, expect, it, vi } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import { App } from '../src/App';
import { useWorkspaceState } from '../src/state';

afterEach(() => { cleanup(); vi.restoreAllMocks(); localStorage.removeItem('oa-second-isolated-demo-v1'); });

const hash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const chats = [
  { title: 'Review React project structure', time: '2 minutes ago' },
  { title: 'Add authentication to dashboard', time: '1 hour ago' },
  { title: 'Create data analysis agent', time: '3 hours ago' },
  { title: 'Plan project architecture', time: 'Yesterday' },
  { title: 'Fix deployment issue', time: 'Monday' },
];

it('opens Chat to a clean offline state instead of replaying shell conversation fixtures', async () => {
  localStorage.setItem('oa-second-data-mode', 'demo');
  localStorage.removeItem('oa-second-isolated-demo-v1');
  useWorkspaceState.getState().setDataMode('demo');
  const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async input => {
    const url = String(input);
    if (url.endsWith('/demo/reference-fixtures.json')) return new Response(JSON.stringify({ sourceSha256: hash, data: { topLevel: {}, advanced: {} } }), { status: 200 });
    if (url.endsWith('/demo/approved-shell-fixtures.json')) return new Response(JSON.stringify({ sourceSha256: hash, data: { chats } }), { status: 200 });
    throw new Error(`Unexpected demo request: ${url}`);
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  try {
    window.history.replaceState({}, '', '/?view=chat');
    render(<QueryClientProvider client={client}><BrowserRouter><App /></BrowserRouter></QueryClientProvider>);
    expect(await screen.findByRole('heading', { name: 'Start a conversation' })).toBeVisible();
    expect(screen.getByRole('combobox', { name: 'Conversation' }).querySelectorAll('option')).toHaveLength(1);
    expect(document.querySelectorAll('#content .chat-thread .chat-message')).toHaveLength(0);
    expect(screen.getByText('Chat is not simulated in offline Demo mode. Switch to Live to use the connected chat service.')).toBeVisible();
    expect(screen.getByText('DESIGN PREVIEW')).toBeVisible();
    expect(screen.getByText('Offline reference')).toBeVisible();
    await waitFor(() => expect(fetchMock.mock.calls.every(([input]) => !String(input).startsWith('/api/'))).toBe(true));
  } finally {
    client.clear();
    useWorkspaceState.getState().setDataMode('live');
  }
});
