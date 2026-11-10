import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ChatScreen } from '../src/screens/HomeChat';
import { useWorkspaceState } from '../src/state';

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
const renderChat = () => render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><ChatScreen /></QueryClientProvider>);

beforeEach(() => useWorkspaceState.getState().setDataMode('live'));
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks(); useWorkspaceState.getState().setDataMode('demo'); });

describe('Chat workspace tools', () => {
  it('keeps tools in Chat and shows genuine knowledge results separately from assistant messages', async () => {
    const urls: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input); urls.push(url);
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations')) return json({ conversations: [] });
      if (url.includes('/api/knowledge/search')) return json({ results: [{ id: 'source-1', title: 'Local guide', snippet: 'A measured finding.' }] });
      return json({ error: `Unexpected ${url}` }, 404);
    }));
    renderChat();
    fireEvent.click(screen.getByRole('button', { name: /Tools/ }));
    fireEvent.click(screen.getByRole('button', { name: /Search knowledge/ }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Knowledge search terms' }), { target: { value: 'local guide' } });
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    expect(await screen.findByRole('region', { name: 'Knowledge search' })).toHaveTextContent('Local guide');
    expect(screen.getByRole('region', { name: 'Knowledge search' })).toHaveTextContent('not saved as an assistant reply');
    expect(document.querySelectorAll('.chat-thread .chat-message')).toHaveLength(0);
    expect(urls.some(url => url.includes('/api/knowledge/search?project=default&q=local%20guide'))).toBe(true);
  });

  it('downloads the backend document bytes and does not claim success for a missing artifact', async () => {
    const downloads: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations')) return json({ conversations: [] });
      if (url.includes('/api/chat/artifact')) return json({ artifact: { filename: 'report.pdf', content_type: 'application/pdf', base64: btoa('real document bytes') } });
      return json({ error: `Unexpected ${url}` }, 404);
    }));
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: vi.fn(() => 'blob:report'), revokeObjectURL: vi.fn() }));
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) { downloads.push(this.download); });
    renderChat();
    fireEvent.click(screen.getByRole('button', { name: /Tools/ }));
    fireEvent.click(screen.getByRole('button', { name: /Make a document/ }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Document title' }), { target: { value: 'Report' } });
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    await waitFor(() => expect(downloads).toContain('report.pdf'));
    expect(screen.getByRole('region', { name: 'Document' })).toHaveTextContent('Downloaded report.pdf');
  });
});
