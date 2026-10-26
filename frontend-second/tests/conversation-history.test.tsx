import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor, cleanup } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ChatScreen } from '../src/screens/HomeChat';
import { useWorkspaceState } from '../src/state';

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
}

function renderChat() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><ChatScreen /></QueryClientProvider>);
}

function openConversationHistory() {
  if (!screen.queryByRole('button', { name: 'Conversation history' })) {
    fireEvent.click(screen.getByRole('button', { name: 'Conversation actions' }));
  }
  fireEvent.click(screen.getByRole('button', { name: 'Conversation history' }));
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); sessionStorage.clear(); history.replaceState({}, '', '/'); useWorkspaceState.getState().setDataMode('demo'); });
beforeEach(() => { useWorkspaceState.getState().setDataMode('live'); });

describe('saved conversation history', () => {
  it('confirms and deletes a saved conversation through the existing Live endpoint', async () => {
    const requests: Array<{ url: string; method: string }> = [];
    let deleted = false;
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = init?.method ?? 'GET'; requests.push({ url, method });
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations/organized')) return json({ conversations: deleted ? [] : [{ id: 'c-delete', title: 'Disposable conversation', pinned: false, archived: false }] });
      if (url.includes('/api/conversations?')) return json({ conversations: deleted ? [] : [{ id: 'c-delete', title: 'Disposable conversation' }] });
      if (url.endsWith('/api/conversations/c-delete') && method === 'DELETE') { deleted = true; return json({ action: 'delete', changed: [{ id: 'c-delete', deleted: true }], count: 1 }); }
      if (url.includes('/api/messages')) return json({ messages: [] });
      return json({ error: `Unexpected ${method} ${url}` }, 404);
    }));

    renderChat();
    openConversationHistory();
    fireEvent.click(await screen.findByRole('button', { name: 'Delete Disposable conversation' }));
    await waitFor(() => expect(requests.some(request => request.method === 'DELETE' && request.url.endsWith('/api/conversations/c-delete'))).toBe(true));
    expect(await screen.findByText('Conversation deleted.')).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Disposable conversation' })).not.toBeInTheDocument();
    expect(confirm).toHaveBeenCalledWith('Delete “Disposable conversation”? This permanently removes the conversation and its messages.');
  });

  it('searches backend history and sends pin, rename and archive changes to the organization API', async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal('prompt', vi.fn(() => 'Renamed from UI'));
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations/organized')) return json({ conversations: [{ id: 'c-1', title: 'Project discussion', pinned: false, archived: false, semantic_score: 2 }] });
      if (url.includes('/api/conversations?')) return json({ conversations: [{ id: 'c-1', title: 'Project discussion' }] });
      if (url.includes('/api/conversations/c-1')) return json({ conversation: { id: 'c-1' } });
      if (url.includes('/api/messages')) return json({ messages: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderChat();
    openConversationHistory();
    const search = screen.getByRole('textbox', { name: 'Search conversation history' });
    fireEvent.change(search, { target: { value: 'project' } });
    expect(await screen.findByRole('button', { name: 'Pin Project discussion' })).toBeVisible();
    await waitFor(() => expect(requests.some(request => request.url.includes('sort=semantic') && request.url.includes('q=project'))).toBe(true));

    fireEvent.click(screen.getByRole('button', { name: 'Pin Project discussion' }));
    await waitFor(() => expect(requests.some(request => request.init?.method === 'PATCH' && request.url.endsWith('/api/conversations/c-1') && JSON.parse(String(request.init.body)).pinned === true)).toBe(true));

    fireEvent.click(screen.getByRole('button', { name: 'Rename Project discussion' }));
    await waitFor(() => expect(requests.some(request => request.init?.method === 'PATCH' && JSON.parse(String(request.init.body)).title === 'Renamed from UI')).toBe(true));

    const archiveButton = screen.getByRole('button', { name: 'Archive Project discussion' });
    // History actions intentionally disable while the previous mutation and
    // its refetch are settling; wait for the real interaction to be enabled.
    await waitFor(() => expect(archiveButton).toBeEnabled());
    fireEvent.click(archiveButton);
    await waitFor(() => expect(requests.some(request => request.init?.method === 'PATCH' && JSON.parse(String(request.init.body)).archived === true)).toBe(true));
  });

  it('selects a saved conversation with a shareable URL and reports backend action failures', async () => {
    let failPin = false;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations/organized')) return json({ conversations: [{ id: 'linked-c', title: 'Linked conversation', pinned: false }] });
      if (url.includes('/api/conversations?')) return json({ conversations: [{ id: 'linked-c', title: 'Linked conversation' }] });
      if (url.includes('/api/messages')) return json({ messages: [{ role: 'user', content: 'Saved question' }] });
      if (url.endsWith('/api/conversations/linked-c') && init?.method === 'PATCH' && failPin) return json({ error: 'Permission denied.' }, 403);
      if (url.endsWith('/api/conversations/linked-c')) return json({ conversation: { id: 'linked-c' } });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderChat();
    openConversationHistory();
    fireEvent.click(await screen.findByRole('button', { name: 'Linked conversation' }));
    expect(location.search).toContain('conversation_id=linked-c');
    expect(await screen.findByText('Saved question')).toBeVisible();

    openConversationHistory();
    failPin = true;
    fireEvent.click(await screen.findByRole('button', { name: 'Pin Linked conversation' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('This account does not have permission for this action.');
  });

  it('exports a conversation through the backend without asking the server to write a file', async () => {
    const created = vi.fn(() => 'blob:export');
    const NativeURL = URL;
    const clicked: string[] = [];
    vi.stubGlobal('URL', Object.assign(NativeURL, { createObjectURL: created, revokeObjectURL: vi.fn() }));
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) { clicked.push(this.download); });
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations?')) return json({ conversations: [{ id: 'c-export', title: 'Export me' }] });
      if (url.includes('/api/messages')) return json({ messages: [] });
      if (url.includes('/api/chat/export')) return json({ export: { format: 'openagent.conversation.v1', conversation: { title: 'Export me' }, messages: [] } });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    history.replaceState({}, '', '/?view=chat&conversation_id=c-export');
    renderChat();
    fireEvent.click(screen.getByRole('button', { name: 'Conversation actions' }));
    fireEvent.click(screen.getByRole('button', { name: 'Export JSON' }));
    await waitFor(() => expect(clicked).toContain('Export-me.openagent.json'));
    expect(requests.some(request => request.url.includes('/api/chat/export?conversation_id=c-export&write_file=false'))).toBe(true);
    expect(created).toHaveBeenCalledOnce();
  });

  it('branches from a persisted message and opens the backend-created conversation', async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations?')) return json({ conversations: [{ id: 'c-source', title: 'Source' }] });
      if (url.includes('/api/messages?conversation_id=c-source')) return json({ messages: [{ id: 'm-1', role: 'user', content: 'Question' }, { id: 'm-2', role: 'assistant', content: 'Answer' }] });
      if (url.endsWith('/api/chat/branch')) return json({ branch: { conversation: { id: 'c-branch' } } }, 201);
      if (url.includes('/api/conversations/organized')) return json({ conversations: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    history.replaceState({}, '', '/?view=chat&conversation_id=c-source');
    renderChat();
    fireEvent.click(await screen.findByRole('button', { name: 'Actions for assistant message 2' }));
    fireEvent.click(screen.getByRole('button', { name: 'Branch from here' }));
    await waitFor(() => expect(requests.some(request => request.url.endsWith('/api/chat/branch') && JSON.parse(String(request.init?.body)).at_message_id === 'm-2')).toBe(true));
    await waitFor(() => expect(location.search).toContain('conversation_id=c-branch'));
  });

  it('validates imported OpenAgent exports before posting them to the backend', async () => {
    const requests: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push(url);
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations?')) return json({ conversations: [] });
      if (url.includes('/api/chat/import')) return json({ imported: { conversation: { id: 'c-imported' } } }, 201);
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderChat();
    fireEvent.click(screen.getByRole('button', { name: 'Conversation actions' }));
    const input = screen.getByLabelText('Choose conversation export');
    fireEvent.change(input, { target: { files: [new File(['{}'], 'invalid.json', { type: 'application/json' })] } });
    expect(await screen.findByRole('alert')).toHaveTextContent('not a supported OpenAgent conversation export');
    expect(requests.some(url => url.includes('/api/chat/import'))).toBe(false);
    const valid = { format: 'openagent.conversation.v1', conversation: { title: 'Imported' }, messages: [{ role: 'user', content: 'Hello' }] };
    fireEvent.change(input, { target: { files: [new File([JSON.stringify(valid)], 'conversation.json', { type: 'application/json' })] } });
    await waitFor(() => expect(requests.some(url => url.includes('/api/chat/import'))).toBe(true));
    await waitFor(() => expect(location.search).toContain('conversation_id=c-imported'));
  });

  it('renders replay events only from the backend timeline response', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations?')) return json({ conversations: [{ id: 'c-replay', title: 'Replay' }] });
      if (url.includes('/api/messages')) return json({ messages: [] });
      if (url.includes('/api/chat/replay')) return json({ replay: { message_count: 1, timeline: [{ role: 'assistant', content: 'Backend timeline entry', created_at: '2026-10-14T00:00:00Z' }] } });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    history.replaceState({}, '', '/?view=chat&conversation_id=c-replay');
    renderChat();
    fireEvent.click(screen.getByRole('button', { name: 'Conversation actions' }));
    fireEvent.click(screen.getByRole('button', { name: 'Replay timeline' }));
    expect(await screen.findByRole('dialog', { name: 'Conversation replay' })).toBeVisible();
    expect(screen.getByText('Backend timeline entry')).toBeVisible();
  });

  it('creates a Markdown artifact from an actual assistant message through the artifact API', async () => {
    const clicked: string[] = [];
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) { clicked.push(this.download); });
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations?')) return json({ conversations: [{ id: 'c-artifact', title: 'Artifact' }] });
      if (url.includes('/api/messages')) return json({ messages: [{ id: 'm-answer', role: 'assistant', content: 'A real answer from the backend.' }] });
      if (url.endsWith('/api/chat/artifact')) return json({ artifact: { filename: 'reply.md', content_type: 'text/markdown', base64: btoa('A real answer from the backend.') } }, 201);
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    history.replaceState({}, '', '/?view=chat&conversation_id=c-artifact');
    renderChat();
    fireEvent.click(await screen.findByRole('button', { name: 'Actions for assistant message 1' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save as Markdown' }));
    await waitFor(() => expect(clicked).toContain('reply.md'));
    const request = requests.find(item => item.url.endsWith('/api/chat/artifact'));
    expect(JSON.parse(String(request?.init?.body))).toMatchObject({ content: 'A real answer from the backend.', format: 'md', project: 'default' });
  });
});
