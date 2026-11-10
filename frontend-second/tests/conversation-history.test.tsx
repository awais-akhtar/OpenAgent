import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor, cleanup, within } from '@testing-library/react';
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

/**
 * Conversation history, export, replay and import moved out of a header
 * "Conversation actions" menu and into the conversations panel beside the
 * thread, so these drive the panel's own controls.
 */
function openConversationHistory() {
  fireEvent.click(screen.getByRole('button', { name: /Archived chats|Load all/ }));
}

/**
 * The conversations panel lists recent chats with their own row controls, so
 * pin, rename, archive and delete each appear twice once history is open.
 * These assertions are about the history panel, so they are scoped to it.
 */
function inHistory() {
  return within(screen.getByLabelText('Conversation history'));
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
    fireEvent.click(await inHistory().findByRole('button', { name: 'Delete Disposable conversation' }));
    fireEvent.click(await inHistory().findByRole('button', { name: 'Delete', exact: true }));
    await waitFor(() => expect(requests.some(request => request.method === 'DELETE' && request.url.endsWith('/api/conversations/c-delete'))).toBe(true));
    expect(screen.queryByRole('button', { name: 'Disposable conversation' })).not.toBeInTheDocument();
    expect(confirm).not.toHaveBeenCalled();
  });

  it('searches backend history and sends pin, rename and archive changes to the organization API', async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
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
    expect(await inHistory().findByRole('button', { name: 'Pin Project discussion' })).toBeVisible();
    await waitFor(() => expect(requests.some(request => request.url.includes('sort=semantic') && request.url.includes('q=project'))).toBe(true));

    fireEvent.click(inHistory().getByRole('button', { name: 'Pin Project discussion' }));
    await waitFor(() => expect(requests.some(request => request.init?.method === 'PATCH' && request.url.endsWith('/api/conversations/c-1') && JSON.parse(String(request.init.body)).pinned === true)).toBe(true));

    fireEvent.click(inHistory().getByRole('button', { name: 'Rename Project discussion' }));
    fireEvent.change(inHistory().getByRole('textbox', { name: 'Rename Project discussion' }), { target: { value: 'Renamed from UI' } });
    fireEvent.click(inHistory().getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(requests.some(request => request.init?.method === 'PATCH' && JSON.parse(String(request.init.body)).title === 'Renamed from UI')).toBe(true));

    const archiveButton = inHistory().getByRole('button', { name: 'Archive Project discussion' });
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
    fireEvent.click(await inHistory().findByRole('button', { name: 'Linked conversation' }));
    expect(location.search).toContain('conversation_id=linked-c');
    expect(await screen.findByText('Saved question')).toBeVisible();

    openConversationHistory();
    failPin = true;
    fireEvent.click(await inHistory().findByRole('button', { name: 'Pin Linked conversation' }));
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
      if (url.includes('/api/conversations')) return json({ conversations: [{ id: 'c-export', title: 'Export me' }] });
      if (url.includes('/api/messages')) return json({ messages: [] });
      if (url.includes('/api/chat/export')) return json({ export: { format: 'openagent.conversation.v1', conversation: { title: 'Export me' }, messages: [] } });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    history.replaceState({}, '', '/?view=chat&conversation_id=c-export');
    renderChat();
    fireEvent.click(await screen.findByRole('button', { name: 'More actions for Export me' }));
    fireEvent.click(screen.getByRole('menuitem', { name: 'Export JSON' }));
    await waitFor(() => expect(clicked).toContain('Export-me.openagent.json'));
    expect(requests.some(request => request.url.includes('/api/chat/export?conversation_id=c-export&write_file=false'))).toBe(true);
    expect(created).toHaveBeenCalledOnce();
  });

  it('exports from a persisted message menu without leaving Chat', async () => {
    const clicked: string[] = [];
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) { clicked.push(this.download); });
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations?')) return json({ conversations: [{ id: 'c-source', title: 'Source' }] });
      if (url.includes('/api/messages?conversation_id=c-source')) return json({ messages: [{ id: 'm-1', role: 'user', content: 'Question' }, { id: 'm-2', role: 'assistant', content: 'Answer' }] });
      if (url.includes('/api/chat/export')) return json({ export: { format: 'openagent.conversation.v1', conversation: { title: 'Source' }, messages: [] } });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    history.replaceState({}, '', '/?view=chat&conversation_id=c-source');
    renderChat();
    fireEvent.click(await screen.findByRole('button', { name: 'Actions for assistant message 2' }));
    fireEvent.click(screen.getByRole('menuitem', { name: 'Export response' }));
    await waitFor(() => expect(clicked).toContain('openagent-response.txt'));
  });

  it('validates imported OpenAgent exports before posting them to the backend', async () => {
    const requests: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push(url);
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations')) return json({ conversations: [] });
      if (url.includes('/api/chat/import')) return json({ imported: { conversation: { id: 'c-imported' } } }, 201);
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderChat();
    const input = screen.getByLabelText('Choose conversation export');
    fireEvent.change(input, { target: { files: [new File(['{}'], 'invalid.json', { type: 'application/json' })] } });
    expect(await screen.findByRole('alert')).toHaveTextContent('not a supported OpenAgent conversation export');
    expect(requests.some(url => url.includes('/api/chat/import'))).toBe(false);
    const valid = { format: 'openagent.conversation.v1', conversation: { title: 'Imported' }, messages: [{ role: 'user', content: 'Hello' }] };
    fireEvent.change(input, { target: { files: [new File([JSON.stringify(valid)], 'conversation.json', { type: 'application/json' })] } });
    await waitFor(() => expect(requests.some(url => url.includes('/api/chat/import'))).toBe(true));
    await waitFor(() => expect(location.search).toContain('conversation_id=c-imported'));
  });


  it('offers delete from an assistant message menu with inline confirmation', async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations?')) return json({ conversations: [{ id: 'c-artifact', title: 'Artifact' }] });
      if (url.includes('/api/messages')) return json({ messages: [{ id: 'm-answer', role: 'assistant', content: 'A real answer from the backend.' }] });
      if (url.endsWith('/api/conversations/c-artifact') && init?.method === 'DELETE') return json({ deleted: true });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    history.replaceState({}, '', '/?view=chat&conversation_id=c-artifact');
    renderChat();
    fireEvent.click(await screen.findByRole('button', { name: 'Actions for assistant message 1' }));
    fireEvent.click(screen.getByRole('menuitem', { name: 'Delete conversation' }));
    fireEvent.click(screen.getByRole('button', { name: 'Delete', exact: true }));
    await waitFor(() => expect(requests.some(item => item.url.endsWith('/api/conversations/c-artifact') && item.init?.method === 'DELETE')).toBe(true));
  });
});
