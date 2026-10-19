import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { FilesScreen } from '../src/screens/WorkspaceScreens';

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

function renderFiles() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><FilesScreen/></QueryClientProvider>);
}
function json(payload: unknown, status = 200) { return new Response(JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json' } }); }

describe('Files workspace links and notes', () => {
  it('creates, lists and updates a workspace link without treating it as a document', async () => {
    let saved: Record<string, unknown> = { id: 'w-link', kind: 'note', title: 'API guide', content: 'Useful docs', metadata: { workspace_type: 'link', url: 'https://docs.example.test' }, updated_at: 'now' };
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/documents/library')) return json({ items: [] });
      if (url.includes('/api/workspace/items?')) return json({ items: [saved] });
      if (url.endsWith('/api/workspace/items') && init?.method === 'POST') {
        const body = JSON.parse(String(init.body));
        saved = { id: 'w-link', kind: body.kind, ...body, updated_at: 'now' };
        return json({ item: saved }, 201);
      }
      if (url.endsWith('/api/workspace/items/update') && init?.method === 'POST') {
        const body = JSON.parse(String(init.body));
        saved = { ...saved, ...body, metadata: body.metadata, updated_at: 'later' };
        return json({ item: saved });
      }
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderFiles();
    fireEvent.click(await screen.findByRole('tab', { name: 'Links' }));
    expect(await screen.findByText('API guide')).toBeVisible();
    expect(screen.getByText(/Workspace items are stored separately/)).toBeVisible();
    fireEvent.change(screen.getByRole('textbox', { name: 'Link URL' }), { target: { value: 'https://updated.example.test/guide' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save link' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/workspace/items/update'), expect.objectContaining({ method: 'POST' })));
    const update = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/api/workspace/items/update'));
    expect(JSON.parse(String(update?.[1]?.body))).toMatchObject({ item_id: 'w-link', metadata: { workspace_type: 'link', url: 'https://updated.example.test/guide' } });

    fireEvent.click(screen.getByRole('button', { name: 'Add link' }));
    const createDialog = within(screen.getByRole('dialog'));
    fireEvent.change(createDialog.getByLabelText('Link title'), { target: { value: 'Release notes' } });
    fireEvent.change(createDialog.getByLabelText('New link URL'), { target: { value: 'https://example.test/releases' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create link' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringMatching(/\/api\/workspace\/items$/), expect.objectContaining({ method: 'POST' })));
    const create = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/workspace/items') && init?.method === 'POST');
    expect(JSON.parse(String(create?.[1]?.body))).toMatchObject({ kind: 'note', title: 'Release notes', metadata: { workspace_type: 'link', url: 'https://example.test/releases' } });
    expect(fetchMock.mock.calls.some(([url]) => String(url).startsWith('/api/document/'))).toBe(false);
  });

  it('creates a workspace note using actual workspace item endpoints and surfaces API failures', async () => {
    let createdNote: Record<string, unknown> | null = null;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/documents/library')) return json({ items: [] });
      if (url.includes('/api/workspace/items?')) return json({ items: createdNote ? [createdNote] : [] });
      if (url.endsWith('/api/workspace/items') && init?.method === 'POST') { createdNote = { id: 'w-note', kind: 'note', title: 'Meeting notes', content: 'Decisions', metadata: { workspace_type: 'note' } }; return json({ item: createdNote }, 201); }
      if (url.endsWith('/api/workspace/items/update')) return json({ error: 'Workspace store unavailable' }, 503);
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderFiles();
    fireEvent.click(await screen.findByRole('tab', { name: 'Notes' }));
    expect(await screen.findByText('No notes yet')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Add note' }));
    fireEvent.change(screen.getByLabelText('Note title'), { target: { value: 'Meeting notes' } });
    fireEvent.change(screen.getByLabelText('Note'), { target: { value: 'Decisions' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create note' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringMatching(/\/api\/workspace\/items$/), expect.objectContaining({ method: 'POST' })));
    const create = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/workspace/items') && init?.method === 'POST');
    expect(JSON.parse(String(create?.[1]?.body))).toMatchObject({ kind: 'note', title: 'Meeting notes', content: 'Decisions', metadata: { workspace_type: 'note' } });
    expect(await screen.findByText('Meeting notes')).toBeVisible();
    fireEvent.change(screen.getByRole('textbox', { name: 'Note content' }), { target: { value: 'Keep this draft after a service error.' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save note' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Workspace store unavailable');
    expect(screen.getByRole('textbox', { name: 'Note content' })).toHaveValue('Keep this draft after a service error.');
  });

  it('rejects unsupported link schemes before sending a workspace mutation', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => String(input).includes('/api/workspace/items?') ? json({ items: [] }) : json({ items: [] }));
    vi.stubGlobal('fetch', fetchMock);
    renderFiles();
    fireEvent.click(await screen.findByRole('tab', { name: 'Links' }));
    fireEvent.click(screen.getByRole('button', { name: 'Add link' }));
    const createDialog = within(screen.getByRole('dialog'));
    fireEvent.change(createDialog.getByLabelText('Link title'), { target: { value: 'Unsafe' } });
    fireEvent.change(createDialog.getByLabelText('New link URL'), { target: { value: 'ftp://unsafe.example.test' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create link' }));
    await screen.findByRole('alert');
    expect(fetchMock).not.toHaveBeenCalledWith(expect.stringMatching(/\/api\/workspace\/items$/), expect.objectContaining({ method: 'POST' }));
  });
});
