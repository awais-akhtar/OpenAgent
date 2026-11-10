import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import AdvancedScreen from '../src/screens/AdvancedScreen';
import { useWorkspaceState } from '../src/state';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  useWorkspaceState.getState().setDataMode('live');
});

it('carries genuine created G25 IDs into the next graph and reasoning actions', async () => {
  useWorkspaceState.getState().setDataMode('live');
  const requests: Array<{ url: string; method: string }> = [];
  let nodesCreated = 0;
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    requests.push({ url, method });
    const body = url.endsWith('/api/g25/graph/nodes')
      ? { id: ++nodesCreated === 1 ? 'node-from-python' : 'target-from-python', label: 'Source' }
      : url.endsWith('/api/g25/reasoning/chains')
        ? { chain_id: 'chain-from-python', topic: 'Local reasoning' }
        : {};
    return new Response(JSON.stringify(body), { status: method === 'POST' ? 201 : 200, headers: { 'Content-Type': 'application/json' } });
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><AdvancedScreen route="g25advanced" /></QueryClientProvider>);

  fireEvent.click(screen.getByRole('tab', { name: 'Knowledge graph' }));
  fireEvent.change(screen.getByRole('textbox', { name: 'Node label' }), { target: { value: 'Source' } });
  fireEvent.click(screen.getByRole('button', { name: 'Create node' }));
  await waitFor(() => expect(screen.getByRole('textbox', { name: 'Source node id' })).toHaveValue('node-from-python'));
  fireEvent.change(screen.getByRole('textbox', { name: 'Node label' }), { target: { value: 'Target' } });
  fireEvent.click(screen.getByRole('button', { name: 'Create node' }));
  await waitFor(() => expect(screen.getByRole('textbox', { name: 'Target node id' })).toHaveValue('target-from-python'));
  fireEvent.change(screen.getByRole('textbox', { name: 'Relationship' }), { target: { value: 'references' } });
  fireEvent.click(screen.getByRole('button', { name: 'Create edge' }));
  await waitFor(() => expect(requests).toContainEqual({ url: '/api/g25/graph/edges', method: 'POST' }));

  fireEvent.click(screen.getByRole('tab', { name: 'Reasoning' }));
  fireEvent.change(screen.getByRole('textbox', { name: 'Chain topic' }), { target: { value: 'Local reasoning' } });
  fireEvent.click(screen.getByRole('button', { name: 'Create chain' }));
  await waitFor(() => expect(screen.getByRole('textbox', { name: 'Chain id' })).toHaveValue('chain-from-python'));
  expect(requests).toEqual(expect.arrayContaining([
    { url: '/api/g25/graph/nodes', method: 'POST' },
    { url: '/api/g25/reasoning/chains', method: 'POST' },
  ]));
});

it('uses the uploaded document ID for the next RAG index action', async () => {
  useWorkspaceState.getState().setDataMode('live');
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const body = url.endsWith('/api/g25/documents/upload') ? { document_id: 'doc-from-python', filename: 'notes.txt' } : {};
    return new Response(JSON.stringify(body), { status: init?.method === 'POST' ? 201 : 200, headers: { 'Content-Type': 'application/json' } });
  });
  vi.stubGlobal('fetch', fetchMock);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><AdvancedScreen route="g25advanced" /></QueryClientProvider>);

  fireEvent.change(screen.getByLabelText('File'), { target: { files: [new File(['local test text'], 'notes.txt', { type: 'text/plain' })] } });
  await waitFor(() => expect(screen.getByRole('button', { name: 'Upload document' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Upload document' }));
  await waitFor(() => expect(fetchMock.mock.calls.some(([url, init]) => String(url).endsWith('/api/g25/documents/upload') && init?.method === 'POST')).toBe(true));
  fireEvent.click(screen.getByRole('tab', { name: 'RAG' }));
  expect(screen.getByRole('tab', { name: 'RAG' })).toHaveAttribute('aria-selected', 'true');
  await waitFor(() => expect(screen.getByLabelText('Document id')).toHaveValue('doc-from-python'));
});
