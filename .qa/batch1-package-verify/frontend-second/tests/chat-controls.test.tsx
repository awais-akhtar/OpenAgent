import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ChatScreen } from '../src/screens/HomeChat';

const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
function renderChat() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><ChatScreen/></QueryClientProvider>);
}
afterEach(() => { cleanup(); vi.unstubAllGlobals(); sessionStorage.clear(); history.replaceState({}, '', '/'); });

describe('chat provider, privacy and attachment controls', () => {
  it('requires explicit egress consent for network providers and submits real provider/model/attachment selections', async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.endsWith('/api/models')) return json({ providers: [
        { id: 'auto', label: 'Auto route', local: true, network: false, available: true },
        { id: 'cloud-test', label: 'Cloud test', local: false, network: true, available: true, default_model: 'model-default', installed_models: ['model-default', 'model-explicit'], privacy_warning: 'External endpoint selected.' },
      ] });
      if (url.endsWith('/api/model-policy')) return json({ routing_policy: 'local-first' });
      if (url.includes('/api/conversations')) return json({ conversations: [] });
      if (url.includes('/api/messages')) return json({ messages: [] });
      if (url.includes('/api/media/ingest')) return json({ asset: { id: 'asset-chat-1', title: 'notes.txt', media_type: 'document', mime_type: 'text/plain', size_bytes: 4 } }, 201);
      if (url.endsWith('/api/chat/stream')) {
        const frames = [
          { type: 'conversation', conversation: { id: 'chat-created' } },
          { type: 'message', message: { id: 'assistant-1', content: 'Backend stream reply.' } },
        ].map(event => `data: ${JSON.stringify(event)}\n\n`).join('');
        return new Response(new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode(frames)); controller.close(); } }), { headers: { 'Content-Type': 'text/event-stream' } });
      }
      if (url.includes('/api/messages')) return json({ messages: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderChat();
    fireEvent.click(screen.getByRole('button', { name: /Select model/ }));
    expect(await screen.findByRole('option', { name: 'Cloud test (external)' })).toBeVisible();
    fireEvent.change(screen.getByLabelText('Model provider'), { target: { value: 'cloud-test' } });
    expect(await screen.findByRole('alert')).toHaveTextContent('prompt and attachment content over the network');
    fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'model-explicit' } });

    fireEvent.change(screen.getByLabelText('Choose files to attach'), { target: { files: [new File(['memo'], 'notes.txt', { type: 'text/plain' })] } });
    expect(await screen.findByTestId('chat-attachment')).toHaveTextContent('notes.txt');
    fireEvent.change(screen.getByLabelText('Message OpenAgent'), { target: { value: 'Summarize this file' } });
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled();
    expect(requests.some(item => item.url.endsWith('/api/chat/stream'))).toBe(false);

    fireEvent.click(screen.getByLabelText('Allow external provider calls'));
    expect(screen.getByRole('button', { name: 'Send message' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }));
    expect(await screen.findByText('Backend stream reply.')).toBeVisible();

    const upload = requests.find(item => item.url.endsWith('/api/media/ingest'));
    expect(JSON.parse(String(upload?.init?.body))).toMatchObject({ filename: 'notes.txt', project: 'default', source: 'chat' });
    const stream = requests.find(item => item.url.endsWith('/api/chat/stream'));
    expect(JSON.parse(String(stream?.init?.body))).toMatchObject({
      message: 'Summarize this file', provider: 'cloud-test', model: 'model-explicit', allow_external: true,
      attachments: [{ id: 'asset-chat-1', name: 'notes.txt', media_type: 'document', mime_type: 'text/plain' }],
    });
  });

  it('keeps external egress disabled for Auto route until the user opts in', async () => {
    let streamBody: Record<string, unknown> | undefined;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/api/models')) return json({ providers: [{ id: 'auto', label: 'Auto route', local: true, network: false, available: true }] });
      if (url.endsWith('/api/model-policy')) return json({ routing_policy: 'local-first' });
      if (url.includes('/api/conversations')) return json({ conversations: [] });
      if (url.endsWith('/api/chat/stream')) {
        streamBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
        const frames = [
          { type: 'conversation', conversation: { id: 'local-chat' } },
          { type: 'message', message: { content: 'Local path response.' } },
        ].map(event => `data: ${JSON.stringify(event)}\n\n`).join('');
        return new Response(new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode(frames)); controller.close(); } }), { headers: { 'Content-Type': 'text/event-stream' } });
      }
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderChat();
    fireEvent.change(screen.getByLabelText('Message OpenAgent'), { target: { value: 'Keep this local' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }));
    expect(await screen.findByText('Local path response.')).toBeVisible();
    await waitFor(() => expect(streamBody).toMatchObject({ provider: 'auto', allow_external: false, message: 'Keep this local' }));
  });

  it('does not confuse a local Ollama server connection with external provider egress', async () => {
    let requestBody: Record<string, unknown> | undefined;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/api/models')) return json({ providers: [{ id: 'auto', label: 'Auto route', local: true, available: true }, { id: 'ollama', label: 'Ollama', local: true, network: true, available: true, default_model: 'local-model' }] });
      if (url.includes('/api/conversations')) return json({ conversations: [] });
      if (url.includes('/api/messages')) return json({ messages: [] });
      if (url.includes('/api/chat/stream')) {
        requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
        const frames = [
          { type: 'conversation', conversation: { id: 'ollama-chat' } },
          { type: 'message', message: { content: 'Local provider response.' } },
        ].map(event => `data: ${JSON.stringify(event)}\n\n`).join('');
        return new Response(new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode(frames)); controller.close(); } }), { headers: { 'Content-Type': 'text/event-stream' } });
      }
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderChat();
    fireEvent.click(screen.getByRole('button', { name: /Select model/ }));
    expect(await screen.findByRole('option', { name: 'Ollama (local server)' })).toBeVisible();
    fireEvent.change(screen.getByLabelText('Model provider'), { target: { value: 'ollama' } });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Message OpenAgent'), { target: { value: 'Use the local server' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }));
    expect(await screen.findByText('Local provider response.')).toBeVisible();
    expect(requestBody).toMatchObject({ provider: 'ollama', allow_external: false, message: 'Use the local server' });
  });
});
