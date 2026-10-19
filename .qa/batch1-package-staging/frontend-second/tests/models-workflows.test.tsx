import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor, cleanup } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ModelsScreen } from '../src/screens/OperationsScreens';

const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
function renderModels() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><ModelsScreen /></QueryClientProvider>);
}
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('model operations workflows', () => {
  it('runs a real backend benchmark and presents returned measurements', async () => {
    const calls: Array<{ url: string; body?: Record<string, unknown> }> = [];
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined; calls.push({ url, body });
      if (url.endsWith('/api/models')) return json({ providers: [{ id: 'ollama', label: 'Ollama', local: true, available: true, default_model: 'llama3', default_model_ready: true, network: false, installed_models: ['llama3', 'qwen2'] }] });
      if (url.endsWith('/api/local-ai/models')) return json({ models: [{ id: 'model-1', name: 'Local Chat', provider: 'ollama', status: 'available', local_path: 'ollama://model-1' }] });
      if (url.endsWith('/api/local-ai/hardware')) return json({ system: 'Windows' });
      if (url.endsWith('/api/model-policy')) return json({ routing_policy: 'local-first' });
      if (url.endsWith('/api/local-ai/setup/status')) return json({ recommended: { chat: 'model-1' } });
      if (url.endsWith('/api/local-ai/marketplace')) return json({ models: [] });
      if (url.endsWith('/api/local-ai/adapters')) return json({ adapters: [] });
      if (url.includes('/api/local-ai/jobs?')) return json({ jobs: [] });
      if (url.endsWith('/api/local-ai/benchmark')) return json({ benchmark: { model_id: 'model-1', model: 'Local Chat', provider: 'ollama', duration_ms: 842, tokens_per_second: 32.5, prompt_tokens: 2, output_tokens: 8, created_at: '2026-10-14T10:00:00Z' } }, 201);
      return json({ error: `Unexpected ${url}` }, 404);
    }));
    renderModels();
    expect((await screen.findAllByText('llama3')).length).toBeGreaterThan(0);
    const providerSetup = screen.getByText('Provider setup');
    if (providerSetup.closest('details')?.open !== true) fireEvent.click(providerSetup);
    const providerAvailability = screen.getByText('Provider availability');
    if (providerAvailability.closest('details')?.open !== true) fireEvent.click(providerAvailability);
    fireEvent.click(screen.getByRole('button', { name: /Ollama · Available/ }));
    if (providerAvailability.closest('details')?.open !== true) fireEvent.click(providerAvailability);
    await waitFor(() => expect(screen.getByText('qwen2')).toBeVisible());
    fireEvent.click(screen.getByText('Model actions'));
    fireEvent.click(await screen.findByRole('button', { name: 'Benchmark' }));
    expect(await screen.findByLabelText('Model benchmark result')).toHaveTextContent('32.5 tokens/s');
    expect(calls.find(call => call.url.endsWith('/api/local-ai/benchmark'))?.body).toEqual({ model_id: 'model-1', prompt: 'Hello OpenAgent' });
  });

  it('requires explicit network consent before remote model download and sends that choice to backend', async () => {
    const calls: Array<{ url: string; body?: Record<string, unknown> }> = [];
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined; calls.push({ url, body });
      if (url.endsWith('/api/models')) return json({ providers: [] });
      if (url.endsWith('/api/local-ai/models')) return json({ models: [{ id: 'remote-1', name: 'Remote model', provider: 'local', status: 'registered', source_url: 'https://models.example/model.gguf' }] });
      if (url.endsWith('/api/local-ai/hardware')) return json({});
      if (url.endsWith('/api/model-policy')) return json({});
      if (url.endsWith('/api/local-ai/setup/status')) return json({});
      if (url.endsWith('/api/local-ai/marketplace')) return json({ models: [] });
      if (url.endsWith('/api/local-ai/adapters')) return json({ adapters: [] });
      if (url.includes('/api/local-ai/jobs?')) return json({ jobs: [] });
      if (url.endsWith('/api/local-ai/models/download')) return json({ model: { id: 'remote-1', status: 'available', local_path: 'managed/model.gguf' } }, 201);
      return json({ error: `Unexpected ${url}` }, 404);
    }));
    renderModels();
    fireEvent.click(screen.getByText('Provider setup'));
    const download = await screen.findByRole('button', { name: 'Download model' });
    expect(download).toBeDisabled();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow network for model downloads and setup' }));
    await waitFor(() => expect(download).toBeEnabled());
    fireEvent.click(download);
    await waitFor(() => expect(calls.some(call => call.url.endsWith('/api/local-ai/models/download'))).toBe(true));
    expect(calls.find(call => call.url.endsWith('/api/local-ai/models/download'))?.body).toMatchObject({ model_id: 'remote-1', allow_network: true });
  });

  it('does not expose model benchmark for a provider known to be external', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/models')) return json({ providers: [{ id: 'cloud', label: 'Cloud', local: false, available: true }] });
      if (url.endsWith('/api/local-ai/models')) return json({ models: [{ id: 'cloud-1', name: 'Cloud Model', provider: 'cloud', status: 'available', local_path: 'cloud://model' }] });
      if (url.endsWith('/api/local-ai/hardware') || url.endsWith('/api/model-policy') || url.endsWith('/api/local-ai/setup/status')) return json({});
      if (url.endsWith('/api/local-ai/marketplace') || url.endsWith('/api/local-ai/adapters') || url.includes('/api/local-ai/jobs?')) return json({ models: [], adapters: [], jobs: [] });
      return json({ error: `Unexpected ${url}` }, 404);
    }));
    renderModels();
    await screen.findByText('Cloud Model');
    fireEvent.click(screen.getByText('Model actions'));
    expect(screen.queryByRole('button', { name: 'Benchmark' })).not.toBeInTheDocument();
  });
});
