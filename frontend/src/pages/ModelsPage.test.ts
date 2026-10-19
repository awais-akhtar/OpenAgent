import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const apiMocks = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock('../lib/react/api', () => ({
  api: apiMocks,
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
  records: (value: unknown, key?: string) => {
    const rows = Array.isArray(value) ? value : key && value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : [];
    return Array.isArray(rows) ? rows.filter(row => row && typeof row === 'object') : [];
  },
}));

describe('Models page', () => {
  let host: HTMLDivElement; let root: Root;
  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    history.replaceState({}, '', '/?project=project%20one');
    localStorage.clear();
    apiMocks.get.mockReset().mockImplementation(async (path: string) => {
      if (path === '/api/models') return { providers: [
        { id: 'ollama', label: 'Ollama', local: true, available: true, network: false, default_model: 'qwen:small', default_model_ready: true, installed_models: ['qwen:small', 'llama:8b'], privacy_warning: 'Connects to the configured local Ollama server.' },
        { id: 'remote', label: 'Remote provider', local: false, available: false, network: true, default_model: 'remote-chat', default_model_ready: false, installed_models: [], privacy_warning: 'Prompts may leave this device.' },
      ] };
      if (path === '/api/local-ai/models') return { models: [{ id: 'qwen:small', name: 'Qwen Small', provider: 'ollama', status: 'ready', local_path: 'ollama://qwen:small', size_bytes: 1024 ** 3 }] };
      if (path === '/api/local-ai/hardware') return { system: 'Linux', machine: 'x86_64', cpu_count: 12, ram_gb: 32, gpu: { available: true, adapter: 'NVIDIA GPU' } };
      if (path === '/api/model-policy') return { routing_policy: 'local-first', fallback_policy: 'offline', routes: { 'local-first': ['ollama', 'local'] } };
      if (path === '/api/local-ai/adapters') return { adapters: [{ id: 'adapter-1', name: 'CUDA', model_id: 'qwen:small', status: 'ready' }] };
      if (path === '/api/local-ai/marketplace') return { models: [{ id: 'catalog-1', name: 'Qwen 3B', provider: 'ollama', recommended_engine: 'Ollama', metadata: { model: 'qwen:3b', ram_gb: 8, quantization: 'Q4' } }] };
      if (path === '/api/local-ai/setup/status') return { ollama: { available: true, server_available: true, model_ready: true, selected_model: 'qwen:small', model: 'qwen:small' }, llamacpp: { available: false, server_available: false, model_ready: false, model: 'local-7b' }, recommended: { chat: 'qwen:small', code: 'qwen-coder:7b' } };
      if (path.startsWith('/api/local-ai/jobs?')) return { jobs: [] };
      if (path.includes('/probe')) return { model_id: 'qwen:small', status: 'ready', local_path: 'ollama://qwen:small', exists: false, size_bytes: 2 * 1024 ** 3, checksum_valid: true, dependents: [{ id: 'chat' }], tools: [{ id: 'assistant' }] };
      return {};
    });
    apiMocks.post.mockReset().mockImplementation(async (path: string) => path.includes('/setup/job') ? { job: { id: 'job-1', model_name: 'qwen:3b', status: 'queued', progress: 0 } } : { discovered: [] });
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

  it('opens with a selected provider and readable availability, locality, and model state', async () => {
    const { ModelsPage } = await import('./ModelsPage');
    await act(async () => root.render(createElement(ModelsPage)));
    expect(host.querySelector('h1')?.textContent).toBe('Models');
    expect(host.querySelector('.models-provider-choice.active strong')?.textContent).toBe('Ollama');
    expect(host.textContent).toContain('Local');
    expect(host.textContent).toContain('Available');
    expect(host.textContent).toContain('qwen:small');
    const remote = [...host.querySelectorAll<HTMLButtonElement>('.models-provider-choice')].find(button => button.textContent?.includes('Remote provider'));
    await act(async () => remote?.click());
    expect(host.textContent).toContain('Prompts may leave this device.');
    expect(host.textContent).toContain('Not installed');
    expect(host.textContent).toContain('NVIDIA GPU');
    expect(host.textContent).not.toMatch(/\btrue\b|\bfalse\b/);
    expect(host.textContent).not.toContain('"ollama"');
    expect(apiMocks.get).toHaveBeenCalledWith('/api/local-ai/jobs?project=project%20one&limit=12');
    expect(host.querySelector('.models-routing')?.hasAttribute('open')).toBe(false);
  });

  it('renders the provider setup capability and routing configuration as readable summaries', async () => {
    const { ModelsPage } = await import('./ModelsPage');
    await act(async () => root.render(createElement(ModelsPage)));
    await act(async () => host.querySelector<HTMLElement>('.models-secondary:last-of-type > summary')?.click());
    expect(host.textContent).toContain('Local setup tools');
    expect(host.textContent).toContain('Local server responding');
    expect(host.textContent).toContain('Runtime not found on this device');
    expect(host.textContent).toContain('Recommended defaults');
    expect(host.textContent).not.toContain('{\n');
  });

  it('shows an honest unavailable state when the local model inventory request fails', async () => {
    apiMocks.get.mockImplementation(async (path: string) => {
      if (path === '/api/local-ai/models') throw new Error('service unavailable');
      if (path === '/api/models') return { providers: [] };
      if (path.startsWith('/api/local-ai/jobs?')) return { jobs: [] };
      return {};
    });
    const { ModelsPage } = await import('./ModelsPage');
    await act(async () => root.render(createElement(ModelsPage)));
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('local-models');
    expect(host.textContent).toContain('Local model inventory could not be read.');
    expect(host.textContent).not.toContain('No local models are registered.');
  });

  it('does not infer local runtime installation or model readiness from an empty status response', async () => {
    apiMocks.get.mockImplementation(async (path: string) => {
      if (path === '/api/local-ai/setup/status') return {};
      if (path === '/api/models') return { providers: [] };
      if (path.startsWith('/api/local-ai/jobs?')) return { jobs: [] };
      return {};
    });
    const { ModelsPage } = await import('./ModelsPage');
    await act(async () => root.render(createElement(ModelsPage)));
    await act(async () => host.querySelector<HTMLElement>('.models-secondary:last-of-type > summary')?.click());
    expect(host.textContent).toContain('Runtime details not reported');
    expect(host.textContent).toContain('Model readiness not reported');
    expect(host.textContent).not.toContain('Runtime not found on this device');
  });

  it('probes a model and shows concise integrity and dependency results', async () => {
    const { ModelsPage } = await import('./ModelsPage');
    await act(async () => root.render(createElement(ModelsPage)));
    const probe = [...host.querySelectorAll<HTMLButtonElement>('button')].find(button => button.textContent === 'Probe');
    await act(async () => probe?.click());
    expect(apiMocks.get).toHaveBeenCalledWith('/api/local-ai/models/qwen%3Asmall/probe');
    expect(host.textContent).toContain('Managed by local provider');
    expect(host.textContent).toContain('Verified');
    expect(host.textContent).toContain('Assigned tools');
    expect(host.textContent).not.toContain('"checksum_valid"');
  });

  it('keeps network downloads opt-in and retains the explicit setup confirmation and project scope', async () => {
    const confirm = vi.fn().mockReturnValue(true);
    vi.stubGlobal('confirm', confirm);
    const { ModelsPage } = await import('./ModelsPage');
    await act(async () => root.render(createElement(ModelsPage)));
    await act(async () => host.querySelector<HTMLElement>('.models-secondary:last-of-type > summary')?.click());
    const network = host.querySelector<HTMLInputElement>('.models-network-toggle input');
    expect(network?.checked).toBe(false);
    const queue = [...host.querySelectorAll<HTMLButtonElement>('button')].find(button => button.textContent === 'Queue setup');
    await act(async () => queue?.click());
    expect(confirm).toHaveBeenCalled();
    expect(apiMocks.post).toHaveBeenCalledWith('/api/local-ai/models/setup/job', { task: 'chat', model: 'qwen:3b', allow_network: false, project: 'project one' });
    expect(host.textContent).toContain('Setup request accepted for qwen:3b.');
  });
});
