import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';
import AdvancedScreen from '../src/screens/AdvancedScreen';
import { useWorkspaceState } from '../src/state';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  useWorkspaceState.getState().setDataMode('live');
});

function renderAdvanced(route: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><AdvancedScreen route={route} /></QueryClientProvider>);
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

describe('Advanced Live workspace actions', () => {
  it('creates a gallery album and adds the selected existing image through the backend routes', async () => {
    const calls: Array<{ url: string; method: string; body: string }> = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input); const method = String(init?.method ?? 'GET'); const body = String(init?.body ?? '');
      calls.push({ url, method, body });
      if (url.includes('/api/gallery/library')) return json({ images: [{ id: 'asset-7', name: 'Sample image' }] });
      if (url.includes('/api/gallery/albums') && method === 'GET') return json({ albums: [{ id: 'album-2', name: 'Research' }] });
      if (url.endsWith('/api/gallery/albums') && method === 'POST') return json({ album: { id: 'album-3', name: 'Launch' } }, 201);
      if (url.includes('/api/gallery/albums/album-2/assets') && method === 'POST') return json({ added: true });
      return json({});
    });
    renderAdvanced('gallery');

    await screen.findByRole('button', { name: /Sample image/i });
    fireEvent.click(screen.getByRole('button', { name: /Sample image/i }));
    fireEvent.click(screen.getByRole('tab', { name: 'Albums' }));
    fireEvent.change(screen.getByLabelText('Album name'), { target: { value: 'Launch' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create album' }));
    await screen.findByText('Album created by the backend.');
    await waitFor(() => expect(calls.some(call => call.url.endsWith('/api/gallery/albums') && call.method === 'POST' && JSON.parse(call.body).name === 'Launch')).toBe(true));

    fireEvent.change(screen.getByLabelText('Album for selected image'), { target: { value: 'album-2' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add selected image' }));
    await waitFor(() => expect(calls.some(call => call.url.includes('/api/gallery/albums/album-2/assets') && call.method === 'POST' && JSON.parse(call.body).asset_id === 'asset-7')).toBe(true));
  });

  it('launches an Ultra Model registered manifest only after explicit local launch approval', async () => {
    const calls: Array<{ url: string; method: string; body: string }> = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input); const method = String(init?.method ?? 'GET'); const body = String(init?.body ?? '');
      calls.push({ url, method, body });
      if (url.includes('/api/ultra-model/manifests')) return json({ manifests: [{ id: 'manifest-4', name: 'Local model' }] });
      if (url.includes('/api/ultra-model/status')) return json({ status: 'ready', version: '1' });
      if (url.includes('/api/ultra-model/sessions')) return json({ sessions: [] });
      if (url.includes('/api/ultra-model/launch') && method === 'POST') return json({ status: 'queued', session: { id: 'session-5', status: 'queued' } }, 202);
      return json({});
    });
    renderAdvanced('ultramodel');

    const launch = await screen.findByRole('button', { name: 'Launch' });
    await screen.findByRole('option', { name: 'Local model' });
    fireEvent.change(screen.getByLabelText('Manifest to launch'), { target: { value: 'manifest-4' } });
    expect(launch).toBeDisabled();
    fireEvent.click(screen.getByLabelText('I approve launching this registered model on the local host.'));
    expect(launch).toBeEnabled();
    fireEvent.click(launch);
    await screen.findByText('Backend launch response: queued.');
    const call = calls.find(item => item.url.includes('/api/ultra-model/launch'));
    expect(call?.method).toBe('POST');
    expect(JSON.parse(call?.body ?? '{}')).toMatchObject({ manifest_id: 'manifest-4', approved: true, host: '127.0.0.1', allow_distributed: false });
  });

  it('certifies a Model Cookbook endpoint only after an explicit submit and displays the backend result', async () => {
    const calls: Array<{ url: string; method: string; body: string }> = [];
    const records: Array<Record<string, unknown>> = [{ id: 'cert-old', model_ref: 'old-model', endpoint: 'http://127.0.0.1:8081', status: 'certified' }];
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input); const method = String(init?.method ?? 'GET'); const body = String(init?.body ?? '');
      calls.push({ url, method, body });
      if (url.includes('/api/model-cookbook/certifications') && method === 'GET') return json({ certifications: records });
      if (url.includes('/api/model-cookbook/hosts')) return json({ hosts: [] });
      if (url.includes('/api/model-cookbook/sessions')) return json({ sessions: [] });
      if (url.includes('/api/model-cookbook/hardware-fit')) return json({});
      if (url.endsWith('/api/model-cookbook/certify') && method === 'POST') {
        const created = { id: 'cert-3', status: 'certified', endpoint: 'http://127.0.0.1:8080' }; records.unshift(created); return json({ certification: created }, 201);
      }
      if (url.endsWith('/api/model-cookbook/certifications/cert-old/revoke') && method === 'POST') { records[0].status = 'revoked'; return json({ certification: records[0] }); }
      return json({});
    });
    renderAdvanced('cookbook');
    fireEvent.click(await screen.findByRole('tab', { name: 'Certifications' }));
    fireEvent.change(await screen.findByLabelText('OpenAI-compatible endpoint'), { target: { value: 'http://127.0.0.1:8080' } });
    fireEvent.click(screen.getByRole('button', { name: 'Certify endpoint' }));
    await screen.findByText('Backend certification status: certified.');
    expect(calls.some(call => call.url.endsWith('/api/model-cookbook/certify') && call.method === 'POST' && JSON.parse(call.body).endpoint === 'http://127.0.0.1:8080' && JSON.parse(call.body).allow_external === false)).toBe(true);
    const revokeButtons = await screen.findAllByRole('button', { name: 'Revoke' });
    fireEvent.click(revokeButtons[revokeButtons.length - 1]);
    await waitFor(() => expect(calls.some(call => call.url.endsWith('/api/model-cookbook/certifications/cert-old/revoke') && call.method === 'POST')).toBe(true));
  });

  it('uses the implemented Personal OS GET export contract and surfaces backend errors', async () => {
    const calls: Array<{ url: string; method: string }> = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input); const method = String(init?.method ?? 'GET'); calls.push({ url, method });
      if (url.includes('/api/personal-os/export')) return json({ error: 'Export unavailable' }, 503);
      return json({ status: 'ready', presets: [], notes: [] });
    });
    renderAdvanced('personalos');
    fireEvent.click(await screen.findByRole('button', { name: 'Export data' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(/Export unavailable|503/i);
    expect(calls.some(call => call.url.includes('/api/personal-os/export?project=default') && call.method === 'GET')).toBe(true);
  });

  it('preserves CRM contact form values when the backend rejects the create request', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes('/api/contacts') && init?.method === 'POST') return json({ error: 'Contact service unavailable' }, 503);
      if (url.includes('/api/communication/contacts')) return json({ contacts: [] });
      return json({ contacts: [] });
    });
    renderAdvanced('communications');
    fireEvent.click(await screen.findByRole('tab', { name: 'Contacts' }));
    fireEvent.click(screen.getByText('Add CRM contact'));
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Avery Example' } });
    fireEvent.change(screen.getByLabelText('Email'), { target: { value: 'avery@example.test' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save contact' }));
    await screen.findByRole('alert');
    expect(screen.getByLabelText('Name')).toHaveValue('Avery Example');
    expect(screen.getByLabelText('Email')).toHaveValue('avery@example.test');
  });
});
