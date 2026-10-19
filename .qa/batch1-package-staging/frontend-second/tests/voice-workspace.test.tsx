import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import AdvancedScreen from '../src/screens/AdvancedScreen';

function renderVoice(tab?: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const rendered = render(<QueryClientProvider client={client}><AdvancedScreen route="voice" /></QueryClientProvider>);
  if (tab && tab !== 'Recordings') fireEvent.click(screen.getByRole('tab', { name: tab }));
  return rendered;
}

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); sessionStorage.clear(); window.history.replaceState({}, '', '/'); });

describe('Voice workspace backend integration', () => {
  it('creates a transcription session and submits selected audio through the upload API', async () => {
    const calls: Array<{ url: string; method: string; body?: BodyInit | null }> = [];
    let sessions: Array<Record<string, unknown>> = [{ id: 'voice-existing', mode: 'transcription', language: 'en', status: 'active', created_at: '2026-10-14T09:00:00Z' }];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = init?.method ?? 'GET'; calls.push({ url, method, body: init?.body });
      if (url.endsWith('/api/voice/status')) return json({ runtime: 'local-session-runtime', microphone_control: 'explicit-client-audio-input', sessions: sessions.length, turns: 0, privacy_boundary: 'Audio and transcripts stay local.' });
      if (url.endsWith('/api/voice/personas')) return json({ personas: [] });
      if (url.endsWith('/api/voice/sessions') && method === 'POST') { const body = JSON.parse(String(init?.body)); const session = { id: 'voice-1', mode: body.mode, language: body.language, status: 'active', created_at: '2026-10-14T10:00:00Z' }; sessions = [session, ...sessions]; return json({ session }, 201); }
      if (url.endsWith('/api/voice/sessions')) return json({ sessions });
      if (url.endsWith('/api/voice/stt') && method === 'POST') return json({ transcription: { transcript_status: 'transcribed', stt_engine: 'vosk', confidence: 0.9, display_text: 'A real backend transcript.', audio_profile: { format: 'wav' } } }, 201);
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));

    const rendered = renderVoice();
    expect(rendered.container.textContent).toMatch(/no live microphone streaming endpoint/i);
    fireEvent.click(screen.getByRole('button', { name: 'Create session' }));
    expect(await screen.findByText(/Session recorded by the backend/)).toBeInTheDocument();
    expect(calls.find(call => call.url.endsWith('/api/voice/sessions') && call.method === 'POST')).toBeDefined();
    expect(JSON.parse(String(calls.find(call => call.url.endsWith('/api/voice/sessions') && call.method === 'POST')?.body))).toMatchObject({ project: 'default', mode: 'transcription', always_listening: false });

    fireEvent.click(screen.getByRole('tab', { name: 'Transcripts' }));
    expect(await screen.findByRole('option', { name: /voice-1/ })).toBeInTheDocument();
    expect(await screen.findByRole('option', { name: /voice-existi/ })).toBeInTheDocument();

    const audio = new File(['local audio bytes'], 'sample.wav', { type: 'audio/wav' });
    fireEvent.change(screen.getByLabelText('Choose audio file'), { target: { files: [audio] } });
    fireEvent.click(screen.getByRole('button', { name: 'Upload and transcribe' }));
    expect(await screen.findByText('A real backend transcript.')).toBeVisible();
    const upload = calls.find(call => call.url.endsWith('/api/voice/stt') && call.method === 'POST');
    const uploadBody = JSON.parse(String(upload?.body));
    expect(uploadBody).toMatchObject({ filename: 'sample.wav', project: 'default', allow_fallback_stt: false });
    expect(['voice-1', 'voice-existing']).toContain(uploadBody.session_id);
    expect(uploadBody.audio_base64).toBeTruthy();
  });

  it('loads transcript turns for a selected backend session and exposes STT provenance', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/voice/status')) return json({ runtime: 'local-session-runtime', microphone_control: 'explicit-client-audio-input' });
      if (url.endsWith('/api/voice/personas')) return json({ personas: [] });
      if (url.endsWith('/api/voice/sessions')) return json({ sessions: [{ id: 'voice-2', mode: 'transcription', language: 'en', status: 'active' }] });
      if (url.includes('/api/voice/turns?session_id=voice-2')) return json({ turns: [{ id: 'turn-1', role: 'user', speaker: 'user', text: 'Meeting starts at noon.', is_final: true, audio_asset_id: 'asset-1', created_at: '2026-10-14T10:05:00Z', metadata: { stt: { status: 'transcribed', provider: 'vosk' } } }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderVoice('Transcripts');
    expect(await screen.findByText('Meeting starts at noon.')).toBeVisible();
    expect(screen.getByText(/STT transcribed \(vosk\)/)).toBeVisible();
    expect(screen.getByText(/audio asset asset-1/)).toBeVisible();
  });

  it('generates a local synthesis artifact and states that playback is unavailable', async () => {
    const calls: Array<{ url: string; body?: unknown }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); calls.push({ url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      if (url.endsWith('/api/voice/status')) return json({ runtime: 'local-session-runtime', microphone_control: 'explicit-client-audio-input' });
      if (url.endsWith('/api/voice/personas')) return json({ personas: [{ id: 'persona-1', name: 'Local Voice' }] });
      if (url.endsWith('/api/voice/tts')) return json({ script: { audio_status: 'generated_wav', audio_asset_id: 'asset-tts-1', audio_mime_type: 'audio/wav', privacy_boundary: 'Generated locally.' } }, 201);
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    const rendered = renderVoice('Voice settings');
    fireEvent.change(screen.getByRole('textbox', { name: 'Text to synthesize' }), { target: { value: 'Hello from the local voice runtime.' } });
    fireEvent.click(screen.getByRole('button', { name: 'Generate local speech artifact' }));
    expect((await screen.findAllByText(/generated_wav/)).length).toBeGreaterThan(0);
    expect(rendered.container.textContent).toMatch(/cannot play it/i);
    expect(calls.find(call => call.url.endsWith('/api/voice/tts'))?.body).toMatchObject({ text: 'Hello from the local voice runtime.', project: 'default', voice: 'local-script' });
  });

  it('surfaces transcription permission or backend errors without reporting a transcript', async () => {
    const calls: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      calls.push(`${init?.method ?? 'GET'} ${url}`);
      if (url.endsWith('/api/voice/status')) return json({ runtime: 'local-session-runtime', microphone_control: 'explicit-client-audio-input' });
      if (url.endsWith('/api/voice/personas')) return json({ personas: [] });
      if (url.endsWith('/api/voice/sessions')) return json({ sessions: [{ id: 'voice-3', mode: 'transcription', language: 'en', status: 'active' }] });
      if (url.includes('/api/voice/turns?session_id=voice-3')) return json({ turns: [] });
      if (url.endsWith('/api/voice/stt') && init?.method === 'POST') return json({ error: 'No local speech-to-text engine is available.' }, 400);
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderVoice('Transcripts');
    expect(await screen.findByRole('option', { name: /voice-3/ })).toBeInTheDocument();
    const audio = new File(['audio'], 'sample.wav', { type: 'audio/wav' });
    fireEvent.change(await screen.findByLabelText('Choose audio file'), { target: { files: [audio] } });
    fireEvent.click(screen.getByRole('button', { name: 'Upload and transcribe' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(/No local speech-to-text engine is available/i);
    expect(calls.some(call => call.startsWith('POST ') && call.includes('/api/voice/stt'))).toBe(true);
    expect(screen.queryByText('Backend transcription result')).not.toBeInTheDocument();
  });
});

