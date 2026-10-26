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
    expect(rendered.container.textContent).toMatch(/Start a recording, then review an accurate transcript/i);
    fireEvent.click(screen.getByRole('button', { name: 'New session' }));
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

  it('records only after microphone permission and uploads the stopped recording to local STT', async () => {
    const calls: Array<{ url: string; body?: Record<string, unknown> }> = [];
    const stopTrack = vi.fn();
    const getUserMedia = vi.fn(async () => ({ getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream));
    vi.stubGlobal('navigator', Object.assign(Object.create(navigator), { mediaDevices: { getUserMedia } }));
    class StubMediaRecorder {
      static isTypeSupported = vi.fn(() => true);
      state: RecordingState = 'inactive';
      mimeType: string;
      ondataavailable: ((event: BlobEvent) => void) | null = null;
      onstop: ((event: Event) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;
      constructor(_stream: MediaStream, options?: MediaRecorderOptions) { this.mimeType = options?.mimeType ?? 'audio/webm'; }
      start() { this.state = 'recording'; }
      stop() {
        this.state = 'inactive';
        this.ondataavailable?.({ data: new Blob(['captured local audio'], { type: this.mimeType }) } as BlobEvent);
        this.onstop?.(new Event('stop'));
      }
    }
    vi.stubGlobal('MediaRecorder', StubMediaRecorder);
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      calls.push({ url, body });
      if (url.endsWith('/api/voice/status')) return json({ runtime: 'local-session-runtime', microphone_control: 'browser buffered capture', privacy_boundary: 'Audio stays in the local service.' });
      if (url.endsWith('/api/voice/personas')) return json({ personas: [] });
      if (url.includes('/api/voice/sessions')) return json({ sessions: [{ id: 'voice-capture-1', mode: 'transcription', status: 'active' }] });
      if (url.endsWith('/api/voice/stt') && init?.method === 'POST') return json({ transcription: { transcript_status: 'transcribed', stt_engine: 'local-fixture', display_text: 'Buffered microphone recording transcript.' } }, 201);
      if (url.includes('/api/voice/turns?')) return json({ turns: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderVoice();
    const startButton = await screen.findByRole('button', { name: 'Start recording' });
    await vi.waitFor(() => expect(startButton).toBeEnabled());
    fireEvent.click(startButton);
    expect(await screen.findByText(/Recording from this browser microphone/)).toBeVisible();
    expect(getUserMedia).toHaveBeenCalledWith({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    expect(screen.getByRole('button', { name: 'Stop and transcribe' })).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Stop and transcribe' }));
    expect(await screen.findByText('Buffered microphone recording transcript.')).toBeVisible();
    expect(stopTrack).toHaveBeenCalledOnce();
    const sttCall = calls.find(call => call.url.endsWith('/api/voice/stt'));
    expect(sttCall?.body).toMatchObject({ project: 'default', session_id: 'voice-capture-1', language: 'en', allow_fallback_stt: false });
    expect(String(sttCall?.body?.filename)).toMatch(/^voice-recording-.*\.webm$/);
    expect(sttCall?.body?.audio_base64).toBeTruthy();
    expect(await screen.findByText('Backend transcription status: transcribed.')).toBeVisible();
  });

  it('keeps microphone denial local and does not submit audio', async () => {
    const calls: string[] = [];
    const getUserMedia = vi.fn(async () => { throw new DOMException('Permission denied', 'NotAllowedError'); });
    vi.stubGlobal('navigator', Object.assign(Object.create(navigator), { mediaDevices: { getUserMedia } }));
    vi.stubGlobal('MediaRecorder', class StubMediaRecorder {
      static isTypeSupported = vi.fn(() => true);
      constructor(_stream: MediaStream, _options?: MediaRecorderOptions) {}
      start() {}
      stop() {}
    });
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input); calls.push(url);
      if (url.endsWith('/api/voice/status')) return json({ runtime: 'local-session-runtime' });
      if (url.endsWith('/api/voice/personas')) return json({ personas: [] });
      if (url.includes('/api/voice/sessions')) return json({ sessions: [{ id: 'voice-denied-1', mode: 'transcription', status: 'active' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderVoice();
    const startButton = await screen.findByRole('button', { name: 'Start recording' });
    await vi.waitFor(() => expect(startButton).toBeEnabled());
    fireEvent.click(startButton);
    expect(await screen.findByText(/Microphone permission was denied/i)).toBeVisible();
    expect(getUserMedia).toHaveBeenCalledOnce();
    expect(calls.some(url => url.endsWith('/api/voice/stt'))).toBe(false);
  });

  it('discards a browser recording without uploading it', async () => {
    const calls: string[] = [];
    const stopTrack = vi.fn();
    const getUserMedia = vi.fn(async () => ({ getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream));
    vi.stubGlobal('navigator', Object.assign(Object.create(navigator), { mediaDevices: { getUserMedia } }));
    class StubMediaRecorder {
      static isTypeSupported = vi.fn(() => true);
      state: RecordingState = 'inactive';
      mimeType = 'audio/webm';
      ondataavailable: ((event: BlobEvent) => void) | null = null;
      onstop: ((event: Event) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;
      constructor(_stream: MediaStream, options?: MediaRecorderOptions) { this.mimeType = options?.mimeType ?? this.mimeType; }
      start() { this.state = 'recording'; }
      stop() {
        this.state = 'inactive';
        this.ondataavailable?.({ data: new Blob(['discarded local audio'], { type: this.mimeType }) } as BlobEvent);
        this.onstop?.(new Event('stop'));
      }
    }
    vi.stubGlobal('MediaRecorder', StubMediaRecorder);
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input); calls.push(url);
      if (url.endsWith('/api/voice/status')) return json({ runtime: 'local-session-runtime' });
      if (url.endsWith('/api/voice/personas')) return json({ personas: [] });
      if (url.includes('/api/voice/sessions')) return json({ sessions: [{ id: 'voice-discard-1', mode: 'transcription', status: 'active' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderVoice();
    const startButton = await screen.findByRole('button', { name: 'Start recording' });
    await vi.waitFor(() => expect(startButton).toBeEnabled());
    fireEvent.click(startButton);
    expect(await screen.findByText(/Recording from this browser microphone/)).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Discard recording' }));
    expect(await screen.findByText(/Recording discarded\. Audio was not uploaded/i)).toBeVisible();
    expect(stopTrack).toHaveBeenCalledOnce();
    expect(calls.some(url => url.endsWith('/api/voice/stt'))).toBe(false);
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

  it('generates and loads the authenticated local synthesis WAV for playback', async () => {
    const calls: Array<{ url: string; body?: unknown }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); calls.push({ url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      if (url.endsWith('/api/voice/status')) return json({ runtime: 'local-session-runtime', microphone_control: 'explicit-client-audio-input' });
      if (url.endsWith('/api/voice/personas')) return json({ personas: [{ id: 'persona-1', name: 'Local Voice' }] });
      if (url.endsWith('/api/voice/tts')) return json({ script: { project: 'default', audio_status: 'generated_wav', audio_asset_id: 'asset-tts-1', audio_mime_type: 'audio/wav', privacy_boundary: 'Generated locally.' } }, 201);
      if (url.includes('/api/voice/assets/asset-tts-1/content')) {
        const audio = new Uint8Array(64); audio.set([82, 73, 70, 70]);
        return { ok: true, status: 200, blob: async () => new Blob([audio], { type: 'audio/wav' }) } as Response;
      }
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    const rendered = renderVoice('Voice settings');
    fireEvent.change(screen.getByRole('textbox', { name: 'Text to synthesize' }), { target: { value: 'Hello from the local voice runtime.' } });
    fireEvent.click(screen.getByRole('button', { name: 'Generate local speech artifact' }));
    expect((await screen.findAllByText(/generated_wav/)).length).toBeGreaterThan(0);
    expect(await screen.findByLabelText('Generated local voice audio')).toHaveAttribute('src', expect.stringMatching(/^blob:/));
    expect(screen.getByRole('link', { name: 'Download WAV' })).toHaveAttribute('download', 'openagent-voice-asset-tts-1.wav');
    expect(calls.some(call => call.url.includes('/api/voice/assets/asset-tts-1/content?project=default'))).toBe(true);
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

