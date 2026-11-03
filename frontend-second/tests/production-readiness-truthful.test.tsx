import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ProductionReadinessPanel from '../src/screens/ProductionReadinessPanel';
import { useWorkspaceState } from '../src/state';

/**
 * The Production Readiness screen is a Live destination whose own description
 * promises "real-time status of all production-ready features". Three of its
 * four tabs used to invent everything they showed: a recorder that touched no
 * microphone, a hardcoded "Sample transcription", a random confidence and
 * random waveform bars, a fixed 24ms latency, three permanent green ticks for
 * Faster Whisper / a live socket / audio buffering, two made-up voice sessions,
 * three made-up reminders with "12 delivered today", three made-up training
 * jobs with progress bars and ETAs, a "2.4GB" checkpoint, and tab badges that
 * read 2/3/3/2 whatever the services held. The Ultra Model tab read its two
 * endpoints through a bare fetch() with no credential and, on failure,
 * substituted two invented sessions.
 *
 * These tests hold the screen to what the backend actually reports.
 */

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  useWorkspaceState.getState().setDataMode('live');
});

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

/** Records every request the panel makes and answers the ones it names. */
function mockBackend(handler: (url: string) => unknown) {
  const calls: Array<{ url: string; headers: Headers }> = [];
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
    const url = String(input);
    calls.push({ url, headers: new Headers(init?.headers) });
    return json(handler(url) ?? {});
  });
  return calls;
}

describe('Production Readiness reports only what the backend returns', () => {
  it('shows the voice counts and installed engines the backend reports', async () => {
    mockBackend(url => url.includes('/api/voice/status')
      ? {
          sessions: 4,
          turns: 9,
          translations: 1,
          stt: { engines: [
            { id: 'faster-whisper', name: 'Faster Whisper', available: true },
            { id: 'vosk', name: 'Vosk', available: false, reason: 'Install the vosk package' },
          ] },
        }
      : {});
    render(<ProductionReadinessPanel tab="Voice STT" />);

    await screen.findByText('Faster Whisper');
    expect(screen.getByText('4')).toBeTruthy();
    expect(screen.getByText('9')).toBeTruthy();
    // An engine that is not installed says why rather than showing a tick.
    expect(screen.getByText('Install the vosk package')).toBeTruthy();
    expect(screen.getByText(/1 engine ready/)).toBeTruthy();
  });

  it('claims no live stream, confidence or latency on the voice tab', async () => {
    mockBackend(() => ({ sessions: 0, turns: 0, translations: 0, stt: { engines: [] } }));
    render(<ProductionReadinessPanel tab="Voice STT" />);

    await waitFor(() => expect(screen.getByText(/no local speech-to-text engines/i)).toBeTruthy());
    expect(screen.queryByText(/Sample transcription/i)).toBeNull();
    expect(screen.queryByText(/24ms/)).toBeNull();
    // The old panel showed a running confidence percentage from Math.random().
    expect(screen.queryByText(/^\d+\.\d%$/)).toBeNull();
    expect(screen.queryByText(/WebSocket streaming with real STT/i)).toBeNull();
    expect(screen.queryByRole('button', { name: /Start Recording/i })).toBeNull();
  });

  it('lists the reminders the calendar system reports and counts them from the records', async () => {
    mockBackend(url => url.includes('/api/calendar/reminders')
      ? { reminders: [
          { id: 'r-1', title: 'Review the release', trigger_at: '2026-11-03T09:00:00Z', delivered: true },
          { id: 'r-2', title: 'Sign the renewal', trigger_at: '2026-11-03T14:00:00Z', delivered: false },
        ] }
      : {});
    render(<ProductionReadinessPanel tab="Auto Scheduling" />);

    await screen.findByText('Review the release');
    expect(screen.getByText('Sign the renewal')).toBeTruthy();
    expect(screen.getByText('Delivered')).toBeTruthy();
    expect(screen.getByText('Not delivered')).toBeTruthy();
    expect(screen.getByText('2 reminders')).toBeTruthy();
    // The old panel showed these regardless of any record.
    expect(screen.queryByText('Team Meeting')).toBeNull();
    expect(screen.queryByText('12')).toBeNull();
  });

  it('lists the fine-tuning jobs the service reports and no invented ones', async () => {
    mockBackend(url => url.includes('/api/finetune/jobs')
      ? { jobs: [{ id: 'job-a', name: 'Local adapter', status: 'running', progress: 40 }] }
      : {});
    render(<ProductionReadinessPanel tab="Fine-tuning" />);

    await screen.findByText('Local adapter');
    expect(screen.getByText('1 job')).toBeTruthy();
    expect(screen.getByText('40%')).toBeTruthy();
    expect(screen.queryByText('GPT-2 Fine-tune')).toBeNull();
    expect(screen.queryByText('2.4GB')).toBeNull();
  });

  it('says a job reports no progress rather than drawing a bar at zero', async () => {
    mockBackend(url => url.includes('/api/finetune/jobs')
      ? { jobs: [{ id: 'job-b', name: 'Queued adapter', status: 'queued' }] }
      : {});
    render(<ProductionReadinessPanel tab="Fine-tuning" />);

    await screen.findByText('Queued adapter');
    expect(screen.getByText('Progress not reported')).toBeTruthy();
  });

  it('reads the ultra model endpoints through the authenticated client', async () => {
    const calls = mockBackend(url => url.includes('/api/ultra-model/sessions')
      ? { sessions: [{ id: 'session-x', model: 'local-small', status: 'planned' }] }
      : { available: false });
    render(<ProductionReadinessPanel tab="Ultra Model" />);

    await waitFor(() => expect(calls.some(call => call.url.includes('/api/ultra-model/status'))).toBe(true));
    await waitFor(() => expect(calls.some(call => call.url.includes('/api/ultra-model/sessions'))).toBe(true));
    // These two reads went through a bare fetch() before, so they carried no
    // headers from the shared client and no credential. Every request the panel
    // makes must now come from that client.
    const ultraCalls = calls.filter(call => call.url.includes('/api/ultra-model/'));
    expect(ultraCalls.length).toBeGreaterThan(0);
    for (const call of ultraCalls) {
      expect(call.headers.get('X-Response-Envelope')).toBe('legacy');
      expect(call.headers.get('Accept')).toBe('application/json');
    }
  });

  it('shows nothing in place of ultra model data it could not read', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => json({ error: 'unauthorized' }, 401));
    render(<ProductionReadinessPanel tab="Ultra Model" />);

    await waitFor(() => expect(screen.queryByText(/Llama-7B/)).toBeNull());
    expect(screen.queryByText(/Llama-13B/)).toBeNull();
    expect(screen.queryByText(/4h 32m/)).toBeNull();
  });

  it('does not report a model format or serve type the backend did not send', async () => {
    mockBackend(() => ({ available: false }));
    render(<ProductionReadinessPanel tab="Ultra Model" />);

    await waitFor(() => expect(screen.queryAllByText('Not reported').length).toBeGreaterThan(0));
    expect(screen.queryByText('GGUF')).toBeNull();
    expect(screen.queryByText('Q4_K_M')).toBeNull();
  });
});
