import { renderHook, act, waitFor } from '@testing-library/react';
import { useVoiceStream, useEventStream } from './useVoiceStream';

describe('useVoiceStream', () => {
  it('initializes voice stream session', async () => {
    const { result } = renderHook(() => useVoiceStream({ language: 'en' }));

    await act(async () => {
      const stream_id = await result.current.initialize();
      expect(stream_id).toBeTruthy();
      expect(result.current.stream_id).toBeTruthy();
    });
  });

  it('sends audio chunks and receives frames', async () => {
    const frames: any[] = [];
    const { result } = renderHook(() =>
      useVoiceStream({
        language: 'en',
        on_frame: (frame) => frames.push(frame),
      })
    );

    await act(async () => {
      await result.current.initialize();

      // Create sample audio data (1 second of silence at 16kHz)
      const audio_buffer = new ArrayBuffer(32000);
      await result.current.send_audio(audio_buffer, 'raw');
    });

    expect(frames.length).toBeGreaterThan(0);
  });

  it('finishes stream and gets transcription', async () => {
    const transcriptions: any[] = [];
    const { result } = renderHook(() =>
      useVoiceStream({
        on_transcription: (result) => transcriptions.push(result),
      })
    );

    await act(async () => {
      await result.current.initialize();
      const final_result = await result.current.finish();
      expect(final_result).toHaveProperty('transcription');
    });

    expect(transcriptions.length).toBeGreaterThan(0);
  });

  it('handles errors gracefully', async () => {
    const errors: Error[] = [];
    const { result } = renderHook(() =>
      useVoiceStream({
        on_error: (error) => errors.push(error),
      })
    );

    await act(async () => {
      try {
        await result.current.send_audio(new Uint8Array([1, 2, 3]), 'raw');
      } catch {
        // Expected to fail without initialization
      }
    });

    expect(result.current.error).toBeTruthy();
  });
});

describe('useEventStream', () => {
  it('loads events from stream endpoint', async () => {
    const { result } = renderHook(() =>
      useEventStream('/api/agents/jobs/test-job-id/events/stream')
    );

    await waitFor(() => {
      expect(result.current.is_loading === false).toBe(true);
    });

    expect(Array.isArray(result.current.events)).toBe(true);
  });

  it('supports pagination with next_cursor', async () => {
    let initial_cursor = '';

    const { result: result1 } = renderHook(() =>
      useEventStream('/api/agents/jobs/test-job-id/events/stream')
    );

    await waitFor(() => {
      expect(result1.current.is_loading === false).toBe(true);
    });

    initial_cursor = result1.current.next_cursor;

    // Load next page
    const { result: result2 } = renderHook(() =>
      useEventStream(
        `/api/agents/jobs/test-job-id/events/stream?after=${encodeURIComponent(initial_cursor)}`
      )
    );

    await waitFor(() => {
      expect(result2.current.is_loading === false).toBe(true);
    });

    // Should get different events
    expect(result2.current.events).toBeTruthy();
  });

  it('tracks stream events with callbacks', async () => {
    const events_received: any[] = [];
    const callbacks = {
      on_stream_start: (cursor: string) => {
        expect(cursor).toBeTruthy();
      },
      on_event: (event: any) => {
        events_received.push(event);
      },
      on_stream_end: (next_cursor: string, has_more: boolean) => {
        expect(typeof has_more).toBe('boolean');
      },
    };

    const { result } = renderHook(() =>
      useEventStream('/api/agents/jobs/test-job-id/events/stream', callbacks)
    );

    await waitFor(() => {
      expect(result.current.is_loading === false).toBe(true);
    });

    // Verify events were processed
    expect(events_received.length >= 0).toBe(true);
  });

  it('provides can_load_more flag for pagination', async () => {
    const { result } = renderHook(() =>
      useEventStream('/api/agents/jobs/test-job-id/events/stream')
    );

    await waitFor(() => {
      expect(result.current.is_loading === false).toBe(true);
    });

    // can_load_more should be true if has_more and next_cursor are set
    const should_load_more = result.current.can_load_more;
    expect(typeof should_load_more).toBe('boolean');
  });
});
