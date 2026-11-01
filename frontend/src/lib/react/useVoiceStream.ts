import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api';
import { getEvents, type SseEvent } from './stream';

export interface VoiceStreamFrame {
  chunk_received: boolean;
  total_bytes: number;
  chunk_count: number;
  status: 'buffering' | 'transcribed' | 'closed';
}

export interface VoiceStreamResult {
  stream_id: string;
  session_id: string;
  transcription: string;
  is_final: boolean;
  stt_engine: string;
  confidence: string;
  transcript_status: string;
}

export interface UseVoiceStreamOptions {
  project?: string;
  owner?: string;
  language?: string;
  stt_engine?: string;
  allow_fallback_stt?: boolean;
  on_frame?: (frame: VoiceStreamFrame) => void;
  on_transcription?: (result: VoiceStreamResult) => void;
  on_error?: (error: Error) => void;
}

/**
 * React hook for real-time voice streaming with incremental transcription.
 *
 * Example:
 * ```tsx
 * const { stream, send_audio, finish } = useVoiceStream({ language: 'en' });
 * // Send audio chunks as they arrive from microphone
 * await send_audio(audioChunk);
 * // Finish streaming to get final transcription
 * const result = await finish();
 * ```
 */
export function useVoiceStream(options: UseVoiceStreamOptions = {}) {
  const [stream_id, set_stream_id] = useState<string>('');
  const [is_streaming, set_is_streaming] = useState(false);
  const [error, set_error] = useState<Error | null>(null);
  const [transcription, set_transcription] = useState('');
  const [is_final, set_is_final] = useState(false);
  const session_id_ref = useRef<string>(generateSessionId());
  const abort_controller_ref = useRef<AbortController | null>(null);

  // Initialize voice stream session
  const initialize = useCallback(async () => {
    try {
      set_error(null);
      set_is_streaming(true);

      // Create a new voice stream session via REST API
      const payload = {
        voice_session_id: session_id_ref.current,
        project: options.project || 'default',
        owner: options.owner || '',
        language: options.language || 'en',
        stt_engine: options.stt_engine || '',
        allow_fallback_stt: options.allow_fallback_stt ?? false,
      };

      const response = await api.post<any>('/api/voice/websocket/create', payload);
      const new_stream_id = response.stream_id || response.voice_session_id;
      set_stream_id(new_stream_id);

      return new_stream_id;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      set_error(error);
      options.on_error?.(error);
      set_is_streaming(false);
      throw error;
    }
  }, [options]);

  // Send audio chunk to stream
  const send_audio = useCallback(
    async (audio_data: ArrayBuffer | Uint8Array | string, encoding: 'base64' | 'raw' = 'base64') => {
      if (!stream_id) {
        throw new Error('Voice stream not initialized. Call initialize() first.');
      }

      try {
        // Encode audio data as base64 if raw
        let encoded_data = audio_data as string;
        if (encoding === 'raw' && !(typeof audio_data === 'string')) {
          const uint8_array = audio_data instanceof ArrayBuffer ? new Uint8Array(audio_data) : audio_data;
          const byte_string = Array.from(uint8_array).map((byte) => String.fromCharCode(byte)).join('');
          encoded_data = btoa(byte_string);
        } else if (typeof audio_data !== 'string') {
          const uint8_array = new Uint8Array(audio_data as ArrayBuffer);
          const byte_string = Array.from(uint8_array).map((byte) => String.fromCharCode(byte)).join('');
          encoded_data = btoa(byte_string);
        }

        const payload = {
          stream_id,
          audio_base64: encoded_data,
        };

        const response = await api.post<VoiceStreamFrame>('/api/voice/websocket/chunk', payload);
        options.on_frame?.(response);

        return response;
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        set_error(error);
        options.on_error?.(error);
        throw error;
      }
    },
    [stream_id, options]
  );

  // Finish streaming and get transcription
  const finish = useCallback(async (): Promise<VoiceStreamResult> => {
    if (!stream_id) {
      throw new Error('Voice stream not initialized.');
    }

    try {
      set_is_streaming(false);

      const payload = { stream_id };
      const response = await api.post<VoiceStreamResult>('/api/voice/websocket/finish', payload);

      if (response.transcription) {
        set_transcription(response.transcription);
        set_is_final(response.is_final);
        options.on_transcription?.(response);
      }

      return response;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      set_error(error);
      options.on_error?.(error);
      set_is_streaming(false);
      throw error;
    }
  }, [stream_id, options]);

  // Get session status
  const get_status = useCallback(async () => {
    if (!stream_id) return null;

    try {
      const response = await api.get<any>(`/api/voice/websocket/status?stream_id=${encodeURIComponent(stream_id)}`);
      return response.session || response;
    } catch (err) {
      console.error('Failed to get stream status:', err);
      return null;
    }
  }, [stream_id]);

  // Stream continuous audio with SSE for real-time transcription updates
  const stream_continuous = useCallback(
    async (on_frame?: (frame: Partial<VoiceStreamFrame>) => void) => {
      if (!stream_id) {
        throw new Error('Voice stream not initialized.');
      }

      try {
        abort_controller_ref.current = new AbortController();

        // Use SSE endpoint for real-time frames
        const events_path = `/api/voice/stream/continuous/${encodeURIComponent(stream_id)}/stream`;
        for await (const event of getEvents(events_path, abort_controller_ref.current.signal)) {
          try {
            const parsed = JSON.parse(event.data);

            if (event.event === 'voice.frame') {
              on_frame?.(parsed);
              options.on_frame?.(parsed as VoiceStreamFrame);
            } else if (event.event === 'voice.transcription') {
              set_transcription(parsed.transcription || '');
              if (parsed.is_final) {
                set_is_final(true);
              }
              options.on_transcription?.(parsed);
            } else if (event.event === 'stream.error') {
              throw new Error(parsed.error || 'Stream error');
            }
          } catch (parse_err) {
            console.warn('Failed to parse voice stream event:', event.data);
          }
        }
      } catch (err) {
        if ((err as any)?.name !== 'AbortError') {
          const error = err instanceof Error ? err : new Error(String(err));
          set_error(error);
          options.on_error?.(error);
        }
      }
    },
    [stream_id, options]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abort_controller_ref.current) {
        abort_controller_ref.current.abort();
      }
    };
  }, []);

  return {
    stream_id,
    is_streaming,
    error,
    transcription,
    is_final,
    session_id: session_id_ref.current,
    initialize,
    send_audio,
    finish,
    get_status,
    stream_continuous,
  };
}

// Utility function to generate a unique session ID
function generateSessionId(): string {
  return `voice-session-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/**
 * Hook for consuming agent/workflow event streams with pagination support.
 *
 * Example:
 * ```tsx
 * const { events, has_more, next_cursor } = useEventStream('/api/agents/jobs/{id}/events/stream');
 * // To resume from a cursor:
 * const { events } = useEventStream(`/api/agents/jobs/{id}/events/stream?after=${cursor}`);
 * ```
 */
export interface EventStreamOptions {
  on_stream_start?: (cursor: string) => void;
  on_event?: (event: any) => void;
  on_stream_end?: (next_cursor: string, has_more: boolean) => void;
  on_error?: (error: Error) => void;
}

export function useEventStream(path: string, options: EventStreamOptions = {}) {
  const [events, set_events] = useState<any[]>([]);
  const [next_cursor, set_next_cursor] = useState<string>('');
  const [has_more, set_has_more] = useState(false);
  const [is_loading, set_is_loading] = useState(false);
  const [error, set_error] = useState<Error | null>(null);
  const abort_controller_ref = useRef<AbortController | null>(null);

  useEffect(() => {
    const stream_events = async () => {
      try {
        set_is_loading(true);
        set_error(null);
        set_events([]);

        abort_controller_ref.current = new AbortController();

        // Collect events from stream
        const collected_events: any[] = [];

        for await (const event of getEvents(path, abort_controller_ref.current.signal)) {
          try {
            const parsed = JSON.parse(event.data);

            if (parsed.type === 'stream.started' || event.event === 'stream.start') {
              options.on_stream_start?.(parsed.cursor);
            } else if (
              parsed.type === 'stream.ended' ||
              event.event === 'stream.end' ||
              parsed.type === 'stream.error' ||
              event.event === 'stream.error'
            ) {
              // End of stream
              if (parsed.type === 'stream.error' || event.event === 'stream.error') {
                throw new Error(parsed.error || 'Stream error');
              }
              set_next_cursor(parsed.next_cursor || parsed.after_id || '');
              set_has_more(parsed.has_more ?? false);
              options.on_stream_end?.(parsed.next_cursor || parsed.after_id || '', parsed.has_more ?? false);
              break;
            } else {
              // Regular event
              collected_events.push(parsed);
              options.on_event?.(parsed);
            }
          } catch (parse_err) {
            console.warn('Failed to parse event stream data:', event.data);
          }
        }

        set_events(collected_events);
        set_is_loading(false);
      } catch (err) {
        if ((err as any)?.name !== 'AbortError') {
          const error = err instanceof Error ? err : new Error(String(err));
          set_error(error);
          options.on_error?.(error);
        }
        set_is_loading(false);
      }
    };

    stream_events();

    return () => {
      if (abort_controller_ref.current) {
        abort_controller_ref.current.abort();
      }
    };
  }, [path, options]);

  return {
    events,
    next_cursor,
    has_more,
    is_loading,
    error,
    can_load_more: has_more && next_cursor,
  };
}
