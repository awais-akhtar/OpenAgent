# Streaming Quickstart Guide

Quick examples to get started with the new voice and event streaming features.

## Voice Streaming - Basic Example

### React Component

```tsx
import { useVoiceStream } from '@/lib/react/useVoiceStream';
import { useRef } from 'react';

export function VoiceRecorder() {
  const { initialize, send_audio, finish, transcription, is_streaming } = useVoiceStream({
    language: 'en',
    project: 'my-project',
    on_transcription: (result) => {
      console.log('Got transcription:', result.transcription);
    },
  });

  const media_recorder_ref = useRef<MediaRecorder | null>(null);

  const start_recording = async () => {
    // Initialize session
    await initialize();

    // Get microphone access
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

    recorder.ondataavailable = async (event) => {
      // Convert blob to ArrayBuffer
      const buffer = await event.data.arrayBuffer();
      // Send audio chunk
      await send_audio(buffer, 'raw');
    };

    // Send chunks every 100ms
    recorder.start(100);
    media_recorder_ref.current = recorder;
  };

  const stop_recording = async () => {
    if (media_recorder_ref.current) {
      media_recorder_ref.current.stop();
      // Get final transcription
      const result = await finish();
      console.log('Transcription:', result.transcription);
    }
  };

  return (
    <div>
      <button onClick={start_recording} disabled={is_streaming}>
        Start Recording
      </button>
      <button onClick={stop_recording} disabled={!is_streaming}>
        Stop & Transcribe
      </button>
      {transcription && <p>Result: {transcription}</p>}
    </div>
  );
}
```

## Voice Streaming - Advanced with Real-time Updates

```tsx
import { useVoiceStream } from '@/lib/react/useVoiceStream';
import { useState } from 'react';

export function AdvancedVoiceRecorder() {
  const [frames, setFrames] = useState<any[]>([]);
  const [current_transcription, setCurrentTranscription] = useState('');

  const { initialize, send_audio, finish, stream_continuous, error } = useVoiceStream({
    language: 'en',
    project: 'my-project',
    on_frame: (frame) => {
      setFrames((prev) => [...prev.slice(-9), frame]); // Keep last 10 frames
    },
    on_transcription: (result) => {
      if (result.is_final) {
        setCurrentTranscription(result.transcription);
      }
    },
    on_error: (err) => {
      console.error('Voice stream error:', err.message);
    },
  });

  const start = async () => {
    await initialize();
    // Get audio stream and send chunks...
  };

  const stream_live = async () => {
    // Open real-time transcription stream
    await stream_continuous((frame) => {
      console.log('Real-time frame:', frame);
    });
  };

  return (
    <div>
      <button onClick={start}>Start</button>
      <button onClick={stream_live}>Enable Live Stream</button>
      <div>
        <p>Frames: {frames.length}</p>
        <p>Transcription: {current_transcription}</p>
        {error && <p style={{ color: 'red' }}>Error: {error.message}</p>}
      </div>
    </div>
  );
}
```

## Agent Job Monitoring

### React Component

```tsx
import { useEventStream } from '@/lib/react/useVoiceStream';
import { useState } from 'react';

export function JobMonitor({ jobId }: { jobId: string }) {
  const [all_events, setAllEvents] = useState<any[]>([]);
  const [cursor, setCursor] = useState<string>('');

  const { events, next_cursor, has_more, is_loading } = useEventStream(
    `/api/agents/jobs/${jobId}/events/stream`,
    {
      on_stream_start: (cursor) => {
        console.log('Stream started at cursor:', cursor);
      },
      on_event: (event) => {
        setAllEvents((prev) => [...prev, event]);
      },
      on_stream_end: (next, has_more_data) => {
        setCursor(next);
        if (has_more_data) {
          console.log('More events available. Use next cursor:', next);
        }
      },
    }
  );

  const load_next_page = () => {
    if (has_more && cursor) {
      window.location.search = `?job=${jobId}&after=${encodeURIComponent(cursor)}`;
    }
  };

  return (
    <div>
      <h2>Job {jobId} Events</h2>
      {is_loading && <p>Loading...</p>}
      <ul>
        {all_events.map((evt, i) => (
          <li key={i}>
            <strong>{evt.event_type}</strong>: {JSON.stringify(evt.payload || evt.data)}
          </li>
        ))}
      </ul>
      {has_more && <button onClick={load_next_page}>Load More Events</button>}
    </div>
  );
}
```

## Workflow Run Monitoring

### React Component

```tsx
import { useEventStream } from '@/lib/react/useVoiceStream';
import { useEffect, useState } from 'react';

export function WorkflowMonitor({ runId, project }: { runId: string; project?: string }) {
  const [status, setStatus] = useState('pending');
  const [result, setResult] = useState<any>(null);

  const { events, has_more, next_cursor } = useEventStream(
    `/api/workflows/runs/${runId}/events/stream?project=${project || 'default'}`,
    {
      on_event: (event) => {
        if (event.event_type === 'run.status') {
          setStatus(event.status);
        } else if (event.event_type === 'run.completed') {
          setResult(event.data.result);
        }
      },
    }
  );

  // Auto-fetch next page if available
  useEffect(() => {
    if (has_more && next_cursor && status === 'completed') {
      // Could implement auto-pagination here
      console.log('More events available:', next_cursor);
    }
  }, [has_more, next_cursor, status]);

  return (
    <div>
      <h2>Workflow Run {runId}</h2>
      <p>Status: {status}</p>
      {result && (
        <div>
          <h3>Result</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
      <div>
        <h3>Events ({events.length})</h3>
        {events.map((evt) => (
          <div key={evt.id} style={{ padding: '10px', border: '1px solid #ccc', marginBottom: '5px' }}>
            <strong>{evt.event_type}</strong> @ {evt.timestamp}
            <pre>{JSON.stringify(evt.data, null, 2)}</pre>
          </div>
        ))}
      </div>
    </div>
  );
}
```

## Combining Voice and Event Streams

### Smart Voice Agent

```tsx
import { useVoiceStream, useEventStream } from '@/lib/react/useVoiceStream';
import { useState } from 'react';

export function VoiceAgent() {
  const [running, setRunning] = useState(false);
  const [job_id, setJobId] = useState<string>('');

  const {
    initialize: init_voice,
    send_audio,
    finish,
    transcription,
  } = useVoiceStream({
    language: 'en',
    project: 'voice-agent',
    on_transcription: async (result) => {
      if (result.is_final) {
        // Send transcription to agent
        const response = await fetch('/api/agents/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            agent: 'voice-processor',
            goal: result.transcription,
            project: 'voice-agent',
          }),
        });
        const data = await response.json();
        setJobId(data.job.id);
      }
    },
  });

  const { events, has_more, next_cursor } = useEventStream(
    job_id ? `/api/agents/jobs/${job_id}/events/stream` : null,
    {
      on_stream_end: (next, has_more_data) => {
        if (!has_more_data) {
          setRunning(false);
        }
      },
    }
  );

  const start_agent = async () => {
    setRunning(true);
    await init_voice();
    // Capture and send audio from microphone
  };

  return (
    <div>
      <button onClick={start_agent} disabled={running}>
        Start Voice Agent
      </button>
      {transcription && <p>You said: {transcription}</p>}
      {job_id && (
        <div>
          <p>Processing job: {job_id}</p>
          <ul>
            {events.map((evt, i) => (
              <li key={i}>{evt.event_type}</li>
            ))}
          </ul>
          {has_more && <button>Load More Results</button>}
        </div>
      )}
    </div>
  );
}
```

## Direct API Usage (Without Hooks)

### Voice Streaming via Fetch

```typescript
// 1. Create session
const session_response = await fetch('/api/voice/websocket/create', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    voice_session_id: 'session-' + Date.now(),
    language: 'en',
    project: 'default',
  }),
});
const { stream_id } = await session_response.json();

// 2. Send audio chunks
const audio_buffer = await fetch('audio.wav').then((r) => r.arrayBuffer());
await fetch('/api/voice/websocket/chunk', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    stream_id,
    audio_base64: btoa(String.fromCharCode(...new Uint8Array(audio_buffer))),
  }),
});

// 3. Finish and get transcription
const result = await fetch('/api/voice/websocket/finish', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ stream_id }),
});
const { transcription } = await result.json();
console.log('Transcription:', transcription);
```

### Event Stream via Fetch

```typescript
// Stream events using EventSource (auto-reconnect available with libraries)
async function* readStream(path: string) {
  const response = await fetch(path, {
    headers: { Accept: 'text/event-stream' },
  });

  if (!response.body) throw new Error('No body');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });

    const parts = buffer.split('\n\n');
    buffer = parts.pop() || '';

    for (const part of parts) {
      if (part.trim()) {
        const lines = part.split('\n');
        let event = 'message';
        let data = '';

        for (const line of lines) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          if (line.startsWith('data:')) data = line.slice(5).trim();
        }

        yield { event, data: JSON.parse(data) };
      }
    }

    if (done) break;
  }
}

// Usage
for await (const { event, data } of readStream('/api/agents/jobs/xyz/events/stream')) {
  console.log(event, data);
}
```

## Error Handling

```typescript
// With hooks
const { initialize, error } = useVoiceStream({
  on_error: (err) => {
    if (err.message.includes('session_id')) {
      console.error('Invalid session:', err);
    } else {
      console.error('Voice error:', err);
    }
  },
});

// Direct fetch
try {
  const response = await fetch('/api/voice/websocket/create', { method: 'POST', body: '...' });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Unknown error');
  }
} catch (err) {
  console.error('Failed to create voice session:', err);
}
```

## Testing in Browser Console

```javascript
// Test voice stream
fetch('/api/voice/websocket/create', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ voice_session_id: 'test-' + Date.now() }),
}).then((r) => r.json().then((d) => console.log('Created:', d)));

// Test agent events stream
const events = [];
fetch('/api/agents/jobs/test-id/events/stream')
  .then((r) => r.body.getReader())
  .then((reader) => {
    function read() {
      return reader.read().then(({ done, value }) => {
        if (!done) {
          console.log('Event:', new TextDecoder().decode(value));
          return read();
        }
      });
    }
    return read();
  });
```

## Troubleshooting

### Voice stream not getting chunks
- Check if `stream_id` is valid
- Verify audio data is base64-encoded correctly
- Check Content-Type header is `application/json`

### Events stream not connecting
- Verify job ID exists: `GET /api/agents/jobs?project=default`
- Check project parameter matches
- Ensure browser supports EventSource (SSE)

### Transcription empty
- Verify audio contains actual voice (not silence)
- Check language setting matches audio language
- Confirm STT engine is available

## Next Steps

1. **Integrate into your components**: Use `useVoiceStream` and `useEventStream` hooks
2. **Handle errors**: Always provide error callbacks
3. **Test with real data**: Use actual microphone and jobs
4. **Monitor performance**: Check browser DevTools for streaming
5. **Implement UI feedback**: Show loading states and event counts

See `STREAMING_IMPLEMENTATION.md` for complete API reference.
