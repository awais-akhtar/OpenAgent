# Real-Time Streaming Implementation Guide

This document describes the real-time streaming features for Voice and Agent/Workflow functionality in OpenAgent.

## Overview

The implementation provides three main streaming capabilities:

1. **Voice Streaming** - Real-time audio streaming with incremental transcription
2. **Agent Event Streaming** - SSE-based streaming of agent job events with pagination
3. **Workflow Event Streaming** - SSE-based streaming of workflow run events with pagination

## Voice Streaming

### Backend Implementation

Voice streaming is handled through HTTP endpoints that work with the `VoiceWebSocketStreamManager`:

```
POST /api/voice/websocket/create        - Create a new voice stream session
POST /api/voice/websocket/chunk         - Send audio chunk to stream
POST /api/voice/websocket/finish        - Finish streaming and get transcription
GET  /api/voice/websocket/status        - Get session status

GET  /api/voice/stream/continuous/{session_id}/stream     - SSE stream for frames
POST /api/voice/stream/continuous/{session_id}/chunk      - Send chunk with session ID
POST /api/voice/stream/continuous/{session_id}/finish     - Finish with session ID
```

### Features

- **Project Scoping**: Each voice session is scoped to a project and owner
- **Incremental Frames**: Audio chunks can be sent incrementally
- **Real-time Transcription**: Stream transcription updates as audio is processed
- **Session Management**: Sessions can be created, monitored, and finalized

### React Hook Usage

```tsx
import { useVoiceStream } from '@/lib/react/useVoiceStream';

function VoiceRecorder() {
  const { initialize, send_audio, finish, transcription, error } = useVoiceStream({
    language: 'en',
    project: 'my-project',
    on_frame: (frame) => console.log('Frame:', frame),
    on_transcription: (result) => console.log('Transcription:', result),
    on_error: (error) => console.error('Error:', error),
  });

  const handleStartRecording = async () => {
    await initialize();
    // Start sending audio chunks from microphone
  };

  const handleAudioChunk = async (audioData: Uint8Array) => {
    await send_audio(audioData, 'raw');
  };

  const handleFinishRecording = async () => {
    const result = await finish();
    console.log('Final transcription:', result.transcription);
  };

  return (
    <div>
      <button onClick={handleStartRecording}>Start Recording</button>
      <button onClick={handleFinishRecording}>Stop & Transcribe</button>
      {transcription && <p>Transcript: {transcription}</p>}
      {error && <p>Error: {error.message}</p>}
    </div>
  );
}
```

## Agent Event Streaming

### Backend Implementation

Agent job events are streamed via Server-Sent Events (SSE):

```
GET /api/agents/jobs/{id}/events              - Get events (JSON)
GET /api/agents/jobs/{id}/events/stream       - Stream events (SSE)
```

### Features

- **Proper SSE Framing**: Events are correctly formatted as ServerSentEvent objects
- **Cursor-based Pagination**: Use `?after=<cursor>` for resumable history
- **Event Filtering**: Filter by event type or job status
- **Deduplication**: Duplicate events are automatically filtered

### Query Parameters

- `after` - Cursor ID for pagination (event ID to resume from)
- `type` - Filter by event type (started, progress, completed, failed)
- `status` - Filter by job status (running, completed, failed, cancelled)
- `limit` - Maximum events to return (1-1000, default 1000)

### React Hook Usage

```tsx
import { useEventStream } from '@/lib/react/useVoiceStream';

function JobMonitor({ jobId }: { jobId: string }) {
  const { events, next_cursor, has_more, can_load_more } = useEventStream(
    `/api/agents/jobs/${jobId}/events/stream`,
    {
      on_stream_start: (cursor) => console.log('Stream started at:', cursor),
      on_event: (event) => console.log('Event:', event),
      on_stream_end: (next, has_more) => console.log('Stream ended. Has more:', has_more),
    }
  );

  const handleLoadMore = async () => {
    // Reload stream starting from next_cursor
    // Component will automatically fetch the next batch
  };

  return (
    <div>
      {events.map((evt) => (
        <div key={evt.id}>
          <span>{evt.event_type}</span>
          <pre>{JSON.stringify(evt.payload || evt.data)}</pre>
        </div>
      ))}
      {can_load_more && <button onClick={handleLoadMore}>Load More Events</button>}
    </div>
  );
}
```

## Workflow Event Streaming

### Backend Implementation

Workflow run events are streamed via SSE with improved pagination:

```
GET /api/workflows/runs/{id}/events         - Get events (JSON)
GET /api/workflows/runs/{id}/events/stream  - Stream events (SSE)
```

### Features

- **Run Status Events**: Status change events for workflow runs
- **Completion Events**: Completion event with final result
- **Pagination Support**: Full pagination with `has_more` flag
- **Project Scoping**: Events are scoped to project

### Workflow Event Structure

Each workflow event includes:

```typescript
{
  id: "run-id:status" | "run-id:completed",
  event_type: "run.status" | "run.completed",
  workflow_id: string,
  run_id: string,
  status: "pending" | "running" | "completed" | "failed",
  timestamp: string,
  created_at: string,
  project: string,
  data: {
    status: string,
    result: object,
    completed_at?: string,
    success?: boolean
  }
}
```

### Query Parameters

- `after` - Cursor for pagination (event ID)
- `limit` - Maximum events (1-500, default 100)
- `project` - Project scope (for security)

### React Hook Usage

```tsx
import { useEventStream } from '@/lib/react/useVoiceStream';

function WorkflowMonitor({ runId }: { runId: string }) {
  const [page, setPage] = useState(0);
  const [all_events, setAllEvents] = useState<any[]>([]);

  const { events, next_cursor, has_more } = useEventStream(
    `/api/workflows/runs/${runId}/events/stream?project=default`,
    {
      on_event: (event) => {
        setAllEvents((prev) => [...prev, event]);
      },
    }
  );

  const loadNextPage = () => {
    if (has_more && next_cursor) {
      // Re-query with next_cursor as the 'after' parameter
      // This will load the next batch of events
    }
  };

  return (
    <div>
      {all_events.map((evt) => (
        <div key={evt.id}>
          <h4>{evt.event_type}</h4>
          <p>Status: {evt.status}</p>
          <pre>{JSON.stringify(evt.data)}</pre>
        </div>
      ))}
      {has_more && <button onClick={loadNextPage}>Load More</button>}
    </div>
  );
}
```

## SSE Event Structure

All streaming endpoints return Server-Sent Events with the following structure:

### Stream Start Event
```json
{
  "event": "stream.start",
  "data": {
    "type": "stream.started",
    "job_id|run_id|session_id": "...",
    "cursor": "...",
    "limit": 1000
  }
}
```

### Regular Event
```json
{
  "event": "job.event|workflow.event|voice.frame",
  "data": {
    "id": "event-id",
    "event_type": "...",
    "payload": {...}
  }
}
```

### Stream End Event
```json
{
  "event": "stream.end",
  "data": {
    "type": "stream.ended",
    "next_cursor": "...",
    "after_id": "...",
    "has_more": true,
    "event_count": 42
  }
}
```

### Stream Error Event
```json
{
  "event": "stream.error",
  "data": {
    "type": "stream.error",
    "error": "error message",
    "message": "stream error message"
  }
}
```

## Pagination Implementation

### Cursor-based Pagination

All streaming endpoints support cursor-based pagination:

1. **First Request**: Call the streaming endpoint without `after` parameter
2. **Get Next Cursor**: From the `stream_ended` event, extract `next_cursor` or `after_id`
3. **Subsequent Requests**: Call with `?after=<next_cursor>`
4. **Check Has More**: The `has_more` field indicates if more data exists

Example:

```typescript
// First page
const response = await fetch('/api/agents/jobs/job-id/events/stream');

// Process events until stream.ended event
// Get next_cursor from stream_ended event

// Next page (if has_more is true)
const nextResponse = await fetch(
  `/api/agents/jobs/job-id/events/stream?after=${encodeURIComponent(next_cursor)}`
);
```

### Multiple Pagination Field Names

For flexibility, pagination fields are provided with multiple names:

- `next` / `next_cursor` / `after_id` - All refer to the cursor for the next batch
- `cursor` - Backward compatibility

## Project and Owner Scoping

All endpoints respect project scoping:

- Voice operations are scoped by `project` and `owner` parameters
- Agent jobs track `owner` for authorization
- Workflow runs are scoped by `project`
- Unauthorized access returns 404 (not found)

### Specifying Project/Owner

**In Request Body:**
```json
{
  "project": "my-project",
  "owner": "user-id",
  "...": "..."
}
```

**In Query Parameters:**
```
?project=my-project&owner=user-id
```

## Error Handling

### Common Error Responses

| Status | Meaning | Solution |
|--------|---------|----------|
| 400 | Bad Request | Check parameter format and required fields |
| 401 | Unauthorized | Provide valid authentication token |
| 404 | Not Found | Check resource ID and project scope |
| 429 | Rate Limited | Implement exponential backoff retry |
| 500 | Server Error | Check server logs, retry after delay |

### SSE Stream Errors

When an error occurs during streaming, a `stream.error` event is sent:

```typescript
// Stream stops after error event
for await (const event of getEvents(path, signal)) {
  if (event.event === 'stream.error') {
    const data = JSON.parse(event.data);
    throw new Error(data.error || 'Stream error');
  }
}
```

## Best Practices

1. **Always Initialize Voice Sessions**: Call `initialize()` before sending audio chunks
2. **Handle Disconnections**: Implement reconnection logic with exponential backoff
3. **Use Pagination for Large Result Sets**: Don't try to fetch all events at once
4. **Monitor Stream Status**: Check the `stream_ended` event for completion
5. **Clean Up Resources**: Call `finish()` or cleanup hooks to release stream resources
6. **Respect Rate Limits**: Implement appropriate throttling for chunk uploads

## Testing

### Voice Stream Test

```typescript
import { useVoiceStream } from '@/lib/react/useVoiceStream';
import { renderHook, act } from '@testing-library/react';

it('streams voice audio and gets transcription', async () => {
  const { result } = renderHook(() => useVoiceStream({ language: 'en' }));

  await act(async () => {
    await result.current.initialize();

    // Create 1 second of audio at 16kHz
    const audioBuffer = new ArrayBuffer(32000);
    const view = new Uint8Array(audioBuffer);
    view.fill(0); // Silence

    await result.current.send_audio(audioBuffer, 'raw');
    const final = await result.current.finish();

    expect(final).toHaveProperty('transcription');
  });
});
```

### Event Stream Test

```typescript
import { useEventStream } from '@/lib/react/useVoiceStream';
import { renderHook, waitFor } from '@testing-library/react';

it('loads agent job events with pagination', async () => {
  const { result } = renderHook(() =>
    useEventStream('/api/agents/jobs/test-id/events/stream')
  );

  await waitFor(() => {
    expect(result.current.is_loading).toBe(false);
  });

  expect(Array.isArray(result.current.events)).toBe(true);
  expect(typeof result.current.has_more).toBe('boolean');
});
```

## Migration Notes

If you're upgrading from older polling-based implementations:

1. **Voice**: Replace polling calls to `get_status` with `stream_continuous()`
2. **Events**: Replace `setInterval` polling with `useEventStream` hook
3. **Pagination**: Replace manual cursor tracking with hook's `next_cursor` and `has_more`
4. **Error Handling**: Add handlers for `stream.error` events instead of checking response status

## Future Enhancements

Planned improvements:

1. **True WebSocket Support**: Replace HTTP polling with WebSocket for lower latency
2. **Bidirectional Voice**: Send and receive audio simultaneously
3. **Workflow Step Events**: Detailed events for each workflow step
4. **Event Filtering**: Server-side event filtering to reduce bandwidth
5. **Reconnection**: Automatic reconnection with cursor resumption
