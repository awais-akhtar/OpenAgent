# Semantic Memory, Media, Voice, and UI Runtime

Stage 11 adds local-first runtime implementations for semantic memory, vector collections, reranking, graph analytics, media ingestion, voice sessions, connector sync workers, command shortcuts, and draggable workflow layouts.

## Privacy Boundary

- Semantic memory, embeddings, vector search, reranking, and graph analytics use deterministic local code and SQLite records.
- Media ingestion reads local files or explicit base64 uploads. Files can be copied into the OpenAgent data directory. No cloud OCR, cloud vision, cloud STT, or cloud TTS provider is called.
- Connector sync consumes an explicit provider payload posted by the user or client. Live external provider polling uses connector-specific adapters, credentials, and explicit external approval.
- Voice sessions store text transcripts, personas, wake-word metadata, local voiceprint clone records, submitted-audio STT records, and local TTS WAV artifacts.
- Workflow layout edits are local UI metadata. Saving a drag/drop layout writes an audit event and does not execute workflow steps.

## Semantic Memory APIs

Create a typed memory:

```bash
curl -X POST http://127.0.0.1:8765/api/semantic/memories \
  -H "Content-Type: application/json" \
  -d "{\"project\":\"core\",\"memory_type\":\"preference\",\"title\":\"Model preference\",\"content\":\"Use local models unless cloud is approved.\",\"tags\":[\"privacy\",\"models\"]}"
```

Search semantic memories:

```bash
curl "http://127.0.0.1:8765/api/semantic/memories?project=core&q=local%20models"
```

Create and search a local vector collection:

```bash
curl -X POST http://127.0.0.1:8765/api/semantic/collections \
  -H "Content-Type: application/json" \
  -d "{\"project\":\"core\",\"name\":\"research\"}"

curl -X POST http://127.0.0.1:8765/api/semantic/vectors \
  -H "Content-Type: application/json" \
  -d "{\"project\":\"core\",\"collection_id\":\"COLLECTION_ID\",\"external_id\":\"note-1\",\"content\":\"OpenAgent stores private context locally.\"}"

curl -X POST http://127.0.0.1:8765/api/semantic/vector-search \
  -H "Content-Type: application/json" \
  -d "{\"collection_id\":\"COLLECTION_ID\",\"query\":\"private context\"}"
```

Rerank local candidates:

```bash
curl -X POST http://127.0.0.1:8765/api/semantic/rerank \
  -H "Content-Type: application/json" \
  -d "{\"project\":\"core\",\"query\":\"private local memory\",\"candidates\":[{\"id\":\"a\",\"title\":\"Local\",\"content\":\"Keep memory on the local machine.\"}]}"
```

## Media APIs

Ingest a local file path:

```bash
curl -X POST http://127.0.0.1:8765/api/media/ingest \
  -H "Content-Type: application/json" \
  -d "{\"project\":\"core\",\"path\":\"D:/openagent/examples/brief.pdf\",\"source\":\"file\"}"
```

Ingest an inline base64 upload:

```bash
curl -X POST http://127.0.0.1:8765/api/media/ingest \
  -H "Content-Type: application/json" \
  -d "{\"project\":\"core\",\"filename\":\"note.txt\",\"data_base64\":\"SGVsbG8gZnJvbSBPcGVuQWdlbnQ=\"}"
```

Media analysis supports:

- Text-like files: local text extraction.
- PDF files: local best-effort text scan.
- Images: format and dimension metadata from file headers.
- Audio and video: local metadata registration.

## Connector Sync APIs

Run a local provider-payload sync:

```bash
curl -X POST http://127.0.0.1:8765/api/communication/sync \
  -H "Content-Type: application/json" \
  -d "{\"project\":\"core\",\"platform\":\"email\",\"payload\":{\"messages\":[{\"sender\":\"lead@example.com\",\"subject\":\"Follow up\",\"body\":\"Please follow up after review.\"}]}}"
```

List sync runs:

```bash
curl "http://127.0.0.1:8765/api/communication/sync-runs?project=core"
```

## Voice APIs

### Local STT Accuracy

OpenAgent does not call browser/cloud speech services by default. Accurate transcription requires a local STT engine. The voice pipeline now:

- records explicit microphone input only,
- normalizes submitted WAV audio,
- trims leading/trailing silence,
- resamples to 16 kHz mono PCM before STT,
- prefers high-quality local engines,
- reports transcript confidence and warnings,
- keeps Windows Speech as an explicit fallback because it can produce inaccurate text.

Recommended local setup:

```powershell
pip install -e ".[voice]"
openagent-workbench
```

After the `voice` extra is installed, OpenAgent uses `base.en` with faster-whisper by default. The first run may download the model into the local model cache. For air-gapped use, set `OPENAGENT_FASTER_WHISPER_MODEL` to a local model path and set `OPENAGENT_STT_OFFLINE_ONLY=1`.

Smooth runtime mode recommends `faster-whisper:base.en` for local CPU transcription because it gives a good accuracy and memory balance for 8 GB systems. The UI exposes the selected STT engine, confidence, warnings, and an editable transcript so users can correct speech recognition before sending it into chat.

Other supported local options:

- `OPENAGENT_STT_COMMAND`: custom local command. Use `{input}` and `{language}` placeholders.
- `OPENAGENT_WHISPER_MODEL`: open-source Whisper model name when the `whisper` package is installed.
- `OPENAGENT_WHISPER_CPP_EXE` plus `OPENAGENT_WHISPER_CPP_MODEL`: local whisper.cpp executable and model file.
- `OPENAGENT_VOSK_MODEL`: local Vosk model directory.
- `OPENAGENT_ENABLE_WINDOWS_SPEECH=1`: allow local Windows Speech fallback. Use this only as a fallback; verify/edit the transcript before sending.

No STT transcript should be treated as final unless the response status is `transcribed` and confidence is acceptable for the use case.

Create a session and add a transcript:

```bash
curl -X POST http://127.0.0.1:8765/api/voice/sessions \
  -H "Content-Type: application/json" \
  -d "{\"project\":\"core\",\"mode\":\"push_to_talk\",\"language\":\"en\"}"

curl -X POST http://127.0.0.1:8765/api/voice/transcript \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"SESSION_ID\",\"speaker\":\"user\",\"text\":\"Summarize the roadmap.\"}"
```

Submit local audio for speech-to-text:

```bash
curl -X POST http://127.0.0.1:8765/api/voice/stt \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"SESSION_ID\",\"filename\":\"speech.wav\",\"audio_base64\":\"BASE64_WAV\",\"stt_engine\":\"faster-whisper\",\"allow_fallback_stt\":false}"
```

Create a local voice clone record from submitted voiceprint audio:

```bash
curl -X POST http://127.0.0.1:8765/api/voice/clone \
  -H "Content-Type: application/json" \
  -d "{\"project\":\"core\",\"name\":\"Local Voice\",\"filename\":\"voiceprint.wav\",\"audio_base64\":\"BASE64_WAV\",\"consent\":\"Owner submitted this sample.\"}"
```

Generate local TTS audio:

```bash
curl -X POST http://127.0.0.1:8765/api/voice/tts \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"SESSION_ID\",\"text\":\"Roadmap summary ready.\"}"
```

The TTS endpoint generates a local WAV media asset plus segment metadata without calling an external TTS provider.

## UI Runtime

The static web workspace now includes:

- `Ctrl+K` command palette.
- `Ctrl+Enter` send chat.
- `Ctrl+Shift+A` run selected agent.
- `Ctrl+Shift+S` save memory.
- `Ctrl+Shift+F` focus memory search.
- Draggable workflow canvas nodes.

Workflow layout saves call `POST /api/workflows/layout`, persist local layout metadata, and write `workflow.layout.save` audit events.
