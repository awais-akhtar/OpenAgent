import { useEffect, useRef, useState } from 'react';
import { api, record, records } from '../lib/react/api';
import { PageFrame } from '../components/PageFrame';

type Row = Record<string, unknown>;
const project = () => new URL(location.href).searchParams.get('project') || localStorage.getItem('oa-project') || 'default';
const val = (row: Row, key: string, fallback = '—') => row[key] == null || row[key] === '' ? fallback : String(row[key]);

export function VoicePage() {
  const [status, setStatus] = useState<Row>({}); const [personas, setPersonas] = useState<Row[]>([]); const [sessions, setSessions] = useState<Row[]>([]); const [turns, setTurns] = useState<Row[]>([]);
  const [selected, setSelected] = useState(''); const [sessionId, setSessionId] = useState(''); const [personaId, setPersonaId] = useState('');
  const [recording, setRecording] = useState(false); const [busy, setBusy] = useState(false); const [error, setError] = useState(''); const [notice, setNotice] = useState('');
  const [transcript, setTranscript] = useState(''); const [manualText, setManualText] = useState(''); const [ttsText, setTtsText] = useState(''); const [ttsResult, setTtsResult] = useState<Row>({}); const [audio, setAudio] = useState<Blob | null>(null);
  const [translateText, setTranslateText] = useState(''); const [translated, setTranslated] = useState<Row>({}); const [targetLanguage, setTargetLanguage] = useState('en');
  const [personaName, setPersonaName] = useState(''); const [personaLanguage, setPersonaLanguage] = useState('en'); const [personaStyle, setPersonaStyle] = useState('clear');
  const recorder = useRef<MediaRecorder | null>(null); const stream = useRef<MediaStream | null>(null); const chunks = useRef<BlobPart[]>([]);

  async function load() {
    setBusy(true); setError('');
    try {
      const p = encodeURIComponent(project());
      const [statusData, personaData, sessionData] = await Promise.all([
        api.get<unknown>(`/api/voice/status?project=${p}`), api.get<unknown>(`/api/voice/personas?project=${p}`), api.get<unknown>(`/api/voice/sessions?project=${p}&limit=30`),
      ]);
      setStatus(record(statusData)); setPersonas(records(personaData, 'personas')); setSessions(records(sessionData, 'sessions'));
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Voice service unavailable.'); }
    finally { setBusy(false); }
  }
  useEffect(() => { void load(); return () => { const current = recorder.current; if (current && current.state !== 'inactive') current.stop(); stream.current?.getTracks().forEach(track => track.stop()); }; }, []);

  async function loadSession(id: string) {
    setSelected(id); setTranscript(''); setTurns([]); setError('');
    try {
      const response = await api.get<unknown>(`/api/voice/turns?session_id=${encodeURIComponent(id)}`);
      const items = records(response, 'turns'); setTurns(items); setTranscript(String([...items].reverse().find(item => item.role === 'user')?.text ?? ''));
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Session transcript could not be loaded.'); }
  }
  async function createSession() {
    setBusy(true); setError('');
    try {
      const response = record(await api.post<unknown>('/api/voice/sessions', { project: project(), mode: 'push_to_talk', persona_id: personaId, language: 'en', always_listening: false, metadata: { client: 'openagent-console', explicit_microphone_start: true } }));
      const id = String(record(response.session).id ?? ''); if (!id) throw new Error('Voice service did not return a session ID.');
      setSessionId(id); setSelected(id); setNotice('Voice session created.'); await load(); return id;
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Voice session could not be created.'); return ''; }
    finally { setBusy(false); }
  }
  async function startRecording() {
    setError(''); setNotice('');
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { setError('Microphone recording is unavailable in this browser.'); return; }
    try {
      const id = sessionId || await createSession(); if (!id) return;
      const live = await navigator.mediaDevices.getUserMedia({ audio: true }); stream.current = live; chunks.current = [];
      const current = new MediaRecorder(live); recorder.current = current;
      current.ondataavailable = event => { if (event.data.size) chunks.current.push(event.data); };
      current.onstop = () => {
        const blob = new Blob(chunks.current, { type: current.mimeType || 'audio/webm' }); setAudio(blob); setRecording(false);
        stream.current?.getTracks().forEach(track => track.stop()); stream.current = null;
        setNotice(`Recording ready · ${Math.ceil(blob.size / 1024)} KB. Transcribe when ready.`);
      };
      current.start(); setRecording(true); setNotice('Microphone active. Select Stop when finished speaking.');
    } catch (cause) { stream.current?.getTracks().forEach(track => track.stop()); stream.current = null; setRecording(false); setError(cause instanceof Error ? cause.message : 'Microphone access was not granted.'); }
  }
  const base64 = (blob: Blob) => new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onerror = () => reject(reader.error); reader.onload = () => resolve(String(reader.result).split(',')[1] || ''); reader.readAsDataURL(blob); });
  async function transcribe() {
    if (!audio || !sessionId) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const response = record(await api.post<unknown>('/api/voice/stt', { project: project(), session_id: sessionId, audio_base64: await base64(audio), filename: `recording.${audio.type.includes('wav') ? 'wav' : audio.type.includes('ogg') ? 'ogg' : audio.type.includes('mp4') ? 'mp4' : 'webm'}`, language: 'en', allow_fallback_stt: false }));
      const result = record(response.transcription); setTranscript(String(result.transcript ?? result.display_text ?? ''));
      setNotice(result.transcript ? `Transcription returned · ${val(result, 'transcript_status', 'status not reported')}.` : val(result, 'stt_message', 'No transcript text returned by the configured local pipeline.'));
      await loadSession(sessionId);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Transcription failed. Captured audio remains available for another attempt.'); }
    finally { setBusy(false); }
  }
  async function createPersona(event: React.FormEvent) {
    event.preventDefault(); if (!personaName.trim()) return;
    setBusy(true); setError('');
    try {
      const response = record(await api.post<unknown>('/api/voice/personas', { name: personaName.trim(), project: project(), language: personaLanguage.trim() || 'en', style: personaStyle.trim() || 'clear', metadata: { created_from: 'openagent-console' } }));
      const persona = record(response.persona); setPersonaName(''); setPersonas(current => [persona, ...current.filter(item => item.id !== persona.id)]); setPersonaId(String(persona.id ?? '')); setNotice('Persona saved by the local voice service.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Persona could not be saved.'); }
    finally { setBusy(false); }
  }
  async function addTranscript(event: React.FormEvent) {
    event.preventDefault(); if (!sessionId || !manualText.trim()) return;
    setBusy(true); setError('');
    try { await api.post('/api/voice/transcript', { session_id: sessionId, text: manualText.trim(), speaker: 'user', role: 'user', is_final: true, metadata: { entered_in_openagent_console: true } }); setManualText(''); setNotice('Transcript turn saved to the active voice session.'); await loadSession(sessionId); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Transcript could not be saved.'); }
    finally { setBusy(false); }
  }
  async function synthesize() {
    if (!ttsText.trim()) return;
    setBusy(true); setError(''); setTtsResult({});
    try { const response = record(await api.post<unknown>('/api/voice/tts', { text: ttsText.trim(), project: project(), session_id: sessionId, persona_id: personaId, voice: val(personas.find(item => item.id === personaId) ?? {}, 'name', 'local-script') })); const script = record(response.script); setTtsResult(script); setNotice('Local WAV cue artifact generated and stored. This backend creates a tone cue, not spoken audio.'); if (sessionId) await loadSession(sessionId); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Local audio artifact could not be generated.'); }
    finally { setBusy(false); }
  }
  async function translate() {
    if (!translateText.trim()) return;
    setBusy(true); setError(''); setTranslated({});
    try {
      const response = record(await api.post<unknown>('/api/voice/translate', { text: translateText.trim(), project: project(), source_language: 'auto', target_language: targetLanguage }));
      const result = record(response.translation); setTranslated(result); setNotice('Translation completed by the configured local voice service.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Translation could not be completed.'); }
    finally { setBusy(false); }
  }
  async function closeSession() {
    if (!sessionId || !confirm('Close the active voice session?')) return;
    setBusy(true); setError('');
    try { await api.post('/api/voice/sessions/close', { session_id: sessionId }); setNotice('Voice session closed.'); setSessionId(''); setAudio(null); await load(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Session could not be closed.'); }
    finally { setBusy(false); }
  }

  const stt = record(status.stt);
  return <PageFrame eyebrow="Local media workspace" title="Voice" description="Select a local persona, capture audio only after explicit permission, inspect transcripts and create local audio cues." actions={<button className="btn" onClick={() => void load()} disabled={busy}>Refresh</button>}>
    {error && <p className="notice error" role="alert">{error}</p>}{notice && <p className="notice success" role="status">{notice}</p>}
    <div className="workspace">
      <section className="panel"><div className="panel-head"><h2>Voice sessions & personas</h2></div><div className="panel-body stack">
        <label>Persona for new sessions<select className="field" value={personaId} onChange={event => setPersonaId(event.target.value)}><option value="">No persona</option>{personas.map(persona => <option key={String(persona.id)} value={String(persona.id)}>{val(persona, 'name')} · {val(persona, 'language')}</option>)}</select></label>
        <div className="list">{sessions.map(session => <button key={String(session.id)} aria-current={selected === session.id} onClick={() => { setSessionId(String(session.id)); void loadSession(String(session.id)); }}><strong>{val(session, 'status', 'Voice session')}</strong><small>{val(session, 'mode', 'push to talk')} · {val(session, 'updated_at', val(session, 'created_at'))}</small></button>)}{!sessions.length && <p className="empty">{busy ? 'Loading sessions…' : 'No saved voice sessions.'}</p>}</div>
        <form className="stack" onSubmit={event => void createPersona(event)}><h3>Create persona</h3><label>Name<input className="field" value={personaName} onChange={event => setPersonaName(event.target.value)} required /></label><div className="two-col"><label>Language<input className="field" value={personaLanguage} onChange={event => setPersonaLanguage(event.target.value)} /></label><label>Style<input className="field" value={personaStyle} onChange={event => setPersonaStyle(event.target.value)} /></label></div><button className="btn" disabled={busy || !personaName.trim()}>Save persona</button></form>
      </div></section>
      <section className="panel"><div className="panel-head"><h2>Recorder & local speech tools</h2><span className="tag">{recording ? 'Recording' : sessionId ? 'Session active' : 'Idle'}</span></div><div className="panel-body stack">
        <div className="panel panel-body" style={{ textAlign: 'center' }}><div aria-hidden="true" style={{ fontSize: 42, color: recording ? 'var(--oa-danger)' : 'var(--oa-primary)' }}>{recording ? '●' : '◉'}</div><p>{recording ? 'Microphone is active' : audio ? 'Recording captured' : sessionId ? 'Ready for an explicit recording' : 'Create a voice session to begin'}</p><div className="toolbar" style={{ justifyContent: 'center' }}>{recording ? <button className="btn danger" onClick={() => recorder.current?.stop()}>Stop recording</button> : <button className="btn primary" onClick={() => void startRecording()} disabled={busy}>Start recording</button>}<button className="btn" onClick={() => void transcribe()} disabled={busy || recording || !audio}>Transcribe audio</button><button className="btn" onClick={() => void createSession()} disabled={busy}>New session</button></div>{sessionId && <button className="btn" onClick={() => void closeSession()} disabled={busy}>Close session</button>}</div>
        <aside className="notice">Microphone privacy: capture starts only after activation and browser permission. Audio is sent only when transcription is requested. Always-listening mode is not started.</aside>
        <small className="tiny">Local speech engines: {records(status.stt, 'engines').filter(engine => engine.available === true).map(engine => val(engine, 'name', val(engine, 'id'))).join(', ') || 'No configured engine reported'}. Audio-to-text fallback is disabled for this request.</small>
        <label>Latest transcript<textarea className="field" style={{ minHeight: 100 }} value={transcript} readOnly placeholder="Returned transcript or local pipeline status" /></label>
        <form className="stack" onSubmit={event => void addTranscript(event)}><label>Add a transcript turn<textarea className="field" rows={2} value={manualText} onChange={event => setManualText(event.target.value)} placeholder="Save text into the active session" /></label><button className="btn" disabled={busy || !sessionId || !manualText.trim()}>Save transcript turn</button></form>
        <form className="stack" onSubmit={event => { event.preventDefault(); void synthesize(); }}><label>Text for a local audio cue<textarea className="field" rows={2} value={ttsText} onChange={event => setTtsText(event.target.value)} placeholder="Create a local cue artifact" /></label><button className="btn" disabled={busy || !ttsText.trim()}>Generate local WAV cue</button>{ttsResult.audio_asset_id != null ? <small className="tiny">Stored media asset · {String(ttsResult.audio_asset_id)} · {val(ttsResult, 'audio_mime_type')}</small> : null}<small className="tiny">This service stores a deterministic tone WAV encoded from the text; it does not perform natural speech synthesis. No media playback/download route is exposed by the backend.</small></form>
        <form className="stack" onSubmit={event => { event.preventDefault(); void translate(); }}><label>Translate text<textarea className="field" rows={2} value={translateText} onChange={event => setTranslateText(event.target.value)} placeholder="Text to translate with the local translation service" /></label><label>Target language<input className="field" value={targetLanguage} onChange={event => setTargetLanguage(event.target.value)} required /></label><button className="btn" disabled={busy || !translateText.trim() || !targetLanguage.trim()}>Translate locally</button>{translated.translated_text != null && <article className="notice" aria-live="polite">{String(translated.translated_text)}<small className="tiny">{val(translated, 'source_language')} → {val(translated, 'target_language')} · {val(translated, 'provider')}</small></article>}</form>
      </div></section>
      <aside className="panel"><div className="panel-head"><h2>Session history</h2></div><div className="panel-body stack">{selected ? <><small className="tiny">Session ID · {selected}</small>{turns.map((turn, index) => <article className="panel panel-body" key={String(turn.id ?? index)}><div className="row"><strong>{val(turn, 'speaker', val(turn, 'role', 'Speaker'))}</strong><small>{val(turn, 'created_at')}</small></div><p style={{ whiteSpace: 'pre-wrap' }}>{val(turn, 'text', 'No transcript text')}</p></article>)}{!turns.length && <p className="empty">No transcript turns recorded for this session.</p>}</> : <p className="empty">Choose a saved session to inspect transcript turns.</p>}</div></aside>
    </div>
  </PageFrame>;
}
