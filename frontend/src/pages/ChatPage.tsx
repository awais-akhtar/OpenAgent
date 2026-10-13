import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react';
import { MessageSquarePlus, Search, Send, Square, RefreshCw, PanelLeftClose, PanelLeftOpen, WifiOff, Paperclip, X, Archive, GitBranch, Download, Upload, FileDown } from 'lucide-react';
import { api, record, records } from '../lib/react/api';
import { postEvents } from '../lib/react/stream';
import './chat-page.css';

type ChatItem = Record<string, unknown>;
type ChatPageProps = { project: string };
const text = (value: unknown, fallback = '') => typeof value === 'string' ? value : fallback;

export function ChatPage({ project }: ChatPageProps) {
  const [conversations, setConversations] = useState<ChatItem[]>([]);
  const [searchResults, setSearchResults] = useState<ChatItem[] | null>(null);
  const [messages, setMessages] = useState<ChatItem[]>([]);
  const [attachments, setAttachments] = useState<ChatItem[]>([]);
  const [providers, setProviders] = useState<ChatItem[]>([]);
  const [selectedId, setSelectedId] = useState(() => new URL(location.href).searchParams.get('conversation_id') || '');
  const [selectedProvider, setSelectedProvider] = useState('auto');
  const [selectedModel, setSelectedModel] = useState('');
  const [draft, setDraft] = useState('');
  const [filter, setFilter] = useState('');
  const [showArchived, setShowArchived] = useState(false);
  const [artifactFormat, setArtifactFormat] = useState('pdf');
  const [replay, setReplay] = useState<ChatItem | null>(null);
  const [allowExternal, setAllowExternal] = useState(false);
  const [localOnly, setLocalOnly] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(() => !window.matchMedia('(max-width: 650px)').matches);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [uploadError, setUploadError] = useState('');
  const controller = useRef<AbortController | null>(null);
  const alive = useRef(true);

  const loadConversations = useCallback(async () => {
    const archived = showArchived ? 'true' : 'false';
    const data = await api.get<unknown>(`/api/conversations/organized?project=${encodeURIComponent(project)}&limit=100&sort=pinned&include_archived=${archived}&archived_only=${archived}`);
    const rows = records(data, 'conversations');
    setConversations(rows);
    setSelectedId(current => current && !rows.some(item => item.id === current) ? '' : current);
  }, [project, showArchived]);
  const loadProviders = useCallback(async () => {
    try { const data = await api.get<unknown>('/api/models'); setProviders(records(data, 'providers')); }
    catch { setProviders([]); }
  }, []);
  const loadPrivacy = useCallback(() => {
    try { const saved = record(JSON.parse(localStorage.getItem('oa-privacy-settings') || '{}')); setLocalOnly(saved.localOnly !== false); }
    catch { setLocalOnly(true); }
  }, []);

  useEffect(() => {
    alive.current = true;
    void Promise.allSettled([loadConversations().catch(cause => { setError(cause instanceof Error ? cause.message : 'Could not load conversation history.'); }), loadProviders(), Promise.resolve(loadPrivacy())]).finally(() => {
      if (!alive.current) return;
      setLoading(false);
      const initialConversation = new URL(location.href).searchParams.get('conversation_id');
      if (initialConversation) void openConversation({ id: initialConversation });
    });
    return () => { alive.current = false; controller.current?.abort(); };
  }, [loadConversations, loadProviders, loadPrivacy]);

  const visibleConversations = useMemo(() => conversations.filter(item => text(item.title, 'Conversation').toLowerCase().includes(filter.toLowerCase())), [conversations, filter]);
  const activeProvider = providers.find(item => item.id === selectedProvider) ?? {};
  const availableModels = [...new Set([text(activeProvider.default_model), ...(Array.isArray(activeProvider.installed_models) ? activeProvider.installed_models.filter((item): item is string => typeof item === 'string') : [])].filter(Boolean))];

  const searchHistory = async (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== 'Enter' || event.nativeEvent.isComposing || !filter.trim()) return;
    try {
      const data = await api.get<unknown>(`/api/chat/search?q=${encodeURIComponent(filter.trim())}&project=${encodeURIComponent(project)}&limit=30`);
      setSearchResults(records(data, 'results').map(item => ({ ...item, id: item.conversation_id, title: item.conversation_title })));
      setError('');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Conversation search failed.'); }
  };
  const openConversation = async (item: ChatItem) => {
    const id = text(item.id); if (!id) return;
    controller.current?.abort(); setSending(false); setError(''); setStatus('Loading saved messages…');
    try {
      const data = await api.get<unknown>(`/api/conversation/messages?conversation_id=${encodeURIComponent(id)}`);
      setMessages(records(data, 'messages')); setSelectedId(id); setStatus('');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not load this conversation.'); setStatus(''); }
  };
  const newConversation = () => { controller.current?.abort(); setSending(false); setSelectedId(''); setMessages([]); setError(''); setStatus(''); setDraft(''); };
  const reload = async () => {
    setError('');
    try { await loadConversations(); if (selectedId) { const selected = conversations.find(item => item.id === selectedId); if (selected) await openConversation(selected); } }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not refresh conversations.'); }
  };

  const archiveConversation = async (item: ChatItem) => {
    const id = text(item.id); if (!id) return;
    const archived = item.archived === true;
    if (!archived && !confirm(`Archive “${text(item.title, 'this conversation')}”? You can restore it from Archived.`)) return;
    try {
      await api.post('/api/conversations/bulk', { conversation_ids: [id], action: archived ? 'unarchive' : 'archive', project });
      if (selectedId === id) newConversation();
      await loadConversations();
      setStatus(archived ? 'Conversation restored.' : 'Conversation archived.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Conversation could not be archived.'); }
  };
  const branchConversation = async () => {
    const message = [...messages].reverse().find(item => item.id && item.role !== 'streaming');
    if (!selectedId || !message) { setError('Open a saved conversation with at least one message before branching.'); return; }
    try {
      const result = record(await api.post<unknown>('/api/chat/branch', { conversation_id: selectedId, at_message_id: String(message.id), title: `Branch: ${text(conversations.find(item => item.id === selectedId)?.title, 'Conversation')}` }));
      const branch = record(result.branch); const conversation = record(branch.conversation);
      if (!conversation.id) throw new Error('The server did not return the new branch conversation.');
      await loadConversations(); setSelectedId(String(conversation.id)); setMessages(records(branch, 'messages')); setStatus('Conversation branch created from the selected history.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Conversation could not be branched.'); }
  };
  const inspectReplay = async () => {
    if (!selectedId) return;
    try {
      const result = record(await api.get<unknown>(`/api/chat/replay?conversation_id=${encodeURIComponent(selectedId)}`));
      setReplay(record(result.replay)); setError('');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Conversation replay could not be loaded.'); }
  };
  const exportConversation = async () => {
    if (!selectedId) return;
    try {
      const result = record(await api.get<unknown>(`/api/chat/export?conversation_id=${encodeURIComponent(selectedId)}&write_file=false`));
      const payload = JSON.stringify(result.export ?? result, null, 2);
      const url = URL.createObjectURL(new Blob([payload], { type: 'application/json' }));
      const link = document.createElement('a'); link.href = url; link.download = `openagent-conversation-${selectedId.slice(0, 10)}.json`; link.click(); URL.revokeObjectURL(url);
      setStatus('Conversation export downloaded.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Conversation export failed.'); }
  };
  const importConversation = async (event: ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget; const file = input.files?.[0]; if (!file) return;
    try {
      const parsed: unknown = JSON.parse(await file.text());
      const result = record(await api.post<unknown>('/api/chat/import', { conversation_export: record(parsed), project }));
      const imported = record(result.imported); const id = text(record(imported.conversation).id);
      await loadConversations();
      if (id) await openConversation({ id });
      setStatus('Conversation imported by the local service.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Conversation import failed.'); }
    finally { input.value = ''; }
  };
  const createArtifact = async () => {
    if (!selectedId) return;
    const lastAssistant = [...messages].reverse().find(item => item.role === 'assistant' && text(item.content).trim());
    if (!lastAssistant) { setError('There is no saved assistant response to render as an artifact.'); return; }
    try {
      const result = record(await api.post<unknown>('/api/chat/artifact', { title: text(conversations.find(item => item.id === selectedId)?.title, 'OpenAgent document'), content: text(lastAssistant.content), format: artifactFormat, project }));
      const artifact = record(result.artifact); const encoded = text(artifact.base64);
      if (!encoded) throw new Error('The document service returned no artifact content.');
      const binary = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
      const url = URL.createObjectURL(new Blob([binary], { type: text(artifact.content_type, 'application/octet-stream') }));
      const link = document.createElement('a'); link.href = url; link.download = text(artifact.filename, `openagent-document.${artifactFormat}`); link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
      setStatus('Artifact created by the document service and downloaded.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Artifact rendering failed.'); }
  };

  const send = async () => {
    const content = draft.trim(); if (!content || sending) return;
    setError(''); setUploadError(''); setDraft(''); setMessages(current => [...current, { role: 'user', content }]);
    const aborter = new AbortController(); controller.current = aborter; setSending(true); setStatus('Connecting to the selected provider…');
    let assistant = ''; let nextConversationId = selectedId;
    const parse = (value: string) => { try { return record(JSON.parse(value)); } catch { return {}; } };
    try {
      const payload = { message: content, project, provider: selectedProvider, model: selectedModel || null, allow_external: !localOnly && allowExternal, conversation_id: selectedId || null, attachments };
      for await (const event of postEvents('/api/chat/stream', payload, aborter.signal)) {
        const data = parse(event.data);
        if (event.event === 'error' || data.type === 'error') throw new Error(text(data.error ?? data.message, 'Chat stream failed.'));
        if (data.type === 'conversation' && record(data.conversation).id) { nextConversationId = text(record(data.conversation).id); setSelectedId(nextConversationId); }
        else if (data.type === 'status') setStatus(`Working${data.model ? ` with ${text(data.model)}` : ''}…`);
        else if (data.type === 'chunk' && typeof data.content === 'string') { assistant += data.content; if (data.conversation_id) nextConversationId = text(data.conversation_id); setStatus('Generating response…'); }
        else if (data.type === 'message') assistant = text(record(data.message).content, assistant);
        else if (typeof data.content === 'string') assistant += data.content;
        if (assistant) setMessages(current => [...current.filter(item => item.role !== 'streaming'), { role: 'streaming', content: assistant }]);
      }
      setMessages(current => assistant ? [...current.filter(item => item.role !== 'streaming'), { role: 'assistant', content: assistant }] : current.filter(item => item.role !== 'streaming'));
      setSelectedId(nextConversationId); setStatus(''); setAttachments([]); await loadConversations();
      if (nextConversationId) { const saved = await api.get<unknown>(`/api/conversation/messages?conversation_id=${encodeURIComponent(nextConversationId)}`); setMessages(records(saved, 'messages')); }
    } catch (cause) {
      const partial = assistant ? { role: 'interrupted', content: assistant } : null;
      setMessages(current => [...current.filter(item => item.role !== 'streaming'), ...(partial ? [partial] : [])]);
      if (aborter.signal.aborted) {
        setStatus('Generation stopped. Earlier messages remain saved.');
        if (nextConversationId) { setSelectedId(nextConversationId); try { const saved = await api.get<unknown>(`/api/conversation/messages?conversation_id=${encodeURIComponent(nextConversationId)}`); const persisted = records(saved, 'messages'); const hasPartial = partial && persisted.some(item => item.role === 'assistant' && item.content === partial.content); setMessages([...persisted, ...(partial && !hasPartial ? [partial] : [])]); await loadConversations(); } catch { /* Preserve partial output until saved messages can be reloaded. */ } }
      } else { setError(cause instanceof Error ? cause.message : 'The stream was interrupted.'); setStatus('Partial output is marked unfinished. Reload saved messages before retrying to avoid duplicating a request.'); }
    } finally { setSending(false); controller.current = null; }
  };
  const addFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget; setUploadError('');
    for (const file of Array.from(input.files ?? [])) {
      const form = new FormData(); form.append('file', file, file.name); form.append('project', project);
      try { const result = record(await api.post<unknown>('/api/knowledge/upload', form)); const item = record(result.attachment); if (!item.name) item.name = file.name; setAttachments(current => [...current, item]); }
      catch (cause) { setUploadError(`${file.name}: ${cause instanceof Error ? cause.message : 'Upload failed.'}`); }
    }
    input.value = '';
  };
  const onComposerKey = (event: KeyboardEvent<HTMLTextAreaElement>) => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send(); } };
  const role = (item: ChatItem) => text(item.role) === 'user' ? 'You' : text(item.role) === 'streaming' ? 'Assistant · generating' : text(item.role) === 'interrupted' ? 'Assistant · unfinished' : 'OpenAgent';

  return <main className="chat-workspace">
    <aside className={`history ${sidebarOpen ? '' : 'closed'}`} aria-label="Conversation history">
      <div className="history-head"><div><p className="eyebrow">WORKSPACE</p><h2>Conversations</h2></div><button className="icon" aria-label="Hide conversation history" onClick={() => setSidebarOpen(false)}><PanelLeftClose size={17}/></button></div>
      <button className="new-chat" onClick={newConversation}><MessageSquarePlus size={16}/> New conversation</button>
      <label className="search"><Search size={15}/><input value={filter} onChange={event => { setFilter(event.currentTarget.value); setSearchResults(null); }} onKeyDown={searchHistory} placeholder="Filter or press Enter to search" aria-label="Search conversations"/></label>
      <button className="refresh" onClick={() => void reload()} aria-label="Refresh conversation history"><RefreshCw size={14}/> Refresh history</button>
      <button className="refresh" aria-pressed={showArchived} onClick={() => { setShowArchived(v => !v); setSearchResults(null); setError(''); }}>{showArchived ? '← Back to conversations' : <><Archive size={14}/> Archived conversations</>}</button>
      <div className="history-list" aria-live="polite">
        {loading ? <p className="muted pad">Loading history…</p> : searchResults ? <>
          {searchResults.map((item, index) => <button key={`${String(item.id)}-${index}`} className={`history-item ${selectedId === item.id ? 'active' : ''}`} onClick={() => void openConversation(item)}><strong>{text(item.title, 'Conversation')}</strong><small>{text(item.content, 'Matching message')}</small></button>)}
          {!searchResults.length && <p className="muted pad">No messages matched that search.</p>}
        </> : visibleConversations.length ? visibleConversations.map(item => <div key={String(item.id)} className="conversation-entry"><button className={`history-item ${selectedId === item.id ? 'active' : ''}`} onClick={() => void openConversation(item)}><strong>{text(item.title, 'Conversation')}</strong><small>{text(item.updated_at ?? item.created_at, 'Saved conversation')}</small></button><button className="archive-item" aria-label={item.archived ? 'Restore conversation' : 'Archive conversation'} title={item.archived ? 'Restore conversation' : 'Archive conversation'} onClick={() => void archiveConversation(item)}><Archive size={13}/></button></div>) : <div className="empty-history"><MessageSquarePlus size={18}/><p>{filter ? 'No matching conversations.' : showArchived ? 'No archived conversations.' : 'Your saved conversations will appear here.'}</p></div>}
      </div>
    </aside>
    <section className="conversation" aria-label="Chat conversation">
      <header className="conversation-head"><button className="icon reveal" aria-label="Show conversation history" onClick={() => setSidebarOpen(true)}><PanelLeftOpen size={17}/></button><div className="heading"><p className="eyebrow">LOCAL-FIRST ASSISTANT</p><h1>{selectedId ? text(conversations.find(item => item.id === selectedId)?.title, 'Conversation') : 'A new conversation'}</h1></div>
        <label className="provider">Provider<select value={selectedProvider} onChange={event => setSelectedProvider(event.currentTarget.value)} aria-label="Chat provider">{providers.map(item => <option key={String(item.id)} value={String(item.id)}>{text(item.label, text(item.id))}{item.available === false ? ' · setup needed' : ''}</option>)}{!providers.length && <option value="auto">Auto Route</option>}</select></label><label className="provider model-select">Model<select value={selectedModel} onChange={event => setSelectedModel(event.currentTarget.value)} aria-label="Chat model"><option value="">Provider default</option>{availableModels.map(model => <option key={model} value={model}>{model}</option>)}</select></label>
        {selectedId && <div className="chat-tools"><button className="tool-button" onClick={() => void branchConversation()} title="Branch at the last saved message"><GitBranch size={14}/> Branch</button><button className="tool-button" onClick={() => void inspectReplay()} title="Inspect saved replay timeline">Replay</button><button className="tool-button" onClick={() => void exportConversation()} title="Export conversation"><Download size={14}/> Export</button><label className="tool-button" title="Import a conversation JSON"><Upload size={14}/> Import<input type="file" accept="application/json,.json" onChange={importConversation} aria-label="Import conversation JSON"/></label><select aria-label="Artifact format" value={artifactFormat} onChange={event => setArtifactFormat(event.currentTarget.value)}>{['pdf','docx','pptx','xlsx','md','html','txt'].map(format => <option key={format}>{format}</option>)}</select><button className="tool-button" onClick={() => void createArtifact()} title="Render last assistant response"><FileDown size={14}/> Artifact</button></div>}
      </header>
      {replay && <aside className="replay-panel" aria-label="Saved conversation replay"><div><strong>Saved replay · {text(replay.message_count, '—')} messages</strong><button className="icon" aria-label="Close replay" onClick={() => setReplay(null)}><X size={15}/></button></div>{records(replay, 'timeline').map((event, index) => <p key={`${String(event.created_at)}-${index}`}><small>{text(event.role)} · {text(event.created_at)}</small><br/>{text(event.content)}</p>)}</aside>}
      <div className="message-scroll" aria-live="polite" aria-relevant="additions text">
        {!messages.length && <div className="welcome"><div className="welcome-mark"><MessageSquarePlus size={21}/></div><h3>What are you working on?</h3><p>Choose a provider and start a conversation. Requests stay local unless you explicitly allow external access.</p><div className="suggestions"><button onClick={() => setDraft('Help me plan my next project')}>Plan a project</button><button onClick={() => setDraft('Summarize the current workspace')}>Summarize workspace</button></div></div>}
        {messages.map((item, index) => <article key={`${String(item.id ?? item.role)}-${index}`} className={`message ${item.role === 'user' ? 'user' : ''} ${item.role === 'streaming' ? 'streaming' : ''} ${item.role === 'interrupted' ? 'interrupted' : ''}`}><span className="speaker">{role(item)}</span><p>{text(item.content ?? item.message)}</p></article>)}
      </div>
      <div className="composer-zone">
        {error && <div className="inline-error" role="alert"><WifiOff size={15}/><span>{error}</span><button className="text-button" onClick={() => void reload()}>Reload</button></div>}
        {status && <div className="stream-status" role="status"><span className={sending ? 'active' : ''}></span>{status}</div>}
        <div className="composer"><textarea value={draft} onChange={event => setDraft(event.currentTarget.value)} onKeyDown={onComposerKey} placeholder="Message OpenAgent…" rows={2} aria-label="Message"/>
          {!!attachments.length && <div className="attachment-tray" aria-label="Uploaded attachments">{attachments.map((item, index) => <span key={`${text(item.name)}-${index}`}>{text(item.name, 'Attachment')}<button aria-label={`Remove ${text(item.name, 'attachment')}`} onClick={() => setAttachments(current => current.filter((_, i) => i !== index))}><X size={12}/></button></span>)}</div>}
          <div className="composer-bottom"><div className="privacy"><label><input type="checkbox" checked={allowExternal} onChange={event => setAllowExternal(event.currentTarget.checked)} disabled={localOnly}/>{localOnly ? 'External providers blocked by local-only policy' : 'Allow external providers for this message'}</label><span>Local-first by default</span></div>
            <div className="composer-actions"><label className="attach" title="Upload attachment"><Paperclip size={15}/><span>Attach</span><input type="file" multiple onChange={addFiles} aria-label="Upload attachments"/></label>{sending ? <button className="stop" onClick={() => controller.current?.abort()} aria-label="Stop generation"><Square size={14} fill="currentColor"/> Stop</button> : <button className="send" onClick={() => void send()} disabled={!draft.trim()} aria-label="Send message"><Send size={15}/> Send</button>}</div>
          </div>
        </div>{uploadError && <p className="upload-error" role="alert">{uploadError}</p>}<p className="hint">Enter to send · Shift + Enter for a new line</p>
      </div>
    </section>
  </main>;
}
