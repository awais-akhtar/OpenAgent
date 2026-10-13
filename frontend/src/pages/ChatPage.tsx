import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react';
import { Ellipsis, GitBranch, Download, Upload, FileDown, WifiOff, Archive } from 'lucide-react';
import { api, record, records } from '../lib/react/api';
import { postEvents } from '../lib/react/stream';
import { ChatComposer } from '../components/chat/ChatComposer';
import { ConversationTimeline } from '../components/chat/ConversationTimeline';
import './chat-page.css';

type ChatItem = Record<string, unknown>;
type ChatPageProps = { project: string };
const text = (value: unknown, fallback = '') => typeof value === 'string' ? value : fallback;

export function ChatPage({ project }: ChatPageProps) {
  const [conversations, setConversations] = useState<ChatItem[]>([]);
  const [messages, setMessages] = useState<ChatItem[]>([]);
  const [attachments, setAttachments] = useState<ChatItem[]>([]);
  const [providers, setProviders] = useState<ChatItem[]>([]);
  const [selectedId, setSelectedId] = useState(() => new URL(location.href).searchParams.get('conversation_id') || '');
  const [selectedProvider, setSelectedProvider] = useState('auto');
  const [selectedModel, setSelectedModel] = useState('');
  const [draft, setDraft] = useState('');
  const [artifactFormat, setArtifactFormat] = useState('pdf');
  const [replay, setReplay] = useState<ChatItem | null>(null);
  const [allowExternal, setAllowExternal] = useState(false);
  const [localOnly, setLocalOnly] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [uploadError, setUploadError] = useState('');
  const controller = useRef<AbortController | null>(null);
  const alive = useRef(true);
  const routeConversation = useRef(new URL(location.href).searchParams.get('conversation_id') || '');

  const loadConversations = useCallback(async () => {
    const data = await api.get<unknown>(`/api/conversations/organized?project=${encodeURIComponent(project)}&limit=100&sort=pinned&include_archived=false&archived_only=false`);
    const rows = records(data, 'conversations');
    setConversations(rows);
    window.dispatchEvent(new Event('openagent:conversations-updated'));
  }, [project]);
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
      const initialConversation = new URL(location.href).searchParams.get('conversation_id');
      if (initialConversation) void openConversation({ id: initialConversation });
    });
    return () => { alive.current = false; controller.current?.abort(); };
  }, [loadConversations, loadProviders, loadPrivacy]);

  useEffect(() => {
    const syncRoute = () => {
      const nextId = new URL(location.href).searchParams.get('conversation_id') || '';
      if (nextId === routeConversation.current) return;
      routeConversation.current = nextId;
      if (!nextId) { controller.current?.abort(); setSending(false); setSelectedId(''); setMessages([]); setError(''); setStatus(''); return; }
      void openConversation({ id: nextId });
    };
    // The shell updates the selected conversation in the URL and dispatches popstate.
    window.addEventListener('popstate', syncRoute);
    return () => window.removeEventListener('popstate', syncRoute);
  }, []);

  useEffect(() => { routeConversation.current = selectedId; }, [selectedId]);

  const activeProvider = providers.find(item => item.id === selectedProvider) ?? {};
  const availableModels = [...new Set([text(activeProvider.default_model), ...(Array.isArray(activeProvider.installed_models) ? activeProvider.installed_models.filter((item): item is string => typeof item === 'string') : [])].filter(Boolean))];

  const openConversation = async (item: ChatItem) => {
    const id = text(item.id); if (!id) return;
    controller.current?.abort(); setSending(false); setError(''); setStatus('Loading saved messages…');
    try {
      const data = await api.get<unknown>(`/api/conversation/messages?conversation_id=${encodeURIComponent(id)}`);
      setMessages(records(data, 'messages')); setSelectedId(id); routeConversation.current = id; setStatus('');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not load this conversation.'); setStatus(''); }
  };
  const reload = async () => {
    setError('');
    try { await loadConversations(); if (selectedId) { const selected = conversations.find(item => item.id === selectedId); if (selected) await openConversation(selected); } }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not refresh conversations.'); }
  };

  const branchConversation = async () => {
    const message = [...messages].reverse().find(item => item.id && item.role !== 'streaming');
    if (!selectedId || !message) { setError('Open a saved conversation with at least one message before branching.'); return; }
    try {
      const result = record(await api.post<unknown>('/api/chat/branch', { conversation_id: selectedId, at_message_id: String(message.id), title: `Branch: ${text(conversations.find(item => item.id === selectedId)?.title, 'Conversation')}` }));
      const branch = record(result.branch); const conversation = record(branch.conversation);
      if (!conversation.id) throw new Error('The server did not return the new branch conversation.');
      const branchId = String(conversation.id); const url = new URL(location.href); url.searchParams.set('conversation_id', branchId); history.pushState({}, '', url); routeConversation.current = branchId;
      await loadConversations(); setSelectedId(branchId); setMessages(records(branch, 'messages')); setStatus('Conversation branch created from the selected history.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Conversation could not be branched.'); }
  };
  const archiveCurrentConversation = async () => {
    if (!selectedId) return;
    const current = conversations.find(item => item.id === selectedId);
    if (!confirm(`Archive “${text(current?.title, 'this conversation')}”? You can restore it from Archived.`)) return;
    try {
      await api.post('/api/conversations/bulk', { conversation_ids: [selectedId], action: 'archive', project });
      controller.current?.abort(); setSelectedId(''); routeConversation.current = ''; setMessages([]);
      const url = new URL(location.href); url.searchParams.delete('conversation_id'); history.replaceState({}, '', url);
      await loadConversations(); setStatus('Conversation archived.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Conversation could not be archived.'); }
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
      if (id) { const url = new URL(location.href); url.searchParams.set('conversation_id', id); history.pushState({}, '', url); routeConversation.current = id; await openConversation({ id }); }
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
        if (data.type === 'conversation' && record(data.conversation).id) { nextConversationId = text(record(data.conversation).id); setSelectedId(nextConversationId); routeConversation.current = nextConversationId; const url = new URL(location.href); url.searchParams.set('conversation_id', nextConversationId); history.replaceState({}, '', url); }
        else if (data.type === 'status') setStatus(`Working${data.model ? ` with ${text(data.model)}` : ''}…`);
        else if (data.type === 'chunk' && typeof data.content === 'string') { assistant += data.content; if (data.conversation_id) nextConversationId = text(data.conversation_id); setStatus('Generating response…'); }
        else if (data.type === 'message') assistant = text(record(data.message).content, assistant);
        else if (typeof data.content === 'string') assistant += data.content;
        if (assistant) setMessages(current => [...current.filter(item => item.role !== 'streaming'), { role: 'streaming', content: assistant }]);
      }
      setMessages(current => assistant ? [...current.filter(item => item.role !== 'streaming'), { role: 'assistant', content: assistant }] : current.filter(item => item.role !== 'streaming'));
      setSelectedId(nextConversationId); routeConversation.current = nextConversationId; if (nextConversationId) { const url = new URL(location.href); url.searchParams.set('conversation_id', nextConversationId); history.replaceState({}, '', url); }
      setStatus(''); setAttachments([]); await loadConversations();
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
  const currentTitle = text(conversations.find(item => item.id === selectedId)?.title, 'Conversation');

  return <main className="oa-chat-page" aria-label="Chat">
    <section className={`oa-chat-surface ${messages.length ? 'conversation-active' : 'conversation-home'}`} aria-label="Chat conversation">
      <div className="oa-chat-topline">
        {selectedId && <span className="oa-chat-title" title={currentTitle}>{currentTitle}</span>}
        {selectedId && <details className="oa-chat-menu"><summary aria-label="Conversation actions" title="Conversation actions"><Ellipsis size={19}/></summary><div className="oa-chat-menu-popover">
          <button onClick={() => void branchConversation()}><GitBranch size={15}/> Branch at last message</button><button onClick={() => void inspectReplay()}>View saved replay</button><button onClick={() => void exportConversation()}><Download size={15}/> Export conversation</button>
          <button onClick={() => void archiveCurrentConversation()}><Archive size={15}/> Archive conversation</button>
          <label><Upload size={15}/> Import conversation<input type="file" accept="application/json,.json" onChange={importConversation} aria-label="Import conversation JSON"/></label>
          <div className="oa-chat-artifact-row"><label htmlFor="artifact-format">Create document</label><select id="artifact-format" aria-label="Artifact format" value={artifactFormat} onChange={event => setArtifactFormat(event.currentTarget.value)}>{['pdf','docx','pptx','xlsx','md','html','txt'].map(format => <option key={format}>{format}</option>)}</select><button onClick={() => void createArtifact()} aria-label="Create document"><FileDown size={15}/></button></div>
        </div></details>}
      </div>
      {error && <div className="oa-chat-error" role="alert"><WifiOff size={15}/><span>{error}</span><button onClick={() => void reload()}>Reload saved conversation</button></div>}
      {status && <div className="oa-chat-status" role="status"><span className={sending ? 'active' : ''}/>{status}</div>}
      <ConversationTimeline messages={messages} sending={sending} replay={replay} onCloseReplay={() => setReplay(null)}/>
      {uploadError && <p className="oa-chat-upload-error" role="alert">{uploadError}</p>}
      <ChatComposer draft={draft} project={project} setDraft={setDraft} onSubmit={() => void send()} onKeyDown={onComposerKey} onFiles={addFiles} attachments={attachments} removeAttachment={index => setAttachments(current => current.filter((_, itemIndex) => itemIndex !== index))} providers={providers} provider={selectedProvider} setProvider={setSelectedProvider} model={selectedModel} setModel={setSelectedModel} models={availableModels} localOnly={localOnly} external={allowExternal} setExternal={setAllowExternal} sending={sending} onStop={() => controller.current?.abort()}/>
    </section>
  </main>;
}
