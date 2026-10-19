import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type Dispatch, type KeyboardEvent, type SetStateAction } from 'react';
import { Ellipsis, GitBranch, Download, Upload, FileDown, WifiOff, Archive, PanelRight, FileText, Eye, Info, Paperclip } from 'lucide-react';
import { api, record, records } from '../lib/react/api';
import { postEvents } from '../lib/react/stream';
import { ChatComposer } from '../components/chat/ChatComposer';
import { ConversationTimeline } from '../components/chat/ConversationTimeline';
import { MessageContent } from '../components/chat/MessageContent';
import './chat-page.css';

type ChatItem = Record<string, unknown>;
type ChatPageProps = {
  project: string;
  draft?: string;
  setDraft?: Dispatch<SetStateAction<string>>;
  attachments?: ChatItem[];
  setAttachments?: Dispatch<SetStateAction<ChatItem[]>>;
};
const text = (value: unknown, fallback = '') => typeof value === 'string' ? value : fallback;
const formatMessageTime = (value: unknown): string => {
  if (typeof value !== 'string' || !value) return '';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

export function ChatPage(props: ChatPageProps) {
  const { project } = props;
  const [conversations, setConversations] = useState<ChatItem[]>([]);
  const [messages, setMessages] = useState<ChatItem[]>([]);
  const [localAttachments, setLocalAttachments] = useState<ChatItem[]>([]);
  const [providers, setProviders] = useState<ChatItem[]>([]);
  const [agents, setAgents] = useState<ChatItem[]>([]);
  const [providersError, setProvidersError] = useState('');
  const [agentsError, setAgentsError] = useState('');
  const [providersLoading, setProvidersLoading] = useState(true);
  const [agentsLoading, setAgentsLoading] = useState(true);
  const [selectedAgent, setSelectedAgent] = useState('');
  const [selectedId, setSelectedId] = useState(() => new URL(location.href).searchParams.get('conversation_id') || '');
  const [selectedProvider, setSelectedProvider] = useState('auto');
  const [selectedModel, setSelectedModel] = useState('');
  const [localDraft, setLocalDraft] = useState('');
  const [artifactFormat, setArtifactFormat] = useState('pdf');
  const [replay, setReplay] = useState<ChatItem | null>(null);
  const [allowExternal, setAllowExternal] = useState(false);
  const [localOnly, setLocalOnly] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [inspectorTab, setInspectorTab] = useState<'files' | 'preview' | 'context'>('preview');
  const [inspectorOpen, setInspectorOpen] = useState(() => typeof window === 'undefined' || window.innerWidth > 680);
  const [uploadError, setUploadError] = useState('');
  const controller = useRef<AbortController | null>(null);
  const openRequest = useRef(0);
  const alive = useRef(true);
  const routeConversation = useRef(new URL(location.href).searchParams.get('conversation_id') || '');
  const draft = props.draft ?? localDraft;
  const setDraft = props.setDraft ?? setLocalDraft;
  const attachments = props.attachments ?? localAttachments;
  const setAttachments = props.setAttachments ?? setLocalAttachments;
  const timelineMessages = useMemo(() => messages.map(message => typeof message.created_at === 'string' ? { ...message, created_at: formatMessageTime(message.created_at) } : message), [messages]);

  const loadConversations = useCallback(async () => {
    const data = await api.get<unknown>(`/api/conversations/organized?project=${encodeURIComponent(project)}&limit=100&sort=pinned&include_archived=false&archived_only=false`);
    const rows = records(data, 'conversations');
    setConversations(rows);
    window.dispatchEvent(new Event('openagent:conversations-updated'));
  }, [project]);
  const loadProviders = useCallback(async () => {
    setProvidersLoading(true);
    try { const data = await api.get<unknown>('/api/models'); setProviders(records(data, 'providers')); setProvidersError(''); }
    catch (cause) { setProvidersError(cause instanceof Error ? cause.message : 'Model options could not be loaded.'); }
    finally { setProvidersLoading(false); }
  }, []);
  const loadAgents = useCallback(async () => {
    setAgentsLoading(true);
    try { const data = await api.get<unknown>('/api/agents'); setAgents(records(data, 'agents')); setAgentsError(''); }
    catch (cause) { setAgentsError(cause instanceof Error ? cause.message : 'Assistant options could not be loaded.'); }
    finally { setAgentsLoading(false); }
  }, []);
  const loadPrivacy = useCallback(() => {
    try { const saved = record(JSON.parse(localStorage.getItem('oa-privacy-settings') || '{}')); setLocalOnly(saved.localOnly !== false); }
    catch { setLocalOnly(true); }
  }, []);

  useEffect(() => {
    alive.current = true;
    // Home hands off its unsent prompt once, without putting private text in the URL.
    const handoffKey = `oa-chat-handoff:${project}`;
    try { const handoff = sessionStorage.getItem(handoffKey); if (handoff) { setLocalDraft(current => current || handoff); sessionStorage.removeItem(handoffKey); } } catch { /* Storage can be disabled; the composer remains available. */ }
    void Promise.allSettled([loadConversations().catch(cause => { setError(cause instanceof Error ? cause.message : 'Could not load conversation history.'); }), loadProviders(), loadAgents(), Promise.resolve(loadPrivacy())]).finally(() => {
      if (!alive.current) return;
      const initialConversation = new URL(location.href).searchParams.get('conversation_id');
      if (initialConversation) void openConversation({ id: initialConversation });
    });
    return () => { alive.current = false; controller.current?.abort(); };
  }, [loadConversations, loadProviders, loadAgents, loadPrivacy]);

  useEffect(() => {
    const syncRoute = () => {
      const nextId = new URL(location.href).searchParams.get('conversation_id') || '';
      if (nextId === routeConversation.current) return;
      routeConversation.current = nextId;
      if (!nextId) {
        openRequest.current++;
        const active = controller.current; controller.current = null; active?.abort();
        setSending(false); setSelectedId(''); setMessages([]); setError(''); setStatus(''); return;
      }
      void openConversation({ id: nextId });
    };
    // The shell updates the selected conversation in the URL and dispatches popstate.
    window.addEventListener('popstate', syncRoute);
    return () => window.removeEventListener('popstate', syncRoute);
  }, []);

  useEffect(() => { routeConversation.current = selectedId; }, [selectedId]);

  const activeProvider = providers.find(item => item.id === selectedProvider) ?? {};
  const selectedConversation = conversations.find(item => text(item.id) === selectedId);
  const activeFiles = useMemo(() => {
    const fromDraft = attachments;
    const fromMessages = messages.flatMap(message => Array.isArray(message.attachments) ? message.attachments.map(record) : []);
    const unique = new Map<string, ChatItem>();
    [...fromDraft, ...fromMessages].forEach((file, index) => unique.set(text(file.id, `${text(file.name, 'File')}-${index}`), file));
    return [...unique.values()];
  }, [attachments, messages]);
  const latestAnswer = [...messages].reverse().find(item => (item.role === 'assistant' || item.role === 'streaming') && text(item.content).trim());
  const latestAnswerText = latestAnswer ? text(latestAnswer.content) : '';
  const latestAnswerWords = latestAnswerText.trim() ? latestAnswerText.trim().split(/\s+/).length : 0;
  const latestAnswerTime = typeof latestAnswer?.created_at === 'string' ? formatMessageTime(latestAnswer.created_at) : '';
  const focusLatestAnswer = () => { const answers = document.querySelectorAll<HTMLElement>('.oa-chat-message.assistant'); answers.item(answers.length - 1)?.scrollIntoView({ behavior: 'smooth', block: 'center' }); };
  const availableModels = [...new Set([text(activeProvider.default_model), ...(Array.isArray(activeProvider.installed_models) ? activeProvider.installed_models.filter((item): item is string => typeof item === 'string') : [])].filter(Boolean))];

  const openConversation = async (item: ChatItem) => {
    const id = text(item.id); if (!id) return;
    const request = ++openRequest.current;
    const active = controller.current; controller.current = null; active?.abort();
    setSending(false); setError(''); setStatus('Loading saved messages…');
    try {
      const data = await api.get<unknown>(`/api/conversation/messages?conversation_id=${encodeURIComponent(id)}`);
      if (request !== openRequest.current) return;
      setMessages(records(data, 'messages')); setSelectedId(id); routeConversation.current = id; setStatus('');
    } catch (cause) {
      if (request !== openRequest.current) return;
      setError(cause instanceof Error ? cause.message : 'Could not load this conversation.'); setStatus('');
    }
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
      if (!encoded) throw new Error('The document could not be created. No file content was returned.');
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
    let assistant = ''; let nextConversationId = selectedId; let serverAcknowledged = false;
    const optimisticMessageId = 'pending-' + Date.now() + '-' + Math.random().toString(36).slice(2);
    setMessages(current => current.map((item, index) => index === current.length - 1 && item.role === 'user' && item.content === content ? { ...item, id: optimisticMessageId } : item));
    if (selectedAgent) {
      try {
        setStatus(`Running ${text(agents.find(agent => agent.id === selectedAgent)?.name, selectedAgent)} through the checked agent service…`);
        const result = record(await api.post<unknown>('/api/agents/run-checked', { agent: selectedAgent, goal: content, project, require_approval: false, verify: true }, aborter.signal));
        if (controller.current !== aborter) return;
        const run = record(result.run);
        const response = text(run.response, text(run.result, text(result.message, 'The agent completed without a text response.')));
        setMessages(current => [...current.filter(item => item.id !== optimisticMessageId), { role: 'assistant', content: response }]);
        setStatus(result.approval ? 'The agent request is waiting for approval in Agents.' : 'Agent run completed.');
        setAttachments([]); setSelectedAgent('');
      } catch (cause) {
        if (controller.current !== aborter) return;
        setMessages(current => current.filter(item => item.id !== optimisticMessageId));
        setDraft(current => current || content);
        if (aborter.signal.aborted) { setError(''); setStatus('Agent run stopped; your prompt was restored.'); }
        else { setError(cause instanceof Error ? cause.message : 'The checked agent run failed.'); setStatus(''); }
      } finally { if (controller.current === aborter) { setSending(false); controller.current = null; } }
      return;
    }
    const parse = (value: string) => { try { return record(JSON.parse(value)); } catch { return {}; } };
    try {
      const payload = { message: content, project, provider: selectedProvider, model: selectedModel || null, allow_external: !localOnly && allowExternal, conversation_id: selectedId || null, attachments };
      for await (const event of postEvents('/api/chat/stream', payload, aborter.signal)) {
        const data = parse(event.data);
        if (event.event === 'error' || data.type === 'error') throw new Error(text(data.error ?? data.message, 'Chat stream failed.'));
        if (data.type === 'conversation' && record(data.conversation).id) { serverAcknowledged = true; nextConversationId = text(record(data.conversation).id); setSelectedId(nextConversationId); routeConversation.current = nextConversationId; const url = new URL(location.href); url.searchParams.set('conversation_id', nextConversationId); history.replaceState({}, '', url); }
        else if (data.type === 'status') setStatus(`Working${data.model ? ` with ${text(data.model)}` : ''}…`);
        else if (data.type === 'chunk' && typeof data.content === 'string') { serverAcknowledged = true; assistant += data.content; if (data.conversation_id) nextConversationId = text(data.conversation_id); setStatus('Generating response…'); }
        else if (data.type === 'message') { serverAcknowledged = true; assistant = text(record(data.message).content, assistant); }
        else if (typeof data.content === 'string') { serverAcknowledged = true; assistant += data.content; }
        if (assistant) setMessages(current => [...current.filter(item => item.role !== 'streaming'), { role: 'streaming', content: assistant }]);
      }
      if (!serverAcknowledged) throw new Error('The local service closed the stream before confirming the message.');
      setMessages(current => assistant ? [...current.filter(item => item.role !== 'streaming'), { role: 'assistant', content: assistant }] : current.filter(item => item.role !== 'streaming'));
      setSelectedId(nextConversationId); routeConversation.current = nextConversationId; if (nextConversationId) { const url = new URL(location.href); url.searchParams.set('conversation_id', nextConversationId); history.replaceState({}, '', url); }
      setStatus(''); setAttachments([]); await loadConversations();
      if (controller.current !== aborter) return;
      if (nextConversationId) {
        const saved = await api.get<unknown>(`/api/conversation/messages?conversation_id=${encodeURIComponent(nextConversationId)}`);
        if (controller.current !== aborter) return;
        setMessages(records(saved, 'messages'));
      }
    } catch (cause) {
      // A route change owns the transcript now; only an explicit Stop button
      // keeps this request current and should render its partial result.
      if (controller.current !== aborter) return;
      const partial = assistant ? { role: 'interrupted', content: assistant } : null;
      setMessages(current => [...current.filter(item => item.role !== 'streaming' && (serverAcknowledged || item.id !== optimisticMessageId)), ...(partial ? [partial] : [])]);
      if (!serverAcknowledged) setDraft(current => current || content);
      if (aborter.signal.aborted) {
        setStatus('Generation stopped. Earlier messages remain saved.');
        if (nextConversationId) { setSelectedId(nextConversationId); try { const saved = await api.get<unknown>(`/api/conversation/messages?conversation_id=${encodeURIComponent(nextConversationId)}`); const persisted = records(saved, 'messages'); const hasPartial = partial && persisted.some(item => item.role === 'assistant' && item.content === partial.content); setMessages([...persisted, ...(partial && !hasPartial ? [partial] : [])]); await loadConversations(); } catch { /* Preserve partial output until saved messages can be reloaded. */ } }
      } else { setError(cause instanceof Error ? cause.message : 'The stream was interrupted.'); setStatus('Partial output is marked unfinished. Reload saved messages before retrying to avoid duplicating a request.'); }
    } finally {
      if (controller.current === aborter) { setSending(false); controller.current = null; }
    }
  };
  const uploadFiles = async (files: File[]) => {
    setUploadError('');
    for (const file of files) {
      const form = new FormData(); form.append('file', file, file.name); form.append('project', project);
      try { const result = record(await api.post<unknown>('/api/knowledge/upload', form)); const item = record(result.attachment); if (!item.name) item.name = file.name; setAttachments(current => [...current, item]); }
      catch (cause) { setUploadError(`${file.name}: ${cause instanceof Error ? cause.message : 'Upload failed.'}`); }
    }
  };
  const addFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget;
    await uploadFiles(Array.from(input.files ?? []));
    input.value = '';
  };
  const onComposerKey = (event: KeyboardEvent<HTMLTextAreaElement>) => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send(); } };
  return <main className="oa-chat-page" aria-label="Chat">
    <section className={`oa-chat-surface ${messages.length ? 'conversation-active' : 'conversation-home'} ${inspectorOpen ? 'has-inspector' : 'inspector-closed'}`} aria-label="Chat conversation">
      <div className="oa-chat-workarea">
      <div className="oa-chat-primary">
      <header className="oa-chat-contextbar"><div><span>Workspace</span><span aria-hidden="true">/</span><strong>{selectedConversation ? text(selectedConversation.title, 'Conversation') : project === 'default' ? 'OpenAgent' : project}</strong></div><div className="oa-chat-contextbar-actions"><button type="button" aria-label={inspectorOpen ? 'Hide context panel' : 'Show context panel'} aria-expanded={inspectorOpen} onClick={() => setInspectorOpen(value => !value)}><PanelRight size={15}/><span>{inspectorOpen ? 'Hide context' : 'Show context'}</span></button>
        {selectedId && <details className="oa-chat-menu"><summary aria-label="Conversation actions" title="Conversation actions"><Ellipsis size={19}/></summary><div className="oa-chat-menu-popover">
          <button onClick={() => void branchConversation()}><GitBranch size={15}/> Branch at last message</button><button onClick={() => void inspectReplay()}>View saved replay</button><button onClick={() => void exportConversation()}><Download size={15}/> Export conversation</button>
          <button onClick={() => void archiveCurrentConversation()}><Archive size={15}/> Archive conversation</button>
          <label><Upload size={15}/> Import conversation<input type="file" accept="application/json,.json" onChange={importConversation} aria-label="Import conversation JSON"/></label>
          <div className="oa-chat-artifact-row"><label htmlFor="artifact-format">Create document</label><select id="artifact-format" aria-label="Artifact format" value={artifactFormat} onChange={event => setArtifactFormat(event.currentTarget.value)}>{['pdf','docx','pptx','xlsx','md','html','txt'].map(format => <option key={format}>{format}</option>)}</select><button onClick={() => void createArtifact()} aria-label="Create document"><FileDown size={15}/></button></div>
        </div></details>}
      </div></header>
      {error && <div className="oa-chat-error" role="alert"><WifiOff size={15}/><span>{error}</span><button onClick={() => void reload()}>Reload saved conversation</button></div>}
      {status && <div className="oa-chat-status" role="status"><span className={sending ? 'active' : ''}/>{sending && !messages.some(item => item.role === 'streaming') && <span className="oa-chat-wait-dots" aria-hidden="true"><i/><i/><i/></span>}{status}</div>}
      <ConversationTimeline messages={timelineMessages} sending={sending} replay={replay} onCloseReplay={() => setReplay(null)}/>
      {uploadError && <p className="oa-chat-upload-error" role="alert">{uploadError}</p>}
      <ChatComposer draft={draft} project={project} setDraft={setDraft} onSubmit={() => void send()} onKeyDown={onComposerKey} onFiles={addFiles} onDropFiles={files => void uploadFiles(files)} attachments={attachments} removeAttachment={index => setAttachments(current => current.filter((_, itemIndex) => itemIndex !== index))} providers={providers} provider={selectedProvider} setProvider={setSelectedProvider} model={selectedModel} setModel={setSelectedModel} models={availableModels} providersError={providersError} providersLoading={providersLoading} onRetryProviders={() => void loadProviders()} agents={agents} agent={selectedAgent} setAgent={setSelectedAgent} agentsError={agentsError} agentsLoading={agentsLoading} onRetryAgents={() => void loadAgents()} localOnly={localOnly} external={allowExternal} setExternal={setAllowExternal} sending={sending} onStop={() => controller.current?.abort()}/>
      </div>
      {inspectorOpen && <aside className="oa-chat-inspector" aria-label="Conversation context">
        <div className="oa-chat-inspector-tabs" role="tablist" aria-label="Conversation panel">
          <button type="button" role="tab" aria-selected={inspectorTab === 'files'} onClick={() => setInspectorTab('files')}><FileText size={14}/>Files{activeFiles.length > 0 && <span>{activeFiles.length}</span>}</button>
          <button type="button" role="tab" aria-selected={inspectorTab === 'preview'} onClick={() => setInspectorTab('preview')}><Eye size={14}/>Preview</button>
          <button type="button" role="tab" aria-selected={inspectorTab === 'context'} onClick={() => setInspectorTab('context')}><Info size={14}/>Context</button>
        </div>
        <div className="oa-chat-inspector-body" role="tabpanel">
          {inspectorTab === 'files' ? <section><h2>Conversation files</h2>{activeFiles.length ? <ul className="oa-chat-file-list">{activeFiles.map((file, index) => <li key={text(file.id, `${text(file.name, 'file')}-${index}`)}><Paperclip size={15}/><span><strong>{text(file.name, 'Attachment')}</strong><small>{text(file.content_type, text(file.type, 'Attached to this conversation'))}</small></span></li>)}</ul> : <div className="oa-chat-inspector-empty"><FileText size={21}/><strong>No files attached</strong><p>Files you add to this conversation will appear here.</p><button type="button" onClick={() => document.querySelector<HTMLButtonElement>('.oa-chat-composer [aria-label="Attach files"]')?.click()}>Add files</button></div>}</section> : inspectorTab === 'preview' ? <section><h2>Latest response</h2>{latestAnswer ? <div className="oa-chat-answer-preview"><p className={`oa-chat-preview-state ${latestAnswer.role === 'streaming' ? 'is-streaming' : ''}`}><span aria-hidden="true"/>{latestAnswer.role === 'streaming' ? 'Generating in conversation' : 'Available in conversation'}</p><dl className="oa-chat-preview-metadata"><div><dt>Length</dt><dd>{latestAnswerText.length.toLocaleString()} characters</dd></div><div><dt>Words</dt><dd>{latestAnswerWords.toLocaleString()}</dd></div>{latestAnswerTime && <div><dt>Received</dt><dd>{latestAnswerTime}</dd></div>}</dl><button type="button" className="oa-chat-preview-jump" onClick={focusLatestAnswer}>View response in conversation</button></div> : <div className="oa-chat-inspector-empty"><Eye size={21}/><strong>No response to preview</strong><p>An assistant response will appear here after one is available.</p></div>}</section> : <section><h2>Context</h2><dl className="oa-chat-context-facts"><div><dt>Workspace</dt><dd>{project === 'default' ? 'Personal workspace' : project}</dd></div><div><dt>Conversation</dt><dd>{selectedConversation ? text(selectedConversation.title, 'Untitled conversation') : selectedId ? 'Saved conversation' : 'New conversation'}</dd></div><div><dt>Messages</dt><dd>{messages.length}</dd></div><div><dt>Assistant</dt><dd>{selectedAgent ? text(agents.find(item => text(item.id) === selectedAgent)?.name, selectedAgent) : 'Chat'}</dd></div><div><dt>Provider</dt><dd>{selectedAgent ? 'Checked agent run' : text(activeProvider.label, selectedProvider === 'auto' ? 'Automatic' : selectedProvider)}</dd></div><div><dt>External access</dt><dd>{localOnly ? 'Blocked by local-only policy' : allowExternal ? 'Allowed for this message' : 'Off for this message'}</dd></div></dl><p className="oa-chat-context-note">Context reflects this conversation and its current privacy settings.</p></section>}
        </div>
      </aside>}
      </div>
    </section>
  </main>;
}
