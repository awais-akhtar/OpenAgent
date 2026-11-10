import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiConnectionMessage, apiDelete, apiGet, apiGetBlob, apiPatch, apiPost, apiStreamPost, type StreamEvent } from '../api/client';
import { useWorkspaceState } from '../state';
import { demoQuery } from '../demo/api';
import {
  Archive, AudioLines, Bot, BookOpen, Check, ChevronDown, ChevronLeft, ChevronRight,
  Copy, FileText, Globe2, Mic, MoreHorizontal, Pin, Square, ThumbsUp,
  Paperclip, Plus, Search, Send, Sparkles, Trash2, Wrench,
} from 'lucide-react';

type Go = (id: string) => void;
type ScreenProps = { go?: Go };
const noNavigation: Go = () => undefined;
let stagedPrompt = '';
const stagedPromptStorageKey = 'openagent.chat.stagedPrompt';
/**
 * End-of-speech detection for hands-free voice. The backend transcribes only
 * when a stream is finished and returns no interim text, so the end of an
 * utterance is decided here from the microphone's own level.
 */
const LIVE_SPEECH_LEVEL = 0.025;
const LIVE_SILENCE_MS = 1200;
let stagedAttachments: ChatAttachment[] = [];
const stagedAttachmentsStorageKey = 'openagent.chat.stagedAttachments';

function ActionButton({ children, onClick, className = '', title, nav }: { children: ReactNode; onClick?: () => void; className?: string; title?: string; nav?: string }) {
  return <button type="button" title={title} data-nav={nav} onClick={onClick} className={className}>{children}</button>;
}
function IconSquare({ children, tone = 'blue', size }: { children: ReactNode; tone?: string; size?: string }) {
  return <span className={`square-icon ${size ?? ''} ${tone}`}>{children}</span>;
}

function renderInlineChatMarkdown(value: string): ReactNode[] {
  const parts = value.split(/(`[^`]+`|\*\*[^*]+\*\*|__[^_]+__)/g).filter(Boolean);
  return parts.map((part, index) => {
    if (part.startsWith('`') && part.endsWith('`')) return <code key={index}>{part.slice(1, -1)}</code>;
    if ((part.startsWith('**') && part.endsWith('**')) || (part.startsWith('__') && part.endsWith('__'))) return <strong key={index}>{part.slice(2, -2)}</strong>;
    return <span key={index}>{part}</span>;
  });
}

function renderChatMarkdown(text: string): ReactNode {
  const blocks: ReactNode[] = [];
  const lines = text.replace(/\r\n?/g, '\n').split('\n');
  let code: string[] = [];
  let language = '';
  let inCode = false;
  let list: string[] = [];
  const flushList = () => { if (list.length) { blocks.push(<ul key={`list-${blocks.length}`}>{list.map((item, index) => <li key={index}>{renderInlineChatMarkdown(item)}</li>)}</ul>); list = []; } };
  const flushCode = () => { if (code.length) { blocks.push(<pre className="chat-code-block" key={`code-${blocks.length}`}><code data-language={language || undefined}>{code.join('\n')}</code></pre>); code = []; language = ''; } };
  lines.forEach((line, index) => {
    if (line.trimStart().startsWith('```')) { if (inCode) { flushCode(); inCode = false; } else { flushList(); language = line.trim().slice(3).trim(); code = []; inCode = true; } return; }
    if (inCode) { code.push(line); return; }
    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    if (bullet) { list.push(bullet[1]); return; }
    flushList();
    const heading = line.match(/^\s*(#{1,3})\s+(.*)$/);
    if (heading) { blocks.push(<h3 key={`heading-${index}`}>{renderInlineChatMarkdown(heading[2])}</h3>); return; }
    if (!line.trim()) return;
    blocks.push(<p key={`paragraph-${index}`}>{renderInlineChatMarkdown(line)}</p>);
  });
  flushList(); flushCode();
  return blocks.length ? blocks : <p />;
}

export { HomeDashboard as HomeScreen } from './HomeDashboard';

type ChatLine = { role: 'user' | 'assistant'; text: string; createdAt?: string };
type ChatLineWithId = ChatLine & { id?: string; serverId?: string };
type ToolOutput = { id: string; title: string; text: string };
type Conversation = { id: string; title?: string; pinned?: boolean; archived?: boolean; updated_at?: string; semantic_score?: number };
type Provider = { id: string; label?: string; default_model?: string; selected_model?: string; installed_models?: string[]; available?: boolean; local?: boolean; network?: boolean; privacy_warning?: string };
/** One selectable entry in the composer's model list, including Auto. */
type ModelChoice = { key: string; providerId: string; model: string; label: string; detail: string; local: boolean; available: boolean };
type ChatAttachment = { id: string; name: string; media_type: string; mime_type: string; size_bytes?: number };
const storedConversationKey = 'oa-second-conversation-id';
function records(value: unknown): Record<string, unknown>[] { return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object') : []; }
function relativeTimestamp(value: unknown) {
  const timestamp = typeof value === 'string' || typeof value === 'number' ? Date.parse(String(value)) : Number.NaN;
  if (!Number.isFinite(timestamp)) return typeof value === 'string' ? value : '';
  const minutes = Math.max(0, Math.floor((Date.now() - timestamp) / 60_000));
  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  if (hours < 48) return 'Yesterday';
  return `${Math.floor(hours / 24)}d ago`;
}
function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
    reader.onload = () => {
      const result = String(reader.result ?? '');
      const comma = result.indexOf(',');
      if (comma < 0) reject(new Error(`Could not encode ${file.name}.`));
      else resolve(result.slice(comma + 1));
    };
    reader.readAsDataURL(file);
  });
}
function fileToText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
    reader.onload = () => resolve(String(reader.result ?? ''));
    reader.readAsText(file);
  });
}
export function ChatScreen({ go = noNavigation }: ScreenProps = {}) {
  const dataMode = useWorkspaceState(state => state.dataMode);
  const [draft, setDraft] = useState('');
  const [lines, setLines] = useState<ChatLineWithId[]>([]);
  const [conversationId, setConversationId] = useState(() => {
    if (dataMode !== 'live') return '';
    const linked = new URL(location.href).searchParams.get('conversation_id');
    if (linked) return linked;
    try { return sessionStorage.getItem(storedConversationKey) ?? ''; } catch { return ''; }
  });
  const [providerId, setProviderId] = useState('auto');
  const [modelId, setModelId] = useState('');
  const [allowExternal, setAllowExternal] = useState(false);
  const [inspector, setInspector] = useState(true);
  const [toolsOpen, setToolsOpen] = useState(false);
  const [toolBusy, setToolBusy] = useState('');
  const [toolError, setToolError] = useState('');
  const [toolQuery, setToolQuery] = useState('');
  const [activeTool, setActiveTool] = useState<'knowledge' | 'research' | 'document' | ''>('');
  const [recording, setRecording] = useState(false);
  const [voiceBusy, setVoiceBusy] = useState(false);
  const [voiceError, setVoiceError] = useState('');
  const [toolOutputs, setToolOutputs] = useState<ToolOutput[]>([]);
  const [recentLimit, setRecentLimit] = useState(8);
  const [liveVoice, setLiveVoice] = useState(false);
  const [liveStatus, setLiveStatus] = useState<'' | 'listening' | 'thinking' | 'speaking'>('');
  const [rowMenuId, setRowMenuId] = useState('');
  const [pendingDeleteId, setPendingDeleteId] = useState('');
  const [renamingId, setRenamingId] = useState('');
  const [renameValue, setRenameValue] = useState('');
  const [streamError, setStreamError] = useState('');
  const [completionNotice, setCompletionNotice] = useState('');
  const [waitingForContent, setWaitingForContent] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [historyOpen, setHistoryOpen] = useState(false);
  const [modelControlsOpen, setModelControlsOpen] = useState(false);
  const [historySearch, setHistorySearch] = useState('');
  const [showArchived, setShowArchived] = useState(false);
  const [historyError, setHistoryError] = useState('');
  const [historyBusyId, setHistoryBusyId] = useState('');
  const [conversationActionMessage, setConversationActionMessage] = useState('');
  const [conversationActionError, setConversationActionError] = useState('');
  const [actionMessageId, setActionMessageId] = useState('');
  const importInputRef = useRef<HTMLInputElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const voiceSessionRef = useRef('');
  const liveActiveRef = useRef(false);
  const liveRecorderRef = useRef<MediaRecorder | null>(null);
  const liveStreamRef = useRef<MediaStream | null>(null);
  const liveContextRef = useRef<AudioContext | null>(null);
  const liveFrameRef = useRef<number | null>(null);
  const speakerRef = useRef<HTMLAudioElement | null>(null);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const requestSequenceRef = useRef(0);
  const waitingTimerRef = useRef<number | null>(null);
  const queryClient = useQueryClient();
  useEffect(() => {
    let prompt = stagedPrompt;
    stagedPrompt = '';
    let pendingAttachments = stagedAttachments;
    stagedAttachments = [];
    try {
      prompt ||= sessionStorage.getItem(stagedPromptStorageKey) ?? '';
      sessionStorage.removeItem(stagedPromptStorageKey);
      const savedAttachments = sessionStorage.getItem(stagedAttachmentsStorageKey);
      sessionStorage.removeItem(stagedAttachmentsStorageKey);
      if (savedAttachments) {
        const parsed: unknown = JSON.parse(savedAttachments);
        if (Array.isArray(parsed)) pendingAttachments = parsed.filter((item): item is ChatAttachment => Boolean(item) && typeof item === 'object' && typeof (item as ChatAttachment).id === 'string' && typeof (item as ChatAttachment).name === 'string' && typeof (item as ChatAttachment).media_type === 'string' && typeof (item as ChatAttachment).mime_type === 'string').slice(0, 12);
      }
    } catch { /* use the in-memory handoff when storage is unavailable */ }
    if (prompt.trim()) setDraft(current => current || prompt);
    if (dataMode === 'live' && pendingAttachments.length) setAttachments(current => current.length ? current : pendingAttachments);
  }, [dataMode]);
  // The recent list reads the organized route because the plain conversations
  // route returns neither `pinned` nor `archived`, so a pinned conversation
  // could never show as pinned or sort to the top.
  const conversationsQuery = useQuery({ queryKey: ['chat', 'conversations', dataMode], queryFn: ({ signal }) => dataMode === 'demo' ? Promise.resolve({ conversations: [] }) : apiGet<{ conversations?: unknown[] }>('/api/conversations/organized?project=default&limit=100&sort=pinned&include_archived=false&archived_only=false', signal), retry: false });
  const historyQuery = useQuery({
    queryKey: ['chat', 'organized-history', dataMode, historySearch, showArchived],
    queryFn: ({ signal }) => dataMode === 'demo' ? Promise.resolve({ conversations: [] }) : apiGet<{ conversations?: unknown[] }>(`/api/conversations/organized?project=default&limit=100&sort=${historySearch.trim() ? 'semantic' : 'pinned'}&include_archived=${showArchived ? 'true' : 'false'}&archived_only=${showArchived ? 'true' : 'false'}&q=${encodeURIComponent(historySearch.trim())}`, signal),
    retry: false,
    enabled: historyOpen,
  });
  const providersQuery = useQuery({
    queryKey: ['chat', 'providers', dataMode],
    queryFn: async ({ signal }): Promise<{ providers?: unknown[] }> => {
      if (dataMode !== 'demo') return apiGet<{ providers?: unknown[] }>('/api/models', signal);
      const models = await demoQuery<Record<string, unknown>>(dataMode, 'shell.models', s => apiGet<Record<string, unknown>>('/api/models', s), signal);
      const configuredModels = records(models).filter(model => String(model.status ?? '').toLowerCase() === 'configured');
      const selectedModel = configuredModels.find(model => model.name === 'GPT-4o') ?? configuredModels[0];
      const modelNames = configuredModels.map(model => String(model.name ?? '')).filter(Boolean);
      return { providers: [{
        id: 'auto',
        label: 'Auto route · local only',
        selected_model: String(selectedModel?.name ?? 'GPT-4o'),
        default_model: String(selectedModel?.name ?? 'GPT-4o'),
        installed_models: modelNames,
        available: true,
        local: true,
      }] };
    },
    retry: false,
  });
  const conversations = useMemo(() => records(conversationsQuery.data?.conversations).flatMap(item => {
    const id = item.id ?? item.conversation_id;
    return id === undefined || id === null || String(id).trim() === '' ? [] : [{ ...item, id: String(id) } as Conversation];
  }), [conversationsQuery.data]);
  const history = useMemo(() => {
    const rows = records(historyQuery.data?.conversations) as Conversation[];
    return historySearch.trim() ? rows.filter(item => Number(item.semantic_score ?? 0) > 0) : rows;
  }, [historyQuery.data, historySearch]);
  const providers = useMemo(() => records(providersQuery.data?.providers) as Provider[], [providersQuery.data]);
  const selectedProvider = providers.find(provider => provider.id === providerId);
  const selectedNeedsEgress = selectedProvider?.local === false && providerId !== 'auto';
  // One flat list of every model the backend reports, the way a provider's own
  // picker reads, instead of a provider select and then a model select. Auto
  // stays first and leaves the choice to the backend's routing policy.
  const modelChoices = useMemo(() => {
    const choices: ModelChoice[] = [{
      key: 'auto', providerId: 'auto', model: '', label: 'Auto',
      detail: 'Let the backend route this request', local: true, available: true,
    }];
    for (const provider of providers) {
      // Only providers that can actually run a request. Listing the rest just
      // showed names that could never be selected.
      if (provider.id === 'auto' || provider.available === false) continue;
      const names = Array.from(new Set([
        provider.selected_model, provider.default_model,
        ...(Array.isArray(provider.installed_models) ? provider.installed_models : []),
      ].filter((name): name is string => Boolean(name))));
      for (const model of names) {
        choices.push({
          key: `${provider.id}:${model}`, providerId: provider.id, model, label: model,
          detail: provider.label || provider.id,
          local: provider.local !== false,
          // Unavailable providers are skipped above, so anything listed runs.
          available: true,
        });
      }
    }
    return choices;
  }, [providers]);
  const selectedChoiceKey = providerId === 'auto'
    ? 'auto'
    : `${providerId}:${modelId || selectedProvider?.selected_model || selectedProvider?.default_model || ''}`;
  const selectedChoice = modelChoices.find(choice => choice.key === selectedChoiceKey);
  const modelButtonLabel = providerId === 'auto'
    ? 'Auto'
    : modelId || selectedProvider?.selected_model || selectedProvider?.default_model || 'Select model';
  const chooseModel = (choice: ModelChoice) => {
    setProviderId(choice.providerId);
    setModelId(choice.model);
    // Selecting a concrete external model is the explicit routing choice;
    // no separate consent checkbox is needed in the model picker.
    setAllowExternal(!choice.local);
    setModelControlsOpen(false);
  };
  const messagesQuery = useQuery({ queryKey: ['chat', 'messages', conversationId, dataMode], queryFn: ({ signal }) => apiGet<{ messages?: unknown[] }>(`/api/messages?conversation_id=${encodeURIComponent(conversationId)}`, signal), enabled: Boolean(conversationId) && dataMode === 'live', retry: false });
  // Do not render any cached Live messages during the render that changes
  // modes; the effect below also clears their selection and local state.
  const visibleLines = dataMode === 'demo' ? [] : lines;
  useEffect(() => {
    if (streaming) return;
    const messages = records(messagesQuery.data?.messages);
    // A stream can return real text before (or without) a persisted
    // conversation ID. Ending that stream must not erase its received rows.
    // Explicit conversation/mode changes clear lines in selectConversation.
    if (!messages.length) return;
    setLines(messages.flatMap(message => {
      const role = message.role === 'user' ? 'user' : message.role === 'assistant' ? 'assistant' : null;
      return role && typeof message.content === 'string' ? [{ role, text: message.content, createdAt: typeof message.created_at === 'string' ? message.created_at : undefined, ...(typeof message.id === 'string' ? { id: message.id, serverId: message.id } : {}) } as ChatLineWithId] : [];
    }));
  }, [messagesQuery.data, conversationId, streaming]);
  const clearWaitingTimer = () => {
    if (waitingTimerRef.current !== null) window.clearTimeout(waitingTimerRef.current);
    waitingTimerRef.current = null;
  };
  useEffect(() => () => { requestSequenceRef.current += 1; clearWaitingTimer(); abortRef.current?.abort(); }, []);
  const selectConversation = useCallback((id: string) => {
    requestSequenceRef.current += 1;
    clearWaitingTimer();
    abortRef.current?.abort(); abortRef.current = null; setStreaming(false); setWaitingForContent(false); setStreamError(''); setCompletionNotice('');
    setConversationId(id); setLines([]); setToolOutputs([]);
    const url = new URL(location.href);
    if (id) url.searchParams.set('conversation_id', id); else url.searchParams.delete('conversation_id');
    window.history.replaceState({}, '', url);
    try { if (id) sessionStorage.setItem(storedConversationKey, id); else sessionStorage.removeItem(storedConversationKey); } catch { /* in-memory selection remains available */ }
  }, []);
  useEffect(() => {
    if (dataMode !== 'demo') return;
    // Live conversations and their associated actions must not remain visible
    // when the mounted Chat workspace switches into its offline Demo mode.
    selectConversation('');
    setAttachments([]); setUploadError(''); setHistoryOpen(false); setHistoryError('');
    setToolOutputs([]);
    setConversationActionMessage(''); setConversationActionError('');
    setActionMessageId(''); setHistoryBusyId('');
  }, [dataMode, selectConversation]);
  useEffect(() => {
    const syncConversation = () => {
      const id = new URL(location.href).searchParams.get('conversation_id') || '';
      if (id !== conversationId) selectConversation(id);
    };
    window.addEventListener('popstate', syncConversation);
    return () => window.removeEventListener('popstate', syncConversation);
  }, [conversationId, selectConversation]);
  const updateConversation = async (item: Conversation, patch: Record<string, unknown>) => {
    if (!item.id) return;
    setHistoryBusyId(item.id); setHistoryError('');
    try {
      await apiPatch(`/api/conversations/${encodeURIComponent(item.id)}`, patch);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['chat', 'organized-history'] }),
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] }),
      ]);
      if (patch.archived === true && item.id === conversationId) selectConversation('');
    } catch (error) {
      setHistoryError(apiConnectionMessage(error));
    } finally { setHistoryBusyId(''); }
  };
  const deleteConversation = async (item: Conversation) => {
    if (!item.id) return;
    setPendingDeleteId(''); setRowMenuId('');
    setHistoryBusyId(item.id); setHistoryError(''); setConversationActionError(''); setConversationActionMessage('');
    try {
      await apiDelete(`/api/conversations/${encodeURIComponent(item.id)}`);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['chat', 'organized-history'] }),
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] }),
        queryClient.removeQueries({ queryKey: ['chat', 'messages', item.id] }),
      ]);
      if (item.id === conversationId) selectConversation('');
    } catch (error) {
      setHistoryError(apiConnectionMessage(error));
    } finally { setHistoryBusyId(''); }
  };
  const renameConversation = async (item: Conversation, nextTitle?: string) => {
    const title = nextTitle ?? renameValue;
    if (nextTitle === undefined && renamingId !== item.id) {
      setRenamingId(item.id); setRenameValue(item.title || 'Conversation'); setRowMenuId('');
      return;
    }
    if (!title.trim()) { setHistoryError('Conversation title cannot be empty.'); return; }
    setRowMenuId(''); setRenamingId(''); setRenameValue('');
    await updateConversation(item, { title: title.trim() });
  };
  const download = (name: string, content: BlobPart, type: string) => {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = name; anchor.click();
    URL.revokeObjectURL(url);
  };
  const exportConversation = async (id = conversationId) => {
    if (!id) return;
    setConversationActionError(''); setConversationActionMessage('Preparing export…');
    try {
      const response = await apiGet<{ export?: Record<string, unknown> }>(`/api/chat/export?conversation_id=${encodeURIComponent(id)}&write_file=false`);
      if (!response.export || response.export.format !== 'openagent.conversation.v1') throw new Error('The backend returned an unsupported conversation export.');
      const title = String((response.export.conversation as Record<string, unknown> | undefined)?.title || 'conversation').replace(/[^a-z0-9-_]+/gi, '-').replace(/^-|-$/g, '').slice(0, 60) || 'conversation';
      download(`${title}.openagent.json`, JSON.stringify(response.export, null, 2), 'application/json');
      setConversationActionMessage('Conversation export downloaded.');
    } catch (error) { setConversationActionMessage(''); setConversationActionError(apiConnectionMessage(error)); }
  };
  const importConversation = async (file?: File) => {
    if (!file) return;
    setConversationActionError(''); setConversationActionMessage('');
    try {
      if (file.size > 1_048_576) throw new Error('Conversation files must be 1 MB or smaller.');
      const data: unknown = JSON.parse(await fileToText(file));
      if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('Choose a valid OpenAgent conversation export.');
      const root = data as Record<string, unknown>;
      const payload = root.conversation_export && typeof root.conversation_export === 'object' ? root.conversation_export as Record<string, unknown> : root;
      const conversation = payload.conversation;
      const messages = payload.messages;
      if (payload.format !== 'openagent.conversation.v1' || !conversation || typeof conversation !== 'object' || !Array.isArray(messages) || messages.length > 500) throw new Error('This file is not a supported OpenAgent conversation export.');
      if (!messages.every(message => message && typeof message === 'object' && ['user', 'assistant'].includes(String((message as Record<string, unknown>).role)) && typeof (message as Record<string, unknown>).content === 'string')) throw new Error('The export contains an unsupported message.');
      const response = await apiPost<{ imported?: { conversation?: { id?: string } } }>('/api/chat/import', { conversation_export: payload, project: 'default' });
      const importedId = response.imported?.conversation?.id;
      if (typeof importedId !== 'string' || !importedId) throw new Error('The backend did not return the imported conversation ID.');
      await Promise.all([queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] }), queryClient.invalidateQueries({ queryKey: ['chat', 'organized-history'] })]);
      selectConversation(importedId); setConversationActionMessage('Conversation imported.');
    } catch (error) { setConversationActionError(error instanceof SyntaxError ? 'The selected file is not valid JSON.' : apiConnectionMessage(error)); }
  };
  useEffect(() => {
    const onSelectConversation = (event: Event) => {
      const id = (event as CustomEvent<string>).detail;
      if (typeof id === 'string') selectConversation(id);
    };
    window.addEventListener('openagent:select-conversation', onSelectConversation);
    return () => window.removeEventListener('openagent:select-conversation', onSelectConversation);
  }, [selectConversation]);
  const stopStream = () => {
    requestSequenceRef.current += 1;
    clearWaitingTimer();
    abortRef.current?.abort(); abortRef.current = null; setStreaming(false); setWaitingForContent(false); setCompletionNotice('Response cancelled.');
    setLines(current => current.at(-1)?.role === 'assistant' && !current.at(-1)?.text ? current.slice(0, -1) : current);
  };
  const uploadFiles = async (files: FileList | File[]) => {
    const selected = Array.from(files);
    if (!selected.length) return;
    setUploadError('');
    const remaining = Math.max(0, 12 - attachments.length);
    if (selected.length > remaining) setUploadError('A chat can include up to 12 attachments.');
    const accepted = selected.slice(0, remaining);
    if (!accepted.length) return;
    setUploading(true);
    for (const file of accepted) {
      try {
        const data_base64 = await fileToBase64(file);
        const response = await apiPost<{ asset?: Record<string, unknown> }>('/api/media/ingest', {
          filename: file.name, data_base64, project: 'default', source: 'chat',
        });
        const asset = response.asset;
        if (!asset || typeof asset.id !== 'string') throw new Error('The backend did not return an uploaded asset ID.');
        setAttachments(current => [...current, {
          id: asset.id as string,
          name: String(asset.title || file.name),
          media_type: String(asset.media_type ?? 'binary'),
          mime_type: String(asset.mime_type ?? file.type ?? 'application/octet-stream'),
          ...(typeof asset.size_bytes === 'number' ? { size_bytes: asset.size_bytes } : {}),
        }]);
      } catch (error) {
        setUploadError(`${file.name}: ${apiConnectionMessage(error)}`);
      }
    }
    setUploading(false);
  };
  /**
   * Adds a line to the thread without sending it to the model. Tool output is
   * real backend output, so it is shown as its own assistant line rather than
   * folded into a reply the model did not produce.
   */
  const appendToolResult = (title: string, text: string) => {
    setToolOutputs(current => [...current, { id: `tool-${Date.now()}-${current.length}`, title, text }]);
  };
  /** Runs one composer tool against its own backend route. */
  const runTool = async (tool: 'knowledge' | 'research' | 'document') => {
    const input = toolQuery.trim();
    if (!input || toolBusy) return;
    setToolBusy(tool); setToolError('');
    try {
      if (tool === 'knowledge') {
        const found = await apiGet<{ results?: unknown[]; documents?: unknown[] }>(
          `/api/knowledge/search?project=default&q=${encodeURIComponent(input)}&limit=5`);
        const rows = records(found.results ?? found.documents);
        appendToolResult('Knowledge search', rows.length
          ? `Knowledge search for "${input}" returned ${rows.length} indexed ${rows.length === 1 ? 'source' : 'sources'}:\n${rows.map((row, index) => `${index + 1}. ${String(row.title ?? row.id ?? 'Untitled')}${row.snippet ? ` — ${String(row.snippet)}` : ''}`).join('\n')}`
          : `Knowledge search for "${input}" returned no indexed sources.`);
      } else if (tool === 'research') {
        const run = await apiPost<Record<string, unknown>>('/api/research/run', { query: input, project: 'default', allow_external: allowExternal });
        const report = String(run.report ?? (run.session as Record<string, unknown> | undefined)?.report ?? '');
        const sources = records(run.sources);
        if (!report) throw new Error('The research service did not return a report.');
        appendToolResult('Research report', `Research on "${input}":\n${report}${sources.length ? `\n\nSources: ${sources.map(source => String(source.title ?? source.url ?? '')).filter(Boolean).join(', ')}` : ''}`);
        if (report && conversationId) {
          // Keep the record with the conversation it was run for.
          await apiPost('/api/chat/research-record', {
            query: input, report, conversation_id: conversationId, project: 'default',
            sources: sources.map(source => ({ title: String(source.title ?? ''), url: String(source.url ?? '') })),
          });
        }
      } else {
        const made = await apiPost<{ artifact?: Record<string, unknown> }>('/api/chat/artifact', {
          title: input.slice(0, 80), content: lines.map(line => `${line.role === 'user' ? 'You' : 'OpenAgent'}: ${line.text}`).join('\n\n') || input,
          format: 'pdf', project: 'default',
        });
        const artifact = made.artifact ?? {};
        if (typeof artifact.base64 !== 'string' || typeof artifact.filename !== 'string') throw new Error('The backend did not return a downloadable document.');
        const bytes = Uint8Array.from(atob(artifact.base64), char => char.charCodeAt(0));
        download(artifact.filename, bytes, String(artifact.content_type ?? 'application/pdf'));
        appendToolResult('Document', `Downloaded ${artifact.filename}`);
      }
      setToolQuery(''); setActiveTool(''); setToolsOpen(false);
    } catch (error) {
      setToolError(apiConnectionMessage(error));
    } finally {
      setToolBusy('');
    }
  };
  /**
   * Speaks a reply through the backend's own synthesiser, resolving when the
   * audio has finished so a hands-free turn does not listen to itself.
   */
  const speak = async (text: string) => {
    try {
      const spoken = await apiPost<{ script?: Record<string, unknown> }>('/api/voice/tts', { text, project: 'default' });
      const script = spoken.script ?? {};
      // Prefer the browser's installed local voice. The backend's legacy
      // generated WAV may be a tone placeholder, so never play it as speech.
      if ('speechSynthesis' in window) {
        await new Promise<void>(resolve => {
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.lang = 'en-US'; utterance.rate = 1; utterance.pitch = 1;
          utterance.onend = () => resolve(); utterance.onerror = () => resolve();
          window.speechSynthesis.cancel(); window.speechSynthesis.speak(utterance);
        });
        return;
      }
      const assetId = String(script.audio_asset_id ?? '');
      if (!assetId) return;
      const blob = await apiGetBlob(`/api/voice/assets/${encodeURIComponent(assetId)}/content?project=default`);
      const player = new Audio(URL.createObjectURL(blob)); speakerRef.current = player;
      await new Promise<void>(resolve => { player.onended = () => resolve(); player.onerror = () => resolve(); void player.play().catch(() => resolve()); });
    } catch { /* a reply that cannot be spoken is still shown as text */ }
    finally { speakerRef.current = null; }
  };
  /** Opens a voice session and a stream session to push chunks into. */
  const openStreamSession = async () => {
    if (!voiceSessionRef.current) {
      const created = await apiPost<{ session?: Record<string, unknown> }>('/api/voice/sessions', { project: 'default', mode: 'always_listening', always_listening: true, language: 'en' });
      const id = String((created.session ?? {}).id ?? (created.session ?? {}).session_id ?? '');
      if (!id) throw new Error('The backend did not return a voice session.');
      voiceSessionRef.current = id;
    }
    const stream = await apiPost<Record<string, unknown>>('/api/voice/websocket/create', {
      voice_session_id: voiceSessionRef.current, project: 'default', language: 'en',
    });
    const streamId = String(stream.stream_id ?? '');
    if (!streamId) throw new Error('The backend did not return a stream session.');
    return streamId;
  };
  /**
   * Hands-free voice: listen, answer aloud, listen again.
   *
   * The backend buffers streamed chunks and transcribes only when the stream
   * is finished, so it returns no interim text. End of speech is therefore
   * detected here from the microphone's own level, and each finished utterance
   * is transcribed, sent as a message, and the reply spoken before listening
   * resumes. Nothing is shown as text until the backend transcribes it.
   */
  const stopLiveVoice = useCallback(() => {
    liveActiveRef.current = false;
    if (liveFrameRef.current !== null) { cancelAnimationFrame(liveFrameRef.current); liveFrameRef.current = null; }
    try { liveRecorderRef.current?.stop(); } catch { /* already stopped */ }
    liveRecorderRef.current = null;
    liveStreamRef.current?.getTracks().forEach(track => track.stop());
    liveStreamRef.current = null;
    void liveContextRef.current?.close().catch(() => undefined);
    liveContextRef.current = null;
    speakerRef.current?.pause();
    speakerRef.current = null;
    setLiveVoice(false); setLiveStatus('');
  }, []);
  const startLiveVoice = async () => {
    if (!navigator.mediaDevices?.getUserMedia) { setVoiceError('This browser offers no microphone access.'); return; }
    setVoiceError('');
    let media: MediaStream;
    try {
      media = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    } catch (error) {
      setVoiceError(error instanceof Error && error.name === 'NotAllowedError' ? 'Microphone permission was declined.' : `Live voice could not start: ${String(error)}`);
      return;
    }
    liveStreamRef.current = media;
    liveActiveRef.current = true;
    setLiveVoice(true);

    const context = new AudioContext();
    liveContextRef.current = context;
    // Chromium can create the analyser in a suspended state until the first
    // user gesture is fully resolved; resume explicitly so speech detection
    // works in the desktop shell as well as a normal browser tab.
    void context.resume().catch(() => undefined);
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    context.createMediaStreamSource(media).connect(analyser);
    const samples = new Uint8Array(analyser.fftSize);

    // One utterance: record until the level stays low long enough to count as
    // the end of speech, then transcribe, answer, and speak the answer.
    const listenOnce = async () => {
      if (!liveActiveRef.current) return;
      setLiveStatus('listening');
      let streamId = '';
      try { streamId = await openStreamSession(); }
      catch (error) { setVoiceError(apiConnectionMessage(error)); stopLiveVoice(); return; }

      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'].find(type => MediaRecorder.isTypeSupported?.(type));
      const recorder = new MediaRecorder(media, mimeType ? { mimeType } : undefined);
      liveRecorderRef.current = recorder;
      const pending: Promise<unknown>[] = [];
      recorder.ondataavailable = async event => {
        if (!event.data.size || !liveActiveRef.current) return;
        const chunk = fileToBase64(new File([event.data], 'chunk.webm', { type: event.data.type || 'audio/webm' }))
          .then(audio_base64 => apiPost('/api/voice/websocket/chunk', { stream_id: streamId, audio_base64 }))
          .catch(() => undefined);
        pending.push(chunk);
      };

      const finished = new Promise<void>(resolve => { recorder.onstop = () => resolve(); });
      recorder.start(400);

      let spokeAtAll = false;
      let quietSince = 0;
      await new Promise<void>(resolve => {
        const tick = () => {
          if (!liveActiveRef.current) { resolve(); return; }
          analyser.getByteTimeDomainData(samples);
          let sum = 0;
          for (const sample of samples) { const centred = (sample - 128) / 128; sum += centred * centred; }
          const level = Math.sqrt(sum / samples.length);
          const now = performance.now();
          if (level > LIVE_SPEECH_LEVEL) { spokeAtAll = true; quietSince = 0; }
          else if (spokeAtAll) {
            if (!quietSince) quietSince = now;
            else if (now - quietSince > LIVE_SILENCE_MS) { resolve(); return; }
          }
          liveFrameRef.current = requestAnimationFrame(tick);
        };
        liveFrameRef.current = requestAnimationFrame(tick);
      });
      if (liveFrameRef.current !== null) { cancelAnimationFrame(liveFrameRef.current); liveFrameRef.current = null; }

      try { recorder.stop(); } catch { /* already stopped */ }
      await finished;
      await Promise.all(pending);
      liveRecorderRef.current = null;
      if (!liveActiveRef.current) return;
      if (!spokeAtAll) { void listenOnce(); return; }

      setLiveStatus('thinking');
      let transcript = '';
      try {
        const result = await apiPost<Record<string, unknown>>('/api/voice/websocket/finish', { stream_id: streamId });
        transcript = String(result.transcription ?? '').trim();
        if (!transcript) setVoiceError(String(result.message ?? 'No local speech-to-text engine returned text. Install one from the Voice screen.'));
      } catch (error) { setVoiceError(apiConnectionMessage(error)); }
      if (!liveActiveRef.current) return;
      if (!transcript) { void listenOnce(); return; }

      const reply = await submit(undefined, transcript);
      if (!liveActiveRef.current) return;
      if (reply) { setLiveStatus('speaking'); await speak(reply); }
      if (liveActiveRef.current) void listenOnce();
    };
    void listenOnce();
  };
  /**
   * Records from the microphone and transcribes it through the buffered
   * speech-to-text route. The continuous socket transport is not an accepted
   * contract, so capture is explicit: press to start, press again to send.
   */
  const toggleRecording = async () => {
    if (recording) {
      recorderRef.current?.stop();
      return;
    }
    setVoiceError('');
    if (!navigator.mediaDevices?.getUserMedia) { setVoiceError('This browser offers no microphone access.'); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = async () => {
        stream.getTracks().forEach(track => track.stop());
        setRecording(false);
        recorderRef.current = null;
        if (!chunks.length) return;
        setVoiceBusy(true);
        try {
          if (!voiceSessionRef.current) {
            const created = await apiPost<{ session?: Record<string, unknown> }>('/api/voice/sessions', { project: 'default', mode: 'push_to_talk', language: 'en' });
            const id = String((created.session ?? {}).id ?? (created.session ?? {}).session_id ?? '');
            if (!id) throw new Error('The backend did not return a voice session.');
            voiceSessionRef.current = id;
          }
          const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
          const audio_base64 = await fileToBase64(new File([blob], 'speech.webm', { type: blob.type }));
          const result = await apiPost<{ transcription?: Record<string, unknown> }>('/api/voice/stt', {
            session_id: voiceSessionRef.current, audio_base64, filename: 'speech.webm', language: 'en', project: 'default',
          });
          const transcription = result.transcription ?? {};
          const text = String(transcription.transcript ?? transcription.display_text ?? '').trim();
          if (text) {
            setDraft(current => (current ? `${current} ${text}` : text));
          }
          else setVoiceError(String(transcription.stt_message ?? 'No local speech-to-text engine returned text. Install one from the Voice screen.'));
        } catch (error) {
          setVoiceError(apiConnectionMessage(error));
        } finally {
          setVoiceBusy(false);
        }
      };
      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch (error) {
      setVoiceError(error instanceof Error && error.name === 'NotAllowedError' ? 'Microphone permission was declined.' : `Recording could not start: ${String(error)}`);
    }
  };
  useEffect(() => () => { recorderRef.current?.stop(); stopLiveVoice(); }, [stopLiveVoice]);
  useEffect(() => {
    if (!toolsOpen && !modelControlsOpen) return;
    const dismiss = (event: MouseEvent) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      // A click inside either panel, or on the button that opened it, keeps it.
      if (target.closest('#chat-tools-menu, #chat-model-controls, .chat-tools-trigger, .chat-model-trigger')) return;
      setToolsOpen(false); setModelControlsOpen(false);
    };
    document.addEventListener('mousedown', dismiss);
    return () => document.removeEventListener('mousedown', dismiss);
  }, [toolsOpen, modelControlsOpen]);
  useEffect(() => {
    if (!rowMenuId && !actionMessageId) return;
    const dismissActionMenus = (event: MouseEvent) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      // Keep an open menu alive while interacting with its items or its
      // trigger; any other click should close it before the next action.
      if (target.closest('.chat-row-menu, .chat-message-menu, [aria-label^="More actions"], [aria-label^="Actions for"]')) return;
      setRowMenuId('');
      setActionMessageId('');
      setPendingDeleteId('');
      setRenamingId('');
      setRenameValue('');
    };
    document.addEventListener('mousedown', dismissActionMenus);
    return () => document.removeEventListener('mousedown', dismissActionMenus);
  }, [rowMenuId, actionMessageId]);
  useEffect(() => {
    if (!liveVoice) return;
    const dismissLiveVoice = (event: MouseEvent) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest('.chat-live-voice-overlay, [aria-label="Stop live voice"]')) return;
      stopLiveVoice();
    };
    document.addEventListener('mousedown', dismissLiveVoice);
    return () => document.removeEventListener('mousedown', dismissLiveVoice);
  }, [liveVoice, stopLiveVoice]);
  const submit = async (event?: FormEvent, spokenText?: string) => {
    event?.preventDefault(); const content = (spokenText ?? draft).trim(); if (!content || streaming || uploading || dataMode !== 'live') return;
    if (selectedNeedsEgress && !allowExternal) { setStreamError('External provider access is off. Opt in before sending prompt or attachment content to a network provider.'); return; }
    clearWaitingTimer();
    const requestId = ++requestSequenceRef.current;
    const requestConversationId = conversationId;
    const userRowId = `local-${requestId}-user`;
    const assistantRowId = `local-${requestId}-assistant`;
    const isCurrentRequest = () => requestSequenceRef.current === requestId;
    setDraft(''); setStreamError(''); setCompletionNotice(''); setWaitingForContent(false);
    setLines(current => [...current, { id: userRowId, role: 'user', text: content }, { id: assistantRowId, role: 'assistant', text: '' }]);
    setStreaming(true);
    const controller = new AbortController(); abortRef.current = controller;
    let assistantText = '';
    const updateAssistant = (text: string) => setLines(current => {
      return current.map(line => line.id === assistantRowId ? { ...line, text } : line);
    });
    waitingTimerRef.current = window.setTimeout(() => {
      if (isCurrentRequest() && !controller.signal.aborted && !assistantText) setWaitingForContent(true);
    }, 500);
    try {
      await apiStreamPost('/api/chat/stream', { message: content, conversation_id: requestConversationId || undefined, project: 'default', provider: providerId, model: modelId || undefined, allow_external: allowExternal, attachments: attachments.map(({ id, name, media_type, mime_type }) => ({ id, name, media_type, mime_type })) }, (event: StreamEvent) => {
        if (!isCurrentRequest() || controller.signal.aborted) return;
        if (event.type === 'conversation') {
          const conversation = event.conversation as Record<string, unknown> | undefined;
          if (typeof conversation?.id === 'string') {
            setConversationId(conversation.id);
            const url = new URL(location.href); url.searchParams.set('conversation_id', conversation.id); window.history.replaceState({}, '', url);
            try { sessionStorage.setItem(storedConversationKey, conversation.id); } catch { /* in-memory selection remains available */ }
          }
        } else if (event.type === 'chunk' && typeof event.content === 'string' && event.content.length) {
          assistantText += event.content;
          clearWaitingTimer(); setWaitingForContent(false);
          updateAssistant(assistantText);
        }
        else if (event.type === 'message') {
          const message = event.message as Record<string, unknown> | undefined;
          if (typeof message?.content === 'string' && message.content.length) {
            assistantText = message.content;
            clearWaitingTimer(); setWaitingForContent(false);
            updateAssistant(assistantText);
            if (typeof message.id === 'string') setLines(current => current.map(line => line.id === assistantRowId ? { ...line, serverId: message.id as string } : line));
          }
        } else if (event.type === 'done' || event.type === 'terminal') {
          clearWaitingTimer(); setWaitingForContent(false); setStreaming(false);
        }
      }, controller.signal);
      if (!isCurrentRequest() || controller.signal.aborted) return;
      clearWaitingTimer(); setWaitingForContent(false); setStreaming(false); setAttachments([]);
      if (!assistantText) {
        setLines(current => current.filter(line => line.id !== assistantRowId));
        setCompletionNotice('The backend completed without returning a response.');
      }
      // Keep terminal UI responsive; refresh persisted history without delaying completion.
      void Promise.all([
        queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] }),
        queryClient.invalidateQueries({ queryKey: ['chat', 'messages'] }),
      ]);
    } catch (error) {
      if (!isCurrentRequest()) return;
      clearWaitingTimer(); setWaitingForContent(false); setStreaming(false);
      if (controller.signal.aborted) setCompletionNotice('Response cancelled.');
      else { setStreamError(apiConnectionMessage(error)); if (!assistantText) setLines(current => current.filter(line => line.id !== assistantRowId)); }
    } finally {
      if (isCurrentRequest()) {
        clearWaitingTimer();
        setWaitingForContent(false); setStreaming(false);
        if (abortRef.current === controller) abortRef.current = null;
      }
    }
    return assistantText;
  };
  const enter = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void submit();
    }
  };
  const regenerateResponse = (index: number) => {
    const prompt = [...visibleLines.slice(0, index)].reverse().find(item => item.role === 'user')?.text;
    setActionMessageId('');
    if (prompt) void submit(undefined, prompt);
  };
  const searchWeb = async (line: ChatLineWithId) => {
    setActionMessageId(''); setToolError(''); setToolBusy('web');
    try {
      const result = await apiPost<Record<string, unknown>>('/api/research/run', { query: line.text, project: 'default', allow_external: true });
      const report = String(result.report ?? '');
      appendToolResult('Web search', report || 'The research service returned no web result.');
    } catch (error) { setToolError(apiConnectionMessage(error)); }
    finally { setToolBusy(''); }
  };
  return <div className={`split-layout chat-demo-${dataMode} ${inspector ? 'has-inspector' : ''}`}>
    <section className="split-main">
      <header className="split-top"><div className="row chat-toolbar-title"/><div className="row">{dataMode === 'demo' && <span className="tag"><span className="status-dot"/>Offline demo · chat not connected</span>}</div></header>
      {liveVoice && <div className="chat-live-voice-overlay" role="status" aria-live="polite">
        <button type="button" className={`chat-live-voice-orb ${liveStatus || 'listening'}`} aria-label="Stop live voice" onClick={stopLiveVoice}>
          <AudioLines size={30} aria-hidden="true" />
        </button>
        <strong>{liveStatus === 'speaking' ? 'Speaking' : liveStatus === 'thinking' ? 'Thinking' : 'Listening'}</strong>
        <span>Tap to stop</span>
      </div>}
      <div className="split-scroll" ref={chatScrollRef}><div className="chat-thread">
        {messagesQuery.isLoading && <div className="caption" role="status">Loading conversation…</div>}
        {messagesQuery.isError && <div className="card padded small" role="alert">{apiConnectionMessage(messagesQuery.error)}</div>}
        {!visibleLines.length && !toolOutputs.length && !messagesQuery.isLoading ? <div className="empty"><IconSquare><Sparkles size={20}/></IconSquare><h2>Start a conversation</h2><p>{dataMode === 'demo' ? 'Chat is not simulated in offline Demo mode. Switch to Live to use the connected chat service.' : 'Send a message to begin.'}</p></div> : visibleLines.map((line, i) => <article className={`chat-message chat-message-${line.role}`} key={line.id ?? `${conversationId}-${i}-${line.role}`}><span className={`avatar ${line.role === 'assistant' ? 'blue' : ''}`}>{line.role === 'user' ? 'Y' : '◉'}</span><div className="flex1"><div className="message-head"><strong>{line.role === 'user' ? 'You' : 'OpenAgent'}</strong><small>{line.role}</small><time>{line.createdAt ? new Date(line.createdAt).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : ''}</time><button type="button" className="icon-button" aria-label={`Actions for ${line.role} message ${i + 1}`} aria-expanded={actionMessageId === line.id} disabled={!conversationId || streaming} onClick={() => setActionMessageId(current => current === line.id ? '' : line.id || '')}><MoreHorizontal size={15}/></button></div><div className="message-body">{line.text && (line.role === 'assistant' ? renderChatMarkdown(line.text) : <p>{line.text}</p>)}{waitingForContent && streaming && line.role === 'assistant' && i === visibleLines.length - 1 && <div className="chat-generation-indicator" role="status"><span aria-hidden="true"/>Thinking…</div>}</div>{line.role === 'assistant' && <div className="message-actions"><button type="button" className="icon-button" aria-label="Copy message" title="Copy message" onClick={() => line.text && void navigator.clipboard.writeText(line.text)}><Copy size={14}/></button><button type="button" className="icon-button" aria-label="Useful response" disabled title="Feedback is not available yet"><ThumbsUp size={14}/></button><button type="button" className="icon-button" aria-label="More actions" aria-expanded={actionMessageId === line.id} disabled={!line.serverId || streaming} onClick={() => setActionMessageId(current => current === line.id ? '' : line.id || '')}><MoreHorizontal size={14}/></button></div>}{actionMessageId === line.id && <div className="chat-message-menu" role="menu" aria-label={`Actions for ${line.role} message`}><button type="button" role="menuitem" onClick={() => { setActionMessageId(''); void speak(line.text); }}>Read aloud</button><button type="button" role="menuitem" onClick={() => regenerateResponse(i)}>Regenerate response</button><button type="button" role="menuitem" onClick={() => void searchWeb(line)}>Search web</button></div>}</div></article>)}
        {toolOutputs.map(output => <section className="card padded chat-tool-output" key={output.id} aria-label={output.title}><strong>{output.title}</strong><p>{output.text}</p><small>Tool output from this session · not saved as an assistant reply</small></section>)}
        {streamError && <div className="card padded small" role="alert">{streamError}</div>}
        {completionNotice && !streaming && <div className="caption" role="status">{completionNotice}</div>}
      </div></div>
      <form className="chat-composer-outer" onSubmit={event => void submit(event)}>
        <div className="chat-composer">
          <textarea value={draft} onChange={e => setDraft(e.target.value)} onKeyDown={enter} onFocus={() => { setToolsOpen(false); setModelControlsOpen(false); }} placeholder="Message OpenAgent..." aria-label="Message OpenAgent" disabled={streaming}/>
          {(attachments.length > 0 || uploading || uploadError) && <div className="row gap6" style={{ flexWrap: 'wrap', padding: '0 12px 8px' }}>{attachments.map(attachment => <span key={attachment.id} className="tag" data-testid="chat-attachment"><Paperclip size={13}/>{attachment.name}<button type="button" className="icon-button" aria-label={`Remove ${attachment.name}`} onClick={() => setAttachments(current => current.filter(item => item.id !== attachment.id))}><span aria-hidden="true">×</span></button></span>)}{uploading && <span className="caption" role="status">Uploading attachment…</span>}{uploadError && <span className="caption" role="alert">{uploadError}</span>}</div>}
          <div className="row between" style={{ flexWrap: 'wrap' }}>
            <div className="row gap6">
              <input ref={fileInputRef} type="file" multiple aria-label="Choose files to attach" style={{ position: 'absolute', width: 1, height: 1, padding: 0, margin: -1, overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', border: 0 }} onChange={event => { if (event.currentTarget.files) void uploadFiles(event.currentTarget.files); event.currentTarget.value = ''; }} />
              <button type="button" className="icon-button" title="Attach files" aria-label="Attach a file" onClick={() => fileInputRef.current?.click()} disabled={dataMode !== 'live' || streaming || uploading || attachments.length >= 12}><Paperclip size={16}/></button>
              <button type="button" className="btn sm chat-model-trigger" aria-expanded={modelControlsOpen} aria-controls="chat-model-controls" onClick={() => { setModelControlsOpen(value => !value); setToolsOpen(false); }}><Bot size={14}/>{modelButtonLabel}{dataMode !== 'demo' && <ChevronDown size={13}/>}</button>
              <button type="button" className="btn sm chat-tools-trigger" aria-expanded={toolsOpen} aria-controls="chat-tools-menu" onClick={() => { setToolsOpen(value => !value); setModelControlsOpen(false); setToolError(''); }}><Wrench size={14}/>Tools{dataMode !== 'demo' && <ChevronDown size={13}/>}</button>
            </div>
            <div className="row gap6">
              <button type="button" className={`icon-button${recording ? ' is-recording' : ''}`} aria-label={recording ? 'Stop recording and transcribe' : 'Record a spoken message'} aria-pressed={recording} title={dataMode === 'demo' ? 'Voice input requires Live mode' : recording ? 'Stop and transcribe' : 'Record a spoken message'} disabled={dataMode !== 'live' || streaming || voiceBusy} onClick={() => void toggleRecording()}>{recording ? <Square size={15}/> : <Mic size={16}/>}</button>
              {/* Hands-free: listen, answer aloud, listen again, until stopped. */}
              {(!draft.trim() || liveVoice) && <button type="button" className={`icon-button${liveVoice ? ' is-live' : ''}`} aria-label={liveVoice ? 'Stop live voice' : 'Start live voice'} aria-pressed={liveVoice} title={dataMode === 'demo' ? 'Live voice requires Live mode' : liveVoice ? 'Stop live voice' : 'Live voice: listen and answer aloud'} disabled={dataMode !== 'live' || recording || voiceBusy} onClick={() => { if (liveVoice) stopLiveVoice(); else void startLiveVoice(); }}>{liveVoice ? <Square size={15}/> : <AudioLines size={16}/>}</button>}
              {streaming ? <button className="btn sm" type="button" onClick={stopStream}>Cancel</button> : draft.trim() ? <button className="send-button" type="submit" aria-label="Send message" title={dataMode === 'demo' ? 'Chat requires Live mode' : undefined} disabled={dataMode !== 'live' || uploading || (selectedNeedsEgress && !allowExternal)}><Send size={16}/></button> : null}
            </div>
          </div>
          {(voiceBusy || voiceError || recording || liveStatus) && <div className="row gap6" style={{ padding: '0 12px 8px' }}>
            {liveStatus === 'listening' && <span className="caption" role="status">Listening… stop speaking and it will answer.</span>}
            {liveStatus === 'thinking' && <span className="caption" role="status">Transcribing and answering…</span>}
            {liveStatus === 'speaking' && <span className="caption" role="status">Speaking the reply…</span>}
            {recording && <span className="caption" role="status">Recording… press the stop button to transcribe.</span>}
            {voiceBusy && <span className="caption" role="status">Transcribing with a local engine…</span>}
            {voiceError && <span className="caption" role="alert">{voiceError}</span>}
          </div>}
          {toolsOpen && <div className="chat-model-controls chat-tools-menu" id="chat-tools-menu" aria-label="Chat tools">
            <div className="chat-tools-list">
              <button type="button" className="chat-tool-row" disabled={dataMode !== 'live' || uploading || attachments.length >= 12} onClick={() => { fileInputRef.current?.click(); setToolsOpen(false); }}>
                <span className="square-icon sm"><Paperclip size={15}/></span>
                <span><strong>Attach files</strong><small>Uploads to this workspace and sends with the message</small></span>
              </button>
              <button type="button" className="chat-tool-row" aria-expanded={activeTool === 'knowledge'} disabled={dataMode !== 'live'} onClick={() => { setActiveTool(current => current === 'knowledge' ? '' : 'knowledge'); setToolError(''); }}>
                <span className="square-icon sm"><BookOpen size={15}/></span>
                <span><strong>Search knowledge</strong><small>Reads this project&apos;s indexed sources</small></span>
              </button>
              <button type="button" className="chat-tool-row" aria-expanded={activeTool === 'research'} disabled={dataMode !== 'live'} onClick={() => { setActiveTool(current => current === 'research' ? '' : 'research'); setToolError(''); }}>
                <span className="square-icon sm"><Globe2 size={15}/></span>
                <span><strong>Research a question</strong><small>Runs the research service and records its report{allowExternal ? '' : ' · external access off'}</small></span>
              </button>
              <button type="button" className="chat-tool-row" aria-expanded={activeTool === 'document'} disabled={dataMode !== 'live'} onClick={() => { setActiveTool(current => current === 'document' ? '' : 'document'); setToolError(''); }}>
                <span className="square-icon sm"><FileText size={15}/></span>
                <span><strong>Make a document</strong><small>Renders this conversation as a file</small></span>
              </button>
            </div>
            {activeTool && <div className="row gap6" style={{ marginTop: 8 }}>
              {/* Deliberately not a form: this panel sits inside the composer's
                  own form, and a nested form is not a valid form owner, so its
                  submit navigated the page instead of running the tool. */}
              <input className="field flex1" aria-label={activeTool === 'knowledge' ? 'Knowledge search terms' : activeTool === 'research' ? 'Research question' : 'Document title'} placeholder={activeTool === 'knowledge' ? 'What to look for in indexed sources' : activeTool === 'research' ? 'What to investigate' : 'Title for the document'} value={toolQuery} onChange={event => setToolQuery(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); void runTool(activeTool); } }} />
              <button type="button" className="btn sm primary" disabled={!toolQuery.trim() || Boolean(toolBusy)} onClick={() => void runTool(activeTool)}>{toolBusy ? 'Running…' : 'Run'}</button>
            </div>}
            {toolError && <p className="caption" role="alert">{toolError}</p>}
          </div>}
          {modelControlsOpen && <div className="chat-model-controls" id="chat-model-controls" aria-label="Model and privacy settings">
            <div className="chat-model-list" role="listbox" aria-label="Model">
              {modelChoices.map(choice => <button
                key={choice.key}
                type="button"
                role="option"
                aria-selected={choice.key === selectedChoiceKey}
                className={`chat-model-option${choice.key === selectedChoiceKey ? ' is-selected' : ''}`}
                disabled={streaming || !choice.available}
                onClick={() => chooseModel(choice)}
              >
                <span className="square-icon sm">{choice.providerId === 'auto' ? <Sparkles size={14}/> : <Bot size={14}/>}</span>
                <span className="flex1"><strong>{choice.label}</strong><small>{choice.detail}</small></span>
                {!choice.available ? <span className="badge">Unavailable</span> : choice.local ? <span className="badge">Local</span> : <span className="badge">External</span>}
                {choice.key === selectedChoiceKey && <Check size={14}/>}
              </button>)}
              {modelChoices.length === 1 && <p className="caption" role="status">No provider reported an installed model. Auto leaves the choice to the backend.</p>}
            </div>
            <div className="row between" style={{ marginTop: 6 }}>
              <ActionButton nav="models" onClick={() => go('models')} className="btn sm">Manage models</ActionButton>
              {selectedChoice && <span className="caption">{selectedChoice.providerId === 'auto' ? 'Routed by the backend policy' : `${selectedChoice.detail} · ${selectedChoice.local ? 'local' : 'external'}`}</span>}
            </div>
            {providersQuery.isError && <span className="caption" role="alert">{apiConnectionMessage(providersQuery.error)}</span>}
          </div>}
        </div>
      </form>
    </section>
    {!inspector && <button type="button" className="chat-panel-edge-toggle" aria-label="Show conversations panel" title="Show conversations panel" onClick={() => setInspector(true)}><ChevronLeft size={15}/></button>}
    {inspector && <aside className="inspector chat-conversation-panel">
      <div className="section-head"><button type="button" className="chat-panel-inline-toggle" aria-label="Hide conversations panel" title="Hide conversations panel" onClick={() => setInspector(false)}><ChevronRight size={15}/></button><button type="button" className="btn sm primary" onClick={() => { selectConversation(''); setRecentLimit(8); }} disabled={dataMode !== 'live'}><Plus size={14}/>New conversation</button></div>
      <div className="inspector-content">
      {historyOpen && <section className="card chat-sidebar-history" aria-label="Conversation history">

        <div className="row between" style={{ gap: 8, flexWrap: 'wrap' }}>
          <label className="row" style={{ flex: '1 1 220px', minWidth: 0, gap: 8 }}><Search size={15}/><input className="field" aria-label="Search conversation history" placeholder="Search conversations" value={historySearch} onChange={event => setHistorySearch(event.target.value)} /></label>
          <button type="button" className="btn sm" aria-pressed={showArchived} onClick={() => setShowArchived(value => !value)}>{showArchived ? 'Archived' : 'Active'}</button>
        </div>
        {historyError && <p className="caption" role="alert">{historyError}</p>}
        {historyQuery.isLoading ? <p className="caption" role="status">Loading conversation history…</p> : historyQuery.isError ? <p className="caption" role="alert">{apiConnectionMessage(historyQuery.error)}</p> : history.length ? <div className="second-search-results" style={{ maxHeight: 220, overflow: 'auto' }}>{history.map(item => <div key={item.id} className="row" style={{ gap: 4, borderBottom: '1px solid var(--line)', padding: '5px 0' }}>
          <button type="button" className="nav-link truncate" aria-current={conversationId === item.id ? 'true' : undefined} style={{ flex: 1, minWidth: 0 }} onClick={() => { selectConversation(item.id); setHistoryOpen(false); }} title={item.title || item.id}>{item.title || item.id}</button>
          <button type="button" className="icon-button" aria-label={`${item.pinned ? 'Unpin' : 'Pin'} ${item.title || 'conversation'}`} title={item.pinned ? 'Unpin conversation' : 'Pin conversation'} disabled={historyBusyId === item.id} onClick={() => void updateConversation(item, { pinned: !item.pinned })}><Pin size={14}/></button>
          <button type="button" className="icon-button" aria-label={`Rename ${item.title || 'conversation'}`} title="Rename conversation" disabled={historyBusyId === item.id} onClick={() => void renameConversation(item)}><MoreHorizontal size={15}/></button>
          {renamingId === item.id && <span className="chat-inline-rename"><input autoFocus aria-label={`Rename ${item.title || 'conversation'}`} value={renameValue} onChange={event => setRenameValue(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') void renameConversation(item, renameValue); if (event.key === 'Escape') { setRenamingId(''); setRenameValue(''); } }} /><button type="button" className="btn sm primary" onClick={() => void renameConversation(item, renameValue)}>Save</button><button type="button" className="btn sm" onClick={() => { setRenamingId(''); setRenameValue(''); }}>Cancel</button></span>}
          <button type="button" className="icon-button" aria-label={`${item.archived ? 'Restore' : 'Archive'} ${item.title || 'conversation'}`} title={item.archived ? 'Restore conversation' : 'Archive conversation'} disabled={historyBusyId === item.id} onClick={() => void updateConversation(item, { archived: !item.archived })}><Archive size={14}/></button>
          {/* Deleting confirms in place, the same way the recent rows do. */}
          <button type="button" className="icon-button" aria-label={`Delete ${item.title || 'conversation'}`} title="Delete conversation" disabled={historyBusyId === item.id} onClick={() => setPendingDeleteId(item.id)}><Trash2 size={14}/></button>
          {pendingDeleteId === item.id && <span className="chat-row-confirm is-inline" role="group" aria-label={`Confirm deleting ${item.title || 'conversation'}`}>
            <button type="button" className="btn sm is-destructive" disabled={historyBusyId === item.id} onClick={() => void deleteConversation(item)}>Delete</button>
            <button type="button" className="btn sm" onClick={() => setPendingDeleteId('')}>Cancel</button>
          </span>}
        </div>)}</div> : <p className="caption" role="status">{historySearch.trim() ? 'No conversations match this search.' : showArchived ? 'No archived conversations.' : 'No saved conversations yet.'}</p>}
      </section>}
      {dataMode !== 'live' ? <div className="empty"><Sparkles size={20}/><strong>Offline demo</strong><p>Switch to Live mode to use saved conversations.</p></div> : <>
        <div className="chat-recent-list">
          {conversationsQuery.isLoading ? <p className="caption" role="status">Loading conversations…</p>
            : conversationsQuery.isError ? <p className="caption" role="alert">{apiConnectionMessage(conversationsQuery.error)}</p>
              : conversations.length ? <>
                {conversations.slice(0, recentLimit).map(item => <div key={item.id} className={`chat-recent-row${conversationId === item.id ? ' is-active' : ''}`}>
                  <button type="button" className="chat-recent-open" aria-current={conversationId === item.id ? 'true' : undefined} title={item.title || item.id} onClick={() => selectConversation(item.id)}>
                    <strong className="truncate">{item.title || item.id}</strong>
                    {item.updated_at ? <small>{relativeTimestamp(item.updated_at)}</small> : null}
                  </button>
                  {/* Pinning shows its result by filling the icon, not by
                      reporting it in a message. */}
                  <button type="button" className={`icon-button${item.pinned ? ' is-pinned' : ''}`} aria-label={`${item.pinned ? 'Unpin' : 'Pin'} ${item.title || 'conversation'}`} aria-pressed={Boolean(item.pinned)} title={item.pinned ? 'Unpin conversation' : 'Pin conversation'} disabled={historyBusyId === item.id} onClick={() => void updateConversation(item, { pinned: !item.pinned })}><Pin size={13}/></button>
                  <button type="button" className="icon-button" aria-label={`More actions for ${item.title || 'conversation'}`} aria-expanded={rowMenuId === item.id} title="More actions" disabled={historyBusyId === item.id} onClick={() => { setRowMenuId(current => current === item.id ? '' : item.id); setPendingDeleteId(''); }}><MoreHorizontal size={14}/></button>
                  {rowMenuId === item.id && <div className="chat-row-menu" role="menu" aria-label={`Actions for ${item.title || 'conversation'}`}>
                    <button type="button" role="menuitem" onClick={() => { setRowMenuId(''); void exportConversation(item.id); }}>Export JSON</button>
                    <button type="button" role="menuitem" onClick={() => void renameConversation(item)}>Rename</button>
                    <button type="button" role="menuitem" onClick={() => { setRowMenuId(''); void updateConversation(item, { archived: true }); }}>Archive</button>
                    <button type="button" role="menuitem" className="is-destructive" onClick={() => { setRowMenuId(''); setPendingDeleteId(item.id); }}>Delete</button>
                  </div>}
                  {pendingDeleteId === item.id && <div className="chat-row-confirm" role="group" aria-label={`Confirm deleting ${item.title || 'conversation'}`}>
                    <span>Delete permanently?</span>
                    <button type="button" className="btn sm is-destructive" disabled={historyBusyId === item.id} onClick={() => void deleteConversation(item)}>Delete</button>
                    <button type="button" className="btn sm" onClick={() => setPendingDeleteId('')}>Cancel</button>
                  </div>}
                  {renamingId === item.id && <span className="chat-inline-rename"><input autoFocus aria-label={`Rename ${item.title || 'conversation'}`} value={renameValue} onChange={event => setRenameValue(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') void renameConversation(item, renameValue); if (event.key === 'Escape') { setRenamingId(''); setRenameValue(''); } }} /><button type="button" className="btn sm primary" onClick={() => void renameConversation(item, renameValue)}>Save</button><button type="button" className="btn sm" onClick={() => { setRenamingId(''); setRenameValue(''); }}>Cancel</button></span>}
                </div>)}
                {conversations.length > recentLimit && <button type="button" className="btn sm chat-recent-more" onClick={() => setRecentLimit(conversations.length)}>Load all {conversations.length} conversations</button>}
              </> : <div className="empty"><Sparkles size={20}/><strong>No conversations yet</strong><p>Send a message to start one.</p></div>}
          {/* Always offered: archived conversations are reachable even when
              nothing is active, which is also what the history panel searches. */}
          <button type="button" className="btn sm chat-recent-more" aria-expanded={historyOpen && showArchived} onClick={() => { setShowArchived(true); setHistoryOpen(true); setHistoryError(''); }}>Archived chats</button>
        </div>
        <div className="chat-conversation-actions">
          <input ref={importInputRef} type="file" accept=".json,application/json" aria-label="Choose conversation export" style={{ position: 'absolute', width: 1, height: 1, padding: 0, margin: -1, overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', border: 0 }} onChange={event => { const file = event.currentTarget.files?.[0]; if (file) void importConversation(file); event.currentTarget.value = ''; }} />
          <button type="button" className="btn sm" onClick={() => importInputRef.current?.click()}>Import conversation…</button>
        </div>
        {conversationActionMessage && <p className="caption" role="status">{conversationActionMessage}</p>}
        {conversationActionError && <p className="caption" role="alert">{conversationActionError}</p>}
      </>}
    </div></aside>}
  </div>;
}
