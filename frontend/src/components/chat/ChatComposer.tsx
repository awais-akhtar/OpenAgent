import { useEffect, useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react';
import { Button } from '@arco-design/web-react';
import { ArrowUp, Paperclip, Square, X } from 'lucide-react';

type ChatItem = Record<string, unknown>;
const label = (value: unknown, fallback = '') => typeof value === 'string' ? value : fallback;

type ChatComposerProps = {
  draft: string;
  project: string;
  setDraft: (value: string) => void;
  onSubmit: () => void;
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  onFiles: (event: ChangeEvent<HTMLInputElement>) => void;
  onDropFiles: (files: File[]) => void;
  attachments: ChatItem[];
  removeAttachment: (index: number) => void;
  providers: ChatItem[];
  provider: string;
  setProvider: (value: string) => void;
  model: string;
  setModel: (value: string) => void;
  models: string[];
  agents: ChatItem[];
  agent: string;
  setAgent: (value: string) => void;
  agentsError: string;
  agentsLoading: boolean;
  onRetryAgents: () => void;
  providersError: string;
  providersLoading: boolean;
  onRetryProviders: () => void;
  localOnly: boolean;
  external: boolean;
  setExternal: (value: boolean) => void;
  sending: boolean;
  onStop: () => void;
};

export function ChatComposer(props: ChatComposerProps) {
  const textarea = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const hasConversationContent = props.draft.length > 0 || props.attachments.length > 0;
  const [draggingFiles, setDraggingFiles] = useState(false);
  const dragDepth = useRef(0);
  const privacyText = props.localOnly ? 'External blocked' : props.external ? 'External access on' : 'External access off';
  const privacyHelp = props.localOnly
    ? 'The local-only policy blocks external model providers for every message.'
    : props.external
      ? 'External model providers may receive this message.'
      : 'External model providers are not approved for this message.';

  useEffect(() => {
    const element = textarea.current;
    if (!element) return;
    element.style.height = '0px';
    element.style.height = `${Math.min(element.scrollHeight, 240)}px`;
  }, [props.draft]);

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter' && !event.nativeEvent.isComposing) {
      event.preventDefault();
      props.onSubmit();
      return;
    }
    props.onKeyDown(event);
  };

  return <div className={`oa-chat-composer ${hasConversationContent ? 'has-draft' : ''} ${draggingFiles ? 'is-drop-target' : ''}`}
    onDragEnter={event => { if (!Array.from(event.dataTransfer.types).includes('Files')) return; event.preventDefault(); dragDepth.current += 1; setDraggingFiles(true); }}
    onDragOver={event => { if (Array.from(event.dataTransfer.types).includes('Files')) event.preventDefault(); }}
    onDragLeave={event => { if (!Array.from(event.dataTransfer.types).includes('Files')) return; event.preventDefault(); dragDepth.current = Math.max(0, dragDepth.current - 1); if (!dragDepth.current) setDraggingFiles(false); }}
    onDrop={event => { if (!Array.from(event.dataTransfer.types).includes('Files')) return; event.preventDefault(); dragDepth.current = 0; setDraggingFiles(false); props.onDropFiles(Array.from(event.dataTransfer.files)); }}>
    {draggingFiles && <div className="oa-chat-drop-hint" aria-hidden="true">Drop files to attach</div>}
    <div className="oa-chat-composer-main">
      <textarea ref={textarea} value={props.draft} onChange={event => props.setDraft(event.currentTarget.value)} onKeyDown={onKeyDown} placeholder="Ask anything…" rows={1} aria-label="Message" />
      {!!props.attachments.length && <div className="oa-chat-attachments" aria-label="Uploaded attachments">{props.attachments.map((item, index) => <span className="oa-chat-attachment" key={`${label(item.name)}-${index}`}><Paperclip size={13}/>{label(item.name, 'Attachment')}<button type="button" aria-label={`Remove ${label(item.name, 'attachment')}`} onClick={() => props.removeAttachment(index)}><X size={13}/></button></span>)}</div>}
      <div className="oa-chat-composer-toolbar">
        <div className="oa-chat-composer-selectors">
          <label className="oa-chat-select oa-chat-agent-select"><span className="sr-only">Assistant</span><select value={props.agent} onChange={event => props.setAgent(event.currentTarget.value)} aria-label="Assistant"><option value="">Chat</option>{props.agents.map(item => <option key={String(item.id)} value={String(item.id)}>{label(item.name, label(item.id))}</option>)}</select></label>
          <span className="oa-chat-select-divider" aria-hidden="true" />
          {props.agent ? <span className="oa-chat-mode-label">Checked run</span> : <>
            <label className="oa-chat-select"><span className="sr-only">Provider</span><select value={props.provider} onChange={event => { props.setProvider(event.currentTarget.value); props.setModel(''); }} aria-label="Chat provider"><option value="auto">Auto</option>{props.providers.filter(item => item.id !== 'auto').map(item => <option key={String(item.id)} value={String(item.id)} disabled={item.available === false}>{label(item.label, label(item.id))}{item.available === false ? ' · setup needed' : ''}</option>)}</select></label>
            <span className="oa-chat-select-divider" aria-hidden="true" />
            <label className="oa-chat-select"><span className="sr-only">Model</span><select value={props.model} onChange={event => props.setModel(event.currentTarget.value)} aria-label="Chat model"><option value="">Default model</option>{props.models.map(model => <option key={model} value={model}>{model}</option>)}</select></label>
          </>}
        </div>
        <div className="oa-chat-composer-actions">
          <label className="oa-chat-privacy" title={privacyHelp}><input type="checkbox" checked={props.external} disabled={props.localOnly} onChange={event => props.setExternal(event.currentTarget.checked)} aria-label="Allow external network access for this message"/><span>{privacyText}</span></label>
          <input ref={fileInput} type="file" multiple hidden onChange={props.onFiles} aria-label="Upload attachments" />
          <button type="button" className="oa-chat-icon-action" onClick={() => fileInput.current?.click()} aria-label="Attach files" title="Attach files"><Paperclip size={17}/></button>
          {props.sending ? <Button type="primary" shape="circle" className="oa-chat-send stop" onClick={props.onStop} aria-label="Stop generation" title="Stop generation"><Square size={14} fill="currentColor"/></Button> : <Button type="primary" shape="circle" className="oa-chat-send" onClick={props.onSubmit} disabled={!props.draft.trim()} aria-label="Send message" title="Send message"><ArrowUp size={18}/></Button>}
        </div>
      </div>
      {(props.agentsError || (!props.agent && props.providersError)) && <div className="oa-chat-catalog-errors">
        {props.agentsError && <p role="status"><span>Assistant options unavailable.</span><button type="button" aria-label="Retry assistant options" onClick={props.onRetryAgents} disabled={props.agentsLoading}>{props.agentsLoading ? 'Retrying…' : 'Retry'}</button></p>}
        {!props.agent && props.providersError && <p role="status"><span>Model options unavailable.</span><button type="button" aria-label="Retry model options" onClick={props.onRetryProviders} disabled={props.providersLoading}>{props.providersLoading ? 'Retrying…' : 'Retry'}</button></p>}
      </div>}
    </div>
    <p className="oa-chat-composer-hint"><span>{props.project === 'default' ? 'Personal workspace' : props.project}</span><span>Shift + Enter for a new line · Ctrl/⌘ + Enter to send</span></p>
  </div>;
}
