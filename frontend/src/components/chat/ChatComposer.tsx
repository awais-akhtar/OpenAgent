import { useEffect, useRef, type ChangeEvent, type KeyboardEvent } from 'react';
import { Button } from '@arco-design/web-react';
import { ArrowUp, FolderOpen, Paperclip, Square, X } from 'lucide-react';

type ChatItem = Record<string, unknown>;
const label = (value: unknown, fallback = '') => typeof value === 'string' ? value : fallback;

type ChatComposerProps = {
  draft: string;
  project: string;
  setDraft: (value: string) => void;
  onSubmit: () => void;
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  onFiles: (event: ChangeEvent<HTMLInputElement>) => void;
  attachments: ChatItem[];
  removeAttachment: (index: number) => void;
  providers: ChatItem[];
  provider: string;
  setProvider: (value: string) => void;
  model: string;
  setModel: (value: string) => void;
  models: string[];
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

  return <div className={`oa-chat-composer ${hasConversationContent ? 'has-draft' : ''}`}>
    <div className="oa-chat-composer-main">
      <textarea ref={textarea} value={props.draft} onChange={event => props.setDraft(event.currentTarget.value)} onKeyDown={onKeyDown} placeholder="Ask anything…" rows={1} aria-label="Message" />
      {!!props.attachments.length && <div className="oa-chat-attachments" aria-label="Uploaded attachments">{props.attachments.map((item, index) => <span className="oa-chat-attachment" key={`${label(item.name)}-${index}`}><Paperclip size={13}/>{label(item.name, 'Attachment')}<button type="button" aria-label={`Remove ${label(item.name, 'attachment')}`} onClick={() => props.removeAttachment(index)}><X size={13}/></button></span>)}</div>}
      <div className="oa-chat-composer-toolbar">
        <div className="oa-chat-composer-selectors">
          <label className="oa-chat-select"><span className="sr-only">Provider</span><select value={props.provider} onChange={event => props.setProvider(event.currentTarget.value)} aria-label="Chat provider"><option value="auto">Auto</option>{props.providers.map(item => <option key={String(item.id)} value={String(item.id)}>{label(item.label, label(item.id))}{item.available === false ? ' · setup needed' : ''}</option>)}</select></label>
          <span className="oa-chat-select-divider" aria-hidden="true" />
          <label className="oa-chat-select"><span className="sr-only">Model</span><select value={props.model} onChange={event => props.setModel(event.currentTarget.value)} aria-label="Chat model"><option value="">Default model</option>{props.models.map(model => <option key={model} value={model}>{model}</option>)}</select></label>
        </div>
        <div className="oa-chat-composer-actions">
          <label className="oa-chat-privacy" title={privacyHelp}><input type="checkbox" checked={props.external} disabled={props.localOnly} onChange={event => props.setExternal(event.currentTarget.checked)} aria-label="Allow external network access for this message"/><span>{privacyText}</span></label>
          <input ref={fileInput} type="file" multiple hidden onChange={props.onFiles} aria-label="Upload attachments" />
          <button type="button" className="oa-chat-icon-action" onClick={() => fileInput.current?.click()} aria-label="Attach files" title="Attach files"><Paperclip size={17}/></button>
          {props.sending ? <Button type="primary" shape="circle" className="oa-chat-send stop" onClick={props.onStop} aria-label="Stop generation" title="Stop generation"><Square size={14} fill="currentColor"/></Button> : <Button type="primary" shape="circle" className="oa-chat-send" onClick={props.onSubmit} disabled={!props.draft.trim()} aria-label="Send message" title="Send message"><ArrowUp size={18}/></Button>}
        </div>
      </div>
    </div>
    <p className="oa-chat-composer-hint"><span><FolderOpen size={12}/>{props.project === 'default' ? 'Default project' : props.project}</span><span>Enter to send · Shift + Enter</span></p>
  </div>;
}
