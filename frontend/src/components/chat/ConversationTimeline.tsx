import { useEffect, useRef, useState } from 'react';
import { Check, Copy, X } from 'lucide-react';
import { MessageContent } from './MessageContent';

type ChatItem = Record<string, unknown>;
const value = (item: ChatItem, key: string, fallback = '') => typeof item[key] === 'string' ? item[key] as string : fallback;

export function ConversationTimeline({ messages, sending, replay, onCloseReplay }: { messages: ChatItem[]; sending: boolean; replay: ChatItem | null; onCloseReplay: () => void }) {
  const viewport = useRef<HTMLDivElement>(null);
  const tail = useRef<HTMLDivElement>(null);
  const shouldStick = useRef(true);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const roleLabel = (message: ChatItem) => message.role === 'user' ? 'You' : message.role === 'interrupted' ? 'OpenAgent · unfinished' : 'OpenAgent';

  useEffect(() => {
    if (!shouldStick.current) return;
    const element = viewport.current;
    if (!element) return;
    // Stream chunks update the message repeatedly; instant scrolling avoids
    // restarting a smooth-scroll animation for every chunk.
    const frame = window.requestAnimationFrame(() => { element.scrollTop = element.scrollHeight; });
    return () => window.cancelAnimationFrame(frame);
  }, [messages]);
  const handleScroll = () => {
    const element = viewport.current;
    if (element) {
      const nearBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 120;
      shouldStick.current = nearBottom;
      setShowJumpToLatest(!nearBottom);
    }
  };
  const jumpToLatest = () => {
    shouldStick.current = true;
    setShowJumpToLatest(false);
    const element = viewport.current;
    if (element) element.scrollTop = element.scrollHeight;
  };
  const copy = (content: string, button: HTMLButtonElement) => {
    void navigator.clipboard?.writeText(content).then(() => {
      button.dataset.copied = 'true';
      window.setTimeout(() => { delete button.dataset.copied; }, 1500);
    });
  };

  return <div ref={viewport} className={`oa-chat-timeline ${messages.length ? 'has-messages' : 'is-empty'}`} onScroll={handleScroll} aria-label="Conversation messages">
    {!messages.length ? <div className="oa-chat-welcome"><h1>What would you like to work on?</h1></div> : <div className="oa-chat-reading-column" aria-live={sending ? 'off' : 'polite'} aria-relevant="additions text">
      {messages.map((item, index) => {
        const content = value(item, 'content', value(item, 'message'));
        const user = item.role === 'user';
        const unfinished = item.role === 'interrupted';
        const isLive = item.role === 'streaming';
        return <article key={`${String(item.id ?? item.role)}-${index}`} className={`oa-chat-message ${user ? 'user' : 'assistant'} ${unfinished ? 'unfinished' : ''} ${isLive ? 'streaming' : ''}`}>
          {!user && <span className="oa-chat-avatar" aria-hidden="true">O</span>}
          <div className="oa-chat-message-body"><div className="oa-chat-message-meta"><span>{roleLabel(item)}</span>{(item.created_at || isLive || unfinished) && <small>{isLive ? 'Generating' : unfinished ? 'Stopped before completion' : value(item, 'created_at')}</small>}</div>
            <div className="oa-chat-message-content"><MessageContent content={content}/></div>
            {!user && content && !isLive && <div className="oa-chat-message-actions"><button type="button" onClick={event => copy(content, event.currentTarget)} aria-label="Copy response" title="Copy response"><Copy size={14}/><Check size={14}/></button></div>}
          </div>
        </article>;
      })}
      {sending && <span className="sr-only" role="status">Assistant is generating a response</span>}
      <div ref={tail}/>
    </div>}
    {messages.length > 0 && showJumpToLatest && <button type="button" className="oa-chat-jump-latest" onClick={jumpToLatest}>Jump to latest</button>}
    {replay && <aside className="oa-chat-replay" aria-label="Saved conversation replay"><div className="oa-chat-replay-head"><strong>Saved replay · {value(replay, 'message_count', '—')} messages</strong><button type="button" onClick={onCloseReplay} aria-label="Close replay"><X size={16}/></button></div>{Array.isArray(replay.timeline) && replay.timeline.map((event, index) => { const row = event as ChatItem; return <p key={`${value(row, 'created_at')}-${index}`}><small>{value(row, 'role')} · {value(row, 'created_at')}</small><br/>{value(row, 'content')}</p>; })}</aside>}
  </div>;
}
