import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const get = vi.fn();
const post = vi.fn();
const postEvents = vi.fn();
vi.mock('../../lib/react/api', () => ({
  api: { get, post },
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
  records: (value: unknown, key?: string) => {
    const rows = Array.isArray(value) ? value : key && value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : [];
    return Array.isArray(rows) ? rows.filter(row => row && typeof row === 'object') : [];
  },
}));
vi.mock('../../lib/react/stream', () => ({ postEvents }));

describe('ChatPage conversation experience', () => {
  let host: HTMLDivElement;
  let root: Root;
  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    get.mockReset(); post.mockReset(); postEvents.mockReset();
    localStorage.setItem('oa-privacy-settings', JSON.stringify({ localOnly: true }));
    history.replaceState({}, '', '/?view=chat');
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/conversations/organized')) return { conversations: [] };
      if (path === '/api/models') return { providers: [{ id: 'local-provider', label: 'Local model', default_model: 'small', installed_models: ['small', 'large'] }] };
      if (path === '/api/agents') return { agents: [{ id: 'executive', name: 'Executive Agent', description: 'Plans work using local context.' }, { id: 'knowledge', name: 'Knowledge Agent', description: 'Searches local memory.' }] };
      if (path.startsWith('/api/conversation/messages')) return { messages: [{ id: 'saved-1', role: 'assistant', content: path.includes('target-conversation') ? 'Target conversation message' : path.includes('conversation-1') ? 'A real streamed response.' : 'Saved assistant response', created_at: '2026-10-14T10:24:00Z' }] };
      return {};
    });
    post.mockResolvedValue({});
    postEvents.mockImplementation(async function* (_path: string, _payload: unknown, _signal: AbortSignal) {
      yield { event: 'message', data: JSON.stringify({ type: 'conversation', conversation: { id: 'conversation-1' } }) };
      yield { event: 'message', data: JSON.stringify({ type: 'chunk', content: 'A real streamed response.' }) };
    });
    HTMLElement.prototype.scrollIntoView = vi.fn();
  });
  afterEach(async () => {
    await act(async () => root.unmount());
    host.remove(); localStorage.clear(); history.replaceState({}, '', '/');
  });

  it('sends the existing project-scoped stream contract and preserves the local-only privacy gate', async () => {
    const { ChatPage } = await import('../../pages/ChatPage');
    await act(async () => { root.render(createElement(ChatPage, { project: 'project-one' })); });
    expect(host.querySelector('aside[aria-label="Conversation history"]')).toBeNull();
    expect(host.querySelector('textarea[aria-label="Message"]')).toBeTruthy();
    const external = host.querySelector<HTMLInputElement>('input[aria-label="Allow external network access for this message"]');
    expect(external?.checked).toBe(false);
    expect(external?.disabled).toBe(true);
    const textarea = host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]');
    await act(async () => {
      if (textarea) {
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
        setter?.call(textarea, 'Explain this task');
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
    const send = host.querySelector<HTMLButtonElement>('button[aria-label="Send message"]');
    await act(async () => { send?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(postEvents).toHaveBeenCalledWith('/api/chat/stream', expect.objectContaining({
      message: 'Explain this task', project: 'project-one', provider: 'auto', model: null,
      allow_external: false, conversation_id: null, attachments: [],
    }), expect.any(AbortSignal));
    expect(host.textContent).toContain('A real streamed response.');
  });

  it('provides a real context inspector and opens it as a panel on narrow screens', async () => {
    const originalWidth = window.innerWidth;
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 390 });
    const { ChatPage } = await import('../../pages/ChatPage');
    await act(async () => { root.render(createElement(ChatPage, { project: 'project-one' })); });
    expect(host.querySelector('.oa-chat-inspector')).toBeNull();
    await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Show context panel"]')?.click(); });
    expect(host.querySelector('.oa-chat-inspector')).toBeTruthy();
    expect(host.querySelector('[role="tab"][aria-selected="true"]')?.textContent).toContain('Preview');
    expect(host.textContent).toContain('project-one');
    await act(async () => { host.querySelector<HTMLButtonElement>('[role="tab"]:nth-child(3)')?.click(); });
    expect(host.querySelector('.oa-chat-inspector-body')?.textContent).toContain('project-one');
    await act(async () => { host.querySelector<HTMLButtonElement>('[role="tab"]:nth-child(2)')?.click(); });
    expect(host.querySelector('.oa-chat-inspector-body')?.textContent).toContain('No response to preview');
    await act(async () => { host.querySelector<HTMLButtonElement>('[role="tab"]:nth-child(1)')?.click(); });
    expect(host.querySelector('.oa-chat-inspector-body')?.textContent).toContain('No files attached');
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: originalWidth });
  });

  it('accepts a one-time Home prompt handoff without placing prompt content in the route', async () => {
    sessionStorage.setItem('oa-chat-handoff:project-one', 'Draft from Home');
    const { ChatPage } = await import('../../pages/ChatPage');
    await act(async () => { root.render(createElement(ChatPage, { project: 'project-one' })); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]')?.value).toBe('Draft from Home');
    expect(sessionStorage.getItem('oa-chat-handoff:project-one')).toBeNull();
    expect(location.search).not.toContain('Draft%20from%20Home');
  });

  it('selects a provider in the composer and sends that provider through the normal chat stream', async () => {
    const { ChatPage } = await import('../../pages/ChatPage');
    await act(async () => { root.render(createElement(ChatPage, { project: 'project-one' })); });
    const provider = host.querySelector<HTMLSelectElement>('select[aria-label="Chat provider"]');
    await act(async () => { if (provider) { provider.value = 'local-provider'; provider.dispatchEvent(new Event('change', { bubbles: true })); } });
    expect(provider?.value).toBe('local-provider');
    const textarea = host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]');
    await act(async () => {
      if (textarea) {
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
        setter?.call(textarea, 'Use this provider');
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
    await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Send message"]')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(postEvents).toHaveBeenCalledWith('/api/chat/stream', expect.objectContaining({ provider: 'local-provider', model: null }), expect.any(AbortSignal));
  });

  it('offers a compact assistant selector and runs the selected assistant through the checked agent API', async () => {
    post.mockResolvedValueOnce({ run: { response: 'A checked local plan.' } });
    const { ChatPage } = await import('../../pages/ChatPage');
    await act(async () => { root.render(createElement(ChatPage, { project: 'project-one' })); });
    expect(host.querySelector('.oa-chat-home-agent-grid')).toBeNull();
    const assistant = host.querySelector<HTMLSelectElement>('select[aria-label="Assistant"]');
    await act(async () => { if (assistant) { assistant.value = 'executive'; assistant.dispatchEvent(new Event('change', { bubbles: true })); } });
    expect(assistant?.value).toBe('executive');
    const textarea = host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]');
    await act(async () => {
      if (textarea) {
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
        setter?.call(textarea, 'Plan this project');
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
    await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Send message"]')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(post).toHaveBeenCalledWith('/api/agents/run-checked', { agent: 'executive', goal: 'Plan this project', project: 'project-one', require_approval: false, verify: true }, expect.any(AbortSignal));
    expect(host.textContent).toContain('A checked local plan.');
    expect(postEvents).not.toHaveBeenCalled();
  });

  it('shows and independently retries model and assistant catalog failures', async () => {
    let modelAttempts = 0;
    let agentAttempts = 0;
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/conversations/organized')) return { conversations: [] };
      if (path === '/api/models') {
        if (modelAttempts++ === 0) throw new Error('Models offline');
        return { providers: [{ id: 'local-provider', label: 'Local model' }] };
      }
      if (path === '/api/agents') {
        if (agentAttempts++ === 0) throw new Error('Agents offline');
        return { agents: [{ id: 'executive', name: 'Executive Agent' }] };
      }
      return {};
    });
    const { ChatPage } = await import('../../pages/ChatPage');
    await act(async () => { root.render(createElement(ChatPage, { project: 'project-one' })); });
    expect(host.textContent).toContain('Model options unavailable.');
    expect(host.textContent).toContain('Assistant options unavailable.');

    await act(async () => {
      host.querySelector<HTMLButtonElement>('button[aria-label="Retry model options"]')?.click();
      host.querySelector<HTMLButtonElement>('button[aria-label="Retry assistant options"]')?.click();
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    expect(host.textContent).not.toContain('Model options unavailable.');
    expect(host.textContent).not.toContain('Assistant options unavailable.');
    expect(host.querySelector('select[aria-label="Chat provider"] option[value="local-provider"]')).toBeTruthy();
    expect(host.querySelector('select[aria-label="Assistant"] option[value="executive"]')).toBeTruthy();
  });

  it('stops a checked assistant run and restores its prompt without leaving an optimistic message', async () => {
    let runSignal: AbortSignal | undefined;
    post.mockImplementationOnce((_path: string, _body: unknown, signal?: AbortSignal) => {
      runSignal = signal;
      return new Promise((_resolve, reject) => signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true }));
    });
    const { ChatPage } = await import('../../pages/ChatPage');
    await act(async () => { root.render(createElement(ChatPage, { project: 'project-one' })); });
    const assistant = host.querySelector<HTMLSelectElement>('select[aria-label="Assistant"]');
    await act(async () => { if (assistant) { assistant.value = 'executive'; assistant.dispatchEvent(new Event('change', { bubbles: true })); } });
    const textarea = host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]');
    await act(async () => {
      if (textarea) {
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
        setter?.call(textarea, 'Stop this checked run');
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
    await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Send message"]')?.click(); });
    expect(runSignal).toBeDefined();
    await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Stop generation"]')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(runSignal?.aborted).toBe(true);
    expect(host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]')?.value).toBe('Stop this checked run');
    expect(host.querySelector('.oa-chat-message.user')).toBeNull();
    expect(host.textContent).toContain('Agent run stopped; your prompt was restored.');
  });

  it('restores text when the stream fails before the server acknowledges the message', async () => {
    postEvents.mockImplementationOnce(async function* () { throw new Error('Connection refused'); });
    const { ChatPage } = await import('../../pages/ChatPage');
    await act(async () => { root.render(createElement(ChatPage, { project: 'project-one' })); });
    const textarea = host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]');
    await act(async () => {
      if (textarea) {
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
        setter?.call(textarea, 'Keep this prompt');
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
    await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Send message"]')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]')?.value).toBe('Keep this prompt');
    expect(host.textContent).toContain('Connection refused');
    expect(host.querySelector('.oa-chat-message.user')).toBeNull();
  });

  it('offers jump-to-latest when the reader scrolls away and jumps instantly on activation', async () => {
    const { ConversationTimeline } = await import('../../components/chat/ConversationTimeline');
    await act(async () => { root.render(createElement(ConversationTimeline, { messages: [{ role: 'assistant', content: 'Answer' }], sending: true, replay: null, onCloseReplay: vi.fn() })); });
    const viewport = host.querySelector<HTMLElement>('.oa-chat-timeline');
    expect(viewport).toBeTruthy();
    if (!viewport) return;
    Object.defineProperty(viewport, 'scrollHeight', { configurable: true, value: 1000 });
    Object.defineProperty(viewport, 'clientHeight', { configurable: true, value: 100 });
    viewport.scrollTop = 0;
    await act(async () => { viewport.dispatchEvent(new Event('scroll', { bubbles: true })); });
    const jump = host.querySelector<HTMLButtonElement>('.oa-chat-jump-latest');
    expect(jump?.textContent).toBe('Jump to latest');
    await act(async () => { jump?.click(); });
    expect(viewport.scrollTop).toBe(1000);
    expect(host.querySelector('.oa-chat-jump-latest')).toBeNull();
  });

  it('accepts dropped files in the composer and uploads them within the active project', async () => {
    post.mockResolvedValueOnce({ attachment: { id: 'attachment-1', name: 'notes.txt' } });
    const { ChatPage } = await import('../../pages/ChatPage');
    await act(async () => { root.render(createElement(ChatPage, { project: 'project-one' })); });
    const composer = host.querySelector<HTMLElement>('.oa-chat-composer');
    expect(composer).toBeTruthy();
    if (!composer) return;
    const file = new File(['hello'], 'notes.txt', { type: 'text/plain' });
    const transfer = { types: ['Files'], files: [file] };
    const dragEnter = new Event('dragenter', { bubbles: true, cancelable: true });
    Object.defineProperty(dragEnter, 'dataTransfer', { value: transfer });
    await act(async () => { composer.dispatchEvent(dragEnter); });
    expect(host.textContent).toContain('Drop files to attach');
    const drop = new Event('drop', { bubbles: true, cancelable: true });
    Object.defineProperty(drop, 'dataTransfer', { value: transfer });
    await act(async () => { composer.dispatchEvent(drop); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(post).toHaveBeenCalledWith('/api/knowledge/upload', expect.any(FormData));
    const form = post.mock.calls[0][1] as FormData;
    expect(form.get('project')).toBe('project-one');
    expect(form.get('file')).toBeInstanceOf(File);
    expect(host.textContent).toContain('notes.txt');
    expect(host.textContent).not.toContain('Drop files to attach');
  });

  it('loads a selected conversation from the shell query parameter', async () => {
    history.replaceState({}, '', '/?view=chat&conversation_id=saved-conversation');
    const { ChatPage } = await import('../../pages/ChatPage');
    await act(async () => { root.render(createElement(ChatPage, { project: 'project-one' })); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(get).toHaveBeenCalledWith('/api/conversation/messages?conversation_id=saved-conversation');
    expect(host.textContent).toContain('Saved assistant response');
    expect(host.textContent).not.toContain('2026-10-14T10:24:00Z');
    expect(host.textContent?.match(/Saved assistant response/g)).toHaveLength(1);
    expect(host.querySelector('.oa-chat-inspector-body')?.textContent).toContain('Words');
    const viewResponse = host.querySelector<HTMLButtonElement>('.oa-chat-preview-jump');
    expect(viewResponse).toBeTruthy();
    await act(async () => { viewResponse?.click(); });
    expect(HTMLElement.prototype.scrollIntoView).toHaveBeenCalled();
    expect(host.querySelector('.oa-chat-title')).toBeNull();
    expect(host.querySelector('.oa-chat-menu summary[aria-label="Conversation actions"]')).toBeTruthy();
  });

  it('sends allow_external only after the user explicitly opts into external access', async () => {
    localStorage.setItem('oa-privacy-settings', JSON.stringify({ localOnly: false }));
    const { ChatPage } = await import('../../pages/ChatPage');
    await act(async () => { root.render(createElement(ChatPage, { project: 'project-one' })); });
    const external = host.querySelector<HTMLInputElement>('input[aria-label="Allow external network access for this message"]');
    expect(external?.disabled).toBe(false);
    await act(async () => { external?.click(); });
    const textarea = host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]');
    await act(async () => {
      if (textarea) {
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
        setter?.call(textarea, 'Use the configured provider');
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
    const send = host.querySelector<HTMLButtonElement>('button[aria-label="Send message"]');
    await act(async () => { send?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(postEvents).toHaveBeenCalledWith('/api/chat/stream', expect.objectContaining({ project: 'project-one', allow_external: true }), expect.any(AbortSignal));
  });

  it('loads a different recent conversation when the shell changes its query selection', async () => {
    const { ChatPage } = await import('../../pages/ChatPage');
    await act(async () => { root.render(createElement(ChatPage, { project: 'project-one' })); });
    await act(async () => {
      history.pushState({}, '', '/?view=chat&conversation_id=next-conversation');
      window.dispatchEvent(new PopStateEvent('popstate'));
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    expect(get).toHaveBeenCalledWith('/api/conversation/messages?conversation_id=next-conversation');
    expect(host.textContent).toContain('Saved assistant response');
  });

  it('does not let an aborted prior stream overwrite the newly selected conversation', async () => {
    postEvents.mockImplementationOnce(async function* (_path: string, _payload: unknown, signal: AbortSignal) {
      yield { event: 'message', data: JSON.stringify({ type: 'conversation', conversation: { id: 'old-conversation' } }) };
      await new Promise<void>((_resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true }));
    });
    const { ChatPage } = await import('../../pages/ChatPage');
    await act(async () => { root.render(createElement(ChatPage, { project: 'project-one' })); });
    const textarea = host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]');
    await act(async () => {
      if (textarea) {
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
        setter?.call(textarea, 'Start a slow response');
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
    await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Send message"]')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => {
      history.pushState({}, '', '/?view=chat&conversation_id=target-conversation');
      window.dispatchEvent(new PopStateEvent('popstate'));
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    expect(host.textContent).toContain('Target conversation message');
    expect(host.textContent).not.toContain('Start a slow response');
    expect(host.textContent).not.toContain('Aborted');
  });

  it('renders response Markdown without injecting HTML or unsafe link targets', async () => {
    const { MessageContent } = await import('../../components/chat/MessageContent');
    await act(async () => { root.render(createElement(MessageContent, { content: '**Bold**\n\n```ts\nconst answer = 42\n```\n\n[unsafe](javascript:alert(1)) <img src=x onerror=alert(1)>\n[docs](https://example.test/docs)' })); });
    expect(host.querySelector('strong')?.textContent).toBe('Bold');
    expect(host.querySelector('pre code')?.textContent).toBe('const answer = 42');
    expect(host.querySelector('a[href^="javascript:"]')).toBeNull();
    expect(host.querySelector('img')).toBeNull();
    expect(host.querySelector('a[href="https://example.test/docs"]')?.getAttribute('rel')).toBe('noopener noreferrer');
    expect(host.textContent).toContain('[unsafe](javascript:alert(1))');
  });
});
