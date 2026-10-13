import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Archive, Bot, CalendarClock, ChevronDown, Command, FolderOpen, Menu, MessageSquarePlus, Moon, MoreHorizontal, PanelLeftClose, Plus, Search, Settings2, Sun, Wrench, X } from 'lucide-react';
import { groups, isViewId, migrateLegacyView, titles, views, type ViewId } from './lib/navigation';
import { api, records } from './lib/react/api';
import { theme } from './state/theme';
import { specialPageIds } from './pages/specialPageIds';
import { Dialog, DialogContent } from './components/ui/dialog';
import './react-shell.css';
import './lib/routes/PageFrame.css';

const ChatPage = lazy(() => import('./pages/ChatPage').then(module => ({ default: module.ChatPage })));
const DashboardPage = lazy(() => import('./pages/DashboardPage').then(module => ({ default: module.DashboardPage })));
const SpecialPages = lazy(() => import('./pages/SpecialPages').then(module => ({ default: module.SpecialPages })));
const ResourcePages = lazy(() => import('./pages/ResourcePages'));
const PlatformWorkspaces = lazy(() => import('./pages/PlatformWorkspaces').then(module => ({ default: module.PlatformWorkspaces })));
const PersonalWorkSurface = lazy(() => import('./pages/PersonalWorkSurfaces').then(module => ({ default: module.PersonalWorkSurface })));
const platformWorkspaceIds = new Set(['developer', 'engine', 'finetune', 'phd', 'science', 'scraper']);
const personalSurfaceIds = new Set(['assistant', 'calendar', 'communications', 'twin']);

type Conversation = Record<string, unknown>;
const value = (input: unknown, fallback = '') => typeof input === 'string' && input ? input : fallback;
const conversationId = (item: Conversation) => value(item.id, value(item.conversation_id));
const savedProject = () => { try { return localStorage.getItem('oa-project') || 'default'; } catch { return 'default'; } };
function routeFromUrl(url: URL): ViewId {
  if (url.pathname.startsWith('/settings')) return 'settings';
  if (url.pathname === '/tasks') return 'automation';
  if (url.pathname === '/assistants') return 'agents';
  if (url.pathname.startsWith('/projects/')) return 'workspace';
  const tool = /^\/tools\/([^/]+)\/?$/.exec(url.pathname)?.[1];
  if (tool && isViewId(tool)) return tool;
  if (/^\/chat\/[^/]+\/?$/.test(url.pathname)) return 'chat';
  return migrateLegacyView(url.searchParams.get('view'));
}
const routeConversation = (url: URL) => url.searchParams.get('conversation_id') || /^\/chat\/([^/]+)\/?$/.exec(url.pathname)?.[1] || '';

function readInitialUrl() {
  const url = new URL(location.href);
  const oldView = url.searchParams.get('view');
  const activityRoute = oldView === 'dashboard' && url.searchParams.get('activity') === '1';
  const view = activityRoute ? 'dashboard' : history.state?.openagentNavigation && isViewId(history.state.view) ? history.state.view : routeFromUrl(url);
  const conversation = routeConversation(url);
  const project = url.searchParams.get('project') || savedProject();
  // Preserve existing query-based links while moving the old dashboard landing to chat.
  if ((!activityRoute && oldView === 'dashboard') || url.pathname !== '/') {
    url.pathname = '/';
    url.searchParams.set('view', view);
    if (conversation) url.searchParams.set('conversation_id', conversation);
    history.replaceState({ openagentNavigation: true, view, project, conversation_id: conversation || null }, '', url);
  }
  return { view, project, conversation };
}

export default function App() {
  const initial = useMemo(readInitialUrl, []);
  const [view, setView] = useState<ViewId>(initial.view);
  const [project, setProject] = useState(initial.project);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [showArchived, setShowArchived] = useState(false);
  const [historyState, setHistoryState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [collapsed, setCollapsed] = useState(false);
  const [mobileNav, setMobileNav] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [paletteQuery, setPaletteQuery] = useState('');
  const [focusIndex, setFocusIndex] = useState(0);
  const [themeMode, setThemeMode] = useState(theme.read());
  const [searchResults, setSearchResults] = useState<Conversation[] | null>(null);
  const [searchState, setSearchState] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const searchRef = useRef<HTMLInputElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);
  const mobileNavOpener = useRef<HTMLElement | null>(null);
  const listController = useRef<AbortController | null>(null);

  const filteredViews = useMemo(() => views.filter(item => `${item.label} ${item.id} ${item.keywords ?? ''}`.toLowerCase().includes(paletteQuery.trim().toLowerCase())), [paletteQuery]);
  const commandCount = (searchResults?.length ?? 0) + filteredViews.length;
  const sortedConversations = useMemo(() => [...conversations].sort((a, b) => Number(b.pinned === true) - Number(a.pinned === true)), [conversations]);
  const pinned = useMemo(() => showArchived ? [] : sortedConversations.filter(item => item.pinned === true), [sortedConversations, showArchived]);
  const recents = useMemo(() => (showArchived ? sortedConversations : sortedConversations.filter(item => item.pinned !== true)).slice(0, 20), [sortedConversations, showArchived]);
  const activeConversation = conversations.find(item => conversationId(item) === new URL(location.href).searchParams.get('conversation_id'));

  const refreshConversations = useCallback(async () => {
    listController.current?.abort();
    const controller = new AbortController();
    listController.current = controller;
    try {
      const archived = showArchived ? 'true' : 'false';
      const query = new URLSearchParams({ project, limit: '100', sort: 'pinned', include_archived: archived, archived_only: archived });
      const data = await api.get<unknown>(`/api/conversations/organized?${query}`, controller.signal);
      if (!controller.signal.aborted) { setConversations(records(data, 'conversations')); setHistoryState('ready'); }
    } catch {
      if (!controller.signal.aborted) setHistoryState('error');
    }
  }, [project, showArchived]);

  useEffect(() => {
    const cleanup = theme.init();
    return cleanup;
  }, []);
  useEffect(() => { void refreshConversations(); return () => listController.current?.abort(); }, [refreshConversations]);
  useEffect(() => {
    const update = () => { void refreshConversations(); };
    window.addEventListener('openagent:conversations-updated', update);
    return () => window.removeEventListener('openagent:conversations-updated', update);
  }, [refreshConversations]);
  useEffect(() => {
    const onPop = () => {
      const url = new URL(location.href);
      const stateView = history.state?.openagentNavigation && isViewId(history.state.view) ? history.state.view : routeFromUrl(url);
      setView(stateView);
      const nextProject = url.searchParams.get('project') || savedProject();
      setProject(nextProject);
      setPaletteOpen(false);
      setMobileNav(false);
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);
  useEffect(() => {
    if (!paletteOpen) return;
    previousFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    searchRef.current?.focus();
    return () => { if (previousFocus.current?.isConnected) previousFocus.current.focus(); previousFocus.current = null; };
  }, [paletteOpen]);
  useEffect(() => {
    if (mobileNav) {
      mobileNavOpener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      requestAnimationFrame(() => document.querySelector<HTMLElement>('.oa-sidebar.mobile-open .oa-new-chat')?.focus());
    } else if (mobileNavOpener.current?.isConnected) {
      mobileNavOpener.current.focus(); mobileNavOpener.current = null;
    }
  }, [mobileNav]);
  useEffect(() => {
    if (!paletteOpen || paletteQuery.trim().length < 2) { setSearchResults(null); setSearchState('idle'); return; }
    const controller = new AbortController();
    setSearchState('loading');
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({ q: paletteQuery.trim(), project, limit: '20' });
        const data = await api.get<unknown>(`/api/chat/search?${params}`, controller.signal);
        if (!controller.signal.aborted) {
          setSearchResults(records(data, 'results').map(row => ({ ...row, id: value(row.conversation_id, value(row.id)), title: value(row.conversation_title, value(row.title, 'Conversation')) })));
          setSearchState('ready');
        }
      } catch { if (!controller.signal.aborted) setSearchState('error'); }
    }, 250);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [paletteOpen, paletteQuery, project]);

  const commitUrl = useCallback((next: ViewId, mode: 'push' | 'replace' = 'push', selectedConversation?: string) => {
    const url = new URL(location.href);
    url.pathname = '/';
    url.searchParams.set('view', next);
    if (next === 'dashboard') url.searchParams.set('activity', '1'); else url.searchParams.delete('activity');
    url.searchParams.set('project', project);
    if (selectedConversation) url.searchParams.set('conversation_id', selectedConversation);
    else if (next === 'chat') url.searchParams.delete('conversation_id');
    else url.searchParams.delete('conversation_id');
    history[mode === 'push' ? 'pushState' : 'replaceState']({ openagentNavigation: true, view: next, project, conversation_id: selectedConversation ?? null }, '', url);
    setView(next);
    window.dispatchEvent(new PopStateEvent('popstate'));
    setMobileNav(false); setPaletteOpen(false); setPaletteQuery(''); setSearchResults(null);
  }, [project]);
  const navigateConversation = (id: string) => { if (id) commitUrl('chat', 'push', id); };
  const newChat = () => commitUrl('chat');
  const changeProject = (next: string) => {
    if (!next || next === project) return;
    setProject(next);
    try { localStorage.setItem('oa-project', next); } catch { /* Project stays in the URL if storage is unavailable. */ }
    const url = new URL(location.href); url.pathname = '/'; url.searchParams.set('project', next); url.searchParams.set('view', view);
    if (view === 'dashboard') url.searchParams.set('activity', '1'); else url.searchParams.delete('activity');
    url.searchParams.delete('conversation_id');
    history.pushState({ openagentNavigation: true, view, project: next }, '', url);
    window.dispatchEvent(new PopStateEvent('popstate'));
  };
  const toggleTheme = () => {
    const next = themeMode === 'dark' ? 'light' : 'dark';
    theme.set(next); setThemeMode(next);
  };
  const openPalette = () => { setPaletteQuery(''); setFocusIndex(0); setPaletteOpen(true); };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); openPalette(); }
      else if (event.key === 'Escape') { if (paletteOpen) setPaletteOpen(false); if (mobileNav) setMobileNav(false); }
      else if (paletteOpen && event.key === 'ArrowDown') { event.preventDefault(); setFocusIndex(index => (index + 1) % Math.max(1, commandCount)); }
      else if (paletteOpen && event.key === 'ArrowUp') { event.preventDefault(); setFocusIndex(index => (index - 1 + commandCount) % Math.max(1, commandCount)); }
      else if (paletteOpen && event.key === 'Enter') {
        const conversationResult = searchResults?.[focusIndex];
        if (conversationResult) openSearchResult(conversationResult);
        else { const item = filteredViews[focusIndex - (searchResults?.length ?? 0)]; if (item) commitUrl(item.id); }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [paletteOpen, mobileNav, filteredViews, searchResults, commandCount, focusIndex, commitUrl]);

  const title = view === 'chat' ? (value(activeConversation?.title, 'New chat')) : titles[view];
  const projectOptions = [...new Set([project, ...conversations.map(item => value(item.project_id, value(item.project))).filter(Boolean)])];
  const openSearchResult = (item: Conversation) => navigateConversation(conversationId(item));
  const archiveConversation = async (item: Conversation, archived: boolean) => {
    const id = conversationId(item); if (!id) return;
    if (!archived && !window.confirm(`Archive “${value(item.title, 'this conversation')}”? You can restore it later.`)) return;
    try {
      await api.post('/api/conversations/bulk', { conversation_ids: [id], action: archived ? 'unarchive' : 'archive', project });
      await refreshConversations();
      window.dispatchEvent(new Event('openagent:conversations-updated'));
    } catch { setHistoryState('error'); }
  };

  return <div className={`oa-app-shell ${collapsed ? 'sidebar-collapsed' : ''}`}>
    {mobileNav && <button className="oa-mobile-scrim" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}
    <aside className={`oa-sidebar ${mobileNav ? 'mobile-open' : ''}`} aria-label="Primary navigation" onKeyDown={event => {
      if (!mobileNav || event.key !== 'Tab') return;
      const items = event.currentTarget.querySelectorAll<HTMLElement>('button:not([disabled]),select:not([disabled])');
      const first = items[0], last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }}>
      <div className="oa-brand-row">
        <button className="oa-brand" onClick={newChat} aria-label="OpenAgent home"><span className="oa-brand-mark"><i /></span><strong>OpenAgent</strong></button>
        <button className="oa-icon-button oa-sidebar-dismiss" aria-label="Close navigation" onClick={() => setMobileNav(false)}><X size={17} /></button>
      </div>
      <button className="oa-new-chat" onClick={newChat} aria-current={view === 'chat' && !new URL(location.href).searchParams.has('conversation_id') ? 'page' : undefined}><MessageSquarePlus size={16} /><span>New chat</span></button>
      <button className="oa-nav-action" onClick={openPalette}><Search size={16} /><span>Search</span><kbd>⌘ K</kbd></button>
      <button className={`oa-nav-action ${view === 'agents' ? 'active' : ''}`} onClick={() => commitUrl('agents')}><Bot size={16} /><span>Assistants</span></button>
      <button className={`oa-nav-action ${view === 'automation' || view === 'calendar' ? 'active' : ''}`} onClick={() => commitUrl('automation')}><CalendarClock size={16} /><span>Scheduled</span></button>

      <div className="oa-workspace-picker">
        <span className="oa-section-label">Workspace</span>
        <label className="oa-project-select"><FolderOpen size={15} /><select aria-label="Project scope" value={project} onChange={event => changeProject(event.currentTarget.value)}>{projectOptions.map(option => <option key={option} value={option}>{option === 'default' ? 'Personal workspace' : option}</option>)}</select><ChevronDown size={13} /></label>
      </div>
      <div className="oa-history-scroll">
        {!!pinned.length && <section className="oa-history-section" aria-label="Pinned conversations"><div className="oa-section-heading"><span>Pinned</span><Archive size={12} /></div>{pinned.slice(0, 8).map(item => <ConversationRow key={conversationId(item)} item={item} archived={false} active={new URL(location.href).searchParams.get('conversation_id') === conversationId(item)} onClick={() => navigateConversation(conversationId(item))} onArchive={() => void archiveConversation(item, false)} />)}</section>}
        <section className="oa-history-section" aria-label={showArchived ? 'Archived conversations' : 'Recent conversations'}><div className="oa-section-heading"><span>{showArchived ? 'Archived' : 'Recent'}</span><button title={showArchived ? 'Show recent conversations' : 'Show archived conversations'} aria-label={showArchived ? 'Show recent conversations' : 'Show archived conversations'} aria-pressed={showArchived} onClick={() => setShowArchived(value => !value)}><Archive size={13} /></button></div>
          {historyState === 'loading' && <p className="oa-history-note">Loading conversations…</p>}
          {historyState === 'error' && <button className="oa-history-note oa-retry" onClick={() => void refreshConversations()}>History unavailable · Retry</button>}
          {historyState === 'ready' && !recents.length && !pinned.length && <p className="oa-history-note">{showArchived ? 'No archived conversations.' : 'Your conversations will appear here.'}</p>}
          {recents.map(item => <ConversationRow key={conversationId(item)} item={item} archived={showArchived} active={new URL(location.href).searchParams.get('conversation_id') === conversationId(item)} onClick={() => navigateConversation(conversationId(item))} onArchive={() => void archiveConversation(item, showArchived)} />)}
        </section>
      </div>
      <div className="oa-sidebar-bottom">
        <button className="oa-nav-action" onClick={openPalette}><Wrench size={16} /><span>Tools</span></button>
        <button className={`oa-nav-action ${view === 'settings' ? 'active' : ''}`} onClick={() => commitUrl('settings')}><Settings2 size={16} /><span>Settings</span></button>
      </div>
    </aside>

    <main className="oa-main-area">
      <a className="oa-skip-link" href="#oa-main-content">Skip to content</a>
      <header className="oa-topbar">
        <div className="oa-topbar-start">
          <button className="oa-icon-button oa-sidebar-toggle" aria-label={window.matchMedia('(max-width: 760px)').matches ? 'Open navigation' : collapsed ? 'Open sidebar' : 'Close sidebar'} aria-expanded={window.matchMedia('(max-width: 760px)').matches ? mobileNav : !collapsed} onClick={() => { if (window.matchMedia('(max-width: 760px)').matches) setMobileNav(true); else setCollapsed(value => !value); }}>{window.matchMedia('(max-width: 760px)').matches ? <Menu size={18} /> : collapsed ? <Menu size={18} /> : <PanelLeftClose size={17} />}</button>
          <span className="oa-page-title" title={title}>{title}</span>
          {view === 'chat' && project !== 'default' && <span className="oa-context-project"><FolderOpen size={13} />{project}</span>}
        </div>
        <div className="oa-topbar-actions">
          <button className="oa-top-search" onClick={openPalette} aria-haspopup="dialog"><Search size={15} /><span>Search</span><kbd>⌘ K</kbd></button>
          <button className="oa-icon-button" onClick={toggleTheme} aria-label={`Use ${themeMode === 'dark' ? 'light' : 'dark'} theme`} title="Toggle theme">{themeMode === 'dark' ? <Sun size={16} /> : <Moon size={16} />}</button>
          {view !== 'chat' && <button className="oa-top-action" onClick={newChat}><Plus size={15} />New chat</button>}
        </div>
      </header>
      <div className="oa-page-stage" id="oa-main-content" tabIndex={-1}>
        <Suspense fallback={<div className="oa-route-loading" role="status" aria-live="polite">Loading {titles[view]}…</div>}>
          {view === 'chat' ? <ChatPage key={`chat:${project}`} project={project} /> : view === 'dashboard' ? <DashboardPage key={`dashboard:${project}`} project={project} /> : platformWorkspaceIds.has(view) ? <PlatformWorkspaces key={`${view}:${project}`} id={view as import('./pages/PlatformWorkspaces').PlatformWorkspaceId} /> : personalSurfaceIds.has(view) ? <PersonalWorkSurface key={`${view}:${project}`} id={view as import('./pages/PersonalWorkSurfaces').PersonalSurfaceId} /> : specialPageIds.has(view) ? <SpecialPages key={`${view}:${project}`} id={view} /> : <ResourcePages key={`${view}:${project}`} id={view} />}
        </Suspense>
      </div>
    </main>

    <Dialog open={paletteOpen} onOpenChange={setPaletteOpen}>
      {paletteOpen && <DialogContent className="oa-command-dialog" aria-label="Search OpenAgent">
        <div className="oa-command-palette">
          <div className="oa-command-input"><Search size={17} /><input ref={searchRef} value={paletteQuery} onChange={event => { setPaletteQuery(event.currentTarget.value); setFocusIndex(0); }} placeholder="Search conversations and tools" aria-label="Search conversations and tools" /><kbd>ESC</kbd></div>
          <div className="oa-command-scroll">
            {searchResults && <section className="oa-command-section"><h2>Conversations</h2>{searchResults.map((item, index) => <button key={`${conversationId(item)}-${index}`} className={`oa-command-row ${focusIndex === index ? 'selected' : ''}`} aria-selected={focusIndex === index} onMouseMove={() => setFocusIndex(index)} onClick={() => openSearchResult(item)}><MessageSquarePlus size={15} /><span>{value(item.title, 'Conversation')}</span><small>{value(item.content, 'Matching message')}</small></button>)}{searchState === 'ready' && !searchResults.length && <p className="oa-command-note">No conversations match.</p>}</section>}
            {paletteQuery.trim().length >= 2 && searchState === 'loading' && <p className="oa-command-note">Searching conversations…</p>}
            {searchState === 'error' && <p className="oa-command-note">Conversation search is unavailable.</p>}
            <section className="oa-command-section"><h2>{paletteQuery.trim() ? 'Tools and screens' : 'Tools'}</h2>{groups.map(group => {
              const items = group.items.filter(item => filteredViews.some(match => match.id === item.id));
              if (!items.length) return null;
              return <div className="oa-command-group" key={group.key}><h3>{group.label}</h3>{items.map(item => { const Icon = item.icon; const index = (searchResults?.length ?? 0) + filteredViews.findIndex(match => match.id === item.id); return <button key={item.id} className={`oa-command-row ${focusIndex === index ? 'selected' : ''}`} aria-selected={focusIndex === index} onMouseMove={() => setFocusIndex(index)} onClick={() => commitUrl(item.id)}><Icon size={15} /><span>{item.label}</span><small>{item.id === 'chat' ? 'New conversation' : item.keywords?.split(' ')[0] ?? 'Open'}</small></button>; })}</div>;
            })}{!filteredViews.length && searchState !== 'loading' && <p className="oa-command-note">No tools match “{paletteQuery}”.</p>}</section>
          </div>
          <div className="oa-command-footer"><span><kbd>↑</kbd><kbd>↓</kbd> Navigate</span><span><kbd>↵</kbd> Open</span><span><Command size={11} /> K Search</span></div>
        </div>
      </DialogContent>}
    </Dialog>
  </div>;
}

function ConversationRow({ item, active, archived, onClick, onArchive }: { item: Conversation; active: boolean; archived: boolean; onClick: () => void; onArchive: () => void }) {
  const title = value(item.title, 'Untitled conversation');
  return <div className={`oa-conversation-entry ${active ? 'active' : ''}`}>
    <button className={`oa-conversation-row ${active ? 'active' : ''}`} onClick={onClick} title={title} aria-current={active ? 'page' : undefined}><MessageSquarePlus size={14} /><span>{title}</span></button>
    <details className="oa-conversation-menu"><summary aria-label={`Actions for ${title}`}><MoreHorizontal size={15} /></summary><div role="menu"><button role="menuitem" onClick={onArchive}>{archived ? 'Restore conversation' : 'Archive conversation'}</button></div></details>
  </div>;
}
