import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import { AnimatePresence, LazyMotion, domAnimation, m, useReducedMotion } from 'motion/react';
import {
  Activity, ArrowRight, BookOpen, CalendarDays, Check, CheckCircle2, ChevronDown, Code2, Cpu,
  FlaskConical, Folder, FolderOpen, Globe2, House, Layers3, LockKeyhole, Mail,
  Menu, MessageCircle, Mic, Network, Package, PanelLeftClose, Command, FileText,
  Plus, Search, Settings2, Shield, SlidersHorizontal, Terminal, Workflow, X, Bell, Sun, Moon,
} from 'lucide-react';
import * as Dialog from './components/ui/dialog';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useLocation, useNavigate } from 'react-router-dom';
import { useWorkspaceState } from './state';
import { apiGet, apiPost, apiConnectionMessage } from './api/client';
import { demoAwareRequest } from './demo/api';
import { LivingActivityCenter } from './components/LivingActivityCenter';

const HomeScreen = lazy(() => import('./screens/HomeChat').then(module => ({ default: module.HomeScreen })));
const ChatScreen = lazy(() => import('./screens/HomeChat').then(module => ({ default: module.ChatScreen })));
const CodeQScreen = lazy(() => import('./screens/WorkspaceScreens').then(module => ({ default: module.CodeQScreen })));
const FilesScreen = lazy(() => import('./screens/WorkspaceScreens').then(module => ({ default: module.FilesScreen })));
const ResearchScreen = lazy(() => import('./screens/WorkspaceScreens').then(module => ({ default: module.ResearchScreen })));
const KnowledgeScreen = lazy(() => import('./screens/WorkspaceScreens').then(module => ({ default: module.KnowledgeScreen })));
const CalendarScreen = lazy(() => import('./screens/WorkspaceScreens').then(module => ({ default: module.CalendarScreen })));
const AgentsScreen = lazy(() => import('./screens/OperationsScreens').then(module => ({ default: module.AgentsScreen })));
const AutomationsScreen = lazy(() => import('./screens/OperationsScreens').then(module => ({ default: module.AutomationsScreen })));
const ModelsScreen = lazy(() => import('./screens/OperationsScreens').then(module => ({ default: module.ModelsScreen })));
const SettingsScreen = lazy(() => import('./screens/OperationsScreens').then(module => ({ default: module.SettingsScreen })));
const AdvancedScreen = lazy(() => import('./screens/AdvancedScreen'));

const primary = [
  ['home', 'Home', House], ['chat', 'Chat', MessageCircle], ['codeq', 'CodeQ', Code2],
  ['agents', 'Agents', Network], ['research', 'Research', Search], ['files', 'Files', Folder],
  ['automations', 'Automations', Workflow], ['calendar', 'Calendar', CalendarDays], ['models', 'Models', Package],
] as const;
const advanced = [
  ['knowledge', 'Knowledge', BookOpen], ['voice', 'Voice', Mic], ['assistant', 'Assistant', CheckCircle2],
  ['communications', 'Communications', Mail], ['security', 'Security', Shield], ['developer', 'Developer', Terminal],
  ['engine', 'Adaptive engine', Cpu], ['science', 'Scientific AI', FlaskConical], ['twin', 'Personal context', Activity],
  ['vault', 'Vault', LockKeyhole], ['scraper', 'Web scraper', Globe2], ['finetune', 'Fine-tuning', SlidersHorizontal],
] as const;
const titles = new Map<string, string>([...primary, ...advanced].map(([id, name]) => [id, name]));
titles.set('settings', 'Settings');
titles.set('tools', 'All tools');
const allViews = [...primary, ...advanced, ['settings', 'Settings', Settings2], ['tools', 'All tools', Layers3]] as const;

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

const getView = (search: string) => {
  const requested = new URLSearchParams(search).get('view') || 'home';
  return titles.has(requested) ? requested : 'home';
};

function ToolsScreen() {
  const [query, setQuery] = useState('');
  const groups = [
    { name: 'Everyday', ids: ['home', 'chat', 'assistant', 'agents', 'calendar'], details: ['Your workspace and recent work', 'Conversations and projects', 'Tasks, goals and priorities', 'Runs and collaboration', 'Plan and schedule'] },
    { name: 'Create & research', ids: ['research', 'files', 'knowledge', 'codeq', 'voice', 'science'], details: ['Evidence and reports', 'Documents and previews', 'Indexed project sources', 'Coding and Git review', 'Recordings and transcripts', 'Datasets and experiments'] },
    { name: 'Run & connect', ids: ['automations', 'communications', 'scraper', 'finetune', 'engine'], details: ['Scheduled workflows', 'Connected messages', 'Permitted extraction jobs', 'Datasets and training', 'Compute compatibility'] },
    { name: 'Platform & privacy', ids: ['models', 'security', 'vault', 'twin', 'developer', 'settings'], details: ['Providers and capabilities', 'Access and audit', 'Protected credentials', 'Personal context', 'Integrations and APIs', 'Workspace preferences'] },
  ];
  const needle = query.trim().toLowerCase();
  return <div className="p-workspace p-hub"><header className="p-titlebar"><div><span className="p-kicker"><Layers3 className="icon xs"/>Browse workspace</span><h1>All tools</h1><p>Everything you can do in OpenAgent, organised around your work.</p></div><span className="p-fixture"><Layers3 className="icon xs"/>22 existing feature views</span></header><label className="p-hub-search"><Search className="icon sm"/><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Find a tool or capability…" aria-label="Find a tool or capability"/><span>{needle ? 'Filtered' : '22 tools'}</span></label>{groups.map(group => {const items = group.ids.map((id, index) => ({ id, detail: group.details[index], item: allViews.find(([key]) => key === id) })).filter(entry => entry.item && (!needle || `${entry.id} ${entry.item[1]} ${entry.detail} ${group.name}`.toLowerCase().includes(needle))); if (!items.length) return null; return <section className="p-tools-group" key={group.name}><div className="p-section-heading"><h2>{group.name}</h2><span>{items.length} {items.length === 1 ? 'tool' : 'tools'}</span></div><div className="p-tools-grid">{items.map(({ id, detail, item }) => { if (!item) return null; const [, routeLabel, Icon] = item; const label = id === 'assistant' ? 'Personal assistant' : routeLabel; return <button key={id} type="button" className="p-tool-link" data-nav={id}><span className="square-icon"><Icon size={18}/></span><span><strong>{label}</strong><small>{detail}</small></span><ArrowRight className="icon sm" aria-hidden="true"/></button>; })}</div></section>; })}{needle && !groups.some(group => group.ids.some((id, i) => `${id} ${titles.get(id)} ${group.details[i]} ${group.name}`.toLowerCase().includes(needle))) && <div className="p-empty" role="status"><Search className="icon lg"/><h3>No matching tools</h3><p>Try another name or capability.</p></div>}</div>;
}

export function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const currentView = getView(location.search);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [mobileLayout, setMobileLayout] = useState(() => typeof window !== 'undefined' && window.matchMedia?.('(max-width: 800px)').matches === true);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [projectPickerOpen, setProjectPickerOpen] = useState(false);
  const [query, setQuery] = useState('');
  const queryClient = useQueryClient();
  const navigationLatch = useRef<string | null>(null);
  const mobileMenuWasOpened = useRef(false);
  const collapsed = useWorkspaceState(state => state.collapsed);
  const toggleCollapsed = useWorkspaceState(state => state.toggleCollapsed);
  const theme = useWorkspaceState(state => state.theme);
  const setTheme = useWorkspaceState(state => state.setTheme);
  const reduceMotionPreference = useWorkspaceState(state => state.reduceMotion);
  const dataMode = useWorkspaceState(state => state.dataMode);
  const setDataMode = useWorkspaceState(state => state.setDataMode);
  const selectedDemoProject = useWorkspaceState(state => state.selectedDemoProject);
  const setSelectedDemoProject = useWorkspaceState(state => state.setSelectedDemoProject);
  const requestDemoProjectCreate = useWorkspaceState(state => state.requestDemoProjectCreate);
  const currentProject = dataMode === 'demo' ? selectedDemoProject : 'Personal Workspace';
  const reduceMotion = useReducedMotion() || reduceMotionPreference;
  const backendHealth = useQuery({
    queryKey: ['backend', 'health'],
    queryFn: ({ signal }) => apiGet('/api/health', signal),
    enabled: dataMode === 'live',
    retry: false,
    refetchInterval: 30_000,
  });
  const recentChatsQuery = useQuery({
    queryKey: ['chat', 'conversations', dataMode],
    queryFn: async ({ signal }): Promise<{ conversations?: unknown[] }> => dataMode === 'demo' ? { conversations: await demoAwareRequest<{ chats?: unknown[] }>('demo', 'GET', 'shell', undefined, signal).then(data => data.chats ?? []) } : apiGet<{ conversations?: unknown[] }>('/api/conversations?project=default&limit=100', signal),
    retry: false,
  });
  const recentChats = Array.isArray(recentChatsQuery.data?.conversations)
    ? recentChatsQuery.data.conversations.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
    : [];
  const demoProjectsQuery = useQuery({
    queryKey: ['home', 'projects', 'demo'],
    queryFn: async ({ signal }) => {
      const shell = await demoAwareRequest<Record<string, unknown>>('demo', 'GET', 'shell', undefined, signal);
      return { projects: Array.isArray(shell.projects) ? shell.projects.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object') : [] };
    },
    enabled: projectPickerOpen && dataMode === 'demo',
    retry: false,
  });
  const projectPickerTriggerRef = useRef<HTMLButtonElement>(null);
  const notificationsQuery = useQuery({
    queryKey: ['scheduler', 'notifications', 'default'],
    queryFn: async ({ signal }): Promise<{ notifications?: Record<string, unknown>[] }> => dataMode === 'demo' ? { notifications: await demoAwareRequest<{ tasks?: Record<string, unknown>[] }>('demo', 'GET', 'collections.assistant', undefined, signal).then(data => data.tasks ?? []) } : apiGet<{ notifications?: Record<string, unknown>[] }>('/api/tasks/notifications?project=default&limit=50', signal),
    enabled: notificationsOpen,
    retry: false,
    refetchInterval: notificationsOpen ? 30_000 : false,
  });
  const notifications = Array.isArray(notificationsQuery.data?.notifications) ? notificationsQuery.data.notifications : [];
  useEffect(() => { void queryClient.invalidateQueries(); }, [dataMode, queryClient]);
  const markNotificationReadMutation = useMutation({
    mutationFn: (id: string) => dataMode === 'demo' ? demoAwareRequest('demo', 'POST', `notifications.${id}`, { read_at: new Date().toISOString() }) : apiPost(`/api/tasks/notifications/${encodeURIComponent(id)}/read`, { project: 'default' }),
    onSuccess: async () => { await queryClient.invalidateQueries({ queryKey: ['scheduler', 'notifications', 'default'] }); },
  });
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);

  const go = useCallback((id: string) => {
    if (!titles.has(id)) return;
    if (navigationLatch.current === id) return;
    navigationLatch.current = id;
    window.setTimeout(() => { navigationLatch.current = null; }, 0);
    navigate(`/?view=${encodeURIComponent(id)}`);
    setMobileOpen(false);
    setPaletteOpen(false);
    setQuery('');
  }, [navigate]);

  const openProjectPicker = (trigger: HTMLButtonElement) => {
    if (dataMode !== 'demo') return;
    projectPickerTriggerRef.current = trigger;
    setProjectPickerOpen(true);
  };
  const closeProjectPicker = () => {
    setProjectPickerOpen(false);
    requestAnimationFrame(() => projectPickerTriggerRef.current?.focus());
  };

  useEffect(() => {
    setTheme(theme);
  // Apply the stored theme once at startup; subsequent changes are handled by the selector.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (theme !== 'system' || typeof window.matchMedia !== 'function') return;
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const update = () => setTheme('system');
    media.addEventListener?.('change', update);
    return () => media.removeEventListener?.('change', update);
  }, [setTheme, theme]);
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return;
    const media = window.matchMedia('(max-width: 800px)');
    const update = () => setMobileLayout(media.matches);
    update();
    media.addEventListener?.('change', update);
    return () => media.removeEventListener?.('change', update);
  }, []);
  useEffect(() => {
    if (mobileOpen) {
      sidebarRef.current?.querySelector<HTMLButtonElement>('.mobile-sidebar-close')?.focus();
      mobileMenuWasOpened.current = true;
    } else if (mobileMenuWasOpened.current && mobileLayout) {
      menuButtonRef.current?.focus();
      mobileMenuWasOpened.current = false;
    }
  }, [mobileLayout, mobileOpen]);
  useEffect(() => {
    document.title = `${titles.get(currentView) ?? 'OpenAgent'} · OpenAgent`;
  }, [currentView]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault(); setPaletteOpen(true);
      }
      const target = event.target;
      const isEditing = target instanceof HTMLElement && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
      if (!isEditing && !event.metaKey && !event.ctrlKey && !event.altKey && event.key.toLowerCase() === 'n') {
        event.preventDefault(); go('chat');
      }
      if (event.key === 'Escape') { setPaletteOpen(false); setMobileOpen(false); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [go]);
  const filtered = useMemo(() => allViews.filter(([id, label]) => `${id} ${label}`.toLowerCase().includes(query.trim().toLowerCase())), [query]);

  const onContentClick = (event: MouseEvent<HTMLElement>) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const nav = target.closest<HTMLElement>('[data-nav]')?.dataset.nav;
    if (nav) go(nav);
  };

  const page = currentView === 'home'
    ? <HomeScreen go={go} />
    : currentView === 'chat'
      ? <ChatScreen go={go} />
      : currentView === 'codeq' ? <CodeQScreen />
        : currentView === 'files' ? <FilesScreen />
          : currentView === 'research' ? <ResearchScreen />
            : currentView === 'knowledge' ? <KnowledgeScreen />
              : currentView === 'calendar' ? <CalendarScreen />
                : currentView === 'agents' ? <AgentsScreen />
                  : currentView === 'automations' ? <AutomationsScreen />
                    : currentView === 'models' ? <ModelsScreen />
                      : currentView === 'settings' ? <SettingsScreen />
                        : currentView === 'tools' ? <ToolsScreen />
                          : <AdvancedScreen route={currentView} />;

  return <div className="app-shell"><div className="app-frame">
    <LazyMotion features={domAnimation}>
      <m.aside ref={sidebarRef} inert={mobileLayout && !mobileOpen} onClick={onContentClick} className={`sidebar ${collapsed ? 'collapsed' : ''} ${mobileOpen ? 'mobile-open' : ''}`} id="sidebar" animate={{ width: collapsed ? 72 : 252, flexBasis: collapsed ? 72 : 252 }} transition={reduceMotion ? { duration: 0 } : { type: 'spring', stiffness: 360, damping: 34 }} aria-label="Primary navigation">
        <div className="sidebar-top">
          <div className="brand"><span className="brand-mark" aria-hidden="true" /><div className="sidebar-copy"><strong>OpenAgent</strong><small>Local AI Workspace</small></div>
            <button className="icon-button sidebar-collapse" aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} onClick={toggleCollapsed}><PanelLeftClose size={17} /></button>
          </div>
          <button ref={projectPickerTriggerRef} className="sidebar-project" type="button" aria-label={dataMode === 'demo' ? `Choose project, current ${currentProject}` : 'Current workspace, switching unavailable'} title={dataMode === 'demo' ? 'Choose a local Demo project' : 'Live project switching is unavailable until a verified workspace scope operation is connected'} disabled={dataMode !== 'demo'} onClick={event => openProjectPicker(event.currentTarget)}><FolderOpen size={15} /><span className="sidebar-copy">{currentProject}</span><ChevronDown className="project-chevron" size={14} /></button>
          <button className="sidebar-new" type="button" title="New chat (N)" onClick={() => {
            if (dataMode === 'live') {
              sessionStorage.removeItem('oa-second-conversation-id');
              window.dispatchEvent(new CustomEvent('openagent:select-conversation', { detail: '' }));
            }
            go('chat');
          }}><Plus size={16} /><span className="nav-label">New chat</span><kbd className="sidebar-shortcut">N</kbd></button>
          <button className="sidebar-search" onClick={() => setPaletteOpen(true)} title="Search pages, files, actions (Ctrl+K)"><Search size={16} /><span className="sidebar-search-label">Search pages, files, actions</span></button>
        </div>
        <div className="sidebar-scroll">
          <nav className="nav-group" aria-label="Main">
            {primary.map(([id, label, Icon]) => <button key={id} className={`nav-link ${currentView === id ? 'active' : ''}`} data-nav={id} aria-current={currentView === id ? 'page' : undefined} title={collapsed ? label : undefined}>{currentView === id && <m.span layoutId="active-navigation" className="nav-active-indicator" transition={reduceMotion ? { duration: 0 } : { type: 'spring', stiffness: 500, damping: 36 }} />}<Icon className="icon" size={18} /><span className="nav-label">{label}</span></button>)}
            <button className={`nav-link ${currentView === 'tools' ? 'active' : ''}`} data-nav="tools">{currentView === 'tools' && <m.span layoutId="active-navigation" className="nav-active-indicator" transition={reduceMotion ? { duration: 0 } : { type: 'spring', stiffness: 500, damping: 36 }} />}<Layers3 size={18} /><span className="nav-label">More tools</span></button>
            <button className={`nav-link ${currentView === 'settings' ? 'active' : ''}`} data-nav="settings">{currentView === 'settings' && <m.span layoutId="active-navigation" className="nav-active-indicator" transition={reduceMotion ? { duration: 0 } : { type: 'spring', stiffness: 500, damping: 36 }} />}<Settings2 size={18} /><span className="nav-label">Settings</span></button>
          </nav>
          <div className="recent-block"><hr className="sidebar-divider" /><div className="sidebar-section-head"><span className="sidebar-heading-text">Recent chats</span><button className="text-link" type="button" onClick={() => go('chat')}>View all</button></div>{recentChatsQuery.isLoading ? <div className="recent-chats-empty" role="status">Loading conversations…</div> : recentChatsQuery.isError ? <button className="recent-chats-empty recent-chat-link" type="button" onClick={() => void recentChatsQuery.refetch()} title={apiConnectionMessage(recentChatsQuery.error)}>Could not load chats · retry</button> : recentChats.length ? recentChats.slice(0, 5).map((conversation, index) => { const title = String(conversation.title ?? conversation.name ?? 'Untitled conversation'); const id = String(conversation.id ?? title); return <button className="recent-item" type="button" key={id || index} title={title} onClick={() => { if (dataMode !== 'demo') { try { sessionStorage.setItem('oa-second-conversation-id', id); } catch { /* in-memory route navigation remains available */ } if (currentView === 'chat') window.dispatchEvent(new CustomEvent('openagent:select-conversation', { detail: id })); } if (currentView !== 'chat') go('chat'); }}><FileText className="icon" size={15}/><span className="flex1"><strong>{title}</strong><span>{String(conversation.time ?? relativeTimestamp(conversation.updated_at ?? conversation.created_at))}</span></span></button>; }) : <div className="recent-chats-empty" role="status">No conversations yet</div>}</div>
        </div>
        <div className="sidebar-foot"><button className="profile-pill" type="button" aria-label={dataMode === 'demo' ? `Choose project, current ${currentProject}` : 'Current workspace, switching unavailable'} title={dataMode === 'demo' ? 'Choose a local Demo project' : 'Live project switching is unavailable until a verified workspace scope operation is connected'} disabled={dataMode !== 'demo'} onClick={event => openProjectPicker(event.currentTarget)}><span className="avatar sm">P</span><span className="sidebar-copy flex1 truncate">{currentProject}</span><ChevronDown className="project-chevron" size={14}/></button></div>
        <button className="icon-button mobile-sidebar-close" aria-label="Close navigation" onClick={() => setMobileOpen(false)}><X size={18} /></button>
      </m.aside>
    </LazyMotion>
    <AnimatePresence>{mobileOpen && <m.button className="mobile-backdrop visible" aria-label="Close navigation" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: reduceMotion ? 0 : 0.18 }} onClick={() => setMobileOpen(false)} />}</AnimatePresence>
    <main className="main">
      <header className="topbar">
        <button ref={menuButtonRef} className="icon-button mobile-menu-button" aria-label="Open navigation" aria-expanded={mobileOpen} aria-controls="sidebar" onClick={() => setMobileOpen(true)}><Menu size={18} /></button>
        <div className="breadcrumbs"><span className="muted"><FolderOpen size={14} /><span>{currentProject}</span></span><span className="slash">/</span><strong>{titles.get(currentView) ?? 'Home'}</strong></div>
        <div className="topbar-spacer" />
        <div className="topbar-right"><LivingActivityCenter route={currentView}/>{import.meta.env.DEV && <button type="button" className="demo-chip demo-mode-toggle" aria-label={`Data mode: ${dataMode}. Click to switch to ${dataMode === 'demo' ? 'live' : 'demo'}`} title="Development demo mode. Demo actions use local fixtures and never call the backend." onClick={() => setDataMode(dataMode === 'demo' ? 'live' : 'demo')}>{dataMode === 'demo' ? 'DESIGN PREVIEW' : 'LIVE MODE'}</button>}{dataMode === 'live' && <button type="button" className={`frontend-status status-text ${backendHealth.isSuccess ? 'connected' : 'disconnected'}`} title={backendHealth.isError ? apiConnectionMessage(backendHealth.error) : backendHealth.isSuccess ? 'Connected to the OpenAgent backend' : 'Checking the OpenAgent backend'} onClick={() => void backendHealth.refetch()}><span aria-hidden="true"/>{backendHealth.isSuccess ? 'Backend connected' : backendHealth.isError ? 'Backend offline' : 'Connecting…'}</button>}{dataMode === 'demo' && <span className="frontend-status status-text disconnected" title="Offline reference. Development demo mode uses local fixtures only; no backend data or actions are used."><CheckCircle2 size={13} aria-hidden="true"/>Offline reference</span>}<button className="icon-button" aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>{theme === 'dark' ? <Sun size={16}/> : <Moon size={16}/>}</button><button className="icon-button notification-trigger" aria-label="Open notifications" aria-haspopup="dialog" aria-expanded={notificationsOpen} onClick={() => setNotificationsOpen(true)}><Bell size={16}/>{notifications.some(item => !item.read_at) && <span className="notification-indicator" aria-label="Unread notifications"/>}</button><button className="icon-button" aria-label="Open command palette" onClick={() => setPaletteOpen(true)}><Command size={16}/></button></div>
      </header>
      <div className="content-area" id="content" onClick={onContentClick}>
        <LazyMotion features={domAnimation}>
          <AnimatePresence mode="wait" initial={false}>
            <m.div className={`reference-screen-motion ${currentView === 'agents' ? 'agent-screen-motion' : ''}`} key={currentView} initial={reduceMotion ? false : { opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: reduceMotion ? 0 : 0.2, ease: 'easeOut' }}>
              <Suspense fallback={<div className="page"><div className="skeleton" aria-label={`Loading ${titles.get(currentView)}`} /></div>}>
                {page}
              </Suspense>
            </m.div>
          </AnimatePresence>
        </LazyMotion>
      </div>
    </main>
    <Dialog.Root open={notificationsOpen} onOpenChange={setNotificationsOpen}>
      <Dialog.Portal>
        <AnimatePresence>
          {notificationsOpen && <Dialog.Overlay asChild><m.div className="second-app-dialog-overlay" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: reduceMotion ? 0 : 0.16 }} /></Dialog.Overlay>}
        </AnimatePresence>
        <Dialog.Content asChild>
          <m.div className="second-app-dialog notification-dialog" initial={reduceMotion ? false : { opacity: 0, y: -8, scale: 0.99 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -5 }} transition={{ duration: reduceMotion ? 0 : 0.18 }}>
            <div className="notification-dialog-header"><div><Dialog.Title className="second-app-dialog-title">Notifications</Dialog.Title><Dialog.Description className="second-app-dialog-description">Updates from your scheduled tasks.</Dialog.Description></div><button className="icon-button" aria-label="Refresh notifications" onClick={() => void notificationsQuery.refetch()} disabled={notificationsQuery.isFetching}><Activity size={15}/></button></div>
            {notificationsQuery.isLoading ? <div className="notification-state" role="status">Loading notifications…</div> : notificationsQuery.isError ? <div className="notification-state" role="alert"><p>{apiConnectionMessage(notificationsQuery.error)}</p><button className="btn sm" onClick={() => void notificationsQuery.refetch()}>Retry</button></div> : notifications.length ? <div className="notification-list" role="list">{notifications.map((item, index) => { const id = String(item.id ?? ''); const title = String(item.title ?? 'Notification'); return <article className="notification-item" role="listitem" key={id || index}><span className={`notification-level ${String(item.level ?? 'info')}`} aria-hidden="true"/><div className="notification-item-copy"><strong>{title}</strong>{typeof item.body === 'string' && item.body && <p>{item.body}</p>}<time>{String(item.created_at ?? '')}</time></div>{!item.read_at && <button className="notification-unread" type="button" disabled={!id || markNotificationReadMutation.isPending} onClick={() => markNotificationReadMutation.mutate(id)}>Mark read</button>}</article>; })}</div> : <div className="notification-state"><Bell size={18}/><strong>No notifications</strong><p>Task updates will appear here when the backend records them.</p></div>}
            {markNotificationReadMutation.error && <p className="notification-error" role="alert">{apiConnectionMessage(markNotificationReadMutation.error)}</p>}
          </m.div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
    <Dialog.Root open={paletteOpen} onOpenChange={setPaletteOpen}>
      <Dialog.Portal>
        <AnimatePresence>
          {paletteOpen && <Dialog.Overlay asChild><m.div className="second-app-dialog-overlay" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: reduceMotion ? 0 : 0.16 }} /></Dialog.Overlay>}
        </AnimatePresence>
        <Dialog.Content asChild>
          <m.div className="second-app-dialog" initial={reduceMotion ? false : { opacity: 0, y: -8, scale: 0.99 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -5 }} transition={{ duration: reduceMotion ? 0 : 0.18 }}>
            <Dialog.Title className="second-app-dialog-title">Search OpenAgent</Dialog.Title>
            <Dialog.Description className="second-app-dialog-description">Find a workspace.</Dialog.Description>
            <label className="second-search-input"><Search size={16} /><input autoFocus value={query} onChange={event => setQuery(event.currentTarget.value)} placeholder="Search workspaces" /></label>
            <div className="second-search-results">{filtered.map(([id, label, Icon]) => <button key={id} className="nav-link" onClick={() => go(id)}><Icon size={16} /><span>{label}</span></button>)}</div>
            <Dialog.Close className="icon-button second-dialog-close" aria-label="Close search"><X size={16} /></Dialog.Close>
          </m.div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
    {projectPickerOpen && <div className="overlay-root" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) closeProjectPicker(); }} onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); closeProjectPicker(); return; } if (event.key === 'Tab') { const focusable = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex="-1"])')); if (!focusable.length) return; const first = focusable[0]; const last = focusable[focusable.length - 1]; if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); } else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); } } }}><section className="palette" role="dialog" aria-modal="true" aria-label="Choose workspace" onMouseDown={event => event.stopPropagation()}><div className="palette-search"><strong className="flex1">Choose a project</strong><button className="icon-button" type="button" aria-label="Close project picker" onClick={closeProjectPicker}><X size={16}/></button></div><div className="palette-list">{demoProjectsQuery.isLoading ? <div className="empty" role="status">Loading example projects…</div> : demoProjectsQuery.isError ? <div className="empty" role="alert">Example projects are unavailable. Close and try again.</div> : <><button autoFocus type="button" className="palette-option" aria-pressed={selectedDemoProject === 'Personal Workspace'} onClick={() => { setSelectedDemoProject('Personal Workspace'); closeProjectPicker(); }}><span className="square-icon sm blue"><House size={16}/></span><strong className="flex1">Personal Workspace</strong>{selectedDemoProject === 'Personal Workspace' && <Check size={15}/>}</button>{(demoProjectsQuery.data?.projects ?? []).map((project, index) => { const name = String(project.name ?? project.title ?? project.id ?? `Project ${index + 1}`); const color = String(project.color ?? 'blue'); return <button key={String(project.id ?? name)} type="button" className="palette-option" aria-pressed={selectedDemoProject === name} onClick={() => { setSelectedDemoProject(name); closeProjectPicker(); }}><span className={`square-icon sm ${color}`}><Folder size={16}/></span><span className="flex1"><strong>{name}</strong><small className="caption" style={{ display: 'block' }}>{String(project.tag ?? project.description ?? 'Demo project')}</small></span>{selectedDemoProject === name && <Check size={15}/>}</button>; })}</>}</div><div style={{ padding: '13px 15px', borderTop: '1px solid var(--line)' }}><button type="button" className="btn sm wide" onClick={() => { closeProjectPicker(); if (currentView !== 'home') go('home'); requestDemoProjectCreate(); }}><Plus size={14}/> New project (preview)</button></div></section></div>}
  </div></div>;
}
