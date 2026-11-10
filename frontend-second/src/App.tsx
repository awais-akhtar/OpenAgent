import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import { AnimatePresence, LazyMotion, domAnimation, m, useReducedMotion } from 'motion/react';
import {
  Activity, ArrowRight, BookOpen, CalendarDays, Check, CheckCircle2, Code2, Cpu,
  FlaskConical, Folder, Globe2, House, Image, Layers3, LockKeyhole, Mail,
  Menu, MessageCircle, Mic, Network, Package, ChevronLeft, ChevronRight, BookMarked,
  Plus, Search, Settings2, Shield, SlidersHorizontal, Terminal, Workflow, X, Bell, Sun, Moon, User, Zap,
} from 'lucide-react';
import * as Dialog from './components/ui/dialog';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useLocation, useNavigate } from 'react-router-dom';
import { useWorkspaceState } from './state';
import { apiGet, apiPost, apiConnectionMessage } from './api/client';
import { demoAwareRequest } from './demo/api';

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
  ['gallery', 'Image Studio', Image], ['cookbook', 'Model Cookbook', BookMarked],
  ['ultramodel', 'Ultra Model', Zap], ['personalos', 'Personal OS', User], ['readiness', 'Production Readiness', Zap],
  ['g25advanced', 'Advanced AI', Layers3], ['voiceconv', 'Voice conversations', Mic], ['modelserve', 'Model serving', Cpu],
  ['factprovenance', 'Fact provenance', Activity],
] as const;
const titles = new Map<string, string>([...primary, ...advanced].map(([id, name]) => [id, name]));
titles.set('settings', 'Settings');
titles.set('tools', 'All tools');
const allViews = [...primary, ...advanced, ['settings', 'Settings', Settings2], ['tools', 'All tools', Layers3]] as const;

const getView = (search: string) => {
  const requested = new URLSearchParams(search).get('view') || 'home';
  return titles.has(requested) ? requested : 'home';
};

function ToolsScreen() {
  const [query, setQuery] = useState('');
  const groups = [
    { name: 'Everyday', ids: ['home', 'chat', 'assistant', 'agents', 'calendar'], details: ['Live local telemetry and runtime', 'Conversations and projects', 'Tasks, goals and priorities', 'Runs and collaboration', 'Plan and schedule'] },
    { name: 'Create & research', ids: ['research', 'files', 'knowledge', 'codeq', 'voice', 'science'], details: ['Evidence and reports', 'Documents and previews', 'Indexed project sources', 'Coding and Git review', 'Recordings and transcripts', 'Datasets and experiments'] },
    { name: 'Run & connect', ids: ['automations', 'communications', 'scraper', 'finetune', 'engine'], details: ['Scheduled workflows', 'Connected messages', 'Permitted extraction jobs', 'Datasets and training', 'Compute compatibility'] },
    { name: 'Platform & privacy', ids: ['models', 'security', 'vault', 'twin', 'factprovenance', 'developer', 'settings'], details: ['Providers and capabilities', 'Access and audit', 'Protected credentials', 'Personal context', 'Fact sources, retraction and restore', 'Integrations and APIs', 'Workspace preferences'] },
    { name: 'Media & models', ids: ['gallery', 'cookbook', 'ultramodel', 'personalos'], details: ['Image library and editing', 'Model hosting and serving', 'Ultra-scale inference control', 'Personal workspace preferences'] },
    { name: 'Advanced services', ids: ['g25advanced', 'voiceconv', 'modelserve'], details: ['Documents, graphs, retrieval and reasoning', 'Multi-turn voice conversations', 'Serving plans and ultra-model sessions'] },
  ];
  // Derived from the groups above: a hardcoded figure silently disagreed
  // with the grid once workspaces were added.
  const viewCount = groups.reduce((total, group) => total + group.ids.length, 0);
  const needle = query.trim().toLowerCase();
  return <div className="p-workspace p-hub"><header className="p-titlebar"><div><span className="p-kicker"><Layers3 className="icon xs"/>Browse workspace</span><h1>All tools</h1><p>Everything you can do in OpenAgent, organised around your work.</p></div><span className="p-fixture"><Layers3 className="icon xs"/>{viewCount} existing feature views</span></header><label className="p-hub-search"><Search className="icon sm"/><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Find a tool or capability…" aria-label="Find a tool or capability"/><span>{needle ? 'Filtered' : `${viewCount} tools`}</span></label>{groups.map(group => {const items = group.ids.map((id, index) => ({ id, detail: group.details[index], item: allViews.find(([key]) => key === id) })).filter(entry => entry.item && (!needle || `${entry.id} ${entry.item[1]} ${entry.detail} ${group.name}`.toLowerCase().includes(needle))); if (!items.length) return null; return <section className="p-tools-group" key={group.name}><div className="p-section-heading"><h2>{group.name}</h2><span>{items.length} {items.length === 1 ? 'tool' : 'tools'}</span></div><div className="p-tools-grid">{items.map(({ id, detail, item }) => { if (!item) return null; const [, routeLabel, Icon] = item; const label = id === 'assistant' ? 'Personal assistant' : routeLabel; return <button key={id} type="button" className="p-tool-link" data-nav={id}><span className="square-icon"><Icon size={18}/></span><span><strong>{label}</strong><small>{detail}</small></span><ArrowRight className="icon sm" aria-hidden="true"/></button>; })}</div></section>; })}{needle && !groups.some(group => group.ids.some((id, i) => `${id} ${titles.get(id)} ${group.details[i]} ${group.name}`.toLowerCase().includes(needle))) && <div className="p-empty" role="status"><Search className="icon lg"/><h3>No matching tools</h3><p>Try another name or capability.</p></div>}</div>;
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
  const selectedDemoProject = useWorkspaceState(state => state.selectedDemoProject);
  const setSelectedDemoProject = useWorkspaceState(state => state.setSelectedDemoProject);
  const requestDemoProjectCreate = useWorkspaceState(state => state.requestDemoProjectCreate);
  const reduceMotion = useReducedMotion() || reduceMotionPreference;
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
    const onOpenPalette = () => setPaletteOpen(true);
    window.addEventListener('openagent:open-command-palette', onOpenPalette);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('openagent:open-command-palette', onOpenPalette);
    };
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
      <m.aside ref={sidebarRef} inert={mobileLayout && !mobileOpen} onClick={onContentClick} className={`sidebar ${collapsed ? 'collapsed' : ''} ${mobileOpen ? 'mobile-open' : ''}`} id="sidebar" animate={{ width: collapsed ? 72 : 170, flexBasis: collapsed ? 72 : 170 }} transition={reduceMotion ? { duration: 0 } : { type: 'spring', stiffness: 360, damping: 34 }} aria-label="Primary navigation">
        <div className="sidebar-top">
          <div className="brand"><span className="brand-mark" aria-hidden="true" /><div className="sidebar-copy"><strong>OpenAgent</strong><small>Local AI</small></div>
            <button className="icon-button sidebar-collapse" aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} onClick={toggleCollapsed}>{collapsed ? <ChevronRight size={17} /> : <ChevronLeft size={17} />}</button>
          </div>
        </div>
        <div className="sidebar-scroll">
          <nav className="nav-group" aria-label="Main">
            {primary.map(([id, label, Icon]) => <button key={id} className={`nav-link ${currentView === id ? 'active' : ''}`} data-nav={id} aria-current={currentView === id ? 'page' : undefined} title={collapsed ? label : undefined}>{currentView === id && <m.span layoutId="active-navigation" className="nav-active-indicator" transition={reduceMotion ? { duration: 0 } : { type: 'spring', stiffness: 500, damping: 36 }} />}<Icon className="icon" size={18} /><span className="nav-label">{label}</span></button>)}
            <button className={`nav-link ${currentView === 'tools' ? 'active' : ''}`} data-nav="tools" title={collapsed ? 'More tools' : undefined}>{currentView === 'tools' && <m.span layoutId="active-navigation" className="nav-active-indicator" transition={reduceMotion ? { duration: 0 } : { type: 'spring', stiffness: 500, damping: 36 }} />}<Layers3 size={18} /><span className="nav-label">More tools</span></button>
          </nav>
        </div>
        <div className="sidebar-foot"><div className="row" style={{ gap: 4 }}><button className={`nav-link ${currentView === 'settings' ? 'active' : ''}`} data-nav="settings" title={collapsed ? 'Settings' : undefined}>{currentView === 'settings' && <m.span layoutId="active-navigation" className="nav-active-indicator" transition={reduceMotion ? { duration: 0 } : { type: 'spring', stiffness: 500, damping: 36 }} />}<Settings2 size={18} /><span className="nav-label">Settings</span></button><button className="icon-button" aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>{theme === 'dark' ? <Sun size={16}/> : <Moon size={16}/>}</button></div></div>
        <button className="icon-button mobile-sidebar-close" aria-label="Close navigation" onClick={() => setMobileOpen(false)}><X size={18} /></button>
      </m.aside>
    </LazyMotion>
    <AnimatePresence>{mobileOpen && <m.button className="mobile-backdrop visible" aria-label="Close navigation" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: reduceMotion ? 0 : 0.18 }} onClick={() => setMobileOpen(false)} />}</AnimatePresence>
    <main className="main">
      <header className="topbar">
        <button ref={menuButtonRef} className="icon-button mobile-menu-button" aria-label="Open navigation" aria-expanded={mobileOpen} aria-controls="sidebar" onClick={() => setMobileOpen(true)}><Menu size={18} /></button>
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
