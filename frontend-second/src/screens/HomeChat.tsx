import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { handleTabListKeyDown, tabIndex } from '../components/tabKeyboard';
import { apiConnectionMessage, apiGet, apiPatch, apiPost, apiStreamPost, type StreamEvent } from '../api/client';
import { useWorkspaceState } from '../state';
import { demoQuery } from '../demo/api';
import { demoProvider } from '../demo/provider';
import {
  Activity, ArrowRight, Archive, Bot, BookOpen, Box, CalendarDays, Check, CheckCircle2, ChevronRight,
  Clock3, Code2, Folder, Globe2, Layers3, MessageCircle, MoreHorizontal, Pin,
  FileCheck2, GitPullRequest, Home, Info, Maximize2, Monitor, Network, Paperclip, PanelRight, Plus, Search, Send, Shield, Sparkles, Sun, Terminal, X,
  Zap, type LucideIcon,
} from 'lucide-react';

type Go = (id: string) => void;
type ScreenProps = { go?: Go };
const noNavigation: Go = () => undefined;
const quickLinks = [
  { title: 'Start a new chat', detail: 'Ask, plan and solve problems', id: 'chat', Icon: MessageCircle, tone: 'blue' },
  { title: 'Create an agent', detail: 'Configure a specialist assistant', id: 'agents', Icon: Network, tone: 'violet' },
  { title: 'Start research', detail: 'Explore a topic with sources', id: 'research', Icon: Search, tone: 'green' },
  { title: 'Set up automation', detail: 'Schedule repeatable work', id: 'automations', Icon: Zap, tone: 'orange' },
];
const pinned = [
  { title: 'Web search', id: 'research', Icon: Globe2 }, { title: 'CodeQ', id: 'codeq', Icon: Code2 },
  { title: 'Files', id: 'files', Icon: Folder }, { title: 'Developer', id: 'developer', Icon: Terminal },
  { title: 'Knowledge', id: 'knowledge', Icon: BookOpen }, { title: 'Calendar', id: 'calendar', Icon: CalendarDays },
];
const homeActivityIcons: Record<string, LucideIcon> = {
  'git-pull-request': GitPullRequest, search: Search, bot: Bot, 'file-check': FileCheck2, activity: Activity,
};
const homeAgentIcons: Record<string, LucideIcon> = { sparkles: Sparkles, search: Search, code: Code2, 'book-open': BookOpen, bot: Bot };
let stagedPrompt = '';
const stagedPromptStorageKey = 'openagent.chat.stagedPrompt';

function ActionButton({ children, onClick, className = '', title, nav }: { children: ReactNode; onClick?: () => void; className?: string; title?: string; nav?: string }) {
  return <button type="button" title={title} data-nav={nav} onClick={onClick} className={className}>{children}</button>;
}
function IconSquare({ children, tone = 'blue', size }: { children: ReactNode; tone?: string; size?: string }) {
  return <span className={`square-icon ${size ?? ''} ${tone}`}>{children}</span>;
}
function EmptyCard({ icon, title, text, className = '' }: { icon: ReactNode; title: string; text: string; className?: string }) {
  return <div className={`card empty ${className}`}><IconSquare>{icon}</IconSquare><strong>{title}</strong><p>{text}</p></div>;
}
function HomeStateCard({ icon, title, text, className = '', error = false, onRetry }: { icon: ReactNode; title: string; text: string; className: string; error?: boolean; onRetry?: () => void }) {
  return <div className={`card empty home-state-card ${className}`} role={error ? 'alert' : 'status'}>
    <IconSquare>{icon}</IconSquare><strong>{title}</strong><p>{text}</p>
    {onRetry && <button type="button" className="btn sm" onClick={onRetry}>Try again</button>}
  </div>;
}

export function HomeScreen({ go = noNavigation }: ScreenProps = {}) {
  const dataMode = useWorkspaceState(state => state.dataMode);
  const [prompt, setPrompt] = useState('');
  const [projectPickerOpen, setProjectPickerOpen] = useState(false);
  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const selectedDemoProject = useWorkspaceState(state => state.selectedDemoProject);
  const setSelectedDemoProject = useWorkspaceState(state => state.setSelectedDemoProject);
  const demoProjectCreateRequest = useWorkspaceState(state => state.demoProjectCreateRequest);
  const consumeDemoProjectCreateRequest = useWorkspaceState(state => state.consumeDemoProjectCreateRequest);
  const [homeNotice, setHomeNotice] = useState('');
  const [newProjectError, setNewProjectError] = useState('');
  const pickerDialogRef = useRef<HTMLElement>(null);
  const newProjectDialogRef = useRef<HTMLElement>(null);
  const queryClient = useQueryClient();
  const completeTask = useMutation({
    mutationFn: async ({ taskId, completed }: { taskId: string; completed: boolean }) => {
      if (dataMode !== 'demo') return apiPost('/api/assistant/tasks/complete', { task_id: taskId, project: 'default' });
      const tasks = await demoProvider.updateShellCollection<Record<string, unknown>>('homeTasks', current => {
        let found = false;
        const updated = current.map(task => {
          if (String(task.id ?? '') !== taskId) return task;
          found = true;
          return { ...task, completed };
        });
        if (!found) throw new Error('That demo task could not be found.');
        return updated;
      });
      const task = tasks.find(item => String(item.id ?? '') === taskId);
      if (!task) throw new Error('That demo task could not be found.');
      return { task };
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['assistant', 'tasks'] }),
  });
  const createDemoProject = useMutation({
    mutationFn: async (name: string) => demoProvider.updateShellCollection<Record<string, unknown>>('projects', current => {
      const normalized = name.trim().toLocaleLowerCase();
      if (current.some(project => String(project.name ?? '').trim().toLocaleLowerCase() === normalized)) {
        throw new Error('A project with this name already exists. Choose a different name.');
      }
      const slug = name.trim().toLocaleLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'project';
      return [...current, { id: `demo-project-${slug}`, name: name.trim(), tag: 'Demo project', color: 'blue', files: 0, updated: 'Just now' }];
    }),
    onSuccess: projects => {
      queryClient.setQueryData(['home', 'projects', 'demo'], { projects });
      void queryClient.invalidateQueries({ queryKey: ['home', 'projects', 'demo'] });
      const added = projects.at(-1);
      if (added) setSelectedDemoProject(String(added.name ?? 'Personal Workspace'));
      setNewProjectName('');
      setNewProjectError('');
      setNewProjectOpen(false);
      setProjectPickerOpen(true);
      setHomeNotice('New local project created.');
    },
    onError: error => setNewProjectError(error instanceof Error ? error.message : 'Could not create this local demo project.'),
  });
  const queryOptions = { retry: false as const, refetchInterval: 30_000 };
  const projectsQuery = useQuery({ queryKey: ['home', 'projects', dataMode], queryFn: ({ signal }) => demoQuery<Record<string, unknown>>(dataMode, 'shell.projects', s => apiGet<{ projects?: unknown[] }>('/api/codebase/projects?project=default', s), signal).then(data => dataMode === 'demo' ? { projects: Array.isArray(data) ? data : Array.isArray(data.projects) ? data.projects : [] } : data), ...queryOptions });
  // Home's reference-only recent-chat fixtures must not share a cache key with
  // Chat's actual conversation collection. Otherwise fixture rows without IDs
  // can leak into Chat's selector and be mistaken for saved conversations.
  const conversationsQuery = useQuery({ queryKey: ['home', 'recent-conversations', dataMode], queryFn: async ({ signal }): Promise<{ conversations?: unknown[] }> => dataMode === 'demo' ? { conversations: await demoQuery<Record<string, unknown>>(dataMode, 'shell.chats', s => apiGet<Record<string, unknown>>('/api/conversations?project=default&limit=5', s), signal).then(data => Array.isArray(data) ? data : Array.isArray(data.chats) ? data.chats : []) } : apiGet<{ conversations?: unknown[] }>('/api/conversations?project=default&limit=5', signal), ...queryOptions });
  const agentsQuery = useQuery({ queryKey: ['agents', dataMode], queryFn: ({ signal }) => demoQuery<Record<string, unknown>>(dataMode, 'shell.agents', s => apiGet<{ agents?: unknown[] }>('/api/agents', s), signal).then(data => dataMode === 'demo' ? { agents: (Array.isArray(data) ? data : Array.isArray(data.agents) ? data.agents : []).map((item, index) => { const agent = item && typeof item === 'object' ? item as Record<string, unknown> : {}; return { ...agent, id: agent.id ?? `demo-agent-${index + 1}`, description: agent.description ?? agent.role ?? '' }; }) } : data), ...queryOptions });
  const runsQuery = useQuery({ queryKey: ['agent-runs', dataMode], queryFn: ({ signal }) => dataMode === 'demo' ? demoQuery<Record<string, unknown>>('demo', 'shell.runs', s => apiGet<{ runs?: unknown[] }>('/api/agent-runs?project=default&limit=5', s), signal).then(data => ({ runs: Array.isArray(data) ? data : Array.isArray(data.runs) ? data.runs : [] })) : Promise.resolve({ runs: [] }), ...queryOptions, enabled: dataMode === 'demo' });
  const eventsQuery = useQuery({ queryKey: ['home', 'activity', dataMode], queryFn: async ({ signal }): Promise<{ events?: unknown[] }> => dataMode === 'demo' ? { events: await demoQuery<Record<string, unknown>>(dataMode, 'shell.activity', s => apiGet<Record<string, unknown>>('/api/events?project=default&limit=5', s), signal).then(data => Array.isArray(data) ? data : Array.isArray(data.activity) ? data.activity : []) } : apiGet<{ events?: unknown[] }>('/api/events?project=default&limit=5', signal), ...queryOptions });
  const tasksQuery = useQuery({ queryKey: ['assistant', 'tasks', dataMode], queryFn: async ({ signal }) => {
    if (dataMode !== 'demo') return apiGet<{ tasks?: unknown[] }>('/api/assistant/tasks?project=default&limit=5', signal);
    const fixture = await demoQuery<Record<string, unknown>>(dataMode, 'shell.homeTasks', s => apiGet<{ tasks?: unknown[] }>('/api/assistant/tasks?project=default&limit=5', s), signal);
    const rows = Array.isArray(fixture) ? fixture : Array.isArray(fixture.homeTasks) ? fixture.homeTasks : [];
    return { tasks: rows.map(item => { const task = item && typeof item === 'object' ? item as Record<string, unknown> : {}; return { ...task, title: task.title ?? 'Task', status: task.completed === true ? 'Completed' : 'Todo' }; }) };
  }, ...queryOptions });
  const today = useMemo(() => {
    const start = new Date(); start.setHours(0, 0, 0, 0);
    const end = new Date(start); end.setDate(end.getDate() + 1); end.setMilliseconds(-1);
    return { key: start.toISOString().slice(0, 10), from: start.toISOString(), to: end.toISOString() };
  }, []);
  const calendarQuery = useQuery({ queryKey: ['calendar', 'events', 'home', today.key, dataMode], queryFn: ({ signal }) => demoQuery<Record<string, unknown>>(dataMode, 'shell.calendar', s => apiGet<{ events?: unknown[] }>(`/api/calendar/events?project=default&from=${encodeURIComponent(today.from)}&to=${encodeURIComponent(today.to)}&limit=5`, s), signal).then(data => dataMode === 'demo' ? { events: Array.isArray(data) ? data : Array.isArray(data.events) ? data.events : [] } : data), ...queryOptions });
  const projects = records(projectsQuery.data?.projects);
  useEffect(() => {
    if (!homeNotice) return;
    const timer = window.setTimeout(() => setHomeNotice(''), 3600);
    return () => window.clearTimeout(timer);
  }, [homeNotice]);
  useEffect(() => {
    if (!demoProjectCreateRequest || dataMode !== 'demo') return;
    setProjectPickerOpen(false);
    setNewProjectName('');
    setNewProjectError('');
    setNewProjectOpen(true);
    consumeDemoProjectCreateRequest(demoProjectCreateRequest);
  }, [consumeDemoProjectCreateRequest, dataMode, demoProjectCreateRequest]);
  useEffect(() => {
    const currentDialog = newProjectOpen ? newProjectDialogRef.current : projectPickerOpen ? pickerDialogRef.current : null;
    if (!currentDialog) return;
    const getFocusable = () => Array.from(currentDialog.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'));
    getFocusable()[0]?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        if (newProjectOpen) setNewProjectOpen(false); else setProjectPickerOpen(false);
      } else if (event.key === 'Tab') {
        const focusable = getFocusable();
        const first = focusable[0]; const last = focusable.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [newProjectOpen, projectPickerOpen]);
  const conversations = records(conversationsQuery.data?.conversations);
  const agents = records(agentsQuery.data?.agents);
  const agentRuns = records(runsQuery.data?.runs);
  const contextEvents = records(eventsQuery.data?.events);
  const assistantTasks = records(tasksQuery.data?.tasks);
  const calendarEvents = records(calendarQuery.data?.events);
  const activity = dataMode === 'demo'
    ? contextEvents.map(item => ({ kind: String(item.detail ?? 'Activity'), label: String(item.title ?? 'Activity'), when: String(item.time ?? ''), detail: String(item.detail ?? ''), icon: String(item.icon ?? 'activity'), tone: String(item.tone ?? 'blue') }))
    : [
      ...agentRuns.map(item => ({ kind: String(item.kind ?? 'Run'), label: String(item.goal ?? item.task ?? item.info ?? item.agent ?? 'Agent run'), when: String(item.created_at ?? item.time ?? ''), detail: '', icon: 'activity', tone: 'blue' })),
      ...contextEvents.map(item => ({ kind: String(item.type ?? item.source ?? 'Activity'), label: String(item.title ?? item.content ?? 'Activity'), when: String(item.created_at ?? item.time ?? ''), detail: String(item.detail ?? ''), icon: 'activity', tone: 'blue' })),
      ...conversations.map(item => ({ kind: 'Conversation', label: String(item.title ?? 'Conversation'), when: String(item.updated_at ?? item.time ?? item.created_at ?? ''), detail: '', icon: 'activity', tone: 'blue' })),
    ].sort((a, b) => (Date.parse(b.when) || 0) - (Date.parse(a.when) || 0)).slice(0, 5);
  const allActivityUnavailable = dataMode === 'live'
    ? eventsQuery.isError && conversationsQuery.isError
    : runsQuery.isError && eventsQuery.isError && conversationsQuery.isError;
  const submit = (event?: FormEvent) => { event?.preventDefault(); if (prompt.trim()) { stagedPrompt = prompt.trim(); try { sessionStorage.setItem(stagedPromptStorageKey, stagedPrompt); } catch { /* in-memory handoff remains available */ } go('chat'); } };
  const enter = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => { if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) submit(); };
  const selectDemoProject = (name: string) => {
    if (dataMode !== 'demo') return;
    setSelectedDemoProject(name);
    setProjectPickerOpen(false);
    setHomeNotice('Project selected in the local demo picker. Live project scope is unchanged.');
  };
  const submitNewDemoProject = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = newProjectName.trim();
    if (dataMode !== 'demo') { setNewProjectError('Project creation is unavailable until a workspace service is connected.'); return; }
    if (!name) { setNewProjectError('Enter a project name.'); return; }
    if (name.length > 80) { setNewProjectError('Project names must be 80 characters or fewer.'); return; }
    setNewProjectError('');
    createDemoProject.mutate(name);
  };
  return <div className="page">
    <div className="home-grid">
      <section>
        <div className="home-intro"><IconSquare tone="orange"><Sun size={19}/></IconSquare><div><h1>Welcome to OpenAgent</h1><p>What would you like to work on?</p></div></div>
        <form className="home-composer" onSubmit={submit}>
          <textarea value={prompt} onChange={e => setPrompt(e.target.value)} onKeyDown={enter} aria-label="Ask OpenAgent" placeholder="Ask OpenAgent anything..." />
          <div className="composer-tools"><button type="button" className="btn sm icon-only" title={dataMode === 'demo' ? 'File attachment is a design-only control' : 'File attachment is unavailable until a workspace service is connected'} aria-label={dataMode === 'demo' ? 'Attach a file' : 'Attach a file unavailable'} disabled={dataMode !== 'demo'} onClick={() => setHomeNotice('File attachment is a design-only control. Connect upload APIs during migration.')}><Plus size={16}/></button>
            {['Agents', 'Web', 'Files', 'Research', 'CodeQ'].map((label, i) => {
              const icons = [Network, Globe2, Folder, Search, Code2]; const Icon = icons[i]; const target = ['agents', 'research', 'files', 'research', 'codeq'][i];
              return <ActionButton key={label} className="btn sm" nav={target} onClick={() => go(target)}><Icon size={14}/>{label}</ActionButton>;
            })}
            <button className="send-button" type="submit" aria-label="Open prompt in chat"><Send size={16}/></button>
          </div>
        </form>
        <div className="grid-4 section-space">{quickLinks.map(({ title, detail, id, Icon, tone }) => <button type="button" data-nav={id} key={id} onClick={() => go(id)} className="card shortcut-card hover-card"><IconSquare tone={tone}><Icon size={18}/></IconSquare><h3>{title}</h3><p>{detail}</p></button>)}</div>
        <div className="section-head section-space"><h2><Box size={18}/> Projects</h2><ActionButton onClick={() => setProjectPickerOpen(true)} className="text-link">View projects <ArrowRight size={13}/></ActionButton></div>
        {projectsQuery.isLoading ? <HomeStateCard className="home-project-empty" icon={<Folder size={20}/>} title="Loading projects" text="Fetching projects for this workspace." /> : projectsQuery.isError ? <HomeStateCard className="home-project-empty" icon={<Folder size={20}/>} title="Projects unavailable" text={apiConnectionMessage(projectsQuery.error)} error onRetry={() => void projectsQuery.refetch()} /> : <div className="grid-4">{projects.length ? projects.slice(0, 4).map((project, index) => <button key={String(project.id ?? project.name ?? index)} className="card project-card hover-card" type="button" onClick={() => go('codeq')}><IconSquare tone={String(project.color ?? 'blue')}><Folder size={18}/></IconSquare><h3>{String(project.name ?? project.title ?? project.id ?? 'Project')}</h3><p className="muted small truncate" title={String(project.tag ?? project.description ?? project.path ?? project.root_path ?? '')}>{String(project.tag ?? project.description ?? project.path ?? project.root_path ?? 'Workspace project')}</p><div className="row between section-space"><span className="caption">{String(project.updated ?? (relativeTimestamp(project.updated_at) || 'Connected project'))}</span><MoreHorizontal size={15}/></div></button>) : <div className="card project-card home-no-project" role="status"><IconSquare><Folder size={18}/></IconSquare><h3>No projects yet</h3><p className="muted small">Projects linked to this workspace will appear here.</p></div>}</div>}
        <div className="section-head section-space"><h2><Clock3 size={18}/> Recent activity</h2><ActionButton nav="tools" onClick={() => go('tools')} className="text-link">View all</ActionButton></div>
        {dataMode === 'live' && <p className="caption" role="status">Agent run history is shown in Agents when this account has the explicit agents:read scope for the active project. Home does not load agent runs.</p>}
        {runsQuery.isLoading || eventsQuery.isLoading || conversationsQuery.isLoading ? <HomeStateCard className="home-activity-empty" icon={<Activity size={20}/>} title="Loading recent activity" text="Checking available events and conversations." /> : allActivityUnavailable ? <HomeStateCard className="home-activity-empty" icon={<Activity size={20}/>} title="Activity unavailable" text={apiConnectionMessage(eventsQuery.error ?? conversationsQuery.error ?? runsQuery.error)} error onRetry={() => { void runsQuery.refetch(); void eventsQuery.refetch(); void conversationsQuery.refetch(); }} /> : <>{(runsQuery.isError || eventsQuery.isError || conversationsQuery.isError) && <p className="caption" role="alert">Some activity could not be loaded: {apiConnectionMessage(eventsQuery.error ?? conversationsQuery.error ?? runsQuery.error)}</p>}{activity.length ? <div className="card activity-list">{activity.map((item, index) => { const Icon = homeActivityIcons[item.icon] ?? Activity; return <div className="list-row" key={`${item.kind}-${item.when}-${index}`}><IconSquare tone={item.tone} size="sm"><Icon size={16}/></IconSquare><div className="flex1"><div className="row-title">{item.label}</div><div className="row-sub">{item.detail || item.kind}</div></div><span className="caption">{item.when}</span></div>; })}</div> : <EmptyCard className="home-activity-empty" icon={<Activity size={20}/>} title="No recent activity" text={dataMode === 'live' ? 'No recent events or conversations were returned.' : 'The backend returned no recent agent runs, events or conversations.'} />}</>}
      </section>
      <aside className="right-stack home-right">
        <section className="card" style={{ padding: 15 }}><div className="section-head"><h2><CheckCircle2 size={18}/> Today</h2><ActionButton nav="calendar" onClick={() => go('calendar')} className="text-link">Calendar</ActionButton></div>
          {tasksQuery.isLoading || calendarQuery.isLoading ? <HomeStateCard className="home-today-empty" icon={<CheckCircle2 size={20}/>} title="Loading today" text="Checking tasks and calendar events." /> : tasksQuery.isError && calendarQuery.isError ? <HomeStateCard className="home-today-empty" icon={<CheckCircle2 size={20}/>} title="Today unavailable" text={apiConnectionMessage(tasksQuery.error ?? calendarQuery.error)} error onRetry={() => { void tasksQuery.refetch(); void calendarQuery.refetch(); }} /> : <>{(tasksQuery.isError || calendarQuery.isError) && <p className="caption" role="alert">Some activity could not be loaded: {apiConnectionMessage(tasksQuery.error ?? calendarQuery.error)}</p>}{completeTask.error && <p className="caption" role="alert">Task could not be completed: {apiConnectionMessage(completeTask.error)}</p>}{assistantTasks.length || calendarEvents.length ? <div>{assistantTasks.slice(0, 4).map(task => {
            const taskId = String(task.id ?? '');
            const taskTitle = String(task.title ?? 'Task');
            const isComplete = typeof task.completed === 'boolean' ? task.completed : String(task.status ?? '').toLowerCase() === 'completed';
            return <label className="task-check" key={taskId || taskTitle}><input className="checkbox" type="checkbox" checked={isComplete} disabled={completeTask.isPending || (dataMode === 'live' && isComplete)} onChange={event => { if (taskId) completeTask.mutate({ taskId, completed: event.currentTarget.checked }); }} aria-label={`Mark ${taskTitle} ${isComplete ? 'incomplete' : 'complete'}`} /><span className="flex1">{taskTitle}</span>{isComplete && <span className="tag green">Done</span>}</label>;
          })}{calendarEvents.slice(0, 4).map(item => <div className="list-row" key={`calendar-${String(item.id)}`}><CalendarDays size={16}/><span className="flex1"><strong>{String(item.title ?? 'Calendar event')}</strong><small>{String(item.starts_at ?? '')}</small></span></div>)}</div> : <div className="empty home-today-empty" role="status"><CheckCircle2 size={20}/><strong>Nothing scheduled today</strong></div>}</>}
        </section>
        <section className="card" style={{ padding: 15 }}><div className="section-head"><h2><Network size={18}/> Your agents</h2><ActionButton nav="agents" onClick={() => go('agents')} className="text-link">View all</ActionButton></div>{agentsQuery.isLoading ? <HomeStateCard className="home-agents-empty" icon={<Bot size={20}/>} title="Loading agents" text="Checking the available workspace agents." /> : agentsQuery.isError ? <HomeStateCard className="home-agents-empty" icon={<Bot size={20}/>} title="Agents unavailable" text={apiConnectionMessage(agentsQuery.error)} error onRetry={() => void agentsQuery.refetch()} /> : agents.length ? agents.map((agent, index) => { const AgentIcon = homeAgentIcons[String(agent.icon ?? '')] ?? Bot; return <div className="agent-mini" key={String(agent.id ?? agent.name ?? index)}><span className={`square-icon sm ${String(agent.tone ?? 'violet')}`}><AgentIcon size={16}/></span><div className="flex1"><div className="agent-name">{String(agent.name ?? agent.id ?? 'Agent')}</div><div className="caption home-agent-description">{String(agent.description ?? agent.role ?? '')}</div></div><span className="status-dot" title="Illustration only"/></div>; }) : <EmptyCard className="home-agents-empty" icon={<Bot size={20}/>} title="No agents available" text="Workspace agents will appear here when available." />}</section>
        <section className="card" style={{ padding: 15 }}><div className="section-head"><h2><Layers3 size={18}/> Pinned tools</h2></div><div className="tool-grid">{pinned.map(({ title, id, Icon }) => <button key={id} type="button" data-nav={id} className="tool-tile" onClick={() => go(id)}><Icon size={21}/>{title}</button>)}</div></section>
      </aside>
    </div>
    {projectPickerOpen && <div className="overlay-root" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) setProjectPickerOpen(false); }}><section className="palette" role="dialog" aria-modal="true" aria-label="Choose workspace" ref={pickerDialogRef} onMouseDown={event => event.stopPropagation()}><div className="palette-search"><strong className="flex1">{dataMode === 'demo' ? 'Choose a project' : 'Project switching unavailable'}</strong><button className="icon-button" type="button" aria-label="Close" onClick={() => setProjectPickerOpen(false)}><X size={16}/></button></div>{dataMode === 'demo' ? <><div className="palette-list"><button type="button" className="palette-option" aria-pressed={selectedDemoProject === 'Personal Workspace'} onClick={() => selectDemoProject('Personal Workspace')}><IconSquare size="sm" tone="blue"><Home size={16}/></IconSquare><strong className="flex1">Personal Workspace</strong>{selectedDemoProject === 'Personal Workspace' && <Check size={15}/>}</button>{projects.map((project, index) => { const name = String(project.name ?? project.title ?? project.id ?? `Project ${index + 1}`); return <button type="button" className="palette-option" aria-pressed={selectedDemoProject === name} key={String(project.id ?? name)} onClick={() => selectDemoProject(name)}><IconSquare size="sm" tone={String(project.color ?? 'blue')}><Folder size={16}/></IconSquare><span className="flex1"><strong>{name}</strong><small className="caption" style={{ display: 'block' }}>{String(project.tag ?? project.description ?? 'Demo project')}</small></span>{selectedDemoProject === name && <Check size={15}/>}</button>; })}{projectsQuery.isLoading && <div className="empty" role="status">Loading example projects…</div>}{projectsQuery.isError && <div className="empty" role="alert">{apiConnectionMessage(projectsQuery.error)}</div>}</div><div style={{ padding: '13px 15px', borderTop: '1px solid var(--line)' }}><button type="button" className="btn sm wide" onClick={() => { setNewProjectName(''); setNewProjectError(''); setProjectPickerOpen(false); setNewProjectOpen(true); }}><Plus size={14}/> New project (preview)</button></div></> : <div className="palette-list"><p className="small muted">The current backend integration does not provide a confirmed project-switch operation. This screen keeps the existing project scope.</p><button type="button" className="btn sm" onClick={() => { setProjectPickerOpen(false); go('codeq'); }}>Open CodeQ projects</button></div>}</section></div>}
    {newProjectOpen && <div className="overlay-root" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) setNewProjectOpen(false); }}><section className="modal" role="dialog" aria-modal="true" aria-labelledby="home-create-project-title" ref={newProjectDialogRef} onMouseDown={event => event.stopPropagation()}><div className="row between"><h2 id="home-create-project-title">Create a project</h2><button className="icon-button" type="button" aria-label="Close dialog" onClick={() => setNewProjectOpen(false)}><X size={16}/></button></div><p>This design preview stores the selected project name locally.</p>{dataMode === 'demo' ? <form onSubmit={submitNewDemoProject}><div className="field"><label htmlFor="home-new-project-name">Project name</label><input id="home-new-project-name" name="name" className="input" autoFocus maxLength={80} required value={newProjectName} onChange={event => { setNewProjectName(event.target.value); setNewProjectError(''); }} placeholder="My new project" /></div>{newProjectError && <p role="alert" className="caption">{newProjectError}</p>}<div className="modal-actions"><button className="btn" type="button" onClick={() => setNewProjectOpen(false)}>Cancel</button><button className="btn primary" type="submit" disabled={createDemoProject.isPending}>{createDemoProject.isPending ? 'Creating…' : 'Create preview'}</button></div></form> : <><p role="alert">Project creation is unavailable until a workspace service is connected.</p><div className="modal-actions"><button className="btn" type="button" onClick={() => setNewProjectOpen(false)}>Close</button></div></>}</section></div>}
    {homeNotice && <div id="toast-region" aria-live="polite" aria-atomic="true"><div className="toast" role="status"><Info size={17}/><span className="flex1">{homeNotice}</span></div></div>}
  </div>;
}

type ChatLine = { role: 'user' | 'assistant'; text: string };
type ChatLineWithId = ChatLine & { id?: string; serverId?: string };
type Conversation = { id: string; title?: string; pinned?: boolean; archived?: boolean; updated_at?: string; semantic_score?: number };
type Provider = { id: string; label?: string; default_model?: string; selected_model?: string; installed_models?: string[]; available?: boolean; local?: boolean; network?: boolean; privacy_warning?: string };
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
  const [tab, setTab] = useState<'Files' | 'Preview' | 'Context'>('Preview');
  const [previewViewport, setPreviewViewport] = useState<'desktop' | 'mobile'>('desktop');
  const [previewExpanded, setPreviewExpanded] = useState(false);
  const [inspector, setInspector] = useState(true);
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
  const [conversationMenuOpen, setConversationMenuOpen] = useState(false);
  const [conversationActionMessage, setConversationActionMessage] = useState('');
  const [conversationActionError, setConversationActionError] = useState('');
  const [actionMessageId, setActionMessageId] = useState('');
  const [replayData, setReplayData] = useState<Record<string, unknown> | null>(null);
  const [replayBusy, setReplayBusy] = useState(false);
  const importInputRef = useRef<HTMLInputElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const requestSequenceRef = useRef(0);
  const waitingTimerRef = useRef<number | null>(null);
  const queryClient = useQueryClient();
  useEffect(() => {
    let prompt = stagedPrompt;
    stagedPrompt = '';
    try {
      prompt ||= sessionStorage.getItem(stagedPromptStorageKey) ?? '';
      sessionStorage.removeItem(stagedPromptStorageKey);
    } catch { /* use the in-memory handoff when storage is unavailable */ }
    if (prompt.trim()) setDraft(current => current || prompt);
  }, []);
  const conversationsQuery = useQuery({ queryKey: ['chat', 'conversations', dataMode], queryFn: ({ signal }) => dataMode === 'demo' ? Promise.resolve({ conversations: [] }) : apiGet<{ conversations?: unknown[] }>('/api/conversations?project=default&limit=100', signal), retry: false });
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
  const modelOptions = Array.from(new Set([selectedProvider?.selected_model, selectedProvider?.default_model, ...(Array.isArray(selectedProvider?.installed_models) ? selectedProvider.installed_models : [])].filter((item): item is string => Boolean(item))));
  const messagesQuery = useQuery({ queryKey: ['chat', 'messages', conversationId, dataMode], queryFn: ({ signal }) => apiGet<{ messages?: unknown[] }>(`/api/messages?conversation_id=${encodeURIComponent(conversationId)}`, signal), enabled: Boolean(conversationId) && dataMode === 'live', retry: false });
  // Do not render any cached Live messages during the render that changes
  // modes; the effect below also clears their selection and local state.
  const visibleLines = dataMode === 'demo' ? [] : lines;
  useEffect(() => {
    if (streaming) return;
    const messages = records(messagesQuery.data?.messages);
    if (!messages.length) { if (!conversationId) setLines([]); return; }
    setLines(messages.flatMap(message => {
      const role = message.role === 'user' ? 'user' : message.role === 'assistant' ? 'assistant' : null;
      return role && typeof message.content === 'string' ? [{ role, text: message.content, ...(typeof message.id === 'string' ? { id: message.id, serverId: message.id } : {}) } as ChatLineWithId] : [];
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
    setConversationId(id); setLines([]);
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
    setConversationMenuOpen(false); setConversationActionMessage(''); setConversationActionError('');
    setActionMessageId(''); setReplayData(null); setReplayBusy(false); setHistoryBusyId('');
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
  const renameConversation = async (item: Conversation) => {
    const title = window.prompt('Rename conversation', item.title || 'Conversation');
    if (title === null) return;
    if (!title.trim()) { setHistoryError('Conversation title cannot be empty.'); return; }
    await updateConversation(item, { title: title.trim() });
  };
  const download = (name: string, content: BlobPart, type: string) => {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = name; anchor.click();
    URL.revokeObjectURL(url);
  };
  const exportConversation = async () => {
    if (!conversationId) return;
    setConversationActionError(''); setConversationActionMessage('Preparing export…');
    try {
      const response = await apiGet<{ export?: Record<string, unknown> }>(`/api/chat/export?conversation_id=${encodeURIComponent(conversationId)}&write_file=false`);
      if (!response.export || response.export.format !== 'openagent.conversation.v1') throw new Error('The backend returned an unsupported conversation export.');
      const title = String((response.export.conversation as Record<string, unknown> | undefined)?.title || 'conversation').replace(/[^a-z0-9-_]+/gi, '-').replace(/^-|-$/g, '').slice(0, 60) || 'conversation';
      download(`${title}.openagent.json`, JSON.stringify(response.export, null, 2), 'application/json');
      setConversationActionMessage('Conversation export downloaded.'); setConversationMenuOpen(false);
    } catch (error) { setConversationActionMessage(''); setConversationActionError(apiConnectionMessage(error)); }
  };
  const replayConversation = async () => {
    if (!conversationId) return;
    setConversationActionError(''); setConversationActionMessage(''); setReplayBusy(true);
    try {
      const response = await apiGet<{ replay?: Record<string, unknown> }>(`/api/chat/replay?conversation_id=${encodeURIComponent(conversationId)}`);
      if (!response.replay || !Array.isArray(response.replay.timeline)) throw new Error('The backend returned an invalid conversation replay.');
      setReplayData(response.replay); setConversationMenuOpen(false);
    } catch (error) { setConversationActionError(apiConnectionMessage(error)); }
    finally { setReplayBusy(false); }
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
      selectConversation(importedId); setConversationMenuOpen(false); setConversationActionMessage('Conversation imported.');
    } catch (error) { setConversationActionError(error instanceof SyntaxError ? 'The selected file is not valid JSON.' : apiConnectionMessage(error)); }
  };
  const branchFromMessage = async (line: ChatLineWithId) => {
    if (!conversationId || !line.serverId) return;
    setConversationActionError(''); setConversationActionMessage('Creating branch…');
    try {
      const response = await apiPost<{ branch?: { conversation?: { id?: string } } }>('/api/chat/branch', { conversation_id: conversationId, at_message_id: line.serverId });
      const branchId = response.branch?.conversation?.id;
      if (typeof branchId !== 'string' || !branchId) throw new Error('The backend did not return the branch conversation ID.');
      await Promise.all([queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] }), queryClient.invalidateQueries({ queryKey: ['chat', 'organized-history'] })]);
      selectConversation(branchId); setConversationActionMessage('Branch created from this message.'); setActionMessageId('');
    } catch (error) { setConversationActionMessage(''); setConversationActionError(apiConnectionMessage(error)); }
  };
  const saveReplyAsMarkdown = async (line: ChatLineWithId) => {
    setConversationActionError('');
    try {
      const response = await apiPost<{ artifact?: { filename?: string; content_type?: string; base64?: string } }>('/api/chat/artifact', { title: 'OpenAgent reply', content: line.text, format: 'md', project: 'default' });
      const artifact = response.artifact;
      if (!artifact || typeof artifact.base64 !== 'string' || typeof artifact.filename !== 'string') throw new Error('The backend did not return the generated document.');
      const bytes = Uint8Array.from(atob(artifact.base64), char => char.charCodeAt(0));
      download(artifact.filename, bytes, artifact.content_type || 'text/markdown'); setConversationActionMessage('Markdown document created and downloaded.'); setActionMessageId('');
    } catch (error) { setConversationActionError(apiConnectionMessage(error)); }
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
  const submit = async (event?: FormEvent) => {
    event?.preventDefault(); const content = draft.trim(); if (!content || streaming || uploading || dataMode !== 'live') return;
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
  };
  const enter = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => { if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) void submit(); };
  return <div className={`split-layout chat-demo-${dataMode} ${inspector ? 'has-inspector' : ''}`}>
    <section className="split-main">
      <header className="split-top"><div className="row chat-toolbar-title"><select aria-label="Conversation" className="select" value={conversationId} onChange={event => selectConversation(event.target.value)}><option key="new-conversation" value="">New conversation</option>{conversations.map(item => <option key={item.id} value={item.id}>{item.title || item.id}</option>)}</select>{conversationId && <span className="tag chat-conversation-tag">Conversation</span>}</div><div className="row"><ActionButton title={inspector ? 'Hide context panel' : 'Show context panel'} onClick={() => setInspector(!inspector)} className="icon-button"><PanelRight size={17}/></ActionButton><button type="button" title="Conversation actions" aria-label="Conversation actions" className="icon-button" aria-expanded={conversationMenuOpen} onClick={() => { setConversationMenuOpen(value => !value); setConversationActionError(''); }}><MoreHorizontal size={19}/></button>{dataMode === 'demo' && <span className="tag"><span className="status-dot"/>Offline demo · chat not connected</span>}</div></header>
      {conversationMenuOpen && <section className="card" aria-label="Conversation actions" style={{ margin: '8px 12px 0', padding: 10 }}>
        <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
          <button type="button" className="btn sm" aria-expanded={historyOpen} onClick={() => { setHistoryOpen(value => !value); setHistoryError(''); }}>Conversation history</button>
          <button type="button" className="btn sm" onClick={() => void exportConversation()} disabled={!conversationId}>Export JSON</button>
          <button type="button" className="btn sm" onClick={() => void replayConversation()} disabled={!conversationId || replayBusy}>{replayBusy ? 'Loading replay…' : 'Replay timeline'}</button>
          <input ref={importInputRef} type="file" accept=".json,application/json" aria-label="Choose conversation export" style={{ position: 'absolute', width: 1, height: 1, padding: 0, margin: -1, overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', border: 0 }} onChange={event => { const file = event.currentTarget.files?.[0]; if (file) void importConversation(file); event.currentTarget.value = ''; }} />
          <button type="button" className="btn sm" onClick={() => importInputRef.current?.click()}>Import conversation…</button>
        </div>
        {conversationActionMessage && <p className="caption" role="status">{conversationActionMessage}</p>}
        {conversationActionError && <p className="caption" role="alert">{conversationActionError}</p>}
      </section>}
      {dataMode === 'live' && replayData && <dialog open aria-label="Conversation replay" className="card" style={{ position: 'fixed', zIndex: 20, inset: '12% 10%', width: '80%', maxWidth: 900, maxHeight: '76vh', overflow: 'auto', padding: 18, boxShadow: '0 16px 48px #0003' }}><div className="row between"><h2>Conversation replay</h2><button type="button" className="icon-button" aria-label="Close replay" onClick={() => setReplayData(null)}>×</button></div><p className="caption">{String(replayData.message_count ?? 0)} messages · backend replay data</p><div>{records(replayData.timeline).map((event, index) => <article key={`${String(event.created_at)}-${index}`} className="chat-message"><span className="avatar">{event.role === 'user' ? 'Y' : '◉'}</span><div className="flex1"><div className="message-head"><strong>{String(event.role ?? 'Message')}</strong><small>{String(event.created_at ?? '')}</small></div><div className="message-body"><p>{String(event.content ?? '')}</p></div></div></article>)}</div></dialog>}
      {historyOpen && <section className="card" aria-label="Conversation history" style={{ margin: '8px 12px 0', padding: 12 }}>
        <div className="row between" style={{ gap: 8, flexWrap: 'wrap' }}>
          <label className="row" style={{ flex: '1 1 220px', minWidth: 0, gap: 8 }}><Search size={15}/><input className="field" aria-label="Search conversation history" placeholder="Search conversations" value={historySearch} onChange={event => setHistorySearch(event.target.value)} /></label>
          <button type="button" className="btn sm" aria-pressed={showArchived} onClick={() => setShowArchived(value => !value)}>{showArchived ? 'Archived' : 'Active'}</button>
        </div>
        {historyError && <p className="caption" role="alert">{historyError}</p>}
        {historyQuery.isLoading ? <p className="caption" role="status">Loading conversation history…</p> : historyQuery.isError ? <p className="caption" role="alert">{apiConnectionMessage(historyQuery.error)}</p> : history.length ? <div className="second-search-results" style={{ maxHeight: 220, overflow: 'auto' }}>{history.map(item => <div key={item.id} className="row" style={{ gap: 4, borderBottom: '1px solid var(--line)', padding: '5px 0' }}>
          <button type="button" className="nav-link truncate" aria-current={conversationId === item.id ? 'true' : undefined} style={{ flex: 1, minWidth: 0 }} onClick={() => { selectConversation(item.id); setHistoryOpen(false); }} title={item.title || item.id}>{item.title || item.id}</button>
          <button type="button" className="icon-button" aria-label={`${item.pinned ? 'Unpin' : 'Pin'} ${item.title || 'conversation'}`} title={item.pinned ? 'Unpin conversation' : 'Pin conversation'} disabled={historyBusyId === item.id} onClick={() => void updateConversation(item, { pinned: !item.pinned })}><Pin size={14}/></button>
          <button type="button" className="icon-button" aria-label={`Rename ${item.title || 'conversation'}`} title="Rename conversation" disabled={historyBusyId === item.id} onClick={() => void renameConversation(item)}><MoreHorizontal size={15}/></button>
          <button type="button" className="icon-button" aria-label={`${item.archived ? 'Restore' : 'Archive'} ${item.title || 'conversation'}`} title={item.archived ? 'Restore conversation' : 'Archive conversation'} disabled={historyBusyId === item.id} onClick={() => void updateConversation(item, { archived: !item.archived })}><Archive size={14}/></button>
        </div>)}</div> : <p className="caption" role="status">{historySearch.trim() ? 'No conversations match this search.' : showArchived ? 'No archived conversations.' : 'No saved conversations yet.'}</p>}
      </section>}
      <div className="split-scroll" ref={chatScrollRef}><div className="chat-thread">
        {messagesQuery.isLoading && <div className="caption" role="status">Loading conversation…</div>}
        {messagesQuery.isError && <div className="card padded small" role="alert">{apiConnectionMessage(messagesQuery.error)}</div>}
        {!visibleLines.length && !messagesQuery.isLoading ? <div className="empty"><IconSquare><Sparkles size={20}/></IconSquare><h2>Start a conversation</h2><p>{dataMode === 'demo' ? 'Chat is not simulated in offline Demo mode. Switch to Live to use the connected chat service.' : 'Send a message to begin.'}</p></div> : visibleLines.map((line, i) => <article className="chat-message" key={line.id ?? `${conversationId}-${i}-${line.role}`}><span className={`avatar ${line.role === 'assistant' ? 'blue' : ''}`}>{line.role === 'user' ? 'Y' : '◉'}</span><div className="flex1"><div className="message-head"><strong>{line.role === 'user' ? 'You' : 'OpenAgent'}</strong><small>{line.role}</small><button type="button" className="icon-button" aria-label={`Actions for ${line.role} message ${i + 1}`} aria-expanded={actionMessageId === line.id} disabled={!line.serverId || streaming} onClick={() => setActionMessageId(current => current === line.id ? '' : line.id || '')}><MoreHorizontal size={15}/></button></div><div className="message-body">{line.text && <p>{line.text}</p>}{waitingForContent && streaming && line.role === 'assistant' && i === visibleLines.length - 1 && <div className="chat-generation-indicator" role="status"><span aria-hidden="true"/>Working…</div>}</div>{actionMessageId === line.id && <div className="row" style={{ gap: 6, marginTop: 6 }}>{line.serverId && <button type="button" className="btn sm" onClick={() => void branchFromMessage(line)}>Branch from here</button>}{line.role === 'assistant' && <button type="button" className="btn sm" onClick={() => void saveReplyAsMarkdown(line)}>Save as Markdown</button>}</div>}</div></article>)}
        {streamError && <div className="card padded small" role="alert">{streamError}</div>}
        {completionNotice && !streaming && <div className="caption" role="status">{completionNotice}</div>}
      </div></div>
      <form className="chat-composer-outer" onSubmit={event => void submit(event)}>
        <div className="chat-composer">
          <textarea value={draft} onChange={e => setDraft(e.target.value)} onKeyDown={enter} placeholder="Message OpenAgent..." aria-label="Message OpenAgent" disabled={streaming}/>
          {(attachments.length > 0 || uploading || uploadError) && <div className="row gap6" style={{ flexWrap: 'wrap', padding: '0 12px 8px' }}>{attachments.map(attachment => <span key={attachment.id} className="tag" data-testid="chat-attachment"><Paperclip size={13}/>{attachment.name}<button type="button" className="icon-button" aria-label={`Remove ${attachment.name}`} onClick={() => setAttachments(current => current.filter(item => item.id !== attachment.id))}><span aria-hidden="true">×</span></button></span>)}{uploading && <span className="caption" role="status">Uploading attachment…</span>}{uploadError && <span className="caption" role="alert">{uploadError}</span>}</div>}
          <div className="row between" style={{ flexWrap: 'wrap' }}>
            <div className="row gap6">
              <input ref={fileInputRef} type="file" multiple aria-label="Choose files to attach" style={{ position: 'absolute', width: 1, height: 1, padding: 0, margin: -1, overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', border: 0 }} onChange={event => { if (event.currentTarget.files) void uploadFiles(event.currentTarget.files); event.currentTarget.value = ''; }} />
              <button type="button" className="icon-button" title="Attach files" aria-label="Attach a file" onClick={() => fileInputRef.current?.click()} disabled={dataMode !== 'live' || streaming || uploading || attachments.length >= 12}><Paperclip size={16}/></button>
              <button type="button" className="btn sm chat-model-trigger" aria-expanded={modelControlsOpen} aria-controls="chat-model-controls" onClick={() => setModelControlsOpen(value => !value)}><Bot size={14}/>{modelId || selectedProvider?.selected_model || selectedProvider?.default_model || 'Select model'}{dataMode !== 'demo' && <ChevronRight size={13}/>}</button>
              <ActionButton nav="tools" onClick={() => go('tools')} className="btn sm"><Globe2 size={14}/> Tools</ActionButton>
            </div>
            {streaming ? <button className="btn sm" type="button" onClick={stopStream}>Cancel</button> : <button className="send-button" type="submit" aria-label="Send message" title={dataMode === 'demo' ? 'Chat requires Live mode' : undefined} disabled={dataMode !== 'live' || !draft.trim() || uploading || (selectedNeedsEgress && !allowExternal)}><Send size={16}/></button>}
          </div>
          {modelControlsOpen && <div className="chat-model-controls" id="chat-model-controls" aria-label="Model and privacy settings">
            <div className="chat-model-selects">
              <label className="caption">Provider <select aria-label="Model provider" className="select" value={providerId} onChange={event => { setProviderId(event.target.value); setModelId(''); }} disabled={streaming}>{providers.length ? providers.map(provider => <option key={provider.id} value={provider.id} disabled={provider.available === false}>{provider.label || provider.id}{provider.id === 'auto' ? '' : provider.available === false ? ' (unavailable)' : provider.local === false ? ' (external)' : provider.network ? ' (local server)' : ' (local)'}</option>) : <option key="auto-local" value="auto">Auto route · local only</option>}</select></label>
              {providerId !== 'auto' && modelOptions.length > 0 && <label className="caption">Model <select aria-label="Model" className="select" value={modelId || selectedProvider?.selected_model || selectedProvider?.default_model || ''} onChange={event => setModelId(event.target.value)} disabled={streaming}><option key="provider-default" value="">Provider default</option>{modelOptions.map(model => <option key={model} value={model}>{model}</option>)}</select></label>}
              <ActionButton nav="models" onClick={() => go('models')} className="btn sm">Browse models</ActionButton>
            </div>
            {providersQuery.isError && <span className="caption" role="alert">{apiConnectionMessage(providersQuery.error)}</span>}
            <div className="row between caption" style={{ gap: 10, flexWrap: 'wrap' }}>
              <label className="row" style={{ gap: 6 }}><input type="checkbox" aria-label="Allow external provider calls" checked={allowExternal} onChange={event => setAllowExternal(event.target.checked)} disabled={streaming}/><span>Allow external provider calls</span></label>
              <span>Off by default · enabled only for this request</span>
            </div>
            {selectedNeedsEgress && !allowExternal && <p className="caption" role="alert">This provider sends prompt and attachment content over the network. Opt in above to continue.</p>}
            {selectedNeedsEgress && allowExternal && <p className="caption" role="status">{String(selectedProvider?.privacy_warning ?? 'External provider access is enabled for this request. Prompt and attachments may leave this server.')}</p>}
            {providerId === 'auto' && allowExternal && <p className="caption" role="status">External routing is permitted for this request when allowed by the backend model policy.</p>}
          </div>}
        </div>
      </form>
    </section>
    {inspector && <aside className="inspector"><div className="inspector-tabs"><div className="tabs" role="tablist" aria-label="Conversation details" onKeyDown={handleTabListKeyDown}>{(['Files', 'Preview', 'Context'] as const).map(name => <button key={name} type="button" role="tab" aria-selected={tab === name} tabIndex={tabIndex(tab === name)} className={`tab ${tab === name ? 'active' : ''}`} onClick={() => setTab(name)}>{name}</button>)}</div></div><div className="inspector-content">
      {tab === 'Files' && <><div className="section-head"><h3>Project files</h3><button type="button" className="icon-button" aria-label="Add files unavailable" title="Adding files requires a connected workspace" disabled><Plus size={15}/></button></div><div className="empty"><Folder size={20}/><strong>No workspace connected</strong><p>Connect a workspace to browse its files.</p></div></>}
      {tab === 'Preview' && <><div className="row between chat-preview-toolbar" style={{ marginBottom: 12 }}><span className="tag"><Globe2 size={13}/> Web preview</span><div className="row gap4"><button type="button" className="icon-button" aria-label={dataMode === 'demo' ? `${previewViewport === 'desktop' ? 'Desktop' : 'Mobile'} preview` : 'Desktop preview unavailable'} title={dataMode === 'demo' ? 'Toggle preview viewport' : 'Desktop preview unavailable'} disabled={dataMode !== 'demo'} aria-pressed={previewViewport === 'mobile'} onClick={() => setPreviewViewport(value => value === 'desktop' ? 'mobile' : 'desktop')}><Monitor size={16}/></button><button type="button" className="icon-button" aria-label={dataMode === 'demo' ? `${previewExpanded ? 'Collapse' : 'Expand'} preview` : 'Expand preview unavailable'} title={dataMode === 'demo' ? 'Toggle expanded preview' : 'Expand preview unavailable'} disabled={dataMode !== 'demo'} aria-pressed={previewExpanded} onClick={() => setPreviewExpanded(value => !value)}><Maximize2 size={16}/></button></div></div>{dataMode === 'demo' ? <><div className={`browser-preview ${previewViewport === 'mobile' ? 'chat-preview-mobile' : ''} ${previewExpanded ? 'chat-preview-expanded' : ''}`}><div className="mock-nav"><strong>◉ OpenAgent</strong><span>Product · Docs · Start</span></div><h2>Build better work with AI</h2><p>One workspace for research, planning and development.</p><button type="button" className="btn primary sm" onClick={() => go('home')}>Explore workspace →</button><div style={{ height: 86, borderRadius: 9, marginTop: 32, background: 'var(--blue-soft)', display: 'grid', placeItems: 'center', color: 'var(--blue)' }}><Layers3 size={32}/></div></div><div className="card padded section-space"><h3>Design notes</h3><p className="small muted" style={{ marginTop: 8 }}>Use a consistent navigation pattern, sensible content widths, and purposeful detail panels.</p></div></> : <div className="browser-preview chat-preview-unavailable"><div className="mock-nav"><strong>◉ Workspace preview</strong><span>Unavailable</span></div><div className="empty"><Layers3 size={28}/><h2>No live preview</h2><p>Connect a running workspace preview to view this panel.</p></div></div>}</>}
      {tab === 'Context' && <div className="card padded"><h3>Conversation context</h3><p className="muted small" style={{ margin: '9px 0 16px' }}>Project scope · default</p><div className="list-row"><Folder size={16}/><span className="flex1">Project file browser</span><span className="badge">No project file endpoint in chat contract</span></div><div className="list-row"><Bot size={16}/><span className="flex1">Provider / model</span><span className="badge">{selectedProvider?.label ?? providerId}{modelId || selectedProvider?.selected_model || selectedProvider?.default_model ? ` · ${modelId || selectedProvider?.selected_model || selectedProvider?.default_model}` : ''}</span></div><div className="list-row"><Shield size={16}/><span className="flex1">External routing</span><span className="badge">{allowExternal ? 'Allowed by request' : 'Blocked by request'}</span></div><div className="list-row"><CheckCircle2 size={16}/><span className="flex1">Backend connection</span><span className="badge">{conversationsQuery.isLoading || providersQuery.isLoading ? 'Checking' : conversationsQuery.isError || providersQuery.isError ? 'Unavailable' : 'Connected'}</span></div></div>}
    </div></aside>}
  </div>;
}

