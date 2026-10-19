import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { Activity, ArrowUpRight, Bot, Code2, Cpu, MessageSquarePlus, RefreshCw, Search, ShieldCheck, Paperclip, Send, Users, Clock3, Sun, Globe, FileText, CalendarDays } from 'lucide-react';
import { api, ApiError, record, records } from '../lib/react/api';
import { auditEventContext, auditEventOutcome, auditEventTitle } from '../lib/react/auditPresentation';
import './DashboardPage.css';

type Row = Record<string, unknown>;
type Snapshot = { health: Row; metrics: Row; modules: Row; inference: Row; localAI: Row; audit: Row; approvals: Row; jobs: Row; conversations: Row; agents: Row };
const empty: Snapshot = { health: {}, metrics: {}, modules: {}, inference: {}, localAI: {}, audit: {}, approvals: {}, jobs: {}, conversations: {}, agents: {} };
const read = (source: unknown, key: string) => record(source)[key];
const display = (value: unknown, suffix = '') => value === undefined || value === null || value === '' ? '—' : `${String(value)}${suffix}`;
const normalizedStatus = (value: unknown) => typeof value === 'string' ? value.trim().toLowerCase().replaceAll('_', ' ') : '';
const healthLabel = (value: unknown) => {
  const status = normalizedStatus(value);
  if (['ok', 'healthy', 'ready', 'available'].includes(status)) return 'Operational';
  if (['degraded', 'limited', 'warning'].includes(status)) return 'Degraded';
  if (['error', 'failed', 'failure', 'unhealthy', 'unavailable', 'down'].includes(status)) return 'Needs attention';
  return status ? status.replace(/\b\w/g, character => character.toUpperCase()) : 'Status unavailable';
};
const moduleLabel = (module: Row) => {
  const status = normalizedStatus(module.status);
  if (module.available === true || ['ok', 'healthy', 'ready', 'available', 'active'].includes(status)) return 'Available';
  if (module.available === false || ['error', 'failed', 'failure', 'unhealthy', 'unavailable', 'down', 'disabled'].includes(status)) return 'Unavailable';
  if (['degraded', 'limited', 'warning'].includes(status)) return 'Degraded';
  if (['', 'unknown', 'registered', 'not reported'].includes(status)) return null;
  return status.replace(/\b\w/g, character => character.toUpperCase());
};
const requestError = (reason: unknown) => {
  if (reason instanceof ApiError && reason.status === 401) return 'Sign in again to read workspace activity.';
  if (reason instanceof ApiError && reason.status === 403) return 'Your account cannot read this workspace activity.';
  if (reason instanceof ApiError && reason.status === 404) return 'An activity service is not available in this server version.';
  if (reason instanceof ApiError && reason.status >= 500) return 'Activity could not be loaded. Retry when it is available.';
  return reason instanceof Error ? reason.message : 'The local service is unavailable.';
};
const timestamp = (value: unknown) => {
  if (value === undefined || value === null || value === '') return 'Time unavailable';
  const date = new Date(String(value));
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
};

const homeUrl = (view: string, project: string, conversationId?: unknown) => {
  const url = new URL('/', window.location.origin);
  url.searchParams.set('view', view);
  url.searchParams.set('project', project || 'default');
  if (view === 'dashboard') url.searchParams.set('activity', '1');
  if (conversationId !== undefined && conversationId !== null && conversationId !== '') url.searchParams.set('conversation_id', String(conversationId));
  return `${url.pathname}${url.search}`;
};

/** Home shows real workspace activity and keeps operational diagnostics secondary. */
export function DashboardPage({ project }: { project: string }) {
  const [data, setData] = useState<Snapshot>(empty);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [updatedAt, setUpdatedAt] = useState('');
  const [homeDraft, setHomeDraft] = useState('');
  const [failedSources, setFailedSources] = useState<string[]>([]);
  const activeRequest = useRef<AbortController | null>(null);
  const requestSequence = useRef(0);

  const refresh = useCallback(async () => {
    if (activeRequest.current && !activeRequest.current.signal.aborted) return;
    const controller = new AbortController();
    activeRequest.current = controller;
    const { signal } = controller;
    const requestId = ++requestSequence.current;
    setRefreshing(true);
    setError('');
    const scopedProject = encodeURIComponent(project || 'default');
    const sourceNames: (keyof Snapshot)[] = ['health', 'metrics', 'modules', 'inference', 'localAI', 'audit', 'approvals', 'jobs', 'conversations', 'agents'];
    const results = await Promise.allSettled([
      api.get<unknown>('/api/health', signal),
      api.get<unknown>('/api/system/metrics', signal),
      api.get<unknown>(`/api/modules/health?project=${scopedProject}`, signal),
      api.get<unknown>('/api/inference/status', signal),
      api.get<unknown>('/api/local-ai/status', signal),
      api.get<unknown>(`/api/audit?project=${scopedProject}&limit=8`, signal),
      api.get<unknown>(`/api/agents/approvals?project=${scopedProject}&status=pending`, signal),
      api.get<unknown>(`/api/worker/jobs?project=${scopedProject}&limit=8`, signal),
      api.get<unknown>(`/api/conversations/organized?project=${scopedProject}&limit=5&sort=pinned&include_archived=false&archived_only=false`, signal),
      api.get<unknown>('/api/agents', signal),
    ]);
    try {
      if (signal.aborted || requestId !== requestSequence.current) return;
      setFailedSources(results.flatMap((result, index) => result.status === 'rejected' ? [sourceNames[index]] : []));
      setData(previous => {
        const next = { ...previous };
        results.forEach((result, index) => { if (result.status === 'fulfilled') next[sourceNames[index]] = record(result.value); });
        return next;
      });
      const rejected = results.filter((result): result is PromiseRejectedResult => result.status === 'rejected');
      if (rejected.length) {
        const representative = results[0].status === 'rejected' ? results[0].reason : rejected.find(result => result.reason instanceof ApiError && [401, 403, 404].includes(result.reason.status))?.reason ?? rejected[0].reason;
        setError(results[0].status === 'rejected' || representative instanceof ApiError ? requestError(representative) : 'Some activity sources could not be reached.');
      }
      setUpdatedAt(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
      setLoading(false);
    } finally {
      if (requestId === requestSequence.current) {
        if (activeRequest.current === controller) activeRequest.current = null;
        setRefreshing(false);
      }
    }
  }, [project]);

  useEffect(() => {
    void refresh();
    return () => {
      requestSequence.current += 1;
      activeRequest.current?.abort();
      activeRequest.current = null;
    };
  }, [refresh]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') void refresh();
    }, 30_000);
    const resume = () => { if (document.visibilityState === 'visible') void refresh(); };
    document.addEventListener('visibilitychange', resume);
    return () => { window.clearInterval(timer); document.removeEventListener('visibilitychange', resume); };
  }, [refresh]);

  const events = useMemo(() => records(data.audit, 'events'), [data.audit]);
  const conversations = useMemo(() => records(data.conversations, 'conversations').slice(0, 5), [data.conversations]);
  const groupedEvents = useMemo(() => {
    const grouped = new Map<string, { event: Row; title: string; context?: string; outcome?: string; count: number }>();
    for (const event of events) {
      const title = auditEventTitle(event);
      const context = auditEventContext(event);
      const outcome = auditEventOutcome(event);
      const key = [title, context ?? '', outcome ?? ''].join('\u0000');
      const current = grouped.get(key);
      if (current) current.count += 1;
      else grouped.set(key, { event, title, ...(context ? { context } : {}), ...(outcome ? { outcome } : {}), count: 1 });
    }
    return [...grouped.values()];
  }, [events]);
  const modules = useMemo(() => records(data.modules, 'modules'), [data.modules]);
  const approvals = useMemo(() => records(data.approvals, 'approvals').filter(item => String(item.status ?? 'pending').toLowerCase() === 'pending'), [data.approvals]);
  const jobs = useMemo(() => records(data.jobs, 'jobs'), [data.jobs]);
  const agents = useMemo(() => records(data.agents, 'agents'), [data.agents]);
  const cpu = read(data.metrics, 'cpu'); const memory = read(data.metrics, 'memory'); const disk = read(data.metrics, 'disk');
  const hardware = record(read(data.localAI, 'hardware'));
  const healthStatus = String(read(data.health, 'status') ?? '').toLowerCase();
  const serviceHealthy = ['ok', 'healthy', 'ready', 'available'].includes(healthStatus);
  const serviceNeedsAttention = ['degraded', 'limited', 'warning', 'error', 'failed', 'failure', 'unhealthy', 'unavailable', 'down'].includes(healthStatus);
  const inferenceStatus = read(data.inference, 'status') ?? read(data.inference, 'provider');
  const hasInferenceStatus = inferenceStatus !== undefined && inferenceStatus !== null && inferenceStatus !== '';
  const hostLabel = [hardware.system, hardware.release].filter(Boolean).join(' ');
  const hasRuntimeDetails = modules.some(module => moduleLabel(module) !== null) || Boolean(hostLabel || hardware.processor || hardware.cpu_count || hasInferenceStatus || cpu || memory || disk);
  const cpuUsage = read(cpu, 'usage_percent');
  const memoryUsage = read(memory, 'used_percent');
  const diskUsage = read(disk, 'used_percent');

  const hasRecentWork = conversations.length > 0 || groupedEvents.length > 0 || jobs.length > 0 || approvals.length > 0;
  const sourcesUnavailable = (source: keyof Snapshot) => failedSources.includes(source);
  const navigate = (view: string, conversationId?: unknown) => {
    const url = homeUrl(view, project, conversationId);
    history.pushState({ openagentNavigation: true, view, project, conversation_id: conversationId ? String(conversationId) : null }, '', url);
    window.dispatchEvent(new PopStateEvent('popstate'));
  };
  const startChat = (event?: FormEvent) => {
    event?.preventDefault();
    if (homeDraft.trim()) { try { sessionStorage.setItem(`oa-chat-handoff:${project}`, homeDraft.trim()); } catch { /* Continue into the composer if session storage is unavailable. */ } }
    navigate('chat');
  };

  return <main className="activity-page home-page">
    <h1 className="sr-only">Home</h1>

    {error && <div className="activity-error" role="alert">
      <span>{failedSources.length < 10 && !/sign in again|cannot read|not available in this server version/i.test(error) ? 'Some workspace information could not be loaded.' : error}</span><button onClick={() => void refresh()} disabled={refreshing}>Try again</button>
    </div>}

    <div className="home-dashboard-grid"><div className="home-main-column">
    <section className="home-start" aria-labelledby="home-start-title">
      <div className="home-intro"><span className="home-sun-icon"><Sun size={23}/></span><div><h2 id="home-start-title">Welcome to OpenAgent</h2><p>What would you like to work on?</p></div><button className="activity-refresh" onClick={() => void refresh()} disabled={refreshing} aria-label="Refresh activity"><RefreshCw size={14} className={refreshing ? 'activity-spin' : ''}/><span>{refreshing ? 'Refreshing' : 'Refresh'}</span></button></div>
      <form className="home-prompt" onSubmit={startChat}><label className="sr-only" htmlFor="home-prompt-input">Ask OpenAgent</label><textarea id="home-prompt-input" rows={2} placeholder="Ask OpenAgent anything…" value={homeDraft} onChange={event => setHomeDraft(event.currentTarget.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); startChat(); } }}/><div className="home-prompt-toolbar"><div className="home-prompt-tools"><button type="button" aria-label="Add files in chat" onClick={() => startChat()}><Paperclip size={14}/></button><button type="button" onClick={() => navigate('agents')}><Users size={14}/>Agents</button><button type="button" onClick={() => navigate('research')}><Globe size={14}/>Web</button><button type="button" onClick={() => navigate('workspace')}><FileText size={14}/>Files</button><button type="button" onClick={() => navigate('research')}><Search size={14}/>Research</button><button type="button" onClick={() => navigate('codebase')}><Code2 size={14}/>CodeQ</button></div><button className="home-primary-action" type="submit" aria-label="Start chat"><Send size={15}/></button></div></form>
      <nav className="home-shortcuts" aria-label="Start a task">
        <button className="home-shortcut-chat" onClick={() => startChat()}><span className="home-shortcut-icon"><MessageSquarePlus size={18}/></span><strong>Start a new chat</strong><small>Ask, plan and solve problems</small><ArrowUpRight size={14}/></button>
        <button className="home-shortcut-agent" onClick={() => navigate('agents')}><span className="home-shortcut-icon"><Users size={18}/></span><strong>Create an agent</strong><small>Configure a specialist assistant</small><ArrowUpRight size={14}/></button>
        <button className="home-shortcut-research" onClick={() => navigate('research')}><span className="home-shortcut-icon"><Search size={18}/></span><strong>Start research</strong><small>Explore a topic with sources</small><ArrowUpRight size={14}/></button>
        <button className="home-shortcut-automation" onClick={() => navigate('automation')}><span className="home-shortcut-icon"><CalendarDays size={18}/></span><strong>Set up automation</strong><small>Schedule repeatable work</small><ArrowUpRight size={14}/></button>
      </nav>
    </section>

    {loading ? <div className="activity-loading" role="status" aria-label="Loading workspace">
      <span className="activity-sr-only">Loading your workspace…</span>
      <i aria-hidden="true"/><i aria-hidden="true"/><i aria-hidden="true"/>
    </div> : <>
      {conversations.length > 0 && <section className="activity-section home-conversations" aria-labelledby="activity-conversations-title">
        <div className="activity-section-head"><h2 id="activity-conversations-title">Recent conversations</h2></div>
        <ol className="activity-list">{conversations.map((conversation, index) => {
          const id = conversation.id ?? conversation.conversation_id;
          const title = conversation.title ?? conversation.name ?? 'Conversation';
          return <li key={String(id ?? index)}>
            <span className="activity-event-mark"><MessageSquarePlus size={14}/></span>
            <div><strong>{display(title)}</strong>{conversation.updated_at !== undefined && conversation.updated_at !== null && conversation.updated_at !== '' && <small>{timestamp(conversation.updated_at)}</small>}</div>
            <button className="home-open-conversation" onClick={() => navigate('chat', id)} aria-label={`Open ${String(title)}`}>Open <ArrowUpRight size={13}/></button>
          </li>;
        })}</ol>
      </section>}

      {groupedEvents.length > 0 && <section className="activity-section" aria-labelledby="activity-events-title">
        <div className="activity-section-head"><h2 id="activity-events-title">Recent activity</h2></div>
        <ol className="activity-list">{groupedEvents.map(({ event, title, context, outcome, count }, index) => {
          const outcomeClass = outcome && /failed|denied|revoked/i.test(outcome) ? 'is-warning' : '';
          return <li key={`${title}-${context ?? ''}-${outcome ?? ''}-${index}`}>
            <span className={`activity-event-mark ${outcomeClass}`}><ShieldCheck size={14}/></span>
            <div><strong>{title}</strong><small>{[context, count > 1 ? `${count} similar events` : undefined].filter(Boolean).join(' · ')}</small></div>
            <span className="activity-event-outcome">{outcome ?? ''}</span>
            <time>{timestamp(event.created_at ?? event.timestamp)}</time>
          </li>;
        })}</ol>
      </section>}

      {!hasRecentWork && !sourcesUnavailable('audit') && !sourcesUnavailable('jobs') && !sourcesUnavailable('conversations') && <section className="home-empty" aria-label="No recent work">
        <div><Bot size={20}/><h2>Nothing here yet</h2><p>Your conversations and activity will appear here.</p></div>
        <button onClick={() => startChat()}>Start a conversation</button>
      </section>}

      {hasRuntimeDetails && <details className="activity-details home-system-details">
        <summary><span><Activity size={15}/>System status</span><span className={`activity-state ${serviceHealthy ? 'is-good' : serviceNeedsAttention ? 'is-warning' : ''}`}><i/>{healthLabel(read(data.health, 'status'))}</span></summary>
        <div className="activity-status">
        <div className="activity-status-title"><Activity size={16}/><span>Live system metrics</span></div>
        <dl>
          {hasInferenceStatus && <div><dt>Inference</dt><dd>{display(inferenceStatus)}{read(data.inference, 'model') ? <small>{display(read(data.inference, 'model'))}</small> : null}</dd></div>}
          {cpuUsage !== undefined && cpuUsage !== null && <div><dt>Processor</dt><dd>{display(cpuUsage, '%')}{read(cpu, 'count') !== undefined && <small>{display(read(cpu, 'count'))} logical cores</small>}</dd></div>}
          {memoryUsage !== undefined && memoryUsage !== null && <div><dt>Memory</dt><dd>{display(memoryUsage, '%')}{read(memory, 'used_gb') !== undefined && <small>{display(read(memory, 'used_gb'), ' GB')} of {display(read(memory, 'total_gb'), ' GB')}</small>}</dd></div>}
          {diskUsage !== undefined && diskUsage !== null && <div><dt>Storage</dt><dd>{display(diskUsage, '%')}{read(disk, 'free_gb') !== undefined && <small>{display(read(disk, 'free_gb'), ' GB')} free</small>}</dd></div>}
        </dl>
        <div className="activity-system-foot">
          {hostLabel && <span>{hostLabel}</span>}
          {hardware.processor !== undefined && hardware.processor !== null && hardware.processor !== '' && <span>{String(hardware.processor)}</span>}
          {hardware.cpu_count !== undefined && hardware.cpu_count !== null && hardware.cpu_count !== '' && <span>{String(hardware.cpu_count)} cores</span>}
          {updatedAt && <span>Updated {updatedAt}</span>}
        </div>
        </div>
        <div className="activity-details-body">
          {modules.some(module => moduleLabel(module) !== null) && <div className="activity-modules"><h3>Module status</h3><ul>{modules.map((module, index) => {
            const state = moduleLabel(module);
            if (!state) return null;
            const ok = state === 'Available';
            return <li key={String(module.id ?? module.name ?? index)}><span>{display(module.label ?? module.name ?? module.id)}</span><span className={ok ? 'is-good' : state === 'Degraded' || state === 'Unavailable' ? 'is-warning' : ''}>{state}</span></li>;
          })}</ul></div>}
          {hasInferenceStatus && <div className="activity-machine"><h3>Model service</h3><p>{display(inferenceStatus)}{read(data.inference, 'model') ? ` · ${display(read(data.inference, 'model'))}` : ''}</p></div>}
        </div>
      </details>}
    </>}
    </div><aside className="home-activity-rail" aria-label="Workspace activity">
      <section className="home-rail-section" aria-labelledby="home-today-title"><header><h2 id="home-today-title"><Clock3 size={16}/>Pending approvals</h2><button onClick={() => navigate('agents')}>View all</button></header>
        {loading ? <div className="home-rail-loading" aria-label="Loading tasks"/> : approvals.length ? <ol className="home-rail-list">{approvals.slice(0, 6).map((item, index) => { const title = item.title ?? item.action ?? item.name ?? item.agent_name ?? item.agent ?? 'Approval request'; return <li key={String(item.id ?? index)}><ShieldCheck size={15}/><span><strong>{display(title)}</strong><small>Pending approval{item.created_at ? ` · ${timestamp(item.created_at)}` : ''}</small></span></li>; })}</ol> : <p className="home-rail-empty">No approvals waiting.</p>}
      </section>
      <section className="home-rail-section" aria-labelledby="home-runs-title"><header><h2 id="home-runs-title"><Activity size={16}/>Recent runs</h2><button onClick={() => navigate('agents')}>View all</button></header>
        {loading ? <div className="home-rail-loading" aria-label="Loading runs"/> : jobs.length ? <ol className="home-rail-list">{jobs.slice(0, 5).map((item, index) => { const title = item.name ?? item.kind ?? item.type ?? item.id; const detail = [item.state ?? item.status, item.updated_at ?? item.created_at ? timestamp(item.updated_at ?? item.created_at) : undefined].filter(Boolean).join(' · '); return <li key={String(item.id ?? index)}><Cpu size={15}/><span><strong>{display(title)}</strong>{detail.length > 0 && <small>{detail}</small>}</span></li>; })}</ol> : <p className="home-rail-empty">No recent runs.</p>}
      </section>
      <section className="home-rail-section" aria-labelledby="home-agents-title"><header><h2 id="home-agents-title"><Users size={16}/>Your agents</h2><button onClick={() => navigate('agents')}>View all</button></header>
        {loading ? <div className="home-rail-loading" aria-label="Loading agents"/> : agents.length ? <ul className="home-agent-list">{agents.slice(0, 4).map((item, index) => <li key={String(item.id ?? index)}><span className="home-agent-mark"><Bot size={15}/></span><span><strong>{display(item.name ?? item.label ?? item.id)}</strong>{typeof item.description === 'string' && item.description.length > 0 && <small>{display(item.description)}</small>}</span></li>)}</ul> : <p className="home-rail-empty">{sourcesUnavailable('agents') ? 'Agent list unavailable.' : 'No agents are configured.'}</p>}
      </section>
      <section className="home-rail-section" aria-labelledby="home-tools-title"><header><h2 id="home-tools-title"><ArrowUpRight size={15}/>Tools</h2><button onClick={() => navigate('developer')}>More tools</button></header><nav className="home-tool-grid" aria-label="Workspace tools"><button onClick={() => navigate('research')}><Globe size={17}/>Web search</button><button onClick={() => navigate('codebase')}><Code2 size={17}/>CodeQ</button><button onClick={() => navigate('workspace')}><FileText size={17}/>Files</button><button onClick={() => navigate('developer')}><Code2 size={17}/>Developer</button><button onClick={() => navigate('knowledge')}><FileText size={17}/>Knowledge</button><button onClick={() => navigate('calendar')}><CalendarDays size={17}/>Calendar</button></nav></section>
    </aside></div>
  </main>;
}
