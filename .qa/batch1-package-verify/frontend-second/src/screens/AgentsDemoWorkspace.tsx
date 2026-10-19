import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Bot, Code2, FileText, MoreHorizontal, Paperclip, Play, Plus, Search, Send, Sparkle, Target } from 'lucide-react';
import { ExecutionTrace } from '../components/ExecutionTrace';

const lanes = [
  {
    name: 'Coordinator', role: 'Plans work and keeps everyone aligned', tone: 'violet', Icon: Sparkle,
    first: 'I’ve divided the work into clear tasks. The team can review each area before a final decision.',
    title: 'Task plan', second: 'The findings from each specialist will be combined into one recommendation.',
  },
  {
    name: 'Researcher', role: 'Finds and checks information and sources', tone: 'green', Icon: Search,
    first: 'I’m comparing documented capabilities and evidence sources.',
    title: 'Research notes', second: 'I’ve grouped key findings with their supporting sources.',
  },
  {
    name: 'Developer', role: 'Analyses code and proposes changes', tone: 'blue', Icon: Code2,
    first: 'I’ve reviewed the proposed component changes and identified areas for testing.',
    title: 'Proposed code review', second: 'Changes are ready to inspect before any files are modified.',
  },
];
const initialTasks = [
  { title: 'Gather market feedback', owner: 'Researcher', done: true },
  { title: 'Review recent changes', owner: 'Developer', done: true },
  { title: 'Analyse competitor activity', owner: 'Researcher', done: true },
  { title: 'Assess risks and opportunities', owner: 'Coordinator', done: false },
  { title: 'Prepare launch recommendation', owner: 'Coordinator', done: false },
  { title: 'Create final report', owner: 'Coordinator', done: false },
];

export function AgentsDemoWorkspace({ onDirectory, directory = false }: { onDirectory: () => void; directory?: boolean }) {
  const [mobilePane, setMobilePane] = useState('Coordinator');
  const [tasks, setTasks] = useState(initialTasks);
  const [traceStartSignal, setTraceStartSignal] = useState(0);
  const [traceStatus, setTraceStatus] = useState<'Preview' | 'Running' | 'Paused' | 'Cancelled' | 'Completed'>('Preview');
  const [traceStep, setTraceStep] = useState(0);
  const [messages, setMessages] = useState<Record<string, string[]>>({});
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [taskDialog, setTaskDialog] = useState(false);
  const [newTask, setNewTask] = useState('');
  const [taskError, setTaskError] = useState('');
  const [attachedFile, setAttachedFile] = useState('');
  const [directorySearch, setDirectorySearch] = useState('');
  const [directoryAgents, setDirectoryAgents] = useState([
    { name: 'Coordinator', role: 'Plans work and keeps everyone aligned', tone: 'violet', Icon: Sparkle },
    { name: 'Researcher', role: 'Finds and checks information and sources', tone: 'green', Icon: Search },
    { name: 'Developer', role: 'Analyses code and proposes changes', tone: 'blue', Icon: Code2 },
    { name: 'Documentation', role: 'Turns technical work into clear guidance', tone: 'orange', Icon: FileText },
  ]);
  const [newAgentOpen, setNewAgentOpen] = useState(false);
  const [newAgentName, setNewAgentName] = useState('');
  const [newAgentRole, setNewAgentRole] = useState('');
  const [agentError, setAgentError] = useState('');
  const [laneToast, setLaneToast] = useState('');
  const laneToastTimer = useRef<number | null>(null);
  const completed = tasks.filter(task => task.done).length;
  useEffect(() => () => { if (laneToastTimer.current !== null) window.clearTimeout(laneToastTimer.current); }, []);
  const onTraceChange = useCallback((status: 'Preview' | 'Recording' | 'Running' | 'Paused' | 'Cancelled' | 'Completed', step: number) => {
    if (status === 'Preview') return;
    setTraceStatus(status === 'Recording' ? 'Running' : status);
    setTraceStep(step);
  }, []);
  const traceSimulation = useMemo(() => ({
    startMessage: 'Multi-agent activity started in the sample job engine',
    pauseMessage: 'Example playback paused',
    resumeMessage: 'Example playback resumed',
    eventMessages: ['Validating configuration', 'Processing example records', 'Generating sample output', 'Completed'],
    tickMilliseconds: 1000,
    progressPerTick: 25,
    onChange: onTraceChange,
  }), [onTraceChange]);
  const running = traceStatus === 'Running' || traceStatus === 'Paused';
  // Keep workflow status banners attached to the completed output, as in the canonical UI.
  const hasPreview = traceStatus !== 'Preview';
  const traceLabel: Record<Exclude<typeof traceStatus, 'Preview'>, string> = {
    Running: 'Running · demo',
    Paused: 'Paused · demo',
    Completed: 'Completed · demo',
    Cancelled: 'Cancelled · demo',
  };
  const startPreview = () => {
    if (running) return;
    setTraceStatus('Running');
    setTraceStep(0);
    setTraceStartSignal(signal => signal + 1);
  };
  const workflowStages = ['Gather information', 'Analyse and discuss', 'Prepare recommendation', 'Finalise report'];
  const stagePresentation = (index: number) => {
    if (traceStatus === 'Preview') return { className: '', label: 'Not started' };
    if (traceStatus === 'Completed' || index < traceStep) return { className: 'complete', label: 'Complete' };
    if (traceStatus === 'Cancelled') return index === traceStep
      ? { className: 'cancelled', label: 'Cancelled' }
      : { className: '', label: 'Not run' };
    if (index === traceStep) return { className: 'active', label: traceStatus === 'Paused' ? 'Paused' : 'In progress' };
    return { className: '', label: 'Pending' };
  };
  const showLaneToast = () => {
    setLaneToast('Design interaction only — connect its real action during migration.');
    if (laneToastTimer.current !== null) window.clearTimeout(laneToastTimer.current);
    laneToastTimer.current = window.setTimeout(() => { setLaneToast(''); laneToastTimer.current = null; }, 3000);
  };

  const sendMessage = (name: string) => {
    const message = (drafts[name] ?? '').trim();
    if (!message) return;
    setMessages(current => ({ ...current, [name]: [...(current[name] ?? []), message] }));
    setDrafts(current => ({ ...current, [name]: '' }));
  };
  const submitTask = () => {
    if (!newTask.trim()) { setTaskError('Enter a task name.'); return; }
    setTasks(current => [...current, { title: newTask.trim(), owner: 'Coordinator', done: false }]);
    setNewTask(''); setTaskError(''); setTaskDialog(false);
  };

  const submitAgent = () => {
    if (!newAgentName.trim() || !newAgentRole.trim()) { setAgentError('Enter both a name and a role.'); return; }
    setDirectoryAgents(current => [...current, { name: newAgentName.trim(), role: newAgentRole.trim(), tone: 'blue', Icon: Bot }]);
    setNewAgentName(''); setNewAgentRole(''); setAgentError(''); setNewAgentOpen(false);
  };

  if (directory) {
    const visibleAgents = directoryAgents.filter(agent => `${agent.name} ${agent.role}`.toLowerCase().includes(directorySearch.toLowerCase()));
    return <div className="page agent-demo-directory">
      <div className="page-heading"><div><h1>AI agents</h1><p>Find the right specialist for a task, or configure a new one.</p></div><div className="row"><button type="button" className="btn sm" onClick={onDirectory}>View example session</button><button type="button" className="btn primary" onClick={() => setNewAgentOpen(true)}><Plus size={15}/> New agent</button></div></div>
      <label className="search-field"><Bot aria-hidden="true"/><input aria-label="Search agents" placeholder="Search agents..." value={directorySearch} onChange={event => setDirectorySearch(event.target.value)}/></label>
      <div className="section-space agent-directory">{visibleAgents.length ? visibleAgents.map(({ name, role, tone, Icon }) => <article className="card directory-card" key={name}><span className={`square-icon ${tone}`}><Icon aria-hidden="true"/></span><div className="flex1"><div className="row between"><h3>{name}</h3><span className="tag green">Available</span></div><p className="muted small" style={{ margin: '7px 0 15px' }}>{role}</p><button type="button" className="btn sm soft" onClick={onDirectory}>View sample workflow</button></div></article>) : <div className="empty"><Bot className="icon lg"/><h3>No agents match</h3><p>Try a different name or role.</p></div>}</div>
      {newAgentOpen && <div className="modal-overlay" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) setNewAgentOpen(false); }}><section className="modal card" role="dialog" aria-modal="true" aria-labelledby="new-demo-agent-title"><header className="modal-head"><div><h2 id="new-demo-agent-title">Create a demo agent</h2><p className="caption">Add a specialist to this local example directory.</p></div><button className="icon-button" type="button" aria-label="Close dialog" onClick={() => setNewAgentOpen(false)}>×</button></header><label className="demo-form-field">Agent name<input autoFocus className="input" value={newAgentName} onChange={event => { setNewAgentName(event.target.value); setAgentError(''); }}/></label><label className="demo-form-field">Role<textarea className="textarea" rows={3} value={newAgentRole} onChange={event => { setNewAgentRole(event.target.value); setAgentError(''); }}/></label>{agentError && <p role="alert" className="form-error">{agentError}</p>}<footer className="modal-actions"><button type="button" className="btn" onClick={() => setNewAgentOpen(false)}>Cancel</button><button type="button" className="btn primary" onClick={submitAgent}>Create agent</button></footer></section></div>}
    </div>;
  }

  return <div className="split-main agent-demo-workspace">
    <div className="agent-top">
      <div className="row between">
        <div className="row" style={{ gap: 15 }}>
          <span className="square-icon"><Target aria-hidden="true"/></span>
          <div><h1 style={{ fontSize: 24 }}>Product launch review</h1><p className="muted small">Example multi-agent workflow with clear task ownership.</p></div>
        </div>
        <div className="row agent-demo-actions"><span className="tag orange">Illustrative session</span><button type="button" className="btn sm" onClick={onDirectory}>Agent directory</button><button type="button" className="btn sm" onClick={startPreview}><Play size={13}/> Preview activity</button></div>
      </div>
      <div className="steps-line" aria-label="Demo execution stages">
        {workflowStages.map((step, index) => { const presentation = stagePresentation(index); return <div className={`step ${presentation.className}`} key={step}><strong>{index + 1}. {step}</strong><div>{presentation.label}</div></div>; })}
      </div>
      <div className="agent-mobile-tabs" role="tablist" aria-label="Agent panels" onKeyDown={event => {
        const tabs = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('[role=tab]'));
        const index = tabs.indexOf(document.activeElement as HTMLButtonElement);
        if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { event.preventDefault(); const next = (index + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length; tabs[next]?.focus(); tabs[next]?.click(); }
      }}>
      {['Coordinator', 'Researcher', 'Developer', 'Tasks'].map(name => <button key={name} type="button" role="tab" aria-selected={mobilePane === name} tabIndex={mobilePane === name ? 0 : -1} className={`btn sm ${mobilePane === name ? 'soft' : ''}`} onClick={() => setMobilePane(name)}>{name}</button>)}
      </div>
    </div>
    {hasPreview && <div className={`live-inline live-inline-${traceStatus.toLowerCase()}`} role="status" aria-live="polite" aria-label={`Agents demo run ${traceStatus.toLowerCase()}`}><span className="live-inline-orb"/><div className="live-inline-label"><strong>{traceLabel[traceStatus as Exclude<typeof traceStatus, 'Preview'>]}</strong><small>{traceStatus === 'Completed' ? 'Coordinator summary is ready to inspect' : traceStatus === 'Cancelled' ? 'Local demo run was cancelled' : 'Coordinator prepares a summary'}</small></div><button type="button" className="btn sm" onClick={() => document.querySelector('.trace-mount.trace-agents')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })}>View steps</button></div>}
    {hasPreview && <ExecutionTrace route="agents" startSignal={traceStartSignal} simulation={traceSimulation}/>}
    <div className="agent-board">
      {lanes.map(({ name, role, tone, Icon, first, title, second }, index) => <section key={name} className={`agent-lane ${mobilePane === name ? 'mobile-selected' : ''}`}>
        <header className="lane-head"><div className="row"><span className={`square-icon ${tone}`}><Icon aria-hidden="true"/></span><div className="flex1"><h3>{name}</h3><p className="caption">{role}</p></div><span className="status-dot"/><button className="icon-button" type="button" aria-label="Agent actions" title="Agent actions" onClick={showLaneToast}><MoreHorizontal size={16}/></button></div></header>
        <div className="lane-body">
          <article className="lane-message"><div className="row"><span className="avatar blue"><Bot aria-hidden="true"/></span><strong>{name}</strong><span className="caption">Preview message</span></div><p>{first}</p></article>
          <div className="lane-note"><h3>{title}</h3>{index === 0 ? <div className="task-list">{['Gather information', 'Analyse and discuss', 'Prepare recommendation', 'Finalise report'].map((item, i) => <label key={item}><input type="checkbox" defaultChecked={i === 0}/>{item}</label>)}</div> : index === 1 ? <><p>Key findings</p><p>• Clear demand for reliable tools<br/>• Integration quality matters<br/>• Privacy needs explicit controls</p></> : <div className="code-sample"><pre>{'src/components/ProductCard.tsx\n\n'}<span style={{ color: 'var(--green)' }}>+ Add component tests</span>{'\n'}<span style={{ color: 'var(--green)' }}>+ Handle loading states</span></pre></div>}</div>
          <article className="lane-message"><div className="row"><span className="avatar blue"><Bot aria-hidden="true"/></span><strong>{name}</strong><span className="caption">Preview message</span></div><p>{second}</p></article>
          {(messages[name] ?? []).map((message, i) => <article className="lane-message demo-user-message" key={`${i}-${message}`}><div className="row"><span className="avatar"><Bot aria-hidden="true"/></span><strong>You</strong><span className="caption">Local demo</span></div><p>{message}</p></article>)}
        </div>
        <div className="chat-composer-outer"><div className="chat-composer"><textarea aria-label={`Message ${name}`} placeholder={`Message ${name}...`} value={drafts[name] ?? ''} onChange={event => setDrafts(current => ({ ...current, [name]: event.target.value }))} onKeyDown={event => { if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); sendMessage(name); } }}/><div className="row between"><button type="button" className="icon-button" aria-label={`Attach a file to ${name}`} onClick={() => document.getElementById(`agent-file-${name}`)?.click()}><Paperclip size={16}/></button><input id={`agent-file-${name}`} type="file" hidden onChange={event => setAttachedFile(event.target.files?.[0]?.name ?? '')}/><button type="button" className="send-button" aria-label={`Send message to ${name}`} disabled={!(drafts[name] ?? '').trim()} onClick={() => sendMessage(name)}><Send size={16}/></button></div></div></div>
      </section>)}
      <aside className={`agent-context ${mobilePane === 'Tasks' ? 'mobile-selected' : ''}`}>
        <div className="section-head"><h3>Tasks</h3><button type="button" className="btn sm" onClick={() => setTaskDialog(true)}><Plus size={13}/> Add</button></div>
        <p className="caption">{completed} of {tasks.length} complete · Example workflow</p><div className="progress"><span style={{ width: `${tasks.length ? completed / tasks.length * 100 : 0}%` }}/></div>
        <div className="task-list">{tasks.map((task, index) => <label key={`${index}-${task.title}`}><input type="checkbox" checked={task.done} onChange={() => setTasks(current => current.map((item, i) => i === index ? { ...item, done: !item.done } : item))}/><span>{task.title}<div className="caption">{task.owner}</div></span></label>)}</div>
        <div className="divider"/><div className="section-head"><h3>Shared files</h3><button type="button" className="icon-button" aria-label="Attach a shared file" onClick={() => document.getElementById('agent-shared-file')?.click()}><Plus size={15}/></button><input id="agent-shared-file" type="file" hidden onChange={event => setAttachedFile(event.target.files?.[0]?.name ?? '')}/></div>
        {['Launch plan draft', 'Market analysis.pdf', 'Competitor comparison.xlsx'].map(file => <div className="row" key={file}><span className="square-icon sm"><FileText aria-hidden="true"/></span><div className="flex1"><strong className="small">{file}</strong><div className="caption">Illustration only</div></div></div>)}
        {attachedFile && <p className="caption" role="status">Attached locally: {attachedFile}</p>}
        <div className="note-panel">This is a multi-agent interaction preview. Real progress requires durable backend run events.</div>
      </aside>
    </div>
    {laneToast && <div id="toast-region" aria-live="polite" aria-atomic="true"><div className="toast"><span>{laneToast}</span></div></div>}
    {taskDialog && <div className="modal-overlay" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) setTaskDialog(false); }}><section className="modal card" role="dialog" aria-modal="true" aria-labelledby="agent-task-title"><header className="modal-head"><div><h2 id="agent-task-title">Add workflow task</h2><p className="caption">Add a task to this local illustrative session.</p></div><button type="button" className="icon-button" aria-label="Close dialog" onClick={() => setTaskDialog(false)}>×</button></header><label className="demo-form-field">Task name<input autoFocus className="input" value={newTask} onChange={event => { setNewTask(event.target.value); setTaskError(''); }} onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); submitTask(); } if (event.key === 'Escape') setTaskDialog(false); }}/></label>{taskError && <p className="form-error" role="alert">{taskError}</p>}<footer className="modal-actions"><button type="button" className="btn" onClick={() => setTaskDialog(false)}>Cancel</button><button type="button" className="btn primary" onClick={submitTask}>Add task</button></footer></section></div>}
  </div>;
}
