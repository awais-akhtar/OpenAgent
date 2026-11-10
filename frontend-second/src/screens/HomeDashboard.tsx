import { useEffect, useMemo, useState, type FormEvent } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Cpu, Database, HardDrive, MemoryStick, Server, ShieldCheck, X, Zap } from 'lucide-react';
import { apiConnectionMessage, apiGet } from '../api/client';
import { demoHomeTelemetry } from '../demo/homeTelemetry';
import { demoProvider } from '../demo/provider';
import { useWorkspaceState } from '../state';

type ObjectData = Record<string, unknown>;
type Sample = { at: number; cpu: number; memory: number };
const asObject = (value: unknown): ObjectData => value && typeof value === 'object' && !Array.isArray(value) ? value as ObjectData : {};
const asRows = (value: unknown): ObjectData[] => Array.isArray(value) ? value.filter((item): item is ObjectData => Boolean(item) && typeof item === 'object') : [];
const number = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : null;
const percent = (value: unknown) => { const result = number(value); return result === null ? null : Math.min(100, Math.max(0, result)); };
const formatPercent = (value: number | null) => value === null ? '—' : `${value.toFixed(1)}%`;
const formatGb = (value: unknown) => { const result = number(value); return result === null ? '—' : `${result.toFixed(1)} GB`; };

function Gauge({ label, value }: { label: string; value: number | null }) {
  const radius = 44;
  const circumference = 2 * Math.PI * radius;
  const tone = value !== null && value >= 90 ? 'critical' : value !== null && value >= 75 ? 'warning' : 'normal';
  return <div className={`home-telemetry-gauge ${tone}`} role="meter" aria-label={`${label} utilization`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={value ?? undefined} aria-valuetext={value === null ? 'Measurement unavailable' : formatPercent(value)}>
    <svg viewBox="0 0 112 112" aria-hidden="true"><circle className="home-telemetry-gauge-track" cx="56" cy="56" r={radius}/><circle className="home-telemetry-gauge-value" cx="56" cy="56" r={radius} strokeDasharray={circumference} strokeDashoffset={circumference * (1 - (value ?? 0) / 100)}/></svg>
    <strong>{formatPercent(value)}</strong>
  </div>;
}

function Trend({ samples, field, label }: { samples: Sample[]; field: 'cpu' | 'memory'; label: string }) {
  const width = 460;
  const height = 76;
  const points = samples.map((sample, index) => `${samples.length < 2 ? 0 : index / (samples.length - 1) * width},${height - sample[field] / 100 * height}`).join(' ');
  return <div className="home-telemetry-trend" aria-label={label} role="img">
    {samples.length < 2 ? <span className="caption">Collecting live samples…</span> : <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true"><line x1="0" y1="38" x2={width} y2="38"/><polyline points={points}/></svg>}
  </div>;
}

function Meter({ title, value, detail, icon: Icon, samples, field }: { title: string; value: number | null; detail: string; icon: typeof Cpu; samples: Sample[]; field: 'cpu' | 'memory' }) {
  return <section className="card home-telemetry-meter" aria-label={`${title} meter`}>
    <div className="row gap6"><span className="square-icon sm blue"><Icon size={16}/></span><h2>{title}</h2></div>
    <div className="home-telemetry-reading"><Gauge label={title} value={value}/><p className="caption">{detail}</p></div>
    <Trend samples={samples} field={field} label={`${title} utilization from ${samples.length} measured samples`}/>
  </section>;
}

export function HomeDashboard(_props: { go?: (id: string) => void } = {}) {
  const dataMode = useWorkspaceState(state => state.dataMode);
  const demo = dataMode === 'demo';
  const [windowMinutes, setWindowMinutes] = useState<1 | 5>(1);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [createProjectOpen, setCreateProjectOpen] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [projectError, setProjectError] = useState('');
  const [creatingProject, setCreatingProject] = useState(false);
  const demoProjectCreateRequest = useWorkspaceState(state => state.demoProjectCreateRequest);
  const consumeDemoProjectCreateRequest = useWorkspaceState(state => state.consumeDemoProjectCreateRequest);
  const setSelectedDemoProject = useWorkspaceState(state => state.setSelectedDemoProject);
  const queryClient = useQueryClient();
  useEffect(() => {
    if (!demo || !demoProjectCreateRequest) return;
    setCreateProjectOpen(true);
    consumeDemoProjectCreateRequest(demoProjectCreateRequest);
  }, [demo, demoProjectCreateRequest, consumeDemoProjectCreateRequest]);
  const createDemoProject = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = newProjectName.trim();
    if (!name || name.length > 80) { setProjectError('Enter a project name of 80 characters or fewer.'); return; }
    setProjectError(''); setCreatingProject(true);
    try {
      await demoProvider.updateShellCollection<ObjectData>('projects', current => {
        if (current.some(project => String(project.name ?? '').trim().toLocaleLowerCase() === name.toLocaleLowerCase())) throw new Error('A project with this name already exists.');
        return [...current, { id: `demo-project-${Date.now().toString(36)}`, name, tag: 'Demo project', color: 'blue', files: 0, updated: 'Just now' }];
      });
      setSelectedDemoProject(name);
      await queryClient.invalidateQueries({ queryKey: ['home', 'projects', 'demo'] });
      setCreateProjectOpen(false); setNewProjectName('');
    } catch (error) { setProjectError(error instanceof Error ? error.message : 'Could not create the local demo project.'); }
    finally { setCreatingProject(false); }
  };
  const metricsQuery = useQuery({ queryKey: ['home-telemetry', 'metrics', dataMode], queryFn: ({ signal }) => demo ? Promise.resolve(demoHomeTelemetry.metrics as ObjectData) : apiGet<ObjectData>('/api/system/metrics', signal), refetchInterval: demo ? false : 3_000, retry: false });
  const healthQuery = useQuery({ queryKey: ['home-telemetry', 'health', dataMode], queryFn: async ({ signal }) => {
    if (demo) return { ...demoHomeTelemetry.health, latency_ms: null } as ObjectData;
    const started = performance.now();
    const result = await apiGet<ObjectData>('/api/health', signal);
    return { ...result, latency_ms: Math.round(performance.now() - started) };
  }, refetchInterval: demo ? false : 10_000, retry: false });
  const setupQuery = useQuery({ queryKey: ['home-telemetry', 'model-setup', dataMode], queryFn: ({ signal }) => demo ? Promise.resolve(demoHomeTelemetry.setup as ObjectData) : apiGet<ObjectData>('/api/local-ai/setup/status', signal), refetchInterval: demo ? false : 15_000, retry: false });
  const agentJobsQuery = useQuery({ queryKey: ['home-telemetry', 'agent-jobs', dataMode], queryFn: ({ signal }) => demo ? Promise.resolve(demoHomeTelemetry.agentJobs as ObjectData) : apiGet<ObjectData>('/api/agents/jobs?project=default&limit=20', signal), refetchInterval: demo ? false : 5_000, retry: false });
  const modelJobsQuery = useQuery({ queryKey: ['home-telemetry', 'model-jobs', dataMode], queryFn: ({ signal }) => demo ? Promise.resolve(demoHomeTelemetry.modelJobs as ObjectData) : apiGet<ObjectData>('/api/local-ai/jobs?limit=20', signal), refetchInterval: demo ? false : 10_000, retry: false });
  const metrics = asObject(metricsQuery.data);
  const cpu = asObject(metrics.cpu);
  const memory = asObject(metrics.memory);
  const disk = asObject(metrics.disk);
  const platform = asObject(metrics.platform);
  const cpuPercent = percent(cpu.usage_percent);
  const memoryPercent = percent(memory.used_percent);
  const diskPercent = percent(disk.used_percent);
  useEffect(() => {
    if (!metricsQuery.dataUpdatedAt || cpuPercent === null || memoryPercent === null) return;
    setSamples(previous => [...previous, { at: metricsQuery.dataUpdatedAt, cpu: cpuPercent, memory: memoryPercent }].slice(-100));
  }, [metricsQuery.dataUpdatedAt, cpuPercent, memoryPercent]);
  useEffect(() => { setSamples([]); }, [dataMode]);
  const visibleSamples = useMemo(() => samples.filter(sample => sample.at >= Date.now() - windowMinutes * 60_000), [samples, windowMinutes]);
  const health = asObject(healthQuery.data);
  const setup = asObject(setupQuery.data);
  const ollama = asObject(setup.ollama);
  const llamaCpp = asObject(setup.llamacpp);
  const agentJobs = asRows(asObject(agentJobsQuery.data).jobs);
  const modelJobs = asRows(asObject(modelJobsQuery.data).jobs);
  const jobRows: ObjectData[] = [...agentJobs.map(job => ({ ...job, source: 'Agent' })), ...modelJobs.map(job => ({ ...job, source: 'Model' }))];
  const activeJobs = jobRows.filter(job => ['queued', 'running', 'analysing', 'waiting_approval', 'pending'].includes(String(job.status ?? '').toLowerCase()));
  return <div className="page home-telemetry-page">
    {demo && <div className="home-telemetry-demo-label"><span className="tag orange">Simulated demo</span></div>}
    <div className="home-telemetry-grid">
      <Meter title="CPU" value={cpuPercent} detail={`${number(cpu.count) ?? '—'} logical cores · system utilization`} icon={Cpu} samples={visibleSamples} field="cpu"/>
      <Meter title="Memory" value={memoryPercent} detail={`${formatGb(memory.used_gb)} used of ${formatGb(memory.total_gb)}`} icon={MemoryStick} samples={visibleSamples} field="memory"/>
      <section className="card home-telemetry-meter" aria-label="Disk meter"><div className="row gap6"><span className="square-icon sm violet"><HardDrive size={16}/></span><h2>Local storage</h2></div><div className="home-telemetry-reading"><Gauge label="Disk" value={diskPercent}/><p className="caption">{formatGb(disk.free_gb)} free of {formatGb(disk.total_gb)}</p></div><div className="home-telemetry-disk"><Database size={19}/><span>OpenAgent data drive<br/><small>{String(platform.system ?? 'Local system')} {String(platform.release ?? '')}</small></span></div></section>
    </div>
    <div className="home-telemetry-controls"><span className="caption">Measured trend</span><div className="row gap6" role="group" aria-label="Trend window"><button type="button" className={`btn sm ${windowMinutes === 1 ? 'selected' : ''}`} aria-pressed={windowMinutes === 1} onClick={() => setWindowMinutes(1)}>1 min</button><button type="button" className={`btn sm ${windowMinutes === 5 ? 'selected' : ''}`} aria-pressed={windowMinutes === 5} onClick={() => setWindowMinutes(5)}>5 min</button></div><span className="caption">{visibleSamples.length} actual {visibleSamples.length === 1 ? 'sample' : 'samples'}</span></div>
    <div className="home-telemetry-lower">
      <section className="card home-telemetry-panel"><div className="section-head"><h2><Server size={18}/> Local runtime</h2></div>{setupQuery.isError ? <p role="alert">Runtime status unavailable: {apiConnectionMessage(setupQuery.error)}</p> : setupQuery.isPending ? <p className="muted small">Checking local runtimes…</p> : <><div className="home-telemetry-runtime-row"><span>Ollama</span><strong>{ollama.server_available === true ? 'Responding' : ollama.available === true ? 'Installed, not responding' : 'Unavailable'}</strong></div><div className="home-telemetry-runtime-row"><span>Selected model</span><strong>{ollama.model_ready === true ? String(ollama.selected_model ?? ollama.model ?? 'Ready') : 'Not ready'}</strong></div><div className="home-telemetry-runtime-row"><span>llama.cpp</span><strong>{llamaCpp.server_available === true ? 'Responding' : llamaCpp.available === true ? 'Installed' : 'Unavailable'}</strong></div></>}</section>
      <section className="card home-telemetry-panel"><div className="section-head"><h2><Zap size={18}/> Active execution</h2><span className="caption">Actual backend statuses</span></div>{agentJobsQuery.isError || modelJobsQuery.isError ? <p role="alert">{apiConnectionMessage(agentJobsQuery.error ?? modelJobsQuery.error)}</p> : agentJobsQuery.isPending || modelJobsQuery.isPending ? <p className="muted small">Checking active jobs…</p> : activeJobs.length ? activeJobs.slice(0, 5).map(job => <div className="home-telemetry-runtime-row" key={`${String(job.source)}-${String(job.id)}`}><span>{String(job.source)} · {String(job.goal ?? job.model_name ?? job.kind ?? job.id)}</span><strong>{String(job.status)}</strong></div>) : <p className="muted small">No Agent or model jobs are currently running.</p>}</section>
      <section className="card home-telemetry-panel"><div className="section-head"><h2><ShieldCheck size={18}/> Connection</h2></div><div className="home-telemetry-runtime-row"><span>OpenAgent backend</span><strong>{health.status === 'ok' ? 'Healthy' : healthQuery.isError ? 'Unavailable' : 'Checking'}</strong></div><div className="home-telemetry-runtime-row"><span>Local response time</span><strong>{number(health.latency_ms) === null ? '—' : `${health.latency_ms} ms`}</strong></div><div className="home-telemetry-runtime-row"><span>Last health check</span><strong>{demo ? 'Demo snapshot' : typeof health.checked_at === 'string' ? new Date(health.checked_at).toLocaleTimeString() : '—'}</strong></div></section>
    </div>
    {createProjectOpen && demo && <div className="overlay-root" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) setCreateProjectOpen(false); }}><section className="modal" role="dialog" aria-modal="true" aria-labelledby="home-create-project-title" onKeyDown={event => { if (event.key === 'Escape') setCreateProjectOpen(false); }}><div className="row between"><h2 id="home-create-project-title">Create a project</h2><button className="icon-button" type="button" aria-label="Close dialog" onClick={() => setCreateProjectOpen(false)}><X size={16}/></button></div><p>This project is stored in the local Demo only.</p><form onSubmit={event => void createDemoProject(event)}><div className="field"><label htmlFor="home-new-project-name">Project name</label><input id="home-new-project-name" className="input" autoFocus required maxLength={80} value={newProjectName} onChange={event => setNewProjectName(event.target.value)} /></div>{projectError && <p role="alert">{projectError}</p>}<div className="modal-actions"><button type="button" className="btn" onClick={() => setCreateProjectOpen(false)}>Cancel</button><button type="submit" className="btn primary" disabled={creatingProject}>{creatingProject ? 'Creating…' : 'Create preview'}</button></div></form></section></div>}
  </div>;
}
