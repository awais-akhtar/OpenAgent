import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, Cpu, HardDrive, RefreshCw, ShieldCheck } from 'lucide-react';
import { api, ApiError, record, records } from '../lib/react/api';
import './DashboardPage.css';

type Row = Record<string, unknown>;
type Snapshot = { health: Row; metrics: Row; modules: Row; inference: Row; localAI: Row; audit: Row; approvals: Row; jobs: Row };
const empty: Snapshot = { health: {}, metrics: {}, modules: {}, inference: {}, localAI: {}, audit: {}, approvals: {}, jobs: {} };
const read = (source: unknown, key: string) => record(source)[key];
const display = (value: unknown, suffix = '') => value === undefined || value === null || value === '' ? '—' : `${String(value)}${suffix}`;
const requestError = (reason: unknown) => {
  if (reason instanceof ApiError && reason.status === 401) return 'Sign in again to read workspace activity.';
  if (reason instanceof ApiError && reason.status === 403) return 'Your account cannot read this workspace activity.';
  if (reason instanceof ApiError && reason.status === 404) return 'An activity service is not available in this server version.';
  if (reason instanceof ApiError && reason.status >= 500) return 'The local service returned an error. Retry when it is available.';
  return reason instanceof Error ? reason.message : 'The local service is unavailable.';
};
const timestamp = (value: unknown) => {
  if (value === undefined || value === null || value === '') return 'Time unavailable';
  const date = new Date(String(value));
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
};

/** The legacy dashboard destination is now a compact, server-backed Activity view. */
export function DashboardPage({ project }: { project: string }) {
  const [data, setData] = useState<Snapshot>(empty);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [updatedAt, setUpdatedAt] = useState('');
  const [history, setHistory] = useState<{ at: number; cpu?: number; memory?: number }[]>([]);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    setRefreshing(true);
    setError('');
    const scopedProject = encodeURIComponent(project || 'default');
    const results = await Promise.allSettled([
      api.get<unknown>('/api/health', signal),
      api.get<unknown>('/api/system/metrics', signal),
      api.get<unknown>(`/api/modules/health?project=${scopedProject}`, signal),
      api.get<unknown>('/api/inference/status', signal),
      api.get<unknown>('/api/local-ai/status', signal),
      api.get<unknown>(`/api/audit?project=${scopedProject}&limit=8`, signal),
      api.get<unknown>(`/api/agents/approvals?project=${scopedProject}&status=pending`, signal),
      api.get<unknown>(`/api/worker/jobs?project=${scopedProject}&limit=8`, signal),
    ]);
    if (signal?.aborted) return;
    const keys: (keyof Snapshot)[] = ['health', 'metrics', 'modules', 'inference', 'localAI', 'audit', 'approvals', 'jobs'];
    setData(previous => {
      const next = { ...previous };
      results.forEach((result, index) => { if (result.status === 'fulfilled') next[keys[index]] = record(result.value); });
      return next;
    });
    const rejected = results.filter((result): result is PromiseRejectedResult => result.status === 'rejected');
    if (rejected.length) {
      const representative = results[0].status === 'rejected' ? results[0].reason : rejected.find(result => result.reason instanceof ApiError && [401, 403, 404].includes(result.reason.status))?.reason ?? rejected[0].reason;
      setError(results[0].status === 'rejected' || representative instanceof ApiError ? requestError(representative) : 'Some activity sources could not be reached.');
    }
    if (results[1].status === 'fulfilled') {
      const metrics = record(results[1].value); const cpu = Number(read(metrics.cpu, 'usage_percent')); const memory = Number(read(metrics.memory, 'used_percent'));
      setHistory(current => [...current, { at: Date.now(), ...(Number.isFinite(cpu) ? { cpu } : {}), ...(Number.isFinite(memory) ? { memory } : {}) }].slice(-20));
    }
    setUpdatedAt(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    setLoading(false);
    setRefreshing(false);
  }, [project]);

  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    return () => controller.abort();
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
  const modules = useMemo(() => records(data.modules, 'modules'), [data.modules]);
  const approvals = useMemo(() => records(data.approvals, 'approvals').filter(item => String(item.status ?? 'pending').toLowerCase() === 'pending'), [data.approvals]);
  const jobs = useMemo(() => records(data.jobs, 'jobs'), [data.jobs]);
  const cpu = read(data.metrics, 'cpu'); const memory = read(data.metrics, 'memory'); const disk = read(data.metrics, 'disk');
  const hardware = record(read(data.localAI, 'hardware'));
  const healthStatus = String(read(data.health, 'status') ?? '').toLowerCase();
  const serviceHealthy = healthStatus === 'ok' || healthStatus === 'healthy';
  const pointLine = (key: 'cpu' | 'memory') => {
    const points = history.filter(item => item[key] !== undefined);
    if (points.length < 2) return '';
    return points.map((item, index) => `${(index / (points.length - 1)) * 100},${100 - Math.max(0, Math.min(100, item[key] as number))}`).join(' ');
  };
  const inferenceStatus = read(data.inference, 'status') ?? read(data.inference, 'provider');

  return <main className="activity-page">
    <header className="activity-header">
      <div><h1>Activity</h1><p>Workspace status and recent events</p></div>
      <button className="activity-refresh" onClick={() => void refresh()} disabled={refreshing} aria-label="Refresh activity">
        <RefreshCw size={15} className={refreshing ? 'activity-spin' : ''}/><span>Refresh</span>
      </button>
    </header>

    {error && <div className="activity-error" role="alert">
      <span>{error}</span><button onClick={() => void refresh()} disabled={refreshing}>Retry</button>
    </div>}

    {loading ? <div className="activity-loading" role="status" aria-label="Loading activity">
      <span className="activity-sr-only">Loading activity…</span>
      <i aria-hidden="true"/><i aria-hidden="true"/><i aria-hidden="true"/>
    </div> : <>
      <section className="activity-status" aria-label="Live workspace status">
        <div className="activity-status-title"><Activity size={16}/><span>System</span><span className={`activity-state ${serviceHealthy ? 'is-good' : ''}`}><i/>{display(read(data.health, 'status'))}</span></div>
        <dl>
          <div><dt>Inference</dt><dd>{display(inferenceStatus)}{read(data.inference, 'model') ? <small>{display(read(data.inference, 'model'))}</small> : null}</dd></div>
          <div><dt>Processor</dt><dd>{display(read(cpu, 'usage_percent'), '%')}{read(cpu, 'count') !== undefined && <small>{display(read(cpu, 'count'))} logical cores</small>}</dd></div>
          <div><dt>Memory</dt><dd>{display(read(memory, 'used_percent'), '%')}{read(memory, 'used_gb') !== undefined && <small>{display(read(memory, 'used_gb'), ' GB')} of {display(read(memory, 'total_gb'), ' GB')}</small>}</dd></div>
          <div><dt>Storage</dt><dd>{display(read(disk, 'used_percent'), '%')}{read(disk, 'free_gb') !== undefined && <small>{display(read(disk, 'free_gb'), ' GB')} free</small>}</dd></div>
          <div><dt>Approvals</dt><dd>{approvals.length}<small>Pending</small></dd></div>
        </dl>
        <div className="activity-system-foot">
          <span>{[hardware.system, hardware.release].filter(Boolean).join(' ') || 'Host details unavailable'}</span>
          <span>{display(read(data.metrics, 'platform') && read(read(data.metrics, 'platform'), 'machine'))}</span>
          <span>Updated {updatedAt || '—'}</span>
        </div>
      </section>

      <section className="activity-section" aria-labelledby="activity-events-title">
        <div className="activity-section-head"><h2 id="activity-events-title">Recent events</h2><span>Project audit</span></div>
        {events.length ? <ol className="activity-list">{events.map((event, index) => {
          const outcome = String(event.status ?? event.outcome ?? '').toLowerCase();
          return <li key={String(event.id ?? `${event.type}-${event.created_at}-${index}`)}>
            <span className={`activity-event-mark ${outcome.includes('fail') || outcome.includes('denied') ? 'is-warning' : ''}`}><ShieldCheck size={14}/></span>
            <div><strong>{display(event.title ?? event.type ?? event.action)}</strong><small>{display(event.resource_type ?? event.resource_id ?? event.project)}{event.project ? ` · ${String(event.project)}` : ''}</small></div>
            <span className="activity-event-outcome">{display(event.status ?? event.outcome)}</span>
            <time>{timestamp(event.created_at ?? event.timestamp)}</time>
          </li>;
        })}</ol> : <p className="activity-empty">No recent audit events.</p>}
      </section>

      <section className="activity-section" aria-labelledby="activity-runs-title">
        <div className="activity-section-head"><h2 id="activity-runs-title">Worker runs</h2><span>{jobs.length} recent</span></div>
        {jobs.length ? <ol className="activity-list">{jobs.map((job, index) => <li key={String(job.id ?? index)}>
          <span className="activity-event-mark"><Cpu size={14}/></span>
          <div><strong>{display(job.kind ?? job.type ?? job.name)}</strong><small>{display(job.id)}</small></div>
          <span className="activity-event-outcome">{display(job.state ?? job.status)}</span>
          <time>{timestamp(job.updated_at ?? job.created_at)}</time>
        </li>)}</ol> : <p className="activity-empty">No worker runs for this project.</p>}
      </section>

      <details className="activity-details">
        <summary><span>Runtime details</span><span>{modules.length} modules · {history.length} samples</span></summary>
        <div className="activity-details-body">
          <div className="activity-modules"><h3>Module status</h3>{modules.length ? <ul>{modules.map((module, index) => {
            const ok = String(module.status).toLowerCase() === 'ok' || module.available === true;
            return <li key={String(module.id ?? module.name ?? index)}><span>{display(module.label ?? module.name ?? module.id)}</span><span className={ok ? 'is-good' : ''}>{display(module.status ?? (module.available === true ? 'Available' : module.available === false ? 'Unavailable' : 'Unknown'))}</span></li>;
          })}</ul> : <p>Module status has not returned any records.</p>}</div>
          <div className="activity-machine"><h3>Host</h3><p>{display(hardware.system)} {display(hardware.release)} · {display(hardware.processor)} · {display(hardware.cpu_count)} cores</p>
            <h3><HardDrive size={14}/> Measured samples</h3>
            {history.length < 2 ? <p>History needs another real metrics sample.</p> : <svg viewBox="0 0 100 100" role="img" aria-label="Measured processor and memory history"><polyline points={pointLine('cpu')} fill="none" stroke="var(--oa-chart-a)" strokeWidth="2" vectorEffect="non-scaling-stroke"/><polyline points={pointLine('memory')} fill="none" stroke="var(--oa-chart-c)" strokeWidth="2" vectorEffect="non-scaling-stroke"/></svg>}
          </div>
        </div>
      </details>
    </>}
  </main>;
}
