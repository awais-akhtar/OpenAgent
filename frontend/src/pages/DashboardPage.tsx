import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, Cpu, HardDrive, RefreshCw, ShieldCheck, Sparkles } from 'lucide-react';
import { api, record, records } from '../lib/react/api';

type Row = Record<string, unknown>;
type Snapshot = { health: Row; metrics: Row; modules: Row; inference: Row; localAI: Row; audit: Row; approvals: Row; jobs: Row };
const empty: Snapshot = { health: {}, metrics: {}, modules: {}, inference: {}, localAI: {}, audit: {}, approvals: {}, jobs: {} };
const read = (source: unknown, key: string) => record(source)[key];
const display = (value: unknown, suffix = '') => value === undefined || value === null || value === '' ? '—' : `${String(value)}${suffix}`;
const errorMessage = (value: unknown) => value instanceof Error ? value.message : 'The local service is unavailable.';

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
    if (results[0].status === 'rejected') setError(errorMessage(results[0].reason));
    else if (results.some(result => result.status === 'rejected')) setError('Some local status sources could not be reached.');
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
  const moduleRows = useMemo(() => records(data.modules, 'modules'), [data.modules]);
  const approvalRows = useMemo(() => records(data.approvals, 'approvals').filter(item => String(item.status ?? 'pending').toLowerCase() === 'pending'), [data.approvals]);
  const jobRows = useMemo(() => records(data.jobs, 'jobs'), [data.jobs]);
  const hardware = record(read(data.localAI, 'hardware'));
  const cpu = read(data.metrics, 'cpu');
  const memory = read(data.metrics, 'memory');
  const disk = read(data.metrics, 'disk');
  const healthStatus = String(read(data.health, 'status') ?? '').toLowerCase();
  const healthy = healthStatus === 'ok' || healthStatus === 'healthy';
  const pct = (value: unknown) => {
    if (value === undefined || value === null || value === '') return undefined;
    const number = Number(value);
    return Number.isFinite(number) ? Math.max(0, Math.min(100, number)) : undefined;
  };
  const line = (key: 'cpu' | 'memory') => {
    const points = history.filter(item => item[key] !== undefined);
    if (points.length < 2) return '';
    const max = 100; const width = 600; const height = 100;
    return points.map((item, index) => `${(index / (points.length - 1)) * width},${height - (Math.max(0, Math.min(max, item[key] as number)) / max) * height}`).join(' ');
  };

  return <main className="dash-page">
    <header className="dash-head"><div><p className="dash-eyebrow">WORKSPACE / OVERVIEW</p><h1>Your local workbench</h1><p className="dash-lede">A clear view of service health, runtime capacity and recent activity.</p></div><button className="dash-button" onClick={() => void refresh()} disabled={refreshing} aria-label="Refresh overview"><RefreshCw size={16} className={refreshing ? 'dash-spin' : ''}/> Refresh</button></header>
    {error && <div className="dash-banner" role="status"><strong>Local service needs attention</strong><span>{error}</span><button className="dash-button dash-quiet" onClick={() => void refresh()}>Retry</button></div>}
    {loading ? <div className="dash-loading" aria-live="polite">Reading local status…</div> : <>
      <section className="dash-summary" aria-label="Workspace status">
        <article className="dash-tile"><span className="dash-glyph"><Activity size={18}/></span><div><span className="dash-label">Local service</span><strong>{healthy ? 'Available' : display(read(data.health, 'status'))}</strong><small>{display(read(data.health, 'version') ?? read(data.health, 'message'))}</small></div><i className={healthy ? 'good' : ''}/></article>
        <article className="dash-tile"><span className="dash-glyph"><Sparkles size={18}/></span><div><span className="dash-label">Inference</span><strong>{display(read(data.inference, 'status') ?? read(data.inference, 'provider'))}</strong><small>{display(read(data.inference, 'model'))}</small></div></article>
        <article className="dash-tile"><span className="dash-glyph"><Cpu size={18}/></span><div><span className="dash-label">Processor load</span><strong>{display(read(cpu, 'usage_percent'), '%')}</strong><small>{display(read(cpu, 'count'), ' logical cores')}</small></div></article>
        <article className="dash-tile"><span className="dash-glyph"><HardDrive size={18}/></span><div><span className="dash-label">Memory in use</span><strong>{display(read(memory, 'used_percent'), '%')}</strong><small>{display(read(memory, 'used_gb'), ' GB')} of {display(read(memory, 'total_gb'), ' GB')}</small></div></article>
        <article className="dash-tile"><span className="dash-glyph"><ShieldCheck size={18}/></span><div><span className="dash-label">Pending approvals</span><strong>{approvalRows.length}</strong><small>From the local approval queue</small></div></article>
      </section>
      <div className="dash-columns">
        <section className="dash-panel"><div className="dash-panel-head"><div><p className="dash-eyebrow">CAPACITY</p><h2>Local resources</h2></div><span className="dash-stamp">OS reported</span></div>
          <div className="dash-resources"><div><div><span>Memory</span><b>{display(read(memory, 'used_percent'), '%')}</b></div>{pct(read(memory, 'used_percent')) === undefined ? <span className="dash-unmeasured">No memory measurement reported</span> : <progress max={100} value={pct(read(memory, 'used_percent'))} aria-label="Memory used percent"/>}<small>{display(read(memory, 'used_gb'), ' GB')} used · {display(read(memory, 'available_gb'), ' GB')} available</small></div>
            <div><div><span>Storage</span><b>{display(read(disk, 'used_percent'), '%')}</b></div>{pct(read(disk, 'used_percent')) === undefined ? <span className="dash-unmeasured">No storage measurement reported</span> : <progress max={100} value={pct(read(disk, 'used_percent'))} aria-label="Storage used percent"/>}<small>{display(read(disk, 'free_gb'), ' GB')} free · {display(read(disk, 'total_gb'), ' GB')} total</small></div></div>
          <div className="dash-machine"><span>Host</span><strong>{display(hardware.system)} {display(hardware.release)}</strong><span>Processor</span><strong>{display(hardware.processor)} · {display(hardware.cpu_count)} cores</strong><span>Platform</span><strong>{display(read(read(data.metrics, 'platform'), 'machine'))}</strong></div>
        </section>
        <section className="dash-panel"><div className="dash-panel-head"><div><p className="dash-eyebrow">CAPABILITIES</p><h2>Module health</h2></div><span className="dash-stamp">{moduleRows.length} reported</span></div>
          {moduleRows.length ? <ul className="dash-modules">{moduleRows.slice(0, 8).map((item, index) => { const good = String(item.status).toLowerCase() === 'ok' || item.available === true; return <li key={String(item.id ?? item.name ?? index)}><span>{display(item.label ?? item.name ?? item.id)}</span><span className={`dash-status ${good ? 'good' : ''}`}>{display(item.status ?? (item.available === true ? 'Available' : item.available === false ? 'Unavailable' : 'Unknown'))}</span></li>; })}</ul> : <p className="dash-empty">Module health has not returned any records.</p>}
          {read(data.localAI, 'models') !== undefined && <p className="dash-footnote">Local model records: {display(read(data.localAI, 'models'))} · Availability depends on the configured runtime.</p>}
        </section>
      </div>
      <section className="dash-panel dash-history"><div className="dash-panel-head"><div><p className="dash-eyebrow">MEASURED SAMPLES</p><h2>Resource history</h2></div><span className="dash-stamp">Local API · 30 second sampling while visible</span></div>
        {history.length < 2 ? <p className="dash-empty">History needs a second real metrics sample. No values are interpolated.</p> : <><svg viewBox="0 0 600 120" role="img" aria-label={`Measured processor history ${line('cpu') ? 'and' : ''} memory history over ${history.length} local samples`}><polyline points={line('cpu')} fill="none" stroke="var(--oa-primary)" strokeWidth="3" vectorEffect="non-scaling-stroke"/><polyline points={line('memory')} fill="none" stroke="#7c8da6" strokeWidth="3" vectorEffect="non-scaling-stroke"/></svg><div className="dash-legend"><span><i/>Processor use</span><span><i/>Memory use</span><small>{new Date(history[0].at).toLocaleTimeString()} – {new Date(history[history.length - 1].at).toLocaleTimeString()}</small></div></>}
      </section>
      <section className="dash-panel dash-activity"><div className="dash-panel-head"><div><p className="dash-eyebrow">AUDIT TRAIL</p><h2>Recent activity</h2></div><span className="dash-stamp">Updated {updatedAt || '—'}</span></div>
        {events.length ? <div className="dash-table-wrap"><table><thead><tr><th scope="col">Event</th><th scope="col">Resource</th><th scope="col">Time</th><th scope="col">Outcome</th></tr></thead><tbody>{events.map((event, index) => <tr key={String(event.id ?? `${event.type}-${event.created_at}-${index}`)}><td><strong>{display(event.title ?? event.type ?? event.action)}</strong></td><td>{display(event.resource_type ?? event.resource_id ?? event.project)}</td><td>{display(event.created_at ?? event.timestamp)}</td><td>{display(event.status ?? event.outcome)}</td></tr>)}</tbody></table></div> : <div className="dash-empty"><ShieldCheck size={20}/><p>No recent audit events were returned for this project.</p></div>}
      </section>
      <section className="dash-panel dash-activity"><div className="dash-panel-head"><div><p className="dash-eyebrow">WORKER QUEUE</p><h2>Recent runs</h2></div><span className="dash-stamp">{jobRows.length} reported</span></div>
        {jobRows.length ? <div className="dash-table-wrap"><table><thead><tr><th scope="col">Job</th><th scope="col">State</th><th scope="col">Updated</th></tr></thead><tbody>{jobRows.map((job, index) => <tr key={String(job.id ?? index)}><td><strong>{display(job.kind ?? job.type ?? job.name)}</strong></td><td>{display(job.state ?? job.status)}</td><td>{display(job.updated_at ?? job.created_at)}</td></tr>)}</tbody></table></div> : <p className="dash-empty">No worker jobs were returned for this project.</p>}
      </section>
    </>}
    <style>{`
      .dash-page{max-width:1440px;margin:0 auto;padding:34px clamp(18px,4vw,54px) 48px;color:var(--oa-text)}.dash-head,.dash-panel-head{display:flex;align-items:flex-start;justify-content:space-between;gap:20px}.dash-head{margin-bottom:26px}.dash-eyebrow{font-size:10px;letter-spacing:.13em;font-weight:700;color:var(--oa-muted);margin:0 0 8px}.dash-page h1{font-size:clamp(27px,3vw,36px);letter-spacing:-.035em;margin:0}.dash-lede{color:var(--oa-muted);margin:8px 0 0;font-size:14px}.dash-button{display:inline-flex;align-items:center;gap:8px;min-height:38px;padding:0 13px;border:1px solid var(--oa-border);border-radius:9px;background:var(--oa-surface);color:var(--oa-text);font:inherit;font-weight:600;cursor:pointer}.dash-button:hover{background:var(--oa-surface-muted)}.dash-button:focus-visible{outline:2px solid var(--oa-primary);outline-offset:2px}.dash-button:disabled{opacity:.6;cursor:wait}.dash-quiet{background:transparent}.dash-summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:16px}.dash-tile,.dash-panel{background:var(--oa-surface);border:1px solid var(--oa-border);border-radius:12px}.dash-tile{display:flex;align-items:flex-start;gap:12px;padding:17px;position:relative;min-width:0}.dash-glyph{display:grid;place-items:center;background:var(--oa-surface-muted);border-radius:9px;width:36px;height:36px;color:var(--oa-primary);flex:none}.dash-tile>div{display:grid;gap:4px;min-width:0}.dash-tile strong{font-size:19px;letter-spacing:-.02em}.dash-tile small,.dash-footnote{font-size:11px;color:var(--oa-muted);overflow-wrap:anywhere}.dash-label{font-size:12px;color:var(--oa-muted)}.dash-tile>i{position:absolute;width:7px;height:7px;border-radius:50%;right:15px;top:17px;background:var(--oa-muted)}.dash-tile>i.good{background:#198a6a}.dash-columns{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}.dash-panel{padding:20px;min-width:0;margin-bottom:16px}.dash-panel-head{align-items:center;margin-bottom:17px}.dash-page h2{margin:0;font-size:17px;letter-spacing:-.02em}.dash-stamp{font-size:11px;color:var(--oa-muted);border:1px solid var(--oa-border);border-radius:999px;padding:5px 9px;white-space:nowrap}.dash-resources{display:grid;gap:17px}.dash-resources>div>div{display:flex;justify-content:space-between;font-size:13px}.dash-resources b{font-variant-numeric:tabular-nums}.dash-resources progress{display:block;width:100%;height:6px;margin:9px 0;background:var(--oa-surface-muted);border:0;border-radius:9px;overflow:hidden}.dash-resources progress::-webkit-progress-bar{background:var(--oa-surface-muted)}.dash-resources progress::-webkit-progress-value{background:var(--oa-primary);border-radius:9px}.dash-resources progress::-moz-progress-bar{background:var(--oa-primary)}.dash-resources .dash-unmeasured{display:block;margin:9px 0;font-size:11px;color:var(--oa-muted)}.dash-resources small{font-size:11px;color:var(--oa-muted)}.dash-machine{border-top:1px solid var(--oa-border);margin-top:19px;padding-top:14px;display:grid;grid-template-columns:100px 1fr;gap:10px;font-size:12px}.dash-machine span{color:var(--oa-muted)}.dash-machine strong{font-weight:550;overflow-wrap:anywhere}.dash-modules{list-style:none;margin:0;padding:0}.dash-modules li{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--oa-border);font-size:12px}.dash-modules li:last-child{border-bottom:0}.dash-status{color:#aa6543}.dash-status.good{color:#138365}.dash-empty{display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px;min-height:90px;color:var(--oa-muted);font-size:13px;text-align:center}.dash-table-wrap{overflow:auto}.dash-page table{border-collapse:collapse;width:100%;font-size:12px;text-align:left}.dash-page th{color:var(--oa-muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em;font-weight:650}.dash-page th,.dash-page td{padding:11px 10px;border-bottom:1px solid var(--oa-border)}.dash-page tbody tr:last-child td{border-bottom:0}.dash-page td strong{font-weight:600}.dash-history svg{width:100%;max-height:150px;overflow:visible}.dash-legend{display:flex;gap:16px;align-items:center;color:var(--oa-muted);font-size:11px}.dash-legend span{display:inline-flex;align-items:center;gap:6px}.dash-legend i{width:10px;height:3px;background:var(--oa-primary)}.dash-legend span+span i{background:#7c8da6}.dash-legend small{margin-left:auto}.dash-banner{display:flex;align-items:center;gap:14px;flex-wrap:wrap;border:1px solid #bb8c5a55;background:#bb8c5a12;padding:12px 14px;border-radius:10px;margin-bottom:16px}.dash-banner strong{font-size:12px}.dash-banner span{font-size:12px;color:var(--oa-muted);flex:1}.dash-loading{padding:60px 0;text-align:center;color:var(--oa-muted)}.dash-spin{animation:dash-spin 1s linear infinite}@keyframes dash-spin{to{transform:rotate(360deg)}}@media(max-width:680px){.dash-page{padding:24px 16px}.dash-head{align-items:flex-start}.dash-head .dash-button{font-size:0;padding:0 10px}.dash-head .dash-button svg{width:17px}.dash-columns{grid-template-columns:1fr}.dash-panel{padding:16px}.dash-machine{grid-template-columns:82px 1fr}}@media(prefers-reduced-motion:reduce){.dash-spin{animation:none}}
    `}</style>
  </main>;
}
