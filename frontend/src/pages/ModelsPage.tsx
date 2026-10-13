import { useEffect, useMemo, useState } from 'react';
import { api, record, records } from '../lib/react/api';
import { PageFrame } from '../components/PageFrame';

type Row = Record<string, unknown>;
const text = (value: unknown, fallback = '—') => value == null || value === '' ? fallback : String(value);
const terminal = (job: Row) => ['completed', 'failed', 'cancelled'].includes(String(job.status).toLowerCase());
const project = () => new URL(location.href).searchParams.get('project') || localStorage.getItem('oa-project') || 'default';

export function ModelsPage() {
  const [providers, setProviders] = useState<Row[]>([]);
  const [models, setModels] = useState<Row[]>([]);
  const [adapters, setAdapters] = useState<Row[]>([]);
  const [marketplace, setMarketplace] = useState<Row[]>([]);
  const [jobs, setJobs] = useState<Row[]>([]);
  const [hardware, setHardware] = useState<Row>({});
  const [policy, setPolicy] = useState<Row>({});
  const [setupStatus, setSetupStatus] = useState<Row>({});
  const [selected, setSelected] = useState<Row>({});
  const [selectedModel, setSelectedModel] = useState<Row>({});
  const [probe, setProbe] = useState<Row>({});
  const [query, setQuery] = useState('');
  const [allowNetwork, setAllowNetwork] = useState(false);
  const [busy, setBusy] = useState(true);
  const [discovering, setDiscovering] = useState(false);
  const [probing, setProbing] = useState(false);
  const [startingJob, setStartingJob] = useState(false);
  const [modelName, setModelName] = useState('');
  const [modelSource, setModelSource] = useState('');
  const [modelPath, setModelPath] = useState('');
  const [modelBusy, setModelBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  async function loadJobs() {
    const response = await api.get<unknown>(`/api/local-ai/jobs?project=${encodeURIComponent(project())}&limit=12`);
    setJobs(records(response, 'jobs'));
  }
  async function refresh() {
    setBusy(true); setError('');
    const response = await Promise.allSettled([
      api.get<unknown>('/api/models'), api.get<unknown>('/api/local-ai/models'),
      api.get<unknown>('/api/local-ai/hardware'), api.get<unknown>('/api/model-policy'), api.get<unknown>('/api/local-ai/adapters'),
      api.get<unknown>('/api/local-ai/marketplace'), api.get<unknown>('/api/local-ai/setup/status'), loadJobs(),
    ]);
    const failed: string[] = [];
    const applyRows = (index: number, setter: (value: Row[]) => void, key: string) => {
      const item = response[index]; if (item.status === 'fulfilled') setter(records(item.value, key)); else failed.push(String(item.reason));
    };
    const applyRecord = (index: number, setter: (value: Row) => void) => {
      const item = response[index]; if (item.status === 'fulfilled') setter(record(item.value)); else failed.push(String(item.reason));
    };
    applyRows(0, setProviders, 'providers');
    applyRows(1, setModels, 'models');
    applyRecord(2, setHardware);
    applyRecord(3, setPolicy);
    applyRows(4, setAdapters, 'adapters');
    applyRows(5, setMarketplace, 'models');
    applyRecord(6, setSetupStatus);
    if (failed.length) setError(`Some model services could not be read: ${failed[0]}`);
    setBusy(false);
  }
  useEffect(() => { void refresh(); }, []);
  useEffect(() => {
    if (!jobs.some(job => !terminal(job))) return;
    const timer = window.setInterval(() => { void loadJobs().catch(cause => setError(cause instanceof Error ? cause.message : 'Model job status unavailable.')); }, 1500);
    return () => window.clearInterval(timer);
  }, [jobs]);

  const shownProviders = useMemo(() => providers.filter(item => `${item.label ?? ''} ${item.id ?? ''}`.toLowerCase().includes(query.toLowerCase())), [providers, query]);
  const shownModels = useMemo(() => models.filter(item => `${item.name ?? ''} ${item.id ?? ''} ${item.provider ?? ''}`.toLowerCase().includes(query.toLowerCase())), [models, query]);
  const shownMarketplace = useMemo(() => marketplace.filter(item => `${item.name ?? ''} ${item.id ?? ''} ${item.provider ?? ''}`.toLowerCase().includes(query.toLowerCase())), [marketplace, query]);
  const gpu = record(hardware.gpu);

  async function discover() {
    setDiscovering(true); setError(''); setNotice('');
    try { const result = record(await api.post<unknown>('/api/models/discover', {})); await refresh(); setNotice(`Discovery finished. ${Array.isArray(result.discovered) ? result.discovered.length : 0} model records reported.`); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Discovery failed.'); }
    finally { setDiscovering(false); }
  }
  async function inspectModel(model: Row) {
    const id = text(model.id, ''); if (!id) return;
    setSelectedModel(model); setProbe({}); setProbing(true); setError('');
    try { setProbe(record(await api.get<unknown>(`/api/local-ai/models/${encodeURIComponent(id)}/probe`))); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Model probe failed.'); }
    finally { setProbing(false); }
  }
  async function setup(item: Row) {
    const metadata = record(item.metadata);
    const modelName = text(metadata.model, '');
    if (!modelName) { setError('This catalog entry has no Ollama model identifier, so the setup job cannot use it.'); return; }
    const networkMessage = allowNetwork ? ' This may download model files over the network.' : ' Network downloads are disabled for this job.';
    if (!confirm(`Queue local chat-model setup for ${modelName}?${networkMessage}`)) return;
    setStartingJob(true); setError(''); setNotice('');
    try {
      const result = record(await api.post<unknown>('/api/local-ai/models/setup/job', { task: 'chat', model: modelName, allow_network: allowNetwork, project: project() }));
      const job = record(result.job); setJobs(current => [job, ...current.filter(existing => existing.id !== job.id)]);
      setNotice(`Setup job queued: ${text(job.id)} · ${text(job.status, 'queued')}.`);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Model setup could not be queued.'); }
    finally { setStartingJob(false); }
  }
  async function cancelJob(id: string) {
    if (!confirm('Cancel this model setup job?')) return;
    try { const result = record(await api.post<unknown>('/api/local-ai/jobs/cancel', { job_id: id })); const job = record(result.job); setJobs(current => current.map(item => item.id === job.id ? job : item)); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Model setup could not be cancelled.'); }
  }
  async function registerModel(event: React.FormEvent) {
    event.preventDefault(); setModelBusy('register'); setError(''); setNotice('');
    try {
      const result = record(await api.post<unknown>('/api/local-ai/models/register', { name: modelName.trim(), provider: 'local', source_url: modelSource.trim(), local_path: modelPath.trim(), metadata: {} }));
      setNotice(`Model record registered: ${text(record(result.model).name, modelName)}.`); setModelName(''); setModelSource(''); setModelPath(''); await refresh();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Model could not be registered.'); }
    finally { setModelBusy(''); }
  }
  async function downloadModel(item: Row) {
    if (!confirm(`Download ${text(item.name, text(item.id))}? The backend verifies any registered checksum. ${allowNetwork ? 'Network use is enabled for this request.' : 'Network downloads are off; remote source URLs will be rejected.'}`)) return;
    setModelBusy(String(item.id)); setError(''); setNotice('');
    try { const result = record(await api.post<unknown>('/api/local-ai/models/download', { model_id: item.id, allow_network: allowNetwork })); setNotice(`Model downloaded to ${text(record(result.model).local_path, 'the configured model store')}.`); await refresh(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Model download failed.'); }
    finally { setModelBusy(''); }
  }
  async function removeModel(item: Row) {
    const deleteFile = confirm(`Remove ${text(item.name, text(item.id))} from the registry and delete its managed local file if it is inside OpenAgent’s model store? Cancel to keep the file and only unregister it.`);
    if (!confirm(`Remove model ${text(item.name, text(item.id))}? Dependent resources may prevent removal.`)) return;
    setModelBusy(String(item.id)); setError(''); setNotice('');
    try { await api.post('/api/local-ai/models/delete', { model_id: item.id, delete_file: deleteFile }); setNotice('Model record removed.'); await refresh(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Model could not be removed.'); }
    finally { setModelBusy(''); }
  }

  return <PageFrame eyebrow="Runtime / model catalog" title="Models & providers" description="See configured providers, installed models, host capability and server-reported setup jobs." actions={<><button className="btn" onClick={() => void refresh()} disabled={busy}>Refresh</button><button className="btn primary" onClick={() => void discover()} disabled={discovering}>{discovering ? 'Discovering…' : 'Discover models'}</button></>}>
    {error && <p className="notice error" role="alert">{error}</p>}{notice && <p className="notice success" role="status">{notice}</p>}
    <section className="panel panel-body row"><strong>{text(hardware.system)} · {text(hardware.machine)}</strong><span>CPU {text(hardware.cpu_count)} · RAM {text(hardware.ram_gb)} GB · GPU {gpu.available === true ? text(gpu.adapter) : 'Not detected'}</span></section>
    <div className="row" style={{ margin: '20px 0' }}><h2>Providers and local models</h2><input className="field" style={{ maxWidth: 300 }} aria-label="Filter models" placeholder="Filter models" value={query} onChange={event => setQuery(event.target.value)} /></div>
    {busy ? <p className="empty">Checking configured runtimes…</p> : <div className="workspace">
      <section className="panel"><div className="panel-head"><h2>Configured providers</h2></div><div className="panel-body list">{shownProviders.map(item => <button key={String(item.id)} aria-current={selected.id === item.id} onClick={() => setSelected(item)}><strong>{text(item.label, text(item.id))}</strong><small>{text(item.privacy_warning, item.local === true ? 'Local provider' : 'Provider status from OpenAgent')} · {item.available === true ? 'Reachable' : item.available === false ? 'Not reachable' : 'Status unknown'}</small></button>)}{!shownProviders.length && <p className="empty">No provider records match this filter.</p>}</div></section>
      <section className="panel"><div className="panel-head"><h2>Provider details</h2></div><div className="panel-body">{selected.id ? <><dl className="facts">{['id', 'default_model', 'network', 'default_model_ready', 'privacy_warning'].map(key => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{text(selected[key])}</dd></div>)}</dl><h3>Installed models</h3><p>{Array.isArray(selected.installed_models) ? selected.installed_models.map(String).join(', ') : 'No installed model list reported.'}</p></> : <p className="empty">Select a provider to inspect its reported configuration.</p>}<details><summary>Runtime setup capability</summary><pre>{JSON.stringify(setupStatus, null, 2)}</pre></details></div></section>
    </div>}
    <section className="panel" style={{ marginTop: 16 }}><div className="panel-head"><h2>Local model inventory</h2><span>{shownModels.length} records</span></div><div className="panel-body">{shownModels.length ? <div className="table-wrap"><table><thead><tr><th>Model</th><th>Provider</th><th>State</th><th>Size</th><th>Actions</th></tr></thead><tbody>{shownModels.map(item => <tr key={String(item.id)}><td><button className="text-button" onClick={() => void inspectModel(item)}>{text(item.name, text(item.id))}</button><small>{text(item.id)}</small></td><td>{text(item.provider)}</td><td>{text(item.status)}</td><td>{item.size_bytes != null && item.size_bytes !== 0 ? `${(Number(item.size_bytes) / 1024 ** 3).toFixed(1)} GB` : '—'}</td><td className="toolbar"><button className="btn" onClick={() => void inspectModel(item)} disabled={probing}>{selectedModel.id === item.id && probing ? 'Checking…' : 'Probe'}</button>{item.local_path ? <button className="btn danger" disabled={!!modelBusy} onClick={() => void removeModel(item)}>Remove</button> : <button className="btn" disabled={!!modelBusy || !item.source_url} onClick={() => void downloadModel(item)}>{modelBusy === item.id ? 'Downloading…' : 'Download'}</button>}</td></tr>)}</tbody></table></div> : <p className="empty">No local model records returned.</p>}{selectedModel.id != null && selectedModel.id !== '' ? <details open><summary>Probe result · {text(selectedModel.name, text(selectedModel.id))}</summary><pre>{JSON.stringify(probe, null, 2)}</pre></details> : null}<details><summary>Active routing policy</summary><pre>{JSON.stringify(policy, null, 2)}</pre></details></div></section>
    <div className="workspace" style={{ marginTop: 16 }}><form className="panel panel-body stack" onSubmit={registerModel}><h2>Register a model source</h2><label>Model name<input className="field" value={modelName} onChange={event => setModelName(event.target.value)} required /></label><label>Local file path<input className="field" value={modelPath} onChange={event => setModelPath(event.target.value)} placeholder="Optional path to an existing model file" /></label><label>Download source URL<input className="field" type="url" value={modelSource} onChange={event => setModelSource(event.target.value)} placeholder="Optional https://…" /></label><small className="tiny">Registration stores model metadata. It does not download the file or contact the URL.</small><button className="btn primary" disabled={modelBusy === 'register'}>{modelBusy === 'register' ? 'Registering…' : 'Register model'}</button></form><section className="panel"><div className="panel-head"><h2>Adapters</h2><span>{adapters.length} records</span></div><div className="panel-body list">{adapters.map((adapter, index) => <article className="record-row" key={text(adapter.id, String(index))}><div><strong>{text(adapter.name, text(adapter.adapter_id, 'Adapter'))}</strong><p>{text(adapter.model_id)} · {text(adapter.status)}</p></div><small>{text(adapter.path, 'Path not reported')}</small></article>)}{!adapters.length && <p className="empty">No acceleration adapters returned.</p>}</div></section></div>
    <section className="panel" style={{ marginTop: 16 }}><div className="panel-head"><h2>Setup catalog</h2><span>Only entries with a returned Ollama model ID can be queued here</span></div><div className="panel-body"><label className="record-row">Permit network download for the next setup job<input type="checkbox" checked={allowNetwork} onChange={event => setAllowNetwork(event.target.checked)} /></label><p className="tiny">Off by default. Every queued setup still asks for confirmation and the backend records `allow_network` with the job.</p><div className="table-wrap"><table><thead><tr><th>Catalog model</th><th>Engine</th><th>Requirements</th><th>Action</th></tr></thead><tbody>{shownMarketplace.map(item => { const metadata = record(item.metadata); return <tr key={String(item.id)}><td>{text(item.name, text(item.id))}<small>{text(metadata.model, 'No direct Ollama identifier')}</small></td><td>{text(item.recommended_engine, text(item.provider))}</td><td>{text(metadata.ram_gb, '—')} GB RAM · {text(metadata.quantization, 'Unreported quantization')}</td><td><button className="btn" disabled={startingJob || !metadata.model} onClick={() => void setup(item)}>{startingJob ? 'Queuing…' : 'Queue chat setup'}</button></td></tr>; })}</tbody></table></div></div></section>
    <section className="panel" style={{ marginTop: 16 }}><div className="panel-head"><h2>Model setup jobs</h2><button className="btn" onClick={() => void loadJobs()}>Refresh jobs</button></div><div className="panel-body list">{jobs.map(job => <article className="record-row" key={String(job.id)}><div><strong>{text(job.model_name, 'Model setup')}</strong><p>{text(job.current_step, text(job.kind))} · {text(job.status)}</p><small>{Math.max(0, Math.min(100, Number(job.progress ?? 0)))}% · {text(job.id)}</small>{job.error != null && job.error !== '' ? <p className="notice error">{String(job.error)}</p> : null}</div>{!terminal(job) && <button className="btn danger" onClick={() => void cancelJob(String(job.id))}>Cancel job</button>}</article>)}{!jobs.length && <p className="empty">No setup jobs reported for this project.</p>}</div></section>
  </PageFrame>;
}
