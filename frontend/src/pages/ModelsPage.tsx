import { useEffect, useMemo, useState } from 'react';
import { api, record, records } from '../lib/react/api';
import { PageFrame } from '../components/PageFrame';
import { Sparkles } from 'lucide-react';
import './ModelsPage.css';

type Row = Record<string, unknown>;
const text = (value: unknown, fallback = '—') => value == null || value === '' ? fallback : String(value);
const terminal = (job: Row) => ['completed', 'failed', 'cancelled'].includes(String(job.status).toLowerCase());
const project = () => new URL(location.href).searchParams.get('project') || localStorage.getItem('oa-project') || 'default';
const stateLabel = (value: unknown) => {
  if (value === true) return 'Available';
  if (value === false) return 'Unavailable';
  if (typeof value !== 'string' || !value.trim()) return 'Not reported';
  const normalized = value.toLowerCase();
  if (['ready', 'healthy', 'available', 'active', 'ok', 'success', 'succeeded'].includes(normalized)) return 'Ready';
  if (['unavailable', 'failed', 'error', 'disabled', 'inactive'].includes(normalized)) return 'Unavailable';
  if (normalized === 'not_installed') return 'Not installed';
  if (['unknown', 'registered'].includes(normalized)) return 'Not reported';
  return normalized.replaceAll('_', ' ').replace(/\b\w/g, character => character.toUpperCase());
};
const jobProgress = (job: Row) => {
  if (job.progress == null || job.progress === '') return undefined;
  const value = Number(job.progress);
  return Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : undefined;
};
const runtimeState = (provider: Row) => provider.available === true ? 'Available' : provider.available === false ? 'Unavailable' : 'Availability not reported';
const locationState = (provider: Row) => provider.local === true ? 'Local' : provider.local === false ? 'External' : 'Routing provider';
const runtimeAvailability = (runtime: Row) => runtime.available === true ? 'Installed' : runtime.available === false ? 'Not installed' : 'Not reported';
const runtimeDescription = (runtime: Row) => {
  if (runtime.server_available === true) return 'Local server responding';
  if (runtime.server_available === false && runtime.available === true) return 'Runtime installed; local server not responding';
  if (runtime.server_available === false && runtime.available === false) return 'Runtime not found on this device';
  if (runtime.available === true) return 'Runtime available; server state not reported';
  if (runtime.available === false) return 'Runtime not found on this device';
  return 'Runtime details not reported';
};
const runtimeModelReadiness = (runtime: Row) => {
  const ready = typeof runtime.default_model_ready === 'boolean' ? runtime.default_model_ready : runtime.model_ready;
  if (ready === true) return `Model ready · ${text(runtime.selected_model, text(runtime.model, 'configured default'))}`;
  if (ready === false) return `No ready model · ${text(runtime.model, 'default not reported')}`;
  return 'Model readiness not reported';
};

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
  const [providerReadError, setProviderReadError] = useState('');
  const [readFailures, setReadFailures] = useState<string[]>([]);
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
    const readAreas = ['providers', 'local-models', 'hardware', 'routing', 'adapters', 'catalog', 'setup', 'jobs'];
    const failedAreas = response.flatMap((item, index) => item.status === 'rejected' ? [readAreas[index]] : []);
    setReadFailures(failedAreas);
    const applyRows = (index: number, setter: (value: Row[]) => void, key: string) => {
      const item = response[index];
      if (item.status === 'fulfilled') {
        const rows = records(item.value, key); setter(rows);
        if (index === 0) setSelected(current => rows.find(provider => provider.id === current.id) ?? rows.find(provider => provider.available === true) ?? rows[0] ?? {});
      } else if (index === 0) setProviderReadError('Provider availability could not be read.');
    };
    const applyRecord = (index: number, setter: (value: Row) => void) => {
      const item = response[index]; if (item.status === 'fulfilled') setter(record(item.value));
    };
    applyRows(0, setProviders, 'providers');
    if (response[0].status === 'fulfilled') setProviderReadError('');
    applyRows(1, setModels, 'models');
    applyRecord(2, setHardware);
    applyRecord(3, setPolicy);
    applyRows(4, setAdapters, 'adapters');
    applyRows(5, setMarketplace, 'models');
    applyRecord(6, setSetupStatus);
    if (failedAreas.length) setError(`Could not read ${failedAreas.join(', ')}. Data from available model services is still shown.`);
    setBusy(false);
  }
  useEffect(() => { void refresh(); }, []);
  useEffect(() => {
    if (!jobs.some(job => !terminal(job))) return;
    const timer = window.setInterval(() => { void loadJobs().catch(() => setError('Model setup activity could not be refreshed.')); }, 1500);
    return () => window.clearInterval(timer);
  }, [jobs]);

  const shownProviders = useMemo(() => providers.filter(item => `${item.label ?? ''} ${item.id ?? ''}`.toLowerCase().includes(query.toLowerCase())), [providers, query]);
  const shownModels = useMemo(() => models.filter(item => `${item.name ?? ''} ${item.id ?? ''} ${item.provider ?? ''}`.toLowerCase().includes(query.toLowerCase())), [models, query]);
  const shownMarketplace = useMemo(() => marketplace.filter(item => `${item.name ?? ''} ${item.id ?? ''} ${item.provider ?? ''}`.toLowerCase().includes(query.toLowerCase())), [marketplace, query]);
  const gpu = record(hardware.gpu);
  const activeProvider = providers.find(item => item.id === selected.id) ?? selected;
  const ollamaSetup = record(setupStatus.ollama);
  const llamaSetup = record(setupStatus.llamacpp);
  const recommendations = record(setupStatus.recommended);
  const setupRuntimes = [['Ollama', ollamaSetup], ['llama.cpp', llamaSetup]] as const;
  const localPolicies = record(policy.routes);
  const policyRoute = Array.isArray(localPolicies['local-first']) ? localPolicies['local-first'].filter(item => typeof item === 'string').join(' → ') : '';
  const hardwareFacts = [
    hardware.system && hardware.machine ? `${String(hardware.system)} · ${String(hardware.machine)}` : hardware.system ?? hardware.machine,
    hardware.cpu_count != null ? `${String(hardware.cpu_count)} CPU cores` : null,
    hardware.ram_gb != null ? `${String(hardware.ram_gb)} GB memory` : null,
    gpu.available === true && gpu.adapter ? `GPU · ${String(gpu.adapter)}` : gpu.available === false ? 'No GPU detected' : null,
  ].filter((value): value is string => typeof value === 'string' && value !== '');
  const probeDependents = records(probe, 'dependents').length;
  const probeTools = records(probe, 'tools').length;
  const providerManagedProbe = ['ollama://', 'llamacpp://'].some(prefix => text(probe.local_path, '').startsWith(prefix));

  async function discover() {
    setDiscovering(true); setError(''); setNotice('');
    try { const result = record(await api.post<unknown>('/api/models/discover', {})); await refresh(); setNotice(Array.isArray(result.discovered) ? `Discovery finished. ${result.discovered.length} model records reported.` : 'Model discovery completed.'); }
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
      setNotice(`Setup request accepted for ${modelName}.`);
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
    try { await api.post<unknown>('/api/local-ai/models/download', { model_id: item.id, allow_network: allowNetwork }); setNotice(`Download completed for ${text(item.name, 'the selected model')}.`); await refresh(); }
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

  return <PageFrame title="Models" actions={<><button className="btn" onClick={() => void refresh()} disabled={busy}>Refresh</button><button className="btn primary" onClick={() => void discover()} disabled={discovering}>{discovering ? 'Discovering…' : 'Discover models'}</button></>}>
    <div className="models-page">
      {error && <p className="notice error" role="alert">{error}</p>}{notice && <p className="notice success" role="status">{notice}</p>}
      <div className="models-primary">
      {hardwareFacts.length > 0 ? <section className="models-host" aria-label="Host capability"><span>Host</span><strong>{hardwareFacts[0]}</strong>{hardwareFacts.slice(1).map(fact => <span key={fact}>{fact}</span>)}</section> : readFailures.includes('hardware') && <section className="models-host" aria-label="Host capability"><span>Host</span><p>Host capabilities could not be read.</p></section>}
      <section className="models-section models-provider-section" aria-labelledby="models-providers-title">
        <div className="models-section-head"><div><h2 id="models-providers-title">Providers</h2></div><input className="field models-filter" aria-label="Filter providers and models" placeholder="Filter providers and models" value={query} onChange={event => setQuery(event.target.value)} /></div>
        {busy ? <div className="models-loading" role="status">Checking provider availability…</div> : <div className="models-provider-layout">
          <div className="models-provider-list" role="group" aria-label="Configured providers">{shownProviders.map(item => <button type="button" key={String(item.id)} aria-current={activeProvider.id === item.id ? 'true' : undefined} className={`models-provider-choice${activeProvider.id === item.id ? ' active' : ''}`} onClick={() => setSelected(item)}><span className="models-provider-copy"><strong>{text(item.label, text(item.id))}</strong><small>{locationState(item)}</small></span><span className={`models-status ${item.available === true ? 'available' : item.available === false ? 'unavailable' : 'unknown'}`}>{runtimeState(item)}</span></button>)}{!shownProviders.length && <p className="models-empty" role={providerReadError ? 'status' : undefined}>{providerReadError ? 'Provider availability could not be loaded.' : query ? 'No providers match this filter.' : 'No provider records are configured.'}</p>}</div>
          <div className="models-provider-detail">{activeProvider.id ? <>
            <div className="models-provider-detail-head"><div><span>{locationState(activeProvider)} provider</span><h3>{text(activeProvider.label, text(activeProvider.id))}</h3></div><span className={`models-status ${activeProvider.available === true ? 'available' : activeProvider.available === false ? 'unavailable' : 'unknown'}`}>{runtimeState(activeProvider)}</span></div>
            <dl className="models-provider-facts"><div><dt>Configured default</dt><dd>{text(activeProvider.default_model, 'Not configured')}</dd></div><div><dt>Default model readiness</dt><dd>{activeProvider.default_model_ready === true ? 'Ready' : activeProvider.default_model_ready === false ? 'Not installed' : 'Not reported'}</dd></div><div><dt>Network access</dt><dd>{activeProvider.network === true ? 'Configured for network access' : activeProvider.network === false ? 'Network disabled' : locationState(activeProvider) === 'External' ? 'External service' : 'Not reported'}</dd></div></dl>
            {Array.isArray(activeProvider.installed_models) && activeProvider.installed_models.length > 0 && <div className="models-installed"><span>Installed with this provider</span><ul>{activeProvider.installed_models.map((model, index) => <li key={`${String(model)}-${index}`}>{String(model)}</li>)}</ul></div>}
            {text(activeProvider.privacy_warning, '') && <p className="models-privacy">{text(activeProvider.privacy_warning, '')}</p>}
            {activeProvider.local === false && <p className="models-privacy">Prompts may leave this device.</p>}
          </> : <p className="models-empty">{providerReadError ? 'Retry to read provider availability.' : 'No configured providers are available.'}</p>}</div>
        </div>}
      </section>

      <section className="models-section" aria-labelledby="models-installed-title">
        <div className="models-section-head"><div><h2 id="models-installed-title">Local models</h2></div><span className="models-count">{readFailures.includes('local-models') ? 'Unavailable' : `${shownModels.length} shown`}</span></div>
        {shownModels.length ? <div className="table-wrap models-table-wrap"><table><thead><tr><th>Model</th><th>Provider</th><th>Availability</th><th>Size</th><th>Actions</th></tr></thead><tbody>{shownModels.map(item => <tr key={String(item.id)}><td><strong>{text(item.name, text(item.id))}</strong>{item.name != null && item.id != null && String(item.name) !== String(item.id) ? <small>{String(item.id)}</small> : null}</td><td>{text(item.provider, 'Local')}</td><td>{stateLabel(item.status)}</td><td>{item.size_bytes != null && item.size_bytes !== 0 ? `${(Number(item.size_bytes) / 1024 ** 3).toFixed(1)} GB` : '—'}</td><td className="toolbar"><button className="btn" onClick={() => void inspectModel(item)} disabled={probing}>{selectedModel.id === item.id && probing ? 'Checking…' : 'Probe'}</button>{item.local_path ? <button className="btn danger" disabled={!!modelBusy} onClick={() => void removeModel(item)}>Remove</button> : <button className="btn" disabled={!!modelBusy || !item.source_url} onClick={() => void downloadModel(item)}>{modelBusy === item.id ? 'Downloading…' : 'Download'}</button>}</td></tr>)}</tbody></table></div> : <p className="models-empty">{busy ? 'Reading local model inventory…' : readFailures.includes('local-models') ? 'Local model inventory could not be read.' : query ? 'No local models match this filter.' : 'No local models are registered.'}</p>}
        {selectedModel.id != null && selectedModel.id !== '' && <section className="models-probe" aria-label="Model probe result"><div><strong>{text(selectedModel.name, text(selectedModel.id))}</strong><span>{probing ? 'Checking local model…' : stateLabel(probe.status)}</span></div>{!probing && <dl><div><dt>File status</dt><dd>{probe.exists === true ? 'Present on this device' : providerManagedProbe ? 'Managed by local provider' : probe.exists === false ? 'Not present at the registered path' : 'Not reported'}</dd></div><div><dt>Size</dt><dd>{probe.size_bytes != null ? `${(Number(probe.size_bytes) / 1024 ** 3).toFixed(2)} GB` : 'Not reported'}</dd></div><div><dt>Checksum</dt><dd>{probe.checksum_valid === true ? 'Verified' : probe.checksum_valid === false ? 'Mismatch' : 'Not available'}</dd></div><div><dt>Dependencies</dt><dd>{probeDependents}</dd></div><div><dt>Assigned tools</dt><dd>{probeTools}</dd></div></dl>}</section>}
        <details className="models-routing"><summary>Model routing settings</summary>{readFailures.includes('routing') ? <p className="models-empty">Routing policy could not be read.</p> : <div className="models-routing-content"><div><span>Routing policy</span><strong>{text(policy.routing_policy, 'Not reported')}</strong></div><div><span>Fallback policy</span><strong>{text(policy.fallback_policy, 'Not reported')}</strong></div>{policyRoute && <div className="models-route-list"><span>Local-first order</span><strong>{policyRoute}</strong></div>}</div>}</details>
      </section>

      <details className="models-secondary" open={jobs.some(job => !terminal(job))}>
        <summary>Setup activity <span>{jobs.length}</span></summary>
        <div className="models-job-list">{jobs.map(job => { const progress = jobProgress(job); return <article className="models-job" key={String(job.id)}><div><strong>{text(job.model_name, 'Model setup')}</strong><p>{text(job.current_step, text(job.kind))} · {stateLabel(job.status)}</p>{progress != null && <progress max="100" value={progress} aria-label={`Setup progress for ${text(job.model_name, 'model')}`} />}{job.error != null && job.error !== '' && <p className="notice error">{String(job.error)}</p>}</div><div><small>{progress != null ? `${progress}%` : 'Progress not reported'}</small>{!terminal(job) && <button className="btn danger" onClick={() => void cancelJob(String(job.id))}>Cancel job</button>}</div></article>; })}{!jobs.length && <p className="models-empty">{readFailures.includes('jobs') ? 'Setup activity could not be read.' : 'No setup jobs have been recorded for this project.'}</p>}</div>
      </details>

      <details className="models-secondary">
        <summary>Setup catalog and model registry</summary>
        <div className="models-management-grid">
          <section><div className="models-section-head"><div><h2>Setup catalog</h2></div></div><label className="models-network-toggle">Permit network download for the next setup job<input type="checkbox" checked={allowNetwork} onChange={event => setAllowNetwork(event.target.checked)} /></label><p className="models-safety-note">Network downloads are disabled by default. Starting a job always requires confirmation; the backend records the chosen policy.</p><div className="models-catalog">{shownMarketplace.map(item => { const metadata = record(item.metadata); return <article className="models-catalog-item" key={String(item.id)}><div><strong>{text(item.name, text(item.id))}</strong><small>{text(metadata.model, 'No direct Ollama identifier')}</small><small>{metadata.ram_gb != null ? `${String(metadata.ram_gb)} GB RAM` : 'Memory requirement not reported'} · {metadata.quantization != null ? String(metadata.quantization) : 'Quantization not reported'}</small></div><button className="btn" disabled={startingJob || !metadata.model} onClick={() => void setup(item)}>{startingJob ? 'Queuing…' : 'Queue setup'}</button></article>; })}{!shownMarketplace.length && <p className="models-empty">{readFailures.includes('catalog') ? 'Setup catalog could not be read.' : query ? 'No catalog models match this filter.' : 'No models are available in the setup catalog.'}</p>}</div></section>
          <section><div className="models-section-head"><div><h2>Register a model</h2><p>Add a local file or optional source URL to the registry.</p></div></div><form className="models-register-form" onSubmit={registerModel}><label>Model name<input className="field" value={modelName} onChange={event => setModelName(event.target.value)} required /></label><label>Local file path<input className="field" value={modelPath} onChange={event => setModelPath(event.target.value)} placeholder="Optional path to an existing model file" /></label><label>Download source URL<input className="field" type="url" value={modelSource} onChange={event => setModelSource(event.target.value)} placeholder="Optional https://…" /></label><small>Registration saves metadata only. It does not download a file or contact the URL.</small><button className="btn primary" disabled={modelBusy === 'register'}>{modelBusy === 'register' ? 'Registering…' : 'Register model'}</button></form></section>
          <section className="models-setup-summary"><div className="models-section-head"><div><h2>Local setup tools</h2></div></div>{readFailures.includes('setup') ? <p className="models-empty">Local setup availability could not be read.</p> : <><div className="models-runtime-grid">{setupRuntimes.map(([label, runtime]) => <article key={label}><div><strong>{label}</strong><span className={`models-status ${runtime.available === true ? 'available' : runtime.available === false ? 'unavailable' : 'unknown'}`}>{runtimeAvailability(runtime)}</span></div><p>{runtimeDescription(runtime)}</p><small>{runtimeModelReadiness(runtime)}</small></article>)}</div>{Object.keys(recommendations).length > 0 && <div className="models-recommendations"><span>Recommended defaults</span><p>{['chat', 'code', 'research', 'speed', 'vision', 'low_resource_code'].filter(key => recommendations[key]).map(key => <span key={key}><b>{key.replaceAll('_', ' ')}</b>{String(recommendations[key])}</span>)}</p></div>}</>}</section>
          <section className="models-adapters"><div className="models-section-head"><div><h2>Acceleration adapters</h2></div><span className="models-count">{readFailures.includes('adapters') ? 'Unavailable' : adapters.length}</span></div><div className="models-adapter-list">{adapters.map((adapter, index) => <article className="models-adapter" key={text(adapter.id, String(index))}><div><strong>{text(adapter.name, text(adapter.adapter_id, 'Adapter'))}</strong><span>{text(adapter.model_id, 'Model not reported')}</span></div><small>{stateLabel(adapter.status)}</small></article>)}{!adapters.length && <p className="models-empty">{readFailures.includes('adapters') ? 'Acceleration adapters could not be read.' : 'No acceleration adapters are reported.'}</p>}</div></section>
        </div>
      </details>
      </div>
      <aside className="models-inspector" aria-label="Model guidance and runtime readiness">
        <section className="models-inspector-card">
          <div className="models-inspector-heading"><span className="models-inspector-icon" aria-hidden="true"><Sparkles size={17}/></span><div><h2>Recommended defaults</h2><p>Based on the runtime configuration.</p></div></div>
          {Object.keys(recommendations).length ? <dl className="models-recommendation-list">{['chat', 'code', 'research', 'speed', 'vision', 'low_resource_code'].filter(key => recommendations[key]).map(key => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{String(recommendations[key])}</dd></div>)}</dl> : <p className="models-inspector-empty">No role recommendations were returned by the runtime.</p>}
        </section>
        <section className="models-inspector-card">
          <div className="models-inspector-heading"><div><h2>Runtime readiness</h2><p>Reported by configured setup tools.</p></div></div>
          {readFailures.includes('setup') ? <p className="models-inspector-empty">Setup availability could not be read.</p> : setupRuntimes.length ? <div className="models-runtime-rail">{setupRuntimes.map(([name, runtime]) => <article key={name}><div><strong>{name}</strong><span className={`models-status ${runtime.available === true ? 'available' : runtime.available === false ? 'unavailable' : 'unknown'}`}>{runtimeAvailability(runtime)}</span></div><small>{runtimeModelReadiness(runtime)}</small></article>)}</div> : <p className="models-inspector-empty">No setup runtimes were reported.</p>}
        </section>
        <section className="models-inspector-note"><strong>Model availability is live</strong><p>Selection, setup, downloads, and removal follow the status and permissions reported by this backend.</p></section>
      </aside>
    </div>
  </PageFrame>;
}
