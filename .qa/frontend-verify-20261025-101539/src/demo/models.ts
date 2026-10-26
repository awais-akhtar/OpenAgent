import { demoProvider } from './provider';

/** Local metadata/job simulation for Models. No provider, download, benchmark, or runtime is called. */
export async function modelDemoAction(path: string, body: unknown): Promise<Record<string, unknown>> {
  const values = body && typeof body === 'object' ? body as Record<string, unknown> : {};
  if (path === '/api/models/discover') return { discovered: true, message: 'Demo catalogue checked; no provider was contacted.', demo: true };
  if (path.endsWith('/probe')) return { id: decodeURIComponent(path.split('/').at(-2) ?? ''), status: 'demo only', exists: false, checksum_valid: null, demo: true };
  if (path === '/api/local-ai/benchmark') return { benchmark: { status: 'not measured', message: 'Benchmarks are disabled in the local demo.' }, demo: true };
  if (path.includes('/local-ai/models/register')) {
    const name = String(values.name ?? '').trim();
    if (!name) throw new Error('Enter a model name.');
    const id = `demo-model-${Date.now().toString(36)}`;
    const model = { id, name, label: name, provider: 'Local demo metadata', local: true, status: 'Registered · demo metadata only', available: false, source_url: '', local_path: '' };
    await demoProvider.updateShellCollection<Record<string, unknown>>('localModels', rows => [model, ...rows]);
    return { model, demo: true };
  }
  if (path.includes('/models/delete')) {
    const id = String(values.model_id ?? '');
    await demoProvider.updateShellCollection<Record<string, unknown>>('localModels', rows => rows.filter(row => String(row.id ?? '') !== id));
    return { removed: true, deleted_file: false, demo: true };
  }
  if (path.includes('/models/setup/job')) {
    const model = String(values.model ?? '');
    if (!model) throw new Error('Choose a model from the catalogue.');
    const job = { id: `demo-setup-${Date.now().toString(36)}`, model_id: model, model_name: model, status: 'preview only', progress: null, allow_network: false, message: 'No files were downloaded or installed.' };
    await demoProvider.updateShellCollection<Record<string, unknown>>('modelJobs', rows => [job, ...rows]);
    return { job, demo: true };
  }
  if (path.includes('/jobs/cancel')) {
    const id = String(values.job_id ?? '');
    await demoProvider.updateShellCollection<Record<string, unknown>>('modelJobs', rows => rows.map(row => String(row.id ?? '') === id ? { ...row, status: 'cancelled · demo' } : row));
    return { cancelled: true, demo: true };
  }
  throw new Error('This Models action is unavailable in local demo mode. No backend request was made.');
}
