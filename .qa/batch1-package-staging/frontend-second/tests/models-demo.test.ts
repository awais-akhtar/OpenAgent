import { afterEach, describe, expect, it, vi } from 'vitest';
import { modelDemoAction } from '../src/demo/models';
import { demoProvider } from '../src/demo/provider';

const hash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const fixtures = { sourceSha256: hash, data: { approvedShellFixtures: { models: [], localModels: [], modelJobs: [] }, advanced: {} } };

afterEach(() => { localStorage.removeItem('oa-second-isolated-demo-v1'); vi.restoreAllMocks(); });

describe('Models local demo adapter', () => {
  it('registers metadata and creates setup previews without contacting model services', async () => {
    const calls: string[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async input => {
      const url = String(input);
      calls.push(url);
      if (url.endsWith('/demo/reference-fixtures.json')) return new Response(JSON.stringify(fixtures), { status: 200 });
      if (url.endsWith('/demo/approved-shell-fixtures.json')) return new Response(JSON.stringify({ sourceSha256: hash, data: {} }), { status: 200 });
      throw new Error(`Unexpected request in model demo: ${url}`);
    });
    await demoProvider.reset();

    const { model } = await modelDemoAction('/api/local-ai/models/register', { name: 'Fixture model' });
    expect(model).toMatchObject({ name: 'Fixture model', status: 'Registered · demo metadata only', available: false });
    expect(await demoProvider.get('shell')).toMatchObject({ localModels: [expect.objectContaining({ name: 'Fixture model' })] });

    const { job } = await modelDemoAction('/api/local-ai/models/setup/job', { model: 'fixture/model', allow_network: true });
    expect(job).toMatchObject({ status: 'preview only', progress: null, allow_network: false, message: 'No files were downloaded or installed.' });
    expect((await demoProvider.get('shell')).modelJobs).toEqual([job]);
    expect(calls.every(url => url.includes('/demo/'))).toBe(true);
  });

  it('reports downloads and benchmarks as unavailable without making a false success', async () => {
    await expect(modelDemoAction('/api/local-ai/models/download', { model_id: 'demo-model' })).rejects.toThrow(/unavailable in local demo/i);
    const result = await modelDemoAction('/api/local-ai/benchmark', { model_id: 'demo-model' });
    expect(result).toMatchObject({ benchmark: { status: 'not measured' }, demo: true });
  });
});
