import { expect, test } from '@playwright/test';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { resolve } from 'node:path';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const root = resolve(process.cwd(), '..');
let backend: ChildProcessWithoutNullStreams;
let fixture: { origin: string; modelDir: string; datasetId: string };
test.describe.configure({ mode: 'serial' });
// This fixture imports PyTorch before it can serve, which takes most of 30
// seconds on its own and overruns that budget when the full suite runs in
// parallel and the import competes for CPU. The allowance covers the import
// only: the server still has to print READY, and a server that fails to start
// still fails the run.
const SERVER_STARTUP_TIMEOUT_MS = 120_000;
test.beforeAll(async () => {
  test.setTimeout(SERVER_STARTUP_TIMEOUT_MS + 30_000);
  const python = process.env.OPENAGENT_MODEL_TEST_PYTHON ?? resolve(root, '.accelerator-venv/Scripts/python.exe');
  backend = spawn(python, ['-u', 'tests/support/tiny_finetune_server.py'], { cwd: root, windowsHide: true });
  fixture = await new Promise((ready, reject) => {
    let output = ''; let errors = '';
    const timer = setTimeout(() => reject(new Error(`Tiny model server startup failed: ${errors}`)), SERVER_STARTUP_TIMEOUT_MS);
    backend.stderr.on('data', chunk => { errors += String(chunk); });
    backend.stdout.on('data', chunk => {
      output += String(chunk);
      const line = output.split('\n').find(row => row.startsWith('READY:'));
      if (line) { clearTimeout(timer); ready(JSON.parse(line.slice(6))); }
    });
    backend.once('error', error => { clearTimeout(timer); reject(error); });
    backend.once('exit', code => { clearTimeout(timer); reject(new Error(`Tiny model server exited ${code}: ${errors}`)); });
  });
});
test.afterEach(async ({ page }) => { await page.unrouteAll({ behavior: 'wait' }); });
test.afterAll(async () => {
  if (!backend || backend.exitCode !== null) return;
  backend.stdin.end('quit\n');
  await new Promise<void>((done, reject) => {
    const timer = setTimeout(() => { backend.kill(); reject(new Error('Tiny model server cleanup timed out')); }, 10000);
    backend.once('exit', code => { clearTimeout(timer); code === 0 ? done() : reject(new Error(`Tiny model cleanup exited ${code}`)); });
  });
});

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark']) {
    test(`tiny actual fine-tune checkpoint ${viewport.width} ${theme}`, async ({ page, request }) => {
      const created = await request.post(fixture.origin + '/api/finetune/jobs', { data: {
        project: 'default', dataset_id: fixture.datasetId, base_model: fixture.modelDir, method: 'lora',
        config: { epochs: 1, rank: 2, alpha: 4, lr: 0.001, accumulation_steps: 1, max_seq_len: 8 },
      } });
      expect(created.status()).toBe(201);
      const job = (await created.json()).job;
      const before = createHash('sha256').update(await readFile(resolve(fixture.modelDir, 'model.safetensors'))).digest('hex');
      await page.setViewportSize(viewport);
      await page.addInitScript(selectedTheme => {
        localStorage.setItem('oa-second-data-mode', 'live'); localStorage.setItem('oa-second-theme', selectedTheme);
      }, theme);
      await page.route(/^https?:\/\/[^/]+\/api\/.*/, async route => {
        const url = new URL(route.request().url());
        const response = await route.fetch({ url: fixture.origin + url.pathname + url.search });
        await route.fulfill({ response });
      });
      await page.goto('/?view=finetune');
      await page.getByRole('tab', { name: 'Jobs', exact: true }).click();
      const panel = page.locator('section.p-card').filter({ has: page.getByRole('heading', { name: 'CPU streaming fine-tune', exact: true }) });
      await panel.getByRole('combobox', { name: 'Job', exact: true }).selectOption(job.id);
      await panel.getByRole('textbox', { name: 'Model directory', exact: true }).fill(fixture.modelDir);
      const started = page.waitForResponse(response => new URL(response.url()).pathname.endsWith(`/${job.id}/run-cpu`));
      page.once('dialog', dialog => dialog.accept());
      await panel.getByRole('button', { name: 'Start CPU fine-tune', exact: true }).click();
      expect((await started).status()).toBe(200);
      await expect(panel.getByRole('status')).toHaveText('Training complete. Adapter registered in Results tab.');
      await expect(panel.getByRole('progressbar')).toHaveAttribute('value', '100');
      const manifestResponse = await request.get(fixture.origin + `/api/finetune/checkpoints?project=default&job_id=${job.id}`);
      const manifest = await manifestResponse.json();
      expect(manifest.checkpoints).toHaveLength(1);
      const bytes = await readFile(manifest.checkpoints[0].adapter_path);
      const adapter = JSON.parse(bytes.toString('utf8'));
      expect(Object.keys(adapter.adapters)).toHaveLength(4);
      expect(Object.values(adapter.adapters).some(value => (value as { B: number[][] }).B.some(row => row.some(item => item !== 0)))).toBe(true);
      expect(createHash('sha256').update(await readFile(resolve(fixture.modelDir, 'model.safetensors'))).digest('hex')).toBe(before);
      await page.reload();
      await page.getByRole('tab', { name: 'Jobs', exact: true }).click();
      await panel.getByRole('combobox', { name: 'Job', exact: true }).selectOption(job.id);
      await panel.getByRole('button', { name: 'Monitor selected job', exact: true }).click();
      await expect(panel.getByRole('status')).toHaveText('Training complete. Adapter registered in Results tab.');
      expect(await readFile(manifest.checkpoints[0].adapter_path)).toEqual(bytes);
    });
  }
}
