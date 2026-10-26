import { test, expect } from '@playwright/test';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { resolve } from 'node:path';

const root = resolve(process.cwd(), '..');
let backend: ChildProcessWithoutNullStreams;
let origin = '';
const serverSource = `
import os, sys, tempfile, threading
os.environ['OPENAGENT_AUTH_ENFORCED'] = '0'
from openagent.storage import OpenAgentStore
from openagent.routes.runtime import build_http_server
with tempfile.TemporaryDirectory(prefix='oa-preset-browser-') as tmp:
    server = build_http_server('127.0.0.1', 0, OpenAgentStore(tmp))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print('READY:http://127.0.0.1:' + str(server.server_address[1]), flush=True)
    sys.stdin.readline()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
`;

test.describe.configure({ mode: 'serial' });
test.beforeAll(async () => {
  backend = spawn(resolve(root, '.venv/Scripts/python.exe'), ['-u', '-c', serverSource], { cwd: root, windowsHide: true });
  origin = await new Promise<string>((resolveReady, reject) => {
    let output = ''; let errors = '';
    const timer = setTimeout(() => reject(new Error(`Python startup timeout: ${errors}`)), 30000);
    backend.stderr.on('data', data => { errors += String(data); });
    backend.stdout.on('data', data => {
      output += String(data);
      const match = output.match(/READY:(http:\/\/127\.0\.0\.1:\d+)/);
      if (match) { clearTimeout(timer); resolveReady(match[1]); }
    });
    backend.once('error', error => { clearTimeout(timer); reject(error); });
    backend.once('exit', code => { clearTimeout(timer); reject(new Error(`Python exited ${code}: ${errors}`)); });
  });
});
test.afterAll(async () => {
  if (!backend || backend.exitCode !== null) return;
  backend.stdin.end('quit\n');
  await new Promise<void>((resolveExit, reject) => {
    const timer = setTimeout(() => { backend.kill(); reject(new Error('Python cleanup timed out')); }, 10000);
    backend.once('exit', code => { clearTimeout(timer); code === 0 ? resolveExit() : reject(new Error(`Python cleanup exited ${code}`)); });
  });
});
for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark']) {
    test(`real Python preset persistence ${viewport.width} ${theme}`, async ({ page }) => {
      await page.setViewportSize(viewport);
      await page.addInitScript(themeName => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', themeName);
      }, theme);
      // Proxy unchanged requests to a real disposable Python service, not response fixtures.
      await page.route('**/api/**', async route => {
        const url = new URL(route.request().url());
        const response = await route.fetch({ url: origin + url.pathname + url.search });
        await route.fulfill({ response });
      });
      const errors: string[] = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('/?view=personalos');
      await page.getByRole('tab', { name: 'Presets', exact: true }).click();
      const label = `Browser ${viewport.width} ${theme}`;
      await page.getByRole('textbox', { name: 'Preset name' }).fill(label);
      await page.getByRole('button', { name: 'Add', exact: true }).click();
      await expect(page.locator('.p-list-row strong').filter({ hasText: label })).toBeVisible();
      await page.getByRole('button', { name: 'Edit', exact: true }).click();
      await page.getByRole('textbox', { name: 'Preset name' }).fill(label + ' updated');
      await page.getByRole('button', { name: 'Save changes', exact: true }).click();
      await expect(page.locator('.p-list-row strong').filter({ hasText: label + ' updated' })).toBeVisible();
      await page.reload();
      await page.getByRole('tab', { name: 'Presets', exact: true }).click();
      await expect(page.locator('.p-list-row strong').filter({ hasText: label + ' updated' })).toBeVisible();
      page.once('dialog', dialog => dialog.accept());
      await page.getByRole('button', { name: 'Delete', exact: true }).click();
      await expect(page.getByText('No presets', { exact: true })).toBeVisible();
      expect(errors).toEqual([]);
    });
    test(`real Python Vault request review ${viewport.width} ${theme}`, async ({ page, request }) => {
      const stored = await request.post(origin + '/api/vault/store', { data: { label: `Synthetic ${viewport.width} ${theme}`, secret: 'noncredential-example', passphrase: 'local-test-passphrase', project: 'default' } });
      expect(stored.status()).toBe(201);
      await page.setViewportSize(viewport);
      await page.addInitScript(themeName => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', themeName);
      }, theme);
      await page.route('**/api/**', async route => {
        const url = new URL(route.request().url());
        const response = await route.fetch({ url: origin + url.pathname + url.search });
        await route.fulfill({ response });
      });
      await page.goto('/?view=vault');
      await page.getByRole('tab', { name: 'Access', exact: true }).click();
      await page.getByRole('button', { name: 'Add', exact: true }).click();
      const dialog = page.getByRole('dialog');
      await dialog.getByRole('textbox', { name: 'Reason', exact: true }).fill('Synthetic browser review');
      await dialog.getByRole('button', { name: 'Create request', exact: true }).click();
      await expect(dialog.getByRole('alert')).toHaveText('Select at least one entry.');
      await dialog.getByRole('checkbox', { name: `Synthetic ${viewport.width} ${theme}`, exact: true }).check();
      await dialog.getByRole('button', { name: 'Create request', exact: true }).click();
      await expect(dialog).not.toBeVisible();
      const row = page.locator('.p-list-row').filter({ hasText: 'Synthetic browser review' }).last();
      await row.getByRole('button', { name: 'Details', exact: true }).click();
      await dialog.getByRole('button', { name: 'Approve', exact: true }).click();
      await expect(dialog).not.toBeVisible();
      await expect(row).toContainText('approved');
      await page.reload();
      await page.getByRole('tab', { name: 'Access', exact: true }).click();
      await expect(page.locator('.p-list-row').filter({ hasText: 'Synthetic browser review' }).last()).toContainText('approved');
    });
  }
}
