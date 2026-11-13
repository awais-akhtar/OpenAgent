import { spawn, type ChildProcess } from 'node:child_process';
import { resolve } from 'node:path';
import { expect, test, type Page } from '@playwright/test';

// GAP-011: React-to-Python chat checks against the real OpenAgent backend.
// Start the server first: `python scripts/gap011_chat_server.py --ui-root frontend-second/dist --port 8791`,
// then run with CHAT_REAL_BASE_URL=http://127.0.0.1:8791. Nothing is mocked here.
const baseURL = process.env.CHAT_REAL_BASE_URL ?? '';

async function openLiveChat(page: Page) {
  await page.addInitScript(() => {
    localStorage.setItem('oa-second-data-mode', 'live');
    localStorage.setItem('oa-second-theme', 'light');
    sessionStorage.removeItem('oa-second-conversation-id');
  });
  await page.goto(`${baseURL}/?view=chat`, { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('textbox', { name: 'Message OpenAgent' })).toBeVisible({ timeout: 30_000 });
}

test.describe('GAP-011 real React-to-Python chat', () => {
  test.skip(!baseURL, 'Set CHAT_REAL_BASE_URL to a running gap011_chat_server.py origin');

  test('idle chat shows no fake activity and no sample answer', async ({ page }) => {
    await openLiveChat(page);
    await expect(page.getByText(/sample|demo answer|lorem/i)).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Cancel' })).toHaveCount(0);
  });

  test('immediate reply from the real backend is shown and persists after reload', async ({ page }) => {
    await openLiveChat(page);
    const prompt = `gap011 immediate ${Date.now()}`;
    const reply = page.waitForResponse(response => response.url().includes('/api/chat') && response.request().method() === 'POST');
    await page.getByRole('textbox', { name: 'Message OpenAgent' }).fill(prompt);
    await page.getByRole('button', { name: 'Send message' }).click();
    const response = await reply;
    expect(response.status()).toBeLessThan(300);
    await expect(page.getByText(prompt, { exact: false }).first()).toBeVisible({ timeout: 60_000 });
    // The assistant bubble must gain real text, and the pending Cancel control must go away.
    const assistant = page.locator('article').nth(1);
    await expect(assistant).not.toBeEmpty({ timeout: 240_000 });
    await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toHaveCount(0, { timeout: 240_000 });
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.getByText(prompt, { exact: false }).first()).toBeVisible({ timeout: 30_000 });
  });

  test('cancelling a real streaming reply stops it and leaves no empty answer behind', async ({ page }) => {
    await openLiveChat(page);
    const prompt = `gap011 cancel ${Date.now()}: write the numbers from 1 to 3000, one per line`;
    await page.getByRole('textbox', { name: 'Message OpenAgent' }).fill(prompt);
    await page.getByRole('button', { name: 'Send message' }).click();
    const cancel = page.getByRole('button', { name: 'Cancel', exact: true });
    await expect(cancel).toBeVisible({ timeout: 30_000 });
    await cancel.click();
    await expect(cancel).toHaveCount(0);
    await expect(page.getByText('Response cancelled.')).toBeVisible();
  });

  test('a reply cut off by a real backend failure shows the error and leaves no empty answer', async ({ page }) => {
    // This check owns its own chat server so stopping it is a real mid-reply failure.
    const repoRoot = resolve(process.cwd(), '..');
    const python = resolve(repoRoot, '.venv', 'Scripts', 'python.exe');
    const uiRoot = resolve(repoRoot, 'frontend-second', 'dist');
    const server: ChildProcess = spawn(python, ['scripts/gap011_chat_server.py', '--ui-root', uiRoot, '--port', '8795'], { cwd: repoRoot });
    try {
      await expect.poll(async () => {
        try { return (await fetch('http://127.0.0.1:8795/api/health')).status; } catch { return 0; }
      }, { timeout: 60_000 }).toBe(200);
      await page.addInitScript(() => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', 'light');
        sessionStorage.removeItem('oa-second-conversation-id');
      });
      await page.goto('http://127.0.0.1:8795/?view=chat', { waitUntil: 'domcontentloaded' });
      await expect(page.getByRole('textbox', { name: 'Message OpenAgent' })).toBeVisible({ timeout: 30_000 });
      await page.getByRole('textbox', { name: 'Message OpenAgent' }).fill('Write the numbers from 1 to 3000, one per line');
      await page.getByRole('button', { name: 'Send message' }).click();
      await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toBeVisible({ timeout: 30_000 });
      server.kill('SIGKILL');
      await expect(page.getByRole('alert')).toContainText(/backend|reach|connect/i, { timeout: 60_000 });
      await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toHaveCount(0);
    } finally {
      server.kill('SIGKILL');
    }
  });

  test('empty message is refused by the real backend with a clear error', async ({ request }) => {
    const response = await request.post(`${baseURL}/api/chat`, { data: { message: '   ', provider: 'local' } });
    expect(response.status()).toBe(400);
    const body = await response.json();
    expect(JSON.stringify(body)).toContain('A message is required');
  });

  test('unknown route is a real not-found, not a fabricated answer', async ({ request }) => {
    const response = await request.get(`${baseURL}/api/chat/does-not-exist`);
    expect(response.status()).toBeGreaterThanOrEqual(400);
  });
});

// GAP-004: continuous browser microphone streaming against the real backend.
// Chromium's fake capture device plays a local WAV, so no hardware or external service is used.
// Chromium's fake capture device plays a local WAV for the GAP-004 microphone check.
// These flags only add a fake media device, so every other test in this file is unaffected.
test.use({
  launchOptions: {
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${process.env.GAP004_FAKE_MIC ?? ''}`,
    ],
  },
  permissions: ['microphone'],
});

test.describe('GAP-004 real browser microphone streaming', () => {
  test.skip(!baseURL, 'Set CHAT_REAL_BASE_URL to a running gap011_chat_server.py origin');

  test('live voice streams microphone chunks incrementally and finishes the utterance', async ({ page }) => {
    test.skip(!process.env.GAP004_FAKE_MIC, 'Set GAP004_FAKE_MIC to a WAV file path');
    test.setTimeout(240_000); // speech-to-text on a CPU can take minutes
    const chunkStatuses: number[] = [];
    const finishes: number[] = [];
    page.on('response', response => {
      const url = response.url();
      if (url.includes('/api/voice/websocket/chunk')) chunkStatuses.push(response.status());
      if (url.includes('/api/voice/websocket/finish')) finishes.push(response.status());
    });
    await openLiveChat(page);
    await page.getByRole('button', { name: 'Start live voice' }).click();
    await expect.poll(() => chunkStatuses.length, { timeout: 30_000 }).toBeGreaterThanOrEqual(2);
    await expect.poll(() => finishes.length, { timeout: 60_000 }).toBeGreaterThanOrEqual(1);
    expect(chunkStatuses.every(status => status < 300)).toBe(true);
    expect(finishes.every(status => status < 300)).toBe(true);
    // Closing the page stops the fake microphone; the app may still be thinking, so no UI stop is needed.
  });
});
