import { expect, test } from '@playwright/test';

test('Chat voice mode transcribes, sends the real text, and speaks the real reply', async ({ page }) => {
  page.on('pageerror', error => process.stderr.write(`PAGEERROR ${error.message}\n`));
  page.on('console', message => { if (message.type() === 'error') process.stderr.write(`CONSOLE ${message.text()}\n`); });
  page.on('response', response => { if (response.status() >= 400) process.stderr.write(`HTTP ${response.status()} ${response.url()}\n`); });
  const requests: Array<{ path: string; body: Record<string, unknown> }> = [];
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: async () => ({ getTracks: () => [{ stop() {} }] }) } });
    class TestRecorder {
      mimeType = 'audio/webm';
      ondataavailable?: (event: { data: Blob }) => void;
      onstop?: () => void;
      start() {}
      stop() { this.ondataavailable?.({ data: new Blob(['local test audio'], { type: this.mimeType }) }); this.onstop?.(); }
    }
    Object.defineProperty(window, 'MediaRecorder', { configurable: true, value: TestRecorder });
    Object.defineProperty(window, 'Audio', { configurable: true, value: class { play() { return Promise.resolve(); } } });
  });
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname;
    const body = route.request().postDataJSON?.() as Record<string, unknown> | undefined;
    if (body) requests.push({ path, body });
    const response = path === '/api/models' ? { providers: [{ id: 'auto', available: true, local: true }] }
      : path.startsWith('/api/conversations') ? { conversations: [] }
        : path === '/api/voice/sessions' ? { session: { id: 'voice-chat-1' } }
          : path === '/api/voice/stt' ? { transcription: { transcript: 'What is two plus two?' } }
            : path === '/api/voice/tts' ? { script: { audio_base64: btoa('audio') } }
              : { error: `Unexpected ${path}` };
    if (path === '/api/chat/stream') {
      await route.fulfill({ status: 200, contentType: 'text/event-stream', body: 'data: {"type":"conversation","conversation":{"id":"spoken-chat"}}\n\ndata: {"type":"chunk","content":"Four."}\n\ndata: {"type":"done"}\n\n' });
      return;
    }
    await route.fulfill({ status: 'error' in response ? 404 : 200, contentType: 'application/json', body: JSON.stringify(response) });
  });
  await page.goto('/?view=chat');
  await expect(page.locator('.split-layout')).toBeVisible({ timeout: 8000 });
  await page.getByRole('combobox', { name: 'Voice mode' }).selectOption('answer');
  await page.getByRole('button', { name: 'Record a spoken message' }).click();
  await page.getByRole('button', { name: 'Stop recording and transcribe' }).click();
  await expect(page.locator('.chat-thread .chat-message')).toContainText(['What is two plus two?', 'Four.']);
  await expect.poll(() => requests.some(request => request.path === '/api/voice/tts')).toBe(true);
  expect(requests.find(request => request.path === '/api/chat/stream')?.body).toMatchObject({ message: 'What is two plus two?' });
  expect(requests.find(request => request.path === '/api/voice/tts')?.body).toMatchObject({ text: 'Four.' });
});
