import { expect, test } from '@playwright/test';

const desktops = [
  { width: 1440, height: 900, name: '1440×900' },
  { width: 1280, height: 800, name: '1280×800' },
];
const themes = ['light', 'dark'] as const;

for (const viewport of desktops) {
  for (const theme of themes) {
    test(`Home attachments hand off to Live Chat at ${viewport.name} ${theme}`, async ({ page }) => {
      await page.setViewportSize(viewport);
      await page.addInitScript(({ themeName }) => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', themeName);
      }, { themeName: theme });

      const streamRequests: Array<Record<string, unknown>> = [];
      await page.route(/^https?:\/\/[^/]+\/api\/.*/, async route => {
        const request = route.request();
        const url = new URL(request.url());
        if (request.method() === 'POST' && url.pathname === '/api/media/ingest') {
          return route.fulfill({ status: 201, contentType: 'application/json', body: JSON.stringify({
            asset: { id: 'home-upload-1', title: 'brief.txt', media_type: 'document', mime_type: 'text/plain', size_bytes: 6 },
          }) });
        }
        if (request.method() === 'POST' && url.pathname === '/api/chat/stream') {
          streamRequests.push(request.postDataJSON() as Record<string, unknown>);
          return route.fulfill({ status: 200, contentType: 'text/event-stream', body: 'data: {"type":"chunk","content":"Attached context received."}\n\ndata: {"type":"done"}\n\n' });
        }
        if (request.method() === 'GET' && url.pathname === '/api/models') {
          return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ providers: [{ id: 'auto', label: 'Local', local: true }], models: [] }) });
        }
        return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
      });

      await page.goto('/?view=home');
      await expect(page.getByRole('textbox', { name: 'Ask OpenAgent' })).toBeVisible({ timeout: 5_000 });
      const fileInput = page.getByLabel('Choose files for chat');
      await fileInput.setInputFiles({ name: 'brief.txt', mimeType: 'text/plain', buffer: Buffer.from('hello\n') });
      await expect(page.getByTestId('home-chat-attachment')).toContainText('brief.txt');
      await page.getByRole('textbox', { name: 'Ask OpenAgent' }).fill('Summarise this file');
      await page.getByRole('button', { name: 'Open prompt in chat' }).click();
      await expect(page).toHaveURL(/view=chat/);
      await expect(page.getByTestId('chat-attachment')).toContainText('brief.txt');
      const composer = page.getByRole('textbox', { name: 'Message OpenAgent' });
      await expect(composer).toHaveValue('Summarise this file');
      const streamRequest = page.waitForRequest(request => {
        const url = new URL(request.url());
        return request.method() === 'POST' && url.pathname === '/api/chat/stream';
      });
      await page.getByRole('button', { name: 'Send message' }).click();
      await streamRequest;
      await expect(page.getByText('Attached context received.')).toBeVisible();
      expect(streamRequests).toHaveLength(1);
      expect(streamRequests[0].attachments).toEqual([{ id: 'home-upload-1', name: 'brief.txt', media_type: 'document', mime_type: 'text/plain' }]);
    });
  }
}

test('Demo Home attachment control stays offline and clearly unavailable', async ({ page }) => {
  const apiCalls: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiCalls.push(request.url()); });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.goto('/?view=home');
  await expect(page.getByRole('button', { name: 'Attachments require Live mode' })).toBeDisabled();
  expect(apiCalls).toEqual([]);
});
