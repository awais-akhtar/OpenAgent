import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Files removes an uploaded stored copy after confirmation (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      await page.setViewportSize(viewport);
      await page.addInitScript(({ selectedTheme }) => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', selectedTheme);
        localStorage.setItem('oa-second-api-token', 'e2e-media-write-token');
      }, { selectedTheme: theme });

      const assets: Array<Record<string, unknown>> = [];
      const requests: Array<{ method: string; url: string }> = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/api/')) requests.push({ method: request.method(), url: request.url() });
      });
      await page.route('**/api/media/assets**', async route => {
        const request = route.request();
        const url = new URL(request.url());
        if (request.method() === 'GET') { await route.fulfill({ json: { assets } }); return; }
        if (request.method() === 'DELETE') {
          expect(url.searchParams.get('project')).toBe('default');
          const index = assets.findIndex(asset => String(asset.id) === decodeURIComponent(url.pathname.split('/').at(-1) ?? ''));
          if (index < 0) { await route.fulfill({ status: 404, json: { error: 'Media asset not found.' } }); return; }
          const [asset] = assets.splice(index, 1);
          await route.fulfill({ json: { deleted: true, ...asset, stored_bytes_removed: true, cleanup_pending: false } });
          return;
        }
        await route.fulfill({ status: 405, json: { error: 'Unsupported media request.' } });
      });
      await page.route('**/api/media/ingest', async route => {
        const body = route.request().postDataJSON() as Record<string, unknown>;
        const asset = { id: 'uploaded-media-1', project: body.project, title: body.filename, source: 'upload',
          media_type: 'document', mime_type: 'text/plain', size_bytes: 22, created_at: '2026-10-24T12:00:00Z' };
        assets.push(asset);
        await route.fulfill({ status: 201, json: { asset } });
      });

      await page.goto('/?view=files');
      await page.getByRole('tab', { name: 'Uploaded files', exact: true }).click();
      await page.getByRole('button', { name: 'Add', exact: true }).click();
      await page.getByLabel('Choose uploaded file').setInputFiles({ name: 'synthetic-chat-file.txt', mimeType: 'text/plain', buffer: Buffer.from('safe local upload') });
      await expect(page.getByRole('list', { name: 'Uploaded files' }).getByText('synthetic-chat-file.txt')).toBeVisible();

      const dialogPromise = page.waitForEvent('dialog');
      const deletion = page.waitForResponse(response => response.url().includes('/api/media/assets/uploaded-media-1?project=default') && response.request().method() === 'DELETE');
      const click = page.getByRole('button', { name: 'Delete synthetic-chat-file.txt' }).click();
      const dialog = await dialogPromise;
      expect(dialog.message()).toContain('Chat message text remains');
      expect(dialog.message()).toMatch(/original source file outside OpenAgent storage/i);
      await dialog.accept();
      await click;
      const response = await deletion;
      expect(response.status()).toBe(200);
      await expect(page.getByRole('list', { name: 'Uploaded files' }).getByText('synthetic-chat-file.txt')).toHaveCount(0);
      await expect(page.getByRole('status')).toContainText('stored copy were deleted');
      expect(requests.filter(request => request.method === 'DELETE')).toHaveLength(1);
      await page.reload();
      await page.getByRole('tab', { name: 'Uploaded files', exact: true }).click();
      await expect(page.getByRole('heading', { name: 'No uploaded files' })).toBeVisible();
    });
  }
}

test('Files retains an asset and reports protected Voice references on conflict', async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('oa-second-data-mode', 'live');
    localStorage.setItem('oa-second-api-token', 'e2e-media-write-token');
  });
  await page.route('**/api/media/assets?**', route => route.fulfill({ json: { assets: [
    { id: 'voice-used-asset', title: 'voice.wav', source: 'voice', media_type: 'audio', mime_type: 'audio/wav', size_bytes: 32 },
  ] } }));
  await page.route('**/api/media/assets/voice-used-asset?project=default', route => route.fulfill({ status: 409, json: { error: 'This media file is still used by a saved Voice persona or session.' } }));
  await page.goto('/?view=files');
  await page.getByRole('tab', { name: 'Uploaded files', exact: true }).click();
  const dialogPromise = page.waitForEvent('dialog');
  const click = page.getByRole('button', { name: 'Delete voice.wav' }).click();
  await (await dialogPromise).accept();
  await click;
  await expect(page.getByRole('alert').filter({ hasText: 'still used by a saved Voice persona or session' })).toBeVisible();
  await expect(page.getByRole('list', { name: 'Uploaded files' }).getByText('voice.wav')).toBeVisible();
});
