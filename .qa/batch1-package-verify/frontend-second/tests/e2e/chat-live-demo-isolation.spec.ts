import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`switching Live Chat to Demo clears backend conversation state (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
      });
      await page.setViewportSize(viewport);
      await page.addInitScript(({ themeName }) => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', themeName);
        sessionStorage.setItem('oa-second-conversation-id', 'private-live-conversation');
      }, { themeName: theme });
      await page.route('**/*', async route => {
        const url = new URL(route.request().url());
        if (!url.pathname.startsWith('/api/')) { await route.continue(); return; }
        if (url.pathname === '/api/messages') {
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ messages: [
              { id: 'live-user-message', role: 'user', content: 'Private live user message' },
              { id: 'live-assistant-message', role: 'assistant', content: 'Private live assistant response' },
            ] }),
          });
          return;
        }
        if (url.pathname === '/api/conversations' || url.pathname === '/api/conversations/') {
          await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ conversations: [{ id: 'private-live-conversation', title: 'Private Live conversation' }] }) });
          return;
        }
        if (url.pathname === '/api/models') {
          await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ providers: [{ id: 'auto', label: 'Automatic', selected_model: 'GPT-4o', default_model: 'GPT-4o', installed_models: ['GPT-4o'] }] }) });
          return;
        }
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) });
      });

      await page.goto('/?view=chat&conversation_id=private-live-conversation');
      await expect(page.getByRole('button', { name: 'Data mode: live. Click to switch to demo' })).toBeVisible();
      await expect(page.getByText('Private live user message', { exact: true })).toBeVisible();
      await expect(page.getByText('Private live assistant response', { exact: true })).toBeVisible();
      await expect(page.getByRole('combobox', { name: 'Conversation' })).toHaveValue('private-live-conversation');

      const initialApiRequestCount = apiRequests.length;
      await page.getByRole('button', { name: 'Data mode: live. Click to switch to demo' }).click();
      await expect(page.getByRole('button', { name: 'Data mode: demo. Click to switch to live' })).toBeVisible();
      await expect(page.getByRole('heading', { name: 'Start a conversation' })).toBeVisible();
      await expect(page.getByRole('combobox', { name: 'Conversation' })).toHaveValue('');
      await expect(page.locator('.chat-thread .chat-message')).toHaveCount(0);
      await expect(page.getByText('Private Live conversation', { exact: true })).toHaveCount(0);
      await expect(page).not.toHaveURL(/conversation_id=/);
      expect(await page.evaluate(() => sessionStorage.getItem('oa-second-conversation-id'))).toBeNull();

      await page.waitForTimeout(700);
      expect(apiRequests.slice(initialApiRequestCount)).toEqual([]);
    });
  }
}
