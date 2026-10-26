import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Live inbox moves a message to backend trash (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      await page.setViewportSize(viewport);
      await page.addInitScript(({ themeName }) => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', themeName);
      }, { themeName: theme });

      let folder = 'inbox';
      const mutations: Record<string, unknown>[] = [];
      const drafts: Record<string, unknown>[] = [];
      let subject = 'Approved local retention notice';
      let sender = 'owner@example.test';
      let body = 'Synthetic test message.';
      let failRestoreOnce = true;
      const edits: Record<string, unknown>[] = [];
      await page.route('**/*', async route => {
        const url = new URL(route.request().url());
        if (!url.pathname.startsWith('/api/')) { await route.continue(); return; }
        if (url.pathname === '/api/health') { await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'ok' }) }); return; }
        if (url.pathname.startsWith('/api/conversations')) { await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ conversations: [] }) }); return; }
        if (url.pathname.endsWith('/inbox')) {
          await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ messages: folder === 'inbox' ? [{ id: 'message-trash-test', subject, sender, body, created_at: '2026-10-24', folder }] : [] }) });
          return;
        }
        if (url.pathname.endsWith('/messages/state') && route.request().method() === 'POST') {
          const body = route.request().postDataJSON() as Record<string, unknown>;
          mutations.push(body);
          if (body.folder === 'inbox' && failRestoreOnce) {
            failRestoreOnce = false;
            await route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ error: 'Temporary mailbox failure.' }) });
            return;
          }
          folder = String(body.folder ?? folder);
          await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ message: { id: 'message-trash-test', folder } }) });
          return;
        }
        if (url.pathname.endsWith('/drafts') && route.request().method() === 'POST') {
          const body = route.request().postDataJSON() as Record<string, unknown>;
          drafts.push(body);
          await route.fulfill({ status: 201, contentType: 'application/json', body: JSON.stringify({ draft: { id: 'draft-copy-test', ...body } }) });
          return;
        }
        if (url.pathname.endsWith('/messages/update') && route.request().method() === 'POST') {
          const update = route.request().postDataJSON() as Record<string, unknown>;
          edits.push(update);
          subject = String(update.subject);
          sender = String(update.sender);
          body = String(update.body);
          await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ message: { id: 'message-trash-test', subject, sender, body, folder } }) });
          return;
        }
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ messages: [], runs: [], drafts: [], sent: [], events: [], connectors: [] }) });
      });

      await page.goto('/?view=communications');
      await expect(page.getByText('Communications', { exact: true })).toBeVisible();
      await page.getByRole('button', { name: /Approved local retention notice/ }).click();
      await expect(page.getByRole('heading', { name: 'Approved local retention notice' })).toBeVisible();
      await page.getByRole('button', { name: 'Copy to drafts' }).click();
      await expect(page.getByText('Draft copy draft-copy-test saved by the backend. It was not sent.')).toBeVisible();
      expect(drafts).toEqual([{ platform: 'email', recipients: [], subject: 'Approved local retention notice', body: 'Synthetic test message.', message_id: 'message-trash-test', project: 'default' }]);
      await page.getByRole('button', { name: 'Edit message' }).click();
      const editDialog = page.getByRole('dialog', { name: 'Edit inbox message' });
      await editDialog.getByLabel('Subject').fill('Approved local retention notice · revised');
      await editDialog.getByLabel('Sender').fill('owner-updated@example.test');
      await editDialog.getByRole('textbox', { name: 'Message' }).fill('Updated synthetic test message.');
      await editDialog.getByRole('button', { name: 'Save changes' }).click();
      await expect(page.getByRole('heading', { name: 'Approved local retention notice · revised' })).toBeVisible();
      expect(edits).toEqual([{ message_id: 'message-trash-test', project: 'default', subject: 'Approved local retention notice · revised', sender: 'owner-updated@example.test', body: 'Updated synthetic test message.' }]);
      page.once('dialog', dialog => dialog.accept());
      await page.getByRole('button', { name: 'Move to trash' }).click();
      await expect(page.getByText('Message moved to trash.')).toBeVisible();
      await expect(page.getByRole('heading', { name: 'Approved local retention notice' })).toHaveCount(0);
      expect(mutations).toEqual([{ message_id: 'message-trash-test', folder: 'trash', project: 'default' }]);
      await page.getByRole('button', { name: 'Undo' }).click();
      await expect(page.locator('.communications-inbox-feedback')).toContainText('Mailbox state could not be updated.');
      await expect(page.getByRole('alert')).toContainText('Temporary mailbox failure.');
      await page.getByRole('button', { name: 'Undo' }).click();
      await expect(page.getByText('Message restored to the inbox.')).toBeVisible();
      await expect(page.getByRole('heading', { name: 'Approved local retention notice · revised' })).toBeVisible();
      expect(mutations).toEqual([
        { message_id: 'message-trash-test', folder: 'trash', project: 'default' },
        { message_id: 'message-trash-test', folder: 'inbox', project: 'default' },
        { message_id: 'message-trash-test', folder: 'inbox', project: 'default' },
      ]);
    });
  }
}
