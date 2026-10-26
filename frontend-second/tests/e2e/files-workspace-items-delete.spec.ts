import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Files Live note can be created and deleted through the workspace API (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      await page.setViewportSize(viewport);
      await page.addInitScript(({ selectedTheme }) => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', selectedTheme);
        localStorage.setItem('oa-second-api-token', 'e2e-local-token');
      }, { selectedTheme: theme });

      const items: Array<Record<string, unknown>> = [];
      const paths: string[] = [];
      page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) paths.push(new URL(request.url()).pathname); });
      await page.route('**/api/workspace/items**', async route => {
        const request = route.request();
        const url = new URL(request.url());
        if (request.method() === 'GET' && url.pathname === '/api/workspace/items') {
          const includeArchived = url.searchParams.get('include_archived') === 'true';
          await route.fulfill({ json: { items: items.filter(item => includeArchived || !item.archived) } });
          return;
        }
        if (request.method() === 'POST' && url.pathname === '/api/workspace/items') {
          const body = request.postDataJSON() as Record<string, unknown>;
          const item = { id: `item-${items.length + 1}`, project: body.project ?? 'default', kind: body.kind ?? 'note', title: body.title, content: body.content, metadata: body.metadata ?? {}, updated_at: '2026-10-24T12:00:00Z' };
          items.push(item);
          await route.fulfill({ status: 201, json: { item } });
          return;
        }
        if (request.method() === 'POST' && url.pathname === '/api/workspace/items/update') {
          const body = request.postDataJSON() as Record<string, unknown>;
          const item = items.find(candidate => candidate.id === body.item_id && candidate.project === body.project);
          if (!item) { await route.fulfill({ status: 404, json: { error: 'Workspace item not found.' } }); return; }
          Object.assign(item, { title: body.title, content: body.content, metadata: body.metadata, updated_at: '2026-10-24T12:01:00Z' });
          await route.fulfill({ json: { item } });
          return;
        }
        if (request.method() === 'POST' && /\/api\/workspace\/items\/[^/]+\/(archive|restore)$/.test(url.pathname)) {
          const [, , , , id, action] = url.pathname.split('/');
          const item = items.find(candidate => candidate.id === id);
          if (!item) { await route.fulfill({ status: 404, json: { error: 'Workspace item not found.' } }); return; }
          item.archived = action === 'archive';
          await route.fulfill({ json: action === 'archive' ? { archived: true, item_id: id } : { restored: true, item_id: id } });
          return;
        }
        if (request.method() === 'DELETE') {
          const id = decodeURIComponent(url.pathname.split('/').at(-1) ?? '');
          const index = items.findIndex(item => item.id === id);
          if (index < 0) { await route.fulfill({ status: 404, json: { error: 'Workspace item not found.' } }); return; }
          items.splice(index, 1);
          await route.fulfill({ json: { deleted: true, id } });
          return;
        }
        await route.fulfill({ status: 405, json: { error: 'Unsupported test request.' } });
      });

      await page.goto('/?view=files');
      await page.getByRole('tab', { name: 'Notes', exact: true }).click();
      const list = page.getByRole('list', { name: 'Workspace notes' });
      const title = `Disposable note ${viewport.width} ${theme}`;
      await page.getByRole('button', { name: 'Add note', exact: true }).click();
      const createDialog = page.getByRole('dialog', { name: 'New workspace note' });
      await createDialog.getByLabel('Note title', { exact: true }).fill(title);
      await createDialog.getByLabel('Note', { exact: true }).fill('Synthetic Live API item for delete verification.');
      await createDialog.getByRole('button', { name: 'Create note', exact: true }).click();
      await expect(createDialog).toBeHidden();
      const row = list.getByRole('button', { name: new RegExp(title) });
      await expect(row).toBeVisible();
      const editedTitle = `${title} updated`;
      await page.getByLabel('Note title', { exact: true }).fill(editedTitle);
      await page.getByLabel('Note content', { exact: true }).fill('Edited synthetic Live API item.');
      await page.getByRole('button', { name: 'Save note', exact: true }).click();
      await expect(page.getByRole('status')).toContainText('Note updated');
      await expect(list.getByRole('button', { name: new RegExp(editedTitle) })).toBeVisible();
      expect(items[0]).toMatchObject({ project: 'default', title: editedTitle, content: 'Edited synthetic Live API item.' });
      const archiveResponse = page.waitForResponse(response => response.url().includes(`/api/workspace/items/${items[0].id}/archive`) && response.request().method() === 'POST');
      await page.getByRole('button', { name: 'Archive note' }).click();
      expect((await archiveResponse).ok()).toBe(true);
      await expect(list.getByRole('button', { name: new RegExp(editedTitle) })).toHaveCount(0);
      await page.getByLabel('Show archived').check();
      const archivedRow = list.getByRole('button', { name: new RegExp(editedTitle) });
      await expect(archivedRow).toContainText('Archived');
      await archivedRow.click();
      const restoreResponse = page.waitForResponse(response => response.url().includes(`/api/workspace/items/${items[0].id}/restore`) && response.request().method() === 'POST');
      await page.getByRole('button', { name: 'Restore note' }).click();
      expect((await restoreResponse).ok()).toBe(true);
      await expect(list.getByRole('button', { name: new RegExp(editedTitle) })).not.toContainText('Archived');
      await page.getByLabel('Show archived').uncheck();
      page.once('dialog', dialog => dialog.accept());
      await page.getByRole('button', { name: 'Delete note' }).click();
      await expect(list.getByRole('button', { name: new RegExp(editedTitle) })).toHaveCount(0);
      await expect(page.getByRole('status')).toContainText('Note deleted');
      await page.reload();
      await page.getByRole('tab', { name: 'Notes', exact: true }).click();
      await expect(page.getByRole('list', { name: 'Workspace notes' }).getByRole('button', { name: new RegExp(title) })).toHaveCount(0);
      expect(items).toHaveLength(0);
      expect(paths).toContain('/api/workspace/items');
      expect(paths).toContain('/api/workspace/items/update');
      expect(paths.some(path => /^\/api\/workspace\/items\/item-1$/.test(path))).toBe(true);
    });
  }
}
