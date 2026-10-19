import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Live Settings notifications replace stale status after refresh (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const demoFixtureRequests: string[] = [];
      const notification = { id: 'notice-1', title: 'Task completed', body: 'Daily review finished.', level: 'info', created_at: '2026-10-18T10:00:00Z', read_at: null as string | null };
      let scenario: 'populated' | 'empty' | 'error' = 'populated';
      let markedRead = false;
      let holdInitialLoad = true;
      let releaseInitialLoad: () => void = () => {};
      const initialLoadGate = new Promise<void>(resolve => { releaseInitialLoad = resolve; });

      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/demo/')) demoFixtureRequests.push(request.url());
      });
      await page.setViewportSize(viewport);
      await page.addInitScript(({ themeName }) => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', themeName);
      }, { themeName: theme });
      await page.route('**/api/tasks/notifications**', async route => {
        const url = new URL(route.request().url());
        if (route.request().method() === 'POST' && url.pathname.endsWith('/read')) {
          markedRead = true;
          await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ notification: { ...notification, read_at: '2026-10-18T10:02:00Z' } }) });
          return;
        }
        if (holdInitialLoad) {
          holdInitialLoad = false;
          await initialLoadGate;
        }
        if (scenario === 'error') {
          await route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ error: 'Notification service unavailable' }) });
          return;
        }
        const rows = scenario === 'empty' ? [] : [{ ...notification, read_at: markedRead ? '2026-10-18T10:02:00Z' : null }];
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ notifications: rows }) });
      });

      await page.goto('/?view=settings');
      await page.getByRole('tab', { name: 'Notifications' }).click();
      await expect(page.getByText('Loading notifications…')).toBeVisible();
      releaseInitialLoad();
      const markRead = page.getByRole('button', { name: 'Mark read' });
      await expect(markRead).toBeVisible();
      await markRead.click();
      await expect(page.getByText('Notification marked as read.', { exact: true })).toBeVisible();
      await expect(page.getByText('info · Read', { exact: true })).toBeVisible();

      scenario = 'empty';
      await page.getByRole('button', { name: 'Refresh' }).click();
      await expect(page.getByText('No notifications have been recorded for this project.', { exact: true })).toBeVisible();
      await expect(page.getByText('Notification marked as read.', { exact: true })).toHaveCount(0);

      scenario = 'error';
      await page.getByRole('button', { name: 'Refresh' }).click();
      await expect(page.getByRole('alert')).toContainText('Notifications unavailable');
      await expect(page.getByText('Notification marked as read.', { exact: true })).toHaveCount(0);
      expect(demoFixtureRequests).toEqual([]);
    });
  }
}
