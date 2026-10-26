import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Live notification preferences save and reload (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      await page.setViewportSize(viewport);
      await page.addInitScript(({ selectedTheme }) => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', selectedTheme);
      }, { selectedTheme: theme });

      let completionEnabled = false;
      const writes: Array<Record<string, unknown>> = [];
      await page.route('**/api/integrations/preferences**', async route => {
        if (route.request().method() === 'GET') {
          await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ preferences: [{ key: 'task_completion', value: completionEnabled }] }) });
          return;
        }
        const body = route.request().postDataJSON() as Record<string, unknown>;
        writes.push(body);
        expect(body).toEqual({ project: 'default', scope: 'notifications', key: 'task_completion', value: true });
        completionEnabled = true;
        await route.fulfill({ status: 201, contentType: 'application/json', body: JSON.stringify({ preference: { ...body, updated_at: '2026-10-24T12:00:00Z' } }) });
      });
      await page.route('**/api/tasks/notifications**', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ notifications: [] }) }));

      await page.goto('/?view=settings');
      await page.getByRole('tab', { name: 'Notifications' }).click();
      const preference = page.getByRole('checkbox', { name: 'Task completion' });
      await expect(preference).toBeVisible();
      await expect(preference).not.toBeChecked();
      await preference.click();
      await expect(preference).toBeChecked();
      await expect(page.getByText('Notification preference saved to the backend.')).toBeVisible();
      expect(writes).toHaveLength(1);

      await page.reload();
      await page.getByRole('tab', { name: 'Notifications' }).click();
      await expect(page.getByRole('checkbox', { name: 'Task completion' })).toBeChecked();
    });
  }
}
