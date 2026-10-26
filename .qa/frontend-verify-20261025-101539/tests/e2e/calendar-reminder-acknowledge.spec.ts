import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Calendar Live reminder acknowledgement waits for backend confirmation (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      await page.setViewportSize(viewport);
      await page.addInitScript(({ selectedTheme }) => {
        localStorage.setItem('oa-second-data-mode', 'demo');
        localStorage.setItem('oa-second-theme', selectedTheme);
      }, { selectedTheme: theme });

      let acknowledged = false;
      const requests: Array<{ method: string; body: string | null }> = [];
      await page.route('**/api/calendar/reminders**', async route => {
        const request = route.request();
        const method = request.method();
        requests.push({ method, body: request.postData() });
        if (method === 'GET') {
          await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ reminders: acknowledged ? [] : [{ reminder_id: 'reminder-1', event_title: 'Planning review', trigger_at: '2026-10-24T09:00:00Z' }] }) });
          return;
        }
        expect(new URL(request.url()).pathname).toBe('/api/calendar/reminders/reminder-1/acknowledge');
        expect(request.postDataJSON()).toEqual({ project: 'default' });
        acknowledged = true;
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ acknowledged: true, reminder_id: 'reminder-1' }) });
      });
      await page.route('**/api/calendar/events**', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ events: [] }) }));

      await page.goto('/?view=calendar');
      await page.getByRole('button', { name: 'Data mode: demo. Click to switch to live' }).click();
      await page.getByText('Calendar tools', { exact: true }).click();
      await page.getByRole('tab', { name: 'Reminders' }).click();
      const reminder = page.locator('article.timeline-row').filter({ hasText: 'Planning review' });
      await expect(reminder).toBeVisible();
      await reminder.getByRole('button', { name: 'Acknowledge' }).click();
      await expect(page.getByText('Reminder acknowledged.')).toBeVisible();
      await expect(reminder).toHaveCount(0);
      expect(requests.some(request => request.method === 'POST')).toBe(true);
      await expect(page.getByText('No reminders are due right now.')).toBeVisible();
    });
  }
}
