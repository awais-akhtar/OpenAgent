import { expect, test } from '@playwright/test';

for (const viewport of [
  { name: '1440x900', width: 1440, height: 900 },
  { name: '1280x800', width: 1280, height: 800 },
]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Live Calendar upcoming events do not carry the Demo Examples label · ${viewport.name} · ${theme}`, async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.addInitScript(({ selectedTheme }) => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', selectedTheme);
      }, { selectedTheme: theme });

      const startsAt = new Date(Date.now() + 24 * 60 * 60 * 1000);
      startsAt.setHours(11, 0, 0, 0);
      await page.route('**/api/calendar/events**', route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ events: [{
          id: 'live-calendar-review', title: 'Backend planning review',
          starts_at: startsAt.toISOString(), ends_at: new Date(startsAt.getTime() + 60 * 60 * 1000).toISOString(),
          location: 'Project room',
        }] }),
      }));

      await page.goto('/?view=calendar');
      const upcoming = page.locator('.calendar-side').locator('section').filter({ has: page.getByRole('heading', { name: 'Upcoming events' }) });
      await expect(upcoming.getByRole('heading', { name: 'Upcoming events' })).toBeVisible();
      await expect(upcoming.getByRole('button', { name: /Backend planning review/ })).toBeVisible();
      await expect(upcoming.getByText('Examples', { exact: true })).toHaveCount(0);
      await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
    });
  }
}
