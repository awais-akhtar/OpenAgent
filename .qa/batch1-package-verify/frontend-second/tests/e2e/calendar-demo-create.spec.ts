import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Calendar event create validates, saves, details and persists locally (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
      });
      await page.setViewportSize(viewport);
      await page.addInitScript(({ selectedTheme }) => {
        localStorage.setItem('oa-second-data-mode', 'demo');
        localStorage.setItem('oa-second-theme', selectedTheme);
      }, { selectedTheme: theme });
      await page.goto('/?view=calendar');
      await page.getByRole('button', { name: 'New', exact: true }).click();

      const dialog = page.getByRole('dialog', { name: 'New calendar event' });
      await expect(dialog).toBeVisible();
      await dialog.getByLabel('Title').fill('Local planning block');
      const starts = dialog.getByLabel('Starts');
      const ends = dialog.getByLabel('Ends');
      const startValue = await starts.inputValue();
      await ends.fill(startValue);
      await dialog.getByRole('button', { name: 'Create event' }).click();
      await expect(dialog.getByRole('alert')).toHaveText('End time must be later than start time.');

      const endDate = new Date(`${startValue}:00`);
      endDate.setHours(endDate.getHours() + 1);
      const laterValue = `${endDate.getFullYear()}-${String(endDate.getMonth() + 1).padStart(2, '0')}-${String(endDate.getDate()).padStart(2, '0')}T${String(endDate.getHours()).padStart(2, '0')}:${String(endDate.getMinutes()).padStart(2, '0')}`;
      await ends.fill(laterValue);
      await dialog.getByLabel('Location').fill('Project room');
      await dialog.getByLabel('Description').fill('Review the next release plan.');
      await dialog.getByLabel('Attendees (comma separated)').fill('Alex Morgan, Jamie Lee');
      await dialog.getByRole('button', { name: 'Create event' }).click();
      await expect(dialog).toBeHidden();

      const eventButton = page.locator('.calendar-upcoming-row').filter({ hasText: 'Local planning block' });
      await expect(eventButton).toBeVisible();
      await eventButton.click();
      const details = page.getByRole('dialog', { name: 'Local planning block' });
      await expect(details).toBeVisible();
      await expect(details).toContainText('This event is a visual fixture; real events must come from the calendar API.');
      await expect(details.getByRole('button', { name: 'Cancel' })).toBeVisible();
      await expect(details.getByRole('button', { name: 'Close', exact: true })).toBeVisible();
      await details.getByRole('button', { name: 'Close', exact: true }).click();
      await expect(details).toBeHidden();
      await expect(page.locator('#toast-region')).toContainText('Visual preview only. No backend changes were saved.');
      const upcomingEventButton = page.locator('.calendar-upcoming').getByRole('button', { name: /Team stand-up/ });
      await upcomingEventButton.click();
      const upcomingDetails = page.getByRole('dialog', { name: 'Team stand-up' });
      await expect(upcomingDetails).toContainText('Team stand-up');
      await page.keyboard.press('Escape');
      await expect(upcomingDetails).toBeHidden();
      await expect(upcomingEventButton).toBeFocused();

      const storageKey = 'oa-second-isolated-demo-v1';
      const persistedBeforeReload = await page.evaluate(key => JSON.parse(localStorage.getItem(key) ?? 'null') as { data?: { workspaceDemo?: { state?: { calendarEvents?: Array<{ title?: string }> } } } } | null, storageKey);
      expect(persistedBeforeReload?.data?.workspaceDemo?.state?.calendarEvents?.map(event => event.title)).toContain('Local planning block');
      await page.reload();
      const persistedAfterReload = await page.evaluate(key => JSON.parse(localStorage.getItem(key) ?? 'null') as { data?: { workspaceDemo?: { state?: { calendarEvents?: Array<{ title?: string }> } } } } | null, storageKey);
      expect(persistedAfterReload?.data?.workspaceDemo?.state?.calendarEvents?.map(event => event.title), JSON.stringify(persistedAfterReload)).toContain('Local planning block');
      await expect(page.locator('.calendar-upcoming-row').filter({ hasText: 'Local planning block' })).toBeVisible();
      expect(apiRequests).toEqual([]);
    });
  }
}

test('Live Calendar detail stays truthful and does not show the Demo preview toast', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  const startsAt = new Date();
  startsAt.setDate(startsAt.getDate() - ((startsAt.getDay() + 6) % 7));
  startsAt.setHours(9, 0, 0, 0);
  const endsAt = new Date(startsAt.getTime() + 60 * 60 * 1000);
  await page.route('**/api/calendar/events**', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ events: [{ id: 'live-event-1', title: 'Live backend review', starts_at: startsAt.toISOString(), ends_at: endsAt.toISOString() }] }) });
  });
  await page.goto('/?view=calendar');
  await expect(page.getByRole('heading', { name: 'Calendar', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Data mode: demo. Click to switch to live' }).click();
  const liveEvent = page.locator('.calendar-event[title="Live backend review"]');
  await expect(liveEvent).toBeVisible();
  await liveEvent.click();
  const dialog = page.getByRole('dialog', { name: 'Live backend review' });
  await expect(dialog).toContainText('Event details returned by the connected calendar.');
  await dialog.getByRole('button', { name: 'Close', exact: true }).click();
  await expect(dialog).toBeHidden();
  await expect(page.getByText('Visual preview only. No backend changes were saved.')).toHaveCount(0);
});
