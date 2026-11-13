import { expect, test } from '@playwright/test';

// Run against a temporary localhost Python server using
// scripts/phase6_work_graph_server.py; this spec intentionally has no API mocks.
test('React Work Graph reads persisted records and tool events from Python HTTP', async ({ page }) => {
  await page.goto('/?view=codeq');
  await page.getByRole('combobox', { name: 'CodeQ project' }).selectOption({ label: 'Phase 6 live fixture' });
  await page.getByRole('tab', { name: 'Work Graph' }).click();

  await expect(page.getByLabel('Project goal')).toContainText('Verify the Work Graph live integration');
  await expect(page.getByRole('combobox', { name: 'Work Graph run history' })).toContainText('interactive');
  await expect(page.getByText(/Persisted run history/)).toBeVisible();
  await expect(page.getByText(/Run activity \(\d+ persisted\/received events\)/)).toBeVisible();
  await expect(page.getByLabel('Recent CodeQ events')).toContainText('run.created');

  await page.getByRole('region', { name: 'Work Graph' }).getByRole('tab', { name: 'Code' }).click();
  await page.getByRole('option', { name: 'src/app.py' }).click();
  await expect(page.getByLabel('Current source')).toContainText('def save_account');
  await expect(page.getByLabel('Node details')).toContainText('fresh');

  await page.getByRole('tab', { name: 'Tasks' }).click();
  await expect(page.getByText('No tasks yet for this run.')).toBeVisible();
  await page.getByRole('tab', { name: 'Evidence' }).click();
  await expect(page.getByText('No executed checks recorded for this run.')).toBeVisible();
});
