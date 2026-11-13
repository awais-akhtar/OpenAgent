import { expect, test } from '@playwright/test';

const projectLabel = process.env.OPENAGENT_CODEQ_RUN_PROJECT_LABEL ?? '';
const goalText = process.env.OPENAGENT_CODEQ_RUN_GOAL ?? '';

test.skip(!projectLabel || !goalText, 'Set the disposable project label and goal for a local-model React-to-Python run.');

test('Live CodeQ starts a selected local model run and shows persisted evidence', async ({ page }) => {
  test.setTimeout(20 * 60 * 1000);
  await page.goto('/?view=codeq');
  await page.getByRole('combobox', { name: 'CodeQ project' }).selectOption({ label: projectLabel });
  await page.getByRole('button', { name: 'Choose CodeQ model' }).click();
  await page.getByRole('option', { name: 'phi4-mini:latest' }).click();
  await page.getByLabel('Ask CodeQ').fill(goalText);
  await expect(page.getByRole('button', { name: 'Send to CodeQ' })).toBeEnabled({ timeout: 120_000 });
  await page.getByRole('button', { name: 'Send to CodeQ' }).click();

  await page.getByRole('tab', { name: 'Work Graph' }).click();
  const graph = page.getByRole('region', { name: 'Work Graph' });
  const history = graph.getByRole('combobox', { name: 'Work Graph run history' });
  await expect.poll(async () => {
    const selected = await history.locator('option:checked').innerText();
    return /\b(completed|failed|blocked|cancelled)\b/i.test(selected) ? selected : '';
  }, { timeout: 15 * 60 * 1000, intervals: [2_000] }).not.toBe('');
  await expect(history.locator('option:checked')).toContainText('completed');
  await expect(page.getByLabel('Recent CodeQ events')).toContainText('run.completed');

  await graph.getByRole('tab', { name: 'Tasks' }).click();
  const tasks = graph.getByRole('region', { name: 'Tasks view' });
  await expect(tasks.getByRole('listitem').filter({ hasText: /completed/i }).first()).toBeVisible();
  await graph.getByRole('tab', { name: 'Evidence' }).click();
  const evidence = graph.getByRole('region', { name: 'Evidence view' });
  await expect(evidence.getByRole('listitem').filter({ hasText: /passed \(exit 0\)/i }).first()).toBeVisible();
});
