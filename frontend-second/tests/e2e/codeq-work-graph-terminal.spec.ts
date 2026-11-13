import { expect, test } from '@playwright/test';

const projectLabel = process.env.OPENAGENT_CODEQ_HISTORY_PROJECT_LABEL ?? '';
const runId = process.env.OPENAGENT_CODEQ_HISTORY_RUN_ID ?? '';
const expectedStatus = process.env.OPENAGENT_CODEQ_HISTORY_STATUS ?? '';

test.skip(!projectLabel || !runId || !expectedStatus, 'Set a disposable project label, real persisted CodeQ run id, and its expected terminal status.');

test('Work Graph refreshes persisted terminal run, task, evidence, and event status', async ({ page }) => {
  await page.goto('/?view=codeq');
  await page.getByRole('combobox', { name: 'CodeQ project' }).selectOption({ label: projectLabel });
  await page.getByRole('tab', { name: 'Work Graph' }).click();

  const graph = page.getByRole('region', { name: 'Work Graph' });
  const history = graph.getByRole('combobox', { name: 'Work Graph run history' });
  await expect(history).toBeVisible();
  await expect.poll(async () => history.locator(`option[value="${runId}"]`).count(), { timeout: 30_000 }).toBe(1);
  await history.selectOption(runId);
  await expect(history.locator('option:checked')).toContainText(expectedStatus);

  await graph.getByRole('tab', { name: 'Tasks' }).click();
  const tasks = graph.getByRole('region', { name: 'Tasks view' });
  await expect(tasks.locator('li').filter({ hasText: /— completed$/i }).first()).toBeVisible();
  await expect(tasks).toContainText('Run project test suite — needs_fix');

  await graph.getByRole('tab', { name: 'Evidence' }).click();
  const evidence = graph.getByRole('region', { name: 'Evidence view' });
  await expect(evidence.getByRole('listitem').filter({ hasText: /test — failed \(exit 1\)/i }).first()).toBeVisible();

  const activity = graph.locator('details.codeq-work-graph-events');
  await activity.locator('summary').click();
  await expect(activity.getByRole('list', { name: 'Recent CodeQ events' })).toContainText(`run.${expectedStatus}`);
});
