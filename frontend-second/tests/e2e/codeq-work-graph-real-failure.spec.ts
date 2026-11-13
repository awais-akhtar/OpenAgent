import { expect, test } from '@playwright/test';

const projectLabel = process.env.OPENAGENT_CODEQ_FAILURE_PROJECT_LABEL ?? '';
const runId = process.env.OPENAGENT_CODEQ_FAILURE_RUN_ID ?? '';

test.skip(!projectLabel || !runId, 'Set a disposable project label and persisted failed CodeQ run id.');

test('Work Graph shows a real failed model run without inventing test evidence', async ({ page }) => {
  await page.goto('/?view=codeq');
  await page.getByRole('combobox', { name: 'CodeQ project' }).selectOption({ label: projectLabel });
  await page.getByRole('tab', { name: 'Work Graph' }).click();

  const graph = page.getByRole('region', { name: 'Work Graph' });
  const history = graph.getByRole('combobox', { name: 'Work Graph run history' });
  await expect(history.locator(`option[value="${runId}"]`)).toHaveCount(1);
  await history.selectOption(runId);
  await expect(history.locator('option:checked')).toContainText('failed');

  await graph.getByRole('tab', { name: 'Tasks' }).click();
  const tasks = graph.getByRole('region', { name: 'Tasks view' });
  await expect(tasks).toContainText('Implement slug normalization logic in src/slug.js');
  await expect(tasks).toContainText('needs_fix');

  await graph.getByRole('tab', { name: 'Evidence' }).click();
  const evidence = graph.getByRole('region', { name: 'Evidence view' });
  await expect(evidence).not.toContainText('test — passed');
  await expect(evidence).not.toContainText('runtime — passed');

  const activity = graph.locator('details.codeq-work-graph-events');
  await activity.locator('summary').click();
  await expect(activity.getByRole('list', { name: 'Recent CodeQ events' })).toContainText('run.failed');
});
