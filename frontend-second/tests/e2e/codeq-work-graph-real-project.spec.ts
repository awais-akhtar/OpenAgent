import { expect, test } from '@playwright/test';

const projectLabel = process.env.OPENAGENT_CODEQ_REAL_PROJECT_LABEL ?? '';
const goal = process.env.OPENAGENT_CODEQ_REAL_GOAL ?? '';
const runId = process.env.OPENAGENT_CODEQ_REAL_RUN_ID ?? '';
const sourceFile = process.env.OPENAGENT_CODEQ_REAL_SOURCE_FILE ?? '';
const sourceMarker = process.env.OPENAGENT_CODEQ_REAL_SOURCE_MARKER ?? '';

test.skip(!projectLabel || !goal || !runId || !sourceFile, 'Set the disposable real-model project, goal, run id, and indexed source file.');

test('Work Graph renders a persisted real-model run and its source evidence from Python', async ({ page }) => {
  await page.goto('/?view=codeq');
  await page.getByRole('combobox', { name: 'CodeQ project' }).selectOption({ label: projectLabel });
  await page.getByRole('tab', { name: 'Work Graph' }).click();

  const graph = page.getByRole('region', { name: 'Work Graph' });
  await expect(graph.getByLabel('Project goal')).toContainText(goal);
  const history = graph.getByRole('combobox', { name: 'Work Graph run history' });
  await expect(history.locator(`option[value="${runId}"]`)).toHaveCount(1);
  await history.selectOption(runId);
  await expect(history.locator('option:checked')).toContainText('completed');
  await expect(graph.getByLabel('Recent CodeQ events')).toContainText('run.completed');

  await graph.getByRole('tab', { name: 'Tasks' }).click();
  const tasks = graph.getByRole('region', { name: 'Tasks view' });
  await expect(tasks.getByRole('listitem').filter({ hasText: /— completed$/i }).first()).toBeVisible();

  await graph.getByRole('tab', { name: 'Evidence' }).click();
  const evidence = graph.getByRole('region', { name: 'Evidence view' });
  await expect(evidence.getByRole('listitem').filter({ hasText: /test — passed \(exit 0\)/i }).first()).toBeVisible();

  await graph.getByRole('tab', { name: 'Code' }).click();
  await graph.getByRole('option', { name: sourceFile, exact: true }).click();
  await expect(graph.getByLabel('Node details')).toContainText('fresh');
  if (sourceMarker) await expect(graph.getByLabel('Current source')).toContainText(sourceMarker);
  await expect(graph.getByLabel('Node details')).toContainText('Linked tasks');
});
