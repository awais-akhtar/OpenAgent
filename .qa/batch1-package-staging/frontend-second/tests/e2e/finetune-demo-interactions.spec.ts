import { expect, test } from '@playwright/test';

const sizes = [
  { width: 1440, height: 900, label: '1440x900' },
  { width: 1280, height: 800, label: '1280x800' },
];
const tabs = [
  { tab: 'Datasets', singular: 'dataset', collection: 'datasets' },
  { tab: 'Jobs', singular: 'job', collection: 'jobs' },
  { tab: 'Checkpoints', singular: 'checkpoint', collection: 'checkpoints' },
  { tab: 'Results', singular: 'result', collection: 'results' },
];

for (const size of sizes) for (const theme of ['light', 'dark']) {
  test(`Fine-tuning demo workflows persist and stay local (${size.label}, ${theme})`, async ({ page }) => {
    const apiRequests: string[] = [];
    page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
    await page.setViewportSize({ width: size.width, height: size.height });
    await page.addInitScript(({ theme }) => {
      localStorage.setItem('oa-second-data-mode', 'demo');
      localStorage.setItem('oa-second-theme', theme);
      if (!sessionStorage.getItem('finetune-demo-seeded')) {
        localStorage.removeItem('oa-second-isolated-demo-v1');
        sessionStorage.setItem('finetune-demo-seeded', '1');
      }
    }, { theme });
    await page.goto('/?view=finetune');
    await expect(page.getByText('Fine-tuning', { exact: true }).first()).toBeVisible();

    for (const { tab, singular, collection } of tabs) {
      await page.getByRole('tab', { name: tab, exact: true }).click();
      const list = page.locator('.demo-records');
      const createdName = `Regression ${tab}`;
      await page.getByRole('button', { name: 'Add', exact: true }).click();
      const createDialog = page.getByRole('dialog', { name: `New ${singular}` });
      await expect(createDialog).toBeVisible();
      const nameField = createDialog.getByLabel('Name');
      await nameField.fill('');
      await createDialog.getByRole('button', { name: 'Create item' }).click();
      expect(await nameField.evaluate(element => !(element as HTMLInputElement).validity.valid)).toBe(true);
      await nameField.fill(createdName);
      await createDialog.getByLabel('Description').fill('Local deterministic example');
      await createDialog.getByLabel('Status').fill('Ready · demo');
      await createDialog.getByRole('button', { name: 'Create item' }).click();
      await expect(list.getByText(createdName, { exact: true })).toBeVisible();
      await expect(page.getByRole('region', { name: 'Fine-tuning activity' })).toContainText('Record created');

      await page.getByRole('button', { name: `More actions for ${createdName}`, exact: true }).click();
      const actions = page.getByRole('dialog', { name: 'Training record actions' });
      await actions.getByRole('button', { name: 'Duplicate record' }).click();
      const copyName = `${createdName} (copy)`;
      await expect(list.getByText(copyName, { exact: true })).toBeVisible();
      await expect(page.getByRole('region', { name: 'Fine-tuning activity' })).toContainText('Record duplicated');

      await page.getByRole('button', { name: `More actions for ${createdName}`, exact: true }).click();
      await page.getByRole('dialog', { name: 'Training record actions' }).getByRole('button', { name: 'Edit record' }).click();
      const editDialog = page.getByRole('dialog', { name: `Edit ${singular}` });
      await editDialog.getByLabel('Description').fill('Edited local fixture');
      await editDialog.getByRole('button', { name: 'Save changes' }).click();
      await expect(list.getByText('Edited local fixture')).toBeVisible();

      const search = page.getByRole('textbox', { name: `Search ${collection}` });
      await search.fill('no matching record in this fixture');
      await expect(list.getByRole('status')).toContainText('No matching records');
      await search.clear();

      const detailsTrigger = page.getByRole('button', { name: `Details for ${createdName}`, exact: true });
      await detailsTrigger.click();
      const detailDialog = page.getByRole('dialog', { name: createdName });
      await expect(detailDialog).toContainText('Edited local fixture');
      await page.keyboard.press('Escape');
      await expect(detailDialog).toBeHidden();
      await expect(detailsTrigger).toBeFocused();

      await page.reload();
      await page.getByRole('tab', { name: tab, exact: true }).click();
      const reloadedList = page.locator('.demo-records');
      await expect(reloadedList.getByText(createdName, { exact: true })).toBeVisible();
      await expect(reloadedList.getByText(copyName, { exact: true })).toBeVisible();
      await expect(page.getByRole('region', { name: 'Fine-tuning activity' })).toContainText('Record duplicated');

      await page.getByRole('button', { name: `More actions for ${createdName}`, exact: true }).click();
      const deleteActions = page.getByRole('dialog', { name: 'Training record actions' });
      page.once('dialog', dialog => dialog.dismiss());
      await deleteActions.getByRole('button', { name: 'Delete' }).click();
      await expect(reloadedList.getByText(createdName, { exact: true })).toBeVisible();
      page.once('dialog', dialog => dialog.accept());
      await deleteActions.getByRole('button', { name: 'Delete' }).click();
      await expect(reloadedList.getByText(createdName, { exact: true })).toHaveCount(0);
      await expect(page.getByRole('region', { name: 'Fine-tuning activity' })).toContainText('Record deleted');
    }

    await page.getByRole('tab', { name: 'Jobs', exact: true }).click();
    await expect(page.getByLabel('Dataset')).not.toHaveValue('');
    await page.getByRole('button', { name: 'Simulate training' }).click();
    const trace = page.locator('.trace-mount.trace-finetune');
    await expect(trace).toBeVisible();
    await expect(page.getByRole('status').filter({ hasText: 'No model was trained' })).toBeVisible();
    await expect(page.locator('.demo-records').getByText('Preview · demo')).toBeVisible();
    await expect(page.locator('.demo-records').getByText(/no model was trained and no checkpoint was created/i)).toBeVisible();
    await trace.getByRole('button', { name: 'Stop', exact: true }).click();
    await expect(trace.locator('.trace-head-state')).toHaveText('Cancelled');

    const guideButton = page.getByRole('button', { name: 'Integration guide' });
    await guideButton.click();
    const guide = page.getByRole('dialog', { name: 'Demo and integration guide' });
    await expect(guide).toContainText('Demo mode is offline');
    await expect(guide).not.toContainText(/Codex should|demoDB|fixtures with API adapters/i);
    const downloadPromise = page.waitForEvent('download');
    await guide.getByRole('button', { name: 'Export sample data' }).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/^openagent-demo-finetune\.json$/);
    await page.keyboard.press('Escape');
    await expect(guide).toBeHidden();
    await expect(guideButton).toBeFocused();
    expect(apiRequests).toEqual([]);
  });
}

test('Fine-tuning search and dialogs reset when changing collections', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.goto('/?view=finetune');
  await page.getByRole('tab', { name: 'Datasets', exact: true }).click();
  const datasetSearch = page.getByRole('textbox', { name: 'Search datasets' });
  await datasetSearch.fill('no dataset matches this');
  await expect(page.locator('.demo-records')).toContainText('No matching records');
  await page.getByRole('tab', { name: 'Jobs', exact: true }).click();
  await expect(page.locator('.demo-records')).not.toContainText('No matching records');
  await expect(page.locator('.demo-records .demo-record').first()).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Search jobs' })).toHaveValue('');
  await page.getByRole('tab', { name: 'Datasets', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Search datasets' })).toHaveValue('');
  await expect(page.locator('.demo-records .demo-record').first()).toBeVisible();
});

test('Fine-tuning records preserve distinct edit events after reload', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.goto('/?view=finetune');
  const name = 'Unique event regression dataset';
  await page.getByRole('tab', { name: 'Datasets', exact: true }).click();
  const firstRecord = page.locator('.demo-record').first();
  await firstRecord.getByRole('button', { name: /^More actions for / }).click();
  await page.getByRole('dialog', { name: 'Training record actions' }).getByRole('button', { name: 'Edit record' }).click();
  await page.getByLabel('Name').fill(name);
  await page.getByRole('button', { name: 'Save changes' }).click();
  await page.getByRole('button', { name: `More actions for ${name}`, exact: true }).click();
  await page.getByRole('dialog', { name: 'Training record actions' }).getByRole('button', { name: 'Edit record' }).click();
  await page.getByLabel('Description').fill('Second distinct edit event');
  await page.getByRole('button', { name: 'Save changes' }).click();
  const readEvents = () => page.evaluate(() => {
    const state = JSON.parse(localStorage.getItem('oa-second-isolated-demo-v1') ?? '{}');
    return state?.data?.localActivity?.finetune ?? [];
  });
  let events = await readEvents();
  const edits = events.filter((event: { item: string; action: string }) => event.item === name && event.action === 'Record edited');
  expect(edits).toHaveLength(2);
  expect(new Set(edits.map((event: { id: string }) => event.id)).size).toBe(2);
  await page.reload();
  events = await readEvents();
  expect(events.filter((event: { item: string; action: string }) => event.item === name && event.action === 'Record edited')).toHaveLength(2);
});

test('Fine-tuning preview does not replay when returning to Jobs after cancel or completion', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.goto('/?view=finetune');
  await page.getByRole('tab', { name: 'Jobs', exact: true }).click();
  const startPreview = async () => {
    await page.getByRole('button', { name: 'Simulate training' }).click();
    const trace = page.locator('.trace-mount.trace-finetune');
    await expect(trace).toBeVisible();
    await expect(trace.locator('.trace-head-state')).toHaveText('Running');
    return trace;
  };
  let trace = await startPreview();
  await trace.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(trace.locator('.trace-head-state')).toHaveText('Cancelled');
  await page.getByRole('tab', { name: 'Results', exact: true }).click();
  await page.getByRole('tab', { name: 'Jobs', exact: true }).click();
  await expect(page.locator('.trace-mount.trace-finetune')).toHaveCount(0);

  trace = await startPreview();
  await expect(trace.locator('.trace-head-state')).toHaveText('Completed', { timeout: 12_000 });
  await page.getByRole('tab', { name: 'Results', exact: true }).click();
  await page.getByRole('tab', { name: 'Jobs', exact: true }).click();
  await expect(page.locator('.trace-mount.trace-finetune')).toHaveCount(0);
});
