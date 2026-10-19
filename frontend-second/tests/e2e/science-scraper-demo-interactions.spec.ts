import { expect, test } from '@playwright/test';

test('Scientific AI and scraper tabs use local fixtures and support record interactions in demo mode', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));

  await page.goto('/?view=science');
  await expect(page.getByRole('heading', { name: 'From hypothesis to results' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Workspace overview' })).toBeVisible();
  await expect(page.getByText('Model reliability study', { exact: true })).toBeVisible();
  await page.getByRole('tab', { name: 'Experiments' }).click();
  await expect(page.getByRole('heading', { name: 'Run an experiment' })).toBeVisible();
  await expect(page.getByLabel('Experiment name')).toHaveValue('Calibration comparison');
  await expect(page.getByLabel('Dataset')).toHaveValue('Validation sample v2');
  await page.getByRole('button', { name: 'Run demo experiment' }).click();
  await expect(page.getByText('Demo experiment saved locally. No scientific service ran.')).toBeVisible();
  await expect(page.getByText('Calibration comparison', { exact: true })).toBeVisible();
  await page.getByRole('tab', { name: 'Reports' }).click();
  await expect(page.getByText('Calibration comparison report', { exact: true })).toBeVisible();
  await page.getByRole('tab', { name: 'Data' }).click();
  await expect(page.getByRole('heading', { name: 'From hypothesis to results' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Datasets · demo records' })).toBeVisible();
  await expect(page.getByText('Validation sample v2', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  await page.getByRole('textbox', { name: 'Name' }).fill('Desktop fixture dataset');
  await page.getByRole('textbox', { name: 'Description' }).fill('Created from the local demo workflow.');
  await page.getByRole('button', { name: 'Create item' }).click();
  await expect(page.getByText('Desktop fixture dataset', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Edit Desktop fixture dataset' }).click();
  await page.getByRole('textbox', { name: 'Description' }).fill('Edited locally.');
  await page.getByRole('button', { name: 'Save changes' }).click();
  await expect(page.getByText('Edited locally.', { exact: true })).toBeVisible();

  await page.goto('/?view=scraper');
  await expect(page.getByLabel('Extraction type')).toHaveValue('Text and metadata');
  await page.getByLabel('Extraction type').selectOption('Structured records');
  await page.getByLabel('Page limit').selectOption('10 pages');
  await page.getByLabel('Page URL').fill('https://docs.example.org/demo-only');
  await page.getByLabel('I confirm I have permission to extract this example content.').check();
  await page.getByRole('button', { name: 'Start demo extraction' }).click();
  await expect(page.getByText('Local extraction preview saved. No network request was sent.')).toBeVisible();
  await expect(page.getByText('https://docs.example.org/demo-only', { exact: true })).toBeVisible();
  await page.getByRole('tab', { name: 'Results' }).click();
  await expect(page.getByText('https://docs.example.org/demo-only · JSON preview', { exact: true })).toBeVisible();
  await expect(page.getByText('Structured web extraction', { exact: true })).toBeVisible();
  await expect(page.getByText('Documentation capture · JSON', { exact: true })).toBeVisible();
  expect(apiRequests).toEqual([]);
});

