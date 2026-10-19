import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Communications Demo row workflows persist locally (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
      });
      await page.setViewportSize(viewport);
      await page.addInitScript(({ themeName }) => {
        localStorage.setItem('oa-second-data-mode', 'demo');
        localStorage.setItem('oa-second-theme', themeName);
      }, { themeName: theme });
      await page.goto('/?view=communications');

      const original = page.locator('.communications-demo-row').filter({ hasText: 'Updated release checklist' });
      await expect(original).toBeVisible();
      // Allow route-level observers to settle before user actions. Demo mode
      // must not make an incidental sync-history request from the right rail.
      await page.waitForTimeout(200);
      expect(apiRequests).toEqual([]);
      await original.getByRole('button', { name: 'Message options for Updated release checklist' }).click();
      const detail = page.getByRole('dialog', { name: 'Updated release checklist' });
      await expect(detail.getByText('Maya Chen', { exact: true })).toBeVisible();
      await expect(detail.getByText(/Please review the new acceptance items/)).toBeVisible();
      await page.keyboard.press('Escape');
      await expect(detail).toHaveCount(0);
      await expect(original.getByRole('button', { name: 'Message options for Updated release checklist' })).toBeFocused();
      await original.getByRole('button', { name: 'Message options for Updated release checklist' }).click();
      await detail.getByRole('button', { name: 'Reply' }).click();
      const reply = page.getByRole('dialog', { name: 'Compose example message' });
      await expect(reply.locator('input[name="recipient"]')).toHaveValue('team@demo.example');
      await expect(reply.locator('input[name="subject"]')).toHaveValue('Re: Updated release checklist');
      await reply.getByRole('button', { name: 'Cancel' }).click();

      await original.getByRole('button', { name: 'Message options for Updated release checklist' }).click();
      await page.getByRole('dialog', { name: 'Updated release checklist' }).getByRole('button', { name: 'Edit' }).click();
      const edit = page.getByRole('dialog', { name: 'Edit example message' });
      await edit.getByRole('textbox', { name: 'Subject' }).fill('Updated release checklist · revised');
      await edit.getByRole('textbox', { name: 'Sender' }).fill('Maya Chen');
      await edit.getByRole('textbox', { name: 'Message' }).fill('Please review the final acceptance items before Friday.');
      await edit.getByRole('button', { name: 'Save changes' }).click();
      const revised = page.locator('.communications-demo-row').filter({ hasText: 'Updated release checklist · revised' });
      await expect(revised).toBeVisible();

      await revised.getByRole('button', { name: 'Message options for Updated release checklist · revised' }).click();
      await page.getByRole('dialog', { name: 'Updated release checklist · revised' }).getByRole('button', { name: 'Archive' }).click();
      await expect(page.locator('.communications-demo-row').filter({ hasText: 'Updated release checklist · revised' })).toHaveCount(0);

      const second = page.locator('.communications-demo-row').filter({ hasText: 'Code review follow-up' });
      await second.getByRole('button', { name: 'More options for Code review follow-up' }).click();
      await page.getByRole('dialog', { name: 'Code review follow-up' }).getByRole('button', { name: 'Duplicate', exact: true }).click();
      await page.getByRole('tab', { name: 'Drafts' }).click();
      const savedDraft = page.getByRole('combobox', { name: 'Saved draft' });
      await expect(savedDraft).toBeVisible();
      const duplicateOption = savedDraft.locator('option').filter({ hasText: 'Code review follow-up (copy)' });
      await expect(duplicateOption).toHaveCount(1);
      await savedDraft.selectOption(await duplicateOption.getAttribute('value') ?? '');
      await expect(page.getByRole('heading', { name: 'Code review follow-up (copy)' })).toBeVisible();

      await page.getByRole('tab', { name: 'Inbox' }).click();
      await page.getByRole('button', { name: 'Message options for Code review follow-up' }).click();
      await page.getByRole('dialog', { name: 'Code review follow-up' }).getByRole('button', { name: 'Delete' }).click();
      const confirm = page.getByRole('dialog', { name: 'Delete “Code review follow-up”?' });
      await expect(confirm).toBeVisible();
      await confirm.getByRole('button', { name: 'Cancel' }).click();
      await expect(page.locator('.communications-demo-row').filter({ hasText: 'Code review follow-up' })).toBeVisible();
      await page.getByRole('dialog', { name: 'Code review follow-up' }).getByRole('button', { name: 'Delete' }).click();
      await page.getByRole('dialog', { name: 'Delete “Code review follow-up”?' }).getByRole('button', { name: 'Delete record' }).click();
      await expect(page.locator('.communications-demo-row').filter({ hasText: 'Code review follow-up' })).toHaveCount(0);

      await page.getByRole('button', { name: 'Compose message' }).click();
      const compose = page.getByRole('dialog', { name: 'Compose example message' });
      await compose.getByRole('button', { name: 'Save draft' }).click();
      await expect(compose.locator('input[name="recipient"]')).toHaveJSProperty('validity.valid', false);
      await compose.locator('input[name="recipient"]').fill('reviewer@demo.example');
      await compose.locator('input[name="subject"]').fill('Desktop acceptance follow-up');
      await compose.locator('textarea[name="body"]').fill('Please confirm the local Demo draft workflow.');
      await compose.getByRole('button', { name: 'Save draft' }).click();
      await expect(page.getByRole('status').filter({ hasText: 'Draft saved locally in demo mode' })).toBeVisible();
      await page.getByRole('tab', { name: 'Drafts' }).click();
      const newlySavedOption = page.getByRole('combobox', { name: 'Saved draft' }).locator('option').filter({ hasText: 'Desktop acceptance follow-up' });
      await expect(newlySavedOption).toHaveCount(1);
      await page.getByRole('combobox', { name: 'Saved draft' }).selectOption(await newlySavedOption.getAttribute('value') ?? '');
      await expect(page.getByRole('heading', { name: 'Desktop acceptance follow-up' })).toBeVisible();
      await page.reload();
      await page.getByRole('tab', { name: 'Drafts' }).click();
      const persistedOption = page.getByRole('combobox', { name: 'Saved draft' }).locator('option').filter({ hasText: 'Desktop acceptance follow-up' });
      await expect(persistedOption).toHaveCount(1);
      await page.getByRole('combobox', { name: 'Saved draft' }).selectOption(await persistedOption.getAttribute('value') ?? '');
      await page.getByRole('tab', { name: 'Inbox' }).click();
      await expect(page.locator('.communications-activity-card')).toContainText('Saved draft Desktop acceptance follow-up');
      const activityRail = page.locator('.communications-rail');
      await expect(activityRail.locator('.demo-metric').filter({ hasText: 'demo activities' }).locator('strong')).not.toHaveText('0');
      await expect(activityRail).toContainText('Saved draft Desktop acceptance follow-up — Saved locally · demo');
      expect(apiRequests).toEqual([]);
    });
  }
}
