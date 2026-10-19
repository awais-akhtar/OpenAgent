import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Security Demo record actions stay local and persist (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
      });
      await page.setViewportSize(viewport);
      await page.addInitScript(({ mode, themeName }) => {
        localStorage.setItem('oa-second-data-mode', mode);
        localStorage.setItem('oa-second-theme', themeName);
      }, { mode: 'demo', themeName: theme });
      await page.goto('/?view=security');
      await page.getByRole('tab', { name: 'Access' }).click();

      const original = page.locator('.security-demo-record').filter({ hasText: 'Workspace administrator' });
      await original.getByRole('button', { name: 'More options for Workspace administrator' }).click();
      await page.getByRole('dialog').getByRole('button', { name: 'Edit record' }).click();
      const editor = page.getByRole('dialog', { name: 'Edit policy' });
      await expect(editor).toBeVisible();
      await editor.getByRole('textbox', { name: 'Name' }).fill('Workspace admin review');
      await editor.getByRole('button', { name: 'Save changes' }).click();
      const edited = page.locator('.security-demo-record').filter({ hasText: 'Workspace admin review' });
      await expect(edited).toBeVisible();

      await edited.getByRole('button', { name: 'More options for Workspace admin review' }).click();
      await page.getByRole('dialog').getByRole('button', { name: 'Duplicate' }).click();
      const duplicate = page.locator('.security-demo-record').filter({ hasText: 'Workspace admin review (copy)' });
      await expect(duplicate).toBeVisible();

      await edited.getByRole('button', { name: 'Configure Workspace admin review', exact: true }).click();
      const details = page.getByRole('dialog', { name: 'Workspace admin review' });
      await expect(details.getByText('Enabled', { exact: true })).toBeVisible();
      await details.getByRole('button', { name: 'Disable' }).click();
      await expect(edited.getByText('Disabled', { exact: true })).toBeVisible();

      await duplicate.getByRole('button', { name: 'More options for Workspace admin review (copy)' }).click();
      await page.getByRole('dialog').getByRole('button', { name: 'Delete', exact: true }).click();
      const confirmation = page.getByRole('dialog', { name: 'Delete “Workspace admin review (copy)”?' });
      await expect(confirmation).toBeVisible();
      await confirmation.getByRole('button', { name: 'Delete record' }).click();
      await expect(page.locator('.security-demo-record').filter({ hasText: 'Workspace admin review (copy)' })).toHaveCount(0);

      await page.reload();
      await page.getByRole('tab', { name: 'Access' }).click();
      const persisted = page.locator('.security-demo-record').filter({ hasText: 'Workspace admin review' });
      await expect(persisted).toBeVisible();
      await expect(persisted.getByText('Disabled', { exact: true })).toBeVisible();
      await expect(page.locator('.security-demo-record').filter({ hasText: 'Workspace admin review (copy)' })).toHaveCount(0);

      await page.getByRole('tab', { name: 'Sessions' }).click();
      const session = page.locator('.security-demo-record').filter({ hasText: 'Safari · macOS' });
      await session.getByRole('button', { name: 'More options for Safari · macOS' }).click();
      await page.getByRole('dialog').getByRole('button', { name: 'Revoke session' }).click();
      await expect(session.getByText('Revoked', { exact: true })).toBeVisible();

      await page.getByRole('tab', { name: 'Audit' }).click();
      const audit = page.locator('.security-demo-record').filter({ hasText: 'Approval granted' });
      await audit.getByRole('button', { name: 'More options for Approval granted' }).click();
      await page.getByRole('dialog').getByRole('button', { name: 'Run / test' }).click();
      await expect(page.getByText('Example check simulated locally. No security service was contacted.', { exact: true })).toBeVisible();
      await expect(audit.getByText('Simulated locally', { exact: true })).toBeVisible();
      expect(apiRequests).toEqual([]);
    });
  }
}
