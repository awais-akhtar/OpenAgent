const { test, expect } = require("@playwright/test");

test("desktop workspace loads and core lifecycle controls are usable", async ({ page }) => {
  const consoleErrors = [];
  page.on("console", message => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", error => consoleErrors.push(error.message));

  await page.goto("/");
  await expect(page.locator("#appShell")).toBeVisible();
  await expect(page.locator("#healthBadge")).not.toContainText("Offline");

  await page.locator('[data-view-link="developer"]').click();
  await expect(page.locator("#skillEditorName")).toBeVisible();
  await page.locator("#skillEditorName").fill("Browser proof skill");
  await page.locator("#skillEditorMarkdown").fill("# Browser proof skill\n\nRuns locally without network access.");
  await page.locator("#saveSkill").click();
  await expect(page.locator("#skillEditorList")).toContainText("Browser proof skill");

  await page.locator('[data-view-link="research"]').click();
  await expect(page.locator("#startResearch")).toBeEnabled();
  await expect(page.locator("#researchLibrary")).toBeVisible();

  await page.locator('[data-view-link="security"]').click();
  await expect(page.locator("#identityAdminCards")).toBeVisible();
  await expect(page.locator("#settingLocalOnly")).toBeAttached();

  expect(consoleErrors).toEqual([]);
});

test("local API exposes lifecycle and privacy contracts", async ({ request }) => {
  const health = await request.get("/api/health");
  expect(health.ok()).toBeTruthy();
  expect((await health.json()).status).toBe("ok");

  const modules = await request.get("/api/modules/health?project=default");
  expect(modules.ok()).toBeTruthy();
  const modulePayload = await modules.json();
  expect(modulePayload.privacy_boundary).toContain("local");

  const research = await request.get("/api/research/sessions?project=default");
  expect(research.ok()).toBeTruthy();
  expect(Array.isArray((await research.json()).sessions)).toBeTruthy();
});
