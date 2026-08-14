const { test, expect } = require("@playwright/test");

test("desktop workspace loads and core lifecycle controls are usable", async ({ page, request }) => {
  const consoleErrors = [];
  page.on("console", message => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", error => consoleErrors.push(error.message));

  await page.goto("/");
  await expect(page.locator("#appShell")).toBeVisible();
  await expect(page.locator("#metricGrid")).toContainText("CPU", { timeout: 20000 });
  await expect(page.locator("#metricGrid")).toContainText("RAM");
  await expect(page.locator("#metricGrid")).toContainText("Disk");

  await page.locator('[data-view-link="codebase"]').click();
  await expect(page.locator("#codebaseWorkspace")).toBeVisible();
  await expect(page.locator("#codebaseSourcePath")).toBeAttached();
  await expect(page.locator("#codebaseBoundary")).toContainText(/workspace/i);
  await expect(page.locator("#codebaseFolderPicker")).toBeVisible();
  await expect(page.locator("#codebaseOpenTabs")).toBeAttached();
  await expect(page.locator("#codebaseCopilotFeed")).toContainText("Work with your repository");
  await expect(page.locator("#codebaseGoal")).toHaveAttribute("placeholder", /build, fix, explain, or review/i);
  await expect(page.locator("#startCodebaseRun")).toBeAttached();
  await page.locator('[data-view-link="chat"]').click();
  await expect(page.locator("#chatLog")).toBeVisible();

  await page.locator('[data-view-link="developer"]').click();
  await expect(page.locator("#skillEditorName")).toBeVisible();
  await page.locator("#skillEditorName").fill("Browser proof skill");
  await page.locator("#skillEditorMarkdown").fill("# Browser proof skill\n\nRuns locally without network access.");
  await page.locator("#saveSkill").click();
  await expect(page.locator("#skillEditorList")).toContainText("Browser proof skill");

  await page.locator('[data-view-link="chat"]').click();
  await expect(page.locator("#chatVoiceButton")).toBeVisible();
  await expect(page.locator("#chatAttachButton")).toBeVisible();
  await expect(page.locator("#chatInput")).toHaveAttribute("placeholder", /research/i);

  const created = await request.post("/api/chat", { data: { message: "Conversation menu browser proof", project: "default", provider: "local" } });
  expect(created.ok()).toBeTruthy();
  await page.locator('[data-view-link="chat"]').click();
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await page.waitForTimeout(300);
  await page.locator("#newConversationButton").click();
  await page.waitForTimeout(300);
  const menuTrigger = page.locator("[data-conversation-menu]").first();
  await expect(menuTrigger).toBeVisible();
  await menuTrigger.click();
  await expect(page.locator("#conversationActionMenu")).toBeVisible();
  await expect(page.locator("#conversationActionMenu")).toContainText("Archive");
  await page.locator("#chatInput").click();
  await expect(page.locator("#conversationActionMenu")).toBeHidden();

  await page.locator("#chatVoiceButton").click();
  await expect(page.locator("#chatVoiceMode")).toBeVisible();
  await page.locator("#endChatVoiceMode").click();
  await expect(page.locator("#chatVoiceMode")).toBeHidden();

  await page.locator('[data-nav-group="platform"]').click();
  await page.locator('[data-view-link="security"]').click();
  await expect(page.locator("#identityStatusCards")).toBeVisible();
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

  const codebase = await request.get("/api/codebase/projects?project=default");
  expect(codebase.ok()).toBeTruthy();
  expect(Array.isArray((await codebase.json()).projects)).toBeTruthy();
});
