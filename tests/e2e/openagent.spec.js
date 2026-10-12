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

  await page.locator('[data-nav-group="work"]').click();
  await expect(page.locator('[data-nav-group="work"]')).toHaveAttribute("aria-expanded", "true");
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
  await page.locator('[data-developer-section="skills"]').click();
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

  await page.locator('[data-nav-group="data"]').click();
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

test("deep links keep navigation and browser history in sync", async ({ page }) => {
  await page.goto("/?view=research");
  await expect(page.locator('[data-view-panel="research"]')).toBeVisible();
  await expect(page.locator('[data-nav-group="work"]')).toHaveAttribute("aria-expanded", "true");

  await page.locator('[data-view-link="voice"]').click();
  await expect(page).toHaveURL(/view=voice/);
  await expect(page.locator('[data-view-panel="voice"]')).toBeVisible();

  await page.goBack();
  await expect(page.locator('[data-view-panel="research"]')).toBeVisible();
  await page.goForward();
  await expect(page.locator('[data-view-panel="voice"]')).toBeVisible();

  await page.goto("/?view=phd");
  await expect(page.locator('[data-view-panel="phd"]')).toBeVisible();
  await expect(page.locator('[data-nav-group="studios"]')).toHaveAttribute("aria-expanded", "true");
});

test("mobile chat stays inside the viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/?view=chat");
  await expect(page.locator("#chatLog")).toBeVisible();
  await expect(page.locator("#chatInput")).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
});

test("theme controls update and persist the page theme", async ({ page }) => {
  await page.goto("/");
  await page.locator('[data-theme-select="light"]').click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(page.locator('[data-theme-select="light"]')).toHaveAttribute("aria-pressed", "true");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.locator('[data-theme-select="dark"]').click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
});

test("chat routing controls are visible in Settings and local-only mode clears external access", async ({ page }) => {
  await page.goto("/?view=settings");
  await expect(page.locator("#settingChatProvider")).toBeVisible();
  await expect(page.locator("#settingChatStreaming")).toBeChecked();
  await expect(page.locator("#settingAllowExternal")).toBeDisabled();
  await page.locator("label:has(#settingLocalOnly)").click();
  await page.locator("label:has(#settingAllowExternal)").click();
  await expect(page.locator("#settingAllowExternal")).toBeChecked();
  await page.locator("label:has(#settingLocalOnly)").click();
  await expect(page.locator("#settingAllowExternal")).toBeDisabled();
  await expect(page.locator("#settingAllowExternal")).not.toBeChecked();
  await page.reload();
  await expect(page.locator("#settingAllowExternal")).not.toBeChecked();
});

test("command palette follows the real navigation and returns focus", async ({ page }) => {
  await page.goto("/");
  await page.locator("#projectInput").focus();
  await page.keyboard.press("Control+k");
  await expect(page.locator("#commandPalette")).toBeVisible();
  await expect(page.locator("#paletteInput")).toBeFocused();
  await page.locator("#paletteInput").fill("Research");
  await expect(page.locator("#paletteResults [role=option]")).toHaveCount(1);
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/view=research/);
  await expect(page.locator("#commandPalette")).toBeHidden();
  await expect(page.locator("#projectInput")).toBeFocused();

  await page.keyboard.press("Control+k");
  await page.locator("#paletteInput").fill("not-a-real-module");
  await expect(page.locator("#paletteResults")).toContainText("No matching commands");
  await page.keyboard.press("Escape");
  await expect(page.locator("#commandPalette")).toBeHidden();
});

test("model policy uses a readable result with optional raw data", async ({ page }) => {
  await page.goto("/?view=models");
  await expect(page.locator("#modelPolicyOutput > .result-fields")).toBeVisible({ timeout: 20000 });
  await expect(page.locator("#modelPolicyOutput .result-raw")).toBeVisible();
  await expect(page.locator("#modelPolicyOutput .result-raw pre")).toBeHidden();
  await page.locator("#modelPolicyOutput .result-raw summary").click();
  await expect(page.locator("#modelPolicyOutput .result-raw pre")).toBeVisible();
});

test("work areas fit tablet and mobile widths", async ({ page }) => {
  for (const width of [768, 390]) {
    await page.setViewportSize({ width, height: 900 });
    for (const view of ["dashboard", "research", "voice", "models", "science"]) {
      await page.goto(`/?view=${view}`);
      await expect(page.locator(`[data-view-panel="${view}"]`)).toBeVisible();
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      expect(overflow, `${view} at ${width}px`).toBeLessThanOrEqual(1);
    }
  }
});
