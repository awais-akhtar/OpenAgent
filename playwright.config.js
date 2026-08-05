const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "./tests/e2e",
  timeout: 60000,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:8765",
    trace: "retain-on-failure",
    screenshot: "only-on-failure"
  },
  webServer: {
    command: "python -m openagent --host 127.0.0.1 --port 8765 --data-dir .tmp-playwright",
    url: "http://127.0.0.1:8765/api/health",
    reuseExistingServer: false,
    timeout: 120000
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }]
});
