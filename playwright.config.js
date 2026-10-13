import { defineConfig } from "@playwright/test";
const port = Number(process.env.OPENAGENT_E2E_PORT || 8899);
const python = process.env.OPENAGENT_E2E_PYTHON || "python";
const baseConfig = {
  testDir: "./tests/e2e",
  timeout: 60000,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  use: { baseURL: `http://127.0.0.1:${port}`, trace: "retain-on-failure", screenshot: "only-on-failure" },
  webServer: {
    command: `"${python}" -m openagent --host 127.0.0.1 --port ${port} --data-dir .tmp-playwright`,
    url: `http://127.0.0.1:${port}/api/health`,
    reuseExistingServer: Boolean(process.env.OPENAGENT_E2E_REUSE),
    timeout: 120000,
  },
};

export default defineConfig({
  ...baseConfig,
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
