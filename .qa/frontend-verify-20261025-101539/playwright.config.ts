import { defineConfig, devices } from '@playwright/test';
import { existsSync } from 'node:fs';

const systemChrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const baseURL = process.env.OPENAGENT_UI_ORIGIN ?? 'http://127.0.0.1:5180';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  // Keep CI parallelism bounded on smaller runners; browser canvases and
  // local timers otherwise become load-sensitive across the full screen suite.
  workers: 4,
  reporter: 'list',
  use: {
    baseURL,
    ...devices['Desktop Chrome'],
    browserName: 'chromium',
    // The existing end-to-end suite verifies authenticated/live API adapters;
    // individual demo coverage can explicitly select demo mode in the test.
    storageState: {
      cookies: [],
      origins: [{ origin: baseURL, localStorage: [{ name: 'oa-second-data-mode', value: 'live' }] }],
    },
    // Prefer the machine's installed Chrome when Playwright's bundled browser
    // has not been installed. This keeps the isolated frontend runnable offline.
    launchOptions: existsSync(systemChrome) ? { executablePath: systemChrome } : undefined,
  },
  webServer: process.env.OPENAGENT_UI_ORIGIN ? undefined : {
    // Launch Vite directly. On Windows, `npm run dev` leaves an npm wrapper
    // process between Playwright and Vite; when the suite ends that wrapper
    // can keep the worker process alive after Chromium has already closed.
    command: 'node ./node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5180 --strictPort',
    url: 'http://127.0.0.1:5180',
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
