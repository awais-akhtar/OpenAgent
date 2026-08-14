# Browser Automation

OpenAgent has two deliberately separate browser-related features.

## Offline Form Simulation

`POST /api/production/automation/browser` works with HTML supplied in the request. It can inspect forms, fill fields in that local HTML, and return the resulting markup. It does not start a browser, navigate to a URL, or make a network request. The desktop workbench labels this action **HTML Form Simulation**.

## Optional Playwright Control

The `openagent.tools.browser.BrowserRuntime` uses Python Playwright when the optional dependency and Chromium are installed:

```bash
pip install "openagent[browser]"
playwright install chromium
```

In the desktop workbench, open **Automation**, enter a loopback URL, and select **Run Browser Test**. Loopback addresses work without additional approval. External URLs need the **Approve external URL** checkbox, and the action is recorded in the local audit trail.

The browser runtime can navigate, click, fill, press keys, collect page and console errors, and save a screenshot in the configured OpenAgent data directory. It does not create public tunnels or bypass OpenAgent's external-boundary policy.

The coding-agent surface also exposes `browser_automate` with the same boundary: loopback navigation is sandboxed, and external navigation requires explicit approval.