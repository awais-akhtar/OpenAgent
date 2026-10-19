# OpenAgent desktop white-screen startup fix — 19 October 2026

## Status and scope

This package contains **actual changed source files**, not a full-repository replacement. It is based on the uploaded Batch 2B source snapshot; Codex must merge these five files into the CURRENT working repository and preserve any newer changes.

The uploaded Batch 2B source archive has **zero files in `web/`**. Its Python desktop launcher serves `web/index.html` by default, while the newer React UI builds independently to `frontend-second/dist/`. The missing production-served static frontend is a confirmed packaging defect in that archive, not definitive proof that the current Windows machine has the same defect.

Changes:
1. `openagent/workbench.py`: fail fast with a clear missing/unbuilt asset message; accept an **explicit, local, opt-in** `--ui-root` pointing to a built frontend; start the local HTTP server and verify health/HTML/JS/CSS before opening Chrome/Edge; prevent a cold-server white window.
2. `openagent/desktop.py`: pass through `--ui-root` and display a Windows error dialog if the app cannot start (installed shortcuts may use `pythonw`).
3. `frontend-second/index.html`: visible loading placeholder and a helpful error message if React never mounts.
4. `frontend-second/src/main.tsx`: crash boundary to render a readable interface error rather than a completely white viewport.
5. `tests/test_desktop_ui_startup.py`: six regression tests; the HTTP startup test uses a synthetic local webpage, NOT the complete React application.

The existing `web/` content is **not overwritten**. The approved canonical HTML reference stays unchanged. Remote access is out of scope.

## Local test on Windows (run from the current repository root)

Close any existing OpenAgent desktop window and server. First check where the installed app expects the UI:

```powershell
.\.venv\Scripts\python.exe -m openagent.workbench --status-json
Test-Path .\web\index.html
Test-Path .\web\app.js
```

Build the React UI into **isolated** `frontend-second/dist`:

```powershell
cd frontend-second
npm ci
npm run check
npm run build
cd ..
```

Run it inside the real local desktop shell *without changing the production `web/` folder*:

```powershell
.\.venv\Scripts\python.exe -m openagent.desktop --ui-root frontend-second\dist
```

If source virtualenv does not exist, use the Python executable recorded by the Windows installer. Keep the server bound to `127.0.0.1`. The above `--ui-root` flag is newly added by this patch: merge the source files before trying it.

If this test works, inspect the page, developer-console errors, Chat/CodeQ and reload in the Chromium app window. Only after an explicit owner approval may Codex promote the new React frontend into the regular production-serving location or modify the installed shortcut/default.

## Verification in the patch creation environment

- Python syntax (`py_compile`) for the modified Python files: PASS
- Six new `test_desktop_ui_startup` unittest cases: PASS
- Real Python localhost HTTP server with synthetic static UI, `GET /`, `/app.js`, `/styles.css`, `/api/health`: all HTTP 200
- The actual full Windows launcher and installed Edge/Chrome window: NOT TESTED here
- The full React TypeScript build and Playwright suite: NOT TESTED here (Node dependencies unavailable in this environment)

## What Codex must do immediately

1. Merge this targeted patch into the current working repository, not the older 2B snapshot wholesale.
2. Run on Windows; reproduce original shortcut blank screen; check status-json, current served index and network/console errors.
3. Build `frontend-second/dist`, use `--ui-root` for an isolated local desktop launch; fix any JS/runtime failure; ensure homepage renders visibly and all static assets return 200.
4. Validate that the installed launcher uses the tested built UI **only after owner approval for production frontend cutover**; preserve backup and rollback for `web/` and all user data.
5. Run `npm run check`, focused frontend tests, real Python integration, desktop Playwright smoke, and `npm run build`. Report exact PASS/FAIL; don't claim success from unrun tests.
6. Once desktop launch works, complete existing core features by implement → test → close. No new audit or architecture redesign.
