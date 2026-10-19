# OpenAgent second frontend

## Latest verification — 2026-10-18

TypeScript check passed; 151/151 unit tests across 22 files passed; full Playwright passed 193/193 in 7.1 minutes; isolated production build passed; production-preview safety passed with 12 offline 503-stubbed requests, zero demo-fixture requests and zero page errors. The Calendar canonical-style regression passed four desktop viewport/theme cases. The fresh 23-screen matrix measures 84.94% strict RGBA; Calendar is 94.92%. This does not prove full visual acceptance or live Python integration. Milestone A remains open.

This is the React 19 + TypeScript + Vite frontend target in `frontend-second/`. The canonical approved HTML is `reference/frontend-final-approved.html`, pinned by SHA-256 `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`. The original `frontend/` remains the functional reference.

Deterministic demo fixtures and local demo actions remain isolated from live API requests. The fixture JSON is served only by the Vite development server and is not copied into production build output. Live mode uses existing OpenAgent API calls and reports unsupported backend capabilities honestly.

## Install and run

From the extracted `frontend-second/` directory:

```powershell
npm ci
npm run dev
```

The development server is at `http://127.0.0.1:5180`. Development starts in clearly labeled local demo mode; use the top-right mode control to switch to live mode. Live API requests proxy `/api/*` to `http://127.0.0.1:8765` by default. To connect to a backend, start it from the full repository with `python -m openagent --port 8765`.

`npm run build` inside this directory (or root `npm run build:web`) writes to the isolated `frontend-second/dist/` directory. It never cuts over the Python-served `web/` directory. A separate deployment decision is required before promoting a build. The original frontend can still be started with `npm run dev:web:legacy`.

Set `OPENAGENT_API_TARGET` in a local `.env` file to proxy to another backend host. For a separately hosted production build, set `VITE_OPENAGENT_API_ORIGIN` to the backend origin (for example `https://openagent.example`) and configure that backend's trusted origins for the frontend host. The default local proxy keeps browser API calls same-origin and avoids weakening backend CORS policy.

If backend authentication is enforced, enter an existing bearer token in **Settings → Advanced → OpenAgent API**. The token is held in this browser tab's `sessionStorage` and cleared when removed; it is never written to build-time configuration or local persistent storage. API read/write scopes are still enforced by the backend.

## Test and build

```powershell
npm run build
npm test
npm run test:e2e -- --workers=1
npm run capture:reference
npm run capture:approved-html
npm run compare:pixels
```

Latest verification for this handoff (2026-10-18): `npm run check` passed; **142/142 unit tests across 22 files**; full Playwright passed **154/154**; and `npm run build` passed to isolated `dist/` (existing Monaco large-chunk warning). Chat runtime E2E passed 48 cases at 1440×900 and 1280×800 in both themes. Communications Demo Inbox record workflows passed four cases covering details, reply prefill, edit, archive, duplicate draft, delete confirmation, compose validation/save, Escape focus recovery, local activity persistence, consistent event display in the right rail, and zero API calls at both desktop sizes/themes. Knowledge Demo create/search/detail/persistence/activity/reset passed and restores seeded data while clearing its local event history with zero API calls. Advanced Demo rail export/reset success and injected failure handling passed eight cases across desktop sizes/themes. The production-preview stale-Demo check passed against the latest build with 8 offline 503-stubbed API requests, 0 demo fixture requests and 0 page errors. Calendar, Personal Assistant Tasks and Security Demo focused checks also passed. These checks do not constitute real Python integration or exhaustive interaction-state/pixel parity. The safe build verifies `../web` is unchanged and demo fixtures are excluded.

Visual baseline settings are in `visual-baseline.config.json`. The latest available broad desktop matrix is `.qa/milestone-a-current-20261017/tab-states/manifest.json`: 344 pairs across 23 workspaces, 1440×900 and 1280×800, both themes; strict RGBA match is **84.48%**. The contextual Living UI evidence separately classifies intentionally changed idle states. These are historical diagnostic scores, not evidence of full fidelity after this dialog correction. Milestone A remains open; meaningful design and interaction gaps remain. See `VISUAL_EVIDENCE_INDEX.md`, `INTERACTION_STATE_COVERAGE_REPORT.md`, and `FRONTEND_BACKEND_MASTER_GAPS.md`.
