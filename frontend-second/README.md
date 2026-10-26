# OpenAgent second frontend

## Latest verification — 2026-10-25

`npm run check` passed; Vitest passed **196/196 across 25 files**; complete Playwright passed **241/241 in 9.2 minutes** (single worker, zero retries); and the isolated production build passed. The current real React-to-Python smoke passed **43 workflows / 429 API responses / zero Demo fixture requests**, with authentication enforced, temporary data, synthetic credentials and external providers disabled. The smoke test's Calendar fixture was moved to a visible 10:00 time after a prior test-data timeout at an off-grid hour; the final run passed. Build output stayed in `dist/`, `../web` was unchanged, and `public/demo` fixtures were excluded. Playwright Live API cases are intercepted/offline; the separate smoke is the real-Python integration evidence. Product completion remains open.

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

Historical 2026-10-18 checks: 142/142 unit and 154/154 Playwright cases passed. Newer verification is summarized above and tracked in `INTERACTION_STATE_COVERAGE_REPORT.md`. The real Python smoke covers selected operations only; it does not verify external providers or every Live interaction. The safe build verifies `../web` is unchanged and demo fixtures are excluded.

Visual baseline settings are in `visual-baseline.config.json`. The latest available broad desktop matrix is `.qa/milestone-a-current-20261017/tab-states/manifest.json`: 344 pairs across 23 workspaces, 1440×900 and 1280×800, both themes; strict RGBA match is **84.48%**. The contextual Living UI evidence separately classifies intentionally changed idle states. These are historical diagnostic scores, not evidence of full fidelity after this dialog correction. Milestone A remains open; meaningful design and interaction gaps remain. See `VISUAL_EVIDENCE_INDEX.md`, `INTERACTION_STATE_COVERAGE_REPORT.md`, and `FRONTEND_BACKEND_MASTER_GAPS.md`.
