# Frontend transformation QA record

Date: 2026-10-13. This record separates successful checks from unverified release gates.

## Commands and outcomes

| Check | Result | Evidence |
|---|---|---|
| `npm run check:web` | PASS | TypeScript `tsc --noEmit` completed after the shell, chat, resource, agent/automation, Workspace and security/settings changes. |
| `npm run test:web` | PASS | Vitest: 16 files, 62 tests passed. Includes navigation, chat streaming/navigation race, dashboard, mobile workspace tabs, resizers, resource actions, settings/security summaries, specialist agent/workflow surfaces and workspace utilities. |
| `npm run build:web` | PASS | Latest Vite production build emitted the static React app to the existing `web/` runtime directory, including `index.html`, `app.js`, `styles.css`, current route chunks and updated service-worker fingerprint. |
| `.venv\\Scripts\\python.exe -m unittest tests.test_web_ui tests.test_route_contracts tests.test_production_proof_expansion` | PASS | 8 focused Python UI/static-route/production-proof tests passed. |
| `node --check tests/e2e/openagent.spec.js` | PASS | Updated Playwright test file parses. |
| `git diff --check` | PASS | No whitespace errors after cleanup. |
| Live browser route tour at `http://127.0.0.1:8899` | PASS (read-only route/render smoke) | After the latest build, navigated all 23 original route IDs through the Python-served UI; no API error or browser page failure was observed. Chat showed a single conversation title with its action menu, Workspace showed the file/editor/preview surface, and Settings, Assistants, Fine-tuning and Activity exposed live service data. This remains read-only smoke, not acceptance evidence for every mutation, stream or security negative case. Dashboard Activity uses `?view=dashboard&activity=1`; legacy `?view=dashboard` resolves to Chat. |
| Live Python API smoke | PASS | E2E agent reported `/api/health`, `/api/capabilities`, and project-scoped `/api/conversations` responses succeeded. |
| Full Playwright browser suite | BLOCKED | The Python API smoke test passed. Ten browser tests could not launch Chromium in this Windows environment (`spawn EPERM`); browser assertions, responsive widths and interaction tests are not verified by Playwright. |
| Screenshot matrix / reference scoring | PARTIAL | Earlier connected Chrome inspection covered chat home in 1440×900 light/dark, Workspace at 1440×900, and representative chat/workspace/settings/agents routes at 390×844 and 320×700; no horizontal overflow was observed. After the recent fixes, Workspace was inspected from the freshly built production bundle at the browser's current 615×604 viewport. Screenshots were not saved as workspace artifacts. The required viewport matrix, complete light/dark route matrix, 200% zoom, and 35-point rubric remain incomplete. |
| Full Python `unittest discover -s tests` | FAIL | Completed 1,396 tests: 32 failures, 193 errors, 84 skipped. Failures include unbound `stat` in untouched `openagent/adaptive_engine/expert_store.py`, Windows source-file identity/race assertions and segment ABI expectations. The outdated production-proof selector/language checks were aligned to the new React surface and its focused test passes; broader backend suite failures remain. |

## Integration details and gaps

- Frontend remains same-origin and the Python API implementation was not changed. Type check, frontend unit tests and focused Python UI/route tests pass.
- Browser-backed view smoke exercises live read paths only. It does not exercise every write, stream cancellation/reconnection, authentication denial, project isolation, file limit, service-worker update or dangerous-action confirmation.
- `/workspace` now has a file list/editor/preview/details layout and responsive Edit/Preview tabs; Workspace API tests cover project/query scope and update version fields. Live GETs returned valid empty lists for the default project, and the 1440×900 empty layout was visually inspected.
- `/agents` and `/automation` have dedicated real-data workspaces. The complete frontend suite has 62 passing tests. The approvals queue filters to pending requests; workflow builder models manual, scheduled and event triggers, ordered multiple steps, and guarded approval resolution. Live mutation acceptance remains unverified.
- All 23 routes map to a dedicated page family (chat, activity, specialist, personal-work or platform-work surfaces); several reuse compact inventory/inspector/action-dialog patterns. Workspace panes support pointer/keyboard resizing and mobile editor/preview switching.
- Security and Settings no longer show raw JSON summaries; values are allowlisted and sensitive identifiers, hashes, secret names, paths, and nested policy internals are omitted. Some API-backed forms/actions remain because these operations exist in the backend.
- Reference reuse audit found no AionUI source/assets/logo imports in `frontend/src`. Dependency SBOM/license scan, complete asset/trademark scan, About attribution UI and distribution packaging checks remain incomplete.

## Final status

The current implementation is materially redesigned and the verified checks above pass, but the supplied acceptance matrix is not fully green. In particular, Playwright/browser interaction, required visual screenshots/scoring, full Python suite, service-worker upgrade, and multiple dossier-level specialist acceptance flows remain open. Do not label the transformation 100% complete until those gates are addressed.
