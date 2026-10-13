# Frontend transformation QA record

Date: 2026-10-13. This record separates successful checks from unverified release gates.

## Commands and outcomes

| Check | Result | Evidence |
|---|---|---|
| `npm run check:web` | PASS | TypeScript `tsc --noEmit` completed after the shell, chat, resource, agent/automation, Workspace and security/settings changes. |
| `npm run test:web` | PASS | Final Vitest: 16 files, 67 tests passed. Includes navigation, chat streaming/navigation race, dashboard, mobile Workspace tabs, resizers, resource actions, Settings/Security summaries, specialist agent/workflow surfaces and workspace utilities. |
| `npm run build:web` | PASS | Latest Vite production build emitted the static React app to the existing `web/` runtime directory, including `index.html`, `app.js`, `styles.css`, current route chunks and updated service-worker fingerprint. |
| `.venv\\Scripts\\python.exe -m unittest tests.test_route_contracts tests.test_production_proof_expansion tests.test_security_acceptance tests.test_auth tests.test_commercial_security tests.test_chat_artifacts tests.test_chat_routes tests.test_web_ui tests.test_desktop` | PASS | Final focused Python route/proof/auth/security/chat/UI/desktop set: 58 tests passed. |
| `node --check tests/e2e/openagent.spec.js` | PASS | Updated Playwright test file parses. |
| `git diff --check` | PASS | No whitespace errors after cleanup. |
| Screenshot capture, `.tmp-ui-evidence-aion/manifest.json` | PASS for render coverage; partial visual acceptance | Final capture contains all 22 retained views in light/dark themes at desktop 1440×900 and mobile 390×844 (88 screenshots); manifest reports no missing headings or page errors. It uses mocked API fixtures, so it verifies rendering, not live backend actions. Workspace has focused 390px/320px responsive Playwright assertions. Final Chat home geometry was compared with the supplied reference; layout is close, not pixel-identical. |
| Live browser route tour at `http://127.0.0.1:8899` | Historical read-only route/render smoke | The earlier build was navigated through the pre-retirement route IDs. PhD Application Studio has since been removed at the user's request; its deep link falls back to Chat. The historical tour is not a current full-route interaction test. Dashboard Activity uses `?view=dashboard&activity=1`; legacy `?view=dashboard` resolves to Chat. |
| Live Python API smoke | PASS | E2E agent reported `/api/health`, `/api/capabilities`, and project-scoped `/api/conversations` responses succeeded. |
| Playwright E2E suite, fresh Python server on port 8906 | PASS | Final run passed all 13 tests, including the 22-view deep-link tour/PhD fallback, chat send/provider/checked-agent flows, conversation history/archive, command palette, themes, responsive routes, mobile Workspace, safety-gated representative actions, and live Python health/capability/conversation contracts. A pre-run on port 8904 exposed the stale mock transcript noted above; that fixture was corrected before the passing run. |
| Screenshot matrix / reference scoring | PARTIAL | All 22 active views were captured at 1440×900 and 390×844 in light/dark. The Playwright suite additionally covers representative routes down to 320px. The evidence does not include 768×1024, 320px screenshots for every route, 200% zoom, full keyboard/accessibility scoring or the supplied full visual rubric. |
| Full Python `unittest discover -s tests` | FAIL | Latest agent discovery on a changing concurrent tree reported 1,392 tests: 16 failures, 46 errors, 84 skipped. Compact report identified adaptive family-conformance/registry issues, expert-runtime file identity/races, low-bit and segment ABI cases, plus optional missing-Torch vision errors; complete IDs were truncated. A focused final set of 58 route/proof/auth/security/chat/UI/desktop tests passed. Do not treat full Python backend verification as green. |

## Integration details and gaps

- Frontend remains same-origin and the Python API implementation was not changed. Type check, frontend unit tests and focused Python UI/route tests pass.
- Browser-backed view smoke exercises live read paths only. It does not exercise every write, stream cancellation/reconnection, authentication denial, project isolation, file limit, service-worker update or dangerous-action confirmation.
- `/workspace` now has a file list/editor/preview/details layout and responsive Edit/Preview tabs; Workspace API tests cover project/query scope and update version fields. Live GETs returned valid empty lists for the default project, and the 1440×900 empty layout was visually inspected.
- `/agents` and `/automation` have dedicated real-data workspaces. The complete frontend suite has 62 passing tests. The approvals queue filters to pending requests; workflow builder models manual, scheduled and event triggers, ordered multiple steps, and guarded approval resolution. Live mutation acceptance remains unverified.
- The active navigation contains 22 view IDs; PhD/Application Studio was retired at the user's request. Destinations map to chat, activity, specialist, personal-work or platform-work page families. Several surfaces reuse compact inventory/inspector/action-dialog patterns. Workspace panes support pointer/keyboard resizing and mobile editor/preview switching.
- Security and Settings no longer show raw JSON summaries; values are allowlisted and sensitive identifiers, hashes, secret names, paths, and nested policy internals are omitted. Some API-backed forms/actions remain because these operations exist in the backend.
- Reference reuse audit found no AionUI source/assets/logo imports in `frontend/src`. Dependency SBOM/license scan, complete asset/trademark scan, About attribution UI and distribution packaging checks remain incomplete.

## Final status

The current implementation is materially redesigned, and the frontend check, 67-test suite, production build, focused 58-test Python set and 22-view screenshot capture pass. The supplied acceptance matrix is not fully green: full Python discovery has failures, and live backend mutation/stream/auth/isolation, 200% zoom/accessibility, dependency-license review, service-worker upgrade, and some dossier-level specialist flows remain open. Do not label the transformation 100% complete until those gates are addressed.
