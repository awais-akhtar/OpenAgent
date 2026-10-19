# Canonical Desktop Interaction-State Inventory

**Canonical source:** `frontend final.html`, SHA-256 `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1` (verified; unchanged).  
**Scope:** inventory discovery used Chrome 155.0.8059.40; the latest broad paired screenshots use Chrome 155.0.8059.80, DPR 1, 1440×900 and 1280×800, light and dark themes.  
**Status:** This is a sampled inventory and capture report, not proof of complete Milestone A acceptance.

## Current evidence

- All 23 route DOMs were inventoried from the canonical HTML and React implementation. The audit records 1,072 unique visible trigger rows, visible forms/fields, topbar controls, tab labels and state signals.
- Latest default-route full-screen comparison (captured 2026-10-18): **92 matched pairs** across 23 routes × 1440×900 and 1280×800 × light/dark, **84.88%** strict RGBA. React capture reports zero route/layout/runtime issues. Evidence: `.qa/desktop-current-20261018/visual-match.json` and full-size pairs. The aggregate includes intentional idle-state differences where canonical sample traces are hidden in React by user direction; this is not active-state or complete interaction acceptance.
- Latest all-tab comparison (captured 2026-10-17): **344 matched pairs** (328 visible-tab states plus 16 no-tab initial route states), **84.48%** strict RGBA at the same desktop sizes/themes; no unmatched tabs, browser errors or Demo API requests. Evidence: `.qa/milestone-a-current-20261017/tab-states/manifest.json`. This also includes intentional idle trace differences.
- The earlier 328-pair / **91.94%** matrix is historical and superseded by later implementation changes: `.qa/desktop-acceptance/continued-milestone-a/tab-states-final-current/manifest.json`.
- The representative interaction comparison covers 140 full-size pairs at a historical strict mean of **93.44%**, evidence `.qa/desktop-acceptance/interactions/manifest.json`. A later current Agents lifecycle capture has 12 pairs at **87.38%** in `.qa/agent-current-20261018/manifest.json`; the earlier 91.03% set in `.qa/desktop-acceptance/milestone-a-resume/agents-run-final/manifest.json` is historical. The Assistant goal dialog has separate evidence at `.qa/desktop-acceptance/assistant-goal-dialog-agent-current/manifest.json`. The Vault interaction set has **56 matched pairs**, 14 states, **85.05%** strict full-frame mean; manifest `.qa/desktop-acceptance/continued-milestone-a/vault-interaction-final/manifest.json`.
- A separate earlier initial-route comparison covers 92 full-size pairs and averaged **92.76%**. It is not the latest broad matrix. Evidence and route scores: `visual-match-v14.json`, `approved-final-v14/`, and `react-final-v14/`.
- `INTERACTION_STATE_CATALOG.md` is the readable route/state map. Its machine-readable data is `.qa/desktop-acceptance/continued-milestone-a/interaction-state-catalog-after-fixes.json`; the source DOM inventory is `.qa/desktop-acceptance/milestone-a-resume/interaction-inventory-final-v13.json`.

## Captured interaction states

The representative interaction matrix includes Voice idle/recording/saved; CodeQ initial, Preview, Terminal, Git, expanded diff, selected preview size, expanded trace/step, running and paused; Agents Preview activity running/paused/completed; and running/paused/completed timeline states for Chat, Research, Models, Automations, Scientific AI, Fine-tuning and Web Scraper. The separate Vault matrix captures create, keyboard focus, invalid submit, populated form, detail, edit, approve, deny, delete confirmation, activity history, Audit create/invalid create/detail and active Run example. All Vault states have full-size pairs at both desktop sizes and themes, using fresh deterministic fixtures for each named state.

The screenshots use the same pinned HTML hash, fixture hashes, Chrome build, DPR, fixed clock and viewport/theme pairs. The CodeQ states retain the actual Monaco editor. The Agents states use the canonical Preview activity action and its inline execution rows. The canonical HTML is checked before and after capture and was not changed.

## Explicitly unverified

The catalog discovers visible controls and state signals, but it does not activate every trigger. The paired screenshots do **not** exhaustively verify every dialog, nested menu, form submission, validation error, loading/submitting/success/failure state, notification state, detail view, approval, or execution error/cancellation/recovery state on every route. Keyboard focus behavior and every responsive edge case are not fully covered either. These remain open acceptance work; presence of a trigger in the inventory is not evidence that its resulting state works or matches.

See [`FRONTEND_BACKEND_MASTER_GAPS.md`](./FRONTEND_BACKEND_MASTER_GAPS.md) for the current Milestone A status, remaining differences, and backend gap map. Backend route wiring in that file is based on source/contracts and tests as labeled; no live Python integration run was done during this frontend continuation.

## Vault Access/Audit activation checklist (desktop demo)

The control inventory is being supplemented with activation tests because visible controls alone do not establish working behavior. The new focused Playwright cases are in `tests/e2e/vault-demo-activation.spec.ts`; they run at 1440×900 and 1280×800 in both themes, explicitly select demo mode, and fail if a request is made to an `/api/` path.

| Control/state | Activation evidence in `tests/e2e/vault-demo-activation.spec.ts` | Current verification status |
| --- | --- | --- |
| Access create, required Name validation, successful local create | Assert submit remains enabled with empty Name, native `required` blocks saving, then create a record and observe it in the list | **Verified** in all four desktop viewport/theme combinations |
| Access edit | Change the description, save, and assert the same record reflects it | **Verified** in all four combinations |
| Access details, Deny | Open a seeded pending request, check canonical action order, deny it, and assert the row changes state | **Verified** in all four combinations |
| Access duplicate | Use the row action dialog and assert a local `(copy)` record appears | **Verified** in all four combinations |
| Access delete confirmation | Open confirmation, cancel and confirm retention; delete from details and assert removal | **Verified** in all four combinations |
| Audit detail / action menu | Open seeded row actions, press Escape and assert focus returns to its trigger; verify canonical detail action order | **Verified** in all four combinations |
| Audit Run example | Start the seeded record, assert the workspace row transitions Running → Completed, inspect the structured activity, and require zero backend requests | **Verified** in all four combinations; simulation remains local |
| Audit create and search | Create a synthetic note and filter it back from the list | **Verified** in all four combinations |
| Structured activity | Verify activity shows `vault · status · HH:MM` fields and the green success artwork | **Verified** in all four combinations |
| Local/offline boundary | Track requests across each scenario and require zero `/api/` calls | **Verified** in all four combinations |

Final verification after the latest Vault component and CSS edits: `npm run check` passed; `npm run test -- --maxWorkers=1` passed **130/130 tests across 21 files**; `npm run test:e2e -- --workers=1` passed **51/51 tests**; the focused Access/Audit suite passed **10/10 tests**; `npm run build` passed to isolated `frontend-second/dist`. The 8 activation scenarios are parameterized across both desktop sizes and themes; all assert zero `/api/` requests. The 56 screenshot pairs verify visual state capture, not that every item in the wider inventory has been exercised.

These cases do not prove screenshot parity for each action state. The refreshed Vault set reports an **85.05%** strict full-frame mean across 56 pairs; at 1440×900 light the Access create/detail/edit pairs score 84.48% / 83.31% / 84.57%, and Audit run-active scores 96.00%. Full-frame scores include the application behind the overlay and are not region-only dialog scores. The measured create/edit dialog box is 347.22px tall at 1440×900 and aligns with the canonical HTML; title, description, input, textarea and action-row bounding boxes also matched in the paired geometry inspection. Pixel and content differences remain. The capture manifest is current for the corrected source, uses the in-repository canonical copy and shows zero browser errors and zero demo API calls.

The current inventory still has substantial unactivated controls outside Vault: row/nested menus, destructive confirmations, settings forms, approvals, notifications, error/retry states, and execution failure/cancel/recovery controls across the remaining routes. The 1,072-trigger source inventory is a discovery list; it is not a 1,072-interaction test result.
