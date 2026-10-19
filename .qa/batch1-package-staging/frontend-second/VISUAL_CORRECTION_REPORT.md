# Desktop visual correction report — current

**Status:** Milestone A remains **OPEN / FAIL**. This report is current for 2026-10-16 and must not be read as a completion claim.

## Baseline

- Canonical HTML SHA-256: `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`.
- Chrome 155.0.8059.40, DPR 1, fixed time/fixtures, desktop 1440×900 and 1280×800, both themes.
- Full-size canonical and React images are compared by strict RGBA equality. Canonical HTML remains unchanged.

## Corrections implemented

- Agents completed traces now record and render the final `Completed` event and keep only the canonical recent-event window.
- Personal Assistant goal-create dialog now matches the canonical 620px geometry, field gaps and border treatment. Its dialog region scores 98.82%, while full-frame remains 73–77% due to the page behind the modal.
- Scientific AI and Web Scraper demo tabs use canonical forms and local deterministic add/edit/detail/result interactions. Scoped rail and heading geometry was corrected; live adapters were preserved.
- Knowledge Collections demo now has searchable records, detail view and a local add dialog; live vector-collection integration is unchanged.
- CodeQ retains Monaco and backend calls; read-only demo rendering suppresses editor chrome not shown in the canonical HTML.
- Vault Access/Audit demo rows now match the fixture-backed canonical workspaces. Details use the approved header metadata and action order; create/edit use canonical label layout and spacing, including the enabled-looking required Name button. Audit Run example updates the row and activity in place through the local Running → Completed lifecycle rather than opening a substitute modal.
- Models `Setup needed` now filters out configured models.

## Current paired evidence

- 328 visible-tab pairs: **91.94%** mean. Evidence: `.qa/desktop-acceptance/continued-milestone-a/tab-states-final-current/manifest.json` and its full-size `approved/` and `react/` directories. This comparison covers visible-tab states; it does not capture every dialog.
- Vault Access/Audit: **56 matched named-state pairs, 85.05%** strict full-frame mean. Evidence: `.qa/desktop-acceptance/continued-milestone-a/vault-interaction-final/manifest.json`. The final modal geometry is measured as 347.22px in both apps at 1440×900; Audit Run active-state match is 96.00–96.14% across all four viewport/theme combinations.
- 140 named interaction pairs: **93.44%** mean, a prior representative set. Evidence: `.qa/desktop-acceptance/interactions/manifest.json`.
- Agents lifecycle: 12 pairs at **91.03%**; completed states range **89.86–92.17%**. Evidence: `.qa/desktop-acceptance/milestone-a-resume/agents-run-final/manifest.json`.
- Personal Assistant goal dialog: exact modal geometry and **98.82% dialog-region** match; full-frame remains **73–77%**. Evidence: `.qa/desktop-acceptance/assistant-goal-dialog-agent-current/manifest.json`.
- Focused Science/Scraper: 32 pairs at **90.50%**. Evidence: `.qa/desktop-acceptance/continued-milestone-a/science-scraper-fixed/manifest.json`.

All comparison captures above reported zero browser errors and zero demo API calls. These subsets overlap and must not be added together as a unique capture count. A visible-tab capture is not proof of all dialogs or state transitions.

## Final checks on the updated source

- `npm run check`: passed.
- `npm run test -- --maxWorkers=1`: **130/130**, 21 files.
- `npm run test:e2e -- --workers=1`: **51/51** after final Vault source/style changes; focused Access/Audit cases are **10/10**.
- `npm run build`: passed to isolated `frontend-second/dist`; verified `../web` unchanged and demo fixture JSON excluded. The existing Monaco chunk-size warning remains.
- No live Python integration or permission-matrix test was run during this frontend continuation.

See `FRONTEND_BACKEND_MASTER_GAPS.md` for the definitive wired/unwired/backend-missing inventory and outstanding work. Earlier 136-, 140-, 152-pair and initial-state reports are historical unless they match the manifests listed above.
