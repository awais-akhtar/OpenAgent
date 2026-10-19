# Desktop visual evidence — latest focused capture 2026-10-18

## Latest Calendar correction and full desktop matrix — 2026-10-18

- Fixed React calendar event-card geometry/style differences caused by the framework reset: cards now match the canonical 62 px height, padding, border treatment, radius and shadow. The canonical source was not modified.
- Four Playwright checks passed at 1440×900 and 1280×800, light/dark; assertions cover card dimensions and offsets relative to the calendar cell. Test: tests/e2e/calendar-event-canonical-style.spec.ts.
- Fresh full-page captures: 92 approved and 92 React screenshots, Chrome 155.0.8059.80, DPR 1, fixed date, theme and fixtures. React capture reported no route/layout/runtime issues. Full-frame exact RGBA is 84.94%; Calendar is 94.92%. Pairs: .qa/approved-after-calendar-fix-20261018/ and .qa/react-after-calendar-fix-20261018/; score: .qa/visual-match-after-calendar-fix-20261018.json.
- Low idle scores still include intentionally hidden canonical example execution panels. This requested contextual behavior is not treated as active-state parity. Other design mismatches remain; do not claim full fidelity.

## Shell project chooser comparison — 2026-10-18

- Added a matched canonical-vs-React capture for the shell workspace chooser at 1440×900 and 1280×800, light and dark, Chrome 155.0.8059.80, DPR 1. All four dialog bounds are identical (610×439.56 px; x/y/width/height deltas are 0) and strict RGBA agreement inside the dialog crop is **99.42%** in every condition.
- Full-size paired crops and machine-readable geometry/pixel measurements: `evidence/project-picker-20261018/`. This is component-region evidence, not whole-screen parity. The capture harness uses each app's theme control so canonical theme state remains stable through its own rerender. Canonical SHA is unchanged.
- Visual review found no meaningful chooser geometry or content difference requiring CSS changes. The remaining 0.58% pixel inequality has not been individually classified; do not label it exact or treat this dialog-region score as proof of app-wide parity.

## Prior 23-screen desktop comparison — superseded by the latest matrix — 2026-10-18

- Refreshed the canonical and React default-state captures using the same Chrome 155.0.8059.80 pin, DPR 1, fixed date, theme, deterministic fixture hashes, and 1440×900 / 1280×800 desktop viewports. Both sides produced 92 screenshots across 23 routes, two sizes and two themes; the React capture reported zero route/layout/runtime issues.
- `npm run compare:pixels` measured **84.88%** strict RGBA across 92 full-frame pairs. Per-route means, pair file paths, and raw pixel counts are in `.qa/desktop-current-20261018/visual-match.json`; PNGs are under `.qa/desktop-current-20261018/approved/` and `react/`.
- Lowest route means: Agents **61.77%**, Models **63.32%**, Fine-tuning **70.80%**, Chat **71.42%**, Research **72.22%**, Web Scraper **73.19%**, Scientific AI **74.65%**, Automations **74.88%**, CodeQ **79.10%**. Full-size comparison inspection shows a major source of the low idle scores is the intentionally hidden canonical sample execution preview in React. This follows the user's contextual Living UI instruction; those results are not equivalent active workflow scores. Focused active Living UI evidence is maintained separately. Remaining subpixel/layout/detail differences still need route-by-route review; 84.88% does not pass visual acceptance.
- The capture utilities were corrected to read the pinned Chrome version from `visual-baseline.config.json`; the old scripts hardcoded 155.0.8059.40 while the approved baseline and installed browser were 155.0.8059.80. Canonical SHA stayed `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`.
- Reproduction (PowerShell, run from `frontend-second/`; the React dev server is isolated on port 5182):
  ```powershell
  $env:APPROVED_CAPTURE_VIEWPORTS='desktop-1440x900,desktop-1280x800'; $env:OPENAGENT_APPROVED_CAPTURE_OUTPUT='.qa/desktop-current-20261018/approved'; npm run capture:approved-html
  $env:OPENAGENT_CAPTURE_VIEWPORTS='desktop-1440x900,desktop-1280x800'; $env:OPENAGENT_CAPTURE_OUTPUT='.qa/desktop-current-20261018/react'; $env:OPENAGENT_UI_ORIGIN='http://127.0.0.1:5182'; npm run capture:reference
  $env:OPENAGENT_COMPARE_REFERENCE='.qa/desktop-current-20261018/approved'; $env:OPENAGENT_COMPARE_TARGET='.qa/desktop-current-20261018/react'; $env:OPENAGENT_COMPARE_OUTPUT='.qa/desktop-current-20261018/visual-match.json'; npm run compare:pixels
  ```

## Agents lifecycle evidence — current source — 2026-10-18

- Fresh canonical/React comparison: **12 full-size matched pairs** (running, paused, completed × 1440×900 and 1280×800 × light/dark), Chrome **155.0.8059.80**, DPR 1, fixed clock, pinned fixture hashes, canonical SHA `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`. Strict RGBA mean: **87.38%**; running **88.89%**, paused **84.10%**, completed **88.36%**. Browser errors: **0**; Demo `/api/` requests: **0**. Evidence and manifest: `.qa/agent-current-20261018/`.
- Owner-requested behavior differences are listed in the manifest: canonical shows two inline workflow status rows while React keeps one contextual banner; canonical's second progress label is static illustrative text while React reports its local run lifecycle; the generic Activity sandbox entry is only in Developer → Design Preview. These are intentional behavior divergences and are not presented as component-matched passes.
- The strict score still includes unclassified component-level pixel differences. This is a focused Agents result, not a refreshed score for the other 22 routes. The broad 344-pair matrix below is from 2026-10-17 and remains historical. Milestone A remains open.

## Earlier desktop evidence — 2026-10-17 continuation

## Targeted correction evidence — 2026-10-17

- `.qa/targeted-corrections-20261017/manifest.json` inventories focused Fine-tuning Datasets, search-reset Jobs, explicitly started simulated preview, Jobs return with no autoplay, and Demo Integration Guide screenshots across 1440×900 and 1280×800, light and dark (20 screenshots; 0 Demo `/api/` calls).
- `integration-guide-production-live-1440x900.png` captures the Live Integration Guide from a production-preview build after seeding stale Demo mode. The associated `npm run verify:production-live` run confirms zero `/demo/` fixture requests, no Demo privacy/export copy or button, and a truthful disabled/unavailable export message. API calls were stubbed to 503, so this is not backend integration evidence.
- These are focused behavior screenshots. No new canonical-vs-React pixel comparison was run for this patch; the latest broad 344-pair score below remains historical to its capture source revision.

**Milestone A remains open.** The approved HTML is unchanged and pinned to SHA-256 `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`.

## Latest full-screen and Living UI captures

- Full-screen desktop tab matrix: **344 matched pairs** covering all 23 routes, visible tabs and initial states at 1440×900 and 1280×800 in both themes. Strict mean **84.48%**; zero unmatched tabs, Demo API calls, or browser errors. Paired full-size PNGs and manifest: `.qa/milestone-a-current-20261017/tab-states/`.
- Contextual Living UI matrix: **144 paired states**, 80 route-content readiness assertions; zero Demo API calls and browser errors. Strict RGBA: **82.50% overall**, with **74.79%** across 40 intentionally changed idle pairs, **83.84%** across 8 explicitly started behavior-divergence pairs, and **85.60%** across 96 active-state pairs. Full-size pairs, region scores and geometry data: `.qa/living-ui-desktop-evidence-20261017-current/`.
- Both captures use Chrome 155.0.8059.80, DPR 1, the pinned canonical hash, deterministic fixtures, fixed clock and the desktop viewport/theme matrix. The active/idle results are separate; no 100% fidelity claim is made.

## Earlier screenshot sets (historical)

The entries below are retained as prior review evidence and are not the current measurements.

## Capture conditions

- Chrome 155.0.8059.40, DPR 1, fixed capture time, deterministic fixture hashes, settled fonts, and both light/dark themes.
- Desktop-only viewports: 1440×900 and 1280×800.
- Full viewport captures are measured with strict RGBA pixel equality at native dimensions; no scaling or montage scoring.
- Canonical captures are read-only. React captures use demo mode. The current demo capture runs made zero `/api/` requests.

## Latest evidence

- Visible tabs: **328 matched pairs**, **91.94%** strict mean; desktop 1440×900 and 1280×800 in both themes. No unmatched tabs, browser errors, or demo API requests. Full-size PNG pairs and manifest: `.qa/desktop-acceptance/continued-milestone-a/tab-states-final-current/`.
- Vault Secrets/Access/Audit: **12 matched tab pairs**, **94.70%** strict mean: `.qa/desktop-acceptance/continued-milestone-a/vault-fix-final/`.
- Vault Access/Audit named interactions after final source/style corrections: **56 matched pairs across 14 states**, **85.05%** strict full-frame mean. Includes create, keyboard focus, invalid and populated forms, details, edit, approved/denied, delete confirmation, activity, Audit detail, and the corrected inline Running example state; screenshot scores and states are in `.qa/desktop-acceptance/continued-milestone-a/vault-interaction-final/manifest.json`.
- Other retained evidence: **140 named interaction pairs** at `.qa/desktop-acceptance/interactions/`; **12 Agents lifecycle pairs** at `.qa/desktop-acceptance/milestone-a-resume/agents-run-final/`; **4 Personal Assistant goal-dialog pairs** at `.qa/desktop-acceptance/assistant-goal-dialog-agent-current/`; **32 Science/Scraper pairs** at `.qa/desktop-acceptance/continued-milestone-a/science-scraper-fixed/`.
- All comparisons use Chrome 155.0.8059.40, DPR 1, fixed clock and fixture hashes, desktop viewports, paired full-size captures, and strict RGBA equality. The Vault named-state set reports zero browser errors and zero demo API requests.

## Current differences and limits

- Fine-tuning, Vault, Models, Voice, Personal Context and other screens still have material differences. The Vault detail/action rows, local Running/Completed example behavior and create/edit dialog geometry were corrected. Vault interaction full-frame mean is now 85.05%; the underlying app behind dialogs and remaining pixel differences keep acceptance open.
- Voice, CodeQ, Agents and full-frame goal-dialog captures still have visible differences. The tab and named-state captures do not cover every dialog, form, notification, validation/loading/error state, keyboard/focus state, or every execution transition.
- Milestone A remains open; do not interpret the captures as completion or 100% fidelity.
- The full verification and detailed backend/UI gap inventory are in `FRONTEND_BACKEND_MASTER_GAPS.md`.
