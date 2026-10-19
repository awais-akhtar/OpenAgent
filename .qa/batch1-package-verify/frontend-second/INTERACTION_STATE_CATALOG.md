# Canonical desktop interaction-state catalog

- Canonical HTML SHA-256: `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1` (validated before generation).
- DOM source: `.qa/desktop-acceptance/milestone-a-resume/interaction-inventory-final-v13.json`; historical paired-state evidence: `.qa/desktop-acceptance/continued-milestone-a/interactions-after-fixes/manifest.json` plus `.qa/desktop-acceptance/milestone-a-resume/agents-run-final/manifest.json`; latest broad tab/initial evidence: `.qa/milestone-a-current-20261017/tab-states/manifest.json`; older 328-pair matrix: `.qa/desktop-acceptance/continued-milestone-a/tab-states-final-current/manifest.json`; Vault named-state evidence: `.qa/desktop-acceptance/continued-milestone-a/vault-interaction-final/manifest.json`.
- Environment: inventory discovery used Chrome 155.0.8059.40 at 1440×900, DPR 1. Latest broad matched screenshots use Chrome 155.0.8059.80, DPR 1, desktop 1440×900 and 1280×800, in both themes.
- Historical representative interaction run: 140 named pairs, 93.44% strict exact pixels, 0 Demo API calls, 0 browser errors. Current Agents lifecycle evidence is a separate 12-pair capture at 87.38% (`.qa/agent-current-20261018/manifest.json`); earlier 91.03% Agents evidence is superseded.
- Latest full-screen default-route matrix (captured 2026-10-18): **92 matched pairs**, **84.88%** strict exact pixels, no route/layout/runtime failures. It covers all 23 route defaults at 1440×900 and 1280×800 in both themes. The aggregate includes intentional idle-state differences from hiding canonical sample execution panels; it is not active-state parity. Evidence: `.qa/desktop-current-20261018/visual-match.json` and paired approved/React PNGs.
- Latest visible-tab matrix remains the earlier 2026-10-17 set: **344 matched pairs**, **84.48%**, 0 unmatched tabs, 0 Demo API calls or browser errors; it covers the 328 visible-tab states plus 16 initial route captures at both desktop sizes and themes. Owner-requested idle trace suppression lowers this full-frame score. Evidence: `.qa/milestone-a-current-20261017/tab-states/manifest.json`.
- Earlier 328-pair visible-tab matrix scored 91.94%, but predates later implementation changes and is historical only: `.qa/desktop-acceptance/continued-milestone-a/tab-states-final-current/manifest.json`.
- Current Vault interaction matrix: 56 pairs across 14 named states; 85.05% strict exact pixels, 0 demo API calls and 0 browser errors. The modal geometry and field positions match in the measured 1440×900 create dialog; full-frame scores remain lower because they include the app behind the overlay.
- This route inventory describes discovered controls, not exhaustive activation. The detailed state rows lower in this file include historic pairs; use the latest manifests and `FRONTEND_BACKEND_MASTER_GAPS.md` for current measurements.

## How to read this inventory

This catalog enumerates the canonical route DOM and all visible canonical tab DOM states, including action triggers, form fields, modal/menu/status signals, and text-level lifecycle markers. It distinguishes discovery from activation: an action listed here is an available trigger, not evidence that its resulting state was exercised. Forms are recorded by fields, not treated as validated workflows. Only states present in the paired screenshot coverage list below have matched-state visual evidence. Unlisted or unpaired states remain unverified; they are not assumed missing or complete.

No controls are activated by the generator. That is intentional: actions such as record deletion, exports, file selection, clipboard operations, and media workflows require explicit deterministic fixture handling before safe matched-state probing. The canonical HTML is read-only and is SHA-checked on every run. Exact accessible labels, record values, and the tab where each trigger appeared are preserved in the JSON catalog.

The `Forms and fields` column lists only forms visible in the initial or tab DOM snapshot. Hidden dialog forms are not counted there; their opener actions are listed, and their fields/validation states remain unverified until each dialog is opened and inspected. The lifecycle column reports words present in visible text and is not proof that transitions were exercised.

## Route-level interaction inventory

| Workspace | Canonical tabs | React tab discovery | Dialog/detail trigger families | Menu/action trigger families | Execution trigger families | Visible forms and fields | Lifecycle words in captured text (not transition proof) |
|---|---|---|---|---|---|---|---|
| Home (`home`) | — | no tabs | projectPicker, palette, notifications | — | livingOpen | — | none detected in captured default/tab text |
| Chat (`chat`) | Files, Preview, Context | 3/3 states paired | modelPicker, palette, notifications | feedback | livingStart, traceToggle, traceStart, traceExport, livingOpen | — | none detected in captured default/tab text |
| Agents (`agents`) | — | no tabs | openTaskModal, notifications, palette | feedback | previewRun, traceToggle, traceSelect, traceStart, traceDetails, traceExport, livingOpen | — | none detected in captured default/tab text |
| Research (`research`) | Summary, Findings, Sources, Notes, Activity | 7/7 states paired | notifications, palette | feedback | livingStart, traceToggle, traceSelect, traceStart, traceDetails, traceExport, previewRun, livingOpen | — | running, completed, pending, unavailable |
| Files (`files`) | Documents, Links, Notes, Uploaded files, Preview, Summary, Details | 7/7 states paired | notifications, palette | feedback | livingOpen | — | none detected in captured default/tab text |
| Automations (`automations`) | Overview, Configuration, Runs, Settings | 4/4 states paired | openAutomationModal, runDetails, notifications, palette | feedback | traceToggle, traceSelect, traceStart, traceDetails, traceExport, previewRun, livingOpen | — | error, running, paused, completed, pending, approval, unavailable |
| Calendar (`calendar`) | — | no tabs | openEventModal, eventDetails, notifications, palette | — | livingOpen | — | none detected in captured default/tab text |
| Models (`models`) | All models, Configured, Local, Setup needed | 4/4 states paired | providerModal, modelDetails:GPT-4o, modelDetails:Claude Sonnet, modelDetails:Gemini Pro, modelDetails:Llama local, notifications, palette | feedback | livingStart, traceToggle, traceSelect, traceStart, traceDetails, traceExport, livingOpen | — | completed, pending |
| Settings (`settings`) | Appearance, Models & providers, Privacy, Notifications, Data & backups, Advanced | 6/6 states paired | palette, notifications | — | livingStart, livingOpen | — | approval |
| CodeQ (`codeq`) | Code, Preview, Terminal, Git | 4/4 states paired | codeApproval, notifications, palette | feedback | livingStart, previewRun, traceToggle, traceStart, traceExport, livingOpen | — | error, approval |
| Knowledge (`knowledge`) | Search, Collections, Ingestion, Retrieval details | 4/4 states paired | demoCreate, demoDetails, demoEdit, demoAllActivity, notifications, palette | demoMenu, demoResetRoute | livingStart, livingOpen | demoKnowledgeSearch: Search sources, documents or concepts…* | success, offline |
| Voice (`voice`) | Recordings, Transcripts, Voice settings | 3/3 states paired | demoCreate, demoDetails, demoEdit, demoAllActivity, notifications, palette, demoTranscript, demoExportTranscript | demoMenu, demoResetRoute | livingStart, traceToggle, traceSelect, traceStart, traceDetails, traceExport, livingOpen | demoSettings: Transcription language English (UK) English (US) Urdu Spanish, Microphone Demo input device System microphone, Playback speed 1× normal 1.25× faster 0.75× slower | completed, pending, recording, offline |
| Personal Assistant (`assistant`) | Today, Tasks, Goals, Daily brief | 4/4 states paired | demoCreate, demoDetails, demoEdit, demoAllActivity, notifications, palette | demoMenu, demoResetRoute | livingStart, livingOpen | — | empty, approval, offline |
| Communications (`communications`) | Inbox, Sent, Drafts, Connected accounts | 4/4 states paired | demoCreate, demoDetails, demoEdit, demoAllActivity, notifications, palette | demoMenu, demoResetRoute | livingStart, livingOpen | — | offline |
| Security (`security`) | Overview, Access, Sessions, Audit | 4/4 states paired | demoCreate, demoDetails, demoEdit, demoAllActivity, notifications, palette | demoMenu, demoResetRoute | livingStart, livingOpen | — | completed, approval, offline |
| Developer (`developer`) | Connections, API keys, Webhooks, MCP, Activity | 5/5 states paired | demoCreate, demoDetails, demoEdit, demoAllActivity, notifications, palette | demoMenu, demoResetRoute | livingStart, livingOpen | — | paused, offline |
| Adaptive Engine (`engine`) | Overview, Hardware, Jobs, Plans | 4/4 states paired | demoCreate, demoDetails, demoEdit, demoAllActivity, notifications, palette | demoMenu, demoResetRoute | livingStart, livingOpen | demoRunForm: Workload Local inference Dataset processing Embedding refresh, Compute target Auto (GPU → CPU) CPU only GPU only, Memory cap 8 GB 4 GB 16 GB | completed, offline |
| Scientific AI (`science`) | Projects, Experiments, Data, Reports | 4/4 states paired | demoCreate, demoDetails, demoEdit, demoAllActivity, notifications, palette | demoMenu, demoResetRoute | livingStart, traceToggle, traceSelect, traceStart, traceDetails, traceExport, livingOpen | demoRunForm: Experiment name*, Dataset Validation sample v2 Benchmark telemetry, Method Baseline evaluation Calibration analysis Ablation study | failure, completed, pending, offline |
| Personal Context (`twin`) | Profile, Preferences, Sources, Data controls | 4/4 states paired | demoCreate, demoDetails, demoEdit, demoAllActivity, notifications, palette | demoMenu, demoResetRoute, demoResetAll | livingStart, livingOpen | — | offline |
| Vault (`vault`) | Secrets, Access, Audit | 3/3 initial tabs paired; 14 named interaction states captured | demoCreate, demoDetails, demoEdit, demoAllActivity, notifications, palette | demoMenu, demoResetRoute | demo Run example state, livingStart, livingOpen | create/edit Name and Description; required Name validation | pending, success, running, completed, offline |
| Web Scraper (`scraper`) | New extraction, Jobs, Results, Rules | 4/4 states paired | demoCreate, demoDetails, demoEdit, demoAllActivity, notifications, palette | demoMenu, demoResetRoute | livingStart, traceToggle, traceSelect, traceStart, traceDetails, traceExport, livingOpen | demoRunForm: Page URL*, Extraction type Text and metadata Structured records Links only, Page limit 5 pages 10 pages 25 pages, I confirm I have permission to extract this example content.* | completed, pending, offline |
| Fine-tuning (`finetune`) | Datasets, Jobs, Checkpoints, Results | 4/4 states paired | demoCreate, demoDetails, demoEdit, demoAllActivity, notifications, palette | demoMenu, demoResetRoute | livingStart, traceToggle, traceSelect, traceStart, traceDetails, traceExport, livingOpen | demoRunForm: Dataset Support instruction set Code review examples, Adapter LoRA · rank 16 LoRA · rank 8 QLoRA · 4-bit, Epochs 3 epochs 2 epochs 1 epoch | completed, pending, offline |
| All Tools (`tools`) | — | no tabs | notifications, palette | — | livingOpen | — | none detected in captured default/tab text |

## Paired interaction states with screenshot evidence

| State | Pairs | Exact RGBA match | Range |
|---|---:|---:|---:|
| `voice/idle` | 4 | 88.9% | 88.19%–89.59% |
| `voice/recording` | 4 | 94.72% | 94.26%–95.05% |
| `voice/saved` | 4 | 94.17% | 92.93%–95.24% |
| `codeq/initial` | 4 | 91.55% | 90.85%–92.43% |
| `codeq/preview` | 4 | 92.55% | 91.92%–93.34% |
| `codeq/terminal` | 4 | 91.89% | 91.34%–92.59% |
| `codeq/git` | 4 | 91.4% | 90.81%–92.15% |
| `codeq/git-diff-expanded` | 4 | 89.88% | 88.23%–91.96% |
| `codeq/preview-size-selected` | 4 | 92.39% | 91.69%–93.26% |
| `codeq/trace-expanded` | 4 | 91.08% | 90.88%–91.28% |
| `codeq/trace-step-expanded` | 4 | 90.88% | 90.69%–91.06% |
| `codeq/trace-running` | 4 | 90.91% | 90.72%–91.1% |
| `codeq/trace-paused` | 4 | 90.94% | 90.75%–91.14% |
| `assistant/goal-create-dialog` | 4 | 74.05% | 72%–75.79% |
| `chat/workflow-running` | 4 | 91.75% | 91%–92.33% |
| `chat/workflow-paused` | 4 | 91.69% | 90.92%–92.28% |
| `chat/workflow-completed` | 4 | 91.73% | 90.95%–92.34% |
| `research/workflow-running` | 4 | 96.12% | 95.73%–96.43% |
| `research/workflow-paused` | 4 | 91.28% | 90.88%–91.57% |
| `research/workflow-completed` | 4 | 91% | 90.76%–91.19% |
| `models/workflow-running` | 4 | 98.11% | 97.98%–98.21% |
| `models/workflow-paused` | 4 | 98.07% | 97.9%–98.21% |
| `models/workflow-completed` | 4 | 97.88% | 97.72%–98.01% |
| `automations/workflow-running` | 4 | 94.84% | 94.43%–95.15% |
| `automations/workflow-paused` | 4 | 94.77% | 94.3%–95.12% |
| `automations/workflow-completed` | 4 | 94.54% | 94.14%–94.86% |
| `science/workflow-running` | 4 | 92.65% | 91.76%–93.35% |
| `science/workflow-paused` | 4 | 92.57% | 91.63%–93.31% |
| `science/workflow-completed` | 4 | 92.37% | 91.45%–93.09% |
| `finetune/workflow-running` | 4 | 95.01% | 94.44%–95.46% |
| `finetune/workflow-paused` | 4 | 94.94% | 94.31%–95.42% |
| `finetune/workflow-completed` | 4 | 94.76% | 94.17%–95.22% |
| `scraper/workflow-running` | 4 | 92.68% | 91.79%–93.38% |
| `scraper/workflow-paused` | 4 | 92.6% | 91.66%–93.34% |
| `scraper/workflow-completed` | 4 | 92.41% | 91.49%–93.14% |
| `agents/preview-running` | 1 | 89.98% | 89.98%–89.98% |
| `agents/preview-paused` | 1 | 89.9% | 89.9%–89.9% |
| `agents/preview-completed` | 1 | 68.7% | 68.7%–68.7% |
| `agents/preview-running` | 1 | 92.23% | 92.23%–92.23% |
| `agents/preview-paused` | 1 | 93.68% | 93.68%–93.68% |
| `agents/preview-completed` | 1 | 57.29% | 57.29%–57.29% |
| `agents/preview-running` | 1 | 90.08% | 90.08%–90.08% |
| `agents/preview-paused` | 1 | 90% | 90%–90% |
| `agents/preview-completed` | 1 | 69.3% | 69.3%–69.3% |
| `agents/preview-running` | 1 | 92.32% | 92.32%–92.32% |
| `agents/preview-paused` | 1 | 93.83% | 93.83%–93.83% |
| `agents/preview-completed` | 1 | 57.64% | 57.64%–57.64% |

These are the only state names recorded by the current paired interaction manifest. The table does not imply that every listed state is exhaustive for that screen or that other states are absent.

## Explicit coverage gaps

The current Vault activation suite adds working evidence beyond the original DOM discovery list. Each visual state below has four matched screenshots (1440×900 and 1280×800, light and dark); ranges are strict **full-frame** exact-pixel percentages, not dialog-region scores.

| Vault state | E2E activation | Pair range |
|---|---|---:|
| Access create dialog | Yes | 82.12–84.48% |
| Access Name keyboard focus | Yes | 82.12–84.48% |
| Access invalid submit | Yes | 81.23–84.20% |
| Access populated form | Yes | 82.24–84.57% |
| Access request detail | Yes | 80.64–83.34% |
| Access edit dialog | Yes | 82.23–84.57% |
| Access approved | Yes | 91.40–92.52% |
| Access denied | Yes | 91.44–92.56% |
| Access delete confirmation | Yes | 80.41–83.68% |
| Access All activity | Yes | 80.14–83.23% |
| Audit create dialog | Yes | 82.25–84.58% |
| Audit invalid submit | Yes | 81.42–84.31% |
| Audit record detail | Yes | 81.01–83.68% |
| Audit Run example active | Yes | 96.00–96.14% |

The test suite exercises local demo CRUD/review actions, required validation, focus restoration, local Run example completion, structured activity fields and a zero-`/api/` boundary. These Vault cases do not establish interaction completeness on the other workspaces; the catalog-wide gaps below remain open.

- No exhaustive activation matrix exists yet for every dialog, nested menu, create/edit/delete form, invalid submission, validation error, submitting indicator, success result, or failure result on all 23 routes.
- Notification loading, empty, populated, unread/read, refresh and error states are not represented in the paired interaction manifest.
- Route-level loading, empty, error, retry, approval, failed, cancelled and recovery states are not comprehensively paired across all workspaces.
- Local demo lifecycle actions are only evidenced where named in the paired interaction manifest; controls visible in the catalog remain unverified unless specifically captured.
- Accessibility activation paths (keyboard tab order, Enter/Space, Escape, outside dismissal, focus trap/restoration) are not proven by this visual inventory.
- React tab discovery means the matching label was found in the React DOM and captured; it does not certify visual parity or all nested controls.

## Machine-readable source

Full controls, accessible labels, tab-level form fields, default status signals, and capture details are in [`.qa/desktop-acceptance/continued-milestone-a/interaction-state-catalog-after-fixes.json`](`.qa/desktop-acceptance/continued-milestone-a/interaction-state-catalog-after-fixes.json`).

Regenerate after refreshing the read-only DOM audit and screenshot manifests with:

```powershell
node scripts/build-interaction-state-catalog.mjs
```

The generator refuses mismatched canonical hashes or non-23-route audits and does not mutate or launch actions in the canonical HTML.
