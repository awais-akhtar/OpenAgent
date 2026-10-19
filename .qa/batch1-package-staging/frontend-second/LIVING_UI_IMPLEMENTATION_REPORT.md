# Living UI implementation report

Status date: 2026-10-17. Scope: `frontend-second/` only. Desktop contextual behavior was changed; the canonical HTML was not edited (SHA-256 `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`). This report describes source behavior and test evidence, not completion of Milestone A or verified production integration.

## Context rule implemented

`ExecutionTrace` is strictly a deterministic Demo player. It renders only after an explicit start action and only in Demo mode. Its synthetic steps, progress, timing, metrics and sample output are never rendered in Live mode. CodeQ and Research Live views remain domain-owned and use their existing API event paths.

`LivingActivityCenter` is now an explicitly separate Demo tutorial sandbox available from Developer → “Design Preview”. It no longer adds a top-bar control to the execution workspaces. Users must open it and explicitly start a sandbox walkthrough. Tutorial playback/history is locally simulated and labeled as such. It is not a substitute for an operation started in Agents, Research or another workspace. It is absent in Live. Voice keeps only its native recording, transcript, timer and playback states.

CodeQ and Research Demo traces start as compact status/timeline summaries. Their Pause/Resume, Stop and Play/replay controls remain accessible in that compact presentation; the detailed trace opens only when the user deliberately selects Expand. This keeps Monaco and the Research report in view during normal active status monitoring.

Demo requests are blocked in `src/api/client.ts` regardless of development or production build. Demo action tests and captures assert zero `/api/` requests. Demo trace labels state that the action is simulated/local. Live mode has no trace playback controls.

Data mode is intentionally available as Demo in development and forced to Live in production by `src/state.ts`. A production preview check therefore verifies the Live screen's absence of sample execution content with API responses offline-stubbed; it does not simulate a production Demo session.

## Screen-by-screen behavior

| Screen | Living UI shown? | When and data source | Live-mode truth |
|---|---|---|---|
| Home / Dashboard | No | Routine status and recent records only. | Existing API-backed summaries; no fabricated run trace. |
| Chat | No generic trace | Conversation stream only. | `/api/chat/stream` currently emits conversation/status/chunk/message events; no verified tool-execution event contract was found. |
| Agents | Yes, inline Demo trace and one status banner | After `Preview activity`; banner state reads Running, Paused, Completed or Cancelled and is mounted once. Local deterministic scenario, explicitly marked demo. | Agent run results use `/api/agents/run` or `/api/agents/run-checked`; no incremental run-event stream was confirmed. No demo trace is shown. |
| Research | Yes, inline Demo trace | Only after `Preview running state · simulated` is explicitly selected. The approved example report is not treated as a running job. | Native research stream/checkpoint UI uses the selected real session. No `ExecutionTrace` fixture is shown. |
| Files / Workspace | No | Opening, editing or previewing files uses standard loading, save feedback and errors. | Existing file/document APIs; no execution trace. |
| Automations | Yes, inline Demo trace | After a local Demo workflow run succeeds and while Overview is selected. | Live workflow run records are shown from backend records; no verified event stream, so no synthetic detailed trace. |
| Calendar | No | Events, reminders and sync use standard progress/status. | Sync status may be polled by its own workspace. No generic trace. |
| Models | Yes, inline Demo trace | After Demo setup is started or its setup preview action is selected, on Setup activity. | Actual local-AI setup jobs are read and polled; no verified SSE job stream or verified percentage/event detail. No demo trace. |
| Settings | No | Routine configuration and backend health status. | No synthetic execution display. |
| CodeQ | Yes, inline Demo trace | After `Preview run` starts the local example. | Real CodeQ run events use the existing events/SSE contract, with run snapshots and actual event payloads; trace fixture is suppressed. |
| Knowledge | No | Search/index/retrieval use ordinary loading, empty and error states. | Backend records/status only; no generic trace. |
| Voice | No generic trace | Native microphone/session, recording waveform/timer, transcript and playback information. Demo recording is labeled simulated. | Backend session, upload/STT/TTS operations are represented by returned statuses. Browser microphone streaming is unavailable; no fabricated trace or recording is shown as real. |
| Personal Assistant | No inline trace. | Routine goal/task planning does not use a generic execution timeline. | Backend task/goal responses; no generic trace. The separate tutorial is only available from the Developer Design Preview entry in Demo and is hidden in Live. |
| Communications | No | Inbox, drafts, calendar and sync have ordinary loading/approval states. | Backend records and explicit external-action consent; no generic trace. |
| Security | No | Policies, access and sessions use permission/status panels. | Security-sensitive controls remain permission/authorization gated; no sample events. |
| Developer | No | API keys, webhooks and integrations use standard controls and delivery history. | Backend records only; no generic trace. |
| Adaptive Engine | No generic trace | Plans are explicit local simulations; routine hardware/status tabs remain compact. | Existing status/run records only; no synthetic event timeline. |
| Scientific AI | Yes, inline Demo trace | After a user starts a local example experiment/run from the demo workflow. | Existing science run response/experiment records are used where supported; no event stream or persistent executed-run history was verified. No sample trace in Live. |
| Personal Context | No | Profile, sources and data-control views use standard record states. | Backend records where available; no generic trace. |
| Vault | No | Secret/access/audit operations use security-specific records and confirmations. | Permission-sensitive API responses only. No generic trace or sample credential data. |
| Web Scraper | Yes, inline Demo trace | After explicit local `Run example` / extraction action. | Real job/status/history endpoints are present; no incremental job event stream was verified. Live results must come from those endpoints; no sample trace. |
| Fine-tuning | Yes, inline Demo trace | After clicking `Simulate training`, on Jobs. | Job creation/list/status exists. No checkpoint manifest, incremental event contract, or real run progress contract was verified. No sample metrics/checkpoints in Live. |
| All Tools | No | Catalog discovery/filter/detail only. | No generic execution panel. |

The ten workflows the canonical design associates with execution visibility are Agents, Chat, Research, CodeQ, Models, Automations, Scientific AI, Fine-tuning, Web Scraper and Voice. Chat is intentionally not assigned the generic trace because current chat streaming does not provide verified tool-call events; Voice uses its native audio state instead.

## Source changes

- `src/components/ExecutionTrace.tsx`: hide the synthetic trace in Live and until an explicit Demo start signal; CodeQ and Research begin compact and keep pause/resume/stop controls accessible without opening step detail.
- `src/screens/AgentsDemoWorkspace.tsx`: mount its inline trace only after `Preview activity`; report the actual Running/Paused/Completed/Cancelled state in one banner and drive the four-stage strip from the same local status and step index.
- `src/screens/HomeChat.tsx`: remove generic trace from Chat; preserve existing chat stream.
- `src/screens/WorkspaceScreens.tsx`: CodeQ trace starts only after Demo `Preview run`; Research trace starts only from explicit simulated preview action (fixed dead control).
- `src/screens/OperationsScreens.tsx`: Automation and Model traces require a successful explicit Demo operation and the relevant tab.
- `src/screens/AdvancedScreen.tsx`: Science, Scraper and Fine-tuning traces require explicit local example/simulation actions; remove generic Voice trace.
- `src/components/LivingActivityCenter.tsx`: move the separate Demo tutorial to Developer → Design Preview, require explicit tutorial start, and label its local playback/history/details. Hidden outside Demo mode.
- `src/api/client.ts` and `src/screens/WorkspaceScreens.tsx`: CodeQ and Research SSE GET readers now use the shared mode guard, authenticated request headers and event parser rather than calling unguarded `fetch` directly.
- `src/api/client.ts`: reject JSON, multipart/form and SSE API requests whenever the authoritative in-memory mode is Demo, even if localStorage is unavailable; production state remains locked to Live.

## Requested contextual deviations

- The canonical HTML's default example traces are intentionally not shown in React idle states, by owner request. This is an intentional behavior difference and is reported separately from active-state matches; the HTML was not changed.
- The tutorial/sandbox launch is accessible only from Developer → Design Preview in Demo. It is absent from ordinary workspace top bars and all Live screens.
- Agents' stage strip follows the local trace stage/status from Not started through Running/Paused/Completed/Cancelled; it no longer keeps an unrelated fixed “In progress” state.
- CodeQ and Research show compact run status/timeline by default after Demo start; the detailed trace appears only after deliberate expansion. This is an owner-requested behavior difference from the canonical HTML trace's automatic expanded start.
- Chat still has no generic trace until an actual tool/agent execution event is available. Voice retains native audio states without a generic trace.

## Acceptance boundary

These changes correct contextual visibility, compact execution controls and API-mode isolation. They do not claim that all screen states match the canonical design, that every dialog/control has been verified, or that any Live workflow was exercised against a running Python service. Full application visual acceptance remains open. See `LIVING_UI_BACKEND_MAPPING.md` and `LIVING_UI_TEST_RESULTS.md`.
