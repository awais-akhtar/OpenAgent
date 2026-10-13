# OpenAgent frontend professional design and reliability audit

**Audit date:** 2026-10-13  
**Scope:** Current React 19, TypeScript, Vite, Arco Design and UnoCSS frontend; all 22 active destinations; preservation of the existing Python API/security boundary. PhD/Application Studio has been retired by request and is out of scope.

## Executive assessment

The frontend is navigable and the Chat landing screen has a recognizable conversation-first composition, but the application does not yet feel like one finished professional product. The most visible issue is not a missing gradient or decorative flourish: most non-chat views are generic endpoint browsers or action-form collections with large blank panes. They expose implementation structure instead of helping a person complete the screen's primary task.

The rendered result is restrained, but too often resembles an empty admin scaffold: inventory lists, repeated project labels, uppercase category labels, repeated descriptions, generic key/value facts, and empty-state explanations occupy the same positions on many unrelated screens. This creates a low-information, repetitive interface. Adding more cards, badges, tabs, marketing copy, or accent color would amplify the problem.

The correct next move is a deliberate product-design pass followed by screen-family implementations. Keep the existing Python services, endpoints, authorization, egress policy, approvals, data scope and supported actions. Replace the generic page presentation with task-centered screens, progressively reveal infrequent controls, and make empty states useful without filling them with promotional content.

## Evidence and confidence

- Reviewed active navigation and shell source in frontend/src/lib/navigation.ts and frontend/src/App.tsx.
- Reviewed Chat, Workspace, Activity, shared resource, personal-work and platform-work page implementations.
- Visually inspected the 22-view screenshot capture at desktop and mobile sizes in light/dark themes, including the captured Chat, Workspace, Security and Adaptive Engine screens. A local contact sheet is .tmp-ui-evidence-aion/contact-sheet.png.
- The screenshot folder also contains one stale phd.png from an earlier capture, despite the current manifest reporting 22 views and 88 screenshots. Treat that image as obsolete evidence; do not restore the retired destination.
- Existing current-code checks at audit time: TypeScript check passed; 67 frontend tests passed; production build passed; 13 Playwright E2E tests passed against a fresh Python server. These checks establish route/render and selected interactions, not visual quality or complete feature correctness.
- Full Python discovery is not green (last recorded concurrent-tree run: 16 failures, 46 errors, 84 skipped). Some failures are engine/runtime/platform dependency issues. Audit findings below do not reclassify those failures as frontend defects.

## Highest-priority product problems

### P0 — Replace the generic surface with task-first information architecture

1. **Remove the nine-card assistant showcase from Chat.** It reads as a promotional feature grid, pushes the composer down, and makes the home screen look like a product demo. Keep one compact assistant/mode chooser in the composer or a single searchable menu. Show a short list only when the user opens that chooser.
2. **Remove the duplicate provider controls on Chat.** Provider pills appear above the composer while provider/model selectors already exist inside the composer. Keep one source of truth in the composer; expose the current choice clearly without repeating it.
3. **Stop rendering generic inventory-and-facts workspaces for unrelated tools.** PlatformWorkspaces currently renders five distinct areas (Developer, Adaptive Engine, Fine-tuning, Science and Scraper) through the same inventory / record list / primitive facts layout. Personal Assistant, Calendar, Communications and Personal Context also share one record-inspector pattern. Reuse low-level primitives, but give each screen a workflow and hierarchy suited to its task.
4. **Replace endpoint navigation as the primary mental model.** A person should not need to understand the API's resource names to use Calendar, Security, Models, Science or Fine-tuning. Present tasks and meaningful records first; place technical inventories and diagnostics under contextual secondary navigation.
5. **Make the persistent shell intentional.** The sidebar has only a few primary destinations while most of the product sits behind Tools. Keep fast access for Chat and the user's active work/history, then use one compact searchable Tools index for specialist destinations. Group by user task, not by backend service taxonomy. Remove redundant Workspace/project context when the same project is already clear in the shell or page.

### P1 — Remove repetitive copy and empty scaffolding

- Delete eyebrow + title + explanatory paragraph combinations when the title and visible controls already explain the screen. Keep one concise title and only task-specific guidance that changes what the user should do.
- Remove repeated statements that the server is authoritative, data is local, the service returned no records, or a feature is “server-backed” from routine UI. Keep safety details at the point of a consequential action and in documentation.
- Do not display API-shaped values like true, ok, Unknown, Registered, or generic “no displayable summary fields” as if they were meaningful content. Map supported values to domain language; hide absent/unsupported facts and expose raw diagnostic detail only in a deliberate diagnostics surface.
- Avoid empty bordered cards and three-column shells when there is no selected record. Collapse or replace empty panes. Offer one contextual first action, not a panel explaining that the panel is empty.
- Use empty states with one sentence at most and one primary action. Avoid helper paragraphs that merely paraphrase the heading.
- Remove repeated Refresh text buttons from every header where background refresh or a quiet icon action is sufficient. Keep visible refresh only where freshness is a real user task.

### P1 — Define a cohesive visual system before adding detail

- Set a deliberate type scale and content measure: strong page title, compact section labels, comfortable body text, readable table rows, and clearly differentiated code/metadata. Current small controls and body copy read thin/miniature in the 1440px captures.
- Use a restrained neutral palette, one purposeful accent, one semantic status palette, and consistent border/radius/shadow rules. Do not solve the generic look by adding gradients, random icon tiles, or colorful cards.
- Establish a spacing scale and apply it to shell, page headers, forms, tables, dialogs and empty states. Align page content to one grid; avoid each feature family inventing a different width and left edge.
- Make density adapt to content: long inventories should use a real table/list with columns, sorting and useful row actions; a singleton service status should not occupy a wide two-pane browser.
- Reserve tabs for a small number of peer content modes the user intentionally switches between. Replace decorative/over-segmented tabs and tabs that simply mirror endpoint names with a contextual subnav or filters.
- Provide consistent selected, hover, focus, loading, empty, error, disabled, and destructive states. Make icons, icon-only actions, focus rings and tooltips consistent.
- Define dark mode from the same tokens and hierarchy, not as a separate collection of overrides.

## Screen-by-screen remediation

| Active view | Current issue observed in source/screens | Recommended direction |
| --- | --- | --- |
| Chat | Nine descriptive assistant tiles, provider pills above composer, and provider/model controls inside composer compete for the main action. The home screenshot has the right basic skeleton but still feels like a showcase. | Composer is the hero. Keep provider/model/locality in one compact composer footer. Move assistants to a searchable selector or compact recent/favorite list. Retain conversation actions in one overflow menu. Remove promotional descriptions and duplicate provider choice. |
| Activity (dashboard) | Good operational data exists, but the screen remains a stack of system summary, recent events, worker runs and expandable runtime facts. Empty fixture data produces many generic “no records” lines. | Prioritize alerts/approvals and recent work, then compact health/resource status. Collapse zero/healthy categories. Make timestamps/status scannable. Keep runtime diagnostics subordinate. |
| CodeQ (codebase) | Screenshot presents three large empty panels and a form-first protected-workspace setup. The core development task is lost when no repository is selected. | Repository chooser as a compact left rail; selected repo opens file/tree and active run timeline. Keep preflight, write approval, terminal and Git as contextual panels. Put add workspace behind one explicit empty-state action. Never relax server-side write checks. |
| Voice | Multi-column panels and long capture/persona forms make setup dominate session use. | Session-first console: recent sessions and status, one focused record/control surface, transcript/audio preview only when available. Put persona creation and advanced options behind a secondary action. Preserve explicit microphone permission and server workflows. |
| Research | Several columns expose session selection, research request form and source/provenance panes at once; explanatory copy adds noise before a session exists. | Start from a centered investigation input or recent session list. Once selected, show findings and citations as the main canvas; keep checkpoints and source provenance as a right-side contextual inspector. Make external research consent explicit at the action. |
| Assistants (agents) | Roster/marketplace/permissions/approvals are presented as a general inventory with operations. Screens risk conflating agent registry with start work. | Default to concise assistant list and active task/approval queue. Put permissions and marketplace into secondary sections. Make run review, approval status, delegation and permission scope legible in the selected assistant context. |
| Knowledge | Search, index status, graph/timeline/tags and ingest controls compete. Some controls appear as a generic form near empty results. | Make project search and sourced results primary. Show citations/provenance with each result. Put graph/timeline/tags in a compact secondary explorer. Keep upload/ingest in a single contextual import action with progress and failure recovery. |
| Workspace | Empty state opens a full editor, preview and details pane for an untitled file. This creates a large empty canvas and looks unfinished. | File/project explorer first. With no selection, show one Create document or import action; do not mount the editor/preview/inspector. When a file is selected, use a clean editor with preview toggle and contextual version history; keep resizing as a power feature. |
| Automations | Workflow library, builder contract and run history are treated like peer inventories; creating a workflow exposes raw JSON steps. | Workflow list and recent runs first. Provide a purpose-built visual step builder with validation and clear dry-run preview. Put raw JSON in an advanced editor only. Separate dry run from confirmed live execution and retain network/approval safeguards. |
| Assistant | Daily brief, task, goal, coach and CRM records share a list/inspector pattern. Several forms/actions are always close to the main view. | Brief and next actions first; tasks as a clean list with inline completion; goals and relationships in their own secondary views. Open create/edit forms on demand and return to the relevant record after save. |
| Communications | Connectors, inbox, contacts, calendar, approvals and scheduled messages compete within one generic record browser. | Inbox/drafts as primary. Show connector health as compact context; use approval queue for outbound communication; create drafts in a focused compose surface. Keep send approval and external-send safety visible at the final step. |
| Models | Provider inventory, installed models, hardware and policy are repeated as inventory panes; configured/empty states expose raw facts. | Provider and model list with clear availability, active/default state and capabilities. Detail pane should show meaningful configuration and setup progress, not primitive key/value dumps. Separate host hardware and policy into compact advanced sections. |
| Security | Captured page repeats Secret records count in both posture and enforcement sections, has eight dense tabs, and repeats explanatory statements about backend authority. | One posture summary with unique metrics and actionable findings. Use a concise section list for audit, egress, retention, shares, secrets, reports and sandbox. Keep decisions and approvals contextual, redact values by default, and remove duplicate counts/copy. |
| Developer | Generic developer inventory browser exposes technical records without a clear “what can I do?” workflow. | API keys, webhooks, tools and extensions should be distinct task sections with state, scope, last-used/health data and contextual create/revoke actions. Keep secret material reveal-once and never imply a display value grants authorization. |
| Settings | Long vertically stacked endpoint settings and helper text create an unbounded form. | Categorized navigation with small, focused forms; show saved state and validation beside the affected field. Put advanced runtime controls behind disclosure, preserve current values on errors, and use a sticky save/cancel bar only for dirty edits. |
| Adaptive Engine (engine) | Screenshot is a generic inventory sidebar plus two true/ok facts stretched across a mostly blank pane. | Show actual runtime readiness, supported hardware, active plan and recent benchmark evidence in a compact technical console. Put inspect/benchmark/compile behind contextual actions with progress/results. Do not fabricate metrics when services return none. |
| Scientific AI (science) | Generic status/inventory pane lacks a recognizable scientific workflow. | Start with dataset/tool selection and a clear analysis/run task. Show method, inputs, safety/license constraints and evidence-bound outputs together. Keep memory/catalog diagnostics in secondary navigation. |
| Calendar | Generic collection inventory and object facts are used where time is the key dimension. | Use a real agenda/week view and chronological event list. Put calendar/provider management in a secondary drawer. Keep timezone, conflicts and sync direction explicit. |
| Personal Context (twin) | Profile and suggestion data is reduced to primitive facts, with generic actions; privacy boundaries are not visually clear. | Use a compact, consent-oriented profile/timeline. Separate observed facts from inferred patterns and user-approved goals. Make source, recency and project scope visible without exposing raw private payloads. |
| Vault | Secret form and inventory are visible together; helper copy repeats protection claims. | Default to masked entries and a deliberate unlock/reveal workflow. Use a focused add-secret dialog; display scope and expiry clearly. Ensure clear-on-leave/timer behavior, prevent accidental copy, and test sensitive state cleanup. |
| Web Scraper (scraper) | Generic jobs and service capability facts dominate; external access configuration is not a task-led flow. | Jobs list plus one create/schedule action. Preview the target and server-reported policy/robots result before execution. Show progress, last result, and retry/cancel only where supported. Keep SSRF/egress enforcement server-side. |
| Fine-tuning (finetune) | Generic inventory facts do not make dataset → job → adapter lifecycle understandable. | Build a clear staged pipeline: dataset readiness, training job, logs/progress, resulting adapter and activation. Gate actions on actual backend capability. Make destructive delete/deactivate contextual and confirmed. |

## Confirmed UI and code findings

| Severity | Finding | Evidence / impact |
| --- | --- | --- |
| P1 | Chat duplicates provider selection and displays a nine-card assistant showcase. | frontend/src/pages/ChatPage.tsx renders .oa-chat-home-models and .oa-chat-home-agents, while ChatComposer also receives provider, setProvider, model and setModel. Verified in the desktop screenshot. |
| P1 | Five unrelated platform tools share one generic layout and primitive-only detail renderer. | frontend/src/pages/PlatformWorkspaces.tsx uses the same platform-frame, platform-nav, Facts, rows and action-dialog presentation for Developer, Engine, Fine-tuning, Science and Scraper. Nested domain data is filtered rather than represented meaningfully. |
| P1 | Four personal-work screens share one record list/inspector/action shell. | frontend/src/pages/PersonalWorkSurfaces.tsx routes Assistant, Calendar, Communications and Twin through one common PersonalWorkSurface; screenshot comparison confirms Calendar does not use a calendar/agenda composition. |
| P1 | Action forms are driven by large configuration objects and expose low-level payload concepts. | frontend/src/pages/ResourcePages.tsx includes “Steps as JSON array” workflow creation and long endpoint-specific action definitions; shared dialog rendering makes distinct operations feel like the same generic form. |
| P1 | Empty Workspace renders the editor, preview and details regions without a selected file. | frontend/src/pages/WorkspacePage.tsx always renders .oa-file-editor and .oa-file-inspector; verified in the empty-state screenshot. |
| P1 | Security duplicates a summary count and repeats policy explanation. | frontend/src/pages/SecurityPage.tsx summary and enforcement sections both surface secret-record information; verified in the captured page. |
| P2 | Provider and agent fetch failures are silently converted to empty arrays. | frontend/src/pages/ChatPage.tsx catches /api/models and /api/agents errors and sets empty lists without a visible status. A transient failure can hide capabilities with no explanation or retry cue. |
| P2 | Activity refresh can overlap. | frontend/src/pages/DashboardPage.tsx polls every 30 seconds and also refreshes on visibility change/manual action; refresh sets a flag but has no request-level deduplication. Test timer/resume overlap and stale-response ordering before changing behavior. |
| P2 | Arco's full stylesheet is imported for a small use of Arco components. | frontend/src/main.tsx imports @arco-design/web-react/dist/css/arco.css; recorded Vite output is about 665 kB CSS (about 79 kB gzip). Audit actual used styles and tree-shaking before replacing or splitting it. |
| P2 | The local screenshot archive has an obsolete PhD view image. | .tmp-ui-evidence-aion includes phd.png while the current manifest is 22 views/88 images. This is stale local test evidence, not an active route. |

### Reliability checks required before labeling further items as bugs

- Repeat Chat stream tests for empty completion, server error after partial output, cancellation, switching conversations mid-stream, slow history refresh and failed post-stream message reload. Verify no duplicate user or assistant messages.
- Exercise every provider/agent unavailable, loading and retry state. Do not report hidden capabilities as none when the API failed.
- Test Activity polling and visibility changes with delayed/failing responses to confirm requests do not overlap or overwrite newer data.
- Exercise Workspace create/open/save/archive/restore under slow network, API failure and project change; verify unsaved edits are not silently discarded.
- Validate the current Python auth and project boundary for each sensitive collection/item route. Frontend project query values are never authorization.
- Check clipboard/reveal/auto-clear for Vault and API keys in supported browsers.
- Run a full browser console/error sweep over all 22 views against real backend capabilities. Current E2E verifies routes and representative actions only.

## Keep / remove / redesign rules

### Remove

- Promotional/showcase grids, repeated assistant descriptions, generic What would you like helper copy on non-chat screens.
- Duplicate provider selectors, duplicate metrics, duplicate active-project labels, repeated title/description blocks, empty bordered containers.
- Backend terminology and raw object keys where the user needs a task label.
- Raw JSON step entry as the default workflow builder and any action form shown before the user requests that action.
- Inventory tabs that only enumerate API endpoints and do not represent a meaningful user task.

### Keep

- All 22 active routes and every currently supported backend workflow/API connection.
- Chat history, project selection, conversation actions, model choice, attachment support, local-first default, explicit external-access choice, and cancel/recovery behavior.
- Approval gates, server-side validation, egress boundaries, secret masking, project data scope and destructive-action confirmation.
- Genuine task navigation, accessible tabs when they represent true peer content, and advanced diagnostics required by technical users.

### Redesign

- Start from task and user intent, then reveal records, configuration and diagnostics as needed.
- Prefer lists, tables, timelines, editors, agendas, chat transcripts and real progress indicators when those match the data.
- Use modals only for short, focused create/edit/confirm flows. Keep long workflows in a dedicated guided surface.
- Design for loading, populated, empty, partial-failure, permission-denied, offline and narrow-screen states. Screenshot fixtures must make empty state recognizable as deliberate rather than broken.
- Use domain components with shared accessible primitives; avoid both duplicated styling and one universal generic page template.

## Recommended implementation sequence

1. **Freeze the design direction:** OpenAgent-branded, quiet editorial workspace; graphite/white/soft-neutral base, one restrained accent, high legibility, firm alignment, deliberate density. Create tokens and example compositions for shell, Chat, list/detail, forms, empty/error/loading and mobile.
2. **Fix navigation and remove obvious clutter:** simplify Chat home; unify provider/model controls; trim repeated eyebrow/subtitle copy; reduce sidebar groups and place the 22 tools in a searchable task taxonomy.
3. **Rebuild the shell and base primitives:** sidebar/history, top bar, responsive drawers, page frame, headings, button hierarchy, menu, dialog, list/table, status, empty/error/loading, form field and toast.
4. **Implement screen families with custom compositions:** Chat; task/approvals; document/file workspace; technical workbench; personal work/calendar/communications; privacy/security; model/runtime; specialized science/research/voice. Share primitives, not whole-screen templates.
5. **Connect each visible action to existing APIs:** add typed response/request contracts and explicit states; preserve endpoint semantics and route tests. Do not alter Python behavior to make the UI appear successful.
6. **Remove transitional/legacy styles and dead assets:** identify actual Vite entry points; remove unused legacy shell CSS/JavaScript only after reference search and build proof; reduce the global Arco CSS footprint if feasible.
7. **Verify each screen:** typecheck, component tests, targeted API contract tests, all-route Playwright, 320/390/768/1280/1440 viewports, both themes, keyboard/focus, 200% zoom, error/empty/populated states, and screenshots compared side-by-side with agreed design examples.

## Acceptance criteria for the redesign phase

- Every active view has a distinct task-focused hierarchy and no generic endpoint browser as its primary experience.
- Chat opens directly to an uncluttered conversation composer; no duplicate provider control or showcase card grid.
- Empty states do not render irrelevant editors, inspectors, tabs or empty cards.
- All visible data is understandable; no generic primitive facts (true, ok, Registered) masquerade as finished product content.
- No repeated explanatory copy, decorative labels or route headings that provide no decision value.
- Active APIs, 22 navigation destinations, project scope, permissions, local-first defaults and server-side safety gates remain intact.
- Every screen has reviewed loading, empty, populated, partial-failure, permission and narrow layouts in light/dark mode.
- CI is green for frontend checks and all relevant backend tests; full browser suite has no console errors; screenshots and manual accessibility review support the release claim.

## Audit conclusion

The current frontend has useful backend connections and a functional navigation baseline, but its visual polish is held back by generic screen templates, over-explanation, redundant controls and empty panes. Prioritize the shared structural causes—especially Chat showcase/duplicate controls, universal inventory layouts, and Workspace's empty editor—before adding decoration. The audit is a plan, not evidence that the redesign itself has been implemented.
