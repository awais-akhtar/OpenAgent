# UI copy and dummy-content decisions

This audit records what the current redesign keeps, removes, or still needs to improve. Live review covered the chat home, Workspace, representative mobile widths and the 23-route render tour; it does not certify every rendered copy state.

| Area | Decision in current source | Status / remaining issue |
|---|---|---|
| Brand | Use OpenAgent name and mark; do not use AionUI branding | Source inspected; built asset/metadata scan remains outstanding. |
| Initial landing | New conversation is the default home; legacy dashboard query migrates to chat | Implemented in `App.tsx` / `navigation.ts`. Activity remains available through Tools. |
| Dashboard copy | Old dashboard-grid framing replaced with a compact Activity/status surface using real health, audit, approval and worker endpoints | Partial: still status-oriented; no fake service success. Not visually accepted. |
| Repeated headings and slogans | Shell uses compact page title and grouped navigation rather than repeating product slogans in every surface | Some specialist content retains route-level titles/descriptions for context; a complete copy/heading pass remains. |
| Chat empty state | Short prompt-led copy (for example, “What would you like to work on?”) and real composer/model controls | Source present; the actual rendering and interaction were not re-audited in a browser. |
| Loading and failure states | History and search expose loading, unavailable/retry and no-results copy | Present in shell source. No live API-failure browser test. |
| Placeholder/sample content | No shipped sample conversation records were identified in the reviewed shell/page source. API-backed lists show returned records and explicit empty states | Source search only; backend-seeded records and final DOM were not audited. Test fixtures are not product content. |
| Raw JSON | `SpecialistInspector` and Security/Settings summaries use allowlisted scalar fields, avoiding arbitrary nested payloads and sensitive internals | Security and Settings no longer show raw JSON. Automation no longer uses a JSON steps textarea; its current builder creates one step at a time. Some generic actions still use IDs. |
| Opaque IDs and action forms | Keep actual backend functionality available through contextual action dialogs and screen-specific workspaces | Remaining parity gap: some actions still require IDs and their mutation flows are not yet browser-verified. |
| Safety copy | Retain explicit, meaningful warnings for external sends, live workflow execution, scraper network access, model/privacy policy, destructive restores, CodeQ source writes, and vault rotate/export | Source config/page copy exists; server behavior and confirmation paths are not runtime verified. |
| Voice claims | State that current TTS creates a tone cue artifact, not spoken audio | Explicitly represented in source; no audio runtime verification. |
| Empty state wording | Prefer truthful “no records returned” or loading/unavailable states over prefilled examples | Present in several views; audit did not inspect every rendered state. |

## Copy/content acceptance status

- **Removed from the primary landing:** generic dashboard/status-card home in favor of chat.
- **Preserved intentionally:** concise security, scope, confirmation, external access, and capability notices needed to make real backend effects legible.
- **Still to remediate:** some specialist explanatory copy and opaque ID fields; multi-step workflow editing and scheduled triggers; live mutation/error copy review.
- **Not claimed:** complete dummy-content purge, final visual polish, localization, or DOM-level copy acceptance. Those require rendered browser inspection after implementation stabilizes.
