# Frontend remaining work

This is a short worklist, not a second backend gap inventory. The authoritative UI/backend mapping is [FRONTEND_BACKEND_MASTER_GAPS.md](FRONTEND_BACKEND_MASTER_GAPS.md).

The requested visual milestone is still open. Correct the approved design without substituting layout or content, then compare using the locked setup in [visual-baseline.config.json](visual-baseline.config.json).

Current priority:

1. CodeQ explorer/editor/preview/terminal/Git geometry and controls at desktop and compact widths; preserve Monaco and existing live API calls.
2. Voice recording, waveform, rail and session/transcript states across both themes.
3. Matched responsive checks at 390×844 and 768×1024 for CodeQ, Vault, Fine-tuning, Agents, Personal Context, Science, Security and Files.
4. Inventory and capture each meaningful interaction state per route, including dialogs, loading/empty/error states and demo execution transitions; keep demo states explicitly separate from live backend status.
5. Close the measured per-route mismatches and retain fresh native-size screenshot evidence.

Do not cite prior visual totals from this file's previous revision; those runs were not acceptance evidence for the current canonical fixture matrix. Current measured values belong in [PIXEL_FIDELITY_GAPS.md](PIXEL_FIDELITY_GAPS.md).
