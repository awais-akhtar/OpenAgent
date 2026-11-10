---
name: Model route failure
about: Report a model, format, backend, or recovery path that fails in OpenAgent
title: "[Model route] "
labels: bug, model-route
---

## Route

- OpenAgent version or commit:
- Model family and exact `model_type`:
- Checkpoint name and revision (do not attach weights):
- Persisted source format and manifest name:
- Backend and device (`native`, `device`, or `public-range`; CPU/GPU model):
- Entry point (desktop, CLI command, API route, or Python factory):

## Reproduction

1.
2.
3.

Expected result:

Actual result, including the full error and whether any output was published:

## State and resources

- Fresh request or warm session:
- Media types and ordering, if applicable:
- Streaming, cancellation, restore, or distributed world size, if applicable:
- Peak RAM and model file size, if measured:
- Exact `phase1_verify.py` result, if this concerns Phase 1 evidence:

## Logs and receipts

Attach a minimal log and any route-specific receipt produced by the capture tool. Remove API keys, tokens, private prompts, local user names, and unrelated file paths. A CPU product smoke receipt does not certify a different publisher checkpoint or backend.
