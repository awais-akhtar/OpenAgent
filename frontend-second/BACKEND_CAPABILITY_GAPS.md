# Backend API surface — generated source inventory

> Supporting route evidence only. [FRONTEND_BACKEND_MASTER_GAPS.md](FRONTEND_BACKEND_MASTER_GAPS.md) is the sole current gap, wiring and test-status map.

Updated: 2026-11-02
Source: `openagent.routes.route_contracts()` and production frontend source at this working-tree revision.

## Registered route inventory

The registry has **788 distinct method/path patterns across 55 groups**. The current scope mapper covers **784/788**. The unmapped routes are `GET /api/health`, `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/refresh`. These are public health and authentication/bootstrap paths, not a count of working private routes. Adaptive Engine declarations are included for registry accounting only and excluded from the owner's frontend/backend gap closure.

The static scan found bounded route-pattern strings for **243/788** entries under `frontend/src` and **413/788** under `frontend-second/src`. A match is not an executed request. A missing literal may be a dynamically constructed URL or a route intentionally reserved for admin/runtime use. The complete route-by-route string matrix is in [FRONTEND_CAPABILITY_GAPS.md](FRONTEND_CAPABILITY_GAPS.md), and handler/scope/file candidates are in the master file.

| Route group | Audience | Declared | Primary refs | Second refs |
| --- | --- | --- | --- | --- |
| health | Internal / operator | 5 | 3 | 1 |
| core | Internal / operator | 9 | 0 | 2 |
| integrations | Advanced / admin | 13 | 0 | 2 |
| chat | User-facing | 12 | 9 | 9 |
| agents | User-facing + advanced permissions | 19 | 9 | 11 |
| workflows | User-facing | 15 | 7 | 7 |
| knowledge | User-facing | 20 | 9 | 14 |
| semantic | Advanced user | 13 | 0 | 5 |
| media | User-facing | 25 | 11 | 16 |
| local_ai | Advanced user / admin actions | 31 | 19 | 20 |
| enterprise | Admin-only | 14 | 0 | 0 |
| developer | Advanced / admin | 25 | 16 | 19 |
| legacy_attribution | Review required | 5 | 0 | 0 |
| security | Admin-only / security-sensitive | 25 | 24 | 23 |
| calendar | User-facing | 24 | 10 | 19 |
| communication | User-facing + external-send approval | 40 | 14 | 24 |
| assistant | User-facing | 18 | 10 | 11 |
| personal_os | Advanced user | 23 | 0 | 21 |
| production | Internal / operator | 34 | 0 | 0 |
| twin | Advanced user / privacy-sensitive | 11 | 10 | 10 |
| fact_provenance | Advanced user / privacy-sensitive | 7 | 0 | 7 |
| vault | Advanced / security-sensitive | 11 | 7 | 11 |
| scraper | Advanced user / network-sensitive | 7 | 5 | 7 |
| research | User-facing | 16 | 2 | 9 |
| finetune | Advanced user / admin actions | 16 | 10 | 14 |
| workspace | User-facing | 9 | 3 | 4 |
| auth | Admin-only / security-sensitive | 18 | 1 | 6 |
| document | User-facing | 10 | 9 | 9 |
| draft | Advanced user | 5 | 0 | 0 |
| worker | Internal / operator | 8 | 2 | 0 |
| vector_store | Internal / operator | 5 | 0 | 0 |
| advanced_rag | Advanced user | 4 | 1 | 2 |
| inference | Internal / operator | 14 | 3 | 1 |
| ultra_model | Advanced user / admin actions | 12 | 0 | 9 |
| adaptive_engine | Advanced user / operator | 29 | 10 | 5 |
| contacts | User-facing | 7 | 0 | 5 |
| backup_restore | Admin-only / destructive confirmation | 4 | 4 | 4 |
| model_cookbook | Advanced user | 12 | 0 | 11 |
| document_processing | Advanced user | 14 | 2 | 2 |
| image_studio | User-facing | 15 | 0 | 12 |
| scheduler | User-facing + admin actions | 25 | 0 | 3 |
| safe_terminal | Advanced user / explicit consent | 5 | 0 | 5 |
| mcp_runtime | Advanced / admin | 7 | 0 | 0 |
| tool_runtime | Internal / operator | 2 | 0 | 0 |
| compute_router | Internal / operator | 2 | 0 | 2 |
| agent_bridge | Internal / operator | 9 | 0 | 0 |
| skills | Advanced user / admin actions | 17 | 0 | 0 |
| conversation-organization | User-facing | 9 | 4 | 3 |
| enterprise-identity | Admin-only / security-sensitive | 15 | 0 | 0 |
| science | Advanced user | 20 | 9 | 9 |
| codebase | User-facing + destructive/approval controls | 25 | 20 | 19 |
| voice | User-facing / microphone-sensitive | 5 | 0 | 1 |
| g25_advanced | Advanced user / privacy-sensitive | 17 | 0 | 17 |
| model_serve | Advanced user / model execution | 15 | 0 | 15 |
| voice_conversation | User-facing / microphone-sensitive | 11 | 0 | 7 |

## Capability and security disposition

The current missing backend contracts, incomplete integrations, backend operations needing UI, permissions and test evidence are classified once in the [master gap map](FRONTEND_BACKEND_MASTER_GAPS.md). This generated file does not repeat those findings because a route-string scan cannot establish a capability gap or a safe UI placement.

## Evidence limits

- The counts come from declared contracts and source text, not all possible dynamic runtime branches.
- Scope mapping does not prove object ownership or approval behavior.
- Source references do not prove a frontend control works or that a Python handler persists data.
- Read the dated commands and current blockers in the master file before treating a capability as verified.
