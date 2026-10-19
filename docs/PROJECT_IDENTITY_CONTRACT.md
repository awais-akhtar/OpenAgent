# CodeQ project identity contract

## Read API

`GET /api/codebase/project-identities` returns minimal CodeQ identity data for
the resolved owner. The result contains `codebase_project_id`, display `name`,
the exact persisted `codebase_namespace`, and timestamps. It deliberately omits
source paths, workspace paths, owner identifiers, and repository maps.

The endpoint is covered by the existing `/api/codebase/` read authorization
scope (`codebase:read`) when auth enforcement is enabled. In that mode, the
runtime resolves the owner from the authenticated bearer session and ignores a
caller-supplied `owner` query value. In local mode, the route follows the
existing CodeQ owner resolution behavior and defaults to `local`.

## Identity boundaries

- `codebase_project_id` is the primary key in `codebase_projects` and must be
  used only with CodeQ project endpoints.
- `codebase_namespace` is the exact value persisted in
  `codebase_projects.project`. CodeQ stores this separately from its project
  ID and workspace ID.
- The response's `cross_service_mapping.available` is `false`. Other services
  generally use project namespace strings and do not store a foreign key from
  their records to `codebase_project_id`. The same namespace may also be used
  by more than one CodeQ project or owner. Therefore the current database
  cannot safely claim that a CodeQ project maps to a particular Chat,
  Research, or other service project identity.
- Do not substitute `workspace_id` for `codebase_project_id`, or use
  `codebase_project_id` as a generic `?project=` value. A cross-service
  identity requires a persisted, owner-scoped mapping contract and corresponding
  migration of the services that need it.

This is a read-only discovery API. It does not create, rename, merge, or
reassign projects and does not change existing APIs.
