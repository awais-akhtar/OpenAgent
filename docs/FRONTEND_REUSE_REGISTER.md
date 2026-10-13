# Frontend reference and reuse register

## Audit decision

The inspected production frontend is an **original OpenAgent implementation informed by AionUI design concepts**. A source review of `frontend/src` found no AionUI renderer imports, copied AionUI component files, AionUI product logos, or AionUI screenshots/assets. The custom shell, routes, API calls, screen implementations, and styles are authored as OpenAgent code. Therefore there are no copied AionUI source files to list as adaptations in the table below.

This is a bounded source review, not a complete build-output, dependency, image/font, trademark, or SBOM clearance. Re-run those checks against the actual release artifact.

## Reuse register

| Upstream file / asset | Source version / license | New production path | Reuse type | Change / notice |
|---|---|---|---|---|
| None identified in inspected `frontend/src` | Supplied AionUI reference source identifies version 2.2.2 and Apache-2.0; see reference pack `06_LICENSE_PROVENANCE.md` | — | No source or asset copying detected | No adapted-file header or per-file AionUI notice applies based on this audit. |

## Audited reference concepts

The development-only reference pack and source atlas were used to study public interaction patterns: conversation-first home, compact grouped sidebar/history, command search, concise header, message composer, grouped conversations, focused settings, responsive workspace, restrained grayscale surfaces, typography and spacing. These are general interface concepts; OpenAgent's JSX/CSS, route model, data shape and backend integration were independently implemented. No AionUI logo or product branding is used in the shell; it presents OpenAgent.

The AionUI reference archive, source, screenshots and sample material were local audit inputs and are excluded from this repository. Production frontend source does not import AionUI assets.

## Project and dependency licensing

- The repository's own license is MIT (as recorded in the supplied provenance specification).
- The supplied AionUI source package identifies Apache License 2.0. This fact does not license its logos, third-party assets, trademarks, or unrelated dependencies by implication.
- The React frontend uses third-party packages including Arco Design, UnoCSS, icons and the existing React/Vite toolchain. Their exact resolved versions and license obligations were **not** exhaustively enumerated here; produce an SBOM/license report for the release bundle.
- This register does not claim all dependencies are MIT/Apache, does not determine trademark clearance, and is not legal advice.

## Remaining provenance checks

1. Scan the production source and built `web/` bundle for AionUI/iOfficeAI names, logos, unique assets, source maps and reference screenshots.
2. Generate an SBOM or dependency-license inventory from the locked dependency graph; review notices for Arco, icons, fonts and transitive packages.
3. Keep upstream reference archives, source and screenshots out of the distributed app and repository.
4. Review final branding, favicon, font, icon and user-facing About/licence presentation before release.

No release-grade license gate is marked PASS by this document.
