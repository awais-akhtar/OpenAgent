# Approved frontend source

The user's attached `frontend final.html` is the canonical product design. The exact attachment is preserved as `reference/frontend-final-approved.html`.

- SHA-256: `BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1`
- Source file: `C:/Users/awais/Downloads/frontend final.html`
- Screen count: 23 navigation destinations (22 workspaces plus All tools)
- The original interactive HTML, including its deterministic local demo behavior and living execution layers, remains byte-for-byte available as the approved source artifact.
- `public/demo/reference-fixtures.json` contains fixture metadata/data extracted from this exact source and is permitted only in local development/demo mode.
- Older `.qa/` captures and reports are not current acceptance evidence. The active runners pin this canonical SHA and refuse the superseded HTML; current initial-state captures are under `.qa/approved-html-current/` and `.qa/react-parity-current/`, with strict scores in `.qa/visual-match-current.json`. The 2026-10-15 React screenshot snapshot is also preserved under `.qa/react-parity-final-20261015/`.

## Implementation status

The React app implements all 23 routes and some live backend flows, but it has not reached visual or interaction parity with this attached source. Demo/live selection and a source-verified local fixture adapter are present; visual correction and matched interaction-state coverage remain incomplete. Milestone A remains open until all screens, interactions, themes, viewports and controlled pixel comparisons are verified against the attachment.
