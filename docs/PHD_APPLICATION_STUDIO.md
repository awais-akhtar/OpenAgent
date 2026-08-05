# PhD Application Studio

The PhD Application Studio is an owner-scoped workspace for preparing doctoral
applications from verified personal evidence. It stores reusable academic data,
source documents, target positions, public research, generated material, and
revision history inside the configured OpenAgent data directory.

## What It Does

- Saves a reusable applicant profile: contact details, education, research
  interests, experience, skills, projects, publications, courses,
  certifications, awards, languages, links, and referees.
- Accepts PDF, DOCX, TXT, Markdown, RTF, CSV, and JSON evidence files up to
  25 MB. Files are hashed for local deduplication, text is extracted locally,
  and each owner's evidence is indexed into an isolated advanced-RAG project.
- Tracks multiple positions with institution, program, supervisor, lab,
  department, deadline, URL, requirements, description, notes, and status.
- Researches professors, labs, programs, and publications from public web
  sources only after approval for that run.
- Generates a tailored CV, statement of purpose, motivation letter, cover
  letter, professor email, research proposal, personal statement, scholarship
  statement, interview brief, application checklist, or a custom document.
- Retrieves only the selected evidence through FTS5, FAISS, Chroma, and the
  local SQLite vector fallback, then records the retrieval run and source IDs.
- Uses a local draft, independent evidence review, writer revision, and final
  proofreading pipeline. When a proofreader requests changes, the writer
  applies them and the independent model confirms the repair. Each artifact
  stores its stage, provider, model, latency, role assignment, and fallback
  trace. The record explicitly reports when only one model was available.
- Runs generation as a durable background job. The workbench displays current
  research, retrieval, drafting, review, revision, length check, proofreading,
  and completion events while SQLite preserves the run history.
- Runs an additional audited length-repair pass when the final model output is
  outside the document type's target range. A document is marked finalized
  only when it is complete, inside that range, and has a professional ending.
- Infers requested output types from a plain-language command, or uses the
  document types explicitly selected in the workbench.
- Keeps every generated and manually edited revision. A local review checks
  unresolved placeholders, unsupported numeric claims, generic phrasing,
  repeated openings, target length, evidence overlap, and similarity to an
  uploaded writing sample's sentence rhythm.
- Exports individual revisions as styled DOCX, selectable-text A4 PDF,
  Markdown, or TXT. A complete ZIP package includes the latest Markdown, DOCX,
  and PDF deliverables, quality reports, the position record, public source
  bibliography, a manifest, and optionally the source documents selected by
  the user for that package.

## Use It

1. Open **PhD Studio** from the Core section.
2. Open **My Data**, complete the reusable profile, and upload source evidence.
3. In **Application Studio**, save the target position and paste its official
   description and requirements.
4. Open **Position Research**. Enter a query or official URLs, approve that
   one public web request, and run the research. **Research & Generate** also
   performs this step automatically when the saved position has an official
   URL; that explicit button is the approval boundary for the public request.
5. Return to **Application Studio**. Write a command such as `Write a concise
   email to the professor and tailor my CV` and leave **Choose from command**
   enabled, or select exact deliverables manually.
6. Choose the evidence files for the run and select a local writer. External
   model use requires a separate per-run approval.
7. Review the result, resolve all `[NEEDS INPUT: ...]` markers, verify claims,
   edit it in **Deliverables**, and save a new revision.
8. Export one document or download the complete application package.

## Privacy Boundaries

The default path is local:

- Profile fields are stored in owner- and project-scoped SQLite records.
- Uploaded files remain under `<data-dir>/phd_workspace/sources/` and extracted
  text remains in the local database.
- Generated artifacts, reviews, public-source snapshots, and action audit events
  remain local.
- Public web research sends only the approved query and public URLs. It never
  sends the applicant profile, uploaded files, extracted private text, or
  private position notes.
- External model generation is disabled unless `allow_external_model` is true
  for that action. When enabled, only the selected evidence for that generation
  is sent to the selected model provider.
- The ZIP package excludes uploaded private source files unless the user keeps
  **Include selected original evidence** enabled for that export. Only the
  currently selected documents are included.

Local files are protected by the host account and filesystem. Use full-disk
encryption and appropriate data-directory permissions when the records are
sensitive.

## Authentic Writing

The writing workflow is designed for truthful authorship: it grounds claims in
the applicant's evidence, follows their writing preferences, can compare basic
style characteristics with their own writing sample, and leaves explicit input
markers where facts are missing. It does not attempt to manipulate or promise
results from AI-detection products. Such detectors are not reliable proof of
authorship. The final submission must be reviewed, corrected, and owned by the
applicant.

## API Surface

Key routes include:

```text
GET    /api/phd/status
GET    /api/phd/profile
POST   /api/phd/profile
GET    /api/phd/documents
POST   /api/phd/documents
DELETE /api/phd/documents/{id}
GET    /api/phd/applications
POST   /api/phd/applications
PATCH  /api/phd/applications/{id}
DELETE /api/phd/applications/{id}
POST   /api/phd/applications/{id}/research
GET    /api/phd/applications/{id}/sources
POST   /api/phd/applications/{id}/generate
POST   /api/phd/applications/{id}/generate/start
POST   /api/phd/applications/{id}/export
POST   /api/phd/command
GET    /api/phd/generation-runs
GET    /api/phd/generation-runs/{id}
GET    /api/phd/artifacts
GET    /api/phd/artifacts/{id}
PATCH  /api/phd/artifacts/{id}
DELETE /api/phd/artifacts/{id}
POST   /api/phd/artifacts/{id}/authenticity
POST   /api/phd/artifacts/{id}/export
```

Under enforced authentication, read routes require `phd:read` and mutations
require `phd:write`. All records are scoped to the authenticated owner.

Set `OPENAGENT_SEARCH_ENDPOINT` to a local or self-hosted SearxNG JSON endpoint
to use it for public position search. The value may contain `{query}`. Without
that setting, the approved research action uses DuckDuckGo's public HTML search
endpoint and the local SSRF-protected scraper.

## Verification

Run the focused contract and workflow tests:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_phd_workspace tests.test_ui_runtime_contract tests.test_web_ui -v
node --check web\app.js
```
