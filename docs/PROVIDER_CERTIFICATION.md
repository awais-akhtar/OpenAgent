# Provider Certification

OpenAgent provider certification is local-first and opt-in. The default runtime
does not contact Gmail, Outlook, IMAP, SMTP, CalDAV, or any other provider
unless the operator configures credentials and explicitly allows the external
operation.

## Implemented Local Certification Surface

- `GET /api/communication/provider-certification`
- `GET /api/communication/oauth/callback`
- `GET /api/communication/oauth/callbacks`
- `GET /api/communication/attachments`
- `POST /api/communication/attachments`

The OAuth callback endpoint stores the returned state, error, redirect URI, and
a SHA-256 fingerprint of the authorization code. The raw authorization code is
not returned by the API and is not written to audit logs.

## Provider Presets

OpenAgent includes local provider presets for:

- Gmail OAuth plus IMAP/SMTP policy
- Microsoft Outlook OAuth plus IMAP/SMTP policy
- Generic OAuth IMAP/SMTP
- Fastmail CalDAV certification profile
- Nextcloud CalDAV certification profile
- Radicale CalDAV certification profile

## Live Certification Gates

Live provider checks must be explicitly enabled:

- `OPENAGENT_CERT_GMAIL=1`
- `OPENAGENT_CERT_OUTLOOK=1`
- `OPENAGENT_CERT_CALDAV=1`
- `OPENAGENT_CERT_IMAP=1`

Without those flags and local credentials, certification routes only report
configuration, expected scopes, callback readiness, and privacy boundaries.
Google and Microsoft certification is external evidence issued against an
operator-owned OAuth application and tenant. The repository includes the
onboarding, refresh, mailbox, attachment, scheduling, approval, and gated test
surfaces; it never labels an untested deployment as provider-certified.

## Attachment Workflow

Mailbox attachments can be ingested through `POST /api/communication/attachments`
with `content_base64`. The file is stored under the local OpenAgent data
directory. Text-like attachments are also ingested into the local knowledge
store with an audit event.

## Privacy Boundary

Provider metadata, OAuth callback records, attachment metadata, and attachment
artifacts stay inside the configured OpenAgent data directory. External provider
network calls require explicit operator approval through the existing
`allow_external` controls.
