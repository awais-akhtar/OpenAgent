# Communication Hub

Stage 5 is implemented as a local-first communication hub.

## Implemented Features

- Email connector manifest
- CalDAV connector manifest
- Slack, Discord, Telegram, Signal, Teams, SMS, and WhatsApp connector manifests
- Contact management
- Unified inbox model
- Message search
- Message summarization
- Follow-up detection
- Draft creation
- Reply drafting
- Approval-gated sending
- Local sent-message records
- Calendar event storage
- Communication analytics
- Provider-payload sync worker for messages, contacts, and calendar events

## APIs

Connectors:

```bash
GET /api/communication/connectors
POST /api/communication/connectors/configure
```

Inbox:

```bash
POST /api/communication/messages/import
GET /api/communication/inbox?project=default&q=follow
```

Contacts:

```bash
GET /api/communication/contacts?project=default
POST /api/communication/contacts
```

Draft and send with approval:

```bash
POST /api/communication/drafts/reply
POST /api/communication/send/request-approval
POST /api/communication/send/resolve
POST /api/communication/send
```

Calendar:

```bash
GET /api/communication/calendar?project=default
POST /api/communication/calendar
```

Analytics:

```bash
GET /api/communication/analytics?project=default
```

Provider-payload sync:

```bash
POST /api/communication/sync
GET /api/communication/sync-runs?project=default
```

## Safety

Sending is simulated locally unless external sending is explicitly allowed and approved. Sync workers consume explicit local provider payloads and write sync-run records plus audit events. Live external provider polling/sending should be added connector by connector with transparent permissions, credentials, and audit logs.
