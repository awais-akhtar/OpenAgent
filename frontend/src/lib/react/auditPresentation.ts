type AuditEvent = Record<string, unknown>;

const opaqueIdentifier = /^(?:\d+|[a-f\d]{8}-(?:[a-f\d]{4}-){3}[a-f\d]{12}|[a-f\d]{16,}|[a-z\d_-]{24,})$/i;
const verbs: Record<string, string> = {
  approve: 'approved', approved: 'approved',
  complete: 'completed', completed: 'completed',
  create: 'created', created: 'created',
  deny: 'denied', denied: 'denied',
  delete: 'deleted', deleted: 'deleted',
  fail: 'failed', failed: 'failed',
  insert: 'created',
  open: 'opened', opened: 'opened',
  remove: 'removed', removed: 'removed',
  request: 'requested', requested: 'requested',
  reveal: 'revealed', revealed: 'revealed',
  revoke: 'revoked', revoked: 'revoked',
  run: 'started', start: 'started', started: 'started',
  send: 'sent', sent: 'sent',
  stop: 'stopped', stopped: 'stopped',
  update: 'updated', updated: 'updated', upsert: 'updated',
};

function humanize(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const raw = value.trim();
  if (!raw || opaqueIdentifier.test(raw) || /^(?:unknown|registered|not reported)$/i.test(raw)) return undefined;
  if (!/[._/:\-]/.test(raw)) return raw;

  const words = raw.split(/[._/:\-]+/).filter(word => word && !opaqueIdentifier.test(word));
  if (!words.length) return undefined;
  const last = words.at(-1)!.toLowerCase();
  if (verbs[last]) words[words.length - 1] = verbs[last];
  return words.map((word, index) => index === 0 ? `${word[0].toUpperCase()}${word.slice(1).toLowerCase()}` : word.toLowerCase()).join(' ');
}

export function auditEventTitle(event: AuditEvent): string {
  return humanize(event.title ?? event.action ?? event.event_type ?? event.type) ?? 'Security event';
}

export function auditEventContext(event: AuditEvent): string | undefined {
  return humanize(event.resource_type) ?? humanize(event.resource) ?? humanize(event.actor);
}

export function auditEventOutcome(event: AuditEvent): string | undefined {
  const status = event.status ?? event.outcome;
  if (typeof status !== 'string' || !status.trim()) return undefined;
  const normalized = status.trim().toLowerCase().replaceAll('_', ' ');
  if (['ok', 'success', 'succeeded', 'complete', 'completed'].includes(normalized)) return 'Completed';
  if (['pending', 'requested', 'awaiting approval'].includes(normalized)) return 'Pending';
  return humanize(normalized) ?? undefined;
}
