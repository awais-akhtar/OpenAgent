import { Check, Circle, CircleAlert, Info } from 'lucide-react';
import { record } from '../../lib/react/api';

const primaryFields = new Set([
  'status', 'state', 'enabled', 'available', 'health', 'version', 'provider', 'model',
  'description', 'summary', 'message', 'created_at', 'updated_at', 'started_at', 'finished_at',
  'count', 'total', 'progress', 'type', 'category', 'priority', 'due_at', 'starts_at', 'ends_at',
  'project', 'owner', 'name', 'title', 'label', 'id', 'url', 'filename', 'kind', 'method',
]);

function friendlyName(value: string) {
  return value.replaceAll('_', ' ').replace(/\b\w/g, char => char.toUpperCase());
}

function printable(value: unknown): string | undefined {
  if (typeof value === 'string') return value.trim() || undefined;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return undefined;
}

export function specialistFacts(value: unknown): Array<[string, string]> {
  return Object.entries(record(value))
    .filter(([key, item]) => primaryFields.has(key.toLowerCase()) && printable(item) !== undefined)
    .map(([key, item]) => [key, printable(item) as string]);
}

export function SpecialistInspector({ value, emptyLabel }: { value: unknown; emptyLabel: string }) {
  const source = record(value);
  const fields = specialistFacts(source).map(([key, value]) => [key, value] as [string, unknown]);
  const description = fields.find(([key]) => ['description', 'summary', 'message'].includes(key.toLowerCase()));
  const status = fields.find(([key]) => ['status', 'state', 'health', 'available', 'enabled'].includes(key.toLowerCase()));
  const highlights = fields.filter(([key]) => !['description', 'summary', 'message'].includes(key.toLowerCase()));

  return <section className="specialist-inspector" aria-label="Selected item details">
    {status && <div className={`specialist-status ${String(status[1]).toLowerCase().includes('error') || String(status[1]).toLowerCase() === 'false' ? 'warning' : ''}`}>
      {String(status[1]).toLowerCase() === 'true' ? <Check size={15}/> : String(status[1]).toLowerCase() === 'false' ? <CircleAlert size={15}/> : <Circle size={9}/>}
      <span>{friendlyName(status[0])}: {printable(status[1])}</span>
    </div>}
    {description && <p className="specialist-description">{printable(description[1])}</p>}
    {highlights.length > 0 ? <dl className="specialist-facts">{highlights.map(([key, item]) => <div key={key}><dt>{friendlyName(key)}</dt><dd>{printable(item)}</dd></div>)}</dl>
      : !description && <div className="specialist-no-details"><Info size={16}/><span>{emptyLabel}</span></div>}
  </section>;
}
