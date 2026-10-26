import { useEffect, useRef, useState, type CSSProperties, type FormEvent } from "react";
import { motion } from "motion/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { handleTabListKeyDown, tabIndex } from "../components/tabKeyboard";
import {
  Activity, AlertCircle, ArrowRight, AudioLines, BookOpen, BookMarked, Braces,
  BarChart3, CalendarClock, CalendarDays, Check, CheckCircle2, CircleHelp, CloudOff, Cpu, Database,
  Copy, FileClock, FileDown, FileText, Fingerprint, FlaskConical, Globe2, HardDrive, Image, Info, Plus, RotateCcw,
  KeyRound, Layers3, Lock, LockKeyhole, Mail, Mic, MoreHorizontal, Network, Play, Search, Server, Settings2,
  Search as SearchIcon, Shield, SlidersHorizontal, Sparkles, SquareTerminal, Terminal, Trash2, Upload, UserRound, X, Zap,
  type LucideIcon,
} from "lucide-react";
import { advancedScreenEndpoints, apiConnectionMessage, apiDelete as backendDelete, apiGet as backendGet, apiGetBlob as backendGetBlob, apiPatch as backendPatch, apiPost as backendPost, fetchAdvancedScreenData, type AdvancedRoute, type JsonRecord } from "../api/client";
import { useWorkspaceState } from "../state";
import { advancedDemoAction, advancedDemoScreen, resetAdvancedDemoRoute } from "../demo/advanced";
import { demoProvider } from "../demo/provider";
import type { DemoRecord } from "../demo/types";
import { ExecutionTrace } from "../components/ExecutionTrace";
import * as Dialog from "../components/ui/dialog";

// All operations in this workspace go through one boundary. Demo mode is backed by
// browser-local fixtures; live mode continues through the authenticated API client.
const apiPost = <T extends JsonRecord = JsonRecord>(path: string, body: unknown, signal?: AbortSignal): Promise<T> =>
  useWorkspaceState.getState().dataMode === "demo" ? advancedDemoAction(path, body) as Promise<T> : backendPost<T>(path, body, signal);
const apiPatch = <T extends JsonRecord = JsonRecord>(path: string, body: unknown): Promise<T> => backendPatch<T>(path, body);
const apiDelete = <T extends JsonRecord = JsonRecord>(path: string): Promise<T> => backendDelete<T>(path);
const apiGet = <T extends JsonRecord = JsonRecord>(path: string, signal?: AbortSignal): Promise<T> => {
  if (useWorkspaceState.getState().dataMode !== "demo") return backendGet<T>(path, signal);
  if (path.startsWith("/api/voice/turns")) return Promise.resolve({ turns: [{ id: "demo-turn-1", speaker: "user", role: "user", text: "Review onboarding feedback and release blockers.", is_final: true, created_at: "Today" }] } as unknown as T);
  if (path.includes("/probe")) return Promise.resolve({ status: "ready", exists: true, checksum_valid: true, demo: true } as unknown as T);
  return Promise.resolve({ demo: true } as unknown as T);
};

const enc = (v: string) => encodeURIComponent(v);

async function recordDemoWorkspaceActivity(route: string, name: string, item: string): Promise<void> {
  if (useWorkspaceState.getState().dataMode !== "demo") return;
  const eventId = globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  await demoProvider.recordActivity(route, {
    id: `${route}-activity-${eventId}`,
    name,
    item,
    route,
    status: "Saved locally · demo",
    time: new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }),
    recorded_at: new Date().toISOString(),
  });
}

type Definition = { title: string; description: string; tabs: string[]; icon: LucideIcon; subject: string; note: string };

const definitions: Record<string, Definition> = {
  knowledge: { title: "Knowledge", description: "Search indexed sources and trace information back to its origin.", tabs: ["Search", "Collections", "Ingestion", "Retrieval details"], icon: BookOpen, subject: "Knowledge sources", note: "Indexed documents and citations appear when a knowledge source is connected." },
  voice: { title: "Voice", description: "Record, review transcripts and manage your voice sessions.", tabs: ["Recordings", "Transcripts", "Voice settings"], icon: Mic, subject: "Voice sessions", note: "Recording controls stay unavailable until microphone permission and the voice service are confirmed." },
  assistant: { title: "Personal assistant", description: "Organise goals, tasks and daily plans.", tabs: ["Today", "Tasks", "Goals", "Daily brief"], icon: Sparkles, subject: "Your priorities", note: "Tasks and goals appear here when the assistant service is connected." },
  communications: { title: "Communications", description: "Review connected messages and prepare drafts in one place.", tabs: ["Inbox", "Contacts", "Calendar", "Sent", "Drafts", "Scheduled", "Connected accounts"], icon: Mail, subject: "Messages", note: "Connector state comes from the backend. External polling and message sends require explicit confirmation." },
  security: { title: "Security", description: "Manage access and inspect audit activity.", tabs: ["Overview", "Access", "Sessions", "Audit", "Privacy", "Retention", "Secrets", "Sandbox"], icon: Shield, subject: "Access and audit", note: "Permissions and audit records must come from the authenticated security service." },
  developer: { title: "Developer tools", description: "Manage integrations, keys, MCP and developer activity.", tabs: ["Connections", "API keys", "Webhooks", "MCP", "Activity"], icon: Terminal, subject: "Developer connections", note: "Credentials, tool schemas and delivery logs are unavailable until the developer service connects." },
  engine: { title: "Adaptive engine", description: "Review system capabilities and tracked execution jobs.", tabs: ["Overview", "Hardware", "Jobs", "Plans"], icon: Cpu, subject: "Device and execution capabilities", note: "Capacity and job status are shown only when the engine reports verified measurements." },
  science: { title: "Scientific AI", description: "Manage reproducible experiments and datasets.", tabs: ["Projects", "Experiments", "Data", "Reports"], icon: FlaskConical, subject: "Scientific projects", note: "Experiments and results require a connected service and recorded provenance." },
  twin: { title: "Personal context", description: "Review and control the information used to personalise your work.", tabs: ["Profile", "Preferences", "Sources", "Data controls"], icon: UserRound, subject: "Personal context", note: "No profile facts or inferred preferences are displayed until the context service provides them." },
  vault: { title: "Vault", description: "Keep credentials and other sensitive values protected.", tabs: ["Secrets", "Access", "Audit"], icon: LockKeyhole, subject: "Protected secrets", note: "Secret names, values and permissions require an authenticated vault response." },
  scraper: { title: "Web scraper", description: "Run permitted extraction jobs and review their output.", tabs: ["New extraction", "Jobs", "Results", "Rules"], icon: Globe2, subject: "Website extraction", note: "No request is sent until domain permissions, extraction scope and backend policy are verified." },
  finetune: { title: "Fine-tuning", description: "Prepare datasets, training jobs and model artefacts.", tabs: ["Datasets", "Jobs", "Checkpoints", "Results"], icon: SlidersHorizontal, subject: "Training workspace", note: "Dataset validation, compatibility plans and training status require backend confirmation." },
  gallery: { title: "Image Studio", description: "Browse, organise and edit your image library.", tabs: ["Library", "Albums", "Upload", "Operations"], icon: Image, subject: "Image library", note: "Images and albums appear when the gallery service is connected." },
  cookbook: { title: "Model Cookbook", description: "Host, certify and serve local model builds.", tabs: ["Hosts", "Cached", "Certifications", "Logs"], icon: BookMarked, subject: "Model hosts", note: "Hosted models and certification records require a connected cookbook service." },
  ultramodel: { title: "Ultra Model", description: "Control ultra-scale inference manifests, plans and sessions.", tabs: ["Status", "Manifests", "Plans", "Sessions"], icon: Zap, subject: "Inference sessions", note: "Manifests, routing plans and sessions appear when the ultra-model service is running." },
  personalos: { title: "Personal OS", description: "Manage personal presets, notes, signatures and drafts.", tabs: ["Status", "Presets", "Notes", "Gallery", "Signatures", "Drafts"], icon: UserRound, subject: "Personal workspace", note: "Presets and notes appear when the personal-os service is connected." },
};

const routeIcons: Record<string, LucideIcon> = {
  knowledge: Search, voice: AudioLines, assistant: CheckCircle2, communications: Mail,
  security: Fingerprint, developer: Braces, engine: Activity, science: Database,
  twin: UserRound, vault: KeyRound, scraper: Globe2, finetune: HardDrive,
  gallery: Image, cookbook: BookMarked, ultramodel: Zap, personalos: UserRound,
};

const securityTabEndpoints: Record<string, Record<string, string> | null> = {
  Overview: { status: "/api/security/status" },
  Access: { status: "/api/security/status", grants: "/api/security/shares", egress: "/api/security/egress" },
  Sessions: { sessions: "/api/auth/sessions" },
  // The endpoint requires security:admin. Always constrain this workspace to
  // its explicit local project; never make an unscoped audit request.
  Audit: { events: "/api/audit?project=default&limit=100" },
  Privacy: { reports: "/api/security/privacy-reports" },
  Retention: { retention: "/api/security/retention" },
  Secrets: { secrets: "/api/security/secrets" },
  Sandbox: { profiles: "/api/security/sandbox-profiles" },
};

async function fetchSecurityScreenData(tab: string, signal?: AbortSignal): Promise<Record<string, JsonRecord>> {
  const endpoints = securityTabEndpoints[tab];
  if (!endpoints) return {};
  const rows = await Promise.all(Object.entries(endpoints).map(async ([key, path]) => [key, await apiGet(path, signal)] as const));
  return Object.fromEntries(rows);
}

function ScreenTabs({ tabs, active, onChange }: { tabs: string[]; active: string; onChange: (tab: string) => void }) {
  return <div className="tabs" role="tablist" aria-label="Workspace sections" onKeyDown={handleTabListKeyDown}>{tabs.map((tab) => <button type="button" role="tab" aria-selected={active === tab} tabIndex={tabIndex(active === tab)} className={`tab ${active === tab ? "active" : ""}`} key={tab} onClick={() => onChange(tab)}>{tab}</button>)}</div>;
}

function EmptyState({ icon: Icon, title, detail }: { icon: LucideIcon; title: string; detail: string }) {
  return <div className="p-empty" role="status"><Icon className="icon lg" aria-hidden="true" /><h3>{title}</h3><p className="muted small">{detail}</p><span className="p-fixture"><CloudOff className="icon xs" aria-hidden="true" />No items in this view</span></div>;
}

function DeveloperResources({ tab, data, demo }: { tab: string; data?: Record<string, JsonRecord>; demo: boolean }) {
  const client = useQueryClient();
  const [connectionQuery, setConnectionQuery] = useState("");
  const [showAddConnection, setShowAddConnection] = useState(false);
  const [connectionName, setConnectionName] = useState("");
  const [connectionDescription, setConnectionDescription] = useState("");
  const [selectedConnection, setSelectedConnection] = useState<JsonRecord | null>(null);
  const [connectionError, setConnectionError] = useState("");
  const createConnection = useMutation({
    mutationFn: () => advancedDemoAction("/api/developer/connections", { name: connectionName.trim(), description: connectionDescription.trim(), status: "Disconnected", scope: "Read only" }),
    onSuccess: async () => { setConnectionName(""); setConnectionDescription(""); setShowAddConnection(false); setConnectionError(""); await client.invalidateQueries({ queryKey: ["advanced-screen", "developer"] }); },
    onError: error => setConnectionError(apiConnectionMessage(error)),
  });
  const rows = (name: string): JsonRecord[] => {
    const collection: Record<string, string> = { keys: "api_keys", webhooks: "webhooks", deliveries: "deliveries", scopes: "scopes", servers: "servers", plugins: "plugins", extensions: "packages" };
    const value = data?.[name]?.[collection[name] ?? name];
    return Array.isArray(value) ? value.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  };
  if (demo && ["API keys", "Webhooks", "MCP", "Activity"].includes(tab)) {
    const collection = tab === "API keys" ? "api_keys" : tab === "Webhooks" ? "webhooks" : tab === "MCP" ? "servers" : "deliveries";
    const items = tab === "Activity" ? rows("deliveries") : tab === "API keys" ? rows("keys") : tab === "MCP" ? rows("servers") : rows("webhooks");
    const heading = tab === "API keys" ? "Scoped API keys" : tab === "Webhooks" ? "Webhook delivery tests" : tab === "MCP" ? "MCP · demo records" : "Activity · demo records";
    return <><section className="demo-feature-hero"><div className="demo-feature-symbol"><SquareTerminal className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">{tab} · Workspace</span><h2>Connect your tools safely</h2><p>Inspect scoped integrations, keys and deliveries.</p></div></section>
      <DeveloperDemoRecords heading={heading} collection={collection} items={items} onTest={tab === "Webhooks" ? () => setConnectionError("Sample delivery recorded locally; no webhook request was sent.") : undefined} feedback={connectionError} />
    </>;
  }
  if (tab === "API keys") {
    const keys = rows("keys");
    const scopes = rows("scopes");
    return <section className="p-card"><div className="p-card-heading"><h3>Issued API keys</h3><span className="p-fixture">{keys.length} records</span></div>
      {!keys.length ? <p className="muted small">No API keys are registered. New key tokens are shown once after creation.</p> : <div className="p-record-sections">{keys.map((key, index) => <article className="p-record-section" key={String(key.id ?? index)}><h4>{String(key.label ?? "API key")}</h4><p className="caption">{String(key.status ?? "unknown")} · created {String(key.created_at ?? "date unavailable")} · last used {String(key.last_used_at ?? "never")}</p><p className="caption">Scopes: {Array.isArray(key.scopes) ? key.scopes.map(String).join(", ") : "not reported"}</p></article>)}</div>}
      <div className="p-note section-space"><Shield className="icon sm" aria-hidden="true" /><span>{scopes.length ? `${scopes.length} permissions are available from the backend scope catalog. Choose only the permissions this key needs.` : "The backend scope catalog could not be read. Key creation is unavailable until the supported permissions can be verified."}</span></div>
    </section>;
  }
  if (tab === "Webhooks" || tab === "Activity") {
    const webhooks = rows("webhooks");
    const deliveries = rows("deliveries");
    return <>
      {tab === "Webhooks" && <section className="p-card"><div className="p-card-heading"><h3>Registered webhooks</h3><span className="p-fixture">{webhooks.length} records</span></div>
        {!webhooks.length ? <p className="muted small">No webhooks registered.</p> : <div className="p-record-sections">{webhooks.map((webhook, index) => <article className="p-record-section" key={String(webhook.id ?? index)}><h4>{String(webhook.event ?? "*")}</h4><p className="caption">{String(webhook.target_url ?? "Target unavailable")} · {webhook.enabled === true ? "enabled" : "disabled"}</p><p className="caption">Updated {String(webhook.updated_at ?? "date unavailable")}</p></article>)}</div>}
      </section>}
      <section className="p-card"><div className="p-card-heading"><h3>Delivery history</h3><span className="p-fixture">{deliveries.length} records</span></div>
        {!deliveries.length ? <p className="muted small">No delivery records. This screen does not send test events.</p> : <div className="p-record-sections">{deliveries.slice(0, 50).map((delivery, index) => <article className="p-record-section" key={String(delivery.id ?? index)}><h4>{String(delivery.event ?? "Webhook event")}</h4><p className="caption">{String(delivery.status ?? "status unavailable")} · webhook {String(delivery.webhook_id ?? "unknown")} · {String(delivery.created_at ?? "date unavailable")}</p></article>)}</div>}
      </section>
    </>;
  }
  if (tab === "MCP") {
    const servers = rows("servers");
    return <section className="p-card"><div className="p-card-heading"><h3>MCP servers</h3><span className="p-fixture">{servers.length} records</span></div>{!servers.length ? <p className="muted small">No MCP servers registered.</p> : <div className="p-record-sections">{servers.map((server, index) => <article className="p-record-section" key={String(server.id ?? index)}><h4>{String(server.name ?? "MCP server")}</h4><p className="caption">{String(server.status ?? (server.enabled === false ? "disabled" : "registered"))}</p></article>)}</div>}<div className="p-note section-space"><LockKeyhole className="icon sm" aria-hidden="true" /><span>Server registration accepts executable commands and environment values. This workspace does not collect or persist command credentials.</span></div></section>;
  }
  if (tab === "Connections") {
    const status = data?.status ?? {};
    const plugins = rows("plugins");
    const extensions = rows("extensions");
    const connections = Array.isArray(data?.connections?.connections) ? data.connections.connections.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
    if (demo) {
      const filtered = connections.filter(row => `${String(row.name ?? "")} ${String(row.description ?? "")} ${String(row.scope ?? "")}`.toLowerCase().includes(connectionQuery.toLowerCase()));
      return <>
        <section className="demo-feature-hero"><div className="demo-feature-symbol"><SquareTerminal className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">Connections · Workspace</span><h2>Connect your tools safely</h2><p>Inspect scoped integrations, keys and deliveries.</p></div></section>
        <section className="p-card p-tight"><div className="p-card-heading"><h3>Connections · demo records</h3></div>
          <div className="demo-record-toolbar"><label className="search-field demo-record-search"><Search className="icon sm" aria-hidden="true"/><input value={connectionQuery} onChange={event => setConnectionQuery(event.target.value)} placeholder="Search connections…" aria-label="Search connections"/></label><span className="caption">{filtered.length} items</span><button className="btn primary" type="button" onClick={() => { setShowAddConnection(value => !value); setConnectionError(""); }}><Plus className="icon sm"/>Add</button></div>
          {showAddConnection && <form className="p-record-sections" onSubmit={event => { event.preventDefault(); if (connectionName.trim()) createConnection.mutate(); }}><label className="demo-form-field">Connection name<input className="input" value={connectionName} onChange={event => setConnectionName(event.target.value)} required maxLength={100}/></label><label className="demo-form-field">Description<input className="input" value={connectionDescription} onChange={event => setConnectionDescription(event.target.value)} maxLength={180}/></label><div className="row"><button className="btn primary" type="submit" disabled={createConnection.isPending || !connectionName.trim()}>{createConnection.isPending ? "Saving…" : "Save local connection"}</button><button className="btn" type="button" onClick={() => setShowAddConnection(false)}>Cancel</button></div>{connectionError && <p className="caption" role="alert">{connectionError}</p>}</form>}
          {filtered.length ? <div>{filtered.map((row, index) => <article className="demo-record developer-demo-record" key={String(row.id ?? index)}><div className="square-icon"><SquareTerminal className="icon sm" aria-hidden="true"/></div><button type="button" className="demo-record-main" onClick={() => setSelectedConnection(row)}><span className="demo-record-text"><strong>{String(row.name ?? "Connection")}</strong><small>{String(row.description ?? row.scope ?? "Scoped integration")}</small></span></button><span className="tag green">{String(row.status ?? "Status unavailable")}</span><button className="icon-button developer-record-action" type="button" aria-label={`View ${String(row.name ?? "connection")} details`} onClick={() => setSelectedConnection(row)}><Settings2 className="icon sm"/></button><button className="icon-button developer-record-action" type="button" aria-label={`More options for ${String(row.name ?? "connection")}`} onClick={() => setSelectedConnection(row)}><MoreHorizontal className="icon sm"/></button></article>)}</div> : <p className="caption">{connectionQuery ? "No matching connections." : "No demo connections are configured."}</p>}
          {selectedConnection && <section className="p-note section-space" role="status"><Network className="icon sm" aria-hidden="true"/><span><strong>{String(selectedConnection.name ?? "Connection")}</strong> · {String(selectedConnection.scope ?? "Scope not specified")} · {String(selectedConnection.status ?? "Status unavailable")}</span><button type="button" className="btn sm" onClick={() => setSelectedConnection(null)}>Close</button></section>}
        </section>
      </>;
    }
    return <section className="p-card"><div className="p-card-heading"><h3>Developer connections</h3></div><div className="p-record-sections"><p className="caption">Backend platform status: {String(status.status ?? "response received; no status field reported")}</p><p className="caption">Registered plugins: {plugins.length} · extensions: {extensions.length}</p></div><div className="p-note section-space"><Network className="icon sm" aria-hidden="true" /><span>No general integration-connect endpoint is exposed by this service.</span></div></section>;
  }
  return <section className="p-card"><div className="p-card-heading"><h3>{tab}</h3></div><p className="muted small">No matching developer API resource is available.</p></section>;
}

function DeveloperDemoRecords({ heading, collection, items, note, onTest, feedback, icon = SquareTerminal }: { heading: string; collection: string; items: JsonRecord[]; note?: string; onTest?: () => void; feedback?: string; icon?: LucideIcon }) {
  const Icon = icon;
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState<JsonRecord | null>(null);
  const [adding, setAdding] = useState(false);
  const [localItems, setLocalItems] = useState<JsonRecord[]>([]);
  const filtered = items.filter(row => `${String(row.name ?? row.event ?? "")} ${String(row.description ?? row.scope ?? "")}`.toLowerCase().includes(filter.toLowerCase()));
  const visible = [...localItems, ...filtered];
  return <section className="p-card"><div className="p-card-heading"><h3>{heading}</h3></div>
    {note && <div className="p-note"><LockKeyhole className="icon sm" aria-hidden="true"/><span>{note}</span></div>}
    <div className="demo-record-toolbar"><label className="search-field demo-record-search"><Search className="icon sm" aria-hidden="true"/><input value={filter} onChange={event => setFilter(event.target.value)} placeholder={`Search ${collection}…`} aria-label={`Search ${collection}`}/></label><span className="caption">{visible.length} item{visible.length === 1 ? "" : "s"}</span><button className="btn primary sm" type="button" onClick={() => setAdding(value => !value)}><Plus className="icon sm"/>Add</button></div>
    {adding && <form className="p-record-sections" onSubmit={event => { event.preventDefault(); const form = event.currentTarget; const values = new FormData(form); const name = String(values.get("name") ?? "").trim(); if (!name) return; setLocalItems(current => [{ id: `demo-local-${current.length + 1}`, name, description: String(values.get("description") ?? ""), status: "Ready" }, ...current]); form.reset(); setAdding(false); }}><label className="demo-form-field">Name<input className="input" name="name" required maxLength={100}/></label><label className="demo-form-field">Description<input className="input" name="description" maxLength={180}/></label><div className="row"><button className="btn primary" type="submit">Create local example</button><button className="btn" type="button" onClick={() => setAdding(false)}>Cancel</button></div></form>}
    <div className="demo-records">{visible.length ? visible.map((row, index) => <article className="demo-record" key={String(row.id ?? index)}>
      <button type="button" className="demo-record-main" onClick={() => setSelected(selected === row ? null : row)}><span className="square-icon sm blue"><Icon className="icon sm" aria-hidden="true"/></span><span className="demo-record-text"><strong>{String(row.name ?? row.event ?? "Example record")}</strong><small>{String(row.description ?? row.scope ?? row.url ?? "Example record")}</small></span></button>
      <span className={`tag ${/active|available|ready|connected/i.test(String(row.status ?? "")) ? "green" : ""}`}>{String(row.status ?? "Ready")}</span>
      <button className="icon-button" type="button" aria-label={`Details for ${String(row.name ?? "record")}`} onClick={() => setSelected(selected === row ? null : row)}><Settings2 className="icon sm" aria-hidden="true"/></button>
      <button className="icon-button" type="button" aria-label={`More actions for ${String(row.name ?? "record")}`} onClick={() => setSelected(selected === row ? null : row)}><MoreHorizontal className="icon sm" aria-hidden="true"/></button>
    </article>) : <div className="p-empty"><Search className="icon lg" aria-hidden="true"/><h3>No matching items</h3><p className="muted small">Try a different search or create a sample record.</p></div>}</div>
    {selected && <div className="p-note section-space" role="status"><Network className="icon sm" aria-hidden="true"/><span><strong>{String(selected.name ?? selected.event ?? "Example record")}</strong> · {Object.entries(selected).filter(([key]) => !["id", "name", "description"].includes(key)).map(([key, value]) => `${key}: ${String(value)}`).join(" · ")}</span><button className="btn sm" type="button" onClick={() => setSelected(null)}>Close</button></div>}
    {onTest && <div className="row section-space"><button type="button" className="btn primary" onClick={onTest}><Play className="icon sm" aria-hidden="true"/>Send test delivery</button></div>}{feedback && <p className="caption" role="status">{feedback}</p>}
  </section>;
}

function humanize(value: string) { return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function recordTitle(record: JsonRecord) {
  for (const key of ["title", "name", "label", "subject", "url", "action", "kind", "id", "session_id", "run_id", "task_id", "goal", "query"]) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return "Backend record";
}
function recordSummary(record: JsonRecord, route: string) {
  const hidden = route === "twin" ? new Set(["content", "context"]) : route === "scraper" ? new Set(["content", "text", "html", "body", "markdown", "extracted_content", "raw_response"]) : route === "vault" || route === "security" ? new Set(["value", "secret", "ciphertext", "passphrase", "token", "key"]) : new Set<string>();
  const values = Object.entries(record)
    .filter(([key, value]) => !hidden.has(key.toLowerCase()) && !["title", "name", "label", "id"].includes(key.toLowerCase()) && (typeof value === "string" || typeof value === "number" || typeof value === "boolean"))
    .slice(0, 3)
    .map(([key, value]) => `${humanize(key)}: ${String(value).slice(0, 90)}`);
  if (route === "twin" && Array.isArray(record.topics) && record.topics.every((topic) => typeof topic === "string")) {
    values.unshift(`Topics: ${(record.topics as string[]).slice(0, 8).join(", ") || "none extracted"}`);
  }
  return values.join(" · ");
}

function ScienceDataWorkspace({ data }: { data?: Record<string, JsonRecord> }) {
  const client = useQueryClient();
  const [datasetId, setDatasetId] = useState("");
  const [allowExternal, setAllowExternal] = useState(false);
  const [licenseAccepted, setLicenseAccepted] = useState(false);
  const [feedback, setFeedback] = useState("");
  const catalog = Array.isArray(data?.catalog?.datasets) ? data.catalog.datasets.filter((row): row is JsonRecord => Boolean(row) && typeof row === "object" && !Array.isArray(row)) : [];
  const mutation = useMutation({
    mutationFn: ({ action }: { action: "import" | "analyse" }) => apiPost(`/api/science/materials/catalog/${action}`, { dataset_id: datasetId, allow_external: allowExternal, license_accepted: licenseAccepted, project: "default" }),
    onSuccess: async (_result, variables) => { setFeedback(variables.action === "import" ? "The backend recorded the catalog import result." : "The backend returned a dataset analysis."); await client.invalidateQueries({ queryKey: ["advanced-screen", "science"] }); },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const submit = (action: "import" | "analyse") => {
    if (!datasetId || !licenseAccepted) return;
    const externalNote = allowExternal ? " External data retrieval is permitted by this request." : " External retrieval is disabled.";
    if (!window.confirm(`${action === "import" ? "Import" : "Analyse"} this public materials dataset?${externalNote} The service will also enforce dataset licensing.`)) return;
    setFeedback(""); mutation.mutate({ action });
  };
  return <section className="p-card"><div className="p-card-heading"><h3>Materials catalog</h3><span className="p-fixture">{catalog.length} backend datasets</span></div>
    <p className="caption">Catalog metadata is bundled with the service. Importing or analysing a selection is a separate backend action; it does not claim experimental validation.</p>
    {!catalog.length ? <p className="muted small">No catalog datasets were returned by the backend.</p> : <label className="demo-form-field">Dataset<select className="select" value={datasetId} onChange={(event) => setDatasetId(event.target.value)}><option value="">Select a catalog dataset</option>{catalog.map((item, index) => <option key={String(item.dataset_id ?? item.id ?? index)} value={String(item.dataset_id ?? item.id ?? "")}>{String(item.name ?? item.title ?? item.dataset_id ?? "Dataset")} · {String(item.license ?? "license unavailable")}</option>)}</select></label>}
    <label className="row"><input type="checkbox" checked={allowExternal} onChange={(event) => setAllowExternal(event.target.checked)} />Allow external dataset retrieval for this operation</label>
    <label className="row"><input type="checkbox" checked={licenseAccepted} onChange={(event) => setLicenseAccepted(event.target.checked)} />I reviewed and accept the selected dataset license</label>
    <div className="row section-space"><button type="button" className="btn" disabled={mutation.isPending || !datasetId || !licenseAccepted} onClick={() => submit("import")}>{mutation.isPending ? "Working…" : "Import selected data"}</button><button type="button" className="btn primary" disabled={mutation.isPending || !datasetId || !licenseAccepted} onClick={() => submit("analyse")}>Analyse selected data</button></div>
    {feedback && <p className="caption" role={mutation.isError ? "alert" : "status"}>{feedback}</p>}{mutation.data && <ActionResult route="science" result={mutation.data} />}
  </section>;
}

const demoWorkspaceCollection: Record<string, Record<string, string>> = {
  science: { Projects: "projects", Experiments: "experiments", Data: "datasets", Reports: "reports" },
  scraper: { "New extraction": "jobs", Jobs: "jobs", Results: "results", Rules: "rules" },
};

function AdvancedDemoRecordWorkspace({ route, tab, data }: { route: "science" | "scraper"; tab: string; data?: Record<string, JsonRecord> }) {
  const client = useQueryClient();
  const collection = demoWorkspaceCollection[route][tab] ?? "items";
  const rows = Array.isArray(data?.[collection]?.[collection]) ? data[collection][collection] as JsonRecord[] : [];
  const [filter, setFilter] = useState("");
  const [dialog, setDialog] = useState<"create" | "edit" | "details" | "actions" | "confirm-delete" | null>(null);
  const [selected, setSelected] = useState<JsonRecord | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [status, setStatus] = useState("Ready · demo");
  const [feedback, setFeedback] = useState("");
  const [traceStartSignal, setTraceStartSignal] = useState(0);
  const [permission, setPermission] = useState(false);
  const [experimentDataset, setExperimentDataset] = useState("");
  const [experimentMethod, setExperimentMethod] = useState("Baseline evaluation");
  const [scraperFormat, setScraperFormat] = useState("Text and metadata");
  const [scraperLimit, setScraperLimit] = useState("5 pages");
  const titleOf = (row: JsonRecord) => String(row.name ?? row.url ?? row.title ?? "Example record");
  useEffect(() => {
    if (experimentDataset) return;
    const datasets = data?.datasets?.datasets;
    if (Array.isArray(datasets) && datasets.length > 0) setExperimentDataset(String((datasets[0] as JsonRecord).name ?? ""));
  }, [data, experimentDataset]);
  const visible = rows.filter(row => `${titleOf(row)} ${String(row.description ?? "")}`.toLowerCase().includes(filter.trim().toLowerCase()));
  const createTitle = route === "scraper" && tab === "New extraction" ? "Set up a permitted extraction" : tab === "Data" ? "Datasets" : tab;
  const isScienceExperiment = route === "science" && tab === "Experiments";
  const isScraperExtraction = route === "scraper" && tab === "New extraction";
  const resetForm = (row?: JsonRecord) => { setSelected(row ?? null); setName(row ? titleOf(row) : ""); setDescription(String(row?.description ?? "")); setStatus(String(row?.status ?? "Ready · demo")); setFeedback(""); setDialog(row ? "edit" : "create"); };
  const save = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!name.trim()) return;
    if (route === "scraper" && tab === "New extraction" && !permission) { setFeedback("Confirm that you have permission before creating this local extraction preview."); return; }
    const id = String(selected?.id ?? `demo-${route}-${collection}-${Date.now().toString(36)}`);
    const saved: JsonRecord = { ...selected, id, name: name.trim(), description: description.trim(), status: route === "scraper" && tab === "New extraction" ? "Completed · demo" : status };
    if (route === "scraper" && tab === "New extraction") saved.description = `${description.trim() || "Example extraction"} · no network request sent`;
    await demoProvider.updateCollection<JsonRecord>(`collections.advanced.${route}.${collection}`, current => selected ? current.map(row => String(row.id) === id ? saved : row) : [saved, ...current]);
    if (route === "scraper" && tab === "New extraction") {
      await demoProvider.updateCollection<JsonRecord>("collections.advanced.scraper.results", current => [{ id: `${id}-result`, name: `${name.trim()} · JSON preview`, description: "Illustrative extracted fields · no network request was sent", status: "Available · demo", count: 0 }, ...current]);
    }
    await client.invalidateQueries({ queryKey: ["advanced-screen", route] });
    setFeedback(route === "scraper" && tab === "New extraction" ? "Local extraction preview saved. No network request was sent." : "Example record saved locally.");
    setDialog(null); setSelected(null); setName(""); setDescription(""); setPermission(false);
  };
  const runCanonicalDemo = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const values = new FormData(event.currentTarget);
    const isScraperPreview = route === "scraper" && tab === "New extraction";
    if (isScraperPreview && !permission) {
      setFeedback("Confirm that you have permission before creating this local extraction preview.");
      return;
    }
    const dataset = String(values.get("dataset") ?? experimentDataset);
    const title = isScraperPreview ? String(values.get("url") ?? "").trim() : String(values.get("name") ?? "").trim();
    if (!title) return;
    const id = `demo-${route}-${collection}-${Date.now().toString(36)}`;
    const detail = isScraperPreview
      ? `format: ${scraperFormat} · limit: ${scraperLimit} · permission: confirmed · no network request sent`
      : `dataset: ${dataset} · method: ${experimentMethod} · local example only`;
    const job: JsonRecord = { id, name: title, description: detail, status: "Completed · demo" };
    await demoProvider.updateCollection<JsonRecord>(`collections.advanced.${route}.${collection}`, current => [job, ...current]);
    if (isScraperPreview) {
      await demoProvider.updateCollection<JsonRecord>("collections.advanced.scraper.results", current => [{ id: `${id}-result`, name: `${title} · JSON preview`, description: "Illustrative extracted fields · no network request was sent", status: "Available · demo", count: 0 }, ...current]);
    } else {
      await demoProvider.updateCollection<JsonRecord>("collections.advanced.science.reports", current => [{ id: `${id}-report`, name: `${title} report`, description: "Example metrics and method summary · no scientific service ran", status: "Draft · demo" }, ...current]);
    }
    await client.invalidateQueries({ queryKey: ["advanced-screen", route] });
    setFeedback(isScraperPreview ? "Local extraction preview saved. No network request was sent." : "Demo experiment saved locally. No scientific service ran.");
    setTraceStartSignal(value => value + 1);
    setName(""); setPermission(false);
  };
  const remove = async () => {
    if (!selected) return;
    if (!window.confirm(`Delete “${titleOf(selected)}” from the local demo?`)) return;
    await demoProvider.updateCollection<JsonRecord>(`collections.advanced.${route}.${collection}`, current => current.filter(row => String(row.id) !== String(selected.id)));
    await client.invalidateQueries({ queryKey: ["advanced-screen", route] });
    setDialog(null); setFeedback("Example record deleted locally.");
  };
  const runExample = async () => {
    if (!selected) return;
    await demoProvider.updateCollection<JsonRecord>(`collections.advanced.${route}.${collection}`, current => current.map(row => String(row.id) === String(selected.id) ? { ...row, status: "Completed · demo" } : row));
    await client.invalidateQueries({ queryKey: ["advanced-screen", route] });
    setDialog(null); setFeedback("Example workflow completed locally; no backend service ran.");
    setTraceStartSignal(value => value + 1);
  };

  return <>
    {traceStartSignal > 0 && <ExecutionTrace route={route} startSignal={traceStartSignal}/>}
    <section className="demo-feature-hero"><div className="demo-feature-symbol"><span className="icon xl">{route === "science" ? <FlaskConical aria-hidden="true"/> : <Globe2 aria-hidden="true"/>}</span></div><div className="flex1"><span className="p-eyebrow">{tab} · Workspace</span><h2>{route === "science" ? "From hypothesis to results" : "Structured web extraction"}</h2><p>{route === "science" ? "Keep experiments, datasets and results reproducible." : "Respect permissions, review the scope and track each result."}</p></div></section>
    <section className="p-card">
      <div className="p-card-heading"><h3>{isScienceExperiment ? "Run an experiment" : isScraperExtraction ? createTitle : `${createTitle} · demo records`}</h3>{!isScienceExperiment && !isScraperExtraction && <span className="p-fixture">{rows.length} sample records</span>}</div>
      {isScienceExperiment && <form className="demo-settings-form" onSubmit={runCanonicalDemo}>
        <div className="p-form-grid"><label className="p-field">Experiment name<input className="input" name="name" required defaultValue="Calibration comparison"/></label><label className="p-field">Dataset<select className="select" name="dataset" value={experimentDataset} onChange={event => setExperimentDataset(event.target.value)}>{(Array.isArray(data?.datasets?.datasets) ? data.datasets.datasets as JsonRecord[] : []).map((dataset, index) => <option key={String(dataset.id ?? index)} value={titleOf(dataset)}>{titleOf(dataset)}</option>)}</select></label></div>
        <label className="p-field">Method<select className="select" name="method" value={experimentMethod} onChange={event => setExperimentMethod(event.target.value)}><option>Baseline evaluation</option><option>Calibration analysis</option><option>Ablation study</option></select></label>
        <button className="btn primary" type="submit"><Play className="icon sm" aria-hidden="true"/>Run demo experiment</button>
      </form>}
      {isScraperExtraction && <form className="demo-settings-form" onSubmit={runCanonicalDemo}>
        <label className="p-field">Page URL<input className="input" name="url" type="url" required defaultValue="https://docs.example.org/getting-started" onChange={() => setFeedback("")}/></label>
        <div className="p-form-grid"><label className="p-field">Extraction type<select className="select" name="format" value={scraperFormat} onChange={event => setScraperFormat(event.target.value)}><option>Text and metadata</option><option>Structured records</option><option>Links only</option></select></label><label className="p-field">Page limit<select className="select" name="limit" value={scraperLimit} onChange={event => setScraperLimit(event.target.value)}><option>5 pages</option><option>10 pages</option><option>25 pages</option></select></label></div>
        <label className="row small"><input type="checkbox" required checked={permission} onChange={event => setPermission(event.target.checked)}/> I confirm I have permission to extract this example content.</label>
        <button className="btn primary" type="submit"><Play className="icon sm" aria-hidden="true"/>Start demo extraction</button>
      </form>}
      <div className="demo-record-toolbar"><label className="search-field demo-record-search"><Search className="icon sm" aria-hidden="true"/><input value={filter} onChange={event => setFilter(event.target.value)} placeholder={`Search ${collection}…`} aria-label={`Search ${collection}`}/></label><span className="caption">{visible.length} item{visible.length === 1 ? "" : "s"}</span><button type="button" className="btn primary sm" onClick={() => resetForm()}><Plus className="icon sm" aria-hidden="true"/>Add</button></div>
      <div className="demo-records">{visible.length ? visible.map((row, index) => <article className="demo-record" key={String(row.id ?? index)}><button type="button" className="demo-record-main" onClick={() => { setSelected(row); setDialog("details"); }}><span className="square-icon sm blue">{route === "science" ? <FlaskConical className="icon sm" aria-hidden="true"/> : <Globe2 className="icon sm" aria-hidden="true"/>}</span><span className="demo-record-text"><strong>{titleOf(row)}</strong><small>{String(row.description ?? "Local demonstration record")}</small></span></button><span className={`tag ${/completed|active|validated|available|enabled|published/i.test(String(row.status ?? "")) ? "green" : ""}`}>{String(row.status ?? "Ready")}</span><button type="button" className="icon-button" aria-label={`Edit ${titleOf(row)}`} onClick={() => resetForm(row)}><Settings2 className="icon sm" aria-hidden="true"/></button><button type="button" className="icon-button" aria-label={`More actions for ${titleOf(row)}`} onClick={() => { setSelected(row); setDialog("actions"); }}><MoreHorizontal className="icon sm" aria-hidden="true"/></button></article>) : <div className="p-empty"><Search className="icon lg" aria-hidden="true"/><h3>No matching items</h3><p>Try a different search or add a sample record.</p></div>}</div>
      {feedback && !dialog && <p className="caption" role="status">{feedback}</p>}
    </section>
    <Dialog.Root open={dialog !== null} onOpenChange={open => { if (!open) setDialog(null); }}><Dialog.Portal><Dialog.Overlay/><Dialog.Content className="advanced-integration-dialog"><div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">{dialog === "details" ? String(selected?.name ?? "Record details") : dialog === "actions" ? "Record actions" : dialog === "edit" ? `Edit ${createTitle.toLowerCase()}` : `New ${createTitle.toLowerCase()}`}</Dialog.Title><Dialog.Close asChild><button className="icon-button" aria-label="Close dialog" type="button"><X size={16}/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">{dialog === "details" ? "Example data stored in this isolated local demonstration." : "Changes are saved locally in this interactive demonstration."}</Dialog.Description>
      {dialog === "details" && selected && <><div className="demo-detail-grid">{Object.entries(selected).filter(([key]) => key !== "id").map(([key, value]) => <div key={key}><small>{humanize(key)}</small><strong>{String(value)}</strong></div>)}</div><div className="advanced-integration-actions"><button className="btn" type="button" onClick={() => setDialog("actions")}>More actions</button><button className="btn" type="button" onClick={() => setDialog(null)}>Close</button></div></>}
      {dialog === "actions" && selected && <div className="demo-action-grid"><button className="btn" type="button" onClick={() => setDialog("details")}>Open details</button><button className="btn" type="button" onClick={() => resetForm(selected)}>Edit record</button><button className="btn" type="button" onClick={() => void runExample()}>Run example</button><button className="btn danger" type="button" onClick={() => void remove()}>Delete</button></div>}
      {(dialog === "create" || dialog === "edit") && <form onSubmit={save}><label className="demo-form-field">Name<input className="input" required maxLength={160} value={name} onChange={event => setName(event.target.value)} placeholder="Short, meaningful name"/></label><label className="demo-form-field">Description<textarea className="textarea" rows={3} maxLength={500} value={description} onChange={event => setDescription(event.target.value)} placeholder="What is this for?"/></label><label className="demo-form-field">Status<input className="input" value={status} onChange={event => setStatus(event.target.value)} maxLength={80}/></label><div className="advanced-integration-actions"><button className="btn" type="button" onClick={() => setDialog(null)}>Cancel</button><button className="btn primary" type="submit">{dialog === "edit" ? "Save changes" : "Create item"}</button></div></form>}
    </Dialog.Content></Dialog.Portal></Dialog.Root>
  </>;
}

function TwinOperations({ tab, data }: { tab: string; data?: Record<string, JsonRecord> }) {
  const client = useQueryClient();
  const [context, setContext] = useState("");
  const [goal, setGoal] = useState("");
  const [feedback, setFeedback] = useState("");
  const [result, setResult] = useState<JsonRecord>({});
  const mutation = useMutation({
    mutationFn: ({ path, body }: { path: string; body: unknown }) => apiPost(path, body),
    onSuccess: async (response, variables) => { setResult(response); setFeedback(variables.path.endsWith("/predict") ? "Prediction returned by the context service. Treat it as a suggestion, not a fact." : variables.path.endsWith("/consolidate") ? "The backend returned its memory-consolidation result." : "The backend saved the selected personal goal."); await client.invalidateQueries({ queryKey: ["advanced-screen", "twin"] }); },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const predictions = tab === "Preferences";
  const controls = tab === "Data controls";
  if (!predictions && !controls) return null;
  return <section className="p-card"><div className="p-card-heading"><h3>{predictions ? "Ask for a context suggestion" : "Personal data controls"}</h3></div>
    {predictions ? <form onSubmit={(event) => { event.preventDefault(); setFeedback(""); setResult({}); mutation.mutate({ path: "/api/twin/predict", body: { context: context.trim(), project: "default" } }); }}><label className="demo-form-field">Context to consider<textarea className="input" rows={3} value={context} onChange={(event) => setContext(event.target.value)} required /></label><p className="caption">Only the context you enter is sent. The result is an inferred suggestion and may be wrong.</p><button className="btn primary" disabled={mutation.isPending || !context.trim()}>{mutation.isPending ? "Requesting suggestion…" : "Get suggestion"}</button></form> : <><form onSubmit={(event) => { event.preventDefault(); setFeedback(""); setResult({}); mutation.mutate({ path: "/api/twin/goals", body: { title: goal.trim(), project: "default" } }); }}><label className="demo-form-field">Goal you choose to track<input className="input" value={goal} onChange={(event) => setGoal(event.target.value)} required /></label><button className="btn" disabled={mutation.isPending || !goal.trim()}>{mutation.isPending ? "Saving…" : "Save personal goal"}</button></form><div className="p-record-section section-space"><p className="caption">Consolidation may update stored personal context. Review the backend response after the operation.</p><button type="button" className="btn" disabled={mutation.isPending} onClick={() => { if (!window.confirm("Consolidate this project's personal context memory now?")) return; setFeedback(""); setResult({}); mutation.mutate({ path: "/api/twin/consolidate", body: { project: "default" } }); }}>Consolidate context memory</button></div></>}
    {feedback && <p className="caption" role={mutation.isError ? "alert" : "status"}>{feedback}</p>}{Object.keys(result).length > 0 && <ActionResult route="twin" result={result} />}
    <span className="sr-only">Loaded goals: {Array.isArray(data?.goals?.goals) ? data.goals.goals.length : 0}</span>
  </section>;
}

function TwinDemoProfile({ data }: { data?: Record<string, JsonRecord> }) {
  const client = useQueryClient();
  const [filter, setFilter] = useState("");
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});
  const [showAdd, setShowAdd] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [error, setError] = useState("");
  const preferences = Array.isArray(data?.preferences?.preferences)
    ? data.preferences.preferences.filter((row): row is JsonRecord => Boolean(row) && typeof row === "object" && !Array.isArray(row))
    : [];
  const matching = preferences.filter((row) => `${String(row.name ?? "")} ${String(row.description ?? "")}`.toLowerCase().includes(filter.trim().toLowerCase()));
  return <>
    <section className="demo-feature-hero twin-demo-hero"><div className="demo-feature-symbol"><UserRound className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">Profile · Workspace</span><h2>Your context, your choice</h2><p>Review what information is available to the assistant.</p></div></section>
    <section className="p-card twin-demo-profile">
      <div className="p-card-heading"><h3>Context profile</h3></div>
      <div className="twin-demo-identity"><span className="twin-demo-initial">P</span><div><strong>Personal Workspace</strong><p>Your local example identity, preferences and source controls.</p></div></div>
      <div className="demo-record-toolbar twin-demo-profile-toolbar"><label className="search-field demo-record-search"><SearchIcon className="icon sm" aria-hidden="true" /><input aria-label="Search preferences" placeholder="Search preferences…" value={filter} onChange={(event) => setFilter(event.target.value)} /></label><span className="caption">{matching.length} items</span><button type="button" className="btn primary sm" onClick={() => { setShowAdd(true); setError(""); }}><Plus className="icon sm" aria-hidden="true" />Add</button></div>
      {showAdd && <form className="twin-demo-add" onSubmit={async (event) => { event.preventDefault(); setError(""); try { await advancedDemoAction("/api/twin/preferences", { name: newName.trim(), description: newDescription.trim(), status: "On" }); setNewName(""); setNewDescription(""); setShowAdd(false); await client.invalidateQueries({ queryKey: ["advanced-screen", "twin"] }); } catch (cause) { setError(apiConnectionMessage(cause)); } }}><label className="demo-form-field">Preference name<input className="input" required value={newName} onChange={(event) => setNewName(event.target.value)} /></label><label className="demo-form-field">Description<input className="input" required value={newDescription} onChange={(event) => setNewDescription(event.target.value)} /></label><div className="row"><button className="btn primary" disabled={!newName.trim() || !newDescription.trim()}>Save preference</button><button className="btn" type="button" onClick={() => setShowAdd(false)}>Cancel</button></div></form>}
    {error && <p className="caption" role="alert">{error}</p>}
      <div className="twin-demo-preferences">{matching.map((row, index) => { const id = String(row.id ?? index); const on = enabled[id] ?? String(row.status ?? "On") === "On"; return <article className="p-item" key={id}><div className="square-icon"><UserRound className="icon sm" aria-hidden="true" /></div><div className="p-item-copy"><strong>{String(row.name ?? "Preference")}</strong><small>{String(row.description ?? "")}</small></div><button type="button" className={`p-item-meta twin-demo-toggle${on ? " is-on" : ""}`} aria-pressed={on} onClick={() => setEnabled((current) => ({ ...current, [id]: !on }))}>{on ? "On" : "Off"}</button><button type="button" className="icon-button" aria-label={`Configure ${String(row.name ?? "preference")}`} onClick={() => setEnabled((current) => ({ ...current, [id]: !on }))}><SlidersHorizontal className="icon sm" aria-hidden="true" /></button><button type="button" className="icon-button" aria-label={`More options for ${String(row.name ?? "preference")}`} onClick={() => setFilter(String(row.name ?? ""))}>···</button></article>; })}</div>
    </section>
  </>;
}

function TwinDemoWorkspace({ tab, data }: { tab: string; data?: Record<string, JsonRecord> }) {
  const client = useQueryClient();
  const [selected, setSelected] = useState<JsonRecord | null>(null);
  const [editorRecord, setEditorRecord] = useState<JsonRecord | null>(null);
  const [editorOpen, setEditorOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [status, setStatus] = useState("Enabled");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const sources = Array.isArray(data?.sources?.sources) ? data.sources.sources as JsonRecord[] : [];
  const preferences = Array.isArray(data?.preferences?.preferences) ? data.preferences.preferences as JsonRecord[] : [];
  const history = Array.isArray(data?.history?.history) ? data.history.history as JsonRecord[] : [];
  const collection = tab === "Sources" ? "sources" : tab === "Preferences" ? "preferences" : "history";
  const records = tab === "Sources" ? sources : tab === "Preferences" ? preferences : history;
  const filtered = records.filter(row => `${String(row.name ?? "")} ${String(row.description ?? "")} ${String(row.status ?? "")}`.toLowerCase().includes(search.trim().toLowerCase()));
  const title = tab === "Data controls" ? "Your data controls" : `${tab} · demo records`;
  const beginCreate = () => { setEditorRecord(null); setName(""); setDescription(""); setStatus(collection === "history" ? "Recorded" : "Enabled"); setError(""); setEditorOpen(true); };
  const beginEdit = (record: JsonRecord) => { setEditorRecord(record); setName(String(record.name ?? "")); setDescription(String(record.description ?? "")); setStatus(String(record.status ?? "Enabled")); setError(""); setEditorOpen(true); setSelected(null); };
  const saveRecord = async (event: FormEvent) => {
    event.preventDefault();
    const cleanName = name.trim();
    if (!cleanName) { setError("Enter a name before saving."); return; }
    try {
      const next: JsonRecord = { ...(editorRecord ?? {}), id: String(editorRecord?.id ?? `demo-twin-${collection}-${Date.now().toString(36)}`), name: cleanName, description: description.trim(), status };
      await demoProvider.updateCollection<JsonRecord>(`collections.advanced.twin.${collection}`, current => editorRecord ? current.map(row => row.id === editorRecord.id ? next : row) : [next, ...current]);
      await recordDemoWorkspaceActivity("twin", editorRecord ? "Updated personal context example" : "Added personal context example", collection);
      setEditorOpen(false); setNotice(`${cleanName} saved in this browser's demo data.`); setError("");
      await client.invalidateQueries({ queryKey: ["advanced-screen", "twin"] });
    } catch (cause) { setError(apiConnectionMessage(cause)); }
  };
  const deleteRecord = async (record: JsonRecord) => {
    if (!window.confirm(`Delete “${String(record.name ?? "this item")}” from the local demo? No backend data will be changed.`)) return;
    await demoProvider.updateCollection<JsonRecord>(`collections.advanced.twin.${collection}`, current => current.filter(row => row.id !== record.id));
    await recordDemoWorkspaceActivity("twin", "Deleted personal context example", collection);
    setSelected(null); setNotice("Local demo record deleted."); await client.invalidateQueries({ queryKey: ["advanced-screen", "twin"] });
  };
  const exportAll = async () => {
    const payload = await demoProvider.get("collections.advanced");
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = "openagent-demo-data.json"; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice("All local demo collections were exported. No backend data was included.");
  };
  const resetAll = async () => {
    if (!window.confirm("Reset all local demo data to the approved sample fixtures? Live backend data will not be changed.")) return;
    await demoProvider.reset(); await client.invalidateQueries(); setNotice("Local demo data was reset to the approved fixtures.");
  };
  return <>
    <section className="demo-feature-hero twin-demo-hero"><div className="demo-feature-symbol"><UserRound className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">{tab} · Workspace</span><h2 style={{ fontWeight: 700 }}>Your context, your choice</h2><p>Review what information is available to the assistant.</p></div></section>
    <section className={`p-card twin-demo-actions${tab === "Profile" ? " twin-demo-profile" : ""}`}>
      <div className="p-card-heading"><h3>{tab === "Profile" ? "Context profile" : title}</h3></div>
      {tab === "Profile" && <div className="twin-demo-identity"><span className="twin-demo-initial">P</span><div><strong>Personal Workspace</strong><p>Your local example identity, preferences and source controls.</p></div></div>}
      {tab === "Data controls" && <>
        <div className="p-note"><Shield className="icon sm" aria-hidden="true"/><span>In this demo, data is stored only in this browser. You can export it or reset all fixtures.</span></div>
        <div className="row section-space"><button type="button" className="btn primary" onClick={() => void exportAll()}><FileDown className="icon sm" aria-hidden="true"/>Export all demo data</button><button type="button" className="btn danger" onClick={() => void resetAll()}><RotateCcw className="icon sm" aria-hidden="true"/>Reset demo data</button></div>
      </>}
      <div className="demo-record-toolbar twin-demo-record-toolbar"><label className="search-field demo-record-search"><SearchIcon className="icon sm" aria-hidden="true"/><input aria-label={`Search ${collection}`} placeholder={`Search ${collection}…`} value={search} onChange={event => setSearch(event.target.value)}/></label><span className="caption">{filtered.length} item{filtered.length === 1 ? "" : "s"}</span><button type="button" className="btn primary sm" onClick={beginCreate}><Plus className="icon sm" aria-hidden="true"/>Add</button></div>
      <div className="demo-records">{filtered.length ? filtered.map((record, index) => {
        const id = String(record.id ?? index); const rowName = String(record.name ?? (tab === "Sources" ? "Interaction source" : "Example record"));
        return <article className="demo-record" key={id}>
          <button type="button" className="demo-record-main" onClick={() => setSelected(record)}><span className="square-icon sm blue"><UserRound className="icon sm" aria-hidden="true"/></span><span className="demo-record-text"><strong>{rowName}</strong><small>{String(record.description ?? record.time ?? "Example record")}</small></span></button>
          <span className={`tag ${/enabled|recorded|on/i.test(String(record.status ?? "")) ? "green" : ""}`}>{String(record.status ?? "Ready")}</span>
          <button type="button" className="icon-button" aria-label={`Edit ${rowName}`} onClick={() => beginEdit(record)}><SlidersHorizontal className="icon sm" aria-hidden="true"/></button>
          <button type="button" className="icon-button" aria-label={`More actions for ${rowName}`} onClick={() => setSelected(record)}><MoreHorizontal className="icon sm" aria-hidden="true"/></button>
        </article>;
      }) : <div className="p-empty"><SearchIcon className="icon lg" aria-hidden="true"/><h3>No matching items</h3><p className="muted small">Try a different search or create a sample record.</p></div>}</div>
      {notice && <p className="caption section-space" role="status">{notice}</p>}{error && <p className="caption" role="alert">{error}</p>}
    </section>
    <Dialog.Root open={editorOpen} onOpenChange={setEditorOpen}><Dialog.Portal><Dialog.Overlay/><Dialog.Content className="advanced-integration-dialog"><div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">{editorRecord ? "Edit" : "New"} {collection === "sources" ? "source" : collection === "history" ? "history item" : "preference"}</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close editor"><X size={16} aria-hidden="true"/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">Changes are saved to this browser's deterministic demo data only.</Dialog.Description><form onSubmit={event => void saveRecord(event)}><label className="demo-form-field">Name<input className="input" value={name} onChange={event => setName(event.target.value)} required maxLength={100}/></label><label className="demo-form-field">Description<textarea className="textarea" rows={3} value={description} onChange={event => setDescription(event.target.value)} maxLength={240}/></label><label className="demo-form-field">Status<select className="select" value={status} onChange={event => setStatus(event.target.value)}>{(collection === "history" ? ["Recorded", "Reviewed"] : ["Enabled", "Disabled"]).map(value => <option key={value}>{value}</option>)}</select></label>{error && <p className="caption" role="alert">{error}</p>}<div className="advanced-integration-actions"><button className="btn" type="button" onClick={() => setEditorOpen(false)}>Cancel</button><button className="btn primary" type="submit" disabled={!name.trim()}>Save locally</button></div></form></Dialog.Content></Dialog.Portal></Dialog.Root>
    <Dialog.Root open={Boolean(selected)} onOpenChange={open => { if (!open) setSelected(null); }}><Dialog.Portal><Dialog.Overlay/><Dialog.Content className="advanced-integration-dialog"><div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">{String(selected?.name ?? "Context details")}</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close context details"><X size={16} aria-hidden="true"/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">{String(selected?.description ?? "Example context details")}</Dialog.Description><div className="demo-detail-grid">{Object.entries(selected ?? {}).filter(([key]) => !["id", "name", "description"].includes(key)).map(([key, value]) => <div key={key}><small>{humanize(key)}</small><strong>{String(value)}</strong></div>)}</div><div className="p-note"><Info className="icon sm" aria-hidden="true"/><span>Local example data only. This action does not read or update backend personal facts.</span></div><div className="advanced-integration-actions"><button type="button" className="btn danger" onClick={() => selected && void deleteRecord(selected)}>Delete record</button><button type="button" className="btn" onClick={() => selected && beginEdit(selected)}>Edit</button><Dialog.Close asChild><button className="btn" type="button">Close</button></Dialog.Close></div></Dialog.Content></Dialog.Portal></Dialog.Root>
  </>;
}

function SecurityDemoWorkspace({ tab, data }: { tab: string; data?: Record<string, JsonRecord> }) {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [showAdd, setShowAdd] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [notice, setNotice] = useState("");
  const [selected, setSelected] = useState<JsonRecord | null>(null);
  const [actionRecord, setActionRecord] = useState<JsonRecord | null>(null);
  const [editingRecord, setEditingRecord] = useState<JsonRecord | null>(null);
  const [editName, setEditName] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [editStatus, setEditStatus] = useState("");
  const [deleteRecord, setDeleteRecord] = useState<JsonRecord | null>(null);
  const [policies, setPolicies] = useState<JsonRecord[]>([]);
  const [sessions, setSessions] = useState<JsonRecord[]>([]);
  const [events, setEvents] = useState<JsonRecord[]>([]);
  useEffect(() => {
    setPolicies(Array.isArray(data?.policies?.policies) ? data.policies.policies as JsonRecord[] : []);
    setSessions(Array.isArray(data?.sessions?.sessions) ? data.sessions.sessions as JsonRecord[] : []);
    setEvents(Array.isArray(data?.events?.events) ? data.events.events as JsonRecord[] : []);
  }, [data]);
  const records = tab === "Sessions" ? sessions : tab === "Audit" ? events : policies;
  const collection = tab === "Sessions" ? "sessions" : tab === "Audit" ? "audit" : "policies";
  const filtered = records.filter(row => `${row.name ?? ""} ${row.description ?? ""} ${row.status ?? ""}`.toLowerCase().includes(search.trim().toLowerCase()));
  const title = tab === "Access" ? "Access" : tab === "Sessions" ? "Sessions" : "Audit";
  const addLabel = tab === "Access" ? "policy" : tab === "Sessions" ? "session" : "audit note";
  const noMatchingRecords = <div className="p-empty" role="status"><SearchIcon className="icon lg" aria-hidden="true"/><h3>No matching items</h3><p>Try a different search or create a sample record.</p></div>;
  const setRecords = (next: JsonRecord[]) => { if (collection === "sessions") setSessions(next); else if (collection === "audit") setEvents(next); else setPolicies(next); };
  const persistRecords = async (update: (current: JsonRecord[]) => JsonRecord[], message: string, activity = "Updated security example") => {
    const next = await demoProvider.updateCollection<JsonRecord>(`collections.advanced.security.${collection}`, update);
    setRecords(next);
    setNotice(message);
    await recordDemoWorkspaceActivity("security", activity, collection);
    await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "security"] });
  };
  const recordId = (record: JsonRecord) => String(record.id ?? "");
  const newRecordId = () => globalThis.crypto?.randomUUID?.() ?? `demo-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  const addRecord = async (event: FormEvent) => {
    event.preventDefault(); if (!name.trim()) return;
    const record = { id: newRecordId(), name: name.trim(), description: description.trim() || "Example record created in this browser", status: collection === "policies" ? "Enabled" : collection === "sessions" ? "Active" : "Example" };
    await persistRecords(current => [record, ...current], `Local demo ${addLabel} added. No security service was changed.`, `Added security ${addLabel}`);
    setName(""); setDescription(""); setShowAdd(false);
  };
  const openEdit = (record: JsonRecord) => { setEditingRecord(record); setEditName(String(record.name ?? "")); setEditDescription(String(record.description ?? "")); setEditStatus(String(record.status ?? "Example")); setActionRecord(null); setSelected(null); };
  const saveEdit = async (event: FormEvent) => {
    event.preventDefault(); if (!editingRecord || !editName.trim()) return;
    const id = recordId(editingRecord);
    await persistRecords(current => current.map(row => recordId(row) === id ? { ...row, name: editName.trim(), description: editDescription.trim(), status: editStatus } : row), "Example record updated locally. No security service was changed.", `Updated security ${addLabel}`);
    setEditingRecord(null);
  };
  const duplicateRecord = async (record: JsonRecord) => {
    const duplicate = { ...record, id: newRecordId(), name: `${String(record.name ?? addLabel)} (copy)` };
    await persistRecords(current => [duplicate, ...current], "Duplicated sample item locally.", `Duplicated security ${addLabel}`);
    setActionRecord(null);
  };
  const changeRecordStatus = async (record: JsonRecord) => {
    const status = collection === "policies" ? String(record.status).toLowerCase() === "enabled" ? "Disabled" : "Enabled" : collection === "sessions" ? "Revoked" : "Simulated locally";
    const id = recordId(record);
    await persistRecords(current => current.map(row => recordId(row) === id ? { ...row, status } : row), collection === "audit" ? "Example check simulated locally. No security service was contacted." : `Example ${collection === "policies" ? "policy" : "session"} marked ${status.toLowerCase()} locally. No security service was changed.`, `Changed security ${addLabel} status`);
    setSelected(null); setActionRecord(null);
  };
  const confirmDelete = async () => {
    if (!deleteRecord) return;
    const id = recordId(deleteRecord);
    await persistRecords(current => current.filter(row => recordId(row) !== id), "Demo record deleted locally. No backend data was changed.", `Deleted security ${addLabel}`);
    setDeleteRecord(null); setSelected(null); setActionRecord(null);
  };
  const showDetails = (record: JsonRecord) => { setActionRecord(null); setSelected(record); };
  const beginDelete = (record: JsonRecord) => { setSelected(null); setActionRecord(null); setDeleteRecord(record); };
  return <>
    <section className="demo-feature-hero security-demo-hero"><div className="demo-feature-symbol"><Shield className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">{tab} · Workspace</span><h2 style={{ fontWeight: 700 }}>Clear visibility and control</h2><p>Understand policy, approvals and activity.</p></div></section>
    {tab === "Overview" ? <section className="p-card p-tight security-demo-summary"><div className="p-card-heading"><h3>Security summary</h3></div><div className="demo-stats-grid"><div><strong>{policies.length}</strong><small>Active policies</small></div><div><strong>{sessions.length}</strong><small>Sessions</small></div><div><strong>{events.length}</strong><small>Audit entries</small></div></div><div className="p-note"><Shield className="icon sm" aria-hidden="true"/><span>This is a simulated security dashboard: no real authorisation, session revocation or audit verification takes place.</span></div><div className="demo-record-toolbar security-demo-toolbar"><label className="search-field demo-record-search"><SearchIcon className="icon sm" aria-hidden="true"/><input aria-label="Search policies" placeholder="Search policies..." value={search} onChange={event => setSearch(event.target.value)}/></label><span className="caption">{filtered.length} items</span><button className="btn primary sm" type="button" onClick={() => { setShowAdd(true); setNotice(""); }}><Plus className="icon sm" aria-hidden="true"/>Add</button></div>{filtered.length ? filtered.map((row, index) => <article className="p-item security-demo-record" key={String(row.id ?? index)}><div className="square-icon"><Shield className="icon sm" aria-hidden="true"/></div><div className="p-item-copy"><strong>{String(row.name ?? "Access policy")}</strong><small>{String(row.description ?? "")}</small></div><span className={`tag ${String(row.status).toLowerCase() === "enabled" ? "green" : ""}`}>{String(row.status ?? "Enabled")}</span><button className="icon-button" type="button" aria-label={`Configure ${String(row.name ?? "policy")}`} onClick={() => setSelected(row)}><Settings2 className="icon sm" aria-hidden="true"/></button><button className="icon-button" type="button" aria-label={`More options for ${String(row.name ?? "policy")}`} onClick={() => setActionRecord(row)}><MoreHorizontal className="icon sm" aria-hidden="true"/></button></article>) : noMatchingRecords}</section>
      : <section className="p-card p-tight security-demo-list"><div className="p-card-heading"><h3>{title} · demo records</h3><span className="p-fixture">{filtered.length} items</span></div><div className="demo-record-toolbar security-demo-toolbar"><label className="search-field demo-record-search"><SearchIcon className="icon sm" aria-hidden="true"/><input aria-label={`Search ${title.toLowerCase()}`} placeholder={`Search ${title.toLowerCase()}...`} value={search} onChange={event => setSearch(event.target.value)}/></label><span className="caption">{filtered.length} items</span><button className="btn primary sm" type="button" onClick={() => { setShowAdd(true); setNotice(""); }}><Plus className="icon sm" aria-hidden="true"/>Add</button></div>{filtered.length ? filtered.map((row, index) => <article className="p-item security-demo-record" key={String(row.id ?? index)}><div className="square-icon"><Shield className="icon sm" aria-hidden="true"/></div><div className="p-item-copy"><strong>{String(row.name ?? (tab === "Audit" ? "Audit event" : title.slice(0, -1)))}</strong><small>{String(row.description ?? "")}</small></div><span className={`tag ${["enabled", "allowed", "active"].includes(String(row.status).toLowerCase()) ? "green" : String(row.status).toLowerCase() === "denied" ? "red" : ""}`}>{String(row.status ?? "Example")}</span><button className="icon-button" type="button" aria-label={`Configure ${String(row.name ?? addLabel)}`} onClick={() => setSelected(row)}><Settings2 className="icon sm" aria-hidden="true"/></button><button className="icon-button" type="button" aria-label={`More options for ${String(row.name ?? addLabel)}`} onClick={() => setActionRecord(row)}><MoreHorizontal className="icon sm" aria-hidden="true"/></button></article>) : noMatchingRecords}</section>}
    {showAdd && <section className="p-card security-demo-add"><form onSubmit={addRecord}><div className="p-card-heading"><h3>Add local {addLabel}</h3></div><label className="demo-form-field">Name<input className="input" value={name} onChange={event => setName(event.target.value)} required/></label><label className="demo-form-field">Description<input className="input" value={description} onChange={event => setDescription(event.target.value)}/></label><div className="row"><button className="btn primary" disabled={!name.trim()}>Save locally</button><button className="btn" type="button" onClick={() => setShowAdd(false)}>Cancel</button></div></form></section>}
    {notice && <p className="caption" role="status">{notice}</p>}
    <Dialog.Root open={Boolean(selected)} onOpenChange={open => { if (!open) setSelected(null); }}><Dialog.Portal><Dialog.Overlay/><Dialog.Content className="advanced-integration-dialog"><div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">{String(selected?.name ?? "Security record")}</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close security details"><X size={16} aria-hidden="true"/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">{String(selected?.description ?? "Example record")}</Dialog.Description><div className="demo-detail-grid"><div><small>Collection</small><strong>{collection}</strong></div><div><small>Status</small><strong>{String(selected?.status ?? "Example")}</strong></div></div><p className="p-note"><Info className="icon sm" aria-hidden="true"/><span>Illustrative local demo data. These actions do not change backend permissions, sessions or trusted audit records.</span></p><div className="advanced-integration-actions">{collection === "policies" && <button type="button" className="btn" onClick={() => selected && void changeRecordStatus(selected)}>{String(selected?.status).toLowerCase() === "enabled" ? "Disable" : "Enable"}</button>}{collection === "sessions" && <button type="button" className="btn danger" onClick={() => selected && void changeRecordStatus(selected)}>Revoke session</button>}{collection === "audit" && <button type="button" className="btn" onClick={() => selected && void changeRecordStatus(selected)}>Run example</button>}<button type="button" className="btn" onClick={() => selected && openEdit(selected)}>Edit</button><button type="button" className="btn danger" onClick={() => selected && beginDelete(selected)}>Delete</button><Dialog.Close asChild><button className="btn" type="button">Close</button></Dialog.Close></div></Dialog.Content></Dialog.Portal></Dialog.Root>
    <Dialog.Root open={Boolean(actionRecord)} onOpenChange={open => { if (!open) setActionRecord(null); }}><Dialog.Portal><Dialog.Overlay/><Dialog.Content className="advanced-integration-dialog"><div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">{String(actionRecord?.name ?? "Item actions")}</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close record actions"><X size={16} aria-hidden="true"/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">Choose an action for this example record.</Dialog.Description><div className="demo-action-grid"><button type="button" className="btn" onClick={() => actionRecord && showDetails(actionRecord)}><FileText className="icon sm" aria-hidden="true"/>Open details</button><button type="button" className="btn" onClick={() => actionRecord && openEdit(actionRecord)}><SlidersHorizontal className="icon sm" aria-hidden="true"/>Edit record</button><button type="button" className="btn" onClick={() => actionRecord && void duplicateRecord(actionRecord)}><Copy className="icon sm" aria-hidden="true"/>Duplicate</button><button type="button" className="btn" onClick={() => actionRecord && void changeRecordStatus(actionRecord)}><Play className="icon sm" aria-hidden="true"/>{collection === "audit" ? "Run / test" : collection === "sessions" ? "Revoke session" : String(actionRecord?.status).toLowerCase() === "enabled" ? "Disable" : "Enable"}</button><button type="button" className="btn danger" onClick={() => actionRecord && beginDelete(actionRecord)}><Trash2 className="icon sm" aria-hidden="true"/>Delete</button></div></Dialog.Content></Dialog.Portal></Dialog.Root>
    <Dialog.Root open={Boolean(editingRecord)} onOpenChange={open => { if (!open) setEditingRecord(null); }}><Dialog.Portal><Dialog.Overlay/><Dialog.Content className="advanced-integration-dialog"><div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">Edit {addLabel}</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close security record editor"><X size={16} aria-hidden="true"/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">Changes are saved to this browser's local demo data only.</Dialog.Description><form onSubmit={event => void saveEdit(event)}><label className="demo-form-field">Name<input className="input" value={editName} onChange={event => setEditName(event.target.value)} required maxLength={120}/></label><label className="demo-form-field">Description<input className="input" value={editDescription} onChange={event => setEditDescription(event.target.value)} maxLength={300}/></label><label className="demo-form-field">Status<select className="select" value={editStatus} onChange={event => setEditStatus(event.target.value)}>{Array.from(new Set([editStatus, ...(collection === "policies" ? ["Enabled", "Disabled"] : collection === "sessions" ? ["Active", "Revoked"] : ["Example", "Simulated locally"])] )).map(value => <option key={value}>{value}</option>)}</select></label><div className="advanced-integration-actions"><button type="button" className="btn" onClick={() => setEditingRecord(null)}>Cancel</button><button type="submit" className="btn primary" disabled={!editName.trim()}>Save changes</button></div></form></Dialog.Content></Dialog.Portal></Dialog.Root>
    <Dialog.Root open={Boolean(deleteRecord)} onOpenChange={open => { if (!open) setDeleteRecord(null); }}><Dialog.Portal><Dialog.Overlay/><Dialog.Content className="advanced-integration-dialog"><div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">Delete “{String(deleteRecord?.name ?? "record")}”?</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close delete confirmation"><X size={16} aria-hidden="true"/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">This deletes the local demo record. It does not affect your backend.</Dialog.Description><div className="advanced-integration-actions"><button type="button" className="btn" onClick={() => setDeleteRecord(null)}>Cancel</button><button type="button" className="btn danger" onClick={() => void confirmDelete()}>Delete record</button></div></Dialog.Content></Dialog.Portal></Dialog.Root>
  </>;
}

function ScraperDueJobs({ data }: { data?: Record<string, JsonRecord> }) {
  const client = useQueryClient();
  const [consent, setConsent] = useState(false);
  const [feedback, setFeedback] = useState("");
  const cancelMutation = useMutation({
    mutationFn: (jobId: string) => apiPost<{ cancelled?: boolean; job?: { status?: string } }>(`/api/scraper/jobs/${encodeURIComponent(jobId)}/cancel`, { project: "default" }),
    onSuccess: async (result) => {
      setFeedback(result.cancelled === true
        ? "Schedule cancelled. The saved job history has been retained."
        : result.job?.status === "running"
          ? "This job is already running. Its current request was not interrupted."
          : `The schedule was not cancelled. Its current status is ${String(result.job?.status ?? "unknown")}. Refresh and try again if it is still pending.`);
      await client.invalidateQueries({ queryKey: ["advanced-screen", "scraper"] });
    },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const mutation = useMutation({
    mutationFn: () => apiPost("/api/scraper/run-due", { project: "default" }),
    onSuccess: async () => { setFeedback("The backend processed its currently due scheduled jobs."); await client.invalidateQueries({ queryKey: ["advanced-screen", "scraper"] }); },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const jobs = Array.isArray(data?.jobs?.jobs) ? data.jobs.jobs : [];
  return <section className="p-card"><div className="p-card-heading"><h3>Scheduled extraction jobs</h3><span className="p-fixture">{jobs.length} backend records</span></div>
    <p className="caption">Pending jobs can be cancelled before their next run. Processing due jobs uses each saved network approval and server-side egress protections.</p>
    {jobs.length > 0 && <div className="p-record-sections" aria-label="Scheduled extraction jobs">{jobs.map((job, index) => {
      const id = String(job.id ?? "");
      const metadata = job.metadata && typeof job.metadata === "object" && !Array.isArray(job.metadata) ? job.metadata as JsonRecord : {};
      const status = String(job.status ?? "unknown").toLowerCase();
      const label = String(metadata.label ?? job.url ?? "Scheduled extraction");
      const canCancel = ["scheduled", "awaiting_approval"].includes(status) && Boolean(id);
      return <article className="p-record-section" key={id || index} data-scraper-job-id={id}>
        <div className="row between"><strong>{label}</strong><span className={`tag ${status === "scheduled" ? "green" : status === "cancelled" ? "" : "orange"}`}>{status.replaceAll("_", " ")}</span></div>
        <p className="caption">{String(job.url ?? "URL unavailable")} · every {String(job.interval_hours ?? "—")} hours · next {String(job.next_run ?? "not reported")}</p>
        <p className="caption">{metadata.external_approved === true ? "Recurring external access approved" : "External access not approved"}{job.last_run ? ` · last run ${String(job.last_run)}` : " · not run yet"}</p>
        {canCancel && <button type="button" className="btn sm danger" aria-label={`Cancel scheduled extraction for ${label}`} disabled={cancelMutation.isPending} onClick={() => {
          if (!window.confirm(`Cancel the recurring extraction for “${label}”? The current project history will be retained.`)) return;
          setFeedback(""); cancelMutation.mutate(id);
        }}>{cancelMutation.isPending ? "Cancelling…" : "Cancel schedule"}</button>}
      </article>;
    })}</div>}
    {!jobs.length && <p className="muted small">No scheduled extraction jobs are stored for this project.</p>}
    <label className="row"><input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />I authorize processing due jobs, which may fetch their configured URLs</label>
    <button type="button" className="btn primary" disabled={!consent || mutation.isPending || !jobs.some(job => String(job.status ?? "") === "scheduled")} onClick={() => { if (!window.confirm("Process all jobs currently due for this project, using each job's saved network policy?")) return; setFeedback(""); mutation.mutate(); }}>{mutation.isPending ? "Processing due jobs…" : "Process due jobs"}</button>
    {feedback && <p className="caption" role={mutation.isError || cancelMutation.isError ? "alert" : "status"}>{feedback}</p>}{mutation.data && <ActionResult route="scraper" result={mutation.data} />}
  </section>;
}

type VaultDemoEvent = { id: string; name: string; route: "vault"; status: string; time: string };
const readVaultDemoEvents = (): VaultDemoEvent[] => {
  try {
    const value: unknown = JSON.parse(localStorage.getItem("oa-second-demo-vault-activity") ?? "[]");
    if (!Array.isArray(value)) return [];
    return value.map((item, index) => typeof item === "string"
      ? { id: `legacy-${index}`, name: item, route: "vault" as const, status: "Completed", time: "12:00" }
      : item && typeof item === "object" && typeof item.name === "string"
        ? { id: String(item.id ?? `activity-${index}`), name: item.name, route: "vault" as const, status: String(item.status ?? "Completed"), time: String(item.time ?? "12:00") }
        : null).filter((item): item is VaultDemoEvent => item !== null).slice(0, 5);
  } catch { return []; }
};

function VaultDemoAccessAudit({ tab, data }: { tab: "Access" | "Audit"; data?: Record<string, JsonRecord> }) {
  const client = useQueryClient();
  const collection = tab === "Access" ? "requests" : "audit";
  const rows = Array.isArray(data?.[collection]?.[collection]) ? data[collection][collection] as JsonRecord[] : [];
  const [filter, setFilter] = useState("");
  const [dialog, setDialog] = useState<"create" | "edit" | "details" | "actions" | "confirm-delete" | null>(null);
  const [selected, setSelected] = useState<JsonRecord | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [feedback, setFeedback] = useState("");
  const [events, setEvents] = useState<VaultDemoEvent[]>(readVaultDemoEvents);
  const runTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastTrigger = useRef<HTMLElement | null>(null);
  const collectionName = tab === "Access" ? "requests" : "audit";
  useEffect(() => {
    const reset = () => { setEvents([]); localStorage.removeItem("oa-second-demo-vault-activity"); };
    window.addEventListener("oa-vault-demo-reset", reset);
    return () => window.removeEventListener("oa-vault-demo-reset", reset);
  }, []);
  const titleOf = (row: JsonRecord) => String(row.name ?? row.title ?? "Vault record");
  const log = (message: string, status = "Completed") => {
    const time = new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit" }).format(new Date());
    const event: VaultDemoEvent = { id: `vault-event-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 5)}`, name: message, route: "vault", status, time };
    const next = [event, ...readVaultDemoEvents()].slice(0, 5);
    setEvents(next);
    try { localStorage.setItem("oa-second-demo-vault-activity", JSON.stringify(next)); } catch { /* Local activity remains visible in this session. */ }
    window.dispatchEvent(new CustomEvent("oa-vault-demo-activity", { detail: next }));
  };
  const visible = rows.filter(row => `${titleOf(row)} ${String(row.description ?? "")} ${String(row.scope ?? "")} ${String(row.status ?? "")}`.toLowerCase().includes(filter.trim().toLowerCase()));
  const start = (mode: "create" | "edit", row?: JsonRecord, trigger?: HTMLElement) => {
    if (trigger) lastTrigger.current = trigger;
    setSelected(row ?? null); setName(row ? titleOf(row) : ""); setDescription(String(row?.description ?? ""));
    setFeedback(""); setDialog(mode);
  };
  const openDialog = (mode: "details" | "actions" | "confirm-delete", row: JsonRecord | null, trigger?: HTMLElement) => {
    if (trigger) lastTrigger.current = trigger;
    setSelected(row); setDialog(mode);
  };
  const restoreFocus = () => { const trigger = lastTrigger.current; window.requestAnimationFrame(() => trigger?.focus()); };
  const persist = async (next: JsonRecord[]) => {
    await demoProvider.updateCollection<JsonRecord>(`collections.advanced.vault.${collection}`, () => next);
    await client.invalidateQueries({ queryKey: ["advanced-screen", "vault"] });
  };
  const save = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const id = String(selected?.id ?? `demo-vault-${collection}-${Date.now().toString(36)}`);
    const record: JsonRecord = { ...selected, id, name: name.trim(), description: description.trim(), ...(selected ? {} : { status: "Ready" }), ...(tab === "Audit" && !selected ? { time: "Just now" } : {}) };
    await persist(selected ? rows.map(row => String(row.id) === id ? record : row) : [record, ...rows]);
    const message = `${selected ? "Updated" : "Added"} ${titleOf(record)}`;
    log(message); setFeedback(`${message} in the local demo.`); setDialog(null); setSelected(null); restoreFocus();
  };
  const updateStatus = async (row: JsonRecord, nextStatus: string) => {
    const updated = { ...row, status: nextStatus };
    await persist(rows.map(item => String(item.id) === String(row.id) ? updated : item));
    log(`${titleOf(row)} · ${nextStatus}`); setFeedback(`${titleOf(row)} marked ${nextStatus} locally.`); setDialog(null); restoreFocus();
  };
  const remove = async () => {
    if (!selected) return;
    await persist(rows.filter(row => String(row.id) !== String(selected.id)));
    log(`Deleted ${titleOf(selected)}`); setFeedback(`${titleOf(selected)} deleted from the local demo.`); setDialog(null); setSelected(null); restoreFocus();
  };
  const duplicate = async () => {
    if (!selected) return;
    const copy = { ...selected, id: `demo-vault-${collection}-${Date.now().toString(36)}`, name: `${titleOf(selected)} (copy)` };
    await persist([copy, ...rows]); log(`Duplicated ${titleOf(selected)}`); setFeedback(`${titleOf(selected)} duplicated locally.`); setDialog(null); restoreFocus();
  };
  const runExample = async (row: JsonRecord) => {
    if (runTimer.current) { setFeedback("Finish the current local example run before starting another."); return; }
    const id = String(row.id);
    const updateRunState = async (status: "Running" | "Completed") => {
      await demoProvider.updateCollection<JsonRecord>(`collections.advanced.vault.${collection}`, current => current.map(item => String(item.id) === id ? { ...item, status } : item));
      await client.invalidateQueries({ queryKey: ["advanced-screen", "vault"] });
    };
    const name = titleOf(row);
    await updateRunState("Running");
    log(`Started ${name}`, "Running");
    setDialog(null); setSelected(null); setFeedback("");
    runTimer.current = setTimeout(() => {
      void updateRunState("Completed").then(() => log(`Completed ${name}`, "Completed")).finally(() => { runTimer.current = null; });
    }, 4000);
  };
  return <>
    <section className="demo-feature-hero vault-demo-hero"><div className="demo-feature-symbol"><LockKeyhole className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">{tab} · Workspace</span><h2>Security without friction</h2><p>Masked references, explicit access and full auditability.</p></div></section>
    <section className="p-card vault-access-card">
      <div className="p-card-heading"><h3>{tab} · demo records</h3></div>
      <div className="demo-record-toolbar"><label className="search-field demo-record-search"><SearchIcon className="icon sm" aria-hidden="true"/><input value={filter} onChange={event => setFilter(event.target.value)} placeholder={`Search ${collection}…`} aria-label={`Search vault ${tab.toLowerCase()}`}/></label><span className="caption">{visible.length} item{visible.length === 1 ? "" : "s"}</span><button type="button" className="btn primary sm" onClick={event => start("create", undefined, event.currentTarget)}><Plus className="icon sm" aria-hidden="true"/>Add</button></div>
      <div className="demo-records vault-access-records">{visible.length ? visible.map((row, index) => {
        const id = String(row.id ?? index); const title = titleOf(row); const state = String(row.status ?? "Ready");
        return <article className="demo-record" key={id}>
        <button className="demo-record-main" type="button" onClick={event => openDialog("details", row, event.currentTarget)}><span className="square-icon sm blue"><LockKeyhole className="icon sm" aria-hidden="true"/></span><span className="demo-record-text"><strong>{title}</strong><small>{String(row.description ?? row.scope ?? "Example record")}</small></span></button>
          <span className={`tag ${/approved|success|allowed/i.test(state) ? "green" : /denied|failed/i.test(state) ? "red" : /pending/i.test(state) ? "orange" : ""}`}>{state}</span>
          <button className="icon-button" type="button" aria-label={`Edit ${title}`} onClick={event => start("edit", row, event.currentTarget)}><SlidersHorizontal className="icon sm" aria-hidden="true"/></button>
          <button className="icon-button" type="button" aria-label={`More actions for ${title}`} onClick={event => openDialog("actions", row, event.currentTarget)}><MoreHorizontal className="icon sm" aria-hidden="true"/></button>
        </article>;
      }) : <div className="p-empty"><SearchIcon className="icon lg" aria-hidden="true"/><h3>No matching items</h3><p>Try a different search or create a sample record.</p></div>}</div>
      {feedback && !dialog && <p className="caption" role="status">{feedback}</p>}
    </section>
    <section className="p-card advanced-demo-activity" aria-label="Vault recent activity"><div className="p-card-heading"><h3>Recent activity</h3><button type="button" className="btn sm" onClick={event => { lastTrigger.current = event.currentTarget; setSelected(null); setDialog("details"); }}>All activity</button></div>{events.length ? events.map(event => <div className="demo-activity-row" key={event.id}><span className="status-dot"/><div className="flex1"><strong>{event.name}</strong><small>{event.status} · {event.time}</small></div></div>) : <p className="caption">No Vault activity has been recorded in this demo session.</p>}</section>

    <Dialog.Root open={dialog !== null} onOpenChange={open => { if (!open) { setDialog(null); restoreFocus(); } }}>
      <Dialog.Portal>
        <Dialog.Overlay className="vault-demo-overlay"/>
        <Dialog.Content className="advanced-integration-dialog vault-demo-dialog">
          {dialog === "details" && selected ? <div className="vault-demo-detail-heading">
            <span className="square-icon blue"><Lock className="icon sm" aria-hidden="true"/></span>
            <div className="vault-demo-detail-heading-copy">
              <Dialog.Title className="second-app-dialog-title">{titleOf(selected)}</Dialog.Title>
              <Dialog.Description className="second-app-dialog-description vault-demo-record-meta">{collectionName} · {String(selected.status ?? "Ready")}</Dialog.Description>
            </div>
            <Dialog.Close asChild><button className="icon-button" aria-label="Close dialog" type="button"><X size={16} aria-hidden="true"/></button></Dialog.Close>
          </div> : <div className="advanced-integration-heading">
            <Dialog.Title className="second-app-dialog-title">{dialog === "details" ? "Demo activity" : dialog === "actions" ? String(selected?.name ?? "Item actions") : dialog === "confirm-delete" ? `Delete “${titleOf(selected ?? {})}”?` : dialog === "edit" ? `Edit ${tab === "Access" ? "request" : "audit"}` : `New ${tab === "Access" ? "request" : "audit"}`}</Dialog.Title>
            <Dialog.Close asChild><button className="icon-button" aria-label="Close dialog" type="button"><X size={16} aria-hidden="true"/></button></Dialog.Close>
          </div>}
          {(dialog !== "details" || !selected) && <Dialog.Description className="second-app-dialog-description">{dialog === "details" ? "Actions completed in this browser, not from a real backend." : dialog === "confirm-delete" ? "This deletes the local demo record. It does not affect your backend." : "Changes are saved locally in this interactive demonstration."}</Dialog.Description>}
          {dialog === "create" || dialog === "edit" ? <form className="vault-request-form" onSubmit={save}>
            <label className="demo-form-field">Name<input className="input" value={name} onChange={event => setName(event.target.value)} required maxLength={100} placeholder="Short, meaningful name"/></label>
            <label className="demo-form-field">Description<textarea className="textarea" rows={3} value={description} onChange={event => setDescription(event.target.value)} maxLength={160} placeholder="What is this for?"/></label>
            <div className="demo-modal-actions"><button className="btn" type="button" onClick={() => { setDialog(null); restoreFocus(); }}>Cancel</button><button className="btn primary" type="submit">{dialog === "edit" ? "Save changes" : "Create item"}</button></div>
          </form> : dialog === "details" && selected ? <>
            <p className="vault-demo-record-description">{String(selected.description ?? "Interactive demo record")}</p>
            <div className="demo-detail-grid">{Object.entries(selected).filter(([key]) => !["id", "name", "description"].includes(key)).map(([key, value]) => <div key={key}><small>{humanize(key)}</small><strong>{String(value)}</strong></div>)}</div>
            <div className="p-note"><Info className="icon sm" aria-hidden="true"/><span>Demo records and actions are local. The React integration must use verified backend handlers, permission checks and audited run IDs.</span></div>
            <div className="modal-actions demo-detail-actions vault-demo-detail-actions">
              {tab === "Access" ? <>
                <button className="btn soft" type="button" onClick={() => updateStatus(selected, "Approved")}><Check className="icon sm" aria-hidden="true"/>Approve</button>
                <button className="btn soft" type="button" onClick={() => updateStatus(selected, "Denied")}><X className="icon sm" aria-hidden="true"/>Deny</button>
              </> : <button className="btn soft" type="button" onClick={() => runExample(selected)}><Play className="icon sm" aria-hidden="true"/>Run example</button>}
              <button className="btn" type="button" onClick={() => start("edit", selected)}><SlidersHorizontal className="icon sm" aria-hidden="true"/>Edit</button>
              <button className="btn danger" type="button" onClick={event => openDialog("confirm-delete", selected, event.currentTarget)}><X className="icon sm" aria-hidden="true"/>Delete</button>
            </div>
          </> : dialog === "actions" && selected ? <><p>Choose an action for this example record.</p><div className="demo-action-grid">
            <button className="btn" type="button" onClick={() => openDialog("details", selected)}>Open details</button>
            <button className="btn" type="button" onClick={() => start("edit", selected)}>Edit record</button>
            <button className="btn" type="button" onClick={duplicate}>Duplicate</button>
            <button className="btn" type="button" onClick={() => runExample(selected)}>Run / test</button>
            <button className="btn danger" type="button" onClick={() => setDialog("confirm-delete")}><X className="icon sm" aria-hidden="true"/>Delete</button>
          </div></> : dialog === "confirm-delete" && selected ? <div className="modal-actions vault-demo-confirm-actions">
            <button className="btn" type="button" onClick={() => { setDialog("details"); }}>Cancel</button>
            <button className="btn danger" type="button" onClick={remove}><X className="icon sm" aria-hidden="true"/>Delete record</button>
          </div> : dialog === "details" ? <div className="demo-event-scroll vault-demo-event-scroll">
            {events.length ? events.map(activity => <div className="list-row" key={activity.id}><span className="square-icon sm green"><Check className="icon sm" aria-hidden="true"/></span><div className="flex1"><strong>{activity.name}</strong><div className="caption">{activity.route} · {activity.status} · {activity.time}</div></div></div>) : <p className="caption">Nothing yet.</p>}
          </div> : null}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  </>;
}

function VaultWorkspace({ tab, data, demo = false }: { tab: string; data?: Record<string, JsonRecord>; demo?: boolean }) {
  const client = useQueryClient();
  const [filter, setFilter] = useState("");
  const [demoCreateOpen, setDemoCreateOpen] = useState(false);
  const [demoName, setDemoName] = useState("");
  const [demoDescription, setDemoDescription] = useState("");
  const [demoScope, setDemoScope] = useState("");
  const [demoEntries, setDemoEntries] = useState<JsonRecord[]>(() => { try { const value = JSON.parse(localStorage.getItem("oa-second-demo-vault-entries") ?? "[]"); return Array.isArray(value) ? value.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : []; } catch { return []; } });
  useEffect(() => { const reset = () => setDemoEntries([]); window.addEventListener("oa-vault-demo-reset", reset); return () => window.removeEventListener("oa-vault-demo-reset", reset); }, []);

  const [label, setLabel] = useState("");
  const [category, setCategory] = useState("secret");
  const [secret, setSecret] = useState("");
  const [storePassphrase, setStorePassphrase] = useState("");
  const [revealTarget, setRevealTarget] = useState("");
  const [revealPassphrase, setRevealPassphrase] = useState("");
  const [revealed, setRevealed] = useState<Record<string, string>>({});
  const revealTimers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const [oldPassphrase, setOldPassphrase] = useState("");
  const [newPassphrase, setNewPassphrase] = useState("");
  const [exportPassphrase, setExportPassphrase] = useState("");
  const [feedback, setFeedback] = useState("");
  const [error, setError] = useState("");
  const [demoDetailsId, setDemoDetailsId] = useState("");
  const entries = [...(Array.isArray(data?.entries?.entries) ? data.entries.entries.filter((row): row is JsonRecord => Boolean(row) && typeof row === "object" && !Array.isArray(row)) : []), ...(demo ? demoEntries : [])];
  const access = Array.isArray(data?.access_log?.access_log) ? data.access_log.access_log.filter((row): row is JsonRecord => Boolean(row) && typeof row === "object" && !Array.isArray(row)) : [];
  const status = data?.status ?? {};
  const clearReveal = (id?: string) => {
    if (id) { const timer = revealTimers.current.get(id); if (timer) clearTimeout(timer); revealTimers.current.delete(id); setRevealed((current) => { const next = { ...current }; delete next[id]; return next; }); return; }
    for (const timer of revealTimers.current.values()) clearTimeout(timer);
    revealTimers.current.clear(); setRevealed({}); setRevealPassphrase(""); setRevealTarget("");
  };
  useEffect(() => {
    const conceal = () => { if (document.visibilityState === "hidden") clearReveal(); };
    document.addEventListener("visibilitychange", conceal);
    return () => { document.removeEventListener("visibilitychange", conceal); for (const timer of revealTimers.current.values()) clearTimeout(timer); revealTimers.current.clear(); };
  }, []);
  const mutation = useMutation({
    mutationFn: ({ path, body }: { path: string; body: unknown }) => apiPost(path, body),
    onSuccess: async (result, variables) => {
      setError("");
      if (variables.path === "/api/vault/retrieve") {
        const secretResult = result.secret && typeof result.secret === "object" ? result.secret as JsonRecord : {};
        const value = secretResult.secret ?? result.secret;
        const id = String((variables.body as JsonRecord).entry_id ?? "");
        setRevealed((current) => ({ ...current, [id]: typeof value === "string" ? value : JSON.stringify(value ?? "") }));
        setRevealTarget(""); setFeedback("Secret revealed briefly. Conceal it when finished; it will also disappear when this tab is hidden.");
        const existing = revealTimers.current.get(id); if (existing) clearTimeout(existing);
        revealTimers.current.set(id, setTimeout(() => { revealTimers.current.delete(id); setRevealed((current) => { const next = { ...current }; delete next[id]; return next; }); }, 15_000));
      } else if (variables.path === "/api/vault/store") { setFeedback("Encrypted entry stored."); await client.invalidateQueries({ queryKey: ["advanced-screen", "vault"] }); }
      else if (variables.path === "/api/vault/rotate-key") { setFeedback(`Passphrase rotation completed for ${String(result.rotated ?? 0)} entries. Store the new passphrase securely.`); clearReveal(); }
      else if (variables.path === "/api/vault/export") {
        const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }); const url = URL.createObjectURL(blob); const link = document.createElement("a");
        link.href = url; link.download = `openagent-vault-${String(status.project ?? "default")}.json`; link.click(); URL.revokeObjectURL(url);
        setFeedback(`Encrypted bundle downloaded (${String(result.entry_count ?? 0)} entries).`);
      }
    },
    onError: (cause) => { setError(apiConnectionMessage(cause)); setFeedback(""); },
  });
  const submit = (path: string, body: JsonRecord) => { setError(""); setFeedback(""); mutation.mutate({ path, body }); };
  const matching = entries.filter((entry) => `${String(entry.label ?? "")} ${String(entry.category ?? "")} ${Array.isArray(entry.tags) ? entry.tags.map(String).join(" ") : ""}`.toLowerCase().includes(filter.trim().toLowerCase()));
  if ((tab === "Access" || tab === "Audit") && demo) return <VaultDemoAccessAudit tab={tab} data={data}/>;
  if (tab === "Access" || tab === "Audit") {
    return <section className="p-card p-tight">
      <div className="p-card-heading"><h3>{tab === "Access" ? "Credential access" : "Vault activity"}</h3><span className="p-fixture">{access.length} backend records</span></div>
      {tab === "Access" ? <>
        <article className="p-item"><div className="square-icon"><Shield className="icon sm" aria-hidden="true" /></div><div className="p-item-copy"><strong>Who can use secrets</strong><small>Service accounts, roles and scopes</small></div><span className="p-item-meta">Policy</span></article>
        <article className="p-item"><div className="square-icon"><CheckCircle2 className="icon sm" aria-hidden="true" /></div><div className="p-item-copy"><strong>Approval requests</strong><small>Sensitive operations require verification</small></div><span className="p-item-meta">Required</span></article>
        <article className="p-item"><div className="square-icon"><KeyRound className="icon sm" aria-hidden="true" /></div><div className="p-item-copy"><strong>Rotation and expiry</strong><small>Only actual expiry dates reported by the backend</small></div><span className="p-item-meta">Not loaded</span></article>
        {access.length > 0 && <div className="p-record-sections section-space" aria-label="Backend access events">{access.map((event, index) => <article className="p-record-section" key={String(event.id ?? index)}><strong>{String(event.action ?? "Vault activity")}</strong><p className="caption">Entry {String(event.entry_id ?? "unavailable")} · {event.success === true ? "success" : event.success === false ? "failed" : "outcome not reported"}</p></article>)}</div>}
      </> : !access.length ? <EmptyState icon={FileClock} title="Vault audit not connected" detail="Actual reveal, use and rotation events appear here after backend verification." /> : <div className="p-record-sections">{access.map((event, index) => <article className="p-record-section" key={String(event.id ?? index)}><strong>{String(event.action ?? "Vault activity")}</strong><p className="caption">Entry {String(event.entry_id ?? "unavailable")} · {event.success === true ? "success" : event.success === false ? "failed" : "outcome not reported"} · {String(event.created_at ?? "date unavailable")}</p></article>)}</div>}
    </section>;
  }
  return <>
    {demo ? <section className="demo-feature-hero vault-demo-hero"><div className="demo-feature-symbol"><LockKeyhole className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">Secrets · Workspace</span><h2>Security without friction</h2><p>Masked references, explicit access and full auditability.</p></div></section> : <section className="p-vault-banner"><span><LockKeyhole className="icon xl" aria-hidden="true" /></span><div><h2>Credentials, kept protected.</h2><p>Work with masked references. Reveal and use secrets only through authorised backend flows.</p></div><button type="button" className="btn primary" onClick={() => { if (demo) { setDemoCreateOpen(true); return; } const panel = document.getElementById("vault-store-entry") as HTMLDetailsElement | null; if (panel) { panel.open = true; panel.scrollIntoView({ behavior: "smooth", block: "center" }); } }}><Plus className="icon sm" aria-hidden="true" />Add credential</button></section>}
    <section className={`p-card p-tight${demo ? " vault-demo-card" : ""}`}><div className="p-card-heading"><h3>{demo ? "Masked credentials" : "Your secrets"}</h3>{!demo && <span className="p-fixture">{String(status.total_entries ?? entries.length)} backend entries</span>}{Object.keys(revealed).length > 0 && <button type="button" className="btn sm" onClick={() => clearReveal()}>Conceal all</button>}</div>
      {demo ? <div className="p-note vault-demo-note"><LockKeyhole className="icon sm" aria-hidden="true" /><span>Credentials are always synthetic placeholders. Never paste real secrets into this demo.</span></div> : <p className="caption">Entry metadata is available to readers. Secret values remain masked until the vault authorises a brief reveal.</p>}
      <div className="row vault-demo-toolbar"><label className="search-field flex1"><SearchIcon className="icon sm" aria-hidden="true" /><input aria-label="Filter vault entries" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder={demo ? "Search secrets…" : "Filter label, category or tags"} /></label>{demo && <><span className="caption">{entries.length} items</span><button type="button" className="btn primary" onClick={() => setDemoCreateOpen(true)}><Plus className="icon sm" aria-hidden="true" />Add</button></>}</div>
      {!matching.length ? <EmptyState icon={LockKeyhole} title={entries.length ? "No matching entries" : "Your vault is empty"} detail={entries.length ? "Adjust the filter to find an entry." : "Store an encrypted credential when you have vault administrator access."} /> : <div className="p-record-sections section-space">{matching.map((entry, index) => { const id = String(entry.id ?? ""); const labelText = String(entry.label ?? "Vault entry"); return <article className="p-item" key={id || index}><div className="square-icon"><LockKeyhole className="icon sm" aria-hidden="true" /></div><div className="p-item-copy"><strong>{labelText}</strong><small>{demo ? String(entry.description ?? entry.category ?? "Synthetic example") : `${String(entry.category ?? "secret")}${Array.isArray(entry.tags) && entry.tags.length ? ` · ${entry.tags.map(String).join(", ")}` : ""}`}</small></div>{demo ? <><span className={`p-item-meta vault-demo-status ${String(entry.status).toLowerCase() === "active" ? "is-active" : ""}`}>{String(entry.status ?? "Masked")}</span><button type="button" className="icon-button" aria-label={`Credential details for ${labelText}`} onClick={() => setDemoDetailsId((current) => current === id ? "" : id)}><SlidersHorizontal className="icon sm" aria-hidden="true" /></button><button type="button" className="icon-button" aria-label={`More options for ${labelText}`} onClick={() => setDemoDetailsId((current) => current === id ? "" : id)}>···</button>{demoDetailsId === id && <div className="p-note vault-demo-detail"><strong>{labelText}</strong><span>{String(entry.scope ?? "Demo-only scope")} · Secret value stays masked.</span></div>}</> : <><span className="p-item-meta" aria-label={revealed[id] !== undefined ? "Secret value revealed temporarily" : "Secret value masked"}>{revealed[id] !== undefined ? "Temporarily revealed" : "••••••••"}</span>
        {revealed[id] !== undefined ? <div className="p-note section-space"><code>{revealed[id]}</code><button type="button" className="btn sm" onClick={() => clearReveal(id)}>Conceal now</button></div> : revealTarget === id ? <form className="row section-space" onSubmit={(event) => { event.preventDefault(); const passphrase = revealPassphrase; setRevealPassphrase(""); submit("/api/vault/retrieve", { entry_id: id, passphrase, project: String(status.project ?? "default") }); }}><label className="demo-form-field">Vault passphrase<input className="input" type="password" autoComplete="current-password" value={revealPassphrase} onChange={(event) => setRevealPassphrase(event.target.value)} required /></label><button className="btn primary" disabled={mutation.isPending || !revealPassphrase.trim()}>{mutation.isPending ? "Checking…" : "Reveal for 15 seconds"}</button><button type="button" className="btn" onClick={() => { setRevealTarget(""); setRevealPassphrase(""); }}>Cancel</button></form> : <button type="button" className="btn sm section-space" onClick={() => { setError(""); setRevealTarget(id); }}>Reveal</button>}</>}</article>; })}</div>}
    </section>
    {!demo && <details className="p-card" id="vault-store-entry"><summary className="p-card-heading"><h3>Store encrypted entry</h3><span className="p-fixture">Requires vault administrator access</span></summary><form onSubmit={(event) => { event.preventDefault(); const payload = { label: label.trim(), category, secret, passphrase: storePassphrase, project: String(status.project ?? "default") }; setSecret(""); setStorePassphrase(""); setLabel(""); submit("/api/vault/store", payload); }}><label className="demo-form-field">Label<input className="input" value={label} onChange={(event) => setLabel(event.target.value)} required /></label><label className="demo-form-field">Category<select className="select" value={category} onChange={(event) => setCategory(event.target.value)}>{["secret", "password", "api_key", "token", "certificate", "private_note", "credential"].map((item) => <option key={item}>{item}</option>)}</select></label><label className="demo-form-field">Secret value<input className="input" type="password" autoComplete="new-password" value={secret} onChange={(event) => setSecret(event.target.value)} required /></label><label className="demo-form-field">Vault passphrase<input className="input" type="password" autoComplete="current-password" value={storePassphrase} onChange={(event) => setStorePassphrase(event.target.value)} required /></label><p className="caption">The value and passphrase are cleared from the form as soon as you submit.</p><button className="btn primary" disabled={mutation.isPending || !label.trim() || !secret || !storePassphrase}>{mutation.isPending ? "Encrypting…" : "Encrypt and store"}</button></form></details>}
    {!demo && <details className="p-card"><summary className="p-card-heading">Vault maintenance</summary><div className="p-record-sections section-space"><form onSubmit={(event) => { event.preventDefault(); if (!window.confirm("Re-encrypt every vault entry in this project with the new passphrase? The old passphrase will stop working after success.")) return; const body = { old_passphrase: oldPassphrase, new_passphrase: newPassphrase, project: String(status.project ?? "default") }; setOldPassphrase(""); setNewPassphrase(""); submit("/api/vault/rotate-key", body); }}><h4>Rotate passphrase</h4><label className="demo-form-field">Current passphrase<input className="input" type="password" autoComplete="current-password" value={oldPassphrase} onChange={(event) => setOldPassphrase(event.target.value)} required /></label><label className="demo-form-field">New passphrase<input className="input" type="password" autoComplete="new-password" value={newPassphrase} onChange={(event) => setNewPassphrase(event.target.value)} required /></label><button className="btn danger" disabled={mutation.isPending || !oldPassphrase || !newPassphrase}>Re-encrypt entries</button></form>
      <form onSubmit={(event) => { event.preventDefault(); if (!window.confirm("Export an encrypted bundle of this project's vault entries? The downloaded file is encrypted with your passphrase.")) return; const body = { passphrase: exportPassphrase, project: String(status.project ?? "default") }; setExportPassphrase(""); submit("/api/vault/export", body); }}><h4>Encrypted export</h4><label className="demo-form-field">Vault passphrase<input className="input" type="password" autoComplete="current-password" value={exportPassphrase} onChange={(event) => setExportPassphrase(event.target.value)} required /></label><button className="btn" disabled={mutation.isPending || !exportPassphrase}>Download encrypted bundle</button></form></div></details>}
    <Dialog.Root open={demoCreateOpen} onOpenChange={(open) => { setDemoCreateOpen(open); if (!open) { setDemoName(""); setDemoDescription(""); setDemoScope(""); } }}>
      <Dialog.Portal><Dialog.Overlay/><Dialog.Content className="voice-demo-dialog demo-large-modal">
        <div className="row between"><Dialog.Title className="second-app-dialog-title">Add secret</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close dialog"><X className="icon sm"/></button></Dialog.Close></div>
        <Dialog.Description className="second-app-dialog-description">Add a synthetic masked reference to this local demo. No secret value is created or sent.</Dialog.Description>
        <form className="demo-modal-form" onSubmit={(event) => { event.preventDefault(); const record: JsonRecord = { id: "demo-vault-" + Date.now().toString(36), label: demoName.trim(), description: demoDescription.trim(), scope: demoScope.trim(), status: "Active", category: "secret", value: "••••••••", demo: true }; setDemoEntries((current) => { const next = [record, ...current]; try { localStorage.setItem("oa-second-demo-vault-entries", JSON.stringify(next)); } catch { /* local demo remains in memory */ } return next; }); setDemoName(""); setDemoDescription(""); setDemoScope(""); setDemoCreateOpen(false); setFeedback("Synthetic masked reference added locally."); }}>
          <label className="demo-form-field">Name<input className="input" value={demoName} onChange={(event) => setDemoName(event.target.value)} required maxLength={80}/></label>
          <label className="demo-form-field">Description<input className="input" value={demoDescription} onChange={(event) => setDemoDescription(event.target.value)} required maxLength={160}/></label>
          <label className="demo-form-field">Scope<input className="input" value={demoScope} onChange={(event) => setDemoScope(event.target.value)} required maxLength={80}/></label>
          <p className="caption">Only the masked label and description are stored in browser demo data.</p>
          <div className="demo-modal-actions"><Dialog.Close asChild><button className="btn" type="button">Cancel</button></Dialog.Close><button className="btn primary" type="submit" disabled={!demoName.trim() || !demoDescription.trim() || !demoScope.trim()}>Add secret</button></div>
        </form>
      </Dialog.Content></Dialog.Portal>
    </Dialog.Root>
    {error && <p className="caption" role="alert">{error}</p>}{feedback && <p className="caption" role="status">{feedback}</p>}
  </>;
}

function FineTuneWorkspace({ tab, data }: { tab: string; data?: Record<string, JsonRecord> }) {
  const client = useQueryClient();
  const [feedback, setFeedback] = useState("");
  const [error, setError] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [jobBaseModel, setJobBaseModel] = useState("");
  const [templateId, setTemplateId] = useState("");
  const [probeModelId, setProbeModelId] = useState("");
  const [probeResult, setProbeResult] = useState<JsonRecord>({});
  const [exampleInput, setExampleInput] = useState("");
  const [exampleOutput, setExampleOutput] = useState("");
  const [datasetName, setDatasetName] = useState("");
  const [jobMethod, setJobMethod] = useState("lora");
  const [adapterName, setAdapterName] = useState("");
  const [adapterModel, setAdapterModel] = useState("");
  const [adapterRank, setAdapterRank] = useState("16");
  const [activeJobIds, setActiveJobIds] = useState<string[]>([]);
  const datasets = Array.isArray(data?.datasets?.datasets) ? data.datasets.datasets.filter((row): row is JsonRecord => Boolean(row) && typeof row === "object" && !Array.isArray(row)) : [];
  const jobs = Array.isArray(data?.jobs?.jobs) ? data.jobs.jobs.filter((row): row is JsonRecord => Boolean(row) && typeof row === "object" && !Array.isArray(row)) : [];
  const adapters = Array.isArray(data?.adapters?.adapters) ? data.adapters.adapters.filter((row): row is JsonRecord => Boolean(row) && typeof row === "object" && !Array.isArray(row)) : [];
  const templates = Array.isArray(data?.templates?.templates) ? data.templates.templates.filter((row): row is JsonRecord => Boolean(row) && typeof row === "object" && !Array.isArray(row)) : [];
  const models = Array.isArray(data?.localModels?.models) ? data.localModels.models.filter((row): row is JsonRecord => Boolean(row) && typeof row === "object" && !Array.isArray(row)) : [];
  const localJobs = Array.isArray(data?.localJobs?.jobs) ? data.localJobs.jobs.filter((row): row is JsonRecord => Boolean(row) && typeof row === "object" && !Array.isArray(row)) : [];
  const status = data?.status ?? {};
  const mutation = useMutation({
    mutationFn: ({ path, body }: { path: string; body: unknown }) => apiPost(path, body),
    onSuccess: async () => { setError(""); setFeedback("The backend accepted the request. Check its returned state before assuming work has started."); await client.invalidateQueries({ queryKey: ["advanced-screen", "finetune"] }); },
    onError: (cause) => { setError(apiConnectionMessage(cause)); setFeedback(""); },
  });
  const runAction = (path: string, body: JsonRecord, onSuccess?: () => void) => {
    setError(""); setFeedback("");
    mutation.mutate({ path, body }, { onSuccess: () => onSuccess?.() });
  };
  const probe = async () => {
    if (!probeModelId) return;
    setError(""); setFeedback(""); setProbeResult({});
    try { const result = await apiGet(`/api/local-ai/models/${encodeURIComponent(probeModelId)}/probe`); setProbeResult(result); }
    catch (cause) { setError(apiConnectionMessage(cause)); }
  };
  const createJob = (event: FormEvent) => { event.preventDefault(); if (!window.confirm("Create this fine-tuning job record? Resource policy determines whether and when training can run.")) return; runAction("/api/finetune/jobs", { dataset_id: datasetId, base_model: jobBaseModel, method: jobMethod, template_id: templateId, project: "default" }); };
  const authNote = <p className="caption">Fine-tuning writes require the backend finetune:write scope. A queued job record can be cancelled; no training worker is currently exposed by the backend.</p>;
  return <>
    {tab === "Datasets" && <>
      <section className="p-finetune-hero"><span className="p-finetune-emblem"><Cpu className="icon xl" aria-hidden="true" /></span><div><span className="p-eyebrow">Model training</span><h2>Prepare a reliable fine-tuning run.</h2><p>Validate data, choose a compatible model, and approve resources before training.</p></div></section>
      <section className="p-card p-tight"><div className="p-card-heading"><h3>Training preparation</h3></div><div className="p-flow" aria-label="Training preparation steps">{["Prepare dataset", "Validate examples", "Select base model", "Review resources", "Start approved job"].map((step, index) => <span className="p-flow-step" key={step}><span className="p-flow-number">{index + 1}</span>{step}</span>)}</div><div className="p-toolbar"><button type="button" className="btn primary" onClick={() => { const panel = document.getElementById("finetune-create-dataset") as HTMLDetailsElement | null; if (panel) { panel.open = true; panel.scrollIntoView({ behavior: "smooth", block: "center" }); } }}><Plus className="icon sm" aria-hidden="true" />Choose dataset</button><button type="button" className="btn" disabled title="The backend does not expose a compatibility planning endpoint">Review compatibility</button></div></section>
      <section className="p-card p-tight"><div className="p-card-heading"><h3>Training datasets</h3><span className="p-fixture">{datasets.length} backend records</span></div>{!datasets.length ? <EmptyState icon={Database} title="No training datasets" detail="Backend dataset records appear here when available." /> : <div className="p-record-sections">{datasets.map((item, index) => <article className="p-item" key={String(item.id ?? index)}><div className="square-icon"><Database className="icon sm" aria-hidden="true" /></div><div className="p-item-copy"><strong>{String(item.name ?? item.id ?? "Dataset")}</strong><small>{String(item.format ?? "format unavailable")} · {String(item.example_count ?? item.examples ?? "example count unavailable")} examples</small></div><span className="p-item-meta">{String(item.updated_at ?? item.created_at ?? "date unavailable")}</span></article>)}</div>}</section>
      <details className="p-card" id="finetune-create-dataset"><summary className="p-card-heading"><h3>Create dataset</h3><span className="p-fixture">Requires finetune:write</span></summary><form onSubmit={(event) => { event.preventDefault(); runAction("/api/finetune/datasets", { name: datasetName.trim(), format: "jsonl", project: "default" }, () => setDatasetName("")); }}><label className="demo-form-field">Dataset name<input className="input" value={datasetName} onChange={(event) => setDatasetName(event.target.value)} required /></label><button className="btn primary" disabled={mutation.isPending || !datasetName.trim()}>Create dataset record</button></form>{authNote}</details>
      <details className="p-card"><summary className="p-card-heading"><h3>Add training example</h3><span className="p-fixture">Append to a backend dataset</span></summary><form onSubmit={(event) => { event.preventDefault(); runAction("/api/finetune/examples", { dataset_id: datasetId, user_input: exampleInput.trim(), assistant_output: exampleOutput.trim(), project: "default" }, () => { setExampleInput(""); setExampleOutput(""); }); }}><label className="demo-form-field">Dataset<select className="select" value={datasetId} onChange={(event) => setDatasetId(event.target.value)} required><option value="">Select a backend dataset</option>{datasets.map((item, index) => <option key={String(item.id ?? index)} value={String(item.id ?? "")}>{String(item.name ?? item.id ?? "Dataset")}</option>)}</select></label><label className="demo-form-field">Training prompt<textarea className="input" rows={3} value={exampleInput} onChange={(event) => setExampleInput(event.target.value)} required /></label><label className="demo-form-field">Expected response<textarea className="input" rows={3} value={exampleOutput} onChange={(event) => setExampleOutput(event.target.value)} required /></label><p className="caption">Submitted examples become part of the selected training dataset.</p><button className="btn" disabled={mutation.isPending || !datasetId || !exampleInput.trim() || !exampleOutput.trim()}>Add example</button></form>{authNote}</details>
    </>}
    {tab === "Jobs" && <><section className="p-card"><div className="p-card-heading"><h3>Training job history</h3><span className="p-fixture">{jobs.length} backend records</span></div>{!jobs.length ? <p className="muted small">No training jobs returned.</p> : <div className="p-record-sections">{jobs.map((job, index) => { const jobId = String(job.id ?? ""); const jobStatus = String(job.status ?? job.state ?? "status unavailable"); return <article className="p-record-section" key={jobId || index}><div className="row between"><strong>{String(job.base_model ?? job.name ?? job.id ?? "Training job")}</strong><span className="caption">{jobStatus}</span></div><p className="caption">Dataset {String(job.dataset_id ?? "unavailable")} · {String(job.method ?? "method unavailable")} · updated {String(job.updated_at ?? job.created_at ?? "date unavailable")}</p>{typeof job.progress === "number" && <progress value={Math.max(0, Math.min(100, job.progress))} max="100" aria-label={`Backend reported progress ${job.progress}%`} />}{jobStatus === "queued" && jobId && <button className="btn sm" type="button" disabled={mutation.isPending} onClick={() => { if (!window.confirm("Cancel this queued fine-tuning job record? This does not stop a training process.")) return; runAction(`/api/finetune/jobs/${encodeURIComponent(jobId)}/cancel`, { project: String(job.project ?? "default") }); }}>Cancel queued job</button>}</article>; })}</div>}</section>
      <section className="p-card"><div className="p-card-heading"><h3>Create training job</h3></div><form onSubmit={createJob}><label className="demo-form-field">Dataset<select className="select" value={datasetId} onChange={(event) => setDatasetId(event.target.value)} required><option value="">Choose a dataset</option>{datasets.map((item, index) => <option key={String(item.id ?? index)} value={String(item.id ?? "")}>{String(item.name ?? item.id ?? "Dataset")}</option>)}</select></label><label className="demo-form-field">Installed base model<select className="select" value={jobBaseModel} onChange={(event) => setJobBaseModel(event.target.value)} required><option value="">Choose an installed model</option>{models.map((item, index) => <option key={String(item.id ?? index)} value={String(item.id ?? "")}>{String(item.name ?? item.id ?? "Model")} · {String(item.status ?? "status unavailable")}</option>)}</select></label><label className="demo-form-field">Training method<select className="select" value={jobMethod} onChange={(event) => setJobMethod(event.target.value)}>{["lora", "qlora", "full-finetune"].map((item) => <option key={item}>{item}</option>)}</select></label>{templates.length > 0 && <label className="demo-form-field">Configuration template<select className="select" value={templateId} onChange={(event) => setTemplateId(event.target.value)}><option value="">No template</option>{templates.map((item, index) => <option key={String(item.id ?? index)} value={String(item.id ?? "")}>{String(item.name ?? item.id ?? "Template")}</option>)}</select></label>}<p className="caption">This creates a job record. Backend status is the only source for execution or progress; admin policy may reject the request.</p><button className="btn primary" disabled={mutation.isPending || !datasetId || !jobBaseModel}>{mutation.isPending ? "Submitting…" : "Create training job"}</button></form>{authNote}</section>
      <details className="p-card"><summary>Local model checks and setup jobs</summary><p className="caption">These are local-AI operations, separate from fine-tuning job records. Model probe requires local_ai:read; cancel requires local_ai:admin. Paths and checksums are not displayed.</p><div className="p-record-section"><h4>Probe installed model</h4><label className="demo-form-field">Local model<select className="select" value={probeModelId} onChange={(event) => setProbeModelId(event.target.value)}><option value="">Select model</option>{models.map((item, index) => <option key={String(item.id ?? index)} value={String(item.id ?? "")}>{String(item.name ?? item.id ?? "Model")}</option>)}</select></label><button className="btn" type="button" disabled={!probeModelId} onClick={() => void probe()}>Probe model readiness</button>{Object.keys(probeResult).length > 0 && <p className="caption" role="status">{String(probeResult.status ?? "status unavailable")} · file exists: {String(probeResult.exists ?? "not reported")} · checksum valid: {String(probeResult.checksum_valid ?? "not reported")}</p>}</div>
        <div className="p-record-section"><h4>Local model setup jobs</h4>{localJobs.map((job, index) => { const id = String(job.id ?? ""); const terminal = ["completed", "failed", "cancelled"].includes(String(job.status ?? "").toLowerCase()); return <div className="row between" key={id || index}><span>{String(job.model_name ?? job.model_id ?? id)} · {String(job.status ?? "unknown")}</span><button className="btn sm danger" type="button" disabled={!id || terminal || mutation.isPending || activeJobIds.includes(id)} onClick={() => { if (!window.confirm("Request cancellation of this local model setup job? Its current engine call may finish first.")) return; setActiveJobIds((current) => [...current, id]); runAction("/api/local-ai/jobs/cancel", { job_id: id }); }}>Cancel local setup</button></div>; })}{!localJobs.length && <p className="caption">No local model setup jobs returned.</p>}</div></details></>}
    {tab === "Results" && <><section className="p-card"><div className="p-card-heading"><h3>Registered adapters</h3><span className="p-fixture">{adapters.length} backend records</span></div>{!adapters.length ? <p className="muted small">No adapters returned.</p> : <div className="p-record-sections">{adapters.map((adapter, index) => { const id = String(adapter.id ?? ""); return <article className="p-record-section" key={id || index}><div className="row between"><strong>{String(adapter.name ?? adapter.base_model ?? id ?? "Adapter")}</strong><span className="caption">{adapter.active === true ? "active" : String(adapter.status ?? "registered")}</span></div><p className="caption">Base model {String(adapter.base_model ?? "unavailable")} · rank {String(adapter.rank ?? "unavailable")}</p><button className="btn sm" type="button" disabled={mutation.isPending || adapter.active === true || !id} onClick={() => { if (!window.confirm(`Activate adapter ${String(adapter.name ?? id)} for this project?`)) return; runAction("/api/finetune/adapters/activate", { adapter_id: id, project: "default" }); }}>Activate for project</button></article>; })}</div>}</section>
      <section className="p-card"><div className="p-card-heading"><h3>Register adapter</h3></div><form onSubmit={(event) => { event.preventDefault(); runAction("/api/finetune/adapters", { name: adapterName.trim(), base_model: adapterModel.trim(), rank: Number(adapterRank), project: "default" }, () => setAdapterName("")); }}><label className="demo-form-field">Adapter name<input className="input" value={adapterName} onChange={(event) => setAdapterName(event.target.value)} required /></label><label className="demo-form-field">Base model<input className="input" value={adapterModel} onChange={(event) => setAdapterModel(event.target.value)} required /></label><label className="demo-form-field">Rank<input className="input" type="number" min="1" max="512" value={adapterRank} onChange={(event) => setAdapterRank(event.target.value)} required /></label><button className="btn primary" disabled={mutation.isPending || !adapterName.trim() || !adapterModel.trim()}>Register adapter metadata</button></form>{authNote}</section></>}
    {tab === "Checkpoints" && <section className="p-card"><div className="p-card-heading"><h3>Checkpoints</h3></div><p className="muted small">The backend exposes no checkpoint manifest or listing route. This view cannot show checkpoint files until that contract exists.</p></section>}
    {(tab === "Datasets" || tab === "Jobs" || tab === "Results") && <>{error && <p className="caption" role="alert">{error}</p>}{feedback && <p className="caption" role="status">{feedback}</p>}</>}
    <span className="sr-only">Backend-reported datasets: {String(status.datasets ?? datasets.length)}; jobs: {String(status.jobs ?? jobs.length)}; adapters: {String(status.adapters ?? adapters.length)}</span>
  </>;
}
function ActionResult({ route, result }: { route: string; result: JsonRecord }) {
  if (route === "science" && Array.isArray(result.experiments)) {
    const proposals = result.experiments.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item));
    return <div className="p-record-sections" role="status"><h4>Backend experiment proposals</h4>{proposals.length ? <ul>{proposals.map((item, index) => <li key={`${recordTitle(item)}-${index}`}><strong>{recordTitle(item)}</strong><span className="caption">{typeof item.rationale === "string" ? item.rationale : recordSummary(item, route)}</span></li>)}</ul> : <p className="caption">The backend returned no proposals for those candidates.</p>}</div>;
  }
  const candidates = ["task", "goal", "brief", "draft", "session", "dataset", "job", "sent", "run", "result", "scrape", "record", "profile", "webhook", "entry", "plan"];
  const records = candidates.flatMap((key) => {
    const item = result[key];
    return item && typeof item === "object" && !Array.isArray(item) ? [{ key, value: item as JsonRecord }] : [];
  });
  const scalars = Object.fromEntries(Object.entries(result).filter(([key, value]) => !["secret", "value", "token", "key", "ciphertext", "content", "text", "html"].includes(key.toLowerCase()) && (typeof value === "string" || typeof value === "number" || typeof value === "boolean")));
  return <div className="p-note" role="status"><CheckCircle2 className="icon sm" aria-hidden="true" /><span>{records.length ? `${humanize(records[0].key)}: ${recordTitle(records[0].value)}${recordSummary(records[0].value, route) ? ` · ${recordSummary(records[0].value, route)}` : ""}` : Object.keys(scalars).length ? Object.entries(scalars).slice(0, 3).map(([key, value]) => `${humanize(key)}: ${String(value).slice(0, 100)}`).join(" · ") : "The backend returned a response."}</span></div>;
}
function LiveData({ route, tab, query, supported }: { route: string; tab: string; query: { data?: Record<string, JsonRecord>; isLoading: boolean; error: unknown }; supported: boolean }) {
  if (route === "communications" || route === "voice" || route === "assistant" || route === "security" || route === "twin" && tab === "Sources") return null;
  if (query.isLoading) return <section className="p-card" aria-live="polite"><div className="p-card-heading"><h3>Loading verified backend data</h3></div><p className="caption">Waiting for the OpenAgent response…</p></section>;
  if (query.error) return <section className="p-card" role="alert"><div className="p-card-heading"><h3>Backend request failed</h3></div><p className="muted small">{apiConnectionMessage(query.error)}</p></section>;
  if (route === "developer") return null;
  if (!supported) {
    const gaps: Record<string, string> = {
      "science:Projects": "The backend exposes scientific status and datasets, but no project-list endpoint.",
      "finetune:Checkpoints": "No fine-tuning checkpoint-list endpoint is exposed.",
    };
    return <section className="p-card"><div className="p-card-heading"><h3>{tab}</h3></div><p className="muted small">{gaps[`${route}:${tab}`] ?? "This tab has no matching backend read endpoint yet."}</p></section>;
  }
  const sections = Object.entries(query.data ?? {}).flatMap(([key, response]) => Object.entries(response).flatMap(([collection, value]) => {
    if (Array.isArray(value)) return [{ name: collection, items: value.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) }];
    if (value && typeof value === "object") {
      const record = value as JsonRecord;
      const arrays = Object.entries(record).filter(([, nested]) => Array.isArray(nested));
      if (arrays.length) return arrays.map(([name, nested]) => ({ name, items: (nested as unknown[]).filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) }));
      const scalars = Object.fromEntries(Object.entries(record).filter(([field, item]) => !["secrets", "secret", "value", "token", "key", "ciphertext"].includes(field.toLowerCase()) && (typeof item === "string" || typeof item === "number" || typeof item === "boolean")));
      return Object.keys(scalars).length ? [{ name: key, items: [scalars] }] : [];
    }
    return [];
  }));
  const hasRows = sections.some((section) => section.items.length > 0);
  if (route === "science" && tab === "Experiments") {
    const records = Object.values(query.data ?? {}).flatMap((response) => Array.isArray(response.experiments)
      ? response.experiments.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item))
      : []);
    return <section id="advanced-live-data" className="p-card" aria-live="polite"><div className="p-card-heading"><h3>Saved proposals</h3><span className="p-fixture">{records.length} records</span></div>{!records.length ? <p className="muted small">No saved experiment proposals for this project.</p> : <div className="p-record-sections">{records.map((record, index) => { const proposals = Array.isArray(record.proposals) ? record.proposals.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : []; return <article className="p-record-section" key={String(record.id ?? index)}><h4>{recordTitle(record)}</h4><p className="caption">{String(record.status ?? "proposed")} · {String(record.execution_status ?? "not_started")} · {String(record.created_at ?? "")}</p>{typeof record.goal === "string" && <p className="caption">Goal: {record.goal}</p>}{proposals.map((proposal, proposalIndex) => <div className="p-record-section" key={String(proposal.id ?? proposalIndex)}><strong>{recordTitle(proposal)}</strong>{typeof proposal.rationale === "string" && <span className="caption">{proposal.rationale}</span>}{typeof proposal.variables === "object" && proposal.variables !== null ? <code className="caption">{JSON.stringify(proposal.variables)}</code> : null}</div>)}</article>; })}</div>}</section>;
  }
  if (route === "science" && tab === "Reports") {
    const executions = Object.values(query.data ?? {}).flatMap(response => Array.isArray(response.executions)
      ? response.executions.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item))
      : []);
    return <section id="advanced-live-data" className="p-card" aria-live="polite"><div className="p-card-heading"><h3>Recorded scientific executions</h3><span className="p-fixture">{executions.length} backend records</span></div>{!executions.length ? <p className="muted small">No bounded code execution results have been recorded for this project.</p> : <div className="p-record-sections">{executions.map((item, index) => <article className="p-record-section" key={String(item.id ?? index)}><div className="row between"><strong>{String(item.status ?? "Execution status unavailable")}</strong><span className="caption">Exit {String(item.returncode ?? "not reported")} · {String(item.created_at ?? "time unavailable")}</span></div><p className="caption">{String(item.elapsed_ms ?? "not reported")} ms · {String(item.attempts ?? "not reported")} attempt(s) · source SHA-256 {String(item.code_sha256 ?? "not recorded")}</p><details><summary>Inspect actual output</summary><pre className="terminal-window" style={{ whiteSpace: "pre-wrap" }}>{String(item.stdout ?? "")}{item.stderr ? `\n${String(item.stderr)}` : ""}</pre></details></article>)}</div>}</section>;
  }
  return <section id="advanced-live-data" className="p-card" aria-live="polite"><div className="p-card-heading"><h3>Live backend data</h3><span className="p-fixture">{hasRows ? "Verified response" : "Connected · no records"}</span></div>{!hasRows ? <p className="muted small">The backend returned no records for this view.</p> : <div className="p-record-sections">{sections.filter((section) => section.items.length).map((section) => <div className="p-record-section" key={section.name}><h4>{humanize(section.name)}</h4><ul>{section.items.slice(0, 12).map((record, index) => <li key={String(record.id ?? record.run_id ?? record.task_id ?? `${section.name}-${index}`)}><strong>{recordTitle(record)}</strong>{recordSummary(record, route) && <span className="caption">{recordSummary(record, route)}</span>}</li>)}</ul></div>)}</div>}</section>;
}

function SecurityAccessWorkspace({ data }: { data?: Record<string, JsonRecord> }) {
  const client = useQueryClient();
  const [purpose, setPurpose] = useState("");
  const [destination, setDestination] = useState("");
  const [resourceType, setResourceType] = useState("");
  const [shareType, setShareType] = useState("document");
  const [shareId, setShareId] = useState("");
  const [grantee, setGrantee] = useState("");
  const [shareExpiresAt, setShareExpiresAt] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const requests = Array.isArray(data?.egress?.requests)
    ? data.egress.requests.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item))
    : [];
  const grants = Array.isArray(data?.grants?.grants)
    ? data.grants.grants.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item))
    : [];
  const mutation = useMutation({
    mutationFn: ({ path, body }: { path: string; body: JsonRecord }) => apiPost(path, body),
    onSuccess: async (_result, variables) => {
      setError("");
      setNotice(variables.path.endsWith("/request") ? "The security service recorded the review request." : variables.path.endsWith("/resolve") ? "The security service recorded the decision." : "The security service updated the grant.");
      if (variables.path === "/api/security/egress/request") { setPurpose(""); setDestination(""); setResourceType(""); }
      if (variables.path === "/api/security/shares") { setShareId(""); setGrantee(""); setShareExpiresAt(""); }
      await client.invalidateQueries({ queryKey: ["advanced-screen", "security"] });
    },
    onError: (cause) => { setNotice(""); setError(apiConnectionMessage(cause)); },
  });
  const pending = (item: JsonRecord) => ["pending", "requested", "review"].includes(String(item.status ?? "").toLowerCase());
  const resolve = (request: JsonRecord, approved: boolean) => {
    const destinationName = String(request.destination ?? "this destination");
    const decision = approved ? "approve" : "deny";
    if (!window.confirm(`${decision === "approve" ? "Approve" : "Deny"} external access to ${destinationName}? The security service will record this decision.`)) return;
    mutation.mutate({ path: "/api/security/egress/resolve", body: { request_id: String(request.id ?? ""), approved, approved_by: "user" } });
  };
  const revokeGrant = (grant: JsonRecord) => {
    const grantId = String(grant.id ?? "");
    if (!grantId || !window.confirm(`Revoke the read grant for ${String(grant.grantee ?? "this grantee")}? The security service will record this decision.`)) return;
    mutation.mutate({ path: "/api/security/shares/resolve", body: { grant_id: grantId, status: "revoked" } });
  };
  const submit = (event: FormEvent) => {
    event.preventDefault();
    setNotice(""); setError("");
    mutation.mutate({ path: "/api/security/egress/request", body: {
      requester: "user", purpose: purpose.trim(), destination: destination.trim(),
      resource_type: resourceType.trim() || "unknown", resource_id: "", risk_level: "medium", metadata: {},
    } });
  };
  const createGrant = (event: FormEvent) => {
    event.preventDefault();
    const expiration = Date.parse(shareExpiresAt);
    if (!Number.isFinite(expiration) || expiration <= Date.now()) { setError("Choose a future expiry for this read grant."); return; }
    setNotice(""); setError("");
    mutation.mutate({ path: "/api/security/shares", body: { resource_type: shareType, resource_id: shareId.trim(), grantee: grantee.trim(), scope: "read", expires_at: new Date(expiration).toISOString(), metadata: {} } });
  };
  return <div className="p-record-sections">
    <section className="p-card">
      <div className="p-card-heading"><h3>External access requests</h3><span className="p-fixture">{requests.length} backend records</span></div>
      <p className="caption">A request asks for review; submitting it does not send data or grant access. Approve or deny only after reviewing the destination and purpose.</p>
      {!requests.length ? <p className="muted small">No external-access requests have been returned by the security service.</p> : requests.map((request, index) => <article className="p-record-section" key={String(request.id ?? index)}>
        <div className="row between"><strong>{String(request.destination ?? "Destination unavailable")}</strong><span className="caption">{String(request.status ?? "status unavailable")}</span></div>
        <p className="caption">{String(request.purpose ?? "Purpose unavailable")}</p>
        <p className="caption">{String(request.resource_type ?? "resource type unavailable")} · {String(request.created_at ?? "date unavailable")}</p>
        {pending(request) && <div className="row section-space"><button type="button" className="btn" disabled={mutation.isPending} onClick={() => resolve(request, true)}>Approve request</button><button type="button" className="btn danger" disabled={mutation.isPending} onClick={() => resolve(request, false)}>Deny request</button></div>}
      </article>)}
    </section>
    <section className="p-card">
      <div className="p-card-heading"><h3>Share grants</h3><span className="p-fixture">{grants.length} backend records</span></div>
      {!grants.length ? <p className="muted small">No share grants have been returned by the security service.</p> : grants.map((grant, index) => <article className="p-record-section" key={String(grant.id ?? index)}>
        <strong>{String(grant.resource_type ?? "resource")} · {String(grant.resource_id ?? "ID unavailable")}</strong>
        <p className="caption">Grantee {String(grant.grantee ?? "unreported")} · {String(grant.scope ?? "scope unavailable")} · {String(grant.status ?? "status unavailable")}</p>
        <p className="caption">Expires {String(grant.expires_at ?? "not reported")}</p>
        {grant.status === "active" && <button type="button" className="btn danger" disabled={mutation.isPending} onClick={() => revokeGrant(grant)}>Revoke read grant</button>}
      </article>)}
      <form className="section-space" onSubmit={createGrant}>
        <h4>Create read grant</h4>
        <label className="demo-form-field">Share resource type<input className="input" value={shareType} onChange={(event) => setShareType(event.target.value)} required maxLength={120} /></label>
        <label className="demo-form-field">Resource ID<input className="input" value={shareId} onChange={(event) => setShareId(event.target.value)} required maxLength={500} /></label>
        <label className="demo-form-field">Grantee<input className="input" value={grantee} onChange={(event) => setGrantee(event.target.value)} required maxLength={500} /></label>
        <label className="demo-form-field">Share expiry<input className="input" type="datetime-local" min={new Date(Date.now() - new Date().getTimezoneOffset() * 60_000 + 60_000).toISOString().slice(0, 16)} value={shareExpiresAt} onChange={(event) => setShareExpiresAt(event.target.value)} required /></label>
        <p className="caption">Creates a read-only grant with a required expiry. Only the backend can enforce whether this identity may receive access.</p>
        <button className="btn primary" disabled={mutation.isPending || !shareId.trim() || !grantee.trim() || !shareExpiresAt}>{mutation.isPending ? "Creating grant…" : "Create read grant"}</button>
      </form>
    </section>
    <section className="p-card">
      <div className="p-card-heading"><h3>Request external access</h3></div>
      <form onSubmit={submit}>
        <label className="demo-form-field">Purpose<input className="input" value={purpose} onChange={(event) => setPurpose(event.target.value)} required maxLength={500} /></label>
        <label className="demo-form-field">Destination<input className="input" value={destination} onChange={(event) => setDestination(event.target.value)} required maxLength={500} /></label>
        <label className="demo-form-field">External-access resource type<input className="input" value={resourceType} onChange={(event) => setResourceType(event.target.value)} maxLength={120} /></label>
        <p className="caption">This creates a review request only. It does not send data or approve access.</p>
        <button className="btn primary" disabled={mutation.isPending || !purpose.trim() || !destination.trim()}>{mutation.isPending ? "Submitting request…" : "Submit for review"}</button>
      </form>
      {error && <p className="caption" role="alert">{error}</p>}
      {notice && <p className="caption" role="status">{notice}</p>}
    </section>
  </div>;
}

const privacySummary = (value: JsonRecord): Array<{ label: string; value: string }> => {
  const secretSection = value.secrets && typeof value.secrets === "object" ? value.secrets as JsonRecord : {};
  const chain = value.audit_chain && typeof value.audit_chain === "object" ? value.audit_chain as JsonRecord : {};
  const inventory = value.data_inventory && typeof value.data_inventory === "object" ? value.data_inventory as JsonRecord : {};
  const inventoryCount = Object.values(inventory).filter((item): item is number => typeof item === "number").reduce((sum, item) => sum + item, 0);
  return [
    ...(value.generated_at != null ? [{ label: "Generated", value: String(value.generated_at) }] : []),
    ...(typeof value.network_default === "boolean" ? [{ label: "Network default", value: value.network_default ? "Allowed" : "Denied" }] : []),
    ...(secretSection.count != null ? [{ label: "Secret records", value: String(secretSection.count) }] : []),
    ...(Array.isArray(value.retention_policies) ? [{ label: "Retention policies", value: String(value.retention_policies.length) }] : []),
    ...(Array.isArray(value.share_grants) ? [{ label: "Share grants", value: String(value.share_grants.length) }] : []),
    ...(Array.isArray(value.egress_requests) ? [{ label: "External-access requests", value: String(value.egress_requests.length) }] : []),
    ...(Array.isArray(value.sandbox_profiles) ? [{ label: "Sandbox profiles", value: String(value.sandbox_profiles.length) }] : []),
    ...(typeof chain.ok === "boolean" ? [{ label: "Audit chain", value: chain.ok ? "Verified" : "Needs review" }] : []),
    ...(inventoryCount ? [{ label: "Tracked data records", value: String(inventoryCount) }] : []),
  ];
};

function SecuritySecretsWorkspace({ data }: { data?: Record<string, JsonRecord> }) {
  const client = useQueryClient();
  const secrets = Array.isArray(data?.secrets?.secrets) ? data.secrets.secrets.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const [name, setName] = useState("");
  const [newValue, setNewValue] = useState("");
  const [newPassphrase, setNewPassphrase] = useState("");
  const [passphrases, setPassphrases] = useState<Record<string, string>>({});
  const [replacements, setReplacements] = useState<Record<string, string>>({});
  const [revealed, setRevealed] = useState<Record<string, string>>({});
  const [feedback, setFeedback] = useState("");
  const [error, setError] = useState("");
  const timers = useRef(new Map<string, number>());
  const [busy, setBusy] = useState(false);
  useEffect(() => () => { for (const timer of timers.current.values()) window.clearTimeout(timer); timers.current.clear(); }, []);
  const refresh = () => client.invalidateQueries({ queryKey: ["advanced-screen", "security"] });
  const store = async (event: FormEvent) => {
    event.preventDefault(); setError(""); setFeedback(""); setBusy(true);
    const value = newValue; const passphrase = newPassphrase;
    setNewValue(""); setNewPassphrase("");
    try {
      await apiPost("/api/security/secrets", { name: name.trim(), value, passphrase, description: "" });
      setName(""); setFeedback("Encrypted secret stored. Its value is not returned to this list."); await refresh();
    } catch (cause) { setError(apiConnectionMessage(cause)); }
    finally { setBusy(false); }
  };
  const reveal = async (secret: JsonRecord) => {
    const id = String(secret.id ?? ""); const passphrase = passphrases[id] ?? "";
    if (!id || !passphrase || !window.confirm(`Reveal “${String(secret.name ?? "this secret")}” for 15 seconds? This access is recorded by the security service.`)) return;
    setError(""); setFeedback(""); setBusy(true);
    try {
      const response = await apiPost("/api/security/secrets/reveal", { secret_id: id, passphrase });
      const revealedSecret = response.secret && typeof response.secret === "object" ? response.secret as JsonRecord : {};
      if (typeof revealedSecret.value !== "string") throw new Error("The security service did not return a secret value.");
      const previous = timers.current.get(id); if (previous) window.clearTimeout(previous);
      setRevealed((current) => ({ ...current, [id]: String(revealedSecret.value) }));
      setPassphrases((current) => ({ ...current, [id]: "" }));
      timers.current.set(id, window.setTimeout(() => { timers.current.delete(id); setRevealed((current) => { const next = { ...current }; delete next[id]; return next; }); }, 15_000));
      setFeedback("Secret revealed briefly. The security service records this access."); await refresh();
    } catch (cause) { setError(apiConnectionMessage(cause)); }
    finally { setBusy(false); }
  };
  const conceal = (id: string) => {
    const timer = timers.current.get(id); if (timer) window.clearTimeout(timer); timers.current.delete(id);
    setRevealed((current) => { const next = { ...current }; delete next[id]; return next; });
  };
  const rotate = async (secret: JsonRecord) => {
    const id = String(secret.id ?? ""); const passphrase = passphrases[id] ?? ""; const replacement = replacements[id] ?? "";
    if (!id || !passphrase || !replacement || !window.confirm(`Replace the encrypted value for “${String(secret.name ?? "this secret")}”? The action is recorded by the security service.`)) return;
    setError(""); setFeedback(""); setBusy(true);
    try {
      await apiPost("/api/security/secrets/rotate", { secret_id: id, value: replacement, passphrase });
      setPassphrases((current) => ({ ...current, [id]: "" })); setReplacements((current) => ({ ...current, [id]: "" })); conceal(id);
      setFeedback("Secret value replaced."); await refresh();
    } catch (cause) { setError(apiConnectionMessage(cause)); }
    finally { setBusy(false); }
  };
  return <div className="p-record-sections">
    <section className="p-card"><div className="p-card-heading"><h3>Encrypted secrets</h3><span className="p-fixture">{secrets.length} backend records</span></div>
      <p className="caption">Secret values are never loaded in the list. Reveal requires the vault passphrase, confirmation and a 15-second conceal timer. Secret changes require the security:admin scope when auth enforcement is enabled.</p>
      {!secrets.length ? <p className="muted small">No secret metadata was returned by the security service.</p> : secrets.map((secret, index) => { const id = String(secret.id ?? index); return <article className="p-record-section" key={id}>
        <div className="row between"><strong>{String(secret.name ?? "Unnamed secret")}</strong><span className="caption">{secret.encrypted === true ? "Encrypted" : "Encryption status unavailable"}</span></div>
        <p className="caption">{String(secret.description ?? "")} · {String(secret.status ?? "status unavailable")} · updated {String(secret.updated_at ?? "date unavailable")}</p>
        <label className="demo-form-field">Passphrase<input className="input" type="password" autoComplete="current-password" value={passphrases[id] ?? ""} onChange={(event) => setPassphrases((current) => ({ ...current, [id]: event.target.value }))} /></label>
        {!revealed[id] && <label className="demo-form-field">Replacement value<input className="input" type="password" autoComplete="new-password" value={replacements[id] ?? ""} onChange={(event) => setReplacements((current) => ({ ...current, [id]: event.target.value }))} /></label>}
        <div className="row">{revealed[id] != null ? <><code style={{ overflowWrap: "anywhere" }}>{revealed[id]}</code><button className="btn" type="button" onClick={() => conceal(id)}>Conceal now</button></> : <button className="btn" type="button" disabled={busy || !passphrases[id]} onClick={() => void reveal(secret)}>Reveal for 15 seconds</button>}
          <button className="btn danger" type="button" disabled={busy || !passphrases[id] || !replacements[id]} onClick={() => void rotate(secret)}>Replace secret value</button></div>
      </article>; })}
    </section>
    <section className="p-card"><div className="p-card-heading"><h3>Store a secret</h3></div>
      <form onSubmit={store}><label className="demo-form-field">Name<input className="input" value={name} onChange={(event) => setName(event.target.value)} required autoComplete="off" maxLength={200} /></label>
        <label className="demo-form-field">Secret value<input className="input" type="password" value={newValue} onChange={(event) => setNewValue(event.target.value)} required autoComplete="new-password" /></label>
        <label className="demo-form-field">Encryption passphrase<input className="input" type="password" value={newPassphrase} onChange={(event) => setNewPassphrase(event.target.value)} required autoComplete="current-password" /></label>
        <p className="caption">The value and passphrase are sent only on submit and cleared from the form immediately afterward. They are never rendered from the response.</p>
        <button className="btn primary" disabled={busy || !name.trim() || !newValue || !newPassphrase}>{busy ? "Saving…" : "Encrypt and store"}</button>
      </form>
    </section>
    {error && <p className="caption" role="alert">{error}</p>}{feedback && <p className="caption" role="status">{feedback}</p>}
  </div>;
}

function SecurityWorkspace({ tab, data, query, demo }: { tab: string; data?: Record<string, JsonRecord>; query: { isLoading: boolean; error: unknown }; demo: boolean }) {
  const client = useQueryClient();
  const [feedback, setFeedback] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [resourceType, setResourceType] = useState("context_events");
  const [retentionDays, setRetentionDays] = useState("365");
  const [profileName, setProfileName] = useState("");
  const [profilePermissions, setProfilePermissions] = useState("");
  const [profileNetwork, setProfileNetwork] = useState(false);
  const [selectedProfile, setSelectedProfile] = useState("");
  const [requestedPermissions, setRequestedPermissions] = useState("");
  const [requestNetwork, setRequestNetwork] = useState(false);
  const [sandboxResult, setSandboxResult] = useState<JsonRecord>({});
  const [dryRunResult, setDryRunResult] = useState<JsonRecord>({});
  const policies = Array.isArray(data?.retention?.policies) ? data.retention.policies.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const profiles = Array.isArray(data?.profiles?.profiles) ? data.profiles.profiles.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const reports = Array.isArray(data?.reports?.reports) ? data.reports.reports.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const auditEvents = Array.isArray(data?.events?.events) ? data.events.events.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const sessions = Array.isArray(data?.sessions?.sessions) ? data.sessions.sessions.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const status = data?.status ?? {};
  const refresh = () => client.invalidateQueries({ queryKey: ["advanced-screen", "security"] });
  const run = async (path: string, body: JsonRecord, actionLabel: string) => {
    setError(""); setFeedback(""); setBusy(true);
    try { const result = await apiPost(path, body); setFeedback(actionLabel); await refresh(); return result; }
    catch (cause) { setError(apiConnectionMessage(cause)); return undefined; }
    finally { setBusy(false); }
  };
  const policyTypes = ["context_events", "agent_runs", "workflow_runs", "developer_webhook_deliveries"];
  const notices = <>{error && <p className="caption" role="alert">{error}</p>}{feedback && <p className="caption" role="status">{feedback}</p>}</>;
  if (demo) return <SecurityDemoWorkspace tab={tab} data={data} />;
  if (query.isLoading) return <section className="p-card" role="status"><p className="caption">Loading security records…</p></section>;
  if (query.error) return <section className="p-card" role="alert"><h3>Security service unavailable</h3><p className="caption">{apiConnectionMessage(query.error)}</p></section>;
  if (tab === "Overview" && demo) {
    const demoPolicies = Array.isArray(data?.policies?.policies) ? data.policies.policies.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
    return <>
      <section className="demo-feature-hero"><div className="demo-feature-symbol"><Shield className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">Overview · Workspace</span><h2>Clear visibility and control</h2><p>Understand policy, approvals and activity.</p></div></section>
      <section className="p-card p-tight"><div className="p-card-heading"><h3>Security summary</h3></div><div className="demo-stats-grid"><div><strong>{demoPolicies.length}</strong><small>Active policies</small></div><div><strong>{sessions.length}</strong><small>Sessions</small></div><div><strong>{auditEvents.length}</strong><small>Audit entries</small></div></div><div className="p-note"><Shield className="icon sm" aria-hidden="true"/><span>This is a simulated security dashboard: no real authorisation, session revocation or audit verification takes place.</span></div></section>
      <section className="p-card p-tight"><div className="p-card-heading"><h3>Access · demo records</h3><span className="p-fixture">{demoPolicies.length} policies</span></div>{demoPolicies.map((policy, index) => <article className="demo-record" key={String(policy.id ?? index)}><div className="square-icon"><Shield className="icon sm" aria-hidden="true"/></div><div className="demo-record-text"><strong>{String(policy.name ?? "Access policy")}</strong><small>{String(policy.description ?? policy.scope ?? "Workspace access")}</small></div><span className="tag green">{String(policy.status ?? "Enabled")}</span></article>)}</section>
    </>;
  }
  if (tab === "Overview") return <>{notices}<div className="p-record-sections">
    <section className="p-card"><div className="p-card-heading"><h3>Security posture</h3><span className="p-fixture">Live backend status</span></div>
      <p className="caption">Outbound network default: {status.network_default === false ? "denied" : status.network_default === true ? "allowed" : "not reported"} · Audit chain: {status.audit_chain_ok === true ? "verified" : status.audit_chain_ok === false ? "needs review" : "not reported"}</p>
      <div className="p-metrics-grid">{[["External-access requests", "egress_requests"], ["Share grants", "share_grants"], ["Encrypted secrets", "secrets"], ["Retention policies", "retention_policies"], ["Privacy reports", "privacy_reports"], ["Sandbox profiles", "sandbox_profiles"]].map(([label, key]) => <div className="p-stat" key={key}><Shield className="p-stat-icon icon" aria-hidden="true" /><small>{label}</small><strong>{typeof status[key] === "number" ? String(status[key]) : "—"}</strong><span>{typeof status[key] === "number" ? "Backend count" : "Not reported"}</span></div>)}</div>
    </section>
  </div></>;
  if (tab === "Access") return <SecurityAccessWorkspace data={data} />;
  if (tab === "Sessions") return <>{notices}<section className="p-card"><div className="p-card-heading"><h3>Authenticated sessions</h3><span className="p-fixture">{sessions.length} backend records</span></div><p className="caption">Session details are returned for the authenticated account. This panel does not expose sessions belonging to other users.</p>{!sessions.length ? <p className="muted small">No active sessions were returned. Authentication may be required for this view.</p> : sessions.map((session, index) => <article className="p-record-section" key={String(session.session_id ?? index)}><strong>{String(session.session_id ?? "Session")}</strong><p className="caption">{String(session.ip ?? "Address unavailable")} · last active {String(session.last_seen_at ?? "not reported")} · expires {String(session.expires_at ?? "not reported")}</p></article>)}</section></>;
  if (tab === "Audit") return <>
    {notices}
    <div className="p-record-sections">
      <section className="p-card">
        <div className="p-card-heading">
          <h3>Audit events</h3>
          <span className="p-fixture">{auditEvents.length} events | project default</span>
          <button className="btn" type="button" onClick={() => void refresh()}>Refresh</button>
          <button className="btn primary" type="button" disabled={busy} onClick={async () => { const result = await run("/api/security/audit-chain/verify", {}, "Audit-chain verification completed."); if (result) setDryRunResult(result); }}>{busy ? "Verifying..." : "Verify chain"}</button>
        </div>
        <p className="caption">Read-only events for project <code>default</code>. Viewing requires <code>security:admin</code>. Event metadata is not displayed.</p>
        {auditEvents.length ? <div className="p-record-sections" aria-label="Security audit events">{auditEvents.map((event, index) => {
          const statusText = String(event.status ?? "Outcome not reported");
          const statusClass = /success|allowed|complete/i.test(statusText) ? "green" : /fail|deny|error/i.test(statusText) ? "red" : "";
          return <article className="p-record-section" data-audit-event-id={String(event.id ?? index)} key={String(event.id ?? index)}>
            <div className="row between"><strong>{String(event.action ?? "Audit event")}</strong><span className={`tag ${statusClass}`}>{statusText}</span></div>
            <p className="caption">{String(event.actor ?? "Actor not reported")} | {String(event.resource ?? "Resource not reported")} | {String(event.created_at ?? "Time not reported")}</p>
          </article>;
        })}</div> : <div className="p-empty" role="status"><Activity className="icon lg" aria-hidden="true"/><h3>No audit events for this project</h3><p className="muted small">The backend returned no audit records for project default.</p></div>}
      </section>
      {Object.keys(dryRunResult).length > 0 && <section className="p-card"><div className="p-card-heading"><h3>Verification result</h3></div><p className="caption">{dryRunResult.ok === true ? "Audit chain verified" : dryRunResult.ok === false ? "Audit chain needs review" : "Backend returned a verification result"} | {String(dryRunResult.sealed ?? "unknown")} sealed records | {Array.isArray(dryRunResult.issues) ? dryRunResult.issues.length : 0} issues</p></section>}
    </div>
  </>;
  if (tab === "Privacy") return <>{notices}<div className="p-record-sections"><section className="p-card"><div className="p-card-heading"><h3>Privacy reports</h3><span className="p-fixture">{reports.length} saved</span><button className="btn primary" type="button" disabled={busy} onClick={async () => { const result = await run("/api/security/privacy-report", {}, "Privacy report generated and stored by the security service."); if (result) setDryRunResult(result); }}>{busy ? "Generating…" : "Generate report"}</button></div><p className="caption">Reports are stored by the backend. This summary shows counts and posture only; secret names and report contents are not displayed. Creating reports requires security:admin when auth enforcement is enabled.</p>{reports.map((item, index) => { const report = item.report && typeof item.report === "object" ? item.report as JsonRecord : {}; return <article className="p-record-section" key={String(item.id ?? index)}><strong>{String(item.created_at ?? report.generated_at ?? "Privacy report")}</strong><div>{privacySummary(report).map((fact) => <p className="caption" key={fact.label}>{fact.label}: {fact.value}</p>)}</div></article>; })}{!reports.length && !Object.keys(dryRunResult).length && <p className="muted small">No privacy report is stored.</p>}{typeof dryRunResult.report === "object" && dryRunResult.report !== null && <div className="p-record-section"><h4>New report summary</h4>{privacySummary(dryRunResult.report as JsonRecord).map((fact) => <p className="caption" key={fact.label}>{fact.label}: {fact.value}</p>)}</div>}</section></div></>;
  if (tab === "Retention") return <>{notices}<div className="p-record-sections"><section className="p-card"><div className="p-card-heading"><h3>Retention policies</h3><span className="p-fixture">{policies.length} backend records</span></div><p className="caption">Preview is read-only. Applying an active policy permanently deletes matching records. Writes require security:admin when auth enforcement is enabled.</p>{policies.map((policy, index) => <article className="p-record-section" key={String(policy.id ?? index)}><div className="row between"><strong>{String(policy.resource_type ?? "Policy")}</strong><span className="caption">{String(policy.status ?? "status unavailable")}</span></div><p className="caption">Keep {String(policy.retention_days ?? "—")} days · {String(policy.action ?? "action unavailable")}</p><div className="row"><button className="btn" type="button" disabled={String(policy.status) !== "active"} onClick={async () => { const result = await run("/api/security/retention/apply", { policy_id: String(policy.id ?? ""), dry_run: true }, "Preview completed. No records were deleted."); if (result) setDryRunResult(result); }}>Preview deletions</button><button className="btn danger" type="button" disabled={String(policy.status) !== "active"} onClick={async () => { if (!window.confirm(`Permanently delete records older than ${String(policy.retention_days ?? "the configured retention period")} days for ${String(policy.resource_type ?? "this category")}? This cannot be undone.`)) return; const result = await run("/api/security/retention/apply", { policy_id: String(policy.id ?? ""), dry_run: false }, "Retention policy applied."); if (result) setDryRunResult(result); }}>Apply deletion</button></div></article>)}</section>
    <section className="p-card"><div className="p-card-heading"><h3>Create retention policy</h3></div><form onSubmit={async (event) => { event.preventDefault(); const result = await run("/api/security/retention", { resource_type: resourceType, retention_days: Number(retentionDays), action: "delete", status: "active", metadata: {} }, "Retention policy saved."); if (result) { setResourceType("context_events"); setRetentionDays("365"); } }}><label className="demo-form-field">Record category<select className="select" value={resourceType} onChange={(event) => setResourceType(event.target.value)}>{policyTypes.map((type) => <option key={type} value={type}>{humanize(type)}</option>)}</select></label><label className="demo-form-field">Keep for days<input className="input" type="number" min="0" value={retentionDays} onChange={(event) => setRetentionDays(event.target.value)} required /></label><button className="btn primary">Save policy</button></form>{dryRunResult.policy_id != null && <p className="caption" role="status">Preview: {String(dryRunResult.planned ?? "—")} records matched · {String(dryRunResult.deleted ?? 0)} deleted · cutoff {String(dryRunResult.cutoff ?? "not reported")}</p>}</section></div></>;
  if (tab === "Sandbox") return <>{notices}<div className="p-record-sections"><section className="p-card"><div className="p-card-heading"><h3>Sandbox profiles</h3><span className="p-fixture">{profiles.length} backend records</span></div><p className="caption">These profiles describe policy only; creating a profile does not create or certify an operating-system sandbox. Changes require security:admin when auth enforcement is enabled.</p>{profiles.map((profile, index) => <article className="p-record-section" key={String(profile.id ?? index)}><label className="row"><input type="radio" name="sandbox-profile" value={String(profile.id ?? "")} checked={selectedProfile === String(profile.id ?? "")} onChange={() => setSelectedProfile(String(profile.id ?? ""))} /><strong>{String(profile.name ?? "Profile")}</strong></label><p className="caption">Permissions: {Array.isArray(profile.permissions) ? profile.permissions.map(String).join(", ") || "none" : "not reported"} · network {profile.network_allowed === true ? "allowed" : "denied"}</p></article>)}</section>
    <section className="p-card"><div className="p-card-heading"><h3>Create policy profile</h3></div><form onSubmit={async (event) => { event.preventDefault(); if (!window.confirm(`Save profile “${profileName.trim()}” with network ${profileNetwork ? "allowed" : "denied"}? This stores policy only; it does not enforce an OS sandbox.`)) return; const result = await run("/api/security/sandbox-profiles", { name: profileName.trim(), permissions: profilePermissions.split(/[\s,]+/).filter(Boolean), network_allowed: profileNetwork, filesystem: {}, env: {}, metadata: {} }, "Sandbox profile saved."); if (result) { setProfileName(""); setProfilePermissions(""); setProfileNetwork(false); } }}><label className="demo-form-field">Profile name<input className="input" value={profileName} onChange={(event) => setProfileName(event.target.value)} required maxLength={200} /></label><label className="demo-form-field">Allowed permissions, separated by commas<textarea className="input" rows={3} value={profilePermissions} onChange={(event) => setProfilePermissions(event.target.value)} placeholder="filesystem.read, process.spawn" /></label><label className="row"><input type="checkbox" checked={profileNetwork} onChange={(event) => setProfileNetwork(event.target.checked)} />Allow network access</label><button className="btn primary" disabled={!profileName.trim()}>Save profile</button></form></section>
    <section className="p-card"><div className="p-card-heading"><h3>Evaluate access request</h3></div><form onSubmit={async (event) => { event.preventDefault(); const result = await run("/api/security/sandbox/evaluate", { profile_id: selectedProfile, permissions: requestedPermissions.split(/[\s,]+/).filter(Boolean), network: requestNetwork }, "Sandbox policy evaluation completed."); if (result) setSandboxResult(result); }}><label className="demo-form-field">Profile<select className="select" value={selectedProfile} onChange={(event) => setSelectedProfile(event.target.value)} required><option value="">Select a profile</option>{profiles.map((profile, index) => <option key={String(profile.id ?? index)} value={String(profile.id ?? "")}>{String(profile.name ?? "Profile")}</option>)}</select></label><label className="demo-form-field">Requested permissions<input className="input" value={requestedPermissions} onChange={(event) => setRequestedPermissions(event.target.value)} placeholder="filesystem.read, process.spawn" /></label><label className="row"><input type="checkbox" checked={requestNetwork} onChange={(event) => setRequestNetwork(event.target.checked)} />Request network access</label><button className="btn" disabled={!selectedProfile}>Evaluate only</button></form>{Object.keys(sandboxResult).length > 0 && <div className="p-note" role="status"><strong>{sandboxResult.allowed === true ? "Allowed by this profile" : "Denied by this profile"}</strong><p className="caption">Requested network: {sandboxResult.requested_network === true ? "yes" : "no"} · Permission requirements: {Array.isArray(sandboxResult.requested_permissions) ? sandboxResult.requested_permissions.join(", ") || "none" : "not reported"}</p></div>}</section></div></>;
  return <SecuritySecretsWorkspace data={data} />;
}

function BackendActions({ route, tab, data }: { route: string; tab: string; data?: Record<string, JsonRecord> }) {
  const queryClient = useQueryClient();
  const [values, setValues] = useState<Record<string, string>>({});
  const [message, setMessage] = useState("");
  const [oneTimeKey, setOneTimeKey] = useState("");
  const [allowExternal, setAllowExternal] = useState(false);
  const [selectedScopes, setSelectedScopes] = useState<string[]>([]);
  const [activateWebhook, setActivateWebhook] = useState(false);
  const [confirmWebhookActivation, setConfirmWebhookActivation] = useState(false);
  useEffect(() => { setValues({}); setMessage(""); setOneTimeKey(""); setAllowExternal(false); setSelectedScopes([]); setActivateWebhook(false); setConfirmWebhookActivation(false); }, [route, tab]);
  const mutation = useMutation({
    mutationFn: ({ path, body }: { path: string; body: unknown }) => apiPost(path, body),
    onSuccess: async (result, variables) => {
      setMessage("The backend accepted the request.");
      const key = result.api_key as JsonRecord | undefined;
      if (variables.path === "/api/developer/api-keys" && key && typeof (key.key ?? key.token) === "string") setOneTimeKey(String(key.key ?? key.token));
      if (route === "vault") setValues((current) => ({ ...current, secret: "", passphrase: "" }));
      await queryClient.invalidateQueries({ queryKey: ["advanced-screen", route] });
    },
    onError: (error) => setMessage(apiConnectionMessage(error)),
  });
  const update = (key: string, value: string) => setValues((current) => ({ ...current, [key]: value }));
  const field = (key: string, label: string, type = "text", placeholder = "") => <label className="demo-form-field" key={key}>{label}<input className="input" type={type} value={values[key] ?? ""} placeholder={placeholder} onChange={(event) => update(key, event.target.value)} required /></label>;
  const submit = (event: FormEvent, path: string, body: JsonRecord) => {
    event.preventDefault();
    if (route === "scraper" && !window.confirm(allowExternal ? "Submit this extraction request with external network permission? The server still applies egress and robots rules." : "Submit this extraction request? External access remains disabled and server policy still applies.")) return;
    setMessage(""); setOneTimeKey(""); mutation.mutate({ path, body });
  };
  let form: React.ReactNode = null;

  if (route === "assistant" && ["Today", "Tasks"].includes(tab)) form = null;
  if (route === "assistant" && tab === "Goals") form = <form onSubmit={(event) => submit(event, "/api/assistant/goals", { title: values.title?.trim(), description: values.description ?? "", project: "default" })}>{field("title", "Goal", "text", "A goal you want to track")}{field("description", "Description")}<button className="btn primary" disabled={mutation.isPending || !values.title?.trim()}>{mutation.isPending ? "Saving…" : "Add goal"}</button></form>;
  if (route === "security" && tab === "Audit") form = <div><p className="caption">Verify the integrity of the audit history using the security service.</p><button type="button" className="btn primary" disabled={mutation.isPending} onClick={() => { setMessage(""); setOneTimeKey(""); mutation.mutate({ path: "/api/security/audit-chain/verify", body: {} }); }}>{mutation.isPending ? "Verifying…" : "Verify audit chain"}</button></div>;
  const availableScopes = Array.isArray(data?.scopes?.scopes) ? data.scopes.scopes.filter((scope): scope is JsonRecord => Boolean(scope) && typeof scope === "object" && !Array.isArray(scope) && typeof (scope as JsonRecord).scope === "string") : [];
  if (route === "developer" && tab === "API keys") form = <form onSubmit={(event) => submit(event, "/api/developer/api-keys", { label: values.label?.trim(), scopes: selectedScopes })}>{field("label", "Key label", "text", "CI integration")}
    {!availableScopes.length ? <p className="caption" role="status">Permission catalog unavailable. Load verified scopes before creating a key.</p> : <fieldset className="p-record-section"><legend>Permissions</legend><div className="p-record-sections">{availableScopes.map((item) => { const scope = String(item.scope); return <label className="row" key={scope}><input type="checkbox" checked={selectedScopes.includes(scope)} onChange={(event) => setSelectedScopes((current) => event.target.checked ? [...current, scope] : current.filter((value) => value !== scope))} /><span><strong>{scope}</strong><span className="caption">{String(item.description ?? "")}</span></span></label>; })}</div></fieldset>}
    <p className="caption">Choose the minimum permissions needed. The token is returned once and cannot be retrieved later.</p><button className="btn primary" disabled={mutation.isPending || !values.label?.trim() || !selectedScopes.length || !availableScopes.length}>{mutation.isPending ? "Creating…" : "Create API key"}</button>{oneTimeKey && <label className="demo-form-field">New API token (shown once)<input className="input" type="password" autoComplete="new-password" readOnly value={oneTimeKey} onFocus={(event) => event.currentTarget.select()} /></label>}</form>;
  if (route === "developer" && tab === "Webhooks") form = <form onSubmit={(event) => submit(event, "/api/developer/webhooks", { event: values.event?.trim() || "*", target_url: values.target_url?.trim(), enabled: activateWebhook, secret_hint: "", metadata: {} })}>{field("event", "Event pattern", "text", "openagent.task.completed")}{field("target_url", "Target URL", "url", "https://example.org/webhook")}
    <label className="row"><input type="checkbox" checked={activateWebhook} onChange={(event) => { setActivateWebhook(event.target.checked); setConfirmWebhookActivation(false); }} />Enable immediately for matching future events</label>
    {activateWebhook && <label className="row"><input type="checkbox" checked={confirmWebhookActivation} onChange={(event) => setConfirmWebhookActivation(event.target.checked)} />I understand this can cause outbound requests to the target URL.</label>}
    <p className="caption">Disabled by default. Registration never sends a test request. The backend has no update endpoint, so activation cannot be changed here after registration.</p><button className="btn primary" disabled={mutation.isPending || !values.target_url?.trim() || (activateWebhook && !confirmWebhookActivation)}>{mutation.isPending ? "Registering…" : "Register webhook"}</button></form>;
  if (route === "developer" && tab === "MCP") form = <form onSubmit={(event) => {
    event.preventDefault();
    try {
      const args = JSON.parse(values.args ?? "[]") as unknown;
      if (!Array.isArray(args) || args.some((arg) => typeof arg !== "string")) throw new Error("Arguments must be a JSON array of strings.");
      setMessage("");
      mutation.mutate({ path: "/api/developer/mcp/servers", body: { name: values.name?.trim(), command: values.command?.trim(), args, env: {}, enabled: false, metadata: {} } });
    } catch (error) { setMessage(error instanceof Error ? error.message : "Enter command arguments as a JSON string array."); }
  }}><p className="caption">Register a disabled definition. This does not execute the command. Environment variables and secret values are intentionally not accepted here.</p>{field("name", "Server name")}{field("command", "Executable command")}
    <label className="demo-form-field">Arguments (JSON string array)<textarea className="input" rows={3} value={values.args ?? "[]"} onChange={(event) => update("args", event.target.value)} placeholder='["--stdio"]' /></label>
    <button className="btn primary" disabled={mutation.isPending || !values.name?.trim() || !values.command?.trim()}>{mutation.isPending ? "Registering…" : "Register disabled server"}</button></form>;
  if (route === "science" && tab === "Experiments") form = <form onSubmit={(event) => {
    event.preventDefault();
    try {
      const candidates = JSON.parse(values.candidates ?? "") as unknown;
      if (!Array.isArray(candidates) || candidates.some((candidate) => !candidate || typeof candidate !== "object" || Array.isArray(candidate))) throw new Error("Enter a JSON array of candidate parameter objects.");
      setMessage("");
      mutation.mutate({ path: "/api/science/experiments", body: { goal: values.goal?.trim(), candidates } });
    } catch (error) { setMessage(error instanceof Error ? error.message : "Candidate parameters must be a JSON array."); }
  }}><p className="caption">The backend stores this proposal as proposed; it is not an executed experiment run.</p>{field("goal", "Research question", "text", "Describe what you want to investigate")}<label className="demo-form-field">Candidate parameter objects<textarea className="input" rows={4} value={values.candidates ?? ""} onChange={(event) => update("candidates", event.target.value)} placeholder={'[{"temperature_c": 40}]'} required /></label><button className="btn primary" disabled={mutation.isPending || !values.goal?.trim() || !values.candidates?.trim()}>{mutation.isPending ? "Requesting proposal…" : "Generate experiment proposal"}</button></form>;
  if (route === "science" && tab === "Reports") form = <form onSubmit={(event) => submit(event, "/api/science/execute", { code: values.code ?? "", project: "default", approved: values.approved === "true", timeout_seconds: 10, memory_limit_mb: 512, retries: 0 })}><label className="demo-form-field">Python source<textarea className="input" rows={8} value={values.code ?? ""} onChange={(event) => update("code", event.target.value)} placeholder="print(1 + 1)" required /></label><p className="caption">Runs only after explicit approval using the backend’s policy-checked bounded runner. Source code is not saved; the actual result and source hash are kept in this project’s history.</p><label className="row"><input type="checkbox" checked={values.approved === "true"} onChange={(event) => update("approved", event.target.checked ? "true" : "false")} />I reviewed this Python source and approve one bounded local execution.</label><button className="btn primary" disabled={mutation.isPending || !values.code?.trim() || values.approved !== "true"}>{mutation.isPending ? "Running approved code…" : "Run approved code"}</button></form>;
  if (route === "scraper" && tab === "New extraction") form = <form onSubmit={(event) => submit(event, "/api/scraper/scrape", { url: values.url?.trim(), project: "default", respect_robots: true, ingest: values.ingest === "true", allow_external: allowExternal })}>{field("url", "Website URL", "url", "https://example.org/page")}<label className="demo-form-field">Knowledge ingestion<select className="select" value={values.ingest ?? "false"} onChange={(event) => update("ingest", event.target.value)}><option value="false">Do not ingest</option><option value="true">Ingest extracted content</option></select></label><label className="row"><input type="checkbox" checked={allowExternal} onChange={(event) => setAllowExternal(event.target.checked)} />I approve an external network request for this URL.</label><p className="caption">Robots rules are respected. The extraction will only be sent after you submit this form.</p><button className="btn primary" disabled={mutation.isPending || !values.url?.trim()}>{mutation.isPending ? "Requesting extraction…" : "Run extraction"}</button></form>;
  if (route === "scraper" && tab === "Jobs") form = <form onSubmit={(event) => submit(event, "/api/scraper/schedule", { url: values.url?.trim(), label: values.label?.trim() ?? "", interval_hours: Number(values.interval_hours || 24), project: "default", allow_external: allowExternal })}>{field("url", "Website URL", "url", "https://example.org/page")}{field("label", "Job label", "text", "Optional label")}{field("interval_hours", "Repeat interval (hours)", "number", "24")}<label className="row"><input type="checkbox" checked={allowExternal} onChange={(event) => setAllowExternal(event.target.checked)} />I approve recurring external network requests for this URL.</label><button className="btn primary" disabled={mutation.isPending || !values.url?.trim()}>{mutation.isPending ? "Scheduling…" : "Schedule extraction"}</button></form>;
  if (route === "finetune" && tab === "Datasets") form = <form onSubmit={(event) => submit(event, "/api/finetune/datasets", { name: values.name?.trim(), description: values.description ?? "", format: "jsonl", project: "default" })}>{field("name", "Dataset name", "text", "Training dataset")}{field("description", "Description")}<button className="btn primary" disabled={mutation.isPending || !values.name?.trim()}>{mutation.isPending ? "Creating…" : "Create dataset"}</button></form>;
  if (route === "finetune" && tab === "Jobs") form = <form onSubmit={(event) => submit(event, "/api/finetune/jobs", { dataset_id: values.dataset_id?.trim(), base_model: values.base_model?.trim(), method: "lora", project: "default" })}>{field("dataset_id", "Dataset ID", "text", "Use an ID from the loaded dataset list")}{field("base_model", "Base model", "text", "Model identifier")}<p className="caption">Starting this job asks the backend to create a fine-tuning job. It does not imply that training has started.</p><button className="btn primary" disabled={mutation.isPending || !values.dataset_id?.trim() || !values.base_model?.trim()}>{mutation.isPending ? "Submitting…" : "Create training job"}</button></form>;
  if (route === "engine" && tab === "Plans") form = <form onSubmit={(event) => submit(event, "/api/compute/plan", { task: values.task?.trim(), model: values.model?.trim() ?? "", allow_external: allowExternal, target_latency_ms: Number(values.target_latency_ms || 0), memory_budget_gb: Number(values.memory_budget_gb || 0), require_streaming: values.require_streaming === "true" })}>{field("task", "Task type", "text", "chat, code, embeddings…")}{field("model", "Preferred model (optional)", "text", "Model identifier")}{field("target_latency_ms", "Target latency (ms)", "number", "0")}{field("memory_budget_gb", "Memory budget (GB)", "number", "0")}<label className="row"><input type="checkbox" checked={allowExternal} onChange={(event) => setAllowExternal(event.target.checked)} />Allow external providers for this plan.</label><label className="row"><input type="checkbox" checked={values.require_streaming === "true"} onChange={(event) => update("require_streaming", event.target.checked ? "true" : "false")} />Require streaming support.</label><button className="btn primary" disabled={mutation.isPending || !values.task?.trim()}>{mutation.isPending ? "Planning…" : "Ask backend for a compute plan"}</button></form>;
  if (route === "vault" && tab === "Secrets") form = <form onSubmit={(event) => submit(event, "/api/vault/store", { label: values.label?.trim(), secret: values.secret ?? "", passphrase: values.passphrase ?? "", category: values.category?.trim() || "secret", project: "default" })}>{field("label", "Secret label", "text", "Credential name")}{field("secret", "Secret value", "password")}{field("passphrase", "Vault passphrase", "password")}<p className="caption">The value and passphrase are sent only when you submit. Secret contents are never shown in the response list.</p><button className="btn primary" disabled={mutation.isPending || !values.label?.trim() || !values.secret || !values.passphrase}>{mutation.isPending ? "Encrypting…" : "Store encrypted secret"}</button></form>;
  if (!form) return null;
  return <section id="advanced-backend-action" className="p-card"><div className="p-card-heading"><h3>{route === "scraper" ? "Submit a permitted request" : "Backend action"}</h3></div>{form}{message && <p className="caption" role={mutation.isError ? "alert" : "status"}>{message}</p>}{mutation.data && <ActionResult route={route} result={mutation.data} />}{data?.datasets && route === "finetune" && <p className="caption">Dataset list has been loaded from the backend above; enter its dataset ID to create a job.</p>}</section>;
}

function scrollToPanel(id: string) { document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "center" }); }

function ConnectionRail({ definition, route, connectionLabel, data, onTabChange, tab = "" }: { definition: Definition; route: string; connectionLabel: string; data?: Record<string, JsonRecord>; onTabChange?: (tab: string) => void; tab?: string }) {
  const Icon = routeIcons[route] ?? definition.icon;
  const dataMode = useWorkspaceState(state => state.dataMode);
  const queryClient = useQueryClient();
  const [vaultActivity, setVaultActivity] = useState<VaultDemoEvent[]>(readVaultDemoEvents);
  const [demoRailStatus, setDemoRailStatus] = useState("");
  const [demoRailBusy, setDemoRailBusy] = useState(false);
  const demoRailActionSequence = useRef(0);
  useEffect(() => {
    demoRailActionSequence.current += 1;
    setDemoRailStatus("");
    setDemoRailBusy(false);
  }, [route, tab, dataMode]);
  useEffect(() => {
    const update = (event: Event) => { const value = (event as CustomEvent<unknown>).detail; setVaultActivity(Array.isArray(value) ? value.filter((item): item is VaultDemoEvent => Boolean(item) && typeof item === "object" && typeof (item as VaultDemoEvent).name === "string").slice(0, 5) : []); };
    const reset = () => setVaultActivity([]);
    window.addEventListener("oa-vault-demo-activity", update);
    window.addEventListener("oa-vault-demo-reset", reset);
    return () => { window.removeEventListener("oa-vault-demo-activity", update); window.removeEventListener("oa-vault-demo-reset", reset); };
  }, []);
  const runDemoRailAction = async (operation: () => Promise<void>, success: string) => {
    if (demoRailBusy || dataMode !== "demo") return;
    const actionSequence = ++demoRailActionSequence.current;
    setDemoRailBusy(true);
    setDemoRailStatus("");
    try {
      await operation();
      if (demoRailActionSequence.current === actionSequence) setDemoRailStatus(success);
    } catch (error) {
      if (demoRailActionSequence.current === actionSequence) setDemoRailStatus(`Could not complete this local demo action: ${error instanceof Error ? error.message : "Please try again."}`);
    } finally {
      if (demoRailActionSequence.current === actionSequence) setDemoRailBusy(false);
    }
  };
  const exportDemoRecords = (exportRoute: string, exportTab: string, filename: string, payload?: unknown) => {
    void runDemoRailAction(async () => {
      const data = payload ?? await advancedDemoScreen(exportRoute, exportTab);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      try {
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        link.click();
      } finally {
        window.setTimeout(() => URL.revokeObjectURL(url), 0);
      }
    }, "Demo data export downloaded locally.");
  };
  const resetDemoRecords = (resetRoute: string, afterReset?: () => void) => {
    void runDemoRailAction(async () => {
      await resetAdvancedDemoRoute(resetRoute);
      afterReset?.();
      await queryClient.invalidateQueries({ queryKey: ["advanced-screen", resetRoute] });
    }, "Local demo data was reset to the approved examples.");
  };
  const demoRailFeedback = demoRailStatus ? <p className="caption section-space" role={demoRailStatus.startsWith("Could not") ? "alert" : "status"}>{demoRailStatus}</p> : null;
  const communicationsActivity = useQuery({
    queryKey: ["communications", "sync-runs", "default"],
    queryFn: ({ signal }) => apiGet<{ runs?: JsonRecord[] }>("/api/communication/sync-runs?project=default&limit=20", signal),
    // The right-rail history belongs to the connected mailbox only. Keep the
    // query disabled in Demo even though the inbox has a separate local feed.
    enabled: route === "communications" && dataMode === "live",
    staleTime: 15_000,
    retry: false,
  });
  const communicationsDemoActivity = useQuery({
    queryKey: ["advanced-screen", "communications", "demo-activity"],
    queryFn: () => demoProvider.getActivity("communications"),
    enabled: route === "communications" && dataMode === "demo",
    staleTime: 0,
  });
  const trackedDemoActivity = useQuery({
    queryKey: ["advanced-screen", route, "demo-activity"],
    queryFn: () => demoProvider.getActivity(route),
    enabled: dataMode === "demo" && ["voice", "twin", "security", "developer", "engine", "science", "scraper"].includes(route),
    staleTime: 0,
  });
  const trackedEvents = trackedDemoActivity.data ?? [];
  const renderTrackedEvents = () => trackedDemoActivity.isLoading
    ? <p className="caption" role="status">Loading local activity…</p>
    : trackedDemoActivity.isError
      ? <p className="caption" role="alert">Local activity history is unavailable.</p>
      : trackedEvents.length
        ? trackedEvents.slice(0, 3).map((event, index) => <div className="p-instruction" key={String(event.id ?? index)}><strong>{String(event.name ?? "Demo action")}</strong><p>{String(event.item ?? "")} · {String(event.status ?? "Recorded locally · demo")}</p></div>)
        : <p className="caption">No activity yet.</p>;
  if ((route === "science" || route === "scraper") && dataMode === "demo") {
    const collectionKey = demoWorkspaceCollection[route]?.[tab] ?? "items";
    const records = data?.[collectionKey]?.[collectionKey];
    const count = Array.isArray(records) ? records.length : 0;
    const label = route === "science" ? ({ projects: "projects", experiments: "experiments", datasets: "datasets", reports: "reports" } as Record<string, string>)[collectionKey] : ({ jobs: "jobs", results: "results", rules: "rules" } as Record<string, string>)[collectionKey];
    return <aside className="p-rail advanced-demo-rail">
      <section className="p-rail-section"><div className="p-rail-heading"><h3>Workspace overview</h3><BarChart3 className="icon sm" aria-hidden="true"/></div><div className="demo-metric"><strong>{count}</strong><span>{label ?? "records"}</span></div><div className="demo-metric"><strong>{trackedEvents.length}</strong><span>demo activities</span></div><button className="btn sm" type="button" disabled={demoRailBusy} onClick={() => exportDemoRecords(route, tab, `openagent-${route}-demo.json`)}><FileDown className="icon sm" aria-hidden="true"/>Export demo data</button></section>
      <section className="p-rail-section"><div className="p-rail-heading"><h3>How it works</h3></div><p className="caption">Create, edit, search and run example workflows. Your changes are saved locally.</p><button className="btn sm section-space" type="button" disabled={demoRailBusy} onClick={() => resetDemoRecords(route)}><RotateCcw className="icon sm" aria-hidden="true"/>Reset this area</button></section>
      <section className="p-rail-section"><div className="p-rail-heading"><h3>Latest events</h3></div>{renderTrackedEvents()}</section>
    {dataMode === "demo" && demoRailFeedback}</aside>;
  }
  if (route === "vault" && dataMode === "demo") {
    const collectionKey = tab === "Secrets" ? "secrets" : tab === "Audit" ? "audit" : "requests";
    const records = data?.[collectionKey]?.[collectionKey];
    const count = Array.isArray(records) ? records.length : 0;
    const label = tab === "Access" ? "requests" : tab === "Audit" ? "audit entries" : "secrets";
    const eventCount = vaultActivity.length;
    return <aside className="p-rail advanced-demo-rail">
      <section className="p-rail-section"><div className="p-rail-heading"><h3>Workspace overview</h3><BarChart3 className="icon sm" aria-hidden="true"/></div><div className="demo-metric"><strong>{count}</strong><span>{label}</span></div><div className="demo-metric"><strong>{eventCount}</strong><span>demo activities</span></div><button className="btn sm" type="button" disabled={demoRailBusy} onClick={() => exportDemoRecords("vault", tab, "openagent-vault-demo.json")}><FileDown className="icon sm" aria-hidden="true"/>Export demo data</button></section>
      <section className="p-rail-section"><div className="p-rail-heading"><h3>How it works</h3></div><p className="caption">Create, edit, search and run example workflows. Your changes are saved locally.</p><button className="btn sm section-space" type="button" disabled={demoRailBusy} onClick={() => resetDemoRecords("vault", () => { try { localStorage.removeItem("oa-second-demo-vault-activity"); } catch { /* activity reset is best-effort */ } window.dispatchEvent(new Event("oa-vault-demo-reset")); })}><RotateCcw className="icon sm" aria-hidden="true"/>Reset this area</button></section>
      <section className="p-rail-section"><h3 style={{ marginBottom: 12 }}>Latest events</h3>{vaultActivity.length ? vaultActivity.slice(0,3).map(event => <div className="p-instruction" key={event.id}><strong>{event.name}</strong><p>{event.time} · {event.status}</p></div>) : <p className="caption">No activity yet.</p>}</section>
    {dataMode === "demo" && demoRailFeedback}</aside>;
  }
  if ((route === "twin" || route === "security") && dataMode === "demo") {
    const collectionKey = route === "twin" ? ({ Profile: "preferences", Preferences: "preferences", Sources: "sources", "Data controls": "history" } as Record<string, string>) : ({ Overview: "policies", Access: "policies", Sessions: "sessions", Audit: "events" } as Record<string, string>);
    const key = collectionKey[tab] ?? "policies";
    const collection = route === "twin" ? data?.[key]?.[key] : data?.[key]?.[key];
    const count = Array.isArray(collection) ? collection.length : route === "twin" ? 3 : 0;
    const label = route === "twin" ? (key === "sources" ? "sources" : key === "history" ? "events" : "preferences") : key;
    return <aside className="p-rail advanced-demo-rail">
      <section className="p-rail-section"><div className="p-rail-heading"><h3>Workspace overview</h3><BarChart3 className="icon sm" aria-hidden="true"/></div><div className="demo-metric"><strong>{count}</strong><span>{label}</span></div><div className="demo-metric"><strong>{trackedEvents.length}</strong><span>demo activities</span></div><button className="btn sm" type="button" disabled={demoRailBusy} onClick={() => exportDemoRecords(route, tab || (route === "twin" ? "Profile" : "Overview"), `openagent-${route}-demo.json`)}><FileDown className="icon sm" aria-hidden="true"/>Export demo data</button></section>
      <section className="p-rail-section"><div className="p-rail-heading"><h3>How it works</h3></div><p className="caption">Create, edit, search and run example workflows. Your changes are saved locally.</p><button className="btn sm section-space" type="button" disabled={demoRailBusy} onClick={() => resetDemoRecords(route)}>Reset this area</button></section>
      <section className="p-rail-section"><div className="p-rail-heading"><h3>Latest events</h3><Activity className="icon sm" aria-hidden="true"/></div>{renderTrackedEvents()}</section>
    {dataMode === "demo" && demoRailFeedback}</aside>;
  }
  if (["developer", "engine", "finetune"].includes(route) && dataMode === "demo") {
    const collectionKey: Record<string, Record<string, string>> = {
      developer: { Connections: "connections", "API keys": "api_keys", Webhooks: "webhooks", MCP: "servers", Activity: "deliveries" },
      engine: { Overview: "jobs", Hardware: "hardware", Jobs: "jobs", Plans: "plans" },
      finetune: { Datasets: "datasets", Jobs: "jobs", Checkpoints: "checkpoints", Results: "adapters" },
    };
    const key = collectionKey[route]?.[tab] ?? "items";
    const child = route === "developer" ? (key === "api_keys" ? "api_keys" : key === "servers" ? "servers" : key) : key;
    const collection = data?.[key]?.[child];
    const count = Array.isArray(collection) ? collection.length : 0;
    const label = route === "developer" ? ({ api_keys: "keys", servers: "MCP servers", deliveries: "events", webhooks: "webhooks", connections: "connections" } as Record<string, string>)[key] ?? "records" : route === "engine" ? ({ hardware: "hardware", jobs: "jobs", plans: "plans" } as Record<string, string>)[key] : ({ datasets: "datasets", jobs: "jobs", checkpoints: "checkpoints", adapters: "results" } as Record<string, string>)[key] ?? "records";
    const recentEvents = route === "finetune" && Array.isArray(data?.activity?.events) ? data.activity.events as JsonRecord[] : [];
    return <aside className="p-rail advanced-demo-rail">
      <section className="p-rail-section"><div className="p-rail-heading"><h3>Workspace overview</h3><BarChart3 className="icon sm" aria-hidden="true"/></div><div className="demo-metric"><strong>{count}</strong><span>{label}</span></div><div className="demo-metric"><strong>{route === "finetune" ? recentEvents.length : trackedEvents.length}</strong><span>demo activities</span></div><button className="btn sm" type="button" disabled={demoRailBusy} onClick={() => exportDemoRecords(route, tab, `openagent-${route}-demo.json`)}><FileDown className="icon sm" aria-hidden="true"/>Export demo data</button></section>
      <section className="p-rail-section"><div className="p-rail-heading"><h3>How it works</h3></div><p className="caption">Create, edit, search and run example workflows. Your changes are saved locally.</p><button className="btn sm section-space" type="button" disabled={demoRailBusy} onClick={() => resetDemoRecords(route)}>Reset this area</button></section>
       <section className="p-rail-section"><div className="p-rail-heading"><h3>Latest events</h3><Activity className="icon sm" aria-hidden="true"/></div>{route === "finetune" ? recentEvents.length ? recentEvents.slice(0, 3).map((event, index) => <div className="p-instruction" key={String(event.id ?? index)}><strong>{String(event.action ?? "Demo action")}</strong><p>{String(event.item ?? "Fine-tuning record")} · {String(event.status ?? "Recorded locally")}</p></div>) : <p className="caption">No fine-tuning actions have been recorded in this demo session.</p> : renderTrackedEvents()}</section>
    {dataMode === "demo" && demoRailFeedback}</aside>;
  }
  if (route === "communications") {
    const messages = Array.isArray(data?.messages?.messages) ? data.messages.messages : [];
    const runs = Array.isArray(communicationsActivity.data?.runs) ? communicationsActivity.data.runs : [];
    const demoEvents = communicationsDemoActivity.data ?? [];
    if (dataMode === "demo") return <aside className="p-rail communications-rail">
      <section className="p-rail-section">
        <div className="p-rail-heading"><h3>Workspace overview</h3><Activity className="icon sm" aria-hidden="true" /></div>
        <div className="demo-metric"><strong>{messages.length}</strong><span>messages</span></div>
        <div className="demo-metric"><strong>{demoEvents.length}</strong><span>demo activities</span></div>
        <button className="btn sm" type="button" disabled={demoRailBusy} onClick={() => exportDemoRecords(route, tab, "openagent-communications-demo.json", { messages })}><FileDown className="icon sm" aria-hidden="true" />Export demo data</button>
      </section>
      <section className="p-rail-section"><div className="p-rail-heading"><h3>How it works</h3></div><p className="caption">Create, edit, search and run example workflows. Your changes are saved locally.</p>
        <button className="btn sm section-space" type="button" disabled={demoRailBusy} onClick={() => resetDemoRecords("communications")}>Reset this area</button></section>
      <section className="p-rail-section"><h3 style={{ marginBottom: 12 }}>Latest events</h3>{communicationsDemoActivity.isLoading ? <p className="caption" role="status">Loading local activity…</p> : demoEvents.length ? demoEvents.slice(0, 3).map((event, index) => <p className="caption communications-rail-event" key={String(event.id ?? index)}>{String(event.name ?? "Demo action")} — {String(event.status ?? "Recorded locally · demo")}</p>) : <p className="caption">No communication activity has been recorded in this demo session.</p>}</section>
      {demoRailFeedback}
    </aside>;
    return <aside className="p-rail communications-rail">
      <section className="p-rail-section">
        <div className="p-rail-heading"><h3>Workspace overview</h3><Activity className="icon sm" aria-hidden="true" /></div>
        <div className="demo-metric"><strong>{messages.length}</strong><span>messages</span></div>
        <div className="demo-metric"><strong>{runs.length}</strong><span>recorded syncs</span></div>
        <button className="btn sm" type="button" disabled title="Export is unavailable until a supported communications export operation is provided"><FileDown className="icon sm" aria-hidden="true" />Export unavailable</button>
      </section>
      <section className="p-rail-section">
        <div className="p-rail-heading"><h3>How it works</h3></div>
        <p className="caption">Review connected messages, prepare drafts and request approval before any external send.</p>
        <p className="caption section-space">Mailbox sync and message delivery follow backend permissions.</p>
      </section>
      <section className="p-rail-section">
        <div className="p-rail-heading"><h3>Latest events</h3><Activity className="icon sm" aria-hidden="true" /></div>
        {communicationsActivity.isLoading ? <p className="caption" role="status">Loading recorded events…</p> : communicationsActivity.isError ? <p className="caption" role="alert">Event history unavailable.</p> : runs.length ? runs.slice(0, 3).map((run, index) => <p className="caption communications-rail-event" key={String(run.id ?? index)}>{String(run.platform ?? "Mailbox sync")} sync — {String(run.status ?? "status unavailable")}</p>) : <p className="caption">No communication activity has been recorded.</p>}
      </section>
    </aside>;
  }
  if (route === "assistant" && dataMode === "demo") {
    const tasks = Array.isArray(data?.tasks?.tasks) ? data.tasks.tasks : [];
    const goals = Array.isArray(data?.goals?.goals) ? data.goals.goals : [];
    const activities = Array.isArray(data?.activity?.events) ? data.activity.events as JsonRecord[] : [];
    const workspaceCount = tab === "Goals" ? goals.length : tasks.length;
    const workspaceLabel = tab === "Goals" ? "goals" : "tasks";
    return <aside className="p-rail assistant-demo-rail">
      <section className="p-rail-section"><div className="p-rail-heading"><h3>Workspace overview</h3><BarChart3 className="icon sm" aria-hidden="true" /></div><div className="assistant-demo-metric"><strong>{workspaceCount}</strong><span>{workspaceLabel}</span></div><div className="assistant-demo-metric"><strong>{activities.length}</strong><span>demo activities</span></div><button className="btn sm" type="button" disabled={demoRailBusy} onClick={() => exportDemoRecords(route, tab, "openagent-assistant-demo.json", { tasks, goals })}><FileDown className="icon sm" aria-hidden="true" />Export demo data</button></section>
      <section className="p-rail-section"><div className="p-rail-heading"><h3>How it works</h3></div><p className="caption">Create, edit, search and run example workflows. Your changes are saved locally.</p><button className="btn sm section-space" type="button" disabled={demoRailBusy} onClick={() => resetDemoRecords("assistant")}>Reset this area</button></section>
      <section className="p-rail-section"><div className="p-rail-heading"><h3>Latest events</h3></div>{activities.length ? activities.slice(0, 3).map((event, index) => <div className="p-instruction" key={String(event.id ?? index)}><strong>{String(event.name ?? event.action ?? "Demo action")}</strong><p>{String(event.time ?? "Recorded locally")} · {String(event.status ?? "Saved locally · demo")}</p></div>) : <p className="caption">No activity yet.</p>}</section>
    {dataMode === "demo" && demoRailFeedback}</aside>;
  }
  if (route === "voice") {
    const sessions = Array.isArray(data?.sessions?.sessions) ? data.sessions.sessions.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
    const transcripts = Array.isArray(data?.transcripts?.transcripts) ? data.transcripts.transcripts : [];
    if (dataMode === "demo") return <aside className="p-rail voice-rail">
      <section className="p-rail-section"><div className="p-rail-heading"><h3>Workspace overview</h3><BarChart3 className="icon sm" aria-hidden="true" /></div>
        <div className="demo-metric"><strong>{sessions.length}</strong><span>recordings</span></div><div className="demo-metric"><strong>{trackedEvents.length}</strong><span>demo activities</span></div>
        <button className="btn sm" type="button" disabled={demoRailBusy} onClick={() => exportDemoRecords(route, tab, "openagent-voice-demo.json", { recordings: sessions, transcripts })}><FileDown className="icon sm" aria-hidden="true" />Export demo data</button>
      </section>
      <section className="p-rail-section"><div className="p-rail-heading"><h3>How it works</h3></div><p className="caption">Create, edit, search and run example workflows. Your changes are saved locally.</p>
        <button className="btn sm section-space" type="button" disabled={demoRailBusy} onClick={() => resetDemoRecords("voice")}>Reset this area</button></section>
      <section className="p-rail-section"><div className="p-rail-heading"><h3>Latest events</h3><Activity className="icon sm" aria-hidden="true" /></div>{renderTrackedEvents()}</section>
      {demoRailFeedback}
    </aside>;
    return <aside className="p-rail voice-rail">
      <section className="p-rail-section">
        <div className="p-rail-heading"><h3>Workspace overview</h3><Activity className="icon sm" aria-hidden="true" /></div>
        <div className="demo-metric"><strong>{sessions.length}</strong><span>{sessions.length === 1 ? "session" : "sessions"}</span></div>
        <div className="demo-metric"><strong>{transcripts.length}</strong><span>{transcripts.length === 1 ? "transcript" : "transcripts"}</span></div>
        <button className="btn sm" type="button" disabled title="A voice export operation is not provided by the backend"><FileDown className="icon sm" aria-hidden="true" />Export unavailable</button>
      </section>
      <section className="p-rail-section">
        <div className="p-rail-heading"><h3>How it works</h3></div>
        <p className="caption">Create a session, then upload audio to transcribe it. Voice services and microphone access are controlled by the backend.</p>
        <button className="btn sm section-space" type="button" onClick={() => onTabChange?.("Transcripts")}>Review transcripts</button>
      </section>
      <section className="p-rail-section">
        <div className="p-rail-heading"><h3>Latest events</h3><Activity className="icon sm" aria-hidden="true" /></div>
        {connectionLabel === "Connection error" ? <p className="caption" role="alert">Voice events are unavailable.</p> : connectionLabel === "Connecting…" ? <p className="caption" role="status">Loading voice sessions…</p> : sessions.length ? sessions.slice(0, 3).map((session, index) => <p className="caption voice-rail-event" key={String(session.id ?? index)}>{String(session.mode ?? "Voice session")} · {String(session.status ?? "status unavailable")}</p>) : <p className="caption">No activity yet.</p>}
      </section>
    </aside>;
  }
  if (route === "vault") return <aside className="p-rail">
    <section className="p-rail-section">
      <div className="p-rail-heading"><h3>Private by design</h3><LockKeyhole className="icon sm" aria-hidden="true" /></div>
      <article className="p-item"><div className="square-icon"><LockKeyhole className="icon sm" aria-hidden="true" /></div><div className="p-item-copy"><strong>Mask by default</strong><small>Never expose secret values in page source</small></div></article>
      <article className="p-item"><div className="square-icon"><FileClock className="icon sm" aria-hidden="true" /></div><div className="p-item-copy"><strong>Time-limited access</strong><small>Reveal only after verified authorization</small></div></article>
      <article className="p-item"><div className="square-icon"><Shield className="icon sm" aria-hidden="true" /></div><div className="p-item-copy"><strong>Trusted audit logs</strong><small>Backend records protected secret operations</small></div></article>
      <div className="p-note section-space"><AlertCircle className="icon sm" aria-hidden="true" /><span>{definition.note}</span></div>
    </section>
    <section className="p-rail-section"><div className="p-rail-heading"><h3>Connection</h3><CloudOff className="icon sm" aria-hidden="true" /></div><div className="row"><div className="square-icon"><Icon className="icon" aria-hidden="true" /></div><div className="flex1"><strong className="small">Vault</strong><div className="caption">{connectionLabel}</div></div></div></section>
    <div className="p-rail-foot"><CircleHelp className="icon sm" aria-hidden="true" /><span>Protected data remains hidden until permission is confirmed.</span></div>
  {dataMode === "demo" && demoRailFeedback}</aside>;
  if (route === "knowledge") return <aside className="p-rail">
    <section className="p-rail-section">
      <div className="p-rail-heading"><h3>Workspace overview</h3><BookOpen className="icon sm" aria-hidden="true" /></div>
      <div className="demo-metric"><strong>—</strong><span>indexed sources · count unavailable</span></div>
      <div className="demo-metric"><strong>—</strong><span>verified activity</span></div>
      <button className="btn sm" type="button" disabled title="Export requires connected indexed sources"><FileDown className="icon sm" />Export unavailable</button>
    </section>
    <section className="p-rail-section">
      <div className="p-rail-heading"><h3>How it works</h3></div>
      <p className="caption">Connect a knowledge source to index documents and search with citations.</p>
      <button className="btn sm section-space" type="button" data-nav="settings">Open settings</button>
    </section>
    <section className="p-rail-section"><div className="p-rail-heading"><h3>Latest events</h3><Activity className="icon sm" aria-hidden="true" /></div><p className="caption">No verified activity loaded.</p></section>
  {dataMode === "demo" && demoRailFeedback}</aside>;
  return <aside className="p-rail">
    <section className="p-rail-section">
      <div className="p-rail-heading"><h3>Connection</h3><CloudOff className="icon sm" aria-hidden="true" /></div>
      <div className="row"><div className="square-icon"><Icon className="icon" aria-hidden="true" /></div><div className="flex1"><strong className="small">{definition.title}</strong><div className="caption">{connectionLabel}</div></div></div>
      <div className="p-note section-space"><AlertCircle className="icon sm" aria-hidden="true" /><span>{definition.note}</span></div>
    </section>
    <section className="p-rail-section"><div className="p-rail-heading"><h3>Workspace</h3><Settings2 className="icon sm" aria-hidden="true" /></div><div className="p-prefs"><div><Layers3 className="icon sm" aria-hidden="true" /><strong>{definition.subject}</strong></div><div><CheckCircle2 className="icon sm" aria-hidden="true" /><strong>Actions require a verified connection</strong></div></div></section>
    <div className="p-rail-foot"><CircleHelp className="icon sm" aria-hidden="true" /><span>{route === "vault" || route === "security" ? "Protected data remains hidden until permission is confirmed." : "This frontend does not simulate service activity."}</span></div>
  {dataMode === "demo" && demoRailFeedback}</aside>;
}

function CommunicationsInbox({ data, query, onTabChange, demo = false }: { data?: Record<string, JsonRecord>; query: { isLoading: boolean; error: unknown }; onTabChange: (tab: string) => void; demo?: boolean }) {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const activityQuery = useQuery({
    queryKey: ["communications", "sync-runs", "default"],
    queryFn: ({ signal }) => apiGet<{ runs?: JsonRecord[] }>("/api/communication/sync-runs?project=default&limit=20", signal),
    // Inbox sync history is Live-only; do not run it through the Demo adapter.
    enabled: !demo,
    staleTime: 15_000,
    refetchInterval: 30_000,
    retry: false,
  });
  const response = data?.messages;
  const messages = Array.isArray(response?.messages)
    ? response.messages.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item))
    : [];
  const filteredMessages = messages.filter((message) => `${message.subject ?? ""} ${message.sender ?? ""} ${message.summary ?? ""} ${message.body ?? ""}`.toLowerCase().includes(search.trim().toLowerCase()));
  const liveInboxMessages = filteredMessages.filter(message => !["archive", "trash", "spam"].includes(String(message.folder ?? "inbox").toLowerCase()) && message.archived !== true && Number(message.archived ?? 0) !== 1);
  const demoInbox = filteredMessages.filter((message) => String(message.status ?? "Inbox").toLowerCase() === "inbox");
  const [selectedId, setSelectedId] = useState("");
  const [editingMessageId, setEditingMessageId] = useState("");
  const [trashUndo, setTrashUndo] = useState<{ id: string; subject: string; folder: string } | null>(null);
  const [editMessageOpen, setEditMessageOpen] = useState(false);
  const [editMessageSubject, setEditMessageSubject] = useState("");
  const [editMessageSender, setEditMessageSender] = useState("");
  const [editMessageBody, setEditMessageBody] = useState("");
  const selected = liveInboxMessages.find((item) => String(item.id ?? "") === selectedId) ?? liveInboxMessages[0];
  const syncRuns = Array.isArray(activityQuery.data?.runs) ? activityQuery.data.runs.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const stateMutation = useMutation({
    mutationFn: (body: JsonRecord) => {
      const { _moveToTrash, _restoreFromTrash, _subject, _previousFolder, ...payload } = body;
      void _moveToTrash;
      void _restoreFromTrash;
      void _subject;
      void _previousFolder;
      return apiPost("/api/communication/messages/state", payload);
    },
    onSuccess: async (_result, body) => {
      if (body._moveToTrash === true) {
        setTrashUndo({ id: String(body.message_id ?? ""), subject: String(body._subject ?? "Message"), folder: String(body._previousFolder ?? "inbox") });
        setInboxFeedback("Message moved to trash.");
      } else if (body._restoreFromTrash === true) {
        setTrashUndo(null);
        setInboxFeedback("Message restored to the inbox.");
      } else {
        setInboxFeedback(body.archived === true ? "Message archived by the backend." : "Message state updated by the backend.");
      }
      await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications", "Inbox"] });
    },
    onError: () => {
      setInboxFeedback("Mailbox state could not be updated.");
    },
  });
  const [inboxFeedback, setInboxFeedback] = useState("");
  const replyMutation = useMutation({
    mutationFn: () => apiPost<{ draft?: JsonRecord }>("/api/communication/drafts/reply", {
      message_id: String(selected?.id ?? ""), project: "default",
      instructions: "Prepare a concise reply draft for review. Do not send it.",
    }),
    onSuccess: async result => {
      setInboxFeedback(`Reply draft ${String(result.draft?.id ?? "created")} saved by the backend. It was not sent.`);
      await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications"] });
    },
  });
  const copyDraftMutation = useMutation({
    mutationFn: () => apiPost<{ draft?: JsonRecord }>("/api/communication/drafts", {
      platform: String(selected?.platform ?? "email"),
      recipients: Array.isArray(selected?.recipients) ? selected.recipients.map(String) : [],
      subject: String(selected?.subject ?? ""),
      body: String(selected?.body ?? selected?.summary ?? ""),
      message_id: String(selected?.id ?? ""),
      project: "default",
    }),
    onSuccess: async result => {
      setInboxFeedback(`Draft copy ${String(result.draft?.id ?? "created")} saved by the backend. It was not sent.`);
      await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications"] });
    },
  });
  const editMessageMutation = useMutation({
    mutationFn: (body: JsonRecord) => apiPost<{ message?: JsonRecord }>("/api/communication/messages/update", body),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications", "Inbox"] });
      setEditMessageOpen(false);
      setInboxFeedback("Message changes saved by the backend.");
    },
  });
  useEffect(() => {
    if (!liveInboxMessages.some((item) => String(item.id ?? "") === selectedId)) setSelectedId(String(liveInboxMessages[0]?.id ?? ""));
  }, [liveInboxMessages, selectedId]);

  if (demo) return <DemoCommunicationsInbox messages={demoInbox} search={search} onSearch={setSearch} onTabChange={onTabChange} />;

  return <>
    <section className="p-card">
      <div className="p-card-heading"><h3>Inbox · mailbox</h3><span className="p-fixture">{messages.length} {messages.length === 1 ? "message" : "messages"}</span></div>
      <div className="communications-inbox-actions">
        <button className="btn primary sm" type="button" onClick={() => onTabChange("Drafts")}><Plus className="icon sm" aria-hidden="true" />Compose message</button>
        <span className="flex1" />
        <button className="btn sm" type="button" disabled={query.isLoading} onClick={() => queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications", "Inbox"] })} title="Refresh messages from the backend"><Activity className="icon sm" aria-hidden="true" />Refresh</button>
      </div>
      {inboxFeedback && <p className="caption communications-inbox-feedback" role="status">{inboxFeedback}{stateMutation.isError && <span className="communications-inbox-state-error" role="alert">{apiConnectionMessage(stateMutation.error)}</span>}{trashUndo && <button className="btn sm" type="button" disabled={stateMutation.isPending} onClick={() => { stateMutation.reset(); stateMutation.mutate({ message_id: trashUndo.id, folder: trashUndo.folder, project: "default", _restoreFromTrash: true }); }}>Undo</button>}</p>}
      <label className="communications-inbox-search"><SearchIcon className="icon sm" aria-hidden="true" /><input aria-label="Search inbox messages" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search messages…" /><span className="caption">{filteredMessages.length} shown</span></label>
      <div className="p-mail-shell p-mail-live">
        <div className="p-mail-list" aria-label="Inbox messages">
          {query.isLoading && !messages.length ? <p className="communications-inbox-state" role="status">Loading messages from the backend…</p> : query.error && !messages.length ? <p className="communications-inbox-state error" role="alert">Messages are unavailable: {apiConnectionMessage(query.error)}</p> : liveInboxMessages.length ? liveInboxMessages.map((message, index) => {
            const id = String(message.id ?? index);
            const active = String(selected?.id ?? "") === id;
            return <button className={`p-item p-mail-item${active ? " active" : ""}`} type="button" key={id} aria-pressed={active} onClick={() => setSelectedId(id)}>
              <span className={`p-mail-unread${message.read ? " read" : ""}`} aria-label={message.read ? "Read" : "Unread"} />
              <span className="p-item-copy"><strong>{String(message.subject ?? "(No subject)")}</strong><small>{String(message.sender ?? "Unknown sender")} · {String(message.summary ?? "")}</small></span>
              <span className="p-item-meta">{String(message.created_at ?? "")}</span>
            </button>;
          }) : filteredMessages.length || messages.length ? <p className="communications-inbox-state" role="status">{search ? `No messages match “${search}”.` : "The inbox has no active messages."}</p> : <EmptyState icon={Mail} title="Inbox is empty" detail="No messages were returned by the backend for this project." />}
        </div>
        <article className="p-mail-reader" aria-live="polite">
          {selected ? <>
            <div className="p-mail-top"><div className="row between"><div><h2>{String(selected.subject ?? "(No subject)")}</h2><p className="caption">From {String(selected.sender ?? "Unknown sender")} · {String(selected.created_at ?? "")}</p></div><div className="row"><button className="btn sm" type="button" disabled={stateMutation.isPending || Boolean(selected.read)} onClick={() => stateMutation.mutate({ message_id: String(selected.id ?? ""), read: true, project: "default" })}>{selected.read ? "Read" : "Mark read"}</button><button className="btn sm" type="button" disabled={replyMutation.isPending || !selected.id} onClick={() => { setInboxFeedback(""); replyMutation.mutate(); }}>{replyMutation.isPending ? "Preparing…" : "Draft reply"}</button><button className="btn sm" type="button" disabled={copyDraftMutation.isPending || !selected.id} onClick={() => { setInboxFeedback(""); copyDraftMutation.mutate(); }}>{copyDraftMutation.isPending ? "Saving copy…" : "Copy to drafts"}</button><button className="btn sm" type="button" disabled={!selected.id || editMessageMutation.isPending} onClick={() => { setEditingMessageId(String(selected.id ?? "")); setEditMessageSubject(String(selected.subject ?? "")); setEditMessageSender(String(selected.sender ?? "")); setEditMessageBody(String(selected.body ?? selected.summary ?? "")); editMessageMutation.reset(); setEditMessageOpen(true); }}>Edit message</button><button className="btn sm" type="button" disabled={stateMutation.isPending || !selected.id} onClick={() => { setInboxFeedback(""); stateMutation.mutate({ message_id: String(selected.id ?? ""), archived: true, project: "default" }); }}>Archive</button><button className="btn sm danger" type="button" disabled={stateMutation.isPending || !selected.id} onClick={() => { if (window.confirm(`Move “${String(selected.subject ?? "this message")}" to trash?`)) { setInboxFeedback(""); stateMutation.mutate({ message_id: String(selected.id ?? ""), folder: "trash", project: "default", _moveToTrash: true, _subject: String(selected.subject ?? "Message"), _previousFolder: String(selected.folder ?? "inbox") }); } }}>Move to trash</button></div></div></div>
            <div className="p-mail-body">{String(selected.body ?? selected.summary ?? "No message body was provided.")}</div>
            {(stateMutation.isError || replyMutation.isError || copyDraftMutation.isError || editMessageMutation.isError) && <p className="caption" role="alert">{apiConnectionMessage(stateMutation.error ?? replyMutation.error ?? copyDraftMutation.error ?? editMessageMutation.error)}</p>}
          </> : <EmptyState icon={Mail} title="Select a message" detail="Choose a message to read its backend-provided content." />}
        </article>
      </div>
    </section>
    <section className="p-card communications-activity-card" aria-label="Recent communication activity">
      <div className="p-card-heading"><h3>Recent activity</h3><span className="p-fixture">{syncRuns.length} recorded syncs</span></div>
      {activityQuery.isLoading ? <p className="caption" role="status">Loading recorded communication activity…</p> : activityQuery.isError ? <p className="caption" role="alert">Activity history unavailable: {apiConnectionMessage(activityQuery.error)}</p> : syncRuns.length ? <div className="p-record-sections">{syncRuns.slice(0, 4).map((run, index) => <article className="p-record-section" key={String(run.id ?? index)}><strong>{String(run.platform ?? "Mailbox sync")} · {String(run.status ?? "status not reported")}</strong><p className="caption">{String(run.started_at ?? run.created_at ?? run.finished_at ?? "Time not reported")}{typeof run.message === "string" ? ` · ${run.message}` : ""}</p></article>)}</div> : <p className="caption">No communication sync activity has been recorded by the backend.</p>}
    </section>
    <Dialog.Root open={editMessageOpen} onOpenChange={open => { if (!open && !editMessageMutation.isPending) setEditMessageOpen(false); }}>
      <Dialog.Portal>
        <Dialog.Overlay />
        <Dialog.Content className="advanced-integration-dialog" aria-describedby="communication-edit-description">
          <div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">Edit inbox message</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close edit message"><X size={16} aria-hidden="true" /></button></Dialog.Close></div>
          <Dialog.Description id="communication-edit-description" className="second-app-dialog-description">Save changes to this message in the selected local mailbox.</Dialog.Description>
          <form onSubmit={event => { event.preventDefault(); editMessageMutation.mutate({ message_id: editingMessageId, project: "default", subject: editMessageSubject.trim(), sender: editMessageSender.trim(), body: editMessageBody.trim() }); }}>
            <label className="demo-form-field">Subject<input className="input" required maxLength={500} value={editMessageSubject} onChange={event => setEditMessageSubject(event.target.value)} /></label>
            <label className="demo-form-field">Sender<input className="input" required maxLength={320} value={editMessageSender} onChange={event => setEditMessageSender(event.target.value)} /></label>
            <label className="demo-form-field">Message<textarea className="textarea" required rows={6} maxLength={100000} value={editMessageBody} onChange={event => setEditMessageBody(event.target.value)} /></label>
            {editMessageMutation.isError && <p className="caption" role="alert">{apiConnectionMessage(editMessageMutation.error)}</p>}
            <div className="advanced-integration-actions"><Dialog.Close asChild><button className="btn" type="button" disabled={editMessageMutation.isPending}>Cancel</button></Dialog.Close><button className="btn primary" type="submit" disabled={editMessageMutation.isPending || !editMessageSubject.trim() || !editMessageSender.trim() || !editMessageBody.trim()}>{editMessageMutation.isPending ? "Saving…" : "Save changes"}</button></div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  </>;
}

function DemoCommunicationsInbox({ messages, search, onSearch, onTabChange }: { messages: JsonRecord[]; search: string; onSearch: (value: string) => void; onTabChange: (tab: string) => void }) {
  const [showCompose, setShowCompose] = useState(false);
  const [composeRecipient, setComposeRecipient] = useState("");
  const [composeSubject, setComposeSubject] = useState("");
  const [composeBody, setComposeBody] = useState("");
  const [feedback, setFeedback] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [dialogMessage, setDialogMessage] = useState<JsonRecord | null>(null);
  const [dialogMode, setDialogMode] = useState<"details" | "actions" | "edit" | "delete" | "run" | null>(null);
  const [editSubject, setEditSubject] = useState("");
  const [editSender, setEditSender] = useState("");
  const [editBody, setEditBody] = useState("");
  const [actionError, setActionError] = useState("");
  const [actionBusy, setActionBusy] = useState(false);
  const messageDialogTrigger = useRef<HTMLElement | null>(null);
  const queryClient = useQueryClient();
  const activityQuery = useQuery({ queryKey: ["advanced-screen", "communications", "demo-activity"], queryFn: () => demoProvider.getActivity("communications"), staleTime: 0 });
  const filtered = messages.filter((message) => `${message.subject ?? ""} ${message.sender ?? ""} ${message.body ?? ""}`.toLowerCase().includes(search.trim().toLowerCase()));
  const recordLocalActivity = async (name: string, status: string) => {
    const eventId = globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    await demoProvider.recordActivity("communications", { id: `communications-activity-${eventId}`, name, route: "communications", status, time: new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }), recorded_at: new Date().toISOString() });
    await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications", "demo-activity"] });
  };
  const openMessageDialog = (message: JsonRecord, mode: NonNullable<typeof dialogMode>, trigger?: HTMLElement) => {
    if (trigger) messageDialogTrigger.current = trigger;
    setDialogMessage(message); setDialogMode(mode); setActionError("");
    setEditSubject(String(message.subject ?? "")); setEditSender(String(message.sender ?? "")); setEditBody(String(message.body ?? ""));
  };
  const openCompose = (message?: JsonRecord) => {
    setFeedback("");
    setComposeRecipient(message ? (String(message.sender ?? "").match(/[\w.+-]+@[\w.-]+\.\w+/)?.[0] ?? "team@demo.example") : "");
    setComposeSubject(message ? `Re: ${String(message.subject ?? "")}` : "");
    setComposeBody(""); setShowCompose(true);
  };
  const updateMessage = async (update: (rows: DemoRecord[]) => DemoRecord[]) => {
    await demoProvider.updateCollection<DemoRecord>("collections.advanced.communications.messages", update);
    await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications"] });
  };
  const applyMessageAction = async (action: "archive" | "duplicate" | "delete" | "run") => {
    if (!dialogMessage) return;
    const id = String(dialogMessage.id ?? "");
    if (!id) { setActionError("This example message no longer has a record ID."); return; }
    setActionBusy(true); setActionError("");
    try {
      const rows = await demoProvider.get("collections.advanced.communications");
      const stored = Array.isArray(rows.messages) ? rows.messages.filter((row): row is DemoRecord => Boolean(row) && typeof row === "object") : [];
      const source = stored.find(row => String(row.id ?? "") === id);
      if (!source) throw new Error("This example message is no longer available.");
      if (action === "archive") {
        await updateMessage(current => current.map(row => String(row.id ?? "") === id ? { ...row, status: "Archived" } : row));
        await recordLocalActivity(`Archived ${String(dialogMessage.subject ?? "message")}`, "Archived locally · demo");
        setFeedback("Message archived locally in Demo. No mailbox was contacted."); setDialogMode(null); setDialogMessage(null);
      } else if (action === "duplicate") {
        const duplicateId = `demo-message-${globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`}`;
        const duplicate = { ...source, id: duplicateId, name: `${String(source.name ?? "Message")} (copy)`, status: "Draft", time: "Just now" };
        await updateMessage(current => [duplicate, ...current]);
        await recordLocalActivity(`Duplicated ${String(dialogMessage.subject ?? "message")}`, "Saved locally · demo");
        setFeedback("A local draft copy was created. No message was sent."); setDialogMode(null); setDialogMessage(null);
      } else if (action === "delete") {
        await updateMessage(current => current.filter(row => String(row.id ?? "") !== id));
        await recordLocalActivity(`Deleted ${String(dialogMessage.subject ?? "message")}`, "Deleted locally · demo");
        setFeedback("Example message deleted from local Demo data."); setDialogMode(null); setDialogMessage(null);
      } else {
        await recordLocalActivity(`Tested ${String(dialogMessage.subject ?? "message")}`, "Simulated locally · no message sent");
        setFeedback("Demo action recorded locally. No email was sent and no connector was contacted."); setDialogMode(null); setDialogMessage(null);
      }
    } catch (error) { setActionError(error instanceof Error ? error.message : "The local message action could not be completed."); }
    finally { setActionBusy(false); }
  };
  const saveMessageEdit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!dialogMessage || actionBusy) return;
    const id = String(dialogMessage.id ?? "");
    const subject = editSubject.trim(); const sender = editSender.trim(); const body = editBody.trim();
    if (!subject || !sender || !body) { setActionError("Complete the subject, sender and message fields."); return; }
    setActionBusy(true); setActionError("");
    try {
      await updateMessage(current => current.map(row => String(row.id ?? "") === id ? { ...row, name: subject, owner: sender, description: body, time: "Just now" } : row));
      await recordLocalActivity(`Edited ${subject}`, "Updated locally · demo");
      setFeedback("Message updated in local Demo data."); setDialogMode(null); setDialogMessage(null);
    } catch (error) { setActionError(error instanceof Error ? error.message : "The local message could not be updated."); }
    finally { setActionBusy(false); }
  };
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (submitting) return;
    const recipient = composeRecipient.trim();
    const subject = composeSubject.trim();
    const body = composeBody.trim();
    if (!recipient || !subject || !body) { setFeedback("Complete the recipient, subject and message fields."); return; }
    setSubmitting(true);
    try {
      const id = `demo-message-${globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`}`;
      await updateMessage(current => [{ id, name: subject, owner: recipient, description: body, status: "Draft", time: "Just now" }, ...current]);
      await recordLocalActivity(`Saved draft ${subject}`, "Saved locally · demo");
      await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications"] });
      setShowCompose(false);
      setFeedback("Draft saved locally in demo mode. No message was sent.");
    } catch { setFeedback("The example draft could not be saved. Please try again."); }
    finally { setSubmitting(false); }
  };
  return <>
    <section className="p-card communications-demo-inbox">
      <div className="p-card-heading"><h3>Inbox · example mailbox</h3></div>
      <div className="communications-inbox-actions">
        <button className="btn primary sm" type="button" onClick={() => openCompose()}><Plus className="icon sm" aria-hidden="true" />Compose message</button>
        <span className="flex1" />
        <button className="btn sm" type="button" onClick={() => setFeedback("Example inbox is up to date.")}><RotateCcw className="icon sm" aria-hidden="true" />Sync demo inbox</button>
      </div>
      <div className="communications-demo-toolbar">
        <label className="communications-inbox-search"><SearchIcon className="icon sm" aria-hidden="true" /><input aria-label="Search inbox messages" value={search} onChange={(event) => onSearch(event.target.value)} placeholder="Search messages…" /></label>
        <span className="caption">{filtered.length} {filtered.length === 1 ? "item" : "items"}</span>
        <button className="btn primary sm" type="button" onClick={() => openCompose()}><Plus className="icon sm" aria-hidden="true" />Add</button>
      </div>
      <div className="communications-demo-list" aria-label="Example inbox messages">
        {filtered.map((message, index) => <article className="communications-demo-row" key={String(message.id ?? index)}>
          <span className="communications-demo-icon"><Mail className="icon sm" aria-hidden="true" /></span>
          <div className="flex1 communications-demo-copy"><strong>{String(message.subject ?? "(No subject)")}</strong><small>{String(message.body ?? message.summary ?? "")}</small></div>
          <span className="p-fixture communications-demo-status">Inbox</span>
          <button className="icon-button communications-demo-row-action" type="button" aria-label={`Message options for ${String(message.subject ?? "message")}`} title="Message options" onClick={event => openMessageDialog(message, "details", event.currentTarget)}><Settings2 className="icon sm" aria-hidden="true" /></button>
          <button className="icon-button communications-demo-row-action" type="button" aria-label={`More options for ${String(message.subject ?? "message")}`} title="More options" onClick={event => openMessageDialog(message, "actions", event.currentTarget)}><span aria-hidden="true">···</span></button>
        </article>)}
        {!filtered.length && <p className="communications-inbox-state" role="status">{messages.length ? `No messages match “${search}”.` : "No example inbox messages are available."}</p>}
      </div>
      {feedback && <p className="caption communications-demo-feedback" role="status">{feedback}</p>}
    </section>
    <section className="p-card communications-activity-card" aria-label="Recent communication activity">
      <div className="p-card-heading"><h3>Recent activity</h3><button className="btn sm" type="button" onClick={() => onTabChange("Connected accounts")}>All activity</button></div>
      {activityQuery.isLoading ? <p className="caption" role="status">Loading local communication activity…</p> : activityQuery.isError ? <p className="caption" role="alert">Local activity is unavailable.</p> : activityQuery.data?.length ? <div className="p-record-sections">{activityQuery.data.slice(0, 5).map((event, index) => <article className="p-record-section" key={String(event.id ?? index)}><strong>{String(event.name ?? "Communication action")}</strong><p className="caption">{String(event.status ?? "Recorded locally")} · {String(event.time ?? "Time not recorded")}</p></article>)}</div> : <p className="caption">No communication activity has been recorded in this demo session.</p>}
    </section>
    <Dialog.Root open={Boolean(dialogMode)} onOpenChange={open => { if (!open && !actionBusy) { setDialogMode(null); setDialogMessage(null); setActionError(""); window.requestAnimationFrame(() => messageDialogTrigger.current?.focus()); } }}><Dialog.Portal><Dialog.Overlay/><Dialog.Content className="advanced-integration-dialog communications-message-dialog"><div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">{dialogMode === "actions" ? String(dialogMessage?.subject ?? "Message actions") : dialogMode === "details" ? String(dialogMessage?.subject ?? "Message details") : dialogMode === "edit" ? "Edit example message" : dialogMode === "delete" ? `Delete “${String(dialogMessage?.subject ?? "example message")}”?` : dialogMode === "run" ? "Run example action?" : "Message details"}</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close message dialog" disabled={actionBusy}><X size={16} aria-hidden="true"/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">{dialogMode === "actions" ? "Choose an action for this local example record." : dialogMode === "edit" ? "Changes are saved only in isolated Demo data." : dialogMode === "delete" ? "This deletes the local demo record. It does not affect your backend." : "This is an example message. No external account is contacted and no email is sent."}</Dialog.Description>
      {dialogMode === "actions" && <div className="communications-message-actions"><button className="btn" type="button" onClick={() => setDialogMode("details")}>Open details</button><button className="btn" type="button" onClick={() => setDialogMode("edit")}>Edit record</button><button className="btn" type="button" onClick={() => void applyMessageAction("duplicate")} disabled={actionBusy}>Duplicate</button><button className="btn" type="button" onClick={() => setDialogMode("run")}>Run / test</button><button className="btn danger" type="button" onClick={() => setDialogMode("delete")}>Delete</button></div>}
      {dialogMode === "details" && dialogMessage && <><div className="p-note"><Info className="icon sm" aria-hidden="true"/><span>Local Demo record · {String(dialogMessage.status ?? "Status unavailable")}</span></div><div className="p-record-sections"><article className="p-record-section"><strong>From</strong><p className="caption">{String(dialogMessage.sender ?? "Sender unavailable")}</p></article><article className="p-record-section"><strong>Message</strong><p className="caption">{String(dialogMessage.body ?? "No message body was provided.")}</p></article></div><div className="advanced-integration-actions"><button className="btn" type="button" onClick={() => { setDialogMode(null); openCompose(dialogMessage); }}>Reply</button>{String(dialogMessage.status ?? "").toLowerCase() !== "archived" && <button className="btn" type="button" disabled={actionBusy} onClick={() => void applyMessageAction("archive")}>Archive</button>}<button className="btn" type="button" onClick={() => setDialogMode("edit")}>Edit</button><button className="btn danger" type="button" onClick={() => setDialogMode("delete")}>Delete</button></div></>}
      {dialogMode === "edit" && dialogMessage && <form className="communications-message-edit" onSubmit={event => void saveMessageEdit(event)}><label className="demo-form-field">Subject<input className="input" required maxLength={180} value={editSubject} onChange={event => setEditSubject(event.target.value)}/></label><label className="demo-form-field">Sender<input className="input" required value={editSender} onChange={event => setEditSender(event.target.value)}/></label><label className="demo-form-field">Message<textarea className="input" rows={4} required value={editBody} onChange={event => setEditBody(event.target.value)}/></label><div className="row between"><button className="btn" type="button" disabled={actionBusy} onClick={() => setDialogMode("details")}>Cancel</button><button className="btn primary" type="submit" disabled={actionBusy}>{actionBusy ? "Saving…" : "Save changes"}</button></div></form>}
      {dialogMode === "delete" && <div className="advanced-integration-actions"><button className="btn" type="button" disabled={actionBusy} onClick={() => setDialogMode("details")}>Cancel</button><button className="btn danger" type="button" disabled={actionBusy} onClick={() => void applyMessageAction("delete")}>{actionBusy ? "Deleting…" : "Delete record"}</button></div>}
      {dialogMode === "run" && <div className="advanced-integration-actions"><span className="p-fixture">Simulated · Demo only</span><button className="btn" type="button" onClick={() => setDialogMode("actions")}>Cancel</button><button className="btn primary" type="button" disabled={actionBusy} onClick={() => void applyMessageAction("run")}>Run local example</button></div>}
      {actionError && <p className="caption" role="alert">{actionError}</p>}
    </Dialog.Content></Dialog.Portal></Dialog.Root>
    {showCompose && <div className="communications-demo-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !submitting) setShowCompose(false); }}>
      <section className="communications-demo-dialog" role="dialog" aria-modal="true" aria-labelledby="communications-compose-title">
        <div className="row between"><h2 id="communications-compose-title">Compose example message</h2><button className="btn icon-only" type="button" aria-label="Close compose message" disabled={submitting} onClick={() => setShowCompose(false)}>×</button></div>
        <p className="caption">Save a local draft. Demo messages are not delivered.</p>
        <form onSubmit={(event) => void submit(event)}>
          <label className="demo-form-field">To<input className="input" name="recipient" type="email" required placeholder="name@demo.example" value={composeRecipient} onChange={event => setComposeRecipient(event.target.value)} /></label>
          <label className="demo-form-field">Subject<input className="input" name="subject" required placeholder="Message subject" value={composeSubject} onChange={event => setComposeSubject(event.target.value)} /></label>
          <label className="demo-form-field">Message<textarea className="input" name="body" rows={5} required placeholder="Write your message…" value={composeBody} onChange={event => setComposeBody(event.target.value)} /></label>
          {feedback && <p className="caption" role="status">{feedback}</p>}
          <div className="row between"><button className="btn" type="button" disabled={submitting} onClick={() => setShowCompose(false)}>Cancel</button><button className="btn primary" type="submit" disabled={submitting}>{submitting ? "Saving…" : "Save draft"}</button></div>
        </form>
      </section>
    </div>}
  </>;
}

function CommunicationsDraftQueue({ data }: { data?: Record<string, JsonRecord> }) {
  const queryClient = useQueryClient();
  const drafts = Array.isArray(data?.drafts?.drafts)
    ? data.drafts.drafts.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item))
    : [];
  const approvals = Array.isArray(data?.approvals?.approvals)
    ? data.approvals.approvals.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item))
    : [];
  const [selectedDraftId, setSelectedDraftId] = useState("");
  const [selectedApprovalId, setSelectedApprovalId] = useState("");
  const [confirmSend, setConfirmSend] = useState(false);
  const [scheduleAt, setScheduleAt] = useState("");
  const [feedback, setFeedback] = useState("");
  const [recipient, setRecipient] = useState("");
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [draftValidationError, setDraftValidationError] = useState(false);
  const selectedDraft = drafts.find((draft) => String(draft.id ?? "") === selectedDraftId) ?? drafts[0];
  const approval = approvals.find((item) => String(item.id ?? "") === selectedApprovalId)
    ?? approvals.find((item) => String(item.draft_id ?? "") === String(selectedDraft?.id ?? "") && item.status === "pending")
    ?? approvals.find((item) => String(item.draft_id ?? "") === String(selectedDraft?.id ?? "") && item.status === "approved");
  const refresh = () => queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications", "Drafts"] });
  const createDraft = useMutation({
    mutationFn: () => apiPost<{ draft?: JsonRecord }>("/api/communication/drafts", {
      platform: "email",
      recipients: recipient.split(/[;,]/).map(value => value.trim()).filter(Boolean),
      subject: subject.trim(),
      body: body.trim(),
      project: "default",
    }),
    onSuccess: async result => {
      const created = result.draft;
      if (created?.id) setSelectedDraftId(String(created.id));
      setRecipient(""); setSubject(""); setBody(""); setConfirmSend(false);
      setFeedback("Draft saved by the backend. It has not been sent.");
      await refresh();
    },
    onError: error => setFeedback(apiConnectionMessage(error)),
  });
  const request = useMutation({
    mutationFn: () => apiPost("/api/communication/send/request-approval", { draft_id: String(selectedDraft?.id ?? ""), reason: "User requested this draft to be reviewed for sending.", project: "default" }),
    onSuccess: async (result) => {
      const next = result.approval as JsonRecord | undefined;
      setSelectedApprovalId(typeof next?.id === "string" ? next.id : "");
      setConfirmSend(false);
      setFeedback("Approval request recorded by the backend.");
      await refresh();
    },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const resolve = useMutation({
    mutationFn: (approved: boolean) => apiPost("/api/communication/send/resolve", { approval_id: String(approval?.id ?? ""), approved, project: "default" }),
    onSuccess: async (result) => {
      const resolved = result.approval as JsonRecord | undefined;
      setSelectedApprovalId(typeof resolved?.id === "string" ? resolved.id : String(approval?.id ?? ""));
      setConfirmSend(false);
      setFeedback(`Backend approval status: ${String(resolved?.status ?? "response received")}.`);
      await refresh();
    },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const send = useMutation({
    mutationFn: () => apiPost("/api/communication/send", { draft_id: String(selectedDraft?.id ?? ""), approval_id: String(approval?.id ?? ""), project: "default", allow_external: false }),
    onSuccess: async () => {
      setConfirmSend(false);
      setFeedback("The backend accepted the approved send request. Check Sent for its recorded result.");
      await refresh();
      await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications", "Sent"] });
    },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const schedule = useMutation({
    mutationFn: () => apiPost("/api/communication/scheduled", { draft_id: String(selectedDraft?.id ?? ""), approval_id: String(approval?.id ?? ""), send_at: new Date(scheduleAt).toISOString(), project: "default" }),
    onSuccess: async () => {
      setScheduleAt(""); setFeedback("The backend recorded this approved draft as scheduled.");
      await refresh(); await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications", "Scheduled"] });
    },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  useEffect(() => {
    if (!drafts.some((draft) => String(draft.id ?? "") === selectedDraftId)) setSelectedDraftId(String(drafts[0]?.id ?? ""));
  }, [drafts, selectedDraftId]);
  useEffect(() => { setConfirmSend(false); }, [selectedDraftId]);

  const pending = request.isPending || resolve.isPending || send.isPending || schedule.isPending || createDraft.isPending;
  return <section className="p-card">
    <div className="p-card-heading"><h3>Draft review</h3><span className="p-fixture">{drafts.length} saved · {approvals.length} approval records</span></div>
    <form className="p-record-sections" aria-label="Create communication draft" onSubmit={event => {
      event.preventDefault();
      const recipients = recipient.split(/[;,]/).map(value => value.trim()).filter(Boolean);
      if (!recipients.length || recipients.some(value => !/^\S+@\S+\.\S+$/.test(value))) {
        setDraftValidationError(true);
        setFeedback("Enter one or more valid email addresses, separated by commas.");
        return;
      }
      setDraftValidationError(false); setFeedback(""); createDraft.mutate();
    }}>
      <h4>Create a draft</h4>
      <label className="demo-form-field">To<input className="input" type="text" required maxLength={1000} value={recipient} onChange={event => { setRecipient(event.target.value); setDraftValidationError(false); }} placeholder="name@example.com" aria-describedby="communications-draft-hint" /></label>
      <p className="caption" id="communications-draft-hint">Separate multiple recipients with commas. Saving a draft does not send it.</p>
      <label className="demo-form-field">Subject<input className="input" required maxLength={240} value={subject} onChange={event => setSubject(event.target.value)} placeholder="Message subject" /></label>
      <label className="demo-form-field">Message<textarea className="input" required maxLength={20000} rows={4} value={body} onChange={event => setBody(event.target.value)} placeholder="Write a draft for review…" /></label>
      <button className="btn primary" type="submit" disabled={pending || !recipient.trim() || !subject.trim() || !body.trim()}>{createDraft.isPending ? "Saving draft…" : "Save draft"}</button>
    </form>
    {!drafts.length ? <EmptyState icon={Mail} title="No saved drafts" detail="Saved drafts appear here. Sending stays unavailable until the backend records and approves a request." /> : <div className="p-record-sections">
      <label className="demo-form-field">Saved draft<select className="select" value={String(selectedDraft?.id ?? "")} onChange={(event) => { setSelectedDraftId(event.target.value); setSelectedApprovalId(""); }}>
        {drafts.map((draft, index) => <option key={String(draft.id ?? index)} value={String(draft.id ?? "")}>{String(draft.subject ?? "(No subject)")} · {String(draft.status ?? "draft")}</option>)}
      </select></label>
      {selectedDraft && <article className="p-record-section"><h4>{String(selectedDraft.subject ?? "(No subject)")}</h4>
        <p className="caption">To {Array.isArray(selectedDraft.recipients) ? selectedDraft.recipients.map(String).join(", ") : "No recipients recorded"} · {String(selectedDraft.status ?? "draft")}</p>
        <p className="p-mail-body">{String(selectedDraft.body ?? "")}</p>
        <div className="row section-space"><button className="btn" type="button" disabled={pending || !selectedDraft.id || selectedDraft.status === "sent" || Boolean(approval && ["pending", "approved"].includes(String(approval.status)))} onClick={() => { setFeedback(""); request.mutate(); }}>{request.isPending ? "Requesting…" : "Request approval"}</button>
          {approval?.status === "pending" && <><button className="btn primary" type="button" disabled={pending} onClick={() => { setFeedback(""); resolve.mutate(true); }}>{resolve.isPending ? "Recording decision…" : "Approve request"}</button><button className="btn" type="button" disabled={pending} onClick={() => { setFeedback(""); resolve.mutate(false); }}>Reject</button></>}
        </div>
        {approval && <div className="p-note section-space"><Shield className="icon sm" aria-hidden="true" /><span>Approval {String(approval.id ?? "").slice(0, 12)} · backend status: <strong>{String(approval.status ?? "unknown")}</strong></span></div>}
        {approval?.status === "approved" && selectedDraft.status !== "sent" && <div className="p-record-section section-space">
          <h4>Send now</h4>
          <label className="row"><input type="checkbox" checked={confirmSend} onChange={(event) => setConfirmSend(event.target.checked)} />I have reviewed this exact draft and explicitly confirm the send request.</label>
          <p className="caption">The request includes <code>allow_external: false</code>. The backend may still refuse delivery if connector policy requires separate authorization.</p>
          <button className="btn primary" type="button" disabled={pending || !confirmSend} onClick={() => { setFeedback(""); send.mutate(); }}>{send.isPending ? "Submitting approved send…" : "Send approved draft"}</button>
          <h4 className="section-space">Schedule</h4>
          <label className="demo-form-field">Send at<input className="input" type="datetime-local" value={scheduleAt} min={new Date(Date.now() + 60_000).toISOString().slice(0, 16)} onChange={(event) => setScheduleAt(event.target.value)} /></label>
          <button className="btn" type="button" disabled={pending || !scheduleAt || new Date(scheduleAt).getTime() <= Date.now()} onClick={() => schedule.mutate()}>{schedule.isPending ? "Scheduling…" : "Schedule approved draft"}</button>
          <p className="caption">Scheduling does not send immediately. The backend requires this approved draft when the due send is processed.</p>
        </div>}
      </article>}
    </div>}
    {feedback && <p className="caption" role={draftValidationError || request.isError || resolve.isError || send.isError || schedule.isError || createDraft.isError ? "alert" : "status"}>{feedback}</p>}
  </section>;
}

function VoiceWorkspace({ tab, data, query }: { tab: string; data?: Record<string, JsonRecord>; query: { isLoading: boolean; error: unknown } }) {
  const queryClient = useQueryClient();
  const dataMode = useWorkspaceState(state => state.dataMode);
  const [sessionId, setSessionId] = useState("");
  const [language, setLanguage] = useState("en");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [synthesisText, setSynthesisText] = useState("");
  const [feedback, setFeedback] = useState("");
  const [latestTranscription, setLatestTranscription] = useState<JsonRecord | null>(null);
  const [latestSynthesis, setLatestSynthesis] = useState<JsonRecord | null>(null);
  const [synthesisAudioUrl, setSynthesisAudioUrl] = useState("");
  const [synthesisAudioBytes, setSynthesisAudioBytes] = useState(0);
  const [synthesisAudioLoading, setSynthesisAudioLoading] = useState(false);
  const [synthesisAudioError, setSynthesisAudioError] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordingWaveform, setRecordingWaveform] = useState<number[]>([]);
  const captureRef = useRef<{
    recorder: MediaRecorder;
    stream: MediaStream;
    chunks: BlobPart[];
    sessionId: string;
    audioContext: AudioContext | null;
    analyser: AnalyserNode | null;
    clockTimer: number | null;
    waveformTimer: number | null;
    discard: boolean;
  } | null>(null);
  const sessions = Array.isArray(data?.sessions?.sessions) ? data.sessions.sessions.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const personas = Array.isArray(data?.personas?.personas) ? data.personas.personas.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const status = data?.status ?? {};
  const selectedSession = sessions.find((session) => String(session.id ?? "") === sessionId) ?? sessions.find((session) => session.status === "active") ?? sessions[0];
  const activeSessionId = String(selectedSession?.id ?? "");
  const turnsQuery = useQuery({
    queryKey: ["voice-turns", activeSessionId],
    queryFn: ({ signal }) => apiGet(`/api/voice/turns?session_id=${encodeURIComponent(activeSessionId)}`, signal),
    enabled: tab === "Transcripts" && Boolean(activeSessionId),
    retry: false,
  });
  const refresh = () => queryClient.invalidateQueries({ queryKey: ["advanced-screen", "voice"] });
  const createSession = useMutation({
    mutationFn: () => apiPost("/api/voice/sessions", { project: "default", mode: "transcription", language: language.trim() || "en", always_listening: false }),
    onSuccess: async (result) => {
      const session = result.session as JsonRecord | undefined;
      await refresh();
      setSessionId(String(session?.id ?? ""));
      setFeedback(`Session recorded by the backend: ${String(session?.status ?? "response received")}. This does not start a microphone.`);
    },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const closeSession = useMutation({
    mutationFn: (id: string) => apiPost("/api/voice/sessions/close", { session_id: id }),
    onSuccess: async (result) => { await refresh(); setFeedback(`Backend session status: ${String((result.session as JsonRecord | undefined)?.status ?? "response received")}.`); },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const transcribe = useMutation({
    mutationFn: async ({ file, targetSessionId }: { file: File; targetSessionId: string }) => {
      if (!file || !targetSessionId) throw new Error("Choose an audio file and an existing voice session.");
      const audioBase64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(",").at(-1) ?? "");
        reader.onerror = () => reject(new Error("The selected audio file could not be read."));
        reader.readAsDataURL(file);
      });
      return apiPost("/api/voice/stt", { audio_base64: audioBase64, filename: file.name, project: "default", session_id: targetSessionId, language: language.trim() || "en", allow_fallback_stt: false });
    },
    onSuccess: async (result, variables) => {
      const transcription = result.transcription as JsonRecord | undefined;
      setLatestTranscription(transcription ?? null);
      setFeedback(`Backend transcription status: ${String(transcription?.transcript_status ?? "response received")}.`);
      setAudioFile(null);
      await refresh();
      await queryClient.invalidateQueries({ queryKey: ["voice-turns", variables.targetSessionId] });
    },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const synthesize = useMutation({
    mutationFn: () => apiPost("/api/voice/tts", { text: synthesisText.trim(), project: "default", session_id: activeSessionId, persona_id: String(selectedSession?.persona_id ?? ""), voice: "local-script" }),
    onSuccess: async (result) => {
      const script = result.script as JsonRecord | undefined;
      setLatestSynthesis(script ?? null);
      setFeedback(`Backend synthesis status: ${String(script?.audio_status ?? "response received")}.`);
      setSynthesisAudioError("");
      setSynthesisAudioBytes(0);
      setSynthesisAudioUrl(current => { if (current) URL.revokeObjectURL(current); return ""; });
      const audioAssetId = String(script?.audio_asset_id ?? "");
      if (dataMode === "live" && audioAssetId) {
        setSynthesisAudioLoading(true);
        try {
          const project = String(script?.project ?? selectedSession?.project ?? "default");
          const blob = await backendGetBlob(`/api/voice/assets/${encodeURIComponent(audioAssetId)}/content?project=${encodeURIComponent(project)}`);
          if (!blob.type.startsWith("audio/")) throw new Error("The backend asset is not reported as playable audio.");
          if (blob.size <= 44) throw new Error("The backend returned an empty or incomplete WAV asset.");
          const signatureBuffer = await new Promise<ArrayBuffer>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => reader.result instanceof ArrayBuffer ? resolve(reader.result) : reject(new Error("Could not inspect the WAV header."));
            reader.onerror = () => reject(new Error("Could not inspect the WAV header."));
            reader.readAsArrayBuffer(blob.slice(0, 4));
          });
          const signature = new Uint8Array(signatureBuffer);
          if (String.fromCharCode(...signature) !== "RIFF") throw new Error("The backend asset is not a valid WAV file.");
          setSynthesisAudioBytes(blob.size);
          setSynthesisAudioUrl(URL.createObjectURL(blob));
        } catch (error) {
          setSynthesisAudioError(`Generated audio could not be loaded: ${apiConnectionMessage(error)}`);
        } finally { setSynthesisAudioLoading(false); }
      }
      setSynthesisText("");
      await refresh();
      if (activeSessionId) await queryClient.invalidateQueries({ queryKey: ["voice-turns", activeSessionId] });
    },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const queryError = query.error ? apiConnectionMessage(query.error) : "";
  const turnError = turnsQuery.error ? apiConnectionMessage(turnsQuery.error) : "";
  const turns = Array.isArray(turnsQuery.data?.turns) ? turnsQuery.data.turns.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const busy = createSession.isPending || closeSession.isPending || transcribe.isPending || synthesize.isPending;
  useEffect(() => () => { if (synthesisAudioUrl) URL.revokeObjectURL(synthesisAudioUrl); }, [synthesisAudioUrl]);
  const finishCapture = (capture: NonNullable<typeof captureRef.current>) => {
    if (capture.clockTimer !== null) window.clearInterval(capture.clockTimer);
    if (capture.waveformTimer !== null) window.clearInterval(capture.waveformTimer);
    capture.stream.getTracks().forEach(track => track.stop());
    if (capture.audioContext && capture.audioContext.state !== "closed") void capture.audioContext.close().catch(() => undefined);
    if (captureRef.current === capture) {
      captureRef.current = null;
      setIsRecording(false);
      setRecordingWaveform([]);
    }
    if (capture.discard) return;
    const mimeType = capture.recorder.mimeType || "audio/webm";
    const audioBlob = new Blob(capture.chunks, { type: mimeType });
    if (!audioBlob.size) { setFeedback("The recording contained no audio data. Try recording again."); return; }
    const extension = mimeType.includes("ogg") ? "ogg" : mimeType.includes("mp4") ? "m4a" : "webm";
    const file = new File([audioBlob], `voice-recording-${new Date().toISOString().replace(/[:.]/g, "-")}.${extension}`, { type: mimeType });
    setAudioFile(file);
    setFeedback("Recording stopped. Sending the captured audio to the local transcription service…");
    transcribe.mutate({ file, targetSessionId: capture.sessionId });
  };
  const startRecording = async () => {
    setFeedback("");
    if (!activeSessionId) { setFeedback("Create or select a voice session before recording."); return; }
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") { setFeedback("This browser does not support microphone recording."); return; }
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"].find(type => typeof MediaRecorder.isTypeSupported !== "function" || MediaRecorder.isTypeSupported(type));
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      const capture = {
        recorder, stream, chunks: [] as BlobPart[], sessionId: activeSessionId,
        audioContext: null as AudioContext | null, analyser: null as AnalyserNode | null,
        clockTimer: null as number | null, waveformTimer: null as number | null, discard: false,
      };
      captureRef.current = capture;
      recorder.ondataavailable = event => { if (event.data.size) capture.chunks.push(event.data); };
      recorder.onstop = () => finishCapture(capture);
      recorder.onerror = () => { capture.discard = true; setFeedback("The browser could not continue recording. No audio was uploaded."); if (recorder.state !== "inactive") recorder.stop(); else finishCapture(capture); };
      recorder.start(250);
      setRecordingSeconds(0);
      setRecordingWaveform([18, 35, 54, 32, 46, 24, 59, 42, 28, 51, 36, 62, 31, 45, 22, 53, 39, 60, 29, 47, 34, 57, 25, 44, 33, 61, 39, 50, 27, 46, 32, 55]);
      setIsRecording(true);
      setFeedback("Recording from this browser microphone. Audio is uploaded to the local service after you stop.");
      capture.clockTimer = window.setInterval(() => setRecordingSeconds(value => value + 1), 1000);
      try {
        if (typeof AudioContext !== "undefined") {
          capture.audioContext = new AudioContext();
          const source = capture.audioContext.createMediaStreamSource(stream);
          capture.analyser = capture.audioContext.createAnalyser();
          capture.analyser.fftSize = 128;
          source.connect(capture.analyser);
          const samples = new Uint8Array(capture.analyser.fftSize);
          capture.waveformTimer = window.setInterval(() => {
            if (!capture.analyser) return;
            capture.analyser.getByteTimeDomainData(samples);
            const levels = Array.from({ length: 32 }, (_, index) => {
              const offset = Math.floor(index * samples.length / 32);
              const amplitude = Math.abs((samples[offset] ?? 128) - 128) / 128;
              return Math.round(12 + Math.min(1, amplitude * 3) * 48);
            });
            setRecordingWaveform(levels);
          }, 90);
        }
      } catch {
        // Capture still works if audio-level visualization is unavailable.
      }
    } catch (error) {
      stream?.getTracks().forEach(track => track.stop());
      const reason = error instanceof DOMException && error.name === "NotAllowedError" ? "Microphone permission was denied. Allow microphone access in your browser to record." : error instanceof Error ? error.message : "Microphone capture could not start.";
      setFeedback(reason);
    }
  };
  const stopRecording = (discard = false) => {
    const capture = captureRef.current;
    if (!capture) return;
    capture.discard = discard;
    if (discard) setFeedback("Recording discarded. Audio was not uploaded.");
    if (capture.recorder.state === "inactive") finishCapture(capture);
    else capture.recorder.stop();
  };
  useEffect(() => () => {
    const capture = captureRef.current;
    if (!capture) return;
    capture.discard = true;
    if (capture.clockTimer !== null) window.clearInterval(capture.clockTimer);
    if (capture.waveformTimer !== null) window.clearInterval(capture.waveformTimer);
    capture.recorder.onstop = null;
    if (capture.recorder.state !== "inactive") { try { capture.recorder.stop(); } catch { /* The device may already have stopped. */ } }
    capture.stream.getTracks().forEach(track => track.stop());
    if (capture.audioContext && capture.audioContext.state !== "closed") void capture.audioContext.close().catch(() => undefined);
    captureRef.current = null;
  }, []);

  if (tab === "Recordings") return <>
    {query.isLoading && <section className="p-card" role="status"><p className="caption">Loading voice sessions and local runtime status…</p></section>}
    <section className="p-card voice-session-card">
      {queryError && <p className="p-note voice-query-status" role="alert"><AlertCircle className="icon sm" aria-hidden="true" />{queryError}</p>}
      <div className="p-voice-hero">
        <div className="p-record-circle"><Mic className="icon xl" aria-hidden="true" /></div>
        <span className="p-eyebrow">Voice workspace</span>
        <h2>Your ideas, captured clearly.</h2>
        <p>Start a recording, then review an accurate transcript before using it in a conversation.</p>
        <div className={`p-wave${isRecording ? " is-recording" : ""}`} aria-hidden="true">{Array.from({ length: 47 }, (_, index) => {
          const measuredHeight = recordingWaveform[index % Math.max(1, recordingWaveform.length)];
          const idleHeight = 15 + (index * 19 + index * index * 3) % 66;
          return <span key={index} style={{ "--wave-height": `${isRecording && measuredHeight ? measuredHeight : idleHeight}px`, "--wave-delay": `${(index % 11) * -0.08}s` } as CSSProperties} />;
        })}</div>
        <div className="row" style={{ justifyContent: "center" }}>
          {isRecording
            ? <><button type="button" className="btn primary" onClick={() => stopRecording(false)}><AudioLines className="icon sm" aria-hidden="true" />Stop and transcribe</button><button type="button" className="btn" onClick={() => stopRecording(true)}>Discard recording</button></>
            : <><button type="button" className="btn primary" disabled={busy || !activeSessionId} title={!activeSessionId ? "Create a voice session first." : undefined} onClick={() => void startRecording()}><Mic className="icon sm" aria-hidden="true" />{busy ? "Preparing…" : "Start recording"}</button><button type="button" className="btn" disabled={busy || isRecording || !language.trim()} onClick={() => { setFeedback(""); createSession.mutate(); }}>{createSession.isPending ? "Creating session…" : "New session"}</button></>
          }
        </div>
        <span className="voice-idle-time" aria-live="polite">{`${String(Math.floor(recordingSeconds / 60)).padStart(2, "0")}:${String(recordingSeconds % 60).padStart(2, "0")}`}</span>
        <small className="p-subtext">Waveform is a visual animation, not live microphone input.</small>
      </div>
      {latestTranscription && <article className="p-record-section"><h4>Backend transcription result</h4><p className="caption">Status: {String(latestTranscription.transcript_status ?? "unreported")} · engine: {String(latestTranscription.stt_engine ?? "unreported")} · confidence: {String(latestTranscription.confidence ?? "unreported")}</p><p>{String(latestTranscription.display_text ?? "No transcript text returned.")}</p>{typeof latestTranscription.stt_message === "string" && latestTranscription.stt_message && <p className="caption">{latestTranscription.stt_message}</p>}{Array.isArray(latestTranscription.warnings) && <p className="caption">{latestTranscription.warnings.map(String).join(" · ")}</p>}</article>}
      {feedback && <p className="caption" role={createSession.isError || closeSession.isError || transcribe.isError || synthesize.isError ? "alert" : "status"}>{feedback}</p>}
    </section>
  </>;

  return <div className="p-record-sections">
    {query.isLoading && <section className="p-card" role="status"><p className="caption">Loading voice sessions and local runtime status…</p></section>}
    <section className={`p-card${tab === "Recordings" ? " voice-session-card" : ""}`}>
      <div className="p-card-heading"><h3>{tab === "Voice settings" ? "Local voice runtime" : tab === "Transcripts" ? "Voice transcript" : "Voice sessions"}</h3><span className="p-fixture">{String(status.runtime ?? "runtime status unavailable")}</span></div>
      {queryError && <p className="p-note voice-query-status" role="alert"><AlertCircle className="icon sm" aria-hidden="true" />{queryError}</p>}
      <p className="caption">{String(status.privacy_boundary ?? "Voice data handling is reported by the backend.")}</p>
      <p className="caption"><strong>Microphone capture:</strong> {String(status.microphone_control ?? "not reported")}. This workspace has no live microphone streaming endpoint.</p>
      <div className="p-record-section"><label className="demo-form-field">Voice session<select className="select" value={activeSessionId} onChange={(event) => setSessionId(event.target.value)} aria-label="Voice session">
        {!sessions.length && <option value="">No sessions</option>}{sessions.map((session, index) => <option key={String(session.id ?? index)} value={String(session.id ?? "")}>{String(session.id ?? "Session").slice(0, 12)} · {String(session.mode ?? "mode unavailable")} · {String(session.status ?? "unknown")}</option>)}
      </select></label>
      <label className="demo-form-field">Language<input className="input" value={language} onChange={(event) => setLanguage(event.target.value)} maxLength={20} /></label>
      <button type="button" className="btn" disabled={busy || !language.trim()} onClick={() => { setFeedback(""); createSession.mutate(); }}>{createSession.isPending ? "Creating session…" : "Create transcription session"}</button>
      {selectedSession?.status === "active" && <button type="button" className="btn section-space" disabled={busy} onClick={() => { setFeedback(""); closeSession.mutate(activeSessionId); }}>{closeSession.isPending ? "Closing…" : "Close session"}</button>}
      {tab === "Transcripts" && <><label className="demo-form-field section-space">Audio file<input aria-label="Choose audio file" className="input" type="file" accept="audio/*,.webm,.ogg,.wav,.mp4,.m4a" onChange={(event) => setAudioFile(event.currentTarget.files?.[0] ?? null)} /></label>{audioFile && <p className="caption">Selected: {audioFile.name} · {audioFile.size.toLocaleString()} bytes</p>}<button type="button" className="btn" disabled={!audioFile || !activeSessionId || busy} onClick={() => { if (!audioFile || !activeSessionId) return; setFeedback(""); transcribe.mutate({ file: audioFile, targetSessionId: activeSessionId }); }}>{transcribe.isPending ? "Uploading and transcribing…" : "Upload and transcribe"}</button></>}
      {tab === "Transcripts" && latestTranscription && <article className="p-record-section"><h4>Backend transcription result</h4><p className="caption">Status: {String(latestTranscription.transcript_status ?? "unreported")} · engine: {String(latestTranscription.stt_engine ?? "unreported")} · confidence: {String(latestTranscription.confidence ?? "unreported")}</p><p>{String(latestTranscription.display_text ?? "No transcript text returned.")}</p>{typeof latestTranscription.stt_message === "string" && latestTranscription.stt_message && <p className="caption">{latestTranscription.stt_message}</p>}{Array.isArray(latestTranscription.warnings) && <p className="caption">{latestTranscription.warnings.map(String).join(" · ")}</p>}</article>}
      </div>
      {sessions.length > 0 && <div className="p-record-sections"><h4>Backend session history</h4>{sessions.map((session, index) => <article className="p-record-section" key={String(session.id ?? index)}><strong>{String(session.mode ?? "Voice session")} · {String(session.status ?? "unknown")}</strong><p className="caption">{String(session.created_at ?? "")} · language {String(session.language ?? "unreported")} · {String(session.id ?? "")}</p></article>)}</div>}
    </section>

    {tab === "Transcripts" && <section className="p-card"><div className="p-card-heading"><h3>Session turns</h3><span className="p-fixture">{turns.length} backend turns</span></div>{turnError && <p className="caption" role="alert">{turnError}</p>}{turnsQuery.isLoading && <p className="caption" role="status">Loading turns…</p>}{!turnError && !turnsQuery.isLoading && !turns.length && <EmptyState icon={AudioLines} title="No transcript turns" detail="Choose a session or upload audio to a transcription session." />}{turns.map((turn, index) => { const metadata = turn.metadata && typeof turn.metadata === "object" ? turn.metadata as JsonRecord : {}; const stt = metadata.stt && typeof metadata.stt === "object" ? metadata.stt as JsonRecord : {}; return <article className="p-record-section" key={String(turn.id ?? index)}><div className="row between"><strong>{String(turn.speaker ?? turn.role ?? "speaker")}</strong><span className="caption">{String(turn.created_at ?? "")}</span></div><p>{String(turn.text ?? "")}</p><p className="caption">{String(turn.role ?? "turn")} · {turn.is_final === true ? "final" : "partial"}{stt.status ? ` · STT ${String(stt.status)} (${String(stt.provider ?? "engine unavailable")})` : ""}{turn.audio_asset_id ? ` · audio asset ${String(turn.audio_asset_id)}` : ""}</p></article>; })}</section>}

    {tab === "Voice settings" && <section className="p-card"><div className="p-card-heading"><h3>Local speech synthesis</h3></div><p className="caption">Generated WAV audio is stored in the local media library and can be played here. No external TTS provider is called.</p>{personas.length > 0 && <p className="caption">Available backend personas: {personas.map((persona) => String(persona.name ?? persona.id)).join(", ")}</p>}<label className="demo-form-field">Text to synthesize<textarea className="input" rows={3} value={synthesisText} onChange={(event) => setSynthesisText(event.target.value)} maxLength={5000} /></label><button type="button" className="btn primary" disabled={!synthesisText.trim() || busy} onClick={() => { setFeedback(""); synthesize.mutate(); }}>{synthesize.isPending ? "Generating local audio…" : "Generate local speech artifact"}</button>{latestSynthesis && <article className="p-record-section section-space"><h4>Backend synthesis artifact</h4><p className="caption">{String(latestSynthesis.audio_status ?? "status unavailable")} · {String(latestSynthesis.audio_mime_type ?? "type unavailable")} · asset {String(latestSynthesis.audio_asset_id ?? "ID unavailable")}</p><p className="caption">{String(latestSynthesis.privacy_boundary ?? "")}</p>{synthesisAudioLoading && <p role="status" className="caption">Loading the authenticated local WAV asset…</p>}{synthesisAudioError && <p role="alert" className="caption">{synthesisAudioError}</p>}{synthesisAudioUrl && <div className="voice-generated-audio"><audio controls preload="metadata" src={synthesisAudioUrl} aria-label="Generated local voice audio" data-audio-bytes={synthesisAudioBytes}/><a className="btn sm" href={synthesisAudioUrl} download={`openagent-voice-${String(latestSynthesis.audio_asset_id ?? "audio")}.wav`}>Download WAV</a></div>}</article>}</section>}

    {feedback && <p className="caption" role={createSession.isError || closeSession.isError || transcribe.isError || synthesize.isError ? "alert" : "status"}>{feedback}</p>}
  </div>;
}

function VoiceDemoTranscripts() {
  const [editing, setEditing] = useState(false);
  const [transcript, setTranscript] = useState("We should make this workflow easy to understand from the first click. The agent timeline needs to show real status, not just a spinner.");
  const [saved, setSaved] = useState("");
  const exportTranscript = () => {
    const contents = `[00:04] ${transcript.split(" The agent timeline")[0]}\n[00:13] The agent timeline${transcript.split(" The agent timeline")[1] ?? " needs to show real status, not just a spinner."}`;
    const url = URL.createObjectURL(new Blob([contents], { type: "text/plain" }));
    const link = document.createElement("a"); link.href = url; link.download = "openagent-demo-transcript.txt"; link.click(); URL.revokeObjectURL(url);
    setSaved("Sample transcript downloaded.");
  };
  return <>
    <section className="p-card p-tight">
      <div className="p-card-heading"><h3>Transcripts</h3></div>
      <div className="p-transcript">
        <div className="p-transcript-row"><span>00:04</span><p>We should make this workflow easy to understand from the first click.</p></div>
        <div className="p-transcript-row"><span>00:13</span><p>The agent timeline needs to show real status, not just a spinner.</p></div>
      </div>
      <div className="row" style={{ marginBottom: 12 }}>
        <button className="btn soft" type="button" onClick={() => setEditing(true)}><SlidersHorizontal className="icon sm"/>Edit transcript</button>
        <button className="btn" type="button" onClick={exportTranscript}><FileDown className="icon sm"/>Export text</button>
      </div>
      {saved && <p className="caption" role="status">{saved}</p>}
      <section className="p-card p-tight voice-demo-recordings">
        <div className="p-card-heading"><h3>Your recordings</h3><span className="p-fixture">2 demo records</span></div>
        <article className="demo-record"><span className="square-icon sm blue"><Mic className="icon sm"/></span><span className="demo-record-text"><strong>Product planning sync</strong><small>Review onboarding feedback and release blockers</small></span><span className="tag green">Transcribed</span></article>
        <article className="demo-record"><span className="square-icon sm blue"><Mic className="icon sm"/></span><span className="demo-record-text"><strong>Quick idea — agent UI</strong><small>Notes about task execution feedback</small></span><span className="tag green">Transcribed</span></article>
      </section>
    </section>
    <Dialog.Root open={editing} onOpenChange={setEditing}>
      <Dialog.Portal><Dialog.Overlay/><Dialog.Content className="second-app-dialog voice-demo-dialog demo-large-modal">
        <div className="row between"><Dialog.Title className="second-app-dialog-title">Edit demo transcript</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close dialog"><X className="icon sm"/></button></Dialog.Close></div>
        <Dialog.Description className="second-app-dialog-description">Edit the transcript text and save a local example.</Dialog.Description>
        <form onSubmit={event => { event.preventDefault(); setEditing(false); setSaved("Transcript saved locally."); }}>
          <textarea className="textarea" rows={8} required value={transcript} onChange={event => setTranscript(event.target.value)}/>
          <div className="row end voice-demo-modal-actions"><button className="btn" type="button" onClick={() => setEditing(false)}>Cancel</button><button className="btn primary" type="submit">Save transcript</button></div>
        </form>
      </Dialog.Content></Dialog.Portal>
    </Dialog.Root>
  </>;
}

function VoiceDemoSettings() {
  const [notice, setNotice] = useState("");
  const [language, setLanguage] = useState("English (UK)");
  const [device, setDevice] = useState("Demo input device");
  const [speed, setSpeed] = useState("1× normal");
  useEffect(() => {
    try { const saved = JSON.parse(localStorage.getItem("oa-demo-preferences-v4") ?? "{}"); const voice = saved.voice ?? {}; if (voice.language) setLanguage(String(voice.language)); if (voice.device) setDevice(String(voice.device)); if (voice.speed) setSpeed(String(voice.speed)); } catch { /* keep deterministic defaults */ }
  }, []);
  return <>
    <section className="p-card p-tight"><div className="p-card-heading"><h3>Voice preferences</h3></div>
      <form className="demo-settings-form" onSubmit={event => { event.preventDefault(); try { const all = JSON.parse(localStorage.getItem("oa-demo-preferences-v4") ?? "{}"); localStorage.setItem("oa-demo-preferences-v4", JSON.stringify({ ...all, voice: { language, device, speed } })); } catch { /* local demo remains usable if storage is blocked */ } setNotice("Preferences saved locally."); }}>
        <div className="p-form-grid"><label className="p-field">Transcription language<select className="select" name="language" value={language} onChange={event => setLanguage(event.target.value)}><option>English (UK)</option><option>English (US)</option><option>Urdu</option><option>Spanish</option></select></label><label className="p-field">Microphone<select className="select" name="device" value={device} onChange={event => setDevice(event.target.value)}><option>Demo input device</option><option>System microphone</option></select></label></div>
        <label className="p-field">Playback speed<select className="select" name="speed" value={speed} onChange={event => setSpeed(event.target.value)}><option>1× normal</option><option>1.25× faster</option><option>0.75× slower</option></select></label>
        <button type="submit" className="btn primary"><CheckCircle2 className="icon sm"/>Save preferences</button>
      </form>
      {notice && <p className="caption" role="status">{notice}</p>}
    </section>
  </>;
}

function VoiceDemoRecordings({ data }: { data?: Record<string, JsonRecord> }) {
  const client = useQueryClient();
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [toastMessage, setToastMessage] = useState("");
  const [filter, setFilter] = useState("");
  const [dialog, setDialog] = useState<"create" | "edit" | "details" | "actions" | "delete" | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [duration, setDuration] = useState("");
  const [formError, setFormError] = useState("");
  const [busy, setBusy] = useState(false);
  const toastTimer = useRef<number | null>(null);
  const [recordingRun, setRecordingRun] = useState<{ id: string; name: string; step: number; status: "Running" | "Completed" } | null>(null);
  const runTimer = useRef<number | null>(null);
  const hideRunTimer = useRef<number | null>(null);
  const recordings = Array.isArray(data?.recordings?.recordings) ? data.recordings.recordings.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const selected = recordings.find(item => String(item.id ?? "") === selectedId);
  const visibleRecordings = recordings.filter(item => !filter.trim() || JSON.stringify(item).toLowerCase().includes(filter.trim().toLowerCase()));
  const refresh = () => client.invalidateQueries({ queryKey: ["advanced-screen", "voice"] });
  const showToast = (message: string) => {
    if (toastTimer.current !== null) window.clearTimeout(toastTimer.current);
    setToastMessage(message);
    toastTimer.current = window.setTimeout(() => { setToastMessage(""); toastTimer.current = null; }, 3600);
  };
  const openRecordDialog = (mode: "create" | "edit" | "details" | "actions" | "delete", item?: JsonRecord) => {
    setSelectedId(String(item?.id ?? ""));
    setName(String(item?.name ?? ""));
    setDescription(String(item?.description ?? ""));
    setDuration(String(item?.duration ?? ""));
    setFormError("");
    setDialog(mode);
  };
  const closeDialog = () => { if (!busy) { setDialog(null); setFormError(""); } };
  const saveRecord = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!name.trim()) { setFormError("Enter a recording name."); return; }
    setBusy(true); setFormError("");
    try {
      if (dialog === "edit") await advancedDemoAction("/api/voice/recordings/update", { recording_id: selectedId, name: name.trim(), description: description.trim(), duration: duration.trim() });
      else await advancedDemoAction("/api/voice/recordings/create", { name: name.trim(), description: description.trim(), duration: duration.trim() || "00:00" });
      await refresh(); setDialog(null); setFeedback(dialog === "edit" ? "Recording updated in the local demo." : "Recording created in the local demo.");
    } catch (cause) { setFormError(cause instanceof Error ? cause.message : "Could not save this demo recording."); }
    finally { setBusy(false); }
  };
  const deleteRecord = async () => {
    setBusy(true); setFormError("");
    try {
      await advancedDemoAction("/api/voice/recordings/delete", { recording_id: selectedId });
      await refresh(); setDialog(null); setFeedback("Demo recording deleted locally.");
    } catch (cause) { setFormError(cause instanceof Error ? cause.message : "Could not delete this demo recording."); }
    finally { setBusy(false); }
  };
  const duplicateRecord = async () => {
    setBusy(true); setFormError("");
    try {
      await advancedDemoAction("/api/voice/recordings/duplicate", { recording_id: selectedId });
      await refresh(); setDialog(null); setFeedback("A copy was added to the local demo recordings.");
    } catch (cause) { setFormError(cause instanceof Error ? cause.message : "Could not duplicate this demo recording."); }
    finally { setBusy(false); }
  };
  const startRecordingRun = async (item: JsonRecord) => {
    const id = String(item.id ?? ""); const itemName = String(item.name ?? "Voice recording");
    if (!id || useWorkspaceState.getState().dataMode !== "demo") return;
    if (runTimer.current !== null) window.clearInterval(runTimer.current);
    if (hideRunTimer.current !== null) window.clearTimeout(hideRunTimer.current);
    setDialog(null); setFormError(""); setFeedback("A local example workflow is running. No microphone, backend or external audio service is used.");
    setRecordingRun({ id, name: itemName, step: 0, status: "Running" });
    await advancedDemoAction("/api/voice/recordings/update", { recording_id: id, status: "Running" });
    await refresh();
    let step = 0;
    runTimer.current = window.setInterval(() => {
      step += 1;
      if (step >= 4) {
        if (runTimer.current !== null) window.clearInterval(runTimer.current);
        runTimer.current = null;
        setRecordingRun({ id, name: itemName, step: 4, status: "Completed" });
        void advancedDemoAction("/api/voice/recordings/update", { recording_id: id, status: "Completed", duration: item.duration || "4s" }).then(refresh);
        setFeedback("The local example workflow completed. No backend or real audio operation occurred.");
        hideRunTimer.current = window.setTimeout(() => setRecordingRun(current => current?.id === id ? null : current), 4200);
      } else setRecordingRun(current => current?.id === id ? { ...current, step } : current);
    }, 1000);
  };
  const cancelRecordingRun = async () => {
    const current = recordingRun; if (!current) return;
    if (runTimer.current !== null) window.clearInterval(runTimer.current);
    if (hideRunTimer.current !== null) window.clearTimeout(hideRunTimer.current);
    runTimer.current = null; hideRunTimer.current = null; setRecordingRun(null);
    await advancedDemoAction("/api/voice/recordings/update", { recording_id: current.id, status: "Cancelled" });
    await refresh(); setFeedback("The local example workflow was cancelled.");
  };
  useEffect(() => {
    if (!recording) return;
    const timer = window.setInterval(() => setSeconds(value => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [recording]);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setTimeout(() => { setPlaying(false); setFeedback("Sample preview finished. No microphone was accessed."); }, 1400);
    return () => window.clearTimeout(timer);
  }, [playing]);
  useEffect(() => () => {
    if (runTimer.current !== null) window.clearInterval(runTimer.current);
    if (hideRunTimer.current !== null) window.clearTimeout(hideRunTimer.current);
    if (toastTimer.current !== null) window.clearTimeout(toastTimer.current);
  }, []);
  const stopAndSave = async () => {
    setRecording(false);
    await advancedDemoAction("/api/voice/stt", { filename: `New recording · ${new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}`, duration: `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`, description: "Illustrative transcription: team updates and next steps." });
    await client.invalidateQueries({ queryKey: ["advanced-screen", "voice"] });
    setSeconds(0);
    setFeedback("");
    showToast("Saved a demo recording and transcript.");
  };
  const time = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  return <>
    <section className="demo-feature-hero"><div className="demo-feature-symbol"><Mic className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">Recordings · Workspace</span><h2 style={{ fontWeight: 700 }}>Capture and understand ideas</h2><p>Record, review and edit the words that matter.</p></div></section>
    <section className="p-voice-hero">
      <div className="p-record-circle"><Mic className="icon xl" aria-hidden="true" /></div><div className="p-eyebrow">Interactive recording simulation</div><h2 style={{ fontWeight: 700 }}>Capture your next idea.</h2>
      <p>Start a simulated recording, watch the waveform react and save a transcript to the demo.</p>
      <div className={`p-wave${recording ? " demo-voice-active" : ""}`} aria-hidden="true">{Array.from({ length: 38 }, (_, index) => <span key={index} style={{ "--wave-height": `${14 + (index * 31 % 68)}px`, "--wave-delay": `${index * 29}ms` } as CSSProperties} />)}</div>
      <div className="demo-rec-timer" aria-live="polite">{time}</div>
      <div className="row">{recording
        ? <button className="btn primary" type="button" onClick={() => void stopAndSave()}><CheckCircle2 className="icon sm" />Stop &amp; save recording</button>
        : <button className="btn primary" type="button" onClick={() => { setSeconds(0); setFeedback(""); setRecording(true); showToast("Demo recording started (no microphone is accessed)."); }}><Mic className="icon sm" />Start recording</button>}
        <button className="btn" type="button" disabled={playing || recording} onClick={() => { setFeedback("Playing a sample preview. No microphone or external audio service is used."); setPlaying(true); }}><Play className="icon sm" />{playing ? "Playing sample…" : "Play sample"}</button>
      </div>
      {recording && <>
        <div className="live-inline" role="status" aria-live="polite"><span className="live-inline-orb"/><div className="live-inline-label"><strong>Recording preview</strong><small>No device microphone is accessed. Timer is synthetic.</small></div></div>
        <span className="live-recording-indicator">Simulated recording in progress</span>
      </>}
      {feedback && <p className="caption" role="status">{feedback}</p>}
    </section>
    <section className="p-card p-tight voice-demo-recordings"><div className="p-card-heading"><h3>Your recordings</h3><span className="p-fixture">{recordings.length} demo records</span></div>
      <div className="demo-record-toolbar"><label className="search-field demo-record-search"><Search className="icon sm" aria-hidden="true"/><input value={filter} onChange={event => setFilter(event.target.value)} placeholder="Search recordings…" aria-label="Search recordings"/></label><span className="caption" aria-live="polite">{visibleRecordings.length} item{visibleRecordings.length === 1 ? "" : "s"}</span><button className="btn primary" type="button" onClick={() => openRecordDialog("create")}><Plus className="icon sm" aria-hidden="true"/>Add</button></div>
      {!visibleRecordings.length ? <p className="caption">{filter ? "No matching items. Try a different search or create a sample record." : "No recordings yet. Create a sample record or start a demo recording above."}</p> : <div className="voice-demo-record-list">{visibleRecordings.map((item, index) => {
        const id = String(item.id ?? index); const itemName = String(item.name ?? item.mode ?? "Demo recording"); const itemStatus = String(item.status ?? "Ready");
        return <article className="demo-record" key={id}><button type="button" className="demo-record-main" aria-label={`Open ${itemName} details`} onClick={() => openRecordDialog("details", item)}><span className="square-icon sm blue"><Mic className="icon sm" aria-hidden="true"/></span><span className="demo-record-text"><strong>{itemName}</strong><small>{String(item.description ?? "Example voice recording")}</small></span></button><span className={`tag ${/transcribed|completed|ready/i.test(itemStatus) ? "green" : ""}`}>{itemStatus}</span><button className="icon-button" type="button" aria-label={`Edit ${itemName}`} title={`Edit ${itemName}`} onClick={() => openRecordDialog("edit", item)}><SlidersHorizontal className="icon sm" aria-hidden="true"/></button><button className="icon-button" type="button" aria-label={`More actions for ${itemName}`} title={`More actions for ${itemName}`} onClick={() => openRecordDialog("actions", item)}><MoreHorizontal className="icon sm" aria-hidden="true"/></button></article>;
      })}</div>}
    </section>
    <Dialog.Root open={dialog !== null} onOpenChange={open => { if (!open) closeDialog(); }}>
      <Dialog.Portal><Dialog.Overlay/><Dialog.Content className="voice-demo-dialog demo-large-modal">
        {dialog === "create" || dialog === "edit" ? <>
          <div className="row between"><Dialog.Title className="second-app-dialog-title">{dialog === "edit" ? "Edit recording" : "New recording"}</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close dialog" disabled={busy}><X className="icon sm"/></button></Dialog.Close></div>
          <Dialog.Description className="second-app-dialog-description">Changes are saved locally in this interactive demonstration.</Dialog.Description>
          <form className="voice-demo-record-form" onSubmit={event => void saveRecord(event)}><label className="demo-form-field"><span>Name</span><input className="input" required autoFocus maxLength={120} value={name} onChange={event => { setName(event.target.value); setFormError(""); }}/></label><label className="demo-form-field"><span>Description</span><textarea className="textarea" rows={3} maxLength={500} value={description} onChange={event => setDescription(event.target.value)}/></label><label className="demo-form-field"><span>Duration</span><input className="input" value={duration} onChange={event => setDuration(event.target.value)} placeholder="e.g. 02:16" maxLength={12}/></label>{formError && <p className="form-error" role="alert">{formError}</p>}<div className="row end voice-demo-modal-actions"><button className="btn" type="button" onClick={closeDialog} disabled={busy}>Cancel</button><button className="btn primary" type="submit" disabled={busy || !name.trim()}>{busy ? "Saving…" : dialog === "edit" ? "Save changes" : "Create item"}</button></div></form>
        </> : dialog === "details" && selected ? <>
          <div className="row between"><div className="row"><span className="square-icon blue"><Mic className="icon sm"/></span><div><Dialog.Title className="second-app-dialog-title">{String(selected.name ?? "Recording")}</Dialog.Title><p className="caption">recordings · {String(selected.status ?? "Ready")}</p></div></div><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close details"><X className="icon sm"/></button></Dialog.Close></div>
          <Dialog.Description className="second-app-dialog-description voice-demo-detail-description">{String(selected.description ?? "Interactive demo record")}</Dialog.Description><div className="demo-detail-grid">{Object.entries(selected).filter(([key]) => !["id", "name", "description"].includes(key)).map(([key, value]) => <div key={key}><small>{key.replaceAll("_", " ")}</small><strong>{String(value)}</strong></div>)}</div><p className="p-note"><Info className="icon sm" aria-hidden="true"/><span>Demo records and actions are local. The live backend is not changed.</span></p><div className="row voice-demo-modal-actions"><button className="btn soft" type="button" onClick={() => void startRecordingRun(selected)}><AudioLines className="icon sm"/>Run example</button><button className="btn" type="button" onClick={() => openRecordDialog("edit", selected)}><SlidersHorizontal className="icon sm"/>Edit</button><button className="btn danger" type="button" onClick={() => openRecordDialog("delete", selected)}><X className="icon sm"/>Delete</button></div>
        </> : dialog === "actions" && selected ? <>
          <div className="row between"><Dialog.Title className="second-app-dialog-title">{String(selected.name ?? "Recording actions")}</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close actions"><X className="icon sm"/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">Choose an action for this example record.</Dialog.Description><div className="demo-action-grid voice-demo-action-grid"><button className="btn" type="button" onClick={() => setDialog("details")}><FileDown className="icon sm"/>Open details</button><button className="btn" type="button" onClick={() => openRecordDialog("edit", selected)}><SlidersHorizontal className="icon sm"/>Edit record</button><button className="btn" type="button" onClick={() => void duplicateRecord()} disabled={busy}><Plus className="icon sm"/>Duplicate</button><button className="btn" type="button" onClick={() => void startRecordingRun(selected)}><AudioLines className="icon sm"/>Run / test</button><button className="btn danger" type="button" onClick={() => openRecordDialog("delete", selected)}><X className="icon sm"/>Delete</button></div>{formError && <p className="form-error" role="alert">{formError}</p>}
        </> : dialog === "delete" && selected ? <>
          <div className="row between"><Dialog.Title className="second-app-dialog-title">Delete “{String(selected.name ?? "recording")}”?</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close delete dialog" disabled={busy}><X className="icon sm"/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">This deletes the local demo record. It does not affect your backend.</Dialog.Description>{formError && <p className="form-error" role="alert">{formError}</p>}<div className="row end voice-demo-modal-actions"><button className="btn" type="button" onClick={closeDialog} disabled={busy}>Cancel</button><button className="btn danger" type="button" onClick={() => void deleteRecord()} disabled={busy}>{busy ? "Deleting…" : "Delete record"}</button></div>
        </> : null}
      </Dialog.Content></Dialog.Portal>
    </Dialog.Root>
    {toastMessage && <div id="toast-region" aria-live="polite" aria-atomic="true"><div className="toast toast-success"><CheckCircle2 className="icon sm"/><span>{toastMessage}</span></div></div>}
    {recordingRun && <section className="voice-demo-run-hud" role="status" aria-label={`Local workflow ${recordingRun.status.toLowerCase()}`}><div className="row between"><div><strong>{recordingRun.name}</strong><small>Simulated Voice recording workflow</small></div><span className={`tag ${recordingRun.status === "Completed" ? "green" : "orange"}`}>{recordingRun.status}</span>{recordingRun.status === "Running" && <button type="button" className="icon-button" aria-label="Cancel local run" title="Cancel local run" onClick={() => void cancelRecordingRun()}><X className="icon sm"/></button>}</div><div className="voice-demo-run-progress" role="progressbar" aria-label="Simulated recording workflow progress" aria-valuemin={0} aria-valuemax={4} aria-valuenow={recordingRun.step}><span style={{ width: `${recordingRun.step / 4 * 100}%` }}/></div><div className="demo-job-stages">{["Prepare", "Validate", "Process", "Finish"].map((stage, index) => <span className={recordingRun.step > index ? "complete" : ""} key={stage}>{recordingRun.step > index ? "✓ " : ""}{stage}</span>)}</div><small>Local example only · no microphone or backend action</small></section>}
  </>;
}

function AssistantWorkspace({ tab, data, query }: { tab: string; data?: Record<string, JsonRecord>; query: { isLoading: boolean; error: unknown } }) {
  const queryClient = useQueryClient();
  const [title, setTitle] = useState("");
  const [priority, setPriority] = useState("normal");
  const [dueAt, setDueAt] = useState("");
  const [goalId, setGoalId] = useState("");
  const [editingTaskId, setEditingTaskId] = useState("");
  const [showGoalForm, setShowGoalForm] = useState(false);
  const [editingGoalId, setEditingGoalId] = useState("");
  const [goalTitle, setGoalTitle] = useState("");
  const [goalDescription, setGoalDescription] = useState("");
  const [goalHorizon, setGoalHorizon] = useState("short");
  const [feedback, setFeedback] = useState("");
  const tasksResponse = data?.tasks ?? {};
  const brief = data?.brief ?? {};
  const tasks = (Array.isArray(tasksResponse.tasks) ? tasksResponse.tasks : Array.isArray(brief.open_tasks) ? brief.open_tasks : [])
    .filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item));
  const goalsResponse = data?.goals ?? {};
  const goals = (Array.isArray(goalsResponse.goals) ? goalsResponse.goals : Array.isArray(brief.active_goals) ? brief.active_goals : [])
    .filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item));
  const recommended = Array.isArray(brief.recommended_focus) ? brief.recommended_focus.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const refresh = () => queryClient.invalidateQueries({ queryKey: ["advanced-screen", "assistant"] });
  const createTask = useMutation({
    mutationFn: () => editingTaskId ? apiPatch(`/api/assistant/tasks/${encodeURIComponent(editingTaskId)}?project=default`, {
      title: title.trim(), priority, due_at: dueAt ? new Date(dueAt).toISOString() : "", goal_id: goalId,
    }) : apiPost("/api/assistant/tasks", {
      project: "default", title: title.trim(), priority,
      due_at: dueAt ? new Date(dueAt).toISOString() : "", goal_id: goalId,
    }),
    onSuccess: async (result) => {
      const task = result.task as JsonRecord | undefined;
      const wasEditing = Boolean(editingTaskId);
      setEditingTaskId(""); setTitle(""); setDueAt(""); setGoalId("");
      setFeedback(`${wasEditing ? "Task updated" : "Task saved"} by the backend${task?.id ? ` (${String(task.id)})` : ""}.`);
      await refresh();
    },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const deleteTask = useMutation({
    mutationFn: (taskId: string) => apiDelete(`/api/assistant/tasks/${encodeURIComponent(taskId)}?project=default`),
    onSuccess: async () => { setFeedback("Task deleted by the backend."); await refresh(); },
    onError: error => setFeedback(apiConnectionMessage(error)),
  });
  const saveGoal = useMutation({
    mutationFn: () => editingGoalId
      ? apiPatch(`/api/assistant/goals/${encodeURIComponent(editingGoalId)}?project=default`, { title: goalTitle.trim(), description: goalDescription.trim(), horizon: goalHorizon })
      : apiPost("/api/assistant/goals", { project: "default", title: goalTitle.trim(), description: goalDescription.trim(), horizon: goalHorizon }),
    onSuccess: async result => { const wasEditing = Boolean(editingGoalId); setShowGoalForm(false); setEditingGoalId(""); setGoalTitle(""); setGoalDescription(""); setFeedback(`${wasEditing ? "Goal updated" : "Goal created"} by the backend${result.goal && typeof result.goal === "object" && (result.goal as JsonRecord).id ? ` (${String((result.goal as JsonRecord).id)})` : ""}.`); await refresh(); },
    onError: error => setFeedback(apiConnectionMessage(error)),
  });
  const deleteGoal = useMutation({
    mutationFn: (goalIdValue: string) => apiDelete(`/api/assistant/goals/${encodeURIComponent(goalIdValue)}?project=default`),
    onSuccess: async () => { setFeedback("Goal deleted by the backend. Linked tasks remain and are unlinked."); await refresh(); },
    onError: error => setFeedback(apiConnectionMessage(error)),
  });
  const completeTask = useMutation({
    mutationFn: (taskId: string) => apiPost("/api/assistant/tasks/complete", { project: "default", task_id: taskId }),
    onSuccess: async (result) => {
      const task = result.task as JsonRecord | undefined;
      setFeedback(`Backend task status: ${String(task?.status ?? "response received")}.`);
      await refresh();
    },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const fail = query.error ? apiConnectionMessage(query.error) : "";
  const taskForm = <form className="p-record-sections" onSubmit={(event) => { event.preventDefault(); setFeedback(""); createTask.mutate(); }}>
    <label className="demo-form-field">{editingTaskId ? "Edit task" : "Task"}<input className="input" value={title} onChange={(event) => setTitle(event.target.value)} maxLength={240} required /></label>
    <div className="row">
      <label className="demo-form-field">Priority<select className="select" value={priority} onChange={(event) => setPriority(event.target.value)}><option value="normal">Normal</option><option value="low">Low</option><option value="high">High</option><option value="urgent">Urgent</option></select></label>
      <label className="demo-form-field">Due date<input className="input" type="datetime-local" value={dueAt} onChange={(event) => setDueAt(event.target.value)} /></label>
    </div>
    <label className="demo-form-field">Related goal<select className="select" value={goalId} onChange={(event) => setGoalId(event.target.value)}><option value="">No goal</option>{goals.map((goal, index) => <option key={String(goal.id ?? index)} value={String(goal.id ?? "")}>{String(goal.title ?? goal.id ?? "Goal")}</option>)}</select></label>
    {!goals.length && <p className="caption">No active goals are available to link. Create a goal in the Goals tab first.</p>}
    <div className="row">{editingTaskId && <button className="btn" type="button" onClick={() => { setEditingTaskId(""); setTitle(""); setDueAt(""); setGoalId(""); }}>Cancel edit</button>}<button className="btn primary" type="submit" disabled={!title.trim() || createTask.isPending}>{createTask.isPending ? "Saving task…" : editingTaskId ? "Save changes" : "Add task"}</button></div>
  </form>;
  const taskList = (records: JsonRecord[], showComplete: boolean) => records.length ? <div className="p-record-sections">{records.map((task, index) => {
    const id = String(task.id ?? "");
    const linkedGoal = goals.find((goal) => String(goal.id ?? "") === String(task.goal_id ?? ""));
    return <article className="p-record-section" key={id || index}><div className="row between"><strong>{String(task.title ?? "Untitled task")}</strong><span className="p-fixture">{String(task.status ?? "status unavailable")}</span></div><p className="caption">{String(task.priority ?? "priority unreported")} priority{task.due_at ? ` · due ${String(task.due_at)}` : " · no due date"}{linkedGoal ? ` · goal: ${String(linkedGoal.title ?? linkedGoal.id)}` : task.goal_id ? ` · goal ${String(task.goal_id)}` : ""}</p><div className="row">{showComplete && task.status !== "completed" && id && <button type="button" className="btn" disabled={completeTask.isPending} onClick={() => { setFeedback(""); completeTask.mutate(id); }}>{completeTask.isPending ? "Updating…" : "Mark complete"}</button>}{id && <button type="button" className="btn" onClick={() => { setEditingTaskId(id); setTitle(String(task.title ?? "")); setPriority(String(task.priority ?? "normal")); setDueAt(task.due_at ? new Date(String(task.due_at)).toISOString().slice(0, 16) : ""); setGoalId(String(task.goal_id ?? "")); window.scrollTo({ top: 0, behavior: "smooth" }); }}>Edit</button>}{id && <button type="button" className="btn" disabled={deleteTask.isPending} onClick={() => { if (window.confirm(`Delete task “${String(task.title ?? "Untitled task") }”?`)) deleteTask.mutate(id); }}>Delete</button>}</div></article>;
  })}</div> : <EmptyState icon={CheckCircle2} title="No tasks returned" detail="The backend has not returned any assistant tasks for this project." />;

  if (tab === "Goals") return <div className="p-record-sections">{fail && <section className="p-card" role="alert"><p>{fail}</p></section>}<section className="p-card"><div className="p-card-heading"><h3>Goals</h3><span className="p-fixture">{goals.length} backend records</span><button className="btn primary sm" type="button" onClick={() => { setEditingGoalId(""); setGoalTitle(""); setGoalDescription(""); setGoalHorizon("short"); setShowGoalForm(true); }}>Add goal</button></div>{query.isLoading ? <p className="caption" role="status">Loading goals…</p> : goals.length ? <div className="p-record-sections">{goals.map((goal, index) => { const id = String(goal.id ?? ""); return <article className="p-record-section" key={id || index}><div className="row between"><strong>{String(goal.title ?? "Untitled goal")}</strong><span className="p-fixture">{String(goal.status ?? "status unavailable")}</span></div><p className="caption">{String(goal.horizon ?? "horizon not reported")}{goal.description ? ` · ${String(goal.description)}` : ""}</p><div className="row">{id && <button className="btn" type="button" onClick={() => { setEditingGoalId(id); setGoalTitle(String(goal.title ?? "")); setGoalDescription(String(goal.description ?? "")); setGoalHorizon(String(goal.horizon ?? "short")); setShowGoalForm(true); }}>Edit</button>}{id && <button className="btn" type="button" disabled={deleteGoal.isPending} onClick={() => { if (window.confirm(`Delete goal “${String(goal.title ?? "Untitled goal") }”? Linked tasks will remain unlinked.`)) deleteGoal.mutate(id); }}>Delete</button>}</div></article>; })}</div> : <EmptyState icon={CheckCircle2} title="No goals returned" detail="Create a goal to organize related tasks." />}{feedback && <p role={saveGoal.isError || deleteGoal.isError ? "alert" : "status"} className="caption">{feedback}</p>}</section><Dialog.Root open={showGoalForm} onOpenChange={setShowGoalForm}><Dialog.Portal><Dialog.Overlay/><Dialog.Content className="advanced-integration-dialog"><div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">{editingGoalId ? "Edit goal" : "New goal"}</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close goal form"><X size={16} aria-hidden="true"/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">Changes are saved to the current project.</Dialog.Description><form onSubmit={event => { event.preventDefault(); setFeedback(""); saveGoal.mutate(); }}><label className="demo-form-field">Name<input className="input" value={goalTitle} onChange={event => setGoalTitle(event.target.value)} required maxLength={160}/></label><label className="demo-form-field">Description<textarea className="textarea" rows={3} value={goalDescription} onChange={event => setGoalDescription(event.target.value)} maxLength={500}/></label><label className="demo-form-field">Horizon<select className="select" value={goalHorizon} onChange={event => setGoalHorizon(event.target.value)}><option value="short">Short term</option><option value="medium">Medium term</option><option value="long">Long term</option></select></label><div className="advanced-integration-actions"><button className="btn" type="button" onClick={() => setShowGoalForm(false)}>Cancel</button><button className="btn primary" type="submit" disabled={!goalTitle.trim() || saveGoal.isPending}>{saveGoal.isPending ? "Saving…" : editingGoalId ? "Save changes" : "Create goal"}</button></div></form></Dialog.Content></Dialog.Portal></Dialog.Root></div>;

  if (tab === "Daily brief") return <section className="p-card"><div className="p-card-heading"><h3>Daily brief</h3><CalendarClock className="icon sm" aria-hidden="true" /></div>{query.isLoading && <p role="status" className="caption">Loading the backend daily brief…</p>}{fail && <p role="alert" className="caption">{fail}</p>}{!query.isLoading && !fail && <>
    <p className="caption">Project: {String(brief.project ?? "not reported")}</p><p className="caption">{String(brief.priority_count ?? 0)} high or urgent open task(s)</p>
    <h4>Recommended focus</h4>{taskList(recommended, false)}
    <h4 className="section-space">Open tasks</h4>{taskList(Array.isArray(brief.open_tasks) ? brief.open_tasks.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [], false)}
    <h4 className="section-space">Active goals</h4>{goals.length ? <div className="p-record-sections">{goals.map((goal, index) => <article className="p-record-section" key={String(goal.id ?? index)}><strong>{String(goal.title ?? "Goal")}</strong><p className="caption">{String(goal.status ?? "status unavailable")} · {String(goal.horizon ?? "horizon unreported")}{goal.description ? ` · ${String(goal.description)}` : ""}</p></article>)}</div> : <p className="caption">No active goals returned.</p>}
    <p className="caption section-space">Productivity coaching and meeting briefs are not enabled here: no user-facing workflow is included in this workspace slice.</p>
  </>}</section>;
  if (tab === "Today") {
    const activityEvents = (Array.isArray((data?.activity as { events?: unknown })?.events) ? (data?.activity as { events: JsonRecord[] }).events : []) as JsonRecord[];
    return <div className="p-record-sections">
      {fail && <section className="p-card" role="alert"><p>{fail}</p></section>}
      <section className="p-card">
        <div className="p-card-heading"><h3>Today</h3><span className="p-fixture">Backend daily brief</span></div>
        {query.isLoading ? <p role="status" className="caption">Loading current tasks and priorities…</p> : <>
          <p className="caption">{String(brief.priority_count ?? 0)} high or urgent open task(s) · {goals.length} active goal(s)</p>
          <h4>Recommended focus</h4>{taskList(recommended, true)}
        </>}
      </section>
      <section className="p-card">
        <div className="p-card-heading"><h3>Add a task</h3></div>{taskForm}
      </section>
      {activityEvents.length > 0 && <section className="p-card">
        <div className="p-card-heading"><h3>Recent activity</h3><span className="p-fixture">{activityEvents.length} audit events</span></div>
        <div className="p-list">{activityEvents.slice(0, 10).map((event, idx) => <div className="p-list-row" key={String(event.id ?? idx)}>
          <span className="status-dot" aria-hidden="true"/>
          <div className="flex1">
            <strong>{String(event.action ?? event.name ?? "Event")}</strong>
            <small className="muted">{String(event.scope ?? event.route ?? "")} · {String(event.created_at ?? event.time ?? "")}</small>
          </div>
        </div>)}</div>
      </section>}
      {feedback && <p role={createTask.isError || completeTask.isError ? "alert" : "status"} className="caption">{feedback}</p>}
    </div>;
  }
  return <div className="p-record-sections">{fail && <section className="p-card" role="alert"><p>{fail}</p></section>}<section className="p-card"><div className="p-card-heading"><h3>Tasks</h3><span className="p-fixture">{tasks.length} backend records</span></div>{query.isLoading ? <p role="status" className="caption">Loading tasks…</p> : taskList(tasks, true)}</section><section className="p-card"><div className="p-card-heading"><h3>Add a task</h3></div>{taskForm}</section>{feedback && <p role={createTask.isError || completeTask.isError ? "alert" : "status"} className="caption">{feedback}</p>}</div>;
}

function AssistantDemoWorkspace({ tab, data, query, onTabChange }: { tab: string; data?: Record<string, JsonRecord>; query: { isLoading: boolean; error: unknown }; onTabChange: (tab: string) => void }) {
  const client = useQueryClient();
  const [search, setSearch] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [showGoalForm, setShowGoalForm] = useState(false);
  const [goalTitle, setGoalTitle] = useState("");
  const [goalDescription, setGoalDescription] = useState("");
  const [goalProgress, setGoalProgress] = useState("65");
  const [editingGoalId, setEditingGoalId] = useState("");
  const [title, setTitle] = useState("");
  const [editingTaskId, setEditingTaskId] = useState("");
  const [priority, setPriority] = useState("Normal");
  const [dueAt, setDueAt] = useState("");
  const [feedback, setFeedback] = useState("");
  const [error, setError] = useState("");
  const [detailsId, setDetailsId] = useState("");
  const tasks = Array.isArray(data?.tasks?.tasks) ? data.tasks.tasks.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const goals = Array.isArray(data?.goals?.goals) ? data.goals.goals.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const brief = data?.brief ?? {};
  const focusTasks = Array.isArray(brief.recommended_focus) ? brief.recommended_focus.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : tasks;
  const filtered = tasks.filter((item) => `${item.title ?? ""} ${item.description ?? ""} ${item.status ?? ""}`.toLowerCase().includes(search.trim().toLowerCase()));
  const refresh = () => client.invalidateQueries({ queryKey: ["advanced-screen", "assistant"] });
  const createTask = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault(); if (!title.trim()) return;
    setError(""); setFeedback("");
    try {
      if (editingTaskId) {
        // Canonical fixtures call this field `name`; the API-facing view model calls it
        // `title`. Keep both in sync so the adapter cannot map the stale fixture name
        // over the edited title on the next query refresh.
        await demoProvider.updateCollection<JsonRecord>("collections.advanced.assistant.tasks", current => current.map(task => String(task.id ?? "") === editingTaskId ? { ...task, name: title.trim(), title: title.trim(), priority, due_at: dueAt } : task));
        await demoProvider.recordActivity("assistant", { id: `assistant-task-edit-${globalThis.crypto?.randomUUID?.() ?? Date.now()}`, name: "Task edited", item: title.trim(), route: "assistant", status: "Saved locally · demo", time: new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }), recorded_at: new Date().toISOString() });
      } else await advancedDemoAction("/api/assistant/tasks", { title: title.trim(), priority, due_at: dueAt || "", status: "Todo", project: "Personal Workspace" });
      setTitle(""); setDueAt(""); setPriority("Normal"); setEditingTaskId(""); setShowForm(false); setFeedback(editingTaskId ? "Task updated in this browser’s demo data." : "Task saved to this browser’s demo data."); await refresh();
    } catch (cause) { setError(apiConnectionMessage(cause)); }
  };
  const createGoal = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!goalTitle.trim()) return;
    setError(""); setFeedback("");
    try {
      const progress = Math.min(100, Math.max(0, Number(goalProgress) || 0));
      if (editingGoalId) {
        await demoProvider.updateCollection<JsonRecord>("collections.advanced.assistant.goals", current => current.map(goal => String(goal.id ?? "") === editingGoalId ? { ...goal, title: goalTitle.trim(), name: goalTitle.trim(), description: goalDescription.trim(), progress, status: "In progress" } : goal));
        const eventId = globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
        await demoProvider.recordActivity("assistant", { id: `assistant-activity-${eventId}`, name: "Goal edited", item: goalTitle.trim(), route: "assistant", status: "Saved locally · demo", time: new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }), recorded_at: new Date().toISOString() });
        setFeedback("Goal updated in this browser’s demo data.");
      } else {
        await advancedDemoAction("/api/assistant/goals", { title: goalTitle.trim(), description: goalDescription.trim(), progress, status: "In progress" });
        setFeedback("Goal saved to this browser’s demo data.");
      }
      setGoalTitle(""); setGoalDescription(""); setGoalProgress("0"); setEditingGoalId(""); setShowGoalForm(false); await refresh();
    } catch (cause) { setError(apiConnectionMessage(cause)); }
  };
  const editGoal = (goal: JsonRecord) => { setEditingGoalId(String(goal.id ?? "")); setGoalTitle(String(goal.title ?? goal.name ?? "")); setGoalDescription(String(goal.description ?? "")); setGoalProgress(String(goal.progress ?? 0)); setShowGoalForm(true); };
  const editTask = (task: JsonRecord) => { setEditingTaskId(String(task.id ?? "")); setTitle(String(task.title ?? "")); setPriority(String(task.priority ?? "Normal")); setDueAt(String(task.due_at ?? "")); setDetailsId(""); setShowForm(true); document.getElementById("assistant-demo-tasks")?.scrollIntoView({ behavior: "smooth", block: "center" }); };
  const deleteTask = async (task: JsonRecord) => { const id = String(task.id ?? ""); if (!id || !window.confirm(`Delete “${String(task.title ?? "this task")}” from local demo data?`)) return; await demoProvider.updateCollection<JsonRecord>("collections.advanced.assistant.tasks", current => current.filter(item => String(item.id ?? "") !== id)); await demoProvider.recordActivity("assistant", { id: `assistant-task-delete-${globalThis.crypto?.randomUUID?.() ?? Date.now()}`, name: "Task deleted", item: String(task.title ?? "Task"), route: "assistant", status: "Deleted locally · demo", time: new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }), recorded_at: new Date().toISOString() }); setFeedback("Task deleted from this browser’s demo data."); setDetailsId(""); await refresh(); };
  const deleteGoal = async (goal: JsonRecord) => { const id = String(goal.id ?? ""); if (!id || !window.confirm(`Delete “${String(goal.title ?? "this goal")}” from local demo data?`)) return; await demoProvider.updateCollection<JsonRecord>("collections.advanced.assistant.goals", current => current.filter(item => String(item.id ?? "") !== id)); await demoProvider.recordActivity("assistant", { id: `assistant-goal-delete-${globalThis.crypto?.randomUUID?.() ?? Date.now()}`, name: "Goal deleted", item: String(goal.title ?? "Goal"), route: "assistant", status: "Deleted locally · demo", time: new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }), recorded_at: new Date().toISOString() }); setFeedback("Goal deleted from this browser’s demo data."); setDetailsId(""); await refresh(); };
  const complete = async (task: JsonRecord) => {
    const id = String(task.id ?? ""); if (!id) return;
    setError(""); setFeedback("");
    try { await advancedDemoAction("/api/assistant/tasks/complete", { task_id: id }); setFeedback("Task updated in this browser’s demo data."); await refresh(); }
    catch (cause) { setError(apiConnectionMessage(cause)); }
  };
  const taskRows = (records: JsonRecord[], allowComplete = true) => <div className="assistant-demo-list">{records.map((task, index) => {
    const id = String(task.id ?? index); const state = String(task.status ?? "Todo"); const done = ["done", "completed"].includes(state.toLowerCase());
    return <article className="assistant-demo-row" key={id}><span className={`assistant-demo-check${done ? " is-done" : ""}`}><CheckCircle2 className="icon sm" aria-hidden="true" /></span><div className="assistant-demo-copy"><strong>{String(task.title ?? "Untitled task")}</strong><small>{String(task.description ?? "")}</small></div><span className={`tag assistant-demo-state ${state.toLowerCase() === "in progress" ? "orange" : done ? "green" : ""}`}>{state}</span><button type="button" className="icon-button" aria-label={done ? `Task details for ${String(task.title)}` : `Mark ${String(task.title)} complete`} onClick={() => allowComplete && !done ? void complete(task) : setDetailsId((current) => current === id ? "" : id)}><SlidersHorizontal className="icon sm" aria-hidden="true" /></button><button type="button" className="icon-button" aria-label={`More options for ${String(task.title)}`} onClick={() => setDetailsId((current) => current === id ? "" : id)}>···</button>{detailsId === id && <div className="assistant-demo-detail"><strong>{String(task.title ?? "Task details")}</strong><span>{String(task.priority ?? "Normal")} priority · {String(task.due_at ?? "No due date")}</span><p>{String(task.description ?? "No additional notes.")}</p><div className="row"><button type="button" className="btn sm" onClick={() => editTask(task)}>Edit</button><button type="button" className="btn sm" onClick={() => void deleteTask(task)}>Delete</button></div></div>}</article>;
  })}</div>;
  if (tab === "Today") return <>
    <section className="demo-feature-hero assistant-demo-hero"><div className="demo-feature-symbol"><CheckCircle2 className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">Today · Workspace</span><h2>Stay on top of your work</h2><p>Manage tasks, goals and the daily plan.</p></div></section>
    <div className="demo-feature-split assistant-demo-split"><section className="p-focus-panel assistant-demo-focus"><div className="p-eyebrow">Today’s focus</div><h2>Make progress on the important things.</h2><p>Three priorities are prepared for your next work session.</p><div className="row p-focus-actions"><button className="btn primary" type="button" onClick={() => { setShowForm(true); document.getElementById("assistant-demo-tasks")?.scrollIntoView({ behavior: "smooth", block: "center" }); }}><Plus className="icon sm" aria-hidden="true" />Add task</button><button className="btn soft" type="button" onClick={() => onTabChange("Daily brief")}><Sparkles className="icon sm" aria-hidden="true" />Generate brief</button></div></section>
    <section className="p-card p-tight assistant-demo-task-card" id="assistant-demo-tasks"><div className="row assistant-demo-toolbar"><label className="search-field flex1"><SearchIcon className="icon sm" aria-hidden="true" /><input aria-label="Search tasks" placeholder="Search tasks…" value={search} onChange={(event) => setSearch(event.target.value)} /></label><span className="caption">{filtered.length} items</span><button className="btn primary" type="button" onClick={() => setShowForm((current) => !current)}><Plus className="icon sm" aria-hidden="true" />Add</button></div>
      {showForm && <form className="assistant-demo-form" onSubmit={(event) => void createTask(event)}><label className="demo-form-field">{editingTaskId ? "Edit task" : "Task"}<input className="input" value={title} onChange={(event) => setTitle(event.target.value)} maxLength={240} required /></label><label className="demo-form-field">Priority<select className="select" value={priority} onChange={(event) => setPriority(event.target.value)}>{["Low", "Normal", "High", "Urgent"].map((item) => <option key={item}>{item}</option>)}</select></label><label className="demo-form-field">Due date<input className="input" type="date" value={dueAt} onChange={(event) => setDueAt(event.target.value)} /></label><div className="row">{editingTaskId && <button className="btn" type="button" onClick={() => { setEditingTaskId(""); setTitle(""); }}>Cancel edit</button>}<button className="btn primary" disabled={!title.trim()}>{editingTaskId ? "Save changes" : "Save locally"}</button><button className="btn" type="button" onClick={() => { setShowForm(false); setEditingTaskId(""); }}>Cancel</button></div></form>}
      {query.isLoading ? <p className="caption" role="status">Loading the local task list…</p> : error ? <p className="caption" role="alert">{error}</p> : filtered.length ? taskRows(filtered) : <EmptyState icon={CheckCircle2} title={tasks.length ? "No matching tasks" : "No tasks yet"} detail="Add a task to keep your plan up to date." />}
      {feedback && <p className="caption" role="status">{feedback}</p>}
    </section></div>
  </>;
  if (tab === "Tasks") return <><section className="demo-feature-hero assistant-demo-hero"><div className="demo-feature-symbol"><CheckCircle2 className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">Tasks · Workspace</span><h2>Stay on top of your work</h2><p>Manage tasks, goals and the daily plan.</p></div></section><section className="p-card p-tight assistant-demo-task-card"><div className="p-card-heading"><h3>Tasks · demo records</h3><span className="p-fixture">{tasks.length} items</span></div><div className="row assistant-demo-toolbar"><label className="search-field flex1"><SearchIcon className="icon sm" aria-hidden="true" /><input aria-label="Search tasks" placeholder="Search tasks…" value={search} onChange={(event) => setSearch(event.target.value)} /></label><button className="btn primary" type="button" onClick={() => { setEditingTaskId(""); setTitle(""); setShowForm(current => !current); }}><Plus className="icon sm" aria-hidden="true" />Add</button></div>{showForm && <form className="assistant-demo-form" onSubmit={(event) => void createTask(event)}><label className="demo-form-field">{editingTaskId ? "Edit task" : "Task"}<input className="input" value={title} onChange={(event) => setTitle(event.target.value)} required /></label><label className="demo-form-field">Priority<select className="select" value={priority} onChange={(event) => setPriority(event.target.value)}>{["Low", "Normal", "High", "Urgent"].map((item) => <option key={item}>{item}</option>)}</select></label><div className="row">{editingTaskId && <button className="btn" type="button" onClick={() => { setEditingTaskId(""); setTitle(""); }}>Cancel edit</button>}<button className="btn primary" type="submit" disabled={!title.trim()}>{editingTaskId ? "Save changes" : "Save locally"}</button></div></form>}{query.isLoading ? <p className="caption" role="status">Loading the local task list…</p> : query.error ? <p className="caption" role="alert">Tasks are unavailable: {apiConnectionMessage(query.error)}</p> : filtered.length ? taskRows(filtered) : <EmptyState icon={CheckCircle2} title={tasks.length ? "No matching tasks" : "No tasks yet"} detail={tasks.length ? "Try another search or clear the filter." : "Add a task to keep your plan up to date."} />}{feedback && <p className="caption" role="status">{feedback}</p>}{error && <p className="caption" role="alert">{error}</p>}</section></>;
  if (tab === "Goals") return <><section className="demo-feature-hero assistant-demo-hero"><div className="demo-feature-symbol"><CheckCircle2 className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">Goals · Workspace</span><h2>Stay on top of your work</h2><p>Manage tasks, goals and the daily plan.</p></div></section><section className="p-card p-tight assistant-demo-task-card"><div className="p-card-heading"><h3>Goals · demo records</h3></div><div className="row assistant-demo-toolbar"><label className="search-field flex1"><SearchIcon className="icon sm" aria-hidden="true" /><input aria-label="Search goals" placeholder="Search goals…" value={search} onChange={event => setSearch(event.target.value)} /></label><span className="caption">{goals.filter(goal => `${goal.title ?? ""} ${goal.description ?? ""}`.toLowerCase().includes(search.trim().toLowerCase())).length} items</span><button className="btn primary" type="button" onClick={() => { setEditingGoalId(""); setGoalTitle(""); setGoalDescription(""); setGoalProgress("65"); setShowGoalForm(true); }}><Plus className="icon sm" aria-hidden="true" />Add</button></div><Dialog.Root open={showGoalForm} onOpenChange={setShowGoalForm}><Dialog.Portal><Dialog.Overlay className="assistant-goal-overlay"/><Dialog.Content className="advanced-integration-dialog assistant-goal-dialog"><div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">{editingGoalId ? "Edit goal" : "New goal"}</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close goal form"><X size={16} aria-hidden="true"/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">Changes are saved locally in this interactive demonstration.</Dialog.Description><form onSubmit={event => void createGoal(event)}><label className="demo-form-field">Name<input className="input" value={goalTitle} onChange={event => setGoalTitle(event.target.value)} required maxLength={160} placeholder="Short, meaningful name"/></label><label className="demo-form-field">Description<textarea className="textarea" rows={3} value={goalDescription} onChange={event => setGoalDescription(event.target.value)} maxLength={500} placeholder="What is this for?"/></label><label className="demo-form-field">Progress<input className="input" type="number" min="0" max="100" value={goalProgress} onChange={event => setGoalProgress(event.target.value)} placeholder="65"/></label><div className="advanced-integration-actions"><button className="btn" type="button" onClick={() => setShowGoalForm(false)}>Cancel</button><button className="btn primary" type="submit">{editingGoalId ? "Save changes" : "Create item"}</button></div></form></Dialog.Content></Dialog.Portal></Dialog.Root><div className="assistant-demo-list">{goals.filter(goal => `${goal.title ?? ""} ${goal.description ?? ""}`.toLowerCase().includes(search.trim().toLowerCase())).map((goal,index)=><article className="assistant-demo-row" key={String(goal.id ?? index)}><span className="assistant-demo-check"><CheckCircle2 className="icon sm" aria-hidden="true"/></span><div className="assistant-demo-copy"><strong>{String(goal.title ?? "Goal")}</strong><small>{String(goal.description ?? "")}</small></div><span className="tag assistant-demo-state orange">{String(goal.status ?? "In progress")}</span><button type="button" className="icon-button" aria-label={`Edit ${String(goal.title ?? "goal")}`} onClick={() => editGoal(goal)}><SlidersHorizontal className="icon sm" aria-hidden="true"/></button><button type="button" className="icon-button" aria-label={`More actions for ${String(goal.title ?? "goal")}`} onClick={() => setDetailsId(current => current === String(goal.id ?? index) ? "" : String(goal.id ?? index))}><MoreHorizontal className="icon sm" aria-hidden="true"/></button>{detailsId === String(goal.id ?? index) && <div className="assistant-demo-detail"><strong>{String(goal.title ?? "Goal details")}</strong><span>{String(goal.status ?? "In progress")} · {String(goal.progress ?? 0)}% complete</span><p>{String(goal.description ?? "No additional notes.")}</p><button type="button" className="btn sm" onClick={() => void deleteGoal(goal)}>Delete</button></div>}</article>)}</div>{feedback && <p className="caption" role="status">{feedback}</p>}</section></>;
  return <><section className="demo-feature-hero assistant-demo-hero"><div className="demo-feature-symbol"><CheckCircle2 className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">Daily brief · Workspace</span><h2>Stay on top of your work</h2><p>Manage tasks, goals and the daily plan.</p></div></section><section className="p-card assistant-demo-brief"><div className="p-card-heading"><h3>Your daily brief</h3><CalendarClock className="icon sm" aria-hidden="true" /></div><div className="p-note"><Sparkles className="icon sm" aria-hidden="true"/><span>Your next priority is checking CodeQ permissions, followed by the research presentation.</span></div><div className="demo-brief">{focusTasks.slice(0,3).map((task,index)=><div className="demo-activity-row" key={String(task.id ?? index)}><span className="status-dot"/><div><strong>{String(task.title ?? "Task")}</strong><small>{String(task.due_at ?? "Today")}, {String(task.status ?? "Todo")}</small></div></div>)}</div><button className="btn primary" type="button" onClick={() => setFeedback("Demo brief refreshed locally.")}><RotateCcw className="icon sm" aria-hidden="true"/>Refresh demo brief</button>{feedback && <p className="caption" role="status">{feedback}</p>}</section></>;
}

function DemoCommunicationsWorkspace({ tab, data }: { tab: string; data?: Record<string, JsonRecord> }) {
  const client = useQueryClient();
  const [search, setSearch] = useState("");
  const [editorOpen, setEditorOpen] = useState(false);
  const [selected, setSelected] = useState<JsonRecord | null>(null);
  const [name, setName] = useState("");
  const [recipient, setRecipient] = useState("team@demo.example");
  const [description, setDescription] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const collection = tab === "Connected accounts" ? "accounts" : "messages";
  const allMessages = Array.isArray(data?.messages?.messages) ? data.messages.messages as JsonRecord[] : [];
  const items = tab === "Connected accounts" ? (Array.isArray(data?.accounts?.accounts) ? data.accounts.accounts as JsonRecord[] : []) : allMessages.filter(row => String(row.status ?? "").toLowerCase() === tab.slice(0, -1).toLowerCase() || String(row.status ?? "").toLowerCase() === tab.toLowerCase());
  const filtered = items.filter(row => `${String(row.name ?? row.subject ?? row.label ?? "")} ${String(row.description ?? row.body ?? row.status ?? "")}`.toLowerCase().includes(search.trim().toLowerCase()));
  const create = async (event: FormEvent, requestedStatus: "Draft" | "Sent" | "Connected") => {
    event.preventDefault();
    const cleanName = name.trim();
    if (!cleanName) { setError("Enter a name before saving."); return; }
    const target = tab === "Connected accounts" ? "accounts" : "messages";
    const record: JsonRecord = tab === "Connected accounts"
      ? { id: `demo-account-${Date.now().toString(36)}`, name: cleanName, description: description.trim() || "Example account · no external connection", status: requestedStatus, owner: "Workspace" }
      : { id: `demo-message-${Date.now().toString(36)}`, name: cleanName, description: `${recipient.trim()} · ${description.trim()}`, status: requestedStatus, owner: recipient.trim() || "team@demo.example", time: "Just now" };
    try {
      await demoProvider.updateCollection<JsonRecord>(`collections.advanced.communications.${target}`, current => [record, ...current]);
      setEditorOpen(false); setName(""); setDescription(""); setError(""); setNotice(requestedStatus === "Sent" ? "Example message marked sent locally; no email was delivered." : `${cleanName} saved in local demo data.`);
      await client.invalidateQueries({ queryKey: ["advanced-screen", "communications"] });
    } catch (cause) { setError(apiConnectionMessage(cause)); }
  };
  const openEditor = () => { setName(""); setDescription(""); setRecipient("team@demo.example"); setError(""); setEditorOpen(true); };
  return <>
    <section className="p-card">
      <div className="p-card-heading"><h3>{tab === "Connected accounts" ? "Connected accounts · demo records" : `${tab} · example mailbox`}</h3></div>
      <div className="row between communications-demo-mailbox-actions" style={{ marginBottom: 12 }}>
        <button type="button" className="btn primary" onClick={openEditor}><Plus className="icon sm" aria-hidden="true"/>{tab === "Connected accounts" ? "Connect account" : "Compose message"}</button>
        {tab !== "Connected accounts" && <button type="button" className="btn" onClick={() => setNotice("Demo inbox sync preview completed locally. No account was polled.")}><RotateCcw className="icon sm" aria-hidden="true"/>Sync demo inbox</button>}
      </div>
      <div className="demo-record-toolbar"><label className="search-field demo-record-search"><SearchIcon className="icon sm" aria-hidden="true"/><input aria-label={`Search ${collection}`} placeholder={`Search ${collection}…`} value={search} onChange={event => setSearch(event.target.value)}/></label><span className="caption">{filtered.length} item{filtered.length === 1 ? "" : "s"}</span><button type="button" className="btn primary sm" onClick={openEditor}><Plus className="icon sm" aria-hidden="true"/>Add</button></div>
      <div className="demo-records">{filtered.length ? filtered.map((row, index) => {
        const rowName = String(row.name ?? row.subject ?? row.label ?? "Example record"); const id = String(row.id ?? index);
        return <article className="demo-record" key={id}><button type="button" className="demo-record-main" onClick={() => setSelected(row)}><span className="square-icon sm blue"><Mail className="icon sm" aria-hidden="true"/></span><span className="demo-record-text"><strong>{rowName}</strong><small>{String(row.description ?? row.body ?? row.scope ?? "Example record")}</small></span></button><span className={`tag ${/sent|connected|active|ready/i.test(String(row.status ?? "")) ? "green" : /draft/i.test(String(row.status ?? "")) ? "orange" : ""}`}>{String(row.status ?? "Ready")}</span><button type="button" className="icon-button" aria-label={`Edit ${rowName}`} onClick={() => setSelected(row)}><SlidersHorizontal className="icon sm" aria-hidden="true"/></button><button type="button" className="icon-button" aria-label={`More actions for ${rowName}`} onClick={() => setSelected(row)}><MoreHorizontal className="icon sm" aria-hidden="true"/></button></article>;
      }) : <div className="p-empty"><SearchIcon className="icon lg" aria-hidden="true"/><h3>No matching items</h3><p className="muted small">Try a different search or create a sample record.</p></div>}</div>
      {notice && <p className="caption section-space" role="status">{notice}</p>}{error && <p className="caption" role="alert">{error}</p>}
    </section>
    <Dialog.Root open={editorOpen} onOpenChange={setEditorOpen}><Dialog.Portal><Dialog.Overlay/><Dialog.Content className="advanced-integration-dialog"><div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">{tab === "Connected accounts" ? "Connect demo account" : "Compose message"}</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close compose dialog"><X size={16} aria-hidden="true"/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">{tab === "Connected accounts" ? "Add a local example account. No external provider will be connected." : "Use demo@example.org addresses only. No email will actually be sent."}</Dialog.Description><form onSubmit={event => void create(event, tab === "Connected accounts" ? "Connected" : tab === "Sent" ? "Sent" : "Draft")}>
      {tab !== "Connected accounts" && <label className="demo-form-field">To<input className="input" type="email" value={recipient} onChange={event => setRecipient(event.target.value)} required/></label>}
      <label className="demo-form-field">{tab === "Connected accounts" ? "Account name" : "Subject"}<input className="input" value={name} onChange={event => setName(event.target.value)} required maxLength={120}/></label>
      <label className="demo-form-field">{tab === "Connected accounts" ? "Description" : "Message"}<textarea className="textarea" rows={4} value={description} onChange={event => setDescription(event.target.value)} required maxLength={1200}/></label>
      {error && <p className="caption" role="alert">{error}</p>}<div className="advanced-integration-actions"><button className="btn" type="button" onClick={() => setEditorOpen(false)}>Cancel</button>{tab !== "Connected accounts" && tab === "Drafts" && <button className="btn" type="button" onClick={event => void create(event as unknown as FormEvent, "Draft")}>Save draft</button>}<button className="btn primary" type="submit">{tab === "Connected accounts" ? "Save example" : tab === "Sent" ? "Send in demo" : "Save draft"}</button></div>
      </form></Dialog.Content></Dialog.Portal></Dialog.Root>
    <Dialog.Root open={Boolean(selected)} onOpenChange={open => { if (!open) setSelected(null); }}><Dialog.Portal><Dialog.Overlay/><Dialog.Content className="advanced-integration-dialog"><div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">{String(selected?.name ?? selected?.subject ?? "Message details")}</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close message details"><X size={16} aria-hidden="true"/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">{String(selected?.description ?? "Local example record")}</Dialog.Description><div className="p-note"><Info className="icon sm" aria-hidden="true"/><span>This is local demo data. No message was sent and no external account was contacted.</span></div><div className="advanced-integration-actions"><Dialog.Close asChild><button className="btn" type="button">Close</button></Dialog.Close></div></Dialog.Content></Dialog.Portal></Dialog.Root>
  </>;
}

function ContactsPanel({ commContacts, data, search, onSearchChange, onFeedback, feedback, addIsPending, addIsError, onAdd }: { commContacts: JsonRecord[]; data?: Record<string, JsonRecord>; search: string; onSearchChange: (v: string) => void; onFeedback: (v: string) => void; feedback: string; addIsPending: boolean; addIsError: boolean; onAdd: (name: string, email: string, notes: string) => void }) {
  const queryClient = useQueryClient();
  const [crmSearch, setCrmSearch] = useState("");
  const [editId, setEditId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [editEmail, setEditEmail] = useState("");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [notes, setNotes] = useState("");
  const crmContacts = Array.isArray(data?.crm_contacts?.contacts) ? data.crm_contacts.contacts.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const contactEmail = (contact: JsonRecord) => String(contact.email ?? (contact.handles && typeof contact.handles === "object" ? (contact.handles as JsonRecord).email ?? "" : ""));
  const deleteMutation = useMutation({
    mutationFn: (id: string) => apiDelete(`/api/contacts/${enc(id)}`),
    onSuccess: async () => { await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications"] }); },
    onError: (error) => onFeedback(apiConnectionMessage(error)),
  });
  const editMutation = useMutation({
    mutationFn: ({ id, body }: { id: string; body: JsonRecord }) => apiPatch(`/api/contacts/${enc(id)}`, body),
    onSuccess: async () => { setEditId(null); onFeedback("Contact updated."); await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications"] }); },
    onError: (error) => onFeedback(apiConnectionMessage(error)),
  });
  const exportMutation = useMutation({
    mutationFn: () => apiGet<JsonRecord>("/api/contacts/export"),
    onSuccess: (result) => {
      const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url; a.download = "contacts-export.json"; a.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    },
    onError: (error) => onFeedback(apiConnectionMessage(error)),
  });
  const filteredComm = commContacts.filter((c) => `${recordTitle(c)} ${JSON.stringify(c.handles ?? {})}`.toLowerCase().includes(search.toLowerCase()));
  const filteredCrm = crmContacts.filter((c) => `${recordTitle(c)} ${contactEmail(c)}`.toLowerCase().includes(crmSearch.toLowerCase()));
  return <section className="p-card">
    <div className="p-card-heading"><h3>Contacts</h3>
      <div className="row" style={{ gap: 8 }}>
        {crmContacts.length > 0 && <button className="btn sm soft" type="button" disabled={exportMutation.isPending} onClick={() => exportMutation.mutate()}><FileDown className="icon sm"/>Export CRM</button>}
        <span className="p-fixture">{commContacts.length + crmContacts.length} total records</span>
      </div>
    </div>
    {(commContacts.length > 0 || !crmContacts.length) && <>
      <h4 className="small" style={{ marginBottom: 6 }}>Communication contacts</h4>
      <label className="search-field"><SearchIcon className="icon sm" aria-hidden="true" /><input aria-label="Search communication contacts" value={search} onChange={(e) => onSearchChange(e.target.value)} placeholder="Filter contacts…" /></label>
      {filteredComm.length ? <div className="p-record-sections">{filteredComm.map((contact, index) => <article className="p-record-section" key={String(contact.id ?? index)}><h4>{recordTitle(contact)}</h4><p className="caption">{recordSummary(contact, "communications")}</p>{typeof contact.notes === "string" && contact.notes && <p className="caption">{contact.notes}</p>}</article>)}</div> : <p className="caption">No matching communication contacts.</p>}
    </>}
    {crmContacts.length > 0 && <>
      <h4 className="small" style={{ margin: "16px 0 6px" }}>CRM contacts</h4>
      <label className="search-field"><SearchIcon className="icon sm" aria-hidden="true" /><input aria-label="Search CRM contacts" value={crmSearch} onChange={(e) => setCrmSearch(e.target.value)} placeholder="Search CRM…" /></label>
      <div className="p-record-sections">{filteredCrm.map((contact, index) => {
        const id = String(contact.id ?? index);
        const isEditing = editId === id;
        return <article className="p-record-section" key={id}>
          {isEditing ? <>
            <input className="input" style={{ marginBottom: 4 }} value={editName} onChange={e => setEditName(e.target.value)} placeholder="Name" />
              <input className="input" style={{ marginBottom: 8 }} value={editEmail} onChange={e => setEditEmail(e.target.value)} placeholder="Email" type="email" />
            <div className="row" style={{ gap: 6 }}>
                <button className="btn sm primary" type="button" disabled={editMutation.isPending || !editName.trim()} onClick={() => editMutation.mutate({ id, body: { name: editName.trim(), handles: { ...(contact.handles && typeof contact.handles === "object" ? contact.handles as JsonRecord : {}), email: editEmail.trim() } } })}>Save</button>
              <button className="btn sm" type="button" onClick={() => setEditId(null)}>Cancel</button>
            </div>
          </> : <>
            <h4>{recordTitle(contact)}</h4>
            <p className="caption">{contactEmail(contact)} {contact.phone ? `· ${String(contact.phone)}` : ""}</p>
            <div className="row" style={{ gap: 6, marginTop: 4 }}>
              <button className="btn sm soft" type="button" onClick={() => { setEditId(id); setEditName(String(contact.name ?? "")); setEditEmail(contactEmail(contact)); }}>Edit</button>
              <button className="btn sm soft" type="button" disabled={deleteMutation.isPending} onClick={() => { if (window.confirm(`Delete CRM contact “${recordTitle(contact)}”? This removes it from the local contacts database.`)) deleteMutation.mutate(id); }}>Delete</button>
            </div>
          </>}
        </article>;
      })}</div>
    </>}
    <details className="p-record-section section-space"><summary>Add CRM contact</summary><form onSubmit={(event) => { event.preventDefault(); onAdd(name.trim(), email.trim(), notes.trim()); setName(""); setEmail(""); setNotes(""); }}><label className="demo-form-field">Name<input className="input" value={name} onChange={(e) => setName(e.target.value)} required /></label><label className="demo-form-field">Email<input className="input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} /></label><label className="demo-form-field">Notes<textarea className="input" rows={2} value={notes} onChange={(e) => setNotes(e.target.value)} /></label><button className="btn primary" disabled={addIsPending || !name.trim()}>{addIsPending ? "Saving…" : "Save contact"}</button></form></details>
    {feedback && <p role={addIsError || deleteMutation.isError || editMutation.isError ? "alert" : "status"} className="caption">{feedback}</p>}
  </section>;
}

function CommunicationRecords({ tab, data, demo = false }: { tab: string; data?: Record<string, JsonRecord>; demo?: boolean }) {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [title, setTitle] = useState("");
  const [startsAt, setStartsAt] = useState("");
  const [endsAt, setEndsAt] = useState("");
  const [allowExternal, setAllowExternal] = useState(false);
  const [feedback, setFeedback] = useState("");
  const contacts = Array.isArray(data?.contacts?.contacts) ? data.contacts.contacts.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const events = Array.isArray(data?.events?.events) ? data.events.events.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const sent = Array.isArray(data?.sent?.sent) ? data.sent.sent.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const scheduled = Array.isArray(data?.scheduled?.scheduled) ? data.scheduled.scheduled.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const add = useMutation({
    mutationFn: ({ path, body }: { path: string; body: JsonRecord }) => apiPost(path, body),
    onSuccess: async () => {
      setFeedback("Saved by the backend."); setTitle(""); setStartsAt(""); setEndsAt("");
      await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications"] });
    },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const sync = useMutation({
    mutationFn: () => apiPost("/api/communication/sync", { platform: "email", project: "default", allow_external: true }),
    onSuccess: async (result) => { setFeedback(`Backend sync result: ${String((result.run as JsonRecord | undefined)?.status ?? "response received")}.`); setAllowExternal(false); await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications"] }); },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });
  const process = useMutation({
    mutationFn: () => apiPost("/api/communication/scheduled/process", { project: "default", allow_external: true }),
    onSuccess: async (result) => { setFeedback(`Backend processed ${String(result.count ?? 0)} due scheduled send(s).`); setAllowExternal(false); await queryClient.invalidateQueries({ queryKey: ["advanced-screen", "communications"] }); },
    onError: (error) => setFeedback(apiConnectionMessage(error)),
  });

  if (demo && ["Sent", "Drafts", "Connected accounts"].includes(tab)) return <DemoCommunicationsWorkspace tab={tab} data={data} />;

  if (tab === "Contacts") return <ContactsPanel commContacts={contacts} data={data} search={search} onSearchChange={setSearch} onFeedback={setFeedback} feedback={feedback} addIsPending={add.isPending} addIsError={add.isError} onAdd={(name, email, notes) => add.mutate({ path: "/api/contacts", body: { name, handles: email ? { email } : {}, notes: notes || undefined, project: "default" } })} />;

  if (tab === "Calendar") return <section className="p-card">
    <div className="p-card-heading"><h3>Communication calendar</h3><span className="p-fixture">{events.length} backend records</span></div>
    {!events.length ? <EmptyState icon={CalendarDays} title="No communication events" detail="Events are loaded from the communications service." /> : <div className="p-record-sections">{events.map((event, index) => <article className="p-record-section" key={String(event.id ?? index)}><h4>{recordTitle(event)}</h4><p className="caption">{String(event.starts_at ?? "Time unavailable")} – {String(event.ends_at ?? "")}</p>{recordSummary(event, "communications") && <p className="caption">{recordSummary(event, "communications")}</p>}</article>)}</div>}
    <details className="p-record-section section-space"><summary>Add local calendar event</summary><form onSubmit={(event) => { event.preventDefault(); setFeedback(""); add.mutate({ path: "/api/communication/calendar", body: { title: title.trim(), starts_at: new Date(startsAt).toISOString(), ends_at: new Date(endsAt).toISOString(), attendees: [], project: "default", calendar_id: "local", source: "local" } }); }}><label className="demo-form-field">Title<input className="input" value={title} onChange={(event) => setTitle(event.target.value)} required /></label><label className="demo-form-field">Starts<input className="input" type="datetime-local" value={startsAt} onChange={(event) => setStartsAt(event.target.value)} required /></label><label className="demo-form-field">Ends<input className="input" type="datetime-local" value={endsAt} onChange={(event) => setEndsAt(event.target.value)} required /></label><button className="btn primary" disabled={add.isPending || !title.trim() || !startsAt || !endsAt || new Date(endsAt) < new Date(startsAt)}>{add.isPending ? "Saving…" : "Save event"}</button></form></details>
    {feedback && <p role={add.isError ? "alert" : "status"} className="caption">{feedback}</p>}
  </section>;

  if (tab === "Sent") return <section className="p-card"><div className="p-card-heading"><h3>Sent messages</h3><span className="p-fixture">{sent.length} backend records</span></div>{!sent.length ? <EmptyState icon={Mail} title="No sent messages" detail="The backend has no sent-message records for this project." /> : <div className="p-record-sections">{sent.map((message, index) => <article className="p-record-section" key={String(message.id ?? index)}><h4>{recordTitle(message)}</h4><p className="caption">To {Array.isArray(message.recipients) ? message.recipients.map(String).join(", ") : "recipient unavailable"} · {String(message.status ?? "status unavailable")} · {String(message.created_at ?? "")}</p>{typeof message.body === "string" && <p className="caption">{message.body}</p>}</article>)}</div>}</section>;

  if (tab === "Scheduled") return <section className="p-card"><div className="p-card-heading"><h3>Scheduled sends</h3><span className="p-fixture">{scheduled.length} backend records</span></div>{!scheduled.length ? <EmptyState icon={CalendarClock} title="No scheduled sends" detail="Schedule an approved draft from Drafts. Due sends are processed only after explicit confirmation." /> : <div className="p-record-sections">{scheduled.map((item, index) => <article className="p-record-section" key={String(item.id ?? index)}><h4>{String(item.draft_id ?? "Draft")}</h4><p className="caption">{String(item.status ?? "status unavailable")} · scheduled for {String(item.send_at ?? "time unavailable")}</p>{Boolean(item.result && typeof item.result === "object") && <p className="caption">{recordSummary(item.result as JsonRecord, "communications")}</p>}</article>)}</div>}
    <div className="p-record-section section-space"><label className="row"><input type="checkbox" checked={allowExternal} onChange={(event) => setAllowExternal(event.target.checked)} />I explicitly authorize processing due sends, which may deliver approved messages externally.</label><button className="btn primary" type="button" disabled={!allowExternal || process.isPending || !scheduled.some((item) => item.status === "scheduled" && Date.parse(String(item.send_at ?? "")) <= Date.now())} onClick={() => { setFeedback(""); process.mutate(); }}>{process.isPending ? "Processing due sends…" : "Process due sends"}</button><p className="caption">Only backend due items are processed. If the checkbox stays off, no send request is made.</p></div>{feedback && <p role={process.isError ? "alert" : "status"} className="caption">{feedback}</p>}</section>;

  if (tab === "Connected accounts") {
    const connectors = Array.isArray(data?.connectors?.connectors) ? data.connectors.connectors.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
    const accounts = Array.isArray(data?.accounts?.accounts) ? data.accounts.accounts.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
    const providers = Array.isArray(data?.providers?.providers) ? data.providers.providers.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
    const runs = Array.isArray(data?.sync_runs?.runs) ? data.sync_runs.runs.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
    const emailEnabled = connectors.some((item) => item.id === "email" && item.enabled === true);
    return <section className="p-card"><div className="p-card-heading"><h3>Connected accounts</h3><span className="p-fixture">{accounts.length} accounts · {connectors.length} connectors</span></div>
      {!accounts.length ? <p className="muted small">No OAuth accounts are configured. Account onboarding requires provider credentials and is not exposed in this panel.</p> : <div className="p-record-sections">{accounts.map((account, index) => <article className="p-record-section" key={String(account.id ?? index)}><h4>{String(account.label ?? account.provider ?? "Account")}</h4><p className="caption">{String(account.platform ?? "platform unavailable")} · {String(account.status ?? "status unavailable")} · token {account.token_present === true ? "present" : "not present"}</p></article>)}</div>}
      <div className="p-record-section section-space"><h4>Connector status</h4>{!connectors.length ? <p className="caption">No connector states returned.</p> : connectors.map((connector, index) => <p className="caption" key={String(connector.id ?? index)}>{String(connector.label ?? connector.platform ?? connector.id)} · {connector.enabled === true ? "enabled" : "disabled"} · capabilities {Array.isArray(connector.capabilities) ? connector.capabilities.map(String).join(", ") : "not reported"}</p>)}</div>
      {providers.length > 0 && <p className="caption">{providers.length} provider definitions are available. Configuration and OAuth authorization require a separate credential flow.</p>}
      <div className="p-record-section section-space"><h4>Email inbox sync</h4><p className="caption">External polling is available only when the backend email connector is enabled and configured. The backend reports the actual run state.</p><label className="row"><input type="checkbox" checked={allowExternal} onChange={(event) => setAllowExternal(event.target.checked)} />I explicitly authorize one external inbox poll now.</label><button className="btn primary" type="button" disabled={!emailEnabled || !allowExternal || sync.isPending} onClick={() => { setFeedback(""); sync.mutate(); }}>{sync.isPending ? "Polling provider…" : "Sync email inbox"}</button>{!emailEnabled && <p className="caption">Email connector is not enabled by the backend; sync is unavailable.</p>}</div>
      <div className="p-record-section section-space"><h4>Recent sync runs</h4>{!runs.length ? <p className="caption">No backend sync runs.</p> : runs.slice(0, 10).map((run, index) => <p className="caption" key={String(run.id ?? index)}>{String(run.platform ?? "connector")} · {String(run.status ?? "unknown")} · {String(run.started_at ?? "")}{run.error ? ` · ${String(run.error)}` : ""}</p>)}</div>
      {feedback && <p role={sync.isError ? "alert" : "status"} className="caption">{feedback}</p>}
    </section>;
  }
  return null;
}

function EngineDemoOverview({ data }: { data?: Record<string, JsonRecord> }) {
  const [benchmarkPreview, setBenchmarkPreview] = useState(false);
  const hardware = Array.isArray(data?.hardware?.hardware) ? data.hardware.hardware.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  return <>
    <section className="demo-feature-hero"><div className="demo-feature-symbol"><Cpu className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">Overview · Workspace</span><h2 style={{ fontWeight: 700 }}>Make resources work smarter</h2><p>Plan workloads with visible constraints.</p></div></section>
    <section className="p-card"><div className="p-card-heading"><h3>Compute overview</h3><span className="p-fixture">Illustrative demo data</span></div><div className="demo-stats-grid"><div><strong>12</strong><small>Logical CPU cores</small></div><div><strong>12 GB</strong><small>Example VRAM</small></div><div><strong>24 GB</strong><small>Memory available</small></div></div><button type="button" className="btn primary" onClick={() => setBenchmarkPreview(true)}><Activity className="icon sm" aria-hidden="true"/>Run benchmark</button>{benchmarkPreview && <p className="p-note section-space" role="status"><CircleHelp className="icon sm" aria-hidden="true"/>Illustrative benchmark preview only; no hardware workload ran and no performance was measured.</p>}<div className="section-space"><h4>Hardware availability</h4>{hardware.map((item, index) => <article className="p-item" key={String(item.id ?? index)}><div className="square-icon"><Cpu className="icon sm" aria-hidden="true"/></div><div className="p-item-copy"><strong>{String(item.name ?? "Hardware resource")}</strong><small>{String(item.description ?? item.memory ?? "Example device")}</small></div><span className="tag green">{String(item.status ?? "Example")}</span></article>)}</div></section>
  </>;
}

function EngineDemoWorkspace({ tab, data }: { tab: string; data?: Record<string, JsonRecord> }) {
  const [message, setMessage] = useState("");
  const items = tab === "Hardware" ? (data?.hardware?.hardware ?? []) as JsonRecord[] : tab === "Jobs" ? (data?.jobs?.jobs ?? []) as JsonRecord[] : (data?.plans?.plans ?? []) as JsonRecord[];
  return <>
    <section className="demo-feature-hero"><div className="demo-feature-symbol"><Cpu className="icon lg" aria-hidden="true"/></div><div><span className="p-eyebrow">{tab} · Workspace</span><h2 style={{ fontWeight: 700 }}>Make resources work smarter</h2><p>Plan workloads with visible constraints.</p></div></section>
    {tab === "Plans" && <section className="p-card"><div className="p-card-heading"><h3>Workload planner</h3></div><form className="demo-settings-form" onSubmit={event => { event.preventDefault(); setMessage("Execution plan simulated locally. No workload was started."); }}>
      <div className="p-form-grid"><label className="p-field">Workload<select className="select"><option>Local inference</option><option>Dataset processing</option><option>Embedding refresh</option></select></label><label className="p-field">Compute target<select className="select"><option>Auto (GPU → CPU)</option><option>CPU only</option><option>GPU only</option></select></label></div>
      <label className="p-field">Memory cap<select className="select"><option>8 GB</option><option>4 GB</option><option>16 GB</option></select></label><button className="btn primary" type="submit"><Play className="icon sm" aria-hidden="true"/>Simulate execution plan</button>{message && <p className="p-note section-space" role="status">{message}</p>}
    </form></section>}
    <DeveloperDemoRecords heading={`${tab} · demo records`} collection={tab.toLowerCase()} items={items} icon={Cpu}/>
  </>;
}

function EngineLiveWorkspace({ tab, data, query }: { tab: string; data?: Record<string, JsonRecord>; query: { isLoading: boolean; error: unknown } }) {
  const status = data?.status ?? {};
  const hardwareResponse = data?.hardware ?? {};
  const profile = (hardwareResponse.profile && typeof hardwareResponse.profile === "object" ? hardwareResponse.profile : status.hardware_profile) as JsonRecord | undefined;
  const devices = Array.isArray(profile?.devices) ? profile.devices.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const backends = Array.isArray(status.backends) ? status.backends.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const plans = Array.isArray(data?.plans?.plans) ? data.plans.plans.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const runs = Array.isArray(data?.runs?.runs) ? data.runs.runs.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
  const hardwareStatus = String(status.status ?? hardwareResponse.status ?? "Status unavailable");
  const formatMemory = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? `${value} GB` : "Not reported";
  const loadError = query.error ? apiConnectionMessage(query.error) : "";

  if (tab === "Overview") {
    const threads = profile?.cpu_threads;
    const availableRam = profile?.available_ram_gb;
    const acceleratorCount = devices.filter(device => String(device.kind ?? "").toLowerCase() !== "cpu").length;
    const inventory = status.inventory && typeof status.inventory === "object" ? status.inventory as JsonRecord : {};
    return <>
      <section className="p-engine-hero"><div><span className="p-eyebrow"><span className="p-eyebrow-dot" />Overview</span><h2>System capabilities</h2><p>Values below come from the connected adaptive engine. No performance estimate is shown unless the service reports one.</p></div><div className="p-engine-art"><Cpu className="icon xl" aria-hidden="true"/><span>{hardwareStatus}</span></div></section>
      {loadError && <p className="p-note" role="alert"><AlertCircle className="icon sm" aria-hidden="true"/>{loadError}</p>}
      {query.isLoading && <p className="p-note" role="status"><Activity className="icon sm" aria-hidden="true"/>Loading adaptive engine data…</p>}
      <section className="p-card"><div className="p-card-heading"><h3>Compute overview</h3><span className="p-fixture">Backend reported</span></div><div className="p-metrics-grid"><div className="p-stat"><Cpu className="p-stat-icon icon" aria-hidden="true"/><small>CPU threads</small><strong>{typeof threads === "number" ? threads : "—"}</strong><span>{typeof threads === "number" ? "Reported by hardware profile" : "Not reported"}</span></div><div className="p-stat"><Database className="p-stat-icon icon" aria-hidden="true"/><small>Available memory</small><strong>{formatMemory(availableRam)}</strong><span>{typeof availableRam === "number" ? "Reported by hardware profile" : "Not reported"}</span></div><div className="p-stat"><Network className="p-stat-icon icon" aria-hidden="true"/><small>Accelerators</small><strong>{profile?.devices ? acceleratorCount : "—"}</strong><span>{profile?.devices ? "Devices reported by hardware profile" : "Not reported"}</span></div></div>
        {Object.keys(inventory).length > 0 && <div className="demo-stats-grid section-space">{Object.entries(inventory).map(([key, value]) => <div key={key}><strong>{String(value)}</strong><small>{humanize(key)}</small></div>)}</div>}
        <div className="p-card-heading section-space"><h3>Hardware availability</h3></div>{typeof profile?.platform === "string" && profile.platform && <p className="caption">{profile.platform} · {formatMemory(profile.total_ram_gb)} total memory</p>}{devices.length ? devices.map((device, index) => <article className="p-item" key={String(device.id ?? index)}><div className="square-icon"><Cpu className="icon sm" aria-hidden="true"/></div><div className="p-item-copy"><strong>{String(device.name ?? device.id ?? "Hardware device")}</strong><small>{[device.kind, device.vendor, formatMemory(device.memory_gb), device.estimate_source].filter(value => value !== undefined && value !== "Not reported").map(String).join(" · ") || "No additional device details reported"}</small></div><span className={`tag ${device.available === true ? "green" : device.available === false ? "red" : ""}`}>{device.available === true ? "Available" : device.available === false ? "Unavailable" : "Status not reported"}</span></article>) : !query.isLoading && <div className="p-empty" role="status"><Cpu className="icon lg" aria-hidden="true"/><h3>No hardware devices reported</h3><p>The connected hardware profile did not include device records.</p></div>}
        {backends.length > 0 && <><div className="p-card-heading section-space"><h3>Compute backends</h3></div>{backends.map((backend, index) => <article className="p-item" key={String(backend.id ?? backend.name ?? index)}><div className="square-icon"><Network className="icon sm" aria-hidden="true"/></div><div className="p-item-copy"><strong>{String(backend.name ?? backend.id ?? "Backend")}</strong><small>{String(backend.status ?? backend.version ?? "Capabilities reported by service")}</small></div><span className="tag">{String(backend.available === true ? "Available" : backend.available === false ? "Unavailable" : "Reported")}</span></article>)}</>}
      </section>
    </>;
  }

  const title = tab === "Hardware" ? "Hardware availability" : tab === "Jobs" ? "Recent execution jobs" : "Execution plans";
  const records = tab === "Hardware" ? devices : tab === "Jobs" ? runs : plans;
  const RecordIcon = tab === "Jobs" ? Activity : tab === "Plans" ? Layers3 : Cpu;
  return <>
    <section className="p-engine-hero"><div><span className="p-eyebrow"><span className="p-eyebrow-dot" />{tab}</span><h2>{tab === "Jobs" ? "Execution jobs" : tab === "Plans" ? "Execution plans" : "System capabilities"}</h2><p>{tab === "Jobs" ? "Execution records returned by the adaptive engine." : tab === "Plans" ? "Saved workload plans returned by the adaptive engine." : "Hardware details reported by the adaptive engine."}</p></div><div className="p-engine-art"><Cpu className="icon xl" aria-hidden="true"/><span>{hardwareStatus}</span></div></section>
    {loadError && <p className="p-note" role="alert"><AlertCircle className="icon sm" aria-hidden="true"/>{loadError}</p>}
    {query.isLoading && <p className="p-note" role="status"><Activity className="icon sm" aria-hidden="true"/>Loading {tab.toLowerCase()}…</p>}
    <section className="p-card"><div className="p-card-heading"><h3>{title}</h3><span className="p-fixture">{records.length} backend records</span></div>
      {tab === "Hardware" && profile && <p className="caption">{String(profile.platform ?? "Platform not reported")} · {formatMemory(profile.total_ram_gb)} total memory · {typeof profile.cpu_threads === "number" ? `${profile.cpu_threads} CPU threads` : "CPU threads not reported"}</p>}
      {records.length ? records.map((record, index) => {
        const heading = String(record.name ?? record.task ?? record.label ?? record.id ?? `${tab.slice(0, -1)} ${index + 1}`);
        const meta = tab === "Hardware"
          ? [record.kind, record.vendor, formatMemory(record.memory_gb), record.estimate_source].filter(value => value !== undefined && value !== "Not reported").map(String).join(" · ")
          : tab === "Jobs"
            ? [record.mode, `Rounds ${String(record.rounds ?? "not reported")}`, record.approved === true ? "Approved" : record.approved === false ? "Not approved" : "Approval not reported", record.completed_at ? `Completed ${String(record.completed_at)}` : ""].filter(Boolean).join(" · ")
            : [record.status, record.model, record.device, record.updated_at].filter(value => value !== undefined && value !== "").map(String).join(" · ");
        const state = String(record.status ?? (record.available === true ? "Available" : record.available === false ? "Unavailable" : "Status not reported"));
        return <article className="p-item" key={String(record.id ?? index)}><div className="square-icon"><RecordIcon className="icon sm" aria-hidden="true"/></div><div className="p-item-copy"><strong>{heading}</strong><small>{meta || "No additional details reported"}</small>{tab === "Jobs" && typeof record.error === "string" && record.error && <small role="alert">{record.error}</small>}{tab === "Jobs" && typeof record.final_output === "string" && record.final_output && <small>{record.final_output}</small>}</div><span className={`tag ${["completed", "success", "available", "ready"].includes(state.toLowerCase()) ? "green" : ["failed", "error", "unavailable"].includes(state.toLowerCase()) ? "red" : ""}`}>{state}</span></article>;
      }) : !query.isLoading && <div className="p-empty" role="status"><FileClock className="icon lg" aria-hidden="true"/><h3>{tab === "Hardware" ? "No hardware devices reported" : tab === "Jobs" ? "No execution jobs returned" : "No execution plans returned"}</h3><p>{tab === "Hardware" ? "The connected hardware profile did not include device records." : tab === "Jobs" ? "The adaptive engine has no execution records for this project." : "The adaptive engine has no saved plans for this project."}</p></div>}
    </section>
  </>;
}

function FineTuneDemoWorkspace({ tab, data }: { tab: string; data?: Record<string, JsonRecord> }) {
  const [message, setMessage] = useState("");
  const [traceStartSignal, setTraceStartSignal] = useState(0);
  const [showTrainingTrace, setShowTrainingTrace] = useState(false);
  const [trainingPending, setTrainingPending] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedAdapter, setSelectedAdapter] = useState("LoRA · rank 16");
  const [selectedEpochs, setSelectedEpochs] = useState("3 epochs");
  const client = useQueryClient();
  const collections: Record<string, { viewKey: string; storageKey: string }> = {
    Datasets: { viewKey: "datasets", storageKey: "datasets" },
    Jobs: { viewKey: "jobs", storageKey: "jobs" },
    Checkpoints: { viewKey: "checkpoints", storageKey: "checkpoints" },
    Results: { viewKey: "adapters", storageKey: "results" },
  };
  const selectedCollection = collections[tab] ?? collections.Datasets;
  const items = (data?.[selectedCollection.viewKey]?.[selectedCollection.viewKey] ?? []) as JsonRecord[];
  useEffect(() => {
    if (selectedDataset) return;
    const firstDataset = ((data?.datasets?.datasets ?? []) as JsonRecord[])[0];
    if (firstDataset) setSelectedDataset(String(firstDataset.name ?? ""));
  }, [data, selectedDataset]);
  useEffect(() => {
    // A trace is a response to an explicit action, not a tab-entry animation.
    // Hide it when leaving Jobs so remounting the tab cannot replay the preview.
    if (tab !== "Jobs") setShowTrainingTrace(false);
  }, [tab]);
  const heading = tab === "Datasets" ? "Training datasets" : tab === "Jobs" ? "Fine-tuning runs" : tab === "Checkpoints" ? "Checkpoints · demo records" : "Results · demo records";
  const startTrainingPreview = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const values = new FormData(event.currentTarget);
    const dataset = String(values.get("dataset") ?? "").trim();
    if (!dataset) { setMessage("Choose a dataset before creating a training preview."); return; }
    setTrainingPending(true);
    try {
      const preview: JsonRecord = {
        id: `demo-finetune-preview-${Date.now().toString(36)}`,
        name: `${dataset} · ${selectedAdapter}`,
        description: `${selectedEpochs} · simulated preview only; no model was trained and no checkpoint was created.`,
        status: "Preview · demo",
        dataset,
        adapter: selectedAdapter,
        epochs: selectedEpochs,
      };
      await demoProvider.updateCollection<JsonRecord>("collections.advanced.finetune.jobs", current => [preview, ...current]);
      await demoProvider.recordActivity("finetune", { id: `activity-${preview.id}`, action: "Training preview saved", item: String(preview.name), status: "Simulated · demo", recorded_at: new Date().toISOString() });
      await client.invalidateQueries({ queryKey: ["advanced-screen", "finetune"] });
      setMessage("Training preview saved locally. No model was trained and no checkpoint was created.");
      setTraceStartSignal(value => value + 1);
      setShowTrainingTrace(true);
    } catch (cause) {
      setMessage(cause instanceof Error ? cause.message : "Could not save the local training preview.");
    } finally { setTrainingPending(false); }
  };
  return <>
    {tab === "Jobs" && showTrainingTrace && traceStartSignal > 0 && <ExecutionTrace route="finetune" startSignal={traceStartSignal}/>}
    <section className="demo-feature-hero"><div className="demo-feature-symbol"><Cpu className="icon lg" aria-hidden="true"/></div><div><span className="p-eyebrow">{tab} · Workspace</span><h2 style={{ fontWeight: 700 }}>Prepare, train and evaluate</h2><p>Version datasets and monitor the path to a checkpoint.</p></div></section>
    {tab === "Jobs" && <section className="p-card"><div className="p-card-heading"><h3>Fine-tuning runs</h3></div><form className="demo-settings-form" onSubmit={event => void startTrainingPreview(event)}>
      <div className="p-form-grid"><label className="p-field">Dataset<select className="select" name="dataset" required value={selectedDataset} onChange={event => setSelectedDataset(event.target.value)}><option value="">Choose a dataset</option>{((data?.datasets?.datasets ?? []) as JsonRecord[]).map((row, index) => <option key={String(row.id ?? index)} value={String(row.name ?? "Example dataset")}>{String(row.name ?? "Example dataset")}</option>)}</select></label><label className="p-field">Adapter<select className="select" value={selectedAdapter} onChange={event => setSelectedAdapter(event.target.value)}><option>LoRA · rank 16</option><option>LoRA · rank 8</option><option>QLoRA · 4-bit</option></select></label></div><label className="p-field">Epochs<select className="select" value={selectedEpochs} onChange={event => setSelectedEpochs(event.target.value)}><option>3 epochs</option><option>2 epochs</option><option>1 epoch</option></select></label><button type="submit" className="btn primary" disabled={trainingPending || !((data?.datasets?.datasets ?? []) as JsonRecord[]).length}><Play className="icon sm" aria-hidden="true"/>{trainingPending ? "Saving preview…" : "Simulate training"}</button>{message && <p className="p-note section-space" role="status">{message}</p>}
    </form></section>}
    <FineTuneDemoRecords key={selectedCollection.storageKey} heading={heading} collection={selectedCollection.storageKey} items={items}/>
    <FineTuneDemoActivity events={(data?.activity?.events ?? []) as JsonRecord[]} />
  </>;
}

function FineTuneDemoActivity({ events }: { events: JsonRecord[] }) {
  if (!events.length) return null;
  return <section className="p-card advanced-demo-activity" aria-label="Fine-tuning activity">
    <div className="p-card-heading"><h3>Recent activity</h3><span className="p-fixture">Local demo actions</span></div>
    <div className="p-record-sections">{events.slice(0, 8).map((event, index) => <article className="p-record-section" key={String(event.id ?? index)}>
      <h4>{String(event.action ?? "Demo action")}</h4><p className="caption">{String(event.item ?? "Fine-tuning record")} · {String(event.status ?? "Recorded locally")}{event.recorded_at ? ` · ${new Date(String(event.recorded_at)).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` : ""}</p>
    </article>)}</div>
  </section>;
}

function AssistantDemoActivity({ events }: { events: JsonRecord[] }) {
  const [open, setOpen] = useState(false);
  const eventName = (event: JsonRecord) => String(event.name ?? event.action ?? "Demo action");
  const eventTime = (event: JsonRecord) => String(event.time ?? "Recorded locally");
  const eventStatus = (event: JsonRecord) => String(event.status ?? "Saved locally · demo");
  return <section className="p-card" aria-label="Personal assistant recent activity">
    <div className="p-card-heading"><h3>Recent activity</h3><Dialog.Root open={open} onOpenChange={setOpen}><Dialog.Trigger asChild><button className="btn sm" type="button">All activity</button></Dialog.Trigger><Dialog.Portal><Dialog.Overlay/><Dialog.Content className="advanced-integration-dialog demo-large-modal">
      <div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">Demo activity</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close activity"><X size={16} aria-hidden="true"/></button></Dialog.Close></div>
      <Dialog.Description className="second-app-dialog-description">Actions completed in this browser, not from a real backend.</Dialog.Description>
      <div className="demo-event-scroll">{events.length ? events.map((event, index) => <div className="row" key={String(event.id ?? index)} style={{ alignItems: "flex-start", gap: 10, padding: "10px 0", borderBottom: "1px solid var(--line)" }}><span className="square-icon sm green"><Check className="icon sm" aria-hidden="true"/></span><div className="flex1"><strong className="small">{eventName(event)}</strong><div className="caption">Personal assistant · {eventStatus(event)} · {eventTime(event)}</div>{event.item !== undefined && event.item !== null && <p className="caption">{String(event.item)}</p>}</div></div>) : <p className="caption">Nothing yet.</p>}</div>
    </Dialog.Content></Dialog.Portal></Dialog.Root></div>
    {events.length ? events.slice(0, 5).map((event, index) => <div className="demo-activity-row" key={String(event.id ?? index)}><span className="status-dot"/><div className="flex1"><strong>{eventName(event)}</strong><small>{eventStatus(event)} · {eventTime(event)}</small></div></div>) : <p className="caption">New actions you take in this demo will appear here.</p>}
  </section>;
}

function FineTuneDemoRecords({ heading, collection, items }: { heading: string; collection: string; items: JsonRecord[] }) {
  const client = useQueryClient();
  const lastTrigger = useRef<HTMLElement | null>(null);
  const [filter, setFilter] = useState("");
  const [dialog, setDialog] = useState<"create" | "edit" | "details" | "actions" | null>(null);
  const [selected, setSelected] = useState<JsonRecord | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [status, setStatus] = useState("Ready · demo");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const titleOf = (item: JsonRecord) => String(item.name ?? item.title ?? item.id ?? "Training record");
  const summaryOf = (item: JsonRecord) => String(item.description ?? item.format ?? item.method ?? item.progress ?? "Local fine-tuning example");
  const filtered = items.filter(item => `${titleOf(item)} ${summaryOf(item)} ${String(item.status ?? "")}`.toLowerCase().includes(filter.trim().toLowerCase()));
  const begin = (mode: "create" | "edit", item?: JsonRecord) => { if (document.activeElement instanceof HTMLElement) lastTrigger.current = document.activeElement; setSelected(item ?? null); setName(item ? titleOf(item) : ""); setDescription(item ? summaryOf(item) : ""); setStatus(String(item?.status ?? "Ready · demo")); setError(""); setNotice(""); setDialog(mode); };
  const closeDialog = () => { setDialog(null); requestAnimationFrame(() => lastTrigger.current?.focus()); };
  const save = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!name.trim()) { setError("Enter a name before saving."); return; }
    const id = String(selected?.id ?? `demo-finetune-${collection}-${Date.now().toString(36)}`);
    const record = { ...(selected ?? {}), id, name: name.trim(), description: description.trim(), status };
    try {
      await demoProvider.updateCollection<JsonRecord>(`collections.advanced.finetune.${collection}`, current => selected ? current.map(row => String(row.id) === id ? record : row) : [record, ...current]);
      const eventId = globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
      await demoProvider.recordActivity("finetune", { id: `activity-finetune-${eventId}`, action: selected ? "Record edited" : "Record created", item: name.trim(), status: "Saved locally · demo", recorded_at: new Date().toISOString() });
      await client.invalidateQueries({ queryKey: ["advanced-screen", "finetune"] });
      setNotice(`${name.trim()} saved to local demo data.`); setDialog(null); setSelected(null);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not save the demo record."); }
  };
  const remove = async () => {
    if (!selected || !window.confirm(`Delete “${titleOf(selected)}” from local demo data? No backend record will change.`)) return;
    await demoProvider.updateCollection<JsonRecord>(`collections.advanced.finetune.${collection}`, current => current.filter(row => String(row.id) !== String(selected.id)));
    await demoProvider.recordActivity("finetune", { id: `activity-${String(selected.id)}-deleted-${Date.now().toString(36)}`, action: "Record deleted", item: titleOf(selected), status: "Deleted locally · demo", recorded_at: new Date().toISOString() });
    await client.invalidateQueries({ queryKey: ["advanced-screen", "finetune"] });
    setNotice("Demo record deleted locally."); setDialog(null); setSelected(null);
  };
  const duplicate = async () => {
    if (!selected) return;
    const copy: JsonRecord = { ...selected, id: `demo-finetune-${collection}-${Date.now().toString(36)}`, name: `${titleOf(selected)} (copy)`, status: "Copy · demo" };
    await demoProvider.updateCollection<JsonRecord>(`collections.advanced.finetune.${collection}`, current => [copy, ...current]);
    await demoProvider.recordActivity("finetune", { id: `activity-${String(copy.id)}`, action: "Record duplicated", item: titleOf(copy), status: "Saved locally · demo", recorded_at: new Date().toISOString() });
    await client.invalidateQueries({ queryKey: ["advanced-screen", "finetune"] });
    setNotice(`${titleOf(selected)} duplicated in local demo data.`); setDialog(null); setSelected(null);
  };
  return <section className="p-card"><div className="p-card-heading"><h3>{heading}</h3><span className="p-fixture">{filtered.length} demo records</span></div>
    <div className="demo-record-toolbar"><label className="search-field demo-record-search"><Search className="icon sm" aria-hidden="true"/><input value={filter} onChange={event => setFilter(event.target.value)} placeholder={`Search ${collection}…`} aria-label={`Search ${collection}`}/></label><button className="btn primary sm" type="button" onClick={event => { lastTrigger.current = event.currentTarget; begin("create"); }}><Plus className="icon sm" aria-hidden="true"/>Add</button></div>
    <div className="demo-records">{filtered.length ? filtered.map((item, index) => { const title = titleOf(item); return <article className="demo-record" key={String(item.id ?? index)}><button type="button" className="demo-record-main" onClick={event => { lastTrigger.current = event.currentTarget; setSelected(item); setDialog("details"); }}><span className="square-icon sm blue"><Database className="icon sm" aria-hidden="true"/></span><span className="demo-record-text"><strong>{title}</strong><small>{summaryOf(item)}</small></span></button><span className="tag">{String(item.status ?? "Ready · demo")}</span><button className="icon-button" type="button" aria-label={`Details for ${title}`} onClick={event => { lastTrigger.current = event.currentTarget; setSelected(item); setDialog("details"); }}><Info className="icon sm" aria-hidden="true"/></button><button className="icon-button" type="button" aria-label={`More actions for ${title}`} aria-haspopup="menu" onClick={event => { lastTrigger.current = event.currentTarget; setSelected(item); setDialog("actions"); }}><MoreHorizontal className="icon sm" aria-hidden="true"/></button></article>; }) : <div className="p-empty" role="status"><Search className="icon lg" aria-hidden="true"/><h3>{filter ? "No matching records" : "No records yet"}</h3><p>{filter ? "Adjust the search to find a training record." : "Add a local example or create a training preview."}</p></div>}</div>
    {notice && <p className="caption" role="status">{notice}</p>}
    <Dialog.Root open={dialog !== null} onOpenChange={open => { if (!open) closeDialog(); }}><Dialog.Portal><Dialog.Overlay/><Dialog.Content className="advanced-integration-dialog"><div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">{dialog === "details" ? titleOf(selected ?? {}) : dialog === "actions" ? "Training record actions" : `${dialog === "edit" ? "Edit" : "New"} ${collection === "results" ? "result" : collection === "checkpoints" ? "checkpoint" : collection.replace(/s$/, "")}`}</Dialog.Title><Dialog.Close asChild><button className="icon-button" aria-label="Close dialog" type="button"><X size={16}/></button></Dialog.Close></div><Dialog.Description className="second-app-dialog-description">{dialog === "details" ? "Deterministic local demo record. No model was trained or checkpoint created." : "Changes are stored only in this browser's local demo data."}</Dialog.Description>
      {dialog === "details" && selected && <><div className="demo-detail-grid">{Object.entries(selected).filter(([key]) => key !== "id").map(([key, value]) => <div key={key}><small>{humanize(key)}</small><strong>{String(value)}</strong></div>)}</div><div className="advanced-integration-actions"><button className="btn" type="button" onClick={() => setDialog("actions")}>More actions</button><button className="btn" type="button" onClick={closeDialog}>Close</button></div></>}
      {dialog === "actions" && selected && <div className="demo-action-grid"><button className="btn" type="button" onClick={() => setDialog("details")}>Open details</button><button className="btn" type="button" onClick={() => begin("edit", selected)}>Edit record</button><button className="btn" type="button" onClick={() => void duplicate()}>Duplicate record</button><button className="btn danger" type="button" onClick={() => void remove()}>Delete</button></div>}
      {(dialog === "create" || dialog === "edit") && <form onSubmit={event => void save(event)}><label className="demo-form-field">Name<input className="input" value={name} onChange={event => setName(event.target.value)} required maxLength={160}/></label><label className="demo-form-field">Description<textarea className="textarea" rows={3} value={description} onChange={event => setDescription(event.target.value)} maxLength={500}/></label><label className="demo-form-field">Status<input className="input" value={status} onChange={event => setStatus(event.target.value)} maxLength={80}/></label>{error && <p role="alert" className="caption">{error}</p>}<div className="advanced-integration-actions"><button className="btn" type="button" onClick={closeDialog}>Cancel</button><button className="btn primary" type="submit">{dialog === "edit" ? "Save changes" : "Create item"}</button></div></form>}
    </Dialog.Content></Dialog.Portal></Dialog.Root>
  </section>;
}

// ─── Image Studio ────────────────────────────────────────────────────────────

function GalleryWorkspace({ tab, data, query: lq }: { tab: string; data?: Record<string, JsonRecord>; query: { isLoading: boolean; error: unknown } }) {
  const client = useQueryClient();
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [opError, setOpError] = useState("");

  const images = (Array.isArray((data?.images as { images?: unknown })?.images) ? (data?.images as { images: JsonRecord[] }).images : Array.isArray(data?.images) ? data.images as JsonRecord[] : []) as JsonRecord[];
  const albums = (Array.isArray((data?.albums as { albums?: unknown })?.albums) ? (data?.albums as { albums: JsonRecord[] }).albums : Array.isArray(data?.albums) ? data.albums as JsonRecord[] : []) as JsonRecord[];

  const uploadMutation = useMutation({
    mutationFn: async () => {
      if (!uploadFile) throw new Error("No file selected");
      const form = new FormData();
      form.append("file", uploadFile);
      form.append("project", "default");
      return apiPost("/api/gallery/upload", form);
    },
    onSuccess: async () => { setUploadFile(null); setUploadError(""); await client.invalidateQueries({ queryKey: ["advanced-screen", "gallery"] }); },
    onError: (e) => setUploadError(e instanceof Error ? e.message : "Upload failed"),
  });

  const runImageOperation = (id: string, op: string) => {
    if (op === "upscale") {
      return apiPost("/api/gallery/upscale", { asset_id: id, project: "default", scale: 2 });
    }
    return apiPost(`/api/gallery/${enc(id)}/${op}`, { project: "default" });
  };

  const opMutation = useMutation({
    mutationFn: ({ id, op }: { id: string; op: string }) => runImageOperation(id, op),
    onSuccess: async () => { setOpError(""); await client.invalidateQueries({ queryKey: ["advanced-screen", "gallery"] }); },
    onError: (e) => setOpError(e instanceof Error ? e.message : "Operation failed"),
  });

  const deduplicateMutation = useMutation({
    mutationFn: () => apiPost("/api/gallery/deduplicate", { project: "default" }),
    onSuccess: async () => { await client.invalidateQueries({ queryKey: ["advanced-screen", "gallery"] }); },
    onError: (e) => setOpError(e instanceof Error ? e.message : "Deduplication failed"),
  });

  if (tab === "Upload") return <section className="p-card">
    <div className="p-card-heading"><h3>Upload images</h3></div>
    <label className="demo-form-field">Select image file<input className="input" type="file" accept="image/*" onChange={e => { setUploadFile(e.target.files?.[0] ?? null); setUploadError(""); }} /></label>
    {uploadFile && <p className="caption">{uploadFile.name} ({Math.round(uploadFile.size / 1024)} KB)</p>}
    {uploadError && <p className="caption" role="alert">{uploadError}</p>}
    <div className="row" style={{ gap: 8, marginTop: 12 }}>
      <button className="btn primary" type="button" disabled={!uploadFile || uploadMutation.isPending} onClick={() => uploadMutation.mutate()}><Upload className="icon sm" aria-hidden="true"/>{uploadMutation.isPending ? "Uploading…" : "Upload"}</button>
    </div>
  </section>;

  if (tab === "Albums") return <section className="p-card">
    <div className="p-card-heading"><h3>Albums</h3></div>
    {lq.isLoading ? <p className="caption">Loading albums…</p> : albums.length === 0
      ? <EmptyState icon={Image} title="No albums" detail="Create an album to organise your images." />
      : <div className="p-list">{albums.map((album, idx) => <div className="p-list-row" key={String(album.id ?? idx)}>
          <span className="square-icon sm"><Image className="icon sm" aria-hidden="true"/></span>
          <div className="flex1"><strong>{String(album.name ?? album.title ?? "Album")}</strong><small className="muted">{String(album.count ?? "")} images</small></div>
        </div>)}</div>}
  </section>;

  if (tab === "Operations") return <section className="p-card">
    <div className="p-card-heading"><h3>Batch operations</h3></div>
    {opError && <p className="caption" role="alert">{opError}</p>}
    <div className="p-list">
      <div className="p-list-row"><div className="flex1"><strong>Deduplicate library</strong><small className="muted">Remove duplicate images from the library</small></div><button className="btn sm" type="button" disabled={deduplicateMutation.isPending} onClick={() => deduplicateMutation.mutate()}>{deduplicateMutation.isPending ? "Running…" : "Run"}</button></div>
      {selectedId && <>
        <div className="p-list-row"><div className="flex1"><strong>Rotate selected</strong><small className="muted">Rotate image 90° clockwise</small></div><button className="btn sm" type="button" disabled={opMutation.isPending} onClick={() => opMutation.mutate({ id: selectedId, op: "rotate" })}>Rotate</button></div>
        <div className="p-list-row"><div className="flex1"><strong>Upscale selected</strong><small className="muted">AI upscale to 2x resolution</small></div><button className="btn sm" type="button" disabled={opMutation.isPending} onClick={() => opMutation.mutate({ id: selectedId, op: "upscale" })}>Upscale</button></div>
        <div className="p-list-row"><div className="flex1"><strong>Undo last operation</strong><small className="muted">Revert to previous version</small></div><button className="btn sm" type="button" disabled={opMutation.isPending} onClick={() => opMutation.mutate({ id: selectedId, op: "undo" })}>Undo</button></div>
      </>}
      {!selectedId && <p className="caption">Select an image in Library to enable per-image operations.</p>}
    </div>
  </section>;

  // Library tab (default)
  return <section className="p-card">
    <div className="p-card-heading"><h3>Image library</h3><span className="caption">{images.length} image{images.length !== 1 ? "s" : ""}</span></div>
    {lq.isLoading ? <p className="caption">Loading library…</p> : lq.error ? <p className="caption" role="alert">{apiConnectionMessage(lq.error)}</p> : images.length === 0
      ? <EmptyState icon={Image} title="No images" detail="Upload images to populate your library." />
      : <div className="p-gallery-grid">{images.map((img, idx) => {
          const id = String(img.id ?? idx);
          const name = String(img.name ?? img.filename ?? "Image");
          const thumb = typeof img.thumbnail_url === "string" ? img.thumbnail_url : typeof img.url === "string" ? img.url : null;
          return <button key={id} type="button" className={`p-gallery-cell ${selectedId === id ? "selected" : ""}`} onClick={() => setSelectedId(selectedId === id ? null : id)} title={name}>
            {thumb ? <img src={thumb} alt={name} className="p-gallery-thumb" /> : <span className="p-gallery-thumb p-gallery-placeholder"><Image className="icon" aria-hidden="true"/></span>}
            <span className="p-gallery-label">{name}</span>
          </button>;
        })}</div>}
    {selectedId && <p className="caption" style={{ marginTop: 8 }}>Selected: {selectedId} · Go to Operations tab for edits.</p>}
  </section>;
}

// ─── Model Cookbook ───────────────────────────────────────────────────────────

function CookbookWorkspace({ tab, data, query: lq }: { tab: string; data?: Record<string, JsonRecord>; query: { isLoading: boolean; error: unknown } }) {
  const client = useQueryClient();
  const [hostName, setHostName] = useState("");
  const [hostUrl, setHostUrl] = useState("");
  const [addError, setAddError] = useState("");
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [modelRef, setModelRef] = useState("");
  const [runner, setRunner] = useState("ollama");
  const [sessionError, setSessionError] = useState("");
  const [sessionNotice, setSessionNotice] = useState("");

  const hosts = (Array.isArray((data?.hosts as { hosts?: unknown })?.hosts) ? (data?.hosts as { hosts: JsonRecord[] }).hosts : Array.isArray(data?.hosts) ? data.hosts as JsonRecord[] : []) as JsonRecord[];
  const hostSessions = (Array.isArray((data?.sessions as { sessions?: unknown })?.sessions) ? (data?.sessions as { sessions: JsonRecord[] }).sessions : Array.isArray(data?.sessions) ? data.sessions as JsonRecord[] : []) as JsonRecord[];
  const cached = (Array.isArray((data?.cached as { files?: unknown })?.files) ? (data?.cached as { files: JsonRecord[] }).files : Array.isArray(data?.cached) ? data.cached as JsonRecord[] : []) as JsonRecord[];
  const certs = (Array.isArray((data?.certifications as { certifications?: unknown })?.certifications) ? (data?.certifications as { certifications: JsonRecord[] }).certifications : Array.isArray(data?.certifications) ? data.certifications as JsonRecord[] : []) as JsonRecord[];

  const addHostMutation = useMutation({
    mutationFn: () => apiPost("/api/model-cookbook/hosts", { label: hostName.trim(), address: hostUrl.trim(), host_type: "local", project: "default" }),
    onSuccess: async () => { setHostName(""); setHostUrl(""); setAddError(""); await client.invalidateQueries({ queryKey: ["advanced-screen", "cookbook"] }); },
    onError: (e) => setAddError(e instanceof Error ? e.message : "Failed to add host"),
  });

  const serveMutation = useMutation({
    mutationFn: ({ id, modelRef, runner }: { id: string; modelRef: string; runner: string }) => apiPost("/api/model-cookbook/serve", { host_id: id, model_ref: modelRef, runner, project: "default", allow_process: false }),
    onSuccess: async (result) => {
      const session = (result as { session?: JsonRecord }).session;
      if (session?.id) setSelectedSessionId(String(session.id));
      setSessionError("");
      setSessionNotice(session ? `Serve plan ${String(session.id ?? "")} · ${String(session.status ?? "planned")}. No process was launched.` : "Serve plan recorded. No process was launched.");
      await client.invalidateQueries({ queryKey: ["advanced-screen", "cookbook"] });
    },
    onError: (e) => setSessionError(e instanceof Error ? e.message : "Could not create serve plan"),
  });

  const stopMutation = useMutation({
    mutationFn: (id: string) => apiPost(`/api/model-cookbook/stop/${enc(id)}`, {}),
    onSuccess: async () => { setSessionError(""); setSessionNotice("Serve session was marked stopped by the backend."); await client.invalidateQueries({ queryKey: ["advanced-screen", "cookbook"] }); },
    onError: (e) => setSessionError(e instanceof Error ? e.message : "Could not stop serve session"),
  });

  if (tab === "Cached") return <section className="p-card">
    <div className="p-card-heading"><h3>Cached model files</h3></div>
    {lq.isLoading ? <p className="caption">Loading…</p> : cached.length === 0
      ? <EmptyState icon={BookMarked} title="No cached files" detail="Cached model files appear here after a model is adopted." />
      : <div className="p-list">{cached.map((f, idx) => <div className="p-list-row" key={String(f.id ?? idx)}>
          <span className="square-icon sm"><HardDrive className="icon sm" aria-hidden="true"/></span>
          <div className="flex1"><strong>{String(f.name ?? f.filename ?? "File")}</strong><small className="muted">{String(f.size ?? "")} · {String(f.status ?? "")}</small></div>
        </div>)}</div>}
  </section>;

  if (tab === "Certifications") return <section className="p-card">
    <div className="p-card-heading"><h3>Model certifications</h3></div>
    {lq.isLoading ? <p className="caption">Loading…</p> : certs.length === 0
      ? <EmptyState icon={BookMarked} title="No certifications" detail="Certified model builds appear here." />
      : <div className="p-list">{certs.map((c, idx) => <div className="p-list-row" key={String(c.id ?? idx)}>
          <span className="square-icon sm green"><Check className="icon sm" aria-hidden="true"/></span>
          <div className="flex1"><strong>{String(c.model ?? c.name ?? "Model")}</strong><small className="muted">{String(c.issued_at ?? c.created_at ?? "")} · {String(c.status ?? "certified")}</small></div>
        </div>)}</div>}
  </section>;

  if (tab === "Logs") return <section className="p-card">
    <div className="p-card-heading"><h3>Serving logs</h3></div>
    {hostSessions.length > 0 && <label className="demo-form-field">Serve session<select className="select" aria-label="Serve session" value={selectedSessionId ?? ""} onChange={event => setSelectedSessionId(event.target.value)}><option value="">Select a session</option>{hostSessions.map(session => <option key={String(session.id)} value={String(session.id)}>{String(session.model_ref ?? "Model")} · {String(session.status ?? "unknown")}</option>)}</select></label>}
    {selectedSessionId
      ? <LogsViewer path={`/api/model-cookbook/logs/${enc(selectedSessionId)}`} label="serve session logs" />
      : <p className="caption">Select a serve session to view its backend logs.</p>}
  </section>;

  // Hosts tab (default)
  return <section className="p-card">
    <div className="p-card-heading"><h3>Model hosts</h3></div>
    <div className="row" style={{ gap: 8, marginBottom: 12 }}>
      <input className="input flex1" placeholder="Host name" value={hostName} onChange={e => setHostName(e.target.value)} />
      <input className="input flex1" placeholder="Host address" value={hostUrl} onChange={e => setHostUrl(e.target.value)} />
      <button className="btn sm primary" type="button" disabled={!hostName.trim() || addHostMutation.isPending} onClick={() => addHostMutation.mutate()}><Plus className="icon sm"/>Add</button>
    </div>
    <div className="row" style={{ gap: 8, marginBottom: 12 }}>
      <input className="input flex1" aria-label="Model reference to serve" placeholder="Model reference" value={modelRef} onChange={event => setModelRef(event.target.value)} />
      <select className="select flex1" aria-label="Model runner" value={runner} onChange={event => setRunner(event.target.value)}>{["ollama", "llama-server", "llama.cpp", "python", "vllm"].map(value => <option key={value} value={value}>{value}</option>)}</select>
    </div>
    <p className="caption">Serve creates a backend plan only. It does not launch a process.</p>
    {addError && <p className="caption" role="alert">{addError}</p>}
    {sessionError && <p className="caption" role="alert">{sessionError}</p>}
    {sessionNotice && <p className="caption" role="status">{sessionNotice}</p>}
    {lq.isLoading ? <p className="caption">Loading hosts…</p> : hosts.length === 0
      ? <EmptyState icon={Server} title="No hosts" detail="Add a model host to get started." />
      : <div className="p-list">{hosts.map((h, idx) => {
          const id = String(h.id ?? idx);
          const sessionsForHost = hostSessions.filter(session => String(session.host_id ?? "") === id);
          const latestSession = sessionsForHost[0];
          const status = latestSession ? String(latestSession.status ?? "unknown") : "No serve session";
          const running = latestSession && ["running", "serving", "planned", "adopted"].includes(status);
          return <div className="p-list-row" key={id}>
            <span className={`square-icon sm ${running ? "green" : ""}`}><Server className="icon sm" aria-hidden="true"/></span>
            <div className="flex1"><strong>{String(h.label ?? h.name ?? "Host")}</strong><small className="muted">{String(h.address ?? h.url ?? "")} · {status}</small></div>
            <div className="row" style={{ gap: 4 }}>
              {!running && <button className="btn sm" type="button" disabled={!modelRef.trim() || serveMutation.isPending} onClick={() => serveMutation.mutate({ id, modelRef: modelRef.trim(), runner })}>Create serve plan</button>}
              {running && latestSession && <button className="btn sm" type="button" disabled={stopMutation.isPending} onClick={() => stopMutation.mutate(String(latestSession.id))}>{status === "planned" ? "Close plan" : "Stop"}</button>}
              {latestSession && <button className="btn sm soft" type="button" onClick={() => setSelectedSessionId(String(latestSession.id))}>Logs</button>}
            </div>
          </div>;
        })}</div>}
  </section>;
}

// ─── Ultra Model ──────────────────────────────────────────────────────────────

function UltraModelWorkspace({ tab, data, query: lq }: { tab: string; data?: Record<string, JsonRecord>; query: { isLoading: boolean; error: unknown } }) {
  const client = useQueryClient();
  const [launchModel, setLaunchModel] = useState("");
  const [launchError, setLaunchError] = useState("");

  const manifests = (Array.isArray((data?.manifests as { manifests?: unknown })?.manifests) ? (data?.manifests as { manifests: JsonRecord[] }).manifests : Array.isArray(data?.manifests) ? data.manifests as JsonRecord[] : []) as JsonRecord[];
  const plans = (Array.isArray((data?.plans as { plans?: unknown })?.plans) ? (data?.plans as { plans: JsonRecord[] }).plans : Array.isArray(data?.plans) ? data.plans as JsonRecord[] : []) as JsonRecord[];
  const sessions = (Array.isArray((data?.sessions as { sessions?: unknown })?.sessions) ? (data?.sessions as { sessions: JsonRecord[] }).sessions : Array.isArray(data?.sessions) ? data.sessions as JsonRecord[] : []) as JsonRecord[];
  const status = data?.status as JsonRecord | undefined;

  const launchMutation = useMutation({
    mutationFn: () => apiPost("/api/ultra-model/launch", { model_id: launchModel.trim() }),
    onSuccess: async () => { setLaunchModel(""); setLaunchError(""); await client.invalidateQueries({ queryKey: ["advanced-screen", "ultramodel"] }); },
    onError: (e) => setLaunchError(e instanceof Error ? e.message : "Launch failed"),
  });

  const stopSessionMutation = useMutation({
    mutationFn: (id: string) => apiPost(`/api/ultra-model/sessions/${enc(id)}/stop`, {}),
    onSuccess: async () => { await client.invalidateQueries({ queryKey: ["advanced-screen", "ultramodel"] }); },
  });

  const calibrateMutation = useMutation({
    mutationFn: () => apiPost("/api/ultra-model/calibrate", {}),
    onSuccess: async () => { await client.invalidateQueries({ queryKey: ["advanced-screen", "ultramodel"] }); },
  });

  if (tab === "Manifests") return <section className="p-card">
    <div className="p-card-heading"><h3>Inference manifests</h3></div>
    {lq.isLoading ? <p className="caption">Loading…</p> : manifests.length === 0
      ? <EmptyState icon={Zap} title="No manifests" detail="Manifests appear after a model is routed or launched." />
      : <div className="p-list">{manifests.map((m, idx) => <div className="p-list-row" key={String(m.id ?? idx)}>
          <span className="square-icon sm"><Zap className="icon sm" aria-hidden="true"/></span>
          <div className="flex1"><strong>{String(m.model_id ?? m.name ?? "Manifest")}</strong><small className="muted">{String(m.status ?? "")} · {String(m.created_at ?? "")}</small></div>
        </div>)}</div>}
  </section>;

  if (tab === "Plans") return <section className="p-card">
    <div className="p-card-heading"><h3>Routing plans</h3></div>
    {lq.isLoading ? <p className="caption">Loading…</p> : plans.length === 0
      ? <EmptyState icon={Zap} title="No plans" detail="Routing plans are computed when a model is routed." />
      : <div className="p-list">{plans.map((p, idx) => <div className="p-list-row" key={String(p.id ?? idx)}>
          <span className="square-icon sm"><Activity className="icon sm" aria-hidden="true"/></span>
          <div className="flex1"><strong>{String(p.name ?? p.plan_id ?? "Plan")}</strong><small className="muted">{String(p.strategy ?? "")} · {String(p.created_at ?? "")}</small></div>
        </div>)}</div>}
  </section>;

  if (tab === "Sessions") return <section className="p-card">
    <div className="p-card-heading"><h3>Active sessions</h3></div>
    {lq.isLoading ? <p className="caption">Loading…</p> : sessions.length === 0
      ? <EmptyState icon={Zap} title="No active sessions" detail="Launch a model to start an inference session." />
      : <div className="p-list">{sessions.map((s, idx) => {
          const id = String(s.id ?? idx);
          return <div className="p-list-row" key={id}>
            <span className="square-icon sm green"><Zap className="icon sm" aria-hidden="true"/></span>
            <div className="flex1"><strong>{String(s.model_id ?? s.name ?? "Session")}</strong><small className="muted">{String(s.status ?? "active")} · {String(s.started_at ?? "")}</small></div>
            <button className="btn sm" type="button" onClick={() => stopSessionMutation.mutate(id)}>Stop</button>
          </div>;
        })}</div>}
  </section>;

  // Status tab (default)
  return <section className="p-card">
    <div className="p-card-heading"><h3>Ultra model status</h3></div>
    {status && <div className="p-kv-grid" style={{ marginBottom: 16 }}>
      <div><small>Service</small><strong>{String(status.status ?? "unknown")}</strong></div>
      <div><small>Version</small><strong>{String(status.version ?? "—")}</strong></div>
      <div><small>Active sessions</small><strong>{String(status.active_sessions ?? sessions.length)}</strong></div>
    </div>}
    <div className="row" style={{ gap: 8, marginBottom: 12 }}>
      <input className="input flex1" placeholder="Model ID to launch" value={launchModel} onChange={e => setLaunchModel(e.target.value)} />
      <button className="btn sm primary" type="button" disabled={!launchModel.trim() || launchMutation.isPending} onClick={() => launchMutation.mutate()}><Play className="icon sm"/>Launch</button>
    </div>
    {launchError && <p className="caption" role="alert">{launchError}</p>}
    <div className="row" style={{ gap: 8, marginTop: 8 }}>
      <button className="btn sm soft" type="button" disabled={calibrateMutation.isPending} onClick={() => calibrateMutation.mutate()}>Calibrate</button>
    </div>
    {sessions.length > 0 && <div style={{ marginTop: 16 }}>
      <div className="p-card-heading"><h3>Active sessions</h3></div>
      <div className="p-list">{sessions.slice(0, 5).map((s, idx) => <div className="p-list-row" key={String(s.id ?? idx)}>
        <span className="status-dot green" aria-hidden="true"/>
        <div className="flex1"><strong>{String(s.model_id ?? "Session")}</strong><small className="muted">{String(s.status ?? "active")}</small></div>
      </div>)}</div>
    </div>}
  </section>;
}

// ─── Personal OS ──────────────────────────────────────────────────────────────

function PersonalOSWorkspace({ tab, data, query: lq }: { tab: string; data?: Record<string, JsonRecord>; query: { isLoading: boolean; error: unknown } }) {
  const client = useQueryClient();
  const [noteText, setNoteText] = useState("");
  const [noteError, setNoteError] = useState("");
  const [presetName, setPresetName] = useState("");
  const [presetError, setPresetError] = useState("");

  const presets = (Array.isArray((data?.presets as { presets?: unknown })?.presets) ? (data?.presets as { presets: JsonRecord[] }).presets : Array.isArray(data?.presets) ? data.presets as JsonRecord[] : []) as JsonRecord[];
  const notes = (Array.isArray((data?.notes as { notes?: unknown })?.notes) ? (data?.notes as { notes: JsonRecord[] }).notes : Array.isArray(data?.notes) ? data.notes as JsonRecord[] : []) as JsonRecord[];
  const galleryPayload = data?.gallery as { gallery?: unknown; assets?: unknown } | undefined;
  const gallery = (Array.isArray(galleryPayload?.assets) ? galleryPayload.assets : Array.isArray(galleryPayload?.gallery) ? galleryPayload.gallery : Array.isArray(data?.gallery) ? data.gallery : []) as JsonRecord[];
  const signatures = (Array.isArray((data?.signatures as { signatures?: unknown })?.signatures) ? (data?.signatures as { signatures: JsonRecord[] }).signatures : Array.isArray(data?.signatures) ? data.signatures as JsonRecord[] : []) as JsonRecord[];
  const drafts = (Array.isArray((data?.drafts as { drafts?: unknown })?.drafts) ? (data?.drafts as { drafts: JsonRecord[] }).drafts : Array.isArray(data?.drafts) ? data.drafts as JsonRecord[] : []) as JsonRecord[];
  const status = data?.status as JsonRecord | undefined;

  const addNoteMutation = useMutation({
    mutationFn: () => apiPost("/api/personal-os/notes", { title: "Personal note", body: noteText.trim(), project: "default" }),
    onSuccess: async () => { setNoteText(""); setNoteError(""); await client.invalidateQueries({ queryKey: ["advanced-screen", "personalos"] }); },
    onError: (e) => setNoteError(e instanceof Error ? e.message : "Failed to save note"),
  });

  const addPresetMutation = useMutation({
    mutationFn: () => apiPost("/api/personal-os/presets", { name: presetName.trim(), project: "default" }),
    onSuccess: async () => { setPresetName(""); setPresetError(""); await client.invalidateQueries({ queryKey: ["advanced-screen", "personalos"] }); },
    onError: (e) => setPresetError(e instanceof Error ? e.message : "Failed to create preset"),
  });

  const exportMutation = useMutation({
    mutationFn: () => apiPost("/api/personal-os/export", { project: "default" }),
    onSuccess: (result) => {
      const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = "personal-os-export.json"; a.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    },
  });

  if (tab === "Presets") return <section className="p-card">
    <div className="p-card-heading"><h3>Presets</h3></div>
    <div className="row" style={{ gap: 8, marginBottom: 12 }}>
      <input className="input flex1" placeholder="Preset name" value={presetName} onChange={e => setPresetName(e.target.value)} />
      <button className="btn sm primary" type="button" disabled={!presetName.trim() || addPresetMutation.isPending} onClick={() => addPresetMutation.mutate()}><Plus className="icon sm"/>Add</button>
    </div>
    {presetError && <p className="caption" role="alert">{presetError}</p>}
    {lq.isLoading ? <p className="caption">Loading…</p> : presets.length === 0
      ? <EmptyState icon={UserRound} title="No presets" detail="Create a preset to save workspace preferences." />
      : <div className="p-list">{presets.map((p, idx) => <div className="p-list-row" key={String(p.id ?? idx)}>
          <span className="square-icon sm"><UserRound className="icon sm" aria-hidden="true"/></span>
          <div className="flex1"><strong>{String(p.name ?? "Preset")}</strong><small className="muted">{String(p.created_at ?? "")}</small></div>
        </div>)}</div>}
  </section>;

  if (tab === "Notes") return <section className="p-card">
    <div className="p-card-heading"><h3>Personal notes</h3></div>
    <textarea className="input" rows={4} placeholder="Write a note…" value={noteText} onChange={e => setNoteText(e.target.value)} style={{ width: "100%", marginBottom: 8 }} />
    {noteError && <p className="caption" role="alert">{noteError}</p>}
    <button className="btn sm primary" type="button" disabled={!noteText.trim() || addNoteMutation.isPending} onClick={() => addNoteMutation.mutate()}>{addNoteMutation.isPending ? "Saving…" : "Save note"}</button>
    {notes.length > 0 && <div style={{ marginTop: 16 }}>
      <div className="p-list">{notes.map((n, idx) => <div className="p-list-row" key={String(n.id ?? idx)}>
        <div className="flex1"><p className="small" style={{ margin: 0 }}>{String(n.body ?? n.content ?? n.text ?? "Note")}</p><small className="muted">{String(n.created_at ?? "")}</small></div>
      </div>)}</div>
    </div>}
  </section>;

  if (tab === "Gallery") return <section className="p-card">
    <div className="p-card-heading"><h3>Personal gallery</h3></div>
    {lq.isLoading ? <p className="caption">Loading…</p> : gallery.length === 0
      ? <EmptyState icon={Image} title="No gallery items" detail="Images saved to Personal OS appear here." />
      : <div className="p-gallery-grid">{gallery.map((img, idx) => {
          const thumb = typeof img.thumbnail_url === "string" ? img.thumbnail_url : null;
          return <div key={String(img.id ?? idx)} className="p-gallery-cell">
            {thumb ? <img src={thumb} alt={String(img.name ?? "")} className="p-gallery-thumb" /> : <span className="p-gallery-thumb p-gallery-placeholder"><Image className="icon" aria-hidden="true"/></span>}
            <span className="p-gallery-label">{String(img.name ?? "Image")}</span>
          </div>;
        })}</div>}
  </section>;

  if (tab === "Signatures") return <section className="p-card">
    <div className="p-card-heading"><h3>Signatures</h3></div>
    {lq.isLoading ? <p className="caption">Loading…</p> : signatures.length === 0
      ? <EmptyState icon={UserRound} title="No signatures" detail="Add a personal signature for use in outgoing messages." />
      : <div className="p-list">{signatures.map((s, idx) => <div className="p-list-row" key={String(s.id ?? idx)}>
          <div className="flex1"><strong>{String(s.label ?? s.name ?? "Signature")}</strong><p className="small muted" style={{ margin: 0 }}>{String(s.content ?? "").slice(0, 80)}</p></div>
        </div>)}</div>}
  </section>;

  if (tab === "Drafts") return <section className="p-card">
    <div className="p-card-heading"><h3>Drafts</h3></div>
    {lq.isLoading ? <p className="caption">Loading…</p> : drafts.length === 0
      ? <EmptyState icon={FileText} title="No drafts" detail="Personal drafts appear here." />
      : <div className="p-list">{drafts.map((d, idx) => <div className="p-list-row" key={String(d.id ?? idx)}>
          <span className="square-icon sm"><FileText className="icon sm" aria-hidden="true"/></span>
          <div className="flex1"><strong>{String(d.name ?? d.title ?? "Draft")}</strong><small className="muted">{String(d.updated_at ?? d.created_at ?? "")}</small></div>
        </div>)}</div>}
  </section>;

  // Status tab (default)
  return <section className="p-card">
    <div className="p-card-heading"><h3>Personal OS status</h3></div>
    {status && <div className="p-kv-grid" style={{ marginBottom: 16 }}>
      <div><small>Service</small><strong>{String(status.status ?? "unknown")}</strong></div>
      <div><small>Presets</small><strong>{String(status.preset_count ?? presets.length)}</strong></div>
      <div><small>Notes</small><strong>{String(status.note_count ?? notes.length)}</strong></div>
    </div>}
    <div className="row" style={{ gap: 8, marginTop: 8 }}>
      <button className="btn sm soft" type="button" disabled={exportMutation.isPending} onClick={() => exportMutation.mutate()}><FileDown className="icon sm"/>Export data</button>
    </div>
  </section>;
}

// ─── Logs viewer helper ───────────────────────────────────────────────────────

function LogsViewer({ path, label }: { path: string; label: string }) {
  const q = useQuery({ queryKey: ["logs", path], queryFn: ({ signal }) => apiGet(path, signal), staleTime: 10_000 });
  const lines = (Array.isArray((q.data as { lines?: unknown })?.lines) ? (q.data as { lines: string[] }).lines : []) as string[];
  return <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, background: "var(--surface)", borderRadius: 6, padding: 12, overflowX: "auto", maxHeight: 300, overflowY: "auto" }}>
    {q.isLoading ? <p className="caption">Loading {label}…</p> : q.error ? <p className="caption">{apiConnectionMessage(q.error)}</p> : lines.length === 0 ? <p className="caption">No log entries.</p> : lines.map((line, i) => <div key={i}>{line}</div>)}
  </div>;
}

function RouteContent({ route, tab, definition, data, query, onTabChange, demo }: { route: string; tab: string; definition: Definition; data?: Record<string, JsonRecord>; query: { isLoading: boolean; error: unknown }; onTabChange: (tab: string) => void; demo: boolean }) {
  const Icon = routeIcons[route] ?? definition.icon;
  if (route === "engine" && tab === "Overview" && demo) return <EngineDemoOverview data={data} />;
  if (route === "engine" && demo) return <EngineDemoWorkspace tab={tab} data={data}/>;
  if (route === "engine" && ["Overview", "Hardware", "Jobs", "Plans"].includes(tab)) return <EngineLiveWorkspace tab={tab} data={data} query={query} />;

  if (route === "voice") return demo
    ? tab === "Recordings" ? <VoiceDemoRecordings data={data} /> : tab === "Transcripts" ? <VoiceDemoTranscripts /> : <VoiceDemoSettings />
    : <VoiceWorkspace tab={tab} data={data} query={query} />;
  if (route === "assistant" && demo) return <AssistantDemoWorkspace tab={tab} data={data} query={query} onTabChange={onTabChange} />;
  if (route === "assistant") return <AssistantWorkspace tab={tab} data={data} query={query} />;
  if (route === "security") return <SecurityWorkspace tab={tab} data={data} query={query} demo={demo} />;
  if (route === "vault") return <VaultWorkspace tab={tab} data={data} demo={demo} />;
  if (route === "finetune" && demo) return <FineTuneDemoWorkspace tab={tab} data={data}/>;
  if (route === "finetune") return <FineTuneWorkspace tab={tab} data={data} />;
  if ((route === "science" || route === "scraper") && demo) return <AdvancedDemoRecordWorkspace route={route} tab={tab} data={data} />;
  if (route === "science" && tab === "Data") return <><ScienceDataWorkspace data={data} /><section className="p-card"><div className="p-card-heading"><h3>Project datasets and scientific notes</h3></div><p className="caption">Saved datasets and memory records below are backend records. A saved proposal or catalog analysis is not a measured experimental result.</p></section></>;
  if (route === "twin" && tab === "Sources" && !demo) {
    const sourcePayload = data?.sources?.sources;
    const sources = Array.isArray(sourcePayload) ? sourcePayload.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
    return <section className="p-card"><div className="p-card-heading"><h3>Interaction source metadata</h3><span className="p-fixture">{sources.length} sources</span></div><p className="muted small">The backend lists recorded interaction type, extracted topics and date. Where an extracted topic contributed to a stored inferred preference, that relationship is shown. Original interaction text and context are omitted.</p>{Boolean(query.error) && <p className="note-panel" role="alert">{apiConnectionMessage(query.error)}</p>}{query.isLoading ? <p className="caption" role="status">Loading source metadata…</p> : sources.length ? <div className="p-record-sections">{sources.map((source, index) => { const signals = Array.isArray(source.preference_signals) ? source.preference_signals.filter((value): value is JsonRecord => Boolean(value) && typeof value === "object" && !Array.isArray(value)) : []; return <article className="p-record-section" data-source-id={String(source.id ?? "")} key={String(source.id ?? index)}><h4>{String(source.kind ?? source.type ?? "Interaction source")}</h4><p className="caption">{recordSummary({ ...source, kind: undefined, type: undefined, preference_signals: undefined }, "twin") || "No extracted topics"}{source.created_at ? ` · ${String(source.created_at)}` : ""}</p>{signals.length > 0 && <p className="caption"><strong>Inferred preference signals:</strong> {signals.map(signal => String(signal.value ?? signal.key ?? "topic")).join(", ")}</p>}</article>; })}</div> : <EmptyState icon={UserRound} title="No context sources" detail="No interaction source metadata has been reported by the context service." />}</section>;
  }
  if (route === "twin" && demo && tab === "Profile") return <TwinDemoProfile data={data} />;
  if (route === "twin" && demo) return <TwinDemoWorkspace key={tab} tab={tab} data={data} />;
  if (route === "twin") return <><section className="p-card"><div className="p-card-heading"><h3>{tab}</h3></div><div className="p-context-intro"><div className="p-profile-avatar"><UserRound className="icon xl" aria-hidden="true" /></div><div><h2>Personal context</h2><p>Profile, patterns, suggestions and goals shown below come from the context service. Inferred suggestions are not verified facts.</p></div></div></section><TwinOperations tab={tab} data={data} /></>;
  if (route === "scraper" && tab === "Jobs") return <ScraperDueJobs data={data} />;

  if (route === "scraper" && tab === "New extraction") return <section className="p-card"><div className="p-card-heading"><h3>Review an extraction request</h3></div><p className="muted small">Provide the URL and network permission in the submission form below. The backend enforces robots and domain policy.</p><label className="demo-form-field">Website URL<input className="input" type="url" placeholder="Enter the URL in the submission form below" disabled /></label><label className="demo-form-field">Extraction scope<select className="select" disabled defaultValue=""><option value="">Backend policy determines allowed scope</option></select></label><button type="button" className="btn primary" onClick={() => scrollToPanel("advanced-backend-action")}>Review permissions</button><div className="p-note section-space"><Shield className="icon sm" aria-hidden="true" />No extraction request is sent until you submit the form below.</div></section>;

  if (route === "science" && tab === "Experiments") return <section className="p-card"><div className="p-card-heading"><h3>Experiment proposals</h3></div><div className="p-science-hero"><div className="p-science-flask"><FlaskConical className="icon lg" aria-hidden="true" /></div><div><h2>Propose a reproducible experiment</h2><p>Backend proposals are saved with project scope and remain marked proposed until an execution operation records a run.</p></div></div><button type="button" className="btn primary" onClick={() => scrollToPanel("advanced-backend-action")} title="Enter a research question and candidate parameters first">Propose experiment</button></section>;

  if (route === "finetune" && tab === "Datasets") return <section className="p-card"><div className="p-card-heading"><h3>Dataset preparation</h3></div><div className="p-dropzone"><HardDrive className="icon lg" aria-hidden="true" /><strong>No dataset selected</strong><span>Create a dataset or select a verified backend record below.</span><button className="btn" type="button" onClick={() => scrollToPanel("advanced-backend-action")}>Create dataset</button></div><EmptyState icon={Database} title="No datasets loaded" detail="Dataset records and validation results have not been provided." /></section>;

  if (route === "security") return <section className="p-card"><div className="p-card-heading"><h3>{tab === "Overview" ? "Access and audit" : tab}</h3></div><div className="p-security-banner"><div className="p-security-emblem"><Shield className="icon xl" aria-hidden="true" /></div><div><span className="p-overline">Protected workspace</span><h2>Security information is not loaded</h2><p>Effective permissions, authenticated sessions and audit events must come from the security service.</p></div></div><EmptyState icon={Fingerprint} title="No security records" detail="There are no verified permission or audit records to display." /></section>;

  if (route === "communications" && tab === "Inbox") return <CommunicationsInbox data={data} query={query} onTabChange={onTabChange} demo={demo} />;
  if (route === "communications" && tab === "Drafts") return <CommunicationsDraftQueue data={data} />;
  if (route === "communications") return <CommunicationRecords tab={tab} data={data} demo={demo} />;

  if (route === "developer") return <DeveloperResources tab={tab} data={data} demo={demo} />;

  if (route === "knowledge") return <>
    <section className="demo-feature-hero"><div className="demo-feature-symbol"><BookOpen className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">Search · Workspace</span><h2>Knowledge that stays useful</h2><p>Search original sources, maintain collections and inspect citations.</p></div></section>
    <section className="p-card">
      <div className="p-card-heading"><h3>{tab === "Search" ? "Search the knowledge base" : tab}</h3></div>
      <label className="p-input-large"><Search className="icon sm" aria-hidden="true" /><input aria-label="Search indexed knowledge" placeholder="Search sources, documents or concepts…" disabled /><button className="btn primary" type="button" disabled>Search</button></label>
      <div className="row section-space" style={{ marginBottom: 12 }}><label className="search-field flex1"><Search className="icon sm" aria-hidden="true" /><input aria-label="Filter indexed documents" placeholder="Search documents…" disabled /></label><span className="caption">Index status unavailable</span><button className="btn primary" type="button" disabled title="Adding sources requires the knowledge service"><Plus className="icon sm" />Add</button></div>
      <div className="p-empty" role="status"><BookOpen className="icon lg" aria-hidden="true" /><h3>No indexed sources</h3><p className="muted small">Connect a knowledge source to search documents and citations.</p><span className="p-fixture"><CloudOff className="icon xs" aria-hidden="true" />Knowledge index not connected</span></div>
    </section>
  </>;

  if (route === "science" && tab !== "Experiments") return <section className="p-card"><div className="p-card-heading"><h3>{tab}</h3></div><EmptyState icon={Icon} title={`No ${tab.toLowerCase()} loaded`} detail="Scientific projects, data provenance and reports require a connected service." /></section>;

  if (route === "gallery") return <GalleryWorkspace tab={tab} data={data} query={query} />;
  if (route === "cookbook") return <CookbookWorkspace tab={tab} data={data} query={query} />;
  if (route === "ultramodel") return <UltraModelWorkspace tab={tab} data={data} query={query} />;
  if (route === "personalos") return <PersonalOSWorkspace tab={tab} data={data} query={query} />;

  return <section className="p-card"><div className="p-card-heading"><h3>{tab}</h3></div><EmptyState icon={Icon} title={`No ${tab.toLowerCase()} loaded`} detail={definition.note} /></section>;
}

export default function AdvancedScreen({ route }: { route: string }) {
  const dataMode = useWorkspaceState(state => state.dataMode);
  const definition = definitions[route] ?? definitions.knowledge;
  const [activeTab, setActiveTab] = useState(definition.tabs[0]);
  const [integrationGuideOpen, setIntegrationGuideOpen] = useState(false);
  const [guideExportError, setGuideExportError] = useState("");
  const integrationGuideTrigger = useRef<HTMLButtonElement | null>(null);
  const currentTab = definition.tabs.includes(activeTab) ? activeTab : definition.tabs[0];
  const supportedRoutes: AdvancedRoute[] = ["voice", "assistant", "communications", "security", "developer", "engine", "science", "twin", "vault", "scraper", "finetune", "gallery", "cookbook", "ultramodel", "personalos"];
  const isAdvanced = supportedRoutes.includes(route as AdvancedRoute);
  const selectedRoute = (isAdvanced ? route : "voice") as AdvancedRoute;
  const endpointMap = isAdvanced ? route === "security" ? securityTabEndpoints[currentTab] : route === "science" && currentTab === "Data" ? { ...(advancedScreenEndpoints(selectedRoute, currentTab) ?? {}), catalog: "/api/science/materials/catalog" } : advancedScreenEndpoints(selectedRoute, currentTab) : null;
  const liveQuery = useQuery({
    queryKey: ["advanced-screen", route, currentTab, dataMode],
    queryFn: async ({ signal }) => {
      if (dataMode === "demo") return advancedDemoScreen(route, currentTab) as Promise<Record<string, JsonRecord>>;
      if (route === "security") return fetchSecurityScreenData(currentTab, signal);
      const result = await fetchAdvancedScreenData(selectedRoute, currentTab, signal);
      if (route === "science" && currentTab === "Data") result.catalog = await apiGet("/api/science/materials/catalog", signal);
      return result;
    },
    enabled: isAdvanced && (dataMode === "demo" || Boolean(endpointMap)),
    staleTime: 15_000,
    refetchInterval: route === "communications" && currentTab === "Inbox" ? 15_000 : 30_000,
  });
  const connectionLabel = dataMode === "demo" ? "Offline visual prototype" : !isAdvanced ? "Backend unavailable" : !endpointMap ? "No read endpoint for this tab" : liveQuery.isError ? "Connection error" : liveQuery.isSuccess ? "Backend connected" : liveQuery.isLoading ? "Connecting…" : "Backend unavailable";
  const showVoiceIntro = route === "voice" && currentTab === "Recordings" && dataMode !== "demo";
  const showCommunicationsIntro = route === "communications" && dataMode === "demo";
  const visibleTabs = route === "communications" && dataMode === "demo" ? ["Inbox", "Sent", "Drafts", "Connected accounts"] : route === "security" && dataMode === "demo" ? ["Overview", "Access", "Sessions", "Audit"] : definition.tabs;
  return <motion.div className={`p-workspace p-workspace--${route}`} initial={{ opacity: 0, y: 7 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.22, ease: "easeOut" }}>
    <header className="p-titlebar">
      <div><span className="p-kicker"><Layers3 className="icon xs" aria-hidden="true" />Advanced tools / {definition.title}</span><h1>{definition.title}</h1><p>{definition.description}</p></div>
      <div className="p-header-actions"><span className="p-fixture" role="status">{dataMode === "demo" ? <Info className="icon xs" aria-hidden="true" /> : !endpointMap || liveQuery.isError ? <CloudOff className="icon xs" aria-hidden="true" /> : liveQuery.isSuccess ? <CheckCircle2 className="icon xs" aria-hidden="true" /> : <Info className="icon xs" aria-hidden="true" />}{connectionLabel}</span><button ref={integrationGuideTrigger} className="btn soft" type="button" onClick={() => setIntegrationGuideOpen(true)}><ArrowRight className="icon sm" aria-hidden="true" />Integration guide</button>{dataMode === "demo" && <span className="btn sm advanced-demo-badge"><span className="advanced-demo-dot" aria-hidden="true"/>Live demo</span>}</div>
    </header>
    <ScreenTabs tabs={visibleTabs} active={currentTab} onChange={setActiveTab} />
    <div className="p-main-grid"><div className="p-primary">
      {showVoiceIntro && <section className="demo-feature-hero" aria-label="Voice workspace"><div className="demo-feature-symbol"><Mic className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">Recordings · Workspace</span><h2>Capture and understand ideas</h2><p>Record, review and edit the words that matter.</p></div></section>}
      {route === "voice" && dataMode === "demo" && currentTab !== "Recordings" && <section className="demo-feature-hero" aria-label="Voice workspace"><div className="demo-feature-symbol"><Mic className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">{currentTab} · Workspace</span><h2>Capture and understand ideas</h2><p>Record, review and edit the words that matter.</p></div></section>}
      {showCommunicationsIntro && <section className="demo-feature-hero" aria-label={`${currentTab} workspace`}><div className="demo-feature-symbol"><Mail className="icon lg" aria-hidden="true" /></div><div><span className="p-eyebrow">{currentTab} · Workspace</span><h2>Everything in one place</h2><p>Read, reply and prepare communications with clear control.</p></div></section>}
      <RouteContent route={route} tab={currentTab} definition={definition} data={liveQuery.data} query={liveQuery} onTabChange={setActiveTab} demo={dataMode === "demo"} />{route === "assistant" && dataMode === "demo" && <AssistantDemoActivity events={(liveQuery.data?.activity?.events ?? []) as JsonRecord[]} />}{isAdvanced && route !== "security" && !(route === "assistant" && dataMode === "demo") && !(route === "twin" && dataMode === "demo") && !(["developer", "engine", "science", "scraper"].includes(route) && dataMode === "demo") && <>{route !== "vault" && route !== "finetune" && <><LiveData route={route} tab={currentTab} query={liveQuery} supported={Boolean(endpointMap)} /><BackendActions route={route} tab={currentTab} data={liveQuery.data} /></>}</>}
    </div><ConnectionRail definition={definition} route={route} tab={currentTab} connectionLabel={connectionLabel} data={liveQuery.data} onTabChange={setActiveTab} /></div>
    <Dialog.Root open={integrationGuideOpen} onOpenChange={open => { setIntegrationGuideOpen(open); if (!open) requestAnimationFrame(() => integrationGuideTrigger.current?.focus()); }}>
      <Dialog.Portal>
        <Dialog.Overlay />
        <Dialog.Content className="advanced-integration-dialog">
          <div className="advanced-integration-heading"><Dialog.Title className="second-app-dialog-title">Demo and integration guide</Dialog.Title><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close integration guide"><X size={16} aria-hidden="true"/></button></Dialog.Close></div>
          <Dialog.Description className="second-app-dialog-description">{dataMode === "demo" ? "Every interaction in this reference runs against local example records. Nothing calls your backend or external accounts." : "This workspace uses connected backend services where available. Sample data export is disabled in Live mode."}</Dialog.Description>
          <div className="p-note advanced-integration-note"><Info className="icon sm" aria-hidden="true"/><span>{dataMode === "demo" ? "Demo mode is offline. Changes apply only to isolated example data stored in this browser." : endpointMap ? "A backend read route is configured for this tab. Available operations continue to follow your authenticated workspace permissions." : "No backend read route is configured for this tab. Live data and unsupported operations remain unavailable."}</span></div>
          <div className="demo-detail-grid"><div><small>Current screen</small><strong>{route}</strong></div><div><small>Record collection</small><strong>{({ knowledge: { Search: "documents", Collections: "collections", Ingestion: "documents", "Retrieval details": "documents" }, voice: { Recordings: "recordings", Transcripts: "recordings", "Voice settings": "settings" }, assistant: { Today: "tasks", Tasks: "tasks", Goals: "goals", "Daily brief": "tasks" }, communications: { Inbox: "messages", Sent: "messages", Drafts: "messages", "Connected accounts": "accounts" }, security: { Overview: "policies", Access: "policies", Sessions: "sessions", Audit: "audit" }, developer: { Connections: "connections", "API keys": "keys", Webhooks: "webhooks", MCP: "mcp", Activity: "connections" }, engine: { Overview: "jobs", Hardware: "hardware", Jobs: "jobs", Plans: "plans" }, science: { Projects: "projects", Experiments: "experiments", Data: "datasets", Reports: "reports" }, twin: { Profile: "preferences", Preferences: "preferences", Sources: "sources", "Data controls": "history" }, vault: { Secrets: "secrets", Access: "requests", Audit: "audit" }, scraper: { "New extraction": "jobs", Jobs: "jobs", Results: "results", Rules: "rules" }, finetune: { Datasets: "datasets", Jobs: "jobs", Checkpoints: "checkpoints", Results: "results" } } as Record<string, Record<string, string>>)[route]?.[currentTab] ?? "items"}</strong></div><div><small>Privacy</small><strong>{dataMode === "demo" ? "Synthetic local examples" : "Backend permissions apply"}</strong></div><div><small>Persistence</small><strong>{dataMode === "demo" ? "Browser demo storage where available" : endpointMap ? "Managed by connected service" : "No live service for this tab"}</strong></div></div>
          {guideExportError && <p className="caption" role="alert">{guideExportError}</p>}
          <div className="advanced-integration-actions">{dataMode === "demo" ? <button className="btn primary" type="button" onClick={async () => { setGuideExportError(""); try { const data = await advancedDemoScreen(route, currentTab); const safeRoute = route.replace(/[^a-z0-9-]/gi, "-"); const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }); const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = `openagent-demo-${safeRoute}.json`; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 0); } catch (cause) { setGuideExportError(cause instanceof Error ? `Sample export failed: ${cause.message}` : "Sample export failed. Try again."); } }}><FileDown className="icon sm" aria-hidden="true"/>Export sample data</button> : <p className="caption" role="status">Live export is unavailable here. No backend export route is configured for this guide.</p>}</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  </motion.div>;
}

