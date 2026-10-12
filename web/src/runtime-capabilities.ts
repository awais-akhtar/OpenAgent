export type CapabilityState =
  | "ready"
  | "setup-needed"
  | "degraded"
  | "sandbox"
  | "experimental"
  | "unavailable"
  | "certified"
  | "unknown";

export interface RuntimeCapability {
  id: string;
  label: string;
  state: CapabilityState;
  reason: string;
  requiredDependency: string;
  configurationAction: string;
  privacyBoundary: string;
  lastCheck: string;
  certification: string;
}

type UnknownRecord = Record<string, unknown>;

function record(value: unknown): UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value as UnknownRecord : {};
}

function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function inferState(module: UnknownRecord): CapabilityState {
  const evidence = record(module.evidence);
  const evidenceStatus = text(evidence.status).toLowerCase();
  if (["mock-completed", "simulated_local", "local_fixture", "demo", "demonstration"].includes(evidenceStatus)) return "sandbox";
  if (text(module.certification) || text(evidence.certification)) return "certified";
  const capability = text(module.capability_status).toLowerCase();
  if (["available", "ready"].includes(capability)) return "ready";
  if (["setup_needed", "requires_setup"].includes(capability)) return "setup-needed";
  if (capability === "experimental") return "experimental";
  if (capability === "unavailable") return "unavailable";
  const probe = text(module.probe_status || module.status).toLowerCase();
  if (["timeout", "degraded", "error", "failed"].includes(probe)) return "degraded";
  return "unknown";
}

export function normalizeRuntimeCapabilities(payload: unknown): RuntimeCapability[] {
  const root = record(payload);
  const modules = Array.isArray(root.modules) ? root.modules : [];
  return modules.map(value => {
    const module = record(value);
    const evidence = record(module.evidence);
    return {
      id: text(module.id),
      label: text(module.label) || text(module.id),
      state: inferState(module),
      reason: text(module.reason) || text(evidence.reason),
      requiredDependency: text(module.required_dependency) || text(evidence.required_dependency),
      configurationAction: text(module.configuration_action) || text(evidence.configuration_action),
      privacyBoundary: text(module.privacy_boundary) || text(evidence.privacy_boundary),
      lastCheck: text(module.last_check) || text(root.checked_at),
      certification: text(module.certification) || text(evidence.certification),
    };
  }).filter(capability => capability.id);
}

const viewAliases: Readonly<Record<string, string>> = {
  adaptive_engine: "engine",
  conversation_organization: "chat",
  enterprise_identity: "security",
  mcp_runtime: "developer",
  memory: "knowledge",
  scheduler: "automation",
  ultra_models: "models",
};

export function applyRuntimeCapabilities(capabilities: readonly RuntimeCapability[], root: ParentNode = document): void {
  for (const capability of capabilities) {
    const view = viewAliases[capability.id] ?? capability.id;
    for (const element of root.querySelectorAll<HTMLElement>(`[data-view-link="${CSS.escape(view)}"], [data-view-panel="${CSS.escape(view)}"]`)) {
      element.dataset.capabilityState = capability.state;
      if (capability.reason) element.dataset.capabilityReason = capability.reason;
      if (element.matches("[data-view-link]") && capability.state !== "ready" && capability.state !== "unknown") {
        const description = [capability.state.replace("-", " "), capability.reason].filter(Boolean).join(": ");
        element.setAttribute("aria-description", description);
        element.title = description;
      }
    }
  }
}
