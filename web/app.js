/**
 * OpenAgent UI — Main Application Script
 * Privacy-first self-hosted AI operating system
 * Original implementation, no third-party dependencies.
 */

// =============================================
// Constants & State
// =============================================

const MODULES = [
  { id: "chat",        label: "AI Workspace",       text: "Streaming chat, provider controls, conversation memory." },
  { id: "codebase",    label: "CodeQ",               text: "Repository-aware coding with protected tools, repair loops, and validation evidence." },
  { id: "agents",      label: "Multi-Agent System",  text: "Specialists, parallel execution, delegation, verification." },
  { id: "automation",  label: "Automation",          text: "Visual workflow builder, triggers, safe actions." },
  { id: "calendar",    label: "Calendar System",     text: "CalDAV-compatible calendar, scheduling intelligence, iCal." },
  { id: "twin",        label: "Digital Twin",        text: "Personal AI memory, intent prediction, goal tracking." },
  { id: "vault",       label: "Encrypted Vault",     text: "Zero-knowledge local vault for secrets and credentials." },
  { id: "scraper",     label: "Web Scraper",         text: "Privacy-first URL scraping and scheduled ingestion." },
  { id: "finetune",    label: "Fine-Tuning",         text: "Dataset curation, LoRA adapters, training job management." },
  { id: "science",     label: "Scientific AI",       text: "Local scientific computing, engineering optimisation, verification, and materials discovery." },
];

const PALETTE_COMMANDS = [
  { label: "Dashboard",              icon: "🏠", group: "Navigate", action: () => setView("dashboard") },
  { label: "Chat",                   icon: "💬", group: "Navigate", action: () => setView("chat") },
  { label: "CodeQ",                  icon: "⌘", group: "Navigate", action: () => setView("codebase") },
  { label: "Agents",                 icon: "🤖", group: "Navigate", action: () => setView("agents") },
  { label: "Knowledge",              icon: "✦", group: "Navigate", action: () => setView("knowledge") },
  { label: "Documents",              icon: "🗂", group: "Navigate", action: () => setView("workspace") },
  { label: "Automation",             icon: "⚡", group: "Navigate", action: () => setView("automation") },
  { label: "Assistant",              icon: "📋", group: "Navigate", action: () => setView("assistant") },
  { label: "Communications",         icon: "📡", group: "Navigate", action: () => setView("communications") },
  { label: "Models",                 icon: "🧠", group: "Navigate", action: () => setView("models") },
  { label: "Security",               icon: "🔒", group: "Navigate", action: () => setView("security") },
  { label: "Adaptive Engine",        icon: "⚙", group: "Navigate", action: () => setView("engine") },
  { label: "Scientific AI",          icon: "🔬", group: "Navigate", action: () => setView("science") },
  { label: "Developer",              icon: "⚙️", group: "Navigate", action: () => setView("developer") },
  { label: "Settings",               icon: "🔧", group: "Navigate", action: () => setView("settings") },
  { label: "Calendar",               icon: "📅", group: "Navigate", action: () => setView("calendar") },
  { label: "Digital Twin",           icon: "🪞", group: "Navigate", action: () => setView("twin") },
  { label: "Encrypted Vault",        icon: "🔐", group: "Navigate", action: () => setView("vault") },
  { label: "Web Scraper",            icon: "🌐", group: "Navigate", action: () => setView("scraper") },
  { label: "Fine-Tuning",            icon: "🎯", group: "Navigate", action: () => setView("finetune") },
  { label: "New chat",               icon: "➕", group: "Actions",  action: () => { setView("chat"); newConversation(); } },
  { label: "Run agent",              icon: "▶️", group: "Actions",  action: () => { setView("agents"); runAgent(); } },
  { label: "Daily brief",            icon: "📰", group: "Actions",  action: () => { setView("assistant"); getDailyBrief(); } },
  { label: "Privacy report",         icon: "🔏", group: "Actions",  action: () => { setView("security"); loadPrivacyReport(); } },
  { label: "Smooth runtime plan",    icon: "🧠", group: "System",   action: () => { setView("models"); loadSmoothRuntime("chat"); } },
  { label: "Auto-quantize model",    icon: "⚙️", group: "System",   action: () => { setView("models"); autoQuantizeRegisteredModel(); } },
  { label: "Run hard check",         icon: "✅", group: "System",   action: () => runHardCheck() },
  { label: "Refresh everything",     icon: "🔄", group: "System",   action: () => refreshAll() },
  { label: "Dark theme",             icon: "🌑", group: "System",   action: () => applyTheme("dark") },
  { label: "Light theme",            icon: "☀️", group: "System",   action: () => applyTheme("light") },
  { label: "System theme",           icon: "💻", group: "System",   action: () => applyTheme("system") },
];

const DEFAULT_MODEL_SETTINGS = {
  chatModel: "phi4-mini",
  codeModel: "qwen2.5-coder:3b",
  researchModel: "phi4-mini",
  voiceModel: "faster-whisper:base.en",
  quantizationMode: "auto",
  targetRamGb: "",
  modelTimeoutSeconds: "35",
  parameterCountB: "",
};

const state = {
  activeView: "dashboard",
  selectedProject: "default",
  health: {},
  systemMetrics: {},
  localAiStatus: {},
  modelPolicy: {},
  moduleHealth: {},
  inferenceStatus: {},
  auditEvents: [],
  dashboardHistory: {
    cpu: [],
    memory: [],
    disk: [],
  },
  dashboardUi: {
    paused: false,
    resourceWindow: 36,
    providerFilter: "all",
    incidentFilter: "all",
  },
  dashboardSources: {
    health: false,
    metrics: false,
    providers: false,
    models: false,
    modules: false,
    inference: false,
    audit: false,
  },
  capabilities: {},
  customAgents: [],
  privacySettings: {
    localOnly: true,
    auditAll: true,
  },
  smoothRuntime: null,
  modelSetup: {},
  modelSetupJobs: [],
  localModels: [],
  integrations: {},
  personalOs: {},
  communication: {},
  calendarSync: {},
  modelSettings: { ...DEFAULT_MODEL_SETTINGS },
  agents: [],
  providers: [],
  events: [],
  workflows: [],
  conversationId: null,
  streamingEnabled: true,
  chatBusy: false,
  chatAbortController: null,
  chatAbortReason: "",
  voiceSessionId: null,
  voiceRecording: false,
  voiceMediaRecorder: null,
  voiceMediaStream: null,
  voiceAudioContext: null,
  voiceAudioSource: null,
  voiceAudioWorklet: null,
  voiceAudioProcessor: null,
  voiceAudioGain: null,
  voicePcmChunks: [],
  voiceSampleRate: 0,
  voicePeakLevel: 0,
  voiceRecordingBackend: "",
  voiceChunks: [],
  voiceSessions: [],
  voiceTarget: "voice",
  voiceReplyPending: false,
  chatVoiceMode: false,
  chatAttachments: [],
  showArchivedConversations: false,
  conversationMenuTarget: "",
  researchRunning: false,
  researchAbortController: null,
  researchSessionId: null,
  lastResearchResult: null,
  knowledge: {
    status: {},
    rag: {},
    documents: [],
    tags: [],
    timeline: [],
    filter: "",
    mode: "search",
    hasResult: false,
    selectedDocumentId: "",
  },
  ultraModel: { status: {}, manifests: [], sessions: [], plan: null, manifestId: "", sessionId: "" },
  adaptiveEngine: { status: {}, plans: [], benchmarks: [], calibrations: [], direct: {}, busy: false },
  engineRoles: [],
  engineRoleOptions: { roles: [], providers: [], models: [] },
  engineRuns: [],
  science: { status: {}, busy: false },
  codebase: {
    projects: [],
    selectedProjectId: "",
    selectedProject: null,
    selectedFile: "",
    selectedFileHash: "",
    selectedTreePath: "",
    openFiles: [],
    fileBuffers: {},
    activeRunId: "",
    activeRun: null,
    eventCursor: 0,
    events: [],
    streamController: null,
    pollTimer: null,
    liveEditToken: 0,
    liveEditQueue: Promise.resolve(),
  },
  selectedEngineRole: "chat",
  theme: "system",
  authToken: (() => { try { return sessionStorage.getItem("oa-auth-token") || ""; } catch (e) { return ""; } })(),
  authUser: null,
  selectedSkillId: "",
  phd: {
    status: {},
    profile: {},
    documents: [],
    applications: [],
    artifacts: [],
    selectedApplicationId: "",
    selectedArtifactId: "",
    generationRun: null,
    busy: false,
  },
};

const activeModelJobPolls = new Set();

// =============================================
// Utilities
// =============================================

const qs  = (sel, root = document) => root.querySelector(sel);
const qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function escapeHtml(val) {
  return String(val ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function getProject() {
  state.selectedProject = qs("#projectInput").value.trim() || "default";
  return state.selectedProject;
}

async function refreshProjectContext() {
  const selected = getProject();
  state.conversationId = null;
  state.voiceSessionId = null;
  state.knowledge.hasResult = false;
  state.knowledge.selectedDocumentId = "";
  state.knowledge.filter = "";
  const loader = VIEW_LOADERS[state.activeView];
  if (loader) await loader();
  toast("Project loaded", selected);
}

function setOutput(sel, val) {
  const el = qs(sel);
  if (!el) return;
  if (typeof val === "string") {
    el.textContent = val;
    return;
  }
  el.textContent = formatStructuredOutput(val);
}

function formatStructuredOutput(value) {
  let payload = value;
  if (payload && typeof payload === "object" && "success" in payload) {
    if (payload.success === false) {
      return `Error\n${payload.error?.message || payload.error || "The action could not be completed."}`;
    }
    if (payload.data && typeof payload.data === "object") payload = payload.data;
  }
  const lines = [];
  const label = key => String(key).replace(/_/g, " ").replace(/\b\w/g, character => character.toUpperCase());
  const write = (key, item, depth = 0) => {
    if (item === null || item === undefined || item === "") return;
    if (Array.isArray(item)) {
      if (!item.length) return;
      lines.push(`${label(key)} (${item.length})`);
      item.slice(0, 20).forEach((entry, index) => {
        if (entry && typeof entry === "object") {
          lines.push(`  ${index + 1}.`);
          Object.entries(entry).forEach(([childKey, child]) => write(childKey, child, depth + 2));
        } else {
          lines.push(`  ${index + 1}. ${String(entry)}`);
        }
      });
      return;
    }
    if (item && typeof item === "object") {
      const entries = Object.entries(item).filter(([, child]) => child !== null && child !== undefined && child !== "");
      if (!entries.length) return;
      if (key) lines.push(`${" ".repeat(depth)}${label(key)}`);
      entries.forEach(([childKey, child]) => write(childKey, child, depth + (key ? 2 : 0)));
      return;
    }
    const formatted = typeof item === "boolean" ? (item ? "Yes" : "No") : String(item);
    lines.push(`${" ".repeat(depth)}${label(key)}: ${formatted}`);
  };
  if (payload && typeof payload === "object") {
    Object.entries(payload).forEach(([key, item]) => write(key, item));
  } else {
    lines.push(String(payload ?? "No result returned."));
  }
  return lines.join("\n") || "Action completed.";
}

function toast(title, body = "") {
  const node = document.createElement("div");
  node.className = "toast";
  node.innerHTML = `<strong>${escapeHtml(title)}</strong><p>${escapeHtml(body)}</p>`;
  qs("#toastStack").appendChild(node);
  setTimeout(() => node.remove(), 4200);
}

function emptyState(text) {
  return `<div class="empty-state">${escapeHtml(text)}</div>`;
}

function listItem(title, body = "", action = "") {
  return `<article class="list-item">
    <div class="list-item-row">
      <div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(body)}</p></div>
      ${action}
    </div></article>`;
}

function readJsonInput(sel, fallback = {}) {
  const raw = qs(sel)?.value.trim() || "";
  if (!raw) return fallback;
  try {
    return JSON.parse(raw);
  } catch (e) {
    throw new Error(`${sel} must contain valid JSON.`);
  }
}

async function api(path, options = {}) {
  const authorization = state.authToken ? { Authorization: `Bearer ${state.authToken}` } : {};
  const contentHeaders = options.body instanceof FormData ? {} : { "Content-Type": "application/json" };
  const res = await fetch(path, {
    headers: { ...contentHeaders, ...authorization, ...(options.headers || {}) },
    ...options,
  });
  const text = await res.text();
  const payload = text ? JSON.parse(text) : {};
  if (!res.ok) {
    const err = payload.error;
    const message = typeof err === "string" ? err : (err?.message || `Request failed: ${res.status}`);
    throw new Error(message);
  }
  return payload;
}

// Lightweight markdown-to-HTML renderer (no external deps)
function renderMarkdown(text) {
  if (!text) return "";
  return escapeHtml(text)
    .replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code}</code></pre>`)
    .replace(/`([^`]+)`/g, (_, code) => `<code>${code}</code>`)
    .replace(/\*\*(.+?)\*\*/g, (_, t) => `<strong>${t}</strong>`)
    .replace(/\*(.+?)\*/g, (_, t) => `<em>${t}</em>`)
    .replace(/^#{3}\s+(.+)$/gm, (_, t) => `<h4>${t}</h4>`)
    .replace(/^#{2}\s+(.+)$/gm, (_, t) => `<h3>${t}</h3>`)
    .replace(/^#{1}\s+(.+)$/gm, (_, t) => `<h3>${t}</h3>`)
    .replace(/\n/g, "<br>");
}

// =============================================
// Theme management
// =============================================

const THEME_PREFERENCES = new Set(["dark", "light", "system"]);
let systemThemeListenerBound = false;

function normalizeThemePreference(theme) {
  return THEME_PREFERENCES.has(theme) ? theme : "system";
}

function resolveTheme(theme) {
  const preference = normalizeThemePreference(theme);
  if (preference !== "system") return preference;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme, { persist = true } = {}) {
  const preference = normalizeThemePreference(theme);
  const resolvedTheme = resolveTheme(preference);
  state.theme = preference;
  document.documentElement.setAttribute("data-theme", resolvedTheme);
  document.documentElement.setAttribute("data-theme-preference", preference);
  if (persist) {
    try { localStorage.setItem("oa-theme", preference); } catch(e) {}
  }
  qsa("[data-theme-select]").forEach(btn => {
    const selected = btn.dataset.themeSelect === preference;
    btn.classList.toggle("active", selected);
    btn.setAttribute("aria-pressed", String(selected));
  });
  const sel = qs("#settingsTheme");
  if (sel) sel.value = preference;
  const meta = qs('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", resolvedTheme === "dark" ? "#0b0f16" : "#f5f7fb");
}

function initTheme() {
  let saved = "system";
  try { saved = localStorage.getItem("oa-theme") || "system"; } catch(e) {}
  applyTheme(saved);
  if (!systemThemeListenerBound && window.matchMedia) {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const refresh = () => {
      if (state.theme === "system") applyTheme("system", { persist: false });
    };
    if (typeof media.addEventListener === "function") media.addEventListener("change", refresh);
    else if (typeof media.addListener === "function") media.addListener(refresh);
    systemThemeListenerBound = true;
  }
}

// =============================================
// View navigation
// =============================================

const VIEW_LOADERS = {
  dashboard:      () => Promise.allSettled([loadDashboardTelemetry(), loadAudit()]),
  chat:           () => Promise.allSettled([loadConversations(), loadProviders(), loadModelSetupStatus(), loadModelJobs()]),
  codebase:       () => loadCodebaseProjects().catch(error => toast("CodeQ unavailable", error.message)),
  voice:          () => loadVoiceStatus().catch(() => {}),
  agents:         () => loadAgents().catch(() => {}),
  knowledge:      () => loadKnowledgeWorkspace().catch(error => toast("Knowledge unavailable", error.message)),
  workspace:      () => loadWorkspaceItems().catch(() => {}),
  automation:     () => Promise.allSettled([loadWorkflows(), loadRuntimeTools()]),
  assistant:      () => Promise.allSettled([getDailyBrief()]),
  communications: () => loadCommunicationHub().catch(() => {}),
  models:         () => Promise.allSettled([loadProviders(), loadModelPolicy(), loadSmoothRuntime("chat")]),
  security:       () => Promise.allSettled([loadAudit(), loadPrivacyReport(), loadIdentityAdministration()]),
  developer:      () => Promise.allSettled([loadDeveloperFeed(), loadDeveloperSdk(), loadIntegrationStatus(), loadSettingsModels(), loadModelSetupStatus(), loadModelJobs(), loadOperationalModules(), loadUltraModelEngine(), loadSkills()]),
  engine:         () => loadEngineWorkspace().catch(() => {}),
  calendar:       () => Promise.allSettled([loadCalendarEvents(), loadCalendarSyncStatus()]),
  twin:           () => Promise.allSettled([loadTwinSuggestions(), loadTwinGoals()]),
  vault:          () => loadVaultEntries().catch(() => {}),
  scraper:        () => loadScraperJobs().catch(() => {}),
  finetune:       () => Promise.allSettled([loadFinetuneStatus(), loadAdapters()]),
  science:        () => loadScienceStatus().catch(() => {}),
};

function setView(view) {
  state.activeView = view;
  document.body.classList.toggle("codebase-mode", view === "codebase");
  qsa("[data-view-panel]").forEach(p => p.classList.toggle("active", p.dataset.viewPanel === view));
  qsa("[data-view-link]").forEach(b => b.classList.toggle("active", b.dataset.viewLink === view));
  const titleEl = qs("#screenTitle");
  const viewTitles = { phd: "PhD Studio", finetune: "Fine-Tuning", communications: "Communications" };
  if (titleEl) titleEl.textContent = viewTitles[view] || view.split("-").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
  // Auto-load view-specific data
  const loader = VIEW_LOADERS[view];
  if (loader) loader();
}

function setScienceSection(section) {
  state.science.section = section;
  qsa("[data-science-section]").forEach(link => link.classList.toggle("active", link.dataset.scienceSection === section));
  qsa("[data-science-capability]").forEach(card => card.classList.toggle("active", card.dataset.scienceCapability === section));
  const details = {
    agents: ["Scientific Agents", "Break a question into steps, select local tools, execute bounded work, inspect the result, and retry with traceback feedback when a repair function is supplied."],
    computing: ["Scientific Computing", "Run deterministic solvers, a trained DeepONet, and transparent surrogate benchmarks. FNO and PINN remain clearly labelled baseline interfaces."],
    engineering: ["Engineering Optimisation", "Define variables, bounds, objectives, and physical constraints, then compare random, differential-evolution, or local-search results with a visible evidence trail."],
    multimodal: ["Multimodal Science", "Fuse papers, equations, tables, sensor CSV, simulation output, and measured image features. A configured native vision provider can add attributed image observations."],
    verification: ["Scientific Verification", "Check finite values, equation residuals, dimensional units, constraints, and claim-to-evidence links before a result is marked reviewable."],
    materials: ["Materials Discovery", "Train a standardized bootstrap ensemble, report held-out metrics, rank candidates, and expose uncertainty as a screening signal rather than a calibrated probability."],
  };
  const detail = details[section] || details.agents;
  const title = qs("#scienceActiveTitle");
  if (title) title.textContent = detail[0];
  const detailEl = qs("#scienceSectionDetail");
  if (detailEl) detailEl.innerHTML = `<h3>${detail[0]}</h3><p>${detail[1]}</p>`;
  renderScienceWorkspace(section);
}

function renderScienceWorkspace(section) {
  const workspace = qs("#scienceSectionWorkspace");
  if (!workspace) return;
  const views = {
    agents: `<div class="form-grid"><label class="field"><span>Memory category</span><select id="scienceMemoryCategory"><option>assumption</option><option>equation</option><option>property</option><option>boundary condition</option><option>experiment</option></select></label><label class="field"><span>Name</span><input id="scienceMemoryName" placeholder="Young modulus"></label><label class="field"><span>Value</span><input id="scienceMemoryValue" placeholder="210"></label><label class="field"><span>Unit</span><input id="scienceMemoryUnit" placeholder="GPa"></label></div><div class="toolbar"><button class="btn btn-soft" data-science-action="save-memory">Save scientific memory</button><button class="btn btn-soft" data-science-action="load-memory">Load project memory</button></div>`,
    computing: `<div class="form-grid"><label class="field"><span>Simulator</span><select id="scienceSimulator"><option value="fenics">FEniCS</option><option value="openfoam">OpenFOAM</option><option value="comsol">COMSOL</option><option value="lammps">LAMMPS</option></select></label><label class="field"><span>Temperature (K)</span><input id="scienceSimulationTemperature" type="number" value="350"></label><label class="field"><span>Duration (s)</span><input id="scienceSimulationDuration" type="number" value="10"></label><label class="field"><span>Conductivity</span><input id="scienceSimulationConductivity" type="number" value="1.5" step="0.1"></label></div><div class="toolbar"><button class="btn btn-primary" data-science-action="simulate">Run simulation example</button></div><label class="field"><span>Approved local Python</span><textarea id="sciencePythonCode" rows="5">print(math.sqrt(16))</textarea></label><div class="toolbar"><button class="btn btn-soft" data-science-action="execute-code">Execute with profiling and repair</button></div>`,
    engineering: `<div class="toolbar"><button class="btn btn-primary" data-science-prompt="optimise a cantilever beam for minimum mass subject to stress and deflection limits">Optimise beam</button><button class="btn btn-soft" data-science-prompt="optimise a heat exchanger and verify the engineering constraints">Optimise heat exchanger</button><button class="btn btn-soft" data-science-prompt="optimise a lighter bridge design and verify stress and deflection">Optimise bridge</button></div>`,
    multimodal: `<label class="field"><span>Evidence files</span><input id="scienceEvidenceFiles" type="file" multiple accept=".pdf,.txt,.md,.csv,.json,image/*"></label><div class="toolbar"><button class="btn btn-primary" data-science-action="analyse-evidence">Analyse</button></div>`,
    verification: `<label class="field"><span>Result JSON</span><textarea id="scienceVerificationPayload" rows="10">{"text":"The calculated stress is within the stated limit.","values":[125000000],"units":{"stress":{"unit":"Pa","expected":"Pa"}},"residuals":[0.0],"equations":[{"expression":"stress = force/area","symbols":{"stress":125000000,"force":250000,"area":0.002}}],"claims":[{"text":"The calculated stress is within the stated limit.","evidence_ids":["calculation:stress","constraint:stress"]}],"evidence":{"calculation:stress":{"value":125000000,"unit":"Pa"},"constraint:stress":{"passed":true,"limit":250000000,"unit":"Pa"}},"citations":[],"constraints":[{"name":"stress","passed":true}]}</textarea></label><div class="toolbar"><button class="btn btn-primary" data-science-action="verify">Run verification</button></div>`,
    materials: `<div class="science-source-band"><div><span class="eyebrow">Verified public catalog</span><strong>Reproducible materials benchmarks</strong><p id="scienceMaterialsCatalogStatus" class="muted">Loading catalog metadata...</p></div><label class="field"><span>Public dataset</span><select id="scienceMaterialsCatalog"><option value="">Loading...</option></select></label><label class="toggle-row"><input id="scienceMaterialsNetworkApproved" type="checkbox"><span><strong>Approve this download</strong><small>Allow one checksum-verified HTTPS dataset request.</small></span></label><label class="toggle-row"><input id="scienceMaterialsLicenseAccepted" type="checkbox"><span><strong>Accept dataset license</strong><small>The selected catalog entry shows its upstream license and citation.</small></span></label><div class="toolbar"><button class="btn btn-soft" data-science-action="import-public-materials">Import verified dataset</button><button class="btn btn-primary" data-science-action="analyse-public-materials">Download and analyse 80/20 split</button></div></div><div class="section-divider"><span>or use a local dataset</span></div><div class="form-grid"><label class="field"><span>Dataset</span><input id="scienceMaterialsFile" type="file" accept=".csv,.json"></label><label class="field"><span>Target property</span><input id="scienceMaterialsTarget" value="strength"></label><label class="field"><span>Dataset name</span><input id="scienceMaterialsName" value="Local materials dataset"></label><label class="field"><span>Objective</span><select id="scienceMaterialsObjective"><option value="maximize">Maximise target</option><option value="minimize">Minimise target</option></select></label></div><div class="toolbar"><button class="btn btn-soft" data-science-action="save-materials">Store dataset</button><button class="btn btn-primary" data-science-action="analyse-materials">Analyse and rank</button></div>`,
  };
  workspace.innerHTML = views[section] || views.agents;
  if (section === "materials") loadPublicMaterialsCatalog().catch(error => {
    const status = qs("#scienceMaterialsCatalogStatus");
    if (status) status.textContent = `Catalog unavailable: ${error.message}`;
  });
}

// =============================================
// Dashboard
// =============================================

async function loadHealth() {
  try {
    state.health = await api("/api/health");
    renderMetrics();
    renderDashboardSystem();
  } catch(e) { console.warn("health", e); }
}

async function loadCapabilities() {
  try {
    state.capabilities = await api("/api/capabilities");
    renderMetrics();
    renderDashboardSystem();
  } catch(e) { console.warn("capabilities", e); }
}

async function loadDashboardTelemetry() {
  const moduleHealthRequest = api(`/api/modules/health?project=${encodeURIComponent(getProject())}`).then(value => {
    state.moduleHealth = value;
    state.dashboardSources.modules = true;
    renderDashboardSystem();
    return value;
  });
  const [health, metrics, providers, localAi, localModels, policy, moduleHealth, uiBuild, inference] = await Promise.allSettled([
    api("/api/health"),
    api("/api/system/metrics"),
    api("/api/models"),
    api("/api/local-ai/status"),
    api("/api/local-ai/models"),
    api("/api/model-policy"),
    moduleHealthRequest,
    api("/api/ui/build"),
    api("/api/inference/status"),
  ]);
  if (health.status === "fulfilled") state.health = health.value;
  if (metrics.status === "fulfilled") state.systemMetrics = metrics.value;
  if (providers.status === "fulfilled") state.providers = providers.value.providers || [];
  if (localAi.status === "fulfilled") state.localAiStatus = localAi.value;
  if (localModels.status === "fulfilled") state.localModels = localModels.value.models || [];
  if (policy.status === "fulfilled") state.modelPolicy = policy.value;
  if (moduleHealth.status === "fulfilled") state.moduleHealth = moduleHealth.value;
  if (uiBuild.status === "fulfilled") state.uiBuild = uiBuild.value;
  if (inference.status === "fulfilled") state.inferenceStatus = inference.value;
  state.dashboardSources = {
    health: health.status === "fulfilled",
    metrics: metrics.status === "fulfilled",
    providers: providers.status === "fulfilled",
    models: localModels.status === "fulfilled",
    modules: moduleHealth.status === "fulfilled",
    inference: inference.status === "fulfilled",
    audit: state.dashboardSources.audit,
  };
  if (metrics.status === "fulfilled") captureDashboardSample(metrics.value);
  renderMetrics();
  renderDashboardSystem();
}

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function captureDashboardSample(metrics) {
  const timestamp = Date.now();
  const append = (series, value) => {
    if (!Number.isFinite(Number(value))) return;
    series.push({ at: timestamp, value: Math.max(0, Math.min(100, Number(value))) });
    if (series.length > 36) series.splice(0, series.length - 36);
  };
  append(state.dashboardHistory.cpu, metrics.cpu?.usage_percent);
  append(state.dashboardHistory.memory, metrics.memory?.used_percent);
  append(state.dashboardHistory.disk, metrics.disk?.used_percent);
  const updated = qs("#dashboardLastUpdated");
  if (updated) updated.textContent = `Updated ${new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
}

function chartPoints(series, width = 600, height = 180, padding = 12, maximum = 100) {
  if (!series.length) return "";
  const usableWidth = width - padding * 2;
  const usableHeight = height - padding * 2;
  return series.map((item, index) => {
    const x = series.length === 1 ? width - padding : padding + (index / (series.length - 1)) * usableWidth;
    const y = height - padding - (Math.max(0, Math.min(maximum, finiteNumber(item.value))) / maximum) * usableHeight;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

function telemetrySvg(series, { label, cssClass = "chart-line-primary", maximum = 100, width = 600, height = 180 } = {}) {
  if (!series.length) return "";
  const points = chartPoints(series, width, height, 12, maximum);
  const last = series[series.length - 1];
  const lastPoint = points.split(" ").at(-1).split(",");
  return `
    <polyline class="${cssClass}" points="${points}" vector-effect="non-scaling-stroke"></polyline>
    <circle class="${cssClass} chart-endpoint" cx="${lastPoint[0]}" cy="${lastPoint[1]}" r="3"></circle>
    <title>${escapeHtml(label)}: ${escapeHtml(String(last.value))}</title>
  `;
}

function renderDashboardResourceChart() {
  const chart = qs("#dashboardResourceChart");
  if (!chart) return;
  const windowSize = state.dashboardUi.resourceWindow;
  const cpu = state.dashboardHistory.cpu.slice(-windowSize);
  const memory = state.dashboardHistory.memory.slice(-windowSize);
  const latestCpu = cpu.at(-1)?.value;
  const latestMemory = memory.at(-1)?.value;
  renderDashboardPressure();
  if (!cpu.length && !memory.length) {
    chart.setAttribute("aria-label", "CPU and memory telemetry are unavailable.");
    chart.innerHTML = `<div class="chart-empty"><strong>Telemetry unavailable</strong><span>No numeric system sample was returned.</span></div>`;
    return;
  }
  chart.setAttribute("aria-label", `CPU ${latestCpu == null ? "unavailable" : formatPercent(latestCpu)} and memory ${latestMemory == null ? "unavailable" : formatPercent(latestMemory)}. ${Math.max(cpu.length, memory.length)} live samples retained in this browser session.`);
  chart.innerHTML = `
    <svg viewBox="0 0 600 180" preserveAspectRatio="none" aria-hidden="true" focusable="false">
      <path class="chart-grid-line" d="M12 12H588 M12 51H588 M12 90H588 M12 129H588 M12 168H588"></path>
      ${telemetrySvg(cpu, { label: "CPU percent", cssClass: "chart-line-cpu" })}
      ${telemetrySvg(memory, { label: "Memory percent", cssClass: "chart-line-memory" })}
    </svg>
    <div class="chart-axis" aria-hidden="true"><span>100%</span><span>50%</span><span>0%</span></div>
  `;
}

function pressureLevel(value) {
  if (!Number.isFinite(value)) return { label: "Unavailable", className: "badge-soft" };
  if (value >= 90) return { label: "Critical", className: "badge-danger" };
  if (value >= 75) return { label: "High", className: "badge-warn" };
  if (value >= 55) return { label: "Elevated", className: "badge-info" };
  return { label: "Nominal", className: "badge-success" };
}

function renderDashboardPressure() {
  const metrics = state.systemMetrics || {};
  const values = [
    ["CPU", Number(metrics.cpu?.usage_percent)],
    ["Memory", Number(metrics.memory?.used_percent)],
    ["Disk", Number(metrics.disk?.used_percent)],
  ].filter(([, value]) => Number.isFinite(value));
  const pressure = values.length ? Math.max(...values.map(([, value]) => value)) : NaN;
  const level = pressureLevel(pressure);
  const badge = qs("#dashboardPressureBadge");
  if (badge) {
    badge.textContent = Number.isFinite(pressure) ? `${level.label} · ${pressure.toFixed(0)}%` : level.label;
    badge.className = `badge ${level.className}`;
  }
  const grid = qs("#dashboardResourcePressure");
  if (!grid) return;
  grid.innerHTML = values.length ? values.map(([label, value]) => `
    <div class="pressure-item">
      <span>${escapeHtml(label)}</span><strong>${formatPercent(value)}</strong>
      <i aria-hidden="true"><b style="width:${Math.max(0, Math.min(100, value))}%"></b></i>
    </div>
  `).join("") : emptyState("Resource pressure is unavailable because no numeric OS metrics were returned.");
  grid.setAttribute("aria-label", values.length
    ? `Current resource pressure is ${level.label.toLowerCase()} at ${pressure.toFixed(0)} percent, based on the highest measured CPU, memory, or disk utilization.`
    : "Current resource pressure is unavailable.");
}

function inferenceThroughputSeries(events) {
  const timed = events
    .map(event => ({ at: finiteNumber(event.at) * 1000, value: finiteNumber(event.latency_ms) }))
    .filter(event => event.at > 0)
    .sort((a, b) => a.at - b.at);
  return timed.map(event => ({
    at: event.at,
    value: timed.filter(candidate => candidate.at <= event.at && candidate.at > event.at - 60000).length,
  }));
}

function renderDashboardInference() {
  const status = state.inferenceStatus || {};
  const events = Array.isArray(status.last_events) ? status.last_events : [];
  const latencySeries = events
    .filter(event => Number.isFinite(Number(event.latency_ms)))
    .map(event => ({ at: finiteNumber(event.at) * 1000, value: finiteNumber(event.latency_ms) }));
  const p50 = status.latency?.p50_ms;
  const firstToken = status.latency?.first_token_p50_ms;
  const p95 = status.latency?.p95_ms;
  const throughputSeries = inferenceThroughputSeries(events);
  const latestThroughput = throughputSeries.at(-1)?.value;
  const maximum = latencySeries.length ? Math.max(1, ...latencySeries.map(item => item.value)) : 1;
  const chart = qs("#dashboardInferenceChart");
  const badge = qs("#dashboardInferenceBadge");
  const stats = qs("#dashboardInferenceStats");
  if (!state.dashboardSources.inference) {
    if (badge) { badge.textContent = "Unavailable"; badge.className = "badge badge-soft"; }
    if (chart) {
      chart.setAttribute("aria-label", "Model inference telemetry is unavailable.");
      chart.innerHTML = `<div class="chart-empty"><strong>Telemetry unavailable</strong><span>The inference status endpoint did not return data.</span></div>`;
    }
    if (stats) stats.innerHTML = "";
    return;
  }
  if (badge) {
    badge.textContent = latencySeries.length ? `Warm · ${latencySeries.length} sample${latencySeries.length === 1 ? "" : "s"}` : "Cold";
    badge.className = `badge ${latencySeries.length ? "badge-success" : "badge-soft"}`;
  }
  if (chart) {
    chart.setAttribute("aria-label", latencySeries.length
      ? `${latencySeries.length} measured inference requests. Median response latency ${p50 ?? "unavailable"} milliseconds and current measured request rate ${latestThroughput ?? 0} per minute.`
      : "No measured model inference requests are available yet.");
    chart.innerHTML = latencySeries.length ? `
      <div class="inference-trend-band"><span>Latency · max ${maximum.toFixed(0)} ms</span><svg viewBox="0 0 600 82" preserveAspectRatio="none" aria-hidden="true" focusable="false"><path class="chart-grid-line" d="M12 12H588 M12 41H588 M12 70H588"></path>${telemetrySvg(latencySeries, { label: "Response latency milliseconds", cssClass: "chart-line-latency", maximum, height: 82 })}</svg></div>
      <div class="inference-trend-band"><span>Completed requests/min · max ${Math.max(1, ...throughputSeries.map(item => item.value))}</span><svg viewBox="0 0 600 82" preserveAspectRatio="none" aria-hidden="true" focusable="false"><path class="chart-grid-line" d="M12 12H588 M12 41H588 M12 70H588"></path>${telemetrySvg(throughputSeries, { label: "Completed requests per minute", cssClass: "chart-line-throughput", maximum: Math.max(1, ...throughputSeries.map(item => item.value)), height: 82 })}</svg></div>
    ` : `<div class="chart-empty"><strong>No measured generation yet</strong><span>Latency appears after a model request completes.</span></div>`;
  }
  if (stats) stats.innerHTML = [
    ["p50 response", p50 == null ? "—" : `${finiteNumber(p50).toFixed(0)} ms`],
    ["p95 response", p95 == null ? "—" : `${finiteNumber(p95).toFixed(0)} ms`],
    ["p50 first token", firstToken == null ? "—" : `${finiteNumber(firstToken).toFixed(0)} ms`],
    ["requests/min", latestThroughput == null ? "—" : String(latestThroughput)],
  ].map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
}

function renderMetrics() {
  const metrics = state.systemMetrics || {};
  const memory = metrics.memory || {};
  const cpu = metrics.cpu || {};
  const disk = metrics.disk || {};
  const sources = state.dashboardSources || {};
  const cpuValue = Number(cpu.usage_percent);
  const memoryValue = Number(memory.used_percent);
  const diskValue = Number(disk.used_percent);
  const inferenceEvents = Array.isArray(state.inferenceStatus?.last_events) ? state.inferenceStatus.last_events : [];
  const inferenceRate = inferenceThroughputSeries(inferenceEvents).at(-1)?.value || 0;
  const activeInference = finiteNumber(state.inferenceStatus?.activity?.active_requests);
  const inferenceCapacity = Math.max(1, finiteNumber(state.inferenceStatus?.activity?.capacity, 1));
  const inferenceLoad = Number.isFinite(Number(state.inferenceStatus?.activity?.load_percent))
    ? finiteNumber(state.inferenceStatus.activity.load_percent)
    : Math.min(100, activeInference / inferenceCapacity * 100);
  const inferenceP50 = Number(state.inferenceStatus?.latency?.p50_ms);
  const items = [
    ["CPU", cpuValue, `${cpu.count || 1} threads`],
    ["RAM", memoryValue, Number.isFinite(Number(memory.available_gb)) ? `${formatGb(memory.available_gb)} free` : "Local memory"],
    ["Disk", diskValue, Number.isFinite(Number(disk.free_gb)) ? `${formatGb(disk.free_gb)} free` : "Data volume"],
    ["AI load", inferenceLoad, activeInference > 0
      ? `${activeInference} active · capacity ${inferenceCapacity}`
      : Number.isFinite(inferenceP50) ? `Idle · ${Math.round(inferenceP50)} ms p50` : "Idle · no measured request"],
  ];
  const el = qs("#metricGrid");
  if (el) el.innerHTML = items.map(([label, rawValue, detail]) => {
    const available = label === "AI load" ? sources.inference : sources.metrics;
    const value = available && Number.isFinite(rawValue) ? Math.max(0, Math.min(100, rawValue)) : 0;
    const circumference = 2 * Math.PI * 48;
    const offset = circumference * (1 - value / 100);
    return `<article class="gauge-card" style="--gauge-value:${value}">
      <div class="gauge-dial"><svg viewBox="0 0 112 112" aria-hidden="true"><circle class="gauge-track" cx="56" cy="56" r="48"></circle><circle class="gauge-value" cx="56" cy="56" r="48" stroke-dasharray="${circumference.toFixed(2)}" stroke-dashoffset="${offset.toFixed(2)}"></circle></svg><strong>${available && Number.isFinite(rawValue) ? `${rawValue.toFixed(0)}%` : "—"}</strong></div>
      <div><span>${escapeHtml(label)}</span><small>${escapeHtml(detail)}</small></div>
    </article>`;
  }).join("");
  renderDashboardResourceChart();
  renderDashboardInference();
}

function renderDashboardSystem() {
  const metrics = state.systemMetrics || {};
  const platform = metrics.platform || {};
  const memory = metrics.memory || {};
  const disk = metrics.disk || {};
  const cpu = metrics.cpu || {};
  const health = state.health || {};
  const policy = state.modelPolicy || {};
  const providers = state.providers || [];
  const models = state.localModels || [];
  const moduleHealth = state.moduleHealth || {};
  const uiBuild = state.uiBuild || {};
  const availableProviders = providers.filter(p => p.id !== "auto" && p.available);
  const runtimeSummary = qs("#dashboardRuntimeSummary");
  if (runtimeSummary) {
    runtimeSummary.textContent = `${platform.system || "Local OS"} ${platform.release || ""} · ${cpu.count || 1} CPU threads · ${formatGb(memory.total_gb)} RAM · data at ${health.database || disk.path || "local store"}`;
  }
  const modelSummary = qs("#dashboardModelSummary");
  if (modelSummary) {
    modelSummary.textContent = state.dashboardSources.providers && state.dashboardSources.models
      ? `${availableProviders.length} live providers · ${models.length} registered local models · ${policy.routing_policy || "local-first"} route`
      : "Provider or model inventory is unavailable.";
  }
  const systemGrid = qs("#dashboardSystemGrid");
  if (systemGrid) {
    const rows = [
      ["Platform", `${platform.system || "Unknown"} ${platform.release || ""}`.trim()],
      ["CPU threads", cpu.count || 1],
      ["RAM", `${formatGb(memory.used_gb)} / ${formatGb(memory.total_gb)}`],
      ["Data disk", `${formatGb(disk.free_gb)} free`],
      ["UI build", uiBuild.version || "checking"],
      ["Database", health.database || "local SQLite"],
    ];
    systemGrid.innerHTML = rows.map(([label, value]) => `
      <div class="system-row">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(String(value))}</strong>
      </div>
    `).join("");
  }
  const modelGrid = qs("#dashboardModelGrid");
  if (modelGrid) {
    const filter = state.dashboardUi.providerFilter;
    const filteredProviders = providers
      .filter(p => p.id !== "auto")
      .filter(p => filter === "all" || (filter === "ready" && p.available) || (filter === "local" && p.local) || (filter === "external" && !p.local));
    const providerItems = filteredProviders
      .map(p => {
        const installed = Array.isArray(p.installed_models) ? p.installed_models : [];
        const modelDetail = p.id === "ollama" && installed.length
          ? `${p.selected_model || installed[0]} · ${installed.length} installed`
          : p.default_model || "policy";
        return `<div class="provider-status-row">
          <i class="provider-indicator ${p.available ? "online" : "offline"}" aria-hidden="true"></i>
          <div><strong>${escapeHtml(p.label || p.id)}</strong><span>${escapeHtml(modelDetail)}</span></div>
          <small>${escapeHtml(p.local ? "LOCAL" : "EXTERNAL")}</small>
        </div>`;
      });
    const modelItems = filter === "all" || filter === "local" ? models.slice(0, 3).map(m => `<div class="provider-status-row model-record">
      <i class="provider-indicator ${m.status === "failed" ? "offline" : "standby"}" aria-hidden="true"></i>
      <div><strong>${escapeHtml(m.name || m.id)}</strong><span>${escapeHtml(m.provider || "local")} · ${escapeHtml(m.status || "registered")}</span></div>
      <small>${escapeHtml(m.metadata?.quantization || m.metadata?.tier || "MODEL")}</small>
    </div>`) : [];
    modelGrid.innerHTML = [...providerItems, ...modelItems].join("") || emptyState("No model providers or registered local models yet.");
  }
  const readiness = qs("#dashboardProviderReadiness");
  if (readiness) {
    const listed = providers.filter(provider => provider.id !== "auto");
    const local = listed.filter(provider => provider.local);
    const external = listed.filter(provider => !provider.local);
    readiness.innerHTML = [
      ["Local ready", local.filter(provider => provider.available).length, local.length],
      ["External ready", external.filter(provider => provider.available).length, external.length],
      ["Unavailable", listed.filter(provider => !provider.available).length, listed.length],
      ["Registered models", models.length, models.length],
    ].map(([label, value, total]) => `<div><span>${escapeHtml(label)}</span><strong>${value}${label === "Registered models" ? "" : ` / ${total}`}</strong></div>`).join("");
  }
  const healthBadge = qs("#dashboardModuleHealthBadge");
  if (healthBadge) {
    const counts = moduleHealth.counts || {};
    healthBadge.textContent = state.dashboardSources.modules ? `${counts.ready || 0} ready` : "Unavailable";
    healthBadge.className = `badge ${!state.dashboardSources.modules ? "badge-soft" : (counts.error || 0) ? "badge-danger" : (counts.degraded || 0) ? "badge-warn" : "badge-success"}`;
  }
  const healthGrid = qs("#dashboardModuleHealth");
  if (healthGrid) {
    const modules = moduleHealth.modules || [];
    const ordered = [...modules].sort((a, b) => (a.status === "ready") - (b.status === "ready"));
    healthGrid.innerHTML = ordered.slice(0, 9).map(module => `
      <div class="module-health-row">
        <i class="health-${escapeHtml(module.status || "unknown")}" aria-hidden="true"></i>
        <span>${escapeHtml(module.label || module.id)}</span>
        <small>${escapeHtml(module.status || "unknown")}</small>
      </div>
    `).join("") || emptyState("Module health has not been loaded yet.");
  }
  const healthChart = qs("#dashboardHealthChart");
  if (healthChart) {
    if (!state.dashboardSources.modules) {
      healthChart.setAttribute("aria-label", "Module health telemetry is unavailable.");
      healthChart.innerHTML = `<div class="chart-empty"><strong>Health unavailable</strong><span>The module health endpoint did not return data.</span></div>`;
    } else {
      const counts = moduleHealth.counts || {};
      const total = Math.max(1, finiteNumber(counts.total) || ((moduleHealth.modules || []).length));
      const ready = finiteNumber(counts.ready);
      const degraded = finiteNumber(counts.degraded);
      const error = finiteNumber(counts.error);
      healthChart.setAttribute("aria-label", `${ready} ready, ${degraded} degraded, and ${error} failed modules out of ${total}.`);
      healthChart.innerHTML = `
        <div class="health-bar" aria-hidden="true">
          <i class="ready" style="width:${ready / total * 100}%"></i>
          <i class="degraded" style="width:${degraded / total * 100}%"></i>
          <i class="error" style="width:${error / total * 100}%"></i>
        </div>
        <div class="health-counts"><span><b>${ready}</b> Ready</span><span><b>${degraded}</b> Degraded</span><span><b>${error}</b> Failed</span></div>
      `;
    }
  }
  const privacy = qs("#privacyBoundary");
  if (privacy) privacy.textContent = metrics.privacy_boundary || "System telemetry remains on this machine.";
}

async function loadIntelligenceOverview() {
  const cachedHealth = Array.isArray(state.moduleHealth?.modules) && state.moduleHealth.modules.length
    ? Promise.resolve(state.moduleHealth)
    : api(`/api/modules/health?project=${encodeURIComponent(getProject())}`);
  const [moduleHealth, agents, workspace, workflows] = await Promise.allSettled([
    cachedHealth,
    api("/api/agents"),
    api(`/api/workspace/items?project=${encodeURIComponent(getProject())}&limit=6`),
    api(`/api/workflows?project=${encodeURIComponent(getProject())}&limit=6`),
  ]);
  const health = moduleHealth.status === "fulfilled" ? moduleHealth.value : {};
  if (moduleHealth.status === "fulfilled") state.moduleHealth = health;
  const modules = health.modules || [];
  const agentsData = agents.status === "fulfilled" ? agents.value.agents || [] : [];
  const workspaceItems = workspace.status === "fulfilled" ? workspace.value.items || [] : [];
  const workflowItems = workflows.status === "fulfilled" ? workflows.value.workflows || [] : [];
  const ready = (health.counts || {}).ready || 0;
  const summary = qs("#intelligenceSummary");
  if (summary) {
    summary.textContent = `${ready} runtime modules ready · ${agentsData.length} built-in agents · ${workspaceItems.length} recent workspace items · ${workflowItems.length} workflows loaded.`;
  }
  const readiness = qs("#intelligenceReadiness");
  if (readiness) {
    readiness.textContent = `${agentsData.length || 0} specialists ready for local work`;
  }
  const moduleById = Object.fromEntries(modules.map(module => [module.id, module]));
  const cards = [
    {
      id: "agents",
      title: "Multi-agent runtime",
      body: `${agentsData.length} specialists with delegation and verification.`,
      status: moduleById.agents?.status || "ready",
      action: "Open Agents",
      view: "agents",
    },
    {
      id: "knowledge",
      title: "Knowledge and memory",
      body: "Hybrid search, semantic memory, graph, and synthesis tools.",
      status: moduleById.knowledge?.status || "ready",
      action: "Open Knowledge",
      view: "knowledge",
    },
    {
      id: "workspace",
      title: "Workspace system",
      body: `${workspaceItems.length} recent items available in this project.`,
      status: moduleById.workspace?.status || "ready",
      action: "Open Workspace",
      view: "workspace",
    },
    {
      id: "automation",
      title: "Workflow automation",
      body: `${workflowItems.length} workflows ready for local execution.`,
      status: moduleById.automation?.status || "ready",
      action: "Open Automation",
      view: "automation",
    },
  ];
  const grid = qs("#intelligenceGrid");
  if (grid) {
    grid.innerHTML = cards.map(card => {
      const badgeClass = card.status === "ready" ? "badge-success" : card.status === "degraded" ? "badge-warning" : "badge-danger";
      return `
        <article class="intelligence-card">
          <div>
            <span class="badge ${badgeClass}">${escapeHtml(card.status)}</span>
            <h3>${escapeHtml(card.title)}</h3>
            <p>${escapeHtml(card.body)}</p>
          </div>
          <button class="btn btn-soft" data-view-jump="${escapeHtml(card.view)}">${escapeHtml(card.action)}</button>
        </article>
      `;
    }).join("");
  }
}

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(number % 1 ? 1 : 0)}%` : "0%";
}

function formatGb(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? `${number.toFixed(number >= 10 ? 1 : 2)} GB` : "0 GB";
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let current = bytes;
  let index = 0;
  while (current >= 1024 && index < units.length - 1) {
    current /= 1024;
    index += 1;
  }
  return `${current >= 10 || index === 0 ? current.toFixed(0) : current.toFixed(1)} ${units[index]}`;
}

function formatDateTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// =============================================
// Chat — supports streaming and non-streaming
// =============================================

function appendChat(role, content, streaming = false) {
  const el = qs("#chatLog");
  if (!el) return null;
  const node = document.createElement("div");
  node.className = `message ${role}${streaming ? " streaming" : ""}`;
  node.innerHTML = renderMarkdown(content);
  el.appendChild(node);
  el.scrollTop = el.scrollHeight;
  return node;
}

function requestedChatArtifactFormat(message) {
  const text = String(message || "").toLowerCase();
  if (!/\b(make|create|generate|write|prepare|export|give)\b/.test(text)) return "";
  if (/\b(pdf)\b/.test(text)) return "pdf";
  if (/\b(docx|word document|word file)\b/.test(text)) return "docx";
  if (/\b(markdown|\.md)\b/.test(text)) return "md";
  if (/\b(html|web document)\b/.test(text)) return "html";
  if (/\b(text file|\.txt)\b/.test(text)) return "txt";
  return "";
}

async function attachRequestedChatArtifact(message, content, node) {
  const format = requestedChatArtifactFormat(message);
  if (!format || !content || !node) return;
  try {
    const title = String(message).replace(/\s+/g, " ").trim().slice(0, 90) || "OpenAgent document";
    const payload = await api("/api/chat/artifact", { method: "POST", body: JSON.stringify({ title, content, format, project: getProject() }) });
    const artifact = payload.artifact || {};
    const bytes = Uint8Array.from(atob(artifact.base64 || ""), character => character.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: artifact.content_type || "application/octet-stream" }));
    const link = document.createElement("a");
    link.className = "chat-artifact-download";
    link.href = url;
    link.download = artifact.filename || `openagent-document.${format}`;
    link.innerHTML = `<span>${escapeHtml(format.toUpperCase())}</span><strong>${escapeHtml(artifact.filename || "Download document")}</strong><small>${escapeHtml(formatBytes(artifact.size_bytes || bytes.length))}</small>`;
    node.appendChild(link);
    window.setTimeout(() => URL.revokeObjectURL(url), 600000);
  } catch (error) {
    toast("Document rendering failed", error.message);
  }
}

function newConversation() {
  stopActiveChat("New conversation started.");
  state.conversationId = null;
  const el = qs("#chatLog");
  if (el) el.innerHTML = "";
  loadConversations().catch(() => {});
  updateChatControls(false);
}

function isResearchRequest(message) {
  const text = String(message || "").toLowerCase();
  return /\b(research|investigate|find (?:me )?(?:the )?(?:latest|recent|newest|best|options?|cars?|products?|papers?)|compare (?:current|available|the best)|latest (?:paper|study|research)|search (?:the )?(?:web|internet|papers?)|literature review|papers? on|studies on|shop for|shortlist)\b/.test(text);
}

function researchUsesPublicWeb(message) {
  return /\b(latest|recent|newest|current|today|web|internet|paper|study|github|kaggle|arxiv|doi)\b/i.test(String(message || ""));
}

async function sendChat() {
  const input = qs("#chatInput");
  const message = input?.value.trim();
  if (!message) return;
  if (state.chatBusy) {
    stopActiveChat("Stopped so the new message can steer the conversation.");
    await new Promise(resolve => setTimeout(resolve, 80));
  }
  const provider = qs("#providerSelect")?.value || "auto";
  const configuredModel = normalizeKnownModelName(state.modelSettings.chatModel || "");
  const model = provider === "auto" ? "" : configuredModel;
  const allowExternal = !state.privacySettings.localOnly && (qs("#allowExternal")?.checked || false);
  const attachments = state.chatAttachments.map(item => ({ ...item }));
  appendChat("user", message);
  input.value = "";
  state.chatAttachments = [];
  renderChatAttachments();

  if (isResearchRequest(message)) {
    await sendChatResearch(message, provider, model, attachments);
    loadConversations().catch(() => {});
    return;
  }

  if (state.streamingEnabled) {
    await sendChatStreaming(message, provider, model, allowExternal, attachments);
  } else {
    await sendChatBlocking(message, provider, model, allowExternal, attachments);
  }
  loadConversations().catch(() => {});
}

async function sendChatStreaming(message, provider, model, allowExternal, attachments = []) {
  const node = appendChat("assistant", `Thinking with ${provider === "auto" ? "Auto Route" : provider}${model ? ` · ${model}` : ""}…`, true);
  let accum = "";
  let finalMessage = "";
  let streamError = "";
  let sseBuffer = "";
  const controller = new AbortController();
  state.chatBusy = true;
  state.chatAbortController = controller;
  state.chatAbortReason = "";
  updateChatControls(true);
  let stallTimer = null;
  const resetStallWatchdog = () => {
    if (stallTimer) clearTimeout(stallTimer);
    stallTimer = setTimeout(() => {
      controller.abort("No chat stream activity for 60 seconds.");
    }, 60000);
  };
  try {
    resetStallWatchdog();
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({ message, project: getProject(), provider, model, allow_external: allowExternal, conversation_id: state.conversationId, attachments }),
    });
    if (!res.ok) {
      let detail = "";
      try { detail = await res.text(); } catch(_) {}
      throw new Error(`Stream error: ${res.status}${detail ? ` ${detail.slice(0, 160)}` : ""}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      resetStallWatchdog();
      sseBuffer += decoder.decode(value, { stream: true });
      const lines = sseBuffer.split(/\r?\n/);
      sseBuffer = lines.pop() || "";
      for (const line of lines) handleChatSseLine(line, node, {
        get accum() { return accum; },
        set accum(value) { accum = value; },
        get finalMessage() { return finalMessage; },
        set finalMessage(value) { finalMessage = value; },
        set streamError(value) { streamError = value; },
      });
      if (streamError) throw new Error(streamError);
      const visible = accum || finalMessage;
      if (node && visible) { node.innerHTML = renderMarkdown(visible); node.scrollIntoView({ block: "nearest" }); }
    }
    if (sseBuffer.trim()) {
      handleChatSseLine(sseBuffer, node, {
        get accum() { return accum; },
        set accum(value) { accum = value; },
        get finalMessage() { return finalMessage; },
        set finalMessage(value) { finalMessage = value; },
        set streamError(value) { streamError = value; },
      });
      if (streamError) throw new Error(streamError);
      const visible = accum || finalMessage;
      if (node && visible) { node.innerHTML = renderMarkdown(visible); node.scrollIntoView({ block: "nearest" }); }
    }
    if (node && !accum && finalMessage) node.innerHTML = renderMarkdown(finalMessage);
    if (node && !accum && !finalMessage) node.innerHTML = "<em>No assistant text was returned. Try Setup Chat Model in Developer.</em>";
    if (node) {
      node.classList.remove("streaming");
      await attachRequestedChatArtifact(message, accum || finalMessage, node);
      speakPendingVoiceReply(accum || finalMessage);
    }
  } catch(e) {
    if (node) {
      node.classList.remove("streaming");
      const stoppedByUser = e.name === "AbortError" && state.chatAbortReason;
      const message = stoppedByUser ? state.chatAbortReason : e.name === "AbortError" ? "Chat stream stalled for 60 seconds." : e.message;
      node.innerHTML = stoppedByUser
        ? `<em>${escapeHtml(message)}</em>`
        : `<em>Error: ${escapeHtml(message)}. Check Chat model readiness above or open Developer > Setup Chat Model.</em>`;
    }
    if (e.name === "AbortError" && state.chatAbortReason) {
      toast("Chat stopped", state.chatAbortReason);
    } else {
      toast("Chat stream error", e.name === "AbortError" ? "No model stream activity for 60 seconds." : e.message);
    }
  } finally {
    if (stallTimer) clearTimeout(stallTimer);
    if (state.chatAbortController === controller) {
      state.chatBusy = false;
      state.chatAbortController = null;
      state.chatAbortReason = "";
      updateChatControls(false);
    }
  }
}

function handleChatSseLine(line, node, stateRef) {
  if (!line.startsWith("data:")) return;
  const raw = line.slice(5).trim();
  if (!raw || raw === "[DONE]") return;
  const obj = JSON.parse(raw);
  if (obj.type === "conversation" && obj.conversation?.id) {
    state.conversationId = obj.conversation.id;
  } else if (obj.type === "status") {
    if (node && !stateRef.accum) node.innerHTML = renderMarkdown(`Thinking locally${obj.model ? ` with ${obj.model}` : ""}…`);
  } else if (obj.type === "chunk") {
    if (obj.content) stateRef.accum += obj.content;
    if (obj.conversation_id) state.conversationId = obj.conversation_id;
  } else if (obj.type === "message") {
    stateRef.finalMessage = obj.message?.content || stateRef.finalMessage;
  } else if (obj.type === "error") {
    stateRef.streamError = obj.error || "Chat stream failed.";
  } else if (obj.content) {
    stateRef.accum += obj.content;
  }
}

async function sendChatBlocking(message, provider, model, allowExternal, attachments = []) {
  const node = appendChat("assistant", "…");
  state.chatBusy = true;
  updateChatControls(true);
  try {
    const payload = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, project: getProject(), provider, model, allow_external: allowExternal, conversation_id: state.conversationId, attachments }),
    });
    state.conversationId = payload.conversation?.id;
    const lastMsg = (payload.messages || []).at(-1);
    if (node && lastMsg) {
      node.innerHTML = renderMarkdown(lastMsg.content);
      await attachRequestedChatArtifact(message, lastMsg.content, node);
      speakPendingVoiceReply(lastMsg.content);
    }
  } catch(e) {
    if (node) node.innerHTML = `<em>Error: ${escapeHtml(e.message)}</em>`;
    toast("Chat error", e.message);
  } finally {
    state.chatBusy = false;
    updateChatControls(false);
  }
}

async function sendChatResearch(message, provider, model, attachments = []) {
  const node = appendChat("assistant", renderInlineResearchProgress(message, [], "Planning the investigation"), true);
  const controller = new AbortController();
  state.chatBusy = true;
  state.chatAbortController = controller;
  state.chatAbortReason = "";
  updateChatControls(true);
  let sessionId = "";
  try {
    const created = await api("/api/research/sessions", {
      method: "POST",
      signal: controller.signal,
      body: JSON.stringify({
        query: message,
        project: getProject(),
        max_iterations: 3,
        provider,
        model,
        allow_external: researchUsesPublicWeb(message),
      }),
    });
    sessionId = created.session?.id || "";
    if (!sessionId) throw new Error("Research did not return a session ID.");
    let session = created.session;
    while (!controller.signal.aborted) {
      const payload = await api(`/api/research/sessions/${encodeURIComponent(sessionId)}`, { signal: controller.signal });
      session = payload.session;
      const checkpoints = session.checkpoints || [];
      if (node) node.innerHTML = renderInlineResearchProgress(message, checkpoints, checkpoints.at(-1)?.message || "Collecting evidence");
      if (["completed", "failed", "cancelled"].includes(session.status)) break;
      await abortableDelay(650, controller.signal);
    }
    if (session.status !== "completed") throw new Error(session.error || `Research ${session.status}.`);
    const result = { ...session, ...(session.result || {}) };
    const report = result.report || "No source-backed report was returned.";
    const sources = result.sources || session.sources || [];
    if (node) {
      node.classList.remove("streaming");
      node.innerHTML = `${renderMarkdown(report)}${renderChatResearchSources(sources)}`;
    }
    const recorded = await api("/api/chat/research-record", {
      method: "POST",
      body: JSON.stringify({
        query: message, report, sources, research_session_id: sessionId,
        conversation_id: state.conversationId, project: getProject(),
      }),
    });
    state.conversationId = recorded.conversation_id || recorded.conversation?.id || state.conversationId;
    speakPendingVoiceReply(report);
  } catch (error) {
    if (sessionId && (error.name === "AbortError" || controller.signal.aborted)) {
      api(`/api/research/sessions/${encodeURIComponent(sessionId)}/cancel`, { method: "POST", body: "{}" }).catch(() => {});
    }
    if (node) {
      node.classList.remove("streaming");
      node.innerHTML = `<em>${escapeHtml(error.name === "AbortError" ? (state.chatAbortReason || "Research stopped.") : error.message)}</em>`;
    }
    if (error.name !== "AbortError") toast("Research failed", error.message);
  } finally {
    if (state.chatAbortController === controller) {
      state.chatBusy = false;
      state.chatAbortController = null;
      state.chatAbortReason = "";
      updateChatControls(false);
    }
  }
}

function abortableDelay(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Stopped", "AbortError"));
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, milliseconds);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function renderInlineResearchProgress(query, checkpoints, status) {
  const completed = new Set((checkpoints || []).map(item => item.stage));
  const steps = [
    ["planning", "Master agent"], ["sources", "Source agents"], ["synthesis", "Review agent"], ["completed", "Cited answer"],
  ];
  return `<div class="chat-research-run"><strong>Researching: ${escapeHtml(query)}</strong><ol>${steps.map(([stage, label], index) => {
    const done = completed.has(stage) || steps.slice(index + 1).some(([later]) => completed.has(later));
    const active = !done && index === steps.findIndex(([candidate]) => !completed.has(candidate));
    return `<li class="${done ? "done" : active ? "active" : ""}"><i>${done ? "✓" : index + 1}</i><span>${label}</span></li>`;
  }).join("")}</ol><small>${escapeHtml(status || "Working")}</small></div>`;
}

function renderChatResearchSources(sources) {
  const linked = (sources || []).filter(source => /^https?:\/\//i.test(String(source.url || "")));
  if (!linked.length) return "";
  return `<details class="chat-research-sources" open><summary>${linked.length} verified source${linked.length === 1 ? "" : "s"}</summary>${linked.map(source => `
    <a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer"><strong>${escapeHtml(source.title || source.url)}</strong><small>${escapeHtml((source.published_at || "date unavailable").slice(0, 10))} · ${escapeHtml(source.type || "web")}</small></a>
  `).join("")}</details>`;
}

function speakPendingVoiceReply(text) {
  if (!state.voiceReplyPending && !state.chatVoiceMode) return;
  state.voiceReplyPending = false;
  if (!("speechSynthesis" in window)) return;
  const spoken = String(text || "").replace(/```[\s\S]*?```/g, " code block omitted ").replace(/\[([^\]]+)\]\([^\)]+\)/g, "$1").replace(/[#*_>`~-]/g, " ").replace(/\s+/g, " ").trim();
  if (!spoken) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(spoken.slice(0, 4000));
  utterance.rate = 1.02;
  setChatVoiceState("Speaking", spoken.slice(0, 180));
  utterance.onend = () => {
    if (!state.chatVoiceMode) return;
    setChatVoiceState("Listening", "Speak when you are ready.");
    window.setTimeout(() => { if (state.chatVoiceMode && !state.voiceRecording) startVoiceRecording(); }, 350);
  };
  utterance.onerror = () => { if (state.chatVoiceMode) setChatVoiceState("Ready", "Speech playback stopped. Tap the microphone to continue."); };
  window.speechSynthesis.speak(utterance);
}

function stopActiveChat(reason = "Response stopped by user.") {
  state.chatAbortReason = reason;
  if (state.chatAbortController) {
    state.chatAbortController.abort(reason);
  }
  state.chatBusy = false;
  updateChatControls(false);
}

function updateChatControls(running = state.chatBusy) {
  const stopBtn = qs("#stopChat");
  const sendBtn = qs("#sendChat");
  if (stopBtn) stopBtn.disabled = !running;
  if (sendBtn) { sendBtn.disabled = running; sendBtn.textContent = "↑"; }
}

async function testChatModel() {
  const node = appendChat("assistant", "Testing the local model route…", true);
  try {
    const payload = await api(`/api/chat/diagnose?project=${encodeURIComponent(getProject())}`);
    const status = payload.status === "online" ? "real model online" : "offline fallback";
    const text = [
      `Chat model diagnosis: ${status}`,
      "",
      `Provider: ${payload.provider || "unknown"}`,
      `Model: ${payload.model || "unknown"}`,
      `Latency: ${payload.duration_ms || 0} ms`,
      "",
      "Model response:",
      payload.content || "(empty)",
    ].join("\n");
    if (node) {
      node.classList.remove("streaming");
      node.innerHTML = renderMarkdown(text);
    }
    toast("Model diagnosis complete", `${payload.provider || "provider"} · ${payload.model || "model"}`);
    await Promise.allSettled([loadModelSetupStatus(), loadProviders(), loadDashboardTelemetry()]);
  } catch(e) {
    if (node) {
      node.classList.remove("streaming");
      node.innerHTML = `<em>Model diagnosis failed: ${escapeHtml(e.message)}</em>`;
    }
    toast("Model diagnosis failed", e.message);
  }
}

function normalizeKnownModelName(modelName) {
  const value = String(modelName || "").trim();
  const aliases = {
    "phi4-mini:3.8b-instruct-q4_k_m": "phi4-mini",
    "phi4-mini:3.8b-instruct-q4_K_M": "phi4-mini",
    "qwen2.5-coder:3b-instruct-q4_k_m": "qwen2.5-coder:3b",
    "qwen2.5-coder:3b-instruct-q4_K_M": "qwen2.5-coder:3b",
    "gemma3:2b-it-q4_k_m": "gemma3:2b",
    "gemma3:2b-it-q4_K_M": "gemma3:2b",
    "deepseek-coder:1.3b-instruct-q4_k_m": "deepseek-coder:1.3b",
    "deepseek-coder:1.3b-instruct-q4_K_M": "deepseek-coder:1.3b",
    "llama3.2:3b-instruct-q4_k_m": "llama3.2:3b",
    "llama3.2:3b-instruct-q4_K_M": "llama3.2:3b",
    "qwen2.5:7b-instruct-q4_k_m": "qwen2.5:7b",
    "qwen2.5:7b-instruct-q4_K_M": "qwen2.5:7b",
  };
  return aliases[value] || aliases[value.toLowerCase()] || value.replace(/:latest$/i, "");
}

async function loadConversations() {
  const sort = qs("#conversationSort")?.value || "pinned";
  const archived = state.showArchivedConversations ? "true" : "false";
  const semanticQuery = sort === "semantic" ? (qs("#chatInput")?.value.trim() || "conversation") : "";
  const payload = await api(`/api/conversations/organized?project=${encodeURIComponent(getProject())}&limit=100&sort=${encodeURIComponent(sort)}&include_archived=${archived}&archived_only=${archived}&q=${encodeURIComponent(semanticQuery)}`);
  const conversations = payload.conversations || [];
  const el = qs("#conversationList");
  if (el) {
    el.innerHTML = conversations.length
      ? conversations.map(c => `
          <article class="conversation-item ${c.id === state.conversationId ? "active" : ""}" data-conversation-id="${escapeHtml(c.id)}">
            <button class="conversation-open" data-conversation-open="${escapeHtml(c.id)}" title="${escapeHtml(c.title || "Conversation")}"><strong>${c.pinned ? '<i class="pin-mark" aria-hidden="true">◆</i>' : ""}${escapeHtml(c.title || "Conversation")}</strong><span>${escapeHtml(formatDateTime(c.updated_at) || "")}</span></button>
            <button class="conversation-menu-trigger" data-conversation-menu="${escapeHtml(c.id)}" aria-label="Conversation actions" aria-haspopup="menu">•••</button>
          </article>
        `).join("")
      : `<div class="conversation-empty">No conversations</div>`;
    qsa("[data-conversation-open]", el).forEach(item => {
      item.addEventListener("click", () => loadConversation(item.dataset.conversationOpen));
    });
    qsa("[data-conversation-menu]", el).forEach(item => {
      item.addEventListener("click", event => {
        event.stopPropagation();
        openConversationActionMenu(item, conversations.find(c => c.id === item.dataset.conversationMenu));
      });
    });
  }
  const archivedButton = qs("#showArchivedConversations");
  if (archivedButton) {
    archivedButton.setAttribute("aria-pressed", String(state.showArchivedConversations));
    archivedButton.classList.toggle("active", state.showArchivedConversations);
    archivedButton.firstElementChild.textContent = state.showArchivedConversations ? "Back to chats" : "Archived";
  }
}

function closeConversationActionMenu() {
  const menu = qs("#conversationActionMenu");
  if (menu) menu.hidden = true;
  state.conversationMenuTarget = "";
}

function openConversationActionMenu(trigger, conversation) {
  const menu = qs("#conversationActionMenu");
  if (!menu || !conversation) return;
  if (!menu.hidden && state.conversationMenuTarget === conversation.id) {
    closeConversationActionMenu();
    return;
  }
  state.conversationMenuTarget = conversation.id;
  menu.innerHTML = `
    <button role="menuitem" data-action="${conversation.pinned ? "unpin" : "pin"}">${conversation.pinned ? "Unpin" : "Pin"}</button>
    <button role="menuitem" data-action="${conversation.archived ? "unarchive" : "archive"}">${conversation.archived ? "Unarchive" : "Archive"}</button>
    <button role="menuitem" class="danger" data-action="delete">Delete</button>`;
  menu.hidden = false;
  const rect = trigger.getBoundingClientRect();
  const width = 150;
  const left = Math.min(window.innerWidth - width - 10, Math.max(10, rect.right - width));
  const top = Math.min(window.innerHeight - 132, rect.bottom + 5);
  Object.assign(menu.style, { left: `${left}px`, top: `${top}px`, width: `${width}px` });
  qsa("[data-action]", menu).forEach(button => button.addEventListener("click", async () => {
    const action = button.dataset.action;
    closeConversationActionMenu();
    await runConversationAction(conversation.id, action);
  }));
}

async function uploadChatAttachments(files) {
  for (const file of Array.from(files || [])) {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("project", getProject());
    try {
      const payload = await api("/api/knowledge/upload", { method: "POST", body: form });
      state.chatAttachments.push(payload.attachment || { name: file.name, mime: file.type, size_bytes: file.size });
      renderChatAttachments();
    } catch (error) {
      toast("Attachment failed", `${file.name}: ${error.message}`);
    }
  }
  const input = qs("#chatAttachmentInput");
  if (input) input.value = "";
}

function renderChatAttachments() {
  const tray = qs("#chatAttachmentTray");
  if (!tray) return;
  tray.hidden = !state.chatAttachments.length;
  tray.innerHTML = state.chatAttachments.map((item, index) => `<span><strong>${escapeHtml(item.name || "Attachment")}</strong><button type="button" data-remove-attachment="${index}" aria-label="Remove ${escapeHtml(item.name || "attachment")}">×</button></span>`).join("");
  qsa("[data-remove-attachment]", tray).forEach(button => button.addEventListener("click", () => {
    state.chatAttachments.splice(Number(button.dataset.removeAttachment), 1);
    renderChatAttachments();
  }));
}

async function runConversationAction(conversationId, action) {
  if (action === "delete" && !window.confirm("Delete this conversation permanently?")) return;
  await api("/api/conversations/bulk", { method: "POST", body: JSON.stringify({ conversation_ids: [conversationId], action, project: getProject() }) });
  if (action === "delete" && conversationId === state.conversationId) newConversation();
  await loadConversations();
}

async function runConversationBulk() {
  const conversationIds = qsa("#conversationList .conversation-select:checked").map(item => item.value);
  if (!conversationIds.length) { toast("Select conversations first"); return; }
  const action = qs("#conversationBulkAction")?.value || "archive";
  if (action === "delete" && !window.confirm(`Delete ${conversationIds.length} conversation(s) permanently?`)) return;
  try {
    await api("/api/conversations/bulk", { method: "POST", body: JSON.stringify({ conversation_ids: conversationIds, action, project: getProject() }) });
    if (conversationIds.includes(state.conversationId) && action === "delete") newConversation();
    await loadConversations();
    toast("Conversation action complete", `${action}: ${conversationIds.length}`);
  } catch (error) { toast("Conversation action failed", error.message); }
}

async function compactCurrentConversation() {
  if (!state.conversationId) { toast("Select a conversation first"); return; }
  try {
    const payload = await api(`/api/conversations/${encodeURIComponent(state.conversationId)}/compact`, { method: "POST", body: JSON.stringify({ preserve: 8, mode: "manual" }) });
    const item = payload.compaction || {};
    toast("Context compacted", `${item.source_message_count || 0} earlier messages summarized; ${item.preserved_message_count || 0} preserved`);
  } catch (error) { toast("Compaction failed", error.message); }
}

async function showCurrentConversationBranches() {
  if (!state.conversationId) { toast("Select a conversation first"); return; }
  try {
    const graph = await api(`/api/conversations/${encodeURIComponent(state.conversationId)}/branches`);
    appendChat("assistant", [`Branch graph: ${graph.nodes?.length || 0} conversations, ${graph.edges?.length || 0} branches.`, ...(graph.edges || []).map(edge => `${edge.source.slice(0, 8)} -> ${edge.target.slice(0, 8)}`)].join("\n"));
  } catch (error) { toast("Branch graph failed", error.message); }
}

async function loadConversation(conversationId) {
  if (!conversationId) return;
  stopActiveChat("Loaded another conversation.");
  qsa("[data-conversation-id]").forEach(item => {
    item.classList.toggle("active", item.dataset.conversationId === conversationId);
  });
  const payload = await api(`/api/conversation/messages?conversation_id=${encodeURIComponent(conversationId)}`);
  state.conversationId = conversationId;
  const log = qs("#chatLog");
  const messages = payload.messages || [];
  if (log) {
    log.innerHTML = "";
    messages.forEach(message => appendChat(message.role === "user" ? "user" : "assistant", message.content || ""));
  }
  await loadConversations();
}

async function searchChatMemory() {
  const query = qs("#chatInput")?.value.trim()
    || qsa("#chatLog .message").slice(-3).map(node => node.textContent.trim()).join(" ").slice(0, 240)
    || "summary";
  const node = appendChat("assistant", "Searching local memory…", true);
  try {
    const payload = await api(`/api/events?project=${encodeURIComponent(getProject())}&q=${encodeURIComponent(query)}&limit=8`);
    const events = payload.events || [];
    const text = events.length
      ? [
          `Found ${events.length} local memory results for: ${query}`,
          "",
          ...events.map((event, index) => `${index + 1}. ${event.title || event.type || "Memory"}\n   ${String(event.content || "").slice(0, 220)}`),
        ].join("\n")
      : `No local memory results found for: ${query}`;
    if (node) {
      node.classList.remove("streaming");
      node.innerHTML = renderMarkdown(text);
    }
  } catch(e) {
    if (node) {
      node.classList.remove("streaming");
      node.innerHTML = `<em>Memory search failed: ${escapeHtml(e.message)}</em>`;
    }
    toast("Memory search failed", e.message);
  }
}

async function exportCurrentChat() {
  if (!state.conversationId) {
    toast("No conversation selected", "Send or load a conversation first.");
    return;
  }
  const node = appendChat("assistant", "Exporting current conversation…", true);
  try {
    const payload = await api(`/api/chat/export?conversation_id=${encodeURIComponent(state.conversationId)}&write_file=true`);
    const exported = payload.export || {};
    const text = [
      "Conversation exported.",
      "",
      `Messages: ${(exported.messages || []).length}`,
      `Path: ${exported.path || "export object returned"}`,
      `Conversation: ${exported.conversation?.title || state.conversationId}`,
    ].join("\n");
    if (node) {
      node.classList.remove("streaming");
      node.innerHTML = renderMarkdown(text);
    }
    toast("Conversation exported", exported.path || state.conversationId.slice(0, 8));
  } catch(e) {
    if (node) {
      node.classList.remove("streaming");
      node.innerHTML = `<em>Export failed: ${escapeHtml(e.message)}</em>`;
    }
    toast("Export failed", e.message);
  }
}

// =============================================
// Autonomous codebase workspace
// =============================================

async function loadCodebaseProjects() {
  const payload = await api(`/api/codebase/projects?project=${encodeURIComponent(getProject())}`);
  state.codebase.projects = payload.projects || [];
  const select = qs("#codebaseProjectSelect");
  if (!select) return;
  const selected = state.codebase.selectedProjectId;
  select.innerHTML = '<option value="">Open a repository</option>' + state.codebase.projects.map(project =>
    `<option value="${escapeHtml(project.id)}">${escapeHtml(project.name)}</option>`
  ).join("");
  if (selected && state.codebase.projects.some(project => project.id === selected)) select.value = selected;
  if (!state.codebase.selectedProjectId && state.codebase.projects.length) {
    select.value = state.codebase.projects[0].id;
    await selectCodebaseProject(select.value);
  }
}

async function createCodebaseProject() {
  const sourcePath = qs("#codebaseSourcePath")?.value.trim();
  if (!sourcePath) {
    toast("Repository folder required", "Enter the local folder that should be copied into an isolated workspace.");
    return;
  }
  const button = qs("#createCodebaseProject");
  if (button) button.disabled = true;
  try {
    const payload = await api("/api/codebase/projects", {
      method: "POST",
      body: JSON.stringify({
        project: getProject(),
        name: qs("#codebaseProjectName")?.value.trim() || sourcePath.split(/[\\/]/).filter(Boolean).pop(),
        source_path: sourcePath,
        backend: qs("#codebaseBackend")?.value || "local",
      }),
    });
    state.codebase.selectedProjectId = payload.project.id;
    toast("Repository ready", "CodeQ will work in a protected copy and publish verified files to this folder.");
    await loadCodebaseProjects();
    await selectCodebaseProject(payload.project.id);
  } catch (error) {
    toast("Repository could not be opened", error.message);
  } finally {
    if (button) button.disabled = false;
  }
}

async function pickCodebaseFolder(openWorkspace = true) {
  const button = qs("#codebaseFolderPicker");
  if (button) button.disabled = true;
  try {
    const payload = await api("/api/codebase/folder-picker", { method: "POST", body: "{}" });
    if (!payload.path) return;
    qs("#codebaseSourcePath").value = payload.path;
    const name = payload.path.split(/[\\/]/).filter(Boolean).pop() || "Project";
    qs("#codebaseProjectName").value = name;
    if (openWorkspace) await createCodebaseProject();
  } catch (error) {
    toast("Folder picker could not open", error.message);
  } finally {
    if (button) button.disabled = false;
  }
}

async function selectCodebaseProject(projectId) {
  if (!projectId) {
    state.codebase.selectedProjectId = "";
    state.codebase.selectedProject = null;
    qs("#codebaseIde")?.setAttribute("hidden", "");
    qs("#codebaseOpenPanel")?.removeAttribute("hidden");
    if (qs("#codebaseBranchStatus")) qs("#codebaseBranchStatus").textContent = "⑂ no repository";
    if (qs("#codebaseSandboxStatus")) qs("#codebaseSandboxStatus").textContent = "Local boundary";
    return;
  }
  const [projectPayload, treePayload] = await Promise.all([
    api(`/api/codebase/projects/${encodeURIComponent(projectId)}`),
    api(`/api/codebase/projects/${encodeURIComponent(projectId)}/tree`),
  ]);
  state.codebase.selectedProjectId = projectId;
  state.codebase.selectedProject = projectPayload.project;
  state.codebase.selectedFile = "";
  state.codebase.selectedFileHash = "";
  state.codebase.openFiles = [];
  state.codebase.fileBuffers = {};
  renderCodebaseTabs();
  qs("#codebaseIde")?.removeAttribute("hidden");
  qs("#codebaseOpenPanel")?.setAttribute("hidden", "");
  renderCodebaseMap(projectPayload.project);
  renderCodebaseTree(treePayload.entries || []);
  const terminal = qs("#codebaseTerminalOutput");
  if (terminal && terminal.textContent.trim() === "Open a repository to begin.") terminal.textContent = `${projectPayload.project.name} terminal ready.\nCommands run inside the copied workspace.`;
  const recent = await api(`/api/codebase/runs?project=${encodeURIComponent(getProject())}&limit=20`);
  const projectRuns = (recent.runs || []).filter(item => item.codebase_project_id === projectId && item.status !== "interactive");
  renderCodebaseRunHistory(projectRuns);
  const run = projectRuns[0];
  if (run) {
    state.codebase.activeRunId = run.id;
    await loadCodebaseRun(run.id, false);
  } else {
    resetCodebaseRunView();
  }
}

function renderCodebaseMap(project) {
  const map = project.map || {};
  const workspace = project.workspace || {};
  const summary = qs("#codebaseMapSummary");
  if (summary) summary.innerHTML = [
    `<strong>${escapeHtml((map.languages || []).join(", ") || "Repository")}</strong>`,
    `${Number(map.file_count || 0).toLocaleString()} files`,
    map.framework ? escapeHtml(map.framework) : "",
    map.database ? escapeHtml(map.database) : "",
    `<span title="${escapeHtml(workspace.workspace_path || "")}">${escapeHtml(workspace.sandbox_backend || "local")} sandbox · ${escapeHtml(workspace.branch || "agent branch")}</span>`,
  ].filter(Boolean).join("<br>");
  const boundary = qs("#codebaseBoundary");
  if (boundary) boundary.textContent = `Protected workspace: ${workspace.workspace_path || "local workspace"}. Verified changes publish to ${project.source_path || "the selected folder"}.`;
  const git = map.git || {};
  if (qs("#codebaseBranchStatus")) qs("#codebaseBranchStatus").textContent = `⑂ ${git.repository ? (git.branch || "detached") : (workspace.branch || "agent workspace")}`;
  if (qs("#codebaseSandboxStatus")) qs("#codebaseSandboxStatus").textContent = `${workspace.sandbox_backend || "local"} · verified publish`;
  const repositoryState = qs("#codebaseRepositoryState");
  if (repositoryState) repositoryState.textContent = git.repository
    ? `${git.branch || "detached"} · ${(git.commit || "no commit").slice(0, 10)} · ${git.dirty ? `${(git.changes || []).length} changed` : "clean"}`
    : "Workspace is not a Git repository.";
}

function renderCodebaseRunHistory(runs) {
  const select = qs("#codebaseRunSelect");
  if (!select) return;
  select.innerHTML = runs.length ? runs.map(run =>
    `<option value="${escapeHtml(run.id)}">${escapeHtml(String(run.status).replace(/_/g, " "))} · ${escapeHtml(run.user_prompt.slice(0, 42))}</option>`
  ).join("") : '<option value="">No previous runs</option>';
  if (state.codebase.activeRunId && runs.some(run => run.id === state.codebase.activeRunId)) select.value = state.codebase.activeRunId;
}

function renderCodebaseTree(entries) {
  const root = qs("#codebaseFileTree");
  if (!root) return;
  const visible = entries
    .filter(item => item.path !== ".openagent" && !item.path.startsWith(".openagent/"))
    .sort((a, b) => a.path.localeCompare(b.path));
  root.innerHTML = visible.length ? visible.slice(0, 5000).map(item =>
    `<button class="codebase-file-item ${item.type === "directory" ? "directory" : ""} ${item.path === state.codebase.selectedTreePath ? "active" : ""}" data-codebase-path="${escapeHtml(item.path)}" data-codebase-kind="${escapeHtml(item.type)}" ${item.type === "file" ? `data-codebase-file="${escapeHtml(item.path)}"` : ""} title="${escapeHtml(item.path)}" style="padding-left:${7 + Math.min(item.path.split("/").length - 1, 8) * 10}px"><span>${item.type === "directory" ? "▾" : "◇"}</span><span>${escapeHtml(item.path.split("/").pop())}</span></button>`
  ).join("") : emptyState("No readable files found in this workspace.");
  qsa("[data-codebase-path]", root).forEach(node => node.addEventListener("click", () => {
    state.codebase.selectedTreePath = node.dataset.codebasePath;
    qsa("[data-codebase-path]", root).forEach(item => item.classList.toggle("active", item === node));
    if (node.dataset.codebaseKind === "file") openCodebaseFile(node.dataset.codebasePath);
  }));
}

function codebaseParentPath(path) {
  const parts = String(path || "").split("/").filter(Boolean);
  if (parts.length && !String(path).endsWith("/")) parts.pop();
  return parts.join("/");
}

async function mutateCodebasePath(action) {
  if (!state.codebase.selectedProjectId) return;
  const selected = state.codebase.selectedTreePath || state.codebase.selectedFile;
  let path = selected;
  let destination = "";
  if (action === "create_file" || action === "create_directory") {
    const parent = selected && qs(`[data-codebase-path="${CSS.escape(selected)}"]`)?.dataset.codebaseKind === "directory" ? selected : codebaseParentPath(selected);
    const name = window.prompt(action === "create_file" ? "New file name" : "New folder name", "");
    if (!name?.trim()) return;
    path = [parent, name.trim()].filter(Boolean).join("/");
  } else {
    if (!selected) { toast("Select a file or folder first"); return; }
    if (action === "rename") {
      destination = window.prompt("New path", selected) || "";
      if (!destination.trim() || destination.trim() === selected) return;
    }
    if (action === "delete" && !window.confirm(`Delete ${selected} from the workspace and selected folder?`)) return;
  }
  try {
    await api(`/api/codebase/projects/${encodeURIComponent(state.codebase.selectedProjectId)}/paths`, {
      method: "POST",
      body: JSON.stringify({ action, path, destination }),
    });
    if (action === "delete" || action === "rename") closeCodebaseFile(selected);
    state.codebase.selectedTreePath = destination || (action.startsWith("create_") ? path : "");
    await refreshCodebaseTree();
    if (action === "create_file") await openCodebaseFile(path);
    toast("Explorer updated", destination || path);
  } catch (error) {
    toast("Explorer action failed", error.message);
  }
}

async function refreshCodebaseTree() {
  if (!state.codebase.selectedProjectId) return;
  const payload = await api(`/api/codebase/projects/${encodeURIComponent(state.codebase.selectedProjectId)}/tree`);
  renderCodebaseTree(payload.entries || []);
}

async function revealCodebaseAgentFile(path) {
  if (!path || !state.codebase.selectedProjectId) return;
  try {
    const payload = await api(`/api/codebase/projects/${encodeURIComponent(state.codebase.selectedProjectId)}/file?path=${encodeURIComponent(path)}`);
    const existing = state.codebase.fileBuffers[path];
    if (existing?.dirty) return;
    const content = payload.file.content || "";
    state.codebase.fileBuffers[path] = { content, sha256: payload.file.sha256 || "", dirty: false };
    if (!state.codebase.openFiles.includes(path)) state.codebase.openFiles.push(path);
    state.codebase.selectedFile = path;
    state.codebase.selectedFileHash = payload.file.sha256 || "";
    qs("#codebaseEditorPath").textContent = path;
    if (qs("#codebaseEditorTab")) qs("#codebaseEditorTab").textContent = path.split("/").pop();
    if (qs("#codebaseLanguageStatus")) qs("#codebaseLanguageStatus").textContent = codebaseLanguage(path);
    if (qs("#saveCodebaseFile")) qs("#saveCodebaseFile").disabled = false;
    renderCodebaseTabs();
    qsa("[data-codebase-file]").forEach(node => node.classList.toggle("active", node.dataset.codebaseFile === path));
    animateCodebaseAgentEdit(path, content);
    refreshCodebaseTree().catch(() => {});
  } catch (_) {}
}

function animateCodebaseAgentEdit(path, content) {
  const editor = qs("#codebaseEditor");
  if (!editor || state.codebase.selectedFile !== path) return;
  const token = ++state.codebase.liveEditToken;
  editor.classList.add("agent-writing");
  editor.value = "";
  let offset = 0;
  const chunk = Math.max(12, Math.ceil(content.length / 90));
  function paint() {
    if (token !== state.codebase.liveEditToken || state.codebase.selectedFile !== path) return;
    offset = Math.min(content.length, offset + chunk);
    editor.value = content.slice(0, offset);
    updateCodebaseLineNumbers();
    editor.scrollTop = editor.scrollHeight;
    if (offset < content.length) requestAnimationFrame(paint);
    else editor.classList.remove("agent-writing");
  }
  requestAnimationFrame(paint);
}

function handleCodebaseLiveEvent(event) {
  if (event.type === "tool.completed" && event.payload?.result?.path && ["write_file", "patch_file", "move_file"].includes(event.payload?.tool)) {
    state.codebase.liveEditQueue = state.codebase.liveEditQueue
      .catch(() => {})
      .then(() => revealCodebaseAgentFile(event.payload.result.path));
  }
  if (event.type === "source.publish.completed") {
    refreshCodebaseTree().catch(() => {});
    toast("Saved to selected folder", `${event.payload?.files?.length || 0} verified file(s) published.`);
  }
}

function updateCodebaseLineNumbers() {
  const editor = qs("#codebaseEditor");
  const gutter = qs("#codebaseLineNumbers");
  if (!editor || !gutter) return;
  const count = Math.max(1, editor.value.split("\n").length);
  gutter.textContent = Array.from({ length: count }, (_, index) => index + 1).join("\n");
  gutter.scrollTop = editor.scrollTop;
  updateCodebaseCursorStatus();
}

function updateCodebaseCursorStatus() {
  const editor = qs("#codebaseEditor");
  if (!editor) return;
  const before = editor.value.slice(0, editor.selectionStart || 0);
  const lines = before.split("\n");
  if (qs("#codebaseCursorStatus")) qs("#codebaseCursorStatus").textContent = `Ln ${lines.length}, Col ${(lines.at(-1) || "").length + 1}`;
}

function codebaseLanguage(path) {
  const extension = String(path || "").split(".").pop().toLowerCase();
  return ({ py: "Python", js: "JavaScript", jsx: "JavaScript React", ts: "TypeScript", tsx: "TypeScript React", json: "JSON", md: "Markdown", html: "HTML", css: "CSS", scss: "SCSS", sql: "SQL", rs: "Rust", go: "Go", java: "Java", cpp: "C++", c: "C", cs: "C#", yaml: "YAML", yml: "YAML", toml: "TOML" })[extension] || "Plain Text";
}

async function openCodebaseFile(path) {
  if (!state.codebase.selectedProjectId) return;
  try {
    rememberCurrentCodebaseBuffer();
    let buffer = state.codebase.fileBuffers[path];
    if (!buffer) {
      const payload = await api(`/api/codebase/projects/${encodeURIComponent(state.codebase.selectedProjectId)}/file?path=${encodeURIComponent(path)}`);
      buffer = { content: payload.file.content || "", sha256: payload.file.sha256 || "", dirty: false };
      state.codebase.fileBuffers[path] = buffer;
    }
    if (!state.codebase.openFiles.includes(path)) state.codebase.openFiles.push(path);
    state.codebase.selectedFile = path;
    state.codebase.selectedFileHash = buffer.sha256 || "";
    qs("#codebaseEditorPath").textContent = path;
    if (qs("#codebaseEditorTab")) qs("#codebaseEditorTab").textContent = path.split("/").pop();
    if (qs("#codebaseLanguageStatus")) qs("#codebaseLanguageStatus").textContent = codebaseLanguage(path);
    qs("#codebaseEditor").value = buffer.content || "";
    updateCodebaseLineNumbers();
    qs("#saveCodebaseFile").disabled = false;
    qsa("[data-codebase-file]").forEach(node => node.classList.toggle("active", node.dataset.codebaseFile === path));
    renderCodebaseTabs();
  } catch (error) {
    toast("File could not be opened", error.message);
  }
}

function rememberCurrentCodebaseBuffer() {
  const path = state.codebase.selectedFile;
  const editor = qs("#codebaseEditor");
  if (!path || !editor || !state.codebase.fileBuffers[path]) return;
  const buffer = state.codebase.fileBuffers[path];
  if (buffer.content !== editor.value) {
    buffer.content = editor.value;
    buffer.dirty = true;
  }
}

function renderCodebaseTabs() {
  const root = qs("#codebaseOpenTabs");
  if (!root) return;
  root.innerHTML = state.codebase.openFiles.map(path => {
    const buffer = state.codebase.fileBuffers[path] || {};
    return `<div class="codebase-editor-tab ${path === state.codebase.selectedFile ? "active" : ""}" role="tab" aria-selected="${path === state.codebase.selectedFile}" data-codebase-tab="${escapeHtml(path)}"><span class="codebase-file-dot"></span><span>${escapeHtml(path.split("/").pop())}${buffer.dirty ? " •" : ""}</span><button type="button" data-codebase-tab-close="${escapeHtml(path)}" aria-label="Close ${escapeHtml(path)}">×</button></div>`;
  }).join("");
  qsa("[data-codebase-tab]", root).forEach(tab => tab.addEventListener("click", () => openCodebaseFile(tab.dataset.codebaseTab)));
  qsa("[data-codebase-tab-close]", root).forEach(button => button.addEventListener("click", event => {
    event.stopPropagation();
    closeCodebaseFile(button.dataset.codebaseTabClose);
  }));
}

function closeCodebaseFile(path) {
  rememberCurrentCodebaseBuffer();
  const index = state.codebase.openFiles.indexOf(path);
  state.codebase.openFiles = state.codebase.openFiles.filter(item => item !== path);
  delete state.codebase.fileBuffers[path];
  if (state.codebase.selectedFile === path) {
    state.codebase.selectedFile = "";
    const next = state.codebase.openFiles[Math.min(index, state.codebase.openFiles.length - 1)];
    if (next) openCodebaseFile(next);
    else {
      qs("#codebaseEditor").value = "";
      qs("#codebaseEditorPath").textContent = "No file open";
      qs("#saveCodebaseFile").disabled = true;
      updateCodebaseLineNumbers();
    }
  }
  renderCodebaseTabs();
}

async function saveCodebaseFile() {
  if (!state.codebase.selectedProjectId || !state.codebase.selectedFile) return;
  try {
    const payload = await api(`/api/codebase/projects/${encodeURIComponent(state.codebase.selectedProjectId)}/file`, {
      method: "POST",
      body: JSON.stringify({ path: state.codebase.selectedFile, content: qs("#codebaseEditor")?.value || "", expected_sha256: state.codebase.selectedFileHash }),
    });
    state.codebase.selectedFileHash = payload.change.new_sha256 || "";
    const buffer = state.codebase.fileBuffers[state.codebase.selectedFile];
    if (buffer) {
      buffer.content = qs("#codebaseEditor")?.value || "";
      buffer.sha256 = state.codebase.selectedFileHash;
      buffer.dirty = false;
    }
    renderCodebaseTabs();
    toast("Saved", state.codebase.selectedFile);
  } catch (error) {
    toast("Save failed", error.message);
  }
}

async function scanCodebaseProject() {
  if (!state.codebase.selectedProjectId) return;
  try {
    await api(`/api/codebase/projects/${encodeURIComponent(state.codebase.selectedProjectId)}/scan`, { method: "POST", body: JSON.stringify({}) });
    await selectCodebaseProject(state.codebase.selectedProjectId);
    toast("Repository map refreshed");
  } catch (error) {
    toast("Scan failed", error.message);
  }
}

async function startCodebaseRun() {
  const prompt = qs("#codebaseGoal")?.value.trim();
  if (!state.codebase.selectedProjectId || !prompt) {
    toast("Project and goal required");
    return;
  }
  const sendButton = qs("#startCodebaseRun");
  const status = qs("#codebaseRunStatus");
  if (sendButton) sendButton.disabled = true;
  if (status) { status.textContent = "Connecting"; status.className = "badge badge-info"; }
  try {
    const preflightPayload = await api("/api/codebase/preflight", {
      method: "POST",
      body: JSON.stringify({ codebase_project_id: state.codebase.selectedProjectId, goal: prompt }),
    });
    renderCodebasePreflight(preflightPayload.preflight || {});
    const payload = await api("/api/codebase/runs", {
      method: "POST",
      body: JSON.stringify({
        codebase_project_id: state.codebase.selectedProjectId,
        prompt,
        provider: "auto",
        model: state.modelSettings.codeModel || "",
        role_models: {},
        allow_external: false,
        max_iterations: 0,
        completion: {},
      }),
    });
    state.codebase.activeRunId = payload.run.id;
    state.codebase.events = [];
    state.codebase.eventCursor = 0;
    renderCodebaseRun(payload.run);
    if (qs("#codebaseGoal")) qs("#codebaseGoal").value = "";
    startCodebaseEventStream(payload.run.id);
    scheduleCodebasePoll();
    toast("Autonomous run started", "Repository analysis and task planning are running.");
  } catch (error) {
    if (/failed to fetch|networkerror|load failed/i.test(String(error.message || error))) {
      const recovered = await recoverCodebaseRun(prompt);
      if (recovered) return;
    }
    if (sendButton) sendButton.disabled = false;
    if (status) { status.textContent = "Ready"; status.className = "badge"; }
    toast("Run could not start", /failed to fetch/i.test(String(error.message || error))
      ? "The CodeQ service disconnected. OpenAgent is still running; retry once after the workspace reconnects."
      : error.message);
  }
}

function renderCodebasePreflight(preflight) {
  const target = qs("#codebaseToolchainStatus");
  if (!target) return;
  const framework = preflight.framework?.label || preflight.profile?.framework || "Model-selected stack";
  const missing = Array.isArray(preflight.missing_toolchains) ? preflight.missing_toolchains.length : 0;
  const hostReady = preflight.host?.supported !== false;
  target.textContent = hostReady && !missing
    ? `${framework} · toolchain ready`
    : `${framework} · ${!hostReady ? "target host required" : `${missing} toolchain item${missing === 1 ? "" : "s"} missing`}`;
  target.title = (preflight.external_boundaries || []).length
    ? "External release operations require their real environment and explicit approval."
    : "Source generation and local validation are evaluated separately.";
}

async function recoverCodebaseRun(prompt) {
  await new Promise(resolve => setTimeout(resolve, 450));
  try {
    const payload = await api(`/api/codebase/runs?project=${encodeURIComponent(getProject())}&limit=20`);
    const run = (payload.runs || []).find(item =>
      item.codebase_project_id === state.codebase.selectedProjectId && item.user_prompt === prompt && item.status !== "interactive"
    );
    if (!run) return false;
    state.codebase.activeRunId = run.id;
    await loadCodebaseRun(run.id, false);
    startCodebaseEventStream(run.id);
    scheduleCodebasePoll();
    if (qs("#codebaseGoal")) qs("#codebaseGoal").value = "";
    toast("CodeQ reconnected", "The run started and its live state was recovered.");
    return true;
  } catch (_) {
    return false;
  }
}

async function cancelCodebaseRun() {
  if (!state.codebase.activeRunId) return;
  try {
    const payload = await api(`/api/codebase/runs/${encodeURIComponent(state.codebase.activeRunId)}/cancel`, { method: "POST", body: JSON.stringify({}) });
    renderCodebaseRun(payload.run);
  } catch (error) {
    toast("Stop failed", error.message);
  }
}

async function resumeCodebaseRun() {
  if (!state.codebase.activeRunId) return;
  try {
    const payload = await api(`/api/codebase/runs/${encodeURIComponent(state.codebase.activeRunId)}/resume`, { method: "POST", body: JSON.stringify({}) });
    renderCodebaseRun(payload.run);
    startCodebaseEventStream(payload.run.id);
    scheduleCodebasePoll();
    toast("Run resumed", "The agent continues from its durable task and workspace state.");
  } catch (error) {
    toast("Resume failed", error.message);
  }
}

async function restoreCodebaseSnapshot() {
  const snapshotId = qs("#codebaseSnapshotSelect")?.value;
  if (!state.codebase.activeRunId || !snapshotId) return;
  try {
    await api(`/api/codebase/runs/${encodeURIComponent(state.codebase.activeRunId)}/snapshots/${encodeURIComponent(snapshotId)}/restore`, { method: "POST", body: JSON.stringify({}) });
    await selectCodebaseProject(state.codebase.selectedProjectId);
    toast("Snapshot restored", "The copied workspace was rolled back. The source repository was not changed.");
  } catch (error) {
    toast("Restore failed", error.message);
  }
}

function scheduleCodebasePoll() {
  clearTimeout(state.codebase.pollTimer);
  state.codebase.pollTimer = setTimeout(async () => {
    if (!state.codebase.activeRunId) return;
    const run = await loadCodebaseRun(state.codebase.activeRunId, false).catch(() => null);
    if (run && !["completed", "failed", "cancelled", "blocked"].includes(run.status)) scheduleCodebasePoll();
  }, 1000);
}

async function loadCodebaseRun(runId, restartStream = true) {
  const payload = await api(`/api/codebase/runs/${encodeURIComponent(runId)}`);
  state.codebase.activeRunId = runId;
  state.codebase.activeRun = payload.run;
  renderCodebaseRun(payload.run);
  if (restartStream && !["completed", "failed", "cancelled"].includes(payload.run.status)) startCodebaseEventStream(runId);
  return payload.run;
}

function renderCodebaseRun(run) {
  state.codebase.activeRun = run;
  const status = qs("#codebaseRunStatus");
  if (status) {
    status.textContent = String(run.status || "idle").replace(/_/g, " ");
    status.className = `badge ${run.status === "completed" ? "badge-success" : run.status === "failed" ? "badge-error" : run.status === "blocked" ? "badge-warn" : "badge-info"}`;
  }
  const active = ["queued", "planning", "running", "repairing", "validating", "reviewing", "cancelling"].includes(run.status);
  if (qs("#startCodebaseRun")) qs("#startCodebaseRun").disabled = active;
  if (qs("#cancelCodebaseRun")) qs("#cancelCodebaseRun").disabled = !active;
  if (qs("#resumeCodebaseRun")) qs("#resumeCodebaseRun").disabled = !["blocked", "failed", "interrupted"].includes(run.status);
  if (qs("#codebaseRunSelect")) qs("#codebaseRunSelect").value = run.id;
  const snapshotSelect = qs("#codebaseSnapshotSelect");
  if (snapshotSelect) {
    snapshotSelect.innerHTML = (run.snapshots || []).length ? (run.snapshots || []).map(item =>
      `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label)} · ${new Date(item.created_at).toLocaleTimeString()}</option>`
    ).join("") : '<option value="">No snapshots</option>';
    qs("#restoreCodebaseSnapshot").disabled = !(run.snapshots || []).length;
  }
  renderCodebaseTasks(run.tasks || []);
  renderCodebaseEvidence(run);
  renderCodebaseApprovals(run.approvals || []);
  renderCodebaseCopilotFeed();
  const latestValidation = (run.validation || []).slice(-1)[0];
  if (latestValidation) renderCodebaseTerminal(latestValidation);
}

function resetCodebaseRunView() {
  state.codebase.activeRunId = "";
  state.codebase.activeRun = null;
  qs("#codebaseRunStatus").textContent = "Ready";
  if (qs("#startCodebaseRun")) qs("#startCodebaseRun").disabled = false;
  if (qs("#cancelCodebaseRun")) qs("#cancelCodebaseRun").disabled = true;
  renderCodebaseCopilotFeed();
  qs("#codebaseTaskGraph").innerHTML = emptyState("Start a run to create a dependency-aware task graph.");
  qs("#codebaseEventStream").innerHTML = emptyState("Agent activity will stream here.");
  qs("#codebaseEvidence").innerHTML = emptyState("Validation evidence appears only after commands run.");
  qs("#codebaseApprovals").innerHTML = "";
  if (qs("#resumeCodebaseRun")) qs("#resumeCodebaseRun").disabled = true;
  if (qs("#restoreCodebaseSnapshot")) qs("#restoreCodebaseSnapshot").disabled = true;
}

function renderCodebaseTasks(tasks) {
  const root = qs("#codebaseTaskGraph");
  if (!root) return;
  root.innerHTML = tasks.length ? tasks.map(task =>
    `<article class="codebase-task ${escapeHtml(task.status)}"><strong>${escapeHtml(task.title)}</strong><small>${escapeHtml(task.assigned_agent)} · ${escapeHtml(task.status.replace(/_/g, " "))}${task.depends_on?.length ? ` · waits for ${task.depends_on.length}` : ""}</small></article>`
  ).join("") : emptyState("Understanding the repository and creating tasks…");
}

function renderCodebaseEvents() {
  const root = qs("#codebaseEventStream");
  if (!root) return;
  const events = state.codebase.events.slice(-250).reverse();
  root.innerHTML = events.length ? events.map(event => {
    const detail = event.payload?.tool || event.payload?.agent || event.payload?.summary || event.payload?.error || "";
    return `<div class="codebase-event"><span class="codebase-event-dot"></span><div><strong>${escapeHtml(event.type.replace(/\./g, " "))}</strong>${detail ? `<small> · ${escapeHtml(String(detail).slice(0, 180))}</small>` : ""}</div><time>${escapeHtml(new Date(event.created_at).toLocaleTimeString())}</time></div>`;
  }).join("") : emptyState("Waiting for agent activity…");
  renderCodebaseCopilotFeed();
}

function codeqEventLabel(event) {
  const labels = {
    "repository.analysed": "Repository understood",
    "environment.preflight.completed": "Host and toolchain checked",
    "specialists.parallel.started": "Specialists reviewing in parallel",
    "specialists.parallel.completed": "Architecture contract ready",
    "agents.parallel.started": "Independent agents generating in parallel",
    "agents.parallel.completed": "Parallel work coordinated",
    "agent.parallel.recompute": "Overlapping edit scheduled safely",
    "plan.created": "Implementation plan ready",
    "agent.handoff": "Specialist assigned",
    "task.completed": "Step completed",
    "task.continued": "Continuing implementation",
    "repair.requested": "Repairing automatically",
    "completion.audit.completed": "Whole project audited",
    "completion.audit.needs_work": "Missing work added to the plan",
    "autonomy.redirected": "Unsafe action replaced automatically",
    "source.publish.started": "Publishing verified files",
    "source.publish.file": "Saving file to selected folder",
    "source.publish.completed": "Selected folder updated",
    "run.completed": "Work completed and verified",
    "run.failed": "Run needs attention",
    "run.cancelled": "Run stopped",
    "agent.started": "Agent started",
    "agent.thinking": "Planning the next action",
    "model.completed": "Model response received",
    "model.action_rejected": "Action adjusted to match the project",
    "model.action_replaced": "Action replaced with a working local step",
    "framework.scaffold.selected": "Framework scaffold selected",
    "tool.started": "Tool running",
    "tool.completed": "Tool completed",
    "tool.failed": "Tool failed; repair started",
    "file.modified": "File updated",
    "command.started": "Command running",
    "command.stdout": "Command produced output",
    "command.failed": "Command failed; investigating",
  };
  return labels[event.type] || event.type.replace(/\./g, " ");
}

function codeqEventDetail(event) {
  const payload = event.payload || {};
  if (payload.preflight) {
    const framework = payload.preflight.framework?.label || payload.preflight.profile?.framework || "selected stack";
    const missing = payload.preflight.missing_toolchains?.length || 0;
    return `${framework} · ${missing ? `${missing} missing toolchain item${missing === 1 ? "" : "s"}` : "local validation ready"}`;
  }
  if (payload.path) return String(payload.path);
  if (payload.tool) return String(payload.tool).replace(/_/g, " ");
  if (payload.agent) return String(payload.agent).replace(/_/g, " ");
  if (payload.command) return Array.isArray(payload.command) ? payload.command.join(" ") : String(payload.command);
  if (payload.summary) return String(payload.summary);
  return "";
}

function renderCodebaseCopilotFeed() {
  const root = qs("#codebaseCopilotFeed");
  if (!root) return;
  const run = state.codebase.activeRun;
  if (!run) {
    root.innerHTML = '<div class="copilot-welcome"><strong>Work with your repository</strong><p>Ask CodeQ to inspect, edit, test, debug, or review this project. It chooses specialist agents only when the task benefits from them.</p></div>';
    return;
  }
  const tasks = run.tasks || [];
  const hiddenNoise = new Set(["stream.heartbeat", "task.continued"]);
  const recentEvents = state.codebase.events.filter(event => !hiddenNoise.has(event.type)).reduce((items, event) => {
    const key = `${event.type}:${codeqEventDetail(event)}`;
    if (items.length && items[items.length - 1].key === key) return items;
    items.push({ key, event });
    return items;
  }, []).slice(-12).map(item => item.event);
  const validation = run.validation || [];
  const pending = (run.approvals || []).filter(item => item.status === "pending");
  root.innerHTML = `
    <article class="copilot-message user"><p>${escapeHtml(run.user_prompt || "Software engineering task")}</p><div class="copilot-message-actions"><button type="button" data-codeq-copy-prompt>Copy</button><button type="button" data-codeq-edit-prompt>Edit</button></div></article>
    <article class="copilot-message agent"><strong>${escapeHtml(String(run.status || "working").replace(/_/g, " "))}</strong>
      <div class="copilot-task-list">${tasks.map(task => `<span class="${escapeHtml(task.status)}"><i>${task.status === "completed" ? "✓" : task.status === "running" ? "●" : "○"}</i>${escapeHtml(task.title)}</span>`).join("") || "<span><i>●</i>Analysing repository</span>"}</div>
      ${recentEvents.length ? `<div class="copilot-activity">${recentEvents.map(event => { const detail = codeqEventDetail(event); return `<p><i></i><span>${escapeHtml(codeqEventLabel(event))}${detail ? `<small>${escapeHtml(detail.slice(0, 120))}</small>` : ""}</span></p>`; }).join("")}</div>` : ""}
      ${validation.length ? `<div class="copilot-validation">${validation.map(item => `<span class="${item.status === "passed" ? "passed" : "failed"}">${item.status === "passed" ? "✓" : "×"} ${escapeHtml(item.kind)}</span>`).join("")}</div>` : ""}
      ${pending.length ? `<p class="copilot-attention">Waiting for approval: ${escapeHtml(pending[0].tool)}</p>` : ""}
      ${run.summary ? `<p class="copilot-summary">${escapeHtml(run.summary)}</p>` : ""}
      ${run.error ? `<p class="copilot-error">${escapeHtml(run.error)}</p>` : ""}
    </article>`;
  root.scrollTop = root.scrollHeight;
}

async function copyCodeqPrompt() {
  const prompt = String(state.codebase.activeRun?.user_prompt || "");
  if (!prompt) return;
  try { await navigator.clipboard.writeText(prompt); }
  catch (_) {
    const input = document.createElement("textarea");
    input.value = prompt; document.body.appendChild(input); input.select(); document.execCommand("copy"); input.remove();
  }
  toast("Copied", "CodeQ request copied to the clipboard.");
}

function editCodeqPrompt() {
  const input = qs("#codebaseGoal");
  if (!input || !state.codebase.activeRun?.user_prompt) return;
  input.value = state.codebase.activeRun.user_prompt;
  input.focus();
  input.setSelectionRange(input.value.length, input.value.length);
}

function renderCodebaseEvidence(run) {
  const root = qs("#codebaseEvidence");
  if (!root) return;
  const rows = (run.validation || []).map(item =>
    `<div class="codebase-evidence-row"><span>${escapeHtml(item.kind)}</span><strong class="${item.status === "passed" ? "status-ok" : "status-error"}">${escapeHtml(item.status)} · exit ${item.exit_code}</strong></div>`
  );
  if (run.summary) rows.push(`<div class="codebase-evidence-row"><span>Final review</span><strong>${escapeHtml(run.summary)}</strong></div>`);
  if (run.error) rows.push(`<div class="codebase-evidence-row"><span>Blocking error</span><strong class="status-error">${escapeHtml(run.error)}</strong></div>`);
  root.innerHTML = rows.length ? rows.join("") : emptyState("Build, test, lint, typecheck, runtime, and review evidence will appear here.");
}

function renderCodebaseApprovals(approvals) {
  const root = qs("#codebaseApprovals");
  if (!root) return;
  const pending = approvals.filter(item => item.status === "pending");
  root.innerHTML = pending.map(item =>
    `<article class="codebase-approval"><strong>Approval required: ${escapeHtml(item.tool)}</strong><p>${escapeHtml(item.reason)}</p><div class="toolbar"><button class="btn btn-mini btn-primary" data-codebase-approval="${escapeHtml(item.id)}" data-approved="true">Approve</button><button class="btn btn-mini btn-soft" data-codebase-approval="${escapeHtml(item.id)}" data-approved="false">Reject</button></div></article>`
  ).join("");
  renderCodebaseCopilotFeed();
}

async function resolveCodebaseApproval(approvalId, approved) {
  qsa(`[data-codebase-approval="${CSS.escape(approvalId)}"]`).forEach(button => { button.disabled = true; });
  state.codebase.streamController?.abort();
  try {
    const payload = await api(`/api/codebase/approvals/${encodeURIComponent(approvalId)}/resolve`, { method: "POST", body: JSON.stringify({ approved }) });
    const resolution = payload.approval || payload;
    if (resolution.status === "failed") toast("Approved action could not run", `${resolution.result?.error || "Execution failed"}. The agent is repairing the task.`);
    else toast(approved ? "Approved" : "Rejected", approved ? "The run is resuming." : "The operation remains blocked.");
    await loadCodebaseRun(state.codebase.activeRunId);
    scheduleCodebasePoll();
  } catch (error) {
    await new Promise(resolve => setTimeout(resolve, 350));
    try {
      const run = await loadCodebaseRun(state.codebase.activeRunId);
      const recorded = (run?.approvals || []).find(item => item.id === approvalId && item.status !== "pending");
      if (recorded) {
        toast("Approval recorded", `The server recorded this action as ${recorded.status}. The run is continuing from durable state.`);
        scheduleCodebasePoll();
        return;
      }
    } catch (_) {}
    qsa(`[data-codebase-approval="${CSS.escape(approvalId)}"]`).forEach(button => { button.disabled = false; });
    toast("Approval could not be recorded", `${error.message}. The action was not retried automatically.`);
  }
}

async function startCodebaseEventStream(runId) {
  state.codebase.streamController?.abort();
  const controller = new AbortController();
  state.codebase.streamController = controller;
  const headers = state.authToken ? { Authorization: `Bearer ${state.authToken}` } : {};
  try {
    const response = await fetch(`/api/codebase/runs/${encodeURIComponent(runId)}/events/stream?after=${state.codebase.eventCursor}&timeout_seconds=120`, { headers, signal: controller.signal });
    if (!response.ok || !response.body) throw new Error(`Event stream failed: ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      lines.forEach(line => {
        if (!line.startsWith("data: ")) return;
        try {
          const event = JSON.parse(line.slice(6));
          if (event.type === "stream.heartbeat") return;
          if (Number(event.sequence) > state.codebase.eventCursor) state.codebase.eventCursor = Number(event.sequence);
          if (event.type !== "stream.closed") state.codebase.events.push(event);
          handleCodebaseLiveEvent(event);
          renderCodebaseEvents();
        } catch (_) {}
      });
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      const payload = await api(`/api/codebase/runs/${encodeURIComponent(runId)}/events?after=${state.codebase.eventCursor}`).catch(() => ({ events: [] }));
      (payload.events || []).forEach(event => {
        state.codebase.eventCursor = Math.max(state.codebase.eventCursor, Number(event.sequence || 0));
        state.codebase.events.push(event);
        handleCodebaseLiveEvent(event);
      });
      renderCodebaseEvents();
    }
  }
}

async function runCodebaseTool(tool, toolArguments = {}) {
  if (!state.codebase.activeRunId) {
    if (!state.codebase.activeProjectId) throw new Error("Open a repository first.");
    const session = await api(`/api/codebase/projects/${encodeURIComponent(state.codebase.activeProjectId)}/tools`, {
      method: "POST",
      body: JSON.stringify({ tool, arguments: toolArguments }),
    });
    state.codebase.activeRunId = session.run?.id || "";
    if (session.run) {
      state.codebase.runs = [session.run, ...state.codebase.runs.filter(item => item.id !== session.run.id)];
      renderCodebaseRunHistory(state.codebase.runs.filter(item => item.status !== "interactive"));
    }
    if (session.result?.approval_required) {
      toast("Approval required", session.result.approval_required.reason || "Review this operation in the agent panel.");
      await loadCodebaseRun(state.codebase.activeRunId, false);
      return null;
    }
    return session.result;
  }
  const payload = await api(`/api/codebase/runs/${encodeURIComponent(state.codebase.activeRunId)}/tools`, {
    method: "POST",
    body: JSON.stringify({ tool, arguments: toolArguments }),
  });
  if (payload.result?.approval_required) {
    toast("Approval required", payload.result.approval_required.reason || "Review this operation in the agent panel.");
    await loadCodebaseRun(state.codebase.activeRunId, false);
    return null;
  }
  return payload.result;
}

async function runCodebaseTests() {
  try {
    const result = await runCodebaseTool("run_tests");
    if (result) renderCodebaseTerminal(result);
  } catch (error) {
    toast("Tests failed to start", error.message);
  }
}

async function loadCodebaseDiff() {
  try {
    const result = await runCodebaseTool("git_diff");
    if (!result) return;
    qs("#codebaseGitOutput").textContent = result.stdout || result.stderr || "No Git changes.";
    showCodebaseOutput("git");
  } catch (error) {
    toast("Git diff failed", error.message);
  }
}

async function loadCodebaseStatus() {
  try {
    const result = await runCodebaseTool("git_status");
    if (!result) return;
    qs("#codebaseGitOutput").textContent = result.stdout || result.stderr || "Working tree clean.";
    showCodebaseOutput("git");
  } catch (error) {
    toast("Git status failed", error.message);
  }
}

async function runCodebaseCommand() {
  const command = qs("#codebaseCommand")?.value.trim();
  if (!command) return;
  try {
    const result = await runCodebaseTool("run_shell", { command, timeout_seconds: 900 });
    if (result) renderCodebaseTerminal(result);
  } catch (error) {
    toast("Command requires attention", error.message);
  }
}

async function inspectCodebaseBrowser() {
  const url = qs("#codebaseBrowserUrl")?.value.trim();
  if (!url) return;
  try {
    const result = await runCodebaseTool("browser_inspect", { url, actions: [], timeout_seconds: 120 });
    if (result) {
      const browser = result.browser || {};
      renderCodebaseTerminal({
        ...result,
        stdout: [`Page: ${browser.title || url}`, `URL: ${browser.url || url}`, `Console errors: ${(browser.console_errors || []).length}`, `Screenshot: ${result.artifact || "not created"}`].join("\n"),
      });
      toast((browser.console_errors || []).length ? "Browser issues found" : "Browser inspection passed", result.artifact || url);
    }
  } catch (error) {
    toast("Browser inspection failed", error.message);
  }
}

function renderCodebaseTerminal(result) {
  const command = Array.isArray(result.command) ? result.command.join(" ") : String(result.command || "validation");
  qs("#codebaseTerminalOutput").textContent = [`> ${command}`, result.stdout || "", result.stderr || "", `exit ${result.exit_code} · ${Math.round(result.duration_ms || 0)} ms`].filter(Boolean).join("\n");
  showCodebaseOutput("terminal");
}

function showCodebaseOutput(name) {
  const git = name === "git";
  qs("#codebaseTerminalOutput")?.toggleAttribute("hidden", git);
  qs("#codebaseGitOutput")?.toggleAttribute("hidden", !git);
  qsa("[data-codebase-output]").forEach(button => button.classList.toggle("active", button.dataset.codebaseOutput === name));
}

function showCodebaseDetail(name) {
  const map = { tasks: "#codebaseTaskGraph", events: "#codebaseEventStream", review: "#codebaseEvidence" };
  Object.entries(map).forEach(([key, selector]) => qs(selector)?.toggleAttribute("hidden", key !== name));
  qsa("[data-codebase-detail]").forEach(button => button.classList.toggle("active", button.dataset.codebaseDetail === name));
}

// =============================================
// Voice
// =============================================

function initVoice() {
  const btn = qs("#voiceRecordBtn");
  qs("#chatVoiceButton")?.addEventListener("click", toggleChatVoiceRecording);
  qs("#endChatVoiceMode")?.addEventListener("click", endChatVoiceConversation);
  if (!btn) return;
  btn.addEventListener("click", toggleVoiceRecording);
  qs("#sendTranscript")?.addEventListener("click", sendTranscriptToChat);
  qs("#clearTranscript")?.addEventListener("click", () => {
    const el = qs("#voiceTranscript");
    if (el) el.textContent = "Transcript will appear here…";
  });
  qs("#createVoiceSession")?.addEventListener("click", createVoiceSession);
  qs("#loadVoiceStatus")?.addEventListener("click", loadVoiceStatus);
  qs("#allowFallbackStt")?.addEventListener("change", () => {
    loadVoiceStatus().catch(() => {});
    if (qs("#allowFallbackStt")?.checked) {
      toast("Fallback STT enabled", "Windows Speech may be inaccurate. Use Whisper/faster-whisper for final transcripts.");
    }
  });
}

async function toggleChatVoiceRecording() {
  state.voiceTarget = "chat";
  if (!state.chatVoiceMode) {
    state.chatVoiceMode = true;
    qs("#chatVoiceMode")?.removeAttribute("hidden");
    setChatVoiceState("Listening", "Speak naturally. Your transcript and the spoken answer stay in this chat.");
  }
  if (state.voiceRecording) stopVoiceRecording();
  else await startVoiceRecording();
}

function endChatVoiceConversation() {
  state.chatVoiceMode = false;
  state.voiceReplyPending = false;
  if (state.voiceRecording) stopVoiceRecording();
  window.speechSynthesis?.cancel?.();
  qs("#chatVoiceMode")?.setAttribute("hidden", "");
  renderVoiceRecordingButtons(false);
}

function setChatVoiceState(status, detail = "") {
  const title = qs("#chatVoiceState");
  const text = qs("#chatVoiceLiveText");
  const mode = qs("#chatVoiceMode");
  if (title) title.textContent = status;
  if (text) text.textContent = detail;
  if (mode) mode.dataset.state = status.toLowerCase();
}

async function toggleVoiceRecording() {
  state.voiceTarget = "voice";
  if (state.voiceRecording) {
    stopVoiceRecording();
  } else {
    await startVoiceRecording();
  }
}

async function startVoiceRecording() {
  try {
    await ensureVoiceSession();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    if (window.AudioContext || window.webkitAudioContext) {
      await startWavVoiceRecorder(stream);
    } else {
      startMediaVoiceRecorder(stream);
    }
    state.voicePeakLevel = 0;
    renderVoiceTranscriptionQuality({ transcript_status: "recording", stt_message: "Recording audio locally..." });
    state.voiceRecording = true;
    if (state.voiceTarget === "chat") setChatVoiceState("Listening", "I’m listening. Tap the microphone when you finish, or pause naturally.");
    renderVoiceRecordingButtons(true);
    const status = qs("#voiceStatus");
    if (status) status.textContent = `Recording ${state.voiceRecordingBackend === "web-audio-wav" ? "WAV" : "audio"}… click again to stop`;
    const badge = qs("#voiceModeBadge");
    if (badge) { badge.textContent = "Recording"; badge.className = "badge badge-error"; }
  } catch(e) {
    toast("Microphone error", e.message);
  }
}

async function startWavVoiceRecorder(stream) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass || !window.AudioWorkletNode) {
    startMediaVoiceRecorder(stream);
    return;
  }
  const context = new AudioContextClass();
  try {
    if (!context.audioWorklet) throw new Error("AudioWorklet is not available.");
    const moduleUrl = URL.createObjectURL(new Blob([voiceRecorderWorkletSource()], { type: "text/javascript" }));
    try {
      await context.audioWorklet.addModule(moduleUrl);
    } finally {
      URL.revokeObjectURL(moduleUrl);
    }
    const source = context.createMediaStreamSource(stream);
    const worklet = new AudioWorkletNode(context, "openagent-voice-recorder", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
      channelInterpretation: "speakers",
    });
    const gain = context.createGain();
    gain.gain.value = 0;
    state.voicePcmChunks = [];
    state.voicePeakLevel = 0;
    state.voiceSampleRate = context.sampleRate;
    state.voiceMediaStream = stream;
    state.voiceAudioContext = context;
    state.voiceAudioSource = source;
    state.voiceAudioWorklet = worklet;
    state.voiceAudioProcessor = null;
    state.voiceAudioGain = gain;
    state.voiceRecordingBackend = "web-audio-wav";
    worklet.port.onmessage = event => {
      if (!state.voiceRecording) return;
      const input = event.data instanceof Float32Array ? event.data : new Float32Array(event.data || []);
      if (!input.length) return;
      state.voicePcmChunks.push(new Float32Array(input));
      renderVoiceLevel(input);
    };
    source.connect(worklet);
    worklet.connect(gain);
    gain.connect(context.destination);
    if (context.state === "suspended") await context.resume();
  } catch (error) {
    await context.close().catch(() => {});
    startMediaVoiceRecorder(stream);
  }
}

function voiceRecorderWorkletSource() {
  return `
class OpenAgentVoiceRecorder extends AudioWorkletProcessor {
  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    const channel = input && input[0];
    if (channel) {
      const copy = new Float32Array(channel.length);
      copy.set(channel);
      this.port.postMessage(copy, [copy.buffer]);
      if (output) {
        for (let i = 0; i < output.length; i += 1) {
          const source = input[Math.min(i, input.length - 1)] || channel;
          output[i].set(source);
        }
      }
    }
    return true;
  }
}
registerProcessor("openagent-voice-recorder", OpenAgentVoiceRecorder);
`;
}

function startMediaVoiceRecorder(stream) {
  state.voiceChunks = [];
  state.voiceMediaStream = stream;
  state.voiceRecordingBackend = "media-recorder";
  const options = MediaRecorder.isTypeSupported("audio/webm") ? { mimeType: "audio/webm" } : undefined;
  state.voiceMediaRecorder = new MediaRecorder(stream, options);
  state.voiceMediaRecorder.ondataavailable = e => { if (e.data.size > 0) state.voiceChunks.push(e.data); };
  state.voiceMediaRecorder.onstop = handleVoiceRecordingStop;
  state.voiceMediaRecorder.start(100);
}

function stopVoiceRecording() {
  if (!state.voiceRecording) return;
  state.voiceRecording = false;
  if (state.voiceRecordingBackend === "web-audio-wav") {
    state.voiceAudioWorklet?.port?.close?.();
    state.voiceAudioWorklet?.disconnect();
    state.voiceAudioProcessor?.disconnect();
    state.voiceAudioSource?.disconnect();
    state.voiceAudioGain?.disconnect();
    state.voiceMediaStream?.getTracks().forEach(t => t.stop());
    state.voiceAudioContext?.close?.();
    state.voiceAudioWorklet = null;
    state.voiceAudioProcessor = null;
    state.voiceAudioSource = null;
    state.voiceAudioGain = null;
    state.voiceAudioContext = null;
    handleVoiceRecordingStop();
  } else if (state.voiceMediaRecorder) {
    state.voiceMediaRecorder.stop();
    state.voiceMediaRecorder.stream?.getTracks().forEach(t => t.stop());
  }
  renderVoiceRecordingButtons(false);
  const status = qs("#voiceStatus");
  if (status) status.textContent = "Processing…";
  if (state.voiceTarget === "chat" && state.chatVoiceMode) setChatVoiceState("Transcribing", "Turning your speech into text locally.");
}

async function handleVoiceRecordingStop() {
  const recording = buildVoiceRecordingBlob();
  if (!recording) return;
  const { blob, filename } = recording;
  try {
    const session = await ensureVoiceSession();
    const audioBase64 = await blobToBase64(blob);
    const payload = await api("/api/voice/stt", {
      method: "POST",
      body: JSON.stringify({
        session_id: session.id,
        audio_base64: audioBase64,
        filename,
        speaker: "user",
        language: "en",
        stt_engine: qs("#voiceEngineSelect")?.value || "",
        allow_fallback_stt: qs("#allowFallbackStt")?.checked || false,
        transcript_hint: "",
        metadata: {
          client: "openagent-workbench",
          mime_type: blob.type || "application/octet-stream",
          size_bytes: blob.size,
        },
      }),
    });
    const transcription = payload.transcription || {};
    const el = qs("#voiceTranscript");
    if (el) {
      if (transcription.transcript_status === "transcribed" && transcription.transcript) {
        el.textContent = transcription.transcript;
      } else {
        el.textContent = transcription.display_text || transcription.stt_message || "No local speech-to-text engine returned text.";
      }
    }
    renderVoiceTranscriptionQuality(transcription);
    renderVoiceCaptureResult(transcription, payload);
    loadVoiceStatus().catch(() => {});
    const status = qs("#voiceStatus");
    if (status) status.textContent = transcription.transcript_status === "transcribed"
      ? `Transcription complete (${transcription.stt_engine}, ${transcription.confidence || "unknown"} confidence)`
      : "Audio captured - configure a local STT engine for real transcript text";
    const badge = qs("#voiceModeBadge");
    if (badge) {
      badge.textContent = transcription.transcript_status === "transcribed" ? "Transcribed" : "Needs STT";
      badge.className = transcription.transcript_status === "transcribed" ? "badge badge-success" : "badge badge-warning";
    }
    if (state.voiceTarget === "chat" && transcription.transcript_status === "transcribed" && transcription.transcript) {
      const input = qs("#chatInput");
      if (input) input.value = transcription.transcript;
      state.voiceReplyPending = true;
      setChatVoiceState("Thinking", transcription.transcript);
      await sendChat();
    } else if (state.voiceTarget === "chat") {
      toast("No reliable transcript", transcription.stt_message || "Configure faster-whisper or Vosk in Developer.");
    }
  } catch(e) {
    toast("Transcription error", e.message);
    const status = qs("#voiceStatus");
    if (status) status.textContent = "Error - transcription request failed";
    const badge = qs("#voiceModeBadge");
    if (badge) { badge.textContent = "Ready"; badge.className = "badge"; }
  }
}

function renderVoiceRecordingButtons(recording) {
  const dedicated = qs("#voiceRecordBtn");
  if (dedicated) {
    dedicated.classList.toggle("recording", recording);
    dedicated.textContent = recording ? "⏹" : "🎙";
    dedicated.setAttribute("aria-pressed", String(recording));
  }
  const chat = qs("#chatVoiceButton");
  if (chat) {
    chat.classList.toggle("recording", recording && state.voiceTarget === "chat");
    chat.textContent = recording && state.voiceTarget === "chat" ? "■" : "🎙";
    chat.setAttribute("aria-pressed", String(recording && state.voiceTarget === "chat"));
    chat.setAttribute("aria-label", recording && state.voiceTarget === "chat" ? "Stop voice recording" : "Start voice conversation");
    chat.title = recording && state.voiceTarget === "chat" ? "Stop and send" : "Start voice conversation";
  }
}

function renderVoiceLevel(input) {
  const bars = qsa("#voiceWaveform span");
  if (!bars.length) return;
  let sum = 0;
  let peak = 0;
  for (const sample of input) {
    const value = Math.abs(sample);
    sum += value * value;
    if (value > peak) peak = value;
  }
  const rms = Math.sqrt(sum / Math.max(1, input.length));
  state.voicePeakLevel = Math.max(state.voicePeakLevel * 0.92, peak);
  const level = Math.min(1, Math.max(rms * 10, state.voicePeakLevel * 1.4));
  bars.forEach((bar, index) => {
    const weight = 0.35 + (index + 1) / bars.length;
    bar.style.height = `${Math.max(8, Math.round(8 + level * 32 * weight))}px`;
    bar.classList.toggle("hot", level > 0.72);
  });
  const orb = qs("#chatVoiceOrb");
  if (orb && state.chatVoiceMode) orb.style.setProperty("--voice-level", String(level));
}

function renderVoiceTranscriptionQuality(transcription) {
  const el = qs("#voiceConfidence");
  if (!el) return;
  if (transcription.transcript_status === "recording") {
    el.className = "mini-status";
    el.innerHTML = `<strong>Recording locally.</strong><span>${escapeHtml(transcription.stt_message || "Speak clearly and keep the meter moving in the middle range.")}</span>`;
    return;
  }
  const confidence = transcription.confidence || "unknown";
  const warnings = transcription.warnings || [];
  const profile = transcription.prepared_audio_profile || transcription.audio_profile || {};
  const duration = profile.duration_seconds ? `${profile.duration_seconds}s` : "unknown duration";
  const engine = transcription.stt_engine || "none";
  if (transcription.transcript_status === "transcribed") {
    el.className = confidence === "high" ? "mini-status mini-status-success" : "mini-status mini-status-warning";
    el.innerHTML = `<strong>${escapeHtml(confidence.toUpperCase())} confidence · ${escapeHtml(engine)}</strong><span>${escapeHtml(duration)} · ${escapeHtml(warnings[0] || "Edit the transcript if needed before sending.")}</span>`;
    return;
  }
  el.className = "mini-status mini-status-warning";
  el.innerHTML = `<strong>No reliable transcript produced.</strong><span>${escapeHtml(transcription.stt_message || "Configure Whisper/faster-whisper/Vosk/whisper.cpp for accurate local transcription.")}</span>`;
}

function buildVoiceRecordingBlob() {
  if (state.voiceRecordingBackend === "web-audio-wav") {
    if (!state.voicePcmChunks.length) return null;
    return {
      blob: encodeWavBlob(state.voicePcmChunks, state.voiceSampleRate || 44100),
      filename: "voice.wav",
    };
  }
  if (!state.voiceChunks.length) return null;
  const blob = new Blob(state.voiceChunks, { type: state.voiceMediaRecorder?.mimeType || "audio/webm" });
  return {
    blob,
    filename: blob.type.includes("webm") ? "voice.webm" : "voice-recording.bin",
  };
}

function encodeWavBlob(chunks, sampleRate) {
  const samples = flattenFloat32(chunks);
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function flattenFloat32(chunks) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const result = new Float32Array(length);
  let offset = 0;
  chunks.forEach(chunk => { result.set(chunk, offset); offset += chunk.length; });
  return result;
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = String(reader.result || "");
      resolve(result.includes(",") ? result.split(",").pop() : result);
    };
    reader.onerror = () => reject(reader.error || new Error("Unable to read audio blob."));
    reader.readAsDataURL(blob);
  });
}

function sendTranscriptToChat() {
  const transcript = qs("#voiceTranscript")?.textContent?.trim();
  if (!transcript || transcript === "Transcript will appear here…") return;
  const input = qs("#chatInput");
  if (input) input.value = transcript;
  setView("chat");
  sendChat();
}

async function createVoiceSession() {
  try {
    const session = await ensureVoiceSession({ force: true });
    toast("Voice session created", session.id.slice(0, 8));
    await loadVoiceStatus();
  } catch(e) { toast("Voice session error", e.message); }
}

async function ensureVoiceSession({ force = false } = {}) {
  if (state.voiceSessionId && !force) {
    return { id: state.voiceSessionId };
  }
  const payload = await api("/api/voice/sessions", {
    method: "POST",
    body: JSON.stringify({
      project: getProject(),
      mode: "push_to_talk",
      metadata: { client: "openagent-workbench", explicit_microphone_start: true },
    }),
  });
  state.voiceSessionId = payload.session.id;
  const badge = qs("#voiceModeBadge");
  if (badge) { badge.textContent = "Session active"; badge.className = "badge badge-success"; }
  return payload.session;
}

async function loadVoiceStatus() {
  try {
    const [statusResult, sessionsResult] = await Promise.allSettled([
      api(`/api/voice/status?project=${encodeURIComponent(getProject())}`),
      api(`/api/voice/sessions?project=${encodeURIComponent(getProject())}&limit=8`),
    ]);
    if (statusResult.status !== "fulfilled") throw statusResult.reason;
    const data = statusResult.value;
    const sessions = sessionsResult.status === "fulfilled" ? sessionsResult.value.sessions || [] : [];
    state.voiceSessions = sessions;
    renderSttEngineStatus((data.stt || {}).engines || []);
    renderVoiceEngineOptions((data.stt || {}).engines || []);
    renderVoiceDashboard(data, sessions);
  } catch(e) { toast("Voice status error", e.message); }
}

function renderVoiceDashboard(data, sessions = []) {
  const el = qs("#voiceOutput");
  if (!el) return;
  const engines = (data.stt || {}).engines || [];
  const highQuality = engines.filter(engine => engine.enabled || (engine.available && engine.quality !== "fallback" && engine.id !== "ffmpeg"));
  const fallback = engines.find(engine => engine.id === "windows-speech" && engine.available);
  const converter = engines.find(engine => engine.id === "ffmpeg");
  const best = highQuality[0] || (fallback && qs("#allowFallbackStt")?.checked ? fallback : null);
  const readiness = best ? "Ready" : "Needs STT setup";
  const readinessClass = best ? "badge-success" : "badge-warning";
  const cards = [
    ["Sessions", data.sessions ?? 0],
    ["Turns", data.turns ?? 0],
    ["Translations", data.translations ?? 0],
    ["Engine", best?.name || "Not configured"],
  ];
  el.innerHTML = `
    <div class="voice-summary">
      ${cards.map(([label, value]) => `<article><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></article>`).join("")}
    </div>
    <div class="voice-runtime-card">
      <div>
        <strong>Speech runtime</strong>
        <p>${escapeHtml(data.privacy_boundary || "Voice data stays local.")}</p>
      </div>
      <span class="badge ${readinessClass}">${escapeHtml(readiness)}</span>
    </div>
    <div class="voice-engine-list">
      ${engines.filter(engine => engine.id !== "ffmpeg").map(renderVoiceEngineRow).join("")}
      ${converter ? renderVoiceEngineRow(converter) : ""}
    </div>
    <h4 class="subsection-title">Recent sessions</h4>
    <div class="stack-list">
      ${sessions.length ? sessions.map(session => `
        <button class="list-item voice-session-item ${session.id === state.voiceSessionId ? "active" : ""}" data-voice-session-id="${escapeHtml(session.id)}">
          <div class="list-item-row">
            <div>
              <strong>${escapeHtml(session.mode || "voice session")}</strong>
              <p>${escapeHtml(session.status || "active")} · ${escapeHtml(formatDateTime(session.updated_at) || session.id.slice(0, 8))}</p>
            </div>
            <span class="badge">${escapeHtml(session.id.slice(0, 8))}</span>
          </div>
        </button>
      `).join("") : emptyState("No voice sessions yet.")}
    </div>
  `;
  qsa("[data-voice-session-id]", el).forEach(item => {
    item.addEventListener("click", () => loadVoiceSession(item.dataset.voiceSessionId));
  });
}

async function loadVoiceSession(sessionId) {
  if (!sessionId) return;
  state.voiceSessionId = sessionId;
  qsa("[data-voice-session-id]").forEach(item => {
    item.classList.toggle("active", item.dataset.voiceSessionId === sessionId);
  });
  const payload = await api(`/api/voice/turns?session_id=${encodeURIComponent(sessionId)}`);
  const turns = payload.turns || [];
  const session = (state.voiceSessions || []).find(item => item.id === sessionId) || { id: sessionId, mode: "voice session" };
  const transcript = turns.filter(turn => turn.role === "user").map(turn => turn.text).filter(Boolean).at(-1)
    || turns.map(turn => turn.text).filter(Boolean).at(-1)
    || "Transcript will appear here…";
  const transcriptBox = qs("#voiceTranscript");
  if (transcriptBox) transcriptBox.textContent = transcript;
  renderVoiceSessionHistory(session, turns);
}

function renderVoiceSessionHistory(session, turns = []) {
  const el = qs("#voiceOutput");
  if (!el) return;
  el.innerHTML = `
    <div class="voice-runtime-card">
      <div>
        <strong>${escapeHtml(session.mode || "Voice session")}</strong>
        <p>${escapeHtml(session.status || "active")} · ${escapeHtml(formatDateTime(session.updated_at) || session.id)}</p>
      </div>
      <span class="badge badge-success">open</span>
    </div>
    <div class="voice-summary">
      <article><span>Turns</span><strong>${escapeHtml(String(turns.length))}</strong></article>
      <article><span>Session</span><strong>${escapeHtml(session.id.slice(0, 8))}</strong></article>
      <article><span>Mode</span><strong>${escapeHtml(session.mode || "voice")}</strong></article>
    </div>
    <h4 class="subsection-title">Transcript history</h4>
    <div class="stack-list">
      ${turns.length ? turns.map(turn => `
        <article class="list-item">
          <div class="list-item-row">
            <div>
              <strong>${escapeHtml(turn.speaker || turn.role || "speaker")}</strong>
              <p>${escapeHtml(turn.text || "")}</p>
            </div>
            <span class="badge">${escapeHtml(formatDateTime(turn.created_at) || turn.id.slice(0, 8))}</span>
          </div>
        </article>
      `).join("") : emptyState("No transcript turns recorded in this session yet.")}
    </div>
  `;
}

function renderVoiceEngineRow(engine) {
  const enabled = engine.enabled || (engine.available && engine.quality !== "fallback" && engine.id !== "ffmpeg");
  const badgeClass = enabled ? "badge-success" : engine.available ? "badge-warning" : "badge-soft";
  const label = enabled ? "ready" : engine.available ? "available" : "setup needed";
  return `
    <article class="voice-engine-row">
      <div>
        <strong>${escapeHtml(engine.name || engine.id)}</strong>
        <p>${escapeHtml(engine.reason || engine.privacy_boundary || "")}</p>
      </div>
      <span class="badge ${badgeClass}">${escapeHtml(label)}</span>
    </article>
  `;
}

function renderVoiceCaptureResult(transcription, payload) {
  const el = qs("#voiceOutput");
  if (!el) return;
  const status = transcription.transcript_status === "transcribed" ? "Transcript ready" : "Audio captured";
  const statusClass = transcription.transcript_status === "transcribed" ? "badge-success" : "badge-warning";
  const profile = transcription.prepared_audio_profile || transcription.audio_profile || {};
  el.innerHTML = `
    <div class="voice-runtime-card">
      <div>
        <strong>${escapeHtml(status)}</strong>
        <p>${escapeHtml(transcription.stt_message || payload.privacy_boundary || "Captured audio stayed in local storage.")}</p>
      </div>
      <span class="badge ${statusClass}">${escapeHtml(transcription.stt_engine || "local")}</span>
    </div>
    <div class="voice-summary">
      <article><span>Confidence</span><strong>${escapeHtml(transcription.confidence || "unknown")}</strong></article>
      <article><span>Duration</span><strong>${escapeHtml(profile.duration_seconds ? `${profile.duration_seconds}s` : "unknown")}</strong></article>
      <article><span>Audio</span><strong>${escapeHtml(payload.audio_asset?.mime_type || "stored")}</strong></article>
      <article><span>Session</span><strong>${escapeHtml((payload.session_id || "").slice(0, 8))}</strong></article>
    </div>
  `;
}

function renderSttEngineStatus(engines) {
  const el = qs("#sttEngineStatus");
  if (!el) return;
  if (!engines.length) {
    el.className = "mini-status mini-status-warning";
    el.innerHTML = `<strong>Local speech-to-text not configured.</strong><span>Recording works, but install or configure Whisper, faster-whisper, Vosk, whisper.cpp, Windows Speech WAV, or OPENAGENT_STT_COMMAND to generate real transcripts.</span>`;
    return;
  }
  const available = engines.filter(engine => engine.enabled || (engine.available && engine.quality !== "fallback" && engine.id !== "ffmpeg"));
  if (!available.length) {
    el.className = "mini-status mini-status-warning";
    const next = engines.find(engine => engine.id !== "ffmpeg" && engine.reason)?.reason || "Install and configure a local Whisper/faster-whisper/Vosk/whisper.cpp engine.";
    el.innerHTML = `<strong>High-quality speech-to-text is not configured.</strong><span>${escapeHtml(next)}</span>`;
    return;
  }
  el.className = "mini-status mini-status-success";
  el.innerHTML = `<strong>Speech-to-text ready.</strong><span>${escapeHtml(available.map(engine => engine.name).join(", "))}</span>`;
}

function renderVoiceEngineOptions(engines) {
  const select = qs("#voiceEngineSelect");
  if (!select) return;
  const current = select.value;
  const usable = engines.filter(engine => engine.id !== "ffmpeg" && (engine.enabled || engine.available));
  select.innerHTML = `<option value="">Best high-quality local engine</option>` + usable.map(engine => {
    const fallback = engine.quality === "fallback" ? " fallback" : "";
    const disabled = engine.quality === "fallback" && !qs("#allowFallbackStt")?.checked ? " disabled" : "";
    const label = `${engine.name || engine.id}${fallback}${engine.enabled ? "" : " (setup needed)"}`;
    return `<option value="${escapeHtml(engine.id)}"${disabled}>${escapeHtml(label)}</option>`;
  }).join("");
  if ([...select.options].some(option => option.value === current && !option.disabled)) select.value = current;
}

// =============================================
// Deep Research
// =============================================

function initResearch() {
  qs("#startResearch")?.addEventListener("click", startResearch);
  qs("#cancelResearch")?.addEventListener("click", cancelResearch);
  qs("#researchToChat")?.addEventListener("click", sendResearchToChat);
  qs("#refreshResearchLibrary")?.addEventListener("click", loadResearchLibrary);
  qs("#researchLibrary")?.addEventListener("click", handleResearchLibraryAction);
  loadResearchLibrary();
}

function updateResearchUI(running) {
  state.researchRunning = running;
  const prog = qs("#researchProgress");
  const badge = qs("#researchStatusBadge");
  const startBtn = qs("#startResearch");
  const cancelBtn = qs("#cancelResearch");
  if (prog) prog.style.display = running ? "block" : "none";
  if (badge) { badge.textContent = running ? "Researching…" : "Idle"; badge.className = running ? "badge badge-info" : "badge"; }
  if (startBtn) startBtn.disabled = running;
  if (cancelBtn) cancelBtn.disabled = !running;
}

async function cancelResearch() {
  const sessionId = state.researchSessionId;
  try {
    if (sessionId) {
      await api(`/api/research/sessions/${encodeURIComponent(sessionId)}/cancel`, {
        method: "POST",
        body: JSON.stringify({ project: getProject() }),
      });
    }
  } catch (error) {
    toast("Cancellation warning", error.message);
  } finally {
    state.researchAbortController?.abort("Research cancelled by user.");
    state.researchRunning = false;
    updateResearchUI(false);
    const output = qs("#researchOutput");
    if (output) output.innerHTML = `<div class="empty-state">Research cancellation requested and recorded locally.</div>`;
    loadResearchLibrary();
    toast("Research cancelled");
  }
}

function setResearchStep(step, status) {
  // status: "active" | "done" | ""
  const el = qs(`#rStep${step} span:first-child`);
  if (!el) return;
  if (status === "done") el.textContent = "✅";
  else if (status === "active") el.textContent = "⏳";
}

async function startResearch() {
  const query = qs("#researchQuery")?.value.trim();
  if (!query) { toast("Research query required"); return; }
  const controller = new AbortController();
  state.researchAbortController = controller;
  state.lastResearchResult = null;
  updateResearchUI(true);
  const reportBtn = qs("#researchToChat");
  if (reportBtn) reportBtn.disabled = true;
  renderResearchLoading(query);
  const sourcesEl = qs("#researchSources");
  if (sourcesEl) sourcesEl.innerHTML = "";

  setResearchStep(1, "active");
  try {
    const allowExternal = !state.privacySettings.localOnly && (
      qs("#researchAllowExternal")?.checked || qs("#allowExternal")?.checked || false
    );
    const created = await api("/api/research/sessions", {
      method: "POST",
      signal: controller.signal,
      body: JSON.stringify({
        query,
        project: getProject(),
        max_iterations: parseInt(qs("#researchDepth")?.value || "3", 10),
        provider: qs("#providerSelect")?.value || "auto",
        model: normalizeKnownModelName(state.modelSettings.researchModel || ""),
        allow_external: allowExternal,
      }),
    });
    const sessionId = created.session?.id;
    if (!sessionId) throw new Error("Research server did not return a durable session ID.");
    state.researchSessionId = sessionId;
    await loadResearchLibrary();
    const session = await waitForResearchSession(sessionId, controller.signal);
    const result = { ...session, ...(session.result || {}), run_id: session.id };
    state.lastResearchResult = result;
    setResearchStep(1, "done");
    setResearchStep(2, "done");
    setResearchStep(3, "done");
    setResearchStep(4, "done");
    renderResearchResult(result);
    const sources = result.sources || [];
    if (sourcesEl && sources.length) {
      sourcesEl.innerHTML = sources.map(s =>
        `<span class="source-chip">${escapeHtml(typeof s === "string" ? s : (s.title || s.url || s.type || "source"))}</span>`
      ).join("");
    }
    if (reportBtn) reportBtn.disabled = false;
    toast("Research complete", `Found ${sources.length} sources`);
  } catch(e) {
    const cancelled = e.name === "AbortError";
    const output = qs("#researchOutput");
    if (output) output.innerHTML = `<div class="empty-state">${cancelled ? "Research cancelled." : `Research error: ${escapeHtml(e.message)}`}</div>`;
    toast(cancelled ? "Research cancelled" : "Research error", cancelled ? "" : e.message);
  } finally {
    if (state.researchAbortController === controller) {
      state.researchAbortController = null;
      updateResearchUI(false);
    }
    loadResearchLibrary();
  }
}

async function waitForResearchSession(sessionId, signal) {
  let cursor = 0;
  while (true) {
    if (signal.aborted) throw new DOMException("Research cancelled", "AbortError");
    const authorization = state.authToken ? { Authorization: `Bearer ${state.authToken}` } : {};
    const response = await fetch(
      `/api/research/sessions/${encodeURIComponent(sessionId)}/stream?after_id=${cursor}&timeout=20`,
      { headers: authorization, signal },
    );
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error?.message || payload.error || `Research stream failed: ${response.status}`);
    }
    if (!response.body) throw new Error("Research progress streaming is unavailable in this runtime.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";
      for (const frame of frames) {
        const raw = frame.split("\n").filter(line => line.startsWith("data:")).map(line => line.slice(5).trim()).join("\n");
        if (!raw) continue;
        const event = JSON.parse(raw);
        cursor = Math.max(cursor, Number(event.cursor || event.checkpoint?.id || 0));
        if (event.type === "checkpoint") {
          const badge = qs("#researchStatusBadge");
          if (badge) badge.textContent = `${event.checkpoint.stage} · ${event.checkpoint.progress}%`;
        }
        if (event.session) renderResearchCheckpointProgress(event.session);
        if (event.type === "complete") {
          const session = event.session;
          if (session.status === "completed") return session;
          if (session.status === "failed") throw new Error(session.error || "Research failed.");
          if (session.status === "cancelled") throw new DOMException("Research cancelled", "AbortError");
        }
      }
      if (done) break;
    }
  }
}

function renderResearchCheckpointProgress(session) {
  const checkpoints = session.checkpoints || [];
  const progress = checkpoints.length ? checkpoints[checkpoints.length - 1].progress : 0;
  const badge = qs("#researchStatusBadge");
  if (badge) badge.textContent = `${session.status} · ${progress}%`;
  const stages = new Set(checkpoints.map(item => item.stage));
  if (stages.has("planning")) setResearchStep(1, "done");
  if (stages.has("sources")) setResearchStep(2, "done");
  if (stages.has("synthesis")) setResearchStep(3, "done");
  if (stages.has("completed")) setResearchStep(4, "done");
}

async function loadResearchLibrary() {
  const feed = qs("#researchLibrary");
  if (!feed) return;
  try {
    const payload = await api(`/api/research/sessions?project=${encodeURIComponent(getProject())}&limit=50`);
    const sessions = payload.sessions || [];
    feed.innerHTML = sessions.length ? sessions.map(session => `
      <article class="list-item research-session-row">
        <div>
          <strong>${escapeHtml(session.query)}</strong>
          <p>${escapeHtml(session.status)} · attempt ${escapeHtml(session.attempt || 0)}/${escapeHtml(session.max_attempts || 3)}${session.parent_id ? " · branch" : ""}</p>
        </div>
        <div class="toolbar slim">
          <button class="btn btn-soft" data-research-action="open" data-session-id="${escapeHtml(session.id)}">Open</button>
          ${["failed", "cancelled"].includes(session.status) ? `<button class="btn btn-outline" data-research-action="retry" data-session-id="${escapeHtml(session.id)}">Retry</button>` : ""}
          <button class="btn btn-outline" data-research-action="branch" data-session-id="${escapeHtml(session.id)}">Branch</button>
          <button class="btn btn-soft" data-research-action="archive" data-session-id="${escapeHtml(session.id)}">Archive</button>
          <button class="btn btn-danger" data-research-action="delete" data-session-id="${escapeHtml(session.id)}">Delete</button>
        </div>
      </article>
    `).join("") : emptyState("No research sessions yet.");
  } catch (error) {
    feed.innerHTML = emptyState(`Research library error: ${error.message}`);
  }
}

async function handleResearchLibraryAction(event) {
  const button = event.target.closest("[data-research-action]");
  if (!button) return;
  const action = button.dataset.researchAction;
  const sessionId = button.dataset.sessionId;
  try {
    if (action === "open") {
      const payload = await api(`/api/research/sessions/${encodeURIComponent(sessionId)}`);
      const session = payload.session;
      state.researchSessionId = session.id;
      state.lastResearchResult = { ...session, ...(session.result || {}), run_id: session.id };
      qs("#researchQuery").value = session.query;
      renderResearchResult(state.lastResearchResult);
      renderResearchCheckpointProgress(session);
      if (qs("#researchToChat")) qs("#researchToChat").disabled = session.status !== "completed";
      return;
    }
    const body = action === "branch" ? { label: "Follow-up branch" } : {};
    await api(`/api/research/sessions/${encodeURIComponent(sessionId)}/${action}`, {
      method: "POST", body: JSON.stringify(body),
    });
    toast(`Research ${action} complete`);
    await loadResearchLibrary();
  } catch (error) {
    toast("Research action failed", error.message);
  }
}

function renderResearchLoading(query) {
  const el = qs("#researchOutput");
  if (!el) return;
  el.innerHTML = `
    <div class="research-query-message">${escapeHtml(query)}</div>
    <div class="research-writing"><span class="spinner"></span><span>Building the evidence set…</span></div>
  `;
}

function renderResearchResult(result) {
  const el = qs("#researchOutput");
  if (!el) return;
  const report = result.report || result.summary || "No report was returned.";
  const sources = result.sources || [];
  el.innerHTML = `
    <div class="research-markdown">${renderMarkdown(report)}</div>
    <details class="research-source-list" ${sources.length ? "open" : ""}>
      <summary>${sources.length} source${sources.length === 1 ? "" : "s"}</summary>
      ${sources.length ? sources.map(source => `
        <article class="list-item">
          <strong>${escapeHtml(source.title || source.type || "Local source")}</strong>
          <p>${escapeHtml(source.snippet || source.url || source.id || "")}</p>
        </article>
      `).join("") : ""}
    </details>
  `;
}

function sendResearchToChat() {
  const result = state.lastResearchResult;
  if (!result) return;
  const input = qs("#chatInput");
  if (input) {
    input.value = `Continue from this research report and help me act on it:\n\n${result.report || result.summary || ""}`;
  }
  setView("chat");
  input?.focus();
}

// =============================================
// Agents
// =============================================

async function loadAgents() {
  try {
    const payload = await api("/api/agents");
    try { state.customAgents = JSON.parse(localStorage.getItem("oa-custom-agent-roles") || "[]"); } catch (_) { state.customAgents = []; }
    state.agents = [...(payload.agents || []), ...state.customAgents];
    const sel = qs("#agentSelect");
    if (sel) sel.innerHTML = state.agents.map(a => `<option value="${escapeHtml(a.id)}">${escapeHtml(a.name)}</option>`).join("");
    const grid = qs("#agentGrid");
    if (grid) {
      grid.innerHTML = state.agents.map((a, index) => `<button class="agent-role-row ${index === 0 ? "active" : ""}" type="button" data-agent-role="${escapeHtml(a.id)}"><span class="agent-avatar">${escapeHtml((a.name || "A").slice(0, 1).toUpperCase())}</span><span><strong>${escapeHtml(a.name)}</strong><small>${escapeHtml(a.id)}</small></span><i></i></button>`).join("");
      qsa("[data-agent-role]", grid).forEach(button => button.addEventListener("click", () => {
        if (sel) sel.value = button.dataset.agentRole;
        qsa("[data-agent-role]", grid).forEach(item => item.classList.toggle("active", item === button));
      }));
    }
  } catch(e) { console.warn("loadAgents", e); }
}

async function runAgent(endpoint = "/api/agents/run") {
  const selectedId = qs("#agentSelect")?.value || "executive";
  const custom = state.customAgents.find(item => item.id === selectedId);
  const agentId = custom?.base_agent || selectedId;
  const requestedGoal = qs("#goalInput")?.value.trim() || "Review OpenAgent status";
  const goal = custom ? `Role: ${custom.name}\nInstructions: ${custom.instructions}\n\nGoal: ${requestedGoal}` : requestedGoal;
  try {
    const payload = await api(endpoint, {
      method: "POST",
      body: JSON.stringify({ agent: agentId, goal, project: getProject() }),
    });
    renderAgentOutput(payload);
  } catch(e) { toast("Agent error", e.message); }
}

function openAgentRoleDialog(role = null) {
  const dialog = qs("#agentRoleDialog");
  if (!dialog) return;
  const builtIns = state.agents.filter(item => !String(item.id).startsWith("custom:"));
  const base = qs("#agentRoleBase");
  if (base) base.innerHTML = builtIns.map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`).join("");
  dialog.dataset.roleId = role?.id || "";
  qs("#agentRoleDialogTitle").textContent = role ? "Edit role" : "Create role";
  qs("#agentRoleName").value = role?.name || "";
  qs("#agentRoleInstructions").value = role?.instructions || "";
  if (base) base.value = role?.base_agent || builtIns[0]?.id || "executive";
  qs("#deleteAgentRole")?.toggleAttribute("hidden", !role);
  dialog.showModal();
}

function saveAgentRole() {
  const dialog = qs("#agentRoleDialog");
  const name = qs("#agentRoleName")?.value.trim();
  const instructions = qs("#agentRoleInstructions")?.value.trim();
  if (!name || !instructions) { toast("Role needs a name and instructions"); return; }
  const previousId = dialog?.dataset.roleId || "";
  const id = previousId || `custom:${crypto.randomUUID()}`;
  const role = { id, name, description: instructions.slice(0, 140), instructions, base_agent: qs("#agentRoleBase")?.value || "executive", custom: true };
  state.customAgents = [...state.customAgents.filter(item => item.id !== id), role];
  localStorage.setItem("oa-custom-agent-roles", JSON.stringify(state.customAgents));
  dialog?.close();
  loadAgents();
  toast("Role saved", name);
}

function deleteAgentRole() {
  const dialog = qs("#agentRoleDialog");
  const id = dialog?.dataset.roleId || "";
  if (!id || !window.confirm("Delete this custom role?")) return;
  state.customAgents = state.customAgents.filter(item => item.id !== id);
  localStorage.setItem("oa-custom-agent-roles", JSON.stringify(state.customAgents));
  dialog?.close();
  loadAgents();
}

async function runParallelAgents() {
  const goal = qs("#goalInput")?.value.trim() || "Review OpenAgent";
  try {
    const tasks = [
      { agent: "research", goal },
      { agent: "planning", goal },
    ];
    const payload = await api("/api/agents/parallel", {
      method: "POST",
      body: JSON.stringify({ tasks, project: getProject() }),
    });
    renderAgentOutput(payload);
  } catch(e) { toast("Parallel agent error", e.message); }
}

function renderAgentOutput(payload = {}) {
  const el = qs("#agentOutput");
  if (!el) return;
  const runs = payload.results || payload.runs || (payload.run ? [payload.run] : []);
  const approval = payload.approval || payload.pending_approval || null;
  const approvalMarkup = approval ? `<div class="voice-runtime-card">
    <div><strong>Approval required</strong><p>${escapeHtml(approval.reason || approval.action || "Human approval checkpoint is waiting.")}</p></div>
    <span class="badge badge-warning">pending</span>
  </div>` : "";
  const renderedRuns = runs.length ? runs.map(item => {
    const run = item.run || item;
    const response = run.response ?? item.response ?? {};
    const content = typeof response === "string"
      ? response
      : response.content || response.summary || run.summary || item.summary || JSON.stringify(response || run, null, 2);
    const agent = run.agent || response.agent || item.task?.agent || payload.agent || "agent";
    const execution = run.metadata || response.metadata || {};
    const runtimeLabel = execution.provider
      ? `${execution.provider} · ${execution.model || "model"}`
      : execution.degraded ? "Local fallback" : "Completed";
    return `<article class="list-item agent-output-card">
      <div class="list-item-row">
        <div>
          <strong>${escapeHtml(agent)}</strong>
          <p>${renderMarkdown(content)}</p>
        </div>
        <span class="badge ${execution.degraded ? "badge-warning" : "badge-success"}">${escapeHtml(runtimeLabel)}</span>
      </div>
    </article>`;
  }).join("") : `<div class="knowledge-summary-card"><strong>Agent response</strong><p>${renderMarkdown(
    typeof payload.response === "string" ? payload.response : payload.response?.content || payload.summary || JSON.stringify(payload, null, 2)
  )}</p></div>`;
  el.innerHTML = `${approvalMarkup}${renderedRuns}`;
}

// =============================================
// Providers / Models
// =============================================

async function loadProviders() {
  try {
    const payload = await api("/api/models");
    state.providers = payload.providers || [];
    const sel = qs("#providerSelect");
    if (sel) sel.innerHTML = state.providers.map(p => {
      const label = `${p.label || p.id}${p.available ? "" : " (setup needed)"}`;
      return `<option value="${escapeHtml(p.id)}">${escapeHtml(label)}</option>`;
    }).join("");
    const cards = qs("#providerCards");
    if (cards) cards.innerHTML = state.providers.map(p => {
      const installed = Array.isArray(p.installed_models) ? p.installed_models : [];
      const detail = p.id === "ollama" && installed.length
        ? `Selected ${p.selected_model || installed[0]} · ${installed.length} installed`
        : p.privacy_warning || "Local-first provider.";
      const badge = p.id === "ollama" && installed.length && !p.default_model_ready
        ? "Installed model"
        : p.available ? "Available" : "Setup needed";
      return `<article class="provider-tile">
        <strong>${escapeHtml(p.label || p.id)}</strong>
        <p>${escapeHtml(detail)}</p>
        <span class="badge ${p.available ? "badge-success" : "badge-warn"}">${escapeHtml(badge)}</span>
       </article>`;
    }).join("");
    updateProviderWarning();
  } catch(e) { console.warn("loadProviders", e); }
}

function updateProviderWarning() {
  const sel = qs("#providerSelect");
  const warn = qs("#providerWarning");
  if (!sel || !warn) return;
  const selected = state.providers.find(p => p.id === sel.value);
  if (!selected) return;
  const installed = Array.isArray(selected.installed_models) ? selected.installed_models : [];
  warn.textContent = selected.id === "ollama" && installed.length
    ? `Ollama local server ready. OpenAgent will use ${selected.selected_model || installed[0]} for this turn.`
    : selected.privacy_warning || "Local-first routing.";
}

async function discoverModels() {
  try {
    const payload = await api("/api/models/discover", { method: "POST" });
    setOutput("#modelPolicyOutput", payload);
    toast("Discovery complete", `Found ${(payload.discovered || []).length} models`);
    loadProviders();
  } catch(e) { toast("Discovery error", e.message); }
}

async function loadModelPolicy() {
  try {
    const payload = await api("/api/model-policy");
    setOutput("#modelPolicyOutput", payload);
  } catch(e) { console.warn("loadModelPolicy", e); }
}

async function loadSmoothRuntime(task = "chat") {
  try {
    const params = new URLSearchParams({ task, quantization_mode: state.modelSettings.quantizationMode || "auto" });
    if (state.modelSettings.targetRamGb) params.set("target_ram_gb", state.modelSettings.targetRamGb);
    const payload = await api(`/api/core/runtime?${params.toString()}`);
    state.smoothRuntime = payload;
    renderSmoothRuntime(payload);
    return payload;
  } catch(e) {
    toast("Smooth mode error", e.message);
    throw e;
  }
}

function renderSmoothRuntime(payload) {
  const profile = payload.profile || {};
  const badge = qs("#smoothRuntimeBadge");
  if (badge) {
    badge.textContent = profile.tier || "Auto";
    badge.className = "badge badge-success";
  }
  const summary = qs("#smoothRuntimeSummary");
  if (summary) {
    const items = [
      ["Task", payload.task || "chat"],
      ["RAM", `${profile.ram_gb || "?"} GB`],
      ["Quant", `${profile.quantization_method || "Q4"} / ${profile.quantization_bits || 4}-bit`],
      ["Context", profile.context_tokens || "auto"],
    ];
    summary.innerHTML = items.map(([label, value]) =>
      `<article class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></article>`
    ).join("");
  }
  const profiles = qs("#smoothModelProfiles");
  if (profiles) {
    const rows = payload.model_profiles || [];
    profiles.innerHTML = `
      <section class="feature-section">
        <div class="feature-section-header">
          <strong>Recommended model: ${escapeHtml(payload.recommended_model || "")}</strong>
          <span class="badge badge-info">${escapeHtml(profile.tier || "auto")}</span>
        </div>
        ${rows.map(row => `
          <article class="feature-row">
            <div>
              <strong>${escapeHtml(row.task)} · ${escapeHtml(row.model)}</strong>
              <p>${escapeHtml(row.notes || "")}</p>
            </div>
            <span class="badge badge-success">${escapeHtml(row.quantization || "local")}</span>
          </article>
        `).join("")}
      </section>`;
  }
  setOutput("#modelPolicyOutput", payload);
}

async function autoQuantizeRegisteredModel() {
  try {
    const modelsPayload = await api("/api/local-ai/models");
    const models = modelsPayload.models || [];
    const model = models.find(m => m.status === "available") || models[0];
    if (!model) {
      const message = "Register or download a local model first, then OpenAgent can record a RAM-safe quantization plan for it.";
      setOutput("#modelPolicyOutput", { error: message });
      toast("No registered model", message);
      return;
    }
    const task = state.smoothRuntime?.task || "chat";
    const targetRam = state.smoothRuntime?.profile?.ram_gb;
    const payload = await api("/api/local-ai/models/auto-quantize", {
      method: "POST",
      body: JSON.stringify({
        model_id: model.id,
        task,
        target_ram_gb: targetRam || undefined,
        quantization_mode: state.modelSettings.quantizationMode || "auto",
        parameter_count_b: state.modelSettings.parameterCountB || undefined,
      }),
    });
    setOutput("#modelPolicyOutput", payload);
    toast(
      "Quantization plan created",
      `${model.name || model.id}: ${payload.profile?.quantization_method || "smooth"}`
    );
  } catch(e) {
    toast("Auto-quantize error", e.message);
  }
}

// =============================================
// Workspace
// =============================================

async function loadWorkspaceItems() {
  const q = qs("#workspaceSearch")?.value.trim() || "";
  const payload = await api(`/api/workspace/items?project=${encodeURIComponent(getProject())}&q=${encodeURIComponent(q)}&limit=12`);
  const hiddenFixtureTitles = new Set(["desktop smoke document", "hard audit", "audit dashboard", "audit template"]);
  const items = (payload.items || []).filter(item => !hiddenFixtureTitles.has(String(item.title || "").trim().toLowerCase()));
  const el = qs("#workspaceItems");
  if (el) el.innerHTML = items.length
    ? items.map(i => `<button class="document-list-item" type="button" data-workspace-title="${escapeHtml(i.title)}" data-workspace-content="${escapeHtml(i.content || "")}"><strong>${escapeHtml(i.title)}</strong><span>${escapeHtml(i.kind || "document")}</span></button>`).join("")
    : `<div class="document-empty">No documents yet</div>`;
  qsa("[data-workspace-title]", el).forEach(item => item.addEventListener("click", () => {
    const title = qs("#workspaceTitle"); const content = qs("#workspaceContent");
    if (title) title.value = item.dataset.workspaceTitle || "";
    if (content) content.value = item.dataset.workspaceContent || "";
  }));
}

async function createWorkspaceItem() {
  const title = qs("#workspaceTitle")?.value.trim();
  const content = qs("#workspaceContent")?.value.trim() || "";
  const kind = qs("#workspaceKind")?.value || "note";
  if (!title) { toast("Title required"); return; }
  try {
    const payload = await api("/api/workspace/items", {
      method: "POST",
      body: JSON.stringify({ title, content, kind, project: getProject() }),
    });
    setOutput("#workspaceOutput", payload);
    qs("#workspaceTitle").value = "";
    qs("#workspaceContent").value = "";
    loadWorkspaceItems();
  } catch(e) { toast("Workspace error", e.message); }
}

// =============================================
// Knowledge & File upload
// =============================================

async function loadKnowledgeStatus() {
  const [payload, rag] = await Promise.all([
    api(`/api/knowledge/status?project=${encodeURIComponent(getProject())}`),
    api(`/api/rag/status?project=${encodeURIComponent(getProject())}`),
  ]);
  state.knowledge.status = payload;
  state.knowledge.rag = rag;
  const backendStatus = qs("#ragBackendStatus");
  const accelerators = rag.vector_store?.active_accelerators || [];
  if (backendStatus) {
    backendStatus.innerHTML = [
      payload.fts_available ? "FTS5 ready" : "Lexical fallback",
      ...(accelerators.length ? accelerators.map(name => `${name.toUpperCase()} ready`) : ["SQLite vector fallback"]),
      rag.algorithm || "Advanced RAG ready",
    ].map(label => `<span>${escapeHtml(label)}</span>`).join("");
  }
  const embeddingSelect = qs("#knowledgeEmbeddingProvider");
  if (embeddingSelect) {
    const providers = rag.vector_store?.embedding_providers || [];
    const fastembed = providers.find(provider => provider.id === "fastembed");
    const option = [...embeddingSelect.options].find(item => item.value === "fastembed");
    if (option) {
      option.disabled = !fastembed?.available;
      option.textContent = fastembed?.available
        ? `FastEmbed · ${fastembed.model || "local"}`
        : "FastEmbed · setup needed";
      if (option.disabled && embeddingSelect.value === "fastembed") embeddingSelect.value = "local-hash";
    }
  }
  renderKnowledgeIndexStatus();
  if (!state.knowledge.hasResult) renderKnowledgeLanding();
  return { payload, rag };
}

function setKnowledgeOutput(html, { result = true } = {}) {
  for (const el of [qs("#knowledgeOutput"), qs("#knowledgeWorkspaceOutput")]) {
    if (!el) continue;
    el.innerHTML = html;
    el.dataset.knowledgeResult = result ? "true" : "false";
  }
  state.knowledge.hasResult = result;
}

function renderKnowledgeLanding() {
  const payload = state.knowledge.status || {};
  const rag = state.knowledge.rag || {};
  setKnowledgeOutput(`
    <div class="knowledge-empty-state">
      <span>✦</span>
      <strong>${payload.documents ? "Search your local sources" : "Build a local source library"}</strong>
      <p>${payload.documents
        ? `${payload.documents} source${payload.documents === 1 ? "" : "s"} and ${payload.chunks || 0} indexed chunks are ready for retrieval.`
        : "Add a PDF, document, note, or dataset to begin."}</p>
    </div>
  `, { result: false });
  const output = qs("#knowledgeOutput");
  if (output) output.innerHTML = `
    <div class="knowledge-summary-card">
      <strong>Knowledge index</strong>
      <p>${escapeHtml(payload.fts_available ? "Full-text retrieval is enabled." : "Fallback lexical retrieval is enabled.")} ${escapeHtml(rag.algorithm || "Local retrieval is ready.")}</p>
    </div>
  `;
}

function renderKnowledgeIndexStatus() {
  const status = state.knowledge.status || {};
  const rag = state.knowledge.rag || {};
  const el = qs("#knowledgeWorkspaceStatus");
  if (!el) return;
  const accelerators = rag.vector_store?.active_accelerators || [];
  const retrieval = [
    status.fts_available ? "Full-text" : "Lexical",
    accelerators.length ? accelerators.map(item => String(item).toUpperCase()).join(" + ") : "Local vectors",
  ];
  el.innerHTML = `
    <div><strong>${escapeHtml(String(status.documents ?? 0))}</strong><span>sources</span></div>
    <div><strong>${escapeHtml(String(status.chunks ?? 0))}</strong><span>chunks</span></div>
    <div><strong>${escapeHtml(String(status.entities ?? 0))}</strong><span>entities</span></div>
    <p>${escapeHtml(retrieval.join(" · "))}</p>
  `;
}

async function loadKnowledgeWorkspace() {
  const project = encodeURIComponent(getProject());
  const [status, documents, tags, timeline] = await Promise.all([
    loadKnowledgeStatus(),
    api(`/api/knowledge/documents?project=${project}&limit=200`),
    api(`/api/knowledge/tags?project=${project}&limit=48`),
    api(`/api/knowledge/timeline?project=${project}&limit=8`),
  ]);
  state.knowledge.status = status.payload;
  state.knowledge.rag = status.rag;
  state.knowledge.documents = documents.documents || [];
  state.knowledge.tags = tags.tags || [];
  state.knowledge.timeline = timeline.timeline || [];
  renderKnowledgeLibrary();
  renderKnowledgeInspector();
}

function renderKnowledgeLibrary() {
  const list = qs("#knowledgeDocumentList");
  const summary = qs("#knowledgeLibrarySummary");
  if (!list || !summary) return;
  const filter = state.knowledge.filter.trim().toLowerCase();
  const documents = state.knowledge.documents.filter(document => {
    if (!filter) return true;
    return [document.title, document.source, ...(document.tags || [])]
      .join(" ").toLowerCase().includes(filter);
  });
  summary.textContent = `${documents.length} of ${state.knowledge.documents.length} sources`;
  list.innerHTML = documents.length ? documents.map(document => `
    <button class="knowledge-document${document.id === state.knowledge.selectedDocumentId ? " active" : ""}" type="button" data-knowledge-document="${escapeHtml(document.id)}">
      <span class="knowledge-document-icon">${escapeHtml(knowledgeSourceIcon(document.source))}</span>
      <span class="knowledge-document-copy"><strong>${escapeHtml(document.title || "Untitled source")}</strong><small>${escapeHtml(String(document.chunk_count || 0))} chunks · ${escapeHtml(formatCompactDate(document.created_at))}</small></span>
    </button>
  `).join("") : `<div class="knowledge-library-empty">${state.knowledge.documents.length ? "No sources match this filter." : "No local sources yet."}</div>`;
}

function knowledgeSourceIcon(source) {
  const value = String(source || "").toLowerCase();
  if (value.includes("pdf")) return "PDF";
  if (value.includes("csv") || value.includes("dataset")) return "CSV";
  if (value.includes("web")) return "WEB";
  if (value.includes("note") || value.includes("manual")) return "NOTE";
  return "DOC";
}

function formatCompactDate(value) {
  if (!value) return "local";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "local";
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

function renderKnowledgeInspector() {
  const tagList = qs("#knowledgeWorkspaceTags");
  const timeline = qs("#knowledgeTimelinePreview");
  if (tagList) {
    const tags = [...new Map(state.knowledge.tags.map(tag => [tag.tag, tag])).values()].slice(0, 14);
    tagList.innerHTML = tags.length ? tags.map(tag => `<button type="button" data-knowledge-tag="${escapeHtml(tag.tag)}">${escapeHtml(tag.tag)}</button>`).join("") : `<span class="knowledge-muted">Tags appear after sources are indexed.</span>`;
  }
  if (timeline) {
    const items = state.knowledge.timeline.slice(0, 5);
    timeline.innerHTML = items.length ? items.map(item => `<button type="button" data-knowledge-timeline="${escapeHtml(item.source_id || "")}"><span>${escapeHtml(formatCompactDate(item.timestamp))}</span><strong>${escapeHtml(item.title || item.type || "Source activity")}</strong></button>`).join("") : `<span class="knowledge-muted">No source activity yet.</span>`;
  }
}

function initKnowledgeWorkspace() {
  const fileInput = qs("#knowledgeLibraryFileInput");
  qs("#knowledgeAddFile")?.addEventListener("click", () => fileInput?.click());
  fileInput?.addEventListener("change", async event => {
    await uploadFiles([...event.target.files]);
    event.target.value = "";
  });
  qs("#knowledgeLibraryFilter")?.addEventListener("input", event => {
    state.knowledge.filter = event.target.value || "";
    renderKnowledgeLibrary();
  });
  qsa("#knowledgeModeSearch, #knowledgeModeSynthesize, #knowledgeModeRag").forEach(button => {
    button.addEventListener("click", () => {
      state.knowledge.mode = button.dataset.knowledgeMode || "search";
      qsa("#knowledgeModeSearch, #knowledgeModeSynthesize, #knowledgeModeRag").forEach(item => item.classList.toggle("active", item === button));
    });
  });
  qs("#runKnowledgeWorkspace")?.addEventListener("click", runKnowledgeWorkspace);
  qs("#knowledgeWorkspaceQuery")?.addEventListener("keydown", event => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      runKnowledgeWorkspace();
    }
  });
  qs("#showKnowledgeGraphWorkspace")?.addEventListener("click", showKnowledgeGraph);
  qs("#showKnowledgeTimeline")?.addEventListener("click", showKnowledgeTimeline);
}

async function runKnowledgeWorkspace() {
  const query = qs("#knowledgeWorkspaceQuery")?.value.trim() || "";
  if (!query) {
    toast("Enter a knowledge question");
    qs("#knowledgeWorkspaceQuery")?.focus();
    return;
  }
  const button = qs("#runKnowledgeWorkspace");
  if (button) button.disabled = true;
  try {
    if (state.knowledge.mode === "synthesize") await synthesizeKnowledge(query);
    else if (state.knowledge.mode === "rag") {
      await runAdvancedRag({
        query,
        button,
        generate: Boolean(qs("#knowledgeGenerateAnswer")?.checked),
      });
    } else await searchKnowledge(query);
  } finally {
    if (button) button.disabled = false;
  }
}

async function showKnowledgeGraph() {
  try {
    const payload = await api(`/api/knowledge/graph?project=${encodeURIComponent(getProject())}`);
    renderKnowledgeGraph(payload);
  } catch (error) {
    toast("Knowledge graph unavailable", error.message);
  }
}

async function showKnowledgeTimeline() {
  try {
    const payload = await api(`/api/knowledge/timeline?project=${encodeURIComponent(getProject())}&limit=40`);
    const items = payload.timeline || [];
    state.knowledge.timeline = items;
    renderKnowledgeInspector();
    setKnowledgeOutput(`
      <div class="knowledge-summary-card"><strong>Source timeline</strong><p>${escapeHtml(items.length)} local events in this project.</p></div>
      <div class="knowledge-result-list">
        ${items.length ? items.map(item => `<article class="list-item"><strong>${escapeHtml(item.title || item.type || "Source activity")}</strong><p>${escapeHtml(formatDateTime(item.timestamp) || "Local source activity")}</p></article>`).join("") : emptyState("No source activity yet.")}
      </div>
    `);
  } catch (error) {
    toast("Timeline unavailable", error.message);
  }
}

function selectKnowledgeDocument(documentId) {
  const document = state.knowledge.documents.find(item => item.id === documentId);
  if (!document) return;
  state.knowledge.selectedDocumentId = document.id;
  renderKnowledgeLibrary();
  const query = qs("#knowledgeWorkspaceQuery");
  if (query && !query.value.trim()) query.value = `What does ${document.title} cover?`;
}

function initUpload() {
  const zone = qs("#uploadZone");
  const input = qs("#fileInput");
  if (!zone || !input) return;

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") input.click(); });
  input.addEventListener("change", e => uploadFiles([...e.target.files]));

  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    uploadFiles([...e.dataTransfer.files]);
  });
}

async function uploadFiles(files) {
  for (const file of files) await uploadFile(file);
  const input = qs("#fileInput");
  if (input) input.value = "";
  await loadKnowledgeWorkspace().catch(() => {});
}

async function uploadFile(file) {
  const status = qs("#uploadStatus");
  if (status) { status.style.display = "block"; status.textContent = `Uploading ${file.name}…`; }
  const formData = new FormData();
  formData.append("file", file);
  formData.append("project", getProject());
  formData.append("embedding_provider", qs("#knowledgeEmbeddingProvider")?.value || "");
  try {
    const res = await fetch("/api/knowledge/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");
    if (status) status.textContent = `✅ Ingested: ${file.name}`;
    renderKnowledgeIngest(data.document || data, `Ingested ${file.name}`);
    toast("File ingested", file.name);
  } catch(e) {
    if (status) status.textContent = `Error: ${e.message}`;
    toast("Upload error", e.message);
  }
}

async function ingestKnowledge() {
  const title = qs("#knowledgeTitle")?.value.trim();
  const content = qs("#knowledgeContent")?.value.trim();
  if (!title || !content) { toast("Title and content required"); return; }
  try {
    const payload = await api("/api/knowledge/ingest", {
      method: "POST",
      body: JSON.stringify({ title, content, project: getProject(), embedding_provider: qs("#knowledgeEmbeddingProvider")?.value || "" }),
    });
    renderKnowledgeIngest(payload.document || payload, "Text ingested");
    await loadKnowledgeWorkspace().catch(() => {});
    toast("Ingested", title);
  } catch(e) { toast("Ingest error", e.message); }
}

async function searchKnowledge(query = "") {
  const q = query.trim() || qs("#knowledgeQuery")?.value.trim() || "";
  if (!q) return;
  try {
    const payload = await api("/api/knowledge/hybrid-search", {
      method: "POST",
      body: JSON.stringify({ query: q, project: getProject(), embedding_provider: qs("#knowledgeEmbeddingProvider")?.value || "" }),
    });
    renderKnowledgeSearch(q, payload.results || []);
  } catch(e) { toast("Search error", e.message); }
}

async function synthesizeKnowledge(query = "") {
  const q = query.trim() || qs("#knowledgeQuery")?.value.trim() || "summary";
  try {
    const payload = await api("/api/knowledge/synthesize", {
      method: "POST",
      body: JSON.stringify({ query: q, project: getProject(), embedding_provider: qs("#knowledgeEmbeddingProvider")?.value || "" }),
    });
    renderKnowledgeSynthesis(payload);
  } catch(e) { toast("Synthesis error", e.message); }
}

async function runAdvancedRag(options = {}) {
  const query = String(options.query || qs("#knowledgeQuery")?.value || "").trim();
  if (!query) { toast("Enter a knowledge question first"); return; }
  const button = options.button || qs("#runAdvancedRag");
  const original = button?.textContent || "Deep RAG";
  if (button) { button.disabled = true; button.textContent = "Retrieving…"; }
  try {
    const generate = options.generate ?? Boolean(qs("#ragGenerate")?.checked);
    const provider = qs("#providerSelect")?.value || "auto";
    const model = provider === "auto" ? "" : normalizeKnownModelName(state.modelSettings.chatModel || "");
    const payload = await api("/api/rag/query", {
      method: "POST",
      body: JSON.stringify({
        query,
        project: getProject(),
        limit: 8,
        context_token_budget: 1200,
        generate,
        provider,
        model,
        allow_external: false,
        use_cache: true,
        embedding_provider: qs("#knowledgeEmbeddingProvider")?.value || "",
      }),
    });
    renderAdvancedRag(payload);
  } catch(e) {
    toast("Deep RAG error", e.message);
  } finally {
    if (button) { button.disabled = false; button.textContent = original; }
  }
}

function renderAdvancedRag(payload) {
  const sources = payload.sources || [];
  const confidence = payload.confidence || {};
  const retrieval = payload.retrieval || {};
  const conflicts = payload.conflicts || [];
  const cache = payload.cache || {};
  setKnowledgeOutput(`
    <div class="knowledge-summary-card rag-answer-card">
      <div class="list-item-row">
        <strong>Grounded answer</strong>
        <span class="badge ${confidence.level === "high" ? "badge-success" : confidence.level === "low" || confidence.level === "insufficient" ? "badge-error" : "badge-info"}">${escapeHtml(confidence.level || "unknown")}</span>
      </div>
      <p>${renderMarkdown(payload.answer || "No grounded answer returned.")}</p>
    </div>
    <div class="research-meta-grid">
      <article><span>Confidence</span><strong>${escapeHtml(confidence.score != null ? `${Math.round(Number(confidence.score) * 100)}%` : "n/a")}</strong></article>
      <article><span>Sources</span><strong>${escapeHtml(sources.length)}</strong></article>
      <article><span>Latency</span><strong>${escapeHtml(Number(payload.latency_ms || 0).toFixed(1))} ms</strong></article>
      <article><span>Cache</span><strong>${cache.hit ? "hit" : "fresh"}</strong></article>
    </div>
    <div class="mini-chip-row">
      ${(retrieval.techniques || []).map(item => `<span>${escapeHtml(item)}</span>`).join("")}
    </div>
    ${conflicts.length ? `<div class="notice notice-warning"><strong>Evidence conflict detected</strong><p>${escapeHtml(conflicts.map(item => item.reason).join(" "))}</p></div>` : ""}
    <div class="knowledge-result-list rag-source-list">
      ${sources.length ? sources.map(source => `
        <article class="list-item">
          <div class="list-item-row">
            <div>
              <strong>[${escapeHtml(source.citation || "S")}] ${escapeHtml(source.title || "Local source")}</strong>
              <p>${escapeHtml((source.retrieval || []).join(" · ") || "local evidence")}</p>
            </div>
            <span class="badge">${escapeHtml(Number(source.score || 0).toFixed(3))}</span>
          </div>
        </article>
      `).join("") : emptyState("No evidence met the grounding threshold.")}
    </div>
  `);
}

function renderKnowledgeIngest(document, title = "Knowledge saved") {
  const tags = document.tags || [];
  setKnowledgeOutput(`
    <div class="knowledge-summary-card">
      <strong>${escapeHtml(title)}</strong>
      <p>${escapeHtml(document.title || "Document")} · ${escapeHtml(document.chunk_count ?? 0)} chunks · source ${escapeHtml(document.source || "local")}</p>
    </div>
    <div class="mini-chip-row">
      ${tags.length ? tags.map(tag => `<span>${escapeHtml(tag)}</span>`).join("") : `<span>No automatic tags yet</span>`}
    </div>
  `);
}

function renderKnowledgeSearch(query, results) {
  setKnowledgeOutput(`
    <div class="knowledge-summary-card">
      <strong>${escapeHtml(results.length)} results</strong>
      <p>${escapeHtml(query)} · ranked with local hybrid retrieval.</p>
    </div>
    <div class="knowledge-result-list">
      ${results.length ? results.map(result => `
        <article class="list-item">
          <div class="list-item-row">
            <div>
              <strong>${escapeHtml(result.title || "Knowledge chunk")}</strong>
              <p>${escapeHtml(result.content || result.snippet || "")}</p>
            </div>
            <span class="badge">${escapeHtml(result.score != null ? Number(result.score).toFixed(2) : result.type || "local")}</span>
          </div>
        </article>
      `).join("") : emptyState("No matching knowledge found yet.")}
    </div>
  `);
}

function renderKnowledgeSynthesis(payload) {
  const sources = payload.sources || payload.results || [];
  const entities = payload.related_entities || [];
  const relationships = payload.related_relationships || [];
  const synthesis = payload.synthesis || payload.summary || payload.answer || "No synthesis text returned.";
  setKnowledgeOutput(`
    <div class="knowledge-summary-card">
      <strong>Source-backed synthesis</strong>
      <p>${renderMarkdown(synthesis)}</p>
    </div>
    <div class="knowledge-steps">
      ${(payload.reasoning_steps || []).map(step => `<span>${escapeHtml(step)}</span>`).join("")}
    </div>
    <div class="research-meta-grid">
      <article><span>Sources</span><strong>${escapeHtml(sources.length)}</strong></article>
      <article><span>Entities</span><strong>${escapeHtml(entities.length)}</strong></article>
      <article><span>Relations</span><strong>${escapeHtml(relationships.length)}</strong></article>
      <article><span>Project</span><strong>${escapeHtml(payload.project || getProject())}</strong></article>
    </div>
    <div class="knowledge-result-list">
      ${sources.length ? sources.slice(0, 8).map(source => `
        <article class="list-item">
          <strong>${escapeHtml(source.title || source.source || "Local source")}</strong>
          <p>${escapeHtml(source.content || source.snippet || source.id || "")}</p>
        </article>
      `).join("") : emptyState("No direct source chunks were returned.")}
    </div>
  `);
}

function renderKnowledgeGraph(payload) {
  const nodes = (payload.nodes || []).slice(0, 32);
  const edges = payload.edges || [];
  const stats = payload.stats || { nodes: nodes.length, edges: edges.length, connected_nodes: 0 };
  if (!nodes.length) {
    setKnowledgeOutput(`
      <div class="knowledge-summary-card"><strong>Knowledge graph</strong><p>No connected concepts yet.</p></div>
      ${emptyState("Add a document with named concepts, then open the graph again.")}
    `);
    return;
  }

  const width = 760;
  const height = 390;
  const centerX = width / 2;
  const centerY = height / 2;
  const positions = new Map(nodes.map((node, index) => {
    if (nodes.length === 1) return [node.id, { x: centerX, y: centerY }];
    const ring = index < 12 ? 0 : 1;
    const ringIndex = ring ? index - 12 : index;
    const ringSize = ring ? Math.max(1, nodes.length - 12) : Math.min(12, nodes.length);
    const angle = (Math.PI * 2 * ringIndex / ringSize) - Math.PI / 2;
    const radiusX = ring ? 325 : 205;
    const radiusY = ring ? 158 : 112;
    return [node.id, { x: centerX + Math.cos(angle) * radiusX, y: centerY + Math.sin(angle) * radiusY }];
  }));
  const visibleEdges = edges.filter(edge => positions.has(edge.source) && positions.has(edge.target));
  const edgeMarkup = visibleEdges.map(edge => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    return `<line class="knowledge-graph-edge" x1="${source.x.toFixed(1)}" y1="${source.y.toFixed(1)}" x2="${target.x.toFixed(1)}" y2="${target.y.toFixed(1)}"><title>${escapeHtml(edge.label || "related to")}</title></line>`;
  }).join("");
  const nodeMarkup = nodes.map(node => {
    const point = positions.get(node.id);
    const radius = Math.min(16, 7 + Number(node.degree || 0) * 1.5);
    const label = String(node.name || "Entity");
    const shortLabel = label.length > 18 ? `${label.slice(0, 16)}…` : label;
    return `<g class="knowledge-graph-node" transform="translate(${point.x.toFixed(1)} ${point.y.toFixed(1)})">
      <circle r="${radius.toFixed(1)}"></circle>
      <text y="${(radius + 14).toFixed(1)}" text-anchor="middle">${escapeHtml(shortLabel)}</text>
      <title>${escapeHtml(label)} · ${escapeHtml(node.entity_type || "entity")} · ${escapeHtml(node.degree || 0)} connection(s)</title>
    </g>`;
  }).join("");
  setKnowledgeOutput(`
    <div class="knowledge-summary-card">
      <strong>Knowledge graph</strong>
      <p>${escapeHtml(stats.nodes || nodes.length)} concepts · ${escapeHtml(stats.edges || edges.length)} links · ${escapeHtml(stats.connected_nodes || 0)} connected.</p>
    </div>
    <div class="knowledge-graph-shell" role="img" aria-label="Knowledge graph with ${escapeHtml(nodes.length)} visible concepts and ${escapeHtml(visibleEdges.length)} visible links">
      <svg class="knowledge-graph-canvas" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
        <g>${edgeMarkup}</g>
        <g>${nodeMarkup}</g>
      </svg>
    </div>
  `);
}

// =============================================
// Workflows
// =============================================

async function loadWorkflows() {
  const payload = await api(`/api/workflows?project=${encodeURIComponent(getProject())}&limit=12`);
  state.workflows = payload.workflows || [];
  const el = qs("#workflows");
  if (el) el.innerHTML = state.workflows.length
    ? state.workflows.map(w =>
        listItem(w.name, `${w.description || "workflow"} — ${(w.steps || []).length} steps`,
          `<div class="toolbar slim">
            <button class="btn btn-mini btn-soft" data-workflow-run="${escapeHtml(w.id)}">Run</button>
            <button class="btn btn-mini btn-outline" data-workflow-visual="${escapeHtml(w.id)}">Visual</button>
          </div>`
        )
      ).join("")
    : emptyState("No workflows yet");
  qsa("[data-workflow-run]").forEach(b => b.addEventListener("click", () => runWorkflow(b.dataset.workflowRun)));
  qsa("[data-workflow-visual]").forEach(b => b.addEventListener("click", () => loadWorkflowVisual(b.dataset.workflowVisual)));
}

async function runWorkflow(id) {
  try {
    const payload = await api(`/api/workflows/${id}/run`, { method: "POST" });
    setOutput("#workflowOutput", payload);
    toast("Workflow ran", id.slice(0,8));
  } catch(e) { toast("Workflow error", e.message); }
}

async function loadWorkflowVisual(id) {
  try {
    const payload = await api(`/api/workflows/visual?workflow_id=${encodeURIComponent(id)}`);
    renderWorkflowCanvas(payload.nodes || [], payload.edges || []);
  } catch(e) { toast("Visual error", e.message); }
}

function renderWorkflowCanvas(nodes, edges) {
  const canvas = qs("#workflowCanvas");
  if (!canvas) return;
  canvas.innerHTML = nodes.map((node, i) =>
    `<div class="canvas-node ${i === 0 ? "trigger" : ""}" data-node-id="${escapeHtml(node.id)}"
          draggable="true"
          style="left:${node.x||i*110}px;top:${node.y||10}px">
       ${escapeHtml(node.label || node.id)}
     </div>`
  ).join("");
  // Persist canvas layout after drag
  qsa(".canvas-node").forEach(el => {
    el.addEventListener("dragend", async () => {
      const layout = Array.from(qsa(".canvas-node")).map(n => ({
        id: n.dataset.nodeId, x: parseInt(n.style.left, 10), y: parseInt(n.style.top, 10),
      }));
      try {
        await api("/api/workflows/layout", {
          method: "POST",
          body: JSON.stringify({ layout, project: getProject() }),
        });
      } catch(_) {}
    });
  });
}

function renderWorkflowAuthoring() {
  const canvas = qs("#workflowCanvas");
  if (!canvas) return;
  canvas.innerHTML = `<div class="workflow-authoring"><input id="workflowName" placeholder="Workflow name"><div class="workflow-node-editor"><label><span>When</span><select id="workflowTrigger"><option value="manual">Started manually</option><option value="scheduled">On a schedule</option><option value="event">When an event arrives</option></select></label><span class="workflow-connector">→</span><label><span>Do</span><select id="workflowAction"><option value="run_agent">Run an agent</option><option value="search_context">Search memory</option><option value="chat">Ask a model</option><option value="report_generate">Generate a report</option><option value="file_list">List local files</option><option value="monitor_check">Check a monitor</option><option value="alert">Create an alert</option></select></label></div><textarea id="workflowInstruction" rows="5" placeholder="Describe the goal, query, message, or report content…"></textarea><button id="saveWorkflow" class="btn btn-primary" type="button">Save workflow</button></div>`;
  qs("#saveWorkflow")?.addEventListener("click", createWorkflowFromEditor);
  qs("#workflowName")?.focus();
}

async function createWorkflowFromEditor() {
  try {
    const name = qs("#workflowName")?.value.trim();
    const action = qs("#workflowAction")?.value || "run_agent";
    const instruction = qs("#workflowInstruction")?.value.trim();
    if (!name || !instruction) { toast("Workflow needs a name and instruction"); return; }
    const fieldByAction = { run_agent: "goal", search_context: "query", chat: "message", report_generate: "content", file_list: "path", monitor_check: "target", alert: "message" };
    const step = { action, [fieldByAction[action] || "content"]: instruction };
    const d = await api("/api/workflows", { method: "POST", body: JSON.stringify({ name, project: getProject(), trigger_type: qs("#workflowTrigger")?.value || "manual", steps: [step] }) });
    setOutput("#workflowOutput", d);
    loadWorkflows();
    toast("Workflow saved", name);
  } catch(e) { toast("Workflow error", e.message); }
}

async function loadRuntimeTools() {
  try {
    const payload = await api("/api/runtime-tools/status");
    setOutput("#runtimeToolsOutput", payload);
  } catch (error) {
    toast("Runtime tools unavailable", error.message);
  }
}

async function runRuntimeBrowserTest() {
  const url = qs("#runtimeBrowserUrl")?.value.trim() || "";
  if (!url) { toast("Browser URL required"); return; }
  try {
    const payload = await api("/api/runtime-tools/browser-test", {
      method: "POST",
      body: JSON.stringify({
        url,
        project: getProject(),
        approved_external: Boolean(qs("#runtimeBrowserApproval")?.checked),
      }),
    });
    setOutput("#runtimeToolsOutput", payload);
    toast("Browser test completed");
  } catch (error) {
    toast("Browser test did not run", error.message);
  }
}

// =============================================
// Assistant
// =============================================

async function createTask() {
  const title = qs("#assistantTitle")?.value.trim();
  if (!title) { toast("Title required"); return; }
  try {
    const payload = await api("/api/assistant/tasks", {
      method: "POST",
      body: JSON.stringify({ title, description: qs("#assistantBody")?.value || "", project: getProject() }),
    });
    setOutput("#assistantOutput", payload);
    toast("Task created", title);
  } catch(e) { toast("Task error", e.message); }
}

async function createGoal() {
  const title = qs("#assistantTitle")?.value.trim();
  if (!title) { toast("Title required"); return; }
  try {
    const payload = await api("/api/assistant/goals", {
      method: "POST",
      body: JSON.stringify({ title, description: qs("#assistantBody")?.value || "", project: getProject() }),
    });
    setOutput("#assistantOutput", payload);
    toast("Goal created", title);
  } catch(e) { toast("Goal error", e.message); }
}

async function getDailyBrief() {
  try {
    const payload = await api(`/api/assistant/daily-brief?project=${encodeURIComponent(getProject())}`);
    setOutput("#assistantOutput", payload);
  } catch(e) { toast("Brief error", e.message); }
}

// =============================================
// Communications
// =============================================

async function importMessage() {
  try {
    const payload = await api("/api/communication/messages/import", {
      method: "POST",
      body: JSON.stringify({
        platform: qs("#commPlatform")?.value || "email",
        sender: qs("#commSender")?.value.trim() || "unknown",
        subject: qs("#commSubject")?.value.trim() || "",
        body: qs("#commBody")?.value.trim() || "",
        project: getProject(),
      }),
    });
    setOutput("#commOutput", payload);
    toast("Message imported");
  } catch(e) { toast("Import error", e.message); }
}

async function loadCommunicationFeed() {
  try {
    const [inbox, calendar] = await Promise.all([
      api(`/api/communication/inbox?project=${encodeURIComponent(getProject())}&limit=5`),
      api(`/api/communication/calendar?project=${encodeURIComponent(getProject())}&limit=5`),
    ]);
    const messages = inbox.messages || [];
    const events = calendar.events || [];
    const el = qs("#communicationFeed");
    if (el) el.innerHTML = (messages.length || events.length)
      ? [
          ...messages.map(m => listItem(m.subject || "Message", `${m.platform||"email"} — ${m.sender||"?"}`)),
          ...events.map(e => listItem(e.title || "Calendar event", `${e.starts_at||""}`)),
        ].join("")
      : emptyState("No messages or events");
  } catch(e) { console.warn("communicationFeed", e); }
}

async function loadCommunicationHub() {
  await Promise.allSettled([
    loadCommunicationFeed(),
    loadCommunicationOAuth(),
    loadCommunicationSyncStatus(),
  ]);
}

async function loadCommunicationOAuth() {
  try {
    const [providersPayload, accountsPayload] = await Promise.all([
      api("/api/communication/oauth/providers"),
      api(`/api/communication/oauth/accounts?project=${encodeURIComponent(getProject())}`),
    ]);
    state.communication.oauthProviders = providersPayload.providers || [];
    state.communication.oauthAccounts = accountsPayload.accounts || [];
    renderCommunicationOAuth();
    return state.communication;
  } catch(e) {
    renderCommunicationOAuth({ error: e.message });
    return null;
  }
}

function renderCommunicationOAuth(error = null) {
  const providers = state.communication.oauthProviders || [];
  const accounts = state.communication.oauthAccounts || [];
  const providerSelect = qs("#communicationOauthProvider");
  if (providerSelect) {
    const current = providerSelect.value;
    providerSelect.innerHTML = providers.length
      ? providers.map(provider => `<option value="${escapeHtml(provider.id)}">${escapeHtml(provider.name || provider.id)}</option>`).join("")
      : '<option value="generic">Generic OAuth</option>';
    if (current) providerSelect.value = current;
  }
  const accountSelect = qs("#communicationAccountSelect");
  if (accountSelect) {
    const current = accountSelect.value;
    accountSelect.innerHTML = accounts.length
      ? accounts.map(account => `<option value="${escapeHtml(account.account_id)}">${escapeHtml(account.label || account.provider)} · ${escapeHtml(account.status || "configured")}</option>`).join("")
      : '<option value="">No OAuth accounts</option>';
    if (current) accountSelect.value = current;
  }
  const cards = qs("#communicationStatusCards");
  if (cards) {
    cards.innerHTML = [
      opsTile("OAuth providers", providers.length || "0", "Gmail, Microsoft, generic"),
      opsTile("Accounts", accounts.length || "0", `${accounts.filter(a => a.token_present).length} token-ready`),
      opsTile("Privacy", "Local", "Secrets never returned by API"),
      opsTile("External calls", state.privacySettings.localOnly ? "Blocked" : "Explicit", "Token exchange requires approval"),
    ].join("");
  }
  const feed = qs("#communicationOAuthFeed");
  if (!feed) return;
  if (error?.error) {
    feed.innerHTML = emptyState(`OAuth status unavailable: ${error.error}`);
    return;
  }
  const providerRows = providers.slice(0, 4).map(provider => {
    const mailboxes = provider.mailbox_policy || {};
    const detail = Object.entries(mailboxes).map(([k, v]) => `${k}: ${v}`).join(" · ");
    return listItem(provider.name || provider.id, detail || "Provider policy ready", `<span class="badge badge-info">${escapeHtml(provider.id)}</span>`);
  });
  const accountRows = accounts.map(account => {
    const tokenBadge = account.token_present ? "badge-success" : "badge-warning";
    const status = `${account.provider} · ${account.platform} · ${account.redirect_uri || "no redirect URI"}`;
    return listItem(account.label || account.account_id, status, `<span class="badge ${tokenBadge}">${account.token_present ? "token ready" : "needs token"}</span>`);
  });
  feed.innerHTML = [...accountRows, ...providerRows].join("") || emptyState("Save an OAuth account to enable provider-backed email sync.");
}

async function createOauthAccount() {
  try {
    const scopes = (qs("#communicationOauthScopes")?.value || "")
      .split(/[,\s]+/)
      .map(item => item.trim())
      .filter(Boolean);
    const payload = await api("/api/communication/oauth/configure", {
      method: "POST",
      body: JSON.stringify({
        provider: qs("#communicationOauthProvider")?.value || "generic",
        platform: "email",
        project: getProject(),
        label: qs("#communicationOauthLabel")?.value.trim() || "",
        client_id: qs("#communicationOauthClientId")?.value.trim() || "",
        client_secret: qs("#communicationOauthClientSecret")?.value || "",
        redirect_uri: qs("#communicationOauthRedirect")?.value.trim() || "",
        auth_endpoint: qs("#communicationOauthAuthEndpoint")?.value.trim() || "",
        token_endpoint: qs("#communicationOauthTokenEndpoint")?.value.trim() || "",
        scopes,
      }),
    });
    setOutput("#commOutput", payload);
    toast("OAuth account saved", payload.account?.label || payload.account?.provider || "local account");
    await loadCommunicationOAuth();
  } catch(e) { toast("OAuth account error", e.message); }
}

async function buildOauthUrl() {
  const accountId = qs("#communicationAccountSelect")?.value;
  if (!accountId) { toast("Select an OAuth account first"); return; }
  try {
    const payload = await api("/api/communication/oauth/authorize-url", {
      method: "POST",
      body: JSON.stringify({ account_id: accountId, state: `openagent-${Date.now()}` }),
    });
    setOutput("#commOutput", payload);
    const feed = qs("#communicationOAuthFeed");
    if (feed) {
      feed.insertAdjacentHTML("afterbegin", listItem(
        "Authorization URL ready",
        payload.privacy_boundary || "Open this URL only when you approve the external provider step.",
        `<a class="btn btn-mini btn-primary" target="_blank" rel="noreferrer" href="${escapeHtml(payload.authorization_url || "#")}">Open</a>`
      ));
    }
    toast("Authorization URL built");
  } catch(e) { toast("Authorization URL error", e.message); }
}

async function refreshOauthToken() {
  const accountId = qs("#communicationAccountSelect")?.value;
  if (!accountId) { toast("Select an OAuth account first"); return; }
  const allowExternal = !state.privacySettings.localOnly && !!qs("#communicationAllowExternal")?.checked;
  try {
    const payload = await api("/api/communication/oauth/refresh", {
      method: "POST",
      body: JSON.stringify({
        account_id: accountId,
        authorization_code: qs("#communicationOauthCode")?.value.trim() || "",
        allow_external: allowExternal,
      }),
    });
    setOutput("#commOutput", payload);
    await loadCommunicationOAuth();
    toast(payload.approval_required ? "External approval required" : "Token status updated", payload.reason || payload.status || "");
  } catch(e) { toast("Token refresh error", e.message); }
}

async function loadCommunicationSyncStatus() {
  try {
    const [runsPayload, deadPayload] = await Promise.all([
      api(`/api/communication/sync-runs?project=${encodeURIComponent(getProject())}&limit=8`),
      api(`/api/communication/dead-letters?project=${encodeURIComponent(getProject())}`),
    ]);
    state.communication.syncRuns = runsPayload.runs || [];
    state.communication.deadLetters = deadPayload.dead_letters || [];
    renderCommunicationSyncStatus();
    return state.communication;
  } catch(e) {
    renderCommunicationSyncStatus({ error: e.message });
    return null;
  }
}

function renderCommunicationSyncStatus(error = null) {
  const syncFeed = qs("#communicationSyncFeed");
  const deadFeed = qs("#communicationDeadLetterFeed");
  if (error?.error) {
    if (syncFeed) syncFeed.innerHTML = emptyState(`Sync status unavailable: ${error.error}`);
    if (deadFeed) deadFeed.innerHTML = "";
    return;
  }
  const runs = state.communication.syncRuns || [];
  const deadLetters = state.communication.deadLetters || [];
  if (syncFeed) {
    syncFeed.innerHTML = runs.length
      ? runs.map(run => {
          const badgeClass = run.status === "ok" || run.status === "completed" ? "badge-success" : run.status === "failed" ? "badge-error" : "badge-warning";
          return listItem(
            `${run.platform || "connector"} sync`,
            `${run.status || "queued"} · ${run.started_at || ""} · imported ${run.imported_messages ?? 0} message(s)`,
            `<span class="badge ${badgeClass}">${escapeHtml(run.status || "unknown")}</span>`
          );
        }).join("")
      : emptyState("No connector sync runs yet.");
  }
  if (deadFeed) {
    deadFeed.innerHTML = deadLetters.length
      ? deadLetters.slice(0, 5).map(item => listItem(
          `Dead letter ${item.platform || "connector"}`,
          `${item.error || item.reason || "Provider payload was retained for review."}`,
          `<span class="badge badge-error">${escapeHtml(item.status || "dead")}</span>`
        )).join("")
      : `<div class="mini-status mini-status-success"><strong>No dead letters</strong><p>Failed provider payloads will appear here without crashing the runtime.</p></div>`;
  }
}

async function syncCommunicationProvider() {
  const allowExternal = !state.privacySettings.localOnly && !!qs("#communicationAllowExternal")?.checked;
  try {
    const payloadJson = readJsonInput("#commSyncPayload", {});
    const payload = await api("/api/communication/sync-resilient", {
      method: "POST",
      body: JSON.stringify({
        platform: qs("#commSyncPlatform")?.value || "email",
        project: getProject(),
        payload: payloadJson,
        allow_external: allowExternal,
        max_attempts: Number(qs("#commSyncAttempts")?.value || 3),
      }),
    });
    setOutput("#commOutput", payload);
    await loadCommunicationSyncStatus();
    toast("Connector sync finished", payload.status || payload.run?.status || "recorded");
  } catch(e) { toast("Connector sync error", e.message); }
}

// =============================================
// Audit / Security
// =============================================

function setAuthSession(payload) {
  state.authToken = payload?.access_token || "";
  state.authUser = payload?.user || null;
  try {
    if (state.authToken) sessionStorage.setItem("oa-auth-token", state.authToken);
    else sessionStorage.removeItem("oa-auth-token");
  } catch (e) {}
}

async function registerIdentityAccount() {
  try {
    const username = qs("#identityUsername")?.value.trim() || "";
    const password = qs("#identityPassword")?.value || "";
    await api("/api/auth/register", { method: "POST", body: JSON.stringify({ username, email: qs("#identityEmail")?.value.trim() || "", password }) });
    const session = await api("/api/auth/login", { method: "POST", body: JSON.stringify({ username_or_email: username, password }) });
    setAuthSession(session);
    await loadIdentityAdministration();
    toast("Local account created", username);
  } catch (error) { toast("Account creation failed", error.message); }
}

async function loginIdentityAccount() {
  try {
    const session = await api("/api/auth/login", { method: "POST", body: JSON.stringify({
      username_or_email: qs("#identityUsername")?.value.trim() || "", password: qs("#identityPassword")?.value || "",
    }) });
    setAuthSession(session);
    await loadIdentityAdministration();
    toast("Signed in", session.user?.display_name || session.user?.username || "Local account");
  } catch (error) { toast("Sign in failed", error.message); }
}

async function logoutIdentityAccount() {
  try { if (state.authToken) await api("/api/auth/logout", { method: "POST", body: "{}" }); } catch (e) {}
  setAuthSession({});
  await loadIdentityAdministration();
  toast("Signed out");
}

function base64urlToBytes(value) {
  const normalized = String(value || "").replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(normalized + "=".repeat((4 - normalized.length % 4) % 4));
  return Uint8Array.from(binary, character => character.charCodeAt(0));
}

function bytesToBase64url(value) {
  if (!value) return "";
  const bytes = new Uint8Array(value);
  let binary = "";
  bytes.forEach(byte => { binary += String.fromCharCode(byte); });
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function prepareCreationOptions(options) {
  return { ...options, challenge: base64urlToBytes(options.challenge),
    user: { ...options.user, id: base64urlToBytes(options.user.id) },
    excludeCredentials: (options.excludeCredentials || []).map(item => ({ ...item, id: base64urlToBytes(item.id) })) };
}

function prepareRequestOptions(options) {
  return { ...options, challenge: base64urlToBytes(options.challenge),
    allowCredentials: (options.allowCredentials || []).map(item => ({ ...item, id: base64urlToBytes(item.id) })) };
}

function serializePublicKeyCredential(credential) {
  const response = credential.response;
  const serialized = {
    id: credential.id, rawId: bytesToBase64url(credential.rawId), type: credential.type,
    authenticatorAttachment: credential.authenticatorAttachment || null,
    clientExtensionResults: credential.getClientExtensionResults(),
    response: { clientDataJSON: bytesToBase64url(response.clientDataJSON) },
  };
  if (response.attestationObject) serialized.response.attestationObject = bytesToBase64url(response.attestationObject);
  if (response.authenticatorData) serialized.response.authenticatorData = bytesToBase64url(response.authenticatorData);
  if (response.signature) serialized.response.signature = bytesToBase64url(response.signature);
  if (response.userHandle) serialized.response.userHandle = bytesToBase64url(response.userHandle);
  if (response.getTransports) serialized.response.transports = response.getTransports();
  return serialized;
}

async function addIdentityPasskey() {
  if (!state.authToken) { toast("Sign in before adding a passkey"); return; }
  if (!window.PublicKeyCredential || !navigator.credentials) { toast("Passkeys are unavailable in this desktop runtime"); return; }
  try {
    const options = await api("/api/auth/passkeys/registration/options", { method: "POST", body: "{}" });
    const credential = await navigator.credentials.create({ publicKey: prepareCreationOptions(options.publicKey) });
    await api("/api/auth/passkeys/registration/verify", { method: "POST", body: JSON.stringify({
      ceremony_id: options.ceremony_id, credential: serializePublicKeyCredential(credential), label: "Windows Hello / security key",
    }) });
    await loadIdentityAdministration();
    toast("Passkey added");
  } catch (error) { toast("Passkey registration failed", error.message); }
}

async function loginWithIdentityPasskey() {
  if (!window.PublicKeyCredential || !navigator.credentials) { toast("Passkeys are unavailable in this desktop runtime"); return; }
  try {
    const options = await api("/api/auth/passkeys/authentication/options", { method: "POST", body: JSON.stringify({ username_or_email: qs("#identityUsername")?.value.trim() || "" }) });
    const credential = await navigator.credentials.get({ publicKey: prepareRequestOptions(options.publicKey) });
    const session = await api("/api/auth/passkeys/authentication/verify", { method: "POST", body: JSON.stringify({ ceremony_id: options.ceremony_id, credential: serializePublicKeyCredential(credential) }) });
    setAuthSession(session);
    await loadIdentityAdministration();
    toast("Passkey sign in complete");
  } catch (error) { toast("Passkey sign in failed", error.message); }
}

async function rotateIdentityRecoveryCodes() {
  try {
    const result = await api("/api/auth/recovery/rotate", { method: "POST", body: JSON.stringify({ count: 8 }) });
    const output = qs("#identitySensitiveOutput");
    if (output) { output.hidden = false; output.textContent = `Recovery codes - shown once\n\n${(result.recovery_codes || []).join("\n")}`; }
    toast("Recovery codes rotated");
  } catch (error) { toast("Recovery rotation failed", error.message); }
}

async function localDeviceFingerprint() {
  const input = `${navigator.userAgent}|${navigator.platform}|${navigator.language}|${screen.width}x${screen.height}`;
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(input));
  return Array.from(new Uint8Array(digest)).map(byte => byte.toString(16).padStart(2, "0")).join("");
}

async function trustCurrentIdentityDevice() {
  try {
    const device = await api("/api/auth/devices/trust", { method: "POST", body: JSON.stringify({
      name: navigator.platform || "Local desktop", fingerprint: await localDeviceFingerprint(), trust_level: "standard", days: 30,
    }) });
    try { sessionStorage.setItem("oa-device-token", device.device?.device_token || ""); } catch (e) {}
    await loadIdentityAdministration();
    toast("Device trusted for 30 days");
  } catch (error) { toast("Device trust failed", error.message); }
}

async function saveIdentityFederationPolicy() {
  const values = selector => (qs(selector)?.value || "").split(",").map(item => item.trim()).filter(Boolean);
  try {
    await api("/api/auth/federation/policies", { method: "POST", body: JSON.stringify({
      provider: qs("#identityFederationProvider")?.value.trim() || "",
      issuer: qs("#identityFederationIssuer")?.value.trim() || "",
      audience: qs("#identityFederationAudience")?.value.trim() || "",
      allowed_domains: values("#identityFederationDomains"), required_groups: values("#identityFederationGroups"),
      default_role: "member", enabled: true,
    }) });
    await loadIdentityAdministration();
    toast("Federation policy saved");
  } catch (error) { toast("Federation policy failed", error.message); }
}

async function loadIdentityAdministration() {
  const status = await api("/api/auth/identity/status").catch(() => ({}));
  let me = null;
  let passkeys = [];
  let devices = [];
  let policies = [];
  if (state.authToken) {
    const results = await Promise.allSettled([api("/api/auth/me"), api("/api/auth/passkeys"), api("/api/auth/devices"), api("/api/auth/federation/policies")]);
    if (results[0].status === "fulfilled") { me = results[0].value.user; state.authUser = me; }
    else { setAuthSession({}); }
    if (results[1].status === "fulfilled") passkeys = results[1].value.passkeys || [];
    if (results[2].status === "fulfilled") devices = results[2].value.devices || [];
    if (results[3].status === "fulfilled") policies = results[3].value.policies || [];
  }
  const badge = qs("#identitySessionBadge");
  if (badge) { badge.textContent = me ? `${me.display_name || me.username} · ${me.role}` : "Signed out"; badge.classList.toggle("badge-success", !!me); }
  const cards = qs("#identityStatusCards");
  if (cards) cards.innerHTML = [
    ["WebAuthn", status.webauthn_available ? "Ready" : "Dependency missing"], ["Passkeys", passkeys.length],
    ["Trusted devices", devices.filter(item => !item.revoked_at).length], ["Federation policies", policies.length],
  ].map(([label, value]) => `<div class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
  const resources = qs("#identityResourceList");
  if (resources) resources.innerHTML = me
    ? [...passkeys.map(item => listItem(item.label || "Passkey", `${item.device_type || "credential"} · counter ${item.sign_count || 0}`)),
       ...devices.map(item => listItem(item.name, `${item.trust_level} · ${item.revoked_at ? "revoked" : "trusted until " + item.expires_at}`))].join("") || emptyState("No passkeys or trusted devices")
    : emptyState("Sign in to manage passkeys, recovery, and trusted devices");
  const federation = qs("#identityFederationList");
  if (federation) federation.innerHTML = policies.length
    ? policies.map(item => listItem(item.provider, `${item.issuer} · ${item.audience} · ${item.enabled ? "enabled" : "disabled"}`)).join("")
    : emptyState(me ? "No federation policies" : "Administrator sign in required");
}

function auditEventTime(event) {
  const parsed = Date.parse(event.created_at || "");
  return Number.isFinite(parsed) ? parsed : 0;
}

function auditIncidentKey(event) {
  return `${event.action || "action"}\u0000${event.resource || "resource"}`;
}

function classifyAuditIncidents(events, now = Date.now()) {
  const ordered = [...events].sort((a, b) => auditEventTime(b) - auditEventTime(a));
  const current = [];
  const history = [];
  const recentWindowMs = 15 * 60 * 1000;
  ordered.filter(event => event.status === "failed").forEach(failure => {
    const failureTime = auditEventTime(failure);
    const recovered = ordered.some(event =>
      event.status !== "failed"
      && auditIncidentKey(event) === auditIncidentKey(failure)
      && auditEventTime(event) > failureTime
    );
    if (!recovered && failureTime > 0 && failureTime <= now + 60000 && now - failureTime <= recentWindowMs) current.push(failure);
    else history.push({ ...failure, incident_state: recovered ? "resolved" : "aged" });
  });
  return { current, history };
}

function renderDashboardAudit(events) {
  const chart = qs("#dashboardAuditChart");
  const failures = events.filter(event => event.status === "failed");
  const incidents = classifyAuditIncidents(events);
  const currentSet = new Set(incidents.current);
  const historyMap = new Map(incidents.history.map(event => [auditIncidentKey(event) + auditEventTime(event), event.incident_state]));
  const allowed = events.filter(event => event.status !== "failed");
  if (chart) {
    chart.setAttribute("aria-label", `${events.length} recent audit actions: ${allowed.length} allowed, ${incidents.current.length} current incidents, and ${incidents.history.length} historical failures.`);
    chart.innerHTML = events.length ? `
      <div class="audit-bars" aria-hidden="true">
        ${[...events].reverse().map(event => {
          const historical = historyMap.has(auditIncidentKey(event) + auditEventTime(event));
          const eventClass = currentSet.has(event) ? "failed" : historical ? "historical" : "allowed";
          return `<i class="${eventClass}" title="${escapeHtml(event.action || "action")}"></i>`;
        }).join("")}
      </div>
      <div class="audit-summary"><span><b>${allowed.length}</b> allowed</span><span class="${incidents.current.length ? "has-failures" : ""}"><b>${incidents.current.length}</b> current</span><span><b>${incidents.history.length}</b> history</span></div>
    ` : `<div class="chart-empty"><strong>No audit activity</strong><span>Actions appear here as they are recorded.</span></div>`;
  }
  const rail = qs("#dashboardFailures");
  if (rail) {
    if (incidents.current.length) {
      const latest = incidents.current[0];
      const detail = latest.metadata?.error || latest.resource || "No failure detail was recorded.";
      rail.innerHTML = `<div><span>Attention required</span><strong>${escapeHtml(latest.action || "Failed action")}</strong><p>${escapeHtml(String(detail))}</p></div><button class="btn btn-mini btn-outline" data-view-jump="security">Inspect</button>`;
      rail.classList.add("visible");
    } else {
      rail.innerHTML = "";
      rail.classList.remove("visible");
    }
  }
  const dashboard = qs("#auditEvents");
  const visibleEvents = events.filter(event => {
    if (state.dashboardUi.incidentFilter === "all") return true;
    if (state.dashboardUi.incidentFilter === "current") return currentSet.has(event);
    return event.status === "failed" && historyMap.has(auditIncidentKey(event) + auditEventTime(event));
  });
  if (dashboard) dashboard.innerHTML = visibleEvents.length ? visibleEvents.slice(0, 8).map(event => {
    const incidentState = currentSet.has(event) ? "current" : historyMap.get(auditIncidentKey(event) + auditEventTime(event));
    return `
    <div class="audit-event-row ${incidentState === "current" ? "failed current" : incidentState ? "historical" : ""}">
      <i aria-hidden="true"></i>
      <div><strong>${escapeHtml(event.action || "action")}</strong><span>${escapeHtml(event.actor || "system")} · ${escapeHtml(event.resource || "local resource")}</span></div>
      <time datetime="${escapeHtml(event.created_at || "")}">${escapeHtml(formatDateTime(event.created_at))}</time>
      <small>${escapeHtml(incidentState || event.status || "allowed")}</small>
    </div>
  `; }).join("") : emptyState(state.dashboardUi.incidentFilter === "all" ? "No audit events yet" : `No ${state.dashboardUi.incidentFilter} incidents in the loaded audit window.`);
}

async function loadAudit() {
  try {
    const payload = await api(`/api/audit?project=${encodeURIComponent(getProject())}&limit=8`);
    const events = payload.events || [];
    state.dashboardSources.audit = true;
    state.auditEvents = events;
    renderDashboardAudit(events);
    renderMetrics();
    const securityHtml = events.length
      ? events.map(e => listItem(e.action, `${e.actor||"system"} — ${e.status||"ok"}`)).join("")
      : emptyState("No audit events yet");
    const s = qs("#securityAuditFeed"); if (s) s.innerHTML = securityHtml;
  } catch(e) {
    state.dashboardSources.audit = false;
    renderMetrics();
    console.warn("loadAudit", e);
  }
}

async function loadPrivacyReport() {
  try {
    const payload = await api(`/api/security/privacy-report?project=${encodeURIComponent(getProject())}`);
    setOutput("#privacyOutput", payload);
  } catch(e) { toast("Privacy report error", e.message); }
}

// =============================================
// Developer
// =============================================

async function loadDeveloperFeed() {
  try {
    const schema = await api("/api/developer/schema");
    const fields = Object.entries(schema.fields || {});
    const el = qs("#developerFeed");
    if (el) el.innerHTML = fields.length
      ? fields.slice(0,12).map(([n,d]) => listItem(n, d)).join("")
      : emptyState("No schema fields");
  } catch(e) { console.warn("developerFeed", e); }
}

async function loadDeveloperSdk() {
  try {
    const sdk = await api("/api/developer/sdk");
    setOutput("#developerOutput", sdk);
  } catch(e) { console.warn("developerSdk", e); }
}

async function createApiKey() {
  try {
    const payload = await api("/api/developer/api-keys", {
      method: "POST",
      body: JSON.stringify({ label: "ui-generated", scopes: ["api.read", "api.write"] }),
    });
    setOutput("#developerOutput", payload);
    toast("API key created");
  } catch(e) { toast("API key error", e.message); }
}

function showSkillResult(value, tone = "") {
  const output = qs("#skillEditorResult");
  if (!output) return;
  output.className = `mini-status${tone ? ` mini-status-${tone}` : ""}`;
  output.textContent = typeof value === "string" ? value : formatStructuredOutput(value);
}

async function loadSkills() {
  try {
    const project = encodeURIComponent(getProject());
    const [payload, marketplace, installations] = await Promise.all([
      api(`/api/skills?project=${project}`),
      api(`/api/skills/marketplace?project=${project}`),
      api(`/api/skills/installations?project=${project}`),
    ]);
    const skills = payload.skills || [];
    const list = qs("#skillEditorList");
    if (list) {
      list.innerHTML = skills.length ? skills.map(skill => listItem(
        skill.name, `${skill.version} · ${skill.status || "draft"}`,
        `<button class="btn btn-mini btn-soft" data-open-skill="${escapeHtml(skill.id)}">Edit</button>`
      )).join("") : emptyState("No local skills yet");
      qsa("[data-open-skill]", list).forEach(button => button.addEventListener("click", () => openSkill(button.dataset.openSkill)));
    }
    const marketList = qs("#skillMarketplaceList");
    if (marketList) {
      const installedListingIds = new Set((installations.installations || []).filter(item => item.status === "installed").map(item => item.listing_id));
      marketList.innerHTML = (marketplace.listings || []).length
        ? (marketplace.listings || []).map(listing => listItem(
            `${listing.name} ${listing.version}`,
            `${listing.visibility} · ${listing.status}`,
            installedListingIds.has(listing.id)
              ? '<span class="badge badge-success">Installed</span>'
              : `<button class="btn btn-mini btn-outline" data-install-skill="${escapeHtml(listing.id)}">Install</button>`,
          )).join("")
        : emptyState("No signed local marketplace listings yet");
      qsa("[data-install-skill]", marketList).forEach(button => button.addEventListener("click", () => installMarketplaceSkill(button.dataset.installSkill)));
    }
  } catch (error) { showSkillResult(`Skill list failed: ${error.message}`, "error"); }
}

async function openSkill(skillId) {
  try {
    const payload = await api(`/api/skills/${encodeURIComponent(skillId)}`);
    const skill = payload.skill || {};
    state.selectedSkillId = skill.id || "";
    if (qs("#skillEditorName")) qs("#skillEditorName").value = skill.name || "";
    if (qs("#skillEditorVersion")) qs("#skillEditorVersion").value = skill.version || "0.1.0";
    if (qs("#skillEditorMarkdown")) qs("#skillEditorMarkdown").value = skill.markdown || "";
    if (qs("#skillEditorManifest")) qs("#skillEditorManifest").value = JSON.stringify(skill.manifest || {}, null, 2);
    const badge = qs("#skillEditorBadge"); if (badge) badge.textContent = skill.status || "draft";
    showSkillResult(`Editing ${skill.name}`);
  } catch (error) { showSkillResult(error.message, "error"); }
}

async function saveSkillEditor() {
  try {
    const manifest = readJsonInput("#skillEditorManifest", {});
    const payload = await api("/api/skills", { method: "POST", body: JSON.stringify({
      skill_id: state.selectedSkillId, project: getProject(), name: qs("#skillEditorName")?.value.trim() || "",
      version: qs("#skillEditorVersion")?.value.trim() || "0.1.0", markdown: qs("#skillEditorMarkdown")?.value || "", manifest,
    }) });
    state.selectedSkillId = payload.skill?.id || state.selectedSkillId;
    showSkillResult(payload.skill || payload, "success");
    await loadSkills();
  } catch (error) { showSkillResult(error.message, "error"); }
}

async function runSkillEditorAction(action) {
  if (!state.selectedSkillId) { showSkillResult("Save or select a skill first.", "error"); return; }
  try {
    let payload;
    if (action === "permissions" || action === "dependency-scan") {
      payload = await api(`/api/skills/${encodeURIComponent(state.selectedSkillId)}/${action}`);
    } else {
      const signingActions = new Set(["package", "publish"]);
      const body = signingActions.has(action)
        ? { signing_key: qs("#skillSigningKey")?.value || "", visibility: "local" }
        : action === "sandbox"
          ? { input: { request: "Local UI sandbox verification" }, approved_permissions: [] }
          : {};
      payload = await api(`/api/skills/${encodeURIComponent(state.selectedSkillId)}/${action}`, { method: "POST", body: JSON.stringify(body) });
    }
    showSkillResult(payload, payload.status === "failed" || payload.error ? "error" : "success");
    const badge = qs("#skillEditorBadge"); if (badge && payload.status) badge.textContent = payload.status;
    await loadSkills();
  } catch (error) { showSkillResult(error.message, "error"); }
}

async function installMarketplaceSkill(listingId) {
  try {
    const payload = await api(`/api/skills/marketplace/${encodeURIComponent(listingId)}/install`, {
      method: "POST",
      body: JSON.stringify({ project: getProject(), signing_key: qs("#skillSigningKey")?.value || "" }),
    });
    showSkillResult(payload, "success");
    await loadSkills();
  } catch (error) { showSkillResult(error.message, "error"); }
}

async function loadIntegrationStatus() {
  try {
    const [status, safe, parity] = await Promise.all([
      api(`/api/integrations/status?project=${encodeURIComponent(getProject())}`),
      api(`/api/integrations/safe-shell?project=${encodeURIComponent(getProject())}&limit=5`),
      api(`/api/integrations/parity?project=${encodeURIComponent(getProject())}`),
    ]);
    state.integrations = { status, safe, parity };
    renderIntegrationStatus(state.integrations);
    return state.integrations;
  } catch(e) {
    renderIntegrationStatus({ error: e.message });
    return null;
  }
}

function renderIntegrationStatus(data = {}) {
  const cards = qs("#integrationStatusCards");
  const feed = qs("#integrationFeed");
  if (data.error) {
    if (cards) cards.innerHTML = `<article class="ops-tile ops-danger"><span>Status</span><strong>Attention</strong><p>${escapeHtml(data.error)}</p></article>`;
    if (feed) feed.innerHTML = emptyState("Integration status is unavailable");
    return;
  }
  const status = data.status || {};
  const safe = data.safe || {};
  const parity = data.parity || {};
  const latest = status.diagnostics?.latest?.report || status.diagnostics?.latest || null;
  if (cards) cards.innerHTML = [
    opsTile("Pairing", `${status.pairing?.active_codes ?? 0} active`, "Hashed one-time codes"),
    opsTile("Backups", status.backups ?? 0, "Local restore points"),
    opsTile("Diagnostics", latest?.status || "ready", "Local module checks"),
    opsTile("Safe Tools", (safe.commands || []).length, "Allowlisted commands"),
  ].join("");
  const items = [];
  if (latest) {
    items.push(listItem(
      "Latest diagnostic",
      `${latest.status || "ok"} — ${(latest.issues || []).length} issue(s)`,
      `<span class="badge ${latest.status === "ok" ? "badge-success" : "badge-warning"}">${escapeHtml(latest.status || "ok")}</span>`
    ));
  }
  (safe.runs || []).slice(0, 3).forEach(run => {
    items.push(listItem(
      `Safe command: ${run.command_key}`,
      `${run.status} — ${run.duration_ms || 0} ms`,
      `<span class="badge ${run.status === "completed" ? "badge-success" : "badge-warning"}">${escapeHtml(run.status)}</span>`
    ));
  });
  (parity.items || []).slice(0, 4).forEach(item => {
    items.push(listItem(item.label, item.endpoint, `<span class="badge badge-success">${escapeHtml(item.status)}</span>`));
  });
  if (feed) feed.innerHTML = items.length ? items.join("") : emptyState("No integration activity yet");
}

function opsTile(label, value, detail) {
  return `<article class="ops-tile">
    <span>${escapeHtml(label)}</span>
    <strong>${escapeHtml(String(value))}</strong>
    <p>${escapeHtml(detail)}</p>
  </article>`;
}

async function createPairingCode() {
  try {
    const payload = await api("/api/integrations/pairing-codes", {
      method: "POST",
      body: JSON.stringify({
        project: getProject(),
        label: "Desktop companion",
        scopes: ["api.read", "workspace.read", "chat.write"],
        ttl_minutes: 15,
      }),
    });
    const code = payload.code || {};
    const feed = qs("#integrationFeed");
    if (feed) feed.innerHTML = listItem(
      "Pairing code created",
      `Expires ${code.expires_at || "soon"}`,
      `<code class="inline-code">${escapeHtml(code.code || "")}</code>`
    ) + feed.innerHTML;
    setOutput("#developerOutput", { pairing_code: { ...code, code: code.code ? `${code.code.slice(0, 8)}…` : "" } });
    toast("Pairing code ready", "Code is shown once in Local operations.");
    loadIntegrationStatus().catch(() => {});
  } catch(e) { toast("Pairing error", e.message); }
}

async function runIntegrationDiagnostics() {
  try {
    const payload = await api("/api/integrations/diagnostics", {
      method: "POST",
      body: JSON.stringify({ project: getProject() }),
    });
    const diagnostic = payload.diagnostic || {};
    const feed = qs("#integrationFeed");
    if (feed) feed.innerHTML = listItem(
      "Diagnostic completed",
      `${diagnostic.status || "ok"} — ${(diagnostic.issues || []).length} issue(s)`,
      `<span class="badge ${diagnostic.status === "ok" ? "badge-success" : "badge-warning"}">${escapeHtml(diagnostic.status || "ok")}</span>`
    ) + feed.innerHTML;
    setOutput("#developerOutput", diagnostic);
    loadIntegrationStatus().catch(() => {});
  } catch(e) { toast("Diagnostics error", e.message); }
}

async function createIntegrationBackup() {
  try {
    const payload = await api("/api/integrations/backups", {
      method: "POST",
      body: JSON.stringify({ project: getProject() }),
    });
    const backup = payload.backup || {};
    const feed = qs("#integrationFeed");
    if (feed) feed.innerHTML = listItem(
      "Backup completed",
      backup.path || "Local backup saved",
      `<span class="badge badge-success">${escapeHtml(backup.status || "completed")}</span>`
    ) + feed.innerHTML;
    setOutput("#developerOutput", payload);
    loadIntegrationStatus().catch(() => {});
  } catch(e) { toast("Backup error", e.message); }
}

async function runSafePythonCheck() {
  try {
    const payload = await api("/api/integrations/safe-shell", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), command_key: "python-version" }),
    });
    const run = payload.run || {};
    const feed = qs("#integrationFeed");
    if (feed) feed.innerHTML = listItem(
      "Runtime check",
      `${run.output || run.error || "No output"} (${run.duration_ms || 0} ms)`,
      `<span class="badge ${run.status === "completed" ? "badge-success" : "badge-warning"}">${escapeHtml(run.status || "done")}</span>`
    ) + feed.innerHTML;
    setOutput("#developerOutput", payload);
    loadIntegrationStatus().catch(() => {});
  } catch(e) { toast("Runtime check error", e.message); }
}

async function saveUiPreference() {
  try {
    const payload = await api("/api/integrations/preferences", {
      method: "POST",
      body: JSON.stringify({
        project: getProject(),
        scope: "ui",
        key: "workspace",
        value: {
          theme: state.theme,
          sidebar_collapsed: document.body.classList.contains("sidebar-collapsed"),
          active_view: state.activeView,
        },
      }),
    });
    const pref = payload.preference || {};
    const feed = qs("#integrationFeed");
    if (feed) feed.innerHTML = listItem("Preference saved", `${pref.scope || "ui"}:${pref.key || "workspace"}`) + feed.innerHTML;
    setOutput("#developerOutput", payload);
    toast("Preference saved", "Stored locally in OpenAgent.");
    loadIntegrationStatus().catch(() => {});
  } catch(e) { toast("Preference error", e.message); }
}

async function loadPersonalOsStatus() {
  try {
    const project = encodeURIComponent(getProject());
    const [status, presets, signatures, drafts, notes, gallery, compares] = await Promise.all([
      api(`/api/personal-os/status?project=${project}`),
      api(`/api/personal-os/presets?project=${project}`),
      api(`/api/personal-os/signatures?project=${project}`),
      api(`/api/personal-os/drafts?project=${project}`),
      api(`/api/personal-os/notes?project=${project}`),
      api(`/api/personal-os/gallery?project=${project}`),
      api(`/api/personal-os/compare?project=${project}`),
    ]);
    state.personalOs = { status, presets, signatures, drafts, notes, gallery, compares };
    renderPersonalOsStatus(state.personalOs);
    return state.personalOs;
  } catch(e) {
    renderPersonalOsStatus({ error: e.message });
    return null;
  }
}

function renderPersonalOsStatus(data = {}) {
  const cards = qs("#personalOsCards");
  const feed = qs("#personalOsFeed");
  if (data.error) {
    if (cards) cards.innerHTML = `<article class="ops-tile ops-danger"><span>Status</span><strong>Attention</strong><p>${escapeHtml(data.error)}</p></article>`;
    if (feed) feed.innerHTML = emptyState("Personal OS status is unavailable");
    return;
  }
  const counts = data.status?.counts || {};
  if (cards) cards.innerHTML = [
    opsTile("Presets", counts.presets ?? 0, "Reusable chat and model profiles"),
    opsTile("Drafts", counts.drafts ?? 0, "Versioned local writing"),
    opsTile("Notes", counts.notes ?? 0, "Tagged tasks and reminders"),
    opsTile("Gallery", counts.gallery ?? 0, "Local media references"),
  ].join("");

  const items = [];
  (data.drafts?.drafts || []).slice(0, 2).forEach(draft => {
    items.push(listItem(`Draft: ${draft.title}`, `v${draft.version} · ${draft.status} · ${draft.updated_at}`));
  });
  (data.notes?.notes || []).slice(0, 2).forEach(note => {
    const tags = (note.tags || []).join(", ") || "no tags";
    items.push(listItem(`Note: ${note.title}`, `${note.status} · ${tags}${note.due_at ? ` · due ${note.due_at}` : ""}`));
  });
  (data.gallery?.assets || []).slice(0, 2).forEach(asset => {
    items.push(listItem(`Gallery: ${asset.title}`, `${asset.media_type} · ${(asset.tags || []).join(", ") || "untagged"}`));
  });
  (data.compares?.compares || []).slice(0, 2).forEach(compare => {
    items.push(listItem(`Compare: ${compare.prompt}`, compare.winner ? `Winner: ${compare.winner}` : "Awaiting vote"));
  });
  if (feed) feed.innerHTML = items.length ? items.join("") : emptyState("Create a preset, draft, note, gallery asset, or compare run to begin.");
}

function personalOsForm() {
  return {
    title: qs("#personalOsTitle")?.value.trim() || "",
    body: qs("#personalOsBody")?.value.trim() || "",
    project: getProject(),
  };
}

function personalOsTags(text) {
  const tags = [];
  String(text || "").replace(/#([a-z0-9_-]+)/gi, (_, tag) => {
    const clean = tag.toLowerCase();
    if (!tags.includes(clean)) tags.push(clean);
    return "";
  });
  return tags.length ? tags.slice(0, 8) : ["personal"];
}

async function savePersonalPreset() {
  const form = personalOsForm();
  try {
    const payload = await api("/api/personal-os/presets", {
      method: "POST",
      body: JSON.stringify({
        project: form.project,
        name: form.title || "Local assistant preset",
        kind: "chat",
        provider: "auto",
        model: state.modelSettings.chatModel || DEFAULT_MODEL_SETTINGS.chatModel,
        instructions: form.body || "Answer with local project context first and ask before external actions.",
        parameters: { temperature: 0.4, privacy: "local-first" },
      }),
    });
    setOutput("#developerOutput", payload);
    toast("Preset saved", payload.preset?.name || "Personal OS");
    await loadPersonalOsStatus();
  } catch(e) { toast("Preset error", e.message); }
}

async function savePersonalSignature() {
  const form = personalOsForm();
  try {
    const payload = await api("/api/personal-os/signatures", {
      method: "POST",
      body: JSON.stringify({
        project: form.project,
        label: form.title || "Default signature",
        content: form.body || "Best regards,\nOpenAgent",
        is_default: true,
      }),
    });
    setOutput("#developerOutput", payload);
    toast("Signature saved", payload.signature?.label || "");
    await loadPersonalOsStatus();
  } catch(e) { toast("Signature error", e.message); }
}

async function savePersonalDraft() {
  const form = personalOsForm();
  try {
    const payload = await api("/api/personal-os/drafts", {
      method: "POST",
      body: JSON.stringify({
        project: form.project,
        title: form.title || "Untitled draft",
        content: form.body || "Draft content",
        status: "draft",
      }),
    });
    setOutput("#developerOutput", payload);
    toast("Draft saved", `Version ${payload.draft?.version || 1}`);
    await loadPersonalOsStatus();
  } catch(e) { toast("Draft error", e.message); }
}

async function savePersonalNote() {
  const form = personalOsForm();
  try {
    const payload = await api("/api/personal-os/notes", {
      method: "POST",
      body: JSON.stringify({
        project: form.project,
        title: form.title || "Quick note",
        body: form.body || "",
        tags: personalOsTags(`${form.title} ${form.body}`),
        status: "open",
      }),
    });
    setOutput("#developerOutput", payload);
    toast("Note saved", payload.note?.title || "");
    await loadPersonalOsStatus();
  } catch(e) { toast("Note error", e.message); }
}

async function addGalleryAsset() {
  const form = personalOsForm();
  try {
    const payload = await api("/api/personal-os/gallery", {
      method: "POST",
      body: JSON.stringify({
        project: form.project,
        title: form.title || "Local gallery asset",
        media_type: "document",
        source_path: form.body.startsWith("/") || /^[A-Za-z]:\\/.test(form.body) ? form.body : "",
        content: form.body,
        tags: personalOsTags(`${form.title} ${form.body}`),
        metadata: { added_from: "developer-ui" },
      }),
    });
    setOutput("#developerOutput", payload);
    toast("Gallery asset saved", payload.asset?.title || "");
    await loadPersonalOsStatus();
  } catch(e) { toast("Gallery error", e.message); }
}

async function startCompareRun() {
  const form = personalOsForm();
  const parts = form.body.split(/\n-{3,}\n/);
  if (!parts[0]?.trim() || !parts[1]?.trim()) {
    toast("Compare needs two outputs", "Put option A, then a line with ---, then option B.");
    return;
  }
  try {
    const payload = await api("/api/personal-os/compare/start", {
      method: "POST",
      body: JSON.stringify({
        project: form.project,
        prompt: form.title || "Local response comparison",
        left_label: "A",
        left_output: parts[0].trim(),
        right_label: "B",
        right_output: parts.slice(1).join("\n---\n").trim(),
      }),
    });
    setOutput("#developerOutput", payload);
    toast("Compare run saved", payload.compare?.prompt || "");
    await loadPersonalOsStatus();
  } catch(e) { toast("Compare error", e.message); }
}

async function exportPersonalOs() {
  try {
    const payload = await api(`/api/personal-os/export?project=${encodeURIComponent(getProject())}`);
    setOutput("#developerOutput", payload);
    toast("Personal OS exported", `${(payload.notes || []).length} notes, ${(payload.drafts || []).length} drafts`);
    await loadPersonalOsStatus();
  } catch(e) { toast("Export error", e.message); }
}

// =============================================
// Deep operational modules
// =============================================

const TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAMAAAAoyzS7AAAAA1BMVEUAAACnej3aAAAAAXRSTlMAQObYZgAAAApJREFUCNdjYAAAAAIAAeIhvDMAAAAASUVORK5CYII=";

async function loadOperationalModules() {
  await Promise.allSettled([
    loadModelCookbook(),
    loadInferenceEngine(),
    loadMcpRuntime(),
    loadTerminalRuns(),
    loadSchedulerStatus(),
    loadImageStudio(),
    loadAgentBridgeStatus(),
    loadProviderCertification(false),
  ]);
}

async function loadModelCookbook() {
  try {
    const project = encodeURIComponent(getProject());
    const [hosts, cached, fit, certifications] = await Promise.all([
      api(`/api/model-cookbook/hosts?project=${project}`),
      api(`/api/model-cookbook/cached?project=${project}`),
      api(`/api/model-cookbook/hardware-fit?project=${project}&model_ref=${encodeURIComponent(state.modelSettings.chatModel || "local-model")}`),
      api(`/api/model-cookbook/certifications?project=${project}`),
    ]);
    const feed = qs("#modelCookbookFeed");
    if (feed) {
      const rows = [
        listItem("Hosts", `${(hosts.hosts || []).length} configured`, "Safe local/remote model endpoints"),
        listItem("Cached model files", `${(cached.files || []).length} discovered`, "Scans local model folders only"),
        listItem("Hardware fit", `${fit.recommended_bits || "auto"} bit`, `${fit.expected_memory_gb || "?"} GB expected memory`),
        listItem("Certified endpoints", `${(certifications.certifications || []).filter(item => item.status === "certified").length} active`, `${(certifications.certifications || []).length} signed probe report(s)`),
      ];
      feed.innerHTML = rows.join("");
    }
    return { hosts, cached, fit, certifications };
  } catch(e) {
    const feed = qs("#modelCookbookFeed");
    if (feed) feed.innerHTML = emptyState(`Cookbook unavailable: ${e.message}`);
    return null;
  }
}

async function certifyModelEndpoint() {
  try {
    const endpoint = qs("#modelCertificationEndpoint")?.value.trim() || "";
    const payload = await api("/api/model-cookbook/certify", {
      method: "POST",
      body: JSON.stringify({
        project: getProject(), endpoint,
        model_ref: qs("#modelCertificationModel")?.value.trim() || "",
        allow_external: !state.privacySettings.localOnly,
        completion_probe: true,
      }),
    });
    setOutput("#developerOutput", payload);
    toast("Endpoint certification complete", payload.certification?.status || "recorded");
    await loadModelCookbook();
  } catch(e) { toast("Endpoint certification failed", e.message); }
}

async function loadInferenceEngine() {
  try {
    const [inference, rag] = await Promise.all([
      api("/api/inference/status"),
      api(`/api/rag/status?project=${encodeURIComponent(getProject())}`),
    ]);
    const feed = qs("#inferenceEngineFeed");
    if (feed) {
      const cache = inference.cache || {};
      const latency = inference.latency || {};
      const vector = rag.vector_store || {};
      feed.innerHTML = [
        listItem("First-token latency", latency.first_token_p50_ms != null ? `${latency.first_token_p50_ms} ms p50` : "No samples yet", `Target ${latency.target_first_token_ms || 1000} ms`),
        listItem("Response cache", `${cache.entries || 0}/${cache.max_entries || 0} entries`, `${Math.round(Number(cache.hit_rate || 0) * 100)}% hit rate`),
        listItem("Vector engines", (vector.active_accelerators || []).join(" + ") || "SQLite fallback", `${vector.chunk_count || 0} indexed chunks`),
        listItem("RAG pipeline", rag.algorithm || "advanced local RAG", `${rag.query_runs || 0} measured queries`),
      ].join("");
    }
    return { inference, rag };
  } catch(e) {
    const feed = qs("#inferenceEngineFeed");
    if (feed) feed.innerHTML = emptyState(`Inference metrics unavailable: ${e.message}`);
    return null;
  }
}

async function runInferenceAction(path, body, successMessage) {
  try {
    const payload = await api(path, { method: "POST", body: JSON.stringify({ project: getProject(), ...body }) });
    setOutput("#developerOutput", payload);
    toast(successMessage, payload.latency_ms != null ? `${payload.latency_ms} ms` : payload.status || "done");
    await loadInferenceEngine();
  } catch(e) { toast("Inference action error", e.message); }
}

function renderAdaptiveEngine() {
  const snapshot = state.adaptiveEngine || {};
  const status = snapshot.status || {};
  const profile = status.hardware_profile || {};
  const devices = profile.devices || [];
  const badge = qs("#adaptiveEngineBadge");
  const usable = devices.filter(device => (device.backends || []).length > 0);
  if (badge) {
    badge.textContent = usable.length ? `${usable.length} runtime${usable.length === 1 ? "" : "s"} exposed` : "Runtime needed";
    badge.className = `badge ${usable.length ? "badge-success" : "badge-warning"}`;
  }
  const grid = qs("#adaptiveHardwareGrid");
  if (grid) {
    grid.innerHTML = devices.map(device => {
      const backend = (device.backends || []).join(" + ") || "No execution backend";
      const memory = Number(device.memory_gb || 0).toFixed(1);
      return `<div class="adaptive-device-tile" data-device-kind="${escapeHtml(device.kind || "device")}">
        <div class="adaptive-device-heading"><span>${escapeHtml((device.kind || "device").toUpperCase())}</span><i class="status-dot ${(device.backends || []).length ? "online" : ""}"></i></div>
        <strong>${escapeHtml(device.name || device.id)}</strong>
        <p>${memory} GB visible · ${escapeHtml(backend)} · verify model before assignment</p>
        <small>${escapeHtml((device.precisions || []).join(" · ") || "Capabilities unavailable")}</small>
      </div>`;
    }).join("") || emptyState("No compute devices discovered");
  }
  const select = qs("#adaptiveDeviceSelect");
  if (select) {
    const selected = select.value;
    select.innerHTML = usable.map(device => `<option value="${escapeHtml(device.id)}">${escapeHtml(device.kind.toUpperCase())} · ${escapeHtml(device.name)}</option>`).join("");
    if (selected && usable.some(device => device.id === selected)) select.value = selected;
  }
  const plan = (snapshot.plans || [])[0] || {};
  const prefill = plan.prefill || {};
  const decode = plan.decode || {};
  const precision = plan.constraints?.precision_summary || {};
  const benchmark = (snapshot.benchmarks || [])[0] || {};
  const evidence = benchmark.evidence || {};
  const direct = snapshot.direct || status.direct_runtime || {};
  const ownership = status.execution_ownership || {};
  const residentModels = direct.models || [];
  const calibration = (snapshot.calibrations || [])[0] || {};
  const feed = qs("#adaptiveEngineFeed");
  if (!feed) return;
  const rows = [
    listItem("Hardware profile", `${profile.cpu_threads || 0} CPU threads · ${Number(profile.available_ram_gb || 0).toFixed(1)} GB RAM free`, `${devices.length} local compute device(s)`),
    listItem(
      "Unified model execution",
      ownership.gateway_bound ? `${(ownership.shared_consumers || []).length} app modules share one resident gateway` : "Model gateway is not bound",
      `<span class="badge ${ownership.gateway_bound ? "badge-success" : "badge-danger"}">${ownership.gateway_bound ? "Engine routed" : "Disconnected"}</span>`
    ),
    listItem("Compiler boundary", "Plans are estimates until a vendor backend accepts the model", "No model weights or telemetry leave this machine"),
    listItem(
      "Direct hardware runtime",
      direct.status === "ready" || direct.running ? `${residentModels.length} resident model(s) · inherited local pipes` : "Available on demand · no native worker kept open",
      `<span class="badge ${direct.status === "ready" || direct.running ? "badge-success" : "badge-soft"}">${escapeHtml(direct.status || "available")}</span>`
    ),
  ];
  if (calibration.id) {
    rows.push(listItem(
      "Exact-model calibration",
      `${calibration.winner?.device || "Unknown"} won after ${calibration.devices?.length || 0} device validation(s)`,
      `<span class="badge badge-success">${escapeHtml(calibration.status || "verified")}</span>`
    ));
  }
  residentModels.slice(0, 4).forEach(model => rows.push(listItem(
    `Resident · ${model.model_id || "model"}`,
    `${String(model.kind || "model").toUpperCase()} · ${model.device || "device"} · ${model.calls || 0} call(s)`,
    `${Number(model.average_infer_ms || model.average_generate_ms || 0).toFixed(3)} ms average`
  )));
  if (plan.id) {
    rows.push(listItem(
      `Plan ${String(plan.id).slice(0, 8)} · ${plan.status || "planned"}`,
      `${plan.model_name || "Local model"} · ${plan.source_format || "unknown"} · ${Number(precision.planned_weight_gb || 0).toFixed(2)} GB planned weights`,
      `<span class="badge ${plan.status === "compiled" ? "badge-success" : "badge-warning"}">${escapeHtml(plan.status || "planned")}</span>`
    ));
    rows.push(listItem(
      "Prefill execution",
      `${Number(prefill.estimated_tokens_per_second || 0).toFixed(2)} token/s estimated · ${prefill.transfer_count || 0} transfers`,
      `${(prefill.device_regions || []).map(region => region.device_id).join(" → ") || "No regions"}`
    ));
    rows.push(listItem(
      "Decode execution",
      `${Number(decode.estimated_tokens_per_second || 0).toFixed(2)} token/s estimated · ${decode.transfer_count || 0} transfers`,
      `${(decode.device_regions || []).map(region => region.device_id).join(" → ") || "No regions"}`
    ));
    (plan.warnings || []).slice(0, 2).forEach(warning => rows.push(listItem("Planner constraint", warning)));
  }
  if (evidence.status) {
    rows.push(listItem(
      `${benchmark.device_name || benchmark.device_id} validation`,
      `${Number(evidence.latency_ms?.mean || 0).toFixed(4)} ms mean · max error ${evidence.max_absolute_error ?? "n/a"}`,
      `<span class="badge ${evidence.status === "verified" ? "badge-success" : "badge-danger"}">${escapeHtml(evidence.status)}</span>`
    ));
  }
  feed.innerHTML = rows.join("");
}

async function loadAdaptiveEngine(refresh = false) {
  try {
    const project = encodeURIComponent(getProject());
    const refreshFlag = refresh ? "&refresh=1" : "";
    const [status, plans, benchmarks, direct, calibrations] = await Promise.all([
      api(`/api/adaptive-engine/status?project=${project}${refreshFlag}`),
      api(`/api/adaptive-engine/plans?project=${project}&limit=8`),
      api(`/api/adaptive-engine/benchmarks?project=${project}&limit=8`),
      api(`/api/adaptive-engine/direct/status?project=${project}`),
      api(`/api/adaptive-engine/direct/calibrations?project=${project}&limit=8`),
    ]);
    state.adaptiveEngine = {
      ...state.adaptiveEngine,
      status,
      plans: plans.plans || [],
      benchmarks: benchmarks.benchmarks || [],
      direct,
      calibrations: calibrations.calibrations || [],
    };
    renderAdaptiveEngine();
    return state.adaptiveEngine;
  } catch(e) {
    const feed = qs("#adaptiveEngineFeed");
    if (feed) feed.innerHTML = emptyState(`Adaptive engine unavailable: ${e.message}`);
    return null;
  }
}

function adaptiveModelPayload() {
  const budget = qs("#adaptiveMemoryBudget")?.value.trim();
  return {
    project: getProject(),
    path: qs("#adaptiveModelPath")?.value.trim() || "",
    prompt_tokens: Number(qs("#adaptivePromptTokens")?.value || 512),
    context_tokens: Number(qs("#adaptiveContextTokens")?.value || 4096),
    memory_budget_gb: budget ? Number(budget) : null,
    quality_loss_limit: Number(qs("#adaptiveQualityLimit")?.value || 0.08),
    power_mode: qs("#adaptivePowerMode")?.value || "balanced",
    max_device_transitions: 4,
    backend_compile: true,
  };
}

async function inspectAdaptiveModel() {
  const body = adaptiveModelPayload();
  if (!body.path) { toast("Model path required", "Choose a local GGUF, SafeTensors, ONNX, or OpenVINO model."); return; }
  try {
    const payload = await api("/api/adaptive-engine/inspect", { method: "POST", body: JSON.stringify(body) });
    setOutput("#developerOutput", payload);
    toast("Model graph inspected", `${payload.graph_node_count || 0} graph nodes · ${payload.tensor_count || 0} tensors`);
    await loadAdaptiveEngine();
  } catch(e) { toast("Inspection failed", e.message); }
}

async function compileAdaptiveModel() {
  const body = adaptiveModelPayload();
  if (!body.path) { toast("Model path required", "Choose a local model before compiling a plan."); return; }
  try {
    const payload = await api("/api/adaptive-engine/compile", { method: "POST", body: JSON.stringify(body) });
    setOutput("#developerOutput", payload);
    toast("Adaptive plan complete", `${payload.status || "planned"} · ${Number(payload.decode?.estimated_tokens_per_second || 0).toFixed(2)} token/s estimated`);
    await loadAdaptiveEngine();
  } catch(e) { toast("Compilation failed", e.message); }
}

async function benchmarkAdaptiveDevice() {
  const deviceId = qs("#adaptiveDeviceSelect")?.value || "";
  if (!deviceId) { toast("No executable device detected"); return; }
  try {
    const payload = await api("/api/adaptive-engine/benchmark", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), device_id: deviceId, iterations: 20, dimension: 256 }),
    });
    setOutput("#developerOutput", payload);
    toast("Device verified", `${payload.evidence?.latency_ms?.mean || 0} ms · ${payload.evidence?.status || "complete"}`);
    await loadAdaptiveEngine();
  } catch(e) { toast("Device validation failed", e.message); }
}

async function autotuneAdaptiveDevice() {
  const deviceId = qs("#adaptiveDeviceSelect")?.value || "";
  if (!deviceId) { toast("No executable device detected"); return; }
  try {
    const payload = await api("/api/adaptive-engine/autotune", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), device_id: deviceId, phase: "decode", iterations: 12 }),
    });
    setOutput("#developerOutput", payload);
    const winner = payload.winner?.configuration || {};
    toast("Autotuning complete", `${winner.performance_hint || "Backend"} · ${winner.num_streams || "default streams"}`);
    await loadAdaptiveEngine();
  } catch(e) { toast("Autotuning failed", e.message); }
}

async function calibrateAdaptiveModel() {
  const body = adaptiveModelPayload();
  if (!body.path) { toast("Model path required", "Choose a local ONNX or OpenVINO XML model."); return; }
  try {
    const payload = await api("/api/adaptive-engine/direct/calibrate", {
      method: "POST",
      body: JSON.stringify({
        project: getProject(), path: body.path, iterations: 10,
        keep_resident: true,
        model_id: qs("#adaptiveResidentModelId")?.value.trim() || "",
      }),
    });
    setOutput("#developerOutput", payload);
    toast("Exact model verified", `${payload.winner?.device || "Device"} · ${Number(payload.winner?.latency_ms?.mean || 0).toFixed(3)} ms mean`);
    await loadAdaptiveEngine();
  } catch(e) { toast("Calibration failed", e.message); }
}

async function loadAdaptiveResident() {
  const path = qs("#adaptiveModelPath")?.value.trim() || "";
  const deviceId = qs("#adaptiveDeviceSelect")?.value || "";
  const modelId = qs("#adaptiveResidentModelId")?.value.trim() || "openagent-direct-chat";
  const kind = qs("#adaptiveModelKind")?.value || "generic";
  if (!path || !deviceId) { toast("Model and device required", "Choose a local model path and an executable device."); return; }
  try {
    const endpoint = kind === "llm" ? "/api/adaptive-engine/direct/load-llm" : "/api/adaptive-engine/direct/load";
    const payload = await api(endpoint, {
      method: "POST",
      body: JSON.stringify({ project: getProject(), path, device_id: deviceId, model_id: modelId }),
    });
    setOutput("#developerOutput", payload);
    toast("Model resident", `${payload.device || deviceId} · ${Number(payload.load_ms || payload.compile_ms || 0).toFixed(1)} ms load`);
    await loadAdaptiveEngine();
  } catch(e) { toast("Direct load failed", e.message); }
}

async function unloadAdaptiveResident() {
  const modelId = qs("#adaptiveResidentModelId")?.value.trim() || "openagent-direct-chat";
  try {
    const payload = await api("/api/adaptive-engine/direct/unload", {
      method: "POST", body: JSON.stringify({ project: getProject(), model_id: modelId }),
    });
    toast("Resident model", payload.status || "unloaded");
    await loadAdaptiveEngine();
  } catch(e) { toast("Unload failed", e.message); }
}

async function stopAdaptiveRuntime() {
  try {
    await api("/api/adaptive-engine/direct/stop", {
      method: "POST", body: JSON.stringify({ project: getProject() }),
    });
    toast("Direct runtime stopped", "Native model memory released");
    await loadAdaptiveEngine();
  } catch(e) { toast("Runtime stop failed", e.message); }
}

async function loadEngineWorkspace(refresh = false) {
  const project = encodeURIComponent(getProject());
  try {
    const [adaptive, rolesPayload, options, runsPayload] = await Promise.all([
      loadAdaptiveEngine(refresh),
      api(`/api/adaptive-engine/roles?project=${project}`),
      api("/api/adaptive-engine/role-options"),
      api(`/api/adaptive-engine/orchestrations?project=${project}&limit=12`),
    ]);
    state.engineRoles = rolesPayload.roles || [];
    state.engineRoleOptions = options || { roles: [], providers: [], models: [] };
    state.engineRuns = runsPayload.runs || [];
    if (!state.engineRoles.some(role => role.role === state.selectedEngineRole)) {
      state.selectedEngineRole = state.engineRoles[0]?.role || "chat";
    }
    renderEngineWorkspace();
    return { adaptive, roles: state.engineRoles, runs: state.engineRuns };
  } catch(e) {
    const feed = qs("#engineRunFeed");
    if (feed) feed.innerHTML = emptyState(`Engine workspace unavailable: ${e.message}`);
    throw e;
  }
}

function renderEngineWorkspace() {
  const roles = state.engineRoles || [];
  const options = state.engineRoleOptions || {};
  const selected = roles.find(role => role.role === state.selectedEngineRole) || roles[0];
  const summary = qs("#engineRoleSummary");
  if (summary) {
    summary.innerHTML = roles.map(role => `
      <button class="engine-role-pill ${role.role === selected?.role ? "active" : ""}" data-engine-role="${escapeHtml(role.role)}" type="button">
        <span>${escapeHtml(role.label || role.role)}</span>
        <strong>${escapeHtml(role.model_id || "Not assigned")}</strong>
        <small>${escapeHtml(role.provider_id)} · ${escapeHtml(role.device_id || "auto")}</small>
        <i class="status-dot ${role.enabled ? "online" : ""}"></i>
      </button>
    `).join("");
  }
  const roleSelect = qs("#engineRoleSelect");
  if (roleSelect) {
    roleSelect.innerHTML = roles.map(role => `<option value="${escapeHtml(role.role)}">${escapeHtml(role.label || role.role)}</option>`).join("");
    if (selected) roleSelect.value = selected.role;
  }
  const providers = options.providers || [];
  const providerSelect = qs("#engineRoleProvider");
  if (providerSelect) {
    providerSelect.innerHTML = providers.map(provider => `<option value="${escapeHtml(provider.id)}">${escapeHtml(provider.label || provider.id)}${provider.available ? " · ready" : ""}</option>`).join("");
    if (selected && providers.some(provider => provider.id === selected.provider_id)) providerSelect.value = selected.provider_id;
  }
  const discovered = options.models || [];
  const modelValues = [];
  discovered.forEach(item => {
    const value = String(item.model || "").trim();
    if (value && !modelValues.some(existing => existing.value === value)) {
      modelValues.push({ value, label: `${value} · ${item.provider || "local"}` });
    }
  });
  if (selected?.model_id && !modelValues.some(item => item.value === selected.model_id)) {
    modelValues.unshift({ value: selected.model_id, label: `${selected.model_id} · configured` });
  }
  const modelSelect = qs("#engineRoleModelSelect");
  if (modelSelect) {
    modelSelect.innerHTML = modelValues.map(item => `<option value="${escapeHtml(item.value)}">${escapeHtml(item.label)}</option>`).join("") || `<option value="">No local models discovered</option>`;
    if (selected && modelValues.some(item => item.value === selected.model_id)) modelSelect.value = selected.model_id;
  }
  const profileDevices = state.adaptiveEngine?.status?.hardware_profile?.devices || [];
  const deviceSelect = qs("#engineRoleDevice");
  if (deviceSelect) {
    deviceSelect.innerHTML = `<option value="auto">Auto measured target</option>` + profileDevices.map(device => `<option value="${escapeHtml(device.id)}">${escapeHtml((device.kind || "device").toUpperCase())} · ${escapeHtml(device.name || device.id)}</option>`).join("");
    if (selected && (selected.device_id === "auto" || profileDevices.some(device => device.id === selected.device_id))) deviceSelect.value = selected.device_id || "auto";
  }
  if (qs("#engineRoleModelCustom")) qs("#engineRoleModelCustom").value = selected?.model_id || "";
  if (qs("#engineRoleModelPath")) qs("#engineRoleModelPath").value = selected?.model_path || "";
  if (qs("#engineRolePrompt")) qs("#engineRolePrompt").value = selected?.system_prompt || "";
  if (qs("#engineRoleEnabled")) qs("#engineRoleEnabled").checked = selected?.enabled !== false;
  const sequence = qs("#engineSequenceRoles");
  if (sequence) {
    sequence.innerHTML = roles.filter(role => role.enabled).map(role => `<option value="${escapeHtml(role.role)}" ${["reasoning", "research", "supervisor"].includes(role.role) ? "selected" : ""}>${escapeHtml(role.label || role.role)}</option>`).join("");
  }
  renderEngineRuns();
  updateEngineModeVisibility();
}

function selectEngineRole(roleId) {
  if (!(state.engineRoles || []).some(role => role.role === roleId)) return;
  state.selectedEngineRole = roleId;
  renderEngineWorkspace();
}

function updateEngineModeVisibility() {
  const sequenceField = qs(".engine-sequence-field");
  if (sequenceField) sequenceField.hidden = (qs("#engineOrchestrationMode")?.value || "supervised-code") !== "sequence";
}

async function saveEngineRole() {
  const role = qs("#engineRoleSelect")?.value || state.selectedEngineRole;
  const customModel = qs("#engineRoleModelCustom")?.value.trim();
  const selectedModel = qs("#engineRoleModelSelect")?.value || "";
  try {
    const payload = await api("/api/adaptive-engine/roles", {
      method: "POST",
      body: JSON.stringify({
        project: getProject(), role,
        provider_id: qs("#engineRoleProvider")?.value || "auto",
        model_id: customModel || selectedModel,
        model_path: qs("#engineRoleModelPath")?.value.trim() || "",
        device_id: qs("#engineRoleDevice")?.value || "auto",
        system_prompt: qs("#engineRolePrompt")?.value.trim() || "",
        enabled: Boolean(qs("#engineRoleEnabled")?.checked),
        config: {
          scheduler: { enabled: true, enable_prefix_caching: true, dynamic_split_fuse: true, max_num_seqs: 4 },
        },
      }),
    });
    state.selectedEngineRole = payload.role;
    toast("Role assignment saved", `${payload.label} · ${payload.model_id || "not assigned"}`);
    await loadEngineWorkspace();
  } catch(e) { toast("Role save failed", e.message); }
}

function renderEngineRuns() {
  const feed = qs("#engineRunFeed");
  const output = qs("#engineRunOutput");
  const runs = state.engineRuns || [];
  if (feed) {
    feed.innerHTML = runs.slice(0, 8).map(run => listItem(
      `${run.mode || "model team"} · ${run.status || "unknown"}`,
      `${run.turns?.length || 0} model turn(s) · ${run.rounds || 0} review round(s)`,
      `<span class="badge ${run.approved ? "badge-success" : run.status === "failed" ? "badge-danger" : "badge-warning"}">${run.approved ? "approved" : escapeHtml(run.status || "review")}</span>`
    )).join("") || emptyState("No model collaboration runs yet.");
  }
  if (!output) return;
  const run = runs[0];
  if (!run) {
    output.innerHTML = "";
    return;
  }
  output.innerHTML = `
    <div class="engine-run-heading"><div><span>${escapeHtml(run.mode)}</span><strong>${escapeHtml(run.task)}</strong></div><span class="badge ${run.approved ? "badge-success" : "badge-warning"}">${run.approved ? "Approved" : escapeHtml(run.status)}</span></div>
    ${(run.turns || []).map(turn => `<section class="engine-turn"><header><strong>${escapeHtml(turn.label || turn.role)}</strong><span>${escapeHtml(turn.provider_id)} · ${escapeHtml(turn.model_id)}</span></header><pre>${escapeHtml(turn.content)}</pre></section>`).join("")}
    <section class="engine-final-output"><header><strong>Final output</strong></header><pre>${escapeHtml(run.final_output || run.error || "No output")}</pre></section>
  `;
}

async function runEngineCollaboration() {
  const task = qs("#engineCollaborationTask")?.value.trim() || "";
  if (!task) { toast("Task required", "Describe what the model team should build or analyze."); return; }
  const button = qs("#runEngineCollaboration");
  const badge = qs("#engineCollaborationBadge");
  const sequence = qs("#engineSequenceRoles");
  const roles = sequence ? Array.from(sequence.selectedOptions).map(option => option.value) : [];
  if (button) { button.disabled = true; button.textContent = "Models working…"; }
  if (badge) { badge.textContent = "Running"; badge.className = "badge badge-warning"; }
  try {
    const payload = await api("/api/adaptive-engine/orchestrate", {
      method: "POST",
      body: JSON.stringify({
        project: getProject(), task,
        mode: qs("#engineOrchestrationMode")?.value || "supervised-code",
        max_rounds: Number(qs("#engineMaxRounds")?.value || 2),
        roles,
        allow_external: Boolean(qs("#engineAllowExternal")?.checked),
      }),
    });
    state.engineRuns = [payload, ...(state.engineRuns || []).filter(item => item.id !== payload.id)];
    renderEngineRuns();
    toast("Model team completed", payload.approved ? "Supervisor approved the result" : "Review limit reached; inspect the final feedback");
  } catch(e) { toast("Model collaboration failed", e.message); }
  finally {
    if (button) { button.disabled = false; button.textContent = "Run Model Team"; }
    if (badge) { badge.textContent = "Ready"; badge.className = "badge badge-soft"; }
  }
}

function ultraModelSpecification() {
  const architecture = qs("#ultraModelArchitecture")?.value || "dense";
  const total = Number(qs("#ultraModelParams")?.value || 0);
  const active = architecture === "dense" ? total : Number(qs("#ultraModelActiveParams")?.value || 0);
  return {
    name: qs("#ultraModelName")?.value.trim() || "Large local model",
    architecture,
    parameter_count_b: total,
    active_parameter_count_b: active,
    quantization_bits: Number(qs("#ultraModelBits")?.value || 4),
    metadata: { kv_type: "q8_0" },
  };
}

function renderUltraModelEngine() {
  const data = state.ultraModel || {};
  const status = data.status || {};
  const manifests = data.manifests || [];
  const sessions = data.sessions || [];
  const active = sessions.filter(session => ["starting", "warming", "ready"].includes(session.status));
  const select = qs("#ultraManifestSelect");
  if (select) {
    const selected = data.manifestId || select.value;
    select.innerHTML = '<option value="">Planning profile (not registered)</option>' + manifests.map(manifest =>
      `<option value="${escapeHtml(manifest.id)}">${escapeHtml(manifest.name)} · ${manifest.parameter_count_b}B ${escapeHtml(manifest.architecture)}</option>`
    ).join("");
    if (selected && manifests.some(manifest => manifest.id === selected)) select.value = selected;
  }
  const badge = qs("#ultraEngineBadge");
  if (badge) {
    const available = Boolean(status.llama_server?.available);
    badge.textContent = available ? "Runner ready" : "Runner needed";
    badge.className = `badge ${available ? "badge-success" : "badge-warning"}`;
  }
  const plan = data.plan || {};
  const performance = plan.performance || {};
  const memory = plan.memory || {};
  const feed = qs("#ultraModelEngineFeed");
  if (!feed) return;
  const rows = [
    listItem(
      "llama.cpp runtime",
      status.llama_server?.available ? status.llama_server.path : "Install llama-server or set OPENAGENT_LLAMA_SERVER",
      `<span class="badge ${status.llama_server?.available ? "badge-success" : "badge-warning"}">${status.llama_server?.available ? "Available" : "Offline"}</span>`
    ),
    listItem("Registered shard sets", `${manifests.length} manifest(s) · ${active.length} active server(s)`, `${status.hardware?.available_ram_gb || 0} GB RAM free`),
  ];
  if (plan.mode) {
    rows.push(listItem(
      `Plan: ${plan.mode}`,
      `${performance.predicted_tokens_per_second ?? 0} token/s predicted · ${performance.bottleneck || "unknown"}`,
      `<span class="badge ${plan.interactive_predicted ? "badge-success" : "badge-warning"}">${plan.interactive_predicted ? "Interactive" : "Slow path"}</span>`
    ));
    rows.push(listItem(
      "Memory working set",
      `${memory.resident_requirement_gb || 0} GB required · ${memory.usable_local_ram_gb || 0} GB usable · ${memory.disk_streamed_active_gb || 0} GB streamed/token`,
      `<span class="badge ${plan.launchable ? "badge-success" : "badge-soft"}">${plan.launchable ? "Launchable" : "Plan only"}</span>`
    ));
    (plan.warnings || []).slice(0, 3).forEach(warning => rows.push(listItem("Constraint", warning)));
  }
  active.slice(0, 2).forEach(session => rows.push(listItem(
    `Server ${session.status}`,
    `PID ${session.pid} · http://${session.host}:${session.port} · ${session.manifest_id.slice(0, 8)}`,
    `<span class="badge badge-success">Warm</span>`
  )));
  feed.innerHTML = rows.join("");
}

async function loadUltraModelEngine() {
  try {
    const project = encodeURIComponent(getProject());
    const [status, manifests, sessions] = await Promise.all([
      api(`/api/ultra-model/status?project=${project}`),
      api(`/api/ultra-model/manifests?project=${project}`),
      api(`/api/ultra-model/sessions?project=${project}`),
    ]);
    state.ultraModel = {
      ...state.ultraModel,
      status,
      manifests: manifests.manifests || [],
      sessions: sessions.sessions || [],
    };
    const selected = qs("#ultraManifestSelect")?.value;
    if (selected) state.ultraModel.manifestId = selected;
    const running = state.ultraModel.sessions.find(session => ["starting", "warming", "ready"].includes(session.status));
    if (running) state.ultraModel.sessionId = running.id;
    renderUltraModelEngine();
    return state.ultraModel;
  } catch(e) {
    const feed = qs("#ultraModelEngineFeed");
    if (feed) feed.innerHTML = emptyState(`Ultra-scale engine unavailable: ${e.message}`);
    return null;
  }
}

async function registerUltraModel() {
  const path = qs("#ultraModelPath")?.value.trim();
  if (!path) { toast("GGUF path required", "Choose the first shard of the local model."); return; }
  const spec = ultraModelSpecification();
  try {
    const payload = await api("/api/ultra-model/manifests", {
      method: "POST",
      body: JSON.stringify({
        project: getProject(),
        ...spec,
        shard_paths: [path],
      }),
    });
    state.ultraModel.manifestId = payload.manifest?.id || "";
    toast("GGUF shards registered", `${payload.manifest?.shard_paths?.length || 0} local shard(s)`);
    await loadUltraModelEngine();
    await planUltraModel();
  } catch(e) { toast("Registration failed", e.message); }
}

async function planUltraModel() {
  const selected = qs("#ultraManifestSelect")?.value || state.ultraModel.manifestId || "";
  try {
    const payload = await api("/api/ultra-model/plan", {
      method: "POST",
      body: JSON.stringify({
        project: getProject(),
        manifest_id: selected,
        specification: selected ? {} : ultraModelSpecification(),
        context_tokens: Number(qs("#ultraModelContext")?.value || 2048),
        expected_output_tokens: 256,
      }),
    });
    state.ultraModel.plan = payload.plan || {};
    if (selected) state.ultraModel.manifestId = selected;
    renderUltraModelEngine();
    const speed = payload.plan?.performance?.predicted_tokens_per_second ?? 0;
    toast("Feasibility plan ready", `${payload.plan?.mode || "planned"} · ${speed} token/s predicted`);
  } catch(e) { toast("Planning failed", e.message); }
}

async function calibrateUltraModel() {
  try {
    const payload = await api("/api/ultra-model/calibrate", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), size_mb: 32 }),
    });
    const result = payload.calibration || {};
    toast("Local I/O calibrated", `${result.memory_copy_gbps || 0} GB/s memory · ${result.planning_disk_gbps || 0} GB/s disk`);
    setOutput("#developerOutput", result);
  } catch(e) { toast("Calibration failed", e.message); }
}

async function launchUltraModel() {
  const manifestId = qs("#ultraManifestSelect")?.value || state.ultraModel.manifestId || "";
  if (!manifestId) { toast("Register a GGUF model first"); return; }
  const approved = Boolean(qs("#ultraLaunchApproval")?.checked);
  if (!approved) { toast("Approval required", "Enable the local process approval control first."); return; }
  try {
    const payload = await api("/api/ultra-model/launch", {
      method: "POST",
      body: JSON.stringify({
        project: getProject(),
        manifest_id: manifestId,
        approved,
        port: Number(qs("#ultraModelPort")?.value || 8081),
        keep_warm: true,
      }),
    });
    state.ultraModel.sessionId = payload.session?.id || "";
    state.ultraModel.plan = payload.plan || state.ultraModel.plan;
    toast("Local model server starting", `PID ${payload.session?.pid || "pending"}`);
    await loadUltraModelEngine();
  } catch(e) { toast("Launch failed", e.message); }
}

async function stopUltraModel() {
  const running = (state.ultraModel.sessions || []).find(session => ["starting", "warming", "ready"].includes(session.status));
  const sessionId = state.ultraModel.sessionId || running?.id || "";
  if (!sessionId) { toast("No active ultra-model server"); return; }
  try {
    await api(`/api/ultra-model/sessions/${encodeURIComponent(sessionId)}/stop`, {
      method: "POST",
      body: JSON.stringify({ project: getProject(), reason: "Stopped from Developer UI" }),
    });
    state.ultraModel.sessionId = "";
    toast("Local model server stopped");
    await loadUltraModelEngine();
  } catch(e) { toast("Stop failed", e.message); }
}

async function createModelHost() {
  try {
    const payload = await api("/api/model-cookbook/hosts", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), label: "CPU host", host_type: "local", address: "127.0.0.1" }),
    });
    setOutput("#developerOutput", payload);
    toast("Model host added", payload.host?.label || "CPU host");
    await loadModelCookbook();
  } catch(e) { toast("Model host error", e.message); }
}

async function planModelServe() {
  try {
    const payload = await api("/api/model-cookbook/serve", {
      method: "POST",
      body: JSON.stringify({
        project: getProject(),
        model_ref: state.modelSettings.chatModel || "local-model",
        runner: "python",
        args: ["--version"],
        allow_process: false,
      }),
    });
    setOutput("#developerOutput", payload);
    toast("Serve plan recorded", payload.session?.status || "planned");
    await loadModelCookbook();
  } catch(e) { toast("Serve plan error", e.message); }
}

async function checkHardwareFit() {
  try {
    const payload = await api(`/api/model-cookbook/hardware-fit?project=${encodeURIComponent(getProject())}&model_ref=${encodeURIComponent(state.modelSettings.chatModel || "local-model")}`);
    setOutput("#developerOutput", payload);
    toast("Hardware fit ready", `${payload.recommended_bits || "auto"} bit profile`);
    await loadModelCookbook();
  } catch(e) { toast("Hardware fit error", e.message); }
}

async function loadMcpRuntime() {
  try {
    const project = encodeURIComponent(getProject());
    const [servers, tools, permissions] = await Promise.all([
      api(`/api/mcp/servers?project=${project}`),
      api(`/api/mcp/tools?project=${project}`),
      api(`/api/mcp/permissions?project=${project}`),
    ]);
    const feed = qs("#mcpRuntimeFeed");
    if (feed) {
      const serverItems = (servers.servers || []).slice(0, 3).map(server =>
        listItem(server.name, `${server.transport} · ${server.status}`, `${(server.command || []).join(" ") || server.endpoint || "configured locally"}`)
      );
      const toolLine = listItem("Tools", `${(tools.tools || []).length} registered`, `${(permissions.permissions || []).length} permission changes`);
      const toolItems = (tools.tools || []).slice(0, 8).map(tool => {
        const action = `<button class="btn btn-soft btn-small" data-mcp-invoke="${escapeHtml(tool.id)}">Test</button>`;
        return listItem(tool.name, `${tool.enabled ? "enabled" : "disabled"} · ${tool.approval_required ? "approval required" : "local call"}`, action);
      });
      feed.innerHTML = [toolLine, ...toolItems, ...serverItems].join("") || emptyState("No MCP servers registered yet.");
    }
    return { servers, tools, permissions };
  } catch(e) {
    const feed = qs("#mcpRuntimeFeed");
    if (feed) feed.innerHTML = emptyState(`MCP runtime unavailable: ${e.message}`);
    return null;
  }
}

async function invokeMcpTool(toolId) {
  const raw = window.prompt("Arguments as JSON", "{}");
  if (raw === null) return;
  let argumentsPayload;
  try { argumentsPayload = raw.trim() ? JSON.parse(raw) : {}; } catch (_) { toast("MCP call stopped", "Arguments must be valid JSON."); return; }
  if (!window.confirm("Allow this registered MCP tool to run locally now?")) return;
  try {
    const payload = await api(`/api/mcp/tools/${encodeURIComponent(toolId)}/invoke`, {
      method: "POST",
      body: JSON.stringify({ project: getProject(), arguments: argumentsPayload, approval_id: "ui-confirmed" }),
    });
    setOutput("#developerOutput", payload);
    toast("MCP tool completed", toolId.slice(0, 8));
    await loadMcpRuntime();
  } catch (e) { toast("MCP tool failed", e.message); }
}

async function loadRuntimeFabric() {
  try {
    const [runtime, compute] = await Promise.all([api("/api/tools/runtime"), api("/api/compute/status")]);
    const providers = compute.providers || [];
    const ready = providers.filter(item => item.available && (item.local || item.id === "auto")).length;
    const local = providers.filter(item => item.available && item.local && item.id !== "auto").length;
    const hardware = compute.hardware || {};
    const devices = hardware.backends || [];
    const badge = qs("#runtimeFabricBadge");
    if (badge) { badge.textContent = `${local} local ready`; badge.className = local ? "badge badge-success" : "badge badge-warning"; }
    const feed = qs("#runtimeFabricFeed");
    if (feed) {
      feed.innerHTML = [
        `<div class="runtime-fabric-metrics"><div><strong>${ready}</strong><span>routable providers</span></div><div><strong>${runtime.tools?.length || 0}</strong><span>shared tools</span></div><div><strong>${devices.length || 0}</strong><span>hardware profiles</span></div></div>`,
        listItem("Policy", compute.policy?.routing_policy || "local-first", "The plan is advisory; ModelGateway owns the actual call."),
        listItem("Hardware evidence", devices.length ? `${devices.length} backend profile(s) reported` : "No refreshed device profile", hardware.status || "capability probe available"),
      ].join("");
    }
    return { runtime, compute };
  } catch (e) {
    const feed = qs("#runtimeFabricFeed");
    if (feed) feed.innerHTML = emptyState(`Runtime capability check unavailable: ${e.message}`);
    const badge = qs("#runtimeFabricBadge");
    if (badge) { badge.textContent = "Unavailable"; badge.className = "badge badge-warning"; }
    return null;
  }
}

async function planRuntimeFabric() {
  try {
    const payload = await api("/api/compute/plan", {
      method: "POST",
      body: JSON.stringify({
        task: qs("#runtimePlanTask")?.value.trim() || "chat",
        model: qs("#runtimePlanModel")?.value.trim() || "",
        allow_external: Boolean(qs("#runtimePlanExternal")?.checked),
      }),
    });
    setOutput("#developerOutput", payload);
    const selected = payload.selected;
    toast("Execution plan ready", selected ? `${selected.provider} · ${selected.model || "automatic model"}` : "Needs local setup");
  } catch (e) { toast("Execution planning failed", e.message); }
}

async function registerMcpServer() {
  try {
    const payload = await api("/api/mcp/servers", {
      method: "POST",
      body: JSON.stringify({
        project: getProject(),
        name: "local-tools",
        transport: "stdio",
        command: ["python", "-m", "openagent"],
        tools: [{ name: "local_search", description: "Local OpenAgent search", approval_required: true }],
      }),
    });
    setOutput("#developerOutput", payload);
    toast("MCP server registered", payload.server?.name || "local-tools");
    await loadMcpRuntime();
  } catch(e) { toast("MCP error", e.message); }
}

async function loadTerminalRuns() {
  try {
    const payload = await api(`/api/terminal/runs?project=${encodeURIComponent(getProject())}&limit=5`);
    const feed = qs("#terminalFeed");
    if (feed) {
      feed.innerHTML = (payload.runs || []).map(run =>
        listItem(`${run.command?.join(" ") || "command"}`, `${run.status} · exit ${run.exit_code}`, run.started_at)
      ).join("") || emptyState("No terminal runs yet.");
    }
    return payload;
  } catch(e) {
    const feed = qs("#terminalFeed");
    if (feed) feed.innerHTML = emptyState(`Terminal unavailable: ${e.message}`);
    return null;
  }
}

async function runTerminalVersion() {
  try {
    const payload = await api("/api/terminal/stream", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), command: ["python", "--version"] }),
    });
    setOutput("#developerOutput", payload);
    toast("Terminal run complete", payload.run?.status || "done");
    await loadTerminalRuns();
  } catch(e) { toast("Terminal error", e.message); }
}

async function loadSchedulerStatus() {
  try {
    const project = encodeURIComponent(getProject());
    const [tasks, notifications] = await Promise.all([
      api(`/api/tasks?project=${project}&limit=5`),
      api(`/api/tasks/notifications?project=${project}&limit=5`),
    ]);
    const feed = qs("#schedulerFeed");
    if (feed) {
      const taskItems = (tasks.tasks || []).slice(0, 4).map(task =>
        listItem(task.title, `${task.status} · ${task.next_run_at}`, task.task_type)
      );
      const noteLine = listItem("Notifications", `${(notifications.notifications || []).length} recent`, "Local notification records");
      feed.innerHTML = [noteLine, ...taskItems].join("") || emptyState("No scheduled tasks yet.");
    }
    return { tasks, notifications };
  } catch(e) {
    const feed = qs("#schedulerFeed");
    if (feed) feed.innerHTML = emptyState(`Scheduler unavailable: ${e.message}`);
    return null;
  }
}

async function createSchedulerTask() {
  try {
    const payload = await api("/api/tasks", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), title: "Developer UI due task", schedule: { delay_seconds: 0 } }),
    });
    setOutput("#developerOutput", payload);
    toast("Task created", payload.task?.title || "Scheduled task");
    await loadSchedulerStatus();
  } catch(e) { toast("Task error", e.message); }
}

async function addSchedulerOutput() {
  try {
    const tasks = await api(`/api/tasks?project=${encodeURIComponent(getProject())}&limit=1`);
    const task = (tasks.tasks || [])[0];
    if (!task) throw new Error("Create a task before adding an output destination.");
    const payload = await api(`/api/tasks/${encodeURIComponent(task.id)}/destinations`, {
      method: "POST",
      body: JSON.stringify({ destination_type: "local_file", config: { filename: `${task.id}.jsonl` } }),
    });
    setOutput("#developerOutput", payload);
    toast("Local output destination added", task.title);
  } catch(e) { toast("Output destination error", e.message); }
}

async function runDueSchedulerTasks() {
  try {
    const payload = await api("/api/tasks/run-due", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), limit: 10 }),
    });
    setOutput("#developerOutput", payload);
    toast("Due tasks processed", `${payload.count || 0} run`);
    await loadSchedulerStatus();
  } catch(e) { toast("Run due error", e.message); }
}

async function loadImageStudio() {
  try {
    const payload = await api(`/api/gallery/library?project=${encodeURIComponent(getProject())}&limit=5`);
    const feed = qs("#imageStudioFeed");
    if (feed) {
      feed.innerHTML = (payload.assets || []).map(asset =>
        listItem(
          asset.filename,
          `${asset.width || 0}x${asset.height || 0} · ${asset.size_bytes} bytes · ${(asset.tags || []).join(", ") || "untagged"}`,
          `<div class="toolbar slim">
            <button class="btn btn-mini btn-soft" data-image-action="favorite" data-image-id="${escapeHtml(asset.id)}">Favorite</button>
            <button class="btn btn-mini btn-outline" data-image-action="restore" data-image-id="${escapeHtml(asset.id)}">Restore</button>
            <button class="btn btn-mini btn-outline" data-image-action="remove-background" data-image-id="${escapeHtml(asset.id)}">Remove BG</button>
            <button class="btn btn-mini btn-soft" data-image-action="undo" data-image-id="${escapeHtml(asset.id)}">Undo</button>
          </div>`
        )
      ).join("") || emptyState("No image assets yet.");
      qsa("[data-image-action]", feed).forEach(button => button.addEventListener("click", () => runImageAction(button.dataset.imageId, button.dataset.imageAction)));
    }
    return payload;
  } catch(e) {
    const feed = qs("#imageStudioFeed");
    if (feed) feed.innerHTML = emptyState(`Image studio unavailable: ${e.message}`);
    return null;
  }
}

async function batchTagImages() {
  try {
    const library = await api(`/api/gallery/library?project=${encodeURIComponent(getProject())}&limit=20`);
    const assetIds = (library.assets || []).map(asset => asset.id);
    if (!assetIds.length) throw new Error("Add an image before using batch tags.");
    const payload = await api("/api/gallery/batch-tags", {
      method: "POST", body: JSON.stringify({ asset_ids: assetIds, tags: ["reviewed"] }),
    });
    setOutput("#developerOutput", payload);
    toast("Images tagged", `${payload.count || 0} updated`);
    await loadImageStudio();
  } catch(e) { toast("Batch tag error", e.message); }
}

async function runImageAction(assetId, action) {
  try {
    let path = `/api/gallery/${encodeURIComponent(assetId)}/transform`;
    let body = { operation: action, params: {} };
    if (action === "favorite") {
      path = `/api/gallery/${encodeURIComponent(assetId)}/metadata`;
      body = { favorite: true };
    } else if (action === "undo") {
      path = `/api/gallery/${encodeURIComponent(assetId)}/undo`;
      body = {};
    }
    const payload = await api(path, { method: "POST", body: JSON.stringify(body) });
    setOutput("#developerOutput", payload);
    toast("Image operation complete", action);
    await loadImageStudio();
  } catch(e) { toast("Image operation failed", e.message); }
}

async function uploadTinyImage() {
  try {
    const payload = await api("/api/gallery/upload", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), filename: "ui-test-pixel.png", content_type: "image/png", content_base64: TINY_PNG_BASE64, tags: ["ui", "test"] }),
    });
    setOutput("#developerOutput", payload);
    toast("Image stored", payload.asset?.filename || "image");
    await loadImageStudio();
  } catch(e) { toast("Image error", e.message); }
}

async function loadAgentBridgeStatus() {
  try {
    const project = encodeURIComponent(getProject());
    const [capabilities, tasks, memory, documents] = await Promise.all([
      api("/api/agent-bridge/capabilities"),
      api(`/api/agent-bridge/tasks?project=${project}&limit=5`),
      api(`/api/agent-bridge/memory?project=${project}&limit=5`),
      api(`/api/agent-bridge/documents?project=${project}&limit=5`),
    ]);
    const feed = qs("#agentBridgeFeed");
    if (feed) {
      feed.innerHTML = [
        listItem("Capabilities", `${(capabilities.capabilities || []).length} scopes`, capabilities.runtime || "local-agent-bridge"),
        listItem("Bridge tasks", `${(tasks.tasks || []).length} recent`, "External agents use scoped local APIs"),
        listItem("Memory events", `${(memory.events || []).length} recent`, "Local context records"),
        listItem("Documents", `${documents.count || 0} visible`, "Local editor documents"),
      ].join("");
    }
    return { capabilities, tasks, memory, documents };
  } catch(e) {
    const feed = qs("#agentBridgeFeed");
    if (feed) feed.innerHTML = emptyState(`Agent bridge unavailable: ${e.message}`);
    return null;
  }
}

async function writeBridgeMemory() {
  try {
    const payload = await api("/api/agent-bridge/memory", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), title: "Developer UI memory", content: "Agent bridge wrote this local memory record." }),
    });
    setOutput("#developerOutput", payload);
    toast("Bridge memory written", payload.event?.title || "memory");
    await loadAgentBridgeStatus();
  } catch(e) { toast("Bridge memory error", e.message); }
}

async function createBridgeTask() {
  try {
    const payload = await api("/api/agent-bridge/tasks", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), goal: "Review local project status", agent: "assistant" }),
    });
    setOutput("#developerOutput", payload);
    toast("Bridge task created", payload.task?.goal || "task");
    await loadAgentBridgeStatus();
  } catch(e) { toast("Bridge task error", e.message); }
}

async function loadProviderCertification(writeOutput = true) {
  try {
    const payload = await api("/api/communication/provider-certification");
    const feed = qs("#providerDocumentFeed");
    if (feed) {
      feed.innerHTML = [
        listItem("Mail providers", `${(payload.providers || []).length} presets`, "Gmail, Microsoft, generic IMAP/SMTP"),
        listItem("Calendar providers", `${(payload.caldav_profiles || []).length} profiles`, "Fastmail, Nextcloud, Radicale"),
      ].join("");
    }
    if (writeOutput) setOutput("#developerOutput", payload);
    return payload;
  } catch(e) {
    const feed = qs("#providerDocumentFeed");
    if (feed) feed.innerHTML = emptyState(`Provider certification unavailable: ${e.message}`);
    return null;
  }
}

async function recordOauthCallbackDemo() {
  try {
    const payload = await api(`/api/communication/oauth/callback?project=${encodeURIComponent(getProject())}&state=ui-demo&code=demo-code`);
    setOutput("#developerOutput", payload);
    toast("OAuth callback recorded", payload.callback?.code_received ? "Code fingerprint stored" : "No code");
    await loadProviderCertification(false);
  } catch(e) { toast("OAuth callback error", e.message); }
}

async function importDeveloperDocument() {
  try {
    const content = btoa("Imported from Developer UI into the local document processor.");
    const payload = await api("/api/documents/import", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), title: "Developer UI import", filename: "developer-import.txt", content_base64: content }),
    });
    setOutput("#developerOutput", payload);
    toast("Document imported", payload.document?.title || "document");
    await loadAgentBridgeStatus();
  } catch(e) { toast("Document import error", e.message); }
}

// =============================================
// Hard check
// =============================================

async function runHardCheck() {
  try {
    toast("Hard check running…");
    const payload = await api("/api/capabilities/hard-audit", {
      method: "POST",
      body: JSON.stringify({ project: "ui-hard-check" }),
    });
    setView("security");
    setOutput("#privacyOutput", payload);
    toast(`Hard check ${payload.ok ? "✅ passed" : "❌ failed"}`, `${payload.passed}/${payload.total}`);
  } catch(e) { toast("Hard check error", e.message); }
}

// =============================================
// Settings
// =============================================

function initSettings() {
  loadModelSettingsFromStorage();
  loadPrivacySettingsFromStorage();
  syncModelSettingsFields();
  syncPrivacySettingsFields();
  const themeSelect = qs("#settingsTheme");
  if (themeSelect) themeSelect.addEventListener("change", e => applyTheme(e.target.value));
  [
    "#settingChatModel", "#settingCodeModel", "#settingResearchModel", "#settingVoiceModel",
    "#settingQuantMode", "#settingTargetRam", "#settingModelTimeout", "#settingModelParameterCount",
  ].forEach(sel => qs(sel)?.addEventListener("change", () => {
    readModelSettingsFromFields();
    persistModelSettings();
  }));
  ["#settingLocalOnly", "#settingAuditAll"].forEach(sel => qs(sel)?.addEventListener("change", () => {
    readPrivacySettingsFromFields();
    persistPrivacySettings();
    applyPrivacySettingsToUi();
  }));
  qs("#saveSettings")?.addEventListener("click", saveSettings);
  qs("#applyCpuModelStack")?.addEventListener("click", applyCpuModelStack);
  qs("#refreshSettingsModels")?.addEventListener("click", loadSettingsModels);
  qs("#registerSettingsModel")?.addEventListener("click", registerSettingsModel);
  qs("#pullRecommendedOllamaModel")?.addEventListener("click", pullRecommendedOllamaModel);
  qs("#setupReasoningModel")?.addEventListener("click", () => setupModelForTask("reasoning"));
  qs("#setupCodeModel")?.addEventListener("click", () => setupModelForTask("code"));
  qs("#setupVisionModel")?.addEventListener("click", () => setupModelForTask("vision"));
  qs("#setupVoskModel")?.addEventListener("click", setupVoskStt);
  qs("#setupVoskFromVoice")?.addEventListener("click", setupVoskStt);
  qs("#registerLlamaCppModel")?.addEventListener("click", registerLlamaCppModel);
  qs("#downloadSettingsModel")?.addEventListener("click", downloadSettingsModel);
  qs("#quantizeSettingsModel")?.addEventListener("click", quantizeSettingsModel);
  qs("#deleteSettingsModel")?.addEventListener("click", deleteSettingsModel);
  qs("#loadSystemStatus")?.addEventListener("click", async () => {
    try {
      const [health, runtime] = await Promise.all([
        api("/api/health"),
        loadSmoothRuntime("chat"),
      ]);
      const d = { health, runtime, settings: state.modelSettings, privacy: state.privacySettings };
      setOutput("#settingsOutput", d);
    } catch(e) { toast("Status error", e.message); }
  });
  Promise.allSettled([loadSettingsModels(), loadModelSetupStatus(), loadModelJobs()]);
  applyPrivacySettingsToUi();
}

async function saveSettings() {
  const theme = qs("#settingsTheme")?.value || "system";
  applyTheme(theme);
  readModelSettingsFromFields();
  readPrivacySettingsFromFields();
  persistModelSettings();
  persistPrivacySettings();
  const voiceEngine = state.modelSettings.voiceModel.split(":")[0];
  const voiceSelect = qs("#voiceEngineSelect");
  if (voiceSelect && voiceEngine) voiceSelect.value = voiceEngine;
  try {
    const payload = await api("/api/settings/runtime", {
      method: "POST",
      body: JSON.stringify({
        settings: {
          ollama_url: qs("#settingOllamaUrl")?.value.trim() || "",
          ollama_model: qs("#settingOllamaModel")?.value.trim() || "",
          llamacpp_url: qs("#settingLlamaCppUrl")?.value.trim() || "",
          llamacpp_model: qs("#settingLlamaCppModel")?.value.trim() || "",
          chat_model: state.modelSettings.chatModel,
          code_model: state.modelSettings.codeModel,
          research_model: state.modelSettings.researchModel,
          voice_model: state.modelSettings.voiceModel,
          quantization_mode: state.modelSettings.quantizationMode,
          target_ram_gb: state.modelSettings.targetRamGb,
          model_timeout_seconds: state.modelSettings.modelTimeoutSeconds,
        },
      }),
    });
    toast("Settings saved", "Local model and quantization controls are active in the running OpenAgent core.");
    setOutput("#settingsOutput", { theme, saved: true, model_settings: state.modelSettings, privacy: state.privacySettings, runtime: payload.settings });
    loadSmoothRuntime("chat").catch(() => {});
  } catch(e) {
    toast("Settings save error", e.message);
  }
}

function applyCpuModelStack() {
  state.modelSettings = { ...state.modelSettings, ...DEFAULT_MODEL_SETTINGS };
  syncModelSettingsFields();
  persistModelSettings();
  toast("CPU model stack applied", "Phi-4 Mini for chat/research, Qwen2.5 Coder 3B for code, Gemma 3 2B for speed.");
  loadSmoothRuntime("chat").catch(() => {});
}

async function loadModelSetupStatus() {
  try {
    const [setupResult, modelsResult] = await Promise.allSettled([
      api("/api/local-ai/setup/status"),
      api("/api/local-ai/models"),
    ]);
    if (setupResult.status !== "fulfilled") throw setupResult.reason;
    const payload = setupResult.value;
    if (modelsResult.status === "fulfilled") {
      state.localModels = modelsResult.value.models || [];
      renderSettingsModels();
    }
    state.modelSetup = payload;
    renderModelSetupStatus(payload);
    return payload;
  } catch(e) {
    renderModelSetupStatus({ error: e.message });
    return null;
  }
}

async function loadModelJobs() {
  try {
    const payload = await api(`/api/local-ai/jobs?project=${encodeURIComponent(getProject())}&limit=8`);
    state.modelSetupJobs = payload.jobs || [];
    renderModelJobs();
    return state.modelSetupJobs;
  } catch(e) {
    renderModelJobs([{ status: "failed", current_step: e.message, progress: 100 }]);
    return [];
  }
}

function renderModelSetupStatus(payload = {}) {
  const el = qs("#modelSetupStatus");
  const chatEl = qs("#chatModelReadiness");
  const ollama = payload.ollama || {};
  const llamacpp = payload.llamacpp || {};
  const latestJob = (state.modelSetupJobs || [])[0] || {};
  const localModels = state.localModels || [];
  const ggufModels = localModels.filter(model =>
    model.status === "available" && String(model.local_path || "").toLowerCase().endsWith(".gguf")
  );
  const roleCount = role => localModels.filter(model => {
    if (model.status !== "available") return false;
    const metadata = model.metadata || {};
    const roles = Array.isArray(metadata.roles) ? metadata.roles : [metadata.role, metadata.registered_for].filter(Boolean);
    return roles.some(item => String(item || "").toLowerCase().includes(role));
  }).length;
  const installedOllamaCount = Number(ollama.installed_model_count || (Array.isArray(ollama.installed_models) ? ollama.installed_models.length : 0));
  const runnerReady = ollama.model_ready || llamacpp.model_ready;
  const runnerInstalled = ollama.available || llamacpp.available;
  const title = runnerReady
    ? "Real local chat model is ready"
    : ggufModels.length
      ? "Local GGUF models are downloaded"
    : runnerInstalled
      ? "Runner installed, chat model not ready yet"
      : "No local model runner found on PATH";
  const next = runnerReady
    ? "Keep Chat provider on Auto. OpenAgent will use the reachable local server."
    : ggufModels.length
      ? "Start llama-server with one downloaded GGUF, or install Ollama and use Setup Chat Model for one-click serving."
    : ollama.available
      ? "Click Setup Chat Model to pull the CPU chat model through Ollama."
      : llamacpp.available
        ? "Paste a .gguf path in Existing local path, click Register GGUF Model, then run the returned llama-server command."
        : "Install Ollama for one-click model downloads, or install llama.cpp and register a GGUF file.";
  const statusMarkup = `
    <div class="mini-status-row">
      <strong>${escapeHtml(title)}</strong>
      <span class="badge ${runnerReady ? "badge-success" : runnerInstalled ? "badge-warn" : ""}">${runnerReady ? "ready" : runnerInstalled ? "setup" : "missing"}</span>
    </div>
    <p>${escapeHtml(next)}</p>
    <div class="mini-chip-row">
      <span>Ollama: ${escapeHtml(ollama.available ? "installed" : "missing")} · ${escapeHtml(ollama.server_available ? "server on" : "server off")} · ${escapeHtml(installedOllamaCount ? `${installedOllamaCount} model(s)` : "no models")} · ${escapeHtml(ollama.selected_model || ollama.model || "")}</span>
      <span>llama.cpp: ${escapeHtml(llamacpp.available ? "installed" : "missing")} · ${escapeHtml(llamacpp.server_available ? "server on" : "server off")} · ${escapeHtml(llamacpp.model_ready ? "model ready" : "model missing")}</span>
      <span>Downloaded: ${escapeHtml(String(localModels.length))} total · ${escapeHtml(String(ggufModels.length))} GGUF · code ${escapeHtml(String(roleCount("code")))} · reasoning ${escapeHtml(String(roleCount("reasoning")))}</span>
    </div>
    ${ggufModels.length ? `<div class="mini-chip-row">${ggufModels.slice(0, 4).map(model => `<span>${escapeHtml(model.name || model.id)} · ${escapeHtml(formatBytes(model.size_bytes))}</span>`).join("")}</div>` : ""}
    ${latestJob.current_step ? `<p><strong>Latest setup job:</strong> ${escapeHtml(latestJob.current_step)}</p>` : ""}`;
  if (el) el.innerHTML = statusMarkup;
  if (chatEl) {
    chatEl.className = runnerReady ? "mini-status mini-status-success" : runnerInstalled ? "mini-status mini-status-warning" : "mini-status";
    chatEl.innerHTML = statusMarkup;
  }
}

function renderModelJobs(overrideJobs = null) {
  const el = qs("#modelSetupJobs");
  if (!el) return;
  const jobs = overrideJobs || state.modelSetupJobs || [];
  if (!jobs.length) {
    el.innerHTML = emptyState("No model setup jobs yet. Click Setup Chat Model to pull and verify the CPU chat model.");
    return;
  }
  el.innerHTML = jobs.map(job => {
    const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
    const terminal = ["completed", "failed", "cancelled"].includes(job.status);
    const badgeClass = job.status === "completed" ? "badge-success" : job.status === "failed" ? "badge-danger" : job.status === "cancelled" ? "badge-warn" : "badge-info";
    const benchmark = job.result?.benchmark || {};
    const benchmarkText = benchmark.tokens_per_second
      ? `${benchmark.tokens_per_second} tok/s · ${benchmark.provider || "local"}`
      : "";
    return listItem(
      job.model_name || "Chat model setup",
      `${job.current_step || job.status || "queued"}${benchmarkText ? ` · ${benchmarkText}` : ""}`,
      `<div class="job-actions">
        <span class="badge ${badgeClass}">${escapeHtml(job.status || "queued")}</span>
        <div class="progress-bar" title="${progress}%"><div class="progress-bar-fill" style="width:${progress}%"></div></div>
        ${terminal ? "" : `<button class="btn btn-mini btn-outline" data-cancel-model-job="${escapeHtml(job.id)}">Cancel</button>`}
      </div>`
    );
  }).join("");
  qsa("[data-cancel-model-job]").forEach(button => {
    button.addEventListener("click", () => cancelModelJob(button.dataset.cancelModelJob));
  });
}

async function pollModelJob(jobId) {
  if (!jobId || activeModelJobPolls.has(jobId)) return;
  activeModelJobPolls.add(jobId);
  const tick = async () => {
    try {
      const payload = await api(`/api/local-ai/jobs/${encodeURIComponent(jobId)}`);
      const job = payload.job;
      const existing = (state.modelSetupJobs || []).filter(item => item.id !== job.id);
      state.modelSetupJobs = [job, ...existing].slice(0, 8);
      renderModelJobs();
      renderModelSetupStatus(state.modelSetup || {});
      setOutput("#developerOutput", payload);
      if (["completed", "failed", "cancelled"].includes(job.status)) {
        activeModelJobPolls.delete(jobId);
        await Promise.allSettled([loadSettingsModels(), loadProviders(), loadModelSetupStatus(), loadDashboardTelemetry()]);
        await loadModelJobs();
        if (job.status === "completed") toast("Chat model ready", job.model_name || "Local chat model");
        if (job.status === "failed") toast("Model setup failed", job.error || job.current_step || "Check Developer output.");
        if (job.status === "cancelled") toast("Model setup cancelled", job.model_name || "");
        return;
      }
    } catch(e) {
      activeModelJobPolls.delete(jobId);
      toast("Model job status error", e.message);
      return;
    }
    setTimeout(tick, 1200);
  };
  tick();
}

async function cancelModelJob(jobId) {
  if (!jobId) return;
  try {
    const payload = await api("/api/local-ai/jobs/cancel", {
      method: "POST",
      body: JSON.stringify({ job_id: jobId }),
    });
    setOutput("#developerOutput", payload);
    await loadModelJobs();
    toast("Cancel requested", payload.job?.model_name || "model setup");
  } catch(e) {
    toast("Cancel error", e.message);
  }
}

async function pullRecommendedOllamaModel() {
  return setupModelForTask("chat");
}

async function setupModelForTask(task = "chat") {
  readModelSettingsFromFields();
  const modelByTask = {
    chat: state.modelSettings.chatModel || DEFAULT_MODEL_SETTINGS.chatModel,
    reasoning: state.modelSettings.researchModel || DEFAULT_MODEL_SETTINGS.researchModel,
    research: state.modelSettings.researchModel || DEFAULT_MODEL_SETTINGS.researchModel,
    code: state.modelSettings.codeModel || DEFAULT_MODEL_SETTINGS.codeModel,
    speed: DEFAULT_MODEL_SETTINGS.speedModel || "gemma3:2b",
    vision: "qwen2.5vl:3b",
  };
  const model = normalizeKnownModelName(modelByTask[task] || modelByTask.chat);
  const label = task === "code" ? "code model" : task === "reasoning" ? "reasoning model" : task === "vision" ? "vision model" : "chat model";
  try {
    toast(`Setting up ${label}`, `Started local setup job for ${model}.`);
    const payload = await api("/api/local-ai/models/setup/job", {
      method: "POST",
      body: JSON.stringify({ task, model, allow_network: true, project: getProject() }),
    });
    setOutput("#developerOutput", payload);
    setOutput("#settingsOutput", payload);
    const job = payload.job || {};
    state.modelSetupJobs = [job, ...(state.modelSetupJobs || []).filter(item => item.id !== job.id)].slice(0, 8);
    renderModelJobs();
    pollModelJob(job.id);
    await Promise.allSettled([loadModelSetupStatus(), loadModelJobs()]);
  } catch(e) {
    const message = e.message || "Ollama pull failed";
    const payload = {
      error: message,
      next_step: "Install Ollama for one-click downloads, or register a local .gguf file for llama.cpp.",
      model,
      task,
    };
    setOutput("#developerOutput", payload);
    setOutput("#settingsOutput", payload);
    loadModelSetupStatus().catch(() => {});
    toast("Model download error", message);
  }
}

async function setupVoskStt() {
  try {
    toast("Setting up Vosk", "Downloading the small local English STT model when needed.");
    const payload = await api("/api/voice/vosk/setup", {
      method: "POST",
      body: JSON.stringify({ allow_network: true, project: getProject() }),
    });
    setOutput("#developerOutput", payload);
    setOutput("#settingsOutput", payload);
    renderSttEngineStatus((payload.stt || {}).engines || []);
    renderVoiceEngineOptions((payload.stt || {}).engines || []);
    await loadVoiceStatus();
    toast("Vosk setup checked", payload.enabled ? "Local Vosk STT is ready." : payload.next_step || "Install voice extras to enable Vosk.");
  } catch(e) {
    const payload = {
      error: e.message,
      next_step: "Install voice extras with python -m pip install \"openagent-ai[voice]\", then run Setup Vosk again.",
    };
    setOutput("#developerOutput", payload);
    setOutput("#settingsOutput", payload);
    toast("Vosk setup error", e.message);
  }
}

function loadModelSettingsFromStorage() {
  try {
    const saved = JSON.parse(localStorage.getItem("oa-model-settings") || "{}");
    state.modelSettings = { ...state.modelSettings, ...saved };
  } catch(_) {}
  normalizeModelSettings();
}

function persistModelSettings() {
  try { localStorage.setItem("oa-model-settings", JSON.stringify(state.modelSettings)); } catch(_) {}
}

function normalizeModelSettings() {
  Object.entries(DEFAULT_MODEL_SETTINGS).forEach(([key, value]) => {
    if (state.modelSettings[key] === undefined || state.modelSettings[key] === null || state.modelSettings[key] === "") {
      state.modelSettings[key] = value;
    }
  });
  ["chatModel", "codeModel", "researchModel"].forEach(key => {
    state.modelSettings[key] = normalizeKnownModelName(state.modelSettings[key]);
  });
}

function loadPrivacySettingsFromStorage() {
  try {
    const saved = JSON.parse(localStorage.getItem("oa-privacy-settings") || "{}");
    state.privacySettings = { ...state.privacySettings, ...saved };
  } catch(_) {}
}

function persistPrivacySettings() {
  try { localStorage.setItem("oa-privacy-settings", JSON.stringify(state.privacySettings)); } catch(_) {}
}

function syncPrivacySettingsFields() {
  const localOnly = qs("#settingLocalOnly");
  const auditAll = qs("#settingAuditAll");
  if (localOnly) localOnly.checked = !!state.privacySettings.localOnly;
  if (auditAll) auditAll.checked = !!state.privacySettings.auditAll;
}

function readPrivacySettingsFromFields() {
  state.privacySettings = {
    localOnly: qs("#settingLocalOnly")?.checked !== false,
    auditAll: qs("#settingAuditAll")?.checked !== false,
  };
}

function applyPrivacySettingsToUi() {
  const allowExternal = qs("#allowExternal");
  if (!allowExternal) return;
  allowExternal.disabled = !!state.privacySettings.localOnly;
  if (state.privacySettings.localOnly) allowExternal.checked = false;
  const label = allowExternal.closest(".switch-row");
  if (label) label.classList.toggle("is-disabled", !!state.privacySettings.localOnly);
}

function syncModelSettingsFields() {
  const map = {
    "#settingChatModel": "chatModel",
    "#settingCodeModel": "codeModel",
    "#settingResearchModel": "researchModel",
    "#settingVoiceModel": "voiceModel",
    "#settingQuantMode": "quantizationMode",
    "#settingTargetRam": "targetRamGb",
    "#settingModelTimeout": "modelTimeoutSeconds",
    "#settingModelParameterCount": "parameterCountB",
  };
  Object.entries(map).forEach(([sel, key]) => {
    const el = qs(sel);
    if (el) el.value = state.modelSettings[key] || "";
  });
}

function readModelSettingsFromFields() {
  state.modelSettings = {
    ...state.modelSettings,
    chatModel: qs("#settingChatModel")?.value.trim() || DEFAULT_MODEL_SETTINGS.chatModel,
    codeModel: qs("#settingCodeModel")?.value.trim() || DEFAULT_MODEL_SETTINGS.codeModel,
    researchModel: qs("#settingResearchModel")?.value.trim() || DEFAULT_MODEL_SETTINGS.researchModel,
    voiceModel: qs("#settingVoiceModel")?.value.trim() || DEFAULT_MODEL_SETTINGS.voiceModel,
    quantizationMode: qs("#settingQuantMode")?.value || "auto",
    targetRamGb: qs("#settingTargetRam")?.value || "",
    modelTimeoutSeconds: qs("#settingModelTimeout")?.value || "35",
    parameterCountB: qs("#settingModelParameterCount")?.value || "",
  };
}

async function loadSettingsModels() {
  const payload = await api("/api/local-ai/models");
  state.localModels = payload.models || [];
  renderSettingsModels();
  return state.localModels;
}

function selectedSettingsModelId() {
  return qs("#settingsModelSelect")?.value || "";
}

function renderSettingsModels() {
  const select = qs("#settingsModelSelect");
  if (select) {
    select.innerHTML = state.localModels.length
      ? state.localModels.map(m => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.name)} · ${escapeHtml(m.status)}</option>`).join("")
      : '<option value="">No registered models</option>';
  }
  const feed = qs("#settingsModelFeed");
  if (feed) {
    feed.innerHTML = state.localModels.length
      ? state.localModels.slice(0, 8).map(m => listItem(
          m.name || m.id,
          `${m.provider} · ${m.status} · ${formatBytes(m.size_bytes || 0)}`,
          `<span class="badge">${escapeHtml((m.metadata?.tier || m.metadata?.quantization || "local"))}</span>`
        )).join("")
      : emptyState("Register a local model, local file path, or local provider model.");
  }
}

async function registerSettingsModel() {
  try {
    const name = qs("#settingsModelName")?.value.trim();
    if (!name) { toast("Model name required"); return; }
    const payload = await api("/api/local-ai/models/register", {
      method: "POST",
      body: JSON.stringify({
        name,
        provider: qs("#settingsModelProvider")?.value.trim() || "local",
        source_url: qs("#settingsModelSource")?.value.trim() || "",
        local_path: qs("#settingsModelLocalPath")?.value.trim() || "",
        metadata: {
          registered_from: "settings",
          quantization_mode: state.modelSettings.quantizationMode,
        },
      }),
    });
    setOutput("#settingsOutput", payload);
    toast("Model registered", name);
    await loadSettingsModels();
  } catch(e) { toast("Register model error", e.message); }
}

async function registerLlamaCppModel() {
  try {
    const localPath = qs("#settingsModelLocalPath")?.value.trim() || qs("#settingsModelSource")?.value.trim() || "";
    const name = qs("#settingsModelName")?.value.trim() || "";
    if (!localPath) {
      toast("GGUF path required", "Paste the local .gguf model file path into Existing local path.");
      return;
    }
    const payload = await api("/api/local-ai/llamacpp/register", {
      method: "POST",
      body: JSON.stringify({ local_path: localPath, name, project: getProject() }),
    });
    setOutput("#developerOutput", payload);
    setOutput("#settingsOutput", payload);
    toast("GGUF registered", payload.model?.name || name || "llama.cpp model");
    await Promise.allSettled([loadSettingsModels(), loadProviders(), loadModelSetupStatus()]);
  } catch(e) {
    toast("GGUF register error", e.message);
    setOutput("#developerOutput", { error: e.message });
  }
}

async function downloadSettingsModel() {
  const modelId = selectedSettingsModelId();
  if (!modelId) { toast("Select a model first"); return; }
  try {
    const payload = await api("/api/local-ai/models/download", {
      method: "POST",
      body: JSON.stringify({ model_id: modelId, allow_network: false }),
    });
    setOutput("#settingsOutput", payload);
    toast("Model copied locally", payload.model?.name || modelId);
    await loadSettingsModels();
  } catch(e) { toast("Download/copy error", e.message); }
}

async function quantizeSettingsModel() {
  readModelSettingsFromFields();
  const modelId = selectedSettingsModelId();
  if (!modelId) { toast("Select a model first"); return; }
  try {
    const payload = await api("/api/local-ai/models/auto-quantize", {
      method: "POST",
      body: JSON.stringify({
        model_id: modelId,
        task: "chat",
        target_ram_gb: state.modelSettings.targetRamGb || undefined,
        quantization_mode: state.modelSettings.quantizationMode,
        parameter_count_b: state.modelSettings.parameterCountB || undefined,
      }),
    });
    setOutput("#settingsOutput", payload);
    toast("Quantization plan ready", payload.profile?.quantization_method || "smooth");
  } catch(e) { toast("Quantization error", e.message); }
}

async function deleteSettingsModel() {
  const modelId = selectedSettingsModelId();
  if (!modelId) { toast("Select a model first"); return; }
  try {
    const payload = await api("/api/local-ai/models/delete", {
      method: "POST",
      body: JSON.stringify({ model_id: modelId, delete_file: false }),
    });
    setOutput("#settingsOutput", payload);
    toast("Model record deleted");
    await loadSettingsModels();
  } catch(e) { toast("Delete model error", e.message); }
}

// =============================================
// PhD Application Studio
// =============================================

const PHD_PROFILE_FIELDS = {
  full_name: "#phdFullName",
  email: "#phdEmail",
  phone: "#phdPhone",
  location: "#phdLocation",
  citizenship: "#phdCitizenship",
  current_role: "#phdCurrentRole",
  links: "#phdLinks",
  summary: "#phdSummary",
  research_interests: "#phdResearchInterests",
  education: "#phdEducation",
  experience: "#phdExperience",
  skills: "#phdSkills",
  projects: "#phdProjects",
  publications: "#phdPublications",
  awards: "#phdAwards",
  courses: "#phdCourses",
  certifications: "#phdCertifications",
  languages: "#phdLanguages",
  referees: "#phdReferees",
};

function setPhdTab(tab) {
  const tabMap = { studio: "#phdTabStudio", data: "#phdTabData", research: "#phdTabResearch", library: "#phdTabLibrary" };
  qsa("[data-phd-panel]").forEach(panel => panel.classList.toggle("active", panel.dataset.phdPanel === tab));
  Object.entries(tabMap).forEach(([name, selector]) => {
    const button = qs(selector);
    if (!button) return;
    const active = name === tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  if (tab === "research") loadPhdResearchForSelection().catch(() => {});
  if (tab === "library") renderPhdArtifactLibrary();
}

async function loadPhdWorkspace() {
  const project = encodeURIComponent(getProject());
  try {
    const [status, profile, documents, applications, artifacts] = await Promise.all([
      api(`/api/phd/status?project=${project}`),
      api(`/api/phd/profile?project=${project}`),
      api(`/api/phd/documents?project=${project}&limit=150`),
      api(`/api/phd/applications?project=${project}&limit=150`),
      api(`/api/phd/artifacts?project=${project}&limit=300`),
    ]);
    state.phd.status = status;
    state.phd.profile = profile.profile || {};
    state.phd.profileCompleteness = profile.completeness || {};
    state.phd.documents = documents.documents || [];
    state.phd.applications = applications.applications || [];
    state.phd.artifacts = artifacts.artifacts || [];
    if (state.phd.selectedApplicationId && !state.phd.applications.some(item => item.id === state.phd.selectedApplicationId)) {
      state.phd.selectedApplicationId = "";
    }
    if (!state.phd.selectedApplicationId && state.phd.applications.length) {
      state.phd.selectedApplicationId = state.phd.applications[0].id;
    }
    renderPhdHeader();
    renderPhdProfile();
    renderPhdDocuments();
    renderPhdApplications();
    renderPhdArtifactLibrary();
    const selected = selectedPhdApplication();
    if (selected) populatePhdApplicationForm(selected);
    if (state.phd.selectedArtifactId) await openPhdArtifact(state.phd.selectedArtifactId, { quiet: true });
    return state.phd;
  } catch (error) {
    toast("PhD workspace unavailable", error.message);
    throw error;
  }
}

function renderPhdHeader() {
  const score = Number(state.phd.profileCompleteness?.score || 0);
  if (qs("#phdProfileScore")) qs("#phdProfileScore").textContent = `${Math.round(score * 100)}%`;
  if (qs("#phdApplicationCount")) qs("#phdApplicationCount").textContent = String(state.phd.applications.length);
  if (qs("#phdArtifactCount")) qs("#phdArtifactCount").textContent = String(state.phd.artifacts.length);
  if (qs("#phdDocumentCount")) qs("#phdDocumentCount").textContent = String(state.phd.documents.length);
  if (qs("#phdLibraryCount")) qs("#phdLibraryCount").textContent = String(state.phd.artifacts.length);
  if (qs("#phdRagChunkCount")) qs("#phdRagChunkCount").textContent = String(state.phd.status?.rag?.indexed_chunks || 0);
}

function renderPhdProfile() {
  const profile = state.phd.profile || {};
  Object.entries(PHD_PROFILE_FIELDS).forEach(([key, selector]) => {
    const input = qs(selector);
    if (input) input.value = typeof profile[key] === "string" ? profile[key] : formatProfileValue(profile[key]);
  });
  const preferences = profile.writing_preferences || {};
  if (qs("#phdWritingTone")) qs("#phdWritingTone").value = preferences.tone || "clear, specific, and professional";
  if (qs("#phdWritingSpelling")) qs("#phdWritingSpelling").value = preferences.spelling || "international English";
}

function formatProfileValue(value) {
  if (!value) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(item => typeof item === "string" ? item : JSON.stringify(item)).join("\n");
  return Object.entries(value).map(([key, item]) => `${key}: ${item}`).join("\n");
}

function profileFromPhdFields() {
  const profile = {};
  Object.entries(PHD_PROFILE_FIELDS).forEach(([key, selector]) => { profile[key] = qs(selector)?.value.trim() || ""; });
  profile.writing_preferences = {
    tone: qs("#phdWritingTone")?.value.trim() || "clear, specific, and professional",
    spelling: qs("#phdWritingSpelling")?.value || "international English",
    avoid: "generic praise, inflated claims, and unsupported superlatives",
  };
  return profile;
}

async function savePhdProfile() {
  try {
    const payload = await api("/api/phd/profile", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), profile: profileFromPhdFields() }),
    });
    state.phd.profile = payload.profile || {};
    state.phd.profileCompleteness = payload.completeness || {};
    renderPhdHeader();
    if (qs("#phdDataSaved")) qs("#phdDataSaved").textContent = "Saved locally";
    toast("My Data saved", "The reusable application profile is available to this project.");
  } catch (error) { toast("Profile save failed", error.message); }
}

async function uploadPhdDocument() {
  const file = qs("#phdDocumentFile")?.files?.[0];
  if (!file) { toast("Choose a document first"); return; }
  if (file.size > 25 * 1024 * 1024) { toast("File is too large", "Maximum upload size is 25 MB."); return; }
  const button = qs("#uploadPhdDocument");
  if (button) button.disabled = true;
  try {
    const contentBase64 = await blobToBase64(file);
    const payload = await api("/api/phd/documents", {
      method: "POST",
      body: JSON.stringify({
        project: getProject(),
        filename: file.name,
        content_type: file.type || "application/octet-stream",
        content_base64: contentBase64,
        category: qs("#phdDocumentCategory")?.value || "other",
        label: qs("#phdDocumentLabel")?.value.trim() || file.name.replace(/\.[^.]+$/, ""),
      }),
    });
    toast(payload.document?.duplicate ? "Document already stored" : "Document added", `${payload.document?.chars_extracted || 0} characters available as evidence.`);
    if (qs("#phdDocumentFile")) qs("#phdDocumentFile").value = "";
    if (qs("#phdDocumentLabel")) qs("#phdDocumentLabel").value = "";
    await loadPhdDocuments();
  } catch (error) { toast("Document upload failed", error.message); }
  finally { if (button) button.disabled = false; }
}

async function loadPhdDocuments() {
  const payload = await api(`/api/phd/documents?project=${encodeURIComponent(getProject())}&limit=150`);
  state.phd.documents = payload.documents || [];
  renderPhdDocuments();
  renderPhdHeader();
}

function renderPhdDocuments() {
  const library = qs("#phdDocumentLibrary");
  if (library) {
    library.innerHTML = state.phd.documents.length ? state.phd.documents.map(document => `
      <div class="phd-record">
        <div><strong>${escapeHtml(document.label)}</strong><p>${escapeHtml(document.category)} · ${formatBytes(document.size_bytes)} · ${document.chars_extracted || 0} extracted chars</p></div>
        <div class="phd-record-actions"><button class="btn btn-mini btn-danger" data-phd-delete-document="${escapeHtml(document.id)}">Delete</button></div>
      </div>`).join("") : emptyState("No personal documents uploaded yet.");
    qsa("[data-phd-delete-document]", library).forEach(button => button.addEventListener("click", () => deletePhdDocument(button.dataset.phdDeleteDocument)));
  }
  if (qs("#phdDocumentCount")) qs("#phdDocumentCount").textContent = String(state.phd.documents.length);
  const selector = qs("#phdSourceSelector");
  if (selector) {
    selector.innerHTML = state.phd.documents.length ? state.phd.documents.map(document => `
      <label class="phd-source-check">
        <input type="checkbox" data-phd-source-id="${escapeHtml(document.id)}" checked>
        <span><strong>${escapeHtml(document.label)}</strong><span>${escapeHtml(document.category)} · ${document.chars_extracted || 0} chars</span></span>
      </label>`).join("") : emptyState("Upload CV, transcripts, or writing samples in My Data.");
  }
}

async function deletePhdDocument(documentId) {
  if (!documentId || !window.confirm("Delete this local source document?")) return;
  try {
    await api(`/api/phd/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" });
    await loadPhdDocuments();
    toast("Document deleted");
  } catch (error) { toast("Delete failed", error.message); }
}

function selectedPhdApplication() {
  return state.phd.applications.find(item => item.id === state.phd.selectedApplicationId) || null;
}

function renderPhdApplications() {
  const applications = state.phd.applications;
  const options = applications.map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label)}</option>`).join("");
  const studio = qs("#phdApplicationSelect");
  const research = qs("#phdResearchApplication");
  const library = qs("#phdLibraryApplication");
  if (studio) {
    studio.innerHTML = `<option value="">New application</option>${options}`;
    studio.value = state.phd.selectedApplicationId || "";
  }
  if (research) {
    research.innerHTML = `<option value="">Select a saved position</option>${options}`;
    research.value = state.phd.selectedApplicationId || "";
  }
  if (library) {
    const current = library.value;
    library.innerHTML = `<option value="">All positions</option>${options}`;
    library.value = current && applications.some(item => item.id === current) ? current : "";
  }
  if (qs("#exportPhdBundle")) qs("#exportPhdBundle").disabled = !state.phd.selectedApplicationId;
  if (qs("#deletePhdApplication")) qs("#deletePhdApplication").disabled = !state.phd.selectedApplicationId;
}

function populatePhdApplicationForm(application) {
  const values = {
    "#phdInstitution": application.institution,
    "#phdProgram": application.program,
    "#phdPositionTitle": application.position_title,
    "#phdSupervisor": application.supervisor,
    "#phdLab": application.lab,
    "#phdDepartment": application.department,
    "#phdDeadline": application.deadline,
    "#phdPositionUrl": application.position_url,
    "#phdPositionDescription": application.description,
    "#phdRequirements": (application.requirements || []).join("\n"),
    "#phdPositionNotes": application.notes,
  };
  Object.entries(values).forEach(([selector, value]) => { if (qs(selector)) qs(selector).value = value || ""; });
  if (qs("#phdPositionStatus")) qs("#phdPositionStatus").textContent = application.status || "draft";
  if (qs("#phdWebQuery") && !qs("#phdWebQuery").value.trim()) {
    qs("#phdWebQuery").value = [application.supervisor, application.lab, application.institution, application.position_title, "research publications PhD"].filter(Boolean).join(" ");
  }
  if (qs("#phdResearchUrls") && application.position_url && !qs("#phdResearchUrls").value.trim()) qs("#phdResearchUrls").value = application.position_url;
}

function clearPhdApplicationForm() {
  state.phd.selectedApplicationId = "";
  ["#phdInstitution", "#phdProgram", "#phdPositionTitle", "#phdSupervisor", "#phdLab", "#phdDepartment", "#phdDeadline", "#phdPositionUrl", "#phdPositionDescription", "#phdRequirements", "#phdPositionNotes"].forEach(selector => { if (qs(selector)) qs(selector).value = ""; });
  if (qs("#phdPositionStatus")) qs("#phdPositionStatus").textContent = "Draft";
  if (qs("#phdApplicationSelect")) qs("#phdApplicationSelect").value = "";
  if (qs("#deletePhdApplication")) qs("#deletePhdApplication").disabled = true;
  if (qs("#exportPhdBundle")) qs("#exportPhdBundle").disabled = true;
}

function phdApplicationFromFields() {
  return {
    institution: qs("#phdInstitution")?.value.trim() || "",
    program: qs("#phdProgram")?.value.trim() || "",
    position_title: qs("#phdPositionTitle")?.value.trim() || "",
    supervisor: qs("#phdSupervisor")?.value.trim() || "",
    lab: qs("#phdLab")?.value.trim() || "",
    department: qs("#phdDepartment")?.value.trim() || "",
    deadline: qs("#phdDeadline")?.value || "",
    position_url: qs("#phdPositionUrl")?.value.trim() || "",
    description: qs("#phdPositionDescription")?.value.trim() || "",
    requirements: (qs("#phdRequirements")?.value || "").split(/\r?\n/).map(item => item.trim()).filter(Boolean),
    notes: qs("#phdPositionNotes")?.value.trim() || "",
    status: selectedPhdApplication()?.status || "draft",
  };
}

async function savePhdApplication({ quiet = false } = {}) {
  const application = phdApplicationFromFields();
  if (!application.institution && !application.program && !application.position_title) {
    if (!quiet) toast("Add a position", "Institution, program, or position title is required.");
    return "";
  }
  try {
    const id = state.phd.selectedApplicationId;
    const payload = await api(id ? `/api/phd/applications/${encodeURIComponent(id)}` : "/api/phd/applications", {
      method: id ? "PATCH" : "POST",
      body: JSON.stringify({ project: getProject(), application }),
    });
    state.phd.selectedApplicationId = payload.application?.id || id;
    await loadPhdApplications();
    if (!quiet) toast(id ? "Position updated" : "Position saved", payload.application?.label || application.institution);
    return state.phd.selectedApplicationId;
  } catch (error) {
    if (!quiet) toast("Position save failed", error.message);
    return "";
  }
}

async function loadPhdApplications() {
  const payload = await api(`/api/phd/applications?project=${encodeURIComponent(getProject())}&limit=150`);
  state.phd.applications = payload.applications || [];
  renderPhdApplications();
  renderPhdHeader();
  const selected = selectedPhdApplication();
  if (selected) populatePhdApplicationForm(selected);
}

async function selectPhdApplication(applicationId) {
  state.phd.selectedApplicationId = applicationId || "";
  if (!applicationId) {
    clearPhdApplicationForm();
    return;
  }
  const application = selectedPhdApplication();
  if (application) populatePhdApplicationForm(application);
  renderPhdApplications();
  renderPhdArtifactLibrary();
  await loadPhdResearchForSelection().catch(() => {});
}

async function deletePhdApplication() {
  const id = state.phd.selectedApplicationId;
  if (!id || !window.confirm("Delete this position, its research sources, and all generated revisions?")) return;
  try {
    await api(`/api/phd/applications/${encodeURIComponent(id)}`, { method: "DELETE" });
    state.phd.selectedApplicationId = "";
    state.phd.selectedArtifactId = "";
    clearPhdApplicationForm();
    await loadPhdWorkspace();
    toast("Position deleted");
  } catch (error) { toast("Delete failed", error.message); }
}

function selectedPhdDocumentIds() {
  return qsa("[data-phd-source-id]:checked").map(input => input.dataset.phdSourceId).filter(Boolean);
}

function renderPhdGenerationRun(run) {
  if (!run) return;
  state.phd.generationRun = run;
  const container = qs("#phdGenerationProgress");
  if (container) container.hidden = false;
  const progress = Math.max(0, Math.min(100, Number(run.progress || 0)));
  if (qs("#phdProgressFill")) qs("#phdProgressFill").style.width = `${progress}%`;
  if (qs("#phdProgressPercent")) qs("#phdProgressPercent").textContent = `${progress}%`;
  if (qs("#phdProgressMessage")) qs("#phdProgressMessage").textContent = run.progress_message || run.status || "Working locally";
  if (qs("#phdGenerationState")) qs("#phdGenerationState").textContent = run.status === "completed" ? "Ready" : run.status === "failed" ? "Failed" : `${progress}%`;
  const timeline = qs("#phdProgressTimeline");
  if (!timeline) return;
  const events = Array.isArray(run.events) ? run.events.slice(-10) : [];
  timeline.innerHTML = events.map(event => {
    const model = [event.provider || event.requested_provider, event.model || event.requested_model].filter(Boolean).join(" · ");
    return `<div class="phd-progress-event ${escapeHtml(event.status || "running")}">
      <strong>${escapeHtml(event.stage || "working")}</strong>
      <span>${escapeHtml(event.message || "Working locally.")}${model ? `<br><small>${escapeHtml(model)}</small>` : ""}</span>
    </div>`;
  }).join("");
  timeline.scrollTop = timeline.scrollHeight;
}

async function pollPhdGenerationRun(runId) {
  while (state.phd.busy && state.phd.generationRun?.id === runId) {
    const payload = await api(`/api/phd/generation-runs/${encodeURIComponent(runId)}?project=${encodeURIComponent(getProject())}`);
    const run = payload.run;
    renderPhdGenerationRun(run);
    if (["completed", "failed"].includes(run.status)) return run;
    await new Promise(resolve => setTimeout(resolve, 650));
  }
  return state.phd.generationRun;
}

async function generatePhdBundle() {
  const applicationId = await savePhdApplication({ quiet: true });
  if (!applicationId) { toast("Save a position first"); return; }
  const autoDetect = Boolean(qs("#phdAutoDetectOutputs")?.checked);
  const artifactTypes = autoDetect ? [] : qsa("[data-phd-artifact-type]:checked").map(input => input.value);
  if (!autoDetect && !artifactTypes.length) {
    toast("Choose an output", "Select one or more documents, or enable Choose from command.");
    return;
  }
  const command = qs("#phdCommand")?.value.trim() || "Prepare the selected application documents using my verified evidence.";
  const allowExternal = Boolean(qs("#phdAllowExternalModel")?.checked);
  const provider = qs("#phdProvider")?.value || "auto";
  const application = selectedPhdApplication();
  const researchBeforeGeneration = Boolean(application?.position_url);
  if (allowExternal && provider !== "openai-compatible") {
    toast("External approval noted", "The selected local route will still keep evidence on device unless it resolves to an external provider.");
  }
  const button = qs("#generatePhdBundle");
  state.phd.busy = true;
  if (button) button.disabled = true;
  if (qs("#phdGenerationState")) qs("#phdGenerationState").textContent = "Starting";
  try {
    const payload = await api(`/api/phd/applications/${encodeURIComponent(applicationId)}/generate/start`, {
      method: "POST",
      body: JSON.stringify({
        project: getProject(), command, artifact_types: artifactTypes,
        document_ids: selectedPhdDocumentIds(), provider, allow_external_model: allowExternal,
        research_before_generation: researchBeforeGeneration,
        allow_external_research: researchBeforeGeneration,
      }),
    });
    renderPhdGenerationRun(payload.run);
    const run = await pollPhdGenerationRun(payload.run.id);
    if (!run || run.status !== "completed") throw new Error(run?.error || run?.progress_message || "Generation did not complete.");
    const result = run.result || {};
    const generated = result.artifacts || [];
    if (result.research) await loadPhdApplications();
    await loadPhdArtifacts();
    if (generated[0]) await openPhdArtifact(generated[0].id, { quiet: true });
    if (qs("#phdGenerationState")) qs("#phdGenerationState").textContent = `${generated.length} ready`;
    if (qs("#exportPhdBundle")) qs("#exportPhdBundle").disabled = !generated.length;
    const sourceCount = result.research?.sources?.length || 0;
    toast("Application documents ready", `${generated.length} independently reviewed deliverable${generated.length === 1 ? "" : "s"} created locally${sourceCount ? ` from ${sourceCount} public sources` : ""}.`);
  } catch (error) {
    if (qs("#phdGenerationState")) qs("#phdGenerationState").textContent = "Failed";
    toast("Generation failed", error.message);
  } finally {
    state.phd.busy = false;
    if (button) button.disabled = false;
  }
}

async function loadPhdArtifacts() {
  const payload = await api(`/api/phd/artifacts?project=${encodeURIComponent(getProject())}&limit=300`);
  state.phd.artifacts = payload.artifacts || [];
  renderPhdArtifactLibrary();
  renderPhdHeader();
}

function renderPhdArtifactLibrary() {
  const container = qs("#phdArtifactLibrary");
  if (!container) return;
  const filter = qs("#phdLibraryApplication")?.value || "";
  const artifacts = state.phd.artifacts.filter(item => !filter || item.application_id === filter);
  container.innerHTML = artifacts.length ? artifacts.map(item => {
    const app = state.phd.applications.find(candidate => candidate.id === item.application_id);
    return `<div class="phd-record">
      <button class="phd-artifact-button ${item.id === state.phd.selectedArtifactId ? "active" : ""}" data-phd-open-artifact="${escapeHtml(item.id)}">
        <strong>${escapeHtml(item.title)}</strong>
        <p>${escapeHtml(app?.institution || "Application")} · revision ${item.revision} · ${item.word_count} words · score ${item.quality?.score ?? "-"}</p>
      </button>
      <div class="phd-record-actions">
        <button class="btn btn-mini btn-soft" data-phd-quick-export="${escapeHtml(item.id)}" data-phd-export-format="docx">DOCX</button>
        <button class="btn btn-mini btn-soft" data-phd-quick-export="${escapeHtml(item.id)}" data-phd-export-format="pdf">PDF</button>
        <button class="btn btn-mini btn-soft" data-phd-quick-export="${escapeHtml(item.id)}" data-phd-export-format="md">MD</button>
      </div>
    </div>`;
  }).join("") : emptyState("No application documents generated yet.");
  qsa("[data-phd-open-artifact]", container).forEach(button => button.addEventListener("click", () => openPhdArtifact(button.dataset.phdOpenArtifact)));
  qsa("[data-phd-quick-export]", container).forEach(button => button.addEventListener("click", () => {
    exportPhdArtifact(button.dataset.phdExportFormat || "docx", button.dataset.phdQuickExport);
  }));
  if (qs("#phdLibraryCount")) qs("#phdLibraryCount").textContent = String(artifacts.length);
}

async function openPhdArtifact(artifactId, { quiet = false } = {}) {
  if (!artifactId) return;
  try {
    const payload = await api(`/api/phd/artifacts/${encodeURIComponent(artifactId)}`);
    const artifact = payload.artifact;
    state.phd.selectedArtifactId = artifact.id;
    renderPhdArtifact(artifact);
    renderPhdArtifactLibrary();
    if (!quiet) toast("Deliverable opened", `Revision ${artifact.revision}`);
  } catch (error) { if (!quiet) toast("Open failed", error.message); }
}

function renderPhdArtifact(artifact) {
  if (qs("#phdPreviewTitle")) qs("#phdPreviewTitle").textContent = artifact.title;
  if (qs("#phdArtifactPreview")) qs("#phdArtifactPreview").innerHTML = renderMarkdown(artifact.content);
  if (qs("#phdEditorTitle")) qs("#phdEditorTitle").textContent = artifact.title;
  if (qs("#phdEditorRevision")) qs("#phdEditorRevision").textContent = `Revision ${artifact.revision}`;
  if (qs("#phdArtifactEditor")) qs("#phdArtifactEditor").value = artifact.content;
  renderPhdQuality(artifact.quality || {}, "#phdQualityPanel", artifact.pipeline || {});
  renderPhdQuality(artifact.quality || {}, "#phdEditorQuality", artifact.pipeline || {});
  ["#phdPreviewDocx", "#phdPreviewPdf", "#phdPreviewMarkdown", "#savePhdArtifactRevision", "#reviewPhdAuthenticity", "#deletePhdArtifact"].forEach(selector => {
    if (qs(selector)) qs(selector).disabled = false;
  });
}

function renderPhdQuality(quality, selector, pipeline = {}) {
  const panel = qs(selector);
  if (!panel) return;
  const flags = quality.flags || [];
  const checks = [
    [quality.length_ok, `Length ${quality.word_count ?? 0} words`],
    [!(quality.placeholders || []).length, `${(quality.placeholders || []).length} unresolved placeholders`],
    [!(quality.unsupported_numeric_claims || []).length, `${(quality.unsupported_numeric_claims || []).length} numeric claims to verify`],
    [Number(quality.evidence_alignment || 0) >= .08, `Evidence alignment ${Math.round(Number(quality.evidence_alignment || 0) * 100)}%`],
  ];
  if (quality.voice_alignment !== null && quality.voice_alignment !== undefined) checks.push([Number(quality.voice_alignment) >= .65, `Writing-sample alignment ${Math.round(Number(quality.voice_alignment) * 100)}%`]);
  panel.innerHTML = `
    <div class="phd-quality-score"><strong>${quality.score ?? "-"}</strong><span>/ 100 local review</span></div>
    <div class="phd-check-list">${checks.map(([ok, label]) => `<div class="phd-check-item ${ok ? "ok" : "warn"}">${ok ? "Pass" : "Review"}: ${escapeHtml(label)}</div>`).join("")}</div>
    ${flags.length ? `<div class="phd-quality-note"><strong>Before submission</strong><br>${flags.map(escapeHtml).join("<br>")}</div>` : '<div class="phd-quality-note">No automatic issues found. A final human review is still required.</div>'}
    ${pipeline.stage_count ? `<div class="phd-pipeline-note"><strong>Writing pipeline</strong><span>${Number(pipeline.stage_count)} stages · ${Number(pipeline.distinct_models || 1)} model${Number(pipeline.distinct_models || 1) === 1 ? "" : "s"} · ${pipeline.quality_gate === "passed" ? "independent proofread passed" : pipeline.finalized ? "review finalized" : "quality gate needs review"}</span></div>` : ""}
    <p class="phd-quality-note">${escapeHtml(quality.detector_note || "This checks grounding and authentic voice; it does not target AI detectors.")}</p>`;
}

async function savePhdArtifactRevision() {
  const artifactId = state.phd.selectedArtifactId;
  const content = qs("#phdArtifactEditor")?.value.trim() || "";
  if (!artifactId || !content) return;
  try {
    const payload = await api(`/api/phd/artifacts/${encodeURIComponent(artifactId)}`, {
      method: "PATCH",
      body: JSON.stringify({ project: getProject(), content }),
    });
    state.phd.selectedArtifactId = payload.artifact.id;
    await loadPhdArtifacts();
    renderPhdArtifact(payload.artifact);
    toast("New revision saved", `Revision ${payload.artifact.revision}`);
  } catch (error) { toast("Revision save failed", error.message); }
}

async function reviewPhdAuthenticity() {
  const artifactId = state.phd.selectedArtifactId;
  if (!artifactId) return;
  try {
    const payload = await api(`/api/phd/artifacts/${encodeURIComponent(artifactId)}/authenticity`, { method: "POST", body: JSON.stringify({ project: getProject() }) });
    renderPhdQuality(payload.quality || {}, "#phdEditorQuality");
    renderPhdQuality(payload.quality || {}, "#phdQualityPanel");
    toast("Grounding review complete", `Score ${payload.quality?.score ?? "-"}/100`);
  } catch (error) { toast("Review failed", error.message); }
}

async function deletePhdArtifact() {
  const artifactId = state.phd.selectedArtifactId;
  if (!artifactId || !window.confirm("Delete this revision? Other revisions will remain.")) return;
  try {
    await api(`/api/phd/artifacts/${encodeURIComponent(artifactId)}`, { method: "DELETE" });
    state.phd.selectedArtifactId = "";
    await loadPhdArtifacts();
    if (qs("#phdArtifactEditor")) qs("#phdArtifactEditor").value = "";
    toast("Revision deleted");
  } catch (error) { toast("Delete failed", error.message); }
}

async function exportPhdArtifact(format, artifactId = "") {
  const id = artifactId || state.phd.selectedArtifactId;
  if (!id) return;
  try {
    const payload = await api(`/api/phd/artifacts/${encodeURIComponent(id)}/export`, {
      method: "POST",
      body: JSON.stringify({ project: getProject(), format }),
    });
    downloadBase64File(payload);
    toast("Export ready", payload.filename);
  } catch (error) { toast("Export failed", error.message); }
}

async function exportPhdBundle() {
  const applicationId = state.phd.selectedApplicationId;
  if (!applicationId) return;
  try {
    const payload = await api(`/api/phd/applications/${encodeURIComponent(applicationId)}/export`, {
      method: "POST",
      body: JSON.stringify({
        project: getProject(),
        include_source_documents: Boolean(qs("#phdIncludeSourcesInPackage")?.checked),
        document_ids: selectedPhdDocumentIds(),
      }),
    });
    downloadBase64File(payload);
    toast("Application package ready", `${payload.artifact_count || 0} latest deliverables included.`);
  } catch (error) { toast("Package export failed", error.message); }
}

function downloadBase64File(payload) {
  const binary = atob(payload.content_base64 || "");
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  const url = URL.createObjectURL(new Blob([bytes], { type: payload.content_type || "application/octet-stream" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = payload.filename || "openagent-export.bin";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

async function runPhdResearch() {
  const applicationId = qs("#phdResearchApplication")?.value || state.phd.selectedApplicationId;
  if (!applicationId) { toast("Select a saved position first"); return; }
  const allowExternal = Boolean(qs("#phdAllowWebResearch")?.checked);
  const urls = (qs("#phdResearchUrls")?.value || "").split(/\r?\n/).map(item => item.trim()).filter(Boolean);
  if (urls.length && !allowExternal) { toast("Approval required", "Enable the per-run web research boundary first."); return; }
  const button = qs("#runPhdResearch");
  if (button) button.disabled = true;
  if (qs("#phdResearchState")) qs("#phdResearchState").textContent = allowExternal ? "Searching public web..." : "Reading local evidence...";
  try {
    const payload = await api(`/api/phd/applications/${encodeURIComponent(applicationId)}/research`, {
      method: "POST",
      body: JSON.stringify({
        project: getProject(), query: qs("#phdWebQuery")?.value.trim() || "", urls,
        allow_external: allowExternal, max_sources: 6,
      }),
    });
    state.phd.selectedApplicationId = applicationId;
    renderPhdResearch(payload.summary || "", payload.sources || []);
    await loadPhdApplications();
    if (qs("#phdResearchState")) qs("#phdResearchState").textContent = `${(payload.sources || []).length} sources`;
    toast("Position research complete", payload.search_error ? `Public search warning: ${payload.search_error}` : `${(payload.new_sources || []).length} new sources saved locally.`);
  } catch (error) {
    if (qs("#phdResearchState")) qs("#phdResearchState").textContent = "Failed";
    toast("Research failed", error.message);
  } finally { if (button) button.disabled = false; }
}

async function loadPhdResearchForSelection() {
  const applicationId = qs("#phdResearchApplication")?.value || state.phd.selectedApplicationId;
  if (!applicationId) {
    renderPhdResearch("", []);
    return;
  }
  const application = state.phd.applications.find(item => item.id === applicationId);
  const payload = await api(`/api/phd/applications/${encodeURIComponent(applicationId)}/sources`);
  renderPhdResearch(application?.research_summary || "", payload.sources || []);
}

function renderPhdResearch(summary, sources) {
  if (qs("#phdResearchSummary")) qs("#phdResearchSummary").innerHTML = summary ? renderMarkdown(summary) : emptyState("Research findings will appear here.");
  if (qs("#phdResearchSourceCount")) qs("#phdResearchSourceCount").textContent = String(sources.length);
  const container = qs("#phdResearchSources");
  if (container) {
    container.innerHTML = sources.length ? sources.map(source => `
      <div class="phd-record"><div><strong>${escapeHtml(source.title)}</strong><p>${escapeHtml(source.source_type)} · ${escapeHtml(source.snippet || "Saved public source")}</p></div>
      <a class="btn btn-mini btn-outline" href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">Open</a></div>`).join("") : emptyState("No public sources saved for this position.");
  }
}

function initPhdWorkspace() {
  qs("#phdTabStudio")?.addEventListener("click", () => setPhdTab("studio"));
  qs("#phdTabData")?.addEventListener("click", () => setPhdTab("data"));
  qs("#phdTabResearch")?.addEventListener("click", () => setPhdTab("research"));
  qs("#phdTabLibrary")?.addEventListener("click", () => setPhdTab("library"));
  qs("#refreshPhdWorkspace")?.addEventListener("click", loadPhdWorkspace);
  qs("#savePhdProfile")?.addEventListener("click", savePhdProfile);
  qs("#uploadPhdDocument")?.addEventListener("click", uploadPhdDocument);
  qs("#phdDocumentFile")?.addEventListener("change", event => {
    const file = event.target.files?.[0];
    if (file && qs("#phdDocumentLabel") && !qs("#phdDocumentLabel").value.trim()) qs("#phdDocumentLabel").value = file.name.replace(/\.[^.]+$/, "");
  });
  qs("#savePhdApplication")?.addEventListener("click", () => savePhdApplication());
  qs("#newPhdApplication")?.addEventListener("click", clearPhdApplicationForm);
  qs("#deletePhdApplication")?.addEventListener("click", deletePhdApplication);
  qs("#phdApplicationSelect")?.addEventListener("change", event => selectPhdApplication(event.target.value));
  qs("#phdResearchApplication")?.addEventListener("change", event => selectPhdApplication(event.target.value));
  qs("#phdLibraryApplication")?.addEventListener("change", renderPhdArtifactLibrary);
  qs("#selectAllPhdSources")?.addEventListener("click", () => {
    const inputs = qsa("[data-phd-source-id]");
    const shouldSelect = inputs.some(input => !input.checked);
    inputs.forEach(input => { input.checked = shouldSelect; });
  });
  qs("#phdAutoDetectOutputs")?.addEventListener("change", event => {
    if (event.target.checked) qsa("[data-phd-artifact-type]").forEach(input => { input.checked = false; });
  });
  qsa("[data-phd-artifact-type]").forEach(input => input.addEventListener("change", event => {
    if (event.target.checked && qs("#phdAutoDetectOutputs")) qs("#phdAutoDetectOutputs").checked = false;
    if (!qsa("[data-phd-artifact-type]:checked").length && qs("#phdAutoDetectOutputs")) qs("#phdAutoDetectOutputs").checked = true;
  }));
  qs("#generatePhdBundle")?.addEventListener("click", generatePhdBundle);
  qs("#exportPhdBundle")?.addEventListener("click", exportPhdBundle);
  qs("#phdPreviewDocx")?.addEventListener("click", () => exportPhdArtifact("docx"));
  qs("#phdPreviewPdf")?.addEventListener("click", () => exportPhdArtifact("pdf"));
  qs("#phdPreviewMarkdown")?.addEventListener("click", () => exportPhdArtifact("md"));
  qs("#runPhdResearch")?.addEventListener("click", runPhdResearch);
  qs("#savePhdArtifactRevision")?.addEventListener("click", savePhdArtifactRevision);
  qs("#reviewPhdAuthenticity")?.addEventListener("click", reviewPhdAuthenticity);
  qs("#deletePhdArtifact")?.addEventListener("click", deletePhdArtifact);
}

// =============================================
// Command Palette
// =============================================

let paletteIndex = -1;

function openCommandPalette() { openPalette(); }

function openPalette() {
  const overlay = qs("#commandPalette");
  overlay?.classList.add("open");
  const input = qs("#paletteInput");
  if (input) { input.value = ""; input.focus(); }
  paletteIndex = -1;
  renderPaletteResults("");
}

function closePalette() {
  qs("#commandPalette")?.classList.remove("open");
}

function renderPaletteResults(query) {
  const filtered = query
    ? PALETTE_COMMANDS.filter(c => c.label.toLowerCase().includes(query.toLowerCase()) || c.group.toLowerCase().includes(query.toLowerCase()))
    : PALETTE_COMMANDS;
  const results = qs("#paletteResults");
  if (!results) return;
  results.innerHTML = filtered.map((c, i) =>
    `<div class="palette-item ${i === paletteIndex ? "focused" : ""}" data-palette-idx="${i}" role="option" aria-selected="${i === paletteIndex}">
       <span class="pi-icon">${c.icon}</span>
       <span class="pi-label">${escapeHtml(c.label)}</span>
       <span class="pi-group">${escapeHtml(c.group)}</span>
     </div>`
  ).join("");
  qsa(".palette-item").forEach(item =>
    item.addEventListener("click", () => {
      const idx = parseInt(item.dataset.paletteIdx, 10);
      if (filtered[idx]) { filtered[idx].action(); closePalette(); }
    })
  );
}

function initPalette() {
  qs("#openPalette")?.addEventListener("click", openPalette);
  qs("#globalSearch")?.addEventListener("focus", openPalette);
  qs("#commandPalette")?.addEventListener("click", e => { if (e.target === qs("#commandPalette")) closePalette(); });

  const input = qs("#paletteInput");
  if (input) {
    input.addEventListener("input", e => { paletteIndex = -1; renderPaletteResults(e.target.value); });
    input.addEventListener("keydown", e => {
      const items = qsa(".palette-item");
      if (e.key === "ArrowDown") {
        e.preventDefault();
        paletteIndex = Math.min(paletteIndex + 1, items.length - 1);
        renderPaletteResults(input.value);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        paletteIndex = Math.max(paletteIndex - 1, 0);
        renderPaletteResults(input.value);
      } else if (e.key === "Enter") {
        if (paletteIndex >= 0 && items[paletteIndex]) items[paletteIndex].click();
      } else if (e.key === "Escape") {
        closePalette();
      }
    });
  }
}

// =============================================
// Global keyboard shortcuts
// =============================================

function initKeyboardShortcuts() {
  document.addEventListener("keydown", e => {
    // Ctrl/Cmd+K — command palette
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      const overlay = qs("#commandPalette");
      if (overlay?.classList.contains("open")) closePalette(); else openPalette();
      return;
    }
    // Escape — close palette
    if (e.key === "Escape") {
      closePalette();
      return;
    }
    // Skip if typing in an input
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    // Single-key shortcuts
    if (e.key === "g" && !e.ctrlKey) { e.preventDefault(); setView("chat"); }
  });
}

// =============================================
// Event delegation for nav
// =============================================

function initNavigation() {
  document.addEventListener("click", e => {
    const codebaseFile = e.target.closest("[data-codebase-file]");
    if (codebaseFile) { openCodebaseFile(codebaseFile.dataset.codebaseFile); return; }
    const codebaseApproval = e.target.closest("[data-codebase-approval]");
    if (codebaseApproval) { resolveCodebaseApproval(codebaseApproval.dataset.codebaseApproval, codebaseApproval.dataset.approved === "true"); return; }
    const codebaseDetail = e.target.closest("[data-codebase-detail]");
    if (codebaseDetail) { showCodebaseDetail(codebaseDetail.dataset.codebaseDetail); return; }
    const codebaseOutput = e.target.closest("[data-codebase-output]");
    if (codebaseOutput) { showCodebaseOutput(codebaseOutput.dataset.codebaseOutput); return; }
    const sciencePrompt = e.target.closest("[data-science-prompt]");
    if (sciencePrompt) {
      const goal = qs("#scienceGoal");
      if (goal) goal.value = sciencePrompt.dataset.sciencePrompt;
      runScienceWorkflow();
      return;
    }
    const scienceAction = e.target.closest("[data-science-action]");
    if (scienceAction) {
      runScienceAction(scienceAction.dataset.scienceAction).catch(error => {
        renderScienceSteps(["Prepare input", "Run local operation", "Verify output", "Render result"], 0, error.message);
        toast("Scientific operation failed", error.message);
      });
      return;
    }
    const scienceCapability = e.target.closest("[data-science-capability]");
    if (scienceCapability) {
      setScienceSection(scienceCapability.dataset.scienceCapability);
      return;
    }
    const knowledgeDocument = e.target.closest("[data-knowledge-document]");
    if (knowledgeDocument) {
      selectKnowledgeDocument(knowledgeDocument.dataset.knowledgeDocument);
      return;
    }
    const knowledgeTag = e.target.closest("[data-knowledge-tag]");
    if (knowledgeTag) {
      const query = qs("#knowledgeWorkspaceQuery");
      if (query) query.value = knowledgeTag.dataset.knowledgeTag || "";
      setView("knowledge");
      runKnowledgeWorkspace();
      return;
    }
    const knowledgeTimeline = e.target.closest("[data-knowledge-timeline]");
    if (knowledgeTimeline) {
      const documentId = knowledgeTimeline.dataset.knowledgeTimeline;
      if (documentId) selectKnowledgeDocument(documentId);
      return;
    }
    const viewLink = e.target.closest("[data-view-link]");
    if (viewLink) { setView(viewLink.dataset.viewLink); if (viewLink.dataset.scienceSection) setScienceSection(viewLink.dataset.scienceSection); return; }
    const viewJump = e.target.closest("[data-view-jump]");
    if (viewJump) { setView(viewJump.dataset.viewJump); return; }
    const command = e.target.closest("[data-command]");
    if (command) { runCommand(command.dataset.command); return; }
    const adapterButton = e.target.closest("[data-adapter-activate]");
    if (adapterButton) { activateAdapter(adapterButton.dataset.adapterActivate); return; }
  });
  qs("#collapseSidebar")?.addEventListener("click", () => {
    const collapsed = document.body.classList.toggle("sidebar-collapsed");
    const button = qs("#collapseSidebar");
    if (button) {
      button.setAttribute("aria-expanded", String(!collapsed));
      button.setAttribute("aria-label", collapsed ? "Expand sidebar" : "Collapse sidebar");
      button.title = collapsed ? "Expand sidebar" : "Collapse sidebar";
    }
    try { localStorage.setItem("oa-sidebar-collapsed", String(collapsed)); } catch (_) {}
  });
  qsa(".nav-group-toggle").forEach(toggle => {
    toggle.addEventListener("click", () => toggleNavGroup(toggle));
  });
}

async function runCommand(command) {
  if (command === "search-memory") {
    await searchChatMemory();
    return;
  }
  if (command === "export-chat") {
    await exportCurrentChat();
    return;
  }
  toast("Command unavailable", command || "unknown");
}

function toggleNavGroup(toggle) {
  const collapsed = toggle.classList.toggle("collapsed");
  toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
  let node = toggle.nextElementSibling;
  while (node) {
    const isGroup = node.classList.contains("nav-group-toggle");
    const belongsToGroup = node.dataset.navParent === toggle.dataset.navGroup;
    if (isGroup && !belongsToGroup) break;
    if (node.classList.contains("nav-link") || (isGroup && belongsToGroup)) {
      node.classList.toggle("nav-group-hidden", collapsed);
    }
    node = node.nextElementSibling;
  }
}

// =============================================
// Refresh helpers
// =============================================

async function refreshWorkspace() {
  await Promise.allSettled([loadAudit(), loadConversations()]);
}

async function refreshAll() {
  await Promise.allSettled([
    loadDashboardTelemetry(), loadAgents(), loadProviders(), loadAudit(),
    loadConversations(), loadWorkflows(), loadCommunicationHub(),
    loadIntegrationStatus(), loadPersonalOsStatus(), loadCalendarEvents(), loadCalendarSyncStatus(), loadTwinSuggestions(), loadVaultEntries(),
    loadScraperJobs(), loadFinetuneStatus(), loadModelJobs(), loadOperationalModules(),
    loadRuntimeFabric(),
  ]);
}

function exposeInlineHandlers() {
  Object.assign(window, {
    loadTwinProfile,
    loadTwinPatterns,
    loadTwinSuggestions,
    loadTwinGoals,
    predictIntent,
    consolidateMemory,
    createTwinGoal,
    loadVaultEntries,
    storeVaultSecret,
    retrieveVaultSecret,
    loadScraperJobs,
    scrapeUrl,
    scheduleUrl,
    loadFinetuneStatus,
    loadAdapters,
    createFtDataset,
    addFtExample,
    startFtJob,
    activateAdapter,
  });
}

// =============================================
// Bootstrap
// =============================================

async function init() {
  try { document.body.classList.toggle("sidebar-collapsed", localStorage.getItem("oa-sidebar-collapsed") === "true"); } catch (_) {}
  initTheme();
  initNavigation();
  initPalette();
  initKeyboardShortcuts();
  initVoice();
  initResearch();
  initUpload();
  initKnowledgeWorkspace();
  initSettings();
  exposeInlineHandlers();

  // Bind buttons
  qs("#projectInput")?.addEventListener("change", refreshProjectContext);
  qs("#projectInput")?.addEventListener("keydown", event => {
    if (event.key === "Enter") {
      event.preventDefault();
      refreshProjectContext();
    }
  });
  qs("#sendChat")?.addEventListener("click", sendChat);
  qs("#newConversationButton")?.addEventListener("click", newConversation);
  qs("#chatAttachButton")?.addEventListener("click", () => qs("#chatAttachmentInput")?.click());
  qs("#chatAttachmentInput")?.addEventListener("change", event => uploadChatAttachments(event.target.files));
  qs("#newCodebaseProject")?.addEventListener("click", () => pickCodebaseFolder(true));
  qs("#refreshCodebase")?.addEventListener("click", loadCodebaseProjects);
  qs("#createCodebaseProject")?.addEventListener("click", createCodebaseProject);
  qs("#codebaseFolderPicker")?.addEventListener("click", () => pickCodebaseFolder(true));
  qs("#codebaseProjectSelect")?.addEventListener("change", event => selectCodebaseProject(event.target.value));
  qs("#scanCodebase")?.addEventListener("click", scanCodebaseProject);
  qs("#newCodebaseFile")?.addEventListener("click", () => mutateCodebasePath("create_file"));
  qs("#newCodebaseFolder")?.addEventListener("click", () => mutateCodebasePath("create_directory"));
  qs("#renameCodebasePath")?.addEventListener("click", () => mutateCodebasePath("rename"));
  qs("#deleteCodebasePath")?.addEventListener("click", () => mutateCodebasePath("delete"));
  qs("#saveCodebaseFile")?.addEventListener("click", saveCodebaseFile);
  qs("#codebaseEditor")?.addEventListener("input", () => { rememberCurrentCodebaseBuffer(); renderCodebaseTabs(); updateCodebaseLineNumbers(); });
  qs("#codebaseEditor")?.addEventListener("scroll", updateCodebaseLineNumbers);
  qs("#codebaseEditor")?.addEventListener("click", updateCodebaseCursorStatus);
  qs("#codebaseEditor")?.addEventListener("keyup", updateCodebaseCursorStatus);
  qs("#codebaseEditor")?.addEventListener("select", updateCodebaseCursorStatus);
  qs("#codebaseEditor")?.addEventListener("keydown", event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveCodebaseFile();
    } else if (event.key === "Tab") {
      event.preventDefault();
      const editor = event.currentTarget;
      const start = editor.selectionStart;
      editor.setRangeText("  ", start, editor.selectionEnd, "end");
      updateCodebaseLineNumbers();
    }
  });
  qs("#startCodebaseRun")?.addEventListener("click", startCodebaseRun);
  qs("#cancelCodebaseRun")?.addEventListener("click", cancelCodebaseRun);
  qs("#resumeCodebaseRun")?.addEventListener("click", resumeCodebaseRun);
  qs("#restoreCodebaseSnapshot")?.addEventListener("click", restoreCodebaseSnapshot);
  qs("#codebaseRunSelect")?.addEventListener("change", event => event.target.value && loadCodebaseRun(event.target.value));
  qs("#codebaseGoal")?.addEventListener("keydown", event => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !state.codebaseRunBusy) {
      event.preventDefault();
      startCodebaseRun();
    }
  });
  qs("#runCodebaseTests")?.addEventListener("click", runCodebaseTests);
  qs("#runCodebaseCommand")?.addEventListener("click", runCodebaseCommand);
  qs("#codebaseCommand")?.addEventListener("keydown", event => { if (event.key === "Enter") { event.preventDefault(); runCodebaseCommand(); } });
  qs("#inspectCodebaseBrowser")?.addEventListener("click", inspectCodebaseBrowser);
  qs("#loadCodebaseStatus")?.addEventListener("click", loadCodebaseStatus);
  qs("#loadCodebaseDiff")?.addEventListener("click", loadCodebaseDiff);
  qs("#codebaseCopilotFeed")?.addEventListener("click", event => {
    if (event.target.closest("[data-codeq-copy-prompt]")) copyCodeqPrompt();
    if (event.target.closest("[data-codeq-edit-prompt]")) editCodeqPrompt();
  });
  qs("#stopChat")?.addEventListener("click", () => stopActiveChat("Response stopped by user."));
  qs("#testChatModel")?.addEventListener("click", testChatModel);
  qs("#conversationSort")?.addEventListener("change", loadConversations);
  qs("#showArchivedConversations")?.addEventListener("click", () => {
    state.showArchivedConversations = !state.showArchivedConversations;
    closeConversationActionMenu();
    loadConversations();
  });
  qs("#runConversationBulk")?.addEventListener("click", runConversationBulk);
  qs("#compactConversation")?.addEventListener("click", compactCurrentConversation);
  qs("#showConversationBranches")?.addEventListener("click", showCurrentConversationBranches);
  qs("#chatInput")?.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } });
  document.addEventListener("click", event => {
    const menu = qs("#conversationActionMenu");
    if (menu && !menu.hidden && !menu.contains(event.target) && !event.target.closest("[data-conversation-menu]")) closeConversationActionMenu();
  });
  window.addEventListener("resize", closeConversationActionMenu);
  document.addEventListener("keydown", event => { if (event.key === "Escape") closeConversationActionMenu(); });
  qs("#researchQuery")?.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); startResearch(); } });
  qs("#streamToggle")?.addEventListener("click", e => {
    state.streamingEnabled = !state.streamingEnabled;
    e.currentTarget.textContent = state.streamingEnabled ? "⚡ Stream" : "⚡ Batch";
    e.currentTarget.classList.toggle("btn-primary", state.streamingEnabled);
  });
  qs("#refreshAll")?.addEventListener("click", refreshAll);
  qs("#runAgent")?.addEventListener("click", () => runAgent());
  qs("#runCheckedAgent")?.addEventListener("click", () => runAgent("/api/agents/run-checked"));
  qs("#runParallelAgents")?.addEventListener("click", runParallelAgents);
  qs("#newAgentRole")?.addEventListener("click", () => openAgentRoleDialog());
  qs("#editAgentRole")?.addEventListener("click", () => {
    const selected = qs("#agentSelect")?.value || "";
    const role = state.customAgents.find(item => item.id === selected);
    if (!role) { toast("Built-in roles are read-only", "Create a custom role to define your own instructions."); return; }
    openAgentRoleDialog(role);
  });
  qs("#closeAgentRole")?.addEventListener("click", () => qs("#agentRoleDialog")?.close());
  qs("#cancelAgentRole")?.addEventListener("click", () => qs("#agentRoleDialog")?.close());
  qs("#saveAgentRole")?.addEventListener("click", saveAgentRole);
  qs("#deleteAgentRole")?.addEventListener("click", deleteAgentRole);
  qs("#createWorkspaceItem")?.addEventListener("click", createWorkspaceItem);
  qs("#workspaceSearch")?.addEventListener("input", loadWorkspaceItems);
  qs("#loadWorkspaceCanvas")?.addEventListener("click", async () => {
    try { const d = await api(`/api/workspace/canvas?project=${encodeURIComponent(getProject())}`); setOutput("#workspaceOutput", d); } catch(e) {}
  });
  qs("#ingestKnowledge")?.addEventListener("click", ingestKnowledge);
  qs("#addSemanticMemory")?.addEventListener("click", async () => {
    const title = qs("#knowledgeTitle")?.value.trim();
    const content = qs("#knowledgeContent")?.value.trim();
    if (!title || !content) { toast("Title and content required"); return; }
    try {
      const d = await api("/api/semantic/memories", { method: "POST", body: JSON.stringify({ title, content, project: getProject(), memory_type: "semantic", tags: [] }) });
      renderKnowledgeIngest({ title: d.memory?.title || title, chunk_count: 1, source: "semantic-memory", tags: d.memory?.tags || [] }, "Semantic memory saved");
      toast("Semantic memory saved");
    } catch(e) { toast("Memory error", e.message); }
  });
  qs("#searchKnowledge")?.addEventListener("click", searchKnowledge);
  qs("#synthesizeKnowledge")?.addEventListener("click", synthesizeKnowledge);
  qs("#runAdvancedRag")?.addEventListener("click", runAdvancedRag);
  qs("#showKnowledgeGraph")?.addEventListener("click", async () => {
    try { const d = await api(`/api/knowledge/graph?project=${encodeURIComponent(getProject())}`); renderKnowledgeGraph(d); } catch(e) {}
  });
  qs("#createWorkflow")?.addEventListener("click", renderWorkflowAuthoring);
  qs("#saveWorkflow")?.addEventListener("click", createWorkflowFromEditor);
  qs("#runDueWorkflows")?.addEventListener("click", async () => {
    try { const d = await api("/api/workflows/run-due", { method: "POST", body: JSON.stringify({ project: getProject() }) }); setOutput("#workflowOutput", d); } catch(e) { toast("Run due error", e.message); }
  });
  qs("#loadRuntimeTools")?.addEventListener("click", loadRuntimeTools);
  qs("#runRuntimeBrowserTest")?.addEventListener("click", runRuntimeBrowserTest);
  qs("#createTask")?.addEventListener("click", createTask);
  qs("#createGoal")?.addEventListener("click", createGoal);
  qs("#dailyBrief")?.addEventListener("click", getDailyBrief);
  qs("#meetingSummary")?.addEventListener("click", async () => {
    const body = qs("#assistantBody")?.value.trim() || "Meeting";
    try { const d = await api("/api/assistant/meeting/summary", { method: "POST", body: JSON.stringify({ transcript: body, project: getProject() }) }); setOutput("#assistantOutput", d); } catch(e) { toast("Meeting summary error", e.message); }
  });
  qs("#importMessage")?.addEventListener("click", importMessage);
  qs("#loadInbox")?.addEventListener("click", loadCommunicationFeed);
  qs("#loadCalendar")?.addEventListener("click", async () => {
    try { const d = await api(`/api/communication/calendar?project=${encodeURIComponent(getProject())}&limit=10`); setOutput("#commOutput", d); } catch(e) {}
  });
  qs("#createOauthAccount")?.addEventListener("click", createOauthAccount);
  qs("#refreshOauthAccounts")?.addEventListener("click", loadCommunicationOAuth);
  qs("#buildOauthUrl")?.addEventListener("click", buildOauthUrl);
  qs("#refreshOauthToken")?.addEventListener("click", refreshOauthToken);
  qs("#syncCommunicationProvider")?.addEventListener("click", syncCommunicationProvider);
  qs("#refreshCommunicationSync")?.addEventListener("click", loadCommunicationSyncStatus);
  qs("#discoverModels")?.addEventListener("click", discoverModels);
  qs("#refreshModels")?.addEventListener("click", loadProviders);
  qs("#loadSmoothRuntime")?.addEventListener("click", () => loadSmoothRuntime("chat"));
  qs("#planChatRuntime")?.addEventListener("click", () => loadSmoothRuntime("chat"));
  qs("#planResearchRuntime")?.addEventListener("click", () => loadSmoothRuntime("research"));
  qs("#planVoiceRuntime")?.addEventListener("click", () => loadSmoothRuntime("voice"));
  qs("#autoQuantizeRegisteredModel")?.addEventListener("click", autoQuantizeRegisteredModel);
  qs("#loadPrivacyReport")?.addEventListener("click", loadPrivacyReport);
  qs("#identityRegister")?.addEventListener("click", registerIdentityAccount);
  qs("#identityLogin")?.addEventListener("click", loginIdentityAccount);
  qs("#identityPasskeyLogin")?.addEventListener("click", loginWithIdentityPasskey);
  qs("#identityLogout")?.addEventListener("click", logoutIdentityAccount);
  qs("#identityAddPasskey")?.addEventListener("click", addIdentityPasskey);
  qs("#identityRotateRecovery")?.addEventListener("click", rotateIdentityRecoveryCodes);
  qs("#identityTrustDevice")?.addEventListener("click", trustCurrentIdentityDevice);
  qs("#identityRefresh")?.addEventListener("click", loadIdentityAdministration);
  qs("#identitySaveFederation")?.addEventListener("click", saveIdentityFederationPolicy);
  qs("#loadAuditChain")?.addEventListener("click", async () => {
    try { const d = await api(`/api/audit?project=${encodeURIComponent(getProject())}&limit=30`); setOutput("#privacyOutput", d); } catch(e) {}
  });
  qs("#loadDevSchema")?.addEventListener("click", loadDeveloperFeed);
  qs("#createApiKey")?.addEventListener("click", createApiKey);
  qs("#saveSkill")?.addEventListener("click", saveSkillEditor);
  qs("#testSkill")?.addEventListener("click", () => runSkillEditorAction("test"));
  qs("#previewSkillPermissions")?.addEventListener("click", () => runSkillEditorAction("permissions"));
  qs("#scanSkillDependencies")?.addEventListener("click", () => runSkillEditorAction("dependency-scan"));
  qs("#packageSkill")?.addEventListener("click", () => runSkillEditorAction("package"));
  qs("#exportSkill")?.addEventListener("click", () => runSkillEditorAction("export"));
  qs("#sandboxSkill")?.addEventListener("click", () => runSkillEditorAction("sandbox"));
  qs("#publishSkill")?.addEventListener("click", () => runSkillEditorAction("publish"));
  qs("#loadIntegrationStatus")?.addEventListener("click", loadIntegrationStatus);
  qs("#createPairingCode")?.addEventListener("click", createPairingCode);
  qs("#runIntegrationDiagnostics")?.addEventListener("click", runIntegrationDiagnostics);
  qs("#createIntegrationBackup")?.addEventListener("click", createIntegrationBackup);
  qs("#runSafePythonCheck")?.addEventListener("click", runSafePythonCheck);
  qs("#saveUiPreference")?.addEventListener("click", saveUiPreference);
  qs("#loadPersonalOsStatus")?.addEventListener("click", loadPersonalOsStatus);
  qs("#savePersonalPreset")?.addEventListener("click", savePersonalPreset);
  qs("#savePersonalSignature")?.addEventListener("click", savePersonalSignature);
  qs("#savePersonalDraft")?.addEventListener("click", savePersonalDraft);
  qs("#savePersonalNote")?.addEventListener("click", savePersonalNote);
  qs("#addGalleryAsset")?.addEventListener("click", addGalleryAsset);
  qs("#startCompareRun")?.addEventListener("click", startCompareRun);
  qs("#exportPersonalOs")?.addEventListener("click", exportPersonalOs);
  qs("#refreshModelCookbook")?.addEventListener("click", loadModelCookbook);
  qs("#certifyModelEndpoint")?.addEventListener("click", certifyModelEndpoint);
  qs("#createModelHost")?.addEventListener("click", createModelHost);
  qs("#planModelServe")?.addEventListener("click", planModelServe);
  qs("#checkHardwareFit")?.addEventListener("click", checkHardwareFit);
  qs("#refreshInferenceEngine")?.addEventListener("click", loadInferenceEngine);
  qs("#warmInferenceEngine")?.addEventListener("click", () => runInferenceAction("/api/inference/warmup", { provider: "auto" }, "Local model warmed"));
  qs("#benchmarkInferenceEngine")?.addEventListener("click", () => runInferenceAction("/api/inference/benchmark", { iterations: 250 }, "Inference benchmark complete"));
  qs("#benchmarkRagEngine")?.addEventListener("click", () => runInferenceAction("/api/rag/benchmark", { iterations: 8 }, "RAG benchmark complete"));
  qs("#clearInferenceCache")?.addEventListener("click", () => runInferenceAction("/api/inference/cache/clear", {}, "Inference cache cleared"));
  qs("#refreshUltraModelEngine")?.addEventListener("click", loadUltraModelEngine);
  qs("#refreshAdaptiveEngine")?.addEventListener("click", () => loadAdaptiveEngine(true));
  qs("#inspectAdaptiveModel")?.addEventListener("click", inspectAdaptiveModel);
  qs("#compileAdaptiveModel")?.addEventListener("click", compileAdaptiveModel);
  qs("#benchmarkAdaptiveDevice")?.addEventListener("click", benchmarkAdaptiveDevice);
  qs("#autotuneAdaptiveDevice")?.addEventListener("click", autotuneAdaptiveDevice);
  qs("#calibrateAdaptiveModel")?.addEventListener("click", calibrateAdaptiveModel);
  qs("#loadAdaptiveResident")?.addEventListener("click", loadAdaptiveResident);
  qs("#unloadAdaptiveResident")?.addEventListener("click", unloadAdaptiveResident);
  qs("#stopAdaptiveRuntime")?.addEventListener("click", stopAdaptiveRuntime);
  qs("#engineRoleSelect")?.addEventListener("change", e => selectEngineRole(e.target.value));
  qs("#engineRoleModelSelect")?.addEventListener("change", e => {
    if (qs("#engineRoleModelCustom")) qs("#engineRoleModelCustom").value = e.target.value;
  });
  qs("#engineRoleSummary")?.addEventListener("click", e => {
    const target = e.target.closest("[data-engine-role]");
    if (target) selectEngineRole(target.dataset.engineRole);
  });
  qs("#saveEngineRole")?.addEventListener("click", saveEngineRole);
  qs("#refreshEngineRoles")?.addEventListener("click", () => loadEngineWorkspace(true));
  qs("#engineOrchestrationMode")?.addEventListener("change", updateEngineModeVisibility);
  qs("#runEngineCollaboration")?.addEventListener("click", runEngineCollaboration);
  qs("#refreshEngineRuns")?.addEventListener("click", () => loadEngineWorkspace());
  qs("#runScienceWorkflow")?.addEventListener("click", runScienceWorkflow);
  qs("#refreshScienceStatus")?.addEventListener("click", loadScienceStatus);
  qs("#registerUltraModel")?.addEventListener("click", registerUltraModel);
  qs("#planUltraModel")?.addEventListener("click", planUltraModel);
  qs("#calibrateUltraModel")?.addEventListener("click", calibrateUltraModel);
  qs("#launchUltraModel")?.addEventListener("click", launchUltraModel);
  qs("#stopUltraModel")?.addEventListener("click", stopUltraModel);
  qs("#ultraManifestSelect")?.addEventListener("change", (event) => {
    state.ultraModel.manifestId = event.target.value || "";
    const manifest = (state.ultraModel.manifests || []).find(item => item.id === state.ultraModel.manifestId);
    if (manifest) {
      qs("#ultraModelName").value = manifest.name;
      qs("#ultraModelArchitecture").value = manifest.architecture;
      qs("#ultraModelParams").value = manifest.parameter_count_b;
      qs("#ultraModelActiveParams").value = manifest.active_parameter_count_b;
      qs("#ultraModelBits").value = manifest.quantization_bits;
      qs("#ultraModelPath").value = manifest.shard_paths?.[0] || "";
    }
  });
  qs("#ultraModelArchitecture")?.addEventListener("change", (event) => {
    if (event.target.value === "dense") qs("#ultraModelActiveParams").value = qs("#ultraModelParams")?.value || "";
  });
  qs("#refreshMcpRuntime")?.addEventListener("click", loadMcpRuntime);
  qs("#registerMcpServer")?.addEventListener("click", registerMcpServer);
  qs("#mcpRuntimeFeed")?.addEventListener("click", event => {
    const button = event.target.closest("[data-mcp-invoke]");
    if (button) invokeMcpTool(button.dataset.mcpInvoke);
  });
  qs("#refreshRuntimeFabric")?.addEventListener("click", loadRuntimeFabric);
  qs("#planRuntimeFabric")?.addEventListener("click", planRuntimeFabric);
  qs("#refreshTerminalRuns")?.addEventListener("click", loadTerminalRuns);
  qs("#runTerminalVersion")?.addEventListener("click", runTerminalVersion);
  qs("#refreshScheduler")?.addEventListener("click", loadSchedulerStatus);
  qs("#createSchedulerTask")?.addEventListener("click", createSchedulerTask);
  qs("#addSchedulerOutput")?.addEventListener("click", addSchedulerOutput);
  qs("#runDueSchedulerTasks")?.addEventListener("click", runDueSchedulerTasks);
  qs("#refreshImageStudio")?.addEventListener("click", loadImageStudio);
  qs("#uploadTinyImage")?.addEventListener("click", uploadTinyImage);
  qs("#batchTagImages")?.addEventListener("click", batchTagImages);
  qs("#refreshAgentBridge")?.addEventListener("click", loadAgentBridgeStatus);
  qs("#writeBridgeMemory")?.addEventListener("click", writeBridgeMemory);
  qs("#createBridgeTask")?.addEventListener("click", createBridgeTask);
  qs("#loadProviderCertification")?.addEventListener("click", () => loadProviderCertification(true));
  qs("#recordOauthCallbackDemo")?.addEventListener("click", recordOauthCallbackDemo);
  qs("#importDeveloperDocument")?.addEventListener("click", importDeveloperDocument);
  qs("#providerSelect")?.addEventListener("change", updateProviderWarning);
  qs("#refreshCalendarEvents")?.addEventListener("click", loadCalendarEvents);
  qs("#exportCalendarIcal")?.addEventListener("click", exportIcal);
  qs("#createCalendarEventBtn")?.addEventListener("click", createCalendarEvent);
  qs("#findFreeSlotsBtn")?.addEventListener("click", loadFreeSlots);
  qs("#registerCalendarProvider")?.addEventListener("click", registerCalendarProvider);
  qs("#refreshCalendarSync")?.addEventListener("click", refreshCalendarSync);
  qs("#runCalendarPull")?.addEventListener("click", () => runCalendarSync("pull"));
  qs("#runCalendarPush")?.addEventListener("click", () => runCalendarSync("push"));
  qs("#runCalendarBoth")?.addEventListener("click", () => runCalendarSync("bidirectional"));

  // Theme buttons in topbar
  qsa("[data-theme-select]").forEach(btn =>
    btn.addEventListener("click", () => applyTheme(btn.dataset.themeSelect))
  );
  qsa("[data-dashboard-window]").forEach(button => button.addEventListener("click", () => {
    state.dashboardUi.resourceWindow = Number(button.dataset.dashboardWindow) || 36;
    qsa("[data-dashboard-window]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
    renderDashboardResourceChart();
  }));
  qsa("[data-provider-filter]").forEach(button => button.addEventListener("click", () => {
    state.dashboardUi.providerFilter = button.dataset.providerFilter || "all";
    qsa("[data-provider-filter]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
    renderDashboardSystem();
  }));
  qsa("[data-incident-filter]").forEach(button => button.addEventListener("click", () => {
    state.dashboardUi.incidentFilter = button.dataset.incidentFilter || "all";
    qsa("[data-incident-filter]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
    renderDashboardAudit(state.auditEvents);
  }));
  qs("#dashboardPauseTelemetry")?.addEventListener("click", event => {
    state.dashboardUi.paused = !state.dashboardUi.paused;
    event.currentTarget.setAttribute("aria-pressed", String(state.dashboardUi.paused));
    event.currentTarget.textContent = state.dashboardUi.paused ? "Resume live" : "Pause live";
    event.currentTarget.setAttribute("aria-label", state.dashboardUi.paused ? "Resume dashboard live updates" : "Pause dashboard live updates");
  });

  // Global search redirect to palette
  qs("#globalSearch")?.addEventListener("keydown", e => {
    if (e.key === "Enter") {
      const q = e.target.value.trim();
      if (q) { closePalette(); /* future: run semantic search */ }
    }
  });

  // Initial load
  await refreshAll();
  window.setInterval(() => {
    if (!document.hidden && state.activeView === "dashboard" && !state.dashboardUi.paused) {
      Promise.allSettled([loadDashboardTelemetry(), loadAudit()]).catch(() => {});
    }
  }, 5000);
  window.setInterval(() => {
    if (!document.hidden && state.activeView !== "dashboard") {
      Promise.allSettled([loadDashboardTelemetry(), loadAudit(), loadProviders()]).catch(() => {});
    }
  }, 30000);

  // Restore initial view from URL params if any
  const urlParams = new URLSearchParams(window.location.search);
  const initView = urlParams.get("view");
  if (initView) setView(initView);

  // PWA install prompt (deferred — available after user gesture)
  let deferredInstall = null;
  window.addEventListener("beforeinstallprompt", e => { e.preventDefault(); deferredInstall = e; });
}

document.addEventListener("DOMContentLoaded", init);

// =============================================
// Scientific AI
// =============================================

async function loadScienceStatus() {
  try {
    const payload = await api("/api/science/status");
    state.science.status = payload;
    const badge = qs("#scienceBadge");
    if (badge) {
      badge.textContent = payload.status === "ready" ? "Ready" : "Needs attention";
      badge.className = `badge ${payload.status === "ready" ? "badge-success" : "badge-error"}`;
    }
    const grid = qs("#scienceWorkflowGrid");
    if (grid) {
      const workflows = payload.workflows || [];
      grid.innerHTML = workflows.map(item => `<article class="metric-card"><span>${escapeHtml(item.label)}</span><strong>Local</strong></article>`).join("");
    }
    const capabilities = [
      ["agents", "Scientific Agents", "Planning, tool calling, execution, analysis, and repair-ready workflows."],
      ["computing", "Scientific Computing", "Solvers, surrogates, operator-learning interfaces, and reproducible runs."],
      ["engineering", "Engineering Optimisation", "Design variables, constraints, objective scoring, and ranked candidates."],
      ["multimodal", "Multimodal Science", "Text, sensor tables, images, simulation output, and evidence fusion."],
      ["verification", "Scientific Verification", "Numeric, residual, unit, equation, and constraint checks."],
      ["materials", "Materials Discovery", "Property prediction, candidate ranking, and next-experiment selection."],
    ];
    const capabilityGrid = qs("#scienceCapabilityGrid");
    if (capabilityGrid) capabilityGrid.innerHTML = capabilities.map(([id, label, text]) => `<button class="science-capability" data-science-capability="${id}" type="button"><strong>${label}</strong><span>${text}</span></button>`).join("");
    setScienceSection(state.science.section || "agents");
  } catch (error) {
    const badge = qs("#scienceBadge");
    if (badge) { badge.textContent = "Unavailable"; badge.className = "badge badge-error"; }
    console.warn("scienceStatus", error);
  }
}

async function runScienceWorkflow() {
  const goal = qs("#scienceGoal")?.value.trim() || "Plan a reproducible scientific investigation";
  const output = qs("#scienceResult");
  const button = qs("#runScienceWorkflow");
  const stages = ["Plan", "Execute", "Verify", "Report"];
  state.science.busy = true;
  renderScienceSteps(stages, 0);
  let visibleStage = 0;
  const progressTimer = window.setInterval(() => {
    visibleStage = Math.min(3, visibleStage + 1);
    renderScienceSteps(stages, visibleStage);
  }, 900);
  if (button) { button.disabled = true; button.textContent = "…"; button.setAttribute("aria-label", "Workflow running"); }
  if (output) output.textContent = "";
  try {
    const payload = await api("/api/science/run", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), goal }),
    });
    const run = payload.run || {};
    renderScienceSteps(stages, 4);
    renderSciencePayload(payload);
    toast("Scientific workflow complete", run.metadata?.science?.status || "Result verified locally");
  } catch (error) {
    renderScienceSteps(stages, visibleStage, error.message);
    if (output) output.textContent = `Scientific workflow failed: ${error.message}`;
    toast("Scientific workflow failed", error.message);
  } finally {
    window.clearInterval(progressTimer);
    state.science.busy = false;
    if (button) { button.disabled = false; button.textContent = "↑"; button.setAttribute("aria-label", "Run workflow"); }
  }
}

function renderScienceSteps(steps, completedCount, failure = "") {
  const container = qs("#scienceRunSteps");
  if (!container) return;
  const runningIndex = failure ? -1 : completedCount < steps.length ? completedCount : -1;
  container.innerHTML = steps.map((step, index) => {
    let status = index < completedCount ? "complete" : index === runningIndex ? "running" : "";
    if (failure && index === Math.max(0, completedCount)) status = "failed";
    return `<div class="science-step ${status}">${escapeHtml(step)}${failure && index === Math.max(0, completedCount) ? `: ${escapeHtml(failure)}` : ""}</div>`;
  }).join("");
}

async function runScienceAction(action) {
  const steps = ["Prepare local input", "Run scientific operation", "Verify output", "Render evidence"];
  renderScienceSteps(steps, 1);
  let payload;
  if (action === "save-memory") {
    payload = await api("/api/science/memory", { method: "POST", body: JSON.stringify({
      project: getProject(), category: qs("#scienceMemoryCategory")?.value,
      name: qs("#scienceMemoryName")?.value, value: qs("#scienceMemoryValue")?.value,
      unit: qs("#scienceMemoryUnit")?.value,
    }) });
  } else if (action === "load-memory") {
    payload = await api(`/api/science/memory?project=${encodeURIComponent(getProject())}`);
  } else if (action === "simulate") {
    payload = await api("/api/science/simulate", { method: "POST", body: JSON.stringify({
      project: getProject(), simulator: qs("#scienceSimulator")?.value,
      inputs: { temperature: Number(qs("#scienceSimulationTemperature")?.value || 350), duration: Number(qs("#scienceSimulationDuration")?.value || 10), conductivity: Number(qs("#scienceSimulationConductivity")?.value || 1.5) },
    }) });
  } else if (action === "execute-code") {
    payload = await api("/api/science/execute", { method: "POST", body: JSON.stringify({ project: getProject(), approved: true, retries: 2, code: qs("#sciencePythonCode")?.value || "" }) });
  } else if (action === "verify") {
    const verification = JSON.parse(qs("#scienceVerificationPayload")?.value || "{}");
    payload = await api("/api/science/verify", { method: "POST", body: JSON.stringify({ project: getProject(), ...verification }) });
  } else if (action === "save-materials" || action === "analyse-materials") {
    const records = await readMaterialsDataset();
    const target = qs("#scienceMaterialsTarget")?.value.trim() || "strength";
    const endpoint = action === "save-materials" ? "/api/science/materials/datasets" : "/api/science/materials/analyse";
    const body = action === "save-materials"
      ? { project: getProject(), name: qs("#scienceMaterialsName")?.value, target, records, source: "local-upload" }
      : { project: getProject(), target, records, maximize: qs("#scienceMaterialsObjective")?.value !== "minimize" };
    payload = await api(endpoint, { method: "POST", body: JSON.stringify(body) });
  } else if (action === "import-public-materials" || action === "analyse-public-materials") {
    const datasetId = qs("#scienceMaterialsCatalog")?.value || "";
    if (!datasetId) throw new Error("Choose a public materials dataset first.");
    if (!qs("#scienceMaterialsNetworkApproved")?.checked) throw new Error("Approve the one-time HTTPS dataset download first.");
    if (!qs("#scienceMaterialsLicenseAccepted")?.checked) throw new Error("Accept the selected dataset license first.");
    const endpoint = action === "import-public-materials"
      ? "/api/science/materials/catalog/import"
      : "/api/science/materials/catalog/analyse";
    payload = await api(endpoint, { method: "POST", body: JSON.stringify({
      project: getProject(), dataset_id: datasetId, allow_external: true,
      license_accepted: true, seed: 31,
    }) });
  } else if (action === "analyse-evidence") {
    payload = await analyseScienceEvidence();
  } else {
    throw new Error(`Unknown scientific action: ${action}`);
  }
  renderScienceSteps(steps, 4);
  renderSciencePayload(payload);
}

async function loadPublicMaterialsCatalog() {
  const select = qs("#scienceMaterialsCatalog");
  if (!select) return;
  const payload = await api("/api/science/materials/catalog");
  const datasets = Array.isArray(payload.datasets) ? payload.datasets : [];
  select.innerHTML = datasets.length
    ? datasets.map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.title)} · ${escapeHtml(item.rows)} rows</option>`).join("")
    : `<option value="">No catalog datasets available</option>`;
  const renderSelection = () => {
    const selected = datasets.find(item => item.id === select.value);
    const status = qs("#scienceMaterialsCatalogStatus");
    if (!status) return;
    status.textContent = selected
      ? `${selected.rows} rows · target ${selected.target} · ${selected.license_id} · records download only after approval; checksum ${String(selected.sha256 || "").slice(0, 12)}...`
      : "No catalog dataset selected.";
  };
  select.addEventListener("change", renderSelection);
  renderSelection();
}

async function readMaterialsDataset() {
  const file = qs("#scienceMaterialsFile")?.files?.[0];
  if (!file) throw new Error("Choose a local CSV or JSON materials dataset first.");
  const text = await file.text();
  if (file.name.toLowerCase().endsWith(".json")) {
    const parsed = JSON.parse(text);
    if (!Array.isArray(parsed)) throw new Error("The materials JSON file must contain an array of records.");
    return parsed;
  }
  return parseScienceCsv(text);
}

function parseScienceCsv(text) {
  const lines = text.split(/\r?\n/).filter(line => line.trim());
  if (lines.length < 2) throw new Error("The CSV needs a header and at least one data row.");
  const headers = lines[0].split(",").map(value => value.trim());
  return lines.slice(1).map(line => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => {
      const raw = (values[index] || "").trim();
      const numeric = Number(raw);
      return [header, raw !== "" && Number.isFinite(numeric) ? numeric : raw];
    }));
  });
}

async function analyseScienceEvidence() {
  const files = Array.from(qs("#scienceEvidenceFiles")?.files || []);
  if (!files.length) throw new Error("Choose at least one local evidence file.");
  const body = { project: getProject(), goal: qs("#scienceGoal")?.value || "Explain the supplied scientific evidence", documents: [], text: [], sensor_csv: [], images: [], simulations: [] };
  for (const file of files) {
    const lower = file.name.toLowerCase();
    if (file.type.startsWith("image/")) {
      body.images.push(await sampleScienceImage(file));
    } else if (lower.endsWith(".csv")) {
      body.sensor_csv.push({ name: file.name, content: await file.text() });
    } else if (lower.endsWith(".json")) {
      body.simulations.push({ name: file.name, content: JSON.parse(await file.text()) });
    } else if (lower.endsWith(".pdf")) {
      body.documents.push({ name: file.name, media_type: file.type || "application/pdf", base64: await fileBase64(file) });
    } else {
      body.text.push({ name: file.name, content: await file.text() });
    }
  }
  return api("/api/science/multimodal", { method: "POST", body: JSON.stringify(body) });
}

function fileBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || "").split(",", 2)[1] || "");
    reader.onerror = () => reject(reader.error || new Error("Could not read the file."));
    reader.readAsDataURL(file);
  });
}

function sampleScienceImage(file) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      const width = Math.min(64, image.naturalWidth || 1);
      const height = Math.min(64, image.naturalHeight || 1);
      const canvas = document.createElement("canvas");
      canvas.width = width; canvas.height = height;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      context.drawImage(image, 0, 0, width, height);
      const rgba = context.getImageData(0, 0, width, height).data;
      const values = [];
      for (let index = 0; index < rgba.length; index += 4) values.push((rgba[index] + rgba[index + 1] + rgba[index + 2]) / 3);
      URL.revokeObjectURL(url);
      resolve({ name: file.name, width, height, values });
    };
    image.onerror = () => { URL.revokeObjectURL(url); reject(new Error(`Could not decode ${file.name}.`)); };
    image.src = url;
  });
}

function renderSciencePayload(payload) {
  const summary = qs("#scienceResultSummary");
  const checks = qs("#scienceVerificationList");
  const table = qs("#scienceResultTable");
  const output = qs("#scienceResult");
  const chart = qs("#scienceResultChart");
  const planList = qs("#sciencePlanList");
  const terminal = qs("#scienceExecutionTimeline");
  if (summary) summary.innerHTML = "";
  if (checks) checks.innerHTML = "";
  if (table) table.innerHTML = "";
  if (chart) chart.classList.remove("visible");
  if (planList) planList.innerHTML = "";
  if (terminal) terminal.innerHTML = "";

  const analysis = payload.analysis || payload.run?.metadata?.science?.result || payload.simulation || payload.execution || payload.verification || payload.context || payload;
  const verification = payload.analysis?.verification || payload.verification?.scientific || payload.run?.metadata?.science?.verification;
  const plan = payload.run?.metadata?.science?.plan;
  const execution = analysis.execution || (payload.execution?.lifecycle ? payload.execution : null);
  const findings = verification?.findings || [];
  const summaryValues = [
    ["Status", analysis.status || (verification?.passed ? "passed" : "complete")],
    ["Verification", verification?.score != null ? `${Math.round(Number(verification.score) * 100)}%` : findings.length ? `${findings.filter(item => item.passed).length}/${findings.length}` : "not run"],
    ["Execution", execution?.elapsed_ms != null ? `${Number(execution.elapsed_ms).toFixed(1)} ms` : "—"],
  ];
  if (summary) summary.innerHTML = summaryValues.map(([label, value]) => `<div class="science-summary-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`).join("");
  if (checks) checks.innerHTML = findings.map(item => `<div class="science-check ${item.passed ? "pass" : "fail"}"><div><strong>${escapeHtml(item.check || "check")}</strong><div class="muted">${escapeHtml(item.message || "")}</div></div></div>`).join("");
  if (planList && plan?.steps?.length) {
    planList.innerHTML = `<div class="science-record-heading">Problem plan</div>${plan.steps.map((step, index) => `<div class="science-plan-row"><span>${index + 1}</span><div>${escapeHtml(step)}</div></div>`).join("")}`;
  }
  if (terminal && execution) {
    const lifecycle = execution.lifecycle || [];
    terminal.innerHTML = `<div class="science-record-heading">Execution terminal</div>${lifecycle.map(item => `<div class="science-terminal-row ${item.status === "complete" ? "complete" : item.status === "captured" ? "failed" : ""}"><span>${escapeHtml(item.stage || "event")}</span><div>${escapeHtml(item.message || "")}</div>${item.attempt ? `<small>Attempt ${Number(item.attempt)}</small>` : ""}</div>`).join("")}<div class="science-profile-grid"><div><span>Attempts</span><strong>${Number(execution.attempts || 0)}</strong></div><div><span>Repairs</span><strong>${Number(execution.repairs || 0)}</strong></div><div><span>Execution</span><strong>${Number(execution.elapsed_ms || 0).toFixed(1)} ms</strong></div><div><span>Peak memory</span><strong>${formatBytes(Number(execution.peak_memory_bytes || 0))}</strong></div></div>`;
  }

  const ranked = analysis.ranked_candidates || [];
  if (table && ranked.length) {
    table.innerHTML = `<table class="data-table"><thead><tr><th>Candidate</th><th>Prediction</th><th>Uncertainty</th></tr></thead><tbody>${ranked.map((item, index) => `<tr><td>${index + 1}</td><td>${Number(item.prediction).toFixed(4)}</td><td>${Number(item.uncertainty).toFixed(4)}</td></tr>`).join("")}</tbody></table>`;
  }
  if (analysis.chart?.length) drawScienceChart(analysis.chart);
  if (output) output.textContent = payload.run?.response || formatStructuredOutput(analysis);
}

function drawScienceChart(points) {
  const canvas = qs("#scienceResultChart");
  const context = canvas?.getContext("2d");
  if (!canvas || !context || !points.length) return;
  canvas.classList.add("visible");
  context.clearRect(0, 0, canvas.width, canvas.height);
  const values = points.map(point => Number(point.prediction) || 0);
  const maximum = Math.max(...values, 1);
  const barWidth = Math.max(12, (canvas.width - 60) / points.length - 12);
  context.font = "12px system-ui";
  points.forEach((point, index) => {
    const height = (values[index] / maximum) * 180;
    const x = 40 + index * (barWidth + 12);
    const y = 220 - height;
    context.fillStyle = "#19c37d";
    context.fillRect(x, y, barWidth, height);
    context.fillStyle = "#9aa7a0";
    context.fillText(String(index + 1), x + barWidth / 2 - 3, 242);
  });
}

// =============================================
// Calendar System
// =============================================

async function loadCalendarEvents() {
  try {
    const payload = await api(`/api/calendar/events?project=${encodeURIComponent(getProject())}&limit=20`);
    const events = payload.events || [];
    const el = qs("#calendarFeed");
    if (el) el.innerHTML = events.length
      ? events.map(e => listItem(
          escapeHtml(e.title),
          `${e.starts_at?.slice(0,16).replace("T"," ")} → ${e.ends_at?.slice(0,16).replace("T"," ")} ${e.location ? "@ " + escapeHtml(e.location) : ""}`,
          `<span class="badge">${escapeHtml(e.status)}</span>${e.rrule ? '<span class="badge" style="margin-left:4px">↻</span>' : ""}`
        )).join("")
      : emptyState("No calendar events");
  } catch(e) { console.warn("calendarFeed", e); }
}

async function createCalendarEvent() {
  const title = qs("#calEventTitle")?.value.trim();
  const startsAt = qs("#calEventStart")?.value;
  const endsAt = qs("#calEventEnd")?.value;
  if (!title || !startsAt) { toast("Title and start time required"); return; }
  try {
    const payload = await api("/api/calendar/events", {
      method: "POST",
      body: JSON.stringify({
        title, starts_at: new Date(startsAt).toISOString(),
        ends_at: endsAt ? new Date(endsAt).toISOString() : new Date(startsAt).toISOString(),
        location: qs("#calEventLocation")?.value.trim() || "",
        description: qs("#calEventDesc")?.value.trim() || "",
        rrule: qs("#calEventRrule")?.value.trim() || "",
        project: getProject(),
      }),
    });
    setOutput("#calendarOutput", payload);
    toast("Event created", title);
    loadCalendarEvents();
  } catch(e) { toast("Calendar error", e.message); }
}

async function loadFreeSlots() {
  const now = new Date();
  const end = new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000);
  try {
    const payload = await api(`/api/calendar/free-slots?project=${encodeURIComponent(getProject())}&from=${encodeURIComponent(now.toISOString())}&to=${encodeURIComponent(end.toISOString())}&duration=60`);
    const slots = payload.slots || [];
    const el = qs("#calFreeSlots");
    if (el) el.innerHTML = slots.length
      ? `<div class="eyebrow" style="margin-bottom:6px">Free 60-min slots this week</div>` +
        slots.slice(0,5).map(s =>
          `<div class="list-item">${s.starts_at.slice(0,16).replace("T"," ")} – ${s.ends_at.slice(11,16)}</div>`
        ).join("")
      : emptyState("No free slots found");
  } catch(e) { toast("Free slots error", e.message); }
}

async function exportIcal() {
  try {
    const payload = await api("/api/calendar/ical-export", {
      method: "POST",
      body: JSON.stringify({ project: getProject() }),
    });
    const blob = new Blob([payload.ical], { type: "text/calendar" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "openagent-calendar.ics"; a.click();
    URL.revokeObjectURL(url);
  } catch(e) { toast("Export error", e.message); }
}

async function loadCalendarSyncStatus() {
  try {
    const [statusPayload, providersPayload, conflictsPayload] = await Promise.all([
      api(`/api/calendar/status?project=${encodeURIComponent(getProject())}`),
      api(`/api/calendar/providers?project=${encodeURIComponent(getProject())}`),
      api(`/api/calendar/conflicts?project=${encodeURIComponent(getProject())}`),
    ]);
    state.calendarSync = {
      status: statusPayload,
      providers: providersPayload.providers || [],
      conflicts: conflictsPayload.conflicts || [],
    };
    renderCalendarSyncStatus();
    return state.calendarSync;
  } catch(e) {
    renderCalendarSyncStatus({ error: e.message });
    return null;
  }
}

function renderCalendarSyncStatus(error = null) {
  const cards = qs("#calendarSyncCards");
  const providerSelect = qs("#calendarProviderSelect");
  const feed = qs("#calendarSyncFeed");
  const conflictFeed = qs("#calendarConflictFeed");
  if (error?.error) {
    if (cards) cards.innerHTML = `<article class="ops-tile ops-danger"><span>Status</span><strong>Attention</strong><p>${escapeHtml(error.error)}</p></article>`;
    if (feed) feed.innerHTML = emptyState("Calendar sync status is unavailable.");
    if (conflictFeed) conflictFeed.innerHTML = "";
    return;
  }
  const providers = state.calendarSync.providers || [];
  const conflicts = state.calendarSync.conflicts || [];
  const status = state.calendarSync.status || {};
  if (providerSelect) {
    const current = providerSelect.value;
    providerSelect.innerHTML = providers.length
      ? providers.map(provider => `<option value="${escapeHtml(provider.provider_id)}">${escapeHtml(provider.provider)} · ${escapeHtml(provider.status || "configured")}</option>`).join("")
      : '<option value="">No sync providers</option>';
    if (current) providerSelect.value = current;
  }
  if (cards) {
    cards.innerHTML = [
      opsTile("Providers", providers.length || "0", "Manual ICS and CalDAV HTTP"),
      opsTile("Open conflicts", conflicts.filter(item => item.status === "open").length || "0", "Manual policy available"),
      opsTile("Local events", status.events ?? status.event_count ?? "0", "SQLite calendar store"),
      opsTile("Sync tokens", providers.filter(provider => provider.sync_token).length || "0", "REPORT cursors tracked"),
    ].join("");
  }
  if (feed) {
    feed.innerHTML = providers.length
      ? providers.map(provider => {
          const badgeClass = provider.status === "connected" ? "badge-success" : provider.status === "error" ? "badge-error" : "badge-info";
          const details = [
            provider.base_url || "local/manual",
            provider.sync_token ? `sync-token ${provider.sync_token}` : "no sync-token yet",
            provider.calendar_id || "default calendar",
          ].join(" · ");
          return listItem(provider.provider, details, `<span class="badge ${badgeClass}">${escapeHtml(provider.status || "configured")}</span>`);
        }).join("")
      : emptyState("Register a manual ICS or CalDAV HTTP provider to enable sync.");
  }
  if (conflictFeed) {
    conflictFeed.innerHTML = conflicts.length
      ? conflicts.slice(0, 6).map(conflict => listItem(
          `Conflict ${conflict.conflict_id?.slice(0, 8) || ""}`,
          `${conflict.status || "open"} · local ${conflict.local_event_id || "?"} · remote ${conflict.remote_uid || "?"}`,
          `<span class="badge ${conflict.status === "resolved" ? "badge-success" : "badge-warning"}">${escapeHtml(conflict.status || "open")}</span>`
        )).join("")
      : `<div class="mini-status mini-status-success"><strong>No calendar conflicts</strong><p>Pull, push, and If-Match failures are recorded here when detected.</p></div>`;
  }
}

async function refreshCalendarSync() {
  await Promise.allSettled([loadCalendarEvents(), loadCalendarSyncStatus()]);
}

async function registerCalendarProvider() {
  try {
    const config = readJsonInput("#calProviderConfig", {});
    const payload = await api("/api/calendar/providers/register", {
      method: "POST",
      body: JSON.stringify({
        provider: qs("#calProviderKind")?.value || "manual",
        project: getProject(),
        base_url: qs("#calProviderUrl")?.value.trim() || "",
        account_ref: qs("#calProviderAccount")?.value.trim() || "",
        calendar_id: qs("#calProviderCalendarId")?.value.trim() || "",
        config,
      }),
    });
    setOutput("#calendarOutput", payload);
    await loadCalendarSyncStatus();
    toast("Calendar provider registered", payload.provider || payload.provider_id || "");
  } catch(e) { toast("Calendar provider error", e.message); }
}

async function runCalendarSync(direction = "pull") {
  const providerId = qs("#calendarProviderSelect")?.value;
  if (!providerId) { toast("Select a calendar provider first"); return; }
  try {
    const payload = await api("/api/calendar/sync", {
      method: "POST",
      body: JSON.stringify({
        provider_id: providerId,
        direction,
        conflict_policy: qs("#calConflictPolicy")?.value || "latest_update",
      }),
    });
    let output = payload;
    if (payload.sync_job_id) {
      try {
        output = await api(`/api/calendar/sync/${encodeURIComponent(payload.sync_job_id)}`);
      } catch (_) {
        output = payload;
      }
    }
    setOutput("#calendarOutput", output);
    await refreshCalendarSync();
    toast("Calendar sync completed", `${direction}: ${payload.state || "recorded"}`);
  } catch(e) { toast("Calendar sync error", e.message); }
}

// =============================================
// Digital Twin
// =============================================

async function loadTwinProfile() {
  try {
    const payload = await api(`/api/twin/profile?project=${encodeURIComponent(getProject())}`);
    setOutput("#twinFeed", payload);
  } catch(e) { toast("Twin profile error", e.message); }
}

async function loadTwinPatterns() {
  try {
    const payload = await api(`/api/twin/patterns?project=${encodeURIComponent(getProject())}`);
    const el = qs("#twinFeed");
    if (!el) return;
    const patterns = payload.patterns || [];
    el.innerHTML = patterns.length
      ? patterns.map(p => listItem(escapeHtml(p.type), escapeHtml(p.description))).join("") +
        `<div class="eyebrow" style="margin-top:12px">Top Topics</div>` +
        (payload.top_topics || []).map(t =>
          `<span class="badge" style="margin:2px">${escapeHtml(t)}</span>`
        ).join("")
      : emptyState("No patterns yet — interact more with OpenAgent");
  } catch(e) { toast("Patterns error", e.message); }
}

async function loadTwinSuggestions() {
  try {
    const payload = await api(`/api/twin/suggestions?project=${encodeURIComponent(getProject())}`);
    const el = qs("#twinFeed");
    if (!el) return;
    const suggestions = payload.suggestions || [];
    el.innerHTML = suggestions.length
      ? suggestions.map(s =>
          listItem(
            escapeHtml(s.title),
            escapeHtml(s.description),
            `<span class="badge badge-${s.priority === "high" ? "accent" : "soft"}">${escapeHtml(s.priority)}</span>`
          )
        ).join("")
      : emptyState("No suggestions at this time");
  } catch(e) { toast("Suggestions error", e.message); }
}

async function predictIntent() {
  const ctx = qs("#twinContext")?.value.trim();
  if (!ctx) { toast("Enter context text first"); return; }
  try {
    const payload = await api("/api/twin/predict", {
      method: "POST",
      body: JSON.stringify({ context: ctx, project: getProject() }),
    });
    setOutput("#twinOutput", payload);
  } catch(e) { toast("Predict error", e.message); }
}

async function consolidateMemory() {
  try {
    const payload = await api("/api/twin/consolidate", {
      method: "POST",
      body: JSON.stringify({ project: getProject() }),
    });
    setOutput("#twinOutput", payload);
    toast("Memory consolidated", `${payload.consolidated} interactions processed`);
  } catch(e) { toast("Consolidate error", e.message); }
}

async function loadTwinGoals() {
  try {
    const payload = await api(`/api/twin/goals?project=${encodeURIComponent(getProject())}`);
    const goals = payload.goals || [];
    const el = qs("#twinGoals");
    if (el) el.innerHTML = goals.length
      ? goals.map(g =>
          listItem(
            escapeHtml(g.title),
            `${Math.round(g.progress * 100)}% — ${escapeHtml(g.status)}`,
            `<div class="progress-bar" style="width:100%;height:4px;background:var(--line);border-radius:2px;margin-top:4px"><div style="width:${Math.round(g.progress*100)}%;height:100%;background:var(--accent);border-radius:2px"></div></div>`
          )
        ).join("")
      : emptyState("No goals set");
  } catch(e) { console.warn("goals", e); }
}

async function createTwinGoal() {
  const title = qs("#twinGoalTitle")?.value.trim();
  if (!title) { toast("Goal title required"); return; }
  try {
    const payload = await api("/api/twin/goals", {
      method: "POST",
      body: JSON.stringify({
        title,
        description: qs("#twinGoalDesc")?.value.trim() || "",
        project: getProject(),
      }),
    });
    setOutput("#twinOutput", payload);
    toast("Goal created", title);
    loadTwinGoals();
  } catch(e) { toast("Goal error", e.message); }
}

// =============================================
// Encrypted Vault
// =============================================

async function loadVaultEntries() {
  try {
    const payload = await api(`/api/vault/entries?project=${encodeURIComponent(getProject())}`);
    const entries = payload.entries || [];
    const el = qs("#vaultFeed");
    if (el) el.innerHTML = entries.length
      ? entries.map(e =>
          listItem(
            escapeHtml(e.label),
            `${escapeHtml(e.category)} · accessed ${e.access_count} times · ${e.last_accessed ? e.last_accessed.slice(0,10) : "never"}`,
            `<code class="badge" style="font-size:10px">${escapeHtml(e.id?.slice(0,8))}</code>`
          )
        ).join("")
      : emptyState("No vault entries");
  } catch(e) { console.warn("vaultFeed", e); }
}

async function storeVaultSecret() {
  const label = qs("#vaultLabel")?.value.trim();
  const secret = qs("#vaultSecret")?.value.trim();
  const passphrase = qs("#vaultPassphrase")?.value;
  const category = qs("#vaultCategory")?.value;
  const notes = qs("#vaultNotes")?.value.trim();
  if (!label || !secret || !passphrase) { toast("Label, secret, and passphrase required"); return; }
  try {
    const payload = await api("/api/vault/store", {
      method: "POST",
      body: JSON.stringify({ label, secret, passphrase, category, notes, project: getProject() }),
    });
    toast("Secret stored", label);
    qs("#vaultSecret").value = "";
    qs("#vaultPassphrase").value = "";
    loadVaultEntries();
  } catch(e) { toast("Vault error", e.message); }
}

async function retrieveVaultSecret() {
  const entryId = qs("#vaultRetrieveId")?.value.trim();
  const passphrase = qs("#vaultRetrievePass")?.value;
  if (!entryId || !passphrase) { toast("Entry ID and passphrase required"); return; }
  try {
    const payload = await api("/api/vault/retrieve", {
      method: "POST",
      body: JSON.stringify({ entry_id: entryId, passphrase, project: getProject() }),
    });
    const el = qs("#vaultOutput");
    if (el) {
      el.textContent = `Label: ${payload.secret?.label}\nCategory: ${payload.secret?.category}\nSecret: ${payload.secret?.secret}\nNotes: ${payload.secret?.notes}`;
      // Auto-clear after 30 seconds
      setTimeout(() => {
        if (el) el.textContent = "Secret cleared from display.";
        qs("#vaultRetrievePass").value = "";
      }, 30000);
    }
  } catch(e) { toast("Vault retrieve error", e.message); }
}

// =============================================
// Web Scraper
// =============================================

async function loadScraperJobs() {
  try {
    const payload = await api(`/api/scraper/jobs?project=${encodeURIComponent(getProject())}`);
    const jobs = payload.jobs || [];
    const el = qs("#scraperJobs");
    if (el) el.innerHTML = jobs.length
      ? jobs.map(j =>
          listItem(
            escapeHtml(j.url.slice(0, 60) + (j.url.length > 60 ? "…" : "")),
            `Every ${j.interval_hours}h · Next: ${j.next_run?.slice(0,16).replace("T"," ")} · ${escapeHtml(j.status)}`,
          )
        ).join("")
      : emptyState("No scheduled scrape jobs");
  } catch(e) { console.warn("scraperJobs", e); }
}

async function scrapeUrl() {
  const url = qs("#scraperUrl")?.value.trim();
  if (!url) { toast("URL required"); return; }
  const ingest = qs("#scraperIngest")?.checked !== false;
  try {
    const payload = await api("/api/scraper/scrape", {
      method: "POST",
      body: JSON.stringify({ url, ingest, project: getProject() }),
    });
    setOutput("#scraperOutput", payload);
    toast("Scraped", payload.title?.slice(0,40) || url);
  } catch(e) { toast("Scrape error", e.message); }
}

async function scheduleUrl() {
  const url = qs("#scraperUrl")?.value.trim();
  if (!url) { toast("URL required"); return; }
  const interval = parseFloat(qs("#scraperInterval")?.value || "24");
  try {
    const payload = await api("/api/scraper/schedule", {
      method: "POST",
      body: JSON.stringify({ url, interval_hours: interval, project: getProject() }),
    });
    toast("Scheduled", url.slice(0,40));
    loadScraperJobs();
  } catch(e) { toast("Schedule error", e.message); }
}

// =============================================
// Fine-Tuning
// =============================================

async function loadFinetuneStatus() {
  try {
    const [status, datasets, jobs, adapters] = await Promise.all([
      api(`/api/finetune/status?project=${encodeURIComponent(getProject())}`),
      api(`/api/finetune/datasets?project=${encodeURIComponent(getProject())}`),
      api(`/api/finetune/jobs?project=${encodeURIComponent(getProject())}`),
      api(`/api/finetune/adapters?project=${encodeURIComponent(getProject())}`),
    ]);
    const el = qs("#fineTuneFeed");
    if (el) {
      const ds = datasets.datasets || [];
      const js = jobs.jobs || [];
      el.innerHTML = `
        <div class="eyebrow">Datasets (${ds.length})</div>
        ${ds.map(d => listItem(escapeHtml(d.name), `${d.num_examples} examples · ${escapeHtml(d.format)}`)).join("") || emptyState("No datasets")}
        <div class="eyebrow" style="margin-top:12px">Jobs (${js.length})</div>
        ${js.map(j => listItem(
          escapeHtml(j.base_model),
          `${escapeHtml(j.method)} · ${escapeHtml(j.status)} · ${Math.round(j.progress*100)}%`,
          `<div class="progress-bar" style="width:100%;height:4px;background:var(--line);border-radius:2px;margin-top:4px"><div style="width:${Math.round(j.progress*100)}%;height:100%;background:var(--accent-3);border-radius:2px"></div></div>`
        )).join("") || emptyState("No jobs")}
      `;
      // Populate dataset selects
      const dsSelects = qsa("#ftExDataset, #ftJobDataset");
      dsSelects.forEach(sel => {
        const current = sel.value;
        sel.innerHTML = '<option value="">Select dataset…</option>' +
          ds.map(d => `<option value="${escapeHtml(d.id)}">${escapeHtml(d.name)}</option>`).join("");
        if (current) sel.value = current;
      });
    }
    const adEl = qs("#adapterFeed");
    if (adEl) {
      const as = adapters.adapters || [];
      adEl.innerHTML = as.length
        ? as.map(a =>
            listItem(
              escapeHtml(a.name),
              `${escapeHtml(a.base_model)} · rank ${a.rank} · score ${a.eval_score?.toFixed(2)}`,
              `<span class="badge ${a.active ? "badge-accent" : "badge-soft"}">${a.active ? "Active" : "Inactive"}</span>` +
              `<button class="btn btn-mini btn-soft" style="margin-left:6px" data-adapter-activate="${escapeHtml(a.id)}">Activate</button>`
            )
          ).join("")
        : emptyState("No LoRA adapters");
    }
  } catch(e) { console.warn("finetuneStatus", e); }
}

async function createFtDataset() {
  const name = qs("#ftDatasetName")?.value.trim();
  if (!name) { toast("Dataset name required"); return; }
  try {
    const payload = await api("/api/finetune/datasets", {
      method: "POST",
      body: JSON.stringify({
        name,
        description: qs("#ftDatasetDesc")?.value.trim() || "",
        project: getProject(),
      }),
    });
    setOutput("#finetuneOutput", payload);
    toast("Dataset created", name);
    loadFinetuneStatus();
  } catch(e) { toast("Dataset error", e.message); }
}

async function addFtExample() {
  const datasetId = qs("#ftExDataset")?.value;
  const userInput = qs("#ftExUser")?.value.trim();
  const assistantOutput = qs("#ftExAssistant")?.value.trim();
  if (!datasetId || !userInput || !assistantOutput) { toast("All fields required"); return; }
  try {
    const payload = await api("/api/finetune/examples", {
      method: "POST",
      body: JSON.stringify({ dataset_id: datasetId, user_input: userInput, assistant_output: assistantOutput, project: getProject() }),
    });
    toast("Example added");
    qs("#ftExUser").value = "";
    qs("#ftExAssistant").value = "";
  } catch(e) { toast("Example error", e.message); }
}

async function startFtJob() {
  const datasetId = qs("#ftJobDataset")?.value;
  const baseModel = qs("#ftJobModel")?.value.trim();
  const templateId = qs("#ftJobTemplate")?.value;
  if (!datasetId || !baseModel) { toast("Dataset and base model required"); return; }
  try {
    const payload = await api("/api/finetune/jobs", {
      method: "POST",
      body: JSON.stringify({ dataset_id: datasetId, base_model: baseModel, template_id: templateId, project: getProject() }),
    });
    setOutput("#finetuneOutput", payload);
    toast("Job queued", baseModel);
    loadFinetuneStatus();
  } catch(e) { toast("Job error", e.message); }
}

async function activateAdapter(adapterId) {
  try {
    await api("/api/finetune/adapters/activate", {
      method: "POST",
      body: JSON.stringify({ adapter_id: adapterId, project: getProject() }),
    });
    toast("Adapter activated");
    loadFinetuneStatus();
  } catch(e) { toast("Activate error", e.message); }
}

// Alias for the onclick button in the fine-tune view
const loadAdapters = () => loadFinetuneStatus().catch(() => {});
