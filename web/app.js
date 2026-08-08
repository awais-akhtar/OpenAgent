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
  { id: "voice",       label: "Voice Interface",     text: "Push-to-talk, transcription, voice sessions." },
  { id: "research",    label: "Deep Research",       text: "Iterative AI-driven research with source synthesis." },
  { id: "phd",         label: "PhD Studio",          text: "Private profile, position research, and grounded application documents." },
  { id: "agents",      label: "Multi-Agent System",  text: "Specialists, parallel execution, delegation, verification." },
  { id: "knowledge",   label: "Knowledge + RAG",     text: "File ingest, hybrid search, graph, semantic memory." },
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
  { label: "Voice Chat",             icon: "🎙", group: "Navigate", action: () => setView("voice") },
  { label: "Deep Research",          icon: "🔍", group: "Navigate", action: () => setView("research") },
  { label: "PhD Application Studio", icon: "🎓", group: "Navigate", action: () => setView("phd") },
  { label: "Intelligence Overview",  icon: "✨", group: "Navigate", action: () => setView("intelligence") },
  { label: "Agents",                 icon: "🤖", group: "Navigate", action: () => setView("agents") },
  { label: "Knowledge Base",         icon: "📚", group: "Navigate", action: () => setView("knowledge") },
  { label: "Workspace",              icon: "🗂", group: "Navigate", action: () => setView("workspace") },
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
  { label: "Start research",         icon: "🔍", group: "Actions",  action: () => { setView("research"); qs("#researchQuery").focus(); } },
  { label: "Run agent",              icon: "▶️", group: "Actions",  action: () => { setView("agents"); runAgent(); } },
  { label: "Search knowledge",       icon: "🔎", group: "Actions",  action: () => { setView("knowledge"); qs("#knowledgeQuery").focus(); } },
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
  capabilities: {},
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
  researchRunning: false,
  researchAbortController: null,
  researchSessionId: null,
  lastResearchResult: null,
  ultraModel: { status: {}, manifests: [], sessions: [], plan: null, manifestId: "", sessionId: "" },
  adaptiveEngine: { status: {}, plans: [], benchmarks: [], calibrations: [], direct: {}, busy: false },
  engineRoles: [],
  engineRoleOptions: { roles: [], providers: [], models: [] },
  engineRuns: [],
  science: { status: {}, busy: false },
  selectedEngineRole: "chat",
  theme: "dark",
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
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...authorization, ...(options.headers || {}) },
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

function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("oa-theme", theme); } catch(e) {}
  qsa("[data-theme-select]").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.themeSelect === theme);
  });
  const sel = qs("#settingsTheme");
  if (sel) sel.value = theme;
}

function initTheme() {
  let saved = "dark";
  try { saved = localStorage.getItem("oa-theme") || "dark"; } catch(e) {}
  applyTheme(saved);
}

// =============================================
// View navigation
// =============================================

const VIEW_LOADERS = {
  dashboard:      () => Promise.allSettled([loadDashboardTelemetry(), loadAudit()]),
  chat:           () => Promise.allSettled([loadConversations(), loadProviders(), loadModelSetupStatus(), loadModelJobs()]),
  voice:          () => loadVoiceStatus().catch(() => {}),
  phd:            () => loadPhdWorkspace().catch(() => {}),
  intelligence:   () => loadIntelligenceOverview().catch(() => {}),
  agents:         () => loadAgents().catch(() => {}),
  knowledge:      () => loadKnowledgeStatus().catch(() => {}),
  workspace:      () => loadWorkspaceItems().catch(() => {}),
  automation:     () => loadWorkflows().catch(() => {}),
  assistant:      () => Promise.allSettled([getDailyBrief()]),
  communications: () => loadCommunicationHub().catch(() => {}),
  models:         () => Promise.allSettled([loadProviders(), loadModelPolicy(), loadSmoothRuntime("chat")]),
  security:       () => Promise.allSettled([loadAudit(), loadPrivacyReport(), loadIdentityAdministration()]),
  developer:      () => Promise.allSettled([loadDeveloperFeed(), loadDeveloperSdk(), loadIntegrationStatus(), loadPersonalOsStatus(), loadSettingsModels(), loadModelSetupStatus(), loadModelJobs(), loadOperationalModules(), loadUltraModelEngine(), loadSkills()]),
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
    computing: ["Scientific Computing", "Run deterministic solvers and transparent surrogate baselines. FNO, DeepONet, PINN, and operator-learning contracts are available for richer local backends."],
    engineering: ["Engineering Optimisation", "Define variables, bounds, objectives, and physical constraints, then compare feasible candidates with a visible method and evidence trail."],
    multimodal: ["Multimodal Science", "Combine paper or note text, equations, tables, sensor CSV, numerical images, and simulation output into one typed evidence context."],
    verification: ["Scientific Verification", "Check finite values, residuals, units, equations, and engineering constraints before a result is marked reviewable."],
    materials: ["Materials Discovery", "Fit a transparent property predictor, rank candidate materials, estimate uncertainty, and choose the next experiment for review."],
  };
  const detail = details[section] || details.agents;
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
    multimodal: `<label class="field"><span>Local evidence files</span><input id="scienceEvidenceFiles" type="file" multiple accept=".pdf,.txt,.md,.csv,.json,image/*"></label><p class="muted">PDFs, notes, sensor CSV, simulation JSON, and images are combined locally.</p><div class="toolbar"><button class="btn btn-primary" data-science-action="analyse-evidence">Analyse evidence</button></div>`,
    verification: `<label class="field"><span>Result JSON</span><textarea id="scienceVerificationPayload" rows="10">{"text":"Stress is within the stated limit [1].","values":[125000000],"units":{"stress":"Pa"},"residuals":[0.0],"equations":[{"expression":"stress = force/area","symbols":{"stress":125000000,"force":250000,"area":0.002}}],"citations":["design calculation"],"evidence_terms":["stress","limit"],"constraints":[{"name":"stress","passed":true}]}</textarea></label><div class="toolbar"><button class="btn btn-primary" data-science-action="verify">Run verification</button></div>`,
    materials: `<div class="form-grid"><label class="field"><span>Dataset</span><input id="scienceMaterialsFile" type="file" accept=".csv,.json"></label><label class="field"><span>Target property</span><input id="scienceMaterialsTarget" value="strength"></label><label class="field"><span>Dataset name</span><input id="scienceMaterialsName" value="Public materials dataset"></label><label class="field"><span>Objective</span><select id="scienceMaterialsObjective"><option value="maximize">Maximise target</option><option value="minimize">Minimise target</option></select></label></div><div class="toolbar"><button class="btn btn-soft" data-science-action="save-materials">Store dataset</button><button class="btn btn-primary" data-science-action="analyse-materials">Analyse and rank</button></div>`,
  };
  workspace.innerHTML = views[section] || views.agents;
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
  const [health, metrics, providers, localAi, localModels, policy, moduleHealth, uiBuild] = await Promise.allSettled([
    api("/api/health"),
    api("/api/system/metrics"),
    api("/api/models"),
    api("/api/local-ai/status"),
    api("/api/local-ai/models"),
    api("/api/model-policy"),
    api(`/api/modules/health?project=${encodeURIComponent(getProject())}`),
    api("/api/ui/build"),
  ]);
  if (health.status === "fulfilled") state.health = health.value;
  if (metrics.status === "fulfilled") state.systemMetrics = metrics.value;
  if (providers.status === "fulfilled") state.providers = providers.value.providers || [];
  if (localAi.status === "fulfilled") state.localAiStatus = localAi.value;
  if (localModels.status === "fulfilled") state.localModels = localModels.value.models || [];
  if (policy.status === "fulfilled") state.modelPolicy = policy.value;
  if (moduleHealth.status === "fulfilled") state.moduleHealth = moduleHealth.value;
  if (uiBuild.status === "fulfilled") state.uiBuild = uiBuild.value;
  renderMetrics();
  renderDashboardSystem();
}

function renderMetrics() {
  const health = state.health;
  const metrics = state.systemMetrics || {};
  const memory = metrics.memory || {};
  const cpu = metrics.cpu || {};
  const disk = metrics.disk || {};
  const localModels = state.localModels || [];
  const availableProviders = (state.providers || []).filter(p => p.id !== "auto" && p.available).length;
  const items = [
    ["CPU",         formatPercent(cpu.usage_percent)],
    ["RAM",         formatPercent(memory.used_percent)],
    ["RAM Free",    formatGb(memory.available_gb)],
    ["Disk Free",   formatGb(disk.free_gb)],
    ["Local Models", localModels.length],
    ["Providers",   availableProviders],
    ["Chats",       health.conversation_count ?? 0],
    ["Audit",       health.audit_event_count ?? 0],
  ];
  const el = qs("#metricGrid");
  if (el) el.innerHTML = items.map(([l,v]) =>
    `<article class="metric-card"><span>${escapeHtml(l)}</span><strong>${escapeHtml(String(v))}</strong></article>`
  ).join("");
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
    modelSummary.textContent = `${availableProviders.length} live providers · ${models.length} registered local models · ${policy.routing_policy || "local-first"} route`;
  }
  const systemGrid = qs("#dashboardSystemGrid");
  if (systemGrid) {
    const rows = [
      ["Platform", `${platform.system || "Unknown"} ${platform.release || ""}`.trim()],
      ["CPU threads", cpu.count || 1],
      ["Process CPU", `${cpu.process_seconds ?? 0}s`],
      ["RAM total", formatGb(memory.total_gb)],
      ["RAM used", `${formatGb(memory.used_gb)} (${formatPercent(memory.used_percent)})`],
      ["Data disk", `${formatGb(disk.free_gb)} free of ${formatGb(disk.total_gb)}`],
      ["Database", health.database || "local SQLite"],
      ["UI build", uiBuild.version ? `${uiBuild.version} · ${uiBuild.web_root || "local web"}` : "checking"],
      ["Privacy", metrics.privacy_boundary || "Local telemetry only"],
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
    const providerItems = providers
      .filter(p => p.id !== "auto")
      .map(p => {
        const installed = Array.isArray(p.installed_models) ? p.installed_models : [];
        const modelDetail = p.id === "ollama" && installed.length
          ? `${p.selected_model || installed[0]} · ${installed.length} installed`
          : p.default_model || "policy";
        return listItem(
          p.label || p.id,
          `${p.available ? "Available" : "Offline"} · ${modelDetail} · ${p.local ? "local boundary" : "external"}`,
          `<span class="badge ${p.available ? "badge-success" : "badge-soft"}">${escapeHtml(p.id)}</span>`
        );
      });
    const modelItems = models.slice(0, 5).map(m => listItem(
      m.name || m.id,
      `${m.provider || "local"} · ${m.status || "registered"}`,
      `<span class="badge">${escapeHtml(m.metadata?.quantization || m.metadata?.tier || "model")}</span>`
    ));
    modelGrid.innerHTML = [...providerItems, ...modelItems].join("") || emptyState("No model providers or registered local models yet.");
  }
  const healthBadge = qs("#dashboardModuleHealthBadge");
  if (healthBadge) {
    const counts = moduleHealth.counts || {};
    healthBadge.textContent = `${counts.ready || 0} ready`;
    healthBadge.className = `badge ${(counts.error || 0) ? "badge-danger" : (counts.degraded || 0) ? "badge-warn" : "badge-success"}`;
  }
  const healthGrid = qs("#dashboardModuleHealth");
  if (healthGrid) {
    const modules = moduleHealth.modules || [];
    healthGrid.innerHTML = modules.slice(0, 8).map(module => {
      const badgeClass = module.status === "ready" ? "badge-success" : module.status === "degraded" ? "badge-warn" : "badge-danger";
      return listItem(
        module.label || module.id,
        module.reason || "Local status API responded.",
        `<span class="badge ${badgeClass}">${escapeHtml(module.status || "unknown")}</span>`
      );
    }).join("") || emptyState("Module health has not been loaded yet.");
  }
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

function newConversation() {
  stopActiveChat("New conversation started.");
  state.conversationId = null;
  const el = qs("#chatLog");
  if (el) el.innerHTML = "";
  loadConversations().catch(() => {});
  updateChatControls(false);
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
  appendChat("user", message);
  input.value = "";

  if (state.streamingEnabled) {
    await sendChatStreaming(message, provider, model, allowExternal);
  } else {
    await sendChatBlocking(message, provider, model, allowExternal);
  }
  loadConversations().catch(() => {});
}

async function sendChatStreaming(message, provider, model, allowExternal) {
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
      body: JSON.stringify({ message, project: getProject(), provider, model, allow_external: allowExternal, conversation_id: state.conversationId }),
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
    if (node) node.classList.remove("streaming");
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

async function sendChatBlocking(message, provider, model, allowExternal) {
  const node = appendChat("assistant", "…");
  state.chatBusy = true;
  updateChatControls(true);
  try {
    const payload = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, project: getProject(), provider, model, allow_external: allowExternal, conversation_id: state.conversationId }),
    });
    state.conversationId = payload.conversation?.id;
    const lastMsg = (payload.messages || []).at(-1);
    if (node && lastMsg) node.innerHTML = renderMarkdown(lastMsg.content);
  } catch(e) {
    if (node) node.innerHTML = `<em>Error: ${escapeHtml(e.message)}</em>`;
    toast("Chat error", e.message);
  } finally {
    state.chatBusy = false;
    updateChatControls(false);
  }
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
  if (sendBtn) sendBtn.textContent = running ? "Steer" : "Send";
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
  const archived = qs("#showArchivedConversations")?.checked ? "true" : "false";
  const semanticQuery = sort === "semantic" ? (qs("#chatInput")?.value.trim() || "conversation") : "";
  const payload = await api(`/api/conversations/organized?project=${encodeURIComponent(getProject())}&limit=100&sort=${encodeURIComponent(sort)}&include_archived=${archived}&q=${encodeURIComponent(semanticQuery)}`);
  const conversations = payload.conversations || [];
  const el = qs("#conversationList");
  if (el) {
    el.innerHTML = conversations.length
      ? conversations.map(c => `
          <article class="conversation-item ${c.id === state.conversationId ? "active" : ""}" data-conversation-id="${escapeHtml(c.id)}">
            <input type="checkbox" class="conversation-select" value="${escapeHtml(c.id)}" aria-label="Select ${escapeHtml(c.title || "conversation")}">
            <button class="conversation-open" data-conversation-open="${escapeHtml(c.id)}">
              <strong>${c.pinned ? "Pinned · " : ""}${escapeHtml(c.title || "Conversation")}</strong>
              <span>${escapeHtml(c.folder || "Unfiled")} · ${escapeHtml(formatDateTime(c.updated_at) || c.id.slice(0, 8))} · ${escapeHtml(c.message_count || 0)} messages</span>
            </button>
            <button class="conversation-action" data-conversation-action="${c.pinned ? "unpin" : "pin"}" data-conversation-target="${escapeHtml(c.id)}" title="${c.pinned ? "Unpin" : "Pin"}">${c.pinned ? "U" : "P"}</button>
            <button class="conversation-action" data-conversation-action="${c.archived ? "unarchive" : "archive"}" data-conversation-target="${escapeHtml(c.id)}" title="${c.archived ? "Unarchive" : "Archive"}">${c.archived ? "R" : "A"}</button>
          </article>
        `).join("")
      : emptyState("No conversations yet");
    qsa("[data-conversation-open]", el).forEach(item => {
      item.addEventListener("click", () => loadConversation(item.dataset.conversationOpen));
    });
    qsa("[data-conversation-action]", el).forEach(item => {
      item.addEventListener("click", () => runConversationAction(item.dataset.conversationTarget, item.dataset.conversationAction));
    });
  }
}

async function runConversationAction(conversationId, action) {
  await api("/api/conversations/bulk", { method: "POST", body: JSON.stringify({ conversation_ids: [conversationId], action, project: getProject() }) });
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
// Voice
// =============================================

function initVoice() {
  const btn = qs("#voiceRecordBtn");
  if (!btn) return;
  btn.addEventListener("click", toggleVoiceRecording);
  qs("#sendTranscript")?.addEventListener("click", sendTranscriptToChat);
  qs("#clearTranscript")?.addEventListener("click", () => {
    const el = qs("#voiceTranscript");
    if (el) el.textContent = "Transcript will appear here…";
  });
  qs("#chatVoiceBtn")?.addEventListener("click", () => setView("voice"));
  qs("#createVoiceSession")?.addEventListener("click", createVoiceSession);
  qs("#loadVoiceStatus")?.addEventListener("click", loadVoiceStatus);
  qs("#allowFallbackStt")?.addEventListener("change", () => {
    loadVoiceStatus().catch(() => {});
    if (qs("#allowFallbackStt")?.checked) {
      toast("Fallback STT enabled", "Windows Speech may be inaccurate. Use Whisper/faster-whisper for final transcripts.");
    }
  });
}

async function toggleVoiceRecording() {
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
    const btn = qs("#voiceRecordBtn");
    if (btn) { btn.classList.add("recording"); btn.textContent = "⏹"; btn.setAttribute("aria-pressed", "true"); }
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
  const btn = qs("#voiceRecordBtn");
  if (btn) { btn.classList.remove("recording"); btn.textContent = "🎙"; btn.setAttribute("aria-pressed", "false"); }
  const status = qs("#voiceStatus");
  if (status) status.textContent = "Processing…";
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
  } catch(e) {
    toast("Transcription error", e.message);
    const status = qs("#voiceStatus");
    if (status) status.textContent = "Error - transcription request failed";
    const badge = qs("#voiceModeBadge");
    if (badge) { badge.textContent = "Ready"; badge.className = "badge"; }
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
    <div class="research-loading">
      <span class="spinner"></span>
      <div>
        <strong>Researching locally</strong>
        <p>${escapeHtml(query)}</p>
      </div>
    </div>
  `;
}

function renderResearchResult(result) {
  const el = qs("#researchOutput");
  if (!el) return;
  const report = result.report || result.summary || "No report was returned.";
  const sources = result.sources || [];
  el.innerHTML = `
    <div class="research-meta-grid">
      <article><span>Provider</span><strong>${escapeHtml(result.provider || "auto")}</strong></article>
      <article><span>Model</span><strong>${escapeHtml(result.model || "auto")}</strong></article>
      <article><span>Iterations</span><strong>${escapeHtml(result.iterations ?? 0)}</strong></article>
      <article><span>Elapsed</span><strong>${escapeHtml(result.elapsed_seconds ?? 0)}s</strong></article>
    </div>
    <div class="research-plan">
      <strong>Plan</strong>
      <p>${renderMarkdown(result.plan || "No plan returned.")}</p>
    </div>
    <div class="research-markdown">${renderMarkdown(report)}</div>
    <div class="research-source-list">
      <strong>Sources used</strong>
      ${sources.length ? sources.map(source => `
        <article class="list-item">
          <strong>${escapeHtml(source.title || source.type || "Local source")}</strong>
          <p>${escapeHtml(source.snippet || source.url || source.id || "")}</p>
        </article>
      `).join("") : `<div class="empty-state">No local sources matched. Report is based on model reasoning and current prompt only.</div>`}
    </div>
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
    state.agents = payload.agents || [];
    const sel = qs("#agentSelect");
    if (sel) sel.innerHTML = state.agents.map(a => `<option value="${escapeHtml(a.id)}">${escapeHtml(a.name)}</option>`).join("");
    const grid = qs("#agentGrid");
    if (grid) grid.innerHTML = state.agents.map(a =>
      `<article class="agent-tile"><strong>${escapeHtml(a.name)}</strong><p>${escapeHtml(a.description)}</p></article>`
    ).join("");
  } catch(e) { console.warn("loadAgents", e); }
}

async function runAgent(endpoint = "/api/agents/run") {
  const agentId = qs("#agentSelect")?.value || "executive";
  const goal = qs("#goalInput")?.value.trim() || "Review OpenAgent status";
  try {
    const payload = await api(endpoint, {
      method: "POST",
      body: JSON.stringify({ agent: agentId, goal, project: getProject() }),
    });
    renderAgentOutput(payload);
  } catch(e) { toast("Agent error", e.message); }
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
  const items = payload.items || [];
  const el = qs("#workspaceItems");
  if (el) el.innerHTML = items.length
    ? items.map(i => listItem(i.title, `${i.kind} — ${(i.content || "").slice(0, 60)}`)).join("")
    : emptyState("No workspace items yet");
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
  const el = qs("#knowledgeOutput");
  const backendStatus = qs("#ragBackendStatus");
  const accelerators = rag.vector_store?.active_accelerators || [];
  if (backendStatus) {
    backendStatus.innerHTML = [
      payload.fts_available ? "FTS5 ready" : "Lexical fallback",
      ...(accelerators.length ? accelerators.map(name => `${name.toUpperCase()} ready`) : ["SQLite vector fallback"]),
      rag.algorithm || "Advanced RAG ready",
    ].map(label => `<span>${escapeHtml(label)}</span>`).join("");
  }
  if (!el) return;
  el.innerHTML = `
    <div class="knowledge-summary-card">
      <strong>Knowledge engine ready</strong>
      <p>${escapeHtml(payload.fts_available ? "Full-text search is available." : "Fallback lexical search is active.")} Vector adapter: ${escapeHtml(payload.vector_adapter || "local")}.</p>
    </div>
    <div class="research-meta-grid">
      <article><span>Documents</span><strong>${escapeHtml(payload.documents ?? 0)}</strong></article>
      <article><span>Chunks</span><strong>${escapeHtml(payload.chunks ?? 0)}</strong></article>
      <article><span>Entities</span><strong>${escapeHtml(payload.entities ?? 0)}</strong></article>
      <article><span>RAG runs</span><strong>${escapeHtml(rag.query_runs ?? 0)}</strong></article>
    </div>
  `;
}

function initUpload() {
  const zone = qs("#uploadZone");
  const input = qs("#fileInput");
  if (!zone || !input) return;

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") input.click(); });
  input.addEventListener("change", e => { if (e.target.files[0]) uploadFile(e.target.files[0]); });

  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
  });
}

async function uploadFile(file) {
  const status = qs("#uploadStatus");
  if (status) { status.style.display = "block"; status.textContent = `Uploading ${file.name}…`; }
  const formData = new FormData();
  formData.append("file", file);
  formData.append("project", getProject());
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
      body: JSON.stringify({ title, content, project: getProject() }),
    });
    renderKnowledgeIngest(payload.document || payload, "Text ingested");
    toast("Ingested", title);
  } catch(e) { toast("Ingest error", e.message); }
}

async function searchKnowledge() {
  const q = qs("#knowledgeQuery")?.value.trim();
  if (!q) return;
  try {
    const payload = await api("/api/knowledge/hybrid-search", {
      method: "POST",
      body: JSON.stringify({ query: q, project: getProject() }),
    });
    renderKnowledgeSearch(q, payload.results || []);
  } catch(e) { toast("Search error", e.message); }
}

async function synthesizeKnowledge() {
  const q = qs("#knowledgeQuery")?.value.trim() || "summary";
  try {
    const payload = await api("/api/knowledge/synthesize", {
      method: "POST",
      body: JSON.stringify({ query: q, project: getProject() }),
    });
    renderKnowledgeSynthesis(payload);
  } catch(e) { toast("Synthesis error", e.message); }
}

async function runAdvancedRag() {
  const query = qs("#knowledgeQuery")?.value.trim();
  if (!query) { toast("Enter a knowledge question first"); return; }
  const button = qs("#runAdvancedRag");
  const original = button?.textContent || "Deep RAG";
  if (button) { button.disabled = true; button.textContent = "Retrieving…"; }
  try {
    const generate = Boolean(qs("#ragGenerate")?.checked);
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
  const el = qs("#knowledgeOutput");
  if (!el) return;
  const sources = payload.sources || [];
  const confidence = payload.confidence || {};
  const retrieval = payload.retrieval || {};
  const conflicts = payload.conflicts || [];
  const cache = payload.cache || {};
  el.innerHTML = `
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
  `;
}

function renderKnowledgeIngest(document, title = "Knowledge saved") {
  const el = qs("#knowledgeOutput");
  if (!el) return;
  const tags = document.tags || [];
  el.innerHTML = `
    <div class="knowledge-summary-card">
      <strong>${escapeHtml(title)}</strong>
      <p>${escapeHtml(document.title || "Document")} · ${escapeHtml(document.chunk_count ?? 0)} chunks · source ${escapeHtml(document.source || "local")}</p>
    </div>
    <div class="mini-chip-row">
      ${tags.length ? tags.map(tag => `<span>${escapeHtml(tag)}</span>`).join("") : `<span>No automatic tags yet</span>`}
    </div>
  `;
}

function renderKnowledgeSearch(query, results) {
  const el = qs("#knowledgeOutput");
  if (!el) return;
  el.innerHTML = `
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
  `;
}

function renderKnowledgeSynthesis(payload) {
  const el = qs("#knowledgeOutput");
  if (!el) return;
  const sources = payload.sources || payload.results || [];
  const entities = payload.related_entities || [];
  const relationships = payload.related_relationships || [];
  const synthesis = payload.synthesis || payload.summary || payload.answer || "No synthesis text returned.";
  el.innerHTML = `
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
  `;
}

function renderKnowledgeGraph(payload) {
  const el = qs("#knowledgeOutput");
  if (!el) return;
  const entities = payload.entities || [];
  const relationships = payload.relationships || [];
  el.innerHTML = `
    <div class="knowledge-summary-card">
      <strong>Knowledge graph</strong>
      <p>${escapeHtml(entities.length)} entities · ${escapeHtml(relationships.length)} relationships.</p>
    </div>
    <div class="knowledge-result-list">
      ${entities.slice(0, 12).map(entity => `
        <article class="list-item">
          <strong>${escapeHtml(entity.name || "Entity")}</strong>
          <p>${escapeHtml(entity.entity_type || "entity")} · ${escapeHtml(formatDateTime(entity.created_at) || "")}</p>
        </article>
      `).join("") || emptyState("No graph entities yet.")}
    </div>
  `;
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

async function loadAudit() {
  try {
    const payload = await api(`/api/audit?project=${encodeURIComponent(getProject())}&limit=8`);
    const events = payload.events || [];
    const html = events.length
      ? events.map(e => listItem(e.action, `${e.actor||"system"} — ${e.status||"ok"}`)).join("")
      : emptyState("No audit events yet");
    const a = qs("#auditEvents"); if (a) a.innerHTML = html;
    const s = qs("#securityAuditFeed"); if (s) s.innerHTML = html;
  } catch(e) { console.warn("loadAudit", e); }
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
      feed.innerHTML = [toolLine, ...serverItems].join("") || emptyState("No MCP servers registered yet.");
    }
    return { servers, tools, permissions };
  } catch(e) {
    const feed = qs("#mcpRuntimeFeed");
    if (feed) feed.innerHTML = emptyState(`MCP runtime unavailable: ${e.message}`);
    return null;
  }
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
  const theme = qs("#settingsTheme")?.value || "dark";
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
  };
  const model = normalizeKnownModelName(modelByTask[task] || modelByTask.chat);
  const label = task === "code" ? "code model" : task === "reasoning" ? "reasoning model" : "chat model";
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
    if (e.key === "v" && !e.ctrlKey) { e.preventDefault(); setView("voice"); }
    if (e.key === "r" && !e.ctrlKey) { e.preventDefault(); setView("research"); }
  });
}

// =============================================
// Event delegation for nav
// =============================================

function initNavigation() {
  document.addEventListener("click", e => {
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
    document.body.classList.toggle("sidebar-collapsed");
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
  ]);
  toast("Refreshed", "All data reloaded from local runtime.");
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
  initTheme();
  initNavigation();
  initPalette();
  initKeyboardShortcuts();
  initVoice();
  initResearch();
  initUpload();
  initSettings();
  initPhdWorkspace();
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
  qs("#stopChat")?.addEventListener("click", () => stopActiveChat("Response stopped by user."));
  qs("#testChatModel")?.addEventListener("click", testChatModel);
  qs("#conversationSort")?.addEventListener("change", loadConversations);
  qs("#showArchivedConversations")?.addEventListener("change", loadConversations);
  qs("#runConversationBulk")?.addEventListener("click", runConversationBulk);
  qs("#compactConversation")?.addEventListener("click", compactCurrentConversation);
  qs("#showConversationBranches")?.addEventListener("click", showCurrentConversationBranches);
  qs("#chatInput")?.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } });
  qs("#streamToggle")?.addEventListener("click", e => {
    state.streamingEnabled = !state.streamingEnabled;
    e.currentTarget.textContent = state.streamingEnabled ? "⚡ Stream" : "⚡ Batch";
    e.currentTarget.classList.toggle("btn-primary", state.streamingEnabled);
  });
  qs("#refreshAll")?.addEventListener("click", refreshAll);
  qs("#runAgent")?.addEventListener("click", () => runAgent());
  qs("#runCheckedAgent")?.addEventListener("click", () => runAgent("/api/agents/run-checked"));
  qs("#runParallelAgents")?.addEventListener("click", runParallelAgents);
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
  qs("#createWorkflow")?.addEventListener("click", async () => {
    try {
      const d = await api("/api/workflows", { method: "POST", body: JSON.stringify({ name: "Sample workflow", project: getProject(), steps: [{ action: "search_context", query: "OpenAgent" }] }) });
      setOutput("#workflowOutput", d);
      loadWorkflows();
    } catch(e) { toast("Workflow error", e.message); }
  });
  qs("#runDueWorkflows")?.addEventListener("click", async () => {
    try { const d = await api("/api/workflows/run-due", { method: "POST", body: JSON.stringify({ project: getProject() }) }); setOutput("#workflowOutput", d); } catch(e) { toast("Run due error", e.message); }
  });
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
    if (!document.hidden) {
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
  state.science.busy = true;
  renderScienceSteps(["Planning the scientific workflow", "Running local tools", "Checking numerical and physical constraints", "Preparing the reviewable report"], 1);
  if (button) { button.disabled = true; button.textContent = "Running locally…"; }
  if (output) output.textContent = "Preparing the local scientific workflow…";
  try {
    const payload = await api("/api/science/run", {
      method: "POST",
      body: JSON.stringify({ project: getProject(), goal }),
    });
    const run = payload.run || {};
    renderScienceSteps(["Planning the scientific workflow", "Running local tools", "Checking numerical and physical constraints", "Preparing the reviewable report"], 4);
    renderSciencePayload(payload);
    toast("Scientific workflow complete", run.metadata?.science?.status || "Result verified locally");
  } catch (error) {
    renderScienceSteps(["Planning the scientific workflow", "Running local tools", "Checking numerical and physical constraints", "Preparing the reviewable report"], 0, error.message);
    if (output) output.textContent = `Scientific workflow failed: ${error.message}`;
    toast("Scientific workflow failed", error.message);
  } finally {
    state.science.busy = false;
    if (button) { button.disabled = false; button.textContent = "Run locally"; }
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
  } else if (action === "analyse-evidence") {
    payload = await analyseScienceEvidence();
  } else {
    throw new Error(`Unknown scientific action: ${action}`);
  }
  renderScienceSteps(steps, 4);
  renderSciencePayload(payload);
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
  const summaryValues = [
    ["Status", analysis.status || (verification?.passed ? "passed" : "complete")],
    ["Project", getProject()],
    ["Local operation", "Yes"],
  ];
  if (summary) summary.innerHTML = summaryValues.map(([label, value]) => `<div class="science-summary-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`).join("");
  const findings = verification?.findings || [];
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
