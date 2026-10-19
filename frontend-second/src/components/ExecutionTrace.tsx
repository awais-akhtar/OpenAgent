import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState, type CSSProperties } from "react";
import { useWorkspaceState } from "../state";

type Route = "agents" | "chat" | "research" | "codeq" | "models" | "automations" | "science" | "finetune" | "scraper" | "voice";
type Step = { name: string; agent: string; detail: string; tool: string; out: string };
type Scene = { title: string; subtitle: string; layout: string; steps: Step[]; metrics: [string, string][]; output: string };
type EventItem = { time: string; message: string };
type RunStatus = "Preview" | "Recording" | "Running" | "Paused" | "Cancelled" | "Completed";
type Run = { status: RunStatus; step: number; progress: number; events: EventItem[] };

const scenes: Record<Route, Scene> = {
  agents: { title: "Multi-agent orchestration", subtitle: "Customer insight · coordinated sample execution", layout: "lanes", steps: [
    { name: "Fetch CRM records", agent: "Data connector", detail: "Retrieve a synthetic lead batch from the local fixture.", tool: "crm.records.list", out: "48 example leads · schema validated" },
    { name: "Analyse with reasoning agent", agent: "Analysis agent", detail: "Segment leads and identify patterns in sample data.", tool: "agent.run_checked", out: "4 example segments · 11 notable patterns" },
    { name: "Generate insights report", agent: "Report agent", detail: "Compose a structured illustrative report and charts.", tool: "report.compose", out: "6 report sections assembled" },
    { name: "Review delivery approval", agent: "Coordinator", detail: "Present an approval preview; do not deliver externally.", tool: "approval.request", out: "Draft prepared for manual review" }], metrics: [["Agents", "3 active roles"], ["Sample tokens", "85K"], ["Example throughput", "31 tok/s"]], output: "Sample insight report prepared. External CRM, model and delivery services were not called." },
  chat: { title: "Chat reasoning & tools", subtitle: "Conversation · sample streaming workflow", layout: "compact", steps: [
    { name: "Understand the message", agent: "Conversation", detail: "Parse the local example prompt.", tool: "chat.prepare", out: "Question and constraints extracted" },
    { name: "Retrieve context", agent: "Knowledge tool", detail: "Find illustrative content from local example records.", tool: "knowledge.hybrid_search", out: "3 synthetic passages" },
    { name: "Compose response", agent: "Assistant", detail: "Stream an example answer in the conversation.", tool: "chat.stream", out: "Example response drafted" },
    { name: "Check response", agent: "Assistant", detail: "Review the sample response before displaying it.", tool: "chat.finish", out: "Example response ready" }], metrics: [["Context", "3 snippets"], ["Generated", "Example"], ["Latency", "Local demo"]], output: "An example response is ready. This is local text rather than a model-generated answer." },
  research: { title: "Research execution", subtitle: "Source discovery · evidence assessment · report", layout: "research", steps: [
    { name: "Define scope", agent: "Planner", detail: "Break the sample question into focused research queries.", tool: "research.plan", out: "4 illustrative questions" },
    { name: "Discover sources", agent: "Source collector", detail: "Load five synthetic source records from local data.", tool: "research.collect", out: "5 illustrative sources" },
    { name: "Evaluate evidence", agent: "Analysis agent", detail: "Compare claims, source dates and evidence strength.", tool: "research.validate", out: "3 claims flagged for review" },
    { name: "Compile report", agent: "Writer", detail: "Assemble an example sourced report with caveats.", tool: "research.report", out: "Draft with sample references" }], metrics: [["Sources", "5 examples"], ["Claims", "3 to review"], ["Report", "1 draft"]], output: "Illustrative research report completed. The listed examples are not verified external sources." },
  codeq: { title: "CodeQ execution trace", subtitle: "Review-before-write · IDE sample session", layout: "code", steps: [
    { name: "Inspect workspace", agent: "CodeQ explorer", detail: "Examine sample project files without modifying disk.", tool: "codebase.tree", out: "12 example paths indexed" },
    { name: "Prepare suggested change", agent: "Coding assistant", detail: "Propose a local example component patch.", tool: "codebase.diff", out: "3 proposed hunks · no write" },
    { name: "Review sample checks", agent: "Test harness", detail: "Display an illustrative validation sequence.", tool: "codebase.preflight", out: "Mock check summary prepared" },
    { name: "Await file-write approval", agent: "Approval gate", detail: "Show how explicit approval would work. No files are written.", tool: "codebase.approval", out: "Awaiting user review (demo)" }], metrics: [["Files", "3 suggested"], ["Diff", "+18 / −6"], ["Tests", "Mock only"]], output: "Example patch is ready for review. No disk changes, tests or terminal commands were executed." },
  models: { title: "Model setup & download", subtitle: "Local model lifecycle · staged sample transfer", layout: "download", steps: [
    { name: "Resolve model manifest", agent: "Model catalogue", detail: "Inspect a synthetic manifest and licence summary.", tool: "models.discover", out: "Example 7B instruct model selected" },
    { name: "Check disk & memory", agent: "Device inspector", detail: "Compare sample model requirements to example hardware.", tool: "local_ai.hardware", out: "Example 8 GB VRAM · 16 GB RAM" },
    { name: "Transfer model files", agent: "Model downloader", detail: "Animate a dummy transfer. No files are downloaded.", tool: "local_ai.download", out: "Illustrative 4.4 GB package staged" },
    { name: "Verify and register", agent: "Local runtime", detail: "Display the hypothetical hash check and registration.", tool: "models.register", out: "Model setup example complete" }], metrics: [["Model", "7B instruct"], ["Package", "4.4 GB"], ["Transfer", "Simulated"]], output: "Synthetic model setup complete. No network transfer, hash validation or model installation occurred." },
  automations: { title: "Automation run timeline", subtitle: "Daily briefing · dry-run simulation", layout: "schedule", steps: [
    { name: "Schedule fires", agent: "Scheduler", detail: "Simulate a scheduled trigger with a sample event.", tool: "scheduler.trigger", out: "Example trigger accepted" },
    { name: "Collect new material", agent: "Research worker", detail: "Load fictional updates for the daily brief.", tool: "workflow.collect", out: "4 example updates collected" },
    { name: "Summarise changes", agent: "Automation agent", detail: "Prepare sample conclusions and follow-up actions.", tool: "workflow.transform", out: "2 priority alerts prepared" },
    { name: "Review delivery", agent: "Approval step", detail: "Show the notification preview; nothing is sent.", tool: "workflow.approval", out: "Awaiting notification preview" }], metrics: [["Trigger", "Dry run"], ["Steps", "4 stages"], ["External", "Blocked"]], output: "A sample daily briefing is ready for review. No scheduled backend workflow ran." },
  science: { title: "Scientific AI experiment", subtitle: "Dataset to analysis · synthetic experiment", layout: "science", steps: [
    { name: "Validate dataset", agent: "Data preparation", detail: "Check a synthetic tabular dataset schema.", tool: "science.dataset.validate", out: "1,280 example observations" },
    { name: "Prepare experiment", agent: "Experiment planner", detail: "Create a local mock proposal and parameter set.", tool: "science.proposal", out: "2 comparison configurations" },
    { name: "Execute example pipeline", agent: "Compute worker", detail: "Simulate computation. No Python/science tool runs.", tool: "science.run", out: "Example evaluation curves generated" },
    { name: "Review findings", agent: "Research report", detail: "Present dummy metrics with uncertainty.", tool: "science.report", out: "Illustrative results compiled" }], metrics: [["Records", "1,280 fake"], ["Trials", "2 example"], ["Confidence", "Demo only"]], output: "A synthetic experiment report was generated. No dataset was analysed by a real compute service." },
  finetune: { title: "Fine-tuning job monitor", subtitle: "PEFT adapter · training preview", layout: "training", steps: [
    { name: "Validate training data", agent: "Dataset service", detail: "Check mock examples and formatting.", tool: "finetune.dataset", out: "2,400 synthetic rows" },
    { name: "Initialise adapter", agent: "PEFT runtime", detail: "Prepare an illustrative LoRA configuration.", tool: "finetune.adapter", out: "r=16 · alpha=32 (example)" },
    { name: "Train example epochs", agent: "Training worker", detail: "Animate dummy batches and checkpoints.", tool: "finetune.train", out: "3 example epochs completed" },
    { name: "Evaluate & package", agent: "Evaluation worker", detail: "Display invented sample loss and adapter packaging.", tool: "finetune.evaluate", out: "Synthetic checkpoint manifest ready" }], metrics: [["Epochs", "3 / 3 demo"], ["Loss", "0.42 sample"], ["Adapter", "LoRA · mock"]], output: "Synthetic adapter checkpoint ready. No GPU training, evaluation, or model modification occurred." },
  scraper: { title: "Web extraction pipeline", subtitle: "Robots check · extraction · structured results", layout: "scrape", steps: [
    { name: "Review URL and policy", agent: "Safety layer", detail: "Check an example URL string; no request is sent.", tool: "scraper.policy", out: "Example policy: external disabled" },
    { name: "Discover page structure", agent: "Parser", detail: "Read a bundled demonstration document.", tool: "scraper.parse", out: "3 sample selectors matched" },
    { name: "Extract structured rows", agent: "Extraction worker", detail: "Map fields to an illustrative output schema.", tool: "scraper.extract", out: "7 synthetic records" },
    { name: "Validate and export", agent: "Result writer", detail: "Prepare exportable JSON fixture data.", tool: "scraper.export", out: "7 mock rows ready" }], metrics: [["Pages", "1 fixture"], ["Records", "7 sample"], ["Egress", "Off"]], output: "Seven synthetic records were assembled locally. No network scraping or external request occurred." },
  voice: { title: "Voice capture & transcription", subtitle: "Microphone-style preview · simulated audio pipeline", layout: "audio", steps: [
    { name: "Begin capture preview", agent: "Browser input", detail: "Show the recording state without accessing the microphone.", tool: "voice.capture", out: "Synthetic audio activity enabled" },
    { name: "Decode speech", agent: "STT pipeline", detail: "Generate sample transcript segments.", tool: "voice.stt", out: "4 example transcript segments" },
    { name: "Clean transcript", agent: "Text postprocessor", detail: "Illustrate speaker segmentation and punctuation.", tool: "voice.transcript", out: "Demo transcript prepared" },
    { name: "Prepare response", agent: "Voice assistant", detail: "Provide a sample summary and ready-to-play state.", tool: "voice.tts", out: "Example voice response available" }], metrics: [["Input", "No microphone"], ["Transcript", "4 sample lines"], ["Audio", "Simulated"]], output: "A synthetic transcript and sample response are available. No microphone data was captured or transcribed." },
};

// The approved trace starts with its first stage complete and the second stage
// marked as the preview. Keep this separate from the running stage index, which
// resets to zero when Play is pressed.
const initial = (): Run => ({ status: "Preview", step: 1, progress: 0, events: [{ time: "11:32:01", message: "Open this sample run and press Play to see the full execution." }] });
const clock = () => new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
// The standalone product HTML uses a small handwritten icon registry and
// falls back to its box mark for unregistered names. Keep that exact contract
// here instead of letting a newer Lucide release silently change the artwork.
const sourceIconPaths: Record<string, string> = {
  network: '<circle cx="12" cy="4" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/><path d="M12 6v7M10 14l-4 3m8-3 4 3M5 13v-2h14v2"/>',
  "message-circle": '<path d="M21 11.5a8.5 8.5 0 0 1-8.5 8.5 9 9 0 0 1-4-.9L3 21l1.8-5.4A9 9 0 1 1 21 11.5Z"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m16 16 5 5"/>',
  "git-pull-request": '<circle cx="6" cy="4" r="2"/><circle cx="6" cy="20" r="2"/><circle cx="18" cy="20" r="2"/><path d="M6 6v12m12 0v-8a5 5 0 0 0-5-5h-2m0 0 3-3m-3 3 3 3"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l4 2"/>',
  flask: '<path d="M9 2h6m-5 0v8L4 20h16l-6-10V2M7 16h10"/>',
  sliders: '<path d="M4 6h16M4 12h16M4 18h16"/><circle cx="9" cy="6" r="2"/><circle cx="15" cy="12" r="2"/><circle cx="8" cy="18" r="2"/>',
  globe: '<circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a17 17 0 0 0 0 20M12 2a17 17 0 0 1 0 20"/>',
  mic: '<rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10a7 7 0 0 0 14 0m-7 7v5m-4 0h8"/>',
  sparkles: '<path d="m12 2 2.2 7.8L22 12l-7.8 2.2L12 22l-2.2-7.8L2 12l7.8-2.2z"/>',
  "file-text": '<path d="M5 2h9l5 5v15H5zM14 2v5h5M8 12h8M8 16h8"/>',
  play: '<path d="m7 4 14 8-14 8z"/>',
  check: '<path d="m5 12 5 5 9-10"/>',
  "check-circle": '<circle cx="12" cy="12" r="10"/><path d="m8 12 3 3 5-6"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 11v6m0-10v1"/>',
  terminal: '<rect x="2" y="3" width="20" height="18" rx="2"/><path d="m6 8 4 4-4 4m7 0h5"/>',
  box: '<path d="m12 2 9 5-9 5-9-5 9-5Zm-9 5v10l9 5 9-5V7M12 12v10"/>',
};
function ReferenceTraceIcon({ name }: { name: string }) {
  return <svg className="icon sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" dangerouslySetInnerHTML={{ __html: sourceIconPaths[name] ?? sourceIconPaths.box }} />;
}
const Glyph = ({ route }: { route: Route }) => {
  switch (route) {
    case "agents": return <ReferenceTraceIcon name="network"/>; case "chat": return <ReferenceTraceIcon name="message-circle"/>; case "research": return <ReferenceTraceIcon name="search"/>; case "codeq": return <ReferenceTraceIcon name="git-pull-request"/>;
    case "models": return <ReferenceTraceIcon name="hard-drive-download"/>; case "automations": return <ReferenceTraceIcon name="clock"/>; case "science": return <ReferenceTraceIcon name="flask"/>;
    // Match the canonical icon registry: this unregistered name falls back to its box mark.
    case "finetune": return <ReferenceTraceIcon name="sliders-horizontal"/>; case "scraper": return <ReferenceTraceIcon name="globe"/>; default: return <ReferenceTraceIcon name="mic"/>;
  }
};

export function ExecutionTrace({ route, startSignal = 0, externalStatus = null, simulation }: { route: Route; startSignal?: number; externalStatus?: "Recording" | "Completed" | null; simulation?: { startMessage?: string; pauseMessage?: string; resumeMessage?: string; eventMessages?: string[]; tickMilliseconds?: number; progressPerTick?: number; onChange?: (status: RunStatus, step: number) => void } }) {
  const mode = useWorkspaceState(state => state.dataMode);
  const scene = scenes[route];
  const [run, setRun] = useState<Run>(initial);
  // Keep execution summaries compact inside workspaces where the primary
  // artifact must remain visible. Users can expand the trace deliberately.
  const compactByDefault = route === "chat" || route === "codeq" || route === "research";
  const [expanded, setExpanded] = useState(!compactByDefault);
  const [selected, setSelected] = useState<number | null>(null);
  const [details, setDetails] = useState(false);
  const lastStartSignal = useRef(0);
  const live = mode === "live";
  // ExecutionTrace is the local demo player. Live workspaces render their
  // domain-owned run/event views instead; this component has no verified live
  // adapter and must never expose its fixture steps or metrics in Live mode.
  const demoOperationStarted = startSignal > 0 || (route === "voice" && externalStatus !== null);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    if (!details) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setDetails(false);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [details]);

  useEffect(() => {
    if (run.status !== "Running" || live) return;
    // Keep the local demo's progression cadence aligned with the approved HTML
    // trace: a 150 ms visual tick with a 2.1 point increment. This is only a
    // deterministic presentation fixture; it never reports a live run.
    const timer = window.setInterval(() => setRun(previous => {
      const progress = Math.min(100, previous.progress + (simulation?.progressPerTick ?? 2.1));
      const step = Math.min(scene.steps.length - 1, Math.floor(progress / (100 / scene.steps.length)));
      // The approved demo workflow emits a final stage event even when its
      // stage index has not changed (the last progress tick completes stage 4).
      const events = progress >= 100
        ? [...previous.events, { time: clock(), message: simulation?.eventMessages?.[scene.steps.length - 1] ?? "Completed" }]
        : step !== previous.step
          ? [...previous.events, { time: clock(), message: simulation?.eventMessages?.[step - 1] ?? `${scene.steps[Math.min(step, scene.steps.length - 1)].agent}: ${scene.steps[Math.min(step, scene.steps.length - 1)].name}` }]
          : previous.events;
      return { ...previous, progress, step: progress >= 100 ? scene.steps.length : step, status: progress >= 100 ? "Completed" : "Running", events };
    }), simulation?.tickMilliseconds ?? 150);
    return () => window.clearInterval(timer);
  }, [run.status, live, scene.steps, simulation?.eventMessages, simulation?.progressPerTick, simulation?.tickMilliseconds]);

  useEffect(() => simulation?.onChange?.(run.status, run.step), [run.status, run.step, simulation]);

  // The approved trace follows the active step while a sample run advances,
  // then leaves the final result expanded for review.
  useEffect(() => {
    if (live) return;
    if (run.status === "Running" || run.status === "Paused") setSelected(run.step);
    else if (run.status === "Completed") setSelected(Math.max(0, scene.steps.length - 1));
  }, [run.status, run.step, live, scene.steps.length]);

  const phase = (index: number) => run.status === "Preview"
    ? index === 0 ? "complete" : index === 1 ? "preview" : "pending"
    : run.status === "Completed" || index < run.step ? "complete"
      : index === run.step && (run.status === "Recording" || run.status === "Running" || run.status === "Paused") ? "active" : "pending";
  const start = () => {
    if (live) return;
    setRun({ status: "Running", step: 0, progress: 0, events: [{ time: clock(), message: simulation?.startMessage ?? "Sample session started · no backend services contacted" }] });
    setExpanded(!compactByDefault);
    setSelected(compactByDefault ? null : 0);
  };
  useEffect(() => {
    if (startSignal > lastStartSignal.current) {
      lastStartSignal.current = startSignal;
      start();
    }
    // The parent increments this signal only for an explicit local-demo action.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [startSignal]);
  useEffect(() => {
    if (route !== "voice" || live || !externalStatus) return;
    setRun(current => ({
      ...current,
      status: externalStatus,
      step: externalStatus === "Recording" ? 0 : scene.steps.length,
      progress: externalStatus === "Recording" ? 20 : 100,
      events: [...current.events, {
        time: clock(),
        message: externalStatus === "Recording"
          ? "Synthetic microphone preview active · no audio hardware used"
          : "Synthetic transcript saved locally",
      }],
    }));
  }, [externalStatus, live, route, scene.steps.length]);
  const pause = () => { setRun(current => { const next = current.status === "Running" ? "Paused" : "Running"; return { ...current, status: next, events: [...current.events, { time: clock(), message: next === "Paused" ? simulation?.pauseMessage ?? "Local sample paused" : simulation?.resumeMessage ?? "Local sample resumed" }] }; }); };
  const stop = () => setRun(current => ({ ...current, status: "Cancelled", events: [...current.events, { time: clock(), message: "Sample stopped · no external operation occurred" }] }));
  const exportExample = () => {
    const blob = new Blob([JSON.stringify({ notice: "LOCAL SYNTHETIC DATA ONLY. No backend run occurred.", route, status: run.status, progress: run.progress, steps: scene.steps, sampleMetrics: scene.metrics, events: run.events, output: scene.output }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob); const anchor = document.createElement("a"); anchor.href = url; anchor.download = `openagent-${route}-sample-trace.json`; anchor.click(); URL.revokeObjectURL(url);
  };

  // Keep the detailed trace contextual: the parent raises startSignal only
  // after an explicit demo operation. Routine page visits and tab changes do
  // not mount a sample workflow. Live workflows use their verified workspace
  // event/status components and therefore do not fall back to this fixture.
  if (live || !demoOperationStarted) return null;

  const researchPoints = [32,38,35,49,48,43,57,52,50,65,59,71,66,73,69,83,80,88,82,94].map((value, index) => `${index * 14},${101 - value + 10}`).join(" ");
  const iconVisual = route === "voice" ? <><div className="trace-visual-top"><ReferenceTraceIcon name="mic"/> Live waveform preview <span>SIMULATED</span></div><div className="trace-voice-wave" aria-hidden="true">{Array.from({ length: 35 }, (_, i) => <i key={i} style={{ "--peak": `${8 + (i * 13 % 43)}px`, "--delay": `${(i % 11) * -0.12}s` } as CSSProperties}/>)}</div><div className="trace-visual-foot"><ReferenceTraceIcon name="audio-lines"/> Synthetic amplitude · {run.status === "Running" || run.status === "Recording" ? "Capturing preview" : "No live microphone input"}</div></>
    : route === "models" ? <><div className="trace-visual-top"><ReferenceTraceIcon name="hard-drive-download"/> Model transfer <span>SIMULATED</span></div><div className="trace-file-row"><div className="trace-model-art">7B</div><div><strong>openagent-7b-instruct.gguf</strong><small>4.4 GB · quantised model · sample</small></div></div><div className="trace-download-track"><i style={{ width: `${run.progress}%` }}/></div><div className="trace-visual-foot">{(4.4 * run.progress / 100).toFixed(2)} / 4.4 GB · {run.status === "Running" ? "Animating file transfer" : "No download performed"}</div></>
    : route === "codeq" ? <><div className="trace-visual-top"><ReferenceTraceIcon name="git-pull-request"/> Proposed change <span>NO WRITE</span></div><pre><span className="trace-dim">src/components/ProductCard.tsx</span>{"\n"}<span className="trace-deleted">- const label = 'Buy now';</span>{"\n"}<span className="trace-added">+ const label = 'Add to basket';</span>{"\n"}<span className="trace-added">+ ariaLabel: product.name;</span></pre></>
    : route === "scraper" ? <><div className="trace-visual-top"><ReferenceTraceIcon name="table-2"/> Extraction preview <span>SIMULATED</span></div><div className="trace-table"><span>Title</span><span>Price</span><span>Available</span><strong>Sample item A</strong><strong>£29.00</strong><strong>Yes</strong><strong>Sample item B</strong><strong>£41.00</strong><strong>No</strong></div></>
    : <><div className="trace-visual-top"><ReferenceTraceIcon name={route === "science" ? "flask-conical" : route === "research" ? "library-big" : route === "automations" ? "calendar-days" : route === "chat" ? "sparkles" : route === "finetune" ? "line-chart" : "activity"}/>{route === "science" ? "Experiment metrics" : route === "research" ? "Evidence graph" : route === "automations" ? "Schedule preview" : route === "chat" ? "Tool-use context" : route === "finetune" ? "Training telemetry" : "Sample activity metrics"}<span>{route === "automations" ? "DRY RUN" : route === "chat" ? "DEMO" : route === "finetune" ? "SIMULATED" : "ILLUSTRATIVE"}</span></div>{route === "research" ? <><div className="trace-source-pills"><span>Question</span><b>→</b><span>5 sample sources</span><b>→</b><span>Draft report</span></div><svg className="trace-chart" viewBox="0 0 266 100" preserveAspectRatio="none" role="img" aria-label="Illustrative metric trend"><defs><linearGradient id="trace-area-research" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#39a9ff" stopOpacity=".22"/><stop offset="1" stopColor="#39a9ff" stopOpacity="0"/></linearGradient></defs><polygon points={`0,100 ${researchPoints} 266,100`} fill="url(#trace-area-research)"/><polyline points={researchPoints} fill="none" stroke="currentColor" strokeWidth="2.7" strokeLinecap="round" strokeLinejoin="round"/></svg></> : route === "automations" ? <div className="trace-schedule-row"><div>08:00<small>Example trigger</small></div><span>→</span><div>Collect<small>4 items</small></div><span>→</span><div>Review<small>No delivery</small></div></div> : route === "chat" ? <div className="trace-chat-context"><p><ReferenceTraceIcon name="search"/> Local knowledge lookup <em>3 examples</em></p><p><ReferenceTraceIcon name="file-text"/> Conversation memory <em>Sample only</em></p></div> : <><svg className="trace-chart" viewBox="0 0 266 100" preserveAspectRatio="none" role="img" aria-label="Illustrative sample chart"><defs><linearGradient id={`trace-area-${route}`} x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#39a9ff" stopOpacity=".22"/><stop offset="1" stopColor="#39a9ff" stopOpacity="0"/></linearGradient></defs><polygon points={`0,100 ${[32,38,35,49,48,43,57,52,50,65,59,71,66,73,69,83,80,88,82,94].map((v, i) => `${i * 14},${101 - v + (['agents','research','chat','codeq','models','automations','science','finetune','scraper','voice'].indexOf(route) % 4) * 5}`).join(' ')} 266,100`} fill={`url(#trace-area-${route})`}/><polyline points={[32,38,35,49,48,43,57,52,50,65,59,71,66,73,69,83,80,88,82,94].map((v, i) => `${i * 14},${101 - v + (['agents','research','chat','codeq','models','automations','science','finetune','scraper','voice'].indexOf(route) % 4) * 5}`).join(' ')} fill="none" stroke="currentColor" strokeWidth="2.7" strokeLinecap="round" strokeLinejoin="round"/></svg><div className="trace-training-markers"><span>{route === "finetune" ? "Epoch 1 / 3" : route === "science" ? "Trial A · 0.81" : "3 agent roles"}</span><span>{route === "finetune" ? "Loss 0.42 (fixture)" : route === "science" ? "Trial B · 0.84" : "4 stages"}</span></div></>}</>;

  return <section className={`trace-mount trace-${route} trace-layout-${scene.layout} ${run.status === "Running" || run.status === "Recording" ? "trace-is-running" : ""}`} aria-label={scene.title}>
    <div className="trace-shell">
      <header className="trace-head"><div className="trace-head-left"><span className="trace-head-icon"><Glyph route={route}/></span><div><div className="trace-title-line"><strong>{scene.title}</strong><span className={`trace-head-state ${run.status === "Running" || run.status === "Recording" ? "trace-state-live" : ""}`}>{live ? "Backend status only" : run.status === "Preview" ? "Interactive preview" : run.status}</span></div><p>{scene.subtitle} <span className="trace-example-mark">· {live ? "LIVE STATE ONLY" : "DEMO DATA"}</span></p></div></div><div className="trace-head-actions"><button className="trace-btn" type="button" aria-expanded={expanded} onClick={() => setExpanded(value => !value)}>{expanded ? <ReferenceTraceIcon name="chevron-up"/> : <ReferenceTraceIcon name="chevron-down"/>}{expanded ? "Collapse" : "Expand"}</button></div></header>
      {expanded ? <div className="trace-layout"><div className="trace-main"><div className="trace-run-row"><div><span className="trace-run-kicker">{live ? "VERIFIED BACKEND STATE" : run.status === "Running" ? <><i className="trace-live-pulse"/> PLAYING LOCAL EXAMPLE</> : "EXECUTION PREVIEW"}</span><p className="trace-run-sub">{live ? "Execution events are not connected to this view yet. No simulated run will start in live mode." : run.status === "Completed" ? scene.output : run.status === "Cancelled" ? "Simulation stopped; no external operation occurred." : run.status === "Preview" ? "Explore the full workflow below, or play the synthetic run." : `Stage ${Math.min(run.step + 1, scene.steps.length)} of ${scene.steps.length} · ${scene.steps[Math.min(run.step, scene.steps.length - 1)].name}`}</p></div><span className="trace-run-percent">{live ? "—" : Math.round(run.progress) + "%"}</span></div><div className="trace-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={live ? undefined : Math.round(run.progress)} aria-label={live ? "Backend progress unavailable" : "Simulated activity progress"}><span style={{ width: live ? "0%" : `${run.progress}%` }}/></div><div className="trace-steps">{scene.steps.map((step, i) => { const status = live ? "pending" : phase(i); const isSelected = selected === i; return <div className={`trace-step trace-step-${status} ${isSelected ? "trace-selected" : ""}`} key={step.name}><div className="trace-connector"><span className="trace-step-node">{status === "complete" ? <ReferenceTraceIcon name="check"/> : status === "active" ? <i className="trace-node-core"/> : i + 1}</span></div><div className="trace-step-body"><button type="button" className="trace-step-head" aria-expanded={isSelected} onClick={() => setSelected(isSelected ? null : i)}><span className="trace-step-title">{i + 1}. {step.name}<small>{step.agent}</small></span><span className={`trace-status trace-status-${status}`}>{status === "active" ? run.status : status === "complete" ? "Completed" : status === "preview" ? "Preview" : "Pending"}</span>{isSelected ? <ReferenceTraceIcon name="chevron-up"/> : <ReferenceTraceIcon name="chevron-down"/>}</button><AnimatePresence initial={false}>{isSelected && <motion.div className="trace-step-extra" initial={reducedMotion ? false : { height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: reducedMotion ? 0 : .2 }}><p>{step.detail}</p><div className="trace-tool"><span><ReferenceTraceIcon name="terminal"/> Tool</span><code>{step.tool}</code></div><div className="trace-step-output"><ReferenceTraceIcon name={status === "complete" ? "check-circle" : "info"}/>{status === "complete" ? step.out : live ? "Waiting for a verified backend event" : "Illustrative operation · select Play to advance"}</div></motion.div>}</AnimatePresence></div></div>; })}</div><div className="trace-controls"><button type="button" className="trace-btn primary" disabled={live} onClick={start}><ReferenceTraceIcon name="play"/> {run.status === "Completed" ? "Replay example" : "Play example"}</button>{!live && (run.status === "Running" || run.status === "Paused") && <><button type="button" className="trace-btn" onClick={pause}>{run.status === "Paused" ? <ReferenceTraceIcon name="play"/> : <ReferenceTraceIcon name="pause"/>}{run.status === "Paused" ? "Resume" : "Pause"}</button><button type="button" className="trace-btn" onClick={stop}><ReferenceTraceIcon name="square"/> Stop</button></>}<button type="button" className="trace-btn" onClick={() => setDetails(true)}><ReferenceTraceIcon name="maximize-2"/> Details</button></div>{run.status === "Completed" && !live && <div className="trace-result"><ReferenceTraceIcon name="check-circle-2"/><span>{scene.output}</span></div>}</div><aside className="trace-rail"><div className={`trace-visual ${route === "voice" ? "trace-visual-audio" : ""}`}>{iconVisual}</div><div className="trace-metric-card"><div className="trace-rail-heading"><ReferenceTraceIcon name="gauge"/><strong>{route === "models" ? "Model details" : route === "voice" ? "Capture status" : route === "finetune" ? "Training summary" : "Sample metrics"}</strong></div><div className="trace-metric-grid">{scene.metrics.map(([label, value]) => <div key={label}><small>{label}</small><strong>{value}</strong></div>)}</div></div><div className="trace-log-panel"><div className="trace-rail-heading"><ReferenceTraceIcon name="activity"/><strong>Execution log</strong><span className="trace-demo-indicator">{live ? "BACKEND EVENTS" : "LOCAL EXAMPLE"}</span></div><div className="trace-log-stream" aria-live="polite">{run.events.slice(-6).map((event, i) => <p key={`${i}-${event.time}`}><time>{event.time}</time> {event.message}</p>)}</div></div></aside></div> : <div className="trace-collapsed"><div className="trace-mini-steps">{scene.steps.map((step, i) => <span className={`trace-mini-stage ${live ? "pending" : phase(i)}`} title={step.name} key={step.name}><i/><em>{step.name}</em></span>)}</div><div className="trace-mini-actions"><span>{live ? "Backend state only" : `${Math.round(run.progress)}% · ${run.status}`}</span>{!live && (run.status === "Running" || run.status === "Paused") ? <><button className="trace-btn" type="button" onClick={pause}>{run.status === "Paused" ? <ReferenceTraceIcon name="play"/> : <ReferenceTraceIcon name="pause"/>}{run.status === "Paused" ? "Resume" : "Pause"}</button><button className="trace-btn" type="button" onClick={stop}><ReferenceTraceIcon name="square"/> Stop</button></> : <button className="trace-btn primary" type="button" disabled={live} onClick={start}><ReferenceTraceIcon name="play"/> {run.status === "Completed" ? "Replay example" : "Play example"}</button>}</div></div>}
      <footer className="trace-foot"><span><ReferenceTraceIcon name="shield-check"/> {live ? "Live mode · this preview never fabricates execution state" : "Sample UI states only · no API calls, model downloads, execution, audio capture, network requests or file writes"}</span><button type="button" className="trace-btn" onClick={exportExample}><ReferenceTraceIcon name="download"/> Export example</button></footer>
    </div>
    {details && <div className="overlay" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) setDetails(false); }}><motion.div className="modal trace-detail-modal" role="dialog" aria-modal="true" aria-labelledby={`trace-detail-title-${route}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}><div className="row between"><div><div className="trace-run-kicker">{live ? "BACKEND STATUS" : "LOCAL WORKFLOW SIMULATION"}</div><h2 id={`trace-detail-title-${route}`}>{scene.title}</h2><p>{scene.subtitle}</p></div><button className="icon-button" type="button" aria-label="Close details" onClick={() => setDetails(false)}><ReferenceTraceIcon name="x"/></button></div><div className="trace-detail-meta"><div className="trace-metric-grid">{scene.metrics.map(([label, value]) => <div key={label}><small>{label}</small><strong>{value}</strong></div>)}</div><p>{live ? "This screen does not yet receive backend execution events." : scene.output}</p></div><h3>Full execution trace</h3>{scene.steps.map((step, i) => <div className="trace-detail-step" key={step.name}><span className={`trace-status trace-status-${live ? "pending" : phase(i)}`}>{i + 1}</span><div><strong>{step.name}</strong><p>{step.detail}</p><code>{step.tool}</code><small>{step.out}</small></div></div>)}<h3>Local event history</h3><div className="trace-detail-logs">{run.events.map((event, i) => <p key={`${event.time}-${i}`}>{event.time} · {event.message}</p>)}</div><div className="row between" style={{ marginTop: 17 }}><small className="muted">{live ? "No simulated actions performed." : "No real service actions performed."}</small><button className="btn" type="button" onClick={() => setDetails(false)}>Close</button></div></motion.div></div>}
  </section>;
}

export type { Route as ExecutionTraceRoute };
