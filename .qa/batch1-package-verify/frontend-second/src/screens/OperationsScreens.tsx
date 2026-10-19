import { useEffect, useState, type Dispatch, type FormEvent, type SetStateAction } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { handleTabListKeyDown, tabIndex } from "../components/tabKeyboard";
import { useWorkspaceState } from "../state";
import { motion } from "motion/react";
import {
  Bot, Check, CloudOff, Cpu, Database,
  FileClock, Info, LockKeyhole, Moon, MoreHorizontal,
  Filter, Search, Shield, Sun, Monitor, Workflow, Wrench, UserRound, Sparkles, Plus,
  MessageCircle, Code2, BookOpen, Brain, Clock3, FileText, BarChart3, Bell, CalendarDays, X,
  type LucideIcon,
} from "lucide-react";
import { ApiError, apiConnectionMessage, apiDelete, apiGet, apiPost, apiPut, getApiToken, setApiToken } from "../api/client";
import { demoAwareRequest, demoCollectionQuery, demoCoreQuery, demoQuery } from "../demo/api";
import { modelDemoAction } from "../demo/models";
import { ExecutionTrace } from "../components/ExecutionTrace";
import { demoProvider } from "../demo/provider";
import { AgentsDemoWorkspace } from "./AgentsDemoWorkspace";

const pageMotion = {
  initial: { opacity: 0, y: 7 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.22, ease: "easeOut" as const },
};

function Icon({ icon: Component, size = "sm" }: { icon: LucideIcon; size?: string }) {
  return <Component aria-hidden="true" className={`icon ${size}`} />;
}

function Tabs({ items, selected, onSelect }: { items: string[]; selected: string; onSelect: (value: string) => void }) {
  return (
    <div className="tabs" role="tablist" aria-label="View options" onKeyDown={handleTabListKeyDown}>
      {items.map((item) => (
        <button
          className={`tab ${selected === item ? "active" : ""}`}
          type="button"
          role="tab"
          aria-selected={selected === item}
          tabIndex={tabIndex(selected === item)}
          key={item}
          onClick={() => onSelect(item)}
        >
          {item}
        </button>
      ))}
    </div>
  );
}

function PageHeading({ title, subtitle }: { title: string; subtitle: string }) {
  return <div className="page-heading"><div><h1>{title}</h1><p>{subtitle}</p></div></div>;
}

function Disconnected({ icon: IconComponent = CloudOff, title, detail }: { icon?: LucideIcon; title: string; detail: string }) {
  return (
    <div className="empty" role="status">
      <Icon icon={IconComponent} size="lg" />
      <h3>{title}</h3>
      <p>{detail}</p>
      <div className="row gap6 caption"><span className="status-dot" style={{ background: "var(--muted-2)" }} />Backend connection not available</div>
    </div>
  );
}

export function AgentsScreen() {
  const dataMode = useWorkspaceState(state => state.dataMode);
  const project = activeProject();
  const [view, setView] = useState<"Workspace" | "Directory" | "Access">("Workspace");
  const [mobilePane, setMobilePane] = useState("Coordinator");
  const [query, setQuery] = useState("");
  const [goal, setGoal] = useState("");
  const [agentId, setAgentId] = useState("executive");
  const [fromAgentId, setFromAgentId] = useState("executive");
  const [runMode, setRunMode] = useState<"run" | "delegate">("run");
  const [requestedApprovalId, setRequestedApprovalId] = useState("");
  const [requestedApprovalStatus, setRequestedApprovalStatus] = useState("");
  const [permissionAgent, setPermissionAgent] = useState("executive");
  const [permissionScope, setPermissionScope] = useState("agent.run");
  const [permissionMessage, setPermissionMessage] = useState("");
  const [approvalMessage, setApprovalMessage] = useState("");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [showAllRuns, setShowAllRuns] = useState(false);
  const queryClient = useQueryClient();
  const agentQuery = useQuery({ queryKey: ["agents", dataMode], queryFn: ({ signal }) => demoQuery<Record<string, unknown>>(dataMode, "shell", s => apiGet<{ agents?: unknown[] }>("/api/agents", s), signal).then(data => dataMode === "demo" ? { agents: Array.isArray(data.agents) ? data.agents.map((item, index) => { const agent = item && typeof item === "object" ? item as Record<string, unknown> : {}; return { ...agent, id: agent.id ?? `demo-agent-${index + 1}`, description: agent.description ?? agent.role }; }) : [] } : data), retry: false });
  const runQuery = useQuery({ queryKey: ["agent-runs", project, dataMode], queryFn: async ({ signal }) => {
    if (dataMode === "demo") return demoQuery("demo", "shell.runs", s => apiGet<{ runs?: unknown[] }>(`/api/agent-runs?project=${encodeURIComponent(project)}&limit=50`, s), signal).then(data => ({ runs: Array.isArray(data.runs) ? data.runs : [] }));
    try { return await apiGet<{ runs?: unknown[] }>(`/api/agent-runs?project=${encodeURIComponent(project)}&limit=50`, signal); }
    catch (error) { if (error instanceof ApiError && error.status === 404) return { runs: [] }; throw error; }
  }, retry: false });
  const permissionsQuery = useQuery({ queryKey: ["agent-permissions", project, dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "shell.agentPermissions", s => apiGet<{ permissions?: Record<string, unknown>[] }>(`/api/agents/permissions?project=${encodeURIComponent(project)}`, s), signal).then(data => dataMode === "demo" ? { permissions: [] } : data), enabled: view === "Access", retry: false });
  const approvalsQuery = useQuery({ queryKey: ["agent-approvals", project, dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "shell.agentApprovals", s => apiGet<{ approvals?: Record<string, unknown>[] }>(`/api/agents/approvals?project=${encodeURIComponent(project)}`, s), signal).then(data => dataMode === "demo" ? { approvals: [] } : data), enabled: view === "Access", retry: false });
  const agents = Array.isArray(agentQuery.data?.agents) ? agentQuery.data.agents.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object") : [];
  const runs = Array.isArray(runQuery.data?.runs) ? runQuery.data.runs.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object") : [];
  const permissions = permissionsQuery.data?.permissions ?? [];
  const approvals = approvalsQuery.data?.approvals ?? [];
  const [runError, setRunError] = useState("");
  const runMutation = useMutation({
    mutationFn: () => dataMode === "demo" ? demoAwareRequest<{ run?: Record<string, unknown>; verification?: Record<string, unknown> }>("demo", "POST", "shell.lastAgentRun", { run: { id: `demo-${Date.now()}`, agent: agentId, goal: goal.trim(), response: "Example run complete. This is local demo output.", status: "Completed", created_at: new Date().toISOString(), next_actions: [] }, verification: { status: "Illustrative only" } }) : apiPost<{ run?: Record<string, unknown>; verification?: Record<string, unknown> }>("/api/agents/run-checked", {
      agent: agentId, goal: goal.trim(), project, require_approval: requestedApprovalStatus === "approved",
      ...(requestedApprovalStatus === "approved" ? { approval_id: requestedApprovalId } : {}), verify: true,
    }),
    onSuccess: async () => { setGoal(""); setRunError(""); setRequestedApprovalId(""); setRequestedApprovalStatus(""); await Promise.all([queryClient.invalidateQueries({ queryKey: ["agent-runs"] }), queryClient.invalidateQueries({ queryKey: ["agent-approvals", project] })]); },
    onError: error => setRunError(apiConnectionMessage(error)),
  });
  const delegateMutation = useMutation({
    mutationFn: () => dataMode === "demo" ? demoAwareRequest<Record<string, unknown>>("demo", "POST", "shell.lastDelegation", { id: `demo-${Date.now()}`, from_agent: fromAgentId, to_agent: agentId, goal: goal.trim(), response: "Example delegated task complete. This is local demo output." }) : apiPost<Record<string, unknown>>("/api/agents/delegate", { from_agent: fromAgentId, to_agent: agentId, goal: goal.trim(), project }),
    onSuccess: async () => { setGoal(""); setRunError(""); await queryClient.invalidateQueries({ queryKey: ["agent-runs"] }); },
    onError: error => setRunError(apiConnectionMessage(error)),
  });
  const requestApprovalMutation = useMutation({
    mutationFn: () => apiPost<{ approval?: Record<string, unknown> }>("/api/agents/approvals/request", { requested_by: "user", action: "agent.run", reason: `Run ${agentId} for the submitted task`, project, payload: { agent: agentId, goal: goal.trim() } }),
    onSuccess: async result => { setRequestedApprovalId(String(result.approval?.id ?? "")); setRequestedApprovalStatus(String(result.approval?.status ?? "pending")); setRunError(""); setApprovalMessage("Approval request recorded by the backend."); await queryClient.invalidateQueries({ queryKey: ["agent-approvals", project] }); },
    onError: error => setRunError(apiConnectionMessage(error)),
  });
  const resolveApprovalMutation = useMutation({
    mutationFn: ({ id, approved }: { id: string; approved: boolean }) => apiPost<{ approval?: Record<string, unknown> }>("/api/agents/approvals/resolve", { approval_id: id, approved }),
    onSuccess: async result => { const nextStatus = String(result.approval?.status ?? "updated"); if (String(result.approval?.id ?? "") === requestedApprovalId) setRequestedApprovalStatus(nextStatus); setApprovalMessage(`Backend approval status: ${nextStatus}.`); await queryClient.invalidateQueries({ queryKey: ["agent-approvals", project] }); },
    onError: error => setApprovalMessage(apiConnectionMessage(error)),
  });
  const grantPermissionMutation = useMutation({
    mutationFn: () => apiPost<{ permission?: Record<string, unknown> }>("/api/agents/permissions/grant", { agent: permissionAgent, scope: permissionScope, project, granted_by: "user" }),
    onSuccess: async result => { setPermissionMessage(`Backend permission recorded: ${String(result.permission?.agent_id ?? permissionAgent)} · ${String(result.permission?.scope ?? permissionScope)}.`); await queryClient.invalidateQueries({ queryKey: ["agent-permissions", project] }); },
    onError: error => setPermissionMessage(apiConnectionMessage(error)),
  });
  const displayedAgents = agents.filter(agent => !query || `${agent.name ?? ""} ${agent.id ?? ""} ${agent.description ?? ""}`.toLowerCase().includes(query.toLowerCase()));
  const selectedRun = runs.find(run => String(run.id ?? "") === selectedRunId) ?? runs[0] ?? runMutation.data?.run;
  const loadApprovalTask = (approval: Record<string, unknown>) => {
    const payload = approval.payload && typeof approval.payload === "object" ? approval.payload as Record<string, unknown> : {};
    const approvedAgent = String(payload.agent ?? "");
    const approvedGoal = String(payload.goal ?? "");
    if (!approvedAgent || !approvedGoal) return;
    setAgentId(approvedAgent);
    setGoal(approvedGoal);
    setRequestedApprovalId(String(approval.id ?? ""));
    setRequestedApprovalStatus(String(approval.status ?? "pending"));
    setRunError("");
    setView("Workspace");
  };
  const panes = ["Coordinator", "Researcher", "Developer", "Tasks"];
  const laneMatchers: Record<string, RegExp> = {
    Coordinator: /executive|planning|coordinat|orchestrat|operation|automation/,
    Researcher: /research|knowledge|scientific|science|writing/,
    Developer: /coding|code|developer|security|communication|voice/,
  };
  const laneProfiles = (pane: string) => displayedAgents.filter(agent => laneMatchers[pane]?.test(`${String(agent.id ?? "")} ${String(agent.name ?? "")} ${String(agent.description ?? "")}`.toLowerCase()));
  const selectedRunAgent = String(selectedRun?.agent ?? "").toLowerCase();
  const selectedRunResponse = String(selectedRun?.response ?? selectedRun?.result ?? selectedRun?.message ?? "").trim();
  const currentVerification = runMutation.data?.run && String(runMutation.data.run.id ?? "") === String(selectedRun?.id ?? "") ? runMutation.data.verification : null;
  const runStages = [
    { label: "Task submitted", detail: selectedRun ? String(selectedRun.created_at ?? "Recorded by backend") : dataMode === "live" ? "History unavailable until project scope is verified" : runQuery.isLoading ? "Loading local examples" : runQuery.isError ? "History unavailable" : "No run recorded", state: selectedRun ? "complete" : "" },
    { label: "Agent response", detail: selectedRun ? (selectedRunResponse ? "Response received" : "No response in record") : "Awaiting a recorded run", state: selectedRunResponse ? "complete" : "" },
    { label: "Verification", detail: currentVerification ? String(currentVerification.status ?? "Backend response") : "Not reported", state: currentVerification ? "complete" : "" },
    { label: "Next actions", detail: Array.isArray(selectedRun?.next_actions) ? `${selectedRun.next_actions.length} returned` : "None returned", state: Array.isArray(selectedRun?.next_actions) && selectedRun.next_actions.length ? "complete" : "" },
  ];

  if (dataMode === "demo" && (view === "Workspace" || view === "Directory")) {
    return <AgentsDemoWorkspace directory={view === "Directory"} onDirectory={() => setView(current => current === "Directory" ? "Workspace" : "Directory")} />;
  }

  if (view === "Access") {
    return (
      <motion.div className="page" {...pageMotion}>
        <div className="page-heading"><div><h1>Agent access</h1><p>Review permission records and pending approvals · {project}.</p></div><button className="btn sm" type="button" onClick={() => setView("Workspace")}>Workspace</button></div>
        <section className="card padded section-space" aria-labelledby="grant-agent-permission">
          <h2 id="grant-agent-permission" className="small">Grant a scoped permission</h2>
          <div className="row section-space" style={{ flexWrap: "wrap" }}>
            <label className="caption">Agent<select className="select" aria-label="Permission agent" value={permissionAgent} onChange={event => setPermissionAgent(event.target.value)} disabled={!agents.length}><option value="">Choose agent</option>{agents.map(agent => <option key={String(agent.id)} value={String(agent.id)}>{String(agent.name ?? agent.id)}</option>)}</select></label>
            <label className="caption">Permission scope<select className="select" aria-label="Permission scope" value={permissionScope} onChange={event => setPermissionScope(event.target.value)}>{["agent.run", "agent.delegate", "agent.verify", "memory.read", "memory.write", "workflow.plan"].map(scope => <option key={scope}>{scope}</option>)}</select></label>
            <button className="btn primary" type="button" disabled={!permissionAgent || grantPermissionMutation.isPending} onClick={() => grantPermissionMutation.mutate()}>{grantPermissionMutation.isPending ? "Saving…" : "Grant permission"}</button>
          </div>
          {agentQuery.isError && <p role="alert" className="caption">Agent list unavailable: {apiConnectionMessage(agentQuery.error)}</p>}
          {permissionMessage && <p role="status" className="caption section-space">{permissionMessage}</p>}
          <h3 className="small section-space">Effective permission records</h3>
          {permissionsQuery.isLoading ? <p role="status" className="caption">Loading permissions…</p> : permissionsQuery.isError ? <p role="alert">{apiConnectionMessage(permissionsQuery.error)}</p> : permissions.length ? permissions.map((permission, index) => <div className="list-row" key={String(permission.id ?? index)}><Icon icon={LockKeyhole}/><span className="flex1">{String(permission.agent_id)} · {String(permission.scope)}</span><span className="caption">{String(permission.project)} · {String(permission.granted_by)}</span></div>) : <Disconnected icon={LockKeyhole} title="No permissions returned" detail="The backend returned no effective permission records for this project." />}
        </section>
        <section className="card padded section-space" aria-labelledby="agent-approval-records">
          <div className="row between"><h2 id="agent-approval-records" className="small">Approval records</h2><button className="btn sm" type="button" disabled={approvalsQuery.isFetching} onClick={() => void approvalsQuery.refetch()}>Refresh</button></div>
          {approvalsQuery.isLoading ? <p role="status" className="caption section-space">Loading approvals…</p> : approvalsQuery.isError ? <p role="alert" className="section-space">{apiConnectionMessage(approvalsQuery.error)}</p> : approvals.length ? approvals.map((approval, index) => { const payload = approval.payload && typeof approval.payload === "object" ? approval.payload as Record<string, unknown> : {}; const hasAgentTask = Boolean(payload.agent && payload.goal); return <article className="list-row" key={String(approval.id ?? index)}><Icon icon={Shield}/><span className="flex1"><strong>{String(approval.action)}</strong><span className="caption" style={{ display: "block" }}>{String(approval.reason)} · {String(approval.status)}</span>{Boolean(payload.agent || payload.goal) && <span className="caption" style={{ display: "block" }}>Requested task: {String(payload.agent ?? "agent")} · {String(payload.goal ?? "No goal recorded")}</span>}</span>{hasAgentTask && approval.status === "pending" && <button className="btn sm" type="button" onClick={() => loadApprovalTask(approval)}>Load task</button>}{hasAgentTask && approval.status === "approved" && <button className="btn sm primary" type="button" onClick={() => loadApprovalTask(approval)}>Run approved task</button>}{approval.status === "pending" && <div className="row"><button className="btn sm" type="button" disabled={resolveApprovalMutation.isPending} onClick={() => resolveApprovalMutation.mutate({ id: String(approval.id), approved: false })}>Reject</button><button className="btn sm primary" type="button" disabled={resolveApprovalMutation.isPending} onClick={() => resolveApprovalMutation.mutate({ id: String(approval.id), approved: true })}>Approve</button></div>}</article>; }) : <Disconnected icon={Shield} title="No approval records" detail="Approval requests and their statuses are loaded from the backend." />}
          {approvalMessage && <p role="status" className="caption section-space">{approvalMessage}</p>}
        </section>
      </motion.div>
    );
  }

  if (view === "Directory") {
    return (
      <motion.div className="page" {...pageMotion}>
        <div className="page-heading">
          <div><h1>AI agents</h1><p>Choose a specialist for a task.</p></div>
          <div className="row"><button className="btn sm" type="button" onClick={() => setView("Access")}>Access & approvals</button><button className="btn sm" type="button" onClick={() => setView("Workspace")}>Workspace</button></div>
        </div>
        <label className="search-field">
          <Icon icon={Search} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search agents..." aria-label="Search agents" />
        </label>
        {agentQuery.isLoading ? <div className="section-space caption" role="status">Loading agents…</div> : agentQuery.isError ? <div className="section-space" role="alert">{apiConnectionMessage(agentQuery.error)}</div> : <div className="section-space agent-directory">{displayedAgents.length ? displayedAgents.map(agent => <article className="card padded section-space" key={String(agent.id)}><div className="row"><span className="avatar blue"><Icon icon={Bot}/></span><div><strong>{String(agent.name ?? agent.id)}</strong><p className="caption">{String(agent.description ?? "")}</p></div></div></article>) : <Disconnected icon={Bot} title={query ? "No agents found" : "No agents available"} detail="The backend returned no agent profiles." />}</div>}
      </motion.div>
    );
  }

  return (
    <motion.div className="split-main agent-workspace-screen" {...pageMotion}>
      <header className="agent-top">
        <div className="row between">
          <div className="row" style={{ gap: 15 }}>
            <div className="square-icon"><Icon icon={Workflow} /></div>
            <div><h1 style={{ fontSize: 24 }}>{selectedRun ? String(selectedRun.goal ?? "Agent workspace") : "Agent workspace"}</h1><p className="muted small">{selectedRun ? `${String(selectedRun.agent ?? "Agent")} · backend run ${String(selectedRun.id ?? "recorded")}` : "Coordinate work across specialist agents."}</p></div>
          </div>
          <div className="row agent-header-actions"><button className="btn sm" type="button" onClick={() => setView("Access")}>Access & approvals</button><button className="btn sm" type="button" onClick={() => setView("Directory")}>Agent directory</button><details className="agent-run-panel"><summary className="btn sm primary">Run or delegate</summary>
            <form className="card padded agent-run-form" onSubmit={event => { event.preventDefault(); if (!goal.trim()) return; runMode === "run" ? runMutation.mutate() : delegateMutation.mutate(); }}>
              <div className="row section-space" role="group" aria-label="Agent execution mode"><button className={`btn sm ${runMode === "run" ? "soft" : ""}`} type="button" aria-pressed={runMode === "run"} onClick={() => setRunMode("run")}>Checked run</button><button className={`btn sm ${runMode === "delegate" ? "soft" : ""}`} type="button" aria-pressed={runMode === "delegate"} onClick={() => setRunMode("delegate")}>Delegate</button></div>
              <label className="demo-form-field">Task goal<textarea className="textarea" rows={2} value={goal} onChange={event => { setGoal(event.target.value); setRequestedApprovalId(""); setRequestedApprovalStatus(""); }} placeholder="Describe the task for an agent…" required /></label>
              <div className="row between" style={{ flexWrap: "wrap" }}>{runMode === "delegate" && <label className="caption">From <select className="select" aria-label="Delegating agent" value={fromAgentId} onChange={event => setFromAgentId(event.target.value)} disabled={!agents.length || delegateMutation.isPending}>{agents.map(agent => <option key={String(agent.id)} value={String(agent.id)}>{String(agent.name ?? agent.id)}</option>)}</select></label>}<label className="caption">{runMode === "delegate" ? "To agent" : "Agent"}<select className="select" aria-label={runMode === "delegate" ? "Target agent" : "Select agent"} value={agentId} onChange={event => { setAgentId(event.target.value); setRequestedApprovalId(""); setRequestedApprovalStatus(""); }} disabled={!agents.length || runMutation.isPending || delegateMutation.isPending}><option value="">Choose agent</option>{agents.map(agent => <option key={String(agent.id)} value={String(agent.id)}>{String(agent.name ?? agent.id)}</option>)}</select></label><div className="row">{runMode === "run" && <button className="btn" type="button" disabled={!goal.trim() || !agentId || requestApprovalMutation.isPending || runMutation.isPending || Boolean(requestedApprovalId)} onClick={() => requestApprovalMutation.mutate()}>{requestApprovalMutation.isPending ? "Requesting…" : requestedApprovalId ? "Request recorded" : "Request approval"}</button>}{runMode === "run" && requestedApprovalId && <button className="btn sm" type="button" onClick={() => { setRequestedApprovalId(""); setRequestedApprovalStatus(""); }}>Clear approval</button>}<button className="btn primary" type="submit" disabled={!goal.trim() || !agentId || runMutation.isPending || delegateMutation.isPending || (runMode === "run" && Boolean(requestedApprovalId) && requestedApprovalStatus !== "approved")}>{runMutation.isPending || delegateMutation.isPending ? "Submitting…" : runMode === "delegate" ? "Delegate task" : requestedApprovalId ? requestedApprovalStatus === "approved" ? "Run with approval" : "Approval pending" : "Run agent"}</button></div></div>
              {requestedApprovalId && <p className="caption section-space" role="status">Approval {requestedApprovalId}: {requestedApprovalStatus}. {requestedApprovalStatus === "approved" ? "This backend-approved task can now be submitted." : "Resolve it in Access & approvals before submitting."}</p>}
              {agentQuery.isError && <p className="caption" role="alert">{apiConnectionMessage(agentQuery.error)}</p>}{runError && <p className="caption" role="alert">{runError}</p>}
            </form>
          </details></div>
        </div>
        {(runMutation.isPending || delegateMutation.isPending) && <p className="caption section-space" role="status">Submitting the request to the backend…</p>}
        <div className="steps-line" aria-label="Backend run record activity">{runStages.map(stage => <div className={`step ${stage.state}`} key={stage.label}><strong>{stage.label}</strong><span>{stage.detail}</span></div>)}</div>
        <div className="agent-mobile-tabs" role="tablist" aria-label="Agent panels" onKeyDown={handleTabListKeyDown}>
          {panes.map((pane) => <button className={`btn sm ${mobilePane === pane ? "soft" : ""}`} type="button" role="tab" aria-selected={mobilePane === pane} tabIndex={tabIndex(mobilePane === pane)} key={pane} onClick={() => setMobilePane(pane)}>{pane}</button>)}
        </div>
      </header>
      {/* Actual backend run records are rendered in the workspace below. */}
      <div className="agent-board">
        {panes.slice(0, 3).map((pane) => (
          <section className={`agent-lane ${mobilePane === pane ? "mobile-selected" : ""}`} key={pane}>
            <header className="lane-head"><div className="row"><span className="avatar blue"><Icon icon={Bot} /></span><div className="flex1"><h3>{pane}</h3><p className="caption">{agentQuery.isLoading ? "Loading backend profiles" : agentQuery.isError ? "Unavailable" : `${laneProfiles(pane).length} matching profiles`}</p></div><Icon icon={MoreHorizontal} /></div></header>
            <div className="lane-body">{agentQuery.isError ? <Disconnected icon={Bot} title="Agent service unavailable" detail={apiConnectionMessage(agentQuery.error)} /> : agentQuery.isLoading ? <div className="caption" role="status">Loading agent profiles…</div> : laneProfiles(pane).length ? laneProfiles(pane).map(agent => <button className="list-row agent-profile-row" type="button" key={String(agent.id)} onClick={() => { setAgentId(String(agent.id ?? "")); setGoal(current => current || String(selectedRun?.goal ?? "")); }} title="Select this backend agent in Run or delegate"><span className="avatar blue"><Icon icon={Bot}/></span><span className="flex1"><strong>{String(agent.name ?? agent.id)}</strong><small className="caption">{String(agent.description ?? "")}</small></span><span className="caption">Select</span></button>) : <div className="empty agent-lane-empty" role="status"><Icon icon={Bot} size="lg"/><h3>No {pane.toLowerCase()} profiles</h3><p>Matching roles from the backend agent directory appear here.</p></div>}
              {selectedRun && (selectedRunAgent === pane.toLowerCase() || laneProfiles(pane).some(agent => String(agent.id ?? "").toLowerCase() === selectedRunAgent)) && <article className="lane-message"><div className="row"><span className="avatar blue"><Icon icon={Bot}/></span><strong>{String(selectedRun.agent ?? "Agent")}</strong><span className="caption">Backend run</span></div>{selectedRunResponse ? <p>{selectedRunResponse}</p> : <p className="caption">The selected run record contains no response text.</p>}{Array.isArray(selectedRun.next_actions) && selectedRun.next_actions.length > 0 && <div className="lane-note"><h3>Next actions from backend</h3>{selectedRun.next_actions.slice(0, 4).map((action, index) => <p key={`${String(action)}-${index}`}>• {String(action)}</p>)}</div>}</article>}
              {!(selectedRun && (selectedRunAgent === pane.toLowerCase() || laneProfiles(pane).some(agent => String(agent.id ?? "").toLowerCase() === selectedRunAgent))) && !agentQuery.isError && !agentQuery.isLoading && <div className="lane-activity-empty" role="status"><strong>No activity recorded</strong><p>{selectedRun ? "The selected run has no output for this specialist." : "Backend run messages appear here when recorded."}</p></div>}
            </div>
          </section>
        ))}
        <aside className={`agent-context ${mobilePane === "Tasks" ? "mobile-selected" : ""}`}>
          <div className="section-head"><div><h3>Tasks</h3><span className="caption">{runMutation.isPending || delegateMutation.isPending ? "Submitting request" : runQuery.isLoading ? "Loading run history" : dataMode === "live" ? `${runs.length} backend runs` : `${runs.length} example runs`}</span></div><button className="btn sm" type="button" title={dataMode === "live" ? "Refresh authorized project run history." : "Refresh example history"} disabled={runQuery.isFetching} onClick={() => void runQuery.refetch()}>Refresh</button></div>
          {runMutation.data?.run && <div className="card padded section-space"><div className="row between"><strong>Latest checked run</strong><span className="caption">{String(runMutation.data.run.status ?? "Returned by backend")}</span></div><p className="small section-space" style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{String(runMutation.data.run.response ?? runMutation.data.run.result ?? runMutation.data.run.message ?? "The backend returned no response text.")}</p>{runMutation.data.verification && <p className="caption">Verification: {String(runMutation.data.verification.status ?? "returned")}{Array.isArray(runMutation.data.verification.findings) ? ` · ${(runMutation.data.verification.findings as unknown[]).map(String).join("; ")}` : ""}</p>}{runMutation.data.run.id != null && <p className="caption">Run ID {String(runMutation.data.run.id)}</p>}</div>}
          {delegateMutation.data && (() => { const result = delegateMutation.data.result && typeof delegateMutation.data.result === "object" ? delegateMutation.data.result as Record<string, unknown> : {}; const delegatedRun = result.run && typeof result.run === "object" ? result.run as Record<string, unknown> : result; const verification = result.verification && typeof result.verification === "object" ? result.verification as Record<string, unknown> : null; return <div className="card padded section-space"><div className="row between"><strong>Delegation result</strong><span className="caption">{String(delegateMutation.data.id ?? "Returned by backend")}</span></div><p className="small section-space" style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{String(delegatedRun.response ?? delegatedRun.message ?? "The backend returned no response text.")}</p>{verification && <p className="caption">Verification: {String(verification.status ?? "returned")}{Array.isArray(verification.findings) ? ` · ${(verification.findings as unknown[]).map(String).join("; ")}` : ""}</p>}</div>; })()}
          {selectedRun && <section className="agent-selected-task"><h4>Selected task</h4><p>{String(selectedRun.goal ?? "Goal not reported")}</p><span className="caption">{String(selectedRun.status ?? "Backend run record")}{selectedRun.created_at ? ` · ${String(selectedRun.created_at)}` : ""}</span>{typeof selectedRun.confidence === "number" && Number.isFinite(selectedRun.confidence) && <div className="settings-line"><span className="flex1">Confidence</span><strong>{String(selectedRun.confidence)}</strong></div>}</section>}
          {Array.isArray(selectedRun?.next_actions) && selectedRun.next_actions.length > 0 && <section className="task-list"><h4>Next actions returned</h4>{selectedRun.next_actions.slice(0, 8).map((action, index) => <div className="agent-task-item" key={`${String(action)}-${index}`}><span className="task-number">{index + 1}</span><span>{String(action)}</span></div>)}</section>}
          {runQuery.isLoading ? <div className="caption" role="status">Loading {dataMode === "demo" ? "example" : "backend"} run history…</div> : runQuery.isError ? <div className="caption" role="alert">{apiConnectionMessage(runQuery.error)}</div> : runs.length ? <details className="agent-run-history"><summary>Run history · {runs.length}</summary>{runs.slice(0, showAllRuns ? runs.length : 8).map((run, index) => <button className="list-row" type="button" aria-pressed={String(selectedRun?.id ?? "") === String(run.id ?? "")} key={String(run.id ?? index)} onClick={() => setSelectedRunId(String(run.id ?? ""))}><Icon icon={FileClock}/><span className="flex1">{String(run.goal ?? run.task ?? run.agent ?? "Agent run")}</span><span className="caption">{String(run.status ?? "Recorded")}</span></button>)}{runs.length > 8 && <button className="btn sm section-space" type="button" onClick={() => setShowAllRuns(value => !value)}>{showAllRuns ? "Show recent only" : `Show all ${runs.length} returned runs`}</button>}</details> : <div className="empty agent-task-empty" role="status"><Icon icon={FileClock} size="lg"/><h3>No tasks yet</h3><p>No agent runs are recorded for this project.</p></div>}
          <section className="task-list" aria-label="Shared context"><h4>Shared context</h4>{selectedRun && Array.isArray(selectedRun.context_used) && selectedRun.context_used.length > 0 ? selectedRun.context_used.slice(0, 8).map((context, index) => <div className="agent-task-item" key={`${String(context)}-${index}`}><Icon icon={Database}/><span>{String(context)}</span></div>) : <p className="caption">{selectedRun ? "No context references were returned by the backend for this run." : "Context references appear after a backend run is recorded."}</p>}</section>
        </aside>
      </div>
    </motion.div>
  );
}

const automationTabs = ["Overview", "Configuration", "Runs", "Settings"];
const automationActionFields: Record<string, Array<{ key: string; label: string; multiline?: boolean; type?: string }>> = {
  add_event: [{ key: "title", label: "Title" }, { key: "content", label: "Content", multiline: true }, { key: "type", label: "Record type" }],
  run_agent: [{ key: "agent", label: "Agent ID" }, { key: "goal", label: "Goal", multiline: true }],
  search_context: [{ key: "query", label: "Search query" }],
  chat: [{ key: "message", label: "Message", multiline: true }, { key: "provider", label: "Provider" }],
  condition: [{ key: "condition", label: "Condition (JSON)", multiline: true }],
  file_read: [{ key: "path", label: "Workflow file path" }],
  file_list: [{ key: "path", label: "Workflow folder path" }],
  file_write: [{ key: "path", label: "Workflow file path" }, { key: "content", label: "Content", multiline: true }],
  spreadsheet_read: [{ key: "path", label: "CSV path" }],
  spreadsheet_write: [{ key: "path", label: "CSV path" }, { key: "rows", label: "Rows (JSON array)", multiline: true }],
  database_query: [{ key: "path", label: "Database path (__openagent__ for app database)" }, { key: "query", label: "Read-only SQL (SELECT)", multiline: true }],
  report_generate: [{ key: "path", label: "Report path" }, { key: "title", label: "Report title" }, { key: "sections", label: "Sections (JSON array)", multiline: true }],
  monitor_check: [{ key: "condition", label: "Condition (JSON)", multiline: true }, { key: "severity", label: "Severity" }, { key: "message", label: "Message" }],
  alert: [{ key: "severity", label: "Severity" }, { key: "message", label: "Message" }],
  browser_fetch: [{ key: "url", label: "URL", type: "url" }],
  api_request: [{ key: "url", label: "URL", type: "url" }, { key: "method", label: "HTTP method" }, { key: "body", label: "Request body (JSON)", multiline: true }],
  audit: [{ key: "audit_action", label: "Audit action" }, { key: "resource", label: "Resource" }, { key: "status", label: "Status" }],
};
const initialAutomationStep = () => ({ action: "add_event", title: "", content: "", type: "automation" });
const automationIcons: Record<string, LucideIcon> = { "file-text": FileText, code: Code2, "book-open": BookOpen, "chart-column": BarChart3, bell: Bell };
function parseStepValue(key: string, value: string): unknown {
  const field = key === "condition" || key === "sections" || key === "rows" || key === "body";
  if (!field || !value.trim()) return value;
  try { return JSON.parse(value); } catch { throw new Error(`${key} must contain valid JSON.`); }
}
function stepFieldText(value: unknown): string {
  return typeof value === "string" ? value : value == null ? "" : JSON.stringify(value, null, 2);
}
function normalizeStepDrafts(steps: Record<string, unknown>[]): Record<string, unknown>[] {
  return steps.map(step => {
    const action = String(step.action ?? "");
    if (!automationActionFields[action]) throw new Error(`Unsupported workflow step: ${action || "missing action"}.`);
    const normalized: Record<string, unknown> = { action };
    for (const field of automationActionFields[action]) {
      const value = step[field.key];
      if (value !== undefined && value !== "") normalized[field.key] = typeof value === "string" ? parseStepValue(field.key, value) : value;
    }
    return normalized;
  });
}
function stepResultRows(result: unknown): Record<string, unknown>[] {
  if (!result || typeof result !== "object") return [];
  const steps = (result as Record<string, unknown>).steps;
  return Array.isArray(steps) ? steps.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object") : [];
}

export function AutomationsScreen() {
  const dataMode = useWorkspaceState((state) => state.dataMode);
  const [tab, setTab] = useState("Overview");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [editName, setEditName] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [triggerType, setTriggerType] = useState("manual");
  const [intervalSeconds, setIntervalSeconds] = useState("3600");
  const [eventName, setEventName] = useState("");
  const [editSteps, setEditSteps] = useState<Record<string, unknown>[]>([initialAutomationStep()]);
  const [createSteps, setCreateSteps] = useState<Record<string, unknown>[]>([initialAutomationStep()]);
  const [allowNetworkConsent, setAllowNetworkConsent] = useState(false);
  const [expandedRunId, setExpandedRunId] = useState("");
  const [automationError, setAutomationError] = useState("");
  const [latestRun, setLatestRun] = useState<Record<string, unknown> | null>(null);
  const [traceStartSignal, setTraceStartSignal] = useState(0);
  const [demoRunDetail, setDemoRunDetail] = useState<Record<string, unknown> | null>(null);
  const [automationMenuOpen, setAutomationMenuOpen] = useState(false);
  useEffect(() => {
    if (!demoRunDetail) return;
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === "Escape") setDemoRunDetail(null); };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [demoRunDetail]);
  const queryClient = useQueryClient();
  const workflowsQuery = useQuery({ queryKey: ["workflows", dataMode], queryFn: ({ signal }) => demoCoreQuery<Record<string, unknown>>(dataMode, "shell", s => apiGet<{ workflows?: unknown[] }>("/api/workflows?project=default&limit=100", s), signal).then(data => dataMode === "demo" ? { workflows: Array.isArray(data.automations) ? data.automations.map((item, index) => { const workflow = item && typeof item === "object" ? item as Record<string, unknown> : {}; return { ...workflow, id: workflow.id ?? `demo-workflow-${index + 1}`, trigger_type: workflow.trigger_type ?? "manual", steps: Array.isArray(workflow.steps) ? workflow.steps : [{ action: "noop" }] }; }) : [] } : data), retry: false });
  const workflowRunsQuery = useQuery({ queryKey: ["workflow-runs", dataMode], queryFn: ({ signal }) => demoCoreQuery(dataMode, "shell", s => apiGet<{ runs?: unknown[] }>("/api/workflow-runs?project=default&limit=100", s), signal).then(data => dataMode === "demo" ? { runs: Array.isArray(data.runs) ? data.runs : [] } : data), retry: false });
  const builderQuery = useQuery({ queryKey: ["workflow-builder-schema", dataMode], queryFn: ({ signal }) => demoCoreQuery(dataMode, "shell", s => apiGet<{ actions?: Record<string, { label?: string; category?: string; requires_network?: boolean }> }>("/api/workflows/builder", s), signal).then(data => dataMode === "demo" ? { actions: { noop: { label: "Prepare example output" } } } : data), retry: false });
  const workflows = Array.isArray(workflowsQuery.data?.workflows) ? workflowsQuery.data.workflows.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object") : [];
  const runs = Array.isArray(workflowRunsQuery.data?.runs) ? workflowRunsQuery.data.runs.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object") : [];
  const filteredWorkflows = workflows.filter(workflow => !query || `${workflow.name ?? ""} ${workflow.description ?? ""}`.toLowerCase().includes(query.toLowerCase()));
  const selected = workflows.find(workflow => String(workflow.id) === selectedId) ?? null;
  const latestRunIsPreview = Boolean(latestRun && (latestRun.dry_run === true || (latestRun.result !== null && typeof latestRun.result === "object" && (latestRun.result as Record<string, unknown>).dry_run === true)));
  const liveOverviewRun = dataMode === "live"
    ? latestRun ?? runs.find(run => !selected || String(run.workflow_id ?? "") === String(selected.id))
    : null;
  const allowedActions = Object.entries(builderQuery.data?.actions ?? {}).filter(([action]) => Boolean(automationActionFields[action]));
  const createStepPayload = () => normalizeStepDrafts(createSteps);
  const editStepPayload = () => normalizeStepDrafts(editSteps);
  const changeStep = (setter: Dispatch<SetStateAction<Record<string, unknown>[]>>, index: number, key: string, value: string) => setter(previous => previous.map((step, position) => position === index ? { ...step, [key]: value } : step));
  const renderStepEditor = (steps: Record<string, unknown>[], setter: Dispatch<SetStateAction<Record<string, unknown>[]>>, prefix: string) => (
    <div className="section-space">
      <div className="row between"><strong>Workflow steps</strong><button className="btn sm" type="button" onClick={() => setter(previous => [...previous, initialAutomationStep()])}>Add step</button></div>
      {steps.map((step, index) => {
        const action = String(step.action ?? "");
        const fields = automationActionFields[action] ?? [];
        return <fieldset className="card padded section-space" key={`${prefix}-${index}`}><legend className="caption">Step {index + 1}</legend><div className="row between"><label className="demo-form-field flex1">Action<select className="select" aria-label={`${prefix} step ${index + 1} action`} value={action} onChange={event => setter(previous => previous.map((item, position) => position === index ? { action: event.target.value } : item))}><option value="">Select action</option>{allowedActions.map(([key, metadata]) => <option value={key} key={key}>{metadata.label ?? key}{metadata.requires_network ? " · network" : ""}</option>)}</select></label><button className="btn danger sm" type="button" aria-label={`Remove ${prefix} step ${index + 1}`} disabled={steps.length === 1} onClick={() => setter(previous => previous.filter((_, position) => position !== index))}>Remove</button></div>{fields.map(field => <label className="demo-form-field" key={field.key}>{field.label}{field.multiline ? <textarea className="textarea" rows={field.key === "content" || field.key === "message" ? 3 : 2} aria-label={`${prefix} step ${index + 1} ${field.label}`} value={stepFieldText(step[field.key])} onChange={event => changeStep(setter, index, field.key, event.target.value)} /> : <input className="input" type={field.type ?? "text"} aria-label={`${prefix} step ${index + 1} ${field.label}`} value={stepFieldText(step[field.key])} onChange={event => changeStep(setter, index, field.key, event.target.value)} />}</label>)}{builderQuery.data?.actions?.[action]?.requires_network && <p className="caption">This step needs network access. A live run keeps network disabled unless you explicitly consent at run time.</p>}</fieldset>;
      })}
    </div>
  );
  const createMutation = useMutation({
    mutationFn: async () => {
      const payload = { name: name.trim(), description: description.trim(), project: "default", trigger_type: "manual", trigger: {}, steps: createStepPayload(), enabled: true };
      if (dataMode !== "demo") return apiPost<{ workflow?: Record<string, unknown> }>("/api/workflows", payload);
      const workflow = { ...payload, id: `demo-workflow-${Date.now()}`, created_at: new Date().toISOString() };
      await demoProvider.updateShellCollection<Record<string, unknown>>("automations", rows => [...rows, workflow]);
      return { workflow };
    },
    onSuccess: async result => { setCreating(false); setName(""); setDescription(""); setCreateSteps([initialAutomationStep()]); setAutomationError(""); if (result.workflow?.id) setSelectedId(String(result.workflow.id)); await queryClient.invalidateQueries({ queryKey: ["workflows"] }); },
    onError: error => setAutomationError(apiConnectionMessage(error)),
  });
  const updateMutation = useMutation({
    mutationFn: async ({ id, patch }: { id: string; patch: Record<string, unknown> }) => {
      if (dataMode !== "demo") return apiPut<{ workflow?: Record<string, unknown> }>(`/api/workflows/${encodeURIComponent(id)}`, patch);
      let workflow: Record<string, unknown> | undefined;
      await demoProvider.updateShellCollection<Record<string, unknown>>("automations", rows => rows.map((row, index) => {
        if (String(row.id ?? `demo-workflow-${index + 1}`) !== id) return row;
        workflow = { ...row, ...patch, id };
        return workflow;
      }));
      if (!workflow) throw new Error("That demo automation no longer exists.");
      return { workflow };
    },
    onSuccess: async result => {
      if (result.workflow) {
        setEditName(String(result.workflow.name ?? ""));
        setEditDescription(String(result.workflow.description ?? ""));
      }
      setAutomationError("");
      await queryClient.invalidateQueries({ queryKey: ["workflows"] });
    },
    onError: error => setAutomationError(apiConnectionMessage(error)),
  });
  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      if (dataMode !== "demo") return apiDelete(`/api/workflows/${encodeURIComponent(id)}`);
      const rows = await demoProvider.updateShellCollection<Record<string, unknown>>("automations", current => current.filter((row, index) => String(row.id ?? `demo-workflow-${index + 1}`) !== id));
      if (rows.some(row => String(row.id ?? "") === id)) throw new Error("That demo automation could not be deleted.");
      return { deleted: true };
    },
    onSuccess: async () => { setSelectedId(""); setLatestRun(null); setAutomationError(""); await queryClient.invalidateQueries({ queryKey: ["workflows"] }); },
    onError: error => setAutomationError(apiConnectionMessage(error)),
  });
  const selectWorkflow = (workflow: Record<string, unknown>) => {
    setSelectedId(String(workflow.id));
    setLatestRun(null);
    setEditName(String(workflow.name ?? ""));
    setEditDescription(String(workflow.description ?? ""));
    setTriggerType(String(workflow.trigger_type ?? "manual"));
    const trigger = workflow.trigger && typeof workflow.trigger === "object" ? workflow.trigger as Record<string, unknown> : {};
    const config = trigger.config && typeof trigger.config === "object" ? trigger.config as Record<string, unknown> : {};
    setIntervalSeconds(String(config.interval_seconds ?? 3600));
    setEventName(String(config.event_name ?? ""));
    setEditSteps(Array.isArray(workflow.steps) ? workflow.steps.filter((step): step is Record<string, unknown> => Boolean(step) && typeof step === "object").map(step => ({ ...step })) : []);
    setAllowNetworkConsent(false);
  };
  useEffect(() => {
    if (dataMode !== "demo" || selectedId || !workflows.length) return;
    selectWorkflow(workflows[0]);
  }, [dataMode, selectedId, workflows]);
  const saveWorkflow = (event: FormEvent) => {
    event.preventDefault();
    if (!selected || !editName.trim()) return;
    const interval = Number(intervalSeconds);
    if (triggerType === "scheduled" && (!Number.isInteger(interval) || interval < 60)) {
      setAutomationError("Scheduled interval must be at least 60 seconds.");
      return;
    }
    if (triggerType === "event" && !eventName.trim()) {
      setAutomationError("Enter the event name that should trigger this workflow.");
      return;
    }
    updateMutation.mutate({ id: String(selected.id), patch: {
      name: editName.trim(), description: editDescription.trim(), trigger_type: triggerType,
      trigger: triggerType === "scheduled" ? { interval_seconds: interval } : triggerType === "event" ? { event_name: eventName.trim() } : {},
      steps: editStepPayload(),
    } });
  };
  const runMutation = useMutation({
    mutationFn: async ({ id, dryRun, allowNetwork = false }: { id: string; dryRun: boolean; allowNetwork?: boolean }) => {
      if (dataMode !== "demo") return apiPost<{ run?: Record<string, unknown> }>(`/api/workflows/${encodeURIComponent(id)}/run`, { allow_network: !dryRun && allowNetwork, dry_run: dryRun, project: "default" });
      const workflow = workflows.find(row => String(row.id) === id);
      if (!workflow) throw new Error("That demo automation no longer exists.");
      const run = {
        id: `demo-run-${Date.now()}`, workflow_id: id, status: "simulated", dry_run: dryRun,
        created_at: new Date().toISOString(),
        result: { dry_run: dryRun, steps: (Array.isArray(workflow.steps) ? workflow.steps : []).map((step, index) => ({ action: String((step as Record<string, unknown>)?.action ?? `Step ${index + 1}`), status: "simulated", output: "Local demo preview; no external action was performed." })), allow_network: false },
        demo: true, allow_network: false,
      };
      await demoProvider.updateShellCollection<Record<string, unknown>>("runs", rows => [run, ...rows]);
      return { run };
    },
    onSuccess: async result => { setLatestRun(result.run ?? result as Record<string, unknown>); setAllowNetworkConsent(false); if (dataMode === "demo" && result.run) setTraceStartSignal(value => value + 1); await queryClient.invalidateQueries({ queryKey: ["workflow-runs"] }); },
    onError: error => setAutomationError(apiConnectionMessage(error)),
  });
  const renderRunHistory = () => {
    if (workflowRunsQuery.isLoading) return <div className="caption" role="status">Loading run history…</div>;
    if (workflowRunsQuery.isError) return <div role="alert">{apiConnectionMessage(workflowRunsQuery.error)}</div>;
    if (dataMode === "demo") return <div className="card padded section-space"><div className="section-head"><h3>Recent runs</h3></div>{runs.map((run, index) => {
      const failed = String(run.status ?? "").toLowerCase() === "error";
      return <div className="run-record" key={String(run.id ?? `${String(run.time ?? "run")}-${index}`)}><span className={`run-dot${failed ? " error" : ""}`}>{failed ? <X className="icon xs"/> : <Check className="icon xs"/>}</span><div><strong>{String(run.time ?? "Example run")}</strong><div className="caption">{String(run.duration ?? "")}</div></div><span className="muted">{String(run.info ?? "")}</span><button className="btn sm" type="button" onClick={() => setDemoRunDetail(run)}>{failed ? "View error" : "View output"}</button></div>;
    })}</div>;
    const matchingRuns = runs.filter(run => !selected || run.workflow_id === selected.id);
    if (!matchingRuns.length) return <Disconnected icon={FileClock} title="No run history" detail="The backend returned no workflow runs."/>;
    return matchingRuns.map((run, index) => {
      const runId = String(run.id ?? index);
      const result = run.result && typeof run.result === "object" ? run.result : {};
      const steps = stepResultRows(result);
      return <article className="card padded section-space" key={runId}><button className="row between" type="button" aria-expanded={expandedRunId === runId} onClick={() => setExpandedRunId(value => value === runId ? "" : runId)} style={{ width: "100%", textAlign: "left" }}><span><strong>{String(run.workflow_id ?? "Workflow run")}</strong><span className="caption" style={{ display: "block" }}>{String(run.created_at ?? "")}</span></span><span className="caption">{String(run.status ?? "recorded")}</span></button>{expandedRunId === runId && <div className="section-space">{steps.length ? steps.map((step, stepIndex) => <article className="list-row" key={`${runId}-${stepIndex}`}><span className="flex1">{String(step.action ?? `Step ${stepIndex + 1}`)}</span><span className="caption">{String(step.status ?? "returned")}</span>{Boolean(step.error) && <span role="alert" className="caption">{String(step.error)}</span>}</article>) : <p className="caption">No step details were included in this backend run record.</p>}</div>}</article>;
    });
  };
  return (
    <motion.div className="automations-shell" {...pageMotion}>
      <section className="automation-list">
        <div className="row between" style={{ marginBottom: 15 }}>
          <div><h1 style={{ fontSize: 24 }}>Automations</h1><p className="muted small">Scheduled work, clearly monitored</p></div>
          <button className="btn primary" type="button" onClick={() => { setCreating(value => !value); setAutomationError(""); }}>{creating ? "Close" : <><Plus className="icon sm"/> New</>}</button>
        </div>
        <label className="search-field"><Icon icon={Search} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search automations..." aria-label="Search automations" /></label>
        {dataMode === "demo" && <div className="row gap6 section-space"><span className="tag green">{workflows.filter(workflow => workflow.enabled !== false).length} enabled</span><span className="tag">{workflows.filter(workflow => workflow.enabled === false).length} paused</span></div>}
        {creating && <form className="card padded section-space" onSubmit={event => { event.preventDefault(); if (name.trim() && allowedActions.length && !builderQuery.isError) createMutation.mutate(); }}><strong>New automation</strong><label className="demo-form-field">Name<input className="input" value={name} onChange={event => setName(event.target.value)} required maxLength={120} /></label><label className="demo-form-field">Description<textarea className="textarea" rows={2} value={description} onChange={event => setDescription(event.target.value)} /></label>{builderQuery.isLoading ? <p className="caption" role="status">Loading supported workflow actions…</p> : builderQuery.isError ? <p className="caption" role="alert">Builder schema unavailable: {apiConnectionMessage(builderQuery.error)}</p> : allowedActions.length ? renderStepEditor(createSteps, setCreateSteps, "New") : <p className="caption" role="alert">No supported workflow step types are available.</p>}<p className="caption">{dataMode === "demo" ? "Local demo saves an isolated workflow definition; no workflow is executed until you request a simulation." : "Steps are validated by the backend. Creation saves the definition; it does not execute it."}</p><button className="btn primary" type="submit" disabled={!name.trim() || !allowedActions.length || createMutation.isPending}>{createMutation.isPending ? "Saving…" : "Save automation"}</button></form>}
        {automationError && <div className="card padded small section-space" role="alert">{automationError}</div>}
        <div className="section-space">{workflowsQuery.isLoading ? <div className="caption" role="status">Loading automations…</div> : workflowsQuery.isError ? <Disconnected icon={Workflow} title="Automation service unavailable" detail={apiConnectionMessage(workflowsQuery.error)} /> : filteredWorkflows.length ? filteredWorkflows.map(workflow => { const id = String(workflow.id ?? ""); const IconComponent = automationIcons[String(workflow.icon ?? "")] ?? Workflow; const schedule = String(workflow.schedule ?? (workflow.trigger_type === "scheduled" ? "Scheduled interval" : workflow.trigger_type === "event" ? "Event triggered" : "Manual run")); const next = String(workflow.next ?? (workflow.enabled === false ? "Paused" : "Ready when requested")); return <div key={id} role="button" tabIndex={0} aria-current={selectedId === id ? "true" : undefined} onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectWorkflow(workflow); } }} onClick={() => selectWorkflow(workflow)} className={`automation-record ${selectedId === id ? "active" : ""}`}><span className="square-icon blue"><IconComponent size={17}/></span><div className="flex1"><div className="row between"><h3>{String(workflow.name ?? workflow.id)}</h3><input type="checkbox" className="toggle" checked={workflow.enabled !== false} aria-label={`Toggle ${String(workflow.name ?? "automation")}`} onClick={event => event.stopPropagation()} onChange={event => updateMutation.mutate({ id, patch: { enabled: event.currentTarget.checked } })}/></div><p className="small muted" style={{ margin: "5px 0" }}>{String(workflow.description ?? "")}</p><div className="row between caption"><Clock3 className="icon sm"/>{schedule}</div><p className="caption">Next: {next}</p></div></div>; }) : <Disconnected icon={Workflow} title={query ? "No matching automations" : "No automations"} detail="The backend returned no workflows for this project." />}</div>
      </section>
      <section className="automation-detail">
        <div className="row between automation-detail-heading" style={{ position: "relative" }}><div className="row align-start">{selected && <span className="square-icon blue"><Icon icon={automationIcons[String(selected.icon ?? "")] ?? Workflow}/></span>}<div>{selected ? <><h2 style={{ fontSize: 21, fontWeight: 700 }}>{String(selected.name ?? selected.id)}</h2><p className="muted small">{String(selected.description ?? "")}</p><div style={{ marginTop: 6 }}><span className={`tag ${selected.enabled === false ? "orange" : "green"}`}>{selected.enabled === false ? "Paused" : "Enabled"}</span></div></> : <><h2 style={{ fontSize: 21 }}>Automation details</h2><p className="muted small">Select an automation to inspect its configuration and run history.</p></>}</div></div>{selected && <div className="row"><button className="btn sm automation-back" type="button" onClick={() => setSelectedId("")}>← Back</button><button className="icon-button" type="button" aria-label="Automation options" aria-expanded={automationMenuOpen} onClick={() => setAutomationMenuOpen(value => !value)}><MoreHorizontal size={18}/></button>{automationMenuOpen && <div className="card" style={{ position: "absolute", top: 36, right: 0, zIndex: 5, display: "grid", gap: 6, padding: 10 }}><button className="btn sm" type="button" disabled={updateMutation.isPending} onClick={() => { updateMutation.mutate({ id: String(selected.id), patch: { enabled: selected.enabled === false } }); setAutomationMenuOpen(false); }}>{selected.enabled === false ? "Enable automation" : "Pause automation"}</button><button className="btn danger sm" type="button" disabled={deleteMutation.isPending} onClick={() => { setAutomationMenuOpen(false); if (window.confirm(`Delete “${String(selected.name ?? selected.id)}”? This cannot be undone.`)) deleteMutation.mutate(String(selected.id)); }}>Delete automation</button></div>}</div>}</div>
        <div className="section-space"><Tabs items={automationTabs} selected={tab} onSelect={setTab} /></div>
        {dataMode === "demo" && tab === "Overview" && traceStartSignal > 0 && <ExecutionTrace route="automations" startSignal={traceStartSignal} />}
        {selected && tab === "Configuration" && <form className="card padded section-space" onSubmit={saveWorkflow}><label className="demo-form-field">Name<input className="input" value={editName} onChange={event => setEditName(event.target.value)} required maxLength={120}/></label><label className="demo-form-field">Description<textarea className="textarea" rows={3} value={editDescription} onChange={event => setEditDescription(event.target.value)}/></label><label className="demo-form-field">Trigger<select className="select" value={triggerType} onChange={event => setTriggerType(event.target.value)}><option value="manual">Manual</option><option value="scheduled">Scheduled interval</option><option value="event">Event</option></select></label>{triggerType === "scheduled" && <label className="demo-form-field">Run every (seconds)<input className="input" type="number" min="60" step="1" value={intervalSeconds} onChange={event => setIntervalSeconds(event.target.value)} required/><span className="caption">{dataMode === "demo" ? "This interval is saved as local demo metadata; no scheduler will run." : "The backend scheduler stores this interval; it does not guarantee an exact start time."}</span></label>}{triggerType === "event" && <label className="demo-form-field">Event name<input className="input" value={eventName} onChange={event => setEventName(event.target.value)} required maxLength={120} placeholder="e.g. document.created"/><span className="caption">{dataMode === "demo" ? "This trigger is an example setting only; no event listener is registered." : "The workflow runs when the backend receives this event name."}</span></label>}{builderQuery.isError ? <p className="caption" role="alert">Builder schema unavailable: {apiConnectionMessage(builderQuery.error)}</p> : renderStepEditor(editSteps, setEditSteps, "Edit")}<div className="row between"><span className="caption">{dataMode === "demo" ? "Changes are saved in isolated local demo data." : "Saving changes the backend definition. No workflow runs until requested."}</span><button className="btn primary" type="submit" disabled={!editName.trim() || updateMutation.isPending || builderQuery.isLoading}>{updateMutation.isPending ? "Saving…" : "Save configuration"}</button></div></form>}
        {selected && tab === "Overview" && <><div className="metric-strip"><div><Clock3 className="icon sm"/><strong>Schedule</strong><p className="muted small">{String(selected.schedule ?? (selected.trigger_type === "scheduled" ? "Scheduled interval" : "Manual run"))}</p></div><div>{dataMode === "demo" || String(liveOverviewRun?.status ?? "").toLowerCase() === "completed" ? <Check className="icon sm"/> : <FileClock className="icon sm"/>}<strong>Last run</strong><p className="muted small">{dataMode === "demo" ? "Example · Completed" : liveOverviewRun ? String(liveOverviewRun.status ?? "Status unavailable") : "No runs reported"}</p></div><div><CalendarDays className="icon sm"/><strong>Next run</strong><p className="muted small">{String(selected.next ?? (selected.enabled === false ? "Paused" : "When requested"))}</p></div></div><div className="card padded"><h3>Instructions</h3><p className="muted small" style={{ marginTop: 9 }}>{String(selected.instruction ?? selected.description ?? "No instructions have been configured.")}</p><div className="row" style={{ marginTop: 13 }}><span className="tag">Research</span><span className="tag">Summarise</span><span className="tag">Save result</span></div></div><div className="card padded section-space row between"><div><h3>Test run</h3><p className="caption">Preview the running animation and status layout.</p></div><button className="btn sm primary" type="button" disabled={runMutation.isPending || selected.enabled === false} onClick={() => { setAutomationError(""); runMutation.mutate({ id: String(selected.id), dryRun: true }); }}>{runMutation.isPending ? "Requesting preview…" : "Preview run"}</button></div>{dataMode === "demo" && <div className="card padded section-space"><div className="section-head"><h3>Recent runs</h3><button className="text-link" type="button" onClick={() => setTab("Runs")}>View all runs</button></div>{runs.slice(0, 4).map((run, index) => { const failed = String(run.status ?? "").toLowerCase() === "error"; return <div className="run-record" key={String(run.id ?? `${String(run.time ?? "run")}-${index}`)}><span className={`run-dot${failed ? " error" : ""}`}>{failed ? <X className="icon xs"/> : <Check className="icon xs"/>}</span><div><strong>{String(run.time ?? "Example run")}</strong><div className="caption">{String(run.duration ?? "")}</div></div><span className="muted">{String(run.info ?? "")}</span><button className="btn sm" type="button" onClick={() => setDemoRunDetail(run)}>{failed ? "View error" : "View output"}</button></div>; })}</div>}</>}
        {selected && tab === "Settings" && <div className="card padded section-space"><h3>Schedule and status</h3><div className="settings-line"><span className="flex1">Automation</span><strong>{String(selected.enabled === false ? "Disabled" : "Enabled")}</strong></div><div className="settings-line"><span className="flex1">Trigger</span><strong>{String(selected.trigger_type ?? "manual")}</strong></div>{Boolean(selected.trigger && typeof selected.trigger === "object" && (selected.trigger as Record<string, unknown>).next_run_at) && <div className="settings-line"><span className="flex1">Next scheduled run</span><strong>{new Date(String((selected.trigger as Record<string, unknown>).next_run_at)).toLocaleString()}</strong></div>}<p className="caption section-space">Change schedule details in Configuration. Disable an automation to prevent scheduled execution.</p></div>}
        {latestRun && <div className="card padded section-space" role="status"><div className="row between"><strong>{latestRunIsPreview ? "Preview result" : "Run result"}</strong><span className="caption">{String(latestRun.status ?? "Response received")}</span></div>{dataMode === "demo" && <p className="caption section-space">Local demo simulation · no backend or external services were contacted.</p>}<p className="caption section-space">Run {String(latestRun.id ?? "recorded by backend")}{latestRun.created_at ? ` · ${String(latestRun.created_at)}` : ""}</p>{stepResultRows(latestRun.result).map((step, index) => <article className="list-row" key={`${String(latestRun.id ?? "latest")}-${index}`}><span className="flex1">{String(step.action ?? `Step ${index + 1}`)}</span><span className="caption">{String(step.status ?? "returned")}</span>{Boolean(step.error) && <span className="caption" role="alert">{String(step.error)}</span>}</article>)}{latestRunIsPreview && <div className="section-space"><p className="caption">{dataMode === "demo" ? "A local demo run is simulated without contacting external services." : "Live execution is a separate request and can create side effects. Network access stays disabled by default."}</p>{dataMode === "live" && <><label className="settings-line"><input type="checkbox" checked={allowNetworkConsent} onChange={event => setAllowNetworkConsent(event.target.checked)} />Allow network access for this live run</label><button className="btn danger sm section-space" type="button" disabled={runMutation.isPending} onClick={() => { const message = allowNetworkConsent ? "Run these workflow steps now with network access enabled? This may create external side effects." : "Run these workflow steps now with network access disabled? This may create side effects in configured services or workflow files."; if (window.confirm(message)) { setAutomationError(""); runMutation.mutate({ id: String(selected?.id ?? latestRun.workflow_id ?? ""), dryRun: false, allowNetwork: allowNetworkConsent }); } }}>Confirm live run</button></>}{dataMode === "demo" && <button className="btn sm section-space" type="button" disabled={runMutation.isPending} onClick={() => runMutation.mutate({ id: String(selected?.id ?? latestRun.workflow_id ?? ""), dryRun: false })}>{runMutation.isPending ? "Simulating…" : "Run local demo"}</button>}</div>}</div>}
        {tab === "Runs" && renderRunHistory()}
        {!selected && tab !== "Runs" && <Disconnected icon={Workflow} title="No automation selected" detail="Choose an automation from the backend list." />}
        <div className="row caption section-space"><Info className="icon sm" aria-hidden="true" />{dataMode === "demo" ? "Automation edits and run records stay in isolated local demo data." : "Runs and progress shown here come from backend responses."}</div>
      </section>
      {demoRunDetail && <div className="overlay-root" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) setDemoRunDetail(null); }}><section className="modal" role="dialog" aria-modal="true" aria-labelledby="automation-run-detail-title"><div className="row between"><h2 id="automation-run-detail-title">Run details</h2><button className="icon-btn" type="button" aria-label="Close" onClick={() => setDemoRunDetail(null)}><X className="icon sm"/></button></div><p>Example run history from your offline workspace.</p><div className="demo-detail-grid"><div><small>Started</small><strong>{String(demoRunDetail.time ?? "Not reported")}</strong></div><div><small>Status</small><strong>{String(demoRunDetail.status ?? "Not reported")}</strong></div><div><small>Summary</small><strong>{String(demoRunDetail.info ?? "No example output was reported")}</strong></div></div></section></div>}
    </motion.div>
  );
}

const modelTabs = ["All models", "Configured", "Local", "Setup needed"];
const modelPanels = ["Inventory", "Setup activity", "Marketplace & runtime"];
type ModelRow = Record<string, unknown>;
const modelRows = (value: unknown, key: string): ModelRow[] => {
  if (!value || typeof value !== "object") return [];
  const rows = (value as Record<string, unknown>)[key];
  return Array.isArray(rows) ? rows.filter((row): row is ModelRow => Boolean(row) && typeof row === "object") : [];
};
const modelRecord = (value: unknown): ModelRow => value && typeof value === "object" ? value as ModelRow : {};
const jobTerminal = (job: ModelRow) => ["completed", "failed", "cancelled"].includes(String(job.status ?? "").toLowerCase());
const activeProject = () => new URL(window.location.href).searchParams.get("project") || localStorage.getItem("oa-project") || "default";
const fixtureModelGlyph = (name: unknown): LucideIcon => ({ bot: Bot, brain: Brain, sparkles: Sparkles, cpu: Cpu }[String(name ?? "").toLowerCase()] ?? Cpu);
const fixtureModelTone = (name: unknown): string => ({ blue: "blue", violet: "violet", green: "green", orange: "orange", red: "red" }[String(name ?? "").toLowerCase()] ?? "blue");

export function ModelsScreen() {
  const dataMode = useWorkspaceState((state) => state.dataMode);
  const [tab, setTab] = useState("All models");
  const [panel, setPanel] = useState("Inventory");
  const [query, setQuery] = useState("");
  const [allowNetwork, setAllowNetwork] = useState(false);
  const [selectedProviderId, setSelectedProviderId] = useState("");
  const [demoSelectedModel, setDemoSelectedModel] = useState(() => {
    try { return localStorage.getItem("oa-second-model-selected") || "GPT-4o"; } catch { return "GPT-4o"; }
  });
  const [demoDetailsModel, setDemoDetailsModel] = useState<ModelRow | null>(null);
  const [traceStartSignal, setTraceStartSignal] = useState(0);
  const [demoFilterOpen, setDemoFilterOpen] = useState(false);
  const [demoStatusFilter, setDemoStatusFilter] = useState("All statuses");
  useEffect(() => {
    if (!demoDetailsModel) return;
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === "Escape") setDemoDetailsModel(null); };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [demoDetailsModel]);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [modelName, setModelName] = useState("");
  const [modelSource, setModelSource] = useState("");
  const [modelPath, setModelPath] = useState("");
  const [probe, setProbe] = useState<ModelRow | null>(null);
  const [benchmarkResult, setBenchmarkResult] = useState<ModelRow | null>(null);
  const queryClient = useQueryClient();
  const providersQuery = useQuery({ queryKey: ["models", "providers", dataMode], queryFn: ({ signal }) => demoCollectionQuery(dataMode, "models", "All models", s => apiGet<{ providers?: unknown[] }>("/api/models", s), signal).then(data => dataMode === "demo" ? { providers: data.providers ?? [] } : data), retry: false });
  const localModelsQuery = useQuery({ queryKey: ["models", "local", dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "shell.localModels", s => apiGet<Record<string, unknown>>("/api/local-ai/models", s), signal).then(data => dataMode === "demo" ? { models: data.models ?? [] } : data), retry: false });
  const hardwareQuery = useQuery({ queryKey: ["models", "hardware", dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "shell.hardware", s => apiGet<Record<string, unknown>>("/api/local-ai/hardware", s), signal), retry: false });
  const policyQuery = useQuery({ queryKey: ["models", "policy", dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "shell.modelPolicy", s => apiGet<Record<string, unknown>>("/api/model-policy", s), signal), retry: false });
  const setupQuery = useQuery({ queryKey: ["models", "setup", dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "shell.modelSetup", s => apiGet<Record<string, unknown>>("/api/local-ai/setup/status", s), signal), retry: false });
  const marketplaceQuery = useQuery({ queryKey: ["models", "marketplace", dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "shell.marketplace", s => apiGet<Record<string, unknown>>("/api/local-ai/marketplace", s), signal).then(data => dataMode === "demo" ? { models: data.models ?? [] } : data), retry: false });
  const adaptersQuery = useQuery({ queryKey: ["models", "adapters", dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "shell.adapters", s => apiGet<Record<string, unknown>>("/api/local-ai/adapters", s), signal).then(data => dataMode === "demo" ? { adapters: data.adapters ?? [] } : data), retry: false });
  const jobsQuery = useQuery({
    queryKey: ["models", "jobs", activeProject(), dataMode],
    queryFn: ({ signal }) => demoQuery(dataMode, "shell.modelJobs", s => apiGet<{ jobs?: unknown[] }>(`/api/local-ai/jobs?project=${encodeURIComponent(activeProject())}&limit=12`, s), signal),
    retry: false,
    refetchInterval: query => modelRows(query.state.data, "jobs").some(job => !jobTerminal(job)) ? 1500 : false,
  });
  const providers = modelRows(providersQuery.data, "providers");
  const selectedProvider = providers.find(provider => String(provider.id ?? "") === selectedProviderId) ?? providers[0];
  const localModels = modelRows(localModelsQuery.data, "models");
  const jobs = modelRows(jobsQuery.data, "jobs");
  const catalog = modelRows(marketplaceQuery.data, "models");
  const adapters = modelRows(adaptersQuery.data, "adapters");
  const hardware = hardwareQuery.data ?? {};
  const policy = policyQuery.data ?? {};
  const setup = setupQuery.data ?? {};
  const loading = providersQuery.isLoading || localModelsQuery.isLoading;
  const mutation = useMutation({
    mutationFn: ({ path, body, method }: { path: string; body?: unknown; method?: "GET" | "POST" }) => dataMode === "demo" ? modelDemoAction(path, body) as Promise<ModelRow> : method === "GET" ? apiGet<ModelRow>(path) : apiPost<ModelRow>(path, body ?? {}),
    onSuccess: async (result, variables) => {
      setError("");
      if (variables.path.endsWith("/probe")) setProbe(result);
      else if (variables.path.includes("/jobs/cancel")) setNotice(dataMode === "demo" ? "Demo setup record marked cancelled locally. No download was running." : "Cancellation request recorded by the backend.");
      else if (variables.path.includes("/setup/job")) { setNotice(dataMode === "demo" ? "Local demo setup preview created. No files were downloaded or installed." : "Model setup job accepted. Progress will update from backend job records."); if (dataMode === "demo") { setPanel("Setup activity"); setTraceStartSignal(value => value + 1); } }
      else if (variables.path.includes("/models/register")) { setNotice(dataMode === "demo" ? "Model metadata added to local demo fixtures. No model was installed." : "Model metadata registered."); setModelName(""); setModelSource(""); setModelPath(""); }
      else if (variables.path.includes("/models/delete")) setNotice(dataMode === "demo" ? "Local demo model metadata removed. No files were deleted." : "Model removal request completed.");
      else if (variables.path.includes("/models/download")) setNotice(dataMode === "demo" ? "Model downloads are unavailable in demo mode; no network request was made." : "Model download request completed.");
      else if (variables.path.includes("/discover")) setNotice(dataMode === "demo" ? "Demo catalogue checked locally. No provider was contacted." : "Discovery response received from the backend.");
      else if (variables.path === "/api/local-ai/benchmark") { setBenchmarkResult(modelRecord(result.benchmark)); setNotice(dataMode === "demo" ? "Benchmark disabled in demo mode; no performance was measured." : "Benchmark result received from the backend."); }
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["models", "providers"] }),
        queryClient.invalidateQueries({ queryKey: ["models", "local"] }),
        queryClient.invalidateQueries({ queryKey: ["models", "jobs"] }),
        queryClient.invalidateQueries({ queryKey: ["models", "marketplace"] }),
        queryClient.invalidateQueries({ queryKey: ["models", "setup"] }),
      ]);
    },
    onError: cause => { setNotice(""); setError(apiConnectionMessage(cause)); },
  });
  const allCards: (ModelRow & { _kind: "provider" | "local-model" })[] = [
    ...providers.map(item => ({ ...item, _kind: "provider" as const })),
    ...localModels.map(item => ({ ...item, _kind: "local-model" as const })),
  ];
  const visibleCards = allCards.filter(item => {
    const label = String(item.label ?? item.name ?? item.id ?? "");
    // Keep the category tabs aligned with the card status badge below. A configured
    // provider is available even when its fixture does not set `available: true`.
    const available = item.available === true || ["ready", "available", "installed", "downloaded", "configured"].includes(String(item.status ?? "").toLowerCase());
    const local = item._kind === "local-model" || item.local === true || ["local", "ollama", "llamacpp", "openvino-direct", "native-publisher"].includes(String(item.id ?? item.provider ?? "").toLowerCase());
    return (!query || `${label} ${item.provider ?? ""} ${item.status ?? ""}`.toLowerCase().includes(query.toLowerCase())) &&
      (dataMode !== "demo" || demoStatusFilter === "All statuses" || String(item.status ?? "") === demoStatusFilter) &&
      (tab === "All models" || tab === "Configured" && available || tab === "Local" && local || tab === "Setup needed" && !available);
  });
  const refresh = () => { void queryClient.invalidateQueries({ queryKey: ["models"] }); };
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!modelName.trim()) return;
    mutation.mutate({ path: "/api/local-ai/models/register", body: { name: modelName.trim(), provider: "local", source_url: modelSource.trim(), local_path: modelPath.trim(), metadata: {} } });
  };
  const requestProbe = (model: ModelRow) => {
    const id = String(model.id ?? ""); if (!id) return;
    setProbe({ id, name: model.name, pending: true });
    mutation.mutate({ path: `/api/local-ai/models/${encodeURIComponent(id)}/probe`, method: "GET" });
  };
  const setupModel = (item: ModelRow) => {
    const name = String(modelRecord(item.metadata).model ?? "");
    if (!name) { setError("This catalog record does not include a supported model identifier."); return; }
    const confirmation = dataMode === "demo"
      ? `Create a local setup preview for ${name}? This will not download or install files.`
      : `Queue local model setup for ${name}? ${allowNetwork ? "Network downloads are explicitly enabled for this request." : "Network downloads remain disabled."}`;
    if (!window.confirm(confirmation)) return;
    mutation.mutate({ path: "/api/local-ai/models/setup/job", body: { task: "chat", model: name, allow_network: allowNetwork, project: activeProject() } });
  };
  const benchmarkModel = (item: ModelRow) => {
    const id = String(item.id ?? "");
    if (!id) return;
    const provider = providers.find(candidate => String(candidate.id ?? "") === String(item.provider ?? ""));
    if (provider?.local === false) { setError("This model is configured with an external provider. Benchmarking is disabled here to avoid sending a prompt without an explicit provider-consent flow."); return; }
    if (!window.confirm(`Run the backend benchmark for ${String(item.name ?? id)}? It sends a short fixed test prompt to this model and records timing and hardware details.`)) return;
    setError(""); setNotice(""); setBenchmarkResult(null);
    mutation.mutate({ path: "/api/local-ai/benchmark", body: { model_id: id, prompt: "Hello OpenAgent" } });
  };
  const removeModel = (model: ModelRow) => {
    const id = String(model.id ?? ""); if (!id) return;
    const name = String(model.name ?? id);
    const deleteFile = window.confirm(`Remove ${name} and delete its managed file if it is inside the OpenAgent model store? Choose Cancel to keep the file and unregister only.`);
    if (!window.confirm(`Confirm removal of ${name}? Dependent resources may prevent this operation.`)) return;
    mutation.mutate({ path: "/api/local-ai/models/delete", body: { model_id: id, delete_file: deleteFile } });
  };
  const hardwareFacts = [hardware.system && hardware.machine ? `${String(hardware.system)} · ${String(hardware.machine)}` : hardware.system ?? hardware.machine, hardware.cpu_count != null ? `${String(hardware.cpu_count)} CPU cores` : null, hardware.ram_gb != null ? `${String(hardware.ram_gb)} GB memory` : null, modelRecord(hardware.gpu).available === true ? `GPU · ${String(modelRecord(hardware.gpu).adapter ?? "Detected")}` : null].filter(Boolean);
  const recommendations = modelRecord(setup.recommended);
  const runtimeEntries = [["Ollama", modelRecord(setup.ollama)], ["llama.cpp", modelRecord(setup.llamacpp)]] as const;
  return (
    <motion.div className="page" {...pageMotion}>
      <div className={`models-heading${dataMode === "demo" ? " models-heading-demo" : ""}`}><PageHeading title="Models" subtitle="Choose the right model for each task." /><div className={dataMode === "demo" ? "models-demo-heading-actions" : undefined}><details className="models-advanced-tools" open={panel !== "Inventory"}><summary className="btn sm primary">{dataMode === "demo" && <Icon icon={Plus}/ >}{dataMode === "demo" ? "Add provider" : "Provider setup"}</summary><div className="models-advanced-panel"><label className="models-network-choice"><input type="checkbox" checked={allowNetwork} onChange={event => setAllowNetwork(event.target.checked)} /><span>Allow network for model downloads and setup</span></label><div className="row"><button className="btn sm" type="button" onClick={refresh} disabled={loading}>Refresh</button><button className="btn sm" type="button" onClick={() => { setError(""); setNotice(""); mutation.mutate({ path: "/api/models/discover", body: {} }); }} disabled={mutation.isPending}>Discover models</button></div><div className="models-advanced-tabs"><Tabs items={modelPanels} selected={panel} onSelect={setPanel} /></div><details className="model-provider-details"><summary><span><strong>Provider availability</strong><small>Live status and setup details</small></span><span className="caption">{providers.length}</span></summary><div className="model-provider-list">{providersQuery.isError ? <p className="caption" role="alert">{apiConnectionMessage(providersQuery.error)}</p> : providers.map(provider => <button className="list-row" type="button" aria-pressed={String(selectedProvider?.id ?? "") === String(provider.id ?? "")} key={String(provider.id)} onClick={() => setSelectedProviderId(String(provider.id ?? ""))}><Icon icon={provider.available === true ? Cpu : CloudOff}/><span className="flex1 small">{String(provider.label ?? provider.id)} · {provider.available === true ? "Available" : provider.available === false ? "Unavailable" : "Not reported"}</span></button>)}</div>{selectedProvider && <div className="model-provider-facts"><div className="settings-line"><span className="flex1">Default model</span><span>{String(selectedProvider.default_model ?? "Not configured")}</span></div><div className="settings-line"><span className="flex1">Readiness</span><span>{selectedProvider.default_model_ready === true ? "Ready" : selectedProvider.default_model_ready === false ? "Not installed" : "Not reported"}</span></div><div className="settings-line"><span className="flex1">Network</span><span>{selectedProvider.network === true ? "Enabled" : selectedProvider.network === false ? "Disabled" : "Not reported"}</span></div>{Array.isArray(selectedProvider.installed_models) && selectedProvider.installed_models.length > 0 && <div className="section-space"><strong className="caption">Installed with provider</strong>{selectedProvider.installed_models.map((model, index) => <div className="list-row small" key={`${String(model)}-${index}`}>{String(model)}</div>)}</div>}</div>}</details></div></details>{dataMode === "demo" && <span className="btn sm models-live-demo"><span className="status-dot" aria-hidden="true" />Live demo</span>}</div></div>
      {(error || notice) && <div className="card padded section-space" role={error ? "alert" : "status"}>{error || notice}</div>}
      {dataMode === "demo" && panel === "Setup activity" && traceStartSignal > 0 && <ExecutionTrace route="models" startSignal={traceStartSignal} />}
      <div className="models-shell">
        <div>
          {panel === "Inventory" && <>
            <div className="models-filterbar"><Tabs items={modelTabs} selected={tab} onSelect={setTab} /><div className="row" style={{ gridColumn: 1, width: "100%", gap: 10, alignItems: "center" }}><label className="search-field" style={dataMode === "demo" ? { flex: "0 0 200px", width: 200 } : undefined}><Icon icon={Search} /><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Search models…" aria-label="Search models" /></label>{dataMode === "demo" && <div style={{ position: "relative" }}><button className="btn sm" type="button" aria-expanded={demoFilterOpen} onClick={() => setDemoFilterOpen(open => !open)}><Icon icon={Filter}/>Filter</button>{demoFilterOpen && <div className="card padded" role="group" aria-label="Filter models by status" style={{ position: "absolute", zIndex: 4, right: 0, top: "calc(100% + 6px)", width: 190, display: "grid", gap: 7 }}><label className="caption">Status<select className="select" value={demoStatusFilter} onChange={event => setDemoStatusFilter(event.target.value)}><option>All statuses</option><option>Configured</option><option>Setup required</option><option>Not installed</option></select></label><button className="btn sm" type="button" onClick={() => { setDemoStatusFilter("All statuses"); setDemoFilterOpen(false); }}>Clear filters</button></div>}</div>}</div></div>
            {loading ? <div className="card padded" role="status">Loading provider and local model inventory…</div> : visibleCards.length ? visibleCards.map((item, index) => {
              const local = item._kind === "local-model";
              const label = String(item.label ?? item.name ?? item.id ?? "Model");
              const remoteSource = /^https?:\/\//i.test(String(item.source_url ?? ""));
              const localPath = typeof item.local_path === "string" && item.local_path.length > 0;
              const sourceUrl = typeof item.source_url === "string" && item.source_url.length > 0;
              const provider = providers.find(candidate => String(candidate.id ?? "") === String(item.provider ?? ""));
              const available = item.available === true || ["ready", "available", "installed", "downloaded", "configured"].includes(String(item.status ?? "").toLowerCase());
              const statusText = item.available === true ? "Available" : item.available === false ? "Unavailable" : String(item.status ?? "Status not reported");
              const capabilities = Array.isArray(item.capabilities) ? item.capabilities.filter((value): value is string => typeof value === "string") : [];
              const locationLabel = local ? `Local model · ${String(provider?.label ?? item.provider ?? "provider not reported")}` : `${item.local === true ? "Local provider" : item.local === false ? "External provider" : "Provider"} · ${String(item.id ?? "unknown")}`;
              const description = String(item.description ?? item.privacy_warning ?? (local ? "Registered model metadata and runtime state from the backend." : "Provider availability and default model from the backend."));
              const fixtureModel = dataMode === "demo" && Array.isArray(item.caps);
              const fixtureCaps = fixtureModel && Array.isArray(item.caps) ? item.caps.filter((value): value is string => typeof value === "string") : capabilities;
              const fixtureStatus = String(item.status ?? "");
              const fixtureConfigured = fixtureModel && fixtureStatus.toLowerCase() === "configured";
              return <article className="card model-row" key={`${item._kind}-${String(item.id ?? index)}`}>
                <div className={`square-icon ${fixtureModel ? fixtureModelTone(item.tone) : available ? "green" : "orange"}`}><Icon icon={fixtureModel ? fixtureModelGlyph(item.icon) : Cpu}/></div>
                <div className="model-row-copy"><div className="model-row-title"><h3>{label}</h3><span className={`tag ${available ? "green" : "orange"}`}>{fixtureModel ? fixtureStatus : statusText}</span>{fixtureConfigured && demoSelectedModel === label && <span className="tag blue">In use</span>}{local && localPath && <span className="tag blue">Installed</span>}</div><p>{description}</p>{fixtureCaps.length > 0 && <div className="model-capabilities">{fixtureCaps.slice(0, 5).map(capability => <span className="tag" key={capability}>{capability}</span>)}</div>}<p className="caption">{fixtureModel ? `${String(item.provider ?? "Provider")} · ${String(item.meta ?? "")}` : `${locationLabel}${typeof item.default_model === "string" ? ` · Default model: ${item.default_model}` : ""}${typeof item.size_bytes === "number" ? ` · ${(item.size_bytes / 1024 ** 3).toFixed(2)} GB` : ""}`}</p></div>
                <div className="model-actions">{fixtureModel ? <><button className={`btn ${fixtureConfigured ? "primary" : ""}`} type="button" onClick={() => { if (fixtureConfigured) { setDemoSelectedModel(label); try { localStorage.setItem("oa-second-model-selected", label); } catch { /* Keep the selection for this view. */ } setNotice(`${label} selected in local demo settings. No backend model was changed.`); } else { setDemoDetailsModel(item); } }}>{fixtureConfigured ? "Use model" : "Set up"}</button><button className="btn sm" type="button" onClick={() => setDemoDetailsModel(item)}>Details</button></> : !local ? <button className="btn primary" type="button" onClick={() => setSelectedProviderId(String(item.id ?? ""))}>View provider</button> : <><button className="btn primary" type="button" onClick={() => localPath ? requestProbe(item) : (() => { const id = String(item.id ?? ""); if (dataMode === "demo") { setError("Model downloads are unavailable in local demo mode. No network request was made."); return; } if (id && sourceUrl && (!remoteSource || allowNetwork) && window.confirm(`Download ${label}? ${remoteSource ? "This contacts the configured remote source; network consent is enabled for this request." : "This copies from the registered local source path."}`)) mutation.mutate({ path: "/api/local-ai/models/download", body: { model_id: id, allow_network: allowNetwork } }); })()} disabled={mutation.isPending || (!localPath && (dataMode === "demo" || !sourceUrl || (remoteSource && !allowNetwork)))}>{localPath ? "Probe model" : dataMode === "demo" ? "Download unavailable in demo" : "Download model"}</button><details className="model-row-more"><summary className="btn sm">Model actions</summary><div className="model-row-more-actions">{localPath && provider?.local !== false && <button className="btn sm" type="button" onClick={() => benchmarkModel(item)} disabled={mutation.isPending}>Benchmark</button>}{localPath && <button className="btn danger sm" type="button" onClick={() => removeModel(item)} disabled={mutation.isPending}>Remove</button>}{!localPath && <button className="btn sm" type="button" onClick={() => requestProbe(item)} disabled={mutation.isPending}>Probe</button>}</div></details></>}</div>
              </article>;
            }) : providersQuery.isError || localModelsQuery.isError ? <Disconnected icon={Cpu} title="Model inventory unavailable" detail={apiConnectionMessage(providersQuery.error ?? localModelsQuery.error)} /> : <div className="empty" role="status"><Icon icon={Search} size="lg"/><h3>{query ? "No models found" : `No ${tab.toLowerCase()} entries`}</h3><p>{query ? "Try another search term or model category." : "No matching provider or local model records were returned by the backend."}</p></div>}
            {probe && <section className="card padded section-space" aria-label="Model probe result"><div className="row between"><strong>{String(probe.name ?? probe.id ?? "Model probe")}</strong><span className="caption">{probe.pending ? "Checking…" : String(probe.status ?? "Response received")}</span></div>{!probe.pending && <div className="caption section-space">File {probe.exists === true ? "present" : probe.exists === false ? "missing" : "status not reported"} · Checksum {probe.checksum_valid === true ? "verified" : probe.checksum_valid === false ? "mismatch" : "not available"} · Dependencies {Array.isArray(probe.dependents) ? probe.dependents.length : "not reported"} · Tools {Array.isArray(probe.tools) ? probe.tools.length : "not reported"}</div>}</section>}
            {benchmarkResult && <section className="card padded section-space" aria-label="Model benchmark result"><div className="row between"><strong>Backend benchmark</strong><span className="caption">{String(benchmarkResult.created_at ?? "Completed")}</span></div><div className="settings-line"><span className="flex1">Model / provider</span><span>{String(benchmarkResult.model ?? "Not reported")} · {String(benchmarkResult.provider ?? "Not reported")}</span></div><div className="settings-line"><span className="flex1">Generation time</span><span>{benchmarkResult.duration_ms == null ? "Not reported" : `${String(benchmarkResult.duration_ms)} ms`}</span></div><div className="settings-line"><span className="flex1">Output throughput</span><span>{benchmarkResult.tokens_per_second == null ? "Not reported" : `${String(benchmarkResult.tokens_per_second)} tokens/s`}</span></div><div className="settings-line"><span className="flex1">Prompt / output tokens</span><span>{String(benchmarkResult.prompt_tokens ?? "Not reported")} / {String(benchmarkResult.output_tokens ?? "Not reported")}</span></div></section>}
            <details className="card padded section-space"><summary>Routing policy</summary><div className="settings-line"><span className="flex1">Routing</span><strong>{String(policy.routing_policy ?? "Not reported")}</strong></div><div className="settings-line"><span className="flex1">Fallback</span><strong>{String(policy.fallback_policy ?? "Not reported")}</strong></div><div className="caption">Policy is read-only here; the backend remains authoritative.</div></details>
          </>}
          {panel === "Setup activity" && <div className="section-space">{jobsQuery.isLoading ? <p role="status">Loading setup activity…</p> : jobsQuery.isError ? <div role="alert">{apiConnectionMessage(jobsQuery.error)}</div> : jobs.length ? jobs.map(job => <article className="card padded section-space" key={String(job.id)}><div className="row between"><div><strong>{String(job.model_name ?? job.model ?? "Model setup")}</strong><p className="caption">{String(job.current_step ?? job.kind ?? "Setup job")} · {String(job.status ?? "Status not reported")}</p></div><span className="caption">{job.progress == null ? "Progress not reported" : `${String(job.progress)}%`}</span></div>{job.error != null && <p className="caption" role="alert">{String(job.error)}</p>}{!jobTerminal(job) && <button className="btn danger sm section-space" type="button" onClick={() => { if (window.confirm("Cancel this model setup job?")) mutation.mutate({ path: "/api/local-ai/jobs/cancel", body: { job_id: job.id } }); }} disabled={mutation.isPending}>Cancel job</button>}</article>) : <Disconnected icon={FileClock} title="No setup jobs" detail="The backend has no recorded model setup jobs for this project." />}</div>}
          {panel === "Marketplace & runtime" && <div className="section-space">
            <div className="card padded section-space"><div className="row between"><h3>Local runtime</h3></div>{runtimeEntries.map(([name, runtime]) => <div className="settings-line" key={name}><span className="flex1"><strong>{name}</strong><small>{runtime.server_available === true ? "Server responding" : runtime.available === true ? "Installed; server not responding" : runtime.available === false ? "Not installed" : "Availability not reported"}</small></span><span className="caption">{runtime.model_ready === true ? `Ready · ${String(runtime.selected_model ?? runtime.model ?? "model")}` : runtime.model_ready === false ? "Model not ready" : "Readiness not reported"}</span></div>)}{hardwareFacts.length > 0 ? <p className="caption section-space">{hardwareFacts.join(" · ")}</p> : hardwareQuery.isError ? <p role="alert">Hardware status unavailable: {apiConnectionMessage(hardwareQuery.error)}</p> : null}</div>
            <div className="card padded section-space"><h3>Setup catalog</h3>{marketplaceQuery.isError ? <p role="alert">{apiConnectionMessage(marketplaceQuery.error)}</p> : catalog.length ? catalog.filter(item => !query || `${item.name ?? ""} ${item.id ?? ""} ${item.provider ?? ""}`.toLowerCase().includes(query.toLowerCase())).map(item => <div className="settings-line" key={String(item.id)}><span className="flex1"><strong>{String(item.name ?? item.id ?? "Model")}</strong><small>{String(modelRecord(item.metadata).model ?? "No direct setup identifier")}{modelRecord(item.metadata).ram_gb != null ? ` · ${String(modelRecord(item.metadata).ram_gb)} GB RAM` : ""}</small></span><button className="btn sm" type="button" disabled={mutation.isPending || !modelRecord(item.metadata).model} onClick={() => setupModel(item)}>Queue setup</button></div>) : <p className="caption">No catalog entries returned by the backend.</p>}</div>
            <form className="card padded section-space" onSubmit={submit}><h3>Register local model</h3><p className="caption">Registration saves metadata only; it does not download files or contact a source URL.</p><label className="demo-form-field">Model name<input className="input" required maxLength={180} value={modelName} onChange={event => setModelName(event.target.value)} /></label><label className="demo-form-field">Local path<input className="input" value={modelPath} onChange={event => setModelPath(event.target.value)} /></label><label className="demo-form-field">Source URL (optional)<input className="input" type="url" value={modelSource} onChange={event => setModelSource(event.target.value)} placeholder="https://…" /></label><button className="btn primary" type="submit" disabled={mutation.isPending || !modelName.trim()}>Register model</button></form>
            <div className="card padded section-space"><h3>Role recommendations</h3>{Object.keys(recommendations).length ? Object.entries(recommendations).map(([role, model]) => <div className="settings-line" key={role}><span className="flex1">{role.replaceAll("_", " ")}</span><strong>{String(model)}</strong></div>) : <p className="caption">No recommendations reported.</p>}<div className="settings-line"><span className="flex1">Acceleration adapters</span><strong>{adaptersQuery.isError ? "Unavailable" : adapters.length}</strong></div>{adapters.map((adapter, index) => <p className="caption" key={String(adapter.id ?? index)}>{String(adapter.name ?? adapter.adapter_id ?? "Adapter")} · {String(adapter.status ?? "Status not reported")}</p>)}</div>
          </div>}
        </div>
        <aside className="model-info-side">
          <section className="card model-guidance-card"><div className="row"><div className="square-icon violet"><Icon icon={Sparkles}/></div><h3>Find the right model</h3></div>
            <p className="muted small model-guidance-intro">Choose based on actual availability, supported tools and privacy requirements.</p>
            <div className="model-usecase-list" aria-label="Model selection guidance">
              {[[MessageCircle, "Everyday chat"], [Code2, "Development and code"], [BookOpen, "Long documents"], [LockKeyhole, "Local privacy"], [Brain, "Reasoning tasks"]].map(([Glyph, label]) => <div className="list-row model-usecase" key={String(label)}><Icon icon={Glyph as LucideIcon}/><span className="flex1 small">{String(label)}</span></div>)}
            </div>
            {Object.entries(recommendations).length > 0 && <div className="model-recommendations"><strong className="caption">Role recommendations from setup service</strong>{Object.entries(recommendations).map(([role, value]) => <div className="settings-line" key={role}><span className="flex1">{role.replaceAll("_", " ")}</span><strong>{String(value)}</strong></div>)}</div>}
          </section>
          <section className="card model-guidance-card"><h3>Privacy reminder</h3>
            <p className="muted small model-privacy-copy">Remote models may send data to external providers. Check your project settings before connecting.</p>
            <button className="btn sm section-space" type="button" data-nav="settings">Privacy settings</button>
          </section>
        </aside>
      </div>
      {demoDetailsModel && dataMode === "demo" && <div className="overlay-root" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) setDemoDetailsModel(null); }}><section className="modal" role="dialog" aria-modal="true" aria-labelledby="demo-model-detail-title"><div className="row between"><h2 id="demo-model-detail-title">{String(demoDetailsModel.name ?? demoDetailsModel.label ?? "Model details")}</h2><button className="icon-btn" type="button" aria-label="Close" onClick={() => setDemoDetailsModel(null)}><X className="icon sm"/></button></div><p>{String(demoDetailsModel.description ?? "Model information from the deterministic local fixture.")}</p><div className="demo-detail-grid"><div><small>Provider</small><strong>{String(demoDetailsModel.provider ?? "Not reported")}</strong></div><div><small>Capabilities</small><strong>{Array.isArray(demoDetailsModel.caps) ? demoDetailsModel.caps.map(String).join(", ") : "Not reported"}</strong></div><div><small>Demo connection</small><strong>{String(demoDetailsModel.status ?? "Not reported")}</strong></div></div><div className="modal-actions"><button className="btn primary" type="button" onClick={() => { const label = String(demoDetailsModel.name ?? demoDetailsModel.label ?? ""); if (String(demoDetailsModel.status ?? "").toLowerCase() === "configured") { setDemoSelectedModel(label); try { localStorage.setItem("oa-second-model-selected", label); } catch { /* Keep the selection for this view. */ } setNotice(`${label} selected in local demo settings. No backend model was changed.`); } else { setNotice(`${label} setup remains a preview in local demo mode; no files were downloaded.`); setPanel("Setup activity"); setTraceStartSignal(value => value + 1); } setDemoDetailsModel(null); }}>{String(demoDetailsModel.status ?? "").toLowerCase() === "configured" ? "Use model" : "Preview setup"}</button></div></section></div>}
    </motion.div>
  );
}

const settingsTabs = ["Appearance", "Models & providers", "Privacy", "Notifications", "Data & backups", "Advanced"];
type ThemeChoice = "light" | "dark" | "system";
type RuntimeSettings = Record<string, string | number>;
type BackupRecord = { name?: string; path?: string; size_bytes?: number; manifest?: Record<string, unknown>; entry_count?: number };
type SettingsNotification = { id?: string; title?: string; body?: string; level?: string; created_at?: string; read_at?: string | null; task_id?: string };
const runtimeFields = [
  ["chat_model", "Chat model"], ["code_model", "Code model"], ["research_model", "Research model"],
  ["voice_model", "Voice model"], ["ollama_url", "Ollama URL"], ["ollama_model", "Ollama model"],
  ["llamacpp_url", "llama.cpp URL"], ["llamacpp_model", "llama.cpp model"],
  ["quantization_mode", "Quantization mode"], ["target_ram_gb", "Target RAM (GB)"],
  ["model_timeout_seconds", "Model timeout (seconds)"],
] as const;

export function SettingsScreen() {
  const [tab, setTab] = useState("Appearance");
  const theme = useWorkspaceState((state) => state.theme);
  const setTheme = useWorkspaceState((state) => state.setTheme);
  const reduceMotion = useWorkspaceState((state) => state.reduceMotion);
  const setReduceMotion = useWorkspaceState((state) => state.setReduceMotion);
  const dataMode = useWorkspaceState((state) => state.dataMode);
  const setDataMode = useWorkspaceState((state) => state.setDataMode);
  const collapsed = useWorkspaceState((state) => state.collapsed);
  const setCollapsed = useWorkspaceState((state) => state.setCollapsed);
  const queryClient = useQueryClient();
  const runtimeQuery = useQuery({ queryKey: ["settings", "runtime", dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "operations.settings", s => apiGet<{ settings?: RuntimeSettings; privacy_boundary?: string }>("/api/settings/runtime", s), signal).then(data => dataMode === "demo" ? { settings: data as RuntimeSettings, privacy_boundary: "Local demo fixture" } : data), enabled: tab === "Advanced" && dataMode !== "demo", retry: false });
  const [runtimeDraft, setRuntimeDraft] = useState<RuntimeSettings>({});
  const [runtimeMessage, setRuntimeMessage] = useState("");
  const saveRuntimeMutation = useMutation({
    mutationFn: (settings: RuntimeSettings) => apiPost<{ settings?: RuntimeSettings }>("/api/settings/runtime", { settings }),
    onSuccess: async result => { setRuntimeMessage("Runtime settings were accepted by the backend for this process."); await queryClient.invalidateQueries({ queryKey: ["settings", "runtime"] }); if (result.settings) setRuntimeDraft({ ...result.settings }); },
    onError: error => setRuntimeMessage(`Runtime settings were not saved: ${apiConnectionMessage(error)}`),
  });
  const backupsQuery = useQuery({ queryKey: ["backups", dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "operations.backups", s => apiGet<{ backups?: BackupRecord[] }>("/api/backups", s), signal).then(data => dataMode === "demo" ? { backups: data.backups ?? [] } : data), enabled: tab === "Data & backups", retry: false });
  const providersQuery = useQuery({ queryKey: ["settings", "providers", dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "operations.providers", s => apiGet<{ providers?: Record<string, unknown>[] }>("/api/models", s), signal).then(data => dataMode === "demo" ? { providers: data.providers ?? [] } : data), enabled: tab === "Models & providers", retry: false });
  const modelPolicyQuery = useQuery({ queryKey: ["settings", "model-policy", dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "operations.modelPolicy", s => apiGet<Record<string, unknown>>("/api/model-policy", s), signal), enabled: tab === "Privacy", retry: false });
  const securityStatusQuery = useQuery({ queryKey: ["settings", "security-status", dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "operations.security", s => apiGet<Record<string, unknown>>("/api/security/status", s), signal), enabled: tab === "Privacy", retry: false });
  const egressQuery = useQuery({ queryKey: ["settings", "egress", dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "operations.egress", s => apiGet<{ requests?: Record<string, unknown>[] }>("/api/security/egress", s), signal), enabled: tab === "Privacy", retry: false });
  const notificationsQuery = useQuery({ queryKey: ["settings", "notifications", "default", dataMode], queryFn: ({ signal }) => demoQuery(dataMode, "operations.notifications", s => apiGet<{ notifications?: SettingsNotification[] }>("/api/tasks/notifications?project=default&limit=100", s), signal).then(data => dataMode === "demo" ? { notifications: data.notifications ?? [] } : data), enabled: tab === "Notifications", retry: false });
  const [notificationMessage, setNotificationMessage] = useState("");
  const [demoSettingsValues, setDemoSettingsValues] = useState<Record<string, boolean>>(() => {
    try {
      const stored = localStorage.getItem("oa-second-settings-demo-v1");
      return stored ? JSON.parse(stored) as Record<string, boolean> : {};
    } catch { return {}; }
  });
  const [demoSettingsMessage, setDemoSettingsMessage] = useState("");
  const setDemoPreference = (group: string, label: string, value: boolean) => {
    setDemoSettingsValues(current => {
      const next = { ...current, [`${group}|${label}`]: value };
      try { localStorage.setItem("oa-second-settings-demo-v1", JSON.stringify(next)); } catch { /* Keep the preference for this view. */ }
      return next;
    });
  };
  const markNotificationRead = useMutation({
    mutationFn: (id: string) => apiPost<{ notification?: SettingsNotification }>(`/api/tasks/notifications/${encodeURIComponent(id)}/read`, { project: "default" }),
    onSuccess: async result => { setNotificationMessage(result.notification?.read_at ? "Notification marked as read." : "The backend accepted the read request."); await queryClient.invalidateQueries({ queryKey: ["settings", "notifications", "default"] }); },
    onError: error => setNotificationMessage(`Could not update notification: ${apiConnectionMessage(error)}`),
  });
  const [selectedBackup, setSelectedBackup] = useState("");
  const [backupMessage, setBackupMessage] = useState("");
  const [restoreDryRun, setRestoreDryRun] = useState<Record<string, unknown> | null>(null);
  const createBackupMutation = useMutation({
    mutationFn: () => apiPost<{ backup?: BackupRecord }>("/api/backups/create", { project: "default", include_media: true }),
    onSuccess: async result => { setBackupMessage(result.backup?.name ? `Backup created: ${result.backup.name}` : "The backend returned a successful backup response."); setRestoreDryRun(null); await queryClient.invalidateQueries({ queryKey: ["backups"] }); },
    onError: error => setBackupMessage(`Backup creation failed: ${apiConnectionMessage(error)}`),
  });
  const dryRunMutation = useMutation({
    mutationFn: (path: string) => apiPost<Record<string, unknown>>("/api/backups/restore-dry-run", { path }),
    onSuccess: result => { setRestoreDryRun(result); setBackupMessage(""); },
    onError: error => { setRestoreDryRun(null); setBackupMessage(`Restore preflight failed: ${apiConnectionMessage(error)}`); },
  });
  const restoreMutation = useMutation({
    mutationFn: (path: string) => apiPost<Record<string, unknown>>("/api/backups/restore", { path, confirm: true }),
    onSuccess: async result => { setBackupMessage(result.restored === true ? `Restore completed according to the backend. Safety copy: ${String(result.safety_copy || "not reported")}` : "The restore request returned, but the backend did not confirm that data was restored."); setRestoreDryRun(null); await queryClient.invalidateQueries(); },
    onError: error => setBackupMessage(`Restore failed: ${apiConnectionMessage(error)}`),
  });
  const backendHealth = useQuery({ queryKey: ["backend", "health"], queryFn: ({ signal }) => apiGet("/api/health", signal), enabled: dataMode !== "demo", retry: false, refetchInterval: 30_000 });
  const [apiToken, setTokenDraft] = useState(getApiToken);
  const [tokenMessage, setTokenMessage] = useState("");
  const [helpfulTooltips, setHelpfulTooltips] = useState(() => {
    try { return localStorage.getItem("oa-second-helpful-tooltips") !== "false"; } catch { return true; }
  });
  const chooseTheme = (value: ThemeChoice) => setTheme(value);
  const setTooltips = (enabled: boolean) => {
    try { localStorage.setItem("oa-second-helpful-tooltips", String(enabled)); } catch { /* This preference still applies for the current screen. */ }
    setHelpfulTooltips(enabled);
  };
  const saveToken = (event: FormEvent) => {
    event.preventDefault();
    setApiToken(apiToken);
    setTokenMessage(apiToken.trim() ? "Bearer token saved for this browser tab." : "Bearer token cleared for this browser tab.");
    void queryClient.invalidateQueries();
  };
  return (
    <motion.div className="page settings-screen-page" {...pageMotion}>
      <PageHeading title="Settings" subtitle="Personalise your workspace and manage your connections." />
      {dataMode === "demo" && <span className="btn sm models-live-demo settings-demo-indicator"><span className="status-dot" aria-hidden="true" />Live demo</span>}
      <Tabs items={settingsTabs} selected={tab} onSelect={setTab} />
      <div className="settings-layout">
        <section>
          {tab === "Advanced" && import.meta.env.DEV && <div className="card setting-group"><h3>Frontend data mode</h3><p className="muted small">Demo mode uses deterministic local fixtures and never calls the backend for workspace data or actions.</p><div className="row section-space"><button type="button" className={`btn ${dataMode === "demo" ? "primary" : ""}`} aria-pressed={dataMode === "demo"} onClick={() => setDataMode("demo")}>Local demo</button><button type="button" className={`btn ${dataMode === "live" ? "primary" : ""}`} aria-pressed={dataMode === "live"} onClick={() => setDataMode("live")}>Live backend</button></div></div>}
          {tab === "Appearance" && <>
            <div className="card setting-group"><h3>Appearance</h3><p className="muted small">Choose how OpenAgent looks on your device.</p><div className="theme-choices">{([["light", Sun, "Light"], ["dark", Moon, "Dark"], ["system", Monitor, "System"]] as const).map(([value, ThemeIcon, label]) => <button type="button" className={`theme-card ${theme === value ? "selected" : ""}`} key={value} aria-pressed={theme === value} onClick={() => chooseTheme(value)}><Icon icon={ThemeIcon} />{label}{theme === value && <Check className="icon sm" aria-hidden="true" />}</button>)}</div></div>
            <div className="card setting-group"><h3>Workspace preferences</h3><div className="settings-line"><div className="flex1"><strong>Reduce interface animation</strong><small>Keep panels calm and accessible.</small></div><input className="toggle" type="checkbox" checked={reduceMotion} onChange={(event) => setReduceMotion(event.target.checked)} aria-label="Reduce interface animation" /></div><div className="settings-line"><div className="flex1"><strong>Compact navigation</strong><small>Show a narrow sidebar.</small></div><input className="toggle" type="checkbox" checked={collapsed} onChange={(event) => setCollapsed(event.target.checked)} aria-label="Compact navigation" /></div><div className="settings-line"><div className="flex1"><strong>Enable helpful tooltips</strong><small>Explain controls without extra page text.</small></div><input className="toggle" type="checkbox" checked={helpfulTooltips} onChange={(event) => setTooltips(event.target.checked)} aria-label="Enable helpful tooltips" /></div></div>
          </>}
          {tab === "Models & providers" && <div className="card setting-group"><div className="row between"><div><h3>Models & providers</h3><p className="muted small">Live provider availability reported by the model gateway.</p></div><button className="btn sm" type="button" disabled={providersQuery.isFetching} onClick={() => void providersQuery.refetch()}>Refresh</button></div>{providersQuery.isLoading ? <p className="caption" role="status">Loading provider status…</p> : providersQuery.isError ? <p role="alert">Provider status unavailable: {apiConnectionMessage(providersQuery.error)}</p> : (providersQuery.data?.providers ?? []).length ? (providersQuery.data?.providers ?? []).map((provider, index) => <div className="settings-line" key={String(provider.id ?? index)}><span className="flex1"><strong>{String(provider.label ?? provider.id ?? "Provider")}</strong><small>{String(provider.privacy_warning ?? (provider.local === true ? "Local provider" : provider.local === false ? "External provider" : "Locality not reported"))}</small></span><span className="caption">{provider.available === true ? "Available" : provider.available === false ? "Unavailable" : "Status not reported"}{provider.default_model ? ` · ${String(provider.default_model)}` : ""}</span></div>) : <Disconnected icon={Cpu} title="No providers returned" detail="The model service returned an empty provider list." />}</div>}
          {tab === "Privacy" && (dataMode === "demo" ? <div className="card setting-group"><h3>Privacy controls</h3><p className="muted small">Settings shown as a visual example; wire consent to existing backend policy APIs.</p>{["Ask before external web access", "Show tool permission prompts", "Keep private content local when supported", "Require confirmation for external sends"].map((label, index) => <div className="settings-line" style={{ padding: "15px 0" }} key={label}><div className="flex1">{label}</div><input className="toggle" type="checkbox" aria-label={label} checked={demoSettingsValues[`Privacy|${label}`] ?? index < 2} onChange={event => setDemoPreference("Privacy", label, event.target.checked)}/></div>)}</div> : <div className="card setting-group"><h3>Privacy & routing</h3><p className="muted small">These are read-only policy and security facts from the active backend.</p>{modelPolicyQuery.isLoading || securityStatusQuery.isLoading || egressQuery.isLoading ? <p className="caption" role="status">Loading privacy status…</p> : <>{modelPolicyQuery.isError && <p role="alert">Model policy unavailable: {apiConnectionMessage(modelPolicyQuery.error)}</p>}{securityStatusQuery.isError && <p role="alert">Security status unavailable: {apiConnectionMessage(securityStatusQuery.error)}</p>}{egressQuery.isError && <p role="alert">Egress requests unavailable: {apiConnectionMessage(egressQuery.error)}</p>}{!modelPolicyQuery.isError && <div className="settings-line"><span className="flex1"><strong>Model routing</strong><small>Fallback: {String(modelPolicyQuery.data?.fallback_policy ?? "not reported")}</small></span><span className="caption">{String(modelPolicyQuery.data?.routing_policy ?? "not reported")}</span></div>}{!securityStatusQuery.isError && Object.entries(securityStatusQuery.data ?? {}).filter(([, value]) => ["string", "number", "boolean"].includes(typeof value)).slice(0, 8).map(([key, value]) => <div className="settings-line" key={key}><span className="flex1">{key.replaceAll("_", " ")}</span><span className="caption">{String(value)}</span></div>)}{!egressQuery.isError && <div className="settings-line"><span className="flex1">Recorded egress requests</span><span className="caption">{(egressQuery.data?.requests ?? []).length} · {(egressQuery.data?.requests ?? []).filter(request => request.status === "pending").length} pending</span></div>}</>}</div>)}
          {tab === "Notifications" && (dataMode === "demo" ? <div className="card setting-group"><h3>Notifications</h3>{["Task completion", "Approval requests", "Automation failures", "Release notes"].map((label, index) => <div className="settings-line" style={{ padding: "15px 0" }} key={label}><div className="flex1">{label}</div><input className="toggle" type="checkbox" aria-label={label} checked={demoSettingsValues[`Notifications|${label}`] ?? index < 3} onChange={event => setDemoPreference("Notifications", label, event.target.checked)}/></div>)}</div> : <div className="card setting-group"><div className="row between"><div><h3>Notifications</h3><p className="muted small">Task and workflow notifications recorded by the backend.</p></div><button className="btn sm" type="button" disabled={notificationsQuery.isFetching} onClick={() => { setNotificationMessage(""); void notificationsQuery.refetch(); }}>Refresh</button></div>{notificationsQuery.isLoading ? <p className="caption" role="status">Loading notifications…</p> : notificationsQuery.isError ? <p role="alert">Notifications unavailable: {apiConnectionMessage(notificationsQuery.error)}</p> : (notificationsQuery.data?.notifications ?? []).length ? (notificationsQuery.data?.notifications ?? []).map((notification, index) => <article className="settings-line" key={String(notification.id ?? index)}><span className="flex1"><strong>{String(notification.title ?? "Notification")}</strong><small>{String(notification.body ?? "")}{notification.created_at ? ` · ${String(notification.created_at)}` : ""}</small></span><span className="caption">{String(notification.level ?? "info")}{notification.read_at ? " · Read" : " · Unread"}</span>{!notification.read_at && notification.id && <button className="btn sm" type="button" disabled={markNotificationRead.isPending} onClick={() => markNotificationRead.mutate(String(notification.id))}>Mark read</button>}</article>) : <p className="caption">No notifications have been recorded for this project.</p>}{notificationMessage && <p className="caption section-space" role={notificationMessage.startsWith("Could not") ? "alert" : "status"}>{notificationMessage}</p>}</div>)}
          {tab === "Data & backups" && (dataMode === "demo" ? <div className="card setting-group"><h3>Data & backups</h3><p className="muted small" style={{ marginBottom: 12 }}>Backups, exports and restore controls must use live backend confirmation.</p><div className="row"><button className="btn" type="button" onClick={() => setDemoSettingsMessage("Example export preview is local only. No data was downloaded or sent.")}>Export data (preview)</button><button className="btn" type="button" onClick={() => setDemoSettingsMessage("Backup review is unavailable in local demo mode; no backend request was made.")}>Review backups</button></div>{demoSettingsMessage && <p className="caption section-space" role="status">{demoSettingsMessage}</p>}</div> : <div className="card setting-group"><h3>Data & backups</h3><p className="muted small" style={{ marginBottom: 12 }}>Backups are stored by the backend in its local data directory. Restoring replaces the current database; the backend creates a safety copy first.</p><div className="row"><button className="btn primary" type="button" disabled={createBackupMutation.isPending} onClick={() => { setBackupMessage(""); createBackupMutation.mutate(); }}>{createBackupMutation.isPending ? "Creating backup…" : "Create backup"}</button><button className="btn" type="button" disabled={backupsQuery.isFetching} onClick={() => void backupsQuery.refetch()}>Refresh backups</button></div>{backupsQuery.isLoading ? <p className="caption section-space" role="status">Loading backups…</p> : backupsQuery.isError ? <p className="section-space" role="alert">Backups unavailable: {apiConnectionMessage(backupsQuery.error)}</p> : (backupsQuery.data?.backups ?? []).length ? <div className="section-space"><label className="demo-form-field">Backup archive<select className="select" value={selectedBackup} onChange={event => { setSelectedBackup(event.target.value); setRestoreDryRun(null); setBackupMessage(""); }}><option value="">Choose an existing backend backup</option>{(backupsQuery.data?.backups ?? []).map((backup, index) => <option key={`${backup.name ?? "backup"}-${index}`} value={String(backup.path ?? "")}>{String(backup.name ?? "Unnamed backup")} · {Number(backup.size_bytes ?? 0).toLocaleString()} bytes</option>)}</select></label><button className="btn" type="button" disabled={!selectedBackup || dryRunMutation.isPending} onClick={() => { setBackupMessage(""); dryRunMutation.mutate(selectedBackup); }}>{dryRunMutation.isPending ? "Checking archive…" : "Run restore preflight"}</button></div> : <p className="caption section-space">{backupsQuery.isSuccess ? "No backups are available from the backend." : "Backup list has not been loaded."}</p>}{restoreDryRun && <div className="card padded section-space" role="status"><strong>Restore preflight: {String(restoreDryRun.status ?? "unknown")}</strong><p className="caption section-space">This check did not modify data. Database entries: {Array.isArray(restoreDryRun.database_entries) ? restoreDryRun.database_entries.length : "not reported"}; media entries: {String(restoreDryRun.media_entries ?? "not reported")}.</p>{Array.isArray(restoreDryRun.conflicts) && restoreDryRun.conflicts.length > 0 && <p className="caption" role="alert">Potential impact: {restoreDryRun.conflicts.map(String).join(", ")}. The current database may be replaced.</p>}<button className="btn danger section-space" type="button" disabled={restoreDryRun.status !== "ready" || restoreMutation.isPending || restoreDryRun.will_write !== false} onClick={() => { const backup = (backupsQuery.data?.backups ?? []).find(item => item.path === selectedBackup); const label = backup?.name ?? "selected archive"; if (window.confirm(`Restore ${label}? This replaces the current database. A backend safety copy is attempted first. Continue only if you have verified this archive.`)) restoreMutation.mutate(selectedBackup); }}>Restore this backup…</button></div>}{backupMessage && <p className="caption section-space" role={backupMessage.includes("failed") ? "alert" : "status"}>{backupMessage}</p>}</div>)}
          {tab === "Advanced" && (dataMode === "demo" ? <div className="card setting-group"><h3>Advanced</h3>{[[Database, "Backend connections", "Inspect which services are ready"], [Shield, "System diagnostics", "View backend status and logs"], [Wrench, "Developer tools", "API and integration management"]].map(([Glyph, label, detail]) => <div className="list-row" key={String(label)}><Icon icon={Glyph as LucideIcon}/><div className="flex1"><strong>{String(label)}</strong><div className="caption">{String(detail)}</div></div><button className="btn sm" type="button" data-nav="developer">Open</button></div>)}</div> : <><div className="card setting-group"><h3>Advanced</h3>{[[Database, "Backend connections"], [Shield, "System diagnostics"], [Wrench, "Developer tools"]].map(([Glyph, label]) => <div className="settings-line" key={String(label)}><div className="row flex1"><Icon icon={Glyph as LucideIcon} /><span>{label as string}</span></div><span className="caption">{backendHealth.isSuccess ? "Connected" : backendHealth.isError ? "Offline" : "Checking"}</span></div>)}</div><div className="card setting-group"><h3>Runtime settings</h3><p className="muted small">These values update the current backend process. Empty values are ignored by the service and cannot clear an existing process setting.</p>{runtimeQuery.isLoading ? <p className="caption" role="status">Loading runtime configuration…</p> : runtimeQuery.isError ? <p role="alert">Runtime settings unavailable: {apiConnectionMessage(runtimeQuery.error)}</p> : <form onSubmit={event => { event.preventDefault(); setRuntimeMessage(""); saveRuntimeMutation.mutate(runtimeDraft); }}><div className="runtime-settings-grid">{runtimeFields.map(([key, label]) => <label className="demo-form-field" key={key}>{label}<input className="input" value={String(runtimeDraft[key] ?? runtimeQuery.data?.settings?.[key] ?? "")} onChange={event => setRuntimeDraft(current => ({ ...current, [key]: event.target.value }))} /></label>)}</div><button className="btn primary" type="submit" disabled={saveRuntimeMutation.isPending}>{saveRuntimeMutation.isPending ? "Saving…" : "Save runtime settings"}</button>{runtimeMessage && <p className="caption section-space" role={runtimeMessage.includes("not saved") ? "alert" : "status"}>{runtimeMessage}</p>}{runtimeQuery.data?.privacy_boundary && <p className="caption section-space">{String(runtimeQuery.data.privacy_boundary)}</p>}</form>}</div><div className="card setting-group"><h3>OpenAgent API</h3><p className="muted small">{backendHealth.isError ? apiConnectionMessage(backendHealth.error) : backendHealth.isSuccess ? "The backend health endpoint is responding." : "Checking the configured backend origin."}</p><form onSubmit={saveToken}><label className="demo-form-field">Optional bearer token<input className="input" type="password" autoComplete="off" value={apiToken} onChange={event => setTokenDraft(event.target.value)} placeholder="Required only when backend authentication is enabled" aria-label="OpenAgent bearer token" /></label><div className="row"><button className="btn primary" type="submit">Save for this tab</button><button className="btn" type="button" onClick={() => { setTokenDraft(""); setApiToken(""); setTokenMessage("Bearer token cleared for this browser tab."); void queryClient.invalidateQueries(); }}>Clear token</button></div>{tokenMessage && <p className="caption section-space" role="status">{tokenMessage}</p>}</form></div></>)}
        </section>
        <aside><div className="card padded"><div className="row"><div className="square-icon"><Icon icon={UserRound} /></div><div><h3>Personal Workspace</h3><p className="caption">{dataMode === "demo" ? "Local reference mode" : "Frontend preview"}</p></div></div><p className="muted small section-space">{dataMode === "demo" ? "Account, permissions, quotas, usage and payments are intentionally not fabricated in this prototype." : "Account, permissions, quotas, usage are intentionally not shown without a backend connection."}</p></div><div className="card padded section-space"><div className="row"><div className="square-icon green"><Icon icon={Shield} /></div><h3>Need help?</h3></div><p className="muted small" style={{ margin: "9px 0 12px" }}>Keyboard shortcuts, feature guidance and connection troubleshooting.</p><button className="btn sm" type="button" data-nav="tools">Open tools</button></div><div className="demo-ribbon"><Info className="icon xs" aria-hidden="true" />Illustrative content · UI reference</div></aside>
      </div>
    </motion.div>
  );
}
