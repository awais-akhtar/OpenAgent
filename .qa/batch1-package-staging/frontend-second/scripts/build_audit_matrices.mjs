import { execFileSync } from 'node:child_process';
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const root = path.resolve(process.cwd(), '..');
const here = process.cwd();
const outDir = path.join(here, '.qa', 'audit-handoff-20261018');
const python = path.join(root, '.venv', 'Scripts', 'python.exe');
const py = String.raw`
import json
from openagent.routes import DOMAIN_ROUTE_MODULES, route_contracts
from openagent.auth import required_scope_for_path
from scripts.audit_frontend_api_coverage import GROUP_AUDIENCE, ROUTE_AUDIENCE_OVERRIDES, route_pattern
rows=[]
for contract, module in zip(route_contracts(), DOMAIN_ROUTE_MODULES, strict=True):
    group=contract['group']
    for route in contract['routes']:
        method, pattern=route['method'], route['path']
        rows.append({'group': group, 'method': method, 'path': pattern,
            'declaring_module': f'openagent/routes/{module}.py',
            'audience': ROUTE_AUDIENCE_OVERRIDES.get((group, pattern), GROUP_AUDIENCE.get(group, 'Review required')),
            'scope': required_scope_for_path(pattern, method),
            'regex': route_pattern(pattern).pattern})
print(json.dumps(rows))
`;
const routeRows = JSON.parse(execFileSync(python, ['-c', py], { cwd: root, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 }));

function collect(rootDir, allowTests) {
  const files = [];
  const visit = (dir) => {
    for (const name of readdirSync(dir, { withFileTypes: true })) {
      if (name.name === 'node_modules' || name.name === 'dist' || name.name === '.qa' || name.name === 'demo') continue;
      const full = path.join(dir, name.name);
      if (name.isDirectory()) visit(full);
      else if (/\.(tsx?|jsx?)$/.test(name.name) && (allowTests || !/\.(test|spec)\.[tj]sx?$/.test(name.name))) {
        let text;
        try { text = readFileSync(full, 'utf8'); } catch { continue; }
        files.push({ full, relative: path.relative(root, full).replaceAll('\\', '/'), text });
      }
    }
  };
  visit(rootDir);
  return files;
}
const secondFiles = collect(path.join(here, 'src'), false);
const originalFiles = collect(path.join(root, 'frontend', 'src'), false);
const testFiles = collect(path.join(here, 'tests'), true);
const lineRefs = (files, regex) => {
  const result = [];
  for (const file of files) {
    const lines = file.text.split(/\r?\n/);
    const refs = [];
    for (let i = 0; i < lines.length; i++) {
      regex.lastIndex = 0;
      if (!regex.test(lines[i])) continue;
      const context = lines.slice(Math.max(0, i - 2), Math.min(lines.length, i + 3)).join('\n');
      const requestCandidate = /\b(?:api(?:Get|Post|Put|Patch|Delete|Request|Stream|Sse)|fetch)\s*\(/i.test(context);
      refs.push({ line: i + 1, requestCandidate });
    }
    if (refs.length) result.push({ file: file.relative, refs });
  }
  return result;
};
const esc = (value) => `"${String(value ?? '').replaceAll('"', '""')}"`;
const csv = (rows, cols) => [cols.map(esc).join(','), ...rows.map(row => cols.map(col => esc(row[col])).join(','))].join('\r\n') + '\r\n';

const auditRows = routeRows.map(route => {
  const regex = new RegExp(route.regex, 'g');
  const secondRefs = lineRefs(secondFiles, regex);
  const originalRefs = lineRefs(originalFiles, new RegExp(route.regex, 'g'));
  const testRefs = lineRefs(testFiles, new RegExp(route.regex, 'g'));
  const hasRequestCandidate = secondRefs.some(f => f.refs.some(r => r.requestCandidate));
  let status = !secondRefs.length ? 'No frontend literal; review candidate only' : hasRequestCandidate ? 'Request callsite candidate; runtime unverified' : 'Static reference only; request not proven';
  const scopeStatus = route.scope ? `Mapped: ${route.scope}` : 'Unmapped (auth enforcement permits this path)';
  const flatten = (refs) => refs.map(file => `${file.file}:${file.refs.map(ref => ref.line).join('|')}`).join('; ');
  return { group: route.group, audience: route.audience, method: route.method, backendRoute: route.path,
    declaredIn: route.declaring_module, runtimeHandlerVerification: 'Not individually verified in this audit; declaration is not handler proof',
    authEnforcementScope: scopeStatus, originalFrontendSourceRefs: flatten(originalRefs), frontendSecondSourceRefs: flatten(secondRefs),
    frontendSecondTestSourceRefs: flatten(testRefs), integrationStatus: status,
    pythonHttpEvidence: route.path === '/api/agent-runs' ? 'Direct isolated Python HTTP security reproduction only; React integration withheld' : 'No real React-to-Python request test in this audit' };
});
writeFileSync(path.join(outDir, 'BACKEND_ROUTE_INTEGRATION_MATRIX.csv'), csv(auditRows,
  ['group','audience','method','backendRoute','declaredIn','runtimeHandlerVerification','authEnforcementScope','originalFrontendSourceRefs','frontendSecondSourceRefs','frontendSecondTestSourceRefs','integrationStatus','pythonHttpEvidence']));

const inventory = JSON.parse(readFileSync(path.join(here, '.qa', 'desktop-acceptance', 'milestone-a-resume', 'interaction-inventory-final-v13.json'), 'utf8'));
const screenSources = {
  home: 'src/screens/HomeChat.tsx', chat: 'src/screens/HomeChat.tsx', agents: 'src/screens/AgentsDemoWorkspace.tsx; src/screens/OperationsScreens.tsx',
  research: 'src/screens/WorkspaceScreens.tsx', files: 'src/screens/WorkspaceScreens.tsx', automations: 'src/screens/OperationsScreens.tsx',
  calendar: 'src/screens/WorkspaceScreens.tsx', models: 'src/screens/OperationsScreens.tsx', settings: 'src/screens/OperationsScreens.tsx',
  codeq: 'src/screens/WorkspaceScreens.tsx', knowledge: 'src/screens/WorkspaceScreens.tsx', voice: 'src/screens/AdvancedScreen.tsx',
  assistant: 'src/screens/AdvancedScreen.tsx', communications: 'src/screens/AdvancedScreen.tsx', security: 'src/screens/AdvancedScreen.tsx',
  developer: 'src/screens/AdvancedScreen.tsx', engine: 'src/screens/AdvancedScreen.tsx', science: 'src/screens/AdvancedScreen.tsx',
  twin: 'src/screens/AdvancedScreen.tsx', vault: 'src/screens/AdvancedScreen.tsx', scraper: 'src/screens/AdvancedScreen.tsx',
  finetune: 'src/screens/AdvancedScreen.tsx', tools: 'src/screens/OperationsScreens.tsx'
};
const matrix = [];
const addControl = (route, state, control, index, kind) => {
  const isNav = Boolean(control.nav);
  const isTab = Boolean(control.tabKind || control.tab);
  const status = isNav ? 'Working — navigation only' : isTab ? 'Working — Demo tab navigation only' : 'Not tested — discovered in canonical DOM; activation not inferred';
  const evidence = isNav || isTab ? 'tests/e2e/all-workspaces-functional-smoke.spec.ts (route/tab activation only)' : 'No per-control activation evidence assigned by this inventory generator; see INTERACTION_STATE_COVERAGE_REPORT.md for manually recorded workflows.';
  matrix.push({ id: `${route}|${state}|${kind}|${index}`, route, canonicalState: state,
    screenTitle: kind === 'default' ? (inventory.routes.find(item => item.route === route)?.canonical?.title ?? '') : '',
    controlKind: kind, tag: control.tag ?? '', role: control.role ?? '', label: control.label ?? '', action: control.action ?? '',
    nav: control.nav ?? '', tabKind: control.tabKind ?? '', tab: control.tab ?? '', inputType: control.type ?? '', name: control.name ?? '',
    disabled: control.disabled ?? '', expanded: control.expanded ?? '', frontendSource: screenSources[route] ?? 'src/App.tsx',
    status, evidence, reproduction: isNav ? 'Run all-workspaces-functional-smoke.spec.ts and follow the route label.' : isTab ? 'Run all-workspaces-functional-smoke.spec.ts; this verifies tab activation only.' : 'Manual reproduction: navigate to the route/tab, activate the control, and compare resulting state with canonical HTML; no pass is claimed.' });
};
for (const route of inventory.routes) {
  (route.canonical?.controls ?? []).forEach((control, i) => addControl(route.route, 'default', control, i, 'default-control'));
  (route.canonical?.forms ?? []).forEach((control, i) => addControl(route.route, 'default', control, i, 'default-form'));
  for (const state of route.tabStates ?? []) {
    const name = state.label ?? `tab-${state.canonicalIndex}`;
    (state.canonicalState?.controls ?? []).forEach((control, i) => addControl(route.route, name, control, i, 'tab-control'));
    (state.canonicalState?.forms ?? []).forEach((control, i) => addControl(route.route, name, control, i, 'tab-form'));
  }
}
writeFileSync(path.join(outDir, 'INTERACTION_CONTROL_MATRIX.csv'), csv(matrix,
  ['id','route','canonicalState','screenTitle','controlKind','tag','role','label','action','nav','tabKind','tab','inputType','name','disabled','expanded','frontendSource','status','evidence','reproduction']));
console.log(JSON.stringify({routeRows: auditRows.length, interactionRows: matrix.length,
  requestCandidates: auditRows.filter(row => row.integrationStatus.startsWith('Request callsite')).length,
  staticOnly: auditRows.filter(row => row.integrationStatus.startsWith('Static reference')).length,
  noLiteral: auditRows.filter(row => row.integrationStatus.startsWith('No frontend')).length,
  tabNavRows: matrix.filter(row => row.status.startsWith('Working — Demo tab')).length}, null, 2));
