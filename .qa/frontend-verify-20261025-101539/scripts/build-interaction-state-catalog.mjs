import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';

// Turns the read-only canonical DOM audit and paired-capture manifests into a
// reviewable state catalog. This deliberately does not click or submit controls:
// the HTML contains destructive-looking demo actions as well as downloads, so
// activation must be an explicit, separately reviewed test.
const root = path.resolve(import.meta.dirname, '..');
const canonicalPath = path.resolve(process.env.OPENAGENT_APPROVED_HTML || 'C:/Users/awais/Downloads/frontend final.html');
const auditPath = path.resolve(process.env.OPENAGENT_INTERACTION_AUDIT_INPUT || '.qa/desktop-acceptance/milestone-a-resume/interaction-inventory-final.json');
const interactionManifestPath = path.resolve(process.env.OPENAGENT_INTERACTION_MANIFEST || '.qa/desktop-acceptance/milestone-a-resume/interactions-final-v8/manifest.json');
const agentsPreviewManifestPath = process.env.OPENAGENT_AGENTS_PREVIEW_MANIFEST ? path.resolve(process.env.OPENAGENT_AGENTS_PREVIEW_MANIFEST) : null;
const tabManifestPath = path.resolve(process.env.OPENAGENT_TAB_MANIFEST || '.qa/desktop-acceptance/milestone-a-resume/tab-states-final-v3/manifest.json');
const mdPath = path.resolve(process.env.OPENAGENT_INTERACTION_CATALOG_MD || 'INTERACTION_STATE_CATALOG.md');
const jsonPath = path.resolve(process.env.OPENAGENT_INTERACTION_CATALOG_JSON || '.qa/desktop-acceptance/milestone-a-resume/interaction-state-catalog.json');
const expectedSha = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';

const sha = file => createHash('sha256').update(readFileSync(file)).digest('hex').toUpperCase();
const canonicalSha = sha(canonicalPath);
if (canonicalSha !== expectedSha) throw new Error(`Canonical hash mismatch: ${canonicalSha}`);
const audit = JSON.parse(readFileSync(auditPath, 'utf8'));
const interactionManifest = JSON.parse(readFileSync(interactionManifestPath, 'utf8'));
const agentsPreviewManifest = agentsPreviewManifestPath ? JSON.parse(readFileSync(agentsPreviewManifestPath, 'utf8')) : null;
const tabManifest = JSON.parse(readFileSync(tabManifestPath, 'utf8'));
for (const [name, data] of [['DOM audit', audit], ['interaction captures', interactionManifest], ...(agentsPreviewManifest ? [['Agents preview captures', agentsPreviewManifest]] : []), ['tab captures', tabManifest]]) {
  if (data.canonicalSourceSha256 !== expectedSha) throw new Error(`${name} was captured against a different canonical HTML`);
}
if (audit.routes.length !== 23) throw new Error(`Expected 23 routes in DOM audit, got ${audit.routes.length}`);

const routeLabels = {
  home: 'Home', chat: 'Chat', agents: 'Agents', research: 'Research', files: 'Files', automations: 'Automations',
  calendar: 'Calendar', models: 'Models', settings: 'Settings', codeq: 'CodeQ', knowledge: 'Knowledge', voice: 'Voice',
  assistant: 'Personal Assistant', communications: 'Communications', security: 'Security', developer: 'Developer',
  engine: 'Adaptive Engine', science: 'Scientific AI', twin: 'Personal Context', vault: 'Vault', scraper: 'Web Scraper',
  finetune: 'Fine-tuning', tools: 'All Tools',
};
const unique = values => [...new Set(values.filter(Boolean))];
const norm = value => String(value || '').replace(/\s+/g, ' ').trim();
const allStateSnapshots = route => [
  { name: 'Initial', state: route.canonical },
  ...route.tabStates.map(tab => ({ name: `Tab: ${tab.label}`, state: tab.canonicalState })),
];
const classify = action => {
  const a = String(action || '');
  if (/^(living|trace|previewRun|stopPreview)/i.test(a)) return 'execution';
  if (/(palette|Picker|Modal|Create|Edit|Details|Approval|Activity|History|AllActivity|notifications|transcript)/i.test(a)) return 'dialog/detail candidate';
  if (/(Menu|more|feedback|options|context)/i.test(a)) return 'menu/action candidate';
  if (/^(search|filter|sort|toggle)/i.test(a)) return 'search/filter/toggle';
  if (/^(quick:|fileFromTree|select|pick|open)/i.test(a)) return 'selection/navigation';
  if (/^(demo|core)/i.test(a)) return 'demo record action';
  return a ? 'other action' : 'control without data-action';
};
const statusTokens = [
  'loading', 'empty', 'error', 'failed', 'failure', 'running', 'paused', 'completed', 'cancelled', 'pending', 'approval',
  'recording', 'transcribing', 'success', 'retry', 'offline', 'unavailable',
];
function statusesFor(snapshots) {
  const text = snapshots.map(({ state }) => state?.text || '').join('\n').toLowerCase();
  const signals = snapshots.flatMap(({ state }) => [
    ...(state?.notifications || []).map(item => item.label),
    ...(state?.loading || []),
    ...(state?.execution || []).map(item => item.label),
  ]).join('\n').toLowerCase();
  return statusTokens.filter(token => new RegExp(`\\b${token}\\b`, 'i').test(`${text}\n${signals}`));
}
function actionsFor(snapshots) {
  const map = new Map();
  for (const { name, state } of snapshots) for (const [place, controls] of [['route/tab panel', state?.controls || []], ['application topbar', state?.shellControls || []]]) for (const control of controls) {
    const action = control.action || '';
    const label = norm(control.label);
    const key = `${place}\u0000${action}\u0000${control.value || ''}\u0000${label}`;
    const row = map.get(key) || { action, value: control.value || '', label, category: classify(action), place, observedIn: [] };
    const observation = `${name} (${place})`;
    if (!row.observedIn.includes(observation)) row.observedIn.push(observation);
    map.set(key, row);
  }
  return [...map.values()];
}
function formsFor(snapshots) {
  const map = new Map();
  for (const { name, state } of snapshots) for (const form of state?.forms || []) {
    const fields = (form.fields || []).map(field => ({
      label: norm(field.label), name: field.name || '', type: field.type || '', required: Boolean(field.required), disabled: Boolean(field.disabled),
    }));
    const key = JSON.stringify([form.action, fields]);
    const row = map.get(key) || { action: form.action || '', fields, submit: form.submit || [], observedIn: [] };
    if (!row.observedIn.includes(name)) row.observedIn.push(name);
    map.set(key, row);
  }
  return [...map.values()];
}
const catalogRoutes = audit.routes.map(route => {
  const snapshots = allStateSnapshots(route);
  const tabs = unique(route.canonical.tabs.map(tab => norm(tab.label)));
  const actions = actionsFor(snapshots);
  return {
    route: route.route,
    workspace: routeLabels[route.route] || route.route,
    canonicalTabs: tabs,
    reactTabs: unique(route.react.tabs.map(tab => norm(tab.label))),
    tabMatch: route.tabStates.map(tab => ({ tab: norm(tab.label), reactFound: Boolean(tab.reactFound), capturedInBoth: Boolean(tab.canonicalState && tab.reactState) })),
    controls: actions,
    forms: formsFor(snapshots),
    observedStatusTokens: statusesFor(snapshots),
    defaultStateSignals: route.canonical.stateSignals.map(signal => ({ role: signal.role, className: signal.className, label: norm(signal.label) })),
    observedDialogOrMenuAtRest: snapshots.some(({ state }) => (state?.dialogs?.length || 0) + (state?.menus?.length || 0) > 0),
    comparison: {
      canonicalDefaultControlCount: route.canonical.controls.length,
      reactDefaultControlCount: route.react.controls.length,
      canonicalForms: route.canonical.forms.length,
      reactForms: route.react.forms.length,
      canonicalMatchedTabStateCount: route.tabStates.filter(tab => tab.reactFound && tab.canonicalState && tab.reactState).length,
    },
  };
});

const captureCoverage = Object.entries(interactionManifest.byRouteState || {}).map(([key, value]) => ({
  state: key,
  pairs: value.captures,
  exactPercent: value.exactPercent,
  minPercent: value.minimumPercent,
  maxPercent: value.maximumPercent,
}));
for (const item of agentsPreviewManifest?.captures ?? []) captureCoverage.push({
  state: `agents/${item.state}`,
  pairs: 1,
  exactPercent: item.exactPercent,
  minPercent: item.exactPercent,
  maxPercent: item.exactPercent,
});
const interactionCaptureSets = [interactionManifest, ...(agentsPreviewManifest ? [agentsPreviewManifest] : [])];
const interactionPixelCount = interactionCaptureSets.reduce((total, manifest) => total + (manifest.captures ?? []).reduce((sum, item) => sum + (item.pixels ?? 0), 0), 0);
const interactionEqualPixelCount = interactionCaptureSets.reduce((total, manifest) => total + (manifest.captures ?? []).reduce((sum, item) => sum + (item.equalPixels ?? 0), 0), 0);
const interactionPairCount = interactionCaptureSets.reduce((total, manifest) => total + (manifest.captures?.length ?? 0), 0);
const interactionExactPercent = interactionPixelCount ? Number((100 * interactionEqualPixelCount / interactionPixelCount).toFixed(2)) : interactionManifest.overall?.exactPercent ?? null;
const catalog = {
  canonicalSourceSha256: canonicalSha,
  generatedFrom: {
    domAudit: path.relative(root, auditPath).replaceAll('\\', '/'),
    interactionManifest: path.relative(root, interactionManifestPath).replaceAll('\\', '/'),
    ...(agentsPreviewManifestPath ? { agentsPreviewManifest: path.relative(root, agentsPreviewManifestPath).replaceAll('\\', '/') } : {}),
    tabManifest: path.relative(root, tabManifestPath).replaceAll('\\', '/'),
  },
  environment: {
    browser: audit.browser,
    defaultInventoryViewport: audit.viewport,
    deviceScaleFactor: audit.deviceScaleFactor,
    interactionViewports: interactionManifest.viewports,
    themes: interactionManifest.themes,
    demoApiCalls: interactionCaptureSets.reduce((sum, manifest) => sum + (manifest.demoApiCalls ?? 0), 0),
    browserErrors: interactionCaptureSets.reduce((sum, manifest) => sum + (manifest.browserErrors ?? 0), 0),
  },
  methodNotes: [
    'DOM inventory includes canonical initial routes and every rendered canonical tab state; the React column only establishes that a corresponding rendered control/tab was found.',
    'Controls are listed as triggers, not as proof their resulting overlay or action was activated.',
    'Forms are field inventories; invalid, submitting, success, and failure submissions are not inferred from the presence of a form.',
    'Status tokens are lexical signals in captured default/tab DOM text, not a verified transition matrix.',
    'Only states in paired screenshot manifests are marked captured. A missing state means unverified, not absent from the product.',
    'This generator is read-only with respect to the canonical HTML and deliberately does not activate controls or submit forms.',
  ],
  routes: catalogRoutes,
  pairedInteractionCaptureCoverage: captureCoverage,
  tabCaptureManifest: {
    file: path.relative(root, tabManifestPath).replaceAll('\\', '/'),
    totalPairs: tabManifest.overall?.captures ?? tabManifest.totalCaptures ?? null,
    exactPercent: tabManifest.overall?.exactPercent ?? null,
  },
};
writeFileSync(jsonPath, `${JSON.stringify(catalog, null, 2)}\n`);

const md = [
  '# Canonical desktop interaction-state catalog', '',
  `- Canonical HTML SHA-256: \`${canonicalSha}\` (validated before generation).`,
  `- DOM source: \`${catalog.generatedFrom.domAudit}\`; paired state evidence: \`${catalog.generatedFrom.interactionManifest}\`${catalog.generatedFrom.agentsPreviewManifest ? ` plus \`${catalog.generatedFrom.agentsPreviewManifest}\`` : ''}; tab evidence: \`${catalog.generatedFrom.tabManifest}\`.`,
  `- Environment: ${audit.browser}, inventory ${audit.viewport}, DPR ${audit.deviceScaleFactor}; matched captures ${interactionManifest.viewports.map(v => `${v.width}×${v.height}`).join(' and ')}, ${interactionManifest.themes.join('/')} themes.`,
  `- Paired interaction captures report ${interactionPairCount} pairs across named states, ${interactionExactPercent ?? 'unknown'}% strict exact pixels, ${catalog.environment.demoApiCalls} demo API calls, ${catalog.environment.browserErrors} browser errors.`,
  `- Tab matrix manifest: ${catalog.tabCaptureManifest.totalPairs ?? 'pair count not recorded'} pairs; ${catalog.tabCaptureManifest.exactPercent ?? 'score not recorded'}% exact pixels.`, '',
  '## How to read this inventory', '',
  'This catalog enumerates the canonical route DOM and all visible canonical tab DOM states, including action triggers, form fields, modal/menu/status signals, and text-level lifecycle markers. It distinguishes discovery from activation: an action listed here is an available trigger, not evidence that its resulting state was exercised. Forms are recorded by fields, not treated as validated workflows. Only states present in the paired screenshot coverage list below have matched-state visual evidence. Unlisted or unpaired states remain unverified; they are not assumed missing or complete.', '',
  'No controls are activated by the generator. That is intentional: actions such as record deletion, exports, file selection, clipboard operations, and media workflows require explicit deterministic fixture handling before safe matched-state probing. The canonical HTML is read-only and is SHA-checked on every run. Exact accessible labels, record values, and the tab where each trigger appeared are preserved in the JSON catalog.', '',
  'The `Forms and fields` column lists only forms visible in the initial or tab DOM snapshot. Hidden dialog forms are not counted there; their opener actions are listed, and their fields/validation states remain unverified until each dialog is opened and inspected. The lifecycle column reports words present in visible text and is not proof that transitions were exercised.', '',
  '## Route-level interaction inventory', '',
  '| Workspace | Canonical tabs | React tab discovery | Dialog/detail trigger families | Menu/action trigger families | Execution trigger families | Visible forms and fields | Lifecycle words in captured text (not transition proof) |',
  '|---|---|---|---|---|---|---|---|',
];
for (const route of catalogRoutes) {
  const byType = type => route.controls.filter(control => control.category === type);
  const triggerSummary = type => unique(byType(type).map(control => control.action || control.label)).join(', ') || '—';
  const formSummary = route.forms.map(form => `${form.action || '(unnamed)'}: ${form.fields.map(field => `${field.label || field.name || field.type}${field.required ? '*' : ''}`).join(', ') || 'no fields'}`).join('<br>') || '—';
  const tabSummary = route.canonicalTabs.join(', ') || '—';
  const tabMatch = route.tabMatch.length ? `${route.comparison.canonicalMatchedTabStateCount}/${route.tabMatch.length} states paired` : 'no tabs';
  md.push(`| ${route.workspace} (\`${route.route}\`) | ${tabSummary} | ${tabMatch} | ${triggerSummary('dialog/detail candidate')} | ${triggerSummary('menu/action candidate')} | ${triggerSummary('execution')} | ${formSummary} | ${route.observedStatusTokens.join(', ') || 'none detected in captured default/tab text'} |`);
}
md.push('', '## Paired interaction states with screenshot evidence', '', '| State | Pairs | Exact RGBA match | Range |', '|---|---:|---:|---:|');
for (const item of captureCoverage) md.push(`| \`${item.state}\` | ${item.pairs} | ${item.exactPercent}% | ${item.minPercent}%–${item.maxPercent}% |`);
md.push('', 'These are the only state names recorded by the current paired interaction manifest. The table does not imply that every listed state is exhaustive for that screen or that other states are absent.', '', '## Explicit coverage gaps', '',
  '- No exhaustive activation matrix exists yet for every dialog, nested menu, create/edit/delete form, invalid submission, validation error, submitting indicator, success result, or failure result on all 23 routes.',
  '- Notification loading, empty, populated, unread/read, refresh and error states are not represented in the paired interaction manifest.',
  '- Route-level loading, empty, error, retry, approval, failed, cancelled and recovery states are not comprehensively paired across all workspaces.',
  '- Local demo lifecycle actions are only evidenced where named in the paired interaction manifest; controls visible in the catalog remain unverified unless specifically captured.',
  '- Accessibility activation paths (keyboard tab order, Enter/Space, Escape, outside dismissal, focus trap/restoration) are not proven by this visual inventory.',
  '- React tab discovery means the matching label was found in the React DOM and captured; it does not certify visual parity or all nested controls.', '',
  '## Machine-readable source', '',
  `Full controls, accessible labels, tab-level form fields, default status signals, and capture details are in [\`${path.relative(root, jsonPath).replaceAll('\\', '/')}\`](\`${path.relative(root, jsonPath).replaceAll('\\', '/')}\`).`, '',
  'Regenerate after refreshing the read-only DOM audit and screenshot manifests with:', '',
  '```powershell',
  'node scripts/build-interaction-state-catalog.mjs',
  '```', '',
  'The generator refuses mismatched canonical hashes or non-23-route audits and does not mutate or launch actions in the canonical HTML.',
);
writeFileSync(mdPath, `${md.join('\n')}\n`);
console.log(`Wrote ${mdPath} and ${jsonPath}; cataloged ${catalogRoutes.length} routes, ${catalogRoutes.reduce((n, route) => n + route.controls.length, 0)} unique visible trigger rows, ${captureCoverage.length} paired named states.`);
