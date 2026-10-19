import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from '@playwright/test';

const routes = ['home','chat','agents','research','files','automations','calendar','models','settings','codeq','knowledge','voice','assistant','communications','security','developer','engine','science','twin','vault','scraper','finetune','tools'];
const canonicalPath = path.resolve(process.env.OPENAGENT_APPROVED_HTML || 'C:/Users/awais/Downloads/frontend final.html');
const canonicalHash = createHash('sha256').update(readFileSync(canonicalPath)).digest('hex').toUpperCase();
const expectedHash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const baseline = JSON.parse(readFileSync('visual-baseline.config.json', 'utf8'));
const expectedBrowserVersion = baseline.browser.version;
if (canonicalHash !== expectedHash) throw new Error(`Canonical source hash mismatch: ${canonicalHash}`);
const baseURL = process.env.OPENAGENT_UI_ORIGIN?.replace(/\/$/, '') || 'http://127.0.0.1:5180';
const outputPath = path.resolve(process.env.OPENAGENT_INTERACTION_AUDIT_OUTPUT || '.qa/desktop-acceptance/continued-milestone-a/interaction-inventory.json');
const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
if (browser.version() !== expectedBrowserVersion) throw new Error(`Expected Chrome ${expectedBrowserVersion}, got ${browser.version()}`);
const browserLabel = `Chrome ${browser.version()}`;

async function inspect(page, route, canonical, theme = 'light') {
  if (canonical) {
    await page.addInitScript(({ selectedTheme }) => localStorage.setItem('oa-demo-theme', JSON.stringify(selectedTheme)), { selectedTheme: theme });
    await page.goto(pathToFileURL(canonicalPath).href, { waitUntil: 'domcontentloaded' });
    await page.evaluate(value => { location.hash = value; }, route);
  } else {
    await page.addInitScript(({ selectedTheme }) => { localStorage.setItem('oa-second-data-mode', 'demo'); localStorage.setItem('oa-second-theme', selectedTheme); }, { selectedTheme: theme });
    await page.goto(`${baseURL}/?view=${route}`, { waitUntil: 'domcontentloaded' });
  }
  await page.evaluate(() => document.fonts.ready);
  const readySelectors = { chat: '#content .split-layout', codeq: '#content .codeq-root', agents: '#content .agent-demo-workspace', automations: '#content .automations-shell', voice: '#content .p-voice-hero' };
  if (!canonical && readySelectors[route]) await page.locator(readySelectors[route]).waitFor({ state: 'visible', timeout: 15000 });
  else if (!canonical) await page.waitForFunction(() => { const root=document.querySelector('#content .reference-screen-motion'); return !!root && root.childElementCount > 0 && root.innerText.trim().length > 12 && !/^Loading screen/i.test(root.innerText.trim()); }, null, { timeout: 15000 });
  await page.waitForTimeout(500);
  return page.evaluate(() => {
    const scope = document.querySelector('#content') || document.querySelector('.content-area') || document.body;
    const visible = element => {
      const rect = element.getBoundingClientRect(), style = getComputedStyle(element);
      return !!(rect.width && rect.height && style.visibility !== 'hidden' && style.display !== 'none');
    };
    const label = element => (element.getAttribute('aria-label') || element.getAttribute('title') || element.getAttribute('placeholder') || element.innerText || element.textContent || '').replace(/\s+/g,' ').trim();
    const controls = [...scope.querySelectorAll('button,a[role],input,select,textarea,summary,[role="tab"],[role="menuitem"]')].filter(visible).map((element,index) => ({
      index, tag: element.tagName.toLowerCase(), role: element.getAttribute('role') || '', label: label(element),
      action: element.getAttribute('data-action') || '', nav: element.getAttribute('data-nav') || '',
      tabKind: element.getAttribute('data-tab-kind') || '', tab: element.getAttribute('data-tab') || '',
      type: element.getAttribute('type') || '', name: element.getAttribute('name') || '',
      disabled: Boolean(element.disabled), expanded: element.getAttribute('aria-expanded'),
    }));
    const forms = [...scope.querySelectorAll('form')].filter(visible).map(form => ({
      action: form.getAttribute('data-form-action') || '', scope: form.getAttribute('data-scope') || '',
      fields: [...form.querySelectorAll('input,select,textarea')].map(field => ({name:field.name,type:field.type || field.tagName.toLowerCase(),label:field.labels?.[0]?.innerText?.trim() || field.getAttribute('aria-label') || field.placeholder || '',required:Boolean(field.required),disabled:Boolean(field.disabled)})),
      submit: [...form.querySelectorAll('button[type="submit"],input[type="submit"]')].map(label),
    }));
    const stateSignals = [...scope.querySelectorAll('[role="dialog"],[role="alert"],[role="status"],[aria-busy="true"],.trace-mount,.live-host,.toast,[role="menu"],details')].filter(visible).map(element => ({role:element.getAttribute('role')||'',className:typeof element.className==='string'?element.className:'',label:label(element).slice(0,180)}));
    const tabs = [...scope.querySelectorAll('[role="tab"],.tab')].filter(visible).map((element,index) => ({index,label:label(element),kind:element.getAttribute('data-tab-kind')||'',tab:element.getAttribute('data-tab')||'',selected:element.getAttribute('aria-selected')==='true'||element.classList.contains('active')}));
    const shell = document.querySelector('.topbar');
    const shellControls = shell ? [...shell.querySelectorAll('button,a[role],input,select,textarea,[role="menuitem"]')].filter(visible).map((element,index) => ({
      index, tag:element.tagName.toLowerCase(), role:element.getAttribute('role')||'', label:label(element),
      action:element.getAttribute('data-action')||'', nav:element.getAttribute('data-nav')||'',
      type:element.getAttribute('type')||'', disabled:Boolean(element.disabled), expanded:element.getAttribute('aria-expanded'),
    })) : [];
    return { title:scope.querySelector('h1')?.innerText?.trim() || document.title, controls, forms, tabs, stateSignals, shellControls };
  });
}

async function captureVisibleState(page) {
  return page.evaluate(() => {
    const scope = document.querySelector('#content') || document.querySelector('.content-area') || document.body;
    const visible = element => {
      const rect = element.getBoundingClientRect(), style = getComputedStyle(element);
      return Boolean(rect.width && rect.height && style.visibility !== 'hidden' && style.display !== 'none');
    };
    const label = element => (element.getAttribute('aria-label') || element.getAttribute('title') || element.getAttribute('placeholder') || element.innerText || element.textContent || '').replace(/\s+/g, ' ').trim();
    const controlSelector = 'button,a[role],input,select,textarea,summary,[role="tab"],[role="menuitem"]';
    const shell = document.querySelector('.topbar');
    const shellControls = shell ? [...shell.querySelectorAll(controlSelector)].filter(visible).map(element => ({
      tag:element.tagName.toLowerCase(),role:element.getAttribute('role')||'',label:label(element),
      action:element.getAttribute('data-action')||'',value:element.getAttribute('data-value')||'',
      disabled:Boolean(element.disabled),expanded:element.getAttribute('aria-expanded'),
    })) : [];
    return {
      text: scope.innerText.slice(0, 8000),
      controls: [...scope.querySelectorAll(controlSelector)].filter(visible).map(element => ({
        tag: element.tagName.toLowerCase(), role: element.getAttribute('role') || '', label: label(element),
        action: element.getAttribute('data-action') || '', value: element.getAttribute('data-value') || '',
        tabKind: element.getAttribute('data-tab-kind') || '', tab: element.getAttribute('data-tab') || '',
        disabled: Boolean(element.disabled), expanded: element.getAttribute('aria-expanded'),
      })),
      forms: [...scope.querySelectorAll('form')].filter(visible).map(form => ({
        action: form.getAttribute('data-form-action') || '', scope: form.getAttribute('data-scope') || '',
        fields: [...form.querySelectorAll('input,select,textarea')].map(field => ({
          name: field.name, type: field.type || field.tagName.toLowerCase(),
          label: field.labels?.[0]?.innerText?.trim() || field.getAttribute('aria-label') || field.placeholder || '',
          required: Boolean(field.required), disabled: Boolean(field.disabled),
        })),
        submit: [...form.querySelectorAll('button[type="submit"],input[type="submit"]')].map(label),
      })),
      shellControls,
      dialogs: [...document.body.querySelectorAll('[role="dialog"],.modal')].filter(visible).map(element => ({
        label: label(element), title: element.querySelector('h1,h2,h3,[role="heading"]')?.textContent?.trim() || '',
        fields: [...element.querySelectorAll('input,select,textarea')].map(field => field.getAttribute('aria-label') || field.labels?.[0]?.innerText?.trim() || field.placeholder || ''),
        controls: [...element.querySelectorAll('button')].filter(visible).map(label),
      })),
      menus: [...document.body.querySelectorAll('[role="menu"],[role="listbox"],[role="menuitem"],details[open],.menu,.context-menu')].filter(visible).map(element => ({ role: element.getAttribute('role') || '', label: label(element), controls: [...element.querySelectorAll('button,[role="option"],[role="menuitem"]')].filter(visible).map(label) })),
      notifications: [...document.body.querySelectorAll('[role="alert"],[role="status"],.toast,.notification,.note-panel')].filter(visible).map(element => ({ role: element.getAttribute('role') || '', className: typeof element.className === 'string' ? element.className : '', label: label(element) })),
      loading: [...scope.querySelectorAll('[aria-busy="true"],.loading,.skeleton')].filter(visible).map(label),
      execution: [...scope.querySelectorAll('.trace-mount,.execution-panel,.live-inline')].filter(visible).map(element => ({ label: label(element).slice(0, 1200), controls: [...element.querySelectorAll('button')].filter(visible).map(label) })),
    };
  });
}

const records = [];
for (const route of routes) {
  const refContext = await browser.newContext({ viewport:{width:1440,height:900},deviceScaleFactor:1 });
  const appContext = await browser.newContext({ viewport:{width:1440,height:900},deviceScaleFactor:1 });
  const refPage = await refContext.newPage(), appPage = await appContext.newPage();
  const canonical = await inspect(refPage,route,true); const react = await inspect(appPage,route,false);
  const canonicalTabs = [...new Set(canonical.tabs.map(tab => `${tab.kind}|${tab.tab}|${tab.label}`))];
  const reactControls = new Set(react.controls.map(control => control.label.toLowerCase().replace(/\s+/g,' ').trim()));
  const unmatchedCanonicalControls = canonical.controls.filter(control => {
    return control.label && !reactControls.has(control.label.toLowerCase().replace(/\s+/g,' ').trim());
  });
  const tabStates = [];
  for (const tab of canonical.tabs) {
    const normalized = tab.label.replace(/\s+/g,' ').trim();
    const target = react.tabs.filter(candidate => candidate.label.replace(/\s+/g,' ').trim() === normalized);
    if (!target.length) { tabStates.push({label:normalized,canonicalIndex:tab.index,reactFound:false}); continue; }
    for (const page of [refPage, appPage]) {
      const allTabs = await page.locator('#content [role="tab"],#content .tab').all();
      const matches = [];
      for (const candidate of allTabs) {
        const text = (await candidate.innerText().catch(()=>'' )).replace(/\s+/g,' ').trim();
        if (text === normalized) matches.push(candidate);
      }
      const index = canonical.tabs.filter(candidate => candidate.label.replace(/\s+/g,' ').trim() === normalized).findIndex(candidate => candidate.index === tab.index);
      if (matches[index]) { await matches[index].click().catch(()=>{}); await page.waitForTimeout(100); }
    }
    const [canonicalState, reactState] = await Promise.all([captureVisibleState(refPage), captureVisibleState(appPage)]);
    tabStates.push({label:normalized,canonicalIndex:tab.index,reactFound:true,canonicalState,reactState});
  }
  records.push({route,canonical,react,tabStates,comparison:{canonicalVisibleControls:canonical.controls.length,reactVisibleControls:react.controls.length,canonicalTabs,unmatchedCanonicalControls}});
  await refContext.close(); await appContext.close();
}
await browser.close();
mkdirSync(path.dirname(outputPath),{recursive:true});
const summary = { canonicalSourceSha256:canonicalHash,browser:browserLabel,viewport:'1440x900',deviceScaleFactor:1,dataMode:'deterministic demo for React; standalone canonical fixtures',method:'Screen-scoped DOM inventory for all 23 default routes plus every visible canonical tab state; records route controls and forms, topbar controls separately, and searches the full document body for open dialogs, menus, alerts, status signals and execution panels. No controls are activated. This is inventory evidence, not proof every dialog state was compared.',routes:records };
writeFileSync(outputPath,`${JSON.stringify(summary,null,2)}\n`);
const mdPath = outputPath.replace(/\.json$/i,'.md');
const lines = ['# Desktop interaction-state inventory','',`- Canonical SHA-256: \`${canonicalHash}\``,`- Browser: ${browserLabel} at 1440×900, DPR 1`,'- React evaluated in deterministic demo mode. Canonical HTML was loaded read-only.','- Route content and visible tab states were inspected in both applications. Topbar controls are recorded separately; open overlays/status regions are searched across the document body. This inventory does not claim every dialog/notification/timer state was activated or screenshot-compared.','- See `INTERACTION_STATE_CATALOG.md` for action trigger families and the complete state/capture boundary.','', '| Workspace | Canonical tabs | React tab match | Forms (canonical / React) | Visible route-content controls (canonical / React) | Unmatched canonical labels |','|---|---|---:|---:|---:|---|'];
for (const route of records) {
 const tabs = route.comparison.canonicalTabs.map(item=>item.split('|').at(-1)).filter(Boolean);
 const missingTabs = [...new Set(tabs)].filter(label=>!route.react.tabs.some(tab=>tab.label.replace(/\s+/g,' ').trim()===label.replace(/\s+/g,' ').trim()));
 lines.push(`| ${route.route} | ${tabs.join(', ')||'—'} | ${missingTabs.length ? `missing: ${missingTabs.join(', ')}` : `${new Set(tabs).size}/${new Set(tabs).size}`} | ${route.canonical.forms.length} / ${route.react.forms.length} | ${route.comparison.canonicalVisibleControls} / ${route.comparison.reactVisibleControls} | ${route.comparison.unmatchedCanonicalControls.map(control=>control.label).filter(Boolean).slice(0,8).join('; ')||'—'} |`);
}
lines.push('','## Default-state detail','');
for (const route of records) {
 lines.push(`### ${route.route}`,'');
 lines.push(`- Canonical controls: ${route.canonical.controls.map(control=>control.label).filter(Boolean).join(' · ')||'none'}`);
 lines.push(`- React controls: ${route.react.controls.map(control=>control.label).filter(Boolean).join(' · ')||'none'}`);
 lines.push(`- Forms: ${route.canonical.forms.map(form=>`${form.fields.length} fields`).join(', ')||'none visible'}; React ${route.react.forms.map(form=>`${form.fields.length} fields`).join(', ')||'none visible'}`);
 lines.push(`- Signals: ${route.canonical.stateSignals.map(signal=>signal.className||signal.role).join(', ')||'none'}; React ${route.react.stateSignals.map(signal=>signal.className||signal.role).join(', ')||'none'}`);
 lines.push(`- Tab states exercised: ${route.tabStates.map(tab=>`${tab.label}${tab.reactFound?'':' (missing in React)'}`).join(', ')||'none'}`,'');
}
lines.push('## Visible controls and state inventory for each tab','');
lines.push('The state snapshots are read from the rendered canonical DOM after selecting each tab. They include visible controls and data-action triggers, forms and field requirements, dialogs, menus, notifications/status/error signals, loading markers, and execution controls. An empty dialog/menu/notification/loading list means none was open in that captured tab state; it does not mean every trigger’s resulting overlay has been captured.','');
for (const route of records) {
  if (!route.tabStates.length) continue;
  lines.push(`### ${route.route}`,'');
  for (const state of route.tabStates) {
    const c=state.canonicalState, a=state.reactState;
    if (!c || !a) {
      lines.push(`- **${state.label}** — not paired: ${!c ? 'canonical state was not captured' : ''}${!c && !a ? '; ' : ''}${!a ? 'React matching tab/state was not captured' : ''}.`);
      continue;
    }
    const labels=items=>items.map(item=>typeof item==='string'?item:item.label||item.title||'').filter(Boolean);
    const forms=items=>items.map(form=>`${form.action||'(no action)'}: ${form.fields.map(field=>field.label||field.name||field.type).join(', ')||'no fields'}`);
    lines.push(`- **${state.label}** — canonical / React: ${c.controls.length} / ${a.controls.length} visible controls; forms ${c.forms.length} / ${a.forms.length}; dialogs ${c.dialogs.length} / ${a.dialogs.length}; menus ${c.menus.length} / ${a.menus.length}; notifications/status ${c.notifications.length} / ${a.notifications.length}; loading markers ${c.loading.length} / ${a.loading.length}; execution regions ${c.execution.length} / ${a.execution.length}.`);
    lines.push(`  - Canonical controls: ${labels(c.controls).join(' · ')||'none'}`);
    lines.push(`  - Canonical forms: ${forms(c.forms).join(' · ')||'none visible'}`);
    lines.push(`  - Canonical data-action triggers: ${[...new Set(c.controls.map(control=>control.action).filter(Boolean))].join(', ')||'none visible'}`);
    const missing=labels(c.controls).filter(label=>!new Set(labels(a.controls).map(value=>value.toLowerCase())).has(label.toLowerCase()));
    lines.push(`  - Control-label differences to review: ${[...new Set(missing)].join(' · ')||'none by exact accessible label'}`);
  }
  lines.push('');
}
writeFileSync(mdPath,`${lines.join('\n')}\n`);
console.log(`Audited ${records.length} routes. Inventory: ${outputPath}`);
