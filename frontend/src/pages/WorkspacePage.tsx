import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from 'react';
import { api, ApiError, record, records } from '../lib/react/api';
import { PageFrame } from '../components/PageFrame';
import { MessageContent } from '../components/chat/MessageContent';
import { FilePlus2 } from 'lucide-react';
import './workspace-page.css';
import { createWorkspaceDocumentPayload, updateWorkspaceDocumentPayload, updateWorkspaceItemPayload, workspaceImportPayload, workspaceImportRoute, workspaceLibraryPaths } from './workspace-utils';

type Row = Record<string, unknown>;
type Pane = 'edit' | 'preview' | 'details';
type ResizablePane = 'files';
const paneLimits = { files: { min: 220, max: 420 } } as const;
const project = () => new URL(location.href).searchParams.get('project') || localStorage.getItem('oa-project') || 'default';
const val = (r: Row, k: string, f = '—') => r[k] == null || r[k] === '' ? f : String(r[k]);
function workspaceError(cause: unknown, subject: string) {
  if (cause instanceof ApiError) {
    if (cause.status === 401) return `Sign in to access ${subject}.`;
    if (cause.status === 403) return `You do not have permission to access ${subject}.`;
    if (cause.status === 404) return `${subject} could not be found. Refresh the project files and try again.`;
    if (cause.status === 408 || cause.status >= 500) return `${subject} is unavailable. Check the local service and retry.`;
  }
  return cause instanceof Error ? cause.message : `${subject} could not be loaded.`;
}

function Preview({ content, format }: { content: string; format: string }) {
  if (!content.trim()) return <p className="oa-document-empty">Preview will appear here as you write.</p>;
  if (format === 'text') return <article className="oa-document-preview oa-document-plain">{content}</article>;
  return <article className="oa-document-preview"><MessageContent content={content} /></article>;
}
export function WorkspacePage() {
  const activeProject = project();
  const [items, setItems] = useState<Row[]>([]), [selected, setSelected] = useState<Row | null>(null), [query, setQuery] = useState(''), [title, setTitle] = useState(''), [content, setContent] = useState(''), [format, setFormat] = useState('markdown'), [versions, setVersions] = useState<Row[]>([]), [dirty, setDirty] = useState(false), [draftOpen, setDraftOpen] = useState(false), [busy, setBusy] = useState(false), [fileState, setFileState] = useState<'loading' | 'ready' | 'error'>('loading'), [error, setError] = useState(''), [notice, setNotice] = useState(''), [pane, setPane] = useState<Pane>('edit');
  const [selectedProject, setSelectedProject] = useState(activeProject);
  const [sortBy, setSortBy] = useState<'updated' | 'name' | 'type'>('updated');
  const visibleSelected = selectedProject === activeProject ? selected : null;
  const showEditor = Boolean(visibleSelected || draftOpen);
  const [filesWidth, setFilesWidth] = useState(430);
  const drag = useRef<{ pane: ResizablePane; pointerId: number; startX: number; startWidth: number } | null>(null);
  const openRevision = useRef(0);
  const importFileRef = useRef<HTMLInputElement>(null);
  function updatePaneWidth(which: ResizablePane, width: number) {
    const limits = paneLimits[which], next = Math.max(limits.min, Math.min(limits.max, width));
    setFilesWidth(next);
  }
  function onSeparatorKey(which: ResizablePane, event: KeyboardEvent<HTMLDivElement>) {
    const current = filesWidth, step = event.shiftKey ? 40 : 16;
    let next = current;
    if (event.key === 'ArrowLeft') next = current - step;
    else if (event.key === 'ArrowRight') next = current + step;
    else if (event.key === 'Home') next = paneLimits[which].min;
    else if (event.key === 'End') next = paneLimits[which].max;
    else return;
    event.preventDefault(); updatePaneWidth(which, next);
  }
  function onSeparatorPointerDown(which: ResizablePane, event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    event.preventDefault();
    drag.current = { pane: which, pointerId: event.pointerId, startX: event.clientX, startWidth: filesWidth };
  }
  function onMobileTabKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    let next: Pane | null = null;
    const tabs: Pane[] = ['preview', 'edit', 'details'], index = tabs.indexOf(pane);
    if (event.key === 'ArrowLeft') next = tabs[Math.max(0, index - 1)];
    else if (event.key === 'ArrowRight') next = tabs[Math.min(tabs.length - 1, index + 1)];
    else if (event.key === 'Home') next = tabs[0];
    else if (event.key === 'End') next = tabs[tabs.length - 1];
    if (!next) return;
    event.preventDefault(); setPane(next);
    hostTabRef.current?.querySelector<HTMLButtonElement>(`[data-pane-tab="${next}"]`)?.focus();
  }
  useEffect(() => {
    const move = (event: globalThis.PointerEvent) => {
      const current = drag.current;
      if (!current || current.pointerId !== event.pointerId) return;
      const delta = event.clientX - current.startX;
      updatePaneWidth(current.pane, current.startWidth + delta);
    };
    const stop = (event: globalThis.PointerEvent) => { if (drag.current?.pointerId === event.pointerId) drag.current = null; };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', stop); window.addEventListener('pointercancel', stop);
    return () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', stop); window.removeEventListener('pointercancel', stop); };
  }, [filesWidth]);
  const gridStyle = { '--oa-files-list-width': `${filesWidth}px` } as CSSProperties;
  const hostTabRef = useRef<HTMLDivElement>(null);
  const orderedItems = [...items].sort((a, b) => {
    const left = sortBy === 'name' ? val(a, 'title', '').toLocaleLowerCase() : sortBy === 'type' ? val(a, 'format', val(a, 'kind', '')).toLocaleLowerCase() : val(a, 'updated_at', val(a, 'created_at', ''));
    const right = sortBy === 'name' ? val(b, 'title', '').toLocaleLowerCase() : sortBy === 'type' ? val(b, 'format', val(b, 'kind', '')).toLocaleLowerCase() : val(b, 'updated_at', val(b, 'created_at', ''));
    return sortBy === 'updated' ? right.localeCompare(left) : left.localeCompare(right);
  });

  // The shell keeps this page mounted while switching projects. Clear the old
  // project's open document before paint so its content cannot linger in the
  // editor/inspector after the active scope changes.
  useLayoutEffect(() => {
    ++openRevision.current; setItems([]); setSelected(null); setDraftOpen(false); setSelectedProject(activeProject); setTitle(''); setContent('');
    setVersions([]); setDirty(false); setError(''); setNotice(''); setPane('edit');
  }, [activeProject]);

  async function load(signal?: AbortSignal) {
    setBusy(true); setError(''); setFileState('loading');
    try {
      const paths = workspaceLibraryPaths(project(), query);
      const [a, b] = await Promise.all([api.get<unknown>(paths.workspace, signal), api.get<unknown>(paths.documents, signal)]);
      if (signal?.aborted) return;
      const nextItems = [...records(b, 'items').map(x => ({ ...x, _source: 'document', id: x.doc_id ?? x.id })), ...records(a, 'items').map(x => ({ ...x, _source: 'workspace' }))];
      setItems(nextItems);
      // Match the files reference: open the first real project item as the
      // initial preview. The selection always comes from this scoped API read.
      if (!selected && !draftOpen && nextItems[0]) void open(nextItems[0]);
      setFileState('ready');
    } catch (e) { if (!signal?.aborted) { setError(workspaceError(e, 'Project files')); setFileState('error'); } }
    finally { if (!signal?.aborted) setBusy(false); }
  }
  useEffect(() => { const controller = new AbortController(); const timer = window.setTimeout(() => void load(controller.signal), 180); return () => { window.clearTimeout(timer); controller.abort(); }; }, [query, project()]);

  async function open(item: Row) {
    const openProject = project();
    const revision = ++openRevision.current;
    setSelected(item); setDraftOpen(true); setSelectedProject(openProject); setTitle(String(item.title ?? '')); setContent(String(item.content ?? '')); setFormat(String(item.format ?? 'markdown')); setDirty(false); setVersions([]); setPane('preview');
    if (item._source === 'document') try {
      const r = record(await api.get(`/api/document/${encodeURIComponent(String(item.id))}`)); const d = record(r.document);
      if (project() !== openProject || revision !== openRevision.current) return;
      setSelected({ ...d, _source: 'document', id: d.doc_id ?? item.id }); setTitle(String(d.title ?? '')); setContent(String(d.content ?? '')); setFormat(String(d.format ?? 'markdown'));
      const v = await api.get(`/api/document/${encodeURIComponent(String(item.id))}/versions`); if (project() === openProject && revision === openRevision.current) setVersions(records(v, 'versions'));
    } catch (e) { if (revision === openRevision.current) setError(workspaceError(e, 'This document')); }
  }
  async function save() {
    if (!title.trim()) return; setBusy(true); setError('');
    const saveProject = project();
    try {
      if (selectedProject !== saveProject) return;
      if (!selected) { const r = record(await api.post<unknown>('/api/document', createWorkspaceDocumentPayload(title.trim(), content, format, saveProject))); if (project() !== saveProject) return; setNotice('Document created.'); await load(); const d = record(r.document); if (d.id || d.doc_id) await open({ ...d, _source: 'document', id: d.doc_id ?? d.id }); }
      else if (selected._source === 'document') { const r = record(await api.put<unknown>(`/api/document/${encodeURIComponent(String(selected.id))}`, updateWorkspaceDocumentPayload(title.trim(), content, Number(selected.version)))); setSelected({ ...record(r.document), _source: 'document', id: selected.id }); setNotice('Changes saved.'); setDirty(false); await load(); }
      else { await api.post('/api/workspace/items/update', updateWorkspaceItemPayload(selected.id, title.trim(), content)); setNotice('Workspace item updated.'); setDirty(false); await load(); }
    } catch (e) { setError(workspaceError(e, 'Saving this document')); } finally { setBusy(false); }
  }
  async function importFile(file?: File) {
    if (!file) return;
    setBusy(true); setError(''); setNotice('');
    const importProject = project();
    try {
      const result = record(await api.post<unknown>(workspaceImportRoute(file.name), await workspaceImportPayload(file, importProject)));
      const imported = record(result.document);
      const id = imported.doc_id ?? imported.id;
      if (!id) throw new Error('The document service did not return an imported document.');
      setNotice(`“${file.name.split(/[\\/]/).pop()}” imported to this project.`);
      await load();
      if (project() === importProject) await open({ ...imported, _source: 'document', id });
    } catch (cause) { setError(workspaceError(cause, 'File import')); }
    finally { setBusy(false); }
  }
  async function restore(version: number) {
    if (!selected || !confirm(`Restore document version ${version}?`)) return; setBusy(true); setError('');
    try { await api.post(`/api/document/${encodeURIComponent(String(selected.id))}/restore/${version}`, {}); setNotice(`Version ${version} restored.`); await load(); const refreshed = record(await api.get(`/api/document/${encodeURIComponent(String(selected.id))}`)); const d = record(refreshed.document); setSelected({ ...d, _source: 'document', id: d.doc_id ?? selected.id }); setTitle(String(d.title ?? '')); setContent(String(d.content ?? '')); setFormat(String(d.format ?? 'markdown')); setDirty(false); const v = await api.get(`/api/document/${encodeURIComponent(String(selected.id))}/versions`); setVersions(records(v, 'versions')); }
    catch (e) { setError(workspaceError(e, 'Restoring this document version')); } finally { setBusy(false); }
  }
  async function archive() {
    if (!selected || !confirm(`Archive “${title}”?`)) return; setBusy(true); setError('');
    try { await api.post(`/api/document/${encodeURIComponent(String(selected.id))}/archive`, { archived: true }); setNotice('Document archived.'); setSelected(null); setDraftOpen(false); setTitle(''); setContent(''); await load(); }
    catch (e) { setError(workspaceError(e, 'Archiving this document')); } finally { setBusy(false); }
  }
  function exportDocument() {
    if (!selected) return; const extension = format === 'text' ? 'txt' : 'md', mime = format === 'text' ? 'text/plain' : 'text/markdown'; const blob = new Blob([content], { type: `${mime};charset=utf-8` }); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = `${title.trim() || 'document'}.${extension}`; link.click(); URL.revokeObjectURL(url); setNotice('Downloaded a local copy.');
  }
  function startNew() { if (dirty && !confirm('Discard the unsaved changes and start a new document?')) return; ++openRevision.current; setSelected(null); setDraftOpen(true); setTitle(''); setContent(''); setFormat('markdown'); setDirty(false); setVersions([]); setNotice(''); }

  return <PageFrame title="Files" >
    {error && <p className="notice error" role="alert">{error}</p>}{notice && <p className="notice success" role="status">{notice}</p>}
    <section className={`oa-files ${showEditor ? 'has-open-document' : 'file-first-empty'}`} aria-label="Project files workspace" style={gridStyle}>
      <nav className="oa-files-list" aria-label="Project files"><div className="oa-files-list-head"><div className="oa-files-heading-copy"><strong>Files &amp; knowledge</strong><small>Documents and project materials</small></div><div className="oa-files-list-actions"><button type="button" className="btn oa-refresh-files" onClick={() => void load()} disabled={busy}>Refresh</button><button type="button" className="btn oa-import-document" title="Import a PDF, Office, or text file (up to 700 KB)" aria-describedby="oa-import-limits" disabled={busy} onClick={() => importFileRef.current?.click()}>Import</button><input ref={importFileRef} type="file" hidden accept=".pdf,.docx,.pptx,.xlsx,.txt,.md,.csv,.json,.html,.htm,.rtf" aria-label="Choose a document to import" disabled={busy} onChange={event => { void importFile(event.currentTarget.files?.[0]); event.currentTarget.value = ''; }} /><button className="btn new-document-action" onClick={startNew} aria-label="New document">＋</button></div></div><p className="sr-only" id="oa-import-limits">PDF, Office, and text file imports are limited to 700 KB because the endpoint uses the configured JSON request limit.</p><label className="oa-files-search"><span className="sr-only">Search files</span><input className="field" aria-label="Search files" value={query} onChange={e => setQuery(e.target.value)} placeholder="Find a file" /></label><label className="oa-files-sort"><span>Sort by</span><select value={sortBy} onChange={event => setSortBy(event.target.value as typeof sortBy)} aria-label="Sort files"><option value="updated">Recently updated</option><option value="name">Name</option><option value="type">Type</option></select></label><div className="oa-files-rows" aria-busy={fileState === 'loading'}>{orderedItems.map(x => <button type="button" key={`${x._source}:${x.id}`} aria-current={selected?.id === x.id} onClick={() => void open(x)}><span className="oa-file-icon">{val(x, 'format', x._source === 'document' ? 'MD' : '↗').slice(0, 3).toUpperCase()}</span><span className="oa-file-name"><strong>{val(x, 'title', 'Untitled')}</strong><small>{val(x, 'format', val(x, 'kind', x._source === 'document' ? 'Markdown document' : 'Workspace item'))}{val(x, 'updated_at', val(x, 'created_at', '')) !== '—' ? ` · ${val(x, 'updated_at', val(x, 'created_at'))}` : ''}</small></span></button>)}{fileState === 'loading' && <p className="oa-files-empty" role="status">Loading files…</p>}{fileState === 'error' && !items.length && <button className="oa-files-empty oa-files-retry" onClick={() => void load()}>Files are unavailable. Retry</button>}{fileState === 'ready' && !items.length && <p className="oa-files-empty">{query.trim() ? 'No files match this search.' : 'No files in this project yet.'}</p>}</div></nav>
      <div className="oa-file-splitter" role="separator" aria-label="Resize project files pane" aria-orientation="vertical" aria-valuemin={paneLimits.files.min} aria-valuemax={paneLimits.files.max} aria-valuenow={filesWidth} tabIndex={0} onKeyDown={e => onSeparatorKey('files', e)} onPointerDown={e => onSeparatorPointerDown('files', e)} />
      {showEditor ? <section className="oa-file-editor" aria-label="Document editor"><header className="oa-file-toolbar"><div className="oa-file-title"><span className="oa-file-dot" aria-hidden="true" />{visibleSelected ? <input aria-label="Document title" value={title} onChange={e => { setTitle(e.target.value); setDirty(true); }} placeholder="Untitled" /> : <input aria-label="Document title" value={title} onChange={e => { setTitle(e.target.value); if (title || content) setDirty(true); }} placeholder="Untitled document" />}{dirty && <span className="oa-file-unsaved">Unsaved</span>}</div><div className="oa-file-actions"><select aria-label="Document format" value={format} onChange={e => { setFormat(e.target.value); setDirty(true); }}><option value="markdown">Markdown</option><option value="text">Plain text</option></select><button className="btn primary" onClick={() => void save()} disabled={busy || !title.trim() || (!!visibleSelected && !dirty)}>{busy ? 'Saving…' : visibleSelected ? 'Save' : 'Create'}</button><button className="btn" onClick={exportDocument} disabled={!visibleSelected}>Download</button><details className="oa-file-more"><summary aria-label="More document actions">•••</summary><div><button onClick={startNew}>New document</button>{visibleSelected?._source === 'document' && <button className="danger" onClick={() => void archive()} disabled={busy}>Archive</button>}</div></details></div></header>
        <div className="oa-file-mobile-tabs" role="tablist" aria-label="Document view" ref={hostTabRef}>{(['preview','edit','details'] as const).map(value => <button type="button" role="tab" data-pane-tab={value} tabIndex={pane === value ? 0 : -1} aria-selected={pane === value} onKeyDown={onMobileTabKeyDown} onClick={() => setPane(value)} key={value}>{value[0].toUpperCase() + value.slice(1)}</button>)}</div>
        <div className={`oa-file-content pane-${pane}`}><div className="oa-file-edit-pane"><label className="sr-only" htmlFor="oa-file-content">Document content</label><textarea id="oa-file-content" value={content} onChange={e => { setContent(e.target.value); setDirty(true); }} placeholder="Start writing…" /></div><div className="oa-file-preview-pane"><div className="oa-preview-head"><span>{format === 'text' ? 'Plain text' : 'Markdown'}</span></div><Preview content={content} format={format} /></div><section className="oa-file-details-panel" aria-label="Document details"><dl><dt>Project</dt><dd>{val(visibleSelected ?? {}, 'project', activeProject)}</dd><dt>Type</dt><dd>{visibleSelected?._source === 'document' ? val(visibleSelected, 'format', 'Markdown') : val(visibleSelected ?? {}, 'kind', 'New document')}</dd><dt>Version</dt><dd>{val(visibleSelected ?? {}, 'version', 'Not saved')}</dd><dt>Updated</dt><dd>{val(visibleSelected ?? {}, 'updated_at', val(visibleSelected ?? {}, 'created_at', 'Not saved'))}</dd></dl>{versions.length > 0 && <div className="oa-file-versions"><span>Version history</span>{versions.map(v => <div key={String(v.version)}><span>Version {String(v.version)}</span><button className="btn" onClick={() => void restore(Number(v.version))} disabled={busy}>Restore</button></div>)}</div>}</section></div>
      </section> : <div className="oa-workspace-create"><div className="oa-workspace-empty"><span className="oa-workspace-empty-icon"><FilePlus2 size={19} aria-hidden="true" /></span><h2>No document selected</h2><p>Choose a project file to open it, or create a document for this workspace.</p><button className="btn primary" onClick={startNew}><span aria-hidden="true">＋</span> Create document</button></div></div>}
    </section>
  </PageFrame>;
}
