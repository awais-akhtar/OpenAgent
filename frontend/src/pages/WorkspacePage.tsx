import { useEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from 'react';
import { api, record, records } from '../lib/react/api';
import { PageFrame } from '../components/PageFrame';
import './workspace-page.css';
import { createWorkspaceDocumentPayload, updateWorkspaceDocumentPayload, updateWorkspaceItemPayload, workspaceLibraryPaths, workspacePreviewBlocks } from './workspace-utils';

type Row = Record<string, unknown>;
type Pane = 'edit' | 'preview';
type ResizablePane = 'files' | 'details';
const paneLimits = { files: { min: 200, max: 360 }, details: { min: 210, max: 340 } } as const;
const project = () => new URL(location.href).searchParams.get('project') || localStorage.getItem('oa-project') || 'default';
const val = (r: Row, k: string, f = '—') => r[k] == null || r[k] === '' ? f : String(r[k]);

function Preview({ content, format }: { content: string; format: string }) {
  if (!content.trim()) return <p className="oa-document-empty">Preview will appear here as you write.</p>;
  if (format === 'text') return <article className="oa-document-preview oa-document-plain">{content}</article>;
  const blocks = workspacePreviewBlocks(content);
  return <article className="oa-document-preview">{blocks.map((block, i) => block.kind === 'heading' ? <h2 key={i} className={`oa-preview-h${block.level}`}>{block.text}</h2> : block.kind === 'code' ? <pre key={i}><code>{block.text}</code></pre> : block.kind === 'list' ? <p key={i} className="oa-preview-list">• {block.text}</p> : <p key={i}>{block.text}</p>)}</article>;
}
export function WorkspacePage() {
  const [items, setItems] = useState<Row[]>([]), [selected, setSelected] = useState<Row | null>(null), [query, setQuery] = useState(''), [title, setTitle] = useState(''), [content, setContent] = useState(''), [format, setFormat] = useState('markdown'), [versions, setVersions] = useState<Row[]>([]), [dirty, setDirty] = useState(false), [busy, setBusy] = useState(false), [error, setError] = useState(''), [notice, setNotice] = useState(''), [pane, setPane] = useState<Pane>('edit');
  const [filesWidth, setFilesWidth] = useState(218), [detailsWidth, setDetailsWidth] = useState(210);
  const drag = useRef<{ pane: ResizablePane; pointerId: number; startX: number; startWidth: number } | null>(null);
  function updatePaneWidth(which: ResizablePane, width: number) {
    const limits = paneLimits[which], next = Math.max(limits.min, Math.min(limits.max, width));
    (which === 'files' ? setFilesWidth : setDetailsWidth)(next);
  }
  function onSeparatorKey(which: ResizablePane, event: KeyboardEvent<HTMLDivElement>) {
    const current = which === 'files' ? filesWidth : detailsWidth, step = event.shiftKey ? 40 : 16;
    let next = current;
    if (event.key === 'ArrowLeft') next = current + (which === 'details' ? step : -step);
    else if (event.key === 'ArrowRight') next = current + (which === 'details' ? -step : step);
    else if (event.key === 'Home') next = paneLimits[which].min;
    else if (event.key === 'End') next = paneLimits[which].max;
    else return;
    event.preventDefault(); updatePaneWidth(which, next);
  }
  function onSeparatorPointerDown(which: ResizablePane, event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    event.preventDefault();
    drag.current = { pane: which, pointerId: event.pointerId, startX: event.clientX, startWidth: which === 'files' ? filesWidth : detailsWidth };
  }
  function onMobileTabKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    let next: Pane | null = null;
    if (event.key === 'ArrowLeft') next = 'edit';
    else if (event.key === 'ArrowRight') next = 'preview';
    else if (event.key === 'Home') next = 'edit';
    else if (event.key === 'End') next = 'preview';
    if (!next) return;
    event.preventDefault(); setPane(next);
    hostTabRef.current?.querySelector<HTMLButtonElement>(`[data-pane-tab="${next}"]`)?.focus();
  }
  useEffect(() => {
    const move = (event: globalThis.PointerEvent) => {
      const current = drag.current;
      if (!current || current.pointerId !== event.pointerId) return;
      const delta = event.clientX - current.startX;
      updatePaneWidth(current.pane, current.startWidth + (current.pane === 'details' ? -delta : delta));
    };
    const stop = (event: globalThis.PointerEvent) => { if (drag.current?.pointerId === event.pointerId) drag.current = null; };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', stop); window.addEventListener('pointercancel', stop);
    return () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', stop); window.removeEventListener('pointercancel', stop); };
  }, [filesWidth, detailsWidth]);
  const gridStyle = { '--oa-files-list-width': `${filesWidth}px`, '--oa-files-details-width': `${detailsWidth}px` } as CSSProperties;
  const hostTabRef = useRef<HTMLDivElement>(null);

  async function load(signal?: AbortSignal) {
    setBusy(true); setError('');
    try {
      const paths = workspaceLibraryPaths(project(), query);
      const [a, b] = await Promise.all([api.get<unknown>(paths.workspace, signal), api.get<unknown>(paths.documents, signal)]);
      if (signal?.aborted) return;
      setItems([...records(b, 'items').map(x => ({ ...x, _source: 'document', id: x.doc_id ?? x.id })), ...records(a, 'items').map(x => ({ ...x, _source: 'workspace' }))]);
    } catch (e) { if (!signal?.aborted) setError(e instanceof Error ? e.message : 'Document library unavailable.'); }
    finally { if (!signal?.aborted) setBusy(false); }
  }
  useEffect(() => { const controller = new AbortController(); const timer = window.setTimeout(() => void load(controller.signal), 180); return () => { window.clearTimeout(timer); controller.abort(); }; }, [query, project()]);

  async function open(item: Row) {
    setSelected(item); setTitle(String(item.title ?? '')); setContent(String(item.content ?? '')); setFormat(String(item.format ?? 'markdown')); setDirty(false); setVersions([]); setPane('edit');
    if (item._source === 'document') try {
      const r = record(await api.get(`/api/document/${encodeURIComponent(String(item.id))}`)); const d = record(r.document);
      setSelected({ ...d, _source: 'document', id: d.doc_id ?? item.id }); setTitle(String(d.title ?? '')); setContent(String(d.content ?? '')); setFormat(String(d.format ?? 'markdown'));
      const v = await api.get(`/api/document/${encodeURIComponent(String(item.id))}/versions`); setVersions(records(v, 'versions'));
    } catch (e) { setError(e instanceof Error ? e.message : 'Document could not be opened.'); }
  }
  async function save() {
    if (!title.trim()) return; setBusy(true); setError('');
    try {
      if (!selected) { const r = record(await api.post<unknown>('/api/document', createWorkspaceDocumentPayload(title.trim(), content, format, project()))); setNotice('Document created.'); await load(); const d = record(r.document); if (d.id || d.doc_id) await open({ ...d, _source: 'document', id: d.doc_id ?? d.id }); }
      else if (selected._source === 'document') { const r = record(await api.put<unknown>(`/api/document/${encodeURIComponent(String(selected.id))}`, updateWorkspaceDocumentPayload(title.trim(), content, Number(selected.version)))); setSelected({ ...record(r.document), _source: 'document', id: selected.id }); setNotice('Changes saved.'); setDirty(false); await load(); }
      else { await api.post('/api/workspace/items/update', updateWorkspaceItemPayload(selected.id, title.trim(), content)); setNotice('Workspace item updated.'); setDirty(false); await load(); }
    } catch (e) { setError(e instanceof Error ? e.message : 'Save failed. Your draft remains here.'); } finally { setBusy(false); }
  }
  async function restore(version: number) {
    if (!selected || !confirm(`Restore document version ${version}?`)) return; setBusy(true); setError('');
    try { await api.post(`/api/document/${encodeURIComponent(String(selected.id))}/restore/${version}`, {}); setNotice(`Version ${version} restored.`); await load(); const refreshed = record(await api.get(`/api/document/${encodeURIComponent(String(selected.id))}`)); const d = record(refreshed.document); setSelected({ ...d, _source: 'document', id: d.doc_id ?? selected.id }); setTitle(String(d.title ?? '')); setContent(String(d.content ?? '')); setFormat(String(d.format ?? 'markdown')); setDirty(false); const v = await api.get(`/api/document/${encodeURIComponent(String(selected.id))}/versions`); setVersions(records(v, 'versions')); }
    catch (e) { setError(e instanceof Error ? e.message : 'Version restore failed.'); } finally { setBusy(false); }
  }
  async function archive() {
    if (!selected || !confirm(`Archive “${title}”?`)) return; setBusy(true); setError('');
    try { await api.post(`/api/document/${encodeURIComponent(String(selected.id))}/archive`, { archived: true }); setNotice('Document archived.'); setSelected(null); setTitle(''); setContent(''); await load(); }
    catch (e) { setError(e instanceof Error ? e.message : 'Archive failed.'); } finally { setBusy(false); }
  }
  function exportDocument() {
    if (!selected) return; const extension = format === 'text' ? 'txt' : 'md', mime = format === 'text' ? 'text/plain' : 'text/markdown'; const blob = new Blob([content], { type: `${mime};charset=utf-8` }); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = `${title.trim() || 'document'}.${extension}`; link.click(); URL.revokeObjectURL(url); setNotice('Downloaded a local copy.');
  }
  function startNew() { if (dirty && !confirm('Discard the unsaved changes and start a new document?')) return; setSelected(null); setTitle(''); setContent(''); setFormat('markdown'); setDirty(false); setVersions([]); setNotice(''); }

  return <PageFrame eyebrow="Project / Files" title="Workspace" description="Documents in the selected project." actions={<button className="btn" onClick={() => void load()} disabled={busy}>Refresh</button>}>
    {error && <p className="notice error" role="alert">{error}</p>}{notice && <p className="notice success" role="status">{notice}</p>}
    <section className="oa-files" aria-label="Project files workspace" style={gridStyle}>
      <nav className="oa-files-list" aria-label="Project files"><div className="oa-files-list-head"><span>Files</span><button className="btn" onClick={startNew} aria-label="New document">＋</button></div><label className="oa-files-search"><span className="sr-only">Search files</span><input className="field" aria-label="Search files" value={query} onChange={e => setQuery(e.target.value)} placeholder="Find a file" /></label><div className="oa-files-rows">{items.map(x => <button type="button" key={`${x._source}:${x.id}`} aria-current={selected?.id === x.id} onClick={() => void open(x)}><span className="oa-file-icon">{x._source === 'document' ? 'MD' : '↗'}</span><span className="oa-file-name"><strong>{val(x, 'title', 'Untitled')}</strong><small>{x._source === 'document' ? `v${val(x, 'version', '1')}` : val(x, 'kind', 'Workspace item')}</small></span></button>)}{!items.length && <p className="oa-files-empty">{busy ? 'Loading files…' : error ? 'Files could not be loaded.' : 'No files in this project yet.'}</p>}</div></nav>
      <div className="oa-file-splitter" role="separator" aria-label="Resize project files pane" aria-orientation="vertical" aria-valuemin={paneLimits.files.min} aria-valuemax={paneLimits.files.max} aria-valuenow={filesWidth} tabIndex={0} onKeyDown={e => onSeparatorKey('files', e)} onPointerDown={e => onSeparatorPointerDown('files', e)} />
      <section className="oa-file-editor" aria-label="Document editor"><header className="oa-file-toolbar"><div className="oa-file-title"><span className="oa-file-dot" aria-hidden="true" />{selected ? <input aria-label="Document title" value={title} onChange={e => { setTitle(e.target.value); setDirty(true); }} placeholder="Untitled" /> : <input aria-label="Document title" value={title} onChange={e => setTitle(e.target.value)} placeholder="Untitled document" />}{dirty && <span className="oa-file-unsaved">Unsaved</span>}</div><div className="oa-file-actions"><select aria-label="Document format" value={format} onChange={e => { setFormat(e.target.value); setDirty(true); }}><option value="markdown">Markdown</option><option value="text">Plain text</option></select><button className="btn primary" onClick={() => void save()} disabled={busy || !title.trim() || (!!selected && !dirty)}>{busy ? 'Saving…' : selected ? 'Save' : 'Create'}</button><button className="btn" onClick={exportDocument} disabled={!selected}>Download</button><details className="oa-file-more"><summary aria-label="More document actions">•••</summary><div><button onClick={startNew}>New document</button>{selected?._source === 'document' && <button className="danger" onClick={() => void archive()} disabled={busy}>Archive</button>}</div></details></div></header>
        <div className="oa-file-mobile-tabs" role="tablist" aria-label="Editor view" ref={hostTabRef}><button type="button" role="tab" data-pane-tab="edit" tabIndex={pane === 'edit' ? 0 : -1} aria-selected={pane === 'edit'} onKeyDown={onMobileTabKeyDown} onClick={() => setPane('edit')}>Edit</button><button type="button" role="tab" data-pane-tab="preview" tabIndex={pane === 'preview' ? 0 : -1} aria-selected={pane === 'preview'} onKeyDown={onMobileTabKeyDown} onClick={() => setPane('preview')}>Preview</button></div>
        <div className={`oa-file-content ${pane === 'preview' ? 'preview-active' : ''}`}><div className="oa-file-edit-pane"><label className="sr-only" htmlFor="oa-file-content">Document content</label><textarea id="oa-file-content" value={content} onChange={e => { setContent(e.target.value); setDirty(true); }} placeholder="Start writing…" /></div><div className="oa-file-preview-pane"><div className="oa-preview-head"><button className="oa-preview-back" onClick={() => setPane('edit')} aria-label="Back to editor">‹ <span>Editor</span></button><span className="oa-preview-label">Preview</span><span>{format === 'text' ? 'Plain text' : 'Markdown'}</span></div><Preview content={content} format={format} /></div></div>
      </section>
      <div className="oa-file-splitter oa-file-splitter-details" role="separator" aria-label="Resize document details pane" aria-orientation="vertical" aria-valuemin={paneLimits.details.min} aria-valuemax={paneLimits.details.max} aria-valuenow={detailsWidth} tabIndex={0} onKeyDown={e => onSeparatorKey('details', e)} onPointerDown={e => onSeparatorPointerDown('details', e)} />
      <aside className="oa-file-inspector" aria-label="Document details"><div className="oa-inspector-head">Details</div>{selected ? <><dl><dt>Project</dt><dd>{val(selected, 'project', project())}</dd><dt>Type</dt><dd>{selected._source === 'document' ? val(selected, 'format', 'Markdown') : val(selected, 'kind', 'Workspace item')}</dd><dt>Version</dt><dd>{val(selected, 'version', '—')}</dd><dt>Updated</dt><dd>{val(selected, 'updated_at', val(selected, 'created_at'))}</dd></dl>{versions.length > 0 && <div className="oa-file-versions"><span>History</span>{versions.map(v => <div key={String(v.version)}><span>Version {String(v.version)}</span><button className="btn" onClick={() => void restore(Number(v.version))} disabled={busy}>Restore</button></div>)}</div>}</> : <p className="oa-files-empty">Select a file to see its project details and version history.</p>}</aside>
    </section>
  </PageFrame>;
}
