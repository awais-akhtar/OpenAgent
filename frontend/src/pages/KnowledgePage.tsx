import { useEffect, useRef, useState } from 'react';
import { api, ApiError, record, records } from '../lib/react/api';
import { PageFrame } from '../components/PageFrame';
import { Tabs, TabsList, TabsTrigger } from '../components/ui/tabs';
import './research-knowledge-voice.css';

type Row = Record<string, unknown>;
type Tab = 'Search' | 'Library' | 'Graph' | 'Timeline';
const tabs: Tab[] = ['Search', 'Library', 'Graph', 'Timeline'];
const project = () => new URL(location.href).searchParams.get('project') || localStorage.getItem('oa-project') || 'default';
const value = (row: Row, key: string, fallback = '—') => row[key] == null || row[key] === '' ? fallback : String(row[key]);
const safeExternalUrl = (candidate: unknown) => {
  if (typeof candidate !== 'string') return '';
  try { const url = new URL(candidate); return url.protocol === 'https:' || url.protocol === 'http:' ? url.href : ''; }
  catch { return ''; }
};
function knowledgeError(cause: unknown, label: string) {
  if (cause instanceof ApiError) {
    if (cause.status === 401) return `Sign in to access ${label}.`;
    if (cause.status === 403) return `You do not have permission to access ${label}.`;
    if (cause.status === 404) return `${label} is not available for this project.`;
    if (cause.status === 408 || cause.status >= 500) return `${label} is unavailable. Check the local service and retry.`;
  }
  return cause instanceof Error ? cause.message : `${label} could not be loaded.`;
}

export function KnowledgePage() {
  const [status, setStatus] = useState<Row>({});
  const [ragStatus, setRagStatus] = useState<Row>({});
  const [docs, setDocs] = useState<Row[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<Row | null>(null);
  const [tags, setTags] = useState<Row[]>([]);
  const [graph, setGraph] = useState<Row>({});
  const [timeline, setTimeline] = useState<Row[]>([]);
  const [tab, setTab] = useState<Tab>('Search');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Row[]>([]);
  const [selectedResult, setSelectedResult] = useState<Row | null>(null);
  const [answer, setAnswer] = useState<Row>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [indexState, setIndexState] = useState<'loading' | 'ready' | 'error'>('loading');
  const loadController = useRef<AbortController | null>(null);

  async function load() {
    loadController.current?.abort();
    const controller = new AbortController();
    loadController.current = controller;
    setBusy(true); setError(''); setIndexState('loading');
    const p = encodeURIComponent(project());
    const settled = await Promise.allSettled([
      api.get<unknown>(`/api/knowledge/status?project=${p}`, controller.signal),
      api.get<unknown>(`/api/knowledge/documents?project=${p}&limit=100`, controller.signal),
      api.get<unknown>(`/api/knowledge/tags?project=${p}&limit=100`, controller.signal),
      api.get<unknown>(`/api/knowledge/graph?project=${p}`, controller.signal),
      api.get<unknown>(`/api/knowledge/timeline?project=${p}&limit=100`, controller.signal),
      api.get<unknown>(`/api/rag/status?project=${p}`, controller.signal),
    ]);
    if (controller.signal.aborted) return;
    const firstFailure = settled.find((item, index) => item.status === 'rejected' && index < 2);
    if (settled[0].status === 'fulfilled') setStatus(record(settled[0].value));
    if (settled[1].status === 'fulfilled') {
      const nextDocs = records(settled[1].value, 'documents'); setDocs(nextDocs);
      setSelectedDoc(current => current ? nextDocs.find(doc => doc.id === current.id) ?? nextDocs[0] ?? null : nextDocs[0] ?? null);
      if (!query.trim()) setSelectedResult(current => current ?? nextDocs[0] ?? null);
    }
    if (settled[2].status === 'fulfilled') setTags(records(settled[2].value, 'tags'));
    if (settled[3].status === 'fulfilled') setGraph(record(settled[3].value));
    if (settled[4].status === 'fulfilled') setTimeline(records(settled[4].value, 'timeline'));
    if (settled[5].status === 'fulfilled') setRagStatus(record(settled[5].value));
    if (firstFailure?.status === 'rejected') { setError(knowledgeError(firstFailure.reason, 'Knowledge index')); setIndexState('error'); }
    else setIndexState('ready');
    setBusy(false);
  }
  useEffect(() => { void load(); return () => loadController.current?.abort(); }, []);

  async function search() {
    if (!query.trim()) return;
    setBusy(true); setError(''); setNotice(''); setAnswer({}); setResults([]); setSelectedResult(null); setSelectedDoc(null);
    try {
      const result = record(await api.post<unknown>('/api/knowledge/hybrid-search', { query: query.trim(), project: project(), limit: 10 }));
      const found = records(result, 'results'); setResults(found); setSelectedResult(found[0] ?? null);
      setNotice(found.length ? `${found.length} source-backed result${found.length === 1 ? '' : 's'} returned.` : 'No indexed sources matched this query.');
    } catch (cause) { setError(knowledgeError(cause, 'Knowledge search')); }
    finally { setBusy(false); }
  }
  async function synthesize() {
    if (!query.trim()) return;
    setBusy(true); setError(''); setNotice(''); setResults([]); setSelectedResult(null); setSelectedDoc(null); setAnswer({});
    try {
      const result = record(await api.post<unknown>('/api/knowledge/synthesize', { query: query.trim(), project: project(), limit: 8 }));
      const sources = records(result, 'sources');
      setAnswer(result); setResults(sources); setSelectedResult(sources[0] ?? null);
      setNotice('Local retrieval synthesis returned. It does not call a language model.');
    } catch (cause) { setError(knowledgeError(cause, 'Local synthesis')); }
    finally { setBusy(false); }
  }
  async function upload(file?: File) {
    if (!file) return;
    setBusy(true); setError(''); setNotice('Uploading and extracting source…');
    try {
      const body = new FormData(); body.append('file', file); body.append('project', project());
      const result = record(await api.post<unknown>('/api/knowledge/upload', body));
      const doc = record(result.document); const chars = Number(result.chars_extracted ?? 0);
      setNotice(chars > 0 ? `${value(result, 'filename', file.name)} extracted and indexed (${chars.toLocaleString()} characters).` : `${value(result, 'filename', file.name)} stored as a media asset. No extractable text was returned, so it was not added to text search.`);
      if (doc.id) { setSelectedDoc(doc); setTab('Library'); }
      await load();
    } catch (cause) { setError(knowledgeError(cause, 'Source upload')); }
    finally { setBusy(false); }
  }
  async function saveSynthesis() {
    const summary = value(answer, 'summary', '');
    if (!query.trim() || !summary) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const out = record(await api.post<unknown>('/api/knowledge/ingest', { title: `Local synthesis: ${query.trim()}`, content: summary, project: project(), source: 'manual', metadata: { query: query.trim(), synthesis_project: project(), source_ids: records(answer, 'sources').map(source => source.document_id).filter(Boolean) } }));
      setNotice(`Synthesis saved as document ${value(record(out.document), 'id')}.`); await load();
    } catch (cause) { setError(knowledgeError(cause, 'Saving the synthesis')); }
    finally { setBusy(false); }
  }

  return <PageFrame title="Knowledge" description="Search your saved sources and review citations." actions={<button className="btn" disabled={busy} onClick={() => void load()}>Refresh</button>}>
    {error && <p className="notice error" role="alert">{error}</p>}{notice && <p className="notice success" role="status">{notice}</p>}
    {(Number(status.documents) > 0 || Number(status.chunks) > 0) && <div className="knowledge-index-line" aria-label="Knowledge index summary">{Number(status.documents) > 0 && <>{Number(status.documents).toLocaleString()} sources</>}{Number(status.documents) > 0 && Number(status.chunks) > 0 && <span> · </span>}{Number(status.chunks) > 0 && <>{Number(status.chunks).toLocaleString()} indexed passages</>}</div>}
    {indexState === 'loading' && <p className="knowledge-load-state" role="status">Loading project sources…</p>}
    <Tabs value={tab} onValueChange={value => setTab(value as Tab)}>
      <TabsList aria-label="Knowledge sections" className="knowledge-tabs">{tabs.map(item => <TabsTrigger key={item} id={`knowledge-tab-${item}`} aria-controls={`knowledge-panel-${item}`} value={item}>{item}</TabsTrigger>)}</TabsList>
    </Tabs>
    <div id={`knowledge-panel-${tab}`} role="tabpanel" aria-labelledby={`knowledge-tab-${tab}`} tabIndex={0} className="knowledge-tab-panel">
    {tab === 'Search' && <div className={`knowledge-search-workspace ${selectedResult ? 'has-selection' : ''}`}><section className="panel knowledge-query"><div className="panel-body stack"><label>Search project knowledge<textarea className="field" rows={3} value={query} onChange={event => setQuery(event.target.value)} onKeyDown={event => { if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') void search(); }} placeholder="Ask a question or find a source" /></label><div className="toolbar"><button className="btn primary" disabled={busy || !query.trim()} onClick={() => void search()}>{busy ? 'Searching…' : 'Search'}</button><button className="btn" disabled={busy || !query.trim()} onClick={() => void synthesize()}>Local synthesis</button></div><p className="knowledge-query-hint">Search uses this project’s indexed sources. Local synthesis uses retrieval only; it does not call a language model.</p></div></section><section className="knowledge-results" aria-live="polite">{answer.summary != null && <article className="panel panel-body knowledge-summary"><div className="knowledge-result-heading"><h2>Summary</h2><span>Local retrieval</span></div><p>{String(answer.summary)}</p><div><button className="btn" disabled={busy} onClick={() => void saveSynthesis()}>Save to library</button></div><details><summary>Source context</summary><ul>{records(answer, 'sources').map((source, index) => <li key={value(source, 'document_id', String(index))}>{value(source, 'title', value(source, 'document_id', 'Source'))}</li>)}</ul><p>{records(answer, 'related_entities').map(item => value(item, 'name', String(item))).join(' · ')}</p></details></article>}{results.map((item, index) => <button type="button" className={`panel panel-body knowledge-result ${selectedResult === item ? 'is-selected' : ''}`} aria-pressed={selectedResult === item} key={value(item, 'id', value(item, 'chunk_id', String(index)))} onClick={() => setSelectedResult(item)}><strong>{value(item, 'title', `Source ${index + 1}`)}</strong><p>{value(item, 'content', value(item, 'text', value(item, 'snippet', '')))}</p><small className="tiny">{value(item, 'document_id', 'Project source')}{item.score != null ? ` · ${value(item, 'score')}` : ''}</small></button>)}{!results.length && answer.summary == null && !query.trim() && docs.length > 0 && <div className="knowledge-indexed-list"><header><strong>Indexed sources</strong><span>{docs.length}</span></header>{docs.slice(0, 12).map(doc => <button type="button" key={value(doc, 'id')} aria-pressed={selectedDoc?.id === doc.id} onClick={() => { setSelectedDoc(doc); setSelectedResult(doc); }}><span><strong>{value(doc, 'title', 'Untitled source')}</strong><small>{value(doc, 'source', 'Project source')}{doc.chunk_count != null ? ` · ${value(doc, 'chunk_count')} passages` : ''}</small></span><span aria-hidden="true">›</span></button>)}</div>}{!results.length && answer.summary == null && (query.trim() || !docs.length) && <div className="knowledge-search-empty"><strong>{indexState === 'error' ? 'Source index unavailable' : query.trim() ? 'No results yet' : 'No indexed sources in this project'}</strong><p>{indexState === 'error' ? 'Refresh to reconnect to the project index.' : query.trim() ? 'Run a search to find matching passages.' : 'Add a source to the Library to make it available here.'}</p><button type="button" className="btn" onClick={() => setTab('Library')}>Open library</button></div>}</section>{selectedResult && <aside className="panel knowledge-inspector" aria-label="Selected source details"><div className="knowledge-inspector-head"><span>Source</span><button className="btn" onClick={() => setSelectedResult(null)} aria-label="Close source details">×</button></div><div className="panel-body stack"><strong>{value(selectedResult, 'title', 'Source')}</strong>{(selectedResult.type ?? selectedResult.source) != null && <small>{value(selectedResult, 'type', value(selectedResult, 'source'))}</small>}<p>{value(selectedResult, 'content', value(selectedResult, 'text', value(selectedResult, 'snippet', 'No excerpt returned for this source.')))}</p>{selectedResult.score != null && <small>Relevance · {value(selectedResult, 'score')}</small>}{(selectedResult.chunk_count != null || selectedResult.created_at != null) && <dl className="knowledge-source-facts"><dt>Indexed passages</dt><dd>{value(selectedResult, 'chunk_count')}</dd><dt>Added</dt><dd>{value(selectedResult, 'created_at')}</dd></dl>}{safeExternalUrl(selectedResult.url) && <a href={safeExternalUrl(selectedResult.url)} target="_blank" rel="noopener noreferrer">Open source ↗</a>}</div></aside>}</div>}
    {tab === 'Library' && <div className="knowledge-library-layout"><section className="panel knowledge-library"><div className="panel-head"><h2>Sources</h2><label className="btn">Add source<input type="file" hidden accept=".txt,.md,.pdf,.docx,.csv,.json,.html,.htm,.rtf" disabled={busy} onChange={event => { void upload(event.target.files?.[0]); event.currentTarget.value = ''; }} /></label></div><div className="panel-body list">{docs.map(doc => <button type="button" className={`record-row knowledge-document ${selectedDoc && selectedDoc.id === doc.id ? 'is-selected' : ''}`} aria-pressed={Boolean(selectedDoc && selectedDoc.id === doc.id)} key={value(doc, 'id')} onClick={() => setSelectedDoc(doc)}><span><strong>{value(doc, 'title', 'Untitled source')}</strong><small>{value(doc, 'source', 'Local source')} · {value(doc, 'chunk_count', '—')} passages · {value(doc, 'created_at')}</small></span><span className="knowledge-document-tags">{records(doc.tags).map((tag, i) => <span className="tag" key={String(tag.id ?? i)}>{value(tag, 'tag', String(tag))}</span>)}</span></button>)}{indexState === 'loading' && <p className="empty" role="status">Loading indexed sources…</p>}{indexState === 'ready' && !docs.length && <div className="knowledge-search-empty"><strong>No sources in this project</strong><p>Add a document to make it available to search.</p></div>}</div></section><aside className="panel knowledge-inspector knowledge-document-inspector" aria-label="Source details">{selectedDoc ? <><div className="knowledge-inspector-head"><span>Source details</span><button className="btn" onClick={() => setSelectedDoc(null)} aria-label="Close source details">×</button></div><div className="panel-body stack"><strong>{value(selectedDoc, 'title', 'Untitled source')}</strong><dl className="knowledge-source-facts"><dt>Origin</dt><dd>{value(selectedDoc, 'source', 'Local source')}</dd><dt>Indexed passages</dt><dd>{value(selectedDoc, 'chunk_count')}</dd><dt>Added</dt><dd>{value(selectedDoc, 'created_at')}</dd></dl>{value(selectedDoc, 'summary', '') && <p>{value(selectedDoc, 'summary', '')}</p>}</div></> : <div className="knowledge-inspector-empty"><strong>Source details</strong><p>Select an indexed source to inspect its metadata.</p></div>}</aside><section className="panel knowledge-tags"><div className="panel-head"><h2>Tags</h2></div><div className="panel-body list">{tags.map((tag, i) => <article className="record-row" key={value(tag, 'id', String(i))}><strong>{value(tag, 'tag')}</strong>{(typeof tag.confidence === 'string' || typeof tag.confidence === 'number') && <span>{value(tag, 'confidence')}</span>}</article>)}{!tags.length && <p className="empty">No tags yet.</p>}</div></section></div>}
    {tab === 'Graph' && <div className="workspace"><section className="panel"><div className="panel-head"><h2>Knowledge entities</h2><span>{records(graph, 'entities').length} nodes</span></div><div className="panel-body list">{records(graph, 'entities').map((entity, i) => <article className="record-row" key={value(entity, 'id', String(i))}><div><strong>{value(entity, 'name')}</strong><p>{value(entity, 'entity_type', 'Entity')}</p></div><span>{value(entity, 'degree', 'Degree not reported')} links</span></article>)}{!records(graph, 'entities').length && <p className="empty">No graph entities are indexed for this project.</p>}</div></section><section className="panel"><div className="panel-head"><h2>Relationships</h2><span>{records(graph, 'relationships').length} edges</span></div><div className="panel-body list">{records(graph, 'relationships').map((edge, i) => <article className="record-row" key={value(edge, 'id', String(i))}><span>{value(edge, 'subject', value(edge, 'source'))}</span><strong>{value(edge, 'relationship', value(edge, 'predicate', 'linked to'))}</strong><span>{value(edge, 'object', value(edge, 'target'))}</span></article>)}{!records(graph, 'relationships').length && <p className="empty">No relationships are indexed for this project.</p>}</div></section></div>}
    {tab === 'Timeline' && <section className="panel"><div className="panel-head"><h2>Project knowledge timeline</h2></div><div className="panel-body list">{timeline.map((item, i) => <article className="record-row" key={value(item, 'source_id', String(i))}><div><strong>{value(item, 'title', value(item, 'name', value(item, 'type', 'Knowledge event')))}</strong><p>{value(item, 'type', 'Index record')} · source {value(item, 'source_id')}</p></div><time>{value(item, 'timestamp')}</time></article>)}{!timeline.length && <p className="empty">No timeline events returned for this project.</p>}</div></section>}
    {tab === 'Library' && <details className="knowledge-diagnostics"><summary>Index details</summary><div>Full-text search · {status.fts_available === true ? 'Available' : status.fts_available === false ? 'Unavailable' : 'Not reported'}</div><div>Vector search · {value(record(status.vector_store), 'available', 'Not reported')}</div><div>Retrieval service · {value(ragStatus, 'status', value(ragStatus, 'available', 'Not reported'))}</div></details>}
    </div>
  </PageFrame>;
}
