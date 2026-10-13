import { useEffect, useState } from 'react';
import { api, record, records } from '../lib/react/api';
import { PageFrame } from '../components/PageFrame';
import { Tabs, TabsList, TabsTrigger } from '../components/ui/tabs';

type Row = Record<string, unknown>;
type Tab = 'Search' | 'Library' | 'Graph' | 'Timeline' | 'Diagnostics';
const tabs: Tab[] = ['Search', 'Library', 'Graph', 'Timeline', 'Diagnostics'];
const project = () => new URL(location.href).searchParams.get('project') || localStorage.getItem('oa-project') || 'default';
const value = (row: Row, key: string, fallback = '—') => row[key] == null || row[key] === '' ? fallback : String(row[key]);

export function KnowledgePage() {
  const [status, setStatus] = useState<Row>({});
  const [ragStatus, setRagStatus] = useState<Row>({});
  const [docs, setDocs] = useState<Row[]>([]);
  const [tags, setTags] = useState<Row[]>([]);
  const [graph, setGraph] = useState<Row>({});
  const [timeline, setTimeline] = useState<Row[]>([]);
  const [tab, setTab] = useState<Tab>('Search');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Row[]>([]);
  const [answer, setAnswer] = useState<Row>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  async function load() {
    setBusy(true); setError('');
    const p = encodeURIComponent(project());
    const settled = await Promise.allSettled([
      api.get<unknown>(`/api/knowledge/status?project=${p}`),
      api.get<unknown>(`/api/knowledge/documents?project=${p}&limit=100`),
      api.get<unknown>(`/api/knowledge/tags?project=${p}&limit=100`),
      api.get<unknown>(`/api/knowledge/graph?project=${p}`),
      api.get<unknown>(`/api/knowledge/timeline?project=${p}&limit=100`),
      api.get<unknown>(`/api/rag/status?project=${p}`),
    ]);
    const firstFailure = settled.find((item, index) => item.status === 'rejected' && index < 2);
    if (settled[0].status === 'fulfilled') setStatus(record(settled[0].value));
    if (settled[1].status === 'fulfilled') setDocs(records(settled[1].value, 'documents'));
    if (settled[2].status === 'fulfilled') setTags(records(settled[2].value, 'tags'));
    if (settled[3].status === 'fulfilled') setGraph(record(settled[3].value));
    if (settled[4].status === 'fulfilled') setTimeline(records(settled[4].value, 'timeline'));
    if (settled[5].status === 'fulfilled') setRagStatus(record(settled[5].value));
    if (firstFailure?.status === 'rejected') setError(firstFailure.reason instanceof Error ? firstFailure.reason.message : 'Knowledge service unavailable.');
    setBusy(false);
  }
  useEffect(() => { void load(); }, []);

  async function search() {
    if (!query.trim()) return;
    setBusy(true); setError(''); setNotice(''); setAnswer({});
    try {
      const result = record(await api.post<unknown>('/api/knowledge/hybrid-search', { query: query.trim(), project: project(), limit: 10 }));
      const found = records(result, 'results'); setResults(found);
      setNotice(found.length ? `${found.length} source-backed result${found.length === 1 ? '' : 's'} returned.` : 'No indexed sources matched this query.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Search failed.'); }
    finally { setBusy(false); }
  }
  async function synthesize() {
    if (!query.trim()) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const result = record(await api.post<unknown>('/api/knowledge/synthesize', { query: query.trim(), project: project(), limit: 8 }));
      setAnswer(result); setResults(records(result, 'sources'));
      setNotice('Local retrieval synthesis returned. It does not call a language model.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Synthesis failed.'); }
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
      if (doc.id) setResults([doc]);
      await load();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Upload failed.'); }
    finally { setBusy(false); }
  }
  async function saveSynthesis() {
    const summary = value(answer, 'summary', '');
    if (!query.trim() || !summary) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const out = record(await api.post<unknown>('/api/knowledge/ingest', { title: `Local synthesis: ${query.trim()}`, content: summary, project: project(), source: 'manual', metadata: { query: query.trim(), synthesis_project: project(), source_ids: records(answer, 'sources').map(source => source.document_id).filter(Boolean) } }));
      setNotice(`Synthesis saved as document ${value(record(out.document), 'id')}.`); await load();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Synthesis could not be saved.'); }
    finally { setBusy(false); }
  }

  return <PageFrame eyebrow="Local retrieval" title="Knowledge" description="Search indexed project sources, review provenance, and inspect the local retrieval index." actions={<button className="btn" disabled={busy} onClick={() => void load()}>Refresh</button>}>
    {error && <p className="notice error" role="alert">{error}</p>}{notice && <p className="notice success" role="status">{notice}</p>}
    <section className="panel panel-body row" aria-label="Knowledge index summary"><div>{[['documents', status.documents], ['chunks', status.chunks], ['entities', status.entities], ['tags', status.tags]].map(([label, count]) => <span key={String(label)} style={{ marginRight: 22 }}><strong>{count == null ? '—' : String(count)}</strong><small style={{ display: 'block' }}>{String(label)}</small></span>)}</div><small>FTS {status.fts_available === true ? 'available' : status.fts_available === false ? 'unavailable' : 'unknown'} · vector index {value(record(status.vector_store), 'available', 'status unknown')}</small></section>
    <Tabs value={tab} onValueChange={value => setTab(value as Tab)}>
      <TabsList aria-label="Knowledge sections" className="knowledge-tabs">{tabs.map(item => <TabsTrigger key={item} id={`knowledge-tab-${item}`} aria-controls={`knowledge-panel-${item}`} value={item}>{item}</TabsTrigger>)}</TabsList>
    </Tabs>
    <div id={`knowledge-panel-${tab}`} role="tabpanel" aria-labelledby={`knowledge-tab-${tab}`} tabIndex={0} className="knowledge-tab-panel">
    {tab === 'Search' && <div className="workspace"><section className="panel"><div className="panel-head"><h2>Ask across indexed sources</h2><span className="tag">Hybrid retrieval</span></div><div className="panel-body stack"><label>Question or search phrase<textarea className="field" rows={4} value={query} onChange={event => setQuery(event.target.value)} placeholder="Search local project material" /></label><div className="toolbar"><button className="btn primary" disabled={busy || !query.trim()} onClick={() => void search()}>Search sources</button><button className="btn" disabled={busy || !query.trim()} onClick={() => void synthesize()}>Synthesize locally</button></div>{answer.summary != null && <article className="panel panel-body"><h3>Local retrieval summary</h3><p>{String(answer.summary)}</p><small>Source IDs and retrieval method are returned by the knowledge service.</small><div><button className="btn" disabled={busy} onClick={() => void saveSynthesis()}>Save summary as knowledge note</button></div><details><summary>Synthesis evidence</summary><pre>{JSON.stringify({ reasoning_steps: answer.reasoning_steps, related_entities: answer.related_entities, related_relationships: answer.related_relationships }, null, 2)}</pre></details></article>}<h3>Results and provenance</h3>{results.map((item, index) => <article className="panel panel-body" key={value(item, 'id', value(item, 'chunk_id', String(index)))}><strong>{value(item, 'title', `Source ${index + 1}`)}</strong><p>{value(item, 'content', value(item, 'text', value(item, 'snippet', 'No source excerpt returned.')))}</p><small className="tiny">Document {value(item, 'document_id')} · score {value(item, 'score')} · retrieval {Array.isArray(item.retrieval) ? item.retrieval.map(String).join(', ') : 'backend source record'}</small></article>)}{!results.length && answer.summary == null && <p className="empty">Run a search to inspect returned records and source context.</p>}</div></section><section className="panel"><div className="panel-head"><h2>Project library</h2></div><div className="panel-body stack"><label className="btn">Add source file<input type="file" hidden accept=".txt,.md,.pdf,.docx,.csv,.json,.html,.htm,.rtf" disabled={busy} onChange={event => { void upload(event.target.files?.[0]); event.currentTarget.value = ''; }} /></label><div className="list scroll">{docs.map(doc => <article className="panel panel-body" key={value(doc, 'id')}><strong>{value(doc, 'title', 'Untitled source')}</strong><small>{value(doc, 'source', 'local')} · {value(doc, 'chunk_count')} chunks</small><div>{records(doc.tags).map((tag, i) => <span className="tag" key={String(tag.id ?? i)}>{value(tag, 'tag', String(tag))}</span>)}</div></article>)}</div>{!docs.length && <p className="empty">{busy ? 'Loading sources…' : 'No indexed source documents.'}</p>}</div></section></div>}
    {tab === 'Library' && <div className="workspace"><section className="panel"><div className="panel-head"><h2>Indexed documents</h2><span>{docs.length} returned</span></div><div className="panel-body list">{docs.map(doc => <article className="record-row" key={value(doc, 'id')}><div><strong>{value(doc, 'title', 'Untitled source')}</strong><p>{value(doc, 'source', 'local')} · {value(doc, 'chunk_count')} chunks · {value(doc, 'created_at')}</p><small>{value(doc, 'source_path', 'No source path reported')}</small></div><div>{records(doc.tags).map((tag, i) => <span className="tag" key={String(tag.id ?? i)}>{value(tag, 'tag', String(tag))}</span>)}</div></article>)}{!docs.length && <p className="empty">No indexed documents.</p>}</div></section><section className="panel"><div className="panel-head"><h2>Tags</h2></div><div className="panel-body list">{tags.map((tag, i) => <article className="record-row" key={value(tag, 'id', String(i))}><strong>{value(tag, 'tag')}</strong><span>{value(tag, 'confidence', 'Confidence not reported')}</span></article>)}{!tags.length && <p className="empty">No tags returned by the local index.</p>}</div></section></div>}
    {tab === 'Graph' && <div className="workspace"><section className="panel"><div className="panel-head"><h2>Knowledge entities</h2><span>{records(graph, 'entities').length} nodes</span></div><div className="panel-body list">{records(graph, 'entities').map((entity, i) => <article className="record-row" key={value(entity, 'id', String(i))}><div><strong>{value(entity, 'name')}</strong><p>{value(entity, 'entity_type', 'Entity')}</p></div><span>{value(entity, 'degree', 'Degree not reported')} links</span></article>)}{!records(graph, 'entities').length && <p className="empty">No graph entities are indexed for this project.</p>}</div></section><section className="panel"><div className="panel-head"><h2>Relationships</h2><span>{records(graph, 'relationships').length} edges</span></div><div className="panel-body list">{records(graph, 'relationships').map((edge, i) => <article className="record-row" key={value(edge, 'id', String(i))}><span>{value(edge, 'subject', value(edge, 'source'))}</span><strong>{value(edge, 'relationship', value(edge, 'predicate', 'linked to'))}</strong><span>{value(edge, 'object', value(edge, 'target'))}</span></article>)}</div></section></div>}
    {tab === 'Timeline' && <section className="panel"><div className="panel-head"><h2>Project knowledge timeline</h2></div><div className="panel-body list">{timeline.map((item, i) => <article className="record-row" key={value(item, 'source_id', String(i))}><div><strong>{value(item, 'title', value(item, 'name', value(item, 'type', 'Knowledge event')))}</strong><p>{value(item, 'type', 'Index record')} · source {value(item, 'source_id')}</p></div><time>{value(item, 'timestamp')}</time></article>)}{!timeline.length && <p className="empty">No timeline events returned for this project.</p>}</div></section>}
    {tab === 'Diagnostics' && <div className="workspace"><section className="panel"><div className="panel-head"><h2>Knowledge index status</h2></div><div className="panel-body"><pre>{JSON.stringify(status, null, 2)}</pre></div></section><section className="panel"><div className="panel-head"><h2>RAG capabilities</h2></div><div className="panel-body"><pre>{JSON.stringify(ragStatus, null, 2)}</pre></div></section></div>}
    </div>
  </PageFrame>;
}
