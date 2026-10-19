import { useEffect, useRef, useState } from 'react';
import { api, record, records } from '../lib/react/api';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '../components/ui/dialog';
import './vault-page.css';

type Row = Record<string, unknown>;
const categories = ['secret', 'password', 'api_key', 'token', 'certificate', 'private_note', 'credential'];
export const vaultProject = () => new URL(location.href).searchParams.get('project') || localStorage.getItem('oa-project') || 'default';
export function vaultEntryMetadata(entry: Row) {
  return { id: String(entry.id ?? entry.label ?? ''), label: String(entry.label ?? 'Untitled entry'), category: String(entry.category ?? 'secret'), tags: Array.isArray(entry.tags) ? entry.tags.map(String) : [] };
}
export function filterVaultEntries(entries: Row[], query: string) {
  const term = query.trim().toLowerCase();
  return entries.filter(entry => { const item = vaultEntryMetadata(entry); return !term || `${item.label} ${item.category} ${item.tags.join(' ')}`.toLowerCase().includes(term); });
}

export function VaultPage() {
  const [status, setStatus] = useState<Row>({});
  const [entries, setEntries] = useState<Row[]>([]);
  const [error, setError] = useState('');
  const [loadError, setLoadError] = useState('');
  const [notice, setNotice] = useState('');
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [label, setLabel] = useState('');
  const [category, setCategory] = useState('secret');
  const [secret, setSecret] = useState('');
  const [passphrase, setPassphrase] = useState('');
  const [reveals, setReveals] = useState<Record<string, string>>({});
  const [revealPass, setRevealPass] = useState<Record<string, string>>({});
  const [revealPrompt, setRevealPrompt] = useState('');
  const [revealBusy, setRevealBusy] = useState('');
  const [filter, setFilter] = useState('');
  const [filterCategory, setFilterCategory] = useState('');
  const [oldPassphrase, setOldPassphrase] = useState('');
  const [newPassphrase, setNewPassphrase] = useState('');
  const [exportPassphrase, setExportPassphrase] = useState('');
  const [addOpen, setAddOpen] = useState(false);
  const timers = useRef(new Map<string, ReturnType<typeof setTimeout>>());

  function clearReveal(id?: string) {
    if (id) {
      const timer = timers.current.get(id);
      if (timer) clearTimeout(timer);
      timers.current.delete(id);
      setReveals(current => { const next = { ...current }; delete next[id]; return next; });
      setRevealPass(current => ({ ...current, [id]: '' }));
      setRevealPrompt(current => current === id ? '' : current);
      return;
    }
    for (const timer of timers.current.values()) clearTimeout(timer);
    timers.current.clear();
    setReveals({}); setRevealPass({}); setRevealPrompt('');
  }

  async function refresh() {
    setLoading(true); setLoadError('');
    try {
      const project = encodeURIComponent(vaultProject()), categoryFilter = encodeURIComponent(filterCategory);
      const [nextStatus, nextEntries] = await Promise.all([
        api.get<unknown>(`/api/vault/status?project=${project}`),
        api.get<unknown>(`/api/vault/entries?project=${project}&limit=100${categoryFilter ? `&category=${categoryFilter}` : ''}`),
      ]);
      setStatus(record(nextStatus)); setEntries(records(nextEntries, 'entries'));
    } catch (cause) { setLoadError(cause instanceof Error ? cause.message : 'Vault could not be reached.'); }
    finally { setLoading(false); }
  }

  useEffect(() => {
    void refresh();
    return () => {
      for (const timer of timers.current.values()) clearTimeout(timer);
      timers.current.clear();
    };
  }, []);

  // A revealed credential should not remain readable after the user switches
  // tabs or minimizes the app. The existing timer still conceals it after 15s
  // while the page remains visible.
  useEffect(() => {
    const concealWhenHidden = () => {
      if (document.visibilityState === 'hidden') clearReveal();
    };
    document.addEventListener('visibilitychange', concealWhenHidden);
    return () => document.removeEventListener('visibilitychange', concealWhenHidden);
  }, []);

  async function store(event: React.FormEvent) {
    event.preventDefault(); setSaving(true); setError(''); setNotice('');
    try {
      await api.post('/api/vault/store', { label, category, secret, passphrase, project: vaultProject() });
      setSecret(''); setPassphrase(''); setLabel(''); setCategory('secret'); setAddOpen(false);
      setNotice('Entry stored.'); await refresh();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'The entry was not stored.'); }
    finally { setSaving(false); }
  }

  async function reveal(id: string) {
    setError(''); setNotice(''); setRevealBusy(id);
    try {
      const result = record(await api.post<unknown>('/api/vault/retrieve', { entry_id: id, passphrase: revealPass[id], project: vaultProject() }));
      const value = result.secret, secretValue = record(value).secret ?? value;
      setReveals(current => ({ ...current, [id]: typeof secretValue === 'string' ? secretValue : JSON.stringify(secretValue ?? '') }));
      setRevealPass(current => ({ ...current, [id]: '' })); setRevealPrompt('');
      const previous = timers.current.get(id); if (previous) clearTimeout(previous);
      timers.current.set(id, setTimeout(() => { timers.current.delete(id); setReveals(current => { const next = { ...current }; delete next[id]; return next; }); }, 15_000));
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'The secret could not be retrieved.'); }
    finally { setRevealBusy(''); }
  }

  async function rotateKey(event: React.FormEvent) {
    event.preventDefault();
    if (!confirm('Re-encrypt every vault entry in this project under a new passphrase? The previous passphrase will stop working after success.')) return;
    setSaving(true); setError(''); setNotice('');
    try {
      const result = record(await api.post<unknown>('/api/vault/rotate-key', { old_passphrase: oldPassphrase, new_passphrase: newPassphrase, project: vaultProject() }));
      setNotice(`${String(result.rotated ?? 0)} entries re-encrypted. Record the new passphrase securely.`); setOldPassphrase(''); setNewPassphrase(''); clearReveal();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Vault key rotation failed.'); }
    finally { setSaving(false); }
  }

  async function exportVault(event: React.FormEvent) {
    event.preventDefault();
    if (!confirm('Export an encrypted bundle of all vault entries for this project? The downloaded file remains encrypted with your passphrase.')) return;
    setSaving(true); setError(''); setNotice('');
    try {
      const bundle = await api.post<unknown>('/api/vault/export', { passphrase: exportPassphrase, project: vaultProject() });
      const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' }), url = URL.createObjectURL(blob), link = document.createElement('a');
      link.href = url; link.download = `openagent-vault-${vaultProject()}.json`; link.click(); URL.revokeObjectURL(url);
      setNotice(`Encrypted vault bundle downloaded (${String(record(bundle).entry_count ?? 0)} entries).`); setExportPassphrase('');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Vault export failed.'); }
    finally { setSaving(false); }
  }

  const visibleEntries = filterVaultEntries(entries, filter).filter(entry => !filterCategory || String(entry.category ?? 'secret') === filterCategory);
  const showFirstEntryAction = !loading && !loadError && entries.length === 0;
  return <main className="vault-page">
    <header className="vault-head"><h1>Vault</h1><div><button className="btn" onClick={() => void refresh()} aria-label="Refresh vault" title="Refresh vault">↻</button>{!showFirstEntryAction&&<button className="btn primary" onClick={() => setAddOpen(true)}>Add entry</button>}</div></header>
    <div className="vault-count"><span>{String(status.total_entries ?? entries.length)} entries</span>{Object.keys(reveals).length > 0 && <button className="btn" onClick={() => clearReveal()}>Conceal all</button>}</div>
    {error && <p className="notice error" role="alert">{error}</p>}{notice && <p className="notice success" role="status">{notice}</p>}
    {loadError && <p className="notice error" role="alert">{loadError} <button className="btn" onClick={() => void refresh()}>Retry</button></p>}
    <section className={`vault-list${!loading && !loadError && entries.length === 0 ? ' is-empty' : ''}`} aria-label="Vault entries" aria-busy={loading}>
      {(entries.length > 0 || loading || loadError) && <div className="vault-list-head"><h2>Entries</h2><div className="vault-filters"><input className="field" aria-label="Filter vault entries" placeholder="Search entries" value={filter} onChange={event => setFilter(event.target.value)} /><select className="field" aria-label="Filter category" value={filterCategory} onChange={event => { setFilterCategory(event.target.value); setTimeout(() => void refresh(), 0); }}><option value="">All categories</option>{categories.map(item => <option key={item}>{item}</option>)}</select></div></div>}
      {visibleEntries.map(entry => {
        const item = vaultEntryMetadata(entry), id = item.id;
        return <article className="vault-entry" key={id}>
          <div className="vault-entry-main"><div><strong>{item.label}</strong><small>{item.category}{item.tags.length ? ` · ${item.tags.join(', ')}` : ''}</small></div><span className="vault-mask" aria-label="Secret value masked">••••••••</span></div>
          {reveals[id] !== undefined ? <div className="vault-revealed"><span className="vault-reveal-time">Revealed briefly</span><code>{reveals[id]}</code><button className="btn" onClick={() => clearReveal(id)}>Conceal</button></div>
            : revealPrompt === id ? <form className="vault-reveal-form" onSubmit={event => { event.preventDefault(); void reveal(id); }}><label htmlFor={`vault-pass-${id}`}>Vault passphrase</label><input id={`vault-pass-${id}`} type="password" autoComplete="current-password" className="field" value={revealPass[id] ?? ''} onChange={event => setRevealPass(current => ({ ...current, [id]: event.target.value }))} autoFocus /><button className="btn primary" disabled={!revealPass[id] || !!revealBusy}>{revealBusy === id ? 'Checking…' : 'Reveal for 15 seconds'}</button><button type="button" className="btn" onClick={() => { setRevealPrompt(''); setRevealPass(current => ({ ...current, [id]: '' })); }}>Cancel</button></form>
            : <button className="btn vault-reveal-trigger" onClick={() => { setError(''); setRevealPrompt(id); }}>Reveal</button>}
        </article>;
      })}
      {loading && <div className="vault-empty" role="status">Loading entries…</div>}
      {!loading && !loadError && !visibleEntries.length && <div className="vault-empty"><strong>{entries.length ? 'No matching entries' : 'Your vault is empty'}</strong>{!entries.length && <button className="btn primary" onClick={() => setAddOpen(true)}>Add your first entry</button>}</div>}
    </section>
    <details className="vault-maintenance"><summary>Vault maintenance</summary><div className="vault-maintenance-grid"><form className="stack" onSubmit={event => void rotateKey(event)}><h3>Rotate passphrase</h3><label>Current passphrase<input className="field" type="password" value={oldPassphrase} onChange={event => setOldPassphrase(event.target.value)} required autoComplete="current-password" /></label><label>New passphrase<input className="field" type="password" value={newPassphrase} onChange={event => setNewPassphrase(event.target.value)} required autoComplete="new-password" /></label><button className="btn danger" disabled={saving}>Re-encrypt entries</button></form><form className="stack" onSubmit={event => void exportVault(event)}><h3>Encrypted export</h3><label>Vault passphrase<input className="field" type="password" value={exportPassphrase} onChange={event => setExportPassphrase(event.target.value)} required autoComplete="current-password" /></label><button className="btn" disabled={saving}>Download bundle</button></form></div></details>
    <Dialog open={addOpen} onOpenChange={open => { setAddOpen(open); if (!open) { setSecret(''); setPassphrase(''); } }}><DialogContent className="vault-dialog"><DialogTitle>Add vault entry</DialogTitle><DialogDescription>Values remain masked until you reveal them.</DialogDescription><form className="stack" onSubmit={event => void store(event)}><label>Label<input className="field" value={label} onChange={event => setLabel(event.target.value)} required autoComplete="off" /></label><label>Category<select className="field" value={category} onChange={event => setCategory(event.target.value)}>{categories.map(item => <option key={item}>{item}</option>)}</select></label><label>Secret value<input className="field" type="password" value={secret} onChange={event => setSecret(event.target.value)} required autoComplete="new-password" /></label><label>Vault passphrase<input className="field" type="password" value={passphrase} onChange={event => setPassphrase(event.target.value)} required autoComplete="current-password" /></label><footer><button type="button" className="btn" onClick={() => { setAddOpen(false); setSecret(''); setPassphrase(''); }}>Cancel</button><button className="btn primary" disabled={saving}>{saving ? 'Encrypting…' : 'Encrypt and store'}</button></footer></form></DialogContent></Dialog>
  </main>;
}
