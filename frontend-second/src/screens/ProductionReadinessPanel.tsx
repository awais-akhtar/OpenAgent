import React, { useState, useEffect } from 'react';
import {
  Mic, Zap, HardDrive, CalendarDays,
  CheckCircle2, Clock, Radio,
} from 'lucide-react';

import { apiGet, apiConnectionMessage } from '../api/client';
import type { JsonRecord } from '../api/client';

interface TabProps {
  tabs: Array<{ id: string; label: string; icon: React.ReactNode; badge?: number }>;
  active: string;
  onChange: (id: string) => void;
}

const TabBar: React.FC<TabProps> = ({ tabs, active, onChange }) => (
  <div className="production-tabs">
    {tabs.map(tab => (
      <button
        key={tab.id}
        onClick={() => onChange(tab.id)}
        className={`production-tab ${active === tab.id ? 'active' : ''}`}
      >
        <span className="tab-icon">{tab.icon}</span>
        <span className="tab-label">{tab.label}</span>
        {tab.badge ? <span className="tab-badge">{tab.badge}</span> : null}
      </button>
    ))}
  </div>
);

const rowsFrom = (payload: unknown, ...keys: string[]): JsonRecord[] => {
  if (!payload || typeof payload !== 'object') return [];
  const source = payload as JsonRecord;
  for (const key of keys) {
    const value = source[key];
    if (Array.isArray(value)) return value.filter((row): row is JsonRecord => Boolean(row) && typeof row === 'object' && !Array.isArray(row));
  }
  return [];
};

/** What the backend reports, or why it could not be read. Never a stand-in. */
function useBackendRead<T extends JsonRecord>(path: string, refreshMs = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    const read = async () => {
      try {
        const payload = await apiGet<T>(path, controller.signal);
        if (!cancelled) { setData(payload); setError(''); }
      } catch (cause) {
        // No fabricated fallback: an unreadable panel says so and shows nothing.
        if (!cancelled) { setData(null); setError(apiConnectionMessage(cause)); }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void read();
    const timer = refreshMs > 0 ? setInterval(() => void read(), refreshMs) : undefined;
    return () => { cancelled = true; controller.abort(); if (timer) clearInterval(timer); };
  }, [path, refreshMs]);

  return { data, error, loading };
}

const ReadState: React.FC<{ loading: boolean; error: string; empty?: boolean; emptyText?: string }> = ({ loading, error, empty, emptyText }) => {
  if (loading) return <p className="read-state" role="status">Reading from the backend…</p>;
  if (error) return <p className="read-state" role="alert">{error}</p>;
  if (empty) return <p className="read-state">{emptyText ?? 'The backend reported no records.'}</p>;
  return null;
};

const VoiceSTTPanel: React.FC = () => {
  // This panel used to invent everything it showed: a recorder that touched no
  // microphone, a hardcoded "Sample transcription", a random confidence, random
  // waveform bars, a fixed "24ms" latency, three green ticks claiming Faster
  // Whisper, a live socket and audio buffering were all running, and two made-up
  // sessions with made-up frame counts. It is a Live screen, so it now shows
  // only what /api/voice/status reports.
  const { data, error, loading } = useBackendRead('/api/voice/status?project=default', 15_000);
  const stt = (data?.stt ?? {}) as JsonRecord;
  const engines = rowsFrom(stt, 'engines');
  const ready = engines.filter(engine => Boolean(engine.available));

  return (
    <div className="panel-container">
      <div className="card-grid-2">
        <div className="card primary-card">
          <div className="card-header">
            <div>
              <h3>Buffered speech-to-text</h3>
              <p className="card-subtitle">Audio is uploaded in chunks and transcribed locally</p>
            </div>
            <div className={`status-badge ${ready.length > 0 ? 'active' : 'idle'}`}>
              <Mic size={14} />
              {ready.length > 0 ? `${ready.length} engine${ready.length === 1 ? '' : 's'} ready` : 'No engine ready'}
            </div>
          </div>
          <p className="control-detail">
            Recording and transcription are driven from the Voice screen, which uploads captured audio
            over authenticated HTTP. There is no accepted continuous socket transport, so no live stream,
            running confidence or latency figure is shown here.
          </p>
          <ReadState loading={loading} error={error} />
          {data && <div className="stats-row">
            <div className="stat">
              <span className="label">Recorded sessions</span>
              <span className="value">{String(data.sessions ?? '—')}</span>
            </div>
            <div className="stat">
              <span className="label">Turns</span>
              <span className="value">{String(data.turns ?? '—')}</span>
            </div>
            <div className="stat">
              <span className="label">Translations</span>
              <span className="value">{String(data.translations ?? '—')}</span>
            </div>
          </div>}
        </div>

        <div className="card">
          <div className="card-header"><h3>Installed STT engines</h3></div>
          <ReadState loading={loading} error={error} empty={Boolean(data) && engines.length === 0} emptyText="The backend reported no local speech-to-text engines." />
          <div className="status-list">
            {engines.map((engine, index) => {
              const available = Boolean(engine.available);
              return <div className="status-item" key={String(engine.id ?? engine.name ?? index)}>
                <div className={`status-indicator ${available ? 'active' : ''}`}></div>
                <div>
                  <h4>{String(engine.name ?? engine.id ?? 'Engine')}</h4>
                  <p>{available ? 'Installed and selectable' : String(engine.reason ?? 'Not installed')}</p>
                </div>
              </div>;
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

const AutonomousSchedulerPanel: React.FC = () => {
  // The three reminders, the "12 delivered today", the "3 pending", and the
  // three green worker indicators were all hardcoded. Reminders have a real
  // backend route, so the list and the counts come from it; the delivered flag
  // is the column the calendar system actually records.
  const { data, error, loading } = useBackendRead('/api/calendar/reminders?project=default', 30_000);
  const reminders = rowsFrom(data, 'reminders', 'items');
  const delivered = reminders.filter(reminder => Boolean(reminder.delivered));
  const pending = reminders.filter(reminder => !reminder.delivered);

  return (
    <div className="panel-container">
      <div className="card-grid-2">
        <div className="card">
          <div className="card-header">
            <h3>Due reminders</h3>
            <span className="fixture">project default</span>
          </div>
          <p className="control-detail">
            This reads the reminders the calendar system reports as due. Delivery preferences and the
            Deliver control live on the Calendar screen, which is where a reminder is acknowledged or
            sent; nothing is delivered from this panel.
          </p>
          <ReadState loading={loading} error={error} />
          {data && <div className="stats-grid">
            <div className="stat-box">
              <div className="stat-value">{reminders.length}</div>
              <div className="stat-label">Due now</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{delivered.length}</div>
              <div className="stat-label">Already delivered</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{pending.length}</div>
              <div className="stat-label">Not yet delivered</div>
            </div>
          </div>}
        </div>

        <div className="card">
          <div className="card-header"><h3>What this panel does not claim</h3></div>
          <p className="control-detail">
            Worker, calendar-integration and hub indicators used to be drawn here as three permanent
            green lights. None of them was read from anything, so they have been removed rather than
            left showing a state no service reported. Scheduler and worker health is on the
            Automations and Developer screens, which read their own routes.
          </p>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>Reminder records</h3>
          <span className="fixture">{reminders.length} {reminders.length === 1 ? 'reminder' : 'reminders'}</span>
        </div>
        <ReadState loading={loading} error={error} empty={Boolean(data) && reminders.length === 0} emptyText="The calendar system reports no due reminders." />
        <div className="table-like">
          {reminders.map((reminder, index) => {
            const isDelivered = Boolean(reminder.delivered);
            return <div key={String(reminder.id ?? index)} className="row">
              <div className="col">
                <div className="row-label">{String(reminder.title ?? reminder.event_title ?? reminder.id ?? 'Reminder')}</div>
                <div className="row-detail">{String(reminder.trigger_at ?? reminder.remind_at ?? '')}</div>
              </div>
              <div className="col">
                <span className={`badge ${isDelivered ? 'success' : 'warning'}`}>
                  {isDelivered ? 'Delivered' : 'Not delivered'}
                </span>
              </div>
            </div>;
          })}
        </div>
      </div>
    </div>
  );
};

const FineTuningPanel: React.FC = () => {
  // The three jobs, their progress bars and ETAs, the "2.4GB checkpoint",
  // "FP32", "1 validated" and three green validation ticks were all invented.
  // Fine-tuning jobs have a real route, so the list and counts come from it and
  // nothing is shown for figures no route reports.
  const { data, error, loading } = useBackendRead('/api/finetune/jobs?project=default', 15_000);
  const jobs = rowsFrom(data, 'jobs', 'items');
  const countOf = (...states: string[]) =>
    jobs.filter(job => states.includes(String(job.status ?? '').toLowerCase())).length;

  return (
    <div className="panel-container">
      <div className="card-grid-2">
        <div className="card">
          <div className="card-header">
            <h3>Training jobs</h3>
            <span className="fixture">project default</span>
          </div>
          <ReadState loading={loading} error={error} />
          {data && <div className="stat-grid-2">
            <div className="stat-box">
              <div className="stat-value">{countOf('running', 'training')}</div>
              <div className="stat-label">Running</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{countOf('queued', 'pending')}</div>
              <div className="stat-label">Queued</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{countOf('completed', 'succeeded')}</div>
              <div className="stat-label">Completed</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{countOf('failed', 'cancelled')}</div>
              <div className="stat-label">Failed or cancelled</div>
            </div>
          </div>}
        </div>

        <div className="card">
          <div className="card-header"><h3>Checkpoints and precision</h3></div>
          <p className="control-detail">
            Checkpoint size, precision and per-precision validation ticks were drawn here as fixed
            values that no route reported. Dataset, job, checkpoint and result records, with the
            controls that act on them, are on the Fine-tuning screen.
          </p>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>Job records</h3>
          <span className="fixture">{jobs.length} {jobs.length === 1 ? 'job' : 'jobs'}</span>
        </div>
        <ReadState loading={loading} error={error} empty={Boolean(data) && jobs.length === 0} emptyText="The fine-tuning service reports no jobs for this project." />
        <div className="job-list">
          {jobs.map((job, index) => {
            const status = String(job.status ?? 'unknown').toLowerCase();
            const progress = Number(job.progress ?? job.progress_percent ?? NaN);
            return <div key={String(job.id ?? index)} className="job-item">
              <div className="job-header">
                <h4>{String(job.name ?? job.base_model ?? job.id ?? 'Job')}</h4>
                <span className={`status-label ${status}`}>
                  {status === 'running' && <Radio size={12} />}
                  {(status === 'completed' || status === 'succeeded') && <CheckCircle2 size={12} />}
                  {(status === 'queued' || status === 'pending') && <Clock size={12} />}
                  {status}
                </span>
              </div>
              {Number.isFinite(progress) && <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${Math.max(0, Math.min(100, progress))}%` }}></div>
              </div>}
              <div className="job-footer">
                <span>{Number.isFinite(progress) ? `${progress}%` : 'Progress not reported'}</span>
                <span>{job.updated_at ? String(job.updated_at) : ''}</span>
              </div>
            </div>;
          })}
        </div>
      </div>
    </div>
  );
};

const UltraModelPanel: React.FC = () => {
  const [sessions, setSessions] = useState<any[]>([]);
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [serverUrl, setServerUrl] = useState<string>('');

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    // These two reads used to go through a bare fetch(), so they carried no
    // credential and were answered 401 on any authenticated install; and on
    // failure the panel substituted two invented sessions ("Llama-7B", "ready",
    // "4h 32m") into a Live screen. It now uses the authenticated client and
    // shows nothing it could not read.
    const read = async () => {
      try {
        const statusData = await apiGet<JsonRecord>('/api/ultra-model/status', controller.signal);
        if (cancelled) return;
        setStatus(statusData);
        setServerUrl(statusData.server_host && statusData.server_port
          ? `${String(statusData.server_host)}:${String(statusData.server_port)}`
          : '');
        const sessionsData = await apiGet<JsonRecord>('/api/ultra-model/sessions', controller.signal);
        if (cancelled) return;
        setSessions(rowsFrom(sessionsData, 'sessions', 'items'));
        setError(null);
      } catch (cause) {
        if (cancelled) return;
        setStatus(null);
        setSessions([]);
        setServerUrl('');
        setError(apiConnectionMessage(cause));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void read();
    const interval = setInterval(() => void read(), 10000);
    return () => { cancelled = true; controller.abort(); clearInterval(interval); };
  }, []);

  const readinessChecks = [
    {
      name: 'llama-server',
      status: status?.available ? 'available' : 'unavailable',
      via: status?.server_host ? 'live' : 'offline',
      icon: status?.available ? '✓' : '✗'
    },
    {
      name: 'GGUF Model',
      status: status?.model_validated ? 'validated' : 'pending',
      format: status?.model_format ? String(status.model_format) : '',
      icon: status?.model_validated ? '✓' : '—'
    },
    {
      name: 'Process Identity',
      status: status?.process_tracked ? 'tracked' : 'pending',
      icon: status?.process_tracked ? '✓' : '—'
    },
    {
      name: 'Manifest Validation',
      status: status?.manifest_valid ? 'passed' : 'pending',
      icon: status?.manifest_valid ? '✓' : '—'
    },
  ];

  return (
    <div className="panel-container">
      <div className="card-grid-2">
        <div className="card primary-card">
          <div className="card-header">
            <h3>Ultra Model Status</h3>
            <div className={`status-badge ${status?.available ? 'active' : 'warning'}`}>
              {loading ? 'Checking…' : status?.available ? 'Ready' : 'Offline'}
            </div>
          </div>
          {error && (
            <div className="error-notice" style={{ padding: '12px', color: '#ff6b6b', fontSize: '13px', marginBottom: '12px' }}>
              ⚠ {error}
            </div>
          )}
          <div className="readiness-checklist">
            {readinessChecks.map((check, idx) => (
              <div key={idx} className="check-item">
                <div className="check-icon">{check.icon}</div>
                <div className="check-details">
                  <h4>{check.name}</h4>
                  <p>{check.status}
                    {check.via && ` · via ${check.via}`}
                    {check.format && ` · ${check.format}`}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Infrastructure</h3>
          </div>
          <div className="infra-details">
            <div className="detail-row">
              <span className="label">Server Address</span>
              <span className="value" title={serverUrl || 'Connecting…'}>
                {loading ? 'Connecting…' : serverUrl || 'Not available'}
              </span>
            </div>
            <div className="detail-row">
              <span className="label">Model Format</span>
              <span className="value">{status?.model_format ? String(status.model_format) : 'Not reported'}</span>
            </div>
            <div className="detail-row">
              <span className="label">Serve Type</span>
              <span className="value">{status?.serve_type ? String(status.serve_type) : 'Not reported'}</span>
            </div>
            <div className="detail-row">
              <span className="label">Sessions</span>
              <span className="value">{sessions.length} {sessions.length === 1 ? 'active' : 'active'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Sessions */}
      <div className="card">
        <div className="card-header">
          <h3>Model Sessions</h3>
          <span className="fixture">{sessions.length} sessions</span>
        </div>
        {loading ? (
          <div style={{ padding: '24px', textAlign: 'center', color: '#999' }}>Loading sessions…</div>
        ) : sessions.length === 0 ? (
          <div style={{ padding: '24px', textAlign: 'center', color: '#999' }}>No active sessions</div>
        ) : (
          <div className="table-like">
            {sessions.map(session => (
              <div key={session.id} className="row">
                <div className="col">
                  <div className="row-label">{session.model || session.name || 'Unknown Model'}</div>
                  <div className="row-detail">ID: {session.id}</div>
                </div>
                <div className="col">
                  <span className={`badge ${session.status === 'ready' || session.status === 'active' ? 'success' : 'warning'}`}>
                    {session.status || 'pending'}
                  </span>
                </div>
                <div className="col">
                  <div className="row-label">{session.uptime || session.duration || '—'}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export const ProductionReadinessPanel: React.FC<{ tab?: string }> = ({ tab }) => {
  const tabMap: Record<string, string> = {
    'Voice STT': 'voice',
    'Auto Scheduling': 'scheduler',
    'Fine-tuning': 'finetune',
    'Ultra Model': 'ultramodel',
  };
  const [activeTab, setActiveTab] = useState(() => tab ? tabMap[tab] || 'voice' : 'voice');

  useEffect(() => {
    if (tab) {
      setActiveTab(tabMap[tab] || 'voice');
    }
  }, [tab]);

  // The badges read 2, 3, 3, 2 regardless of what any service held, so each tab
  // advertised a count of records that did not exist. A count belongs to the
  // panel that read it, and each panel now shows its own.
  const tabs = [
    { id: 'voice', label: 'Voice STT', icon: <Mic size={18} /> },
    { id: 'scheduler', label: 'Auto Scheduling', icon: <CalendarDays size={18} /> },
    { id: 'finetune', label: 'Fine-tuning', icon: <HardDrive size={18} /> },
    { id: 'ultramodel', label: 'Ultra Model', icon: <Zap size={18} /> },
  ];

  return (
    <div className="production-panel-wrapper">
      <style>{`
        .production-panel-wrapper {
          background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
          color: #e0e0e0;
          padding: 24px;
          border-radius: 12px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        .production-tabs {
          display: flex;
          gap: 8px;
          margin-bottom: 32px;
          border-bottom: 1px solid rgba(255,255,255,0.1);
          overflow-x: auto;
          padding-bottom: 16px;
        }

        .production-tab {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 16px;
          background: transparent;
          border: none;
          color: #999;
          cursor: pointer;
          border-bottom: 2px solid transparent;
          transition: all 0.2s;
          white-space: nowrap;
          font-size: 14px;
          font-weight: 500;
        }

        .production-tab:hover {
          color: #e0e0e0;
        }

        .production-tab.active {
          color: #00d9ff;
          border-bottom-color: #00d9ff;
        }

        .tab-icon {
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .tab-badge {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          background: rgba(0, 217, 255, 0.2);
          color: #00d9ff;
          border-radius: 10px;
          min-width: 20px;
          height: 20px;
          font-size: 12px;
          font-weight: 600;
        }

        .panel-container {
          display: flex;
          flex-direction: column;
          gap: 24px;
        }

        .card-grid-2 {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 20px;
        }

        .card {
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 8px;
          padding: 20px;
          backdrop-filter: blur(10px);
          transition: all 0.3s;
        }

        .card:hover {
          border-color: rgba(0, 217, 255, 0.3);
          box-shadow: 0 0 20px rgba(0, 217, 255, 0.1);
        }

        .card.primary-card {
          border: 1px solid rgba(0, 217, 255, 0.2);
          background: linear-gradient(135deg, rgba(0,217,255,0.1), rgba(0,100,200,0.05));
        }

        .card-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
          gap: 16px;
        }

        .card-header h3 {
          margin: 0;
          font-size: 16px;
          font-weight: 600;
        }

        .card-subtitle {
          color: #888;
          font-size: 13px;
          margin-top: 4px;
        }

        .fixture {
          font-size: 12px;
          color: #666;
          background: rgba(255,255,255,0.05);
          padding: 4px 8px;
          border-radius: 4px;
        }

        .status-badge {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 4px 12px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 500;
          background: rgba(255,255,255,0.1);
          color: #aaa;
        }

        .status-badge.active {
          background: rgba(34, 197, 94, 0.2);
          color: #22c55e;
        }

        .status-badge.disabled {
          background: rgba(239, 68, 68, 0.2);
          color: #ef4444;
        }

        .status-badge.idle {
          background: rgba(107, 114, 128, 0.2);
          color: #9ca3af;
        }

        .recording-display {
          background: rgba(0,217,255,0.1);
          border: 1px solid rgba(0,217,255,0.2);
          border-radius: 6px;
          padding: 16px;
          margin-bottom: 16px;
        }

        .waveform {
          display: flex;
          align-items: flex-end;
          justify-content: center;
          gap: 4px;
          height: 100px;
        }

        .wave-bar {
          width: 4px;
          background: linear-gradient(180deg, #00d9ff, #00a3cc);
          border-radius: 2px;
          transition: height 0.1s;
        }

        .waveform.animating .wave-bar {
          animation: wave 0.3s ease-in-out;
        }

        @keyframes wave {
          0%, 100% { transform: scaleY(0.3); }
          50% { transform: scaleY(1); }
        }

        .transcript-box {
          background: rgba(0,0,0,0.3);
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 6px;
          padding: 12px;
          min-height: 60px;
          font-size: 14px;
          line-height: 1.5;
          color: #e0e0e0;
          margin-bottom: 16px;
        }

        .placeholder {
          color: #666;
          font-style: italic;
        }

        .stats-row, .stats-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
          gap: 12px;
          margin: 16px 0;
        }

        .stat {
          background: rgba(0,0,0,0.2);
          padding: 12px;
          border-radius: 6px;
          text-align: center;
        }

        .stat .label {
          display: block;
          font-size: 11px;
          color: #888;
          margin-bottom: 4px;
          text-transform: uppercase;
        }

        .stat .value {
          display: block;
          font-size: 18px;
          font-weight: 600;
          color: #00d9ff;
        }

        .button-group {
          display: flex;
          gap: 10px;
        }

        .btn {
          flex: 1;
          padding: 10px 16px;
          border: none;
          border-radius: 6px;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 6px;
          transition: all 0.2s;
        }

        .btn-primary {
          background: linear-gradient(135deg, #00d9ff, #0088cc);
          color: white;
        }

        .btn-primary:hover {
          box-shadow: 0 0 20px rgba(0, 217, 255, 0.4);
        }

        .btn-danger {
          background: #ef4444;
          color: white;
        }

        .btn-danger:hover {
          background: #dc2626;
        }

        .status-list {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .status-item {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px;
          background: rgba(0,0,0,0.2);
          border-radius: 6px;
        }

        .status-check {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 24px;
          height: 24px;
          background: rgba(34, 197, 94, 0.2);
          color: #22c55e;
          border-radius: 4px;
          font-size: 12px;
          font-weight: 600;
        }

        .status-item h4 {
          margin: 0 0 4px 0;
          font-size: 13px;
        }

        .status-item p {
          margin: 0;
          font-size: 12px;
          color: #888;
        }

        .status-indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #666;
        }

        .status-indicator.active {
          background: #22c55e;
          box-shadow: 0 0 10px rgba(34, 197, 94, 0.5);
        }

        .stat-box {
          background: rgba(0,0,0,0.2);
          padding: 16px;
          border-radius: 6px;
          text-align: center;
        }

        .stat-icon {
          font-size: 24px;
          margin-bottom: 8px;
        }

        .stat-value {
          display: block;
          font-size: 20px;
          font-weight: 600;
          color: #00d9ff;
          margin-bottom: 4px;
        }

        .stat-label {
          display: block;
          font-size: 12px;
          color: #888;
        }

        .stat-grid-2 {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 12px;
        }

        .notification-control {
          padding: 16px;
          background: rgba(0,0,0,0.2);
          border-radius: 6px;
          margin-bottom: 16px;
        }

        .control-header {
          font-size: 13px;
          font-weight: 500;
          margin-bottom: 12px;
        }

        .toggle-switch {
          position: relative;
          width: 50px;
          height: 26px;
          background: #555;
          border: none;
          border-radius: 13px;
          cursor: pointer;
          transition: background 0.3s;
          margin-bottom: 8px;
        }

        .toggle-switch.on {
          background: #22c55e;
        }

        .toggle-thumb {
          position: absolute;
          width: 22px;
          height: 22px;
          background: white;
          border-radius: 50%;
          top: 2px;
          left: 2px;
          transition: left 0.3s;
        }

        .toggle-switch.on .toggle-thumb {
          left: 26px;
        }

        .control-detail {
          font-size: 12px;
          color: #888;
          margin: 0;
        }

        .read-state {
          font-size: 12px;
          color: #9a9ab0;
          margin: 12px 0 0;
        }

        .read-state[role="alert"] {
          color: #ff8a8a;
        }

        .table-like {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .row {
          display: grid;
          grid-template-columns: 1fr auto auto;
          align-items: center;
          gap: 16px;
          padding: 12px;
          background: rgba(0,0,0,0.2);
          border-radius: 6px;
          border-left: 3px solid rgba(0,217,255,0.2);
        }

        .col {
          display: flex;
          flex-direction: column;
        }

        .row-label {
          font-size: 14px;
          font-weight: 500;
        }

        .row-detail {
          font-size: 12px;
          color: #888;
          margin-top: 2px;
        }

        .badge {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 4px 10px;
          background: rgba(107, 114, 128, 0.3);
          color: #9ca3af;
          border-radius: 4px;
          font-size: 12px;
          font-weight: 500;
          width: fit-content;
        }

        .badge.success {
          background: rgba(34, 197, 94, 0.2);
          color: #22c55e;
        }

        .badge.warning {
          background: rgba(251, 146, 60, 0.2);
          color: #fb923c;
        }

        .job-list {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .job-item {
          padding: 16px;
          background: rgba(0,0,0,0.2);
          border-radius: 6px;
          border-left: 3px solid rgba(0,217,255,0.3);
        }

        .job-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }

        .job-header h4 {
          margin: 0;
          font-size: 14px;
        }

        .status-label {
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: 12px;
          padding: 4px 8px;
          background: rgba(0,217,255,0.2);
          color: #00d9ff;
          border-radius: 4px;
          text-transform: capitalize;
        }

        .progress-bar {
          width: 100%;
          height: 6px;
          background: rgba(0,0,0,0.3);
          border-radius: 3px;
          overflow: hidden;
          margin-bottom: 8px;
        }

        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #00d9ff, #0088cc);
          border-radius: 3px;
          transition: width 0.3s;
        }

        .job-footer {
          display: flex;
          justify-content: space-between;
          font-size: 12px;
          color: #888;
        }

        .readiness-checklist {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .check-item {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px;
          background: rgba(34, 197, 94, 0.1);
          border: 1px solid rgba(34, 197, 94, 0.2);
          border-radius: 6px;
        }

        .check-icon {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 28px;
          height: 28px;
          background: rgba(34, 197, 94, 0.2);
          color: #22c55e;
          border-radius: 4px;
          font-weight: 600;
          font-size: 16px;
        }

        .check-details h4 {
          margin: 0 0 2px 0;
          font-size: 13px;
        }

        .check-details p {
          margin: 0;
          font-size: 12px;
          color: #888;
        }

        .infra-details {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .detail-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 10px;
          background: rgba(0,0,0,0.2);
          border-radius: 4px;
        }

        .detail-row .label {
          font-size: 13px;
          color: #888;
        }

        .detail-row .value {
          font-size: 13px;
          font-weight: 500;
          color: #00d9ff;
        }

        @media (max-width: 768px) {
          .production-panel-wrapper {
            padding: 16px;
          }

          .card-grid-2 {
            grid-template-columns: 1fr;
          }

          .stat-grid-2 {
            grid-template-columns: repeat(2, 1fr);
          }

          .row {
            grid-template-columns: 1fr;
            gap: 8px;
          }
        }
      `}</style>

      <div className="card-header" style={{ marginBottom: '24px' }}>
        <div>
          <h2 style={{ margin: '0 0 8px 0', fontSize: '24px' }}>Production Readiness</h2>
          <p style={{ margin: 0, color: '#888', fontSize: '14px' }}>
            Complete feature implementation with real-time monitoring
          </p>
        </div>
      </div>

      <TabBar tabs={tabs} active={activeTab} onChange={setActiveTab} />

      {activeTab === 'voice' && <VoiceSTTPanel />}
      {activeTab === 'scheduler' && <AutonomousSchedulerPanel />}
      {activeTab === 'finetune' && <FineTuningPanel />}
      {activeTab === 'ultramodel' && <UltraModelPanel />}
    </div>
  );
};

export default ProductionReadinessPanel;
