import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Mic, Zap, HardDrive, CalendarDays, Play, Square, AlertCircle,
  CheckCircle2, Clock, Radio, Cpu, Database, Settings2, ChevronRight,
  Activity, BarChart3, TrendingUp, RotateCcw, Download, Upload,
  Lock, Shield, Eye, EyeOff, Filter, SortAsc, Plus, X,
} from 'lucide-react';

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

const VoiceSTTPanel: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [sessions, setSessions] = useState([
    { id: 'voice-1', started: '2 min ago', frames: 45, status: 'Active' },
    { id: 'voice-2', started: '15 min ago', frames: 120, status: 'Completed' },
  ]);

  const startRecording = () => {
    setIsRecording(true);
    setFrameCount(0);
    setTranscript('');
    const interval = setInterval(() => {
      setFrameCount(f => f + 1);
      setConfidence(Math.min(0.99, 0.7 + Math.random() * 0.25));
    }, 100);
    return () => clearInterval(interval);
  };

  const stopRecording = () => {
    setIsRecording(false);
    setTranscript('Sample transcription: "This is a test voice message with real-time STT"');
  };

  return (
    <div className="panel-container">
      {/* Live Recording Panel */}
      <div className="card-grid-2">
        <div className="card primary-card">
          <div className="card-header">
            <div>
              <h3>Live Transcription</h3>
              <p className="card-subtitle">WebSocket streaming with real STT</p>
            </div>
            <div className={`status-badge ${isRecording ? 'active' : 'idle'}`}>
              {isRecording ? <Radio size={14} /> : <Mic size={14} />}
              {isRecording ? 'Recording' : 'Ready'}
            </div>
          </div>

          <div className="recording-display">
            <div className={`waveform ${isRecording ? 'animating' : ''}`}>
              {Array(20).fill(0).map((_, i) => (
                <div key={i} className="wave-bar" style={{
                  height: isRecording ? `${Math.random() * 100}%` : '20%',
                }}></div>
              ))}
            </div>
          </div>

          <div className="transcript-box">
            {transcript || <span className="placeholder">Waiting for audio input...</span>}
          </div>

          <div className="stats-row">
            <div className="stat">
              <span className="label">Confidence</span>
              <span className="value">{(confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="stat">
              <span className="label">Frames</span>
              <span className="value">{frameCount}</span>
            </div>
            <div className="stat">
              <span className="label">Latency</span>
              <span className="value">24ms</span>
            </div>
          </div>

          <div className="button-group">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`btn ${isRecording ? 'btn-danger' : 'btn-primary'}`}
            >
              {isRecording ? <Square size={16} /> : <Play size={16} />}
              {isRecording ? 'Stop' : 'Start'} Recording
            </button>
          </div>
        </div>

        {/* STT Status */}
        <div className="card">
          <div className="card-header">
            <h3>STT Engine Status</h3>
          </div>
          <div className="status-list">
            <div className="status-item">
              <div className="status-check">✓</div>
              <div>
                <h4>Faster Whisper</h4>
                <p>Primary engine - Ready</p>
              </div>
            </div>
            <div className="status-item">
              <div className="status-check">✓</div>
              <div>
                <h4>WebSocket Connection</h4>
                <p>Live streaming - Active</p>
              </div>
            </div>
            <div className="status-item">
              <div className="status-check">✓</div>
              <div>
                <h4>Audio Buffering</h4>
                <p>Chunk accumulation - Running</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Sessions */}
      <div className="card">
        <div className="card-header">
          <h3>Active Sessions</h3>
          <span className="fixture">{sessions.length} sessions</span>
        </div>
        <div className="table-like">
          {sessions.map(session => (
            <div key={session.id} className="row">
              <div className="col">
                <div className="row-label">{session.id}</div>
                <div className="row-detail">Started {session.started}</div>
              </div>
              <div className="col">
                <div className="row-label">{session.frames} frames</div>
              </div>
              <div className="col">
                <span className={`badge ${session.status === 'Active' ? 'success' : 'default'}`}>
                  {session.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const AutonomousSchedulerPanel: React.FC = () => {
  const [reminders, setReminders] = useState([
    { id: 'r1', event: 'Team Meeting', time: '10:00 AM', delivered: true, status: 'delivered' },
    { id: 'r2', event: 'Project Review', time: '2:30 PM', delivered: false, status: 'pending' },
    { id: 'r3', event: 'Daily Standup', time: '9:00 AM', delivered: true, status: 'delivered' },
  ]);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);

  return (
    <div className="panel-container">
      <div className="card-grid-2">
        <div className="card">
          <div className="card-header">
            <h3>Autonomous Delivery</h3>
            <div className={`status-badge ${notificationsEnabled ? 'active' : 'disabled'}`}>
              {notificationsEnabled ? 'Enabled' : 'Disabled'}
            </div>
          </div>

          <div className="notification-control">
            <div className="control-header">
              <span>Notification Preferences</span>
            </div>
            <button
              onClick={() => setNotificationsEnabled(!notificationsEnabled)}
              className={`toggle-switch ${notificationsEnabled ? 'on' : 'off'}`}
            >
              <span className="toggle-thumb"></span>
            </button>
            <p className="control-detail">
              {notificationsEnabled
                ? 'Reminders will be delivered autonomously'
                : 'Reminders are disabled'}
            </p>
          </div>

          <div className="stats-grid">
            <div className="stat-box">
              <div className="stat-value">12</div>
              <div className="stat-label">Delivered today</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">3</div>
              <div className="stat-label">Pending</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">60s</div>
              <div className="stat-label">Check interval</div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Worker Status</h3>
          </div>
          <div className="status-list">
            <div className="status-item">
              <div className="status-indicator active"></div>
              <div>
                <h4>Background Worker</h4>
                <p>Running - 1 worker</p>
              </div>
            </div>
            <div className="status-item">
              <div className="status-indicator active"></div>
              <div>
                <h4>Calendar Integration</h4>
                <p>Connected</p>
              </div>
            </div>
            <div className="status-item">
              <div className="status-indicator active"></div>
              <div>
                <h4>Integration Hub</h4>
                <p>Preference-aware</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Reminders List */}
      <div className="card">
        <div className="card-header">
          <h3>Scheduled Reminders</h3>
          <span className="fixture">{reminders.length} reminders</span>
        </div>
        <div className="table-like">
          {reminders.map(reminder => (
            <div key={reminder.id} className="row">
              <div className="col">
                <div className="row-label">{reminder.event}</div>
                <div className="row-detail">{reminder.time}</div>
              </div>
              <div className="col">
                <span className={`badge ${reminder.status === 'delivered' ? 'success' : 'warning'}`}>
                  {reminder.status === 'delivered' ? '✓ Delivered' : '⏱ Pending'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const FineTuningPanel: React.FC = () => {
  const [jobs, setJobs] = useState([
    { id: 'job-1', name: 'GPT-2 Fine-tune', status: 'running', progress: 65, eta: '2h 15m' },
    { id: 'job-2', name: 'BERT Adapter', status: 'completed', progress: 100, eta: '—' },
    { id: 'job-3', name: 'T5 Training', status: 'queued', progress: 0, eta: '—' },
  ]);

  return (
    <div className="panel-container">
      <div className="card-grid-2">
        <div className="card">
          <div className="card-header">
            <h3>Training Workspace</h3>
          </div>
          <div className="stat-grid-2">
            <div className="stat-box">
              <div className="stat-icon">📊</div>
              <div className="stat-value">3</div>
              <div className="stat-label">Active jobs</div>
            </div>
            <div className="stat-box">
              <div className="stat-icon">💾</div>
              <div className="stat-value">2.4GB</div>
              <div className="stat-label">Checkpoint size</div>
            </div>
            <div className="stat-box">
              <div className="stat-icon">⚡</div>
              <div className="stat-value">FP32</div>
              <div className="stat-label">Precision</div>
            </div>
            <div className="stat-box">
              <div className="stat-icon">✓</div>
              <div className="stat-value">1</div>
              <div className="stat-label">Validated</div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Checkpoint Status</h3>
          </div>
          <div className="status-list">
            <div className="status-item">
              <div className="status-check">✓</div>
              <div>
                <h4>FP32 Validation</h4>
                <p>Full precision - Verified</p>
              </div>
            </div>
            <div className="status-item">
              <div className="status-check">✓</div>
              <div>
                <h4>BF16 Validation</h4>
                <p>Brain float precision - Ready</p>
              </div>
            </div>
            <div className="status-item">
              <div className="status-check">✓</div>
              <div>
                <h4>Worker Failures</h4>
                <p>Tracked and logged</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Training Jobs */}
      <div className="card">
        <div className="card-header">
          <h3>Training Jobs</h3>
          <span className="fixture">{jobs.length} jobs</span>
        </div>
        <div className="job-list">
          {jobs.map(job => (
            <div key={job.id} className="job-item">
              <div className="job-header">
                <h4>{job.name}</h4>
                <span className={`status-label ${job.status}`}>
                  {job.status === 'running' && <Radio size={12} />}
                  {job.status === 'completed' && <CheckCircle2 size={12} />}
                  {job.status === 'queued' && <Clock size={12} />}
                  {job.status}
                </span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${job.progress}%` }}></div>
              </div>
              <div className="job-footer">
                <span>{job.progress}%</span>
                <span>ETA: {job.eta}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const UltraModelPanel: React.FC = () => {
  const [sessions, setSessions] = useState([
    { id: 'session-1', model: 'Llama-7B', status: 'ready', uptime: '4h 32m' },
    { id: 'session-2', model: 'Llama-13B', status: 'planning', uptime: '—' },
  ]);
  const [readinessChecks] = useState([
    { name: 'llama-server', status: 'available', via: 'mock', icon: '✓' },
    { name: 'GGUF Model', status: 'validated', format: 'Q4_K_M', icon: '✓' },
    { name: 'Process Identity', status: 'tracked', icon: '✓' },
    { name: 'Manifest Validation', status: 'passed', icon: '✓' },
  ]);

  return (
    <div className="panel-container">
      <div className="card-grid-2">
        <div className="card primary-card">
          <div className="card-header">
            <h3>Ultra Model Status</h3>
            <div className="status-badge active">Ready</div>
          </div>
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
              <span className="label">Mock Server</span>
              <span className="value">127.0.0.1:8000</span>
            </div>
            <div className="detail-row">
              <span className="label">Model Format</span>
              <span className="value">GGUF</span>
            </div>
            <div className="detail-row">
              <span className="label">Serve Type</span>
              <span className="value">llama-server</span>
            </div>
            <div className="detail-row">
              <span className="label">Sessions</span>
              <span className="value">{sessions.length} active</span>
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
        <div className="table-like">
          {sessions.map(session => (
            <div key={session.id} className="row">
              <div className="col">
                <div className="row-label">{session.model}</div>
                <div className="row-detail">ID: {session.id}</div>
              </div>
              <div className="col">
                <span className={`badge ${session.status === 'ready' ? 'success' : 'warning'}`}>
                  {session.status}
                </span>
              </div>
              <div className="col">
                <div className="row-label">{session.uptime}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export const ProductionReadinessPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState('voice');

  const tabs = [
    { id: 'voice', label: 'Voice STT', icon: <Mic size={18} />, badge: 2 },
    { id: 'scheduler', label: 'Auto Scheduling', icon: <CalendarDays size={18} />, badge: 3 },
    { id: 'finetune', label: 'Fine-tuning', icon: <HardDrive size={18} />, badge: 3 },
    { id: 'ultramodel', label: 'Ultra Model', icon: <Zap size={18} />, badge: 2 },
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
