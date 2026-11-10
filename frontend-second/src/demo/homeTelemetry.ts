/** Local-only dashboard fixtures. Never presented as measurements from Python. */
export const demoHomeTelemetry = {
  metrics: {
    cpu: { count: 16, usage_percent: 24.6, process_seconds: 184.2 },
    memory: { total_gb: 32, used_gb: 13.8, available_gb: 18.2, used_percent: 43.1 },
    disk: { total_gb: 512, used_gb: 226, free_gb: 286, used_percent: 44.1 },
    platform: { system: 'Demo workstation', release: 'Preview', machine: 'x64' },
  },
  health: { status: 'ok', checked_at: 'Demo snapshot' },
  setup: { ollama: { available: true, server_available: true, model_ready: true, selected_model: 'Example local model' }, llamacpp: { available: false, server_available: false, model_ready: false } },
  agentJobs: { jobs: [] },
  modelJobs: { jobs: [] },
} as const;
