# Windows Desktop App

OpenAgent ships a Windows desktop entry point for users who want the modern local workbench to launch as an app window.

The launcher uses one stable runtime at `127.0.0.1:8765`. A user-level lock
prevents source and installed launchers from opening competing servers. Running
the shortcut again focuses the existing window, and closing the app window
stops the HTTP runtime and its accelerator worker.

## What It Installs

- `openagent-desktop`, which opens the modern localhost workbench UI backed by the same SQLite data store.
- `openagent-workbench`, the direct modern workbench launcher.
- A private virtual environment under `%LOCALAPPDATA%\OpenAgent\.venv`.
- A local data directory under `%LOCALAPPDATA%\OpenAgent\data`.
- Start Menu and Desktop shortcuts that launch the desktop shell directly.

The installer does not bundle proprietary code, does not copy third-party UI framework code, and does not start telemetry.

## Install

Open PowerShell in the repository root:

```powershell
.\installers\windows\install.ps1
```

Install from another source path:

```powershell
.\installers\windows\install.ps1 -SourcePath "D:\openagent"
```

Install without a Desktop shortcut:

```powershell
.\installers\windows\install.ps1 -NoDesktopShortcut
```

The default install includes local voice/STT extras for Vosk and faster-whisper support. For a minimal install without those optional packages:

```powershell
.\installers\windows\install.ps1 -SkipVoiceExtras
```

## Launch

Use the Start Menu shortcut, Desktop shortcut, or:

```powershell
%LOCALAPPDATA%\OpenAgent\OpenAgent.cmd
```

Developer launch:

```powershell
python -m openagent.desktop
```

Run the latest source without reinstalling while developing:

```powershell
.\.venv\Scripts\python.exe -m openagent.workbench
```

The source launcher and installed app use the same
`%LOCALAPPDATA%\OpenAgent\data` by default. Source edits appear immediately in
the developer launch. Rerun the installer only when the installed private
environment itself should be updated.

Legacy Tk fallback:

```powershell
openagent-desktop --legacy-tk
```

Headless status check:

```powershell
openagent-desktop --status-json
```

Executable hard audit from the desktop command:

```powershell
openagent-desktop --hard-audit-json --project hard-audit
```

Executable hard audit from the dedicated hard-check command:

```powershell
openagent-hard-check --pretty --project hard-audit
```

Show the desktop workbench module manifest:

```powershell
openagent-desktop --feature-manifest-json
```

Run the no-GUI desktop feature smoke:

```powershell
openagent-desktop --feature-smoke-json --project core
```

## Desktop Features

- Local runtime status
- Local memory note creation
- Local memory search
- Local chat through the local model gateway
- Built-in agent runs
- Workspace folders, documents, notes, wiki pages, dashboards, canvases, whiteboards, mind maps, and templates
- Knowledge ingestion, hybrid search, RAG synthesis, graph, timeline, and semantic memory
- Assistant tasks, daily briefs, meeting summaries, inbox import, calendar events, and CRM records
- Local AI model registry, hardware detection, benchmarks, and GPU planning
- Voice sessions, local STT sample, local TTS WAV, voiceprint clone records, and media status
- Workflow runs, browser form automation, desktop file automation, queue jobs, and app packaging
- Privacy reports, sandbox checks, API keys, SDK artifacts, and capability ledger
- Audit log viewing
- Executable hard audit

## Privacy Boundary

OpenAgent Desktop serves the workbench from localhost and reads/writes only the configured local data directory. It does not start telemetry or call external providers unless a configured local action explicitly allows that boundary crossing.

Desktop launch, local notes, chat, agent runs, workspace actions, knowledge actions, communication actions, voice/media actions, automation actions, security/developer actions, and hard-audit actions create local records or audit events.

## Port Or Stale-UI Troubleshooting

- The supported desktop port is `8765`; routine launches do not select a new
  random port.
- `openagent-workbench --status-json` reports the data directory, runtime
  directory, asset fingerprint, and configured port without starting a server.
- Every desktop launch includes a content fingerprint in its app URL. The
  service worker caches only versioned static files. HTML navigation is always
  fetched from the live runtime, so a stopped historical port cannot display a
  cached copy that looks operational.
- Historical addresses such as `8777`, `8789`, or `8897` are unsupported. Close
  those old tabs. If a browser retained an old service worker for one of those
  origins, clearing that origin's site data removes the offline shell; it is
  not an active OpenAgent server.
- Do not run `python -m openagent`, `openagent-workbench`, and the installed
  shortcut as separate products. They are alternative launchers for one local
  runtime. The single-instance lock will keep only one desktop workbench active.

## Uninstall

Remove shortcuts and the private virtual environment while keeping local data:

```powershell
.\installers\windows\uninstall.ps1
```

Remove local data too:

```powershell
.\installers\windows\uninstall.ps1 -RemoveData
```
