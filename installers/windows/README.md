# OpenAgent Windows Desktop Installer

This installer creates a local Windows desktop app entry for OpenAgent. The modern workbench opens in an isolated Edge app-mode window, not a normal browser tab. It installs the Python package into a private virtual environment and creates Start Menu and Desktop shortcuts for `openagent-desktop`.

## Install

Open PowerShell in the repository root and run:

```powershell
.\installers\windows\install.ps1
```

Requirements:

- Python 3.11 or newer
- Microsoft Edge or another supported Chromium app-mode browser for the modern window
- A Python build with `tkinter` only when the optional `--legacy-tk` fallback is used

The default install location is:

```text
%LOCALAPPDATA%\OpenAgent
```

The default local data directory is:

```text
%LOCALAPPDATA%\OpenAgent\data
```

Installs to a custom `-InstallDir` do not change the user's normal Start Menu or Desktop shortcuts. To create shortcuts for an intentional alternate install, pass `-CreateShortcutsForCustomInstall` explicitly.

The installer uses the local source tree by default and installs the voice/STT extras so microphone transcription can use local Vosk or faster-whisper engines when configured.

It does not upgrade `pip` unless you explicitly pass:

```powershell
.\installers\windows\install.ps1 -UpgradePip
```

For a minimal install without voice extras:

```powershell
.\installers\windows\install.ps1 -SkipVoiceExtras
```

## Launch

Use the Start Menu shortcut, Desktop shortcut, or run:

```powershell
%LOCALAPPDATA%\OpenAgent\OpenAgent.cmd
```

The launcher owns one local runtime at `127.0.0.1:8765`. Reopening the shortcut focuses the existing instance instead of creating another port. Closing the app window closes that runtime. Source and installed launches share `%LOCALAPPDATA%\OpenAgent\data` and the same user-level instance lock unless a developer explicitly overrides the data or runtime directory.

## Uninstall

To remove shortcuts and the private virtual environment while keeping local data:

```powershell
.\installers\windows\uninstall.ps1
```

To also remove local OpenAgent data:

```powershell
.\installers\windows\uninstall.ps1 -RemoveData
```

## Privacy Boundary

OpenAgent Desktop reads and writes only the configured local data directory and serves its packaged UI on localhost inside the app-mode window. It does not start telemetry or call external providers unless a configured local action explicitly allows that boundary crossing.
