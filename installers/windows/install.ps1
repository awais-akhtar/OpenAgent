param(
    [string]$SourcePath = "",
    [string]$InstallDir = "$env:LOCALAPPDATA\OpenAgent",
    [switch]$NoDesktopShortcut,
    [switch]$UpgradePip,
    [switch]$SkipVoiceExtras
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($SourcePath)) {
    $SourcePath = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
} else {
    $SourcePath = (Resolve-Path $SourcePath).Path
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python 3.11 or newer is required. Install Python, then run this installer again."
}

$pythonExe = if ($python.Path) { $python.Path } elseif ($python.Source) { $python.Source } else { "python" }
$version = & $pythonExe -c "import json,sys; print(json.dumps({'major': sys.version_info[0], 'minor': sys.version_info[1], 'version': sys.version.split()[0]}))" | ConvertFrom-Json
if (($version.major -lt 3) -or (($version.major -eq 3) -and ($version.minor -lt 11))) {
    throw "Python 3.11 or newer is required. Found $($version.version)."
}

# Production UI is the approved second React frontend, not the legacy web/ bundle.
# A successful Python package install alone is NOT sufficient for the desktop UI.
$frontendSource = Join-Path $SourcePath "frontend-second"
$frontendLock = Join-Path $frontendSource "package-lock.json"
if (-not (Test-Path -LiteralPath $frontendLock)) {
    throw "The approved frontend source is missing: $frontendLock. Use the complete repository."
}
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npm) { $npm = Get-Command npm -ErrorAction SilentlyContinue }
if (-not $npm) {
    throw "Node.js/npm is required to build the approved desktop UI. Install Node.js LTS first."
}
Push-Location $frontendSource
try {
    & $npm.Source ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw "Failed to install locked frontend dependencies (npm ci)." }
    & $npm.Source run build
    if ($LASTEXITCODE -ne 0) { throw "The approved frontend build failed. Installation stopped to avoid an outdated UI." }
} finally {
    Pop-Location
}
$builtUI = Join-Path $frontendSource "dist"
if (-not (Test-Path -LiteralPath (Join-Path $builtUI "index.html"))) {
    throw "The frontend build did not produce dist/index.html; refusing to install the legacy UI."
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$venv = Join-Path $InstallDir ".venv"
$dataDir = Join-Path $InstallDir "data"

& $pythonExe -m venv $venv
$venvPython = Join-Path $venv "Scripts\python.exe"
$venvPythonw = Join-Path $venv "Scripts\pythonw.exe"
$desktopExe = Join-Path $venv "Scripts\openagent-desktop.exe"

if ($UpgradePip) {
    & $venvPython -m pip install --upgrade pip
}
$installTarget = if ($SkipVoiceExtras) { "$SourcePath[media,rag]" } else { "$SourcePath[full]" }
& $venvPython -m pip install $installTarget
if ($LASTEXITCODE -ne 0) { throw "OpenAgent Python package installation failed." }

# Stage and validate complete built assets, then atomically select the new UI.
# Keep one previous UI for rollback; do not modify the existing data directory.
$uiRoot = Join-Path $InstallDir "ui"
$uiStaging = Join-Path $InstallDir "ui.staging"
$uiRollback = Join-Path $InstallDir "ui.previous"
if (Test-Path -LiteralPath $uiStaging) { Remove-Item -LiteralPath $uiStaging -Recurse -Force }
New-Item -ItemType Directory -Force -Path $uiStaging | Out-Null
Copy-Item -Path (Join-Path $builtUI "*") -Destination $uiStaging -Recurse -Force
& $venvPython -c "import sys; from pathlib import Path; from openagent.workbench import validate_desktop_ui; validate_desktop_ui(Path(sys.argv[1]))" $uiStaging
if ($LASTEXITCODE -ne 0) { throw "Staged frontend assets failed startup validation. Existing UI is unchanged." }
if (Test-Path -LiteralPath $uiRollback) { Remove-Item -LiteralPath $uiRollback -Recurse -Force }
if (Test-Path -LiteralPath $uiRoot) { Move-Item -LiteralPath $uiRoot -Destination $uiRollback }
try {
    Move-Item -LiteralPath $uiStaging -Destination $uiRoot
} catch {
    if (-not (Test-Path -LiteralPath $uiRoot) -and (Test-Path -LiteralPath $uiRollback)) {
        Move-Item -LiteralPath $uiRollback -Destination $uiRoot
    }
    throw
}

New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

$packageVersion = & $venvPython -c "import openagent; print(openagent.__version__)"
$installManifest = @{
    product = "OpenAgent"
    version = $packageVersion.Trim()
    python = $version.version
    source = $SourcePath
    installed_at = [DateTimeOffset]::UtcNow.ToString("o")
    data_dir = $dataDir
    ui_root = $uiRoot
    ui_source = "frontend-second/dist"
} | ConvertTo-Json
$installManifest | Set-Content -Encoding UTF8 -Path (Join-Path $InstallDir "install-manifest.json")

$launcher = Join-Path $InstallDir "OpenAgent.cmd"
$launcherContent = @"
@echo off
"$desktopExe" --data-dir "$dataDir" --ui-root "$uiRoot" %*
"@
$launcherContent | Set-Content -Encoding ASCII -Path $launcher

function New-OpenAgentShortcut {
    param(
        [string]$ShortcutPath
    )
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $venvPythonw
    $shortcut.Arguments = "-m openagent.desktop --data-dir `"$dataDir`" --ui-root `"$uiRoot`""
    $shortcut.WorkingDirectory = $InstallDir
    $shortcut.WindowStyle = 7
    $shortcut.Description = "OpenAgent native local desktop"
    $shortcut.Save()
}

$programsDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
New-Item -ItemType Directory -Force -Path $programsDir | Out-Null
New-OpenAgentShortcut -ShortcutPath (Join-Path $programsDir "OpenAgent.lnk")

if (-not $NoDesktopShortcut) {
    $desktopDir = [Environment]::GetFolderPath("Desktop")
    New-OpenAgentShortcut -ShortcutPath (Join-Path $desktopDir "OpenAgent.lnk")
}

Write-Host "OpenAgent Desktop installed."
Write-Host "Launch it from the Start Menu, Desktop shortcut, or:"
Write-Host "  $launcher"
Write-Host "Local data directory:"
Write-Host "  $dataDir"
Write-Host "Installed version: $($packageVersion.Trim())"
Write-Host "Approved UI: $uiRoot"
