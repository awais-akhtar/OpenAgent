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

New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

$packageVersion = & $venvPython -c "import openagent; print(openagent.__version__)"
$installManifest = @{
    product = "OpenAgent"
    version = $packageVersion.Trim()
    python = $version.version
    source = $SourcePath
    installed_at = [DateTimeOffset]::UtcNow.ToString("o")
    data_dir = $dataDir
} | ConvertTo-Json
$installManifest | Set-Content -Encoding UTF8 -Path (Join-Path $InstallDir "install-manifest.json")

$launcher = Join-Path $InstallDir "OpenAgent.cmd"
$launcherContent = @"
@echo off
"$desktopExe" --data-dir "$dataDir" %*
"@
$launcherContent | Set-Content -Encoding ASCII -Path $launcher

function New-OpenAgentShortcut {
    param(
        [string]$ShortcutPath
    )
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $venvPythonw
    $shortcut.Arguments = "-m openagent.desktop --data-dir `"$dataDir`""
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
