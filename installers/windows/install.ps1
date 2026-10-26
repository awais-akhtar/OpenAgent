param(
    [string]$SourcePath = "",
    [string]$InstallDir = "$env:LOCALAPPDATA\OpenAgent",
    [switch]$NoDesktopShortcut,
    [switch]$CreateShortcutsForCustomInstall,
    [switch]$UpgradePip,
    [switch]$SkipVoiceExtras
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($SourcePath)) {
    $SourcePath = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
} else {
    $SourcePath = (Resolve-Path $SourcePath).Path
}

$normalizedInstallRoot = [System.IO.Path]::GetFullPath($InstallDir).TrimEnd('\')
$managedRuntimeRoot = [System.IO.Path]::GetFullPath((Join-Path $normalizedInstallRoot ".venv\Scripts")).TrimEnd('\') + '\'
$activeManagedProcessIds = @()
foreach ($process in Get-Process -ErrorAction SilentlyContinue) {
    try { $processPath = $process.Path } catch { continue }
    if ([string]::IsNullOrWhiteSpace($processPath)) { continue }
    try { $normalizedProcessPath = [System.IO.Path]::GetFullPath($processPath) } catch { continue }
    if ($normalizedProcessPath.StartsWith($managedRuntimeRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        $activeManagedProcessIds += $process.Id
    }
}
if ($activeManagedProcessIds.Count -gt 0) {
    $processList = ($activeManagedProcessIds | Sort-Object -Unique) -join ", "
    throw "OpenAgent is running from the target installation (process IDs: $processList). Close its desktop window and local server before upgrading; user data is untouched."
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
$tempRoot = [System.IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') + '\'
$frontendBuildRoot = [System.IO.Path]::GetFullPath((Join-Path $env:TEMP ("OpenAgent-frontend-build-" + [Guid]::NewGuid().ToString("N"))))
if (-not $frontendBuildRoot.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to build the frontend outside the temporary directory: $frontendBuildRoot"
}
$frontendBuild = Join-Path $frontendBuildRoot "frontend-second"
$frontendBuildAssets = Join-Path $frontendBuild "dist"
$frontendNpmCache = Join-Path $frontendBuildRoot "npm-cache"
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npm) { $npm = Get-Command npm -ErrorAction SilentlyContinue }
if (-not $npm) {
    throw "Node.js/npm is required to build the approved desktop UI. Install Node.js LTS first."
}
try {
    New-Item -ItemType Directory -Force -Path $frontendBuild | Out-Null
    foreach ($relativeFile in @("package.json", "package-lock.json", "tsconfig.json", "index.html", "vite.config.ts")) {
        Copy-Item -LiteralPath (Join-Path $frontendSource $relativeFile) -Destination $frontendBuild
    }
    foreach ($relativeDirectory in @("src", "public", "scripts", "reference")) {
        Copy-Item -LiteralPath (Join-Path $frontendSource $relativeDirectory) -Destination $frontendBuild -Recurse
    }
    Push-Location $frontendBuild
    try {
        # Do not mutate or contend with a developer's shared npm cache while
        # upgrading the desktop from a source checkout.
        $previousNpmCache = $env:npm_config_cache
        $env:npm_config_cache = $frontendNpmCache
        & $npm.Source ci --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { throw "Failed to install locked frontend dependencies (npm ci)." }
        & $npm.Source run build
        if ($LASTEXITCODE -ne 0) { throw "The approved frontend build failed. Installation stopped to avoid an outdated UI." }
    } finally {
        $env:npm_config_cache = $previousNpmCache
        Pop-Location
    }
} catch {
    if (Test-Path -LiteralPath $frontendBuildRoot) {
        Remove-Item -LiteralPath $frontendBuildRoot -Recurse -Force
    }
    throw
}
$builtUI = $frontendBuildAssets
if (-not (Test-Path -LiteralPath (Join-Path $builtUI "index.html"))) {
    if (Test-Path -LiteralPath $frontendBuildRoot) { Remove-Item -LiteralPath $frontendBuildRoot -Recurse -Force }
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
Remove-Item -LiteralPath $frontendBuildRoot -Recurse -Force
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
    $expectedPythonw = (Resolve-Path -LiteralPath $venvPythonw).Path
    $shortcut.TargetPath = $expectedPythonw
    $shortcut.Arguments = "-m openagent.desktop --data-dir `"$dataDir`" --ui-root `"$uiRoot`""
    $shortcut.WorkingDirectory = $InstallDir
    $shortcut.WindowStyle = 7
    $shortcut.Description = "OpenAgent native local desktop"
    $shortcut.Save()

    # Verify the saved link itself, not only the source path. This catches
    # profile redirection and stale target paths that can make an installed
    # shortcut fail even though the UI bundle and Python package are present.
    $savedShortcut = $shell.CreateShortcut($ShortcutPath)
    if (-not (Test-Path -LiteralPath $savedShortcut.TargetPath) -or
        [System.IO.Path]::GetFullPath($savedShortcut.TargetPath) -ne $expectedPythonw) {
        throw "The OpenAgent shortcut target is invalid: $($savedShortcut.TargetPath)"
    }
}

# A developer/test install at a custom path must not replace the user's normal
# OpenAgent shortcuts with links to a temporary data directory. Custom installs
# create shortcuts only after an explicit opt-in.
$normalizedInstallDir = [System.IO.Path]::GetFullPath($InstallDir).TrimEnd('\')
$defaultInstallDir = [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "OpenAgent")).TrimEnd('\')
$isDefaultInstall = [string]::Equals(
    $normalizedInstallDir,
    $defaultInstallDir,
    [System.StringComparison]::OrdinalIgnoreCase
)
$shouldCreateShortcuts = $isDefaultInstall -or $CreateShortcutsForCustomInstall
if ($shouldCreateShortcuts) {
    # An elevated installer can have LOCALAPPDATA and APPDATA from different
    # profiles. For the standard per-user destination, derive the Start Menu
    # path from InstallDir rather than mixing profiles.
$defaultInstallSuffix = "\AppData\Local\OpenAgent"
if ($isDefaultInstall -and $normalizedInstallDir.EndsWith($defaultInstallSuffix, [System.StringComparison]::OrdinalIgnoreCase)) {
    # Use the profile encoded by the actual destination, even if elevation
    # changed the installer process's APPDATA/LOCALAPPDATA environment.
    $userProfileRoot = $normalizedInstallDir.Substring(0, $normalizedInstallDir.Length - $defaultInstallSuffix.Length)
    $programsDir = Join-Path $userProfileRoot "AppData\Roaming\Microsoft\Windows\Start Menu\Programs"
} else {
    $programsDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
}
New-Item -ItemType Directory -Force -Path $programsDir | Out-Null
New-OpenAgentShortcut -ShortcutPath (Join-Path $programsDir "OpenAgent.lnk")

if (-not $NoDesktopShortcut) {
    $desktopDir = [Environment]::GetFolderPath("Desktop")
    New-OpenAgentShortcut -ShortcutPath (Join-Path $desktopDir "OpenAgent.lnk")
}
} else {
    Write-Host "Custom install path detected; existing Start Menu and Desktop shortcuts were left unchanged. Pass -CreateShortcutsForCustomInstall to opt in."
}

Write-Host "OpenAgent Desktop installed."
Write-Host "Launch it from the Start Menu, Desktop shortcut, or:"
Write-Host "  $launcher"
Write-Host "Local data directory:"
Write-Host "  $dataDir"
Write-Host "Installed version: $($packageVersion.Trim())"
Write-Host "Approved UI: $uiRoot"
