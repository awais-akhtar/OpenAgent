param(
    [string]$InstallDir = "$env:LOCALAPPDATA\OpenAgent",
    [switch]$RemoveData
)

$ErrorActionPreference = "Stop"

$programsShortcut = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\OpenAgent.lnk"
$desktopShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "OpenAgent.lnk"

foreach ($shortcut in @($programsShortcut, $desktopShortcut)) {
    if (Test-Path -LiteralPath $shortcut) {
        Remove-Item -LiteralPath $shortcut -Force
    }
}

$venv = Join-Path $InstallDir ".venv"
$launcher = Join-Path $InstallDir "OpenAgent.cmd"
$dataDir = Join-Path $InstallDir "data"
$runtimeDir = Join-Path $InstallDir "runtime"
$statePath = Join-Path $runtimeDir "workbench-state.json"
$installManifest = Join-Path $InstallDir "install-manifest.json"

if (Test-Path -LiteralPath $statePath) {
    try {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $process = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
        $stateData = [IO.Path]::GetFullPath([string]$state.data_dir)
        $expectedData = [IO.Path]::GetFullPath($dataDir)
        $startedAt = [double]$state.started_at_unix
        $processStartedAt = if ($process) { [DateTimeOffset]($process.StartTime.ToUniversalTime()) } else { $null }
        $startMatches = $processStartedAt -and ([Math]::Abs($processStartedAt.ToUnixTimeSeconds() - $startedAt) -le 5)
        if ($process -and $stateData.Equals($expectedData, [StringComparison]::OrdinalIgnoreCase) -and $startMatches) {
            Stop-Process -Id $process.Id -Force
            $process.WaitForExit(5000)
        }
    } catch {
        Write-Warning "The prior OpenAgent process state could not be verified. Close the desktop app before retrying if uninstall reports a locked file."
    }
}

if (Test-Path -LiteralPath $venv) {
    Remove-Item -LiteralPath $venv -Recurse -Force
}

if (Test-Path -LiteralPath $launcher) {
    Remove-Item -LiteralPath $launcher -Force
}

if (Test-Path -LiteralPath $installManifest) {
    Remove-Item -LiteralPath $installManifest -Force
}

if (Test-Path -LiteralPath $runtimeDir) {
    Remove-Item -LiteralPath $runtimeDir -Recurse -Force
}

if ($RemoveData) {
    if (Test-Path -LiteralPath $InstallDir) {
        Remove-Item -LiteralPath $InstallDir -Recurse -Force
    }
    Write-Host "OpenAgent Desktop uninstalled and local data removed."
} else {
    Write-Host "OpenAgent Desktop uninstalled. Local data was kept at:"
    Write-Host "  $(Join-Path $InstallDir "data")"
    Write-Host "Run with -RemoveData only when you intentionally want to delete local OpenAgent data."
}
