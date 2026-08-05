param(
    [Parameter(Mandatory = $true)][string]$Model,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [ValidateSet("fp16", "int8", "int4", "mxfp4", "nf4", "cb4")][string]$WeightFormat = "int4",
    [string]$Task = "text-generation-with-past",
    [switch]$AllowNetwork,
    [switch]$TrustRemoteCode
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $root ".accelerator-venv\Scripts\python.exe"
$optimum = Join-Path $root ".accelerator-venv\Scripts\optimum-cli.exe"

if (-not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath $optimum)) {
    throw "Conversion tools are missing. Run .\installers\windows\setup-accelerator.ps1 -WithConversionTools first."
}

$localModel = Test-Path -LiteralPath $Model
if (-not $localModel -and -not $AllowNetwork) {
    throw "Model is not a local path. Pass -AllowNetwork to authorize a remote model download."
}
if ($TrustRemoteCode -and -not $AllowNetwork -and -not $localModel) {
    throw "Remote custom code requires both -AllowNetwork and -TrustRemoteCode."
}

$output = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $output -Force | Out-Null
$arguments = @(
    "export", "openvino", "--model", $Model,
    "--task", $Task,
    "--weight-format", $WeightFormat
)
if ($TrustRemoteCode) { $arguments += "--trust-remote-code" }
$arguments += $output

Write-Host "Converting model to OpenVINO IR ($WeightFormat)..."
& $optimum @arguments
if ($LASTEXITCODE -ne 0) { throw "OpenVINO model conversion failed." }

$modelXml = Get-ChildItem -LiteralPath $output -Filter "*.xml" -File | Select-Object -First 1
if (-not $modelXml) { throw "Conversion completed without an OpenVINO XML graph." }
$modelBin = [System.IO.Path]::ChangeExtension($modelXml.FullName, ".bin")
if (-not (Test-Path -LiteralPath $modelBin)) { throw "The converted graph is missing its BIN weights file." }

$manifest = [ordered]@{
    format = "openvino-genai"
    source = if ($localModel) { [System.IO.Path]::GetFullPath($Model) } else { $Model }
    task = $Task
    weight_format = $WeightFormat
    graph = $modelXml.Name
    graph_sha256 = (Get-FileHash -LiteralPath $modelXml.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    weights = [System.IO.Path]::GetFileName($modelBin)
    weights_sha256 = (Get-FileHash -LiteralPath $modelBin -Algorithm SHA256).Hash.ToLowerInvariant()
    network_authorized = [bool]$AllowNetwork
    remote_code_authorized = [bool]$TrustRemoteCode
    created_at = [DateTimeOffset]::UtcNow.ToString("o")
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $output ".openagent-conversion.json") -Encoding utf8

$env:PYTHONPATH = $root
& $python -m openagent.adaptive_engine.probe compile --model $modelXml.FullName --device CPU --cache-dir (Join-Path $output ".compile-cache")
if ($LASTEXITCODE -ne 0) { throw "The converted model failed the local CPU compiler verification." }

Write-Host "Converted and compiler-verified model: $output"
Write-Host "Load this directory as an OpenVINO GenAI model in Developer > Adaptive execution engine."
