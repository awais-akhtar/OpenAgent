param(
    [string]$PythonVersion = "3.12",
    [string]$BasePython = "",
    [string]$EnvironmentPath = "",
    [switch]$WithConversionTools
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$environment = if ($EnvironmentPath) {
    [System.IO.Path]::GetFullPath((Join-Path $root $EnvironmentPath))
} else {
    Join-Path $root ".accelerator-venv"
}

$workspacePython = Join-Path $root ".python312\python.exe"
if ([string]::IsNullOrWhiteSpace($BasePython) -and (Test-Path -LiteralPath $workspacePython)) {
    $BasePython = $workspacePython
}
if ([string]::IsNullOrWhiteSpace($BasePython)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python $PythonVersion is required. Pass -BasePython with a standalone interpreter path."
    }
    $resolvedBasePython = & py "-$PythonVersion" -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($resolvedBasePython)) {
        throw "Python $PythonVersion could not be started. Pass -BasePython with a standalone interpreter path."
    }
    $BasePython = $resolvedBasePython.Trim()
}
if (-not (Test-Path -LiteralPath $BasePython)) {
    throw "Base Python interpreter does not exist: $BasePython"
}

$python = Join-Path $environment "Scripts\python.exe"
$runtimeHealthy = $false
if (Test-Path -LiteralPath $python) {
    $probeOut = Join-Path $env:TEMP ("openagent-accelerator-probe-" + [guid]::NewGuid().ToString("N") + ".txt")
    $probeErr = Join-Path $env:TEMP ("openagent-accelerator-probe-" + [guid]::NewGuid().ToString("N") + ".err")
    try {
        $probe = Start-Process -FilePath $python `
            -ArgumentList @("-c", "import sys; assert sys.version_info[:2] == ($($PythonVersion.Replace('.', ',')))") `
            -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput $probeOut -RedirectStandardError $probeErr
        $runtimeHealthy = $probe.ExitCode -eq 0
    } catch {
        $runtimeHealthy = $false
    } finally {
        Remove-Item -LiteralPath $probeOut,$probeErr -Force -ErrorAction SilentlyContinue
    }
}

if (-not $runtimeHealthy) {
    $fullRoot = [System.IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
    $fullEnvironment = [System.IO.Path]::GetFullPath($environment)
    if (-not $fullEnvironment.StartsWith($fullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to replace an accelerator environment outside the OpenAgent workspace: $fullEnvironment"
    }
    if (Test-Path -LiteralPath $fullEnvironment) {
        Write-Host "Replacing an unusable accelerator environment at $fullEnvironment"
        Remove-Item -LiteralPath $fullEnvironment -Recurse -Force
    }
    Write-Host "Creating isolated accelerator environment at $environment"
    & $BasePython -m venv $environment
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Python $PythonVersion accelerator environment." }
}

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip in the accelerator environment." }

& $python -m pip install `
    "openvino==2026.2.1" `
    "openvino-tokenizers==2026.2.1" `
    "openvino-genai==2026.2.1" `
    "onnx==1.18.0" `
    "huggingface-hub==0.33.0"
if ($LASTEXITCODE -ne 0) { throw "Could not install the local accelerator runtime." }

if ($WithConversionTools) {
    & $python -m pip install "optimum-intel[openvino]"
    if ($LASTEXITCODE -ne 0) { throw "Could not install the optional model conversion tools." }
}

$env:PYTHONPATH = $root
& $python -m openagent.adaptive_engine.probe devices
if ($LASTEXITCODE -ne 0) { throw "The accelerator packages installed, but hardware discovery failed." }

Write-Host "OpenAgent accelerator runtime is ready."
Write-Host "Environment: $environment"
Write-Host "Conversion tools: $([bool]$WithConversionTools)"
Write-Host "No telemetry or model data was uploaded by the hardware probe."
