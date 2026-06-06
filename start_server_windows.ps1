param(
    [Alias("port")]
    [int]$ServerPort = $(if ($env:INDEX_TTS_PORT) { [int]$env:INDEX_TTS_PORT } elseif ($env:PORT) { [int]$env:PORT } else { 6006 }),

    [Alias("host")]
    [string]$ServerHost = $(if ($env:INDEX_TTS_HOST) { $env:INDEX_TTS_HOST } else { "0.0.0.0" }),

    [Alias("model_dir")]
    [string]$ModelDir = $(if ($env:INDEX_TTS_MODEL_DIR) { $env:INDEX_TTS_MODEL_DIR } elseif ($env:MODEL_DIR) { $env:MODEL_DIR } else { "checkpoints" }),

    [Alias("gpu_name")]
    [string]$GpuName = $(if ($env:INDEX_TTS_GPU_NAME) { $env:INDEX_TTS_GPU_NAME } else { "5070 Ti" }),

    [Alias("gpu_uuid")]
    [string]$GpuUuid = $env:INDEX_TTS_GPU_UUID,

    [Alias("use_cuda_kernel", "cuda_kernel")]
    [switch]$UseCudaKernel,

    [Alias("fp16")]
    [switch]$UseFp16,

    [switch]$VerboseMode,

    [Alias("diagnose_mode")]
    [switch]$DiagnoseMode,

    [Alias("use_deepspeed", "deepspeed")]
    [switch]$IgnoredDeepSpeed,

    [switch]$NoKillPort,
    [switch]$DryRun,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Test-TruthyEnv {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $false
    }

    return @("1", "true", "yes", "on", "y") -contains $Value.Trim().ToLowerInvariant()
}

function Get-NvidiaGpus {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw "nvidia-smi was not found in PATH. Install or repair the NVIDIA driver first."
    }

    $queryOutput = & nvidia-smi --query-gpu=index,name,pci.bus_id,uuid --format=csv,noheader
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi failed with exit code $LASTEXITCODE."
    }

    $rows = $queryOutput | ConvertFrom-Csv -Header Index, Name, PciBusId, UUID
    $gpus = @()
    foreach ($row in $rows) {
        if ([string]::IsNullOrWhiteSpace($row.UUID)) {
            continue
        }

        $gpus += [pscustomobject]@{
            Index    = $row.Index.Trim()
            Name     = $row.Name.Trim()
            PciBusId = $row.PciBusId.Trim()
            UUID     = $row.UUID.Trim()
        }
    }

    return $gpus
}

function Select-TargetGpu {
    param(
        [string]$Name,
        [string]$UUID
    )

    $gpus = @(Get-NvidiaGpus)
    if ($gpus.Count -eq 0) {
        throw "No NVIDIA GPUs were returned by nvidia-smi."
    }

    if (-not [string]::IsNullOrWhiteSpace($UUID)) {
        $matches = @($gpus | Where-Object { $_.UUID -eq $UUID -or $_.UUID.StartsWith($UUID, [System.StringComparison]::OrdinalIgnoreCase) })
        $selector = "UUID '$UUID'"
    }
    else {
        $matches = @($gpus | Where-Object { $_.Name.IndexOf($Name, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 })
        $selector = "name containing '$Name'"
    }

    if ($matches.Count -eq 0) {
        Write-Err "No NVIDIA GPU matched $selector. Available GPUs:"
        foreach ($gpu in $gpus) {
            Write-Host ("  {0}: {1} ({2}, {3})" -f $gpu.Index, $gpu.Name, $gpu.PciBusId, $gpu.UUID)
        }
        exit 1
    }

    if ($matches.Count -gt 1) {
        Write-Err "Multiple NVIDIA GPUs matched $selector. Set -GpuUuid to choose one:"
        foreach ($gpu in $matches) {
            Write-Host ("  {0}: {1} ({2}, {3})" -f $gpu.Index, $gpu.Name, $gpu.PciBusId, $gpu.UUID)
        }
        exit 1
    }

    return $matches[0]
}

function Stop-ListenersOnPort {
    param([int]$Port)

    Write-Info "Checking port $Port..."
    $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -eq 0) {
        Write-Info "Port $Port is free."
        return
    }

    $processIds = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
    foreach ($listenerProcessId in $processIds) {
        if ($listenerProcessId -eq 0) {
            continue
        }

        $process = Get-Process -Id $listenerProcessId -ErrorAction SilentlyContinue
        if (-not $process) {
            continue
        }

        Write-Warn ("Port {0} is used by PID {1} ({2}); stopping it..." -f $Port, $listenerProcessId, $process.ProcessName)
        Stop-Process -Id $listenerProcessId -Force
    }

    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 250
        $remaining = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
        if ($remaining.Count -eq 0) {
            Write-Info "Port $Port has been released."
            return
        }
    }

    throw "Port $Port is still in use after stopping listeners."
}

function Test-RequiredFiles {
    param([string]$Root)

    Write-Info "Checking required files..."
    $requiredFiles = @(
        "api_server.py",
        "indextts\infer_v2_subtitle.py"
    )

    foreach ($file in $requiredFiles) {
        $path = Join-Path $Root $file
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required file not found: $path"
        }
    }

    $modelPath = Join-Path $Root $ModelDir
    if (-not (Test-Path -LiteralPath $modelPath -PathType Container)) {
        Write-Warn "Model directory was not found: $modelPath"
    }
}

function Remove-WindowsUnsupportedArgs {
    param([string[]]$Args)

    $filtered = @()
    foreach ($arg in @($Args)) {
        if ([string]::IsNullOrWhiteSpace($arg)) {
            continue
        }

        if ($arg -match '^--(use_)?deepspeed($|=)' -or $arg -eq "--no-deepspeed") {
            Write-Warn "Ignoring unsupported Windows DeepSpeed argument: $arg"
            continue
        }

        if ($arg -eq "--cuda_kernel") {
            Write-Warn "Translating --cuda_kernel to api_server.py argument --use_cuda_kernel."
            $filtered += "--use_cuda_kernel"
            continue
        }

        if ($arg -eq "--no-cuda_kernel") {
            Write-Warn "Ignoring unsupported argument: $arg"
            continue
        }

        $filtered += $arg
    }

    return $filtered
}

if (Test-TruthyEnv $env:USE_CUDA_KERNEL) {
    $UseCudaKernel = $true
}
if (Test-TruthyEnv $env:USE_FP16) {
    $UseFp16 = $true
}
if (Test-TruthyEnv $env:USE_VERBOSE) {
    $VerboseMode = $true
}
if ($PSBoundParameters.ContainsKey("Verbose")) {
    $VerboseMode = $true
}
if (Test-TruthyEnv $env:DIAGNOSE_MODE) {
    $DiagnoseMode = $true
}
if ($IgnoredDeepSpeed -or (Test-TruthyEnv $env:USE_DEEPSPEED)) {
    Write-Warn "DeepSpeed is disabled by this Windows launcher and will not be passed to api_server.py."
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $scriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Info "======================================"
Write-Info "IndexTTS api_server Windows launcher"
Write-Info "======================================"
Write-Info "Project root: $scriptRoot"
Write-Info "Host: $ServerHost"
Write-Info "Port: $ServerPort"
Write-Info "Model dir: $ModelDir"

Test-RequiredFiles -Root $scriptRoot

$targetGpu = Select-TargetGpu -Name $GpuName -UUID $GpuUuid
$env:CUDA_VISIBLE_DEVICES = $targetGpu.UUID
Write-Info ("Using NVIDIA GPU {0}: {1} ({2}, {3})" -f $targetGpu.Index, $targetGpu.Name, $targetGpu.PciBusId, $targetGpu.UUID)
Write-Info "CUDA_VISIBLE_DEVICES was set to the selected GPU UUID; PyTorch will see it as cuda:0."

if (-not $NoKillPort -and -not $DryRun) {
    Stop-ListenersOnPort -Port $ServerPort
}
elseif ($NoKillPort) {
    Write-Warn "Skipping port cleanup because -NoKillPort was set."
}

$launchArgs = @(
    "api_server.py",
    "--host", $ServerHost,
    "--port", [string]$ServerPort,
    "--model_dir", $ModelDir
)

if ($UseCudaKernel) {
    $launchArgs += "--use_cuda_kernel"
}
if ($UseFp16) {
    $launchArgs += "--fp16"
}
if ($VerboseMode) {
    $launchArgs += "--verbose"
}
if ($DiagnoseMode) {
    $launchArgs += "--diagnose_mode"
}

$launchArgs += @(Remove-WindowsUnsupportedArgs -Args $ExtraArgs)

if (Get-Command uv -ErrorAction SilentlyContinue) {
    $command = "uv"
    $commandArgs = @("run") + $launchArgs
}
else {
    Write-Warn "uv was not found in PATH; falling back to python."
    $command = "python"
    $commandArgs = $launchArgs
}

Write-Info "Launch command: $command $($commandArgs -join ' ')"
if ($DryRun) {
    Write-Info "Dry run complete. api_server.py was not started."
    exit 0
}

Write-Info "Starting api_server.py. Press Ctrl+C to stop."
& $command @commandArgs
exit $LASTEXITCODE
