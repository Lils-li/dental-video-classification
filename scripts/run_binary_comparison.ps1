# LEGACY LAUNCHER: historical settings; review paths, deadlines and hashes before use.
# Run manually in a standalone PowerShell process; no Codex connection is needed.
# Existing configurations, source, datasets, and results are never edited.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ExperimentRoot = Split-Path -Parent $PSScriptRoot
$PythonExecutable = Join-Path $ExperimentRoot '.venv\Scripts\python.exe'
$RunnerPath = Join-Path $ExperimentRoot 'src\run_binary_comparison.py'
$ResultsRoot = Join-Path $ExperimentRoot 'outputs\runs\binary_comparison_2026-10-14'
$ExperimentJobs = @(
    [pscustomobject]@{
        Name = 'original'
        Config = Join-Path $ExperimentRoot 'configs\binary_comparison_2026-10-14\original.json'
        RunDirectory = Join-Path $ResultsRoot 'original_81_seed42'
    }
    [pscustomobject]@{
        Name = 'cleaned'
        Config = Join-Path $ExperimentRoot 'configs\binary_comparison_2026-10-14\cleaned.json'
        RunDirectory = Join-Path $ResultsRoot 'cleaned_53_seed42'
    }
)

function Invoke-Experiment {
    param($ExperimentJob, [string]$LogDirectory)
    $ConsoleLog = Join-Path $LogDirectory ($ExperimentJob.Name + '.console.log')
    if (Test-Path -LiteralPath $ConsoleLog) {
        throw "Refusing to overwrite console log: $ConsoleLog"
    }
    if (Test-Path -LiteralPath $ExperimentJob.RunDirectory) {
        throw "Refusing to reuse an existing run: $($ExperimentJob.RunDirectory)"
    }
    $PythonArguments = @(
        '-B', '-u', $RunnerPath,
        '--config', $ExperimentJob.Config,
        '--run-dir', $ExperimentJob.RunDirectory,
        '--approve-full-training'
    )
    Write-Host "Starting $($ExperimentJob.Name) at $(Get-Date -Format o)"
    Write-Host "Console log: $ConsoleLog"
    Write-Host "Run directory: $($ExperimentJob.RunDirectory)"

    # Windows PowerShell can turn native stderr into ErrorRecord objects.
    # Continue captures warnings/tracebacks without treating stderr alone as failure.
    # Tee failures still terminate; the actual Python exit code decides success.
    $SavedErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $PythonExecutable @PythonArguments 2>&1 |
            Tee-Object -FilePath $ConsoleLog -ErrorAction Stop
        $PythonExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $SavedErrorPreference
    }
    if ($PythonExitCode -ne 0) {
        throw "$($ExperimentJob.Name) failed with Python exit code $PythonExitCode. See $ConsoleLog. No subsequent experiment will start."
    }
    $CompletionPath = Join-Path $ExperimentJob.RunDirectory 'completed.json'
    if (-not (Test-Path -LiteralPath $CompletionPath -PathType Leaf)) {
        throw "$($ExperimentJob.Name) exited without its completion marker. No subsequent experiment will start."
    }
    $Completion = Get-Content -LiteralPath $CompletionPath -Raw | ConvertFrom-Json
    $Config = Get-Content -LiteralPath $ExperimentJob.Config -Raw | ConvertFrom-Json
    if ($Completion.smoke_only -ne $false -or $Completion.test_used -ne $false -or $Completion.epochs -ne $Config.epochs) {
        throw "$($ExperimentJob.Name) has an invalid completion marker. No subsequent experiment will start."
    }
    Write-Host "Completed $($ExperimentJob.Name) at $(Get-Date -Format o)"
}

$OriginalLocation = Get-Location
try {
    # Preflight both experiments before creating logs or starting Python.
    foreach ($RequiredPath in @($PythonExecutable, $RunnerPath, $ExperimentJobs[0].Config, $ExperimentJobs[1].Config)) {
        if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
            throw "Required file is missing: $RequiredPath"
        }
    }
    foreach ($ExperimentJob in $ExperimentJobs) {
        if (Test-Path -LiteralPath $ExperimentJob.RunDirectory) {
            throw "Run already exists; nothing will be overwritten: $($ExperimentJob.RunDirectory)"
        }
    }
    $LogIdentifier = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
    $LogDirectory = Join-Path $ResultsRoot ('launcher_logs_' + $LogIdentifier)
    if (Test-Path -LiteralPath $LogDirectory) {
        throw "Log directory already exists: $LogDirectory"
    }
    New-Item -ItemType Directory -Path $LogDirectory | Out-Null
    $LaunchManifest = [ordered]@{
        started = Get-Date -Format o
        interpreter = $PythonExecutable
        runner_sha256 = (Get-FileHash -LiteralPath $RunnerPath -Algorithm SHA256).Hash
        original_config_sha256 = (Get-FileHash -LiteralPath $ExperimentJobs[0].Config -Algorithm SHA256).Hash
        cleaned_config_sha256 = (Get-FileHash -LiteralPath $ExperimentJobs[1].Config -Algorithm SHA256).Hash
        order = @('original', 'cleaned')
        logs = $LogDirectory
    }
    $LaunchManifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $LogDirectory 'launch.json') -Encoding UTF8
    Set-Location -LiteralPath $ExperimentRoot
    Write-Host 'Estimated sequential runtime: approximately 3-5 hours.'
    Write-Host 'Keep this PowerShell window open and prevent sleep; locking the session is fine.'
    Invoke-Experiment -ExperimentJob $ExperimentJobs[0] -LogDirectory $LogDirectory
    Invoke-Experiment -ExperimentJob $ExperimentJobs[1] -LogDirectory $LogDirectory
    Write-Host "Both experiments completed successfully. Logs: $LogDirectory"
}
catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
finally {
    Set-Location -LiteralPath $OriginalLocation.Path
}
exit 0
