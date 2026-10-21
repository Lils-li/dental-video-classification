# LEGACY LAUNCHER: historical settings; review paths, deadlines and hashes before use.
# Standalone overnight pipeline. Launch manually; this script never prompts.
# No security/power changes, installs, Git operations, test access, or adaptive parameters.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ExperimentRoot = Split-Path -Parent $PSScriptRoot
$PythonExecutable = Join-Path $ExperimentRoot '.venv\Scripts\python.exe'
$Runner = Join-Path $ExperimentRoot 'src\run_binary_comparison.py'
$Helper = Join-Path $ExperimentRoot 'src\overnight_binary_analysis.py'
$OriginalConfig = Join-Path $ExperimentRoot 'configs\binary_comparison_2026-10-14\original.json'
$CleanedConfig = Join-Path $ExperimentRoot 'configs\binary_comparison_2026-10-14\cleaned.json'
$LaunchIdentifier = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8)
$PipelineRoot = Join-Path $ExperimentRoot ('outputs\runs\overnight_' + $LaunchIdentifier)
$Logs = Join-Path $PipelineRoot 'logs'
$OriginalRun = Join-Path $PipelineRoot 'original_81_seed42'
$CleanedRun = Join-Path $PipelineRoot 'cleaned_53_seed42'
$Analysis = Join-Path $PipelineRoot 'analysis'
$Occlusion = Join-Path $PipelineRoot 'occlusion'
$Report = Join-Path $PipelineRoot 'report'
$CurrentStage = 'preflight'
$PipelineClock = [Diagnostics.Stopwatch]::StartNew()
$OriginalLocation = Get-Location
# These hashes lock the previously approved source, configurations, and inputs.
$ExpectedHashes = [ordered]@{
    'configs\binary_comparison_2026-10-14\original.json' = '0AF5601C59A1E9B8F0F2695E83E42F59E5E0AAF23BB824FAE98A3DD62773B58C'
    'configs\binary_comparison_2026-10-14\cleaned.json' = '4633E79DF7F20B5AA35C735E7AF07CC1DB1C3010F188F3EAD7DFA502D05B89BA'
    'src\run_binary_comparison.py' = 'A004F5B358E0F7F72BA8B79F56899546084847C71B8F080343767A6273BC23AB'
    'src\data\dataset.py' = 'AD84A107C6A93123A7DCF0869EEB0DCA6947EB377DB027A4EC7105A4BB1DD292'
    'data\labels\quadrant_metadata.csv' = '34961A06F9F9C2A7DD6330C8C389D218E9DA961E2883FD5E934D86B3572429F3'
    'data\labels\binary_clean_metadata.csv' = 'F450328DA49D0AE34ACC7A67FEFB46897EE51A8790C41D480B96AAF6B91E58F8'
    'outputs\data_quality\manual_segment_audit_colored.xlsx' = '7DE378AC49E3DEAD09C0471B8D5A2308C08D809EA017E346DFBD1FDEFE995B56'
    'src\overnight_binary_analysis.py' = 'FECF4D683281C4B93E79B5BEC818951FD8C3A4B631B69C5A69AC12D608A409DB'
}

function Save-NewJson {
    param([string]$Path, $Value)
    $Stream = [IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try {
        $Writer = New-Object IO.StreamWriter($Stream,(New-Object Text.UTF8Encoding($false)))
        try { $Writer.WriteLine(($Value | ConvertTo-Json -Depth 10)) }
        finally { $Writer.Dispose() }
    }
    finally { $Stream.Dispose() }
}

function Confirm-Inputs {
    foreach ($RelativePath in $ExpectedHashes.Keys) {
        $Source = Join-Path $ExperimentRoot $RelativePath
        if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) { throw "Missing required input: $Source" }
        if ((Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash -ne $ExpectedHashes[$RelativePath]) {
            throw "Approved input changed: $Source. Refusing to change or reinterpret the experiment."
        }
    }
}

function Invoke-PythonStage {
    param([string]$Name,[string[]]$PythonArguments)
    $Log = Join-Path $Logs ($Name + '.console.log')
    if (Test-Path -LiteralPath $Log) { throw "Console log already exists: $Log" }
    Save-NewJson -Path (Join-Path $Logs ($Name + '.started.json')) -Value @{stage=$Name;time=(Get-Date -Format o);arguments=$PythonArguments}
    Write-Host "Starting $Name; log: $Log"
    $SavedPreference = $ErrorActionPreference
    try {
        # Native stderr is captured, including tracebacks; it is not a success signal.
        $ErrorActionPreference = 'Continue'
        & $PythonExecutable @PythonArguments 2>&1 | Tee-Object -FilePath $Log -ErrorAction Stop
        $PythonExitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $SavedPreference }
    if ($PythonExitCode -ne 0) { throw "Stage $Name failed with Python exit code $PythonExitCode. See $Log. Remaining stages will not run." }
    Save-NewJson -Path (Join-Path $Logs ($Name + '.succeeded.json')) -Value @{stage=$Name;time=(Get-Date -Format o);exit_code=$PythonExitCode}
}

function Confirm-Training {
    param([string]$Directory)
    $Completion = Get-Content -LiteralPath (Join-Path $Directory 'completed.json') -Raw | ConvertFrom-Json
    if ($Completion.smoke_only -ne $false -or $Completion.test_used -ne $false -or $Completion.epochs -ne 15) {
        throw "Invalid training completion marker: $Directory"
    }
}

try {
    if (Test-Path -LiteralPath $PipelineRoot) { throw "Output root already exists: $PipelineRoot" }
    New-Item -ItemType Directory -Path $Logs | Out-Null
    Set-Location -LiteralPath $ExperimentRoot
    Write-Host "Unique overnight output root: $PipelineRoot"
    Write-Host 'Expected runtime: 3-5 hours plus up to 10 minutes for optional occlusion.'
    Write-Host 'Keep this window open and prevent sleep/sign-out. No system settings will be changed.'
    if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw "Missing repository interpreter: $PythonExecutable" }
    Confirm-Inputs
    foreach ($Destination in @($OriginalRun,$CleanedRun,$Analysis,$Occlusion,$Report)) {
        if (Test-Path -LiteralPath $Destination) { throw "Destination exists: $Destination" }
    }
    $Drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($PipelineRoot))
    if ($Drive.AvailableFreeSpace -lt 6GB) { throw 'Less than 6 GiB free output space; stopping before training.' }
    Save-NewJson -Path (Join-Path $Logs 'launch.json') -Value @{time=(Get-Date -Format o);interpreter=$PythonExecutable;output=$PipelineRoot;locked_input_hashes=$ExpectedHashes;occlusion_policy=@{window_hours=8;stage_budget_minutes=10;minimum_free_gpu_gib=1.5;max_clips=4;clip_number=4;regions='fixed 2x2 grid, 112x112 pixels';fill='RGB training mean across all frames';selection='lowest-ID correct and incorrect segment per best model, where present'};test_used=$false}
    Invoke-PythonStage -Name '00_preflight' -PythonArguments @('-B','-u',$Helper,'preflight','--original',$OriginalRun,'--cleaned',$CleanedRun,'--output',(Join-Path $PipelineRoot 'preflight'))

    $CurrentStage = '01_original_training'
    Confirm-Inputs
    Invoke-PythonStage -Name $CurrentStage -PythonArguments @('-B','-u',$Runner,'--config',$OriginalConfig,'--run-dir',$OriginalRun,'--approve-full-training')
    Confirm-Training -Directory $OriginalRun

    $CurrentStage = '02_cleaned_training'
    Confirm-Inputs
    Invoke-PythonStage -Name $CurrentStage -PythonArguments @('-B','-u',$Runner,'--config',$CleanedConfig,'--run-dir',$CleanedRun,'--approve-full-training')
    Confirm-Training -Directory $CleanedRun

    $CurrentStage = '03_quadrant_analysis'
    Confirm-Inputs
    Invoke-PythonStage -Name $CurrentStage -PythonArguments @('-B','-u',$Helper,'analyze','--original',$OriginalRun,'--cleaned',$CleanedRun,'--output',$Analysis)

    $CurrentStage = '04_exploratory_occlusion'
    Confirm-Inputs
    $RemainingMinutes = [math]::Max(0,480 - $PipelineClock.Elapsed.TotalMinutes).ToString('F3',[Globalization.CultureInfo]::InvariantCulture)
    Invoke-PythonStage -Name $CurrentStage -PythonArguments @('-B','-u',$Helper,'occlude','--original',$OriginalRun,'--cleaned',$CleanedRun,'--output',$Occlusion,'--remaining-minutes',$RemainingMinutes)

    $CurrentStage = '05_supervisor_report'
    Confirm-Inputs
    Invoke-PythonStage -Name $CurrentStage -PythonArguments @('-B','-u',$Helper,'report','--original',$OriginalRun,'--cleaned',$CleanedRun,'--analysis',$Analysis,'--occlusion',$Occlusion,'--output',$Report)
    Save-NewJson -Path (Join-Path $PipelineRoot 'completed.json') -Value @{completed=(Get-Date -Format o);elapsed_minutes=$PipelineClock.Elapsed.TotalMinutes;report=(Join-Path $Report 'supervisor_report.md');test_used=$false}
    Write-Host "Overnight pipeline completed. Report: $(Join-Path $Report 'supervisor_report.md')"
}
catch {
    $Failure = $_.Exception.Message
    [Console]::Error.WriteLine("Pipeline stopped at ${CurrentStage}: $Failure")
    if (Test-Path -LiteralPath $Logs -PathType Container) {
        Save-NewJson -Path (Join-Path $Logs 'pipeline_failure.json') -Value @{stage=$CurrentStage;time=(Get-Date -Format o);message=$Failure;elapsed_minutes=$PipelineClock.Elapsed.TotalMinutes}
    }
    exit 1
}
finally {
    $PipelineClock.Stop()
    Set-Location -LiteralPath $OriginalLocation.Path
}
exit 0
