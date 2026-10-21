# LEGACY LAUNCHER: historical settings; review paths, deadlines and hashes before use.
param(
    [string]$CurrentRun = (Join-Path (Split-Path -Parent $PSScriptRoot) 'outputs/runs/overnight_20261014_201445_87e92c7a'),
    [switch]$CheckOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$ExperimentRoot=Split-Path -Parent $PSScriptRoot
$Python=Join-Path $ExperimentRoot '.venv\Scripts\python.exe'
$Runner=Join-Path $ExperimentRoot 'src\run_followup_experiment.py'
$Reporter=Join-Path $ExperimentRoot 'src\build_followup_report.py'
$WeightedConfig=Join-Path $ExperimentRoot 'configs\followup_2026-10-15\corrected_weighted.json'
$PartialConfig=Join-Path $ExperimentRoot 'configs\followup_2026-10-15\partial_finetune.json'
$StartCutoff=[DateTimeOffset]::Parse('2026-10-15T11:00:00+05:00')
$FinishCutoff=[DateTimeOffset]::Parse('2026-10-15T11:40:00+05:00')
$ExpectedHashes=[ordered]@{
    'src/run_followup_experiment.py' = 'E821F7140F2CB2205B30C13E39C2BA571DACFB521588B6A36727BFA99B7C927A'
    'src/build_followup_report.py' = 'FE00AA7DEF6ECB32EA43431A0015D7CDF5590D9CB3228ACCA070ED96222AF172'
    'src/data/dataset.py' = 'AD84A107C6A93123A7DCF0869EEB0DCA6947EB377DB027A4EC7105A4BB1DD292'
    'configs/followup_2026-10-15/corrected_weighted.json' = '719F3EFDB82C2B628F8E4C9469AE780AF14ECE3EF7BB6F4D74707C9B73B77F76'
    'configs/followup_2026-10-15/partial_finetune.json' = '70885EC5B167D064E95DB6B72F996F31F779DE0D7FE2B347A6FB61404945EAA0'
    'data/labels/quadrant_metadata.csv' = '34961A06F9F9C2A7DD6330C8C389D218E9DA961E2883FD5E934D86B3572429F3'
    'data/labels/binary_clean_metadata.csv' = 'F450328DA49D0AE34ACC7A67FEFB46897EE51A8790C41D480B96AAF6B91E58F8'
}

function Confirm-Inputs {
    foreach($Relative in $ExpectedHashes.Keys){
        $Path=Join-Path $ExperimentRoot $Relative
        if((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $ExpectedHashes[$Relative]){throw "Input changed: $Path"}
    }
}
function Get-ResearchPython {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe','pythonw.exe') -and $_.CommandLine -and $_.CommandLine.IndexOf($ExperimentRoot,[StringComparison]::OrdinalIgnoreCase) -ge 0
    })
}
function Get-CurrentState {
    $Jobs=@(Get-ResearchPython)
    $Completed=Test-Path -LiteralPath (Join-Path $CurrentRun 'completed.json')
    $Failed=Test-Path -LiteralPath (Join-Path $CurrentRun 'logs\pipeline_failure.json')
    $Owners=@(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('powershell.exe','pwsh.exe') -and $_.CommandLine -match '(?i)-File\s+(?:"[^"\r\n]*[\\/]run_overnight_supervisor(?:_with_fourclass)?\.ps1"|[^\s"]*[\\/]run_overnight_supervisor(?:_with_fourclass)?\.ps1)(?:\s|$)'
    })
    [pscustomobject]@{Ready=(($Completed -or $Failed) -and $Jobs.Count -eq 0);Completed=$Completed;Failed=$Failed;PythonCount=$Jobs.Count;OwnerCount=$Owners.Count}
}
function Can-StartGpu {
    param([double]$EstimatedMinutes)
    $Now=[DateTimeOffset]::UtcNow
    $Now -lt $StartCutoff -and ($FinishCutoff-$Now).TotalMinutes -ge $EstimatedMinutes -and @(Get-ResearchPython).Count -eq 0
}
function Save-NewJson {
    param([string]$Path,$Value)
    $Stream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try{$Writer=New-Object IO.StreamWriter($Stream,(New-Object Text.UTF8Encoding($false)));try{$Writer.WriteLine(($Value|ConvertTo-Json -Depth 12))}finally{$Writer.Dispose()}}finally{$Stream.Dispose()}
}
function Add-Stage {
    param([string]$Stage,[string]$Status,[string]$Reason)
    $Item=[ordered]@{stage=$Stage;status=$Status;reason=$Reason;time=(Get-Date -Format o)}
    $Stages.Add($Item)
    Save-NewJson -Path (Join-Path $Logs ('{0:D2}_{1}.json' -f $Stages.Count,$Stage)) -Value $Item
    Write-Host "$Stage : $Status : $Reason"
}
function Invoke-NewPython {
    param([string]$Stage,[string[]]$Arguments,[bool]$Gpu=$true)
    if($Gpu){
        if(-not (Can-StartGpu -EstimatedMinutes 0)){throw 'GPU start gate closed: cutoff, remaining time, or active repository Python process.'}
        Confirm-Inputs
    }
    $Quoted=@($Arguments | ForEach-Object {if($_.Contains('"')){throw 'Unexpected double quote in argument.'};'"'+$_+'"'}) -join ' '
    $Stdout=Join-Path $Logs ($Stage+'.stdout.log');$Stderr=Join-Path $Logs ($Stage+'.stderr.log')
    if((Test-Path $Stdout) -or (Test-Path $Stderr)){throw 'Log exists; refusing overwrite.'}
    $Process=Start-Process -FilePath $Python -ArgumentList $Quoted -WorkingDirectory $ExperimentRoot -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
    # Retain the native handle so Windows PowerShell preserves ExitCode after exit.
    $ProcessHandle = $Process.Handle
    Save-NewJson -Path (Join-Path $Logs ($Stage+'.process.json')) -Value @{pid=$Process.Id;arguments=$Arguments;gpu=$Gpu;started_utc=$Process.StartTime.ToUniversalTime().ToString('o')}
    while(-not $Process.WaitForExit(10000)){
        if($Gpu -and [DateTimeOffset]::UtcNow -ge $FinishCutoff.AddMinutes(2)){
            # Only this newly spawned process tree can be terminated, never the existing pipeline.
            $Live=Get-CimInstance Win32_Process -Filter "ProcessId=$($Process.Id)"
            if($Live -and $Live.CommandLine -and $Live.CommandLine.IndexOf($Runner,[StringComparison]::OrdinalIgnoreCase) -ge 0){
                & "$env:SystemRoot\System32\taskkill.exe" /PID $Process.Id /T /F | Out-Null
            }
            throw 'Follow-up GPU process exceeded the reporting cutoff; its new process tree was stopped. Existing pipeline untouched.'
        }
    }
    $Process.WaitForExit()
    if($Process.ExitCode -ne 0){throw "Python failed, exit $($Process.ExitCode); see $Stdout and $Stderr"}
}

Confirm-Inputs
if(-not (Test-Path -LiteralPath $CurrentRun -PathType Container)){throw 'Specified current run directory is missing.'}
if($CheckOnly){
    $State=Get-CurrentState
    $State|Format-List
    Write-Output "GPU start cutoff UTC: $($StartCutoff.UtcDateTime.ToString('o'))"
    Write-Output "GPU finish cutoff UTC: $($FinishCutoff.UtcDateTime.ToString('o'))"
    Write-Output "Weighted gate currently open: $(Can-StartGpu -EstimatedMinutes 110)"
    Write-Output 'Check-only finished: no waiting, files, processes, GPU work or training started.'
    exit 0
}
$Mutex=New-Object Threading.Mutex($false,'Local\DentalVideoFollowup_20261015')
if(-not $Mutex.WaitOne(0)){throw 'Another follow-up launcher already owns the lock.'}
$Identifier=(Get-Date -Format 'yyyyMMdd_HHmmss')+'_'+[guid]::NewGuid().ToString('N').Substring(0,8)
$Followup=Join-Path $ExperimentRoot ('outputs\runs\followup_'+$Identifier)
$Logs=Join-Path $Followup 'logs'
$Stages=New-Object 'Collections.Generic.List[object]'
$Weighted=Join-Path $Followup 'corrected_weighted'
$Probe=Join-Path $Followup 'partial_feasibility_smoke'
$Partial=Join-Path $Followup 'partial_finetune'
$ExistingLocation=Get-Location
try{
    if(Test-Path $Followup){throw 'Follow-up output root already exists.'}
    New-Item -ItemType Directory -Path $Logs|Out-Null
    Save-NewJson -Path (Join-Path $Logs 'launch.json') -Value @{current=$CurrentRun;followup=$Followup;start_cutoff=$StartCutoff.ToString('o');finish_cutoff=$FinishCutoff.ToString('o');weighted_budget_minutes=110;partial_initial_budget_minutes=45;locked_inputs=$ExpectedHashes}
    Write-Host "Follow-up output root: $Followup"
    Write-Host 'Waiting for the existing pipeline; no concurrent training. Keep the window open and prevent sleep.'
    $UnmarkedExitCount=0;$Ready=$false
    while([DateTimeOffset]::UtcNow -lt $StartCutoff){
        $State=Get-CurrentState
        if($State.Ready){$Ready=$true;Add-Stage 'wait' 'finished' "Current terminal marker present; active repository Python count is zero. Current success=$($State.Completed)";break}
        if(-not $State.Completed -and -not $State.Failed -and $State.OwnerCount -eq 0 -and $State.PythonCount -eq 0){$UnmarkedExitCount++}else{$UnmarkedExitCount=0}
        if($UnmarkedExitCount -ge 3){$Ready=$true;Add-Stage 'wait' 'aborted_current' 'Current launcher and Python processes absent for three polls; no terminal marker. Existing artifacts preserved.';break}
        Start-Sleep -Seconds 10
    }
    if(-not $Ready){Add-Stage 'wait' 'cutoff' '11:00 Astana reached before idle terminal state; no additional GPU work will start.'}
    $WeightedSuccess=$false
    if($Ready -and (Can-StartGpu -EstimatedMinutes 110)){
        try{
            Invoke-NewPython 'weighted' @('-B','-u',$Runner,'--config',$WeightedConfig,'--run-dir',$Weighted,'--approve-full-training','--start-before',$StartCutoff.ToString('o'),'--finish-by',$FinishCutoff.ToString('o'))
            $Done=Get-Content (Join-Path $Weighted 'completed.json') -Raw|ConvertFrom-Json
            if($Done.smoke_only -or $Done.test_used -or $Done.epochs -ne 15){throw 'Weighted completion marker invalid.'}
            $WeightedSuccess=$true;Add-Stage 'weighted' 'completed' '15 epochs; corrected weighted singleton loss; no oversampling.'
        }catch{Add-Stage 'weighted' 'failed' $_.Exception.Message}
    }else{Add-Stage 'weighted' 'skipped' 'Current not idle, start cutoff reached, or fewer than 110 minutes remain before 11:40.'}
    if($WeightedSuccess -and (Can-StartGpu -EstimatedMinutes 45)){
        try{
            $Best=Get-Content (Join-Path $Weighted 'best_checkpoint.json') -Raw|ConvertFrom-Json
            if([IO.Path]::GetFileName($Best.path) -ne $Best.path){throw 'Unsafe warm-start filename.'}
            $WarmStart=Join-Path $Weighted $Best.path
            Invoke-NewPython 'partial_probe' @('-B','-u',$Runner,'--config',$PartialConfig,'--run-dir',$Probe,'--smoke','--warm-start',$WarmStart,'--start-before',$StartCutoff.ToString('o'),'--finish-by',$FinishCutoff.ToString('o'))
            $Timing=Get-Content (Join-Path $Probe 'timing.json') -Raw|ConvertFrom-Json
            $TrainMaximum=($Timing.train_segment_seconds|Measure-Object -Maximum).Maximum
            $ValMaximum=($Timing.val_segment_seconds|Measure-Object -Maximum).Maximum
            $EstimatedMinutes=3*(53*$TrainMaximum+13*$ValMaximum)/60*1.5+5
            Add-Stage 'partial_probe' 'completed' "Estimated three-epoch runtime with margin: $EstimatedMinutes minutes; reserved bytes=$($Timing.peak_gpu_reserved_bytes)."
            if($Timing.peak_gpu_reserved_bytes -le 4GB -and (Can-StartGpu -EstimatedMinutes $EstimatedMinutes)){
                Invoke-NewPython 'partial' @('-B','-u',$Runner,'--config',$PartialConfig,'--run-dir',$Partial,'--approve-full-training','--warm-start',$WarmStart,'--start-before',$StartCutoff.ToString('o'),'--finish-by',$FinishCutoff.ToString('o'))
                $Done=Get-Content (Join-Path $Partial 'completed.json') -Raw|ConvertFrom-Json
                if($Done.smoke_only -or $Done.test_used -or $Done.epochs -ne 3){throw 'Partial completion marker invalid.'}
                Add-Stage 'partial' 'completed' 'Three fixed epochs; only last block and head trainable.'
            }else{Add-Stage 'partial' 'skipped' 'Measured memory/runtime or hard start cutoff failed the predefined gate.'}
        }catch{Add-Stage 'partial' 'failed_or_skipped' $_.Exception.Message}
    }else{Add-Stage 'partial' 'skipped' 'Weighted run incomplete or fewer than 45 minutes available; no GPU probe.'}
}catch{
    if(Test-Path $Logs){Add-Stage 'controller' 'failed' $_.Exception.Message}else{[Console]::Error.WriteLine($_.Exception.Message)}
}finally{
    if(Test-Path $Logs){
        Save-NewJson -Path (Join-Path $Followup 'status.json') -Value @{current=$CurrentRun;followup=$Followup;stages=@($Stages.ToArray());test_used=$false}
        try{
            Invoke-NewPython 'report' @('-B','-u',$Reporter,'--current',$CurrentRun,'--followup',$Followup) -Gpu $false
            Write-Host "Updated report: $(Join-Path $Followup 'report\supervisor_report.md')"
        }catch{
            $Fallback=@('# Supervisor report: follow-up reporting failure','','Existing pipeline was not interrupted. Consult preserved run metrics and logs.','',"Report generator error: $($_.Exception.Message)",'',"Current pipeline: $CurrentRun","Follow-up: $Followup",'',($Stages|ConvertTo-Json -Depth 8)) -join [Environment]::NewLine
            $Fallback|Out-File -LiteralPath (Join-Path $Followup 'fallback_report.md') -Encoding UTF8 -NoClobber
            [Console]::Error.WriteLine('Detailed report failed; fallback report and all logs preserved.')
        }
    }
    Set-Location -LiteralPath $ExistingLocation.Path
    $Mutex.ReleaseMutex();$Mutex.Dispose()
}