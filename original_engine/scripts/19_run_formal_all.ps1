param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Config = 'config\formal_local_v1.toml'
$FreezeManifest = Join-Path $ProjectRoot 'frozen\formal_v1\FREEZE_MANIFEST.json'
$LogDir = Join-Path $ProjectRoot 'outputs\formal_local_v1\logs'
$OrchestratorLog = Join-Path $LogDir 'formal_orchestrator.log'
$StageLedger = Join-Path $LogDir 'formal_stage_ledger.jsonl'

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python environment is missing: $Python"
}
if (-not (Test-Path -LiteralPath $FreezeManifest)) {
    throw "Freeze manifest is missing: $FreezeManifest"
}

$Freeze = Get-Content -LiteralPath $FreezeManifest -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $Freeze.formal_generation_ready) {
    throw 'Formal generation is blocked because formal_generation_ready is not true.'
}

$Stages = @(
    [pscustomobject]@{ name = 'part1_gpt_oss_20b'; tasks = 'outputs\formal_local_v1\tasks_part1.csv'; model = 'gpt_oss_20b' },
    [pscustomobject]@{ name = 'part1_mistral_small_32_24b'; tasks = 'outputs\formal_local_v1\tasks_part1.csv'; model = 'mistral_small_32_24b' },
    [pscustomobject]@{ name = 'part1_qwen35_27b_q4km'; tasks = 'outputs\formal_local_v1\tasks_part1.csv'; model = 'qwen35_27b_q4km' },
    [pscustomobject]@{ name = 'part2_gpt_oss_20b'; tasks = 'outputs\formal_local_v1\tasks_part2.csv'; model = 'gpt_oss_20b' },
    [pscustomobject]@{ name = 'part2_mistral_small_32_24b'; tasks = 'outputs\formal_local_v1\tasks_part2.csv'; model = 'mistral_small_32_24b' },
    [pscustomobject]@{ name = 'part2_qwen35_27b_q4km'; tasks = 'outputs\formal_local_v1\tasks_part2.csv'; model = 'qwen35_27b_q4km' }
)

function Write-StageRecord {
    param(
        [string]$Stage,
        [string]$Status,
        [int]$ExitCode
    )
    [pscustomobject]@{
        timestamp_local = (Get-Date).ToString('o')
        stage = $Stage
        status = $Status
        exit_code = $ExitCode
    } | ConvertTo-Json -Compress | Add-Content -LiteralPath $StageLedger -Encoding UTF8
}

Set-Location -LiteralPath $ProjectRoot
Add-Content -LiteralPath $OrchestratorLog -Encoding UTF8 -Value ("{0} FORMAL ORCHESTRATOR START" -f (Get-Date).ToString('o'))

foreach ($Stage in $Stages) {
    Add-Content -LiteralPath $OrchestratorLog -Encoding UTF8 -Value ("{0} STAGE START {1}" -f (Get-Date).ToString('o'), $Stage.name)
    Write-StageRecord -Stage $Stage.name -Status 'started' -ExitCode 0
    # Python's progress logger intentionally writes to stderr. Windows PowerShell 5
    # wraps native stderr as non-terminating ErrorRecord objects, so temporarily use
    # Continue and rely only on the native process exit code for stage control.
    $SavedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Python 'scripts\04_run_generation.py' '--config' $Config '--tasks' $Stage.tasks '--model' $Stage.model '--confirm-local-run' 'I_UNDERSTAND_LOCAL_MODEL_RUN' *>> $OrchestratorLog
    $Code = $LASTEXITCODE
    $ErrorActionPreference = $SavedErrorActionPreference
    if ($Code -ne 0) {
        # A completed stage can return non-zero because it contains a terminal
        # refusal or generation failure. Continue only when every planned task
        # for this model/stage has a terminal record.
        & $Python 'scripts\21_check_stage_terminal.py' '--tasks' $Stage.tasks '--model' $Stage.model *>> $OrchestratorLog
        $StageCheckCode = $LASTEXITCODE
        if ($StageCheckCode -ne 0) {
            Write-StageRecord -Stage $Stage.name -Status 'needs_attention_incomplete' -ExitCode $Code
            Add-Content -LiteralPath $OrchestratorLog -Encoding UTF8 -Value ("{0} STAGE STOPPED INCOMPLETE {1} generator_exit={2}" -f (Get-Date).ToString('o'), $Stage.name, $Code)
            exit $Code
        }
        Write-StageRecord -Stage $Stage.name -Status 'completed_with_terminal_non_success' -ExitCode $Code
        Add-Content -LiteralPath $OrchestratorLog -Encoding UTF8 -Value ("{0} STAGE COMPLETE WITH TERMINAL NON-SUCCESS {1} generator_exit={2}" -f (Get-Date).ToString('o'), $Stage.name, $Code)
        continue
    }
    Write-StageRecord -Stage $Stage.name -Status 'completed' -ExitCode 0
    Add-Content -LiteralPath $OrchestratorLog -Encoding UTF8 -Value ("{0} STAGE COMPLETE {1}" -f (Get-Date).ToString('o'), $Stage.name)
}

$SavedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $Python 'scripts\05_quality_control.py' '--config' $Config *>> $OrchestratorLog
$QcCode = $LASTEXITCODE
$ErrorActionPreference = $SavedErrorActionPreference
Write-StageRecord -Stage 'final_quality_control' -Status $(if ($QcCode -eq 0) { 'completed' } else { 'needs_attention' }) -ExitCode $QcCode
Add-Content -LiteralPath $OrchestratorLog -Encoding UTF8 -Value ("{0} FORMAL ORCHESTRATOR END qc_exit={1}" -f (Get-Date).ToString('o'), $QcCode)
exit $QcCode
