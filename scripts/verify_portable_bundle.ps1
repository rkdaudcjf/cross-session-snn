$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$requiredFiles = @(
    'pyproject.toml',
    'uv.lock',
    'HANDOFF_KO.md',
    'scripts\train_transfer_channel_selection.py',
    'scripts\train_transfer_adaptation_search.py',
    'scripts\summarize_transfer_norm20_multiseed.py',
    'scripts\summarize_transfer_adaptation_search.py',
    'configs\transfer_sutl_norm20_seed42.yaml',
    'configs\transfer_sutl_norm20_seed43.yaml',
    'configs\transfer_sutl_norm20_seed44.yaml',
    'data\sabes_zenodo\master_mat\indy_20160622_01.mat',
    'data\sabes_zenodo\master_mat\indy_20160630_01.mat',
    'data\sabes_zenodo\master_mat\indy_20170124_01.mat',
    'data\sabes_zenodo\master_mat\indy_20170127_03.mat',
    'data\sabes_zenodo\master_mat\indy_20170131_02.mat',
    'outputs\transfer_sutl_norm20_multiseed\README.md'
)

$missing = @()
foreach ($relativePath in $requiredFiles) {
    $path = Join-Path $repoRoot $relativePath
    if (-not (Test-Path -LiteralPath $path)) {
        $missing += $relativePath
    }
}

$pairDirectories = @(
    'indy_20160622_01_to_indy_20160630_01',
    'indy_20170124_01_to_indy_20170127_03',
    'indy_20170127_03_to_indy_20170131_02'
)
$completeRuns = 0
foreach ($seed in 42, 43, 44) {
    foreach ($pairDirectory in $pairDirectories) {
        $summaryPath = Join-Path `
            $repoRoot `
            "outputs\transfer_sutl_norm20_seed$seed\$pairDirectory\top64\run_summary.json"
        if (-not (Test-Path -LiteralPath $summaryPath)) {
            continue
        }
        try {
            $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
            if ($summary.status -eq 'complete') {
                $completeRuns += 1
            }
        } catch {
            Write-Warning "Could not parse $summaryPath"
        }
    }
}

$resumeCheckpoints = 0
foreach ($seed in 42, 43, 44) {
    $checkpoints = Get-ChildItem `
        (Join-Path $repoRoot "outputs\transfer_adaptation_search_seed$seed") `
        -Recurse `
        -Filter 'last_checkpoint.pt' `
        -ErrorAction SilentlyContinue
    if ($checkpoints) {
        $resumeCheckpoints += 1
    }
}

Write-Host "Repository root: $repoRoot"
Write-Host "Required files missing: $($missing.Count)"
if ($missing) {
    $missing | ForEach-Object { Write-Host "  MISSING: $_" }
}
Write-Host "Completed normalized transfer runs found: $completeRuns (expected 9)"
Write-Host "Seeds with resumable adaptation checkpoints: $resumeCheckpoints (expected 3)"

if ($missing.Count -gt 0 -or $completeRuns -lt 9 -or $resumeCheckpoints -lt 3) {
    throw 'Portable bundle verification failed.'
}

Write-Host 'Portable bundle verification passed.'
