$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found. Run 'uv sync --dev' in $repoRoot first."
}

$alreadyRunning = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and
    $_.CommandLine -like '*train_transfer_adaptation_search.py*'
}
if ($alreadyRunning) {
    throw 'An adaptation-search process is already running. Stop it or wait for completion.'
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runs = @()
foreach ($seed in 42, 43, 44) {
    $outputRoot = Join-Path $repoRoot "outputs\transfer_adaptation_search_seed$seed"
    New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
    $stdout = Join-Path $outputRoot "resume_$stamp.stdout.log"
    $stderr = Join-Path $outputRoot "resume_$stamp.stderr.log"
    $arguments = @(
        'scripts\train_transfer_adaptation_search.py',
        '--seed', "$seed",
        '--cpu-threads', '4',
        '--resume'
    )
    $process = Start-Process `
        -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    $runs += [pscustomobject]@{
        Seed = $seed
        LauncherPID = $process.Id
        Stdout = $stdout
        Stderr = $stderr
    }
}

$runs | Format-Table -AutoSize
Write-Host 'Started three resume runs. Check outputs\transfer_adaptation_search_seed* for progress.'
