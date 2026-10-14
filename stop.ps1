$ErrorActionPreference = 'Stop'
try {
    $statePath = Join-Path $PSScriptRoot '.runtime/server.json'
    if (-not (Test-Path -LiteralPath $statePath)) { Write-Host 'No launcher-managed FishLens server found.'; exit 0 }
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $process = Get-Process -Id $state.pid -ErrorAction SilentlyContinue
    if (-not $process) { Write-Host 'FishLens is already stopped.'; exit 0 }
    if ($process.Path -ne $state.executable -or $process.StartTime.ToUniversalTime().ToString('o') -ne $state.started) { throw 'The saved process no longer matches FishLens. No process was stopped.' }
    Stop-Process -Id $process.Id
    Write-Host 'FishLens stopped.' -ForegroundColor Green
} catch { Write-Host $_.Exception.Message -ForegroundColor Red; exit 1 }
