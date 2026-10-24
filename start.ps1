param([switch]$NoBrowser, [switch]$SetupOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtimeDir = Join-Path $PSScriptRoot '.runtime'
$venvPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$requirements = Join-Path $PSScriptRoot 'requirements.txt'
$lockStream = $null
function Test-Python([string]$Executable) {
    if (-not (Test-Path -LiteralPath $Executable)) { return $false }
    try {
        & $Executable -c 'import sys,venv; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}
function Find-Python {
    $candidates = @()
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        try {
            $found = & $py.Source -3 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0) { $candidates += [string]$found }
        } catch { }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python -and $python.Source -notlike '*WindowsApps*') { $candidates += $python.Source }
    foreach ($version in @('314','313','312','311')) {
        $candidates += Join-Path $env:LOCALAPPDATA "Programs/Python/Python$version/python.exe"
    }
    $candidates += Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    foreach ($candidate in $candidates) { if (Test-Python $candidate) { return $candidate } }
    return $null
}
function Get-Health([int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $pending = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $pending.AsyncWaitHandle.WaitOne(150)) { return $null }
        $client.EndConnect($pending)
    } catch { return $null } finally { $client.Dispose() }
    try { return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 1 }
    catch { return $null }
}
try {
    Write-Host 'FishLens - preparing your local website' -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    try { $lockStream = [System.IO.File]::Open((Join-Path $runtimeDir 'launcher.lock'), 'OpenOrCreate', 'ReadWrite', 'None') }
    catch { throw 'Another launcher is preparing this folder. Wait for it to finish.' }
    if (-not (Test-Python $venvPython)) {
        $basePython = Find-Python
        if (-not $basePython) {
            $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
            if (-not $winget) { throw 'Python 3.11+ and WinGet were not found. Install Python from https://www.python.org/downloads/windows/ and retry.' }
            Write-Host 'Installing Python 3.12 for your Windows user via WinGet...'
            & $winget.Source install --id Python.Python.3.12 --exact --source winget --scope user --silent --disable-interactivity | Out-Host
            if ($LASTEXITCODE -ne 0) { throw 'Python installation could not finish (network, policy or agreement may need attention). Install Python from https://www.python.org/downloads/windows/ and retry.' }
            $basePython = Find-Python
            if (-not $basePython) { throw 'Python could not be located. Restart this launcher.' }
        }
        Write-Host 'Creating the project virtual environment...'
        & $basePython -m venv (Join-Path $PSScriptRoot '.venv') | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv. Check folder permissions and Python.' }
    }
    $hash = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash
    $stamp = Join-Path $runtimeDir 'requirements.sha256'
    $previousHash = if (Test-Path -LiteralPath $stamp) { (Get-Content -LiteralPath $stamp -Raw).Trim() } else { '' }
    $dependenciesReady = $false
    try {
        & $venvPython -c "from importlib.metadata import version; from pathlib import Path; requirements=Path('requirements.txt').read_text().splitlines(); assert all(version(name.strip()).split('+')[0] == expected.strip() for line in requirements if line.strip() and not line.lstrip().startswith('#') for name,expected in [line.split('==',1)])" 2>$null | Out-Null
        $dependenciesReady = $LASTEXITCODE -eq 0
    } catch { }
    if ($previousHash -ne $hash -or -not $dependenciesReady) {
        Write-Host 'Installing project dependencies (internet required on first launch)...'
        & $venvPython -m pip install --disable-pip-version-check -r $requirements | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check your internet connection and retry.' }
        & $venvPython -m pip check | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed. Review requirements.txt.' }
        Set-Content -LiteralPath $stamp -Value $hash -Encoding ASCII
    }
    if ($SetupOnly) { Write-Host 'Environment ready.' -ForegroundColor Green; exit 0 }
    # Compute the same checkout identity without loading all models twice.
    $instance = (& $venvPython -c 'import hashlib; from pathlib import Path; print(hashlib.sha256(str(Path.cwd().resolve()).casefold().encode()).hexdigest()[:16])').Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Could not compute the project identity.' }
    $port = $null
    foreach ($candidatePort in 8765..8785) {
        $health = Get-Health $candidatePort
        if ($health -and $health.app -eq 'fishlens' -and $health.instance_id -eq $instance) {
            $url = "http://127.0.0.1:$candidatePort"
            Write-Host "Already running: $url" -ForegroundColor Green
            if (-not $NoBrowser) { Start-Process $url }
            exit 0
        }
        if ($null -eq $port) {
            $listener = $null
            try {
                $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $candidatePort)
                $listener.Start(); $port = $candidatePort
            } catch { } finally { if ($listener) { $listener.Stop() } }
        }
    }
    if ($null -eq $port) { throw 'Ports 8765-8785 are busy. Close another local service and retry.' }
    $env:PORT = [string]$port
    $env:FISHLENS_LAUNCH_ID = [guid]::NewGuid().ToString()
    $stdout = Join-Path $runtimeDir 'server.stdout.log'
    $stderr = Join-Path $runtimeDir 'server.stderr.log'
    Write-Host "Starting http://127.0.0.1:$port ..."
    $serverProcess = Start-Process -FilePath $venvPython -ArgumentList '-m','backend.server' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $ready = $false
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        $serverProcess.Refresh()
        if ($serverProcess.HasExited) { break }
        $health = Get-Health $port
        if ($health -and $health.app -eq 'fishlens' -and $health.instance_id -eq $instance -and $health.launch_id -eq $env:FISHLENS_LAUNCH_ID) { $ready = $true; break }
        Start-Sleep -Milliseconds 300
    }
    if (-not $ready) {
        $serverProcess.Refresh()
        if (-not $serverProcess.HasExited) { $serverProcess.Kill() }
        throw "Website failed to become ready. Read $stderr"
    }
    $actualProcess = Get-Process -Id $health.pid
    @{ pid = $actualProcess.Id; port = $port; instance_id = $instance; executable = $actualProcess.Path; started = $actualProcess.StartTime.ToUniversalTime().ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeDir 'server.json') -Encoding UTF8
    $url = "http://127.0.0.1:$port"
    Write-Host "Ready: $url" -ForegroundColor Green
    Write-Host 'Use Stop-FishLens.cmd to stop the local server.'
    if (-not $NoBrowser) { Start-Process $url }
} catch {
    Write-Host "FishLens: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally { if ($lockStream) { $lockStream.Dispose() } }
