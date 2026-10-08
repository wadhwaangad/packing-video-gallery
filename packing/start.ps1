$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (Test-Path -LiteralPath $bundledPython) {
    $pythonPath = $bundledPython
} else {
    $python = Get-Command py -ErrorAction SilentlyContinue
    if (-not $python) { $python = Get-Command python -ErrorAction SilentlyContinue }
    if ($python) { $pythonPath = $python.Source }
}
if (-not $pythonPath) { throw 'Python was not found. Install Python 3, then run this launcher again.' }

$port = 8011
$url = "http://127.0.0.1:$port/packing/"
$sceneUrl = "${url}immersive/"
$serverReady = $false
try {
    $response = Invoke-WebRequest -Uri $sceneUrl -UseBasicParsing
    $serverReady = $response.StatusCode -eq 200
} catch {
    Start-Process -FilePath $pythonPath -ArgumentList @('-m','http.server',"$port",'--directory',"`"$projectRoot`"") -WindowStyle Hidden
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try { Invoke-WebRequest -Uri $sceneUrl -UseBasicParsing | Out-Null; $serverReady = $true; break }
        catch { Start-Sleep -Milliseconds 200 }
    }
}
if (-not $serverReady) { throw "Could not start the local server. Open $projectRoot manually and check port $port." }
try { Start-Process $url } catch { Write-Output "Open this URL in your browser: $url" }
Write-Output "Packing gallery: $url"
Write-Output "Detailed 3D room: $sceneUrl"
