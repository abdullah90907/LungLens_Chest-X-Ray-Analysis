param([ValidateRange(1024, 65535)][int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$lunglensPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $lunglensPython)) {
    throw 'Create the project .venv and install requirements.txt first; see README.md.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $lunglensPython -m uvicorn backend:app --host 127.0.0.1 --port $Port
}
finally {
    Pop-Location
}
