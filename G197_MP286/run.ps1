$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Project environment missing. Run .\setup.ps1 first."
}
& ".\.venv\Scripts\Activate.ps1"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
