$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

py -3.11 --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11 is not installed. Install Python 3.11 (64-bit), then reopen PowerShell."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.11 -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

& ".\.venv\Scripts\python.exe" -c "import sys, pandas, sklearn, mlflow; print('Python:', sys.version); print('pandas:', pandas.__version__); print('scikit-learn:', sklearn.__version__); print('MLflow:', mlflow.__version__); assert sys.version_info[:2] == (3, 11); assert sklearn.__version__ == '1.5.2'"
Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
