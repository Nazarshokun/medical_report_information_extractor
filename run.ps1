#Requires -Version 5.1
<#
  Windows launcher for the Medical Report Information Extractor.
  Mirrors run.sh: always launches Streamlit from the project venv (.venv-surya),
  so a stray `streamlit` on PATH can't pick the wrong interpreter.

  Usage:
    .\run.ps1                       # launch on the default port (8501)
    .\run.ps1 --server.port 8502    # extra args pass straight through to Streamlit

  If PowerShell blocks the script ("running scripts is disabled on this system"):
    powershell -ExecutionPolicy Bypass -File .\run.ps1
#>
$ErrorActionPreference = "Stop"

$DIR  = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV = Join-Path $DIR ".venv-surya"
$PY   = Join-Path $VENV "Scripts\python.exe"

function Test-PyImport([string]$module) {
  & $PY -c "import $module" 2>$null
  return ($LASTEXITCODE -eq 0)
}

if (-not (Test-Path $PY)) {
  Write-Host "error: $PY not found." -ForegroundColor Red
  Write-Host "Create the env (Python 3.10-3.13, NOT 3.14) and install dependencies:"
  Write-Host "  py -3.12 -m venv `"$VENV`""
  Write-Host "  & `"$PY`" -m pip install --upgrade pip"
  Write-Host "  & `"$PY`" -m pip install -r `"$DIR\requirements.txt`" surya-ocr"
  exit 1
}

if (-not (Test-PyImport "streamlit")) {
  Write-Host "error: streamlit is not installed in $VENV." -ForegroundColor Red
  Write-Host "  & `"$PY`" -m pip install -r `"$DIR\requirements.txt`""
  exit 1
}

# Surya OCR is optional: without it, only the "Surya OCR" PDF modes are hidden;
# the rest of the app still runs (native PDF text, LM Studio, uploaded .txt, etc.).
$suryaOk = Test-PyImport "surya"
$llamaOk = [bool]((Get-Command llama-server.exe -ErrorAction SilentlyContinue) `
             -or (Get-Command llama-server -ErrorAction SilentlyContinue))
if ($suryaOk -and $llamaOk) {
  Write-Host "Surya OCR:    available (on-device, llama.cpp)"
} else {
  Write-Host "Surya OCR:    NOT ready  ->  put llama-server.exe on PATH; & `"$PY`" -m pip install surya-ocr" -ForegroundColor Yellow
}
Write-Host ("Interpreter:  " + (& $PY -c "import sys; print(sys.executable, '(Python %d.%d.%d)' % sys.version_info[:3])"))
Write-Host "Tip: stop any other Streamlit server first, or pass --server.port to use another port."
Write-Host ""

Set-Location $DIR
& $PY -m streamlit run (Join-Path $DIR "app.py") @args
