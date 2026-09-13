<#
.SYNOPSIS
    One-command setup for the Diabetic Retinopathy project on Windows.
    File location: <project_root>\setup.ps1

.DESCRIPTION
    Creates a virtual environment in .venv, installs PyTorch with the correct
    CUDA wheels, installs the remaining requirements, and runs the environment
    self-check.

.EXAMPLE
    # From VS Code: Terminal > New Terminal, then:
    powershell -ExecutionPolicy Bypass -File .\setup.ps1

.EXAMPLE
    # Force a specific CUDA build, or CPU-only:
    powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Cuda cu118
    powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Cuda cpu

.NOTES
    Safe to re-run. If .venv already exists it is reused unless -Recreate is passed.
#>

param(
    [ValidateSet("auto", "cu126", "cu121", "cu118", "cpu")]
    [string]$Cuda = "auto",

    [switch]$Recreate,

    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ProjectRoot

function Write-Step($msg)  { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "    OK  $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "    !   $msg" -ForegroundColor Yellow }
function Fail($msg)        { Write-Host "`nFAILED: $msg" -ForegroundColor Red; exit 1 }

Write-Host "======================================================================"
Write-Host "  Diabetic Retinopathy project - Windows setup"
Write-Host "  Project root: $ProjectRoot"
Write-Host "======================================================================"

# --- 1. Python -------------------------------------------------------------
Write-Step "Checking Python"
try {
    $pyVersion = & $Python --version 2>&1
} catch {
    Fail "'$Python' was not found on PATH. Install Python 3.10-3.12 from python.org and tick 'Add python.exe to PATH'."
}
Write-Ok $pyVersion
if ($pyVersion -match "3\.(1[3-9]|[0-9]{3})") {
    Write-Warn2 "Python 3.13+ often has no prebuilt torch/timm wheels yet. Python 3.11 is the safe choice."
}

# --- 2. Virtual environment ------------------------------------------------
Write-Step "Creating virtual environment (.venv)"
if ($Recreate -and (Test-Path ".venv")) {
    Write-Warn2 "Removing existing .venv (-Recreate)"
    Remove-Item -Recurse -Force ".venv"
}
if (Test-Path ".venv") {
    Write-Ok ".venv already exists - reusing it (pass -Recreate to rebuild)"
} else {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "venv creation failed." }
    Write-Ok ".venv created"
}

$VenvPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { Fail "Expected interpreter not found at $VenvPy" }

Write-Step "Upgrading pip"
& $VenvPy -m pip install --upgrade pip --quiet
Write-Ok "pip upgraded"

# --- 3. Detect CUDA --------------------------------------------------------
Write-Step "Detecting NVIDIA driver / CUDA"
$Target = $Cuda
if ($Cuda -eq "auto") {
    $smi = $null
    try { $smi = & nvidia-smi 2>$null } catch { }
    if (-not $smi) {
        Write-Warn2 "nvidia-smi not found. Falling back to the CPU-only build."
        Write-Warn2 "If you DO have an NVIDIA GPU, install its driver from nvidia.com, then re-run with -Cuda cu121."
        $Target = "cpu"
    } else {
        $line = ($smi | Select-String "CUDA Version").ToString()
        if ($line -match "CUDA Version:\s*([0-9]+)\.([0-9]+)") {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            Write-Ok "Driver reports CUDA $major.$minor"
            # Pick the newest wheel set the driver can run.
            if     ($major -ge 12 -and $minor -ge 6) { $Target = "cu126" }
            elseif ($major -ge 12)                   { $Target = "cu121" }
            elseif ($major -eq 11 -and $minor -ge 8) { $Target = "cu118" }
            else {
                Write-Warn2 "Driver is older than CUDA 11.8 - update your NVIDIA driver. Using CPU build for now."
                $Target = "cpu"
            }
        } else {
            Write-Warn2 "Could not parse the CUDA version from nvidia-smi. Defaulting to cu121."
            $Target = "cu121"
        }
    }
}
Write-Ok "PyTorch build selected: $Target"

# --- 4. Install PyTorch ----------------------------------------------------
Write-Step "Installing PyTorch ($Target) - this downloads ~2.5 GB, be patient"
if ($Target -eq "cpu") {
    & $VenvPy -m pip install torch torchvision
} else {
    & $VenvPy -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Target"
}
if ($LASTEXITCODE -ne 0) {
    Fail "PyTorch install failed. Try a different build, e.g.: .\setup.ps1 -Cuda cu118"
}
Write-Ok "PyTorch installed"

# --- 5. Install everything else -------------------------------------------
Write-Step "Installing project requirements"
& $VenvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  Requirements install failed. Most common cause on Windows:" -ForegroundColor Red
    Write-Host "    'Microsoft Visual C++ 14.0 or greater is required'"
    Write-Host "  That means pip could not find a prebuilt wheel for your Python version"
    Write-Host "  and fell back to compiling C++ source, which needs a compiler."
    Write-Host ""
    Write-Host "  Your interpreter: " -NoNewline
    & $VenvPy --version
    Write-Host "  Python 3.13+ is missing wheels for several packages. Python 3.11 is safest."
    Write-Host "  Fix: install Python 3.11 from python.org, then:"
    Write-Host "       .\setup.ps1 -Recreate -Python `"C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python311\python.exe`""
    Fail "requirements.txt install failed - read the error above."
}
Write-Ok "Requirements installed"

# Optional extras: nice to have, never required. A failure here is NOT fatal.
Write-Step "Installing optional extras (albumentations, ruff) - failures are safe to ignore"
& $VenvPy -m pip install -r requirements-optional.txt
if ($LASTEXITCODE -ne 0) {
    Write-Warn2 "Optional extras failed to build - skipping them."
    Write-Warn2 "Nothing in this project depends on them; augmentation uses torchvision.transforms.v2."
} else {
    Write-Ok "Optional extras installed"
}

# --- 6. Kaggle credentials reminder ---------------------------------------
Write-Step "Checking Kaggle credentials"
$KagglePath = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (Test-Path $KagglePath) {
    Write-Ok "Found $KagglePath"
} else {
    Write-Warn2 "kaggle.json not found at $KagglePath"
    Write-Host "    1. Open https://www.kaggle.com/competitions/aptos2019-blindness-detection and click 'Join Competition'"
    Write-Host "    2. Kaggle > Account > Create New API Token (downloads kaggle.json)"
    Write-Host "    3. Run:  mkdir `"$env:USERPROFILE\.kaggle`" ; move `"$env:USERPROFILE\Downloads\kaggle.json`" `"$env:USERPROFILE\.kaggle\`""
}

# --- 7. Self-check ---------------------------------------------------------
Write-Step "Running environment self-check"
$env:PYTHONPATH = $ProjectRoot
& $VenvPy -m scripts.check_env

Write-Host "`n======================================================================"
Write-Host "  Setup finished."
Write-Host "  In VS Code: Ctrl+Shift+P > 'Python: Select Interpreter' > .venv"
Write-Host "  Then press F5 and choose a milestone to run."
Write-Host "======================================================================`n"
