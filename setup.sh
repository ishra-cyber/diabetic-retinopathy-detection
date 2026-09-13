#!/usr/bin/env bash
# One-command setup for Linux / macOS.
# File location: <project_root>/setup.sh
# Usage:  bash setup.sh [cu126|cu121|cu118|cpu]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
TARGET="${1:-auto}"

step() { printf "\n==> %s\n" "$1"; }
ok()   { printf "    OK  %s\n" "$1"; }
warn() { printf "    !   %s\n" "$1"; }
fail() { printf "\nFAILED: %s\n" "$1"; exit 1; }

step "Checking Python"
command -v python3 >/dev/null || fail "python3 not found."
ok "$(python3 --version)"

step "Creating virtual environment (.venv)"
[ -d .venv ] || python3 -m venv .venv
ok ".venv ready"
VENV_PY="$PROJECT_ROOT/.venv/bin/python"
"$VENV_PY" -m pip install --upgrade pip --quiet

step "Detecting CUDA"
if [ "$TARGET" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_LINE="$(nvidia-smi | grep -o 'CUDA Version: [0-9]*\.[0-9]*' || true)"
    MAJOR="$(echo "$CUDA_LINE" | grep -o '[0-9]*\.[0-9]*' | cut -d. -f1 || echo 0)"
    MINOR="$(echo "$CUDA_LINE" | grep -o '[0-9]*\.[0-9]*' | cut -d. -f2 || echo 0)"
    if   [ "${MAJOR:-0}" -ge 12 ] && [ "${MINOR:-0}" -ge 6 ]; then TARGET=cu126
    elif [ "${MAJOR:-0}" -ge 12 ]; then TARGET=cu121
    elif [ "${MAJOR:-0}" -eq 11 ] && [ "${MINOR:-0}" -ge 8 ]; then TARGET=cu118
    else warn "Driver older than CUDA 11.8 - using CPU build"; TARGET=cpu; fi
  else
    warn "nvidia-smi not found - using CPU build"; TARGET=cpu
  fi
fi
ok "PyTorch build: $TARGET"

step "Installing PyTorch"
if [ "$TARGET" = "cpu" ]; then
  "$VENV_PY" -m pip install torch torchvision
else
  "$VENV_PY" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$TARGET"
fi

step "Installing project requirements"
"$VENV_PY" -m pip install -r requirements.txt

step "Installing optional extras (failures are safe to ignore)"
"$VENV_PY" -m pip install -r requirements-optional.txt || \
  warn "Optional extras failed - skipping. Nothing in the project depends on them."

step "Running environment self-check"
PYTHONPATH="$PROJECT_ROOT" "$VENV_PY" -m scripts.check_env

printf "\nSetup finished. Activate with:  source .venv/bin/activate\n"
