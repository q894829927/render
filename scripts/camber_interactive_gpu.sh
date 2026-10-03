#!/usr/bin/env bash
set -euo pipefail

SPP="${1:-4}"
SAMPLES_PER_PASS="${2:-2}"
MAX_DEPTH="${3:-4}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== Camber interactive GPU workflow =="

if command -v camber >/dev/null 2>&1; then
  camber team select --personal >/dev/null 2>&1 || true
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: GPU is not visible. Select GPU Xsmall in the Camber Hub first."
  exit 10
fi

# New Camber sessions may contain only the GPU driver/runtime.
# Bootstrap the user-space CUDA/C++ toolchain when required.
need_bootstrap=0
command -v cmake >/dev/null 2>&1 || need_bootstrap=1
command -v nvcc >/dev/null 2>&1 || [[ -x /opt/conda/bin/nvcc ]] || need_bootstrap=1
command -v g++ >/dev/null 2>&1 || [[ -x /opt/conda/bin/x86_64-conda-linux-gnu-g++ ]] || need_bootstrap=1

if [[ $need_bootstrap -eq 1 || ! -f "$ROOT_DIR/.camber_toolchain.env" ]]; then
  bash "$ROOT_DIR/scripts/camber_bootstrap.sh"
fi

bash "$ROOT_DIR/scripts/camber_gpu_build.sh" "$SPP" "$SAMPLES_PER_PASS" "$MAX_DEPTH"

echo
echo "== Persist outputs to Stash =="

if command -v camber >/dev/null 2>&1; then
  USERNAME="$(camber me 2>/dev/null | awk -F: '/Username:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')"
  STASH_ROOT="${CAMBER_STASH_ROOT:-}"

  if [[ -z "$STASH_ROOT" && -n "$USERNAME" ]]; then
    STASH_ROOT="stash://$USERNAME/renders/cornell/"
  fi

  if [[ -n "$STASH_ROOT" ]]; then
    camber stash mkdir "$STASH_ROOT" >/dev/null 2>&1 || true
    shopt -s nullglob
    outputs=( *.ppm *.png )
    for file in "${outputs[@]}"; do
      echo "Uploading $file -> $STASH_ROOT"
      camber stash cp "$file" "$STASH_ROOT"
    done
    shopt -u nullglob
  else
    echo "WARNING: could not resolve personal Stash path."
  fi
else
  echo "WARNING: Camber CLI unavailable; output files remain in the current workspace."
fi

echo
echo "Render workflow complete."
echo "IMPORTANT: this is an interactive Hub GPU. Stop/Disconnect the GPU connection when finished."
