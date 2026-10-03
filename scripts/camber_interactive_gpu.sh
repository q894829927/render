#!/usr/bin/env bash
set -euo pipefail

SPP="${1:-4}"
SAMPLES_PER_PASS="${2:-2}"
MAX_DEPTH="${3:-4}"
BUILD_DIR="${BUILD_DIR:-build-camber-interactive}"

echo "== Camber interactive GPU setup =="
echo "Working directory: $(pwd)"
echo

# Keep work in the personal workspace when the CLI is available.
if command -v camber >/dev/null 2>&1; then
  echo "== Camber identity =="
  camber me || true
  echo
  echo "== Select personal workspace =="
  camber team select --personal || true
  echo
fi

echo "== GPU check =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found."
  echo "Connect the Hub to GPU Xsmall first, then run this script again."
  exit 10
fi
nvidia-smi

echo
echo "== CUDA compiler check =="

# Some environments have CUDA installed but nvcc is not on PATH.
if ! command -v nvcc >/dev/null 2>&1; then
  for candidate in     /usr/local/cuda/bin/nvcc     /usr/local/cuda-12/bin/nvcc     /usr/local/cuda-12.8/bin/nvcc     /usr/local/cuda-12.6/bin/nvcc     /usr/local/cuda-12.4/bin/nvcc
  do
    if [[ -x "$candidate" ]]; then
      export PATH="$(dirname "$candidate"):$PATH"
      break
    fi
  done
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: NVIDIA GPU is visible, but nvcc is not available."
  echo "This Hub image has a CUDA runtime/driver but not a CUDA Toolkit compiler."
  echo "Do not switch to CPU. Use a Camber environment/image that includes nvcc."
  exit 11
fi

nvcc --version

echo
echo "== Host toolchain check =="
for tool in cmake g++; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: required tool '$tool' is missing."
    exit 12
  fi
done
cmake --version | head -n 1
g++ --version | head -n 1

echo
echo "== Configure CUDA backend =="
cmake -S . -B "$BUILD_DIR"   -DRENDER_ENABLE_CUDA=ON   -DCMAKE_BUILD_TYPE=Release

echo
echo "== Build =="
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"

BIN="$BUILD_DIR/render_cuda"
if [[ ! -x "$BIN" && -x "$BUILD_DIR/Release/render_cuda" ]]; then
  BIN="$BUILD_DIR/Release/render_cuda"
fi

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: render_cuda executable was not produced."
  exit 13
fi

echo
echo "== Render =="
echo "SPP=$SPP SamplesPerPass=$SAMPLES_PER_PASS MaxDepth=$MAX_DEPTH"
"$BIN" "$SPP" "$SAMPLES_PER_PASS" "$MAX_DEPTH"

echo
echo "== Persist outputs to Stash =="

STASH_ROOT="${CAMBER_STASH_ROOT:-}"

if command -v camber >/dev/null 2>&1; then
  if [[ -z "$STASH_ROOT" ]]; then
    # Resolve the username from 'camber me' output when possible.
    USERNAME="$(camber me 2>/dev/null | awk -F: '/Username:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')"
    if [[ -n "$USERNAME" ]]; then
      STASH_ROOT="stash://$USERNAME/renders/cornell/"
    fi
  fi

  if [[ -n "$STASH_ROOT" ]]; then
    camber stash mkdir "$STASH_ROOT" >/dev/null 2>&1 || true

    shopt -s nullglob
    outputs=( *.ppm *.png )
    if (( ${#outputs[@]} > 0 )); then
      for file in "${outputs[@]}"; do
        echo "Uploading $file -> $STASH_ROOT"
        camber stash cp "$file" "$STASH_ROOT"
      done
    else
      echo "WARNING: no .ppm/.png outputs were found in the current directory."
    fi
    shopt -u nullglob
  else
    echo "WARNING: could not resolve a Stash path automatically."
    echo "Set CAMBER_STASH_ROOT=stash://<username>/renders/cornell/ and rerun."
  fi
else
  echo "WARNING: Camber CLI not found, so outputs were not copied to Stash."
fi

echo
echo "== Done =="
echo "The renderer finished successfully."
echo "When you are done using the interactive GPU, stop/disconnect the Hub GPU connection manually to stop consuming GPU quota."
