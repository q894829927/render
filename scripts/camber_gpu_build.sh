#!/usr/bin/env bash
set -euo pipefail

SPP="${1:-256}"
SAMPLES_PER_PASS="${2:-8}"
MAX_DEPTH="${3:-16}"
BUILD_DIR="${BUILD_DIR:-build-camber}"

echo "== Camber GPU environment check =="
echo "Host: $(hostname)"
echo

for tool in git cmake g++; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: required tool '$tool' was not found in PATH."
    exit 2
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "ERROR: nvidia-smi not found. This job does not appear to have an NVIDIA GPU."
  exit 3
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo
  echo "ERROR: nvcc was not found."
  echo "The GPU is visible, but a CUDA Toolkit with the CUDA compiler is required to build Backend/CUDA/main.cu."
  echo "Run this script in a Camber GPU environment/image that includes the CUDA Toolkit."
  exit 4
fi

echo
nvcc --version
cmake --version | head -n 1
g++ --version | head -n 1

echo
echo "== Configure =="
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
  exit 5
fi

echo
echo "== Render =="
echo "SPP=$SPP SamplesPerPass=$SAMPLES_PER_PASS MaxDepth=$MAX_DEPTH"
"$BIN" "$SPP" "$SAMPLES_PER_PASS" "$MAX_DEPTH"

echo
echo "== Outputs =="
find . -maxdepth 2 -type f \( -name "*.ppm" -o -name "*.png" \) -print
