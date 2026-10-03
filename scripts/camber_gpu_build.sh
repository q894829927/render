#!/usr/bin/env bash
set -euo pipefail

SPP="${1:-256}"
SAMPLES_PER_PASS="${2:-8}"
MAX_DEPTH="${3:-16}"
BUILD_DIR="${BUILD_DIR:-build-camber}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.camber_toolchain.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.camber_toolchain.env"
fi

CMAKE_BIN="${CAMBER_CMAKE:-$(command -v cmake || true)}"
NVCC_BIN="${CAMBER_NVCC:-$(command -v nvcc || true)}"
CXX_BIN="${CAMBER_CXX:-$(command -v g++ || true)}"
PYTHON_BIN="$(command -v python3 || command -v python || true)"

if [[ -z "$CXX_BIN" && -x /opt/conda/bin/x86_64-conda-linux-gnu-g++ ]]; then
  CXX_BIN=/opt/conda/bin/x86_64-conda-linux-gnu-g++
fi
if [[ -z "$CMAKE_BIN" && -x /opt/conda/bin/cmake ]]; then
  CMAKE_BIN=/opt/conda/bin/cmake
fi
if [[ -z "$NVCC_BIN" && -x /opt/conda/bin/nvcc ]]; then
  NVCC_BIN=/opt/conda/bin/nvcc
fi

echo "== Camber GPU environment check =="
echo "Host: $(hostname)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Connect to a GPU Xsmall environment."
  exit 3
fi
nvidia-smi

for pair in "cmake:$CMAKE_BIN" "nvcc:$NVCC_BIN" "g++:$CXX_BIN"; do
  name="${pair%%:*}"
  path="${pair#*:}"
  if [[ -z "$path" || ! -x "$path" ]]; then
    echo "ERROR: $name is unavailable."
    echo "Run: bash scripts/camber_bootstrap.sh"
    exit 4
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: Python is unavailable; PNG conversion requires Python 3."
  exit 4
fi
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' || {
  echo "ERROR: PNG conversion requires Python 3.8 or newer."
  exit 4
}

echo
"$NVCC_BIN" --version
"$CMAKE_BIN" --version | head -n 1
"$CXX_BIN" --version | head -n 1

CUDA_FLAGS=""
if [[ "${CAMBER_NVCC_ALLOW_UNSUPPORTED:-0}" == "1" ]]; then
  CUDA_FLAGS="--allow-unsupported-compiler"
fi

echo
echo "== Configure =="
"$CMAKE_BIN" -S . -B "$BUILD_DIR"   -DRENDER_ENABLE_CUDA=ON   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_CXX_COMPILER="$CXX_BIN"   -DCMAKE_CUDA_COMPILER="$NVCC_BIN"   -DCMAKE_CUDA_HOST_COMPILER="$CXX_BIN"   -DCMAKE_CUDA_FLAGS="$CUDA_FLAGS"

echo
echo "== Build =="
"$CMAKE_BIN" --build "$BUILD_DIR" --config Release -j"$(nproc)"

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
outputs=(
  cornell_cuda_raw.ppm
  cornell_cuda_diffuse.ppm
  cornell_cuda_specular.ppm
  cornell_cuda_final.ppm
)
for file in "${outputs[@]}"; do
  if [[ ! -s "$file" ]]; then
    echo "ERROR: missing or empty GPU render output: $file"
    exit 6
  fi
done

"$PYTHON_BIN" scripts/ppm_to_png.py "${outputs[@]}"
for file in "${outputs[@]}"; do
  png="${file%.ppm}.png"
  if [[ ! -s "$png" ]]; then
    echo "ERROR: missing or empty PNG output: $png"
    exit 7
  fi
  echo "$png"
done
