#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.camber_toolchain.env"
CONDA_PREFIX_VALUE="${CONDA_PREFIX:-/opt/conda}"

echo "== Camber toolchain bootstrap =="

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: NVIDIA GPU is not visible. Connect Camber Hub to GPU Xsmall first."
  exit 20
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available. Expected Camber's /opt/conda environment."
  exit 21
fi

install_if_missing() {
  local tool="$1"
  shift
  if command -v "$tool" >/dev/null 2>&1; then
    return 0
  fi
  echo "Installing packages for $tool: $*"
  conda install -y "$@"
}

# Camber's base GPU image exposes the NVIDIA driver/runtime but may omit build tools.
install_if_missing cmake -c conda-forge cmake

if ! command -v g++ >/dev/null 2>&1 &&    [[ ! -x "$CONDA_PREFIX_VALUE/bin/x86_64-conda-linux-gnu-g++" ]]; then
  echo "Installing C++ compiler..."
  conda install -y -c conda-forge gxx_linux-64
fi

if ! command -v nvcc >/dev/null 2>&1 &&    [[ ! -x "$CONDA_PREFIX_VALUE/bin/nvcc" ]]; then
  echo "Installing CUDA compiler..."
  conda install -y -c nvidia cuda-nvcc
fi

CMAKE_BIN="$(command -v cmake || true)"
[[ -n "$CMAKE_BIN" ]] || CMAKE_BIN="$CONDA_PREFIX_VALUE/bin/cmake"

NVCC_BIN="$(command -v nvcc || true)"
[[ -n "$NVCC_BIN" ]] || NVCC_BIN="$CONDA_PREFIX_VALUE/bin/nvcc"

GXX_BIN="$(command -v g++ || true)"
if [[ -z "$GXX_BIN" && -x "$CONDA_PREFIX_VALUE/bin/x86_64-conda-linux-gnu-g++" ]]; then
  GXX_BIN="$CONDA_PREFIX_VALUE/bin/x86_64-conda-linux-gnu-g++"
fi

for pair in   "cmake:$CMAKE_BIN"   "nvcc:$NVCC_BIN"   "g++:$GXX_BIN"
do
  name="${pair%%:*}"
  path="${pair#*:}"
  if [[ -z "$path" || ! -x "$path" ]]; then
    echo "ERROR: $name is still unavailable after bootstrap."
    exit 22
  fi
done

echo
echo "== Toolchain =="
"$NVCC_BIN" --version
"$CMAKE_BIN" --version | head -n 1
"$GXX_BIN" --version | head -n 1

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
cat > "$TMP_DIR/smoke.cu" <<'EOF'
#include <cuda_runtime.h>
#include <cstdio>
__global__ void k(int* p) { if (threadIdx.x == 0) *p = 42; }
int main() {
    int* d = nullptr;
    int h = 0;
    if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) return 2;
    k<<<1,1>>>(d);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    if (cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
    cudaFree(d);
    std::printf("%d\n", h);
    return h == 42 ? 0 : 5;
}
EOF

ALLOW_UNSUPPORTED=0
echo
echo "== nvcc compile/run smoke test =="
set +e
"$NVCC_BIN" -ccbin "$GXX_BIN" "$TMP_DIR/smoke.cu" -o "$TMP_DIR/smoke" >"$TMP_DIR/build.log" 2>&1
status=$?
set -e

if [[ $status -ne 0 ]]; then
  if grep -qi "unsupported GNU version\|unsupported compiler" "$TMP_DIR/build.log"; then
    echo "Host compiler is newer than nvcc's validated range; retrying with --allow-unsupported-compiler."
    "$NVCC_BIN" --allow-unsupported-compiler -ccbin "$GXX_BIN"       "$TMP_DIR/smoke.cu" -o "$TMP_DIR/smoke"
    ALLOW_UNSUPPORTED=1
  else
    cat "$TMP_DIR/build.log"
    echo "ERROR: nvcc smoke compilation failed."
    exit 23
  fi
fi

"$TMP_DIR/smoke"

cat > "$ENV_FILE" <<EOF
export CAMBER_CMAKE="$CMAKE_BIN"
export CAMBER_NVCC="$NVCC_BIN"
export CAMBER_CXX="$GXX_BIN"
export CAMBER_NVCC_ALLOW_UNSUPPORTED="$ALLOW_UNSUPPORTED"
EOF

echo
echo "Toolchain ready."
echo "Saved: $ENV_FILE"
