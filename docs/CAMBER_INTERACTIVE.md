# Camber interactive GPU workflow

Use this workflow when Camber's background Job API returns `403 code 1005` but the Hub UI lets you select **GPU Xsmall**.

This path does **not** call `CamberJob.create()`.

## 1. Connect the Hub to GPU Xsmall

In the Camber left sidebar / compute selector:

```text
GPU
Xsmall — 8 CPU / 32GB RAM / 1 GPU
```

Wait until the connection is active.

## 2. Open a Terminal in the same Hub session

From the repository root, run:

```bash
bash scripts/camber_interactive_gpu.sh 4 2 4
```

This smoke test checks:

- current Camber identity / personal workspace
- `nvidia-smi`
- `nvcc`
- `cmake`
- `g++`
- CUDA CMake configuration
- `render_cuda` build
- a 4 SPP render
- upload of generated PPM/PNG files to personal Stash

If the smoke test passes:

```bash
bash scripts/camber_interactive_gpu.sh 256 8 16
```

For the high-quality test:

```bash
bash scripts/camber_interactive_gpu.sh 1024 8 16
```

## 3. Important: stop the interactive GPU afterwards

The Hub GPU connection is interactive and may remain allocated after the render command exits.

After results are safely in Stash, use the Camber UI to **Stop connection / Disconnect** the GPU Xsmall session.

Do not rely on closing the browser tab to release the GPU.

## Why this workflow exists

The Student plan can expose GPU Xsmall in the Hub while the background compute Job API may still reject `CamberJob.create(..., with_gpu=True)` with:

```text
HTTP 403
code 1005
Job creation is not available on the free tier
```

That is a backend entitlement/API issue, not a renderer issue. The interactive workflow bypasses that API path while still using the Student GPU quota.
