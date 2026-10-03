# Camber GPU setup

This repository can run its CUDA backend on a Camber GPU node.

## Expected hardware

Camber currently documents NVIDIA L4 (24 GB VRAM) as the on-demand GPU. The L4 is Ada-generation hardware and is compatible with this project's current CUDA architecture setting (`CUDA_ARCHITECTURES 89`).

## 1. Activate the student benefit

Use the Camber offer from GitHub Student Developer Pack and sign in to Camber with the GitHub account that owns the active student benefit.

Account activation and OAuth authorization are interactive account actions and must be completed by the account owner.

## 2. Select the personal workspace

After signing in, use your personal workspace unless you intentionally want a team workspace.

With Camber CLI:

```bash
camber team list
camber team select --personal
```

## 3. Start a GPU environment

In Camber/Nova, request an XSMALL node with GPU. For this renderer, one L4 is sufficient.

Suggested Nova request:

```text
Run this project on an XSMALL node with one GPU. I need an NVIDIA GPU plus a CUDA Toolkit containing nvcc, CMake, Git, and a C++17 compiler.
```

Camber documents `with_gpu=True` for GPU jobs.

## 4. Get the repository

The repository is public:

```bash
git clone https://github.com/q894829927/render.git
cd render
```

## 5. Validate and build

Run:

```bash
bash scripts/camber_gpu_build.sh 256 8 16
```

The script checks:

- `nvidia-smi`
- `nvcc --version`
- `cmake`
- `g++`

Then it configures and builds the CUDA target and runs the Cornell Box renderer.

For a very cheap smoke test first:

```bash
bash scripts/camber_gpu_build.sh 4 2 4
```

For the current high-quality test:

```bash
bash scripts/camber_gpu_build.sh 1024 8 16
```

## 6. Important failure case

Seeing an NVIDIA L4 in `nvidia-smi` does not prove that `nvcc` is installed. This project compiles CUDA C++, so the full CUDA Toolkit (or at least a toolchain containing `nvcc`) is required.

If `nvidia-smi` succeeds but `nvcc --version` fails, use a Camber environment/image with the CUDA Toolkit rather than changing the renderer to a CPU fallback.

## Output

The renderer currently writes its PPM outputs to the working directory. Copy important results into Camber Stash before ending an ephemeral job if the environment does not persist its working directory.
