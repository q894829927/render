# render

一个用于学习现代路径追踪（Path Tracing）的 CPU / CUDA 双后端渲染器。

当前实现：

- Cook–Torrance PBR
- GGX VNDF Sampling
- Diffuse + Specular 混合 BSDF
- 矩形面光源（Area Light）
- 下一事件估计（NEE）
- 多重重要性采样（MIS）
- 俄罗斯轮盘赌（Russian Roulette）
- 渐进式累积（Progressive Accumulation）
- 确定性 SampleGenerator + 4 个 RQMC Replicates
- Owen-scrambled Sobol（默认）+ deterministic hash reference sampler
- Visibility-aware Reconstruction（coverage variance / confidence）
- Film Reconstruction Filter：Tent（默认）/ Box（reference）
- 同采样域 Primary Guide
- RQMC sampling-variance 驱动的 Adaptive A-Trous Denoiser
- Linear HDR PFM 输出与 ROI Quantitative Regression
- Cornell Box
- CPU 多线程后端
- CUDA 后端

详细目录职责见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 构建 CPU

```bash
cmake -S . -B build -DRENDER_ENABLE_CUDA=OFF
cmake --build build --config Release
```

运行：

```bash
./build/render_cpu 256 8 16
```

前三个参数依次是：`SPP`、`Samples Per Pass`、`Max Depth`。

第四个可选参数选择采样器，第五个可选参数选择 Film Reconstruction Filter；第六个可选参数 `debug` 输出 Debug AOV：

```bash
./build/render_cpu 256 8 16 owen tent   # 默认
./build/render_cpu 256 8 16 owen box    # Box reconstruction reference
./build/render_cpu 256 8 16 hash tent   # deterministic sampler reference
./build/render_cpu 256 8 16 owen tent debug
```

Tent 使用归一化三角核，并直接在 Camera sample 阶段 importance-sample reconstruction kernel；不是对最终 PNG 做 blur。

`Samples Per Pass` 只影响调度；相同 SPP 下不会改变样本集合或最终结果。

## 构建 CUDA

需要 CUDA Toolkit 与支持 CUDA 的 NVIDIA GPU。

```bash
cmake -S . -B build -DRENDER_ENABLE_CUDA=ON
cmake --build build --config Release
```

Windows / Visual Studio 2022 可使用 CMake 打开文件夹，也可以直接让 VS 生成 CMake 工程。

运行：

```bash
./build/render_cuda 256 8 16
```

RTX 40 系使用 Ada（sm_89）。如果不是 RTX 40 系，请修改 `CMakeLists.txt` 中的 `CUDA_ARCHITECTURES`。

## 输出

CPU：

```text
cornell_cpu_raw.ppm
cornell_cpu_diffuse.ppm
cornell_cpu_specular.ppm
cornell_cpu_final.ppm
```

CUDA：

```text
cornell_cuda_raw.ppm
cornell_cuda_diffuse.ppm
cornell_cuda_specular.ppm
cornell_cuda_final.ppm
```

在 Camber GPU 会话中运行 `bash scripts/camber_interactive_gpu.sh 256 8 16`
会构建 CUDA 后端、实际使用 GPU 渲染，并把这四张图转换为 PNG 上传到个人 Stash。
GitHub Actions 的 `CUDA Compile Test` 只验证 CUDA 编译；普通托管 runner 不执行 GPU 渲染。


## Linear HDR Validation

CPU renderer 同时输出 display PPM 和未 tone-map 的 Linear HDR PFM：

```text
cornell_cpu_raw.pfm
cornell_cpu_diffuse.pfm
cornell_cpu_specular.pfm
cornell_cpu_final.pfm
```

普通 GitHub Render Validation 会对 16 / 64 / 256 SPP 生成 PFM + PNG，并由 `scripts/render_metrics.py` 计算：

```text
MSE / RMSE / MAE / MaxAbs / NRMSE
Tone-mapped PSNR
```

指标覆盖 full frame、Ceiling Light Border、Short Box Silhouette、Tall Box Silhouette。

普通 push 的 256 SPP Raw 只作为同次运行的 provisional convergence reference。需要更高质量 reference 时，手动运行 `High Quality Reference` workflow，选择 1024 或 2048 SPP。


## Debug AOV

CPU debug 模式：

```bash
./build/render_cpu 256 8 16 owen tent debug
```

额外生成：

```text
cornell_cpu_coverage.ppm
cornell_cpu_coverage_confidence.ppm
cornell_cpu_variance.ppm
cornell_cpu_surface_group.ppm
cornell_cpu_normal.ppm
cornell_cpu_depth.ppm
cornell_cpu_filter_strength.ppm
```

`filter_strength` 是四轮 Adaptive A-Trous 的累计有效强度，而不是最后一轮瞬时值。

GitHub Render Validation 会在 runner 中验证这些 AOV。普通成功 push 只上传核心 PFM/PNG/metrics；失败或手动 workflow_dispatch 时上传完整 Debug AOV artifact。

## Phase E Status

Sampling → SurfaceIdentity → Visibility Reconstruction → Tent Film Filter → Adaptive A-Trous → Linear HDR Regression → Debug AOV 的 Phase E 已完成。

自动验收包括：

```text
CPU Smoke
Sample Determinism
16 / 64 / 256 SPP Render Validation
Full-frame + ROI HDR Regression
CUDA Compile Test
```

下一阶段可以开始 Triangle / Mesh / OBJ / BVH。
