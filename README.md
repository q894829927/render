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

第四个可选参数选择采样器，第五个可选参数选择 Film Reconstruction Filter：

```bash
./build/render_cpu 256 8 16 owen tent   # 默认
./build/render_cpu 256 8 16 owen box    # Box reconstruction reference
./build/render_cpu 256 8 16 hash tent   # deterministic sampler reference
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
