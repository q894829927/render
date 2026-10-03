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
- 同采样域 Primary Guide
- 方差引导 A-Trous Denoiser
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

三个参数依次是：`SPP`、`Samples Per Pass`、`Max Depth`。

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
cornell_cpu_denoised.ppm
cornell_cpu_guide_confidence.ppm
```

CUDA：

```text
cornell_cuda_raw.ppm
cornell_cuda_denoised.ppm
```
