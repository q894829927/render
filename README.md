# render

一个用于学习和验证现代路径追踪（Path Tracing）的实验性渲染器。

当前目标：

- CPU 与 CUDA 双后端
- Cook–Torrance PBR
- GGX VNDF Sampling
- Diffuse + Specular 混合 BSDF
- 面光源（Area Light）与下一事件估计（NEE）
- 多重重要性采样（MIS）
- 俄罗斯轮盘赌（Russian Roulette）
- 渐进式累积（Progressive Accumulation）
- 同采样域 Guide
- 方差引导 A-Trous 降噪

## 目录

```text
Core/
├─ Math
├─ Scene
├─ Material
├─ BSDF
├─ Integrator
├─ Sampling
└─ Denoiser

Backend/
├─ CPU
└─ CUDA
```

## 构建

CPU：

```bash
cmake -S . -B build -DRENDER_ENABLE_CUDA=OFF
cmake --build build --config Release
```

CUDA：

```bash
cmake -S . -B build -DRENDER_ENABLE_CUDA=ON
cmake --build build --config Release
```
