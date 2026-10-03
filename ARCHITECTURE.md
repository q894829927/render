# Architecture

本项目把“渲染算法”和“执行后端”分离：Core 只描述数学、场景、材质、采样、积分器与降噪规则；Backend 决定这些规则由 CPU 线程还是 CUDA Kernel 执行。

```text
Core/
├─ Math/Math.h               # Vec3、Ray、ONB、常量与基础数学
├─ Scene/Scene.h             # 几何、求交、SceneView
├─ Scene/CornellBox.h        # Cornell Box 与相机配置
├─ Material/Material.h       # Metallic/Roughness PBR 材质
├─ BSDF/BSDF.h               # Cook–Torrance、GGX、Fresnel
├─ Sampling/Sampling.h       # Diffuse/GGX VNDF、Light Sampling、MIS
├─ Integrator/Integrator.h   # Path Tracing、RR、Primary Guide
└─ Denoiser/Denoiser.h       # Guide、Confidence、A-Trous 权重规则

Backend/
├─ CPU/main.cpp              # 多线程 Progressive + CPU A-Trous
└─ CUDA/main.cu              # CUDA Progressive + CUDA A-Trous
```

## 关键约束

1. **CPU/CUDA 使用同一份 PBR、BSDF、VNDF、MIS 和 Integrator 逻辑。**
2. **Radiance 与 Guide 来自同一条 jittered primary ray。** 不再单独生成单样本 G-Buffer。
3. **不使用二值 Edge Mask。** 边缘由 Coverage、Normal Coherence、Depth Coherence 形成连续 Confidence。
4. **降噪在 Linear HDR 中进行。** ACES 与 sRGB 只在最终输出阶段执行。
5. 后续 BVH、Triangle Mesh、OBJ/GLTF 应优先加入 Core/Scene；后端只负责数据上传和并行调度。
