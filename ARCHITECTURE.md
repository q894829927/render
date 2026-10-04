# Architecture

本项目把 **渲染算法（Core）** 与 **执行后端（Backend）** 分离。Phase E 完成后，CPU 是图像形成与自动化验收的运行时基准后端；CUDA 保留同一套核心算法定义并做 compile-only CI，GPU runtime 暂不作为 Phase E 验收条件。

## 目录与职责

~~~text
Core/
├─ Math/
│  └─ Math.h
│     Vec3 / Ray / ONB / 数学常量
│
├─ Scene/
│  ├─ Scene.h
│  │  Geometry / HitRecord / SurfaceIdentity / SceneView
│  └─ CornellBox.h
│     Cornell Box / Camera / stable instance-material-group IDs
│
├─ Material/
│  └─ Material.h
│     baseColor / metallic / roughness
│
├─ BSDF/
│  └─ BSDF.h
│     Cook-Torrance / GGX / Smith / Schlick
│
├─ Sampling/
│  ├─ SampleDimensions.h
│  │  固定 Camera / Light / BSDF / RR dimension registry
│  ├─ SobolParameters.h
│  ├─ Sobol.h
│  │  Sobol + fast Owen-style scramble
│  ├─ SampleGenerator.h
│  │  pixel / replicate / sample-within-replicate / dimension
│  └─ Sampling.h
│     Light sampling / cosine / GGX VNDF / MIS
│
├─ Integrator/
│  ├─ PathSample.h
│  │  Diffuse / Specular / Emission / Primary Surface
│  └─ Integrator.h
│     Path tracing / MIS / RR / signal classification
│
├─ Reconstruction/
│  ├─ ReconstructionFilter.h
│  │  Box / Tent filter
│  ├─ Film.h
│  │  continuous raster sample
│  └─ Reconstruction.h
│     surface layers / visibility / replicate uncertainty / confidence
│
├─ Denoiser/
│  ├─ Denoiser.h
│  │  geometry/color weights + adaptive filter decision
│  └─ ATrous.h
│     diffuse/specular independent adaptive A-Trous
│
└─ Output/
   ├─ ImageIO.h
   │  ACES display PPM + unit-range debug PPM
   ├─ PFM.h
   │  Linear HDR RGB float32
   └─ DebugAOV.h
      Coverage / Confidence / Variance / SurfaceGroup /
      Normal / Depth / Effective Filter Strength

Backend/
├─ CPU/main.cpp
│  Progressive render + Reconstruction + Adaptive A-Trous +
│  PFM/PPM/Debug AOV output
└─ CUDA/main.cu
   CUDA execution backend，使用同一 Core 算法定义

Tests/
├─ SampleGeneratorTest.cpp
├─ SurfaceIdentityTest.cpp
├─ ReconstructionTest.cpp
├─ AdaptiveDenoiserTest.cpp
├─ PFMTest.cpp
└─ DebugAOVTest.cpp

scripts/
├─ ppm_to_png.py
└─ render_metrics.py

.github/workflows/
├─ cpu-smoke.yml
├─ sample-determinism.yml
├─ render-preview.yml
├─ render-reference.yml
└─ cuda-compile.yml
~~~

## 图像形成数据流

~~~text
Global Sample Index
        ↓
SampleGenerator
pixel / replicate / sample / dimension
        ↓
Owen-scrambled Sobol
        ↓
Film Reconstruction Filter
Tent(default) / Box(reference)
        ↓
Continuous Raster Sample
        ↓
Primary Ray
        ↓
Path Integrator
        ↓
PathSample
├─ Diffuse
├─ Specular
├─ Emission
└─ PrimarySurfaceSample
        ↓
SurfaceIdentity
instance / primitive / material / surfaceGroup
        ↓
Visibility-aware Reconstruction
├─ Surface Layers
├─ Coverage
├─ Coverage Variance
├─ Coverage Confidence
└─ RQMC Replicate Statistics
        ↓
Diffuse demodulation     Specular signal
        ↓                     ↓
Adaptive A-Trous        Adaptive A-Trous
        └──────────┬──────────┘
                   ↓
                Compose
                   ↓
        Linear HDR Framebuffer
          ├─ PFM float32
          └─ ACES + gamma → PPM/PNG
~~~

## Sampling invariants

### Deterministic dimension model

随机变量不能依赖“之前消费了多少随机数”。每一个用途都有固定 dimension：

~~~text
Camera X/Y
Light U/V
Direct-BSDF
Path-BSDF
Russian Roulette
~~~

`SamplesPerPass` 只决定调度，不改变：

- sample identity；
- sample sequence；
- accumulation order；
- final Linear framebuffer。

CI 会在 64 SPP 下对 pass = 1 / 2 / 4 / 8 / 16 做 exact regression。

### RQMC replicates

当前固定使用 4 个独立 replicate：

~~~text
global sample
    ↓ round-robin
replicateId
    ↓
sampleIndexWithinReplicate
    ↓
independent pixel/replicate scramble
~~~

RQMC uncertainty 来自 **replicate means 之间的差异**，不把同一 Sobol replicate 内的样本错误解释成 IID。

## Surface identity

`SurfaceIdentity`：

~~~cpp
instanceId
primitiveId
materialId
surfaceGroupId
~~~

职责分离：

- `primitiveId`：精确几何 primitive identity；
- `instanceId`：对象实例；
- `materialId`：材质边界；
- `surfaceGroupId`：Reconstruction / Denoiser 连续表面。

Reconstruction layer key 使用：

~~~text
(instanceId, surfaceGroupId)
~~~

因此未来 Triangle Mesh 中，相邻 triangle 可以共享一个连续 surface group，而 Box 的不同 face 仍保持独立。

## Image Reconstruction

Primary visibility 是正式 Reconstruction 输入，不是最终 Compose 时临时乘一个 coverage。

每个 layer 保存：

~~~text
hit count
replicate hit count
coverage
coverage variance
coverage confidence
primary sample count
~~~

Background / miss 与 layer overflow 独立记录，便于检查 visibility mass。

Film 层默认使用 Tent reconstruction filter。Camera sample 直接 importance-sample 归一化 reconstruction kernel，所以不需要跨像素 atomic splat，也不在最终 PNG 上做 AA blur。

## Adaptive A-Trous

Diffuse 与 Specular 独立过滤。

连续 `filterStrength ∈ [0,1]` 由以下证据控制：

~~~text
RQMC RGB sampling uncertainty
primary sample count
coverage confidence
depth continuity
normal continuity
surface/material compatibility
roughness
signal type
A-Trous radius
~~~

两种 variance 必须分离：

~~~text
samplingVariance
  = Reconstruction-time RQMC uncertainty
  = immutable
  = Adaptive Filter Strength 的统计依据

workingVariance
  = A-Trous 中传播的局部 variance
  = 只用于邻域 color weighting
~~~

空间过滤不会创造新的独立 Path Samples，因此不能用 workingVariance 的下降宣称 Monte Carlo sampling uncertainty 已下降。

## Output 与 Regression

### Linear HDR

CPU 输出：

~~~text
cornell_cpu_raw.pfm
cornell_cpu_diffuse.pfm
cornell_cpu_specular.pfm
cornell_cpu_final.pfm
~~~

PFM 为 RGB float32 Linear HDR，不做：

- ACES；
- gamma；
- 8-bit quantization。

### Display Preview

~~~text
Linear HDR
  ↓
ACES
  ↓
gamma
  ↓
PPM
  ↓
PNG artifact
~~~

### Quantitative Regression

`scripts/render_metrics.py` 对 16 / 64 / 256 SPP 计算：

~~~text
Linear HDR
- MSE
- RMSE
- MAE
- Max Absolute Error
- NRMSE

Display Space
- Tone-mapped PSNR
~~~

ROI：

~~~text
Full Frame
Ceiling Light Border
Short Box Silhouette
Tall Box Silhouette
~~~

普通 push 使用同次 256 SPP Raw 作为 provisional convergence reference。1024 / 2048 SPP 高质量 reference 通过独立手动 workflow 生成。

## Debug AOV

CPU CLI 追加 `debug`：

~~~bash
./build/render_cpu 256 8 16 owen tent debug
~~~

生成：

~~~text
Coverage
Coverage Confidence
Variance / Noise Estimate
Surface Group
Normal
Depth
Effective Adaptive Filter Strength
~~~

其中 Filter Strength 是四轮 A-Trous 的连续强度累计：

~~~text
effective = 1 - Π(1 - strength_i)
~~~

普通成功 push 只上传核心 PFM / PNG / metrics；validation 失败或手动 workflow_dispatch 时上传完整 Debug AOV artifact。

## CI 责任边界

~~~text
CPU Smoke
    ↓
Unit Tests
    ↓
SamplesPerPass Determinism
    ↓
16 / 64 / 256 CPU Render
    ↓
PFM + PNG
    ↓
HDR Metrics + ROI Regression
    ↓
Artifact

CUDA
    ↓
Compile Test only
~~~

Phase E 完成后仍明确不包含：

~~~text
Triangle
Mesh
OBJ Loader
BVH
Sponza
Stanford Bunny
GPU runtime automation
~~~

这些功能在图像形成、Reconstruction 与 Denoiser 架构稳定后继续扩展。
