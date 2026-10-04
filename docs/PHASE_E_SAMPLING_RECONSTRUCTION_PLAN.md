# Phase E — Sampling & Reconstruction 重构计划

## 1. 背景与目标

D1–D5 已完成信号拆分、RGB moments、Primary Surface 分层、Diffuse/Specular 独立降噪，以及 CPU/CUDA 共用核心算法。

当前 16 / 64 / 256 SPP 验证图仍有以下问题：

- 顶部面积光边缘仍有明显锯齿，256 SPP 仍可见。
- 灯边缘存在 Primary Visibility / Coverage 的随机噪声。
- 低 SPP 下大面积区域仍有一定过平滑。
- primitiveId 同时承担 Primary Sample 分类和跨像素降噪边界两个职责，无法自然扩展到 Triangle / Mesh。
- 当前随机采样序列受 samplesPerPass 组织方式影响，不适合作为稳定 regression 基线。
- PNG 已经过 Tone Mapping、Gamma 和 8-bit 量化，不适合作为唯一数值验证依据。

Phase E 的目标不是针对面积光打补丁，而是把采样、Primary Visibility、重建、降噪和验证组织成一条可扩展到 Triangle / Mesh / BVH 的稳定管线。

## 2. 目标架构

~~~text
SampleGenerator
├─ Pixel ID
├─ Replicate ID
├─ Sample Index Within Replicate
└─ Dimension
        │
        ├──────── Camera Sample
        │         └─ Owen-Sobol / CMJ
        │
        └──────── Path Sample
                  ├─ Light dimensions
                  ├─ BSDF dimensions
                  └─ RR dimensions
                         │
                         ▼
                     PathSample
                         │
                         ▼
                  SurfaceIdentity
 instance / primitive / material / surfaceGroup
                         │
                         ▼
             Primary Visibility Layers
            coverage + coverage variance
                         │
                         ▼
                  Signal Accumulation
      RGB moments + per-layer reconstruction data
                         │
                         ▼
                    Reconstruction
        ├─ visibility / coverage reconstruction
        └─ film reconstruction filter
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         Diffuse Signal        Specular Signal
       albedo-demodulated     roughness-aware
              │                     │
              └──────────┬──────────┘
                         ▼
                 Adaptive A-Trous
                         │
                         ▼
                      Compose
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
     Linear HDR / PFM               PNG Preview
          │
          ▼
   Quantitative Regression
~~~

核心约束：

1. 不为面积光、Cornell Box 或某个特定场景添加特殊补丁。
2. Sample sequence 只能由 Pixel、Global Sample Index、Dimension 决定。
3. samplesPerPass 只能影响调度，不允许改变样本集合或最终结果。
4. Primary Surface 分类和 Denoiser 邻域判定必须使用不同语义。
5. Coverage / Visibility 是 Reconstruction 的正式输入，而不是 Compose 阶段的附加系数。
6. Denoiser 强度必须由噪声统计决定，而不是所有像素固定执行同样的四轮过滤。
7. 任何视觉优化都必须同时接受 Linear HDR 数值回归验证。
8. Owen-Sobol 等低差异序列的噪声估计不得直接沿用 IID 样本的 variance / N 假设。
9. Image Reconstruction 必须发生在 sample → film → pixel 阶段，禁止通过最终图像 blur 伪装抗锯齿。
10. primitiveId 只表示精确几何元素，不直接决定 reconstruction continuity 或 denoiser continuity。

# E1 — 统一 SampleGenerator

## E1 实施状态

状态：**已完成并通过 CI 验收**。

实现提交：`15379426a0b9175a845f7d7ea94bf22cb856b117`

已完成：

- `SampleDimensions.h`：固定 Camera / Light / BSDF / RR dimension registry。
- `SampleGenerator.h`：无状态、确定性的 pixel / replicate / sample / dimension 接口。
- 固定 4 个 replicate，global sample 按 round-robin 映射到 replicate。
- CPU / CUDA 移除依赖随机数消费顺序的 stateful RNG。
- PathSample 携带 replicateId，PixelAccumulator 保存 replicate signal accumulator。
- 新增 `Sample Determinism` CI。
- 64 SPP 下 samplesPerPass = 1 / 2 / 4 / 8 / 16 的 Linear Hash 与 PPM 输出完全一致。
- CPU Smoke、Render Validation、CUDA Compile Test 全部通过。

E1 仍使用 deterministic hash 作为 reference sampler；Owen-scrambled Sobol 的实际序列切换属于 E2。

## 目标

建立 CPU / CUDA 共用的确定性 SampleGenerator，使结果与 samplesPerPass 无关。

建议接口：

~~~cpp
struct SampleGenerator {
    uint32_t pixelId;
    uint32_t replicateId;
    uint32_t sampleIndexWithinReplicate;
    uint32_t scramble;

    RENDER_HD float Sample1D(uint32_t dimension) const;
    RENDER_HD Vec2  Sample2D(uint32_t dimension) const;
};
~~~

逻辑上的全局样本必须由 replicateId 与 sampleIndexWithinReplicate 唯一确定，不能使用 pass 内局部 index。

固定 RQMC replicate 数量，建议第一版使用 4 个独立 scramble：

~~~text
16 SPP  = 4 replicates × 4 samples
64 SPP  = 4 replicates × 16 samples
256 SPP = 4 replicates × 64 samples
~~~

每个 replicate 内使用同一类低差异序列，不同 replicate 使用独立 scramble。后续噪声估计优先使用 replicate mean 之间的统计，而不是把 Sobol 序列内部样本当作 IID。

## Dimension Layout

为每个随机变量预留固定维度，避免控制流变化导致后续随机维度整体错位。

建议第一版：

~~~text
Dimension 0–1      Camera jitter

Per-bounce block:
  +0–1             Light sample UV
  +2               Direct-BSDF lobe selection
  +3–4             Direct-BSDF sample UV
  +5               Path-BSDF lobe selection
  +6–7             Path-BSDF sample UV
  +8               Russian Roulette
  +9...            Reserved
~~~

每个 bounce 使用固定大小的 dimension block。

## 实现任务

- 新增 Core/Sampling/SampleGenerator.h。
- 移除 Integrator 对有状态 RNG 消耗顺序的依赖。
- Camera、Light、BSDF、RR 均改为显式 dimension sampling。
- CPU/CUDA 共用 SampleGenerator 算法。
- Progressive accumulation 按稳定的逻辑样本顺序逐样本累积，不能因 pass 分组改变浮点累加顺序。
- 禁止先生成 pass-local sum 再归并，因为浮点加法不满足结合律。
- samplesPerPass 只允许决定一次调度/进度更新包含多少逻辑样本，不允许改变样本集合、样本顺序或 reduce 树。
- 为 RQMC 预留 replicate accumulator，保存每个 replicate 的 signal mean。

## 验收标准

同一 backend、同一编译配置下：

~~~text
64 SPP / samplesPerPass = 1
64 SPP / samplesPerPass = 2
64 SPP / samplesPerPass = 4
64 SPP / samplesPerPass = 8
64 SPP / samplesPerPass = 16
~~~

Linear HDR 输出目标为 bitwise identical。若编译器浮点行为导致无法保证，则 max absolute error <= 1e-7，并记录原因。

额外要求：

- 不同 samplesPerPass 下每个 pixel / replicate / sample / dimension 返回完全相同的 sample。
- 4 个 replicate 必须使用互相独立的 scramble。
- 同一输入重复运行必须得到完全相同的 sample sequence。

CI 增加 SampleSequenceDeterminism 检查。

# E2 — Camera jitter 改为 CMJ / Owen-Sobol

## E2 实施状态

状态：**已完成并通过 CI 验收**。

实现提交：

- `a11fd9f3b74aa885afca4d4c48c1170e9300b7a7`：Owen-Sobol / replicate uncertainty 主实现。
- `212b131d8c617f2f0e96987afee600eb454d0798`：修正 CUDA Sobol 常量表定义，CUDA compile CI 通过。

已完成：

- 默认 sampler 从 deterministic hash 切换为 Owen-scrambled Sobol。
- 保留 `hash / pseudo` reference sampler，可通过第四个 CLI 参数做 A/B。
- 使用 Joe-Kuo D(6) 前 256 维 Sobol 参数，覆盖当前默认 `maxDepth=16` 的 Camera / Light / BSDF / RR dimensions。
- Camera X/Y 固定使用 Dimension 0/1；所有 Path dimensions 继续通过统一 SampleGenerator 获取。
- 4 个 RQMC replicate 使用独立 pixel/replicate scramble。
- 使用 practical hash-based fast Owen-style scramble，避免逐 bit nested permutation 带来的路径追踪成本。
- Surface signal 增加 per-replicate accumulation；当前 diffuse/specular uncertainty 改为 replicate means 之间估计，不再把 Sobol 序列内部样本直接当 IID。
- replicate 样本不足时使用 conservative variance，而不是错误返回 zero-noise。
- 新增 `SampleGeneratorTest`：校验基础 Sobol 序列、determinism、replicate scramble 独立性，以及 Camera 前 16 个样本的 4×4 stratification。
- `Sample Determinism` CI 继续验证 64 SPP 下 samplesPerPass = 1 / 2 / 4 / 8 / 16 输出完全一致。
- CPU Smoke、Render Validation、Sample Determinism、CUDA Compile Test 全部通过。

视觉验证：

- 16 / 64 / 256 SPP 下灯边 coverage 的随机抖动较 E1 有下降。
- 灯边仍保留约一像素尺度的 staircase；这是 Box-like pixel reconstruction 的问题，按计划由 E4 Film Reconstruction Filter 解决，E2 不做后处理模糊补丁。

## 目标

减少 Primary Visibility 的白噪声，使灯边缘、箱子轮廓、墙体边界在相同 SPP 下得到更均匀的 coverage 估计。

## 方案

默认目标使用 Owen-scrambled Sobol。Camera X/Y 固定使用 Dimension 0/1。

保留 CMJ 作为参考实现或调试模式：

~~~cpp
enum class SamplerType {
    OwenSobol,
    CMJ,
    PseudoRandomReference
};
~~~

默认使用 OwenSobol。

## 实现任务

- 实现 Sobol 1D/2D 基础序列。
- 加入基于 pixel ID 的 Owen-style scramble。
- Camera jitter 迁移到低差异序列。
- Path dimensions 同样通过统一 SampleGenerator 读取。
- 每个 RQMC replicate 使用独立 Owen-style scramble。
- 保留 deterministic pseudo-random sampler 作为 A/B reference，不作为默认路径。
- 不再用普通 IID 的 sampleVariance / N 解释 Sobol 序列内部噪声。
- 每个 signal 保存 replicate mean，并用 replicate mean 之间的差异估计 uncertainty。

## 验收标准

不修改 Denoiser 参数，比较 16 / 64 / 256 SPP。

重点观察：

- 面积光四条边。
- 短箱顶部和侧面轮廓。
- 长箱顶部边缘。
- Cornell Box 外轮廓。

要求同 SPP 下 coverage 噪声明显低于旧随机 jitter，且不能出现规则网格、条纹或重复 pattern。

# E3 — SurfaceIdentity 拆分

## E3 实施状态

状态：**已完成并通过 CI 验收**。

实现提交：`2a5a078a3aa9aad0bcbebb99e3f0f74b9e13bf2a`

已完成：

- 新增 `SurfaceIdentity { instanceId, primitiveId, materialId, surfaceGroupId }`。
- Primary Sample、HitRecord、LightHit 全链路携带完整 SurfaceIdentity。
- Reconstruction layer key 改为 `(instanceId, surfaceGroupId)`，不再由 primitiveId 决定。
- 同一 surfaceGroup 中命中多个 primitive 时，Resolved Layer 会把 representative primitiveId 标记为 invalid，避免伪装成单一 primitive。
- Cornell Room、Short Box、Tall Box、Area Light 使用稳定 instance/material/group ID。
- OrientedBox 现在精确识别六个 Face；每个 Face 有独立 primitiveId 和 surfaceGroupId。
- AABB slab intersection 同时支持从盒外进入和从盒内射出时的正确 Face identity。
- Denoiser 移除 `primitiveId == primitiveId` hard gate。
- Denoiser 邻域改为 material compatibility + instance/group preference + depth/normal/albedo/roughness continuity。
- 同一平滑 surfaceGroup 中的不同 primitive 可以共享降噪信息；硬折角仍由 normal discontinuity 阻断。
- 新增 `SurfaceIdentityTest`，验证 layer grouping、Box Face identity、material boundary、hard-normal boundary 和 Cornell ID 规则。
- SampleGeneratorTest + SurfaceIdentityTest 均通过，64 SPP samplesPerPass invariance 保持通过。
- CPU Smoke、Render Validation、Sample Determinism、CUDA Compile Test 全部通过。

视觉验证：

- E3 没有改变 Camera sampling 或 Film reconstruction，因此顶部灯剩余 staircase 仍然存在，这是预期行为。
- Box 顶面 / 侧面现在拥有明确不同的 surface groups，后续 E4 不会再把 Box 不同 Face 的 visibility/reconstruction 数据混成一个层。

## 目标

把“样本属于哪个几何表面”和“两个像素能否互相降噪”从同一个 primitiveId 判断中拆开。

建议数据结构：

~~~cpp
struct SurfaceIdentity {
    uint32_t instanceId;
    uint32_t primitiveId;
    uint32_t materialId;
    uint32_t surfaceGroupId;
};
~~~

语义：

~~~text
instanceId
  几何实例 / 对象归属，例如一个 Box、一个 Mesh Instance。

primitiveId
  精确 Primary Surface。
  Rect = Rect ID
  Box = Face ID
  Mesh = Triangle ID

materialId
  材质身份，用于材质连续性和跨 primitive filtering 判断。

surfaceGroupId
  Reconstruction continuity domain。
  Box 的不同 Face 使用不同 group。
  平滑 Mesh 中多个相邻 Triangle 可以属于同一个 group。
~~~

## Primary Classification

一个像素内部 jitter samples 仍保留精确 primitiveId 用于调试、picking 和 exact identity，但 Reconstruction layer key 不直接使用 triangle primitiveId。

建议：

~~~text
Primary exact identity:
(instanceId, primitiveId)

Reconstruction layer key:
(instanceId, surfaceGroupId)
~~~

Box 的不同 face 必须分属不同 surfaceGroupId。

未来 Mesh 中连续平滑、材质连续的一组 triangles 可以共享 surfaceGroupId，避免一个像素因为命中多个相邻 triangle 就耗尽固定 layer slot。

## Denoiser Neighborhood

Denoiser 不再使用 primitiveId equality 作为硬条件。

邻域判断综合：

~~~text
depth continuity
normal continuity
material compatibility
instance continuity（按需要作为强约束）
roughness / metallic compatibility
~~~

这样未来同一个 Mesh 上不同 triangle 可以共享降噪信息，同时几何折角仍由 normal/depth 阻止跨边污染。

## 实现任务

- HitRecord 返回完整 SurfaceIdentity。
- Rect、Box Face 分配稳定 ID。
- 为未来 Triangle / Mesh 预留 ID 规则。
- 增加 surfaceGroupId，并定义稳定分组规则。
- Reconstruction layer key 使用 instanceId + surfaceGroupId。
- primitiveId 保留 exact geometry identity，但不作为 reconstruction / denoiser continuity 的硬边界。
- Denoiser guide 去除 primitive equality hard gate。

## 验收标准

- Box 顶面 / 正面 / 侧面不会在单像素 layer resolve 中错误平均。
- 同一平面上相邻 primitive 可以合理过滤。
- 几何硬折角无明显跨边 bleeding。
- 数据结构可直接扩展到 Mesh Triangle，无需再次改接口。
- 高模 Mesh 中多个相邻 triangle 不会因为 primitiveId 不同而被拆成大量 reconstruction layers。

# E4 — Image Reconstruction：Coverage / Visibility + Film Filter

## E4 实施状态

状态：**已完成并通过 CI 验收**。

实现提交：

- `11a9a6500bf72f332cc26ef40e473df6f6d037f0`：Visibility moments + Film reconstruction 主实现。
- `ec89db0b0870ac6b817979d32bd7df5ae6c7a1a1`：修正 Compose residual 命名并完成全套 CI。

已完成：

- 新增 `ReconstructionFilterType { Box, Tent }`，默认使用 Tent，Box 保留为 reference/debug。
- 新增 `FilmSample`，Camera Dimension 0/1 不再直接解释为 pixel 内 uniform jitter，而是先通过 reconstruction filter 生成连续 raster sample。
- Film reconstruction 采用 **per-output-pixel gather / normalized-kernel importance sampling**，而不是跨像素 splat。每个输出像素直接采样其归一化 Box/Tent kernel，因此 `filter / pdf = 1`，不需要跨像素 atomic，也不存在 splat 边界的额外 weight normalization。
- Tent kernel 使用标准 1D 三角核 `p(x)=1-|x|`、support `[-1,1]`，通过 inverse CDF 直接采样；2D 使用可分离 Tent。
- Visibility 从单一 hitCount 升级为 `VisibilityAccumulator / VisibilityEstimate`。
- 每个 Surface Layer 记录 per-replicate hit count，并使用独立 RQMC replicate coverage estimates 计算 coverage variance。
- Visibility 输出正式包含 `coverage / variance / confidence / hitCount / sampleCount`。
- 单 replicate / 单样本情况使用 conservative coverage variance，confidence=0，不再被视为 zero-noise。
- Background / miss visibility 与 layer-overflow visibility 分开记录；surface + background + overflow 的 visibility mass 可以显式核对。
- SurfaceGuide 现在携带 `coverageVariance / coverageConfidence / primarySampleCount`，并传入 Denoiser guide weighting。
- Emission 继续通过 Primary Surface Layer 的 reconstructed coverage 合成，没有针对 Area Light 增加特殊 AA 分支。
- CPU/CUDA 使用相同 Film / Visibility reconstruction 定义。
- 新增 `ReconstructionTest`，覆盖 Box/Tent support、Tent inverse CDF 对称性、FilmSample footprint、balanced/imbalanced replicate coverage、background mass conservation 和 single-sample low confidence。
- SampleGeneratorTest、SurfaceIdentityTest、ReconstructionTest 全部通过。
- 64 SPP 下 samplesPerPass = 1 / 2 / 4 / 8 / 16 determinism 继续完全一致。
- CPU Smoke、Render Validation、Sample Determinism、CUDA Compile Test 全部通过。

视觉验收：

- 16 SPP 顶灯下边缘在 E3 中存在明显逐像素上下跳动；E4 Tent 后中央边缘明显稳定，随机 staircase 显著减少。
- 64 SPP 的中央灯下边缘在当前阈值检查中已经稳定为单一 scanline；E3 同区域仍存在多次一像素跳变。
- 256 SPP 下 E3/E4 都已稳定收敛，但 E4 保持更平滑的 pixel reconstruction，未观察到明显 halo。
- 短箱 / 长箱轮廓未出现新的 coverage hole、黑边或跨 Face bleeding。

CLI：

~~~bash
./build/render_cpu 256 8 16 owen tent   # 默认
./build/render_cpu 256 8 16 owen box    # reconstruction reference
~~~

E4 只解决 visibility/image reconstruction；A-Trous 仍然固定执行 step 1/2/4/8，Adaptive Filter Strength 属于 E5。

## 目标

把 Primary Visibility 从简单的 sample count / SPP 升级为带统计语义的 Reconstruction 输入，同时正式加入 Film Reconstruction Filter。

E4 必须解决两个不同问题：

~~~text
Sampling variance
  → 边缘 coverage 是否稳定

Pixel reconstruction
  → 连续图像信号如何重建为离散像素
~~~

低差异采样只能降低前者，不能单独消除高对比边缘的 staircase。

建议数据：

~~~cpp
struct VisibilityMoments {
    uint32_t hitCount;
    uint32_t sampleCount;

    float coverage;
    float coverageVariance;
    float confidence;
};
~~~

每一个 Primary Surface Layer 独立统计 visibility。

RQMC 模式下 coverage uncertainty 应优先从独立 replicate 的 coverage mean 估计，而不是把同一 Sobol replicate 内的样本直接视为 IID。

## Film Reconstruction Filter

新增正式的 Film / Reconstruction Filter 层：

~~~text
continuous camera sample position
        ↓
PathSample / Visibility sample
        ↓
FilmSample
        ↓
Reconstruction Filter
        ↓
discrete pixel
~~~

第一版至少支持：

~~~text
Box   — reference / debug
Tent  — default
~~~

Mitchell-Netravali 可以作为后续可选实现，不作为 E4 完成条件。

Film Filter 必须参与 radiance、visibility 和 layer signal 的 sample-to-pixel reconstruction，禁止在 Final PNG 上直接 blur。

## 设计原则

Coverage 不应该只在最终 Compose 时乘一下，而应该参与：

~~~text
Primary Layer reconstruction
Denoiser confidence
Adaptive filter radius
Final composition
~~~

低 coverage / 低 sample count 表示统计不确定性高，不能解释为 variance = 0。

特别禁止：

~~~text
count <= 1
=> variance = 0
=> high confidence
~~~

## 实现任务

- 为 layer 添加 Bernoulli visibility moments。
- Resolve 计算 coverage uncertainty。
- 把 sample count / coverage confidence 传入 Denoiser。
- Emission 与 Primary Visibility 的组合通过 Reconstruction 输出，不为灯单独添加 edge fix。
- 明确 background / miss layer 的 visibility 语义。
- 新增 FilmSample / ReconstructionFilter 抽象。
- 实现 Box 与 Tent filter，默认使用 Tent。
- Reconstruction Filter 必须作用于连续 sample position 到 pixel 的重建，而不是后处理模糊。
- 记录每个 pixel 的有效 filter weight，避免边界归一化错误。

## 验收标准

- 灯边缘随着 16 → 64 → 256 SPP 稳定收敛。
- 低 coverage 像素不能因为单样本 variance=0 被判定为高可信。
- Box silhouette 不出现 coverage hole、halo 或不连续暗边。
- Final 能量与 Raw 的高 SPP reference 保持一致趋势。
- Tent filter 相比 Box filter 应减少高对比灯边的 staircase，同时不能产生明显 halo。
- 相同 sample sequence 下切换 Box / Tent 时，差异必须来自 reconstruction kernel，而不是 sample sequence 改变。

# E5 — Adaptive A-Trous

## E5 实施状态

状态：**已完成并通过 CI 验收**。

实现提交：

- `fc5574df6035621958b8859b793b934161b67fce`：Adaptive A-Trous 主实现。
- `1638f32fc319ac52d07f8ec8d09d2f5bd6e2a85c`：修正非饱和 radius test。
- `518af6f308146b9f92d9c2cf5b2dd36b29f836c2`：增强低 SPP 自适应过滤。
- `6452674900bcd465f1e487361a5a7b53e3a4c25f`：把 reconstruction-time sampling variance 与 propagated filter variance 正式拆分。

已完成：

- 固定全强度 step 1 / 2 / 4 / 8 改为连续 `filterStrength ∈ [0,1]`。
- 每轮仍生成 A-Trous candidate，但最终通过连续强度在 center 与 filtered candidate 之间混合，不使用 per-pixel hard STOP。
- Adaptive controller 正式使用：
  - RQMC RGB sampling variance
  - Primary sample count
  - Coverage confidence
  - Depth continuity
  - Normal continuity
  - Material / instance / surfaceGroup compatibility
  - Roughness
  - Diffuse / Specular signal type
  - 当前 A-Trous radius
- Geometry confidence 从当前 A-Trous footprint 内的 geometry support 计算；硬折角、depth discontinuity 和缺失 surface support 会降低宽半径过滤强度。
- Specular 使用 roughness-dependent radius gate；光滑高光比 rough diffuse 更严格地限制宽范围传播。
- Low-SPP sample scarcity 只增强已有 noise evidence，不把 zero-variance signal 强行模糊。
- 关键统计语义修正：**空间滤波后的 working variance 不再替代原始 RQMC sampling uncertainty**。
  - `samplingVariance`：immutable，来自 Reconstruction 的 replicate means，用于决定 Adaptive Filter Strength。
  - `working / propagated variance`：只用于当前邻域的 variance-aware color weighting。
  - 原因：空间滤波没有增加新的独立 Path Samples，不能因此宣称 Monte Carlo estimator uncertainty 已下降。
- CPU / CUDA 共用同一 Adaptive A-Trous 算法；CUDA 额外保存 immutable sampling variance buffer。
- 新增 `AdaptiveDenoiserTest`，验证：
  - variance 越高，过滤需求越强；
  - 宽 radius 需要更强 noise evidence；
  - geometry discontinuity 降低过滤；
  - coverage confidence 低时降低过滤；
  - low sample count 增强 noise demand；
  - smooth specular 比 rough specular 更严格；
  - zero sampling variance 为 exact no-op；
  - working variance 被过滤为 0 时仍不能抹掉原始 sampling uncertainty。
- Render Validation 现在保存每个 SPP 的 `render.log`，并自动验证 16 SPP 的平均 filter strength 大于 256 SPP。
- SampleGenerator / SurfaceIdentity / Reconstruction / AdaptiveDenoiser 四组 tests 全部通过。
- 64 SPP 下 samplesPerPass = 1 / 2 / 4 / 8 / 16 determinism 保持通过。
- CPU Smoke、Render Validation、Sample Determinism、CUDA Compile Test 在 `6452674900...` 全部通过。

实际平均过滤强度（最终 E5）：

~~~text
16 SPP
  step 1: diffuse 0.8410 / specular 0.3636
  step 2: diffuse 0.7126 / specular 0.2596
  step 4: diffuse 0.5525 / specular 0.1708
  step 8: diffuse 0.3917 / specular 0.1065

64 SPP
  step 1: diffuse 0.5009 / specular 0.1360
  step 2: diffuse 0.3467 / specular 0.0890
  step 4: diffuse 0.2246 / specular 0.0552
  step 8: diffuse 0.1401 / specular 0.0333

256 SPP
  step 1: diffuse 0.2659 / specular 0.0717
  step 2: diffuse 0.1762 / specular 0.0464
  step 4: diffuse 0.1121 / specular 0.0285
  step 8: diffuse 0.0694 / specular 0.0171
~~~

这验证了两层自适应关系：

~~~text
SPP 越高
→ sampling uncertainty 越低
→ filterStrength 越低

radius 越大
→ 需要更强 noise evidence
→ filterStrength 连续下降
~~~

视觉状态：

- 16 / 64 SPP 相比 E4 固定四轮全强度过滤保留了更多高频细节，同时也暴露出更多残余颗粒。
- 256 SPP 明显减少了不必要的宽半径平滑，Box 轮廓、墙面渐变和灯边没有观察到新的 halo / bleeding。
- E5 暂不把“更锐”或“更平”作为最终优劣判断；E6 将引入 Linear HDR/PFM reference 与 NRMSE/RMSE/MAE，决定当前 noise/detail trade-off 是否需要进一步校准。

## 目标

从固定 step 1 / 2 / 4 / 8 的四轮过滤，升级为由噪声与几何可信度控制的自适应过滤。

## 控制输入

至少考虑：

~~~text
RGB variance
sample count
coverage confidence
depth gradient
normal gradient
roughness
signal type (Diffuse / Specular)
~~~

## 策略

Diffuse：

- 高 variance 允许更大过滤半径。
- 低 variance 提前停止。
- 几何高频区域缩小过滤半径。
- 继续在 albedo-demodulated illumination space 工作。

Specular：

- 保持独立 filtering。
- 粗糙度越低，normal/depth 条件越严格。
- 高光细节区不允许使用和 diffuse 相同的宽松半径。

建议提供连续强度决策，而不是把 maxIteration 作为主要算法边界：

~~~cpp
struct AdaptiveFilterDecision {
    float filterStrength;
    float geometryConfidence;
    float noiseConfidence;
};
~~~

每一轮 A-Trous 都计算 filterStrength ∈ [0, 1]。

低噪声区域的后续迭代应自然趋近于 0，而不是突然 STOP。maxIteration 可以保留为性能优化上限，但不能成为主要重建逻辑。

噪声输入优先使用 RQMC replicate mean 之间估计出的 uncertainty；若使用 pseudo-random reference sampler，才允许使用对应的 IID variance estimator。

## 验收标准

- 16 SPP 相比当前版本保留更多墙面、阴影和箱子局部细节。
- 256 SPP 不再接受和 16 SPP 相同强度的过滤。
- 不引入 halo、edge darkening、color bleeding。
- Quantitative regression 不允许为了视觉“更平”而显著增加 HDR error。

# E6 — Linear HDR / PFM + Quantitative Regression

## E6 实施状态

状态：**已完成并通过 CI 验收**。

实现提交：

- `23142c5c060cd24ddd1806fc4741002c407737cc`：PFM 输出、HDR metrics、ROI regression、高质量 reference workflow 主实现。
- `701c0db0d794049e995b8330d6e6c18dd18df7db`：把低 SPP denoising improvement 正式加入 HDR CI gate。

已完成：

- 新增 `Core/Output/PFM.h`。
- CPU 正式输出四路 Linear HDR float32 PFM：
  - `cornell_cpu_raw.pfm`
  - `cornell_cpu_diffuse.pfm`
  - `cornell_cpu_specular.pfm`
  - `cornell_cpu_final.pfm`
- PFM 为 RGB float32、Linear HDR，不经过 ACES、gamma 或 8-bit quantization。
- PFM endian/row layout 使用显式二进制写入，不依赖宿主机字节序。
- 新增 `PFMTest`，验证 Header、little-endian scale 和 float payload 不发生数值变换。
- 新增 `scripts/render_metrics.py`，无第三方 Python 依赖，直接读取 PFM。
- Linear HDR 指标：
  - MSE
  - RMSE
  - MAE
  - Max Absolute Error
  - NRMSE = RMSE / RMS(reference)
- Display-space 指标：
  - 与 renderer 相同 ACES + gamma 后的 PSNR。
- ROI：
  - full frame
  - ceiling light border
  - short-box silhouette
  - tall-box silhouette
- 普通 push 使用同次 256 SPP Raw 作为 **provisional convergence reference**，避免把 256 SPP 描述成最终 ground truth。
- 新增手动 `High Quality Reference` workflow，可选择 1024 / 2048 SPP，生成独立 Linear HDR PFM + PNG artifact；不会在普通 push 自动消耗 CPU 时间。
- `Sample Determinism` 现在同时 exact-compare PPM、PFM 和 Linear framebuffer hash，Correctness Regression 与 Render Quality Regression 保持分离。
- Render Validation 自动上传：
  - 16 / 64 / 256 SPP × Raw / Diffuse / Specular / Final PNG
  - 同矩阵 PFM
  - per-SPP render.log
  - metrics.json
- CI 自动检查：
  - 64 SPP Raw / Final full-frame RMSE 必须优于 16 SPP。
  - 16 / 64 SPP 的 Final 在 full-frame + 三个关键 ROI 中，Linear HDR RMSE 必须优于同 SPP Raw。
  - 同时要求 tone-mapped PSNR 不退化。

本次普通 validation 的 full-frame 结果（对同次 256 SPP Raw reference）：

~~~text
16 SPP
  Raw   RMSE  0.035680
        NRMSE 0.026565
        PSNR  26.94 dB

  Final RMSE  0.029983
        NRMSE 0.022324
        PSNR  36.66 dB

64 SPP
  Raw   RMSE  0.011591
        NRMSE 0.008630
        PSNR  34.06 dB

  Final RMSE  0.009226
        NRMSE 0.006869
        PSNR  38.87 dB
~~~

关键 ROI 也通过 Final < Raw 的 Linear HDR RMSE gate：

~~~text
Ceiling Light Border
16 SPP: Raw 0.16357 → Final 0.16038
64 SPP: Raw 0.04413 → Final 0.04276

Short Box Silhouette
16 SPP: Raw 0.01481 → Final 0.00551
64 SPP: Raw 0.00657 → Final 0.00401

Tall Box Silhouette
16 SPP: Raw 0.01953 → Final 0.00590
64 SPP: Raw 0.00880 → Final 0.00497
~~~

说明：

- 当前 256 SPP Raw 只是普通 CI 的临时 reference，用于自动检查收敛方向和明显退化。
- 真正用于参数定标的高质量 reference 应运行手动 1024 / 2048 SPP workflow。
- E7 将在此基础上加入 Debug AOV、最终视觉矩阵和完整 Phase E 验收。

## 目标

PNG 只作为视觉预览，Linear HDR 作为数值验证基础。

## 输出

CPU 至少输出：

~~~text
cornell_cpu_raw.pfm
cornell_cpu_diffuse.pfm
cornell_cpu_specular.pfm
cornell_cpu_final.pfm

cornell_cpu_raw.png
cornell_cpu_diffuse.png
cornell_cpu_specular.png
cornell_cpu_final.png
~~~

PFM 要求：

- RGB float32。
- Linear HDR。
- 不经过 ACES。
- 不经过 gamma。
- 不做 8-bit quantization。

## Regression Metrics

新增 scripts/render_metrics.py，指标分为三类：

~~~text
Correctness Regression
- max absolute error
- deterministic hash / exact comparison

Linear HDR Quality
- MSE
- RMSE
- MAE
- NRMSE = RMSE / RMS(reference)

Displayed Image
- tone-mapped PSNR
~~~

Linear HDR 不把 PSNR 作为主要指标，因为 HDR 没有稳定的固定 MAX_VALUE。Tone-mapped PSNR 只用于显示空间对比。

支持 ROI：

~~~text
full frame
ceiling light border ROI
short-box silhouette ROI
tall-box silhouette ROI
~~~

避免 full-frame metric 掩盖局部边缘退化。

## Reference 策略

普通 push：

~~~text
16 / 64 / 256 SPP
~~~

生成 PFM + PNG，并检查收敛趋势。

高质量 reference：

~~~text
1024 或 2048 SPP Raw Linear HDR
~~~

放到手动 workflow_dispatch 或独立 reference workflow，避免每次 push 浪费 CPU CI 时间。

## 验收标准

- PFM 稳定读写并保持 Linear HDR。
- 相同输入 deterministic run 指标稳定。
- 16 → 64 → 256 SPP 的 Raw error 整体下降。
- Denoised Final 在低 SPP 下相对 Raw 有更低的 HDR NRMSE / RMSE，同时不能通过明显过平滑换取单一指标优势。
- deterministic regression 与 render quality metric 分开报告，避免“数值完全一致”和“视觉质量更高”混成一个指标。

# E7 — 16 / 64 / 256 SPP 最终验证

## 输出矩阵

每个 SPP 输出：

~~~text
Raw
Diffuse
Specular
Final
~~~

同时输出：

~~~text
PNG preview
PFM Linear HDR
metrics.json
~~~

调试模式额外输出 Debug AOV：

~~~text
Coverage
Coverage Confidence
RGB Variance / Noise Estimate
Surface Group
Normal
Depth
Adaptive Filter Strength
~~~

普通 push 不要求上传全部 Debug AOV；当 validation 失败或手动 workflow_dispatch 时上传完整 Debug AOV。

建议目录：

~~~text
validation/
├─ 16spp/
│  ├─ raw.png
│  ├─ diffuse.png
│  ├─ specular.png
│  ├─ final.png
│  ├─ raw.pfm
│  ├─ diffuse.pfm
│  ├─ specular.pfm
│  └─ final.pfm
├─ 64spp/
├─ 256spp/
└─ metrics.json
~~~

## 强制视觉检查区域

### A. Ceiling Light Border

检查：

- 四条灯边是否连续。
- 是否仍存在明显 stair-step。
- 是否存在随机亮点。
- 是否出现 halo。
- 色彩渗透是否保持自然梯度。

### B. Short Box

检查：

- 顶面与侧面边界。
- Box face 是否错误合层。
- 阴影边缘是否被过度模糊。

### C. Tall Box

检查：

- 顶部斜边。
- 侧面转折。
- 高光 / Diffuse 分离是否稳定。

### D. Large Flat Regions

检查 ceiling、back wall、floor：

- 无明显低频 blotching。
- 无塑料式过平滑。
- 16 → 64 → 256 SPP 细节与稳定性应单调改善。

## 自动验收

GitHub Actions 最终流程：

~~~text
CPU Build
    ↓
Deterministic Sample Test
    ↓
16 / 64 / 256 SPP Render
    ↓
PFM + PNG
    ↓
Metrics
    ↓
ROI Regression
    ↓
Artifact Upload
~~~

Phase E 完成前只使用 CPU 作为自动化运行后端。

CUDA 保留 compile-only CI，暂不作为 Phase E 的运行时验收条件。

# 3. 实施顺序与依赖

严格按照：

~~~text
E1 Deterministic SampleGenerator
   + Dimension Registry
   + RQMC Replicate Architecture
        ↓
E2 Owen-Sobol / CMJ
   + Independent Scramble per Replicate
        ↓
E3 SurfaceIdentity
   + surfaceGroupId
        ↓
E4 Image Reconstruction
   + Visibility / Coverage Moments
   + Film Filter
        ↓
E5 Adaptive A-Trous
   + Replicate-based Noise Estimate
   + Continuous Filter Strength
        ↓
E6 PFM + Quantitative Regression
        ↓
E7 Validation + Debug AOV
~~~

不建议并行跳步：

- E2 依赖 E1 的 dimension model。
- E4 依赖 E3 的 surface identity。
- E5 依赖 E4 的 coverage confidence。
- E7 依赖 E6 的 Linear HDR metrics，否则会退化成只看 PNG。

# 4. 预计文件调整

~~~text
Core/
├─ Sampling/
│  ├─ SampleGenerator.h          # E1
│  ├─ SampleDimensions.h         # E1
│  ├─ Sobol.h                    # E2
│  └─ Sampling.h
│
├─ Scene/
│  └─ Scene.h                    # E3
│
├─ Integrator/
│  ├─ PathSample.h
│  └─ Integrator.h
│
├─ Reconstruction/
│  ├─ Reconstruction.h           # E4
│  ├─ Film.h                     # E4
│  └─ ReconstructionFilter.h     # E4
│
├─ Denoiser/
│  ├─ Denoiser.h
│  └─ ATrous.h                   # E5
│
└─ Output/
   ├─ ImageIO.h
   └─ PFM.h                      # E6

scripts/
├─ ppm_to_png.py
└─ render_metrics.py             # E6

.github/workflows/
├─ cpu-smoke.yml
├─ render-preview.yml            # E6 / E7
└─ cuda-compile.yml
~~~

Phase E 完成后同步更新 ARCHITECTURE.md，使文档与真实数据流一致。

# 5. 明确不在 Phase E 中做的内容

本阶段不进入：

~~~text
Triangle
Mesh
OBJ Loader
BVH
Sponza
Stanford Bunny
GPU runtime automation
新的材质模型
~~~

这些内容在 E1–E7 验收通过后继续。

原因是 Mesh/BVH 会放大当前 SurfaceIdentity 和 Reconstruction 的设计问题。先稳定图像形成与降噪架构，再扩展场景复杂度。

# 6. Phase E Definition of Done

只有同时满足以下条件才算完成：

- [ ] samplesPerPass 不再改变样本集合、样本顺序、reduce 顺序和最终输出。
- [ ] SampleGenerator 显式支持 pixel / replicate / sample-within-replicate / dimension。
- [ ] 固定数量的独立 RQMC replicates 可用于 noise estimation。
- [ ] 默认 Camera sampler 使用 Owen-scrambled Sobol 或验收通过的低差异实现。
- [ ] RQMC uncertainty 不再使用简单 IID variance / N 解释。
- [ ] SurfaceIdentity 已拆分为 instance / primitive / material / surfaceGroup。
- [ ] Box 不同 face 能精确分类，Mesh 相邻 triangle 可共享合理的 surfaceGroup。
- [ ] Denoiser 不再依赖 primitiveId equality hard gate。
- [ ] Coverage / Visibility 具备 sample count、replicate uncertainty 和 confidence 语义。
- [ ] 单样本 layer 不再被错误视为 zero-noise high-confidence。
- [ ] Film Reconstruction Filter 已进入 sample-to-pixel 正式管线。
- [ ] 至少实现 Box 与 Tent filter，默认使用 Tent。
- [ ] A-Trous 根据 noise / geometry / coverage confidence 使用连续 filter strength。
- [ ] CPU 输出 Linear HDR PFM。
- [ ] CI 区分 deterministic correctness 与 render quality metrics。
- [ ] CI 生成 Linear HDR MSE / RMSE / MAE / NRMSE，以及 tone-mapped PSNR。
- [ ] CI 自动生成 16 / 64 / 256 SPP 的 Raw / Diffuse / Specular / Final。
- [ ] validation 失败或手动运行时可输出 Coverage / Confidence / Variance / SurfaceGroup / Normal / Depth / FilterStrength Debug AOV。
- [ ] 顶部灯边缘锯齿与随机噪声相较当前版本明显改善。
- [ ] 短箱和长箱轮廓没有新的 halo / bleeding / surface mixing。
- [ ] 16 → 64 → 256 SPP 在视觉和 Linear HDR metric 上表现出稳定收敛。
- [ ] ARCHITECTURE.md 与最终实现同步。
