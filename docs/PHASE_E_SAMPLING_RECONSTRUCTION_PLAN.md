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
├─ Global Sample Index
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
          instance / primitive / material
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
           visibility + signal reconstruction
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

# E1 — 统一 SampleGenerator

## 目标

建立 CPU / CUDA 共用的确定性 SampleGenerator，使结果与 samplesPerPass 无关。

建议接口：

~~~cpp
struct SampleGenerator {
    uint32_t pixelId;
    uint32_t sampleIndex;
    uint32_t scramble;

    RENDER_HD float Sample1D(uint32_t dimension) const;
    RENDER_HD Vec2  Sample2D(uint32_t dimension) const;
};
~~~

sampleIndex 必须是全局 sample index，而不是 pass 内 index。

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
- Progressive accumulation 按 global sample index 顺序逐样本累积，不能因 pass 分组改变浮点累加顺序。

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

CI 增加 SampleSequenceDeterminism 检查。

# E2 — Camera jitter 改为 CMJ / Owen-Sobol

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
- 保留 deterministic pseudo-random sampler 作为 A/B reference，不作为默认路径。

## 验收标准

不修改 Denoiser 参数，比较 16 / 64 / 256 SPP。

重点观察：

- 面积光四条边。
- 短箱顶部和侧面轮廓。
- 长箱顶部边缘。
- Cornell Box 外轮廓。

要求同 SPP 下 coverage 噪声明显低于旧随机 jitter，且不能出现规则网格、条纹或重复 pattern。

# E3 — SurfaceIdentity 拆分

## 目标

把“样本属于哪个几何表面”和“两个像素能否互相降噪”从同一个 primitiveId 判断中拆开。

建议数据结构：

~~~cpp
struct SurfaceIdentity {
    uint32_t instanceId;
    uint32_t primitiveId;
    uint32_t materialId;
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
~~~

## Primary Classification

一个像素内部 jitter samples 的 Primary Surface 分层使用精确身份：

~~~text
(instanceId, primitiveId)
~~~

禁止把 Box 的不同 face 合并成同一层。

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
- Reconstruction layer key 使用精确 Primary Surface identity。
- Denoiser guide 去除 primitive equality hard gate。

## 验收标准

- Box 顶面 / 正面 / 侧面不会在单像素 layer resolve 中错误平均。
- 同一平面上相邻 primitive 可以合理过滤。
- 几何硬折角无明显跨边 bleeding。
- 数据结构可直接扩展到 Mesh Triangle，无需再次改接口。

# E4 — Coverage / Visibility 正式进入 Reconstruction

## 目标

把 Primary Visibility 从简单的 sample count / SPP 升级为带统计语义的 Reconstruction 输入。

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

## 验收标准

- 灯边缘随着 16 → 64 → 256 SPP 稳定收敛。
- 低 coverage 像素不能因为单样本 variance=0 被判定为高可信。
- Box silhouette 不出现 coverage hole、halo 或不连续暗边。
- Final 能量与 Raw 的高 SPP reference 保持一致趋势。

# E5 — Adaptive A-Trous

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

建议提供：

~~~cpp
struct AdaptiveFilterDecision {
    int maxIteration;
    float varianceScale;
    float geometryConfidence;
};
~~~

## 验收标准

- 16 SPP 相比当前版本保留更多墙面、阴影和箱子局部细节。
- 256 SPP 不再接受和 16 SPP 相同强度的过滤。
- 不引入 halo、edge darkening、color bleeding。
- Quantitative regression 不允许为了视觉“更平”而显著增加 HDR error。

# E6 — Linear HDR / PFM + Quantitative Regression

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

新增 scripts/render_metrics.py，至少计算：

~~~text
MSE
RMSE
PSNR
max absolute error
mean absolute error
~~~

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
- Denoised Final 在低 SPP 下相对 Raw 有更低误差，同时不能通过明显过平滑换取单一指标优势。

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
Artifact Upload
~~~

Phase E 完成前只使用 CPU 作为自动化运行后端。

CUDA 保留 compile-only CI，暂不作为 Phase E 的运行时验收条件。

# 3. 实施顺序与依赖

严格按照：

~~~text
E1 SampleGenerator
        ↓
E2 Owen-Sobol / CMJ
        ↓
E3 SurfaceIdentity
        ↓
E4 Visibility Reconstruction
        ↓
E5 Adaptive A-Trous
        ↓
E6 PFM + Metrics
        ↓
E7 Validation
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
│  └─ Reconstruction.h           # E4
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

- [ ] samplesPerPass 不再改变最终样本集合和输出结果。
- [ ] 默认 Camera sampler 使用 Owen-scrambled Sobol 或验收通过的低差异实现。
- [ ] SurfaceIdentity 已拆分为 instance / primitive / material。
- [ ] Box 不同 face 能精确分类。
- [ ] Denoiser 不再依赖 primitiveId equality hard gate。
- [ ] Coverage / Visibility 具备 sample count、variance/confidence 语义。
- [ ] 单样本 layer 不再被错误视为 zero-noise high-confidence。
- [ ] A-Trous 根据 variance / geometry confidence 自适应停止或缩小半径。
- [ ] CPU 输出 Linear HDR PFM。
- [ ] CI 生成 MSE / RMSE / PSNR 等 regression metrics。
- [ ] CI 自动生成 16 / 64 / 256 SPP 的 Raw / Diffuse / Specular / Final。
- [ ] 顶部灯边缘锯齿与随机噪声相较当前版本明显改善。
- [ ] 短箱和长箱轮廓没有新的 halo / bleeding / surface mixing。
- [ ] 16 → 64 → 256 SPP 在视觉和 Linear HDR metric 上表现出稳定收敛。
- [ ] ARCHITECTURE.md 与最终实现同步。
