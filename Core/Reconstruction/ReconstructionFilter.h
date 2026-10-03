#pragma once

#include "Core/Math/Math.h"

namespace render {

enum class ReconstructionFilterType : int {
    Box = 0,
    Tent = 1
};

RENDER_HD inline float ReconstructionFilterRadius(
    ReconstructionFilterType type)
{
    return type == ReconstructionFilterType::Tent
        ? 1.0f
        : 0.5f;
}

RENDER_HD inline float SampleBoxFilterOffset(float u) {
    return u - 0.5f;
}

RENDER_HD inline float SampleTentFilterOffset(float u) {
    // Inverse CDF of the normalized 1D tent kernel:
    // p(x) = 1 - |x|, x in [-1, 1].
    if (u < 0.5f)
        return sqrtf(fmaxf(2.0f * u, 0.0f)) - 1.0f;
    return 1.0f -
           sqrtf(fmaxf(2.0f * (1.0f - u), 0.0f));
}

RENDER_HD inline float SampleReconstructionFilterOffset(
    ReconstructionFilterType type,
    float u)
{
    u = Clamp(u, 0.0f, 0.99999994f);
    return type == ReconstructionFilterType::Tent
        ? SampleTentFilterOffset(u)
        : SampleBoxFilterOffset(u);
}

RENDER_HD inline float EvaluateReconstructionFilter1D(
    ReconstructionFilterType type,
    float x)
{
    if (type == ReconstructionFilterType::Tent) {
        float ax = fabsf(x);
        return ax < 1.0f ? 1.0f - ax : 0.0f;
    }

    return fabsf(x) <= 0.5f ? 1.0f : 0.0f;
}

} // namespace render
