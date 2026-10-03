#pragma once

#include "Core/Reconstruction/ReconstructionFilter.h"
#include "Core/Sampling/SampleGenerator.h"

namespace render {

struct FilmSample {
    float rasterX = 0.0f;
    float rasterY = 0.0f;
    float reconstructionWeight = 1.0f;
};

RENDER_HD inline FilmSample GenerateFilmSample(
    int pixelX,
    int pixelY,
    const SampleGenerator& samples,
    ReconstructionFilterType filterType)
{
    const float u = samples.Sample1D(kCameraJitterXDimension);
    const float v = samples.Sample1D(kCameraJitterYDimension);

    const float offsetX =
        SampleReconstructionFilterOffset(filterType, u);
    const float offsetY =
        SampleReconstructionFilterOffset(filterType, v);

    // We importance-sample the normalized reconstruction kernel directly.
    // Therefore filter / pdf = 1 and no extra radiance weight is required.
    return {
        static_cast<float>(pixelX) + 0.5f + offsetX,
        static_cast<float>(pixelY) + 0.5f + offsetY,
        1.0f
    };
}

} // namespace render
