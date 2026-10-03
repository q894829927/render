#pragma once
#include <cstdint>
#include "Core/Math/Math.h"

namespace render {

constexpr std::uint32_t kCameraJitterXDimension = 0u;
constexpr std::uint32_t kCameraJitterYDimension = 1u;
constexpr std::uint32_t kPathDimensionBase = 2u;
constexpr std::uint32_t kBounceDimensionStride = 10u;

enum class BounceSampleOffset : std::uint32_t {
    LightU = 0u,
    LightV = 1u,
    DirectBsdfLobe = 2u,
    DirectBsdfU = 3u,
    DirectBsdfV = 4u,
    PathBsdfLobe = 5u,
    PathBsdfU = 6u,
    PathBsdfV = 7u,
    RussianRoulette = 8u,
    Reserved = 9u
};

RENDER_HD inline std::uint32_t BounceSampleDimension(
    int bounce,
    BounceSampleOffset offset)
{
    return kPathDimensionBase +
           static_cast<std::uint32_t>(bounce) * kBounceDimensionStride +
           static_cast<std::uint32_t>(offset);
}

RENDER_HD inline std::uint32_t RequiredSampleDimensionCount(int maxDepth) {
    return kPathDimensionBase +
           static_cast<std::uint32_t>(maxDepth) * kBounceDimensionStride;
}

static_assert(
    static_cast<std::uint32_t>(BounceSampleOffset::Reserved) <
        kBounceDimensionStride,
    "Bounce sample dimensions must fit inside the per-bounce stride.");

} // namespace render
