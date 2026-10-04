#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "Core/Reconstruction/Reconstruction.h"

namespace render {

struct DebugAOVSet {
    std::vector<Vec3> coverage;
    std::vector<Vec3> coverageConfidence;
    std::vector<Vec3> variance;
    std::vector<Vec3> surfaceGroup;
    std::vector<Vec3> normal;
    std::vector<Vec3> depth;
    std::vector<Vec3> filterStrength;
};

inline float CombineFilterStrength(
    float accumulated,
    float current)
{
    accumulated = Saturate(accumulated);
    current = Saturate(current);
    return 1.0f -
           (1.0f - accumulated) *
           (1.0f - current);
}

inline int DominantResolvedLayerSlot(
    const ResolvedPixel& pixel)
{
    int bestSlot = -1;
    float bestCoverage = -1.0f;

    for (int slot = 0;
         slot < kPrimarySurfaceSlots;
         ++slot)
    {
        const ResolvedLayer& layer =
            pixel.layers[slot];

        if (!layer.valid)
            continue;

        if (layer.guide.coverage > bestCoverage) {
            bestCoverage =
                layer.guide.coverage;
            bestSlot = slot;
        }
    }

    return bestSlot;
}

inline std::uint32_t DebugHash32(std::uint32_t value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

inline Vec3 SurfaceGroupDebugColor(
    const SurfaceIdentity& identity)
{
    if (identity.instanceId == kInvalidSurfaceId ||
        identity.surfaceGroupId == kInvalidSurfaceId)
        return Vec3(0.0f);

    std::uint32_t hash =
        DebugHash32(
            identity.instanceId *
                0x9e3779b9u ^
            identity.surfaceGroupId *
                0x85ebca6bu);

    // Keep colors away from black so neighboring surface groups are easy to
    // inspect while remaining deterministic across runs.
    float r =
        0.20f +
        0.80f *
        static_cast<float>(
            (hash >> 0u) & 0xffu) /
        255.0f;
    float g =
        0.20f +
        0.80f *
        static_cast<float>(
            (hash >> 8u) & 0xffu) /
        255.0f;
    float b =
        0.20f +
        0.80f *
        static_cast<float>(
            (hash >> 16u) & 0xffu) /
        255.0f;

    return Vec3(r, g, b);
}

inline float RawNoiseDisplayValue(
    const Vec3& variance)
{
    Vec3 v = MaxZero(variance);
    float standardError =
        std::sqrt(
            std::max(
                (v.x + v.y + v.z) /
                    3.0f,
                0.0f));

    // Monotonic compression for visualization only. The quantitative
    // regression continues to use the unmodified Linear HDR variance.
    constexpr float kNoiseDisplayScale =
        0.02f;

    return Saturate(
        standardError /
        (standardError +
         kNoiseDisplayScale));
}

inline DebugAOVSet BuildDebugAOVs(
    const std::vector<ResolvedPixel>& resolved,
    const std::vector<float>& diffuseEffectiveStrength,
    const std::vector<float>& specularEffectiveStrength,
    int width,
    int height)
{
    const std::size_t pixelCount =
        static_cast<std::size_t>(width) *
        static_cast<std::size_t>(height);

    DebugAOVSet out;
    out.coverage.resize(pixelCount);
    out.coverageConfidence.resize(pixelCount);
    out.variance.resize(pixelCount);
    out.surfaceGroup.resize(pixelCount);
    out.normal.resize(pixelCount);
    out.depth.resize(pixelCount);
    out.filterStrength.resize(pixelCount);

    float minDepth = kInf;
    float maxDepth = 0.0f;

    for (std::size_t pixel = 0;
         pixel < pixelCount;
         ++pixel)
    {
        int slot =
            DominantResolvedLayerSlot(
                resolved[pixel]);

        if (slot < 0)
            continue;

        float depth =
            resolved[pixel]
                .layers[slot]
                .guide
                .depth;

        if (IsFinite(depth)) {
            minDepth =
                std::min(minDepth, depth);
            maxDepth =
                std::max(maxDepth, depth);
        }
    }

    const bool hasDepthRange =
        IsFinite(minDepth) &&
        maxDepth > minDepth;

    for (std::size_t pixel = 0;
         pixel < pixelCount;
         ++pixel)
    {
        const ResolvedPixel& resolvedPixel =
            resolved[pixel];

        float noise =
            RawNoiseDisplayValue(
                resolvedPixel.rawVariance);
        out.variance[pixel] =
            Vec3(noise);

        int slot =
            DominantResolvedLayerSlot(
                resolvedPixel);

        if (slot < 0)
            continue;

        const ResolvedLayer& layer =
            resolvedPixel.layers[slot];

        float coverage =
            Saturate(
                layer.guide.coverage);

        float confidence =
            Saturate(
                layer.guide.coverageConfidence);

        out.coverage[pixel] =
            Vec3(coverage);

        out.coverageConfidence[pixel] =
            Vec3(confidence);

        out.surfaceGroup[pixel] =
            SurfaceGroupDebugColor(
                layer.guide.identity);

        out.normal[pixel] =
            Vec3(
                0.5f *
                    (layer.guide.normal.x + 1.0f),
                0.5f *
                    (layer.guide.normal.y + 1.0f),
                0.5f *
                    (layer.guide.normal.z + 1.0f));

        float depthDisplay = 0.0f;
        if (hasDepthRange &&
            IsFinite(layer.guide.depth))
        {
            float normalized =
                (layer.guide.depth -
                 minDepth) /
                (maxDepth -
                 minDepth);

            // Near = white, far = black.
            depthDisplay =
                1.0f -
                Saturate(normalized);
        }

        out.depth[pixel] =
            Vec3(depthDisplay);

        const int signalIndex =
            LayerIndex(
                static_cast<int>(pixel),
                slot);

        float diffuseStrength =
            signalIndex <
                    static_cast<int>(
                        diffuseEffectiveStrength.size())
                ? diffuseEffectiveStrength[
                    signalIndex]
                : 0.0f;

        float specularStrength =
            signalIndex <
                    static_cast<int>(
                        specularEffectiveStrength.size())
                ? specularEffectiveStrength[
                    signalIndex]
                : 0.0f;

        float strength =
            std::max(
                diffuseStrength,
                specularStrength);

        out.filterStrength[pixel] =
            Vec3(Saturate(strength));
    }

    return out;
}

} // namespace render
