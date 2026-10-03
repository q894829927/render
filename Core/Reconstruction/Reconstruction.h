#pragma once
#include <cstdint>
#include "Core/Integrator/PathSample.h"
#include "Core/Sampling/SampleGenerator.h"

namespace render {

constexpr int kPrimarySurfaceSlots = 4;

RENDER_HD inline Vec3 Square(const Vec3& v) {
    return {v.x*v.x, v.y*v.y, v.z*v.z};
}

RENDER_HD inline Vec3 MaxZero(const Vec3& v) {
    return {fmaxf(v.x, 0.0f), fmaxf(v.y, 0.0f), fmaxf(v.z, 0.0f)};
}

RENDER_HD inline Vec3 SafeDivideColor(const Vec3& value, const Vec3& divisor) {
    return {
        fabsf(divisor.x) > 1e-4f ? value.x / divisor.x : 0.0f,
        fabsf(divisor.y) > 1e-4f ? value.y / divisor.y : 0.0f,
        fabsf(divisor.z) > 1e-4f ? value.z / divisor.z : 0.0f
    };
}

struct ColorMoments {
    Vec3 sum;
    Vec3 squareSum;

    RENDER_HD void Add(const Vec3& v) {
        sum += v;
        squareSum += Square(v);
    }
};

struct ReplicateSignalAccumulator {
    Vec3 rawSum;
    Vec3 diffuseSum;
    Vec3 specularSum;
    Vec3 emissionSum;
    unsigned int sampleCount = 0;

    RENDER_HD void Add(const PathSample& sample) {
        rawSum += sample.Total();
        diffuseSum += sample.diffuse;
        specularSum += sample.specular;
        emissionSum += sample.emission;
        ++sampleCount;
    }
};

struct SurfaceLayerAccumulator {
    std::uint32_t primitiveId = kInvalidPrimitiveId;
    unsigned int count = 0;

    Vec3 normalSum;
    Vec3 albedoSum;
    float depthSum = 0.0f;
    float depthSqSum = 0.0f;
    float roughnessSum = 0.0f;
    float metallicSum = 0.0f;

    ColorMoments diffuseIllumination;
    ColorMoments specular;
    Vec3 emissionSum;
};

struct PixelAccumulator {
    ColorMoments raw;
    SurfaceLayerAccumulator layers[kPrimarySurfaceSlots];
    ReplicateSignalAccumulator replicates[kSampleReplicateCount];
    Vec3 overflowSum;
    unsigned int sampleCount = 0;
};

struct SurfaceGuide {
    Vec3 normal;
    Vec3 albedo;
    float depth = kInf;
    float roughness = 0.0f;
    float metallic = 0.0f;
    float coverage = 0.0f;
    std::uint32_t primitiveId = kInvalidPrimitiveId;
};

struct ResolvedLayer {
    SurfaceGuide guide;
    Vec3 diffuseIllumination;
    Vec3 diffuseVariance;
    Vec3 specular;
    Vec3 specularVariance;
    Vec3 emission;
    int valid = 0;
};

struct ResolvedPixel {
    Vec3 raw;
    Vec3 rawVariance;
    Vec3 overflow;
    ResolvedLayer layers[kPrimarySurfaceSlots];
};

RENDER_HD inline int FindOrCreateLayer(PixelAccumulator& pixel, std::uint32_t primitiveId) {
    for (int i = 0; i < kPrimarySurfaceSlots; ++i) {
        if (pixel.layers[i].count > 0 && pixel.layers[i].primitiveId == primitiveId)
            return i;
    }
    for (int i = 0; i < kPrimarySurfaceSlots; ++i) {
        if (pixel.layers[i].count == 0) {
            pixel.layers[i].primitiveId = primitiveId;
            return i;
        }
    }
    return -1;
}

RENDER_HD inline void AccumulatePathSample(PixelAccumulator& pixel, const PathSample& sample) {
    Vec3 total = sample.Total();
    pixel.raw.Add(total);
    ++pixel.sampleCount;

    if (sample.replicateId < kSampleReplicateCount)
        pixel.replicates[sample.replicateId].Add(sample);

    if (!sample.primary.valid) {
        pixel.overflowSum += total;
        return;
    }

    int slot = FindOrCreateLayer(pixel, sample.primary.primitiveId);
    if (slot < 0) {
        pixel.overflowSum += total;
        return;
    }

    SurfaceLayerAccumulator& layer = pixel.layers[slot];
    ++layer.count;
    layer.normalSum += sample.primary.normal;
    layer.albedoSum += sample.primary.albedo;
    layer.depthSum += sample.primary.depth;
    layer.depthSqSum += sample.primary.depth * sample.primary.depth;
    layer.roughnessSum += sample.primary.roughness;
    layer.metallicSum += sample.primary.metallic;

    Vec3 diffuseIllumination =
        SafeDivideColor(sample.diffuse, sample.primary.albedo);
    layer.diffuseIllumination.Add(diffuseIllumination);
    layer.specular.Add(sample.specular);
    layer.emissionSum += sample.emission;
}

RENDER_HD inline Vec3 Mean(const ColorMoments& m, unsigned int count) {
    return count > 0 ? m.sum / static_cast<float>(count) : Vec3(0.0f);
}

RENDER_HD inline Vec3 VarianceOfMean(const ColorMoments& m, unsigned int count) {
    if (count <= 1) return Vec3(0.0f);
    float n = static_cast<float>(count);
    Vec3 mean = m.sum / n;
    Vec3 sampleVariance = MaxZero(m.squareSum / n - Square(mean));
    return sampleVariance / n;
}

RENDER_HD inline ResolvedPixel ResolvePixel(const PixelAccumulator& pixel) {
    ResolvedPixel out{};
    if (pixel.sampleCount == 0) return out;

    out.raw = Mean(pixel.raw, pixel.sampleCount);
    out.rawVariance = VarianceOfMean(pixel.raw, pixel.sampleCount);
    out.overflow = pixel.overflowSum / static_cast<float>(pixel.sampleCount);

    for (int i = 0; i < kPrimarySurfaceSlots; ++i) {
        const SurfaceLayerAccumulator& src = pixel.layers[i];
        if (src.count == 0) continue;

        float inv = 1.0f / static_cast<float>(src.count);
        ResolvedLayer& dst = out.layers[i];
        dst.valid = 1;
        dst.guide.primitiveId = src.primitiveId;
        dst.guide.normal = Normalize(src.normalSum * inv);
        dst.guide.albedo = src.albedoSum * inv;
        dst.guide.depth = src.depthSum * inv;
        dst.guide.roughness = src.roughnessSum * inv;
        dst.guide.metallic = src.metallicSum * inv;
        dst.guide.coverage =
            static_cast<float>(src.count) / static_cast<float>(pixel.sampleCount);

        dst.diffuseIllumination = Mean(src.diffuseIllumination, src.count);
        dst.diffuseVariance = VarianceOfMean(src.diffuseIllumination, src.count);
        dst.specular = Mean(src.specular, src.count);
        dst.specularVariance = VarianceOfMean(src.specular, src.count);
        dst.emission = src.emissionSum * inv;
    }

    return out;
}

RENDER_HD inline int FindResolvedLayer(
    const ResolvedPixel& pixel, std::uint32_t primitiveId)
{
    for (int i = 0; i < kPrimarySurfaceSlots; ++i) {
        if (pixel.layers[i].valid &&
            pixel.layers[i].guide.primitiveId == primitiveId)
            return i;
    }
    return -1;
}

RENDER_HD inline int LayerIndex(int pixelIndex, int slot) {
    return pixelIndex * kPrimarySurfaceSlots + slot;
}

} // namespace render
