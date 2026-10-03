#pragma once

#include "Core/Reconstruction/Reconstruction.h"

namespace render {

enum class DenoiseSignal : int {
    DiffuseIllumination,
    Specular
};

struct DenoiseSettings {
    int iterations = 4;
    float phiColorDiffuse = 3.0f;
    float phiColorSpecular = 2.5f;
    float phiDepth = 0.020f;
    float phiCoverage = 0.20f;
    float phiAlbedo = 0.25f;
    float phiRoughness = 0.20f;
    float diffuseNormalPower = 8.0f;
    float specularNormalPowerMin = 12.0f;
    float specularNormalPowerMax = 64.0f;
};

RENDER_HD inline float SurfaceGuideWeight(
    const SurfaceGuide& center,
    const SurfaceGuide& sample,
    const DenoiseSettings& settings,
    DenoiseSignal signal)
{
    if (center.primitiveId != sample.primitiveId) return 0.0f;

    float coverageWeight = expf(
        -fabsf(sample.coverage - center.coverage) /
        fmaxf(settings.phiCoverage, 1e-5f));

    bool centerFinite = IsFinite(center.depth);
    bool sampleFinite = IsFinite(sample.depth);
    if (centerFinite != sampleFinite) return 0.0f;

    float depthWeight = 1.0f;
    if (centerFinite && sampleFinite) {
        float depthScale = fmaxf(fminf(center.depth, sample.depth), 1.0f);
        float relativeDepthDiff = fabsf(sample.depth - center.depth) / depthScale;
        depthWeight = expf(
            -relativeDepthDiff / fmaxf(settings.phiDepth, 1e-6f));
    }

    float normalDot = Saturate(Dot(center.normal, sample.normal));
    float normalPower = settings.diffuseNormalPower;
    if (signal == DenoiseSignal::Specular) {
        float smoothness = 1.0f - Saturate(center.roughness);
        normalPower = settings.specularNormalPowerMin +
            (settings.specularNormalPowerMax - settings.specularNormalPowerMin) *
            smoothness;
    }
    float normalWeight = powf(normalDot, normalPower);

    float albedoDiff = Length(sample.albedo - center.albedo);
    float albedoWeight = expf(
        -albedoDiff / fmaxf(settings.phiAlbedo, 1e-5f));

    float roughnessWeight = 1.0f;
    if (signal == DenoiseSignal::Specular) {
        roughnessWeight = expf(
            -fabsf(sample.roughness - center.roughness) /
            fmaxf(settings.phiRoughness, 1e-5f));
    }

    return coverageWeight * depthWeight * normalWeight *
           albedoWeight * roughnessWeight;
}

RENDER_HD inline float VarianceAwareColorWeight(
    const Vec3& center,
    const Vec3& sample,
    const Vec3& centerVariance,
    const Vec3& sampleVariance,
    float phi)
{
    Vec3 d = sample - center;
    Vec3 sigma2 = centerVariance + sampleVariance + Vec3(1e-6f);

    float normalizedDistanceSquared =
        d.x*d.x / sigma2.x +
        d.y*d.y / sigma2.y +
        d.z*d.z / sigma2.z;

    float normalizedDistance = sqrtf(
        fmaxf(normalizedDistanceSquared / 3.0f, 0.0f));

    return expf(-normalizedDistance / fmaxf(phi, 1e-5f));
}

} // namespace render
