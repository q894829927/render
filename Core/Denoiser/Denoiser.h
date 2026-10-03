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

    // E5 adaptive filtering controls. These operate on the variance of the
    // reconstructed mean, not on per-path IID variance.
    float adaptiveNoiseThresholdDiffuse = 0.12f;
    float adaptiveNoiseThresholdSpecular = 0.16f;
    float adaptiveSignalFloorDiffuse = 0.05f;
    float adaptiveSignalFloorSpecular = 0.02f;
    float adaptiveSampleTarget = 64.0f;
    float adaptiveLowSampleBoost = 0.50f;
    float adaptiveGeometryFloor = 0.15f;
    float adaptiveCoverageFloor = 0.25f;
    float adaptiveSpecularMinRadiusFactor = 0.15f;
};

// Continuous E5 decision: signal noise determines demand while geometry,
 // visibility confidence and specular roughness gate how far that demand may
 // propagate. No per-pixel hard iteration cutoff is used.
struct AdaptiveFilterDecision {
    float filterStrength = 0.0f;
    float geometryConfidence = 0.0f;
    float noiseConfidence = 0.0f;
};

RENDER_HD inline float SurfaceIdentityCompatibilityWeight(
    const SurfaceIdentity& center,
    const SurfaceIdentity& sample)
{
    if (HasValidMaterialId(center) &&
        HasValidMaterialId(sample) &&
        center.materialId != sample.materialId)
        return 0.0f;

    if (SameReconstructionSurface(center, sample))
        return 1.0f;

    if (center.instanceId == sample.instanceId)
        return 0.75f;

    return 0.35f;
}

RENDER_HD inline float SurfaceGeometryWeight(
    const SurfaceGuide& center,
    const SurfaceGuide& sample,
    const DenoiseSettings& settings,
    DenoiseSignal signal)
{
    float identityWeight =
        SurfaceIdentityCompatibilityWeight(
            center.identity,
            sample.identity);
    if (identityWeight <= 0.0f)
        return 0.0f;

    bool centerFinite = IsFinite(center.depth);
    bool sampleFinite = IsFinite(sample.depth);
    if (centerFinite != sampleFinite)
        return 0.0f;

    float depthWeight = 1.0f;
    if (centerFinite && sampleFinite) {
        float depthScale =
            fmaxf(
                fminf(center.depth, sample.depth),
                1.0f);
        float relativeDepthDiff =
            fabsf(sample.depth - center.depth) /
            depthScale;
        depthWeight = expf(
            -relativeDepthDiff /
            fmaxf(settings.phiDepth, 1e-6f));
    }

    float normalDot =
        Saturate(Dot(center.normal, sample.normal));

    float normalPower =
        settings.diffuseNormalPower;
    if (signal == DenoiseSignal::Specular) {
        float smoothness =
            1.0f - Saturate(center.roughness);
        normalPower =
            settings.specularNormalPowerMin +
            (settings.specularNormalPowerMax -
             settings.specularNormalPowerMin) *
            smoothness;
    }

    float normalWeight =
        powf(normalDot, normalPower);

    float albedoDiff =
        Length(sample.albedo - center.albedo);
    float albedoWeight = expf(
        -albedoDiff /
        fmaxf(settings.phiAlbedo, 1e-5f));

    float roughnessWeight = 1.0f;
    if (signal == DenoiseSignal::Specular) {
        roughnessWeight = expf(
            -fabsf(
                sample.roughness -
                center.roughness) /
            fmaxf(
                settings.phiRoughness,
                1e-5f));
    }

    return identityWeight *
           depthWeight *
           normalWeight *
           albedoWeight *
           roughnessWeight;
}

RENDER_HD inline float SurfaceGuideWeight(
    const SurfaceGuide& center,
    const SurfaceGuide& sample,
    const DenoiseSettings& settings,
    DenoiseSignal signal)
{
    float geometryWeight =
        SurfaceGeometryWeight(
            center,
            sample,
            settings,
            signal);
    if (geometryWeight <= 0.0f)
        return 0.0f;

    float coverageWeight = expf(
        -fabsf(sample.coverage - center.coverage) /
        fmaxf(settings.phiCoverage, 1e-5f));

    float coverageConfidence =
        fminf(
            center.coverageConfidence,
            sample.coverageConfidence);

    float coverageConfidenceWeight =
        settings.adaptiveCoverageFloor +
        (1.0f - settings.adaptiveCoverageFloor) *
        Saturate(coverageConfidence);

    return geometryWeight *
           coverageWeight *
           coverageConfidenceWeight;
}

RENDER_HD inline int FindBestDenoiseLayer(
    const ResolvedPixel& pixel,
    const SurfaceGuide& centerGuide,
    const DenoiseSettings& settings,
    DenoiseSignal signal)
{
    int bestSlot = -1;
    float bestWeight = 0.0f;

    for (int slot = 0;
         slot < kPrimarySurfaceSlots;
         ++slot)
    {
        const ResolvedLayer& layer =
            pixel.layers[slot];
        if (!layer.valid)
            continue;

        float weight =
            SurfaceGuideWeight(
                centerGuide,
                layer.guide,
                settings,
                signal);

        if (weight > bestWeight) {
            bestWeight = weight;
            bestSlot = slot;
        }
    }

    return bestSlot;
}

RENDER_HD inline float VarianceAwareColorWeight(
    const Vec3& center,
    const Vec3& sample,
    const Vec3& centerVariance,
    const Vec3& sampleVariance,
    float phi)
{
    Vec3 d = sample - center;
    Vec3 sigma2 =
        centerVariance +
        sampleVariance +
        Vec3(1e-6f);

    float normalizedDistanceSquared =
        d.x*d.x / sigma2.x +
        d.y*d.y / sigma2.y +
        d.z*d.z / sigma2.z;

    float normalizedDistance = sqrtf(
        fmaxf(
            normalizedDistanceSquared / 3.0f,
            0.0f));

    return expf(
        -normalizedDistance /
        fmaxf(phi, 1e-5f));
}

RENDER_HD inline float SignalRms(const Vec3& value) {
    return sqrtf(
        fmaxf(
            (value.x*value.x +
             value.y*value.y +
             value.z*value.z) / 3.0f,
            0.0f));
}

RENDER_HD inline float StandardErrorRms(
    const Vec3& variance)
{
    return sqrtf(
        fmaxf(
            (variance.x +
             variance.y +
             variance.z) / 3.0f,
            0.0f));
}

RENDER_HD inline AdaptiveFilterDecision
ComputeAdaptiveFilterDecision(
    const SurfaceGuide& guide,
    const Vec3& color,
    const Vec3& variance,
    float geometryConfidence,
    int step,
    DenoiseSignal signal,
    const DenoiseSettings& settings)
{
    AdaptiveFilterDecision decision{};

    decision.geometryConfidence =
        Saturate(geometryConfidence);

    float signalFloor =
        signal == DenoiseSignal::DiffuseIllumination
            ? settings.adaptiveSignalFloorDiffuse
            : settings.adaptiveSignalFloorSpecular;

    float threshold =
        signal == DenoiseSignal::DiffuseIllumination
            ? settings.adaptiveNoiseThresholdDiffuse
            : settings.adaptiveNoiseThresholdSpecular;

    float relativeNoise =
        StandardErrorRms(variance) /
        fmaxf(
            SignalRms(color) + signalFloor,
            1e-5f);

    float sampleConfidence =
        Saturate(
            static_cast<float>(
                guide.primarySampleCount) /
            fmaxf(
                settings.adaptiveSampleTarget,
                1.0f));

    float sampleScarcity =
        1.0f - sampleConfidence;

    float scarcityBoost =
        1.0f +
        settings.adaptiveLowSampleBoost *
        sampleScarcity;

    // Wider A-Trous radii require progressively stronger evidence that the
    // signal is still noisy. This makes later iterations naturally fade out.
    float radiusThresholdScale =
        sqrtf(
            static_cast<float>(
                step > 0 ? step : 1));

    decision.noiseConfidence =
        Saturate(
            relativeNoise *
            scarcityBoost /
            fmaxf(
                threshold *
                radiusThresholdScale,
                1e-5f));

    float geometryFactor =
        settings.adaptiveGeometryFloor +
        (1.0f - settings.adaptiveGeometryFloor) *
        decision.geometryConfidence;

    float coverageFactor =
        settings.adaptiveCoverageFloor +
        (1.0f - settings.adaptiveCoverageFloor) *
        Saturate(guide.coverageConfidence);

    float signalRadiusFactor = 1.0f;
    if (signal == DenoiseSignal::Specular) {
        signalRadiusFactor =
            settings.adaptiveSpecularMinRadiusFactor +
            (1.0f -
             settings.adaptiveSpecularMinRadiusFactor) *
            Saturate(guide.roughness);
    }

    decision.filterStrength =
        Saturate(
            decision.noiseConfidence *
            geometryFactor *
            coverageFactor *
            signalRadiusFactor);

    return decision;
}

} // namespace render
