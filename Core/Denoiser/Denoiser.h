#pragma once

#include "Core/Math/Math.h"

namespace render {

struct DenoiseSettings {
    int iterations = 4;
    float phiColor = 4.0f;
    float phiDepth = 0.020f;
    float normalPower = 8.0f;
    float phiAlbedo = 0.22f;
    float phiMaterial = 0.20f;
    float phiCoverage = 0.25f;
};

struct GuideValue {
    Vec3 normal;
    Vec3 albedo;
    Vec3 material;
    float depth = kInf;
    float coverage = 0.0f;
    float confidence = 0.0f;
};

RENDER_HD inline float ComputeGuideConfidence(
    const Vec3& meanNormal,
    float meanDepth,
    float depthVariance,
    float coverage)
{
    float normalCoherence = Saturate(Length(meanNormal));
    float relativeDepthStd = sqrtf(fmaxf(depthVariance, 0.0f)) / fmaxf(meanDepth, 1.0f);
    float depthCoherence = expf(-8.0f * relativeDepthStd);
    return Saturate(coverage * (0.35f + 0.65f * normalCoherence) * depthCoherence);
}

RENDER_HD inline float GuideSimilarityWeight(
    const GuideValue& center,
    const GuideValue& sample,
    const DenoiseSettings& settings)
{
    float coverageWeight = expf(-fabsf(sample.coverage - center.coverage) / fmaxf(settings.phiCoverage, 1e-5f));

    bool centerFinite = IsFinite(center.depth);
    bool sampleFinite = IsFinite(sample.depth);
    if (!centerFinite && !sampleFinite) return coverageWeight;
    if (centerFinite != sampleFinite) return coverageWeight * 0.01f;

    float guideTrust = 0.35f + 0.65f * fminf(center.confidence, sample.confidence);
    float effectiveDepthPhi = settings.phiDepth / fmaxf(guideTrust, 0.25f);
    float depthScale = fmaxf(fminf(center.depth, sample.depth), 1.0f);
    float relativeDepthDiff = fabsf(sample.depth - center.depth) / depthScale;
    float depthWeight = expf(-relativeDepthDiff / fmaxf(effectiveDepthPhi, 1e-6f));

    float normalDot = Saturate(Dot(center.normal, sample.normal));
    float normalWeight = powf(normalDot, settings.normalPower * guideTrust);

    float albedoDiff = Length(sample.albedo - center.albedo);
    float albedoWeight = expf(-albedoDiff / fmaxf(settings.phiAlbedo / fmaxf(guideTrust, 0.25f), 1e-5f));

    float materialDiff = Length(sample.material - center.material);
    float materialWeight = expf(-materialDiff / fmaxf(settings.phiMaterial / fmaxf(guideTrust, 0.25f), 1e-5f));

    return coverageWeight * depthWeight * normalWeight * albedoWeight * materialWeight;
}

} // namespace render
