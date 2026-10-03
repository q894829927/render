#include <cmath>
#include <iostream>

#include "Core/Denoiser/ATrous.h"

using namespace render;

namespace {

bool Check(bool condition, const char* message) {
    if (condition) return true;
    std::cerr << "FAIL: " << message << '\n';
    return false;
}

SurfaceGuide MakeGuide(
    float coverageConfidence,
    unsigned int primarySampleCount,
    float roughness)
{
    SurfaceGuide guide{};
    guide.normal = Vec3(0,1,0);
    guide.albedo = Vec3(0.7f);
    guide.depth = 2.0f;
    guide.roughness = roughness;
    guide.coverage = 1.0f;
    guide.coverageConfidence = coverageConfidence;
    guide.primarySampleCount = primarySampleCount;
    guide.identity = {1u, 2u, 3u, 4u};
    return guide;
}

} // namespace

int main() {
    bool ok = true;
    DenoiseSettings settings{};

    SurfaceGuide stable64 =
        MakeGuide(1.0f, 64u, 0.75f);

    Vec3 color(0.5f);
    Vec3 lowVariance(0.0001f);
    Vec3 highVariance(0.04f);

    AdaptiveFilterDecision lowNoise =
        ComputeAdaptiveFilterDecision(
            stable64,
            color,
            lowVariance,
            1.0f,
            1,
            DenoiseSignal::DiffuseIllumination,
            settings);

    AdaptiveFilterDecision highNoise =
        ComputeAdaptiveFilterDecision(
            stable64,
            color,
            highVariance,
            1.0f,
            1,
            DenoiseSignal::DiffuseIllumination,
            settings);

    ok &= Check(
        highNoise.filterStrength >
            lowNoise.filterStrength,
        "Higher variance must produce stronger filtering.");

    AdaptiveFilterDecision wideRadius =
        ComputeAdaptiveFilterDecision(
            stable64,
            color,
            highVariance,
            1.0f,
            8,
            DenoiseSignal::DiffuseIllumination,
            settings);

    ok &= Check(
        wideRadius.filterStrength <
            highNoise.filterStrength,
        "Wide A-Trous radii must require stronger noise evidence.");

    AdaptiveFilterDecision weakGeometry =
        ComputeAdaptiveFilterDecision(
            stable64,
            color,
            highVariance,
            0.1f,
            1,
            DenoiseSignal::DiffuseIllumination,
            settings);

    ok &= Check(
        weakGeometry.filterStrength <
            highNoise.filterStrength,
        "Geometry discontinuity must reduce filter strength.");

    SurfaceGuide uncertainCoverage =
        MakeGuide(0.1f, 64u, 0.75f);

    AdaptiveFilterDecision lowCoverageConfidence =
        ComputeAdaptiveFilterDecision(
            uncertainCoverage,
            color,
            highVariance,
            1.0f,
            1,
            DenoiseSignal::DiffuseIllumination,
            settings);

    ok &= Check(
        lowCoverageConfidence.filterStrength <
            highNoise.filterStrength,
        "Low coverage confidence must reduce filter strength.");

    SurfaceGuide scarceSamples =
        MakeGuide(1.0f, 8u, 0.75f);

    AdaptiveFilterDecision scarce =
        ComputeAdaptiveFilterDecision(
            scarceSamples,
            color,
            Vec3(0.004f),
            1.0f,
            1,
            DenoiseSignal::DiffuseIllumination,
            settings);

    AdaptiveFilterDecision many =
        ComputeAdaptiveFilterDecision(
            stable64,
            color,
            Vec3(0.004f),
            1.0f,
            1,
            DenoiseSignal::DiffuseIllumination,
            settings);

    ok &= Check(
        scarce.filterStrength >
            many.filterStrength,
        "Low primary sample count must increase noise demand.");

    SurfaceGuide smoothSpecular =
        MakeGuide(1.0f, 64u, 0.05f);

    SurfaceGuide roughSpecular =
        MakeGuide(1.0f, 64u, 0.9f);

    AdaptiveFilterDecision smooth =
        ComputeAdaptiveFilterDecision(
            smoothSpecular,
            color,
            highVariance,
            1.0f,
            4,
            DenoiseSignal::Specular,
            settings);

    AdaptiveFilterDecision rough =
        ComputeAdaptiveFilterDecision(
            roughSpecular,
            color,
            highVariance,
            1.0f,
            4,
            DenoiseSignal::Specular,
            settings);

    ok &= Check(
        smooth.filterStrength <
            rough.filterStrength,
        "Smooth specular must reject wide filtering more strongly.");

    AdaptiveFilterDecision zeroNoise =
        ComputeAdaptiveFilterDecision(
            stable64,
            color,
            Vec3(0.0f),
            1.0f,
            1,
            DenoiseSignal::DiffuseIllumination,
            settings);

    ok &= Check(
        zeroNoise.filterStrength == 0.0f,
        "Zero measured variance must not be blurred.");

    ResolvedPixel pixels[1]{};
    pixels[0].layers[0].valid = 1;
    pixels[0].layers[0].guide = stable64;

    Vec3 inputColor[4] = {
        Vec3(0.5f),
        Vec3(0.0f),
        Vec3(0.0f),
        Vec3(0.0f)
    };

    Vec3 inputVariance[4] = {
        Vec3(0.0f),
        Vec3(0.0f),
        Vec3(0.0f),
        Vec3(0.0f)
    };

    FilteredSignal noNoise =
        ATrousLayerAt(
            pixels,
            inputColor,
            inputVariance,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            DenoiseSignal::DiffuseIllumination,
            settings);

    ok &= Check(
        noNoise.filterStrength == 0.0f &&
        fabsf(noNoise.color.x - 0.5f) < 1e-6f,
        "Adaptive A-Trous must be an exact no-op for zero-noise input.");

    if (!ok) return 1;
    std::cout << "Adaptive denoiser tests passed.\n";
    return 0;
}
