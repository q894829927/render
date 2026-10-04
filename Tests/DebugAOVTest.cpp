#include <cmath>
#include <iostream>
#include <vector>

#include "Core/Output/DebugAOV.h"

using namespace render;

namespace {

bool Check(bool condition, const char* message) {
    if (condition) return true;
    std::cerr << "FAIL: " << message << '\n';
    return false;
}

bool Near(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

} // namespace

int main() {
    bool ok = true;

    ResolvedPixel pixel{};

    pixel.layers[0].valid = 1;
    pixel.layers[0].guide.coverage = 0.25f;
    pixel.layers[0].guide.coverageConfidence = 0.4f;
    pixel.layers[0].guide.normal = Vec3(1,0,0);
    pixel.layers[0].guide.depth = 4.0f;
    pixel.layers[0].guide.identity =
        {1u, 10u, 3u, 2u};

    pixel.layers[1].valid = 1;
    pixel.layers[1].guide.coverage = 0.75f;
    pixel.layers[1].guide.coverageConfidence = 0.8f;
    pixel.layers[1].guide.normal = Vec3(0,1,0);
    pixel.layers[1].guide.depth = 2.0f;
    pixel.layers[1].guide.identity =
        {1u, 11u, 3u, 5u};

    pixel.rawVariance = Vec3(0.0004f);

    ok &= Check(
        DominantResolvedLayerSlot(pixel) == 1,
        "Dominant debug layer must use maximum reconstructed coverage.");

    ok &= Check(
        Near(
            CombineFilterStrength(
                0.5f,
                0.5f),
            0.75f),
        "Effective filter strength composition is incorrect.");

    Vec3 colorA =
        SurfaceGroupDebugColor(
            pixel.layers[1].guide.identity);
    Vec3 colorB =
        SurfaceGroupDebugColor(
            pixel.layers[1].guide.identity);
    Vec3 colorOther =
        SurfaceGroupDebugColor(
            SurfaceIdentity{
                1u,
                11u,
                3u,
                6u
            });

    ok &= Check(
        Near(colorA.x, colorB.x) &&
        Near(colorA.y, colorB.y) &&
        Near(colorA.z, colorB.z),
        "Surface group debug color must be deterministic.");

    ok &= Check(
        !(
            Near(colorA.x, colorOther.x) &&
            Near(colorA.y, colorOther.y) &&
            Near(colorA.z, colorOther.z)
        ),
        "Different surface groups should not share the same debug color.");

    std::vector<ResolvedPixel> resolved(2);
    resolved[0] = pixel;
    resolved[1] = pixel;
    resolved[1].layers[1].guide.depth = 6.0f;

    std::vector<float> diffuse(
        2 * kPrimarySurfaceSlots,
        0.0f);
    std::vector<float> specular(
        2 * kPrimarySurfaceSlots,
        0.0f);

    diffuse[LayerIndex(0, 1)] = 0.6f;
    specular[LayerIndex(0, 1)] = 0.2f;

    DebugAOVSet aovs =
        BuildDebugAOVs(
            resolved,
            diffuse,
            specular,
            2,
            1);

    ok &= Check(
        Near(aovs.coverage[0].x, 0.75f),
        "Coverage AOV is incorrect.");

    ok &= Check(
        Near(
            aovs.coverageConfidence[0].x,
            0.8f),
        "Coverage confidence AOV is incorrect.");

    ok &= Check(
        Near(aovs.normal[0].x, 0.5f) &&
        Near(aovs.normal[0].y, 1.0f) &&
        Near(aovs.normal[0].z, 0.5f),
        "Normal AOV encoding is incorrect.");

    ok &= Check(
        Near(
            aovs.filterStrength[0].x,
            0.6f),
        "Filter strength AOV must expose the stronger effective signal.");

    ok &= Check(
        aovs.depth[0].x >
            aovs.depth[1].x,
        "Depth AOV must encode nearer surfaces brighter.");

    ok &= Check(
        aovs.variance[0].x > 0.0f &&
        aovs.variance[0].x < 1.0f,
        "Variance/noise AOV compression is outside display range.");

    if (!ok) return 1;

    std::cout
        << "Debug AOV tests passed.\n";
    return 0;
}
