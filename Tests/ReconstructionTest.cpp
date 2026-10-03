#include <cmath>
#include <cstdint>
#include <iostream>

#include "Core/Reconstruction/Film.h"
#include "Core/Reconstruction/Reconstruction.h"

using namespace render;

namespace {

bool Check(bool condition, const char* message) {
    if (condition) return true;
    std::cerr << "FAIL: " << message << '\n';
    return false;
}

bool Near(float a, float b, float eps = 1e-6f) {
    return fabsf(a - b) <= eps;
}

PathSample MakeHit(
    const SurfaceIdentity& identity,
    std::uint32_t replicateId)
{
    PathSample sample{};
    sample.primary.valid = 1;
    sample.primary.identity = identity;
    sample.primary.normal = Vec3(0,1,0);
    sample.primary.albedo = Vec3(0.7f);
    sample.primary.depth = 2.0f;
    sample.primary.roughness = 0.7f;
    sample.diffuse = Vec3(0.2f);
    sample.replicateId = replicateId;
    return sample;
}

PathSample MakeMiss(std::uint32_t replicateId) {
    PathSample sample{};
    sample.primary.valid = 0;
    sample.replicateId = replicateId;
    return sample;
}

} // namespace

int main() {
    bool ok = true;

    ok &= Check(
        Near(
            SampleReconstructionFilterOffset(
                ReconstructionFilterType::Box,
                0.0f),
            -0.5f),
        "Box filter lower support is incorrect.");

    ok &= Check(
        Near(
            SampleReconstructionFilterOffset(
                ReconstructionFilterType::Box,
                0.5f),
            0.0f),
        "Box filter center is incorrect.");

    ok &= Check(
        Near(
            SampleReconstructionFilterOffset(
                ReconstructionFilterType::Tent,
                0.5f),
            0.0f),
        "Tent filter center is incorrect.");

    ok &= Check(
        Near(
            SampleReconstructionFilterOffset(
                ReconstructionFilterType::Tent,
                0.125f),
            -0.5f),
        "Tent inverse CDF lower branch is incorrect.");

    ok &= Check(
        Near(
            SampleReconstructionFilterOffset(
                ReconstructionFilterType::Tent,
                0.875f),
            0.5f),
        "Tent inverse CDF upper branch is incorrect.");

    for (int i = 1; i < 100; ++i) {
        float u = static_cast<float>(i) / 100.0f;
        float a = SampleReconstructionFilterOffset(
            ReconstructionFilterType::Tent,
            u);
        float b = SampleReconstructionFilterOffset(
            ReconstructionFilterType::Tent,
            1.0f - u);
        ok &= Check(
            Near(a, -b, 2e-6f),
            "Tent filter lost symmetry.");
    }

    SampleGenerator generator =
        MakeSampleGenerator(12u, 0u, 123456u);
    FilmSample boxFilm =
        GenerateFilmSample(
            10,
            20,
            generator,
            ReconstructionFilterType::Box);
    ok &= Check(
        boxFilm.rasterX >= 10.0f &&
        boxFilm.rasterX < 11.0f &&
        boxFilm.rasterY >= 20.0f &&
        boxFilm.rasterY < 21.0f,
        "Box film sample escaped its pixel footprint.");

    SurfaceIdentity surface{1u, 2u, 3u, 4u};

    PixelAccumulator balanced{};
    for (std::uint32_t r = 0; r < kSampleReplicateCount; ++r) {
        for (int i = 0; i < 2; ++i)
            AccumulatePathSample(
                balanced,
                MakeHit(surface, r));
        for (int i = 0; i < 2; ++i)
            AccumulatePathSample(
                balanced,
                MakeMiss(r));
    }

    ResolvedPixel balancedResolved =
        ResolvePixel(balanced);

    const SurfaceGuide& balancedGuide =
        balancedResolved.layers[0].guide;

    ok &= Check(
        Near(balancedGuide.coverage, 0.5f),
        "Balanced surface coverage is incorrect.");
    ok &= Check(
        Near(balancedGuide.coverageVariance, 0.0f),
        "Equal replicate coverage should have zero replicate variance.");
    ok &= Check(
        Near(balancedGuide.coverageConfidence, 1.0f),
        "Stable 16-sample coverage should reach full confidence.");
    ok &= Check(
        Near(
            balancedResolved.backgroundVisibility.coverage,
            0.5f),
        "Background visibility coverage is incorrect.");
    ok &= Check(
        Near(
            balancedGuide.coverage +
            balancedResolved.backgroundVisibility.coverage +
            balancedResolved.overflowVisibility.coverage,
            1.0f),
        "Visibility mass does not sum to one.");

    PixelAccumulator imbalanced{};
    for (std::uint32_t r = 0; r < kSampleReplicateCount; ++r) {
        const bool hitReplicate = (r % 2u) == 0u;
        for (int i = 0; i < 4; ++i) {
            AccumulatePathSample(
                imbalanced,
                hitReplicate
                    ? MakeHit(surface, r)
                    : MakeMiss(r));
        }
    }

    ResolvedPixel imbalancedResolved =
        ResolvePixel(imbalanced);

    const SurfaceGuide& imbalancedGuide =
        imbalancedResolved.layers[0].guide;

    ok &= Check(
        Near(imbalancedGuide.coverage, 0.5f),
        "Imbalanced test should preserve mean coverage.");
    ok &= Check(
        imbalancedGuide.coverageVariance > 0.0f,
        "Disagreeing replicates should produce non-zero coverage variance.");
    ok &= Check(
        imbalancedGuide.coverageConfidence <
            balancedGuide.coverageConfidence,
        "Disagreeing replicates should reduce coverage confidence.");

    PixelAccumulator oneSample{};
    AccumulatePathSample(
        oneSample,
        MakeHit(surface, 0u));
    ResolvedPixel oneResolved =
        ResolvePixel(oneSample);

    ok &= Check(
        oneResolved.layers[0].guide.coverageConfidence == 0.0f,
        "Single-replicate coverage must not be treated as high confidence.");
    ok &= Check(
        Near(
            oneResolved.layers[0].guide.coverageVariance,
            0.25f),
        "Single-replicate coverage requires conservative variance.");

    if (!ok) return 1;
    std::cout << "Reconstruction tests passed.\n";
    return 0;
}
