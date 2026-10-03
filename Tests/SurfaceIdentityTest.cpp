#include <cmath>
#include <cstdint>
#include <iostream>

#include "Core/Denoiser/Denoiser.h"
#include "Core/Scene/CornellBox.h"

using namespace render;

namespace {

bool Check(bool condition, const char* message) {
    if (condition) return true;
    std::cerr << "FAIL: " << message << '\n';
    return false;
}

PathSample MakePrimarySample(
    const SurfaceIdentity& identity,
    const Vec3& normal = Vec3(0,1,0))
{
    PathSample sample{};
    sample.primary.valid = 1;
    sample.primary.identity = identity;
    sample.primary.normal = normal;
    sample.primary.albedo = Vec3(0.7f);
    sample.primary.depth = 2.0f;
    sample.primary.roughness = 0.7f;
    sample.primary.metallic = 0.0f;
    sample.diffuse = Vec3(0.2f);
    sample.replicateId = 0u;
    return sample;
}

SurfaceGuide MakeGuide(
    const SurfaceIdentity& identity,
    const Vec3& normal = Vec3(0,1,0))
{
    SurfaceGuide guide{};
    guide.identity = identity;
    guide.normal = normal;
    guide.albedo = Vec3(0.7f);
    guide.depth = 2.0f;
    guide.roughness = 0.7f;
    guide.metallic = 0.0f;
    guide.coverage = 1.0f;
    return guide;
}

} // namespace

int main() {
    bool ok = true;

    SurfaceIdentity triA{10u, 100u, 3u, 7u};
    SurfaceIdentity triB{10u, 101u, 3u, 7u};
    SurfaceIdentity otherGroup{10u, 102u, 3u, 8u};

    ok &= Check(
        SameReconstructionSurface(triA, triB),
        "Adjacent primitives in the same surface group must share a reconstruction layer.");
    ok &= Check(
        !SameReconstructionSurface(triA, otherGroup),
        "Different surface groups must remain separate reconstruction layers.");

    PixelAccumulator pixel{};
    AccumulatePathSample(pixel, MakePrimarySample(triA));
    AccumulatePathSample(pixel, MakePrimarySample(triB));
    ok &= Check(
        pixel.layers[0].count == 2u,
        "Same surfaceGroup samples were not merged into one layer.");
    ok &= Check(
        pixel.layers[1].count == 0u,
        "Same surfaceGroup unexpectedly consumed a second layer.");
    ok &= Check(
        pixel.layers[0].identity.primitiveId == kInvalidSurfaceId,
        "Merged reconstruction layer should not pretend to have one primitiveId.");

    AccumulatePathSample(pixel, MakePrimarySample(otherGroup));
    ok &= Check(
        pixel.layers[1].count == 1u,
        "Different surfaceGroup did not allocate a separate layer.");

    Material white{Vec3(0.73f), 0.0f, 0.75f};
    OrientedBox box{
        Vec3(0.0f),
        Vec3(1.0f),
        0.0f,
        Vec3(0.0f),
        white,
        42u,
        5u,
        100u
    };

    HitRecord topHit;
    Ray topRay{
        Vec3(0.5f, 2.0f, 0.5f),
        Vec3(0.0f, -1.0f, 0.0f)
    };
    ok &= Check(
        HitOrientedBox(box, topRay, 0.001f, kInf, topHit),
        "Top-face ray missed the test box.");
    ok &= Check(
        topHit.identity.instanceId == 42u &&
        topHit.identity.primitiveId ==
            static_cast<std::uint32_t>(BoxFace::PosY) &&
        topHit.identity.surfaceGroupId ==
            100u + static_cast<std::uint32_t>(BoxFace::PosY),
        "Top box face identity is incorrect.");

    HitRecord sideHit;
    Ray sideRay{
        Vec3(-1.0f, 0.5f, 0.5f),
        Vec3(1.0f, 0.0f, 0.0f)
    };
    ok &= Check(
        HitOrientedBox(box, sideRay, 0.001f, kInf, sideHit),
        "Side-face ray missed the test box.");
    ok &= Check(
        sideHit.identity.primitiveId ==
            static_cast<std::uint32_t>(BoxFace::NegX) &&
        sideHit.identity.surfaceGroupId ==
            100u + static_cast<std::uint32_t>(BoxFace::NegX),
        "Side box face identity is incorrect.");
    ok &= Check(
        !SameReconstructionSurface(
            topHit.identity,
            sideHit.identity),
        "Different box faces must not share one reconstruction layer.");

    DenoiseSettings settings{};
    SurfaceGuide guideA = MakeGuide(triA);
    SurfaceGuide guideB = MakeGuide(triB);
    float sameGroupWeight = SurfaceGuideWeight(
        guideA,
        guideB,
        settings,
        DenoiseSignal::DiffuseIllumination);
    ok &= Check(
        sameGroupWeight > 0.99f,
        "Different primitiveIds in one surfaceGroup should denoise together.");

    SurfaceIdentity differentMaterial{10u, 103u, 9u, 7u};
    SurfaceGuide materialMismatch =
        MakeGuide(differentMaterial);
    ok &= Check(
        SurfaceGuideWeight(
            guideA,
            materialMismatch,
            settings,
            DenoiseSignal::DiffuseIllumination) == 0.0f,
        "Different materialIds must not be treated as the same denoise surface.");

    SurfaceGuide hardCorner =
        MakeGuide(otherGroup, Vec3(1,0,0));
    float cornerWeight = SurfaceGuideWeight(
        guideA,
        hardCorner,
        settings,
        DenoiseSignal::DiffuseIllumination);
    ok &= Check(
        cornerWeight < 1e-6f,
        "Hard normal discontinuity should block cross-face filtering.");

    SceneStorage cornell = MakeCornellBox();
    ok &= Check(
        cornell.boxes[0].instanceId != cornell.boxes[1].instanceId,
        "Cornell boxes require distinct instanceIds.");
    ok &= Check(
        cornell.boxes[0].materialId == cornell.boxes[1].materialId,
        "Cornell white boxes should share a materialId.");
    ok &= Check(
        cornell.rects[2].identity.instanceId ==
            kCornellRoomInstanceId &&
        cornell.rects[2].identity.surfaceGroupId !=
            cornell.rects[3].identity.surfaceGroupId,
        "Cornell floor and ceiling require distinct surface groups.");

    if (!ok) return 1;
    std::cout << "SurfaceIdentity tests passed.\n";
    return 0;
}
