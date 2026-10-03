#pragma once

#include "Core/Sampling/Sampling.h"

namespace render {

struct PrimaryGuide {
    Vec3 normal;
    Vec3 albedo;
    Vec3 material;
    float depth = kInf;
    int valid = 0;
};

template <typename RNG>
RENDER_DEVICE inline Vec3 TracePath(
    Ray ray,
    const SceneView& scene,
    int maxDepth,
    RNG& rng,
    PrimaryGuide* primaryGuide)
{
    Vec3 radiance(0.0f);
    Vec3 throughput(1.0f);

    if (primaryGuide) *primaryGuide = PrimaryGuide{};

    for (int bounce = 0; bounce < maxDepth; ++bounce) {
        SceneHit hitInfo;
        if (!IntersectSceneWithLight(scene, ray, kEpsilon, kInf, hitInfo)) break;

        if (hitInfo.type == HitType::Light) {
            if (bounce == 0 && primaryGuide) {
                primaryGuide->normal = Normalize(hitInfo.lightHit.normal);
                primaryGuide->albedo = Vec3(1.0f);
                primaryGuide->material = Vec3(0.0f, 0.0f, 1.0f);
                primaryGuide->depth = hitInfo.lightHit.t;
                primaryGuide->valid = 1;
            }
            if (bounce == 0) radiance += throughput * hitInfo.lightHit.emission;
            break;
        }

        const HitRecord& hit = hitInfo.surfaceHit;
        if (bounce == 0 && primaryGuide) {
            primaryGuide->normal = Normalize(hit.normal);
            primaryGuide->albedo = hit.material.baseColor;
            primaryGuide->material = Vec3(hit.material.metallic, hit.material.roughness, 0.0f);
            primaryGuide->depth = hit.t;
            primaryGuide->valid = 1;
        }

        Vec3 V = Normalize(-ray.direction);
        radiance += throughput * EstimateDirectMIS(hit, V, scene, rng);

        float bsdfPdf = 0.0f;
        Vec3 wi = SampleBSDF(hit.material, hit.normal, V, rng, bsdfPdf);
        float cosTheta = fmaxf(Dot(hit.normal, wi), 0.0f);
        if (bsdfPdf <= 1e-12f || cosTheta <= 0.0f) break;

        Vec3 f = EvaluateBRDF(hit.material, hit.normal, V, wi);
        throughput = throughput * f * (cosTheta / bsdfPdf);

        if (bounce >= kRussianRouletteStartBounce) {
            float survive = Clamp(MaxComponent(throughput), 0.05f, 0.95f);
            if (rng.NextFloat() > survive) break;
            throughput = throughput / survive;
        }

        if (!IsFinite(throughput) || MaxComponent(throughput) <= 1e-8f) break;

        ray.origin = hit.position + hit.normal * kEpsilon;
        ray.direction = wi;
    }

    return radiance;
}

} // namespace render
