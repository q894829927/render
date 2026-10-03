#pragma once
#include "Core/Integrator/PathSample.h"
#include "Core/Sampling/Sampling.h"

namespace render {

RENDER_DEVICE inline PathSample TracePath(
    Ray ray, const SceneView& scene, int maxDepth,
    const SampleGenerator& samples)
{
    PathSample result{};
    result.replicateId = samples.replicateId;

    Vec3 diffuseThroughput(0.0f);
    Vec3 specularThroughput(0.0f);

    for (int bounce = 0; bounce < maxDepth; ++bounce) {
        SceneHit hitInfo;
        if (!IntersectSceneWithLight(scene, ray, kEpsilon, kInf, hitInfo)) break;

        if (hitInfo.type == HitType::Light) {
            if (bounce == 0) {
                result.emission += hitInfo.lightHit.emission;
                result.primary.normal = Normalize(hitInfo.lightHit.normal);
                result.primary.albedo = Vec3(1.0f);
                result.primary.depth = hitInfo.lightHit.t;
                result.primary.roughness = 0.0f;
                result.primary.metallic = 0.0f;
                result.primary.primitiveId = kLightPrimitiveId;
                result.primary.valid = 1;
            }
            break;
        }

        const HitRecord& hit = hitInfo.surfaceHit;
        if (bounce == 0) {
            result.primary.normal = Normalize(hit.normal);
            result.primary.albedo = hit.material.baseColor;
            result.primary.depth = hit.t;
            result.primary.roughness = hit.material.roughness;
            result.primary.metallic = hit.material.metallic;
            result.primary.primitiveId = hit.primitiveId;
            result.primary.valid = 1;
        }

        Vec3 V = Normalize(-ray.direction);
        DirectLightingSample direct = EstimateDirectMIS(hit, V, scene, samples, bounce);

        if (bounce == 0) {
            result.diffuse += direct.diffuse;
            result.specular += direct.specular;
        } else {
            Vec3 localDirect = direct.Total();
            result.diffuse += diffuseThroughput * localDirect;
            result.specular += specularThroughput * localDirect;
        }

        BsdfDirectionSample bs = SampleBSDF(
            hit.material, hit.normal, V,
            samples.Sample1D(BounceSampleDimension(bounce, BounceSampleOffset::PathBsdfLobe)),
            samples.Sample1D(BounceSampleDimension(bounce, BounceSampleOffset::PathBsdfU)),
            samples.Sample1D(BounceSampleDimension(bounce, BounceSampleOffset::PathBsdfV)));

        float cosTheta = fmaxf(Dot(hit.normal, bs.direction), 0.0f);
        if (bs.pdf <= 1e-12f || cosTheta <= 0.0f) break;

        BRDFLobes f = EvaluateBRDFLobes(
            hit.material, hit.normal, V, bs.direction);

        if (bounce == 0) {
            diffuseThroughput = f.diffuse * (cosTheta / bs.pdf);
            specularThroughput = f.specular * (cosTheta / bs.pdf);
        } else {
            Vec3 factor = f.Total() * (cosTheta / bs.pdf);
            diffuseThroughput = diffuseThroughput * factor;
            specularThroughput = specularThroughput * factor;
        }

        Vec3 totalThroughput = diffuseThroughput + specularThroughput;

        if (bounce >= kRussianRouletteStartBounce) {
            float survive = Clamp(MaxComponent(totalThroughput), 0.05f, 0.95f);
            float rrSample = samples.Sample1D(
                BounceSampleDimension(bounce, BounceSampleOffset::RussianRoulette));
            if (rrSample > survive) break;
            diffuseThroughput = diffuseThroughput / survive;
            specularThroughput = specularThroughput / survive;
            totalThroughput = diffuseThroughput + specularThroughput;
        }

        if (!IsFinite(totalThroughput) || MaxComponent(totalThroughput) <= 1e-8f)
            break;

        ray.origin = hit.position + hit.normal * kEpsilon;
        ray.direction = bs.direction;
    }

    return result;
}

} // namespace render
