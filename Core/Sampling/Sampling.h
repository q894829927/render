#pragma once

#include "Core/BSDF/BSDF.h"
#include "Core/Scene/Scene.h"

namespace render {

RENDER_HD inline float SpecularSampleProbability(const Material& mat) {
    return Clamp(0.25f + 0.75f * mat.metallic, 0.05f, 1.0f);
}

template <typename RNG>
RENDER_DEVICE inline Vec3 SampleCosineHemisphere(const Vec3& N, RNG& rng, float& pdf) {
    float u1 = rng.NextFloat();
    float u2 = rng.NextFloat();
    float r = sqrtf(u1);
    float phi = 2.0f * kPi * u2;

    Vec3 local(
        r * cosf(phi),
        r * sinf(phi),
        sqrtf(fmaxf(0.0f, 1.0f - u1))
    );

    ONB basis(N);
    Vec3 wi = Normalize(basis.LocalToWorld(local));
    pdf = fmaxf(Dot(N, wi), 0.0f) / kPi;
    return wi;
}

RENDER_HD inline float DiffusePdf(const Vec3& N, const Vec3& L) {
    return fmaxf(Dot(N, L), 0.0f) / kPi;
}

template <typename RNG>
RENDER_DEVICE inline Vec3 SampleGGXVNDF(
    const Vec3& N,
    const Vec3& V,
    float roughness,
    RNG& rng,
    float& pdf)
{
    ONB basis(N);
    Vec3 Ve(Dot(V, basis.t), Dot(V, basis.b), Dot(V, basis.n));
    if (Ve.z <= 0.0f) { pdf = 0.0f; return Vec3(0.0f); }

    float alpha = fmaxf(roughness * roughness, 0.0025f);
    Vec3 Vh = Normalize(Vec3(alpha * Ve.x, alpha * Ve.y, Ve.z));

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    Vec3 T1 = lensq > 1e-12f
        ? Vec3(-Vh.y, Vh.x, 0.0f) / sqrtf(lensq)
        : Vec3(1.0f, 0.0f, 0.0f);
    Vec3 T2 = Cross(Vh, T1);

    float u1 = rng.NextFloat();
    float u2 = rng.NextFloat();
    float r = sqrtf(u1);
    float phi = 2.0f * kPi * u2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);

    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1*t1)) + s * t2;

    float nhz = sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2));
    Vec3 Nh = T1*t1 + T2*t2 + Vh*nhz;
    Vec3 Ne = Normalize(Vec3(alpha*Nh.x, alpha*Nh.y, fmaxf(0.0f, Nh.z)));
    Vec3 H = Normalize(basis.LocalToWorld(Ne));

    float VdotH = fmaxf(Dot(V, H), 0.0f);
    if (VdotH <= 0.0f) { pdf = 0.0f; return Vec3(0.0f); }

    Vec3 L = Normalize(Reflect(-V, H));
    float NdotL = fmaxf(Dot(N, L), 0.0f);
    float NdotV = fmaxf(Dot(N, V), 0.0f);
    if (NdotL <= 0.0f || NdotV <= 0.0f) { pdf = 0.0f; return Vec3(0.0f); }

    float D = DistributionGGX(N, H, roughness);
    float G1V = GeometrySmithG1(NdotV, roughness);
    pdf = D * G1V / fmaxf(4.0f * NdotV, 1e-12f);
    return L;
}

RENDER_HD inline float SpecularGGXVNDFPdf(
    const Material& mat,
    const Vec3& N,
    const Vec3& V,
    const Vec3& L)
{
    float NdotL = fmaxf(Dot(N, L), 0.0f);
    float NdotV = fmaxf(Dot(N, V), 0.0f);
    if (NdotL <= 0.0f || NdotV <= 0.0f) return 0.0f;

    Vec3 H = Normalize(V + L);
    float VdotH = fmaxf(Dot(V, H), 0.0f);
    float NdotH = fmaxf(Dot(N, H), 0.0f);
    if (VdotH <= 0.0f || NdotH <= 0.0f) return 0.0f;

    float D = DistributionGGX(N, H, mat.roughness);
    float G1V = GeometrySmithG1(NdotV, mat.roughness);
    return D * G1V / fmaxf(4.0f * NdotV, 1e-12f);
}

RENDER_HD inline float BSDFPdf(const Material& mat, const Vec3& N, const Vec3& V, const Vec3& L) {
    if (Dot(N, L) <= 0.0f || Dot(N, V) <= 0.0f) return 0.0f;
    float pSpec = SpecularSampleProbability(mat);
    return (1.0f - pSpec) * DiffusePdf(N, L) + pSpec * SpecularGGXVNDFPdf(mat, N, V, L);
}

template <typename RNG>
RENDER_DEVICE inline Vec3 SampleBSDF(
    const Material& mat,
    const Vec3& N,
    const Vec3& V,
    RNG& rng,
    float& pdf)
{
    Vec3 L;
    float ignored = 0.0f;
    if (rng.NextFloat() < SpecularSampleProbability(mat))
        L = SampleGGXVNDF(N, V, mat.roughness, rng, ignored);
    else
        L = SampleCosineHemisphere(N, rng, ignored);

    if (Dot(N, L) <= 0.0f) { pdf = 0.0f; return Vec3(0.0f); }
    pdf = BSDFPdf(mat, N, V, L);
    return L;
}

RENDER_HD inline float PowerHeuristic(float pdfA, float pdfB) {
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / fmaxf(a2 + b2, 1e-12f);
}

template <typename RNG>
RENDER_DEVICE inline Vec3 SampleLightPoint(const RectLight& light, RNG& rng) {
    float u = rng.NextFloat() * 2.0f - 1.0f;
    float v = rng.NextFloat() * 2.0f - 1.0f;
    return light.center + light.axisU*u + light.axisV*v;
}

RENDER_HD inline float LightPdf(const RectLight& light, const Vec3& shadingPoint, const Vec3& wi) {
    LightHit lightHit;
    Ray ray{ shadingPoint + wi*kEpsilon, wi };
    if (!HitRectLight(light, ray, kEpsilon, kInf, lightHit)) return 0.0f;

    Vec3 toLight = lightHit.position - shadingPoint;
    float dist2 = LengthSquared(toLight);
    float cosLight = fmaxf(Dot(light.normal, -wi), 0.0f);
    if (cosLight <= 0.0f) return 0.0f;
    return dist2 / fmaxf(cosLight * light.Area(), 1e-12f);
}

template <typename RNG>
RENDER_DEVICE inline Vec3 EstimateDirectMIS(
    const HitRecord& hit,
    const Vec3& V,
    const SceneView& scene,
    RNG& rng)
{
    Vec3 direct(0.0f);

    {
        Vec3 lightPoint = SampleLightPoint(scene.light, rng);
        Vec3 toLight = lightPoint - hit.position;
        float dist2 = LengthSquared(toLight);
        float dist = sqrtf(dist2);
        Vec3 L = toLight / dist;

        float NdotL = fmaxf(Dot(hit.normal, L), 0.0f);
        float cosLight = fmaxf(Dot(scene.light.normal, -L), 0.0f);
        if (NdotL > 0.0f && cosLight > 0.0f) {
            Ray shadowRay{ hit.position + hit.normal*kEpsilon, L };
            if (!Occluded(scene, shadowRay, dist)) {
                float lightPdf = dist2 / fmaxf(cosLight * scene.light.Area(), 1e-12f);
                float bsdfPdf = BSDFPdf(hit.material, hit.normal, V, L);
                Vec3 f = EvaluateBRDF(hit.material, hit.normal, V, L);
                float w = PowerHeuristic(lightPdf, bsdfPdf);
                direct += scene.light.emission * f * NdotL * (w / fmaxf(lightPdf, 1e-12f));
            }
        }
    }

    {
        float bsdfPdf = 0.0f;
        Vec3 L = SampleBSDF(hit.material, hit.normal, V, rng, bsdfPdf);
        float NdotL = fmaxf(Dot(hit.normal, L), 0.0f);
        if (bsdfPdf > 1e-12f && NdotL > 0.0f) {
            Ray sampledRay{ hit.position + hit.normal*kEpsilon, L };
            LightHit lightHit;
            if (HitRectLight(scene.light, sampledRay, kEpsilon, kInf, lightHit)) {
                HitRecord blocker;
                bool blocked = HitScene(scene, sampledRay, kEpsilon, lightHit.t - kEpsilon, blocker);
                if (!blocked) {
                    float lightPdf = LightPdf(scene.light, hit.position, L);
                    Vec3 f = EvaluateBRDF(hit.material, hit.normal, V, L);
                    float w = PowerHeuristic(bsdfPdf, lightPdf);
                    direct += scene.light.emission * f * NdotL * (w / fmaxf(bsdfPdf, 1e-12f));
                }
            }
        }
    }

    return direct;
}

} // namespace render
