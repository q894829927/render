#pragma once
#include "Core/BSDF/BSDF.h"
#include "Core/Sampling/SampleGenerator.h"
#include "Core/Scene/Scene.h"

namespace render {

struct DirectLightingSample {
    Vec3 diffuse;
    Vec3 specular;
    RENDER_HD Vec3 Total() const { return diffuse + specular; }
};

struct BsdfDirectionSample {
    Vec3 direction;
    float pdf = 0.0f;
};

RENDER_HD inline float SpecularSampleProbability(const Material& mat) {
    return Clamp(0.25f + 0.75f * mat.metallic, 0.05f, 1.0f);
}

RENDER_HD inline Vec3 SampleCosineHemisphere(
    const Vec3& N, float u1, float u2, float& pdf)
{
    float r = sqrtf(u1);
    float phi = 2.0f * kPi * u2;
    Vec3 local(r * cosf(phi), r * sinf(phi), sqrtf(fmaxf(0.0f, 1.0f - u1)));
    ONB basis(N);
    Vec3 wi = Normalize(basis.LocalToWorld(local));
    pdf = fmaxf(Dot(N, wi), 0.0f) / kPi;
    return wi;
}

RENDER_HD inline float DiffusePdf(const Vec3& N, const Vec3& L) {
    return fmaxf(Dot(N, L), 0.0f) / kPi;
}

RENDER_HD inline Vec3 SampleGGXVNDF(
    const Vec3& N, const Vec3& V, float roughness,
    float u1, float u2, float& pdf)
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
    const Material& mat, const Vec3& N, const Vec3& V, const Vec3& L)
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

RENDER_HD inline float BSDFPdf(
    const Material& mat, const Vec3& N, const Vec3& V, const Vec3& L)
{
    if (Dot(N, L) <= 0.0f || Dot(N, V) <= 0.0f) return 0.0f;
    float pSpec = SpecularSampleProbability(mat);
    return (1.0f - pSpec) * DiffusePdf(N, L) +
           pSpec * SpecularGGXVNDFPdf(mat, N, V, L);
}

RENDER_HD inline BsdfDirectionSample SampleBSDF(
    const Material& mat, const Vec3& N, const Vec3& V,
    float lobeSample, float u1, float u2)
{
    BsdfDirectionSample result;
    float ignored = 0.0f;
    if (lobeSample < SpecularSampleProbability(mat))
        result.direction = SampleGGXVNDF(N, V, mat.roughness, u1, u2, ignored);
    else
        result.direction = SampleCosineHemisphere(N, u1, u2, ignored);

    if (Dot(N, result.direction) <= 0.0f) return {};
    result.pdf = BSDFPdf(mat, N, V, result.direction);
    return result;
}

RENDER_HD inline float PowerHeuristic(float pdfA, float pdfB) {
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / fmaxf(a2 + b2, 1e-12f);
}

RENDER_HD inline Vec3 SampleLightPoint(const RectLight& light, float u, float v) {
    float su = u * 2.0f - 1.0f;
    float sv = v * 2.0f - 1.0f;
    return light.center + light.axisU*su + light.axisV*sv;
}

RENDER_HD inline float LightPdf(
    const RectLight& light, const Vec3& shadingPoint, const Vec3& wi)
{
    LightHit lightHit;
    Ray ray{ shadingPoint + wi*kEpsilon, wi };
    if (!HitRectLight(light, ray, kEpsilon, kInf, lightHit)) return 0.0f;

    Vec3 toLight = lightHit.position - shadingPoint;
    float dist2 = LengthSquared(toLight);
    float cosLight = fmaxf(Dot(light.normal, -wi), 0.0f);
    if (cosLight <= 0.0f) return 0.0f;
    return dist2 / fmaxf(cosLight * light.Area(), 1e-12f);
}

RENDER_HD inline DirectLightingSample EstimateDirectMIS(
    const HitRecord& hit, const Vec3& V, const SceneView& scene,
    const SampleGenerator& samples, int bounce)
{
    DirectLightingSample direct{};

    {
        float lightU = samples.Sample1D(
            BounceSampleDimension(bounce, BounceSampleOffset::LightU));
        float lightV = samples.Sample1D(
            BounceSampleDimension(bounce, BounceSampleOffset::LightV));
        Vec3 lightPoint = SampleLightPoint(scene.light, lightU, lightV);
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
                BRDFLobes f = EvaluateBRDFLobes(hit.material, hit.normal, V, L);
                float w = PowerHeuristic(lightPdf, bsdfPdf);
                float scale = NdotL * (w / fmaxf(lightPdf, 1e-12f));
                direct.diffuse += scene.light.emission * f.diffuse * scale;
                direct.specular += scene.light.emission * f.specular * scale;
            }
        }
    }

    {
        BsdfDirectionSample bs = SampleBSDF(
            hit.material, hit.normal, V,
            samples.Sample1D(BounceSampleDimension(bounce, BounceSampleOffset::DirectBsdfLobe)),
            samples.Sample1D(BounceSampleDimension(bounce, BounceSampleOffset::DirectBsdfU)),
            samples.Sample1D(BounceSampleDimension(bounce, BounceSampleOffset::DirectBsdfV)));

        Vec3 L = bs.direction;
        float NdotL = fmaxf(Dot(hit.normal, L), 0.0f);
        if (bs.pdf > 1e-12f && NdotL > 0.0f) {
            Ray sampledRay{ hit.position + hit.normal*kEpsilon, L };
            LightHit lightHit;
            if (HitRectLight(scene.light, sampledRay, kEpsilon, kInf, lightHit)) {
                HitRecord blocker;
                bool blocked = HitScene(
                    scene, sampledRay, kEpsilon, lightHit.t - kEpsilon, blocker);
                if (!blocked) {
                    float lightPdf = LightPdf(scene.light, hit.position, L);
                    BRDFLobes f = EvaluateBRDFLobes(hit.material, hit.normal, V, L);
                    float w = PowerHeuristic(bs.pdf, lightPdf);
                    float scale = NdotL * (w / fmaxf(bs.pdf, 1e-12f));
                    direct.diffuse += scene.light.emission * f.diffuse * scale;
                    direct.specular += scene.light.emission * f.specular * scale;
                }
            }
        }
    }

    return direct;
}

} // namespace render
