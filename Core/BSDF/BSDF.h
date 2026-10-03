#pragma once

#include "Core/Material/Material.h"

namespace render {

RENDER_HD inline Vec3 FresnelSchlick(float cosTheta, const Vec3& F0) {
    float f = powf(1.0f - Saturate(cosTheta), 5.0f);
    return F0 + (Vec3(1.0f) - F0) * f;
}

RENDER_HD inline float DistributionGGX(const Vec3& N, const Vec3& H, float roughness) {
    float alpha = fmaxf(roughness * roughness, 0.0025f);
    float a2 = alpha * alpha;
    float NdotH = fmaxf(Dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
    return a2 / fmaxf(kPi * denom * denom, 1e-12f);
}

RENDER_HD inline float GeometrySmithG1(float NdotX, float roughness) {
    float alpha = fmaxf(roughness * roughness, 0.0025f);
    float a2 = alpha * alpha;
    float x2 = NdotX * NdotX;
    return (2.0f * NdotX) /
           fmaxf(NdotX + sqrtf(a2 + (1.0f - a2) * x2), 1e-12f);
}

RENDER_HD inline float GeometrySmith(const Vec3& N, const Vec3& V, const Vec3& L, float roughness) {
    float NdotV = fmaxf(Dot(N, V), 0.0f);
    float NdotL = fmaxf(Dot(N, L), 0.0f);
    return GeometrySmithG1(NdotV, roughness) * GeometrySmithG1(NdotL, roughness);
}

RENDER_HD inline Vec3 EvaluateBRDF(const Material& mat, const Vec3& N, const Vec3& V, const Vec3& L) {
    float NdotV = fmaxf(Dot(N, V), 0.0f);
    float NdotL = fmaxf(Dot(N, L), 0.0f);
    if (NdotV <= 0.0f || NdotL <= 0.0f) return Vec3(0.0f);

    Vec3 H = Normalize(V + L);
    Vec3 F0 = Mix(Vec3(0.04f), mat.baseColor, mat.metallic);
    float D = DistributionGGX(N, H, mat.roughness);
    float G = GeometrySmith(N, V, L, mat.roughness);
    Vec3 F = FresnelSchlick(fmaxf(Dot(V, H), 0.0f), F0);

    Vec3 specular = F * (D * G / fmaxf(4.0f * NdotV * NdotL, 1e-12f));
    Vec3 kD = (Vec3(1.0f) - F) * (1.0f - mat.metallic);
    Vec3 diffuse = kD * mat.baseColor * (1.0f / kPi);
    return diffuse + specular;
}

} // namespace render
