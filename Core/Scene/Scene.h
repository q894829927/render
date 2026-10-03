#pragma once

#include <cstdint>

#include "Core/Math/Math.h"
#include "Core/Material/Material.h"

namespace render {

constexpr std::uint32_t kInvalidSurfaceId = 0xffffffffu;

struct SurfaceIdentity {
    std::uint32_t instanceId = kInvalidSurfaceId;
    std::uint32_t primitiveId = kInvalidSurfaceId;
    std::uint32_t materialId = kInvalidSurfaceId;
    std::uint32_t surfaceGroupId = kInvalidSurfaceId;
};

RENDER_HD inline bool SameReconstructionSurface(
    const SurfaceIdentity& a,
    const SurfaceIdentity& b)
{
    return a.instanceId == b.instanceId &&
           a.surfaceGroupId == b.surfaceGroupId;
}

RENDER_HD inline bool HasValidMaterialId(const SurfaceIdentity& identity) {
    return identity.materialId != kInvalidSurfaceId;
}

enum class RectAxis : int { XY, XZ, YZ };

enum class BoxFace : std::uint32_t {
    NegX = 0u,
    PosX = 1u,
    NegY = 2u,
    PosY = 3u,
    NegZ = 4u,
    PosZ = 5u
};

struct Rect {
    RectAxis axis;
    float a0, a1;
    float b0, b1;
    float k;
    Vec3 outwardNormal;
    Material material;
    SurfaceIdentity identity;
};

struct OrientedBox {
    Vec3 minCorner;
    Vec3 maxCorner;
    float rotationYDeg;
    Vec3 translation;
    Material material;
    std::uint32_t instanceId = kInvalidSurfaceId;
    std::uint32_t materialId = kInvalidSurfaceId;
    std::uint32_t surfaceGroupBase = 0u;
};

struct RectLight {
    Vec3 center;
    Vec3 axisU;
    Vec3 axisV;
    Vec3 normal;
    Vec3 emission;
    SurfaceIdentity identity;

    RENDER_HD float Area() const {
        return Length(Cross(axisU * 2.0f, axisV * 2.0f));
    }
};

struct SceneView {
    const Rect* rects = nullptr;
    int rectCount = 0;
    const OrientedBox* boxes = nullptr;
    int boxCount = 0;
    RectLight light{};
};

struct HitRecord {
    Vec3 position;
    Vec3 normal;
    Material material;
    float t = 0.0f;
    bool frontFace = false;
    SurfaceIdentity identity{};

    RENDER_HD void SetFaceNormal(const Ray& ray, const Vec3& outwardNormal) {
        frontFace = Dot(ray.direction, outwardNormal) < 0.0f;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

struct LightHit {
    float t = 0.0f;
    Vec3 position;
    Vec3 normal;
    Vec3 emission;
    SurfaceIdentity identity{};
};

enum class HitType : int { None, Surface, Light };

struct SceneHit {
    HitType type = HitType::None;
    HitRecord surfaceHit{};
    LightHit lightHit{};
};

RENDER_HD inline Vec3 RotateY(const Vec3& v, float degrees) {
    float radians = degrees * kPi / 180.0f;
    float c = cosf(radians);
    float s = sinf(radians);
    return {c*v.x + s*v.z, v.y, -s*v.x + c*v.z};
}

RENDER_HD inline Vec3 InverseRotateY(const Vec3& v, float degrees) {
    return RotateY(v, -degrees);
}

RENDER_HD inline bool HitRect(
    const Rect& rect,
    const Ray& ray,
    float tMin,
    float tMax,
    HitRecord& hit)
{
    float t = 0.0f, a = 0.0f, b = 0.0f;
    switch (rect.axis) {
        case RectAxis::XY:
            if (fabsf(ray.direction.z) < 1e-8f) return false;
            t = (rect.k - ray.origin.z) / ray.direction.z;
            if (t < tMin || t > tMax) return false;
            a = ray.origin.x + t * ray.direction.x;
            b = ray.origin.y + t * ray.direction.y;
            if (a < rect.a0 || a > rect.a1 || b < rect.b0 || b > rect.b1) return false;
            break;
        case RectAxis::XZ:
            if (fabsf(ray.direction.y) < 1e-8f) return false;
            t = (rect.k - ray.origin.y) / ray.direction.y;
            if (t < tMin || t > tMax) return false;
            a = ray.origin.x + t * ray.direction.x;
            b = ray.origin.z + t * ray.direction.z;
            if (a < rect.a0 || a > rect.a1 || b < rect.b0 || b > rect.b1) return false;
            break;
        case RectAxis::YZ:
            if (fabsf(ray.direction.x) < 1e-8f) return false;
            t = (rect.k - ray.origin.x) / ray.direction.x;
            if (t < tMin || t > tMax) return false;
            a = ray.origin.y + t * ray.direction.y;
            b = ray.origin.z + t * ray.direction.z;
            if (a < rect.a0 || a > rect.a1 || b < rect.b0 || b > rect.b1) return false;
            break;
    }

    hit.t = t;
    hit.position = ray.At(t);
    hit.SetFaceNormal(ray, rect.outwardNormal);
    hit.material = rect.material;
    hit.identity = rect.identity;
    return true;
}

RENDER_HD inline void AxisFaceData(
    int axis,
    bool positiveDirection,
    Vec3& nearNormal,
    Vec3& farNormal,
    std::uint32_t& nearFace,
    std::uint32_t& farFace)
{
    if (axis == 0) {
        nearNormal = positiveDirection ? Vec3(-1,0,0) : Vec3(1,0,0);
        farNormal  = -nearNormal;
        nearFace = static_cast<std::uint32_t>(
            positiveDirection ? BoxFace::NegX : BoxFace::PosX);
        farFace = static_cast<std::uint32_t>(
            positiveDirection ? BoxFace::PosX : BoxFace::NegX);
    } else if (axis == 1) {
        nearNormal = positiveDirection ? Vec3(0,-1,0) : Vec3(0,1,0);
        farNormal  = -nearNormal;
        nearFace = static_cast<std::uint32_t>(
            positiveDirection ? BoxFace::NegY : BoxFace::PosY);
        farFace = static_cast<std::uint32_t>(
            positiveDirection ? BoxFace::PosY : BoxFace::NegY);
    } else {
        nearNormal = positiveDirection ? Vec3(0,0,-1) : Vec3(0,0,1);
        farNormal  = -nearNormal;
        nearFace = static_cast<std::uint32_t>(
            positiveDirection ? BoxFace::NegZ : BoxFace::PosZ);
        farFace = static_cast<std::uint32_t>(
            positiveDirection ? BoxFace::PosZ : BoxFace::NegZ);
    }
}

RENDER_HD inline bool HitLocalAABB(
    const Vec3& minCorner,
    const Vec3& maxCorner,
    const Ray& ray,
    float tMin,
    float tMax,
    float& tHit,
    Vec3& outwardNormal,
    std::uint32_t& faceIndex)
{
    float tEnter = -kInf;
    float tExit = kInf;
    Vec3 enterNormal(0.0f);
    Vec3 exitNormal(0.0f);
    std::uint32_t enterFace = kInvalidSurfaceId;
    std::uint32_t exitFace = kInvalidSurfaceId;

    for (int axis = 0; axis < 3; ++axis) {
        float origin = axis == 0
            ? ray.origin.x
            : (axis == 1 ? ray.origin.y : ray.origin.z);
        float direction = axis == 0
            ? ray.direction.x
            : (axis == 1 ? ray.direction.y : ray.direction.z);
        float mn = axis == 0
            ? minCorner.x
            : (axis == 1 ? minCorner.y : minCorner.z);
        float mx = axis == 0
            ? maxCorner.x
            : (axis == 1 ? maxCorner.y : maxCorner.z);

        if (fabsf(direction) < 1e-8f) {
            if (origin < mn || origin > mx) return false;
            continue;
        }

        float nearT = (mn - origin) / direction;
        float farT = (mx - origin) / direction;
        Vec3 nearNormal, farNormal;
        std::uint32_t nearFace, farFace;
        bool positiveDirection = direction > 0.0f;
        AxisFaceData(
            axis,
            positiveDirection,
            nearNormal,
            farNormal,
            nearFace,
            farFace);

        if (nearT > farT) {
            float temp = nearT;
            nearT = farT;
            farT = temp;
        }

        if (nearT > tEnter) {
            tEnter = nearT;
            enterNormal = nearNormal;
            enterFace = nearFace;
        }
        if (farT < tExit) {
            tExit = farT;
            exitNormal = farNormal;
            exitFace = farFace;
        }

        if (tExit < tEnter) return false;
    }

    if (tEnter >= tMin && tEnter <= tMax) {
        tHit = tEnter;
        outwardNormal = enterNormal;
        faceIndex = enterFace;
        return true;
    }

    if (tExit >= tMin && tExit <= tMax) {
        tHit = tExit;
        outwardNormal = exitNormal;
        faceIndex = exitFace;
        return true;
    }

    return false;
}

RENDER_HD inline SurfaceIdentity BoxSurfaceIdentity(
    const OrientedBox& box,
    std::uint32_t faceIndex)
{
    return {
        box.instanceId,
        faceIndex,
        box.materialId,
        box.surfaceGroupBase + faceIndex
    };
}

RENDER_HD inline bool HitOrientedBox(
    const OrientedBox& box,
    const Ray& ray,
    float tMin,
    float tMax,
    HitRecord& hit)
{
    Ray localRay{
        InverseRotateY(ray.origin - box.translation, box.rotationYDeg),
        InverseRotateY(ray.direction, box.rotationYDeg)
    };

    float tHit = 0.0f;
    Vec3 localNormal;
    std::uint32_t faceIndex = kInvalidSurfaceId;
    if (!HitLocalAABB(
            box.minCorner,
            box.maxCorner,
            localRay,
            tMin,
            tMax,
            tHit,
            localNormal,
            faceIndex))
        return false;

    Vec3 worldNormal = Normalize(
        RotateY(localNormal, box.rotationYDeg));

    hit.t = tHit;
    hit.position = ray.At(tHit);
    hit.SetFaceNormal(ray, worldNormal);
    hit.material = box.material;
    hit.identity = BoxSurfaceIdentity(box, faceIndex);
    return true;
}

RENDER_HD inline bool HitScene(
    const SceneView& scene,
    const Ray& ray,
    float tMin,
    float tMax,
    HitRecord& hit)
{
    bool found = false;
    float closest = tMax;
    HitRecord temp;

    for (int i = 0; i < scene.rectCount; ++i) {
        if (HitRect(scene.rects[i], ray, tMin, closest, temp)) {
            found = true;
            closest = temp.t;
            hit = temp;
        }
    }

    for (int i = 0; i < scene.boxCount; ++i) {
        if (HitOrientedBox(scene.boxes[i], ray, tMin, closest, temp)) {
            found = true;
            closest = temp.t;
            hit = temp;
        }
    }

    return found;
}

RENDER_HD inline bool HitRectLight(
    const RectLight& light,
    const Ray& ray,
    float tMin,
    float tMax,
    LightHit& outHit)
{
    float denom = Dot(ray.direction, light.normal);
    if (fabsf(denom) < 1e-8f) return false;

    float t = Dot(light.center - ray.origin, light.normal) / denom;
    if (t < tMin || t > tMax) return false;

    Vec3 p = ray.At(t);
    Vec3 rel = p - light.center;
    float u = Dot(rel, light.axisU) / Dot(light.axisU, light.axisU);
    float v = Dot(rel, light.axisV) / Dot(light.axisV, light.axisV);
    if (fabsf(u) > 1.0f || fabsf(v) > 1.0f) return false;
    if (Dot(light.normal, -ray.direction) <= 0.0f) return false;

    outHit.t = t;
    outHit.position = p;
    outHit.normal = light.normal;
    outHit.emission = light.emission;
    outHit.identity = light.identity;
    return true;
}

RENDER_HD inline bool IntersectSceneWithLight(
    const SceneView& scene,
    const Ray& ray,
    float tMin,
    float tMax,
    SceneHit& result)
{
    HitRecord surfaceHit;
    bool hitSurface = HitScene(scene, ray, tMin, tMax, surfaceHit);
    float closest = hitSurface ? surfaceHit.t : tMax;

    LightHit lightHit;
    bool hitLight = HitRectLight(scene.light, ray, tMin, closest, lightHit);

    if (hitLight) {
        result.type = HitType::Light;
        result.lightHit = lightHit;
        return true;
    }
    if (hitSurface) {
        result.type = HitType::Surface;
        result.surfaceHit = surfaceHit;
        return true;
    }
    result.type = HitType::None;
    return false;
}

RENDER_HD inline bool Occluded(
    const SceneView& scene,
    const Ray& ray,
    float maxDistance)
{
    HitRecord temp;
    return HitScene(
        scene,
        ray,
        kEpsilon,
        maxDistance - kEpsilon,
        temp);
}

} // namespace render
