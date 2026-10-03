#pragma once

#include "Core/Math/Math.h"
#include "Core/Material/Material.h"

namespace render {

enum class RectAxis : int { XY, XZ, YZ };

struct Rect {
    RectAxis axis;
    float a0, a1;
    float b0, b1;
    float k;
    Vec3 outwardNormal;
    Material material;
};

struct OrientedBox {
    Vec3 minCorner;
    Vec3 maxCorner;
    float rotationYDeg;
    Vec3 translation;
    Material material;
};

struct RectLight {
    Vec3 center;
    Vec3 axisU;
    Vec3 axisV;
    Vec3 normal;
    Vec3 emission;

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

RENDER_HD inline bool HitRect(const Rect& rect, const Ray& ray, float tMin, float tMax, HitRecord& hit) {
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
    return true;
}

RENDER_HD inline bool HitLocalAABB(
    const Vec3& minCorner,
    const Vec3& maxCorner,
    const Ray& ray,
    float tMin,
    float tMax,
    float& tHit,
    Vec3& outwardNormal)
{
    float t0 = tMin;
    float t1 = tMax;
    Vec3 normal(0.0f);

    for (int axis = 0; axis < 3; ++axis) {
        float origin = axis == 0 ? ray.origin.x : (axis == 1 ? ray.origin.y : ray.origin.z);
        float direction = axis == 0 ? ray.direction.x : (axis == 1 ? ray.direction.y : ray.direction.z);
        float mn = axis == 0 ? minCorner.x : (axis == 1 ? minCorner.y : minCorner.z);
        float mx = axis == 0 ? maxCorner.x : (axis == 1 ? maxCorner.y : maxCorner.z);

        if (fabsf(direction) < 1e-8f) {
            if (origin < mn || origin > mx) return false;
            continue;
        }

        float invD = 1.0f / direction;
        float tNear = (mn - origin) * invD;
        float tFar = (mx - origin) * invD;
        Vec3 nearNormal(0.0f);
        if (axis == 0) nearNormal = {-1.0f, 0.0f, 0.0f};
        if (axis == 1) nearNormal = {0.0f, -1.0f, 0.0f};
        if (axis == 2) nearNormal = {0.0f, 0.0f, -1.0f};

        if (invD < 0.0f) {
            float temp = tNear; tNear = tFar; tFar = temp;
            nearNormal = -nearNormal;
        }

        if (tNear > t0) {
            t0 = tNear;
            normal = nearNormal;
        }
        t1 = fminf(t1, tFar);
        if (t1 <= t0) return false;
    }

    tHit = t0;
    outwardNormal = normal;
    return true;
}

RENDER_HD inline bool HitOrientedBox(const OrientedBox& box, const Ray& ray, float tMin, float tMax, HitRecord& hit) {
    Ray localRay{
        InverseRotateY(ray.origin - box.translation, box.rotationYDeg),
        InverseRotateY(ray.direction, box.rotationYDeg)
    };

    float tHit = 0.0f;
    Vec3 localNormal;
    if (!HitLocalAABB(box.minCorner, box.maxCorner, localRay, tMin, tMax, tHit, localNormal)) return false;

    Vec3 localPos = localRay.At(tHit);
    Vec3 worldPos = RotateY(localPos, box.rotationYDeg) + box.translation;
    Vec3 worldNormal = Normalize(RotateY(localNormal, box.rotationYDeg));

    hit.t = tHit;
    hit.position = worldPos;
    hit.SetFaceNormal(ray, worldNormal);
    hit.material = box.material;
    return true;
}

RENDER_HD inline bool HitScene(const SceneView& scene, const Ray& ray, float tMin, float tMax, HitRecord& hit) {
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

RENDER_HD inline bool HitRectLight(const RectLight& light, const Ray& ray, float tMin, float tMax, LightHit& outHit) {
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

RENDER_HD inline bool Occluded(const SceneView& scene, const Ray& ray, float maxDistance) {
    HitRecord temp;
    return HitScene(scene, ray, kEpsilon, maxDistance - kEpsilon, temp);
}

} // namespace render
