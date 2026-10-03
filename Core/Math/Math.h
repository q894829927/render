#pragma once

#include <cmath>

#ifdef __CUDACC__
#define RENDER_HD __host__ __device__
#define RENDER_DEVICE __device__
#else
#define RENDER_HD
#define RENDER_DEVICE
#endif

namespace render {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kInf = 1.0e30f;
constexpr float kEpsilon = 1.0e-3f;
constexpr int kRussianRouletteStartBounce = 3;

struct Vec3 {
    float x, y, z;

    RENDER_HD Vec3(float v = 0.0f) : x(v), y(v), z(v) {}
    RENDER_HD Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    RENDER_HD Vec3 operator-() const { return {-x, -y, -z}; }
    RENDER_HD Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    RENDER_HD Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    RENDER_HD Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    RENDER_HD Vec3 operator/(float s) const { return *this * (1.0f / s); }
    RENDER_HD Vec3 operator*(const Vec3& v) const { return {x * v.x, y * v.y, z * v.z}; }

    RENDER_HD Vec3& operator+=(const Vec3& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
};

RENDER_HD inline Vec3 operator*(float s, const Vec3& v) { return v * s; }
RENDER_HD inline float Dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
RENDER_HD inline Vec3 Cross(const Vec3& a, const Vec3& b) {
    return {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
}
RENDER_HD inline float LengthSquared(const Vec3& v) { return Dot(v, v); }
RENDER_HD inline float Length(const Vec3& v) { return sqrtf(LengthSquared(v)); }
RENDER_HD inline Vec3 Normalize(const Vec3& v) {
    float len = Length(v);
    return len > 0.0f ? v / len : Vec3(0.0f);
}
RENDER_HD inline Vec3 Reflect(const Vec3& v, const Vec3& n) { return v - 2.0f * Dot(v, n) * n; }
RENDER_HD inline float Clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
RENDER_HD inline float Saturate(float x) { return Clamp(x, 0.0f, 1.0f); }
RENDER_HD inline Vec3 Mix(const Vec3& a, const Vec3& b, float t) { return a * (1.0f - t) + b * t; }
RENDER_HD inline float MaxComponent(const Vec3& v) { return fmaxf(v.x, fmaxf(v.y, v.z)); }
RENDER_HD inline float Luminance(const Vec3& c) { return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z; }

RENDER_HD inline bool IsFinite(float x) {
#ifdef __CUDA_ARCH__
    return isfinite(x);
#else
    return std::isfinite(x);
#endif
}

RENDER_HD inline bool IsFinite(const Vec3& v) {
    return IsFinite(v.x) && IsFinite(v.y) && IsFinite(v.z);
}

struct ONB {
    Vec3 t, b, n;

    RENDER_HD explicit ONB(const Vec3& normal) {
        n = Normalize(normal);
        Vec3 helper = fabsf(n.z) < 0.999f ? Vec3(0.0f, 0.0f, 1.0f) : Vec3(1.0f, 0.0f, 0.0f);
        t = Normalize(Cross(helper, n));
        b = Cross(n, t);
    }

    RENDER_HD Vec3 LocalToWorld(const Vec3& v) const {
        return t * v.x + b * v.y + n * v.z;
    }
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
    RENDER_HD Vec3 At(float t) const { return origin + direction * t; }
};

} // namespace render
