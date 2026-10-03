#pragma once

#include <cstdint>

#include "Core/Scene/Scene.h"

namespace render {

struct PrimarySurfaceSample {
    Vec3 normal;
    Vec3 albedo;
    float depth = kInf;
    float roughness = 0.0f;
    float metallic = 0.0f;
    std::uint32_t primitiveId = kInvalidPrimitiveId;
    int valid = 0;
};

struct PathSample {
    Vec3 diffuse;
    Vec3 specular;
    Vec3 emission;
    PrimarySurfaceSample primary;

    RENDER_HD Vec3 Total() const {
        return diffuse + specular + emission;
    }
};

} // namespace render
