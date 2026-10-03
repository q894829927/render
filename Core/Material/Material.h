#pragma once
#include "Core/Math/Math.h"

namespace render {

struct Material {
    Vec3 baseColor;
    float metallic = 0.0f;
    float roughness = 0.5f;
};

} // namespace render
