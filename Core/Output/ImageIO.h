#pragma once

#include <cmath>
#include <fstream>
#include <vector>

#include "Core/Math/Math.h"

namespace render {

inline float ACESFilm(float x) {
    const float a=2.51f,b=0.03f,c=2.43f,d=0.59f,e=0.14f;
    return Saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

inline Vec3 DisplayTransform(Vec3 c) {
    c = {ACESFilm(c.x), ACESFilm(c.y), ACESFilm(c.z)};
    return {
        std::pow(Saturate(c.x), 1.0f/2.2f),
        std::pow(Saturate(c.y), 1.0f/2.2f),
        std::pow(Saturate(c.z), 1.0f/2.2f)
    };
}

inline void SavePPM(
    const char* filename,
    const std::vector<Vec3>& fb,
    int width,
    int height)
{
    std::ofstream f(filename);
    f << "P3\n" << width << ' ' << height << "\n255\n";
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            Vec3 c = DisplayTransform(
                fb[static_cast<std::size_t>(y) * width + x]);
            f << static_cast<int>(255.999f*Saturate(c.x)) << ' '
              << static_cast<int>(255.999f*Saturate(c.y)) << ' '
              << static_cast<int>(255.999f*Saturate(c.z)) << '\n';
        }
    }
}

} // namespace render
