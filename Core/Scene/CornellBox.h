#pragma once

#include <vector>
#include "Core/Scene/Scene.h"

namespace render {

struct Camera {
    Vec3 position;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float viewportWidth;
    float viewportHeight;
};

struct SceneStorage {
    std::vector<Rect> rects;
    std::vector<OrientedBox> boxes;
    RectLight light;

    SceneView View() const {
        return { rects.data(), static_cast<int>(rects.size()), boxes.data(), static_cast<int>(boxes.size()), light };
    }
};

inline SceneStorage MakeCornellBox() {
    Material red   { Vec3(0.65f, 0.05f, 0.05f), 0.0f, 0.8f };
    Material green { Vec3(0.12f, 0.45f, 0.15f), 0.0f, 0.8f };
    Material white { Vec3(0.73f, 0.73f, 0.73f), 0.0f, 0.75f };

    SceneStorage scene;
    scene.rects = {
        { RectAxis::YZ, 0,555, 0,555, 0,   Vec3( 1, 0, 0), red   },
        { RectAxis::YZ, 0,555, 0,555, 555, Vec3(-1, 0, 0), green },
        { RectAxis::XZ, 0,555, 0,555, 0,   Vec3( 0, 1, 0), white },
        { RectAxis::XZ, 0,555, 0,555, 555, Vec3( 0,-1, 0), white },
        { RectAxis::XY, 0,555, 0,555, 555, Vec3( 0, 0,-1), white }
    };
    scene.boxes = {
        { Vec3(0), Vec3(165,165,165), -18.0f, Vec3(130,0,65), white },
        { Vec3(0), Vec3(165,330,165),  15.0f, Vec3(265,0,295), white }
    };
    scene.light = {
        Vec3(278.0f, 554.0f, 279.5f),
        Vec3(65.0f, 0.0f, 0.0f),
        Vec3(0.0f, 0.0f, 52.5f),
        Vec3(0.0f, -1.0f, 0.0f),
        Vec3(18.0f)
    };
    return scene;
}

inline Camera MakeCornellCamera(int width, int height) {
    Vec3 position(278.0f, 278.0f, -800.0f);
    Vec3 target(278.0f, 278.0f, 0.0f);
    Vec3 forward = Normalize(target - position);
    Vec3 worldUp(0.0f, 1.0f, 0.0f);
    Vec3 right = Normalize(Cross(worldUp, forward));
    Vec3 up = Normalize(Cross(forward, right));

    float aspect = static_cast<float>(width) / static_cast<float>(height);
    float fovY = 40.0f * kPi / 180.0f;
    float viewportHeight = 2.0f * tanf(fovY * 0.5f);
    return { position, forward, right, up, viewportHeight * aspect, viewportHeight };
}

} // namespace render
