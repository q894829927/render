#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

#include "Core/Math/Math.h"

namespace render {

inline void WriteFloat32LittleEndian(
    std::ofstream& stream,
    float value)
{
    static_assert(
        sizeof(float) == sizeof(std::uint32_t),
        "PFM output requires 32-bit float.");

    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));

    const unsigned char bytes[4] = {
        static_cast<unsigned char>( bits        & 0xffu),
        static_cast<unsigned char>((bits >> 8u) & 0xffu),
        static_cast<unsigned char>((bits >> 16u)& 0xffu),
        static_cast<unsigned char>((bits >> 24u)& 0xffu)
    };

    stream.write(
        reinterpret_cast<const char*>(bytes),
        sizeof(bytes));
}

inline bool SavePFM(
    const char* filename,
    const std::vector<Vec3>& framebuffer,
    int width,
    int height)
{
    if (width <= 0 ||
        height <= 0 ||
        framebuffer.size() !=
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height))
        return false;

    std::ofstream stream(
        filename,
        std::ios::binary);

    if (!stream)
        return false;

    // Negative scale means little-endian according to the PFM convention.
    stream
        << "PF\n"
        << width << ' ' << height << "\n"
        << "-1.0\n";

    // Match the display output orientation: framebuffer y=0 is the bottom
    // row, and PFM rows are written bottom-to-top.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const Vec3& value =
                framebuffer[
                    static_cast<std::size_t>(y) *
                    width + x];

            WriteFloat32LittleEndian(
                stream,
                value.x);
            WriteFloat32LittleEndian(
                stream,
                value.y);
            WriteFloat32LittleEndian(
                stream,
                value.z);
        }
    }

    return static_cast<bool>(stream);
}

} // namespace render
