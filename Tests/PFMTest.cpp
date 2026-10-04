#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Core/Output/PFM.h"

using namespace render;

namespace {

bool Check(bool condition, const char* message) {
    if (condition) return true;
    std::cerr << "FAIL: " << message << '\n';
    return false;
}

float ReadLittleEndianFloat(std::ifstream& stream) {
    unsigned char bytes[4] = {};
    stream.read(
        reinterpret_cast<char*>(bytes),
        sizeof(bytes));

    std::uint32_t bits =
        static_cast<std::uint32_t>(bytes[0]) |
        (static_cast<std::uint32_t>(bytes[1]) << 8u) |
        (static_cast<std::uint32_t>(bytes[2]) << 16u) |
        (static_cast<std::uint32_t>(bytes[3]) << 24u);

    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool Near(float a, float b) {
    return std::fabs(a - b) < 1e-7f;
}

} // namespace

int main() {
    bool ok = true;

    const char* filename =
        "pfm_roundtrip_test.pfm";

    std::vector<Vec3> image = {
        Vec3(1.0f, 2.0f, 3.0f),
        Vec3(4.0f, 5.0f, 6.0f),
        Vec3(7.0f, 8.0f, 9.0f),
        Vec3(10.0f, 11.0f, 12.0f)
    };

    ok &= Check(
        SavePFM(
            filename,
            image,
            2,
            2),
        "SavePFM failed.");

    std::ifstream stream(
        filename,
        std::ios::binary);

    std::string magic;
    std::string dimensions;
    std::string scale;

    std::getline(stream, magic);
    std::getline(stream, dimensions);
    std::getline(stream, scale);

    ok &= Check(
        magic == "PF",
        "PFM magic is incorrect.");
    ok &= Check(
        dimensions == "2 2",
        "PFM dimensions are incorrect.");
    ok &= Check(
        scale == "-1.0",
        "PFM endian scale is incorrect.");

    for (const Vec3& expected : image) {
        float r = ReadLittleEndianFloat(stream);
        float g = ReadLittleEndianFloat(stream);
        float b = ReadLittleEndianFloat(stream);

        ok &= Check(
            Near(r, expected.x) &&
            Near(g, expected.y) &&
            Near(b, expected.z),
            "PFM linear float payload changed.");
    }

    char extra = 0;
    ok &= Check(
        !stream.read(&extra, 1),
        "PFM contains unexpected trailing bytes.");

    if (!ok)
        return 1;

    std::cout
        << "PFM output tests passed.\n";
    return 0;
}
