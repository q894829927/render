#include <cstdint>
#include <iostream>

#include "Core/Sampling/SampleGenerator.h"

using namespace render;

namespace {

bool Check(bool condition, const char* message) {
    if (condition) return true;
    std::cerr << "FAIL: " << message << '\n';
    return false;
}

} // namespace

int main() {
    const std::uint32_t expectedDim0[8] = {
        0x00000000u, 0x80000000u, 0xc0000000u, 0x40000000u,
        0x60000000u, 0xe0000000u, 0xa0000000u, 0x20000000u
    };
    const std::uint32_t expectedDim1[8] = {
        0x00000000u, 0x80000000u, 0x40000000u, 0xc0000000u,
        0x60000000u, 0xe0000000u, 0x20000000u, 0xa0000000u
    };

    bool ok = true;
    for (std::uint32_t i = 0; i < 8u; ++i) {
        ok &= Check(SobolUInt(i, 0u) == expectedDim0[i],
                    "Sobol dimension 0 reference sequence mismatch");
        ok &= Check(SobolUInt(i, 1u) == expectedDim1[i],
                    "Sobol dimension 1 reference sequence mismatch");
    }

    SampleGenerator a = MakeSampleGenerator(77u, 0u, 123456u);
    SampleGenerator b = MakeSampleGenerator(77u, 0u, 123456u);
    ok &= Check(a.Sample1D(0u) == b.Sample1D(0u),
                "Sampler is not deterministic");

    SampleGenerator replicate0 = MakeSampleGenerator(77u, 0u, 123456u);
    SampleGenerator replicate1 = MakeSampleGenerator(77u, 1u, 123456u);
    ok &= Check(replicate0.scramble != replicate1.scramble,
                "Independent replicates share the same scramble");

    bool occupied[16] = {};
    for (std::uint32_t sample = 0; sample < 16u; ++sample) {
        const std::uint32_t globalSample =
            sample * kSampleReplicateCount;
        SampleGenerator g =
            MakeSampleGenerator(91u, globalSample, 123456u);
        Sample2DValue p = g.Sample2D(kCameraJitterXDimension);
        int bx = static_cast<int>(p.x * 4.0f);
        int by = static_cast<int>(p.y * 4.0f);
        if (bx < 0 || bx >= 4 || by < 0 || by >= 4) {
            ok &= Check(false, "Camera sample outside [0,1)");
            continue;
        }
        int bin = by * 4 + bx;
        ok &= Check(!occupied[bin],
                    "First 16 Owen-Sobol camera samples lost 4x4 stratification");
        occupied[bin] = true;
    }
    for (bool seen : occupied)
        ok &= Check(seen, "First 16 Owen-Sobol camera samples miss a 4x4 stratum");

    SampleGenerator hash =
        MakeSampleGenerator(
            91u, 0u, 123456u,
            SamplerType::PseudoRandomReference);
    ok &= Check(hash.Sample1D(0u) >= 0.0f && hash.Sample1D(0u) < 1.0f,
                "Hash reference sampler outside [0,1)");

    if (!ok) return 1;
    std::cout << "SampleGenerator tests passed.\n";
    return 0;
}
