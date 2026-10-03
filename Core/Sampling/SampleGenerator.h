#pragma once
#include <cstdint>

#include "Core/Sampling/Sobol.h"

namespace render {

constexpr std::uint32_t kSampleReplicateCount = 4u;

enum class SamplerType : std::uint32_t {
    OwenSobol = 0u,
    PseudoRandomReference = 1u
};

struct Sample2DValue {
    float x = 0.0f;
    float y = 0.0f;
};

RENDER_HD inline std::uint32_t Hash32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

RENDER_HD inline std::uint32_t HashCombine32(
    std::uint32_t seed,
    std::uint32_t value)
{
    return Hash32(seed ^ (Hash32(value + 0x9e3779b9u) + 0x85ebca6bu));
}

struct SampleGenerator {
    std::uint32_t pixelId = 0u;
    std::uint32_t replicateId = 0u;
    std::uint32_t sampleIndexWithinReplicate = 0u;
    std::uint32_t scramble = 0u;
    SamplerType samplerType = SamplerType::OwenSobol;

    RENDER_HD float Sample1D(std::uint32_t dimension) const {
        if (samplerType == SamplerType::OwenSobol &&
            dimension < kSobolMaxDimensions)
        {
            const std::uint32_t sobol =
                SobolUInt(sampleIndexWithinReplicate, dimension);
            const std::uint32_t dimensionSeed =
                HashCombine32(scramble, dimension);
            return SobolToUnitFloat(
                FastOwenScramble(sobol, dimensionSeed));
        }

        std::uint32_t key = HashCombine32(
            sampleIndexWithinReplicate,
            dimension * 0x9e3779b9u + 0x68bc21ebu);
        key = HashCombine32(key, scramble);
        const std::uint32_t mantissa24 = Hash32(key) >> 8u;
        return static_cast<float>(mantissa24) * (1.0f / 16777216.0f);
    }

    RENDER_HD Sample2DValue Sample2D(std::uint32_t firstDimension) const {
        return {
            Sample1D(firstDimension),
            Sample1D(firstDimension + 1u)
        };
    }
};

RENDER_HD inline SampleGenerator MakeSampleGenerator(
    std::uint32_t pixelId,
    std::uint32_t globalSampleIndex,
    std::uint32_t baseSeed,
    SamplerType samplerType = SamplerType::OwenSobol)
{
    SampleGenerator generator;
    generator.pixelId = pixelId;
    generator.replicateId =
        globalSampleIndex % kSampleReplicateCount;
    generator.sampleIndexWithinReplicate =
        globalSampleIndex / kSampleReplicateCount;
    generator.samplerType = samplerType;

    std::uint32_t scramble = HashCombine32(baseSeed, pixelId);
    scramble = HashCombine32(scramble, generator.replicateId);
    generator.scramble = scramble;
    return generator;
}

} // namespace render
