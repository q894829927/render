#pragma once
#include <cstdint>

#include "Core/Sampling/SampleDimensions.h"
#include "Core/Sampling/SobolParameters.h"

namespace render {

RENDER_HD inline std::uint32_t ReverseBits32(std::uint32_t x) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    return (x >> 16) | (x << 16);
}

RENDER_HD inline std::uint16_t SobolPolynomial(std::uint32_t dimension) {
#ifdef __CUDA_ARCH__
    return gSobolPolynomialsDevice[dimension];
#else
    return kSobolPolynomialsHost[dimension];
#endif
}

RENDER_HD inline std::uint16_t SobolVInitOffset(std::uint32_t dimension) {
#ifdef __CUDA_ARCH__
    return gSobolVInitOffsetsDevice[dimension];
#else
    return kSobolVInitOffsetsHost[dimension];
#endif
}

RENDER_HD inline std::uint16_t SobolVInitValue(std::uint32_t index) {
#ifdef __CUDA_ARCH__
    return gSobolVInitDevice[index];
#else
    return kSobolVInitHost[index];
#endif
}

RENDER_HD inline int SobolPolynomialDegree(std::uint32_t polynomial) {
    int degree = -1;
    while (polynomial != 0u) {
        polynomial >>= 1u;
        ++degree;
    }
    return degree;
}

RENDER_HD inline std::uint32_t SobolUInt(
    std::uint32_t sampleIndex,
    std::uint32_t dimension)
{
    if (dimension >= kSobolMaxDimensions) return 0u;

    std::uint32_t gray = sampleIndex ^ (sampleIndex >> 1u);
    if (gray == 0u) return 0u;

    if (dimension == 0u) return ReverseBits32(gray);

    const std::uint32_t polynomial = SobolPolynomial(dimension);
    const int degree = SobolPolynomialDegree(polynomial);
    const std::uint32_t a = degree > 1
        ? ((polynomial >> 1u) & ((1u << (degree - 1)) - 1u))
        : 0u;
    const std::uint32_t initOffset = SobolVInitOffset(dimension);

    std::uint32_t history[kSobolMaxDegree] = {};
    std::uint32_t result = 0u;

    int highestBit = 0;
    for (std::uint32_t t = gray; t != 0u; t >>= 1u) ++highestBit;

    for (int j = 1; j <= highestBit; ++j) {
        std::uint32_t direction = 0u;
        if (j <= degree) {
            direction = static_cast<std::uint32_t>(
                SobolVInitValue(initOffset + static_cast<std::uint32_t>(j - 1)))
                << (32 - j);
        } else {
            const int oldIndex = (j - degree - 1) % degree;
            direction = history[oldIndex] ^ (history[oldIndex] >> degree);
            for (int k = 1; k < degree; ++k) {
                if ((a >> (degree - 1 - k)) & 1u) {
                    direction ^= history[(j - k - 1) % degree];
                }
            }
        }

        history[(j - 1) % degree] = direction;
        if (gray & (1u << (j - 1))) result ^= direction;
    }

    return result;
}

// Practical hash-based Owen-style scrambling used for rendering:
// it preserves the important base-2 stratification of Sobol while avoiding
// a per-bit nested-permutation traversal for every path dimension.
RENDER_HD inline std::uint32_t FastOwenScramble(
    std::uint32_t value,
    std::uint32_t seed)
{
    value = ReverseBits32(value);
    value ^= value * 0x3d20adeau;
    value += seed;
    value *= (seed >> 16u) | 1u;
    value ^= value * 0x05526c56u;
    value ^= value * 0x53a22864u;
    return ReverseBits32(value);
}

RENDER_HD inline float SobolToUnitFloat(std::uint32_t value) {
    const std::uint32_t mantissa24 = value >> 8u;
    return static_cast<float>(mantissa24) * (1.0f / 16777216.0f);
}

} // namespace render
