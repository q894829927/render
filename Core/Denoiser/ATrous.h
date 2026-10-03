#pragma once

#include "Core/Denoiser/Denoiser.h"

namespace render {

struct FilteredSignal {
    Vec3 color;
    Vec3 variance;
};

RENDER_HD inline FilteredSignal ATrousLayerAt(
    const ResolvedPixel* pixels,
    const Vec3* inputColor,
    const Vec3* inputVariance,
    int pixelIndex,
    int layerSlot,
    int x,
    int y,
    int width,
    int height,
    int step,
    DenoiseSignal signal,
    const DenoiseSettings& settings)
{
    const ResolvedLayer& centerLayer = pixels[pixelIndex].layers[layerSlot];
    int centerSignalIndex = LayerIndex(pixelIndex, layerSlot);
    Vec3 center = inputColor[centerSignalIndex];
    Vec3 centerVar = inputVariance[centerSignalIndex];

    if (!centerLayer.valid) return {center, centerVar};

    const float kernel[5] = {
        1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f
    };

    Vec3 sum(0.0f);
    Vec3 varianceSum(0.0f);
    float weightSum = 0.0f;

    float phiColor = signal == DenoiseSignal::DiffuseIllumination
        ? settings.phiColorDiffuse
        : settings.phiColorSpecular;

    for (int ky = -2; ky <= 2; ++ky) {
        for (int kx = -2; kx <= 2; ++kx) {
            int sx = x + kx * step;
            int sy = y + ky * step;
            if (sx < 0 || sx >= width || sy < 0 || sy >= height) continue;

            int samplePixelIndex = sy * width + sx;
            int sampleSlot = FindBestDenoiseLayer(
                pixels[samplePixelIndex],
                centerLayer.guide,
                settings,
                signal);
            if (sampleSlot < 0) continue;

            const ResolvedLayer& sampleLayer =
                pixels[samplePixelIndex].layers[sampleSlot];
            int sampleSignalIndex = LayerIndex(samplePixelIndex, sampleSlot);

            Vec3 sampleColor = inputColor[sampleSignalIndex];
            Vec3 sampleVar = inputVariance[sampleSignalIndex];

            float spatial = kernel[kx + 2] * kernel[ky + 2];
            float guideWeight = SurfaceGuideWeight(
                centerLayer.guide, sampleLayer.guide, settings, signal);
            float colorWeight = VarianceAwareColorWeight(
                center, sampleColor, centerVar, sampleVar, phiColor);

            float weight = spatial * guideWeight * colorWeight;
            sum += sampleColor * weight;
            varianceSum += sampleVar * (weight * weight);
            weightSum += weight;
        }
    }

    if (weightSum <= 1e-8f) return {center, centerVar};
    return {
        sum / weightSum,
        varianceSum / (weightSum * weightSum)
    };
}

struct ComposedSignals {
    Vec3 raw;
    Vec3 diffuse;
    Vec3 specular;
    Vec3 finalColor;
};

RENDER_HD inline ComposedSignals ComposePixel(
    const ResolvedPixel& pixel,
    const Vec3* filteredDiffuse,
    const Vec3* filteredSpecular,
    int pixelIndex)
{
    ComposedSignals out{};
    out.raw = pixel.raw;
    out.finalColor = pixel.overflow;

    for (int slot = 0; slot < kPrimarySurfaceSlots; ++slot) {
        const ResolvedLayer& layer = pixel.layers[slot];
        if (!layer.valid) continue;

        int signalIndex = LayerIndex(pixelIndex, slot);
        Vec3 diffuse = filteredDiffuse[signalIndex] * layer.guide.albedo;
        Vec3 specular = filteredSpecular[signalIndex];
        float coverage = layer.guide.coverage;

        out.diffuse += diffuse * coverage;
        out.specular += specular * coverage;
        out.finalColor +=
            (diffuse + specular + layer.emission) * coverage;
    }

    return out;
}

} // namespace render
