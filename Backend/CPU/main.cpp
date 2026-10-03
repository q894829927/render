#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "Core/Denoiser/ATrous.h"
#include "Core/Integrator/Integrator.h"
#include "Core/Output/ImageIO.h"
#include "Core/Reconstruction/Reconstruction.h"
#include "Core/Scene/CornellBox.h"

using namespace render;

namespace {

template <typename Fn>
void ParallelForRows(int height, unsigned workers, Fn&& fn) {
    workers = std::max(1u, std::min(workers, static_cast<unsigned>(height)));
    std::atomic<int> next{0};
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (unsigned i = 0; i < workers; ++i) {
        threads.emplace_back([&] {
            for (;;) {
                int y = next.fetch_add(1, std::memory_order_relaxed);
                if (y >= height) break;
                fn(y);
            }
        });
    }
    for (auto& t : threads) t.join();
}

void ProgressivePass(
    std::vector<PixelAccumulator>& accumulation,
    int width, int height, int samplesThisPass, int sampleOffset,
    int maxDepth, const SceneView& scene, const Camera& camera,
    std::uint32_t baseSeed, unsigned workers)
{
    ParallelForRows(height, workers, [&](int y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            for (int s = 0; s < samplesThisPass; ++s) {
                std::uint32_t globalSampleIndex =
                    static_cast<std::uint32_t>(sampleOffset + s);
                SampleGenerator samples = MakeSampleGenerator(
                    static_cast<std::uint32_t>(idx), globalSampleIndex, baseSeed);
                Sample2DValue cameraJitter =
                    samples.Sample2D(kCameraJitterXDimension);

                float px =
                    (2.0f * ((x + cameraJitter.x) / static_cast<float>(width)) - 1.0f) *
                    camera.viewportWidth * 0.5f;
                float py =
                    (2.0f * ((y + cameraJitter.y) / static_cast<float>(height)) - 1.0f) *
                    camera.viewportHeight * 0.5f;

                Ray ray{
                    camera.position,
                    Normalize(camera.forward + camera.right*px + camera.up*py)
                };
                AccumulatePathSample(
                    accumulation[idx], TracePath(ray, scene, maxDepth, samples));
            }
        }
    });
}

void InitializeSignals(
    const std::vector<ResolvedPixel>& resolved,
    std::vector<Vec3>& diffuse, std::vector<Vec3>& diffuseVariance,
    std::vector<Vec3>& specular, std::vector<Vec3>& specularVariance,
    int width, int height, unsigned workers)
{
    ParallelForRows(height, workers, [&](int y) {
        for (int x = 0; x < width; ++x) {
            int pixel = y * width + x;
            for (int slot = 0; slot < kPrimarySurfaceSlots; ++slot) {
                int i = LayerIndex(pixel, slot);
                const ResolvedLayer& layer = resolved[pixel].layers[slot];
                if (!layer.valid) {
                    diffuse[i] = diffuseVariance[i] =
                        specular[i] = specularVariance[i] = Vec3(0.0f);
                    continue;
                }
                diffuse[i] = layer.diffuseIllumination;
                diffuseVariance[i] = layer.diffuseVariance;
                specular[i] = layer.specular;
                specularVariance[i] = layer.specularVariance;
            }
        }
    });
}

void DenoisePass(
    const std::vector<ResolvedPixel>& resolved,
    const std::vector<Vec3>& inputColor, std::vector<Vec3>& outputColor,
    const std::vector<Vec3>& inputVariance, std::vector<Vec3>& outputVariance,
    DenoiseSignal signal, const DenoiseSettings& settings,
    int step, int width, int height, unsigned workers)
{
    ParallelForRows(height, workers, [&](int y) {
        for (int x = 0; x < width; ++x) {
            int pixel = y * width + x;
            for (int slot = 0; slot < kPrimarySurfaceSlots; ++slot) {
                int i = LayerIndex(pixel, slot);
                FilteredSignal filtered = ATrousLayerAt(
                    resolved.data(), inputColor.data(), inputVariance.data(),
                    pixel, slot, x, y, width, height, step, signal, settings);
                outputColor[i] = filtered.color;
                outputVariance[i] = filtered.variance;
            }
        }
    });
}

std::uint64_t HashLinearBuffer(const std::vector<Vec3>& values) {
    constexpr std::uint64_t kOffset = 1469598103934665603ULL;
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    std::uint64_t hash = kOffset;
    for (const Vec3& value : values) {
        const float channels[3] = {value.x, value.y, value.z};
        for (float channel : channels) {
            std::uint32_t bits = 0u;
            std::memcpy(&bits, &channel, sizeof(bits));
            hash ^= bits;
            hash *= kPrime;
        }
    }
    return hash;
}

} // namespace

int main(int argc, char** argv) {
    const int width = 600;
    const int height = 600;
    int targetSpp = argc > 1 ? std::max(1, std::atoi(argv[1])) : 256;
    int samplesPerPass = argc > 2 ? std::max(1, std::atoi(argv[2])) : 8;
    int maxDepth = argc > 3 ? std::max(1, std::atoi(argv[3])) : 16;
    unsigned workers = std::max(1u, std::thread::hardware_concurrency());

    SceneStorage storage = MakeCornellBox();
    SceneView scene = storage.View();
    Camera camera = MakeCornellCamera(width, height);
    DenoiseSettings denoise{};

    std::cout << "Backend: CPU\nThreads: " << workers
              << "\nResolution: " << width << 'x' << height
              << "\nTarget Samples: " << targetSpp
              << "\nSamples / Pass: " << samplesPerPass
              << "\nMax Depth: " << maxDepth
              << "\nSampler: deterministic dimensioned hash"
              << "\nSample Replicates: " << kSampleReplicateCount
              << "\nArchitecture: layered primary-surface reconstruction"
              << "\nDenoiser: diffuse/specular variance-guided A-Trous x"
              << denoise.iterations << "\n\n";

    std::size_t n = static_cast<std::size_t>(width) * height;
    std::vector<PixelAccumulator> accumulation(n);

    auto renderStart = std::chrono::steady_clock::now();
    int accumulated = 0;
    while (accumulated < targetSpp) {
        int pass = std::min(samplesPerPass, targetSpp - accumulated);
        ProgressivePass(
            accumulation, width, height, pass, accumulated, maxDepth,
            scene, camera, 123456u, workers);
        accumulated += pass;
        std::cout << "\rAccumulating: " << accumulated << '/' << targetSpp
                  << " SPP" << std::flush;
    }
    auto renderStop = std::chrono::steady_clock::now();

    std::vector<ResolvedPixel> resolved(n);
    ParallelForRows(height, workers, [&](int y) {
        for (int x = 0; x < width; ++x) {
            int i = y * width + x;
            resolved[i] = ResolvePixel(accumulation[i]);
        }
    });

    std::size_t signalCount = n * kPrimarySurfaceSlots;
    std::vector<Vec3> diffuseA(signalCount), diffuseB(signalCount);
    std::vector<Vec3> diffuseVarA(signalCount), diffuseVarB(signalCount);
    std::vector<Vec3> specularA(signalCount), specularB(signalCount);
    std::vector<Vec3> specularVarA(signalCount), specularVarB(signalCount);

    InitializeSignals(
        resolved, diffuseA, diffuseVarA, specularA, specularVarA,
        width, height, workers);

    auto denoiseStart = std::chrono::steady_clock::now();

    auto* diffIn = &diffuseA;
    auto* diffOut = &diffuseB;
    auto* diffVarIn = &diffuseVarA;
    auto* diffVarOut = &diffuseVarB;
    auto* specIn = &specularA;
    auto* specOut = &specularB;
    auto* specVarIn = &specularVarA;
    auto* specVarOut = &specularVarB;

    for (int iteration = 0; iteration < denoise.iterations; ++iteration) {
        int step = 1 << iteration;
        DenoisePass(
            resolved, *diffIn, *diffOut, *diffVarIn, *diffVarOut,
            DenoiseSignal::DiffuseIllumination, denoise, step,
            width, height, workers);
        DenoisePass(
            resolved, *specIn, *specOut, *specVarIn, *specVarOut,
            DenoiseSignal::Specular, denoise, step,
            width, height, workers);

        std::swap(diffIn, diffOut);
        std::swap(diffVarIn, diffVarOut);
        std::swap(specIn, specOut);
        std::swap(specVarIn, specVarOut);
    }

    auto denoiseStop = std::chrono::steady_clock::now();

    std::vector<Vec3> raw(n), diffuse(n), specular(n), finalColor(n);
    ParallelForRows(height, workers, [&](int y) {
        for (int x = 0; x < width; ++x) {
            int i = y * width + x;
            ComposedSignals c = ComposePixel(
                resolved[i], diffIn->data(), specIn->data(), i);
            raw[i] = c.raw;
            diffuse[i] = c.diffuse;
            specular[i] = c.specular;
            finalColor[i] = c.finalColor;
        }
    });

    SavePPM("cornell_cpu_raw.ppm", raw, width, height);
    SavePPM("cornell_cpu_diffuse.ppm", diffuse, width, height);
    SavePPM("cornell_cpu_specular.ppm", specular, width, height);
    SavePPM("cornell_cpu_final.ppm", finalColor, width, height);

    double renderMs = std::chrono::duration<double, std::milli>(
        renderStop - renderStart).count();
    double denoiseMs = std::chrono::duration<double, std::milli>(
        denoiseStop - denoiseStart).count();

    std::cout << "\nRender time: " << renderMs
              << " ms\nDenoise time: " << denoiseMs
              << " ms\nLinear hashes:"
              << " raw=" << HashLinearBuffer(raw)
              << " diffuse=" << HashLinearBuffer(diffuse)
              << " specular=" << HashLinearBuffer(specular)
              << " final=" << HashLinearBuffer(finalColor)
              << "\nSaved: raw / diffuse / specular / final\n";
    return 0;
}
