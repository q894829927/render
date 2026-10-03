#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <vector>

#include "Core/Denoiser/ATrous.h"
#include "Core/Integrator/Integrator.h"
#include "Core/Output/ImageIO.h"
#include "Core/Reconstruction/Film.h"
#include "Core/Reconstruction/Reconstruction.h"
#include "Core/Scene/CornellBox.h"

using namespace render;

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){ \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) \
                  << " (" << __FILE__ << ':' << __LINE__ << ")\n"; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

namespace {

__global__ void ProgressiveRenderKernel(
    PixelAccumulator* accumulation,
    int width, int height, int samplesThisPass, int sampleOffset,
    int maxDepth, SceneView scene, Camera camera, std::uint32_t baseSeed,
    SamplerType samplerType, ReconstructionFilterType filterType)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y*width + x;

    for (int s = 0; s < samplesThisPass; ++s) {
        std::uint32_t globalSampleIndex =
            static_cast<std::uint32_t>(sampleOffset + s);
        SampleGenerator samples = MakeSampleGenerator(
            static_cast<std::uint32_t>(idx), globalSampleIndex, baseSeed, samplerType);
        FilmSample filmSample =
            GenerateFilmSample(
                x, y, samples, filterType);

        float px =
            (2.0f*(filmSample.rasterX/static_cast<float>(width))-1.0f) *
            camera.viewportWidth * 0.5f;
        float py =
            (2.0f*(filmSample.rasterY/static_cast<float>(height))-1.0f) *
            camera.viewportHeight * 0.5f;
        Ray ray{
            camera.position,
            Normalize(camera.forward + camera.right*px + camera.up*py)
        };
        AccumulatePathSample(
            accumulation[idx], TracePath(ray, scene, maxDepth, samples));
    }
}

__global__ void ResolveKernel(
    const PixelAccumulator* accumulation, ResolvedPixel* resolved, int pixelCount)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= pixelCount) return;
    resolved[i] = ResolvePixel(accumulation[i]);
}

__global__ void InitializeSignalsKernel(
    const ResolvedPixel* resolved,
    Vec3* diffuse, Vec3* diffuseVariance,
    Vec3* specular, Vec3* specularVariance,
    int pixelCount)
{
    int pixel = blockIdx.x*blockDim.x + threadIdx.x;
    if (pixel >= pixelCount) return;

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

__global__ void ATrousKernel(
    const ResolvedPixel* resolved,
    const Vec3* inputColor, Vec3* outputColor,
    const Vec3* inputVariance, Vec3* outputVariance,
    DenoiseSignal signal, DenoiseSettings settings,
    int step, int width, int height)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixel = y*width + x;
    for (int slot = 0; slot < kPrimarySurfaceSlots; ++slot) {
        int i = LayerIndex(pixel, slot);
        FilteredSignal filtered = ATrousLayerAt(
            resolved, inputColor, inputVariance,
            pixel, slot, x, y, width, height, step, signal, settings);
        outputColor[i] = filtered.color;
        outputVariance[i] = filtered.variance;
    }
}

__global__ void ComposeKernel(
    const ResolvedPixel* resolved,
    const Vec3* diffuse, const Vec3* specular,
    Vec3* rawOut, Vec3* diffuseOut, Vec3* specularOut, Vec3* finalOut,
    int pixelCount)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= pixelCount) return;
    ComposedSignals c = ComposePixel(resolved[i], diffuse, specular, i);
    rawOut[i] = c.raw;
    diffuseOut[i] = c.diffuse;
    specularOut[i] = c.specular;
    finalOut[i] = c.finalColor;
}

template<typename T>
T* Alloc(std::size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), sizeof(T)*count));
    return p;
}

void Zero(void* p, std::size_t bytes) {
    CUDA_CHECK(cudaMemset(p, 0, bytes));
}

} // namespace

int main(int argc, char** argv) {
    const int width = 600;
    const int height = 600;
    int targetSpp = argc > 1 ? std::max(1, std::atoi(argv[1])) : 256;
    int samplesPerPass = argc > 2 ? std::max(1, std::atoi(argv[2])) : 8;
    int maxDepth = argc > 3 ? std::max(1, std::atoi(argv[3])) : 16;
    SamplerType samplerType = SamplerType::OwenSobol;
    if (argc > 4) {
        if (std::strcmp(argv[4], "hash") == 0 ||
            std::strcmp(argv[4], "pseudo") == 0)
            samplerType = SamplerType::PseudoRandomReference;
        else if (std::strcmp(argv[4], "owen") != 0) {
            std::cerr << "Unknown sampler '" << argv[4]
                      << "'. Use owen or hash.\n";
            return 2;
        }
    }
    ReconstructionFilterType filterType =
        ReconstructionFilterType::Tent;
    if (argc > 5) {
        if (std::strcmp(argv[5], "box") == 0)
            filterType = ReconstructionFilterType::Box;
        else if (std::strcmp(argv[5], "tent") != 0) {
            std::cerr << "Unknown reconstruction filter '" << argv[5]
                      << "'. Use tent or box.\n";
            return 2;
        }
    }

    if (samplerType == SamplerType::OwenSobol &&
        RequiredSampleDimensionCount(maxDepth) > kSobolMaxDimensions)
    {
        std::cerr << "Owen-Sobol supports maxDepth <= "
                  << ((kSobolMaxDimensions - kPathDimensionBase) /
                      kBounceDimensionStride)
                  << " with the current direction table.\n";
        return 2;
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemcpyToSymbol(
        render::gSobolPolynomialsDevice,
        render::kSobolPolynomialsHost,
        sizeof(render::kSobolPolynomialsHost)));
    CUDA_CHECK(cudaMemcpyToSymbol(
        render::gSobolVInitOffsetsDevice,
        render::kSobolVInitOffsetsHost,
        sizeof(render::kSobolVInitOffsetsHost)));
    CUDA_CHECK(cudaMemcpyToSymbol(
        render::gSobolVInitDevice,
        render::kSobolVInitHost,
        sizeof(render::kSobolVInitHost)));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    SceneStorage hostStorage = MakeCornellBox();
    Camera camera = MakeCornellCamera(width, height);
    DenoiseSettings denoise{};

    Rect* dRects = Alloc<Rect>(hostStorage.rects.size());
    OrientedBox* dBoxes = Alloc<OrientedBox>(hostStorage.boxes.size());
    CUDA_CHECK(cudaMemcpy(
        dRects, hostStorage.rects.data(),
        sizeof(Rect)*hostStorage.rects.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        dBoxes, hostStorage.boxes.data(),
        sizeof(OrientedBox)*hostStorage.boxes.size(), cudaMemcpyHostToDevice));

    SceneView scene{
        dRects, static_cast<int>(hostStorage.rects.size()),
        dBoxes, static_cast<int>(hostStorage.boxes.size()),
        hostStorage.light
    };

    std::size_t n = static_cast<std::size_t>(width)*height;
    std::size_t signalCount = n * kPrimarySurfaceSlots;

    PixelAccumulator* dAccumulation = Alloc<PixelAccumulator>(n);
    Zero(dAccumulation, sizeof(PixelAccumulator)*n);
    ResolvedPixel* dResolved = Alloc<ResolvedPixel>(n);

    Vec3* dDiffuseA = Alloc<Vec3>(signalCount);
    Vec3* dDiffuseB = Alloc<Vec3>(signalCount);
    Vec3* dDiffuseVarA = Alloc<Vec3>(signalCount);
    Vec3* dDiffuseVarB = Alloc<Vec3>(signalCount);
    Vec3* dSpecularA = Alloc<Vec3>(signalCount);
    Vec3* dSpecularB = Alloc<Vec3>(signalCount);
    Vec3* dSpecularVarA = Alloc<Vec3>(signalCount);
    Vec3* dSpecularVarB = Alloc<Vec3>(signalCount);

    Vec3* dRawOut = Alloc<Vec3>(n);
    Vec3* dDiffuseOut = Alloc<Vec3>(n);
    Vec3* dSpecularOut = Alloc<Vec3>(n);
    Vec3* dFinalOut = Alloc<Vec3>(n);

    std::cout << "GPU: " << prop.name
              << "\nResolution: " << width << 'x' << height
              << "\nTarget Samples: " << targetSpp
              << "\nSamples / Pass: " << samplesPerPass
              << "\nMax Depth: " << maxDepth
              << "\nSampler: "
              << (samplerType == SamplerType::OwenSobol
                    ? "Owen-scrambled Sobol"
                    : "deterministic dimensioned hash")
              << "\nSample Replicates: " << kSampleReplicateCount
              << "\nFilm Filter: "
              << (filterType == ReconstructionFilterType::Tent
                    ? "Tent"
                    : "Box")
              << "\nArchitecture: visibility-aware image reconstruction"
              << "\nDenoiser: diffuse/specular variance-guided A-Trous x"
              << denoise.iterations << "\n\n";

    dim3 block(16,16);
    dim3 grid((width+15)/16, (height+15)/16);
    int linearBlock = 256;
    int linearGrid = (static_cast<int>(n)+linearBlock-1)/linearBlock;

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int accumulated = 0;
    while (accumulated < targetSpp) {
        int pass = std::min(samplesPerPass, targetSpp-accumulated);
        ProgressiveRenderKernel<<<grid,block>>>(
            dAccumulation, width, height, pass, accumulated, maxDepth,
            scene, camera, 123456u, samplerType, filterType);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        accumulated += pass;
        std::cout << "\rAccumulating: " << accumulated << '/'
                  << targetSpp << " SPP" << std::flush;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float renderMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&renderMs, start, stop));

    ResolveKernel<<<linearGrid,linearBlock>>>(
        dAccumulation, dResolved, static_cast<int>(n));
    CUDA_CHECK(cudaGetLastError());

    InitializeSignalsKernel<<<linearGrid,linearBlock>>>(
        dResolved, dDiffuseA, dDiffuseVarA, dSpecularA, dSpecularVarA,
        static_cast<int>(n));
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t ds{}, de{};
    CUDA_CHECK(cudaEventCreate(&ds));
    CUDA_CHECK(cudaEventCreate(&de));
    CUDA_CHECK(cudaEventRecord(ds));

    Vec3* diffIn = dDiffuseA;
    Vec3* diffOut = dDiffuseB;
    Vec3* diffVarIn = dDiffuseVarA;
    Vec3* diffVarOut = dDiffuseVarB;
    Vec3* specIn = dSpecularA;
    Vec3* specOut = dSpecularB;
    Vec3* specVarIn = dSpecularVarA;
    Vec3* specVarOut = dSpecularVarB;

    for (int iteration = 0; iteration < denoise.iterations; ++iteration) {
        int step = 1 << iteration;
        ATrousKernel<<<grid,block>>>(
            dResolved, diffIn, diffOut, diffVarIn, diffVarOut,
            DenoiseSignal::DiffuseIllumination, denoise, step, width, height);
        CUDA_CHECK(cudaGetLastError());
        ATrousKernel<<<grid,block>>>(
            dResolved, specIn, specOut, specVarIn, specVarOut,
            DenoiseSignal::Specular, denoise, step, width, height);
        CUDA_CHECK(cudaGetLastError());

        std::swap(diffIn, diffOut);
        std::swap(diffVarIn, diffVarOut);
        std::swap(specIn, specOut);
        std::swap(specVarIn, specVarOut);
    }

    CUDA_CHECK(cudaEventRecord(de));
    CUDA_CHECK(cudaEventSynchronize(de));
    float denoiseMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&denoiseMs, ds, de));

    ComposeKernel<<<linearGrid,linearBlock>>>(
        dResolved, diffIn, specIn,
        dRawOut, dDiffuseOut, dSpecularOut, dFinalOut,
        static_cast<int>(n));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Vec3> raw(n), diffuse(n), specular(n), finalColor(n);
    CUDA_CHECK(cudaMemcpy(
        raw.data(), dRawOut, sizeof(Vec3)*n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        diffuse.data(), dDiffuseOut, sizeof(Vec3)*n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        specular.data(), dSpecularOut, sizeof(Vec3)*n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        finalColor.data(), dFinalOut, sizeof(Vec3)*n, cudaMemcpyDeviceToHost));

    SavePPM("cornell_cuda_raw.ppm", raw, width, height);
    SavePPM("cornell_cuda_diffuse.ppm", diffuse, width, height);
    SavePPM("cornell_cuda_specular.ppm", specular, width, height);
    SavePPM("cornell_cuda_final.ppm", finalColor, width, height);

    std::cout << "\nRender time: " << renderMs
              << " ms\nDenoise time: " << denoiseMs
              << " ms\nSaved: raw / diffuse / specular / final\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(ds);
    cudaEventDestroy(de);

    cudaFree(dFinalOut);
    cudaFree(dSpecularOut);
    cudaFree(dDiffuseOut);
    cudaFree(dRawOut);
    cudaFree(dSpecularVarB);
    cudaFree(dSpecularVarA);
    cudaFree(dSpecularB);
    cudaFree(dSpecularA);
    cudaFree(dDiffuseVarB);
    cudaFree(dDiffuseVarA);
    cudaFree(dDiffuseB);
    cudaFree(dDiffuseA);
    cudaFree(dResolved);
    cudaFree(dAccumulation);
    cudaFree(dBoxes);
    cudaFree(dRects);
    return 0;
}
