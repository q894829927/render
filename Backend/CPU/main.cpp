#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "Core/Denoiser/Denoiser.h"
#include "Core/Integrator/Integrator.h"
#include "Core/Scene/CornellBox.h"

using namespace render;

namespace {

struct CpuRng {
    std::uint64_t state = 1;

    static std::uint64_t SplitMix64(std::uint64_t& x) {
        std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    void Init(std::uint64_t seed, std::uint64_t sequence) {
        std::uint64_t x = seed ^ (sequence + 0x9E3779B97F4A7C15ULL);
        state = SplitMix64(x);
        if (state == 0) state = 0x853C49E6748FEA9BULL;
    }

    std::uint64_t Next64() {
        std::uint64_t x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        return x * 0x2545F4914F6CDD1DULL;
    }

    float NextFloat() {
        std::uint32_t bits = static_cast<std::uint32_t>(Next64() >> 40);
        return static_cast<float>(bits) * (1.0f / 16777216.0f);
    }
};

struct AccumulationBuffer {
    std::vector<Vec3> radianceSum;
    std::vector<float> luminanceSum;
    std::vector<float> luminanceSqSum;
    std::vector<Vec3> normalSum;
    std::vector<Vec3> albedoSum;
    std::vector<Vec3> materialSum;
    std::vector<float> depthSum;
    std::vector<float> depthSqSum;
    std::vector<unsigned int> guideHitCount;

    explicit AccumulationBuffer(std::size_t n)
        : radianceSum(n, Vec3(0)), luminanceSum(n, 0), luminanceSqSum(n, 0),
          normalSum(n, Vec3(0)), albedoSum(n, Vec3(0)), materialSum(n, Vec3(0)),
          depthSum(n, 0), depthSqSum(n, 0), guideHitCount(n, 0) {}
};

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
    AccumulationBuffer& acc,
    int width,
    int height,
    int samplesThisPass,
    int sampleOffset,
    int maxDepth,
    const SceneView& scene,
    const Camera& camera,
    std::uint64_t baseSeed,
    unsigned workers)
{
    const std::uint64_t passSeed = baseSeed +
        static_cast<std::uint64_t>(sampleOffset) * 0x9E3779B97F4A7C15ULL;

    ParallelForRows(height, workers, [&](int y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            CpuRng rng;
            rng.Init(passSeed, static_cast<std::uint64_t>(idx));

            Vec3 radiancePass(0), normalPass(0), albedoPass(0), materialPass(0);
            float lumPass = 0, lumSqPass = 0, depthPass = 0, depthSqPass = 0;
            unsigned guideHits = 0;

            for (int s = 0; s < samplesThisPass; ++s) {
                float px = (2.0f * ((x + rng.NextFloat()) / static_cast<float>(width)) - 1.0f)
                         * camera.viewportWidth * 0.5f;
                float py = (2.0f * ((y + rng.NextFloat()) / static_cast<float>(height)) - 1.0f)
                         * camera.viewportHeight * 0.5f;

                Ray ray{camera.position, Normalize(camera.forward + camera.right*px + camera.up*py)};
                PrimaryGuide guide{};
                Vec3 sample = TracePath(ray, scene, maxDepth, rng, &guide);
                radiancePass += sample;
                float lum = Luminance(sample);
                lumPass += lum;
                lumSqPass += lum * lum;

                if (guide.valid) {
                    normalPass += guide.normal;
                    albedoPass += guide.albedo;
                    materialPass += guide.material;
                    depthPass += guide.depth;
                    depthSqPass += guide.depth * guide.depth;
                    ++guideHits;
                }
            }

            acc.radianceSum[idx] += radiancePass;
            acc.luminanceSum[idx] += lumPass;
            acc.luminanceSqSum[idx] += lumSqPass;
            acc.normalSum[idx] += normalPass;
            acc.albedoSum[idx] += albedoPass;
            acc.materialSum[idx] += materialPass;
            acc.depthSum[idx] += depthPass;
            acc.depthSqSum[idx] += depthSqPass;
            acc.guideHitCount[idx] += guideHits;
        }
    });
}

void Resolve(
    const AccumulationBuffer& acc,
    int samples,
    std::vector<Vec3>& raw,
    std::vector<float>& variance,
    std::vector<GuideValue>& guides,
    unsigned workers,
    int width,
    int height)
{
    float sampleCount = static_cast<float>(std::max(samples, 1));
    ParallelForRows(height, workers, [&](int y) {
        for (int x = 0; x < width; ++x) {
            int i = y * width + x;
            raw[i] = acc.radianceSum[i] / sampleCount;

            float meanLum = acc.luminanceSum[i] / sampleCount;
            float second = acc.luminanceSqSum[i] / sampleCount;
            float sampleVariance = std::max(second - meanLum*meanLum, 0.0f);
            variance[i] = sampleVariance / sampleCount;

            unsigned hits = acc.guideHitCount[i];
            GuideValue g{};
            g.coverage = Saturate(static_cast<float>(hits) / sampleCount);
            if (hits == 0) {
                g.depth = kInf;
                g.confidence = 0.0f;
                guides[i] = g;
                continue;
            }

            float inv = 1.0f / static_cast<float>(hits);
            Vec3 meanNormal = acc.normalSum[i] * inv;
            g.normal = Normalize(meanNormal);
            g.albedo = acc.albedoSum[i] * inv;
            g.material = acc.materialSum[i] * inv;
            g.depth = acc.depthSum[i] * inv;
            float secondDepth = acc.depthSqSum[i] * inv;
            float depthVariance = std::max(secondDepth - g.depth*g.depth, 0.0f);
            g.confidence = ComputeGuideConfidence(meanNormal, g.depth, depthVariance, g.coverage);
            guides[i] = g;
        }
    });
}

void PrefilterVariance(
    const std::vector<float>& in,
    std::vector<float>& out,
    const std::vector<GuideValue>& guides,
    const DenoiseSettings& settings,
    int width,
    int height,
    unsigned workers)
{
    ParallelForRows(height, workers, [&](int y) {
        for (int x = 0; x < width; ++x) {
            int c = y*width + x;
            float sum = 0, weightSum = 0;
            for (int oy=-1; oy<=1; ++oy) for (int ox=-1; ox<=1; ++ox) {
                int sx=x+ox, sy=y+oy;
                if (sx<0 || sx>=width || sy<0 || sy>=height) continue;
                int s=sy*width+sx;
                float spatial = (ox==0 && oy==0) ? 4.0f : ((ox==0 || oy==0) ? 2.0f : 1.0f);
                float w = spatial * GuideSimilarityWeight(guides[c], guides[s], settings);
                sum += in[s] * w;
                weightSum += w;
            }
            out[c] = weightSum > 1e-8f ? sum/weightSum : in[c];
        }
    });
}

void ATrous(
    const std::vector<Vec3>& inColor,
    std::vector<Vec3>& outColor,
    const std::vector<float>& inVariance,
    std::vector<float>& outVariance,
    const std::vector<GuideValue>& guides,
    const DenoiseSettings& settings,
    int step,
    int width,
    int height,
    unsigned workers)
{
    static constexpr float kernel[5] = {1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16};

    ParallelForRows(height, workers, [&](int y) {
        for (int x=0; x<width; ++x) {
            int c=y*width+x;
            Vec3 center=inColor[c];
            float centerLum=Luminance(center);
            float centerVar=std::max(inVariance[c],0.0f);
            Vec3 sum(0);
            float varSum=0, weightSum=0;

            for (int ky=-2; ky<=2; ++ky) for (int kx=-2; kx<=2; ++kx) {
                int sx=x+kx*step, sy=y+ky*step;
                if (sx<0 || sx>=width || sy<0 || sy>=height) continue;
                int s=sy*width+sx;
                float spatial=kernel[kx+2]*kernel[ky+2];
                float guideW=GuideSimilarityWeight(guides[c],guides[s],settings);
                float sampleVar=std::max(inVariance[s],0.0f);
                float noiseSigma=std::sqrt(std::max(centerVar+sampleVar,1e-10f));
                float threshold=settings.phiColor*noiseSigma+1e-3f;
                float colorW=std::exp(-std::fabs(Luminance(inColor[s])-centerLum)/threshold);
                float w=spatial*guideW*colorW;
                sum += inColor[s]*w;
                varSum += sampleVar*w*w;
                weightSum += w;
            }

            if (weightSum>1e-8f) {
                outColor[c]=sum/weightSum;
                outVariance[c]=varSum/(weightSum*weightSum);
            } else {
                outColor[c]=center;
                outVariance[c]=centerVar;
            }
        }
    });
}

float ACESFilm(float x) {
    const float a=2.51f,b=0.03f,c=2.43f,d=0.59f,e=0.14f;
    return Saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

void SavePPM(const char* filename, const std::vector<Vec3>& fb, int width, int height) {
    std::ofstream f(filename);
    f << "P3\n" << width << ' ' << height << "\n255\n";
    for (int y=height-1; y>=0; --y) for (int x=0; x<width; ++x) {
        Vec3 c=fb[static_cast<std::size_t>(y)*width+x];
        c={ACESFilm(c.x),ACESFilm(c.y),ACESFilm(c.z)};
        c={std::pow(Saturate(c.x),1.0f/2.2f),std::pow(Saturate(c.y),1.0f/2.2f),std::pow(Saturate(c.z),1.0f/2.2f)};
        f << static_cast<int>(255.999f*Saturate(c.x)) << ' '
          << static_cast<int>(255.999f*Saturate(c.y)) << ' '
          << static_cast<int>(255.999f*Saturate(c.z)) << '\n';
    }
}

void SaveConfidence(const char* filename, const std::vector<GuideValue>& guides, int width, int height) {
    std::ofstream f(filename);
    f << "P3\n" << width << ' ' << height << "\n255\n";
    for (int y=height-1; y>=0; --y) for (int x=0; x<width; ++x) {
        int v=static_cast<int>(255.999f*Saturate(guides[static_cast<std::size_t>(y)*width+x].confidence));
        f << v << ' ' << v << ' ' << v << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    const int width=600, height=600;
    int targetSpp = argc>1 ? std::max(1,std::atoi(argv[1])) : 256;
    int samplesPerPass = argc>2 ? std::max(1,std::atoi(argv[2])) : 8;
    int maxDepth = argc>3 ? std::max(1,std::atoi(argv[3])) : 16;
    unsigned workers = std::max(1u,std::thread::hardware_concurrency());

    SceneStorage storage=MakeCornellBox();
    SceneView scene=storage.View();
    Camera camera=MakeCornellCamera(width,height);
    DenoiseSettings denoise{};

    std::cout << "Backend: CPU\nThreads: " << workers
              << "\nResolution: " << width << 'x' << height
              << "\nTarget Samples: " << targetSpp
              << "\nSamples / Pass: " << samplesPerPass
              << "\nMax Depth: " << maxDepth
              << "\nDenoiser: variance-guided A-Trous x" << denoise.iterations << "\n\n";

    std::size_t n=static_cast<std::size_t>(width)*height;
    AccumulationBuffer acc(n);
    auto renderStart=std::chrono::steady_clock::now();
    int accumulated=0;
    while (accumulated<targetSpp) {
        int pass=std::min(samplesPerPass,targetSpp-accumulated);
        ProgressivePass(acc,width,height,pass,accumulated,maxDepth,scene,camera,123456ULL,workers);
        accumulated+=pass;
        std::cout << "\rAccumulating: " << accumulated << '/' << targetSpp << " SPP" << std::flush;
    }
    auto renderStop=std::chrono::steady_clock::now();

    std::vector<Vec3> raw(n), denoiseA(n), denoiseB(n);
    std::vector<float> varianceRaw(n), varianceA(n), varianceB(n);
    std::vector<GuideValue> guides(n);
    Resolve(acc,accumulated,raw,varianceRaw,guides,workers,width,height);

    auto denoiseStart=std::chrono::steady_clock::now();
    PrefilterVariance(varianceRaw,varianceA,guides,denoise,width,height,workers);
    denoiseA=raw;
    auto* colorIn=&denoiseA; auto* colorOut=&denoiseB;
    auto* varIn=&varianceA; auto* varOut=&varianceB;
    for (int i=0;i<denoise.iterations;++i) {
        ATrous(*colorIn,*colorOut,*varIn,*varOut,guides,denoise,1<<i,width,height,workers);
        std::swap(colorIn,colorOut); std::swap(varIn,varOut);
    }
    auto denoiseStop=std::chrono::steady_clock::now();

    SavePPM("cornell_cpu_raw.ppm",raw,width,height);
    SavePPM("cornell_cpu_denoised.ppm",*colorIn,width,height);
    SaveConfidence("cornell_cpu_guide_confidence.ppm",guides,width,height);

    auto renderMs=std::chrono::duration<double,std::milli>(renderStop-renderStart).count();
    auto denoiseMs=std::chrono::duration<double,std::milli>(denoiseStop-denoiseStart).count();
    std::cout << "\nRender time: " << renderMs << " ms\nDenoise time: " << denoiseMs
              << " ms\nSaved: cornell_cpu_raw.ppm / cornell_cpu_denoised.ppm\n";
    return 0;
}
