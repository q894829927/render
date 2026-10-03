#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "Core/Denoiser/Denoiser.h"
#include "Core/Integrator/Integrator.h"
#include "Core/Scene/CornellBox.h"

using namespace render;

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){ \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) << " (" << __FILE__ << ':' << __LINE__ << ")\n"; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

namespace {

struct CudaRng {
    curandStatePhilox4_32_10_t state;

    __device__ CudaRng(unsigned long long seed, unsigned long long sequence) {
        curand_init(seed, sequence, 0ULL, &state);
    }

    __device__ float NextFloat() {
        return fminf(curand_uniform(&state), 0.99999994f);
    }
};

struct DeviceAccumulation {
    Vec3* radianceSum;
    float* luminanceSum;
    float* luminanceSqSum;
    Vec3* normalSum;
    Vec3* albedoSum;
    Vec3* materialSum;
    float* depthSum;
    float* depthSqSum;
    unsigned int* guideHitCount;
};

__global__ void ProgressiveRenderKernel(
    DeviceAccumulation acc,
    int width,
    int height,
    int samplesThisPass,
    int sampleOffset,
    int maxDepth,
    SceneView scene,
    Camera camera,
    unsigned long long baseSeed)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=width || y>=height) return;
    int idx=y*width+x;

    unsigned long long passSeed=baseSeed+
        static_cast<unsigned long long>(sampleOffset)*0x9E3779B97F4A7C15ULL;
    CudaRng rng(passSeed, static_cast<unsigned long long>(idx));

    Vec3 radiancePass(0), normalPass(0), albedoPass(0), materialPass(0);
    float lumPass=0, lumSqPass=0, depthPass=0, depthSqPass=0;
    unsigned guideHits=0;

    for(int s=0;s<samplesThisPass;++s){
        float px=(2.0f*((x+rng.NextFloat())/static_cast<float>(width))-1.0f)*camera.viewportWidth*0.5f;
        float py=(2.0f*((y+rng.NextFloat())/static_cast<float>(height))-1.0f)*camera.viewportHeight*0.5f;
        Ray ray{camera.position,Normalize(camera.forward+camera.right*px+camera.up*py)};
        PrimaryGuide guide{};
        Vec3 sample=TracePath(ray,scene,maxDepth,rng,&guide);
        radiancePass+=sample;
        float lum=Luminance(sample);
        lumPass+=lum;
        lumSqPass+=lum*lum;
        if(guide.valid){
            normalPass+=guide.normal;
            albedoPass+=guide.albedo;
            materialPass+=guide.material;
            depthPass+=guide.depth;
            depthSqPass+=guide.depth*guide.depth;
            ++guideHits;
        }
    }

    acc.radianceSum[idx]+=radiancePass;
    acc.luminanceSum[idx]+=lumPass;
    acc.luminanceSqSum[idx]+=lumSqPass;
    acc.normalSum[idx]+=normalPass;
    acc.albedoSum[idx]+=albedoPass;
    acc.materialSum[idx]+=materialPass;
    acc.depthSum[idx]+=depthPass;
    acc.depthSqSum[idx]+=depthSqPass;
    acc.guideHitCount[idx]+=guideHits;
}

__global__ void ResolveKernel(
    DeviceAccumulation acc,
    Vec3* raw,
    float* variance,
    GuideValue* guides,
    int pixelCount,
    int samples)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=pixelCount) return;
    float n=static_cast<float>(samples > 1 ? samples : 1);
    raw[i]=acc.radianceSum[i]/n;

    float meanLum=acc.luminanceSum[i]/n;
    float second=acc.luminanceSqSum[i]/n;
    float sampleVar=fmaxf(second-meanLum*meanLum,0.0f);
    variance[i]=sampleVar/n;

    unsigned hits=acc.guideHitCount[i];
    GuideValue g{};
    g.coverage=Saturate(static_cast<float>(hits)/n);
    if(hits==0){
        g.depth=kInf;
        g.confidence=0.0f;
        guides[i]=g;
        return;
    }

    float inv=1.0f/static_cast<float>(hits);
    Vec3 meanNormal=acc.normalSum[i]*inv;
    g.normal=Normalize(meanNormal);
    g.albedo=acc.albedoSum[i]*inv;
    g.material=acc.materialSum[i]*inv;
    g.depth=acc.depthSum[i]*inv;
    float secondDepth=acc.depthSqSum[i]*inv;
    float depthVar=fmaxf(secondDepth-g.depth*g.depth,0.0f);
    g.confidence=ComputeGuideConfidence(meanNormal,g.depth,depthVar,g.coverage);
    guides[i]=g;
}

__global__ void PrefilterVarianceKernel(
    const float* input,
    float* output,
    const GuideValue* guides,
    DenoiseSettings settings,
    int width,
    int height)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=width || y>=height) return;
    int c=y*width+x;

    float sum=0, weightSum=0;
    for(int oy=-1;oy<=1;++oy) for(int ox=-1;ox<=1;++ox){
        int sx=x+ox,sy=y+oy;
        if(sx<0||sx>=width||sy<0||sy>=height) continue;
        int s=sy*width+sx;
        float spatial=(ox==0&&oy==0)?4.0f:((ox==0||oy==0)?2.0f:1.0f);
        float w=spatial*GuideSimilarityWeight(guides[c],guides[s],settings);
        sum+=input[s]*w;
        weightSum+=w;
    }
    output[c]=weightSum>1e-8f?sum/weightSum:input[c];
}

__global__ void ATrousKernel(
    const Vec3* inputColor,
    Vec3* outputColor,
    const float* inputVariance,
    float* outputVariance,
    const GuideValue* guides,
    DenoiseSettings settings,
    int step,
    int width,
    int height)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=width||y>=height) return;
    const float kernel[5]={1.0f/16,4.0f/16,6.0f/16,4.0f/16,1.0f/16};
    int c=y*width+x;
    Vec3 center=inputColor[c];
    float centerLum=Luminance(center);
    float centerVar=fmaxf(inputVariance[c],0.0f);
    Vec3 sum(0);
    float varSum=0,weightSum=0;

    for(int ky=-2;ky<=2;++ky) for(int kx=-2;kx<=2;++kx){
        int sx=x+kx*step,sy=y+ky*step;
        if(sx<0||sx>=width||sy<0||sy>=height) continue;
        int s=sy*width+sx;
        float spatial=kernel[kx+2]*kernel[ky+2];
        float guideW=GuideSimilarityWeight(guides[c],guides[s],settings);
        float sampleVar=fmaxf(inputVariance[s],0.0f);
        float noiseSigma=sqrtf(fmaxf(centerVar+sampleVar,1e-10f));
        float threshold=settings.phiColor*noiseSigma+1e-3f;
        float colorW=expf(-fabsf(Luminance(inputColor[s])-centerLum)/threshold);
        float w=spatial*guideW*colorW;
        sum+=inputColor[s]*w;
        varSum+=sampleVar*w*w;
        weightSum+=w;
    }

    if(weightSum>1e-8f){
        outputColor[c]=sum/weightSum;
        outputVariance[c]=varSum/(weightSum*weightSum);
    }else{
        outputColor[c]=center;
        outputVariance[c]=centerVar;
    }
}

float ACESFilm(float x){
    const float a=2.51f,b=0.03f,c=2.43f,d=0.59f,e=0.14f;
    return Saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

void SavePPM(const char* filename,const std::vector<Vec3>& fb,int width,int height){
    std::ofstream f(filename);
    f<<"P3\n"<<width<<' '<<height<<"\n255\n";
    for(int y=height-1;y>=0;--y) for(int x=0;x<width;++x){
        Vec3 c=fb[static_cast<std::size_t>(y)*width+x];
        c={ACESFilm(c.x),ACESFilm(c.y),ACESFilm(c.z)};
        c={powf(Saturate(c.x),1.0f/2.2f),powf(Saturate(c.y),1.0f/2.2f),powf(Saturate(c.z),1.0f/2.2f)};
        f<<static_cast<int>(255.999f*Saturate(c.x))<<' '
         <<static_cast<int>(255.999f*Saturate(c.y))<<' '
         <<static_cast<int>(255.999f*Saturate(c.z))<<'\n';
    }
}

void Zero(void* p,std::size_t bytes){CUDA_CHECK(cudaMemset(p,0,bytes));}

template<typename T>
T* Alloc(std::size_t count){
    T* p=nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p),sizeof(T)*count));
    return p;
}

} // namespace

int main(int argc,char** argv){
    const int width=600,height=600;
    int targetSpp=argc>1?std::max(1,std::atoi(argv[1])):256;
    int samplesPerPass=argc>2?std::max(1,std::atoi(argv[2])):8;
    int maxDepth=argc>3?std::max(1,std::atoi(argv[3])):16;

    int device=0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop,device));

    SceneStorage hostStorage=MakeCornellBox();
    Camera camera=MakeCornellCamera(width,height);
    DenoiseSettings denoise{};

    Rect* dRects=Alloc<Rect>(hostStorage.rects.size());
    OrientedBox* dBoxes=Alloc<OrientedBox>(hostStorage.boxes.size());
    CUDA_CHECK(cudaMemcpy(dRects,hostStorage.rects.data(),sizeof(Rect)*hostStorage.rects.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBoxes,hostStorage.boxes.data(),sizeof(OrientedBox)*hostStorage.boxes.size(),cudaMemcpyHostToDevice));
    SceneView scene{dRects,static_cast<int>(hostStorage.rects.size()),dBoxes,static_cast<int>(hostStorage.boxes.size()),hostStorage.light};

    std::size_t n=static_cast<std::size_t>(width)*height;
    DeviceAccumulation acc{};
    acc.radianceSum=Alloc<Vec3>(n); acc.luminanceSum=Alloc<float>(n); acc.luminanceSqSum=Alloc<float>(n);
    acc.normalSum=Alloc<Vec3>(n); acc.albedoSum=Alloc<Vec3>(n); acc.materialSum=Alloc<Vec3>(n);
    acc.depthSum=Alloc<float>(n); acc.depthSqSum=Alloc<float>(n); acc.guideHitCount=Alloc<unsigned int>(n);
    Zero(acc.radianceSum,sizeof(Vec3)*n); Zero(acc.luminanceSum,sizeof(float)*n); Zero(acc.luminanceSqSum,sizeof(float)*n);
    Zero(acc.normalSum,sizeof(Vec3)*n); Zero(acc.albedoSum,sizeof(Vec3)*n); Zero(acc.materialSum,sizeof(Vec3)*n);
    Zero(acc.depthSum,sizeof(float)*n); Zero(acc.depthSqSum,sizeof(float)*n); Zero(acc.guideHitCount,sizeof(unsigned int)*n);

    Vec3* dRaw=Alloc<Vec3>(n); Vec3* dA=Alloc<Vec3>(n); Vec3* dB=Alloc<Vec3>(n);
    float* dVarRaw=Alloc<float>(n); float* dVarA=Alloc<float>(n); float* dVarB=Alloc<float>(n);
    GuideValue* dGuides=Alloc<GuideValue>(n);

    std::cout<<"GPU: "<<prop.name<<"\nResolution: "<<width<<'x'<<height
             <<"\nTarget Samples: "<<targetSpp<<"\nSamples / Pass: "<<samplesPerPass
             <<"\nMax Depth: "<<maxDepth<<"\nDenoiser: variance-guided A-Trous x"<<denoise.iterations<<"\n\n";

    dim3 block(16,16),grid((width+15)/16,(height+15)/16);
    int linearBlock=256,linearGrid=(static_cast<int>(n)+linearBlock-1)/linearBlock;

    cudaEvent_t start{},stop{};
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    int accumulated=0;
    while(accumulated<targetSpp){
        int pass=std::min(samplesPerPass,targetSpp-accumulated);
        ProgressiveRenderKernel<<<grid,block>>>(acc,width,height,pass,accumulated,maxDepth,scene,camera,123456ULL);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        accumulated+=pass;
        std::cout<<"\rAccumulating: "<<accumulated<<'/'<<targetSpp<<" SPP"<<std::flush;
    }
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float renderMs=0; CUDA_CHECK(cudaEventElapsedTime(&renderMs,start,stop));

    ResolveKernel<<<linearGrid,linearBlock>>>(acc,dRaw,dVarRaw,dGuides,static_cast<int>(n),accumulated);
    CUDA_CHECK(cudaGetLastError());
    PrefilterVarianceKernel<<<grid,block>>>(dVarRaw,dVarA,dGuides,denoise,width,height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dA,dRaw,sizeof(Vec3)*n,cudaMemcpyDeviceToDevice));

    cudaEvent_t ds{},de{}; CUDA_CHECK(cudaEventCreate(&ds)); CUDA_CHECK(cudaEventCreate(&de)); CUDA_CHECK(cudaEventRecord(ds));
    Vec3* colorIn=dA; Vec3* colorOut=dB; float* varIn=dVarA; float* varOut=dVarB;
    for(int i=0;i<denoise.iterations;++i){
        ATrousKernel<<<grid,block>>>(colorIn,colorOut,varIn,varOut,dGuides,denoise,1<<i,width,height);
        CUDA_CHECK(cudaGetLastError());
        std::swap(colorIn,colorOut); std::swap(varIn,varOut);
    }
    CUDA_CHECK(cudaEventRecord(de)); CUDA_CHECK(cudaEventSynchronize(de));
    float denoiseMs=0; CUDA_CHECK(cudaEventElapsedTime(&denoiseMs,ds,de));

    std::vector<Vec3> raw(n),denoised(n);
    CUDA_CHECK(cudaMemcpy(raw.data(),dRaw,sizeof(Vec3)*n,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(denoised.data(),colorIn,sizeof(Vec3)*n,cudaMemcpyDeviceToHost));
    SavePPM("cornell_cuda_raw.ppm",raw,width,height);
    SavePPM("cornell_cuda_denoised.ppm",denoised,width,height);

    std::cout<<"\nRender time: "<<renderMs<<" ms\nDenoise time: "<<denoiseMs
             <<" ms\nSaved: cornell_cuda_raw.ppm / cornell_cuda_denoised.ppm\n";

    cudaEventDestroy(start); cudaEventDestroy(stop); cudaEventDestroy(ds); cudaEventDestroy(de);
    cudaFree(dGuides); cudaFree(dVarB); cudaFree(dVarA); cudaFree(dVarRaw); cudaFree(dB); cudaFree(dA); cudaFree(dRaw);
    cudaFree(acc.guideHitCount); cudaFree(acc.depthSqSum); cudaFree(acc.depthSum); cudaFree(acc.materialSum);
    cudaFree(acc.albedoSum); cudaFree(acc.normalSum); cudaFree(acc.luminanceSqSum); cudaFree(acc.luminanceSum); cudaFree(acc.radianceSum);
    cudaFree(dBoxes); cudaFree(dRects);
    return 0;
}
