#pragma once

#include "CUDACommon.h"
#include "Types.hpp"
#include "CUDABuffer.hpp"

namespace CUDATracer {

    class IPathTracer;

    struct PathTraceSettings {
        Camera cam;
        unsigned viewportWidth, viewportHeight;
        unsigned frameID;
        unsigned seed;
        unsigned maxDepth;
    };
    

    class ITraceable {

    public:
        virtual ~ITraceable() {}
    };

    class IPathTracer {

    public:
        struct DeviceAttribute
        {
            char name[1024];

            int maxThreadsPerBlock;
            int maxBlockDimX;
            int maxBlockDimY;
            int maxBlockDimZ;
            int maxGridDimX;
            int maxGridDimY;
            int maxGridDimZ;
            int maxSharedMemoryPerBlock;
            int sharedMemoryPerBlock;
            int totalConstantMemory;
            int warpSize;
            int maxPitch;
            int maxRegistersPerBlock;
            int registersPerBlock;
            int clockRate;
            int textureAlignment;
            int gpuOverlap;
            int multiprocessorCount;
            int kernelExecTimeout;
            int integrated;
            int canMapHostMemory;
            int computeMode;
            int maximumTexture1dWidth;
            int maximumTexture2dWidth;
            int maximumTexture2dHeight;
            int maximumTexture3dWidth;
            int maximumTexture3dHeight;
            int maximumTexture3dDepth;
            int maximumTexture2dLayeredWidth;
            int maximumTexture2dLayeredHeight;
            int maximumTexture2dLayeredLayers;
            int maximumTexture2dArrayWidth;
            int maximumTexture2dArrayHeight;
            int maximumTexture2dArrayNumslices;
            int surfaceAlignment;
            int concurrentKernels;
            int eccEnabled;
            int pciBusId;
            int pciDeviceId;
            int tccDriver;
            int memoryClockRate;
            int globalMemoryBusWidth;
            int l2CacheSize;
            int maxThreadsPerMultiprocessor;
            int asyncEngineCount;
            int unifiedAddressing;
            int maximumTexture1dLayeredWidth;
            int maximumTexture1dLayeredLayers;
            int canTex2dGather;
            int maximumTexture2dGatherWidth;
            int maximumTexture2dGatherHeight;
            int maximumTexture3dWidthAlternate;
            int maximumTexture3dHeightAlternate;
            int maximumTexture3dDepthAlternate;
            int pciDomainId;
            int texturePitchAlignment;
            int maximumTexturecubemapWidth;
            int maximumTexturecubemapLayeredWidth;
            int maximumTexturecubemapLayeredLayers;
            int maximumSurface1dWidth;
            int maximumSurface2dWidth;
            int maximumSurface2dHeight;
            int maximumSurface3dWidth;
            int maximumSurface3dHeight;
            int maximumSurface3dDepth;
            int maximumSurface1dLayeredWidth;
            int maximumSurface1dLayeredLayers;
            int maximumSurface2dLayeredWidth;
            int maximumSurface2dLayeredHeight;
            int maximumSurface2dLayeredLayers;
            int maximumSurfacecubemapWidth;
            int maximumSurfacecubemapLayeredWidth;
            int maximumSurfacecubemapLayeredLayers;
            int maximumTexture1dLinearWidth;
            int maximumTexture2dLinearWidth;
            int maximumTexture2dLinearHeight;
            int maximumTexture2dLinearPitch;
            int maximumTexture2dMipmappedWidth;
            int maximumTexture2dMipmappedHeight;
            int computeCapabilityMajor;
            int computeCapabilityMinor;
            int maximumTexture1dMipmappedWidth;
            int streamPrioritiesSupported;
            int globalL1CacheSupported;
            int localL1CacheSupported;
            int maxSharedMemoryPerMultiprocessor;
            int maxRegistersPerMultiprocessor;
            int managedMemory;
            int multiGpuBoard;
            int multiGpuBoardGroupId;
            int hostNativeAtomicSupported;
            int singleToDoublePrecisionPerfRatio;
            int pageableMemoryAccess;
            int concurrentManagedAccess;
            int computePreemptionSupported;
            int canUseHostPointerForRegisteredMem;
            int canUse64BitStreamMemOps;
            int canUseStreamWaitValueNor;
            int cooperativeLaunch;
            int cooperativeMultiDeviceLaunch;
            int maxSharedMemoryPerBlockOptin;
            int canFlushRemoteWrites;
            int hostRegisterSupported;
            int pageableMemoryAccessUsesHostPageTables;
            int directManagedMemAccessFromHost;
        };

        std::vector<DeviceAttribute> _devAttrs;
        
    public:
        IPathTracer();
        virtual ~IPathTracer() {}

        virtual ITraceable* CreateTraceable(const Scene& scene) const = 0;

    
        virtual void Trace(
            //const TypedBuffer<Camera>& cam,
            const ITraceable& scn,
            TypedBuffer<PathTraceSettings>& settings,
            float* accBuffer,
            char* buffer
        ) = 0;
    };

    IPathTracer* MakeCUDATracerProg();
    IPathTracer* MakeOptixTracerProg();
}
