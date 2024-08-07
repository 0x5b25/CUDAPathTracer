#include "CUDATracer.hpp"

#include "CUDATracerProg/CUDATracerProg.hpp"
#include "OptixTracerProg/OptixTracerProg.hpp"

namespace CUDATracer {

    PathTracer::PathTracer() {

        CU_CHECK( cuInit(0) ); // Initialize CUDA driver API.

        // Get the maximum CUDA version supported by the installed display driver.
        int versionDriver = 0;
        CU_CHECK( cuDriverGetVersion(&versionDriver) ); 

        // The CUDA version is returned as (1000 * major + 10 * minor).
        int major =  versionDriver / 1000;
        int minor = (versionDriver - major * 1000) / 10;
        std::cout << "CUDA Driver Version = " << major << "." << minor << '\n';

        int countDevices = 0;
        CU_CHECK( cuDeviceGetCount(&countDevices) );
        std::cout << "CUDA Device Count = " << countDevices << '\n';

        //char name[1024];
        //name[1023] = 0;

        for (int ordinal = 0; ordinal < countDevices; ++ordinal)
        {
            CUdevice device = 0;
            CU_CHECK( cuDeviceGet(&device, ordinal) );
            
            DeviceAttribute attr = {};

            CU_CHECK( cuDeviceGetName(attr.name, 1023, device) );
            std::cout << "CUDA Device " << device << ": " << attr.name << '\n';


            CU_CHECK( cuDeviceGetAttribute(&attr.maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maxBlockDimX, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maxBlockDimY, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maxBlockDimZ, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maxGridDimX, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maxGridDimY, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maxGridDimZ, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maxSharedMemoryPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.sharedMemoryPerBlock, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.totalConstantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maxPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maxRegistersPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.registersPerBlock, CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.textureAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.gpuOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.multiprocessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.kernelExecTimeout, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.canMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dDepth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLayeredHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dArrayWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dArrayHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dArrayNumslices, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.surfaceAlignment, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.concurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.eccEnabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.pciBusId, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.pciDeviceId, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.tccDriver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.memoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.globalMemoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.l2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maxThreadsPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.asyncEngineCount, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.unifiedAddressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.canTex2dGather, CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dGatherWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dGatherHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dWidthAlternate, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dHeightAlternate, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dDepthAlternate, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.pciDomainId, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.texturePitchAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexturecubemapWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexturecubemapLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexturecubemapLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface1dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface3dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface3dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface3dDepth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface1dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface1dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dLayeredHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurfacecubemapWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurfacecubemapLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurfacecubemapLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dLinearWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLinearWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLinearHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLinearPitch, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dMipmappedWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dMipmappedHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dMipmappedWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.streamPrioritiesSupported, CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.globalL1CacheSupported, CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.localL1CacheSupported, CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maxSharedMemoryPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maxRegistersPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.managedMemory, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.multiGpuBoard, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.multiGpuBoardGroupId, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.hostNativeAtomicSupported, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.singleToDoublePrecisionPerfRatio, CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.pageableMemoryAccess, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.concurrentManagedAccess, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.computePreemptionSupported, CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.canUseHostPointerForRegisteredMem, CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.canUse64BitStreamMemOps, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.canUseStreamWaitValueNor, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.cooperativeLaunch, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.cooperativeMultiDeviceLaunch, CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.maxSharedMemoryPerBlockOptin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.canFlushRemoteWrites, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.hostRegisterSupported, CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.pageableMemoryAccessUsesHostPageTables, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, device) );
            CU_CHECK( cuDeviceGetAttribute(&attr.directManagedMemAccessFromHost, CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, device) );

            _devAttrs.push_back(attr);
        }

    }


    PathTracer* MakeCUDATracerProg() {
        return new CUDATracerProg();
    }


    PathTracer* MakeOptixTracerProg() {
        
        return new OptixTracerProg();
    }


}