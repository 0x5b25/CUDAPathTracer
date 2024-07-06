#include "OptixTracerProg.hpp"

#ifdef _WIN32
// The cfgmgr32 header is necessary for interrogating driver information in the registry.
#include <cfgmgr32.h>
// For convenience the library is also linked in automatically using the #pragma command.
#pragma comment(lib, "Cfgmgr32.lib")
#else
#include <dlfcn.h>
#endif
#include <cstring>

#include "OptixCommon.h"
#include "Shaders/DefaultShader.compiled.h"

namespace CUDATracer
{
    
#ifdef _WIN32
    // Code based on helper function in optix_stubs.h
    static void* optixLoadWindowsDll(void)
    {
        const char* optixDllName = "nvoptix.dll";
        void* handle = NULL;

        // Get the size of the path first, then allocate
        unsigned int size = GetSystemDirectoryA(NULL, 0);
        if (size == 0)
        {
            // Couldn't get the system path size, so bail
            return NULL;
        }

        size_t pathSize = size + 1 + strlen(optixDllName);
        char*  systemPath = (char*) malloc(pathSize);

        if (GetSystemDirectoryA(systemPath, size) != size - 1)
        {
            // Something went wrong
            free(systemPath);
            return NULL;
        }

        strcat(systemPath, "\\");
        strcat(systemPath, optixDllName);

        handle = LoadLibraryA(systemPath);

        free(systemPath);

        if (handle)
        {
            return handle;
        }

        // If we didn't find it, go looking in the register store.  Since nvoptix.dll doesn't
        // have its own registry entry, we are going to look for the OpenGL driver which lives
        // next to nvoptix.dll. 0 (null) will be returned if any errors occured.

        static const char* deviceInstanceIdentifiersGUID = "{4d36e968-e325-11ce-bfc1-08002be10318}";
        const ULONG        flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
        ULONG              deviceListSize = 0;

        if (CM_Get_Device_ID_List_SizeA(&deviceListSize, deviceInstanceIdentifiersGUID, flags) != CR_SUCCESS)
        {
            return NULL;
        }

        char* deviceNames = (char*) malloc(deviceListSize);

        if (CM_Get_Device_ID_ListA(deviceInstanceIdentifiersGUID, deviceNames, deviceListSize, flags))
        {
            free(deviceNames);
            return NULL;
        }

        DEVINST devID = 0;

        // Continue to the next device if errors are encountered.
        for (char* deviceName = deviceNames; *deviceName; deviceName += strlen(deviceName) + 1)
        {
            if (CM_Locate_DevNodeA(&devID, deviceName, CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS)
            {
                continue;
            }

            HKEY regKey = 0;
            if (CM_Open_DevNode_Key(devID, KEY_QUERY_VALUE, 0, RegDisposition_OpenExisting, &regKey, CM_REGISTRY_SOFTWARE) != CR_SUCCESS)
            {
                continue;
            }

            const char* valueName = "OpenGLDriverName";
            DWORD       valueSize = 0;

            LSTATUS     ret = RegQueryValueExA(regKey, valueName, NULL, NULL, NULL, &valueSize);
            if (ret != ERROR_SUCCESS)
            {
                RegCloseKey(regKey);
                continue;
            }

            char* regValue = (char*) malloc(valueSize);
            ret = RegQueryValueExA(regKey, valueName, NULL, NULL, (LPBYTE) regValue, &valueSize);
            if (ret != ERROR_SUCCESS)
            {
                free(regValue);
                RegCloseKey(regKey);
                continue;
            }

            // Strip the OpenGL driver dll name from the string then create a new string with
            // the path and the nvoptix.dll name
            for (int i = valueSize - 1; i >= 0 && regValue[i] != '\\'; --i)
            {
                regValue[i] = '\0';
            }

            size_t newPathSize = strlen(regValue) + strlen(optixDllName) + 1;
            char*  dllPath = (char*) malloc(newPathSize);
            strcpy(dllPath, regValue);
            strcat(dllPath, optixDllName);

            free(regValue);
            RegCloseKey(regKey);

            handle = LoadLibraryA((LPCSTR) dllPath);
            free(dllPath);

            if (handle)
            {
                break;
            }
        }

        free(deviceNames);

        return handle;
    }
#endif

    
    static OptixResult InitOptiXFunctionTable( OptixFunctionTable& api)
    {
    #ifdef _WIN32
        void* handle = optixLoadWindowsDll();
        if (!handle)
        {
            return OPTIX_ERROR_LIBRARY_NOT_FOUND;
        }

        void* symbol = reinterpret_cast<void*>(GetProcAddress((HMODULE) handle, "optixQueryFunctionTable"));
        if (!symbol)
        {
            return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
        }
    #else
        void* handle = dlopen("libnvoptix.so.1", RTLD_NOW);
        if (!handle)
        {
            return OPTIX_ERROR_LIBRARY_NOT_FOUND;
        }

        void* symbol = dlsym(handle, "optixQueryFunctionTable");
        if (!symbol)
        {
            return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
        }
    #endif

        OptixQueryFunctionTable_t* optixQueryFunctionTable = reinterpret_cast<OptixQueryFunctionTable_t*>(symbol);

        return optixQueryFunctionTable(OPTIX_ABI_VERSION, 0, 0, 0, &api, sizeof(OptixFunctionTable));
    }
    




    /*! SBT record for a raygen program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void* data;
    };

    /*! SBT record for a miss program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void* data;
    };

    /*! SBT record for a hitgroup program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        int objectID;
    };


    
    bool OptixTracerProg::InitOptiX()
    {
        //CUresult cuRes = cuInit(0); // Initialize CUDA driver API.
        //if (cuRes != CUDA_SUCCESS)
        //{
        //  std::cerr << "ERROR: initOptiX() cuInit() failed: " << cuRes << '\n';
        //  return false;
        //}

        //getSystemInformation(); // Get device attributes of all found devices. Fills m_deviceAttributes.

        CUdevice device = 0;

        CUresult cuRes = cuDeviceGet(&device, 0); // Get the CUdevice from device ordinal 0. (These are usually identical.)
        if (cuRes != CUDA_SUCCESS)
        {
          std::cerr << "ERROR: initOptiX() cuDeviceGet() failed: " << cuRes << '\n';
          return false;
        }

        cuRes = cuCtxCreate(&m_cudaContext, CU_CTX_SCHED_SPIN, device); // DEBUG What is the best CU_CTX_SCHED_* setting here.
        if (cuRes != CUDA_SUCCESS)
        {
          std::cerr << "ERROR: initOptiX() cuCtxCreate() failed: " << cuRes << '\n';
          return false;
        }

        // PERF Use CU_STREAM_NON_BLOCKING if there is any work running in parallel on multiple streams.
        cuRes = cuStreamCreate(&m_cudaStream, CU_STREAM_DEFAULT);
        if (cuRes != CUDA_SUCCESS)
        {
          std::cerr << "ERROR: initOptiX() cuStreamCreate() failed: " << cuRes << '\n';
          return false;
        }

        OptixResult res = InitOptiXFunctionTable(_api);
        if (res != OPTIX_SUCCESS)
        {
          std::cerr << "ERROR: initOptiX() initOptiXFunctionTable() failed: " << res << '\n';
          return false;
        }

        OptixDeviceContextOptions options = {};

        options.logCallbackFunction = &Logger::callback;
        options.logCallbackData     = &m_logger;
        options.logCallbackLevel    = 3; // Keep at warning level to suppress the disk cache messages.

        res = _api.optixDeviceContextCreate(m_cudaContext, &options, &m_context);
        if (res != OPTIX_SUCCESS)
        {
          std::cerr << "ERROR: initOptiX() optixDeviceContextCreate() failed: " << res << '\n';
          return false;
        }

        //return InitRenderer(); // Initialize all the rest.

        return true;
    }

    
    void OptixTracerProg::TearDownOptiX() {

        
        OPTIX_CHECK( _api.optixDeviceContextDestroy(m_context) );
        
        CU_CHECK( cuStreamDestroy(m_cudaStream) );
        CU_CHECK( cuCtxDestroy(m_cudaContext) );
    }

    
    bool OptixTracerProg::InitRenderer() {

        std::cout << "#osc: setting up module ..." << std::endl;
        /**************************************
         *        createModule                *
         **************************************/

        OptixModuleCompileOptions moduleCompileOptions{ };

        moduleCompileOptions.maxRegisterCount = 50;

#ifdef NDEBUG
        //Release config
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#else // !NDEBUG
        //Debug config
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

#endif // !NDEBUG

        OptixPipelineCompileOptions pipelineCompileOptions = {};
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.numPayloadValues = 2;
        pipelineCompileOptions.numAttributeValues = 2;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
        //const std::string ptxCode = embedded_ptx_code;

        char log[2048];
        size_t sizeof_log = sizeof(log);
//#if OPTIX_VERSION >= 70700
        OPTIX_CHECK(_api.optixModuleCreate(m_context,
            &moduleCompileOptions,
            &pipelineCompileOptions,
            (const char*)DefaultShader,//ptxCode.c_str(),
            sizeof(DefaultShader),
            log, &sizeof_log,
            &_module
        ));
//#else
//        OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
//            &moduleCompileOptions,
//            &pipelineCompileOptions,
//            ptxCode.c_str(),
//            ptxCode.size(),
//            log,      // Log string
//            &sizeof_log,// Log string sizse
//            &module
//        ));
//#endif
        if (sizeof_log > 1) PRINT(log);

        auto _CreateProgram = [&](OptixProgramGroupKind kind, const std::string& entry1, const std::string& entry2)-> OptixProgramGroup {

            OptixProgramGroupOptions pgOptions { };
            OptixProgramGroupDesc pgDesc { };
            pgDesc.kind = kind;// OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            switch (kind) {
            case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
                pgDesc.raygen.module = _module;
                pgDesc.raygen.entryFunctionName = entry1.c_str();// "__raygen__renderFrame";
                break;

            case OPTIX_PROGRAM_GROUP_KIND_MISS:
                pgDesc.miss.module = _module;
                pgDesc.miss.entryFunctionName = entry1.c_str();// "__raygen__renderFrame";
                break;

            case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
                pgDesc.hitgroup.moduleCH = _module;
                pgDesc.hitgroup.entryFunctionNameCH = entry1.c_str();
                pgDesc.hitgroup.moduleAH = _module;
                pgDesc.hitgroup.entryFunctionNameAH = entry2.c_str();
                break;
            }

            OptixProgramGroup pg;

            char log[2048];
            size_t sizeof_log = sizeof(log);
            auto res = _api.optixProgramGroupCreate(m_context,
                &pgDesc,
                1,
                &pgOptions,
                log, &sizeof_log,
                &pg
            );
            if (res == OPTIX_SUCCESS) {
                return pg;
            }
            else {
                if (sizeof_log > 1) PRINT(log);
                DebugBreak();
                return nullptr;
            }
        };

        std::cout << "#osc: creating raygen programs ..." << std::endl;
        _prog_raygen = _CreateProgram(OPTIX_PROGRAM_GROUP_KIND_RAYGEN, "__raygen__renderFrame", "");
        std::cout << "#osc: creating miss programs ..." << std::endl;
        _prog_raymiss = _CreateProgram(OPTIX_PROGRAM_GROUP_KIND_MISS, "__miss__radiance", "");
        std::cout << "#osc: creating hitgroup programs ..." << std::endl;
        _prog_rayhit = _CreateProgram(OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "__closesthit__radiance", "__anyhit__radiance");

        std::cout << "#osc: setting up optix pipeline ..." << std::endl;

        /**************************************
         *        createPipeline              *
         **************************************/

        std::vector<OptixProgramGroup> programGroups{
            _prog_raygen, _prog_raymiss, _prog_rayhit
        };

        OptixPipelineLinkOptions pipelineLinkOptions { };

        pipelineLinkOptions.maxTraceDepth = RT_TRACE_DEPTH;

        memset(log, 0, sizeof(log));
        sizeof_log = sizeof(log);
        OPTIX_CHECK(_api.optixPipelineCreate(m_context,
            &pipelineCompileOptions,
            &pipelineLinkOptions,
            programGroups.data(),
            (int)programGroups.size(),
            log, &sizeof_log,
            &_pipeline
        ));
        if (sizeof_log > 1) PRINT(log);

        OPTIX_CHECK(_api.optixPipelineSetStackSize
        (/* [in] The pipeline to configure the stack size for */
            _pipeline,
            /* [in] The direct stack size requirement for direct
               callables invoked from IS or AH. */
            2 * 1024,
            /* [in] The direct stack size requirement for direct
               callables invoked from RG, MS, or CH.  */
            2 * 1024,
            /* [in] The continuation stack requirement. */
            2 * 1024,
            /* [in] The maximum depth of a traversable graph
               passed to trace. */
            1));

        std::cout << "#osc: building SBT ..." << std::endl;

        /**************************************
         *        buildSBT                    *
         **************************************/
        _sbt = {};

        std::size_t totalSize = 0;

        RaygenRecord raygenRecord{ };
        std::size_t raygenRecordOffset = totalSize; totalSize += sizeof(RaygenRecord);
        OPTIX_CHECK(_api.optixSbtRecordPackHeader(_prog_raygen, &raygenRecord));

        MissRecord missRecord{ };
        std::size_t missRecordOffset = totalSize; totalSize += sizeof(MissRecord);
        OPTIX_CHECK(_api.optixSbtRecordPackHeader(_prog_raymiss, &missRecord));

        HitgroupRecord hitgroupRecord{ };
        std::size_t hitRecordOffset = totalSize; totalSize += sizeof(HitgroupRecord);
        OPTIX_CHECK(_api.optixSbtRecordPackHeader(_prog_rayhit, &hitgroupRecord));

        _pSbtRecords = new CUDABuffer{ totalSize };
        auto currPtr = (char*)_pSbtRecords->mutable_cpu_data();
        memcpy(currPtr + raygenRecordOffset, &raygenRecord, sizeof(RaygenRecord));
        memcpy(currPtr + missRecordOffset, &missRecord, sizeof(MissRecord));
        memcpy(currPtr + hitRecordOffset, &hitgroupRecord, sizeof(HitgroupRecord));

        auto devPtr = (std::size_t)_pSbtRecords->gpu_data();

        _sbt.raygenRecord = (CUdeviceptr)devPtr + raygenRecordOffset;

        _sbt.missRecordBase = (CUdeviceptr)devPtr + missRecordOffset;
        _sbt.missRecordStrideInBytes = sizeof(MissRecord);
        _sbt.missRecordCount = 1;

        _sbt.hitgroupRecordBase = (CUdeviceptr)devPtr + hitRecordOffset;
        _sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        _sbt.hitgroupRecordCount = 1;

        //_launchParamsBuffer = std::move(CUDABuffer(sizeof(LaunchParams)));
        _pLaunchParamsBuffer = new TypedBuffer<LaunchParams>();
        _pLaunchParamsBuffer->GetMutable<0>() = {};
        std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

        //std::cout << GDT_TERMINAL_GREEN;
        std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
        //std::cout << GDT_TERMINAL_DEFAULT;

        return true;
    }


    void OptixTracerProg::TearDownRenderer() {
        OPTIX_CHECK(_api.optixPipelineDestroy(_pipeline));

        for (auto& pg : { _prog_raygen, _prog_raymiss, _prog_rayhit }) {
            OPTIX_CHECK(_api.optixProgramGroupDestroy(pg));
        }

        OPTIX_CHECK(_api.optixModuleDestroy(_module));

        delete _pSbtRecords;
        delete _pLaunchParamsBuffer;        
    }


    OptixTracerProg::OptixTracerProg() 
        : PathTracer()
        , m_logger(std::cerr)
    {
        //cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 8192);
        InitOptiX();
        InitRenderer();
    }


    OptixTracerProg:: ~OptixTracerProg() {
        TearDownRenderer();

        TearDownOptiX();
    }

    void OptixTracerProg::Trace(
        CUDAScene& scn, TypedBuffer<PathTraceSettings>& settings,
        float* accBuffer, char* buffer
    ){
        auto stdata = (PathTraceSettings*)settings.cpu_data();

        auto w = stdata->viewportWidth;
        auto h = stdata->viewportHeight;

        if (w * h == 0) return;
//
//
        //const unsigned batchW = BATCH_W, batchH = BATCH_H;
        //const unsigned blockW = (w + batchW - 1) / batchW;
        //const unsigned blockH = (h + batchH - 1) / batchH;
//
        //uint3 blkPerGrid{ blockW, blockH, 1 };
        ////uint3 blkPerGrid{ 2, 2, 1 };
        //uint3 threadPerBlk{batchW, batchH, 1};
//
        //auto scene_gpu = (SceneInfoHeader*)scn.GetGPUData();
        //auto settings_gpu = (PathTraceSettings*)settings.gpu_data();
        ////auto cache_gpu = (std::uint8_t*)_rayCache->gpu_data();
        //RTProgram prog{ scene_gpu, settings_gpu};
//
        //GenRay << <blkPerGrid, threadPerBlk >> > (prog, accBuffer, buffer);

        // sanity check: make sure we launch only after first resize is
    // already done:
        {
            auto& launchParams = _pLaunchParamsBuffer->GetMutable<0>();
            launchParams.fbSize.x = w;
            launchParams.fbSize.y = h;
            launchParams.colorBuffer = buffer;
        }

        //Upload to GPU
        auto devPtr = (CUdeviceptr)_pLaunchParamsBuffer->gpu_data();

        {
            auto& launchParams = _pLaunchParamsBuffer->GetMutable<0>();
            launchParams.frameID++;
        }
        OPTIX_CHECK(_api.optixLaunch(/*! pipeline we're launching launch: */
            _pipeline, m_cudaStream,
            /*! parameters and SBT */
            devPtr,
            _pLaunchParamsBuffer->size(),
            &_sbt,
            /*! dimensions of the launch: */
            w,
            h,
            1
        ));
        // sync - make sure the frame is rendered before we download and
        // display (obviously, for a high-performance application you
        // want to use streams and double-buffering, but for this simple
        // example, this will have to do)
        CUDA_CHECK(cudaDeviceSynchronize());
        
    }

} // namespace CUDAPathTracer

