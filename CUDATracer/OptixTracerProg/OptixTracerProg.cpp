#include "OptixTracerProg.hpp"

#ifdef _WIN32
// The cfgmgr32 header is necessary for interrogating driver information in the registry.
#include <cfgmgr32.h>
// For convenience the library is also linked in automatically using the #pragma command.
#pragma comment(lib, "Cfgmgr32.lib")
#else
#include <dlfcn.h>
#endif


#include "OptixCommon.h"


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
    
    
    bool OptixTracerProg::InitOptiX()
    {
        CUresult cuRes = cuInit(0); // Initialize CUDA driver API.
        if (cuRes != CUDA_SUCCESS)
        {
          std::cerr << "ERROR: initOptiX() cuInit() failed: " << cuRes << '\n';
          return false;
        }

        //getSystemInformation(); // Get device attributes of all found devices. Fills m_deviceAttributes.

        CUdevice device = 0;

        cuRes = cuDeviceGet(&device, 0); // Get the CUdevice from device ordinal 0. (These are usually identical.)
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

        OptixResult res = InitOptiXFunctionTable(m_api);
        if (res != OPTIX_SUCCESS)
        {
          std::cerr << "ERROR: initOptiX() initOptiXFunctionTable() failed: " << res << '\n';
          return false;
        }

        OptixDeviceContextOptions options = {};

        options.logCallbackFunction = &Logger::callback;
        options.logCallbackData     = &m_logger;
        options.logCallbackLevel    = 3; // Keep at warning level to suppress the disk cache messages.

        res = m_api.optixDeviceContextCreate(m_cudaContext, &options, &m_context);
        if (res != OPTIX_SUCCESS)
        {
          std::cerr << "ERROR: initOptiX() optixDeviceContextCreate() failed: " << res << '\n';
          return false;
        }

        return InitRenderer(); // Initialize all the rest.

        //return true;
    }

    
    void OptixTracerProg::TearDownOptiX() {

        
        OPTIX_CHECK( m_api.optixDeviceContextDestroy(m_context) );
        
        CU_CHECK( cuStreamDestroy(m_cudaStream) );
        CU_CHECK( cuCtxDestroy(m_cudaContext) );
    }

    
    bool OptixTracerProg::InitRenderer() {
        return true;
    }

    OptixTracerProg::OptixTracerProg() 
        : PathTracer()
        , m_logger(std::cerr)
    {
        //cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 8192);
        InitOptiX();
    }

    void OptixTracerProg::Trace(
        CUDAScene& scn, TypedBuffer<PathTraceSettings>& settings,
        float* accBuffer, char* buffer
    ){
        //auto stdata = (PathTraceSettings*)settings.cpu_data();
//
        //auto w = stdata->viewportWidth;
        //auto h = stdata->viewportHeight;
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

        
    }

} // namespace CUDAPathTracer

