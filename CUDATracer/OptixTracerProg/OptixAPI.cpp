#include "OptixAPI.hpp"

#include "OptixCommon.h"

#ifdef _WIN32
#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
// The cfgmgr32 header is necessary for interrogating driver information in the registry.
#include <cfgmgr32.h>
// For convenience the library is also linked in automatically using the #pragma command.
#pragma comment(lib, "Cfgmgr32.lib")
#else
#include <dlfcn.h>
#endif


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
    



namespace CUDATracer
{
    OptixAPI::OptixAPI() {
        OPTIX_CHECK(InitOptiXFunctionTable(_api));
    }

} // namespace CUDATracer

