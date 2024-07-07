#pragma once

#include "CUDATracer.hpp"

#include "optix.h"

// OptiX 7 function table structure.
#include "optix_function_table.h"

#include "Logger.hpp"
#include "OptixAPI.hpp"
#include "OptixSBT.hpp"

#include "Shaders/ShaderTypes.h"

namespace CUDATracer {
    
    class OptixTracerProg : public IPathTracer {
    
    private:
        // CUDA native types are prefixed with "cuda".
        CUcontext m_cudaContext;
        CUstream  m_cudaStream;

        // The handle for the registered OpenGL PBO when using interop.
        CUgraphicsResource m_cudaGraphicsResource;

        // All others are OptiX types.
        OptixAPI _api;
        OptixDeviceContext m_context;

        Logger m_logger;

        OptixTraversableHandle m_root;  // Scene root
        CUdeviceptr            m_d_ias; // Scene root's IAS (instance acceleration structure).

        OptixModule _module;
        OptixProgramGroup _prog_raygen, _prog_raymiss, _prog_rayhit;
        
        OptixPipeline _pipeline;

        TypedBuffer<LaunchParams>* _pLaunchParamsBuffer;

    private:

        bool InitOptiX();
        void TearDownOptiX();

        bool InitRenderer();
        void TearDownRenderer();
    
    public:
        OptixTracerProg();
        virtual ~OptixTracerProg() override;

        virtual ITraceable* CreateTraceable(const Scene& scene) const override;
    
        virtual void Trace(
            //const TypedBuffer<Camera>& cam,
            ITraceable& scn,
            TypedBuffer<PathTraceSettings>& settings,
            float* accBuffer,
            char* buffer
        ) override;

    private:

    };


} // namespace CUDATracer
