#pragma once

#include "CUDATracer.hpp"

#include "optix.h"

// OptiX 7 function table structure.
#include "optix_function_table.h"

#include "Logger.hpp"

namespace CUDATracer {
    
    class OptixTracerProg : public PathTracer {
    
    private:
        // CUDA native types are prefixed with "cuda".
        CUcontext m_cudaContext;
        CUstream  m_cudaStream;

        // The handle for the registered OpenGL PBO when using interop.
        CUgraphicsResource m_cudaGraphicsResource;

        // All others are OptiX types.
        OptixFunctionTable m_api;
        OptixDeviceContext m_context;

        Logger m_logger;

        OptixTraversableHandle m_root;  // Scene root
        CUdeviceptr            m_d_ias; // Scene root's IAS (instance acceleration structure).


    private:

        bool InitOptiX();
        void TearDownOptiX();

        bool InitRenderer();
    
    public:
        OptixTracerProg();
        virtual ~OptixTracerProg() override {}

    
        virtual void Trace(
            //const TypedBuffer<Camera>& cam,
            CUDAScene& scn,
            TypedBuffer<PathTraceSettings>& settings,
            float* accBuffer,
            char* buffer
        ) override;

    private:

    };


} // namespace CUDATracer
