#pragma once

#include "CUDATracer.hpp"

namespace CUDATracer {
    
    class CUDATracerProg : public PathTracer {
    public:
        CUDATracerProg() ;
        virtual ~CUDATracerProg() override {}

    
        virtual void Trace(
            //const TypedBuffer<Camera>& cam,
            CUDAScene& scn,
            TypedBuffer<PathTraceSettings>& settings,
            float* accBuffer,
            char* buffer
        ) override;
    };


} // namespace CUDATracer {

