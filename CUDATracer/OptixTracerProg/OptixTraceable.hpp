#pragma once

#include "CUDATracer.hpp"

#include "OptixAPI.hpp"
#include "OptixSBT.hpp"
#include "OptixCommon.h"

namespace CUDATracer {

    class OptixTraceable : public ITraceable {
    
        DISABLE_COPY_AND_ASSIGN(OptixTraceable);
    
        const OptixAPI& _api;
        OptixDeviceContext _ctx;
    
    
        CUDABuffer* _vertBuffer;
        CUDABuffer* _indBuffer;
        CUDABuffer* _asBuffer;
        OptixTraversableHandle _asHandle;
    
        OptixSBT _sbt;
    
    public:
        OptixTraceable(
            const OptixAPI& api,
            OptixDeviceContext ctx,        
            OptixProgramGroup _prog_raygen, 
            OptixProgramGroup _prog_raymiss,
            OptixProgramGroup _prog_rayhit,
            const Scene& scene
        );
    
        virtual ~OptixTraceable() override;
    
        OptixTraversableHandle GetHandle() const { return _asHandle; }
    
        CUDABuffer& GetVertexBuffer() const {return *_vertBuffer;}
        CUDABuffer& GetIndexBuffer() const {return *_indBuffer;}
    
        const OptixShaderBindingTable& GetSBT() const {return _sbt.GetBindingTable(); }
    
    };

}

