#pragma once

#include "CUDATracer.hpp"

#include "OptixAPI.hpp"
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

public:
    OptixTraceable(const OptixAPI& api, OptixDeviceContext ctx, const Scene& scene);
    virtual ~OptixTraceable() override;

    OptixTraversableHandle GetHandle() const { return _asHandle; }

};

}

