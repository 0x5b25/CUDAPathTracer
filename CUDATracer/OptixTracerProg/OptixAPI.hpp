#pragma once

#include "optix_function_table.h"

namespace CUDATracer
{
    class OptixAPI {
        
        OptixFunctionTable _api;
    public:

        OptixAPI();

        const OptixFunctionTable& Get() const {return _api;}

    };

} // namespace CUDATracer

