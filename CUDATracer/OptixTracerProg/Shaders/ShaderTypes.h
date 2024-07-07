#pragma once

#include "Types.hpp"
#include "optix.h"
#include "Math/Vec.hpp"

namespace CUDATracer{

    enum {RT_TRACE_DEPTH = 4};

    struct LaunchParams
    {
        struct {
            char* colorBuffer;
            float* accBuffer;
            Math::vec2i     size;
            unsigned frameID;
        } frame;

        Camera camera;

        OptixTraversableHandle traversable;
    };

}