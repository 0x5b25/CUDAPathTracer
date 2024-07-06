#pragma once

#include "Vec.hpp"

namespace CUDATracer{

    enum {RT_TRACE_DEPTH = 4};

    struct LaunchParams
    {
        int       frameID { 0 };
        char      *colorBuffer;
        Math::vec2i     fbSize;
    };

}