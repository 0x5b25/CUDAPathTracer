#pragma once

#include "Vec.hpp"

namespace CUDATracer{

struct LaunchParams
{
    int       frameID { 0 };
    uint32_t *colorBuffer;
    Math::vec2i     fbSize;
};

}