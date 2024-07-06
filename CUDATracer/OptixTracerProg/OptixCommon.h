#pragma once

#include "CUDACommon.h"


#define OPTIX_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    const OptixResult stat = condition; \
    if(stat != OPTIX_SUCCESS) {\
      CHECK_EQ(stat, OPTIX_SUCCESS) << " failed with (" << stat << ")\n"; \
    }\
  } while (0)

