

add_library(CUDATracer
       #"CUDACommon.h" 
       #      "Math/Constants.h"
       #      "Math/Matrix.hpp"
       #      "Math/Quaternion.hpp"
       #      "CUDABuffer.hpp"
       #      "Types.hpp"
       #      "CUDATracer.hpp"
      #"CUDABuffer.cpp"  
      #       "CUDATracer.cu"    
      #       "CUDACommon.cpp"  
             )

find_package(CUDAToolkit REQUIRED)

target_sources(CUDATracer
       #"CUDACommon.h" 
       #      "Math/Constants.h"
       #      "Math/Matrix.hpp"
       #      "Math/Quaternion.hpp"
       #      "CUDABuffer.hpp"
       #      "Types.hpp"
       #      "CUDATracer.hpp"
      PRIVATE "CUDABuffer.cpp"  
             "CUDATracer.cu"    
             "CUDACommon.cpp"  
             #"SplitBVHBuilder.cpp"

)

target_include_directories(CUDATracer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(CUDATracer PUBLIC cxx_std_14)
#target_compile_features(CUDATracer PUBLIC cuda_std_17)

set_target_properties(CUDATracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(CUDATracer PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -gencode arch=compute_75,code=sm_75")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

target_link_libraries(CUDATracer 
    CUDA::cudart_static
    CUDA::cuda_driver
    common
)

#if(CMAKE_BUILD_TYPE EQUAL "Debug")
#    target_compile_options(CUDATracer PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)
#endif()


