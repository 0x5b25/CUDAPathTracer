

# Generate a custom build rule to translate *.cu files to *.ptx or *.optixir files.
# NVCUDA_COMPILE_MODULE(
#   SOURCES file1.cu file2.cu
#   DEPENDENCIES header1.h header2.h
#   TARGET_PATH <path where output files should be stored>
#   EXTENSION ".ptx" | ".optixir"
#   GENERATED_FILES program_modules
#   NVCC_OPTIONS -arch=sm_50
# )

# Generates *.ptx or *.optixir files for the given source files.
# The program_modules argument will receive the list of generated files.
# DAR Using this because I do not want filenames like "cuda_compile_ptx_generated_raygeneration.cu.ptx" but just "raygeneration.ptx".

FUNCTION(NVCUDA_COMPILE_MODULE)
  if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "ERROR: Only 64-bit programs supported.")
  endif()

  set(options "")
  set(oneValueArgs TARGET_PATH GENERATED_FILES)
  set(multiValueArgs NVCC_OPTIONS SOURCES DEPENDENCIES)

  CMAKE_PARSE_ARGUMENTS(NVCUDA_COMPILE_MODULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if (NOT WIN32) # Do not create a folder with the name ${ConfigurationName} under Windows.
    # Under Linux make sure the target directory exists. 
    FILE(MAKE_DIRECTORY ${NVCUDA_COMPILE_MODULE_TARGET_PATH})
  endif()
  
  # Custom build rule to generate either *.ptx or *.optixir files from *.cu files.
  FOREACH(input ${NVCUDA_COMPILE_MODULE_SOURCES})
    get_filename_component(input_we "${input}" NAME_WE)
    get_filename_component(ABS_PATH "${input}" ABSOLUTE)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" REL_PATH "${ABS_PATH}")

    # Generate the output *.ptx or *.optixir files directly into the executable's selected target directory.
    set(ptx_file "${NVCUDA_COMPILE_MODULE_TARGET_PATH}/${input_we}.ptx")
    set(embedded_file "${NVCUDA_COMPILE_MODULE_TARGET_PATH}/${input_we}.compiled.h")
    # message("output = ${output}")

    LIST(APPEND OUTPUT_FILES "${embedded_file}")
    
    # This prints the standalone NVCC command line for each CUDA file.
    # CUDAToolkit_NVCC_EXECUTABLE has been set with FindCUDAToolkit.cmake in CMake 3.17 and newer.
    # message("${CUDAToolkit_NVCC_EXECUTABLE} " "${NVCUDA_COMPILE_MODULE_NVCC_OPTIONS} " "${input} " "-o " "${output}")    

    add_custom_command(
      OUTPUT  ${ptx_file} ${embedded_file}
      DEPENDS "${input}" ${NVCUDA_COMPILE_MODULE_DEPENDENCIES}
      COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} ${NVCUDA_COMPILE_MODULE_NVCC_OPTIONS} --ptx "${input}" -o "${ptx_file}"
      COMMAND ${CUDAToolkit_BIN_DIR}/bin2c -c --padd 0 --type char --name ${input_we} ${ptx_file} > ${embedded_file}
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  ENDFOREACH( )

  set(${NVCUDA_COMPILE_MODULE_GENERATED_FILES} ${OUTPUT_FILES} PARENT_SCOPE)
ENDFUNCTION()



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

set(CUDATRACER_CUDAPROG_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/CUDATracerProg)
set(CUDATRACER_OPTIXPROG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/OptixTracerProg)

set(CUDATRACER_CUDAPROG_SRCS
    "${CUDATRACER_CUDAPROG_DIR}/CUDATracerProg.cu"
)

set(CUDATRACER_OPTIXPROG_SRCS
    "${CUDATRACER_OPTIXPROG_DIR}/OptixTracerProg.cpp"

)

set(CUDATRACER_OPTIXPROG_SHADER_SRCS
    "${CUDATRACER_OPTIXPROG_DIR}/Shaders/DefaultShader.cu"
)

set(CUDATRACER_OPTIXPROG_SHADER_HEADERS
    "${CUDATRACER_OPTIXPROG_DIR}/Shaders/ShaderTypes.h"
)


# When using OptiX SDK 7.5.0 and CUDA 11.7 or higher, the modules can either be built from OptiX IR input or from PTX input.
# OPTIX_PROGRAM_TARGET and OPTIX_MODULE_EXTENSION switch the NVCC compilation between the two options.
NVCUDA_COMPILE_MODULE(
  SOURCES ${CUDATRACER_OPTIXPROG_SHADER_SRCS}
  DEPENDENCIES ${CUDATRACER_OPTIXPROG_SHADER_HEADERS}
  TARGET_PATH ${CMAKE_CURRENT_BINARY_DIR}/Shaders
  #EXTENSION ".ptx"#"${OPTIX_MODULE_EXTENSION}"
  GENERATED_FILES PROGRAM_MODULES
  NVCC_OPTIONS "${OPTIX_PROGRAM_TARGET}" "--machine=64"
                                         "--gpu-architecture=compute_50"
                                         "--use_fast_math"
                                         "--relocatable-device-code=true"
                                         "--generate-line-info"
                                         "-Wno-deprecated-gpu-targets"
                                         "--allow-unsupported-compiler"
                                         "-I${OPTIX_INCLUDE_DIR}"
                                         "-I${CMAKE_CURRENT_SOURCE_DIR}/OptixTraceProg/Shaders"
                                         "-I${CMAKE_CURRENT_SOURCE_DIR}/Math"
                                         "-I${CMAKE_CURRENT_SOURCE_DIR}/../Common"
                                         "-I${CMAKE_CURRENT_SOURCE_DIR}/../3rd-party/optix_8_0_0/include"
)

target_sources(CUDATracer
       #"CUDACommon.h" 
       #      "Math/Constants.h"
       #      "Math/Matrix.hpp"
       #      "Math/Quaternion.hpp"
       #      "CUDABuffer.hpp"
       #      "Types.hpp"
       #      "CUDATracer.hpp"
      PRIVATE "CUDABuffer.cpp"  
             #"CUDATracer.cu"    
             "CUDATracer.cpp"    
             "CUDACommon.cpp"  
             #"SplitBVHBuilder.cpp"
             ${CUDATRACER_CUDAPROG_SRCS}
             ${CUDATRACER_OPTIXPROG_SRCS}
             ${PROGRAM_MODULES}

)

target_include_directories(CUDATracer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
           ${CMAKE_CURRENT_SOURCE_DIR}/Math
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_features(CUDATracer PUBLIC cxx_std_17)
#target_compile_features(CUDATracer PUBLIC cuda_std_17)

set_target_properties(CUDATracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(CUDATracer PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -gencode arch=compute_75,code=sm_75")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

target_link_libraries(CUDATracer 
    CUDA::cudart_static
    CUDA::cuda_driver
    OptiX80
    common
)

#add_dependencies(CUDATracer )

#if(CMAKE_BUILD_TYPE EQUAL "Debug")
#    target_compile_options(CUDATracer PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)
#endif()


