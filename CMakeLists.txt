cmake_minimum_required(VERSION 3.18)

project(CUDAPathTracer
    VERSION 0.1
    LANGUAGES CXX CUDA)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#Enable cuda kernel debugging for debug builds

add_subdirectory("3rd-party/glfw")
add_subdirectory("CUDATracer")
add_subdirectory("Common")

add_executable(SimpleTracer 
    tracer.cpp
    rgbe.c
)

target_include_directories(SimpleTracer PUBLIC ${PROJECT_BINARY_DIR})

target_link_libraries(SimpleTracer 
    PUBLIC glfw
           ${OPENGL_gl_LIBRARY}
           CUDATracer
           common
           )

target_compile_features(SimpleTracer PUBLIC cxx_std_14)

#copy default skydome
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/sky.hdr"
        "${CMAKE_CURRENT_BINARY_DIR}/sky.hdr" COPYONLY)
