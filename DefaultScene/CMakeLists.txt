

add_executable(DefaultScene 
    tracer.cpp
)

target_include_directories(DefaultScene PUBLIC ${PROJECT_BINARY_DIR})

target_link_libraries(DefaultScene 
    PUBLIC glfw
           ${OPENGL_gl_LIBRARY}
           CUDATracer
           common
           )

target_compile_features(DefaultScene PUBLIC cxx_std_14)

#copy default skydome
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/sky.hdr"
        "${CMAKE_CURRENT_BINARY_DIR}/sky.hdr" COPYONLY)
