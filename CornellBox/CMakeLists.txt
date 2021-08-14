

add_executable(CornellBox 
    tracer.cpp
)

target_include_directories(CornellBox PUBLIC ${PROJECT_BINARY_DIR})

target_link_libraries(CornellBox 
    PUBLIC glfw
           ${OPENGL_gl_LIBRARY}
           CUDATracer
           common
           )

target_compile_features(CornellBox PUBLIC cxx_std_14)

