
add_library(common)


find_package(OpenGL REQUIRED)

target_include_directories(common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(common 
    PUBLIC glfw
           ${GLFW_LIBRARIES}
           #${OPENGL_gl_LIBRARY}
           OpenGL::GL
           CUDATracer
           )

target_sources(common
    PRIVATE "common.cpp"
 "GLFWWindow.hpp" "GLFWWindow.cpp" "rgbe.h" "rgbe.c" "RendererWindow.hpp")
