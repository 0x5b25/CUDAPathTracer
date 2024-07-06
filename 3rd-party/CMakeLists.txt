
#GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("glfw")

#IMGUI


#Optix 8.0.0
#Create the folder and copy the Optix sdk into it 
add_library(OptiX80 INTERFACE)
target_include_directories(OptiX80 INTERFACE "optix_8_0_0/include")
