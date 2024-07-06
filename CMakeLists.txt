cmake_minimum_required(VERSION 3.18)

project(CUDAPathTracer
    VERSION 0.1
    LANGUAGES CUDA CXX C)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#Enable cuda kernel debugging for debug builds

add_subdirectory("3rd-party")
add_subdirectory("CUDATracer")
add_subdirectory("Common")

add_subdirectory("DefaultScene")
add_subdirectory("CornellBox")
