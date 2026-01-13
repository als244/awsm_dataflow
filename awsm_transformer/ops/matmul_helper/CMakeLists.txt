cmake_minimum_required(VERSION 3.18)

project(matmul_dispatcher_lib LANGUAGES CXX CUDA)

# --- Configuration ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags (Release build default)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# --- Dependencies ---
# Find the CUDA Toolkit (handles paths for cublasLt, cudart, etc.)
find_package(CUDAToolkit REQUIRED)

# --- Build Target ---
add_library(matmul_dispatcher SHARED src/dispatch.cpp)

target_link_libraries(matmul_dispatcher PRIVATE 
    CUDA::cublasLt 
    CUDA::cudart
)

# --- Output ---
set_target_properties(matmul_dispatcher PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/matmul_dispatcher"
    PREFIX "lib" 
)