# CMake Minimum version
cmake_minimum_required (VERSION 2.8)
# Allow gdb debug
#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# Name
project (proj)
# cmake find cuda
FIND_PACKAGE(CUDA REQUIRED)
if (NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)
# Compute capability
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;)

file(GLOB CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu *.cc)

include_directories(
    /vol/cuda/11.1.0-cudnn8.0.4.30/include
    /vol/intel/oneapi/mkl/latest
)

CUDA_ADD_EXECUTABLE(
    output ${CURRENT_HEADERS} ${CURRENT_SOURCES}
    OPTIONS -arch=sm_61
    )

set_property(TARGET output PROPERTY ENABLE_EXPORTS true)

target_link_libraries(
    output
    blas
    mkl_rt
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    )

