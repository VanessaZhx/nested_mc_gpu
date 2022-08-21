# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)
# 允许gdb debug
#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# 项目名称
project (proj)
# cmake寻找cuda，并且CUDA_ADD_EXECUTABLE等cuda前缀的命令是由cuda提供的，cmake本身不支持
FIND_PACKAGE(CUDA REQUIRED)
if (NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)
# C++和CUDA的编译参数，可选。如果CMAKE_CXX_FLAGS和CUDA_NVCC_FLAGS有相同参数会报错
# 61表示这台机器显卡的算力为6.1（GTX1050）
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;)
# 查找目录下的所有头、源文件
# 并将名称保存到CURRENT_HEADERS、CURRENT_SOURCES变量
file(GLOB CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu *.cc)
# include头文件目录
include_directories(
    /vol/cuda/11.1.0-cudnn8.0.4.30/include
    /vol/intel/oneapi/mkl/latest
)
# 调用nvcc进行编译，指定生成目标output
CUDA_ADD_EXECUTABLE(
    output ${CURRENT_HEADERS} ${CURRENT_SOURCES}
    OPTIONS -arch=sm_61
    )
# 允许output链接静态库
set_property(TARGET output PROPERTY ENABLE_EXPORTS true)
# 链接CMake目录下的静态库
target_link_libraries(
    output
    blas
    mkl_rt
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    )

