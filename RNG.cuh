#pragma once
#ifndef RNG_H
#define RNG_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <curand.h>

#include "cuda_helper.cuh"


class RNG{
public:
    // m - simulation times
    // n - dimention
    ~RNG();
    int RNG::init();
    void set_offset(int ofs) { offset = ofs; }
    int generate_sobol_cpu(float*& data, int m, int n);
    int convert_normal(float*& data, int length, float sigma = 1);
private:
    curandGenerator_t gen;
    int offset = 1024;
};

#endif

