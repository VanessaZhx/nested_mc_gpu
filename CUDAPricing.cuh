#pragma once
#ifndef CUDAPRICING_H
#define CUDAPRICING_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <curand.h>

#define DATA_BLOCK 32

__global__ void moro_inv(float* data, int cnt, float mean, float std);
__global__ void moro_inv_v2(float* data, int cnt, float mean, float std);
__global__ void price_bond(float* rn, int cnt,
	float bond_par, float bond_c, int bond_m, float* bond_y,
	float* prices);
__global__ void price_stock(float* rn, int cnt,
	float s0, float mean, float std, int x, int t,
	float* prices);

#endif