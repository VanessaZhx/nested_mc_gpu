#pragma once
#ifndef CUDAPRICING_H
#define CUDAPRICING_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <curand.h>

#include <cstdio>

#define DATA_BLOCK 32

__global__ void moro_inv(float* data, int cnt, float mean, float std);
__global__ void moro_inv_v2(float* data, int cnt, float mean, float std);
__global__ void price_bond(float* rn, int cnt,
	float bond_par, float bond_c, int bond_m, float* bond_y,
	float* prices);
__global__ void price_stock(
	const float* rn,
	const int cnt,
	const float s0,
	const float mean,
	const float std,
	const int x,
	const int t,
	float* prices
);

__global__ void price_bskop(
	const int cnt,
	const int path_int,
	const float* rn,
	float* s0_rn,
	const int stock_n,
	const int var_t,
	const float* s0,
	const float* mean,
	const float* std,
	const float* stock_x,
	const int bskop_t,
	const float bskop_k,
	const float* w,
	float* prices
);

__global__ void price_bskop_reverse(
	const int cnt,
	const int path_int,
	const float* rn,
	float* s0_rn,
	const int stock_n,
	const int var_t,
	const float* s0,
	const float* mean,
	const float* std,
	const float* stock_x,
	const int bskop_t,
	const float bskop_k,
	const float* w,
	float* value_weighted,
	float* prices
);

__global__ void price_barrier(
	const float* ext_rn,
	const float* int_rn,
	const int cnt,
	const int path_int,
	const float s0,
	const float mean,
	const float std,
	const int x,
	const int var_t,
	const int barop_t,
	const int barop_h,
	const int barop_k,
	float* prices
);

#endif