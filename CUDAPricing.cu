#include "CUDAPricing.cuh"

__global__ void moro_inv(float* data, int cnt, float mean, float std) {
	// Each thread will handle one transfer
	size_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= cnt) return;

	data[Idx] = normcdfinvf(data[Idx]) * std + mean;
}

__global__ void moro_inv_v2(float* data, int cnt, float mean, float std) {
	// Each thread will handle DATA_BLOCK transfer
	size_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= cnt) return;

	float* data_p = &data[Idx * DATA_BLOCK];

	for (int i = 0; i < DATA_BLOCK; i++) {
		data_p[i] = normcdfinvf(data_p[i]) * std + mean;
	}

}

__global__ void price_bond(float* rn, int cnt,
	float bond_par, float bond_c, int bond_m, float* bond_y,
	float* prices) {

	// Naive implementation
	// Each thread handle one outter, calculate the price and store it

	size_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= cnt) return;

	float d = rn[Idx];
	float price = 0.0f;
	// Loop to sum the coupon price until the maturity
	for (int i = 0; i < bond_m; i++) {
		price += bond_c /
			powf((float)(1.0f + (bond_y[i] + d) / 100.0f), i + 1);
	}
	// Add the face value
	price += bond_par /
		powf((float)(1.0f + (bond_y[bond_m - 1] + d) / 100.0f), bond_m);

	// Store in prices matrix
	prices[Idx] = price;
}

__global__ void price_stock(
	const float* rn,
	const int cnt,
	const float s0, 
	const float mean, 
	const float std, 
	const int x,
	const int t,
	float* prices
) {

	// Naive implementation
	// Each thread handle one outter, calculate the price and store it

	size_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= cnt) return;

	const float tmp1 = x * s0 ;
	const float tmp2 = (mean - 0.5f * std * std) * t;
	const float tmp3 = std * sqrtf(float(t));

	// Store in prices matrix
	prices[Idx] =  tmp1 * exp(tmp2 + tmp3 * rn[Idx]);
}

// TODO: stock info is commonly used, can use shared memory to reduce time
// No reverse
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
	const float *w,
	float* prices
) {

	// Each thread handle one outter, calculate the price and store it
	size_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= cnt) return;


	// Inner loop
	// The random number has already been transformed with cov
	// Calculate each value with the correlated-random numbers
	// random numbers[path_int + 1][n]
	// First part is for s0
	const float tmp1 = sqrtf(float(var_t));
	// Get start value for each asset, save in s0_rn array
	for (int i = 0; i < stock_n; i++) {
		s0_rn[Idx * stock_n + i] = s0[i] * exp((mean[i] - 0.5f * std[i] * std[i]) * var_t
			+ std[i] * tmp1 * s0_rn[Idx * stock_n + i]);
	}

	const float* rn_p = &rn[Idx * path_int * stock_n];
	float price = 0.0f;
	float call = 0.0f;
	const float tmp2 = sqrtf(float(bskop_t));

	for (int i = 0; i < path_int; i++) {
		for (int j = 0; j < stock_n; j++) {
			price += stock_x[j] *w[j] * s0_rn[Idx * stock_n + j]
				* exp((mean[j] - 0.5f * std[j] * std[j]) * bskop_t
					+ std[j] * tmp2 * rn_p[i * stock_n + j]);
			//std::printf("idx-%d\n", Idx);
			//std::printf("i-%d j-%d s0-%f rn-%f\n", i, j, s0_rn[Idx * stock_n + j], rn_p[i * stock_n + j]);
		}
		call += (price > bskop_k) ? (price - bskop_k) : 0.0f;
		price = 0.0f;
	}
	prices[Idx] = call / path_int;
}


// Can't use : Require too much space to start up
// value weighted is too large to store(will be contented)
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
	const float* x,
	const int bskop_t,
	const float bskop_k,
	const float* w,
	float* value_weighted,
	float* prices
) {

	// Each thread handle one outter, calculate the price and store it
	size_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= cnt) return;


	// Inner loop
	// The random number has already been transformed with cov
	// Calculate each value with the correlated-random numbers
	// random numbers[n][path_int]

	const float* rn_p = &rn[Idx * path_int * stock_n];
	float* value_p = &value_weighted[Idx * path_int];
	// reverse
	for (int i = 0; i < stock_n; i++) {
		float tmp1 = (mean[i] - 0.5f * std[i] * std[i]) * var_t;
		float tmp2 = std[i] * sqrtf(float(var_t));
	
		// Get start value for each asset
		// To reduce calculation, pre multiply weight and shares
		// tmp3 = weight[i] * x[i] * start_price[i]
		float tmp3 = w[i] * x[i] * (s0[i] * exp(tmp1 +  tmp2 * s0_rn[Idx * stock_n + i]));
	
		
		// random numbers[n][path_int]
		
		for (int j = 0; j < path_int; j++) {
			//printf("s0-%f \n", tmp3);
			value_p[j] += tmp3 * exp(tmp1 + tmp2 * rn_p[j * stock_n + i]);
			std::printf("i-%d j-%d rn-%f\n", i, j, value_p[j]);
		}
	}
	
	// Store in prices matrix
	float call = 0.0f;
	for (int i = 0; i < path_int; i++) {
		std::printf("- %d %f\n", Idx, value_weighted[i]);
		call += (value_p[i] > bskop_k) ? (value_weighted[i] - bskop_k) : 0;
	}
	
	prices[Idx] = call / path_int;
}

	