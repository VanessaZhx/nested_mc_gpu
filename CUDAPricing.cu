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

__global__ void price_bond(
	const float* rn,
	const int cnt,
	const float bond_par,
	const float bond_c,
	const int bond_m,
	const float bond_sig,
	const float* bond_y,
	float* prices){

	// Naive implementation
	// Each thread handle one outter, calculate the price and store it

	size_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= cnt) return;

	float d = rn[Idx] * bond_sig;
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
    float s00 = s0[0] * exp((mean[0] - 0.5f * std[0] * std[0]) * var_t
			+ std[0] * tmp1 * s0_rn[Idx]);
    float s01 = s0[1] * exp((mean[1] - 0.5f * std[1] * std[1]) * var_t
			+ std[1] * tmp1 * s0_rn[Idx]);
//for (int i = 0; i < stock_n; i++) {
//		s0_rn[Idx * stock_n ] = s0[i] * exp((mean[i] - 0.5f * std[i] * std[i]) * var_t
//			+ std[i] * tmp1 * s0_rn[Idx * stock_n + i]);y
//	}

	const float* rn_p = &rn[Idx * path_int * stock_n];
	float price = 0.0f;
	float call = 0.0f;
	const float tmp2 = sqrtf(float(bskop_t));

	for (int i = 0; i < path_int; i++) {
		//for (int j = 0; j < stock_n; j++) {
			price += stock_x[0] *w[0] * s00
				* exp((mean[0] - 0.5f * std[0] * std[0]) * bskop_t
					+ std[0] * tmp2 * rn_p[i * stock_n + 0]);
			price += stock_x[1] *w[1] * s01
				* exp((mean[1] - 0.5f * std[1] * std[1]) * bskop_t
					+ std[1] * tmp2 * rn_p[i * stock_n + 1]);
//std::printf("idx-%d\n", Idx);
			//std::printf("i-%d j-%d s0-%f rn-%f\n", i, j, s0_rn[Idx * stock_n + j], rn_p[i * stock_n + j]);
		//}
		call += (price > bskop_k) ? (price - bskop_k) : 0.0f;
		price = 0.0f;
	}
	prices[Idx] = call / path_int;
}

__global__ void price_bskop_sameRN(
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

	const float* rn_p = rn;
	float price = 0.0f;
	float call = 0.0f;
	const float tmp2 = sqrtf(float(bskop_t));

	for (int i = 0; i < path_int; i++) {
		for (int j = 0; j < stock_n; j++) {
			price += stock_x[j] * w[j] * s0_rn[Idx * stock_n + j]
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



// No early stop
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
) {
	// Each thread handle one outter, calculate the price and store it
	size_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= cnt) return;

	// reprice underlying stocks
	// will be used in inner as s0
	float stock_price = s0 * exp((mean - 0.5f * std * std) * var_t
		+ std * sqrtf(float(var_t)) * ext_rn[Idx]);
	float max_price = stock_price;

	// Inner loop
	float call = 0.0f;
	float barop_price = 0.0f;
	float tmp1 = mean - 0.5f * std * std;
	for (int j = 0; j < path_int; j++) {
		// Loop over steps in one path, get max price
		for (int k = 0; k < barop_t; k++) {
			// Calculate price at this step
			barop_price = stock_price * exp(tmp1 * k
				+ std * sqrtf(float(k)) * int_rn[Idx * path_int * barop_t + j * barop_t + k]);
			// Check maximum
			if (barop_price > max_price) {
				max_price = barop_price;
			}
		}
		//cout << endl<< barop_price << endl;

		// Compare with barrier, the option exists if max price is larger than barrier
		if (max_price > barop_h) {
			// barop_price will be the last price
			// max{St-K, 0}
			call += (barop_price > barop_k) ? barop_price - barop_k : 0;
		}
	}
	//cout << call << endl;

	// Get expected price at var_t
	prices[Idx] = x * (call / path_int);
}
	
// with early stop
__global__ void price_barrier_early(
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
) {
	// Each thread handle one outter, calculate the price and store it
	size_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= cnt) return;

	// reprice underlying stocks
	// will be used in inner as s0
	float stock_price = s0 * exp((mean - 0.5f * std * std) * var_t
		+ std * sqrtf(float(var_t)) * ext_rn[Idx]);

	// Inner loop
	bool acted = false;
	float call = 0.0f;
	float barop_price = 0.0f;
	float tmp1 = mean - 0.5f * std * std;
	for (int j = 0; j < path_int; j++) {
		acted = false;
		// Loop over steps in one path 
		for (int k = 0; k < barop_t; k++) {
			// Calculate price at this step
			barop_price = stock_price * exp(tmp1 * k
				+ std * sqrtf(float(k)) * int_rn[Idx * path_int * barop_t + j * barop_t + k]);

			// Check if exceed H, early stop
			if (barop_price > barop_h) {
				acted = true;
				break;
			}
		}
		//cout << endl<< barop_price << endl;
		if (acted) {
			// barop_price will be the last price
			barop_price = stock_price * exp(tmp1 * (barop_t-1) + std * 
				sqrtf(float((barop_t - 1))) * int_rn[Idx * path_int * barop_t +
												j * barop_t + barop_t - 1]);
			call += (barop_price > barop_k) ? barop_price - barop_k : 0;
		}
	}
	//cout << call << endl;

	// Get expected price at var_t
	prices[Idx] = x * (call / path_int);
}


__global__ void price_barrier_sameRN(
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
) {
	// Each thread handle one outter, calculate the price and store it
	size_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= cnt) return;

	// reprice underlying stocks
	// will be used in inner as s0
	float stock_price = s0 * exp((mean - 0.5f * std * std) * var_t
		+ std * sqrtf(float(var_t)) * ext_rn[Idx]);
	float max_price = stock_price;

	// Inner loop
	float call = 0.0f;
	float barop_price = 0.0f;
	float tmp1 = mean - 0.5f * std * std;
	for (int j = 0; j < path_int; j++) {
		// Loop over steps in one path, get max price
		for (int k = 0; k < barop_t; k++) {
			// Calculate price at this step
			barop_price = stock_price * exp(tmp1 * k
				+ std * sqrtf(float(k)) * int_rn[j * barop_t + k]);
			// Check maximum
			if (barop_price > max_price) {
				max_price = barop_price;
			}
		}
		//cout << endl<< barop_price << endl;

		// Compare with barrier, the option exists if max price is larger than barrier
		if (max_price > barop_h) {
			// barop_price will be the last price
			// max{St-K, 0}
			call += (barop_price > barop_k) ? barop_price - barop_k : 0;
		}
	}
	//cout << call << endl;

	// Get expected price at var_t
	prices[Idx] = x * (call / path_int);
}

__global__ void price_barrier_early_sameRN(
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
) {
	// Each thread handle one outter, calculate the price and store it
	size_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= cnt) return;

	// reprice underlying stocks
	// will be used in inner as s0
	float stock_price = s0 * exp((mean - 0.5f * std * std) * var_t
		+ std * sqrtf(float(var_t)) * ext_rn[Idx]);

	// Inner loop
	bool acted = false;
	float call = 0.0f;
	float barop_price = 0.0f;
	float tmp1 = mean - 0.5f * std * std;
	for (int j = 0; j < path_int; j++) {
		acted = false;
		// Loop over steps in one path 
		for (int k = 0; k < barop_t; k++) {
			// Calculate price at this step
			barop_price = stock_price * exp(tmp1 * k
				+ std * sqrtf(float(k)) * int_rn[j * barop_t + k]);

			// Check if exceed H, early stop
			if (barop_price > barop_h) {
				acted = true;
				break;
			}
		}
		//cout << endl<< barop_price << endl;
		if (acted) {
			// barop_price will be the last price
			barop_price = stock_price * exp(tmp1 * (barop_t - 1) + std *
				sqrtf(float((barop_t - 1))) * int_rn[j * barop_t + barop_t - 1]);
			call += (barop_price > barop_k) ? barop_price - barop_k : 0;
		}
	}
	//cout << call << endl;

	// Get expected price at var_t
	prices[Idx] = x * (call / path_int);
}
