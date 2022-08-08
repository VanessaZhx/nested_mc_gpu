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

__global__ void price_stock(float* rn, int cnt,
	float s0, float mean, float std, int x, int t,
	float* prices) {

	// Naive implementation
	// Each thread handle one outter, calculate the price and store it

	size_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (Idx >= cnt) return;

	// Store in prices matrix
	prices[Idx] = x * s0 * exp((mean - 0.5f * std * std) * t
		+ std * sqrtf(float(t)) * rn[Idx]);


}