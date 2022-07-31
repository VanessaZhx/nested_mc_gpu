#pragma once
#ifndef BASKET_OPTION
#define BASKET_OPTION

#include "Stock.h"
#include <mkl.h>
#include<math.h>
#include <curand_kernel.h>

#include<iostream>

using namespace std;

class BasketOption {
public:

	BasketOption(int n, Stock* stocks, float* cov, float k, float* w) {
		this->n = n;
		this->stocks = stocks;
		this->k = k;
		this->w = w;

		//// Cholesky decompose by lapack
		// The result will be storeed in cov
		// However the upper part won't be set to 0 automatically
		LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', n, cov, n);

		// Copy the cholesky result to A
		A = (float*)malloc((size_t)n * n * sizeof(float));
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				A[i * n + j] = (i < j) ? 0 : cov[i * n + j];
			}
		}
		return;
	}

	~BasketOption() {
		free(A);
	}

	int n;				// Number of stocks in the basket
	
	float k;			// Option execute price(strike)
	float* w;			// Weight of each stock
	Stock* stocks;		// Underlying stocks
	float* A;			// Cholesky decomposistion of the covariance matrix
	//float* rn;			// Random number list

};


#endif // !BASKET_OPTION

