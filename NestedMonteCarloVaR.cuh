#pragma once
#ifndef NMC_H
#define NMC_H

#include "cublas_v2.h"

#include "RNG.cuh"
#include "Bond.h"
#include "Stock.h"
#include "BasketOption.h"
#include "BarrierOption.h"

#include "CUDAPricing.cuh"

#include <mkl.h>
#include <math.h>
#include <algorithm>
#include <chrono>

#include <iostream>
#include <fstream>


using namespace std;

class NestedMonteCarloVaR {
public:
	NestedMonteCarloVaR(int pext, int pint,
		int t, float per, int port_n, float* weight, float risk_free);

	void bond_init(float bond_par, float bond_c, int bond_m,
		float* bond_y, float sig, int idx);

	void stock_init(float stock_s0, float stock_mu, float stock_var,
		int stock_x, int idx);

	void bskop_init(int bskop_n, Stock* bskop_stocks, float* bskop_cov,
		float bskop_k, float* bskop_w, int bskop_t, int idx);

	void barop_int(Stock* barop_stock, float barop_k, float barop_h, int barop_t, int idx);

	double execute();

	void output_res(float* data, int len);

private:
	int path_ext = 0;  // Number of the outer MC loops
	int path_int = 0;  // Number of the inner MC loops

	int var_t = 0;			// VaR duration
	float var_per = 0;		// 1-percentile

	int port_n = 0;			// Number of products in the portfolio
	float* port_w = 0;		// Weights of the products in the portfolio
	float port_p0 = 0;		// Today's price of the portfolio

	float risk_free = 0;	// Risk free rate

	Bond* bond = NULL;
	float* bond_rn = NULL;	   // Pointer to the bond's RN sequence

	Stock* stock = NULL;
	float* stock_rn = NULL;		// Pointer to the stock's RN sequence

	BasketOption* bskop = NULL;
	int bskop_t = 0;					// Maturity of option
	float* bskop_rn = NULL;		// Pointer to the basket option's RN sequence

	BarrierOption* barop = NULL;
	int barop_t = 0;			// Maturity of option
	float* barop_rn = NULL;		// Pointer to barrier option's RN sequence


	float* prices = NULL; // Pointer to the matrix of each of the product's prices
	RNG* rng;				// Random number generator
};


#endif // !NMC_H

