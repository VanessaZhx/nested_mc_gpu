#include "NestedMonteCarloVaR.cuh"
#define DATA_BLOCK 32

using namespace std;

NestedMonteCarloVaR::NestedMonteCarloVaR(int pext, int pint,
	int t, float per, int port_n, float* weight, float risk_free) {
	this->path_ext = pext;
	this->path_int = pint;
	this->var_t = t;
	this->var_per = per;
	this->port_n = port_n;
	this->port_w = weight;
	this->risk_free = risk_free;

	rng = new RNG;
	rng->init_gpu();
	rng->set_offset(1024);
}

void NestedMonteCarloVaR::bond_init(float bond_par, float bond_c, int bond_m,
	float* bond_y, float sig, int idx) {

	// Product initiation
	bond = new Bond(bond_par, bond_c, bond_m, bond_y, sig);

	// Add start price to the portfolio start price
	// Start price for bond is priced with original yield curve
	float price = 0.0f;
	for (int i = 0; i < bond_m; i++) {
		price += bond_c / powf(1.0f + (bond_y[i]) / 100, float(i + 1));
	}
	price += bond_par / powf(1.0f + (bond_y[bond_m - 1]) / 100, float(bond_m));

	this->port_p0 += price * port_w[idx];
}

void NestedMonteCarloVaR::stock_init(float stock_s0, float stock_mu,
	float stock_var, int stock_x, int idx) {

	// Product initiation
	stock = new Stock(stock_s0, stock_mu, stock_var, stock_x);

	// add to the portfolio price
	this->port_p0 += stock_s0 * stock_x * port_w[idx];
}

void NestedMonteCarloVaR::bskop_init(int bskop_n, Stock* bskop_stocks,
	float* bskop_cov, float bskop_k, float* bskop_w, int bskop_t, int idx) {

	// Product initiation
	bskop = new BasketOption(bskop_n, bskop_stocks, bskop_cov, bskop_k, bskop_w);

	this->bskop_t = bskop_t;

	// Simulate start price
	float* rn = (float*)malloc((size_t)path_int * bskop_n * sizeof(float));
	float* tmp_rn = (float*)malloc((size_t)path_int * bskop_n * sizeof(float));
	rng->generate_sobol(tmp_rn, bskop_n, path_int);
	rng->convert_normal(tmp_rn, path_int * bskop_n);
	cblas_sgemm(CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		bskop_n,								// result row
		path_int,								// result col
		bskop_n,								// length of "multiple by"
		1,										// alpha
		bskop->A,								// A
		bskop_n,								// col of A
		tmp_rn,									// B
		bskop_n,								// col of B
		0,										// beta
		rn,										// C
		path_int								// col of C
	);

	// Price the start value
	float* value_each = (float*)malloc((size_t)path_int * bskop_n * sizeof(float));
	float* value_weighted = (float*)malloc((size_t)path_int * sizeof(float));
	Stock* s;

	for (int i = 0; i < bskop_n; i++) {
		s = &(bskop_stocks[i]);
		for (int j = 0; j < path_int; j++) {
			value_each[i * path_int + j] = s->x * s->s0
				* exp((s->mu - 0.5f * s->var * s->var) * bskop_t
					+ s->var * sqrtf(float(bskop_t)) * rn[i * path_int + j]);
		}
	}

	cblas_sgemv(CblasRowMajor,				// Specifies row-major
		CblasTrans,							// Specifies whether to transpose matrix A.
		bskop_n,							// A rows
		path_int,							// A col
		1,									// alpha	
		value_each,							// A
		path_int,							// The size of the first dimension of matrix A.
		bskop_w,							// Vector X.
		1,									// Stride within X. 
		0,									// beta
		value_weighted,						// Vector Y
		1									// Stride within Y
	);

	float call = 0.0f;
	for (int i = 0; i < path_int; i++) {
		call += (value_weighted[i] > bskop_k) ? (value_weighted[i] - bskop_k) : 0;
	}
	call /= path_int;

	free(value_each);
	free(value_weighted);
	free(tmp_rn);
	free(rn);

	// add to the portfolio price
	this->port_p0 += call * port_w[idx];
}


void NestedMonteCarloVaR::barop_int(Stock* barop_stock, float barop_k,
	float barop_h, int barop_t, int idx) {
	// Product initiation
	barop = new BarrierOption(barop_stock, barop_k, barop_h);

	this->barop_t = barop_t;

	// Simulate start price
	// Need one round of inner loop to price start value
	float* temp_rn = (float*)malloc((size_t)path_int * barop_t * sizeof(float));
	rng->generate_sobol(temp_rn, barop_t, path_int);
	rng->convert_normal(temp_rn, path_int * barop_t);

	Stock* s = barop_stock;
	float barop_max_price = 0.0f;		// Max price throughout the path
	float barop_price = 0.0f;			// option price at one step
	float call = 0.0f;					// Accumulated price for a inner path

	for (int j = 0; j < path_int; j++) {
		// Loop over steps in one path, get max price
		for (int k = 0; k < barop_t; k++) {
			// Calculate price at this step
			barop_price = s->s0 * exp((s->mu - 0.5f * s->var * s->var) * k
				+ s->var * sqrtf(float(k)) * temp_rn[j * barop_t + k]);

			// Check maximum
			if (barop_price > barop_max_price) {
				barop_max_price = barop_price;
			}
		}

		// Compare with barrier, the option exists if max price is larger than barrier
		if (barop_max_price > barop->h) {
			// barop_price will be the last price
			// max{St-K, 0}
			call += (barop_price > barop->k) ? barop_price - barop->k : 0;
		}
	}
	free(temp_rn);


	// Add start price to the portfolio start price
	this->port_p0 += s->x * (call / path_int) * port_w[idx];
}

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


double NestedMonteCarloVaR::execute() {
	chrono::steady_clock::time_point start = chrono::steady_clock::now();

	// ====================================================
	//             Random number preperation
	// ====================================================
	int blocksize = 1024;
	dim3 block(blocksize, 1);

	/* == BOND ==
	** RN is used to move yield curve up/down, N~(0, sigma^2)
	** [path_ext, 1]
	*/
	CUDA_CALL(cudaMalloc((void**)&bond_rn, path_ext * sizeof(float)));
	rng->generate_sobol(bond_rn, 1, path_ext);
	dim3 grid_v1((path_ext - 1) / block.x + 1, 1);
	moro_inv << < grid_v1, block >> > (bond_rn, path_ext, 0, bond->sigma);

	//dim3 grid_v2((path_ext - 1) / (block.x * DATA_BLOCK) + 1, 1);
	//moro_inv_v2 << < grid_v2, block >> > (bond_rn, path_ext, 0, bond->sigma);
	//rng->generate_sobol_normal(bond_rn, 1, path_ext, bond->sigma);

	////* == STOCK ==
	//** RN is used as the external path
	//** For stock pricing only, there's no need to generate the inner path now
	//** [path_ext, var_t]
	//*/
	//CUDA_CALL(cudaMalloc((void**)&stock_rn, path_ext * sizeof(float)));
	//rng->generate_sobol(stock_rn, var_t, path_ext);
	//rng->convert_normal(stock_rn, var_t * path_ext);
//
//	/* == Basket Option ==
//	** Need two set of random numbers
//	** First: to reprice underlying stocks at H(use as S0 in inner)
//	** [path_ext, n]
//	** Second: inner loop, to reprice option
//	** [path_ext, [path_int, n]] => [path_ext * path_int, n]
//	*/
//	int bsk_n = bskop->n;
//	// Random number sequence for basket option(outer loop)
//	float* bskop_ext_rn;
//	CUDA_CALL(cudaMalloc((void**)&bskop_ext_rn, path_ext * bsk_n * sizeof(float)));
//	rng->generate_sobol(bskop_ext_rn, bsk_n, path_ext);
//	rng->convert_normal(bskop_ext_rn, path_ext * bsk_n);
//
//	// Random number sequence for basket option(inner loop)
//	//bskop_rn = (float*)malloc((size_t)path_ext * path_int * bsk_n * sizeof(float));
//	CUDA_CALL(cudaMalloc((void**)&bskop_rn, path_ext * path_ext * bsk_n * sizeof(float)));
//	//float* bskop_tmp_rn = (float*)malloc((size_t)path_ext * path_int * bsk_n * sizeof(float));
//	float* bskop_tmp_rn;
//	CUDA_CALL(cudaMalloc((void**)&bskop_tmp_rn, path_ext * path_ext * bsk_n * sizeof(float)));
//	rng->generate_sobol(bskop_tmp_rn, bsk_n, path_ext * path_int);
//	rng->convert_normal(bskop_tmp_rn, path_ext * path_int * bsk_n);
//
//	// Covariance transformation
//	// A[n * n]*rn[n * (path_ext * path_int)]
//	cublasHandle_t handle;
//	CUBLAS_CALL(cublasCreate(&handle));
//	/*		
////(m*n) =((n*n)*(n*m))^T
//	CUBLAS_CALL(cublasSgemm(handle,		//handle to the cuBLAS library context.
//		CUBLAS_OP_T,					//operation op(A) that is non- or (conj.) transpose.
//		CUBLAS_OP_T,					//operation op(B) that is non- or (conj.) transpose.
//		N,								//number of rows of matrix op(A) and C.
//		M,								//number of columns of matrix op(B) and C.
//		N,								//number of columns of op(A) and rows of op(B).
//		&one,							//<type> scalar used for multiplication.
//		Q_dev,							//<type> array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise
//		N,								//leading dimension of two-dimensional array used to store the matrix A.
//		x_dev,							//<type> array of dimension ldb x n with ldb>=max(1,k) if transb == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
//		M,								//leading dimension of two-dimensional array used to store matrix B.
//		&zero,							//<type> scalar used for multiplication. If beta==0, C does not have to be a valid input.
//		Y_dev,							//<type> array of dimensions ldc x n with ldc>=max(1,m).
//		N));							//	leading dimension of a two - dimensional array used to store the matrix C.
//
//	cublasSgemm(
//		handle,
//
//		)
//
//	cblas_sgemm(CblasRowMajor,
//		CblasNoTrans,
//		CblasTrans,
//		bsk_n,								// result row
//		path_ext * path_int,				// result col
//		bsk_n,								// length of "multiple by"
//		1,									// alpha
//		bskop->A,							// A
//		bsk_n,								// col of A
//		bskop_tmp_rn,						// B
//		bsk_n,								// col of B
//		0,									// beta
//		bskop_rn,							// C
//		path_ext * path_int					// col of C
//	);
	//
//	free(bskop_tmp_rn);
//
//	/* == Barrier Option ==
//	** Need two set of random numbers
//	** First: to reprice underlying stocks at H(use as S0 in inner)
//	** [path_ext, var_t]
//	** Second: inner loop, to reprice option
//	** [path_ext, [path_int, steps]] => [path_ext * path_int, steps]
//	*/
//	// Random number sequence for barrier option(outer loop)
//	float* barop_ext_rn = (float*)malloc((size_t)path_ext * var_t * sizeof(float));
//	rng->generate_sobol(barop_ext_rn, var_t, path_ext);
//	rng->convert_normal(barop_ext_rn, path_ext * var_t);
//
//	// Random number sequence for barrier option(inner loop)
//	barop_rn = (float*)malloc((size_t)path_ext * path_int * barop_t * sizeof(float));
//	rng->generate_sobol(barop_rn, barop_t, path_ext * path_int);
//	rng->convert_normal(barop_rn, path_ext * path_int * barop_t);
//
//	/*cout << "Random Numbers:\n";
//	for (int i = 0; i < path_ext * path_int; i++) {
//		for (int j = 0; j < barop_t; j++) {
//			cout << barop_rn[i * barop_t + j] << " ";
//		}
//		cout << endl;
//	}*/
//
//

	// ====================================================
	//            Outter Monte Carlo Simulation
	// ====================================================
	// Store by row
	int row_idx = 0;
	prices = (float*)malloc((size_t)path_ext * port_n * sizeof(float));


//	/* ====================
//	** ==      Bond      ==
//	** ==================== */
//	float price = 0.0f;
//	for (int i = 0; i < path_ext; i++) {
//		price = 0.0f;
//		// Loop to sum the coupon price until the maturity
//		for (int i = 0; i < bond->bond_m; i++) {
//			price += bond->bond_c /
//				powf(1.0f + (bond->bond_y[i] + bond_rn[i]) / 100, float(i + 1));
//		}
//		// Add the face value
//		price += bond->bond_par /
//			powf(1.0f + (bond->bond_y[bond->bond_m - 1] + bond_rn[i]) / 100, float(bond->bond_m));
//
//		// Store to first row of prices matrix
//		prices[row_idx * path_ext + i] = price;
//
//	}
//	row_idx++;
//
//
//
//
//	/* ====================
//	** ==     Stock      ==
//	** ==================== */
//	for (int i = 0; i < path_ext; i++) {
//		// Store to the next row of price matrix
//		prices[row_idx * path_ext + i] = stock->x * stock->s0
//			* exp((stock->mu - 0.5f * stock->var * stock->var) * var_t
//				+ stock->var * sqrtf(float(var_t)) * stock_rn[i]);
//	}
//	row_idx++;
//
//
//
//
//	/* ====================
//	** == Basket Option ==
//	** ==================== */
//	// TODO: don't add x when comparing S and K
//	float* bskop_stock_price = (float*)malloc((size_t)bsk_n * sizeof(float));
//	float* value_each = (float*)malloc((size_t)path_int * bskop->n * sizeof(float));
//	float* value_weighted = (float*)malloc((size_t)path_int * sizeof(float));
//	Stock* s;
//
//	for (int i = 0; i < path_ext; i++) {
//		// reprice underlying stocks
//		// will be used in inner as s0
//		for (int j = 0; j < bskop->n; j++) {
//			s = &(bskop->stocks[j]);
//			bskop_stock_price[j] = s->s0
//				* exp((s->mu - 0.5f * s->var * s->var) * var_t
//					+ s->var * sqrtf(float(var_t)) * bskop_ext_rn[i * bskop->n + j]);
//		}
//
//		// Inner loop
//		// The random number has already been transformed with cov
//		// Calculate each value with the correlated-random numbers
//		// random numbers[n][path_int]
//		for (int j = 0; j < bskop->n; j++) {
//			s = &(bskop->stocks[j]);
//			for (int k = 0; k < path_int; k++) {
//				int rn_offset = i * path_int * bskop->n + j * path_int + k;
//				value_each[j * path_int + k] = s->x *
//					bskop_stock_price[j] * exp((s->mu - 0.5f * s->var * s->var) * bskop_t
//						+ s->var * sqrtf(float(bskop_t)) * bskop_rn[rn_offset]);
//			}
//		}
//
//		// Multiply with weight
//		// Value_each[n][path_int]
//		// value_each[inter * n] * weight[n * 1]
//		cblas_sgemv(CblasRowMajor,		// Specifies row-major
//			CblasTrans,					// Specifies whether to transpose matrix A.
//			bskop->n,					// A rows
//			path_int,					// A col
//			1,							// alpha	
//			value_each,					// A
//			path_int,					// The size of the first dimension of matrix A.
//			bskop->w,					// Vector X.
//			1,							// Stride within X. 
//			0,							// beta
//			value_weighted,				// Vector Y
//			1							// Stride within Y
//		);
//
//		// Determine the price with stirke
//		// If the price is less than k, don't execute
//		float call = 0.0f;
//		for (int i = 0; i < path_int; i++) {
//			call += (value_weighted[i] > bskop->k) ? (value_weighted[i] - bskop->k) : 0;
//		}
//
//		// Store to the next row of price matrix
//		prices[row_idx * path_ext + i] = call / path_int;
//	}
//
//	free(bskop_ext_rn);
//	free(value_each);
//	free(value_weighted);
//	free(bskop_stock_price);
//	row_idx++;
//
//
//
//	/* ====================
//	** == Barrier Option ==
//	** ==================== */
//	// Up-in-call
//	float barop_stock_price = 0.0f;		// Stock price at var_t
//	float barop_max_price = 0.0f;		// Max price throughout the path
//	float barop_price = 0.0f;			// option price at one step
//	float call = 0.0f;					// Accumulated price for a inner path
//
//	for (int i = 0; i < path_ext; i++) {
//		s = barop->s;
//
//		// reprice underlying stocks
//		// will be used in inner as s0
//		barop_stock_price = s->s0 * exp((s->mu - 0.5f * s->var * s->var) * var_t
//			+ s->var * sqrtf(float(var_t)) * barop_ext_rn[i]);
//		// For consistancy with gpu implementation (calculate every path in parallel)
//		// So here we don't use early stop, just record the max
//		barop_max_price = barop_stock_price;
//
//		// Inner loop
//		call = 0.0f;
//		for (int j = 0; j < path_int; j++) {
//			// Loop over steps in one path, get max price
//			for (int k = 0; k < barop_t; k++) {
//				// Calculate price at this step
//				barop_price = barop_stock_price * exp((s->mu - 0.5f * s->var * s->var) * k
//					+ s->var * sqrtf(float(k)) * barop_rn[i * path_int * barop_t + j * barop_t + k]);
//
//				// Check maximum
//				if (barop_price > barop_max_price) {
//					barop_max_price = barop_price;
//				}
//			}
//			//cout << endl<< barop_price << endl;
//
//			// Compare with barrier, the option exists if max price is larger than barrier
//			if (barop_max_price > barop->h) {
//				// barop_price will be the last price
//				// max{St-K, 0}
//				call += (barop_price > barop->k) ? barop_price - barop->k : 0;
//			}
//		}
//		//cout << call << endl;
//
//		// Get expected price at var_t
//		prices[row_idx * path_ext + i] = s->x * (call / path_int);
//	}
//	free(barop_ext_rn);
//
//	// Reset
//	row_idx = 0;
//	rng->set_offset(1024);
//
//	cout << endl << "Prices:" << endl;
//	for (int i = 0; i < port_n; i++) {
//		for (int j = 0; j < path_ext; j++) {
//			cout << prices[i * path_ext + j] << " ";
//		}
//		cout << endl;
//	}
//	cout << endl << "Start Price:" << endl;
//	cout << port_p0 << endl;
//
//
//
//
//	// ====================================================
//	//						Loss
//	// ====================================================
//	// Fill loss with negtive today's price of the portfolio
//	float* loss = (float*)malloc((size_t)path_ext * sizeof(float));
//	for (int i = 0; i < path_ext; i++) {
//		loss[i] = port_p0;
//	}
//
//	// prices[port_n][path_ext]
//	// prices[path_ext * port_n] * w[port_n * 1]
//	// Loss = -(p-p0) = p0-p
//	// Loss = p0 - e^-rT * (price*w) = loss + (- e^-rT) *(price*w)
//	// haven't get ln here, but doesn't affect sorting
//	cblas_sgemv(CblasRowMajor,			// Specifies row-major
//		CblasTrans,						// Specifies whether to transpose matrix A.
//		port_n,							// A rows
//		path_ext,						// A col
//		-exp(-1 * risk_free * var_t),	// alpha
//		prices,							// A
//		path_ext,						// The size of the first dimension of matrix A.
//		port_w,							// Vector X.
//		1,								// Stride within X. 
//		1,								// beta
//		loss,							// Vector Y
//		1);								// Stride within Y
//
//	/*cout << endl << "Loss:" << endl;
//	for (int i = 0; i < path_ext; i++) {
//		cout << loss[i] << " ";
//	}
//	cout << endl;*/
//
//
//
//
//	// ====================================================
//	//						Sort
//	// ====================================================
//	std::sort(loss, loss + path_ext);
//
//	/*cout << endl << "Sorted Loss:" << endl;
//	for (int i = 0; i < path_ext; i++) {
//		std::cout << loss[i] << " ";
//	}
//	cout << endl;*/
//
//	//output_res(loss, path_ext);
//
//
//	// ====================================================
//	//				Calculate var and cvar
//	// ====================================================
//	int pos = (int)floor(path_ext * var_per);
//
//	float var = loss[pos];
//	float cvar = 0;
//	for (int i = pos; i < path_ext; i++) {
//		cvar += loss[i];
//	}
//	cvar /= path_ext - pos;
//
//	/*cout << endl;
//	cout << "var:" << var << endl;
//	cout << "cvar:" << cvar << endl;*/
//
//	free(loss);
//	free(prices);
//	free(stock_rn);
//	free(bskop_rn);
//	free(bond_rn);
//	free(barop_rn);
//	//delete(rng);
//
	chrono::steady_clock::time_point end = chrono::steady_clock::now();
	chrono::duration<double, std::milli> elapsed = end - start;

	return elapsed.count();
}

void NestedMonteCarloVaR::output_res(float* data, int len) {
	// open a file for outputting the matrix
	ofstream outputfile;
	outputfile.open("C:/Users/windows/Desktop/res.txt");

	// output the matrix to the file
	if (outputfile.is_open()) {
		for (int i = 0; i < len; i++) {
			outputfile << data[i] << " ";
		}
	}
	outputfile.close();
	cout << "Result outputed!" << endl;
}
