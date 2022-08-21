#include "NestedMonteCarloVaR.cuh"

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

void NestedMonteCarloVaR::optimise_init(bool combined_rng, bool barrier_early, bool same_rn) {
	this->combined_rng = combined_rng;
	this->barrier_early = barrier_early;
	this->same_rn = same_rn;
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
	float* tmp_rn;
	cudaError x = cudaMallocManaged((void**)&tmp_rn, path_int * bskop_n * sizeof(float));
	if (x != cudaSuccess) {
		std::cout << "\nError at " << __FILE__ << ":"
			<< __LINE__ << ": " << cudaGetErrorString(x) << "\n";
		return;
	}

	// Generate and convert with cholesky matrix
	rng->generate_sobol_normal(tmp_rn, bskop_n, path_int);
	cudaDeviceSynchronize();

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
				* exp((s->mean - 0.5f * s->std * s->std) * bskop_t
					+ s->std * sqrtf(float(bskop_t)) * rn[i * path_int + j]);
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
	cudaFree(tmp_rn);
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
	//float* temp_rn = (float*)malloc((size_t)path_int * barop_t * sizeof(float));
	float* temp_rn;
	cudaError x = cudaMallocManaged((void**)&temp_rn, path_int * barop_t * sizeof(float));
	if (x != cudaSuccess) {
		std::cout << "\nError at " << __FILE__ << ":"
			<< __LINE__ << ": " << cudaGetErrorString(x) << "\n";
		return;
	}
	rng->generate_sobol_normal(temp_rn, barop_t, path_int);
	cudaDeviceSynchronize();
	

	Stock* s = barop_stock;
	float barop_max_price = 0.0f;		// Max price throughout the path
	float barop_price = 0.0f;			// option price at one step
	float call = 0.0f;					// Accumulated price for a inner path

	for (int j = 0; j < path_int; j++) {
		// Loop over steps in one path, get max price
		for (int k = 0; k < barop_t; k++) {
			// Calculate price at this step
			barop_price = s->s0 * exp((s->mean - 0.5f * s->std * s->std) * k
				+ s->std * sqrtf(float(k)) * temp_rn[j * barop_t + k]);

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
	cudaFree(temp_rn);


	// Add start price to the portfolio start price
	this->port_p0 += s->x * (call / path_int) * port_w[idx];
}




double NestedMonteCarloVaR::execute() {
	chrono::steady_clock::time_point start = chrono::steady_clock::now();

	// ====================================================
	//             Random number preperation
	// ====================================================
	int blocksize = 1024;
	dim3 block(blocksize, 1);

	dim3 grid_ext_rn((var_t * path_ext - 1) / block.x + 1, 1);
	CUDA_CALL(cudaMallocManaged((void**)&ext_rn, var_t * path_ext * sizeof(float)));
	if (!combined_rng) {
		rng->generate_sobol(ext_rn, 1, path_ext);
		moro_inv << < grid_ext_rn, block >> > (ext_rn, path_ext, 0, 1);
	}
	else {
		rng->generate_sobol_normal(ext_rn, 1, path_ext, 1);
	}


	/* == BOND ==
	** RN is used to move yield curve up/down, N~(0, sigma^2)
	** [path_ext, 1]
	*/
	bond_rn = ext_rn;


	/* == STOCK ==
	** RN is used as the external path
	** For stock pricing only, there's no need to generate the inner path now
	** [path_ext, var_t]
	*/
	stock_rn = ext_rn;


	/* == Barrier Option ==
	** Need two set of random numbers
	** First: to reprice underlying stocks at H(use as S0 in inner)
	** [path_ext, var_t]
	** Second: inner loop, to reprice option
	** [path_ext, [path_int, steps]] => [path_ext * path_int, steps]
	*/
	// Random number sequence for barrier option

	int ext = path_ext;
	if (same_rn) {
		ext = 1;
	}

	dim3 grid_barop_rn_int((ext * path_int * barop_t - 1) / block.x + 1, 1);

	//float* barop_ext_rn;	
	CUDA_CALL(cudaMallocManaged((void**)&barop_rn, ext * path_int * barop_t * sizeof(float)));

	if (!combined_rng) {
		rng->generate_sobol(barop_rn, barop_t, ext * path_int);
		moro_inv << < grid_barop_rn_int, block >> > (barop_rn, ext * path_int * barop_t, 0, 1);

	}
	else {
		rng->generate_sobol_normal(barop_rn, barop_t, ext * path_int);
	}

	
	

	/* == Basket Option ==
	** Need two set of random numbers
	** First: to reprice underlying stocks at H(use as S0 in inner)
	** [path_ext, n]
	** Second: inner loop, to reprice option
	** [path_ext, [path_int, n]] => [path_ext * path_int, n]
	*/
	int bsk_n = bskop->n;
	dim3 grid_bsk_rn((ext * path_int * bsk_n - 1) / block.x + 1, 1);

	// Random number sequence for basket option(inner loop)
	float* bskop_tmp_rn;
	CUDA_CALL(cudaMallocManaged((void**)&bskop_rn, ext * path_int * bsk_n * sizeof(float)));
	CUDA_CALL(cudaMallocManaged((void**)&bskop_tmp_rn, ext * path_int * bsk_n * sizeof(float)));

	if (!combined_rng) {
		rng->generate_sobol(bskop_tmp_rn, bsk_n, ext * path_int);
		moro_inv << < grid_bsk_rn, block >> > (bskop_tmp_rn, path_ext * path_int * bsk_n, 0, 1);

	}
	else {
		rng->generate_sobol_normal(bskop_tmp_rn, bsk_n, ext * path_int);
	}


	cudaDeviceSynchronize();


	// Covariance transformation
	// A[n * n]*rn[n * (path_ext * path_int)]
	float one = 1.0f;
	float zero = 0.0f;
	cublasHandle_t handle;
	CUBLAS_CALL(cublasCreate(&handle));

	float* A = NULL;
	CUDA_CALL(cudaMallocManaged((void**)&A, bsk_n * bsk_n * sizeof(float)));
	for (int i = 0; i < bsk_n * bsk_n; i++) {
		A[i] = bskop->A[i];
	}
	
	// A B = (BT AT)T
	// A(m,k), B(k, n)
	//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, *B, n, *A, k, &beta, *C, n);
	CUBLAS_CALL(cublasSgemm(handle,		// handle to the cuBLAS library context.
		CUBLAS_OP_N,					// operation op(A) that is non- or (conj.) transpose.
		CUBLAS_OP_N,					// operation op(B) that is non- or (conj.) transpose.
		bsk_n, 							// row of C
		ext * path_int,					// col of C
		bsk_n,							// multi by
		&one,							// alpha
		A,								// A
		bsk_n,							// A row
		bskop_tmp_rn,					// B
		bsk_n,							// B row
		&zero,							// beta
		bskop_rn,						// C
		bsk_n							// C row
	));

	cudaDeviceSynchronize();

	CUDA_CALL(cudaFree(bskop_tmp_rn));
	CUDA_CALL(cudaFree(A));


	// ====================================================
	//            Outter Monte Carlo Simulation
	// ====================================================
	// Store by row
	int row_idx = 0;

	// Save the prices
	CUDA_CALL(cudaMallocManaged((void**)&prices, path_ext* port_n * sizeof(float)));

	
	/* ====================
	** ==      Bond      ==
	** ==================== */
	dim3 grid_bond((path_ext - 1) / block.x + 1, 1);

	// Use Unified Memory and copy bond info to device
	float* bond_y;
	CUDA_CALL(cudaMallocManaged((void**)&bond_y, bond->bond_m * sizeof(float)));
	for (int i = 0; i < bond->bond_m; i++) {
		bond_y[i] = bond->bond_y[i];
	}
	
	// Pricing bond
	price_bond << <grid_bond, block >> > (
		bond_rn, 
		path_ext, 
		bond->bond_par, bond->bond_c, bond->bond_m, bond->sigma,
		bond_y,
		&prices[row_idx * path_ext]
	);
	
	cudaDeviceSynchronize();

	CUDA_CALL(cudaFree(bond_y));
	//CUDA_CALL(cudaFree(bond_rn));

	

	//cout << endl << "Bond:" << endl;
	//for (int i = 0; i < path_ext; i++) {
	//	cout << prices[row_idx * path_ext + i] << " ";
	//}
	//cout << endl;

	row_idx++;

	/* ====================
	** ==     Stock      ==
	** ==================== */
	dim3 grid_stock((path_ext - 1) / block.x + 1, 1);

	// Pricing bond
	price_stock << <grid_stock, block >> > (
		stock_rn,
		path_ext,
		stock->s0, stock->mean, stock->std,
		stock->x, var_t,
		&prices[row_idx * path_ext]
		);

	cudaDeviceSynchronize();

	//CUDA_CALL(cudaFree(stock_rn));

	row_idx++;

	/* ====================
	** == Barrier Option ==
	** ==================== */
	// Up-in-call
	dim3 grid_barop((path_ext - 1) / block.x + 1, 1);
	/*cout << endl << "RN:" << endl;
	for (int i = 0; i < path_ext; i++) {
		cout << ext_rn[i] << " ";
	}
	cout << endl;*/


	if (!barrier_early && !same_rn) {
		price_barrier << <grid_barop, block >> > (
			ext_rn,
			barop_rn,
			path_ext,
			path_int,
			barop->s->s0,
			barop->s->mean,
			barop->s->std,
			barop->s->x,
			var_t,
			barop_t,
			barop->h,
			barop->k,
			&prices[row_idx * path_ext]
			);
	}
	else if (barrier_early && !same_rn) {
		price_barrier_early << <grid_barop, block >> > (
			ext_rn,
			barop_rn,
			path_ext,
			path_int,
			barop->s->s0,
			barop->s->mean,
			barop->s->std,
			barop->s->x,
			var_t,
			barop_t,
			barop->h,
			barop->k,
			&prices[row_idx * path_ext]
			);
	}
	else if (!barrier_early && same_rn) {
		price_barrier_sameRN << <grid_barop, block >> > (
			ext_rn,
			barop_rn,
			path_ext,
			path_int,
			barop->s->s0,
			barop->s->mean,
			barop->s->std,
			barop->s->x,
			var_t,
			barop_t,
			barop->h,
			barop->k,
			&prices[row_idx * path_ext]
			);
	}
	else {
		price_barrier_early_sameRN << <grid_barop, block >> > (
			ext_rn,
			barop_rn,
			path_ext,
			path_int,
			barop->s->s0,
			barop->s->mean,
			barop->s->std,
			barop->s->x,
			var_t,
			barop_t,
			barop->h,
			barop->k,
			&prices[row_idx * path_ext]
			);
	}

	cudaDeviceSynchronize();
	CUDA_CALL(cudaFree(ext_rn));
	CUDA_CALL(cudaFree(barop_rn));

	/*cout << endl << "Barop:" << endl;
	for (int i = 0; i < path_ext; i++) {
		cout << prices[row_idx * path_ext + i] << " ";
	}
	cout << endl;*/

	row_idx++;


	/* ====================
	** == Basket Option ==
	** ==================== */
	dim3 grid_bskop((path_ext - 1) / block.x + 1, 1);
	
	float* list = NULL;
	float* value_weighted = NULL;
	CUDA_CALL(cudaMallocManaged((void**)&list, bsk_n * 5 * sizeof(float)));
	CUDA_CALL(cudaMallocManaged((void**)&value_weighted, path_int * path_ext * sizeof(float)));
	//[s0, mean, std, x, w]
	for (int i = 0; i < bsk_n; i++) {
		list[i + bsk_n * 0] = bskop->stocks[i].s0;
		list[i + bsk_n * 1] = bskop->stocks[i].mean;
		list[i + bsk_n * 2] = bskop->stocks[i].std;
		list[i + bsk_n * 3] = (float)bskop->stocks[i].x;
		list[i + bsk_n * 4] = bskop->w[i];
	}
	for (int i = 0; i < path_int * path_ext; i++) {
		value_weighted[i] = 0.0f;
	}

	if (!same_rn) {
		price_bskop << <grid_bskop, block >> > (
			path_ext,
			path_int,
			bskop_rn,
			ext_rn,
			bskop->n,
			var_t,
			list,				//s0
			&list[bsk_n * 1],	//mean
			&list[bsk_n * 2],	//std
			&list[bsk_n * 3],	//x
			bskop_t,
			bskop->k,
			&list[bsk_n * 4],	//w
			//value_weighted,
			&prices[row_idx * path_ext]
			);
	}
	else {
		price_bskop_sameRN << <grid_bskop, block >> > (
			path_ext,
			path_int,
			bskop_rn,
			ext_rn,
			bskop->n,
			var_t,
			list,				//s0
			&list[bsk_n * 1],	//mean
			&list[bsk_n * 2],	//std
			&list[bsk_n * 3],	//x
			bskop_t,
			bskop->k,
			&list[bsk_n * 4],	//w
			//value_weighted,
			&prices[row_idx * path_ext]
			);
	}
	
	
	cudaDeviceSynchronize();
	
	CUDA_CALL(cudaFree(list));
	CUDA_CALL(cudaFree(value_weighted));
	CUDA_CALL(cudaFree(bskop_rn));

	//cout << endl << "Bskop:" << endl;
	//for (int i = 0; i < path_ext; i++) {
	//	cout << prices[row_idx * path_ext + i] << " ";
	//}
	//cout << endl;
	
	row_idx++;



	

	// Reset
	row_idx = 0;
	rng->set_offset(1024);

	/*cout << endl << "Prices:" << endl;
	for (int i = 0; i < path_ext * 4; i++) {
		cout << prices[i] << " ";
	}
	cout << endl;*/

	cout << endl << "Start Price: "<< port_p0 << endl;


	// ====================================================
	//						Loss
	// ====================================================
	// Fill loss with negtive today's price of the portfolio
	//cublasHandle_t handle;
	//CUBLAS_CALL(cublasCreate(&handle));
	
	float* loss = NULL, *w = NULL;
	CUDA_CALL(cudaMallocManaged((void**)&loss, path_ext * sizeof(float)));
	CUDA_CALL(cudaMallocManaged((void**)&w, port_n * sizeof(float)));
	for (int i = 0; i < path_ext; i++) {
		loss[i] = port_p0;
	}
	for (int i = 0; i < port_n; i++) {
		w[i] = port_w[i];
	}

	// prices[port_n][path_ext]
	// prices[path_ext * port_n] * w[port_n * 1]
	// Loss = -(p-p0) = p0-p
	// Loss = p0 - e^-rT * (price*w) = loss + (- e^-rT) *(price*w)
	const float discount = -exp(-1.0f * risk_free * var_t);
	
	CUBLAS_CALL(cublasSgemv(handle,
		CUBLAS_OP_N,	// Use storage by row
		path_ext,		// cols of A
		port_n,			// rows of A
		&discount,		// alpha
		//&one,		// alpha
		prices,			// A
		path_ext,			// leading dimension of two-dimensional array used to store matrix A.
		w,				// x
		1,				// stride of x
		&one,			// beta
		//&zero,			// beta
		loss,			// y
		1				// stride of y
	));
	CUDA_CALL(cudaDeviceSynchronize());

	CUDA_CALL(cudaFree(w));
	CUBLAS_CALL(cublasDestroy(handle));


	cout << endl << "Loss:" << endl;
	for (int i = 0; i < path_ext; i++) {
		cout << loss[i] << " ";
	}
	cout << endl;


	// ====================================================
	//						Sort
	// ====================================================
	std::sort(loss, loss + path_ext);

	/*cout << endl << "Sorted Loss:" << endl;
	for (int i = 0; i < path_ext; i++) {
		std::cout << loss[i] << " ";
	}
	cout << endl;*/

	output_res(loss, path_ext);


	// ====================================================
	//				Calculate var and cvar
	// ====================================================
	int pos = (int)floor(path_ext * var_per);

	float var = loss[pos];
	float cvar = 0;
	for (int i = pos; i < path_ext; i++) {
		cvar += loss[i];
	}
	cvar /= path_ext - pos;

	/*cout << endl;
	cout << "var:" << var << endl;
	cout << "cvar:" << cvar << endl;*/

	
	CUDA_CALL(cudaFree(prices));
	CUDA_CALL(cudaFree(loss));
	

	chrono::steady_clock::time_point end = chrono::steady_clock::now();
	chrono::duration<double, std::milli> elapsed = end - start;

	return elapsed.count();
}

void NestedMonteCarloVaR::output_res(float* data, int len) {
	// open a file for outputting the matrix
	ofstream outputfile;
	outputfile.open("C:/Users/windows/Desktop/res_gpu.txt");

	// output the matrix to the file
	if (outputfile.is_open()) {
		for (int i = 0; i < len; i++) {
			outputfile << data[i] << " ";
		}
	}
	outputfile.close();
	cout << "Result outputed!" << endl;
}
