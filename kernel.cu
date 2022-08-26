#include "NestedMonteCarloVaR.cuh"

int main(int argc, char* argv[])
{
	int exp_times = 0;   // Total times of MC

	int path_ext = 10;  // Number of the outer MC loops
	int path_int = 10;  // Number of the inner MC loops

	// input: path_ext, path_int, exp_times
	bool combined_rng = false;
	bool barrier_early = false;
	bool same_rn = false;
	int cnt = 0;

	for (int i = 1; i < argc; i++) {
		char* pchar = argv[i];
		switch (pchar[0]) {			// Decide option type
		case '-': {					// For option
			switch (pchar[1]) {
			case 'c':		// combined_rng
				combined_rng = true;
				break;
			case 'e':		// barrier early stop
				barrier_early = true;
				break;
			case 's':		// Same RN sequence
				same_rn = true;
				break;
			default:		// unrecognisable - show usage
				cout << endl << "===================== USAGE =====================" << endl;
				cout << "\t-c\tUse combined sobol RNG and normal transfer" << endl;
				cout << "\t-e\tUse early stop strategy for barrier option" << endl;
				cout << "\t-s\tUse same RN for inner loop" << endl;
				cout << "Enter up to 3 numbers for [path_ext, path_int, exp_times]" << endl;
				cout << "Default setup: [" << path_ext << ", " << path_int << ", "
					<< exp_times << "]" << endl;
				return;
			}
			break;
		}
		default:				// For numbers(not concerning rubust so
								// char will be treated as numbers)
			switch (cnt) {
			case 0:
				path_ext = atoi(argv[i]);
				break;
			case 1:
				path_int = atoi(argv[i]);
				break;
			case 2:
				exp_times = atoi(argv[i]);
				break;
			default:
				break;
			}
			cnt++;
		}
	}


	cout << endl << "== SET UP ==" << endl;
	cout << "Experiment Times: " << exp_times << endl;
	cout << "Path External: " << path_ext << endl;
	cout << "Path Internal: " << path_int << endl;
	cout << "Optimisation: Combined RNG - " << combined_rng << endl;
	cout << "              Barrier Early Stop - " << barrier_early << endl;
	cout << "              Same Inner RN - " << same_rn << endl;

	/*
    cout << endl << "== DEVICE ==" << endl;

	int deviceCount;

	CUDA_CALL(cudaGetDeviceCount(&deviceCount));

	printf("Number of CUDA devices %d.\n", deviceCount);

	for (int dev = 0; dev < deviceCount; dev++) {
		cudaDeviceProp deviceProp;

		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, dev));

		if (dev == 0) {
			if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
				cout << "No CUDA GPU has been detected\n";
				return -1;
			}
			else if (deviceCount == 1) {
				cout << "There is 1 device supporting CUDA\n";
			}
			else {
				cout << "There are " << deviceCount << " devices supporting CUDA\n";
			}
		}

		printf("For device #%d\n", dev);
		printf("Device name:                %s\n", deviceProp.name);
		printf("Major revision number:      %d\n", deviceProp.major);
		printf("Minor revision Number:      %d\n", deviceProp.minor);
		printf("Total Global Memory:        %zd\n", deviceProp.totalGlobalMem);
		printf("Total shared mem per block: %zd\n", deviceProp.sharedMemPerBlock);
		printf("Total const mem size:       %zd\n", deviceProp.totalConstMem);
		printf("Warp size:                  %d\n", deviceProp.warpSize);
		printf("Maximum block dimensions:   %d x %d x %d\n", deviceProp.maxThreadsDim[0], \
			deviceProp.maxThreadsDim[1], \
			deviceProp.maxThreadsDim[2]);

		printf("Maximum grid dimensions:    %d x %d x %d\n", deviceProp.maxGridSize[0], \
			deviceProp.maxGridSize[1], \
			deviceProp.maxGridSize[2]);
		printf("Clock Rate:                 %d\n", deviceProp.clockRate);
		printf("Number of muliprocessors:   %d\n", deviceProp.multiProcessorCount);

	}

    */

	const int var_t = 1;					// VaR duration
	const float var_per = 0.95f;				// 1-percentile

	const int port_n = 4;					// Number of products in the portfolio
	float port_w[port_n] = { 0.2f, 0.1f, 0.35f, 0.35f };		// Weights of the products in the portfolio
														// { bond, stock, basket option, barrier option}
	//float port_w[port_n] = { 0.0f, 0.0f, 0.0f, 1.0f };
	const float risk_free = 0.01f;

	const float bond_par = 1000.0f;			// Par value of bond
	const float bond_c = 100.0f;			// Coupon
	const int bond_m = 10;					// Maturity
	float bond_y[bond_m] = {
			5.00f, 5.69f, 6.09f, 6.38f, 6.61f,
			6.79f, 6.94f, 7.07f, 7.19f, 7.30f
	};										// yeild curve
	const float sigma = 1.5f;				// sigma

	const float stock_s0 = 300.0f;			// Start value of stock
	const float stock_mu = risk_free;			// risk free(or mean)
	const float stock_var = 0.03f;			// Volatility
	const int stock_x = 1;					// Number of shares

	Stock* s0 = new Stock(stock_s0, stock_mu, stock_var, 10);
	Stock* s1 = new Stock(500.0f, risk_free, 0.02f, 10);
	Stock* s2 = new Stock(700.0f, risk_free, 0.01f, 10);
	const int bskop_n = 2;								// Number of stocks in basket
	const float bskop_k = 390.0f;						// Execution price
	const int bskop_t = 1;								// Maturity of basket option
	Stock bskop_stocks[bskop_n] = { *s1, *s2 };			// List of stocks
	float bskop_cov[bskop_n * bskop_n] = { 1.0f, 0.5f,
										   0.5f, 1.0f };	// Covariance matrix
	float bskop_w[bskop_n] = { 0.8f, 0.2f };				// weight

	const float barop_k = 310.0f;				// Execution price
	const float barop_h = 320.0f;				// Barrier
	const int barop_t = 30;						// Maturity(steps of inner path)

	NestedMonteCarloVaR* mc = new NestedMonteCarloVaR(
		path_ext, path_int,
		var_t, var_per,
		port_n, port_w,
		risk_free
	);

	mc->optimise_init(combined_rng, barrier_early, same_rn);
	mc->bond_init(bond_par, bond_c, bond_m, bond_y, sigma, 0);
	mc->stock_init(stock_s0, stock_mu, stock_var, stock_x, 1);
	mc->bskop_init(bskop_n, bskop_stocks, bskop_cov, bskop_k, bskop_w, bskop_t, 2);
	mc->barop_int(s0, barop_k, barop_h, barop_t, 3);
	cout << endl << "== EXECUTION ==" << endl;

	// Warm up
	mc->execute();

	double exe_time = 0.0;
	for (int i = 0; i < exp_times; i++) {
		exe_time += mc->execute();
		if ((i == 0) || (i + 1 == exp_times)
			||(exp_times <= 10)
			|| (exp_times > 10 && i % (exp_times / 10) == 0)) {
			cout << "Experiment # " << i << " finished." << endl;
		}
	}

	cout << endl << "== RESULT ==" << endl;
	cout << "AVG EXECUTION TIME: " << exe_time / exp_times << " ms." << endl;
	return 0;
}
