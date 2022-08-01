#include "RNG.cuh"

int RNG::init_cpu() {
    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGeneratorHost(&gen, CURAND_RNG_QUASI_SOBOL32));
}

int RNG::init_gpu() {
    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL32));
}

RNG::~RNG() {
    curandDestroyGenerator(gen);
}

int RNG::generate_sobol(float*& data, int m, int n) {
    // M-dim of N numbers
    /* Set offset*/
    CURAND_CALL(curandSetGeneratorOffset(gen, this->offset));
    
    /* Set dimention m */
    CURAND_CALL(curandSetQuasiRandomGeneratorDimensions(gen, m));

    /* Generate n floats on device */
    CURAND_CALL(curandGenerateUniform(gen, data, n * m));

    // offset will be added up with each rng call
    this->offset += n * m;
    return 0;
}

int RNG::convert_normal(float*& data, int length, float sigma) {
    for (int i = 0; i < length; i++) {
        data[i] = normcdfinvf(data[i]) * sigma;
    }
    
    return 0;
}

int RNG::generate_sobol_normal(float*& data, int m, int n, float sigma) {
    // M-dim of N numbers
    /* Set offset*/
    CURAND_CALL(curandSetGeneratorOffset(gen, this->offset));

    /* Set dimention m */
    CURAND_CALL(curandSetQuasiRandomGeneratorDimensions(gen, m));

    /* Generate n floats on device */
    CURAND_CALL(curandGenerateNormal(gen, data, n * m, 0, sigma));

    // offset will be added up with each rng call
    this->offset += n * m;
    return 0;
}