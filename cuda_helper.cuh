#pragma once
#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include<iostream>

using namespace std;

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    std::cout << "\nError at "<<__FILE__<<":"<<__LINE__<<": "<<x<<"\n"; \
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    std::cout << "\nError at "<<__FILE__<<":"<<__LINE__<<": "<<x<<"\n"; \
    return EXIT_FAILURE;}} while(0)

#define CUBLAS_CALL(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \
    std::cout << "\nError at "<<__FILE__<<":"<<__LINE__<<": "<<x<<"\n"; \
    return EXIT_FAILURE;}} while(0)

//void output(float* data, int m, int n, string title) {
//	std::cout << title << ":" << std::endl;
//	for (int i = 0; i < m; i++) {
//		for (int j = 0; j < n; j++) {
//			std::cout << data[i * n + j] << " ";
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//	return;
//}
//
//void output(double* data, int m, int n, std::string title) {
//	std::cout << title << ":" << std::endl;
//	for (int i = 0; i < m; i++) {
//		for (int j = 0; j < n; j++) {
//			std::cout << data[i * n + j] << " ";
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//	return;
//}

#endif