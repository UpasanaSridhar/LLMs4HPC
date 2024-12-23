
#include <iostream>
#include <chrono>
#include <cassert>

// #include "generated.h"

typedef double BufferT;
// #include <immintrin.h>

void dgemm_avx2(double* A, double* B, double* C) {
    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 8; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 256; ++k) {
                sum += A[i * 256 + k] * B[k * 8 + j];
            }
            C[i * 8 + j] = sum;
        }
    }
}

// void dgemm_avx2(double *A, double *B, double *C) {
//     int i, j, k;

//     // Loop over the rows of A and B
//     for (i = 0; i < 6; i++) {
//         // Loop over the columns of A and B
//         for (j = 0; j < 8; j++) {
//             // Initialize the dot product to zero
//             __m256d dot = _mm256_setzero_pd();

//             // Loop over the rows of A and B, calculating the dot product
//             for (k = 0; k < 256; k += 8) {
//                 // Load two 256-bit blocks of values from A and B into AVX registers
//                 __m256d a_vals = _mm256_loadu_pd(&A[i * 256 + k]);
//                 __m256d b_vals = _mm256_loadu_pd(&B[k * 8 + j]);

//                 // Calculate the dot product of the two AVX registers
//                 dot = _mm256_fmadd_pd(a_vals, b_vals, dot);
//             }

//             // Store the final dot product in C
//             _mm256_storeu_pd(&C[i * 8 + j], dot);
//         }
//     }
// }


int main() {
    //print the type of kernel

    alignas(64) BufferT A[6 * 256];
    alignas(64) BufferT B[256 * 8];
    alignas(64) BufferT C[6 * 8] = {0}; // 6x8 output matrix

    // Initialize A and B with some values
    for (int i = 0; i < 6 * 256; ++i) A[i] = static_cast<BufferT>(i);
    for (int i = 0; i < 256 * 8; ++i) B[i] = static_cast<BufferT>(0.0001*i);

    //Check correctness against naive
    BufferT C_naive[6 * 8] = {0};
    for (int k = 0; k < 256; ++k) {
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 8; ++j) {
                C_naive[i * 8 + j] += A[i * 256 + k] * B[k * 8 + j];
            }
        }
    }
    //Call the kernel
    dgemm_avx2(A, B, C);

    // Compare the results
    bool check = true;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 8; ++j) {
            // printf("C[%d][%d] = %f, C_naive[%d][%d] = %f\n", i, j, C[i * 8 + j], i, j, C_naive[i * 8 + j]);
            assert(C[i * 8 + j] == C_naive[i * 8 + j]);
        }
    }

    // Run 10 iterations and take min of average time
    float min_time = 1e9;
    for(int r = 0; r < 1000; r++) {
        auto start = std::chrono::high_resolution_clock::now();

        // Run the matrix multiplication 100 times
        for (int iter = 0; iter < 1000; ++iter) {
            dgemm_avx2(A, B, C);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<BufferT> elapsed = end - start;
        if(elapsed.count() < min_time) min_time = elapsed.count();
    }


    std::cout << "Time taken for 1000 runs: " << min_time << " seconds. Throughput "<< float(6*8*256*2.0*1000)/(min_time*5.183e9) << std::endl;


    // Print the result
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 8; ++j) {
            std::cout << C[i * 8 + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

