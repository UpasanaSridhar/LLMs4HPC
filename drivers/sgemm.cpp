
#include <iostream>
#include <chrono>
#include <cassert>

#include "generated.h"

typedef float BufferT;

#ifdef AVX2
# define M 6
# define N 8
# define K 256
# define CALL sgemm_avx2
#endif


#ifdef AVX512
# define M 8
# define N 64
# define K 512
# define CALL sgemm_avx512
#endif



int main() {
    //print the type of kernel

    alignas(64) BufferT A[M * K];
    alignas(64) BufferT B[K * N];
    alignas(64) BufferT C[M * N] = {0}; // 6x16 output matrix

    // Initialize A and B with some values
    for (int i = 0; i < M * K; ++i) A[i] = static_cast<BufferT>(1);
    for (int i = 0; i < K * N; ++i) B[i] = static_cast<BufferT>(0.0001*i);

    //Check correctness against naive
    BufferT C_naive[M * N] = {0};
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C_naive[i * N + j] += A[k * M + i] * B[k * N + j];
            }
        }
    }
    //Call the kernel
    CALL(A, B, C);

    // Compare the results
    bool check = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("C[%d][%d] = %f, C_naive[%d][%d] = %f\n", i, j, C[i * N + j], i, j, C_naive[i * N + j]);
            assert(C[i * N + j] == C_naive[i * N + j]);
        }
    }

    // Run 10 iterations and take min of average time
    float min_time = 1e9;
    for(int r = 0; r < 1000; r++) {
        auto start = std::chrono::high_resolution_clock::now();

        // Run the matrix multiplication 100 times
        for (int iter = 0; iter < 1000; ++iter) {
            CALL(A, B, C);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<BufferT> elapsed = end - start;
        if(elapsed.count() < min_time) min_time = elapsed.count();
    }


    std::cout << "Time taken for 1000 runs: " << min_time << " seconds. Throughput "<< float(M*N*K*2.0*1000)/(min_time*5.183e9) << std::endl;


    // Print the result
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

