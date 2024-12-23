#include <immintrin.h>

void dgemm_avx2(const double* A, const double* B, double* C) {
    __m256d c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
    __m256d b0, b1, a_broadcast;

    // Initialize C registers to zero
    c00 = c01 = c10 = c11 = c20 = c21 = _mm256_setzero_pd();
    c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_pd();

    for (int k = 0; k < 256; ++k) {
        b0 = _mm256_load_pd(B + k * 8 + 0);  // Load B[k,0:3]
        b1 = _mm256_load_pd(B + k * 8 + 4);  // Load B[k,4:7]

        // Broadcast A[i, k] to all elements of a register and FMA with B columns
        a_broadcast = _mm256_broadcast_sd(A + k * 6 + 0);
        c00 = _mm256_fmadd_pd(a_broadcast, b0, c00);
        c01 = _mm256_fmadd_pd(a_broadcast, b1, c01);

        a_broadcast = _mm256_broadcast_sd(A + k * 6 + 1);
        c10 = _mm256_fmadd_pd(a_broadcast, b0, c10);
        c11 = _mm256_fmadd_pd(a_broadcast, b1, c11);

        a_broadcast = _mm256_broadcast_sd(A + k * 6 + 2);
        c20 = _mm256_fmadd_pd(a_broadcast, b0, c20);
        c21 = _mm256_fmadd_pd(a_broadcast, b1, c21);

        a_broadcast = _mm256_broadcast_sd(A + k * 6 + 3);
        c30 = _mm256_fmadd_pd(a_broadcast, b0, c30);
        c31 = _mm256_fmadd_pd(a_broadcast, b1, c31);

        a_broadcast = _mm256_broadcast_sd(A + k * 6 + 4);
        c40 = _mm256_fmadd_pd(a_broadcast, b0, c40);
        c41 = _mm256_fmadd_pd(a_broadcast, b1, c41);

        a_broadcast = _mm256_broadcast_sd(A + k * 6 + 5);
        c50 = _mm256_fmadd_pd(a_broadcast, b0, c50);
        c51 = _mm256_fmadd_pd(a_broadcast, b1, c51);
    }

    // Store C registers back to memory
    _mm256_store_pd(C + 0 * 8 + 0, c00);
    _mm256_store_pd(C + 0 * 8 + 4, c01);
    _mm256_store_pd(C + 1 * 8 + 0, c10);
    _mm256_store_pd(C + 1 * 8 + 4, c11);
    _mm256_store_pd(C + 2 * 8 + 0, c20);
    _mm256_store_pd(C + 2 * 8 + 4, c21);
    _mm256_store_pd(C + 3 * 8 + 0, c30);
    _mm256_store_pd(C + 3 * 8 + 4, c31);
    _mm256_store_pd(C + 4 * 8 + 0, c40);
    _mm256_store_pd(C + 4 * 8 + 4, c41);
    _mm256_store_pd(C + 5 * 8 + 0, c50);
    _mm256_store_pd(C + 5 * 8 + 4, c51);
}
