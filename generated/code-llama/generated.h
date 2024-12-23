#include <immintrin.h>

void sgemm_avx2(const float* A, const float* B, float* C) {
    __m256 c00, c01, c10, c11, c20, c21, c30, c31;
    __m256 c40, c41, c50, c51;
    __m256 b0, b1;
    __m256 a0,  a1,  a2,  a3,  a4,  a5;

    // // set to zero
    c00 = _mm256_setzero_ps();
    c01 = _mm256_setzero_ps();
    c10 = _mm256_setzero_ps();
    c11 = _mm256_setzero_ps();
    c20 = _mm256_setzero_ps();
    c21 = _mm256_setzero_ps();
    c30 = _mm256_setzero_ps();
    c31 = _mm256_setzero_ps();
    c40 = _mm256_setzero_ps();
    c41 = _mm256_setzero_ps();
    c50 = _mm256_setzero_ps();
    c51 = _mm256_setzero_ps();

    const float *a_ptr = A;
    const float *b_ptr = B;

    for (int k = 0; k < 256; ++k) {
        b0 = _mm256_loadu_ps(b_ptr);
        b1 = _mm256_loadu_ps(b_ptr + 8);

        a0 = _mm256_broadcast_ss(a_ptr + 0);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        a1 = _mm256_broadcast_ss(a_ptr + 1);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);

        a2 = _mm256_broadcast_ss(a_ptr + 2);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);

        a3 = _mm256_broadcast_ss(a_ptr + 3);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);

        a4 = _mm256_broadcast_ss(a_ptr + 4);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);

        a5 = _mm256_broadcast_ss(a_ptr + 5);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);

        a_ptr += 6;
        b_ptr += 16;
    }

    // Store the results
    // _mm256_storeu_ps(C + 0 * 16 + 0, c00);
    // _mm256_storeu_ps(C + 0 * 16 + 4, c01);
    // _mm256_storeu_ps(C + 1 * 16 + 0, c10);
    // _mm256_storeu_ps(C + 1 * 16 + 4, c11);
    // _mm256_storeu_ps(C + 2 * 16 + 0, c20);
    // _mm256_storeu_ps(C + 2 * 16 + 4, c21);
    // _mm256_storeu_ps(C + 3 * 16 + 0, c30);
    // _mm256_storeu_ps(C + 3 * 16 + 4, c31);
    _mm256_storeu_ps(C + 0 * 16 + 0, c00);
    _mm256_storeu_ps(C + 0 * 16 + 8, c01);
    _mm256_storeu_ps(C + 1 * 16 + 0, c10);
    _mm256_storeu_ps(C + 1 * 16 + 8, c11);
    _mm256_storeu_ps(C + 2 * 16 + 0, c20);
    _mm256_storeu_ps(C + 2 * 16 + 8, c21);
    _mm256_storeu_ps(C + 3 * 16 + 0, c30);
    _mm256_storeu_ps(C + 3 * 16 + 8, c31);

    _mm256_storeu_ps(C + 4 * 16 + 0, c40);
    _mm256_storeu_ps(C + 4 * 16 + 8, c41);
    _mm256_storeu_ps(C + 5 * 16 + 0, c50);
    _mm256_storeu_ps(C + 5 * 16 + 8, c51);
}
