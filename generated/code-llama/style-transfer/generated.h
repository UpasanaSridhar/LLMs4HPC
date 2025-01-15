
#include <immintrin.h>


void sgemm_avx2(float const* A, float const * B,  float *C) {
    __m256 c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c3_0, c3_1, c4_0, c4_1, c5_0, c5_1;
    __m256 b0, b1;
    __m256 a0, a1;

    //set to zero
    c0_0 = _mm256_setzero_ps();
    c0_1 = _mm256_setzero_ps();
    c1_0 = _mm256_setzero_ps();
    c1_1 = _mm256_setzero_ps();
    c2_0 = _mm256_setzero_ps();
    c2_1 = _mm256_setzero_ps();
    c3_0 = _mm256_setzero_ps();
    c3_1 = _mm256_setzero_ps();
    c4_0 = _mm256_setzero_ps();
    c4_1 = _mm256_setzero_ps();
    c5_0 = _mm256_setzero_ps();
    c5_1 = _mm256_setzero_ps();

    //ptr to cols of A, rows of B
    const float *a_ptr = A;
    const float *b_ptr = B;

    for (int k = 0; k < 256; ++k) {
        
        // load 1 row of B into vector registers
        b0 = _mm256_load_ps(b_ptr + 0);
        b1 = _mm256_load_ps(b_ptr + 8);

        //Increment pointers
        b_ptr += 16; 
        a_ptr += 6;

        for (int j = 0; j < 6; ++j) {
            // load a column of A into vector registers
            a0 = _mm256_load_ps(a_ptr + 0);
            a1 = _mm256_load_ps(a_ptr + 8);

            // multiply and add
            c0_0 = _mm256_fmadd_ps(b0, a0, c0_0);
            c1_0 = _mm256_fmadd_ps(b1, a0, c1_0);
            c2_0 = _mm256_fmadd_ps(b0, a1, c2_0);
            c3_0 = _mm256_fmadd_ps(b1, a1, c3_0);
        }
    }

    //Stores of C
    _mm256_store_ps(C + 0 * 16 + 0, c0_0);
    _mm256_store_ps(C + 0 * 16 + 8, c0_1);
    _mm256_store_ps(C + 1 * 16 + 0, c1_0);
    _mm256_store_ps(C + 1 * 16 + 8, c1_1);
    _mm256_store_ps(C + 2 * 16 + 0, c2_0);
    _mm256_store_ps(C + 2 * 16 + 8, c2_1);
    _mm256_store_ps(C + 3 * 16 + 0, c3_0);
    _mm256_store_ps(C + 3 * 16 + 8, c3_1);
    _mm256_store_ps(C + 4 * 16 + 0, c4_0);
    _mm256_store_ps(C + 4 * 16 + 8, c4_1);
    _mm256_store_ps(C + 5 * 16 + 0, c5_0);
    _mm256_store_ps(C + 5 * 16 + 8, c5_1);
}


void dgemm_avx2(double const* A, double const * B, double *C) {
    __m256d c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c3_0, c3_1, c4_0, c4_1, c5_0, c5_1;
    __m256d b0, b1;
    __m256d a0, a1, a2, a3, a4, a5;

    //set to zero
    c0_0 = _mm256_setzero_pd();
    c0_1 = _mm256_setzero_pd();
    c1_0 = _mm256_setzero_pd();
    c1_1 = _mm256_setzero_pd();
    c2_0 = _mm256_setzero_pd();
    c2_1 = _mm256_setzero_pd();
    c3_0 = _mm256_setzero_pd();
    c3_1 = _mm256_setzero_pd();
    c4_0 = _mm256_setzero_pd();
    c4_1 = _mm256_setzero_pd();
    c5_0 = _mm256_setzero_pd();
    c5_1 = _mm256_setzero_pd();

    const double *a_row = A;
    const double *b_col = B;

    for (int k = 0; k < 256; ++k) {
    
        //Load B
        b0 = _mm256_load_pd(b_col + 0);
        b1 = _mm256_load_pd(b_col + 4);

        //6x8 outer product
        a0 = _mm256_broadcast_sd(a_row + 0);
        c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
        c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);
        a1 = _mm256_broadcast_sd(a_row +  1);
        c1_0 = _mm256_fmadd_pd(a1, b0, c1_0);
        c1_1 = _mm256_fmadd_pd(a1, b1, c1_1);
        a2 = _mm256_broadcast_sd(a_row +  2);
        c2_0 = _mm256_fmadd_pd(a2, b0, c2_0);
        c2_1 = _mm256_fmadd_pd(a2, b1, c2_1);
        a3 = _mm256_broadcast_sd(a_row + 3);
        c3_0 = _mm256_fmadd_pd(a3, b0, c3_0);
        c3_1 = _mm256_fmadd_pd(a3, b1, c3_1);
        a4 = _mm256_broadcast_sd(a_row + 4);
        c4_0 = _mm256_fmadd_pd(a4, b0, c4_0);
        c4_1 = _mm256_fmadd_pd(a4, b1, c4_1);
        a5 = _mm256_broadcast_sd(a_row + 5);
        c5_0 = _mm256_fmadd_pd(a5, b0, c5_0);
        c5_1 = _mm256_fmadd_pd(a5, b1, c5_1);

        //Store C after all k-loop updates
        _mm256_store_pd(C + 0 * 8 + 0, c0_0);
        _mm256_store_pd(C + 0 * 8 + 4, c0_1);
        _mm256_store_pd(C + 1 * 8 + 0, c1_0);
        _mm256_store_pd(C + 1 * 8 + 4, c1_1);
        _mm256_store_pd(C + 2 * 8 + 0, c2_0);
        _mm256_store_pd(C + 2 * 8 + 4, c2_1);
        _mm256_store_pd(C + 3 * 8 + 0, c3_0);
        _mm256_store_pd(C + 3 * 8 + 4, c3_1);
        _mm256_store_pd(C + 4 * 8 + 0, c4_0);
        _mm256_store_pd(C + 4 * 8 + 4, c4_1);
        _mm256_store_pd(C + 5 * 8 + 0, c5_0);
        _mm256_store_pd(C + 5 * 8 + 4, c5_1);

        //Update pointers for next iteration
        b_col += 8; 
        a_row += 6;
    }
}
