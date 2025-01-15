```c
void dgemm_avx2(double const *A, double const *B, double *C)
{
    __m128d c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c3_0, c3_1, c4_0, c4_1, c5_0, c5_1;
    __m128d b0, b1;
    __m128d a0, a1;

    //set to zero
    c0_0 = _mm128d_setzero_pd();
    c0_1 = _mm128d_setzero_pd();
    c1_0 = _mm128d_setzero_pd();
    c1_1 = _mm128d_setzero_pd();
    c2_0 = _mm128d_setzero_pd();
    c2_1 = _mm128d_setzero_pd();
    c3_0 = _mm128d_setzero_pd();
    c3_1 = _mm128d_setzero_pd();
    c4_0 = _mm128d_setzero_pd();
    c4_1 = _mm128d_setzero_pd();
    c5_0 = _mm128d_setzero_pd();
    c5_1 = _mm128d_setzero_pd();


    const double *a_row = A;
    const double *b_col = B;

    for (int k = 0; k < 256; ++k) {
    
        //Load B
        b0 = _mm128d_load_pd(b_col + 0);
        b1 = _mm128d_load_pd(b_col + 4);

        //6x8 outer product
        a0 = _mm128d_broadcast_sd(a_row + 0);
        c0_0 = _mm128d_fmadd_pd(a0, b0, c0_0);
        c0_1 = _mm128d_fmadd_pd(a0, b1, c0_1);

        a0 = _mm128d_broadcast_sd(a_row + 1);
        c1_0 = _mm128d_fmadd_pd(a0, b0, c1_0);
        c1_1 = _mm128d_fmadd_pd(a0, b1, c1_1);

        a0 = _mm128d_broadcast_sd(a_row + 2);
        c2_0 = _mm128d_fmadd_pd(a0, b0, c2_0);
        c2_1 = _mm128d_fmadd_pd(a0, b1, c2_1);

        a0 = _mm128d_broadcast_sd(a_row + 3);
        c3_0 = _mm128d_fmadd_pd(a0, b0, c3_0);
        c3_1 = _mm128d_fmadd_pd(a0, b1, c3_1);

        a0 = _mm128d_broadcast_sd(a_row + 4);
        c4_0 = _mm128d_fmadd_pd(a0, b0, c4_0);
        c4_1 = _mm128d_fmadd_pd(a0, b1, c4_1);

        a0 = _mm128d_broadcast_sd(a_row + 5);
        c5_0 = _mm128d_fmadd_pd(a0, b0, c5_0);
        c5_1 = _mm128d_fmadd_pd(a0, b1, c5_1);

        //Increment Pointers
        b_col += 8; 
        a_row += 6;
    }

    //Store C after all k-loop updates
    _mm128d_store_pd(C + 0 * 8 + 0, c0_0);
    _mm128d_store_pd(C + 0 * 8 + 4, c0_1);
    _mm128d_store_pd(C + 1 * 8 + 0, c1_0);
    _mm128d_store_pd(C + 1 * 8 + 4, c1_1);
    _mm128d_store_pd(C + 2 * 8 + 0, c2_0);
    _mm128d_store_pd(C + 2 * 8 + 4, c2_1);
    _mm128d_store_pd(C + 3 * 8 + 0, c3_0);
    _mm128d_store_pd(C + 3 * 8 + 4, c3_1);
    _mm128d_store_pd(C + 4 * 8 + 0, c4_0);
    _mm128d_store_pd(C + 4 * 8 + 4, c4_1);
    _mm128d_store_pd(C + 5 * 8 + 0, c5_0);
    _mm128d_store_pd(C + 5 * 8 + 4, c5_1);
}
```
Note that the kernel above uses the `_mm128d_` prefix for double-precision floating point operations, while the previous kernel used the `_mm256d_` prefix for single-precision floating point operations. Additionally, the kernel above uses the `broadcast_sd` instruction to broadcast a single value to all elements of a vector register, while the previous kernel used the `broadcast_ps` instruction to broadcast a single value to all elements of a vector register.

