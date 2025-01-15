Here is an example of a high-performance C function that implements a 6x8 kernel for multiplying a 6x256 matrix ( A ) and a 256x8 matrix ( B ), resulting in a 6x8 output matrix ( C ).
```c
void dgemm_avx2(double const* A, double const *B, double *C) {
    __m256d c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c3_0, c3_1, c4_0, c4_1, c5_0, c5_1;
    __m256d b0, b1;
    __m256d a0, a1;
    
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
    
    const double* a_row = A;
    const double* b_col = B;
    
    for (int k = 0; k < 256; ++k) {
        
        //Load B
        b0 = _mm256_loadu_pd(b_col + 0);
        b1 = _mm256_loadu_pd(b_col + 4);
        
        //6x16 outer product
        a0 = _mm256_set_pd(*a_row, *(a_row+1), *(a_row+2), *(a_row+3));
        a1 = _mm256_set_pd(*(a_row+4), *(a_row+5), *(a_row+6), *(a_row+7));
        
        c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
        c1_0 = _mm256_fmadd_pd(a1, b0, c1_0);
        c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);
        c1_1 = _mm256_fmadd_pd(a1, b1, c1_1);
        
        //Update pointers
        b_col += 8;
        a_row += 6;
    }
    
    //Store C after all k-loop updates
    _mm256_storeu_pd(C + 0 * 8 + 0, c0_0);
    _mm256_storeu_pd(C + 0 * 8 + 4, c0_1);
    _mm256_storeu_pd(C + 1 * 8 + 0, c1_0);
    _mm256_storeu_pd(C + 1 * 8 + 4, c1_1);
    
}
```
Note that this implementation uses the AVX2 vector extensions to perform the matrix-matrix multiplication. Specifically, it uses the `fmadd` instruction to add the dot product of two vectors to a third vector, and the `loadu` and `storeu` instructions to load and store packed double-precision floating-point values.

The function first initializes the output matrix ( C ) with zeros using the `_mm256_setzero_pd` intrinsic. Then it iterates over each column of the input matrices A and B, performing a 6x16 outer product for each column. This involves loading two columns from A and two columns from B using the `loadu` instruction, multiplying them together using the `fmadd` instruction, and accumulating the results in C using the `storeu` instruction.

The function then returns the updated output matrix ( C ).

