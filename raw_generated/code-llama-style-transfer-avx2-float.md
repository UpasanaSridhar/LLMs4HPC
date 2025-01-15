
Here is a possible implementation of the `sgemm_avx2` function using AVX2 instructions and the C++ language:
```c
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
```
This implementation uses the `_mm256_load_ps` instruction to load a column of `A` into vector registers, and the `_mm256_fmadd_ps` instruction to perform the dot product and add operations. The `sgemm_avx2` function then processes 16 columns of `B` at a time, using the `for` loop to iterate over each column. The output matrix `C` is stored in memory using the `_mm256_store_ps` instruction.

Note that this implementation assumes that the input matrices are stored in memory in row-major order (i.e., each row of the matrix is contiguous in memory). If the input matrices are stored in column-major order, then the function would need to be modified accordingly.

