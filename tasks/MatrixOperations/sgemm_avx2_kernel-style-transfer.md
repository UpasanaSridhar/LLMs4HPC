

This is an example of a high-performance kernel using single-precision AVX2 instructions to multiply 2 matrices. Use this as a template for writing a similar kernel described below.

## Kernel
```c
void sgemm_avx2(float const* A, float const * B,  float *C)
{
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

        // rank  1 update of C registers
        // 1 register re-used M-times to broadcast column of A
        a0 = _mm256_broadcast_ss(a_ptr + 0);
        c0_0 = _mm256_fmadd_ps(a0, b0, c0_0);
        c0_1 = _mm256_fmadd_ps(a0, b1, c0_1);
        a0 = _mm256_broadcast_ss(a_ptr + 1);
        c1_0 = _mm256_fmadd_ps(a0, b0, c1_0);
        c1_1 = _mm256_fmadd_ps(a0, b1, c1_1);
        a0 = _mm256_broadcast_ss(a_ptr + 2);
        c2_0 = _mm256_fmadd_ps(a0, b0, c2_0);
        c2_1 = _mm256_fmadd_ps(a0, b1, c2_1);
        a0 = _mm256_broadcast_ss(a_ptr + 3);
        c3_0 = _mm256_fmadd_ps(a0, b0, c3_0);
        c3_1 = _mm256_fmadd_ps(a0, b1, c3_1);
        a0 = _mm256_broadcast_ss(a_ptr + 4);
        c4_0 = _mm256_fmadd_ps(a0, b0, c4_0);
        c4_1 = _mm256_fmadd_ps(a0, b1, c4_1);
        a0 = _mm256_broadcast_ss(a_ptr + 5);
        c5_0 = _mm256_fmadd_ps(a0, b0, c5_0);
        c5_1 = _mm256_fmadd_ps(a0, b1, c5_1);

        //Increment pointers
        b_ptr += 16; 
        a_ptr += 6;
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


# Task
Please write a high-performance C function that implements a 6x16 kernel for multiplying a 6x256 matrix \( A \) and a 256x16 matrix \( B \), resulting in a 6x16 output matrix \( C \). 

# Context
Architecture: AVX2 vector extensions. 
Data-type: float 
Function-signature: `sgemm_avx2(double const *A, double const *B, double *C)`. 