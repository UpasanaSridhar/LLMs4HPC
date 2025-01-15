


# Task
Please write a high-performance C function that implements a 6x8 kernel for multiplying a 6x256 matrix \( A \) and a 256x8 matrix \( B \), resulting in a 6x8 output matrix \( C \). 
 
# Context
Architecture: AVX2 vector extensions. 
Data-type: double 
Function-signature: `dgemm_avx2(double const *A, double const *B, double *C)`. 

## Kernel
This is an example of a high-performance kernel using single-precision AVX2 instructions to multiply 2 matrices. Use this as a template for writing a similar kernel described below.

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


    const float *a_row = A;
    const float *b_col = B;

    for (int k = 0; k < 256; ++k) {
    
        //Load B
        b0 = _mm256_load_ps(b_col + 0);
        b1 = _mm256_load_ps(b_col + 8);

        //6x16 outer product
        a0 = _mm256_broadcast_ss(a_row + 0);
        c0_0 = _mm256_fmadd_ps(a0, b0, c0_0);
        c0_1 = _mm256_fmadd_ps(a0, b1, c0_1);

        a1 = _mm256_broadcast_ss(a_row + 1);
        c1_0 = _mm256_fmadd_ps(a1, b0, c1_0);
        c1_1 = _mm256_fmadd_ps(a1, b1, c1_1);

        a0 = _mm256_broadcast_ss(a_row + 2);
        c2_0 = _mm256_fmadd_ps(a0, b0, c2_0);
        c2_1 = _mm256_fmadd_ps(a0, b1, c2_1);

        a1 = _mm256_broadcast_ss(a_row + 3);
        c3_0 = _mm256_fmadd_ps(a1, b0, c3_0);
        c3_1 = _mm256_fmadd_ps(a1, b1, c3_1);

        a0 = _mm256_broadcast_ss(a_row + 4);
        c4_0 = _mm256_fmadd_ps(a0, b0, c4_0);
        c4_1 = _mm256_fmadd_ps(a0, b1, c4_1);

        a1 = _mm256_broadcast_ss(a_row + 5);
        c5_0 = _mm256_fmadd_ps(a1, b0, c5_0);
        c5_1 = _mm256_fmadd_ps(a1, b1, c5_1);

        //Increment Pointers
        b_col += 16; 
        a_row += 6;
    }

    //Store C after all k-loop updates
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

# Challenge
Use the sample kernel above as an example an write the dgemm_avx2 funtion below
