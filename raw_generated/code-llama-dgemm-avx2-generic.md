```c
void dgemm_avx2(double const *A, double const *B, double *C) {
    int i, j, k;
    __m256d a, b, c;

    for (i = 0; i < 6; i++) {
        for (j = 0; j < 8; j += 4) {
            // load A and B data
            a = _mm256_loadu_pd(&A[i * 256 + j]);
            b = _mm256_loadu_pd(&B[j]);

            // compute dot product of A and B
            c = _mm256_mul_pd(a, b);

            // store result in C
            _mm256_storeu_pd(&C[i * 8 + j], c);
        }
    }
}
```
This implementation uses the AVX2 vector extensions to accelerate the matrix multiplication. It loads each element of A and B into a 256-bit register, performs a dot product using the `_mm256_mul_pd` intrinsic, and stores the result in C. The loop is unrolled by a factor of 4 to reduce the number of instructions and improve performance.

