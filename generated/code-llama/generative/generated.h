
#include <immintrin.h>
void dgemm_avx2(double const *A, double const *B, double *C) {
    // Get the number of rows and columns of A and B
    int m = 6;
    int n = 8;
    int k = 256;

    // Calculate the strides for each matrix
    int a_stride = 6 * k;
    int b_stride = k * n;
    int c_stride = m * n;

    // Define the AVX2 registers
    __m256d a, b, c;
    __m256i idx;

    // Loop over each row of C
    for (int i = 0; i < m; i++) {
        // Calculate the address of the first element of the current row of C
        double *c_row = C + i * c_stride;

        // Initialize the AVX2 registers with the values of A and B for the current row
        a = _mm256_loadu_pd(A);
        b = _mm256_loadu_pd(B);

        // Loop over each column of C
        for (int j = 0; j < n; j++) {
            // Calculate the address of the current element of the current row of C
            double *c_ij = c_row + j;

            // Perform the matrix multiplication using AVX2 instructions
            idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            a = _mm256_shuffle_pd(a, idx);
            b = _mm256_shuffle_pd(b, idx);
            c = _mm256_mul_pd(a, b);

            // Store the result in the current element of C
            *c_ij = _mm256_extract_epi32(_mm256_castpd_si256(c), 0);
        }
    }
}
