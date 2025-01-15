#include <immintrin.h>

// void sgemm_avx2(const float *A, const float *B, float *C)
// {
//     __m256 c00, c01, c10, c11, c20, c21, c30, c31;
//     __m256 c40, c41, c50, c51;
//     __m256 b0, b1;
//     __m256 a0, a1, a2, a3, a4, a5;

//     // // set to zero
//     c00 = _mm256_setzero_ps();
//     c01 = _mm256_setzero_ps();
//     c10 = _mm256_setzero_ps();
//     c11 = _mm256_setzero_ps();
//     c20 = _mm256_setzero_ps();
//     c21 = _mm256_setzero_ps();
//     c30 = _mm256_setzero_ps();
//     c31 = _mm256_setzero_ps();
//     c40 = _mm256_setzero_ps();
//     c41 = _mm256_setzero_ps();
//     c50 = _mm256_setzero_ps();
//     c51 = _mm256_setzero_ps();

//     const float *a_ptr = A;
//     const float *b_ptr = B;

//     for (int k = 0; k < 256; ++k)
//     {
//         b0 = _mm256_loadu_ps(b_ptr);
//         b1 = _mm256_loadu_ps(b_ptr + 8);

//         a0 = _mm256_broadcast_ss(a_ptr + 0);
//         c00 = _mm256_fmadd_ps(a0, b0, c00);
//         c01 = _mm256_fmadd_ps(a0, b1, c01);

//         a1 = _mm256_broadcast_ss(a_ptr + 1);
//         c10 = _mm256_fmadd_ps(a1, b0, c10);
//         c11 = _mm256_fmadd_ps(a1, b1, c11);

//         a2 = _mm256_broadcast_ss(a_ptr + 2);
//         c20 = _mm256_fmadd_ps(a2, b0, c20);
//         c21 = _mm256_fmadd_ps(a2, b1, c21);

//         a3 = _mm256_broadcast_ss(a_ptr + 3);
//         c30 = _mm256_fmadd_ps(a3, b0, c30);
//         c31 = _mm256_fmadd_ps(a3, b1, c31);

//         a4 = _mm256_broadcast_ss(a_ptr + 4);
//         c40 = _mm256_fmadd_ps(a4, b0, c40);
//         c41 = _mm256_fmadd_ps(a4, b1, c41);

//         a5 = _mm256_broadcast_ss(a_ptr + 5);
//         c50 = _mm256_fmadd_ps(a5, b0, c50);
//         c51 = _mm256_fmadd_ps(a5, b1, c51);

//         a_ptr += 6;
//         b_ptr += 16;
//     }

//     // Store the results
//     // _mm256_storeu_ps(C + 0 * 16 + 0, c00);
//     // _mm256_storeu_ps(C + 0 * 16 + 4, c01);
//     // _mm256_storeu_ps(C + 1 * 16 + 0, c10);
//     // _mm256_storeu_ps(C + 1 * 16 + 4, c11);
//     // _mm256_storeu_ps(C + 2 * 16 + 0, c20);
//     // _mm256_storeu_ps(C + 2 * 16 + 4, c21);
//     // _mm256_storeu_ps(C + 3 * 16 + 0, c30);
//     // _mm256_storeu_ps(C + 3 * 16 + 4, c31);
//     _mm256_storeu_ps(C + 0 * 16 + 0, c00);
//     _mm256_storeu_ps(C + 0 * 16 + 8, c01);
//     _mm256_storeu_ps(C + 1 * 16 + 0, c10);
//     _mm256_storeu_ps(C + 1 * 16 + 8, c11);
//     _mm256_storeu_ps(C + 2 * 16 + 0, c20);
//     _mm256_storeu_ps(C + 2 * 16 + 8, c21);
//     _mm256_storeu_ps(C + 3 * 16 + 0, c30);
//     _mm256_storeu_ps(C + 3 * 16 + 8, c31);

//     _mm256_storeu_ps(C + 4 * 16 + 0, c40);
//     _mm256_storeu_ps(C + 4 * 16 + 8, c41);
//     _mm256_storeu_ps(C + 5 * 16 + 0, c50);
//     _mm256_storeu_ps(C + 5 * 16 + 8, c51);
// }

// Purely generative prompt
// void dgemm_avx2(double const *A, double const *B, double *C) {
//     int i, j, p;
//     const int M = 6;
//     const int N = 8;
//     const int K = 256;
//     for (i = 0; i < M; ++i) {
//         for (j = 0; j < N; ++j) {
//             C[i*N+j] = 0.0;
//             for (p = 0; p < K; ++p) {
//                 C[i*N+j] += A[i*K+p] * B[p*N+j];
//             }
//         }
//     }
// }

// From generative prompt
// void dgemm_avx2(double const *A, double const *B, double *C) {
//     int i, j, k;

//     // Iterate over the rows of A and B in chunks of 8
//     for (i = 0; i < 6; i += 8) {
//         // Iterate over the columns of A and B in chunks of 4
//         for (j = 0; j < 256; j += 4) {
//             // Compute the dot product of the current row and column
//             __m256d a_vec, b_vec, c_vec;
//             a_vec = _mm256_loadu_pd(A + i);
//             b_vec = _mm256_loadu_pd(B + j);
//             c_vec = _mm256_set1_pd(0.0);
//             for (k = 0; k < 8; k++) {
//                 c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
//                 a_vec = _mm256_loadu_pd(A + i + k * 8);
//                 b_vec = _mm256_loadu_pd(B + j + k * 4);
//             }

//             // Store the result in C
//             _mm256_storeu_pd(C + i * 256 + j, c_vec);
//         }
//     }
// }

// // From Scaffolded prompt
// void dgemm_avx2(double const *A, double const *B, double *C) {
//     int m = 6; // number of rows in A and C
//     int n = 8; // number of columns in B and C
//     int k = 256; // number of columns in A and rows in B

//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j += 4) {
//             __m256d C_vec = _mm256_setzero_pd();
//             for (int p = 0; p < k; p++) {
//                 __m256d A_vec = _mm256_loadu_pd(A + i * k + p);
//                 __m256d B_vec = _mm256_loadu_pd(B + j + p * n);
//                 C_vec = _mm256_fmadd_pd(A_vec, B_vec, C_vec);
//             }
//             _mm256_storeu_pd(C + i * n + j, C_vec);
//         }
//     }
// }

// From style-transfer prompt
// void dgemm_avx2(double const* A, double const *B, double *C) {
//     __m256d c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c3_0, c3_1, c4_0, c4_1, c5_0, c5_1;
//     __m256d b0, b1;
//     __m256d a0;

//     // Set to zero
//     c0_0 = _mm256_setzero_pd();
//     c0_1 = _mm256_setzero_pd();
//     c1_0 = _mm256_setzero_pd();
//     c1_1 = _mm256_setzero_pd();
//     c2_0 = _mm256_setzero_pd();
//     c2_1 = _mm256_setzero_pd();
//     c3_0 = _mm256_setzero_pd();
//     c3_1 = _mm256_setzero_pd();
//     c4_0 = _mm256_setzero_pd();
//     c4_1 = _mm256_setzero_pd();
//     c5_0 = _mm256_setzero_pd();
//     c5_1 = _mm256_setzero_pd();

//     const double *a_ptr = A;
//     const double *b_ptr = B;
//     for (int k = 0; k < 256; ++k) {
//         b0 = _mm256_load_pd(b_ptr + 0);
//         b1 = _mm256_load_pd(b_ptr + 4); // This indexing was wrong

//         a0 = _mm256_broadcast_sd(a_ptr + 0);

//         c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
//         c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);

//         a0 = _mm256_broadcast_sd(a_ptr + 1);

//         c1_0 = _mm256_fmadd_pd(a0, b0, c1_0);
//         c1_1 = _mm256_fmadd_pd(a0, b1, c1_1);

//         a0 = _mm256_broadcast_sd(a_ptr + 2);

//         c2_0 = _mm256_fmadd_pd(a0, b0, c2_0);
//         c2_1 = _mm256_fmadd_pd(a0, b1, c2_1);

//         a0 = _mm256_broadcast_sd(a_ptr + 3);

//         c3_0 = _mm256_fmadd_pd(a0, b0, c3_0);
//         c3_1 = _mm256_fmadd_pd(a0, b1, c3_1);

//         a0 = _mm256_broadcast_sd(a_ptr + 4);

//         c4_0 = _mm256_fmadd_pd(a0, b0, c4_0);
//         c4_1 = _mm256_fmadd_pd(a0, b1, c4_1);

//         a0 = _mm256_broadcast_sd(a_ptr + 5);

//         c5_0 = _mm256_fmadd_pd(a0, b0, c5_0);
//         c5_1 = _mm256_fmadd_pd(a0, b1, c5_1);

//         a_ptr += 6;
//         b_ptr += 8;// This was not included in the original code
//     }

//     // Stores of C
//     _mm256_storeu_pd(C + 0 * 8 + 0, c0_0);
//     _mm256_storeu_pd(C + 0 * 8 + 4, c0_1);
//     _mm256_storeu_pd(C + 1 * 8 + 0, c1_0);
//     _mm256_storeu_pd(C + 1 * 8 + 4, c1_1);
//     _mm256_storeu_pd(C + 2 * 8 + 0, c2_0);
//     _mm256_storeu_pd(C + 2 * 8 + 4, c2_1);
//     _mm256_storeu_pd(C + 3 * 8 + 0, c3_0);
//     _mm256_storeu_pd(C + 3 * 8 + 4, c3_1);
//     _mm256_storeu_pd(C + 4 * 8 + 0, c4_0);
//     _mm256_storeu_pd(C + 4 * 8 + 4, c4_1);
//     _mm256_storeu_pd(C + 5 * 8 + 0, c5_0);
//     _mm256_storeu_pd(C + 5 * 8 + 4, c5_1);
// }

// from table filling+code-generated

// dgemm_avx2 kernel
// M: 6, N: 8, unroll_factor: 1
// K must be a multiple of 1
// computes C += A(mxk) * B(kxn)
// A is column major, B is row major, C is row major
void dgemm_avx2(const double *A, const double *B, double *C)
{
    // define register variables
    __m256d c_0_0;
    __m256d c_0_1;
    __m256d c_1_0;
    __m256d c_1_1;
    __m256d c_2_0;
    __m256d c_2_1;
    __m256d c_3_0;
    __m256d c_3_1;
    __m256d c_4_0;
    __m256d c_4_1;
    __m256d c_5_0;
    __m256d c_5_1;
    __m256d b_0;
    __m256d b_1;
    __m256d a_i;

    // load values into C registers
    c_0_0 = _mm256_setzero_pd();
    c_0_1 = _mm256_setzero_pd();
    c_1_0 = _mm256_setzero_pd();
    c_1_1 = _mm256_setzero_pd();
    c_2_0 = _mm256_setzero_pd();
    c_2_1 = _mm256_setzero_pd();
    c_3_0 = _mm256_setzero_pd();
    c_3_1 = _mm256_setzero_pd();
    c_4_0 = _mm256_setzero_pd();
    c_4_1 = _mm256_setzero_pd();
    c_5_0 = _mm256_setzero_pd();
    c_5_1 = _mm256_setzero_pd();

    // init vector pointers
    const double *a_ptr = A;
    const double *b_ptr = B;

    // outer product loop over K, unrolled by 1
    for (int k = 0; k < 256; k += 1)
    {
        // load values into B registers

        b_0 = _mm256_load_pd(b_ptr + 0);
        b_1 = _mm256_load_pd(b_ptr + 4);

        a_i = _mm256_broadcast_sd(a_ptr + 0 * 1);
        c_0_0 = _mm256_fmadd_pd(a_i, b_0, c_0_0);
        c_0_1 = _mm256_fmadd_pd(a_i, b_1, c_0_1);
        a_i = _mm256_broadcast_sd(a_ptr + 1 * 1);
        c_1_0 = _mm256_fmadd_pd(a_i, b_0, c_1_0);
        c_1_1 = _mm256_fmadd_pd(a_i, b_1, c_1_1);
        a_i = _mm256_broadcast_sd(a_ptr + 2 * 1);
        c_2_0 = _mm256_fmadd_pd(a_i, b_0, c_2_0);
        c_2_1 = _mm256_fmadd_pd(a_i, b_1, c_2_1);
        a_i = _mm256_broadcast_sd(a_ptr + 3 * 1);
        c_3_0 = _mm256_fmadd_pd(a_i, b_0, c_3_0);
        c_3_1 = _mm256_fmadd_pd(a_i, b_1, c_3_1);
        a_i = _mm256_broadcast_sd(a_ptr + 4 * 1);
        c_4_0 = _mm256_fmadd_pd(a_i, b_0, c_4_0);
        c_4_1 = _mm256_fmadd_pd(a_i, b_1, c_4_1);
        a_i = _mm256_broadcast_sd(a_ptr + 5 * 1);
        c_5_0 = _mm256_fmadd_pd(a_i, b_0, c_5_0);
        c_5_1 = _mm256_fmadd_pd(a_i, b_1, c_5_1);

        a_ptr += 6;
        b_ptr += 8;
    }

    // store C registers back to memory
    _mm256_store_pd(C + 0 * 8 + 0, c_0_0);
    _mm256_store_pd(C + 0 * 8 + 4, c_0_1);
    _mm256_store_pd(C + 1 * 8 + 0, c_1_0);
    _mm256_store_pd(C + 1 * 8 + 4, c_1_1);
    _mm256_store_pd(C + 2 * 8 + 0, c_2_0);
    _mm256_store_pd(C + 2 * 8 + 4, c_2_1);
    _mm256_store_pd(C + 3 * 8 + 0, c_3_0);
    _mm256_store_pd(C + 3 * 8 + 4, c_3_1);
    _mm256_store_pd(C + 4 * 8 + 0, c_4_0);
    _mm256_store_pd(C + 4 * 8 + 4, c_4_1);
    _mm256_store_pd(C + 5 * 8 + 0, c_5_0);
    _mm256_store_pd(C + 5 * 8 + 4, c_5_1);
}

// sgemm_avx512 kernel
// M: 8, N: 64, unroll_factor: 1
// K must be a multiple of 1
// computes C += A(mxk) * B(kxn)
// A is column major, B is row major, C is row major
void sgemm_avx512(const float *A, const float *B, float *C)
{
    // define register variables
    __m512 c_0_0;
    __m512 c_0_1;
    __m512 c_0_2;
    __m512 c_0_3;
    __m512 c_1_0;
    __m512 c_1_1;
    __m512 c_1_2;
    __m512 c_1_3;
    __m512 c_2_0;
    __m512 c_2_1;
    __m512 c_2_2;
    __m512 c_2_3;
    __m512 c_3_0;
    __m512 c_3_1;
    __m512 c_3_2;
    __m512 c_3_3;
    __m512 c_4_0;
    __m512 c_4_1;
    __m512 c_4_2;
    __m512 c_4_3;
    __m512 c_5_0;
    __m512 c_5_1;
    __m512 c_5_2;
    __m512 c_5_3;
    __m512 c_6_0;
    __m512 c_6_1;
    __m512 c_6_2;
    __m512 c_6_3;
    __m512 c_7_0;
    __m512 c_7_1;
    __m512 c_7_2;
    __m512 c_7_3;
    __m512 b_0;
    __m512 b_1;
    __m512 b_2;
    __m512 b_3;
    __m512 a_i;

    // load values into C registers
    c_0_0 = _mm512_setzero_ps();
    c_0_1 = _mm512_setzero_ps();
    c_0_2 = _mm512_setzero_ps();
    c_0_3 = _mm512_setzero_ps();
    c_1_0 = _mm512_setzero_ps();
    c_1_1 = _mm512_setzero_ps();
    c_1_2 = _mm512_setzero_ps();
    c_1_3 = _mm512_setzero_ps();
    c_2_0 = _mm512_setzero_ps();
    c_2_1 = _mm512_setzero_ps();
    c_2_2 = _mm512_setzero_ps();
    c_2_3 = _mm512_setzero_ps();
    c_3_0 = _mm512_setzero_ps();
    c_3_1 = _mm512_setzero_ps();
    c_3_2 = _mm512_setzero_ps();
    c_3_3 = _mm512_setzero_ps();
    c_4_0 = _mm512_setzero_ps();
    c_4_1 = _mm512_setzero_ps();
    c_4_2 = _mm512_setzero_ps();
    c_4_3 = _mm512_setzero_ps();
    c_5_0 = _mm512_setzero_ps();
    c_5_1 = _mm512_setzero_ps();
    c_5_2 = _mm512_setzero_ps();
    c_5_3 = _mm512_setzero_ps();
    c_6_0 = _mm512_setzero_ps();
    c_6_1 = _mm512_setzero_ps();
    c_6_2 = _mm512_setzero_ps();
    c_6_3 = _mm512_setzero_ps();
    c_7_0 = _mm512_setzero_ps();
    c_7_1 = _mm512_setzero_ps();
    c_7_2 = _mm512_setzero_ps();
    c_7_3 = _mm512_setzero_ps();

    // init vector pointers
    const float *a_ptr = A;
    const float *b_ptr = B;

    // outer product loop over K, unrolled by 1
    for (int k = 0; k < 512; k += 1)
    {
        // load values into B registers

        b_0 = _mm512_load_ps(b_ptr + 0);
        b_1 = _mm512_load_ps(b_ptr + 16);
        b_2 = _mm512_load_ps(b_ptr + 32);
        b_3 = _mm512_load_ps(b_ptr + 48);

        a_i = _mm512_set1_ps(*(a_ptr + 0 * 1));
        c_0_0 = _mm512_fmadd_ps(a_i, b_0, c_0_0);
        c_0_1 = _mm512_fmadd_ps(a_i, b_1, c_0_1);
        c_0_2 = _mm512_fmadd_ps(a_i, b_2, c_0_2);
        c_0_3 = _mm512_fmadd_ps(a_i, b_3, c_0_3);
        a_i = _mm512_set1_ps(*(a_ptr + 1 * 1));
        c_1_0 = _mm512_fmadd_ps(a_i, b_0, c_1_0);
        c_1_1 = _mm512_fmadd_ps(a_i, b_1, c_1_1);
        c_1_2 = _mm512_fmadd_ps(a_i, b_2, c_1_2);
        c_1_3 = _mm512_fmadd_ps(a_i, b_3, c_1_3);
        a_i = _mm512_set1_ps(*(a_ptr + 2 * 1));
        c_2_0 = _mm512_fmadd_ps(a_i, b_0, c_2_0);
        c_2_1 = _mm512_fmadd_ps(a_i, b_1, c_2_1);
        c_2_2 = _mm512_fmadd_ps(a_i, b_2, c_2_2);
        c_2_3 = _mm512_fmadd_ps(a_i, b_3, c_2_3);
        a_i = _mm512_set1_ps(*(a_ptr + 3 * 1));
        c_3_0 = _mm512_fmadd_ps(a_i, b_0, c_3_0);
        c_3_1 = _mm512_fmadd_ps(a_i, b_1, c_3_1);
        c_3_2 = _mm512_fmadd_ps(a_i, b_2, c_3_2);
        c_3_3 = _mm512_fmadd_ps(a_i, b_3, c_3_3);
        a_i = _mm512_set1_ps(*(a_ptr + 4 * 1));
        c_4_0 = _mm512_fmadd_ps(a_i, b_0, c_4_0);
        c_4_1 = _mm512_fmadd_ps(a_i, b_1, c_4_1);
        c_4_2 = _mm512_fmadd_ps(a_i, b_2, c_4_2);
        c_4_3 = _mm512_fmadd_ps(a_i, b_3, c_4_3);
        a_i = _mm512_set1_ps(*(a_ptr + 5 * 1));
        c_5_0 = _mm512_fmadd_ps(a_i, b_0, c_5_0);
        c_5_1 = _mm512_fmadd_ps(a_i, b_1, c_5_1);
        c_5_2 = _mm512_fmadd_ps(a_i, b_2, c_5_2);
        c_5_3 = _mm512_fmadd_ps(a_i, b_3, c_5_3);
        a_i = _mm512_set1_ps(*(a_ptr + 6 * 1));
        c_6_0 = _mm512_fmadd_ps(a_i, b_0, c_6_0);
        c_6_1 = _mm512_fmadd_ps(a_i, b_1, c_6_1);
        c_6_2 = _mm512_fmadd_ps(a_i, b_2, c_6_2);
        c_6_3 = _mm512_fmadd_ps(a_i, b_3, c_6_3);
        a_i = _mm512_set1_ps(*(a_ptr + 7 * 1));
        c_7_0 = _mm512_fmadd_ps(a_i, b_0, c_7_0);
        c_7_1 = _mm512_fmadd_ps(a_i, b_1, c_7_1);
        c_7_2 = _mm512_fmadd_ps(a_i, b_2, c_7_2);
        c_7_3 = _mm512_fmadd_ps(a_i, b_3, c_7_3);

        a_ptr += 8;
        b_ptr += 64;
    }

    // store C registers back to memory
    _mm512_store_ps(C + 0 * 64 + 0, c_0_0);
    _mm512_store_ps(C + 0 * 64 + 16, c_0_1);
    _mm512_store_ps(C + 0 * 64 + 32, c_0_2);
    _mm512_store_ps(C + 0 * 64 + 48, c_0_3);
    _mm512_store_ps(C + 1 * 64 + 0, c_1_0);
    _mm512_store_ps(C + 1 * 64 + 16, c_1_1);
    _mm512_store_ps(C + 1 * 64 + 32, c_1_2);
    _mm512_store_ps(C + 1 * 64 + 48, c_1_3);
    _mm512_store_ps(C + 2 * 64 + 0, c_2_0);
    _mm512_store_ps(C + 2 * 64 + 16, c_2_1);
    _mm512_store_ps(C + 2 * 64 + 32, c_2_2);
    _mm512_store_ps(C + 2 * 64 + 48, c_2_3);
    _mm512_store_ps(C + 3 * 64 + 0, c_3_0);
    _mm512_store_ps(C + 3 * 64 + 16, c_3_1);
    _mm512_store_ps(C + 3 * 64 + 32, c_3_2);
    _mm512_store_ps(C + 3 * 64 + 48, c_3_3);
    _mm512_store_ps(C + 4 * 64 + 0, c_4_0);
    _mm512_store_ps(C + 4 * 64 + 16, c_4_1);
    _mm512_store_ps(C + 4 * 64 + 32, c_4_2);
    _mm512_store_ps(C + 4 * 64 + 48, c_4_3);
    _mm512_store_ps(C + 5 * 64 + 0, c_5_0);
    _mm512_store_ps(C + 5 * 64 + 16, c_5_1);
    _mm512_store_ps(C + 5 * 64 + 32, c_5_2);
    _mm512_store_ps(C + 5 * 64 + 48, c_5_3);
    _mm512_store_ps(C + 6 * 64 + 0, c_6_0);
    _mm512_store_ps(C + 6 * 64 + 16, c_6_1);
    _mm512_store_ps(C + 6 * 64 + 32, c_6_2);
    _mm512_store_ps(C + 6 * 64 + 48, c_6_3);
    _mm512_store_ps(C + 7 * 64 + 0, c_7_0);
    _mm512_store_ps(C + 7 * 64 + 16, c_7_1);
    _mm512_store_ps(C + 7 * 64 + 32, c_7_2);
    _mm512_store_ps(C + 7 * 64 + 48, c_7_3);
}

// dgemm_avx512 kernel
// M: 8, N: 32, unroll_factor: 1
// K must be a multiple of 1
// computes C += A(mxk) * B(kxn)
// A is column major, B is row major, C is row major
void dgemm_avx512(const double *A, const double *B, double *C)
{
    // define register variables
    __m512d c_0_0;
    __m512d c_0_1;
    __m512d c_0_2;
    __m512d c_0_3;
    __m512d c_1_0;
    __m512d c_1_1;
    __m512d c_1_2;
    __m512d c_1_3;
    __m512d c_2_0;
    __m512d c_2_1;
    __m512d c_2_2;
    __m512d c_2_3;
    __m512d c_3_0;
    __m512d c_3_1;
    __m512d c_3_2;
    __m512d c_3_3;
    __m512d c_4_0;
    __m512d c_4_1;
    __m512d c_4_2;
    __m512d c_4_3;
    __m512d c_5_0;
    __m512d c_5_1;
    __m512d c_5_2;
    __m512d c_5_3;
    __m512d c_6_0;
    __m512d c_6_1;
    __m512d c_6_2;
    __m512d c_6_3;
    __m512d c_7_0;
    __m512d c_7_1;
    __m512d c_7_2;
    __m512d c_7_3;
    __m512d b_0;
    __m512d b_1;
    __m512d b_2;
    __m512d b_3;
    __m512d a_i;

    // load values into C registers
    c_0_0 = _mm512_setzero_pd();
    c_0_1 = _mm512_setzero_pd();
    c_0_2 = _mm512_setzero_pd();
    c_0_3 = _mm512_setzero_pd();
    c_1_0 = _mm512_setzero_pd();
    c_1_1 = _mm512_setzero_pd();
    c_1_2 = _mm512_setzero_pd();
    c_1_3 = _mm512_setzero_pd();
    c_2_0 = _mm512_setzero_pd();
    c_2_1 = _mm512_setzero_pd();
    c_2_2 = _mm512_setzero_pd();
    c_2_3 = _mm512_setzero_pd();
    c_3_0 = _mm512_setzero_pd();
    c_3_1 = _mm512_setzero_pd();
    c_3_2 = _mm512_setzero_pd();
    c_3_3 = _mm512_setzero_pd();
    c_4_0 = _mm512_setzero_pd();
    c_4_1 = _mm512_setzero_pd();
    c_4_2 = _mm512_setzero_pd();
    c_4_3 = _mm512_setzero_pd();
    c_5_0 = _mm512_setzero_pd();
    c_5_1 = _mm512_setzero_pd();
    c_5_2 = _mm512_setzero_pd();
    c_5_3 = _mm512_setzero_pd();
    c_6_0 = _mm512_setzero_pd();
    c_6_1 = _mm512_setzero_pd();
    c_6_2 = _mm512_setzero_pd();
    c_6_3 = _mm512_setzero_pd();
    c_7_0 = _mm512_setzero_pd();
    c_7_1 = _mm512_setzero_pd();
    c_7_2 = _mm512_setzero_pd();
    c_7_3 = _mm512_setzero_pd();

    // init vector pointers
    const double *a_ptr = A;
    const double *b_ptr = B;

    // outer product loop over K, unrolled by 1
    for (int k = 0; k < 256; k += 1)
    {
        // load values into B registers

        b_0 = _mm512_load_pd(b_ptr + 0);
        b_1 = _mm512_load_pd(b_ptr + 8);
        b_2 = _mm512_load_pd(b_ptr + 16);
        b_3 = _mm512_load_pd(b_ptr + 24);

        a_i = _mm512_set1_pd(*(a_ptr + 0 * 1));
        c_0_0 = _mm512_fmadd_pd(a_i, b_0, c_0_0);
        c_0_1 = _mm512_fmadd_pd(a_i, b_1, c_0_1);
        c_0_2 = _mm512_fmadd_pd(a_i, b_2, c_0_2);
        c_0_3 = _mm512_fmadd_pd(a_i, b_3, c_0_3);
        a_i = _mm512_set1_pd(*(a_ptr + 1 * 1));
        c_1_0 = _mm512_fmadd_pd(a_i, b_0, c_1_0);
        c_1_1 = _mm512_fmadd_pd(a_i, b_1, c_1_1);
        c_1_2 = _mm512_fmadd_pd(a_i, b_2, c_1_2);
        c_1_3 = _mm512_fmadd_pd(a_i, b_3, c_1_3);
        a_i = _mm512_set1_pd(*(a_ptr + 2 * 1));
        c_2_0 = _mm512_fmadd_pd(a_i, b_0, c_2_0);
        c_2_1 = _mm512_fmadd_pd(a_i, b_1, c_2_1);
        c_2_2 = _mm512_fmadd_pd(a_i, b_2, c_2_2);
        c_2_3 = _mm512_fmadd_pd(a_i, b_3, c_2_3);
        a_i = _mm512_set1_pd(*(a_ptr + 3 * 1));
        c_3_0 = _mm512_fmadd_pd(a_i, b_0, c_3_0);
        c_3_1 = _mm512_fmadd_pd(a_i, b_1, c_3_1);
        c_3_2 = _mm512_fmadd_pd(a_i, b_2, c_3_2);
        c_3_3 = _mm512_fmadd_pd(a_i, b_3, c_3_3);
        a_i = _mm512_set1_pd(*(a_ptr + 4 * 1));
        c_4_0 = _mm512_fmadd_pd(a_i, b_0, c_4_0);
        c_4_1 = _mm512_fmadd_pd(a_i, b_1, c_4_1);
        c_4_2 = _mm512_fmadd_pd(a_i, b_2, c_4_2);
        c_4_3 = _mm512_fmadd_pd(a_i, b_3, c_4_3);
        a_i = _mm512_set1_pd(*(a_ptr + 5 * 1));
        c_5_0 = _mm512_fmadd_pd(a_i, b_0, c_5_0);
        c_5_1 = _mm512_fmadd_pd(a_i, b_1, c_5_1);
        c_5_2 = _mm512_fmadd_pd(a_i, b_2, c_5_2);
        c_5_3 = _mm512_fmadd_pd(a_i, b_3, c_5_3);
        a_i = _mm512_set1_pd(*(a_ptr + 6 * 1));
        c_6_0 = _mm512_fmadd_pd(a_i, b_0, c_6_0);
        c_6_1 = _mm512_fmadd_pd(a_i, b_1, c_6_1);
        c_6_2 = _mm512_fmadd_pd(a_i, b_2, c_6_2);
        c_6_3 = _mm512_fmadd_pd(a_i, b_3, c_6_3);
        a_i = _mm512_set1_pd(*(a_ptr + 7 * 1));
        c_7_0 = _mm512_fmadd_pd(a_i, b_0, c_7_0);
        c_7_1 = _mm512_fmadd_pd(a_i, b_1, c_7_1);
        c_7_2 = _mm512_fmadd_pd(a_i, b_2, c_7_2);
        c_7_3 = _mm512_fmadd_pd(a_i, b_3, c_7_3);

        a_ptr += 8;
        b_ptr += 32;
    }

    // store C registers back to memory
    _mm512_store_pd(C + 0 * 32 + 0, c_0_0);
    _mm512_store_pd(C + 0 * 32 + 8, c_0_1);
    _mm512_store_pd(C + 0 * 32 + 16, c_0_2);
    _mm512_store_pd(C + 0 * 32 + 24, c_0_3);
    _mm512_store_pd(C + 1 * 32 + 0, c_1_0);
    _mm512_store_pd(C + 1 * 32 + 8, c_1_1);
    _mm512_store_pd(C + 1 * 32 + 16, c_1_2);
    _mm512_store_pd(C + 1 * 32 + 24, c_1_3);
    _mm512_store_pd(C + 2 * 32 + 0, c_2_0);
    _mm512_store_pd(C + 2 * 32 + 8, c_2_1);
    _mm512_store_pd(C + 2 * 32 + 16, c_2_2);
    _mm512_store_pd(C + 2 * 32 + 24, c_2_3);
    _mm512_store_pd(C + 3 * 32 + 0, c_3_0);
    _mm512_store_pd(C + 3 * 32 + 8, c_3_1);
    _mm512_store_pd(C + 3 * 32 + 16, c_3_2);
    _mm512_store_pd(C + 3 * 32 + 24, c_3_3);
    _mm512_store_pd(C + 4 * 32 + 0, c_4_0);
    _mm512_store_pd(C + 4 * 32 + 8, c_4_1);
    _mm512_store_pd(C + 4 * 32 + 16, c_4_2);
    _mm512_store_pd(C + 4 * 32 + 24, c_4_3);
    _mm512_store_pd(C + 5 * 32 + 0, c_5_0);
    _mm512_store_pd(C + 5 * 32 + 8, c_5_1);
    _mm512_store_pd(C + 5 * 32 + 16, c_5_2);
    _mm512_store_pd(C + 5 * 32 + 24, c_5_3);
    _mm512_store_pd(C + 6 * 32 + 0, c_6_0);
    _mm512_store_pd(C + 6 * 32 + 8, c_6_1);
    _mm512_store_pd(C + 6 * 32 + 16, c_6_2);
    _mm512_store_pd(C + 6 * 32 + 24, c_6_3);
    _mm512_store_pd(C + 7 * 32 + 0, c_7_0);
    _mm512_store_pd(C + 7 * 32 + 8, c_7_1);
    _mm512_store_pd(C + 7 * 32 + 16, c_7_2);
    _mm512_store_pd(C + 7 * 32 + 24, c_7_3);
}

// sgemm_avx2 kernel
// M: 6, N: 16, unroll_factor: 1
// K must be a multiple of 1
// computes C += A(mxk) * B(kxn)
// A is column major, B is row major, C is row major
void sgemm_avx2(const float *A, const float *B, float *C)
{
    // define register variables
    __m256 c_0_0;
    __m256 c_0_1;
    __m256 c_1_0;
    __m256 c_1_1;
    __m256 c_2_0;
    __m256 c_2_1;
    __m256 c_3_0;
    __m256 c_3_1;
    __m256 c_4_0;
    __m256 c_4_1;
    __m256 c_5_0;
    __m256 c_5_1;
    __m256 b_0;
    __m256 b_1;
    __m256 a_i;

    // load values into C registers
    c_0_0 = _mm256_setzero_ps();
    c_0_1 = _mm256_setzero_ps();
    c_1_0 = _mm256_setzero_ps();
    c_1_1 = _mm256_setzero_ps();
    c_2_0 = _mm256_setzero_ps();
    c_2_1 = _mm256_setzero_ps();
    c_3_0 = _mm256_setzero_ps();
    c_3_1 = _mm256_setzero_ps();
    c_4_0 = _mm256_setzero_ps();
    c_4_1 = _mm256_setzero_ps();
    c_5_0 = _mm256_setzero_ps();
    c_5_1 = _mm256_setzero_ps();

    // init vector pointers
    const float *a_ptr = A;
    const float *b_ptr = B;

    // outer product loop over K, unrolled by 1
    for (int k = 0; k < 256; k += 1)
    {
        // load values into B registers

        b_0 = _mm256_load_ps(b_ptr + 0);
        b_1 = _mm256_load_ps(b_ptr + 8);

        a_i = _mm256_set1_ps(*(a_ptr + 0 * 1));
        c_0_0 = _mm256_fmadd_ps(a_i, b_0, c_0_0);
        c_0_1 = _mm256_fmadd_ps(a_i, b_1, c_0_1);
        a_i = _mm256_set1_ps(*(a_ptr + 1 * 1));
        c_1_0 = _mm256_fmadd_ps(a_i, b_0, c_1_0);
        c_1_1 = _mm256_fmadd_ps(a_i, b_1, c_1_1);
        a_i = _mm256_set1_ps(*(a_ptr + 2 * 1));
        c_2_0 = _mm256_fmadd_ps(a_i, b_0, c_2_0);
        c_2_1 = _mm256_fmadd_ps(a_i, b_1, c_2_1);
        a_i = _mm256_set1_ps(*(a_ptr + 3 * 1));
        c_3_0 = _mm256_fmadd_ps(a_i, b_0, c_3_0);
        c_3_1 = _mm256_fmadd_ps(a_i, b_1, c_3_1);
        a_i = _mm256_set1_ps(*(a_ptr + 4 * 1));
        c_4_0 = _mm256_fmadd_ps(a_i, b_0, c_4_0);
        c_4_1 = _mm256_fmadd_ps(a_i, b_1, c_4_1);
        a_i = _mm256_set1_ps(*(a_ptr + 5 * 1));
        c_5_0 = _mm256_fmadd_ps(a_i, b_0, c_5_0);
        c_5_1 = _mm256_fmadd_ps(a_i, b_1, c_5_1);

        a_ptr += 6;
        b_ptr += 16;
    }

    // store C registers back to memory
    _mm256_store_ps(C + 0 * 16 + 0, c_0_0);
    _mm256_store_ps(C + 0 * 16 + 8, c_0_1);
    _mm256_store_ps(C + 1 * 16 + 0, c_1_0);
    _mm256_store_ps(C + 1 * 16 + 8, c_1_1);
    _mm256_store_ps(C + 2 * 16 + 0, c_2_0);
    _mm256_store_ps(C + 2 * 16 + 8, c_2_1);
    _mm256_store_ps(C + 3 * 16 + 0, c_3_0);
    _mm256_store_ps(C + 3 * 16 + 8, c_3_1);
    _mm256_store_ps(C + 4 * 16 + 0, c_4_0);
    _mm256_store_ps(C + 4 * 16 + 8, c_4_1);
    _mm256_store_ps(C + 5 * 16 + 0, c_5_0);
    _mm256_store_ps(C + 5 * 16 + 8, c_5_1);
}
python_snippets = parse_markdown_file(file_path, language=language)