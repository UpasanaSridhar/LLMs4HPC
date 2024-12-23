#include <immintrin.h>


void dgemm_avx2(const double* A, const double* B, double* C) {
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 256; ++k) {
                C[i * 8 + j] += A[k * 6 + i] * B[k * 8 + j];
                // C[i * 8 + j] += A[i * 256 + k] * B[k * 8 + j];

            }
        }
    }
}



// void dgemm_avx2(const double* A, const double* B, double* C) {
// for (int k = 0; k < 256; ++k) {
//     for (int i = 0; i < 6; ++i) {
//         for (int j = 0; j < 8; ++j) {
            
//                 C[i * 8 + j] += A[k * 6 + i] * B[k * 8 + j];
//                 // C[i * 8 + j] += A[i * 256 + k] * B[k * 8 + j];

//             }
//         }
//     }
// }

// //vectorized version of the kernel
// void dgemm_avx2(const double* A, const double* B, double* C) {
// __m256d c, a, b;
// for (int k = 0; k < 256; ++k) {
//     for (int i = 0; i < 6; ++i) {
//         //
//         a = _mm256_broadcast_sd(A + k * 6 + i);
//         for (int j = 0; j < 8; j+=4) {
            
//                 //C[i * 8 + j] += A[k * 6 + i] * B[k * 8 + j];
//                 b = _mm256_load_pd(B + k * 8 + j);
//                 c = _mm256_load_pd(C + i * 8 + j);
//                 c = _mm256_fmadd_pd(a, b, c);
//                 _mm256_store_pd(C + i * 8 + j, c);


//             }
//         }
//     }
// }

//vectorized version of the kernel
//unroll vectorized loop( loads hoisted)
// void dgemm_avx2(const double* A, const double* B, double* C) {
// __m256d c0, c1, a, b0, b1;
// for (int k = 0; k < 256; ++k) {
//     // Loads of B can be hoisted
//     b0 = _mm256_load_pd(B + k * 8 + 0);
//     b1 = _mm256_load_pd(B + k * 8 + 4);
//     for (int i = 0; i < 6; ++i) {
//         //
//         a = _mm256_broadcast_sd(A + k * 6 + i);
//         // for (int j = 0; j < 8; j+=4) {
            
//                 //C[i * 8 + j] += A[k * 6 + i] * B[k * 8 + j];


//                 c0 = _mm256_load_pd(C + i * 8 + 0);
//                 c1 = _mm256_load_pd(C + i * 8 + 4);
//                 c0 = _mm256_fmadd_pd(a, b0, c0);
//                 c1 = _mm256_fmadd_pd(a, b1, c1);
//                 _mm256_store_pd(C + i * 8 + 0, c0);
//                 _mm256_store_pd(C + i * 8 + 4, c1);


//             // }
//         }
//     }
// }


// //vectorized version of the kernel
// //unroll both independent loops (spill regis)
// void dgemm_avx2(const double* A, const double* B, double* C) {
// __m256d c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
// __m256d b0, b1;
// __m256d a0,  a1,  a2,  a3,  a4,  a5;
// for (int k = 0; k < 256; ++k) {
//     // Loads of B can be hoisted
//     b0 = _mm256_load_pd(B + k * 8 + 0);
//     b1 = _mm256_load_pd(B + k * 8 + 4);
//     //Unrolled
//     // for (int i = 0; i < 6; ++i) {
//         a0 = _mm256_broadcast_sd(A + k * 6 + 0);
//         a1 = _mm256_broadcast_sd(A + k * 6 + 1);
//         a2 = _mm256_broadcast_sd(A + k * 6 + 2);
//         a3 = _mm256_broadcast_sd(A + k * 6 + 3);
//         a4 = _mm256_broadcast_sd(A + k * 6 + 4);
//         a5 = _mm256_broadcast_sd(A + k * 6 + 5);


//         // for (int j = 0; j < 8; j+=4) {
            
//                 //C[i * 8 + j] += A[k * 6 + i] * B[k * 8 + j];


//                 c00= _mm256_load_pd(C + 0* 8 + 0);
//                 c01 = _mm256_load_pd(C + 0* 8 + 4);
//                 c00 = _mm256_fmadd_pd(a0, b0, c00);
//                 c01 = _mm256_fmadd_pd(a0, b1, c01);
//                 _mm256_store_pd(C + 0* 8 + 0, c00);
//                 _mm256_store_pd(C + 0* 8 + 4, c01);

//                 c10 = _mm256_load_pd(C + 1 * 8 + 0);
//                 c11 = _mm256_load_pd(C + 1 * 8 + 4);
//                 c10 = _mm256_fmadd_pd(a1, b0, c10);
//                 c11 = _mm256_fmadd_pd(a1, b1, c11);
//                 _mm256_store_pd(C + 1 * 8 + 0, c10);
//                 _mm256_store_pd(C + 1 * 8 + 4, c11);

//                 c20 = _mm256_load_pd(C + 2 * 8 + 0);
//                 c21 = _mm256_load_pd(C + 2 * 8 + 4);
//                 c20 = _mm256_fmadd_pd(a2, b0, c20);
//                 c21 = _mm256_fmadd_pd(a2, b1, c21);
//                 _mm256_store_pd(C + 2 * 8 + 0, c20);
//                 _mm256_store_pd(C + 2 * 8 + 4, c21);
                
//                 c30 = _mm256_load_pd(C + 3 * 8 + 0);
//                 c31 = _mm256_load_pd(C + 3 * 8 + 4);
//                 c30 = _mm256_fmadd_pd(a3, b0, c30);
//                 c31 = _mm256_fmadd_pd(a3, b1, c31);
//                 _mm256_store_pd(C + 3 * 8 + 0, c30);
//                 _mm256_store_pd(C + 3 * 8 + 4, c31);
                
//                 c40 = _mm256_load_pd(C + 4 * 8 + 0);
//                 c41 = _mm256_load_pd(C + 4 * 8 + 4);
//                 c40 = _mm256_fmadd_pd(a4, b0, c40);
//                 c41 = _mm256_fmadd_pd(a4, b1, c41);
//                 _mm256_store_pd(C + 4 * 8 + 0, c40);
//                 _mm256_store_pd(C + 4 * 8 + 4, c41);

//                 c50 = _mm256_load_pd(C + 5 * 8 + 0);
//                 c51 = _mm256_load_pd(C + 5 * 8 + 4);
//                 c50 = _mm256_fmadd_pd(a5, b0, c50);
//                 c51 = _mm256_fmadd_pd(a5, b1, c51);
//                 _mm256_store_pd(C + 5 * 8 + 0, c50);
//                 _mm256_store_pd(C + 5 * 8 + 4, c51);


//             // }
//         // }
//     }
// }

// //vectorized version of the kernel
// //unroll both independent loops (spill regis)
// // Hoist loads and stores of C (register resident)
// void dgemm_avx2(const double* A, const double* B, double* C) {
// __m256d c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
// __m256d b0, b1;
// __m256d a0,  a1,  a2,  a3,  a4,  a5;

// //Loads of C

// c00= _mm256_load_pd(C + 0* 8 + 0);
// c01 = _mm256_load_pd(C + 0* 8 + 4);

// c10 = _mm256_load_pd(C + 1 * 8 + 0);
// c11 = _mm256_load_pd(C + 1 * 8 + 4);

// c20 = _mm256_load_pd(C + 2 * 8 + 0);
// c21 = _mm256_load_pd(C + 2 * 8 + 4);

// c30 = _mm256_load_pd(C + 3 * 8 + 0);
// c31 = _mm256_load_pd(C + 3 * 8 + 4);

// c40 = _mm256_load_pd(C + 4 * 8 + 0);
// c41 = _mm256_load_pd(C + 4 * 8 + 4);

// c50 = _mm256_load_pd(C + 5 * 8 + 0);
// c51 = _mm256_load_pd(C + 5 * 8 + 4);

// for (int k = 0; k < 256; ++k) {
//     // Loads of B can be hoisted
//     b0 = _mm256_load_pd(B + k * 8 + 0);
//     b1 = _mm256_load_pd(B + k * 8 + 4);
//     //Unrolled
//     // for (int i = 0; i < 6; ++i) {
//         a0 = _mm256_broadcast_sd(A + k * 6 + 0);
//         a1 = _mm256_broadcast_sd(A + k * 6 + 1);
//         a2 = _mm256_broadcast_sd(A + k * 6 + 2);
//         a3 = _mm256_broadcast_sd(A + k * 6 + 3);
//         a4 = _mm256_broadcast_sd(A + k * 6 + 4);
//         a5 = _mm256_broadcast_sd(A + k * 6 + 5);


//         // for (int j = 0; j < 8; j+=4) {
            
//                 //C[i * 8 + j] += A[k * 6 + i] * B[k * 8 + j];


//                 c00 = _mm256_fmadd_pd(a0, b0, c00);
//                 c01 = _mm256_fmadd_pd(a0, b1, c01);

//                 c10 = _mm256_fmadd_pd(a1, b0, c10);
//                 c11 = _mm256_fmadd_pd(a1, b1, c11);


//                 c20 = _mm256_fmadd_pd(a2, b0, c20);
//                 c21 = _mm256_fmadd_pd(a2, b1, c21);
                

//                 c30 = _mm256_fmadd_pd(a3, b0, c30);
//                 c31 = _mm256_fmadd_pd(a3, b1, c31);
                

//                 c40 = _mm256_fmadd_pd(a4, b0, c40);
//                 c41 = _mm256_fmadd_pd(a4, b1, c41);


//                 c50 = _mm256_fmadd_pd(a5, b0, c50);
//                 c51 = _mm256_fmadd_pd(a5, b1, c51);


//             // }
//         // }
//     }

//     //Stores of C
//     _mm256_store_pd(C + 0* 8 + 0, c00);
//     _mm256_store_pd(C + 0* 8 + 4, c01);
//     _mm256_store_pd(C + 1 * 8 + 0, c10);
//     _mm256_store_pd(C + 1 * 8 + 4, c11);
//     _mm256_store_pd(C + 2 * 8 + 0, c20);
//     _mm256_store_pd(C + 2 * 8 + 4, c21);
//     _mm256_store_pd(C + 3 * 8 + 0, c30);
//     _mm256_store_pd(C + 3 * 8 + 4, c31);
//     _mm256_store_pd(C + 4 * 8 + 0, c40);
//     _mm256_store_pd(C + 4 * 8 + 4, c41);
//     _mm256_store_pd(C + 5 * 8 + 0, c50);
//     _mm256_store_pd(C + 5 * 8 + 4, c51);

// }


//vectorized version of the kernel
//unroll both independent loops (spill regis)
// Hoist loads and stores of C (register resident)
//Only use max 16 registers (reuse for A)
// void dgemm_avx2(const double* A, const double* B, double* C) {
// __m256d c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
// __m256d b0, b1;
// __m256d a0,  a1,  a2,  a3,  a4,  a5;

// //Loads of C

// c00= _mm256_load_pd(C + 0* 8 + 0);
// c01 = _mm256_load_pd(C + 0* 8 + 4);

// c10 = _mm256_load_pd(C + 1 * 8 + 0);
// c11 = _mm256_load_pd(C + 1 * 8 + 4);

// c20 = _mm256_load_pd(C + 2 * 8 + 0);
// c21 = _mm256_load_pd(C + 2 * 8 + 4);

// c30 = _mm256_load_pd(C + 3 * 8 + 0);
// c31 = _mm256_load_pd(C + 3 * 8 + 4);

// c40 = _mm256_load_pd(C + 4 * 8 + 0);
// c41 = _mm256_load_pd(C + 4 * 8 + 4);

// c50 = _mm256_load_pd(C + 5 * 8 + 0);
// c51 = _mm256_load_pd(C + 5 * 8 + 4);

// for (int k = 0; k < 256; ++k) {
//     // Loads of B can be hoisted
//     b0 = _mm256_load_pd(B + k * 8 + 0);
//     b1 = _mm256_load_pd(B + k * 8 + 4);
//     //Unrolled
//     // for (int i = 0; i < 6; ++i) {
      



//         // for (int j = 0; j < 8; j+=4) {
            
//                 //C[i * 8 + j] += A[k * 6 + i] * B[k * 8 + j];

//                 a0 = _mm256_broadcast_sd(A + k * 6 + 0);
//                 a1 = _mm256_broadcast_sd(A + k * 6 + 1);

//                 c00 = _mm256_fmadd_pd(a0, b0, c00);
//                 c01 = _mm256_fmadd_pd(a0, b1, c01);
//                 c10 = _mm256_fmadd_pd(a1, b0, c10);
//                 c11 = _mm256_fmadd_pd(a1, b1, c11);


//                 a0 = _mm256_broadcast_sd(A + k * 6 + 2);
//                 a1 = _mm256_broadcast_sd(A + k * 6 + 3);

//                 c20 = _mm256_fmadd_pd(a0, b0, c20);
//                 c21 = _mm256_fmadd_pd(a0, b1, c21);
//                 c30 = _mm256_fmadd_pd(a1, b0, c30);
//                 c31 = _mm256_fmadd_pd(a1, b1, c31);
                
//                 a0 = _mm256_broadcast_sd(A + k * 6 + 4);
//                 a1 = _mm256_broadcast_sd(A + k * 6 + 5);
//                 c40 = _mm256_fmadd_pd(a0, b0, c40);
//                 c41 = _mm256_fmadd_pd(a0, b1, c41);


//                 c50 = _mm256_fmadd_pd(a1, b0, c50);
//                 c51 = _mm256_fmadd_pd(a1, b1, c51);


//             // }
//         // }
//     }

//     //Stores of C
//     _mm256_store_pd(C + 0* 8 + 0, c00);
//     _mm256_store_pd(C + 0* 8 + 4, c01);
//     _mm256_store_pd(C + 1 * 8 + 0, c10);
//     _mm256_store_pd(C + 1 * 8 + 4, c11);
//     _mm256_store_pd(C + 2 * 8 + 0, c20);
//     _mm256_store_pd(C + 2 * 8 + 4, c21);
//     _mm256_store_pd(C + 3 * 8 + 0, c30);
//     _mm256_store_pd(C + 3 * 8 + 4, c31);
//     _mm256_store_pd(C + 4 * 8 + 0, c40);
//     _mm256_store_pd(C + 4 * 8 + 4, c41);
//     _mm256_store_pd(C + 5 * 8 + 0, c50);
//     _mm256_store_pd(C + 5 * 8 + 4, c51);

// }


//vectorized version of the kernel
//unroll both independent loops (spill regis)
// Hoist loads and stores of C (register resident)
//Only use max 16 registers (reuse for A)
//POinter arithmetic
// void dgemm_avx2(const double* A, const double* B, double* C) {
// __m256d c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
// __m256d b0, b1;
// __m256d a0,  a1,  a2,  a3,  a4,  a5;

// //Loads of C

// // c00= _mm256_load_pd(C + 0* 8 + 0);
// // c01 = _mm256_load_pd(C + 0* 8 + 4);

// // c10 = _mm256_load_pd(C + 1 * 8 + 0);
// // c11 = _mm256_load_pd(C + 1 * 8 + 4);

// // c20 = _mm256_load_pd(C + 2 * 8 + 0);
// // c21 = _mm256_load_pd(C + 2 * 8 + 4);

// // c30 = _mm256_load_pd(C + 3 * 8 + 0);
// // c31 = _mm256_load_pd(C + 3 * 8 + 4);

// // c40 = _mm256_load_pd(C + 4 * 8 + 0);
// // c41 = _mm256_load_pd(C + 4 * 8 + 4);

// // c50 = _mm256_load_pd(C + 5 * 8 + 0);
// // c51 = _mm256_load_pd(C + 5 * 8 + 4);

// // set to zero
// c00 = _mm256_setzero_pd();
// c01 = _mm256_setzero_pd();
// c10 = _mm256_setzero_pd();
// c11 = _mm256_setzero_pd();
// c20 = _mm256_setzero_pd();
// c21 = _mm256_setzero_pd();
// c30 = _mm256_setzero_pd();
// c31 = _mm256_setzero_pd();
// c40 = _mm256_setzero_pd();
// c41 = _mm256_setzero_pd();
// c50 = _mm256_setzero_pd();
// c51 = _mm256_setzero_pd();


// const double *a_ptr = A;
// const double *b_ptr = B;
// for (int k = 0; k < 256; ++k) {
//     // Loads of B can be hoisted
//     b0 = _mm256_load_pd(b_ptr + 0);
//     b1 = _mm256_load_pd(b_ptr + 4);
//     //Unrolled
//     // for (int i = 0; i < 6; ++i) {
      



//         // for (int j = 0; j < 8; j+=4) {
            
//                 //C[i * 8 + j] += A[k * 6 + i] * B[k * 8 + j];

//                 a0 = _mm256_broadcast_sd(a_ptr + 0);

//                 c00 = _mm256_fmadd_pd(a0, b0, c00);
//                 c01 = _mm256_fmadd_pd(a0, b1, c01);
//                 a1 = _mm256_broadcast_sd(a_ptr + 1);

//                 c10 = _mm256_fmadd_pd(a1, b0, c10);
//                 c11 = _mm256_fmadd_pd(a1, b1, c11);


//                 a0 = _mm256_broadcast_sd(a_ptr + 2);

//                 c20 = _mm256_fmadd_pd(a0, b0, c20);
//                 c21 = _mm256_fmadd_pd(a0, b1, c21);
//                 a1 = _mm256_broadcast_sd(a_ptr + 3);

//                 c30 = _mm256_fmadd_pd(a1, b0, c30);
//                 c31 = _mm256_fmadd_pd(a1, b1, c31);
                
//                 a0 = _mm256_broadcast_sd(a_ptr + 4);
//                 c40 = _mm256_fmadd_pd(a0, b0, c40);
//                 c41 = _mm256_fmadd_pd(a0, b1, c41);

//                 a1 = _mm256_broadcast_sd(a_ptr + 5);

//                 c50 = _mm256_fmadd_pd(a1, b0, c50);
//                 c51 = _mm256_fmadd_pd(a1, b1, c51);


//             // }
//         // }
//         b_ptr += 8; 
//         a_ptr += 6;
//     }

//     //Stores of C
//     _mm256_store_pd(C + 0* 8 + 0, c00);
//     _mm256_store_pd(C + 0* 8 + 4, c01);
//     _mm256_store_pd(C + 1 * 8 + 0, c10);
//     _mm256_store_pd(C + 1 * 8 + 4, c11);
//     _mm256_store_pd(C + 2 * 8 + 0, c20);
//     _mm256_store_pd(C + 2 * 8 + 4, c21);
//     _mm256_store_pd(C + 3 * 8 + 0, c30);
//     _mm256_store_pd(C + 3 * 8 + 4, c31);
//     _mm256_store_pd(C + 4 * 8 + 0, c40);
//     _mm256_store_pd(C + 4 * 8 + 4, c41);
//     _mm256_store_pd(C + 5 * 8 + 0, c50);
//     _mm256_store_pd(C + 5 * 8 + 4, c51);

// }


// //vectorized version of the kernel
// //unroll both independent loops (spill regis)
// // Hoist loads and stores of C (register resident)
// //Only use max 16 registers (reuse for A)
// //POinter arithmetic
// void dgemm_avx2(const double* A, const double* B, double* C) {
// __m256d c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
// __m256d b0, b1;
// __m256d a0,  a1,  a2,  a3,  a4,  a5;

// //Loads of C

// c00= _mm256_load_pd(C + 0* 8 + 0);
// c01 = _mm256_load_pd(C + 0* 8 + 4);

// c10 = _mm256_load_pd(C + 1 * 8 + 0);
// c11 = _mm256_load_pd(C + 1 * 8 + 4);

// c20 = _mm256_load_pd(C + 2 * 8 + 0);
// c21 = _mm256_load_pd(C + 2 * 8 + 4);

// c30 = _mm256_load_pd(C + 3 * 8 + 0);
// c31 = _mm256_load_pd(C + 3 * 8 + 4);

// c40 = _mm256_load_pd(C + 4 * 8 + 0);
// c41 = _mm256_load_pd(C + 4 * 8 + 4);

// c50 = _mm256_load_pd(C + 5 * 8 + 0);
// c51 = _mm256_load_pd(C + 5 * 8 + 4);

// // set to zero
// // c00 = _mm256_setzero_pd();
// // c01 = _mm256_setzero_pd();
// // c10 = _mm256_setzero_pd();
// // c11 = _mm256_setzero_pd();
// // c20 = _mm256_setzero_pd();
// // c21 = _mm256_setzero_pd();
// // c30 = _mm256_setzero_pd();
// // c31 = _mm256_setzero_pd();
// // c40 = _mm256_setzero_pd();
// // c41 = _mm256_setzero_pd();
// // c50 = _mm256_setzero_pd();
// // c51 = _mm256_setzero_pd();


// const double *a_ptr = A;
// const double *b_ptr = B;
// for (int k = 0; k < 256; ++k) {
//     // Loads of B can be hoisted
//     b0 = _mm256_load_pd(b_ptr + 0);
//     b1 = _mm256_load_pd(b_ptr + 4);
//     //Unrolled
//     // for (int i = 0; i < 6; ++i) {
      



//         // for (int j = 0; j < 8; j+=4) {
            
//                 //C[i * 8 + j] += A[k * 6 + i] * B[k * 8 + j];

//                 a0 = _mm256_broadcast_sd(a_ptr + 0);

//                 c00 = _mm256_fmadd_pd(a0, b0, c00);
//                 c01 = _mm256_fmadd_pd(a0, b1, c01);
//                 a1 = _mm256_broadcast_sd(a_ptr + 1);

//                 c10 = _mm256_fmadd_pd(a1, b0, c10);
//                 c11 = _mm256_fmadd_pd(a1, b1, c11);


//                 a0 = _mm256_broadcast_sd(a_ptr + 2);

//                 c20 = _mm256_fmadd_pd(a0, b0, c20);
//                 c21 = _mm256_fmadd_pd(a0, b1, c21);
//                 a1 = _mm256_broadcast_sd(a_ptr + 3);

//                 c30 = _mm256_fmadd_pd(a1, b0, c30);
//                 c31 = _mm256_fmadd_pd(a1, b1, c31);
                
//                 a0 = _mm256_broadcast_sd(a_ptr + 4);
//                 c40 = _mm256_fmadd_pd(a0, b0, c40);
//                 c41 = _mm256_fmadd_pd(a0, b1, c41);

//                 a1 = _mm256_broadcast_sd(a_ptr + 5);

//                 c50 = _mm256_fmadd_pd(a1, b0, c50);
//                 c51 = _mm256_fmadd_pd(a1, b1, c51);


//             // }
//         // }
//         b_ptr += 8; 
//         a_ptr += 6;
//     }

//     //Stores of C
//     _mm256_store_pd(C + 0* 8 + 0, c00);
//     _mm256_store_pd(C + 0* 8 + 4, c01);
//     _mm256_store_pd(C + 1 * 8 + 0, c10);
//     _mm256_store_pd(C + 1 * 8 + 4, c11);
//     _mm256_store_pd(C + 2 * 8 + 0, c20);
//     _mm256_store_pd(C + 2 * 8 + 4, c21);
//     _mm256_store_pd(C + 3 * 8 + 0, c30);
//     _mm256_store_pd(C + 3 * 8 + 4, c31);
//     _mm256_store_pd(C + 4 * 8 + 0, c40);
//     _mm256_store_pd(C + 4 * 8 + 4, c41);
//     _mm256_store_pd(C + 5 * 8 + 0, c50);
//     _mm256_store_pd(C + 5 * 8 + 4, c51);

// }