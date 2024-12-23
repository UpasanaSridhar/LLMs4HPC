 Below is a high-performance C++ function implementing a 6x8 General Matrix Multiplication (GEMM) kernel using AVX2 vector instructions. The function assumes matrices `AA` and `BB` are stored in row-major order, aligned on 64-byte boundaries, and contains double precision floating point numbers.

```cpp
#include <immintrin.h> // For AVX2 intrinsics

void matmul_avx2(double* AA, double* BB, double* CC) {
    for (int ii = 0; ii < 6; ++ii) {
        for (int jj = 0; jj < 8; ++jj) {
            __m256d c0 = _mm256_setzero_pd(); // Register for accumulating results, initializing to zero
            __m256d c1 = _mm256_setzero_pd(); // Another register for the same purpose

            for (int kk = 0; kk < 256; kk += 8) {
                __m256d bb = _mm256_loadu_pd(BB + kk * 8 + jj); // Load BB[kk][jj] into a vector register
                __m256d aa0 = _mm256_broadcast_sd(&AA[kk * 6 + ii * 1]); // Broadcast AA[kk][ii] across the vector
                __m256d aa1 = _mm256_broadcast_sd(&AA[(kk + 1) * 6 + ii * 1]); // Next row in AA, same column
                __m256d aa2 = _mm256_broadcast_sd(&AA[(kk + 2) * 6 + ii * 1]);
                __m256d aa3 = _mm256_broadcast_sd(&AA[(kk + 3) * 6 + ii * 1]);
                __m256d aa4 = _mm256_broadcast_sd(&AA[(kk + 4) * 6 + ii * 1]);
                __m256d aa5 = _mm256_broadcast_sd(&AA[(kk + 5) * 6 + ii * 1]);
                __m256d aa6 = _mm256_broadcast_sd(&AA[(kk + 6) * 6 + ii * 1]);
                __m256d aa7 = _mm256_broadcast_sd(&AA[(kk + 7) * 6 + ii * 1]);

                c0 = _mm256_fmadd_pd(aa0, bb, c0); // Multiply-accumulate for the first four elements of bb
                c1 = _mm256_fmadd_pd(aa1, bb, c1); // Same operation for the next four elements and registers
                c0 = _mm256_fmadd_pd(aa2, bb, c0);
                c1 = _mm256_fmadd_pd(aa3, bb, c1);
                c0 = _mm256_fmadd_pd(aa4, bb, c0);
                c1 = _mm256_fmadd_pd(aa5, bb, c1);
                c0 = _mm256_fmadd_pd(aa6, bb, c0);
                c1 = _mm256_fmadd_pd(aa7, bb, c1);
            }

            // Store the results from both registers into CC[ii][jj] and CC[ii][jj+1]
            _mm256_storeu_pd(&CC[ii * 8 + jj], c0);
            _mm256_storeu_pd(&CC[ii * 8 + jj + 4], c1);
        }
    }
}
```

This function fully unrolls the innermost loop and uses AVX2 instructions to perform vectorized computations efficiently. It loads, broadcasts, and accumulates data in a way that aligns with AVX2's capabilities for parallel processing of multiple data points at once. The comments provide explanations for each step, detailing how data is processed and stored across registers and memory locations.


### Systematic Approach to the transformations required in the kernel

1. Vectorization (AVX2): Use AVX2 vector instructions for high performance. Load data using `_mm256_load_pd` and store using `_mm256_store_pd`. Use `_mm256_fmadd_pd` for fused multiply-add operations.
2. Register Usage: Use no more than the avilable architectural registers. Use a total of 16 registers:
   - 12 registers to hold the output matrix \( C \) (6 rows x 2 columns per row).
   - 2 registers to load \( B \) (8 elements, split across two registers).
   - 2 registers to broadcast values from \( A \) along the \( m \)-dimension.
3. Loop Order: Make the `k`-loop the outermost loop. Completely unroll the `i`- and `j`-loops, manually specifying each operation without using runtime loops for `i` and `j`.
4. Manual Unrolling:
   - For the `i`-loop, manually write out all operations for each of the six rows in the output matrix.
   - For the `j`-loop, manually handle the two chunks of four columns each in the output matrix.
5. Broadcast and Indexing:
   - Use `_mm256_broadcast_sd` to broadcast values from \( A \).
   - For \( A \), the indexing into the broadcast should be `A + k * 6 + i`, where \( i \) ranges from 0 to 5.
   - For \( B \), load directly from `B + k * 8 + j`, where \( j \) ranges from 0 to 7, loading in two chunks of four elements.
6. Pointer Arithmetic: Use linear pointer arithmetic instead of multidimensional indexing for \( A \), \( B \), and \( C \).
7. Final Storage: Store the \( C \) registers to the output matrix only after completing the \( k \)-loop. Store the results directly into the appropriate positions in \( C \) (6 rows x 8 columns).
8. Data Layout Assumptions:
   - All matrices are aligned to 64-byte boundaries for optimal memory access.
   - \( A \) and \( B \) are accessed linearly, ensuring efficient memory access patterns.

## Dependencies in the transformations
We need to apply these transformations in a specific order. So we make the following assumptions for the ordering of transformations.

The data layout assumption must first be encoded.

Then:
1. We vectorize and unroll inner loops - so inner loops should be "parallel" loops
2. Outerloops are sequential loops
Thus, loop reordering is the first optimization. (it may happen even just with the compiler. Spoiler it doesn't)