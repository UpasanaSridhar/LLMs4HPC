**Prompt**

You are an expert in high-performance computing and low-level programming. Please write a high-performance C++ function that implements a 6x8 General Matrix Multiplication (GEMM) kernel for multiplying a 6x256 matrix \( A \) and a 256x8 matrix \( B \), resulting in a 6x8 output matrix \( C \). The matrices are stored as arrays of doubles in row-major order, and all arrays are aligned on 64-byte boundaries.
Follow these guidelines:
1. AVX2 Vectorization: Use AVX2 vector instructions for high performance. Load data using `_mm256_load_pd` and store using `_mm256_store_pd`. Use `_mm256_fmadd_pd` for fused multiply-add operations.
2. Register Usage: Use a total of 16 registers:
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
8. Assumptions:
   - All matrices are aligned to 64-byte boundaries for optimal memory access.
   - \( A \) and \( B \) are accessed linearly, ensuring efficient memory access patterns.

Please provide the complete and optimized code snippet in a function named `matmul_avx2`. Ensure correctness and optimal usage of AVX2 instructions, explicitly unrolling the `i` and `j` loops as described.
