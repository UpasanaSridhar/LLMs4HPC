
# Task
You are an exprt in high-performance computing, C and assembly. Please write a high-performance C function that implements a 6x8 kernel for multiplying a 6x256 matrix \( A[M][K] \) and a 256x8 matrix \( B[K][N] \), resulting in a 6x8 output matrix \( C[M][N] \). 


# Context
Architecture: AVX2 vector extensions. 
Data-type: double 
Function-signature: `dgemm_avx2(double const *A, double const *B, double *C)`.  
Use i, j, and k as loop iterator variables for the M, N and K dimensions respectively.

# Kernel Writing Steps

Follow these steps in order:
1. Loop Order: 
   - Make the `K`-loop the outermost loop. Then have the loop over M, then N as the innermost loop.

2. Data Layout and Indexing:
   -  \( A \) -- column-major. 
   -  \( B \) -- row-major.
   -  \( C \) -- row-major.

3. Vectorization:  
   - Vectorize the N-loop.

4. Manual Unrolling:
   - Unroll all iterations of the M-loop and vectorized N-loop

5. Reuse in Register: 
   - Elide redundant loads and stores to the same memory location.

6. Register Usage: There are 16 registers in the AVX2 ISA.
   The unrolled code should use no more than 16 register type variables.

7. Pointer Arithmetic: Use linear pointer arithmetic instead of multidimensional indexing for \( A \), \( B \), and \( C \).

8. Assumptions:
   - All matrices are aligned to 64-byte boundaries for optimal memory access.

