
# Task
You are an exprt in high-performance computing, C and assembly. Please write a high-performance C function that implements a 6x8 kernel for multiplying a 6x256 matrix \( A \) and a 256x8 matrix \( B \), resulting in a 6x8 output matrix \( C \). 


# Context
Architecture: AVX2 vector extensions. 
Data-type: double 
Function-signature: `dgemm_avx2(double const *A, double const *B, double *C)`.  

# Kernel Writing Steps

Follow these steps in order:
1. Loop Order: 
   - Make the `K`-loop the outermost loop. Then have the loop over M, then N as the innermost loop.

2. Data Layout and Indexing:
   - For \( A \) -- column-major. 
      the indexing should be `A + k * 6 + i`, where \( i \) ranges from 0 to 5.
   - For \( B \) -- row-major.
      The indexing should be `B + k * 8 + j`, where \( j \) ranges from 0 to 7.
   - For \( C \) -- row-major.
      The indexing should be `B + k * 8 + j`, where \( j \) ranges from 0 to 7.

3. AVX2 Vectorization:  
   - Vectorize the innermost loop.
   - Increment the vectortized loop by the vector length.
   - Memory Interaction: Load data using a *packed-vector load* and store using *packed-store*.
   - Broadcast : If a single value is reused in a vectorized loop
      Use a *broadcast_instruction* to broadcast values from \( A \).
   - Computation: The original operation is a multiply-accumulate. Use a *fused multiply-add* if available.

4. Manual Unrolling:
   - For the `i`-loop, manually write out all operations for each of the 6 rows in the output matrix.
   - For the `j`-loop, manually handle the two chunks of 4 columns each in the output matrix.
5. Reuse in Register: 
   - Store the \( C \) registers to the output matrix only after completing the \( k \)-loop. Store the results directly into the appropriate positions in \( C \) (6 rows x 8 columns).
   - Loads of A and B must happen once per iteration of the k-loop.
6. Register Usage: There are 16 registers in the AVX2 ISA.
   The unrolled code should use no more than 16 register type variables.
   - 12 registers to hold the output matrix \( C \) (6 rows x 2 columns per row).
   - 2 registers to load \( B \) (8 elements, split across two registers).
   - 2 registers to broadcast values from \( A \) along the \( m \)-dimension.


7. Pointer Arithmetic: Use linear pointer arithmetic instead of multidimensional indexing for \( A \), \( B \), and \( C \).

8. Assumptions:
   - All matrices are aligned to 64-byte boundaries for optimal memory access.
   - \( A \) and \( B \) are accessed linearly, ensuring efficient memory access patterns.


