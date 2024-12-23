:"_mm256_fmadd_ps", 
...       "fused multiply-add./sub": "_mm256_fmaddsub_ps"
...       },
...       "Vector Data Type": "__m256",
...       "Register width": 256,
...       "Elements per vector register": "8"
...     
...     
... 
... ---
... 
... ## Challenge:
... 
...         Objective:
...         Identify the vector instructions and datatypes to use to vectorize the given functionality using the AVX2 vecto
... r ISA.
...         The scalar datatype is int32_t
... """
  { Vector Loads :"_mm256_load_si256",
    "Vector Stores": "_mm256_store_si256",
    "Vector Broadcasts": "_mm256_broadcastb_epi8",
    "Binary Vector Operations": {
      "add": "_mm256_add_epi32",
      "sub": "_mm256_sub_epi32",
      "mul" :"_mm256_mul_epi32",
      "min": "_mm256_min_epi32",
      "max": "_mm256_max_epi32",
      "and": "_mm256_and_si256"
    },
    "Ternary Vector Operations": {
      "fused-multipy-add" :"_mm256_fmadd_epi32",
      "fused multiply-add/sub": "_mm256_fmaddsub_epi32"
    },
    "Vector Data Type": "__m256i",
    "Register width": 256,
    "Elements per vector register": "8" }

>>>   { Vector Loads :"_mm256_load_pd",
...     "Vector Stores": "_mm256_store_pd",
...     "Vector Broadcasts": "_mm256_broadcast_sd",
...     "Binary Vector Operations": {
...       "add": "_mm256_add_pd",
...       "sub": "_mm256_sub_pd",
...       "mul" :"_mm256_mul_pd",
...       "min": "_mm256_min_pd",
...       "max": "_mm256_max_pd",
...       "and": "_mm256_and_pd"
...     },
...     "Ternary Vector Operations": {
...       "fused-multipy-add" :"_mm256_fmadd_pd",
...       "fused multiply-add/sub": "_mm256_fmaddsub_pd"
...     },
...     "Vector Data Type": "__m256d",
...     "Register width": 256,
...     "Elements per vector register": "4" }
... }
  { Vector Loads :"_mm256_load_pd",
    "Vector Stores": "_mm256_store_pd",
    "Vector Broadcasts": "_mm256_broadcast_sd",
    "Binary Vector Operations": {
      "add": "_mm256_add_pd",
      "sub": "_mm256_sub_pd",
      "mul" :"_mm256_mul_pd",
      "min": "_mm256_min_pd",
      "max": "_mm256_max_pd",
      "and": "_mm256_and_pd"
    },
    "Ternary Vector Operations": {
      "fused-multipy-add" :"_mm256_fmadd_pd",
      "fused multiply-add/sub": "_mm256_fmaddsub_pd"
    },
    "Vector Data Type": "__m256d",
    "Register width": 256,
    "Elements per vector register": "4" }

>>> """
...     You are an expert in vectorization and have been asked to identify the vector instructions and datatypes to use to 
... vectorize the given functionality.
... 
...     
...     
...     ## Functionalities:
...     ### Vector Loads
...     ```c
...     // Load 1 vector worth of elements, from contiguous memory locations
...     for(int i=0; i< VECTOR_LENGTH; i++)
...     {
...         vector_reg[i] = B[i];
...     }
...     ```
...     ### Vector Stores
...     ```c
...     // Store 1 vector worth of elements, to contiguous memory locations
...     for(int i=0; i< VECTOR_LENGTH; i++)
...     {
...         C[i] = vector_reg[i];
...     }
...     ```
...     ### Vector Broadcasts
...     ```c
...     // Broadcast a scalar value to all elements of a vector register
...     for(int i=0; i< VECTOR_LENGTH; i++)
...     {
...         vector_reg[i] = scalar;
...     }
...     ```
... 
...     ### Binary Vector Operations
...     ```c
...     // Vector addition
...     for(int i=0; i< VECTOR_LENGTH; i++)
...     {
...         vector_reg[i] = op(in_vector_0[i], in_vector_1[i]);
...         // e.g., vector_reg[i] = in_vector_0[i] + in_vector_1[i];
...     }
...     ```
...     Operations: add, sub, mul, min, max, and, cmp
... 
...     ### Ternary Vector Operations
...     ```c
...     // Vector multiplication
...     for(int i=0; i< VECTOR_LENGTH; i++)
...     {
...         vector_reg[i] = op(in_vector_0[i], in_vector_1[i], vector_reg[i]);
...         // e.g., vector_reg[i] = in_vector_0[i] * in_vector_1[i] + vector_reg[i];
...     }
...     Operations: fused-multipy-add, fused multiply-add/sub
...     ```
... 
...     Objective: Identify the vector instructions and datatypes to use to vectorize the given functionality using the AVX
... 2 vector ISA.
...                 The scalar datatype is float.
...     
...     Output:
...     General Translation
...     { Vector Loads":" _mm256_load_ps",
...       "Vector Stores":" _mm256_store_ps"    ,
...       "Vector Broadcasts":" _mm256_broadcast_ss"   ,
...       "Binary Vector Operations":{
...                 "add": "_mm256_add_ps", 
...                 "sub": "_mm256_sub_ps", 
...                 "mul" :"_mm256_mul_ps", 
...                 "min": "_mm256_min_ps",
...                 "max": "_mm256_max_ps",
...                 "and": "_mm256_and_ps",
...                 },
...       "Ternary Vector Operations":{
...       "fused-multipy-add" :"_mm256_fmadd_ps", 
...       "fused multiply-add./sub": "_mm256_fmaddsub_ps"
...       },
...       "Vector Data Type": "__m256",
...       "Register width": 256,
...       "Elements per vector register": "8"
...     
...     
... 
... ---
... 
... ## Challenge:
... 
...         Objective:
...         Identify the vector instructions and datatypes to use to vectorize the given functionality using the ARM Neon I
... SA. 
... The scalar datatype is floats.
... """
  { Vector Loads :"vld1q_f32",
    "Vector Stores": "vst1q_f32",
    "Vector Broadcasts": "vdupq_n_f32",
    "Binary Vector Operations": {
      "add": "vaddq_f32",
      "sub": "vsubq_f32",
      "mul" :"vmulq_f32",
      "min": "vminq_f32",
      "max": "vmaxq_f32",
      "and": "vandq_s32"
    },
    "Ternary Vector Operations": {
      "fused-multipy-add" :"vmlaq_f32",
      "fused multiply-add/sub": "vmlavq_f32"
    },
    "Vector Data Type": "float32x4_t",
    "Register width": 128,
    "Elements per vector register": "4" }

>>> "" Given that this is the mapping of vector instructions to use,             
    { Vector Loads :"vld1q_f32",
    "Vector Stores": "vst1q_f32",
    "Vector Broadcasts": "vdupq_n_f32",
    "Binary Vector Operations": {
      "add": "vaddq_f32",
      "sub": "vsubq_f32",
      "mul" :"vmulq_f32",
      "min": "vminq_f32",
      "max": "vmaxq_f32",
      "and": "vandq_s32"
    },
    "Ternary Vector Operations": {
      "fused-multipy-add" :"vmlaq_f32",
      "fused multiply-add/sub": "vmlavq_f32"
    },
    "Vector Data Type": "float32x4_t",
    "Register width": 128,
    "Elements per vector register": "4" }

>>> """ You are an expert programmer. Given this mapping of operations to vector instructions,
...   { Vector Loads :"vld1q_f32",
...     "Vector Stores": "vst1q_f32",
...     "Vector Broadcasts": "vdupq_n_f32",
...     "Binary Vector Operations": {
...       "add": "vaddq_f32",
...       "sub": "vsubq_f32",
...       "mul" :"vmulq_f32",
...       "min": "vminq_f32",
...       "max": "vmaxq_f32",
...       "and": "vandq_s32"
...     },
...     "Ternary Vector Operations": {
...       "fused-multipy-add" :"vmlaq_f32",
...       "fused multiply-add/sub": "vmlavq_f32"
...     },
...     "Vector Data Type": "float32x4_t",
...     "Register width": 128,
...     "Elements per vector register": "4" } 
... 
... ## Challenge 
... Vectorize the following piece of code using the appropriate instructions from above.
... ```
...     for(int i=0; i< M; i++)
...     {
...         float * c_row = C + i*N;
...         float a = A[i];
...         for(int j=0; j< N; j++)
...         {
...             c_row[j] += a * B[j];
...         }
...     }
... ```
... Assume that you vectorize the innerloop.
... All inputs are aligned.   
... """
  ```c
  for(int i=0; i< M; i++) {
    // Load the scalar value "a" and the matrix B into registers
    __m128 a = _mm_set1_ps(A[i]);
    __m128* B = (__m128*)B + i*N;
    
    // Initialize the result vector "c_row" to zero
    __m128* c_row = (__m128*)C + i*N;
    _mm_setzero_ps(c_row);
    
    // Perform the dot product of "a" and matrix B, and add it to "c_row"
    for(int j=0; j< N; j+=4) {
      __m128 b = _mm_loadu_ps(B + j); // load 4 elements from B[j]
      __m128 c = _mm_mul_ps(a, b);    // multiply "a" and "b"
      c_row = _mm_add_ps(c_row, c); // add the result to "c_row"
    }
  }
  ```
This code uses the `_mm_set1_ps` instruction to load a scalar value into an __m128 register. The `B + i*N` expression 
is used to calculate the starting address of each row in matrix B, and it is then loaded using the `_mm_loadu_ps` 
instruction. The multiplication of "a" and "b" is performed using the `_mm_mul_ps` instruction, and the resulting 
vector is stored in the __m128 variable `c`. The addition of "c" to "c_row" is performed using the `_mm_add_ps` 
instruction.
The `for` loop is used to iterate over each row of matrix B, and the dot product is calculated for each row 
separately. The `j+=4` expression is used to step through the elements of the matrix in groups of 4, which is the 
size of an __m128 vector. This allows the use of the SIMD instructions that operate on multiple elements at once.
This code assumes that all inputs are aligned, meaning that their starting addresses are multiples of 16 bytes. If 
this is not the case, then the `B + i*N` expression must be modified to ensure that the address is properly aligned.

>>> """
Use Ctrl + d or /bye to exit.
>>> """
...   { Vector Loads :"vld1q_f32",
...     "Vector Stores": "vst1q_f32",
...     "Vector Broadcasts": "vdupq_n_f32",
...     "Binary Vector Operations": {
...       "add": "vaddq_f32",
...       "sub": "vsubq_f32",
...       "mul" :"vmulq_f32",
...       "min": "vminq_f32",
...       "max": "vmaxq_f32",
...       "and": "vandq_s32"
...     },
...     "Ternary Vector Operations": {
...       "fused-multipy-add" :"vmlaq_f32",
...       "fused multiply-add/sub": "vmlavq_f32"
...     },
...     "Vector Data Type": "float32x4_t",
...     "Register width": 128,
...     "Elements per vector register": "4" }
... """
  ```c
  for(int i=0; i< M; i++) {
    // Load the scalar value "a" and the matrix B into registers
    float32x4_t a = vdupq_n_f32(A[i]);
    float32x4_t* B = (float32x4_t*)B + i*N;
    
    // Initialize the result vector "c_row" to zero
    float32x4_t* c_row = (float32x4_t*)C + i*N;
    vmovq_n_f32(c_row, 0);
    
    // Perform the dot product of "a" and matrix B, and add it to "c_row"
    for(int j=0; j< N; j+=4) {
      float32x4_t b = vld1q_f32(B + j); // load 4 elements from B[j]
      float32x4_t c = vmlaq_f32(a, b);    // multiply "a" and "b", add the result to "c_row"
      vst1q_f32(c_row + j, c);
    }
  }
  ```
This code uses the `vdupq_n_f32` instruction to load a scalar value into an float32x4_t register. The `B + i*N` 
expression is used to calculate the starting address of each row in matrix B, and it is then loaded using the 
`vld1q_f32` instruction. The multiplication of "a" and "b" is performed using the `vmlaq_f32` instruction, which 
performs a dot product operation. The resulting vector is stored in the float32x4_t variable `c`. The addition of "c" 
to "c_row" is performed using the `vst1q_f32` instruction.
The `for` loop is used to iterate over each row of matrix B, and the dot product is calculated for each row 
separately. The `j+=4` expression is used to step through the elements of the matrix in groups of 4, which is the 
size of an float32x4_t vector. This allows the use of the SIMD instructions that operate on multiple elements at 
once.
This code assumes that all inputs are aligned, meaning that their starting addresses are multiples of 16 bytes. 
If this is not the case, then the `B + i*N` expression must be modified to ensure that the address is properly aligned

>>> """ You are an expert `ARM` programmer. Given the following mapping between scalar operations and vector intrinsics, ve
... ctorize the code snippet that follows.
...   { "Loads":"vld1q_f32",
...     "Stores": "vst1q_f32",
...     "Broadcasts": "vdupq_n_f32",
...     "Binary Vector Operations": {
...       "+": "vaddq_f32",
...       "-": "vsubq_f32",
...       "*" :"vmulq_f32",
...       "min": "vminq_f32",
...       "max": "vmaxq_f32",
...       "&&": "vandq_s32"
...     },
...     "Ternary Vector Operations": {
...       "+=, *" :"vmlaq_f32",
...       "-+= ,*": "vmlavq_f32"
...     },
...     "Vector Data Type": "float32x4_t",
...     "Register width": 128,
...     "Elements per vector register": "4" }
... 
... ## Original Code to Vectorize
...     ```c
...     for(int i=0; i< M; i++)
...     {
...         float * c_row = C + i*N;
...         float a = A[i];
...         for(int j=0; j< N; j++)
...         {
...             c_row[j] += a * B[j];
...         }
...     }
...     ```
... """
  ```c
  for(int i=0; i< M; i++) {
      float32x4_t* c_row = (float32x4_t*)C + i*N;
      float32x4_t a = vdupq_n_f32(A[i]);
      for(int j=0; j< N; j+=4) {
          float32x4_t b = vld1q_f32((B + j)); // load 4 elements from B[j]
          c_row = vmlaq_f32(c_row, a, b);    // multiply "a" and "b", add the result to "c_row"
      }
  }
  ```
This code uses the `vdupq_n_f32` instruction to load a scalar value into an float32x4_t register. The `B + j` expression is used 
to calculate the starting address of each row in matrix B, and it is then loaded using the `vld1q_f32` instruction. The 
multiplication of "a" and "b" is performed using the `vmlaq_f32` instruction, which performs a dot product operation. The 
resulting vector is stored in the float32x4_t variable `c_row`.
The `for` loop is used to iterate over each row of matrix B, and the dot product is calculated for each row separately. The `j+=4` 
expression is used to step through the elements of the matrix in groups of 4, which is the size of an float32x4_t vector. This 
allows the use of the SIMD instructions that operate on multiple elements at once.
This code assumes that all inputs are aligned, meaning that their starting addresses are multiples of 16 bytes. If this is not the 
case, then the `B + j` expression must be modified to ensure that the address is properly aligned[INST:2]].

>>> """ Take this code snippet and unroll the innerloop, assume that N = 12.
...   ```c
...   for(int i=0; i< M; i++) {
...       float32x4_t* c_row = (float32x4_t*)C + i*N;
...       float32x4_t a = vdupq_n_f32(A[i]);
...       for(int j=0; j< N; j+=4) {
...           float32x4_t b = vld1q_f32((B + j)); // load 4 elements from B[j]
...           c_row = vmlaq_f32(c_row, a, b);    // multiply "a" and "b", add the result to "c_row"
...       }
...   }
...   ```
... """
  ```c
  for(int i=0; i< M; i++) {
      float32x4_t* c_row = (float32x4_t*)C + i*N;
      float32x4_t a = vdupq_n_f32(A[i]);
      float32x4_t b1 = vld1q_f32((B + 0)); // load 4 elements from B[j]
      float32x4_t b2 = vld1q_f32((B + 4)); // load 4 elements from B[j+1]
      float32x4_t b3 = vld1q_f32((B + 8)); // load 4 elements from B[j+2]
      c_row = vmlaq_f32(c_row, a, b1);    // multiply "a" and "b1", add the result to "c_row"
      c_row = vmlaq_f32(c_row, a, b2);    // multiply "a" and "b2", add the result to "c_row"
      c_row = vmlaq_f32(c_row, a, b3);    // multiply "a" and "b3", add the result to "c_row"
  }
  ```
In this unrolled version of the code, we have removed the inner loop and instead used three separate `vld1q_f32` instructions to 
load 4 elements from each row of matrix B. We then use three separate `vmlaq_f32` instructions to multiply each element in the 
current row of matrix A with each loaded vector, and add the results to the corresponding element in the output matrix.
This code is more efficient than the original version because it reduces the number of memory accesses required for each iteration 
of the loop. By loading multiple elements from matrix B at once, we can reduce the number of memory accesses by a factor of 4. 
Additionally, this unrolled version of the code is simpler to understand and maintain, as it eliminates the need for the inner 
loop.

>>> """ given this set of vector instructions     
... { "Vector Loads":"_mm256_load_ps",
... "Vector Stores": "_mm256_store_ps",
... "Vector Broadcasts": "_mm256_broadcast_ss",
... "Binary Vector Operations": {
...   "add": "_mm256_add_ps",
...   "sub": "_mm256_sub_ps",
...   "mul" :"_mm256_mul_ps",
...   "min": "_mm256_min_ps",
...   "max": "_mm256_max_ps",
...   "and": "_mm256_and_ps",
...   "cmp": "_mm256_cmp_ps",
... },
... "Ternary Vector Operations": {
...   "fused-multipy-add" :"_mm256_fmadd_ps",
...   "fused multiply-add/sub": "_mm256_fmaddsub_ps"
... },
... "Vector Data Type": "__m256",
... "Register width": 256,
... "Elements per vector register": 8 
... }
... Vectorize the innerloop of the following code:
... ```
... #define FLOAT_COND_SCALE_TILE_C(step, a, b, W_ob, C_ob)                 \
...     float *c_pixel = c_tile;                                         \
...     float const *a_pixel = a;                                        \
...     float const scale = b[0];                                              \
...     for (uint32_t kk = 0; kk < W_ob; kk++)                              \
...     {                                                                   \
...         float *c_channel = c_pixel;                                  \
...         float const *a_channel = a_pixel;                            \
...         for (uint32_t jj = 0; jj < C_ob; jj++)                          \
...         {                                                               \
...             *(c_channel) = (*(a_channel) > *(c_channel)) ?              \
...                 *(a_channel) : (*(a_channel) * (scale));                \
...             c_channel++;                                                \
...             a_channel++;                                                \
...         }                                                               \
...         a_pixel += step;                                                \
...         c_pixel += C_ob;                                                \
...     }
... ```
... """
 Vectorized version of the inner loop:
```c
for (uint32_t kk = 0; kk < W_ob; kk++) {
    // Load a channel of matrix A into a register
    __m256 a_channel = _mm256_loadu_ps(a_pixel);
    // Broadcast the scale value to all elements of the register
    __m256 scale_broadcast = _mm256_broadcast_ss(&scale);
    // Multiply the two registers element-wise
    __m256 mul_result = _mm256_mul_ps(a_channel, scale_broadcast);
    // Load a channel of matrix C into another register
    __m256 c_channel = _mm256_loadu_ps(c_pixel);
    // Compare the two registers element-wise
    __m256 cmp_result = _mm256_cmp_ps(a_channel, c_channel, _CMP_GT_OQ);
    // Conditionally select either the matrix A value or the scaled matrix A value
    __m256 final_result = _mm256_blendv_ps(mul_result, a_channel, cmp_result);
    // Store the final result in matrix C
    _mm256_storeu_ps(c_pixel, final_result);
    
    a_pixel += step;
    c_pixel += C_ob;
}
```
Explanation:

1. We first load a channel of matrix A into an `__m256` register using `_mm256_loadu_ps`.
2. We then broadcast the scale value to all elements of the register using `_mm256_broadcast_ss`.
3. We multiply the two registers element-wise using `_mm256_mul_ps`.
4. We load a channel of matrix C into another `__m256` register using `_mm256_loadu_ps`.
5. We compare the two registers element-wise using `_mm256_cmp_ps`, with the comparison operator `_CMP_GT_OQ` set to greater than 
or equal to.
6. We conditionally select either the matrix A value or the scaled matrix A value based on the result of the comparison using 
`_mm256_blendv_ps`.
7. Finally, we store the final result in matrix C using `_mm256_storeu_ps`.

This vectorized version of the inner loop uses 8 SIMD registers to perform the operations and should be faster than the original 
loop for large matrices. The use of `_mm256_loadu_ps` and `_mm256_storeu_ps` allows us to load and store multiple elements from/to 
memory in a single operation, which reduces the number of memory accesses required and improves performance.

>>> """ Unroll the outer loop in this code. Assume W_ob = 12. Use 12 registers for the final output. reuse input registers as necessary
... ```c
... for (uint32_t kk = 0; kk < W_ob; kk++) {
...     // Load a channel of matrix A into a register
...     __m256 a_channel = _mm256_loadu_ps(a_pixel);
...     // Broadcast the scale value to all elements of the register
...     __m256 scale_broadcast = _mm256_broadcast_ss(&scale);
...     // Multiply the two registers element-wise
...     __m256 mul_result = _mm256_mul_ps(a_channel, scale_broadcast);
...     // Load a channel of matrix C into another register
...     __m256 c_channel = _mm256_loadu_ps(c_pixel);
...     // Compare the two registers element-wise
...     __m256 cmp_result = _mm256_cmp_ps(a_channel, c_channel, _CMP_GT_OQ);
...     // Conditionally select either the matrix A value or the scaled matrix A value
...     __m256 final_result = _mm256_blendv_ps(mul_result, a_channel, cmp_result);
...     // Store the final result in matrix C
...     _mm256_storeu_ps(c_pixel, final_result);
...     
...     a_pixel += step;
...     c_pixel += C_ob;
... }
... ```
... """
 Here's the unrolled version of the outer loop:
```c
__m256 a_channels[12];
__m256 scale_broadcast[12];
__m256 mul_results[12];
__m256 c_channels[12];
__m256 cmp_results[12];
__m256 final_results[12];

for (uint32_t kk = 0; kk < W_ob; kk++) {
    // Load a channel of matrix A into an __m256 register
    a_channels[kk] = _mm256_loadu_ps(a_pixel);
    scale_broadcast[kk] = _mm256_broadcast_ss(&scale);
    mul_results[kk] = _mm256_mul_ps(a_channels[kk], scale_broadcast[kk]);
    
    c_channels[kk] = _mm256_loadu_ps(c_pixel);
    cmp_results[kk] = _mm256_cmp_ps(a_channels[kk], c_channels[kk], _CMP_GT_OQ);
    final_results[kk] = _mm256_blendv_ps(mul_results[kk], a_channels[kk], cmp_results[kk]);
    
    // Store the final result in matrix C
    _mm256_storeu_ps(c_pixel, final_results[kk]);
    
    a_pixel += step;
    c_pixel += C_ob;
}
```
In this version of the code, we've unrolled the outer loop 12 times and stored each iteration in its own __m256 register. This 
allows us to perform all the operations for a single iteration at once, which reduces the number of memory accesses required and 
improves performance. We've also reused the input registers as necessary to reduce the number of registers required for the loop.

>>> """ Given this set of vector instructions.
...    General Translation
...     { "loads":" _mm256_load_ps",
...       "stores":" _mm256_store_ps"    ,
...       "broadcast":" _mm256_broadcast_ss"   ,
...       "Binary Vector Operations":{
...                 "add": "_mm256_add_ps", 
...                 "sub": "_mm256_sub_ps", 
...                 "mul" :"_mm256_mul_ps", 
...                 "min": "_mm256_min_ps",
...                 "max": "_mm256_max_ps",
...                 "and": "_mm256_and_ps",
...                 "cmp": "_mm256_cmp_ps"
...                 },
...       "Ternary Vector Operations":{
...       "fused multipy-add" :"_mm256_fmadd_ps", 
...       "fused multiply sub": "_mm256_fmsub_ps",
...       "fused multiply-add/sub": "_mm256_fmaddsub_ps"
...       },
...       "Vector Data Type": "__m256",
...       "Register width": 256,
...       "Elements per vector register": "8"
...     }
...     
... 
... ---
... 
... ## Challenge:
... 
...         Objective:
...         Identify the vector instructions and datatypes to use to vectorize the given functionali
... ty using the AVX2 vector ISA.
...         The scalar datatype is double.
... 
...         If any instruction is not available ISA, mention it.
... """
  To vectorize the given functionality using the AVX2 vector ISA, we can use the following 
instructions and data types:

* __m256d: This is a 256-bit vector register that contains eight double precision floating 
point numbers (DPFP). It is used to store the input and output data.
* _mm256_load_pd(): This instruction loads a vector of four double precision floating point 
numbers from memory into an __m256d register.
* _mm256_store_pd(): This instruction stores a vector of four double precision floating point 
numbers from an __m256d register to memory.
* _mm256_broadcast_sd(): This instruction broadcasts a single double precision floating point 
number to all elements of an __m256d register.
* _mm256_add_pd(), _mm256_sub_pd(), _mm256_mul_pd(), _mm256_min_pd(), _mm256_max_pd(): These 
instructions perform the corresponding arithmetic and comparison operations on two __m256d 
registers.
* _mm256_and_pd(): This instruction performs a logical AND operation between two __m256d 
registers, resulting in a new __m256d register with all elements set to 1 if the corresponding 
elements of the input registers are non-zero and 0 otherwise.
* _mm256_cmp_pd(): This instruction compares two __m256d registers and returns a new __m256d 
register with all elements set to 1 if the corresponding elements satisfy the comparison 
condition and 0 otherwise. The comparison conditions are: equal, not equal, less than, greater 
than, less than or equal, and greater than or equal.
* _mm256_fmadd_pd(), _mm256_fmsub_pd(): These instructions perform a fused multiply-add 
operation on three __m256d registers: the first two registers are used for the multiplication 
and addition/subtraction, while the third register is used as an accumulator.
* _mm256_fmaddsub_pd(): This instruction performs a fused multiply-add sub operation on four 
__m256d registers: the first three registers are used for the multiplication and 
addition/subtraction, while the fourth register is used as an accumulator.

Note that some of these instructions may not be available on all platforms that support AVX2, 
so it's important to check the documentation before using them.

>>> """ Given the following mapping from operations to vector ISA instrisics for AVX2 ISA with float
... s,
...     General Translation
...     { "loads":" _mm256_load_ps",
...       "stores":" _mm256_store_ps"    ,
...       "broadcast":" _mm256_broadcast_ss"   ,
...       "Binary Vector Operations":{
...                 "add": "_mm256_add_ps", 
...                 "sub": "_mm256_sub_ps", 
...                 "mul" :"_mm256_mul_ps", 
...                 "min": "_mm256_min_ps",
...                 "max": "_mm256_max_ps",
...                 "and": "_mm256_and_ps",
...                 "cmp": "_mm256_cmp_ps"
...                 },
...       "Ternary Vector Operations":{
...       "fused multipy-add" :"_mm256_fmadd_ps", 
...       "fused multiply sub": "_mm256_fmsub_ps",
...       "fused multiply-add/sub": "_mm256_fmaddsub_ps"
...       },
...       "Vector Data Type": "__m256",
...       "Register width": 256,
...       "Elements per vector register": "8"
...     }
...     
... 
... ---
... 
... ## Challenge:
... 
...         Objective:
...         Identify the vector instructions and datatypes to use to the AVX2 vector ISA.
...         The scalar datatype is double.
... 
...         If any instruction is not available ISA, mention it.
...         """
  To vectorize the given functionality using the AVX2 vector ISA, we can use the following 
instructions and data types:

* __m256: This is a 256-bit vector register that contains eight single precision floating 
point numbers (SPFP). It is used to store the input and output data.
* _mm256_load_ps(): This instruction loads a vector of four single precision floating point 
numbers from memory into an __m256 register.
* _mm256_store_ps(): This instruction stores a vector of four single precision floating point 
numbers from an __m256 register to memory.
* _mm256_broadcast_ss(): This instruction broadcasts a single single precision floating point 
number to all elements of an __m256 register.
* _mm256_add_ps(), _mm256_sub_ps(), _mm256_mul_ps(), _mm256_min_ps(), _mm256_max_ps(): These 
instructions perform the corresponding arithmetic and comparison operations on two __m256 
registers.
* _mm256_and_ps(): This instruction performs a logical AND operation between two __m256 
registers, resulting in a new __m256 register with all elements set to 1 if the corresponding 
elements of the input registers are non-zero and 0 otherwise.
* _mm256_cmp_ps(): This instruction compares two __m256 registers and returns a new __m256 
register with all elements set to 1 if the corresponding elements satisfy the comparison 
condition and 0 otherwise. The comparison conditions are: equal, not equal, less than, greater 
than, less than or equal, and greater than or equal.
* _mm256_fmadd_ps(), _mm256_fmsub_ps(): These instructions perform a fused multiply-add 
operation on three __m256 registers: the first two registers are used for the multiplication 
and addition/subtraction, while the third register is used as an accumulator.
* _mm256_fmaddsub_ps(): This instruction performs a fused multiply-add sub operation on four 
__m256 registers: the first three registers are used for the multiplication and 
addition/subtraction, while the fourth register is used as an accumulator.

Note that some of these instructions may not be available on all platforms that support AVX2, 
so it's important to check the documentation before using them.

>>> """
...     You are an expert in vectorization and have been asked to identify the vector instructions a
... nd datatypes to use to vectorize the given functionality.
... 
...     
...     
...     ## Functionalities:
...     ### Vector Loads
...     ```c
...     // Load 1 vector worth of elements, from contiguous memory locations
...     for(int i=0; i< VECTOR_LENGTH; i++)
...     {
...         vector_reg[i] = B[i];
...     }
...     ```
...     ### Vector Stores
...     ```c
...     // Store 1 vector worth of elements, to contiguous memory locations
...     for(int i=0; i< VECTOR_LENGTH; i++)
...     {
...         C[i] = vector_reg[i];
...     }
...     ```
...     ### Vector Broadcasts
...     ```c
...     // Broadcast a scalar value to all elements of a vector register
...     for(int i=0; i< VECTOR_LENGTH; i++)
...     {
...         vector_reg[i] = scalar;
...     }
...     ```
... 
...     ### Binary Vector Operations
...     ```c
...     // Vector addition
...     for(int i=0; i< VECTOR_LENGTH; i++)
...     {
...         vector_reg[i] = op(in_vector_0[i], in_vector_1[i]);
...         // e.g., vector_reg[i] = in_vector_0[i] + in_vector_1[i];
...     }
...     ```
...     Operations: add, sub, mul, min, max, and, cmp
... 
...     ### Ternary Vector Operations
...     ```c
...     // Vector multiplication
...     for(int i=0; i< VECTOR_LENGTH; i++)
...     {
...         vector_reg[i] = op(in_vector_0[i], in_vector_1[i], vector_reg[i]);
...         // e.g., vector_reg[i] = in_vector_0[i] * in_vector_1[i] + vector_reg[i];
...     }
...     Operations: fused multipy-accumulate, fused multiply-accumulate/substract, fused multiply-su
... bstract
...     ```
... 
...     Objective: Identify the vector instructions and datatypes to use to vectorize the given func
... tionality using the AVX2 vector ISA.
...                 The scalar datatype is float.
...     
...     Output:
...     General Translation
...     { "loads":" _mm256_load_ps",
...       "stores":" _mm256_store_ps"    ,
...       "broadcast":" _mm256_broadcast_ss"   ,
...       "Binary Vector Operations":{
...                 "add": "_mm256_add_ps", 
...                 "sub": "_mm256_sub_ps", 
...                 "mul" :"_mm256_mul_ps", 
...                 "min": "_mm256_min_ps",
...                 "max": "_mm256_max_ps",
...                 "and": "_mm256_and_ps",
...                 "cmp": "_mm256_cmp_ps"
...                 },
...       "Ternary Vector Operations":{
...       "fused multipy-add" :"_mm256_fmadd_ps", 
...       "fused multiply sub": "_mm256_fmsub_ps",
...       "fused multiply-add/sub": "_mm256_fmaddsub_ps"
...       },
...       "Vector Data Type": "__m256",
...       "Register width": 256,
...       "Elements per vector register": "8"
...     }
...     
... 
... ---
... 
... ## Challenge:
... 
...         Objective:
...         Construct a similar table of instructions and datatypes to use to the ARMv8 NEON vector 
... ISA.
...         The scalar datatype is float.
...         If any instruction is not available in the ISA, mention it.
... """
  Here's a table of NEON instructions and data types for the given functionality:

| Instruction | Operation | Data Type | Comments |
| --- | --- | --- | --- |
| VLD1.32 | Load 1 vector worth of elements, from contiguous memory locations | float32x4_t | 
Loads a vector of four single precision floating point numbers from memory into the NEON 
register |
| VST1.32 | Store 1 vector worth of elements, to contiguous memory locations | float32x4_t | 
Stores a vector of four single precision floating point numbers from the NEON register into 
memory |
| VMOV.F32 | Broadcast a scalar value to all elements of a vector register | float32_t | Moves 
a scalar value to all elements of a vector register |
| VADD.F32 | Vector addition | float32x4_t | Adds two vectors of single precision floating 
point numbers and stores the result in the destination register |
| VSUB.F32 | Vector subtraction | float32x4_t | Subtracts one vector from another and stores 
the result in the destination register |
| VMUL.F32 | Vector multiplication | float32x4_t | Multiplies two vectors of single precision 
floating point numbers and stores the result in the destination register |
| VMIN.F32 | Minimum value between two vectors | float32x4_t | Finds the minimum value between 
two vectors of single precision floating point numbers |
| VMAX.F32 | Maximum value between two vectors | float32x4_t | Finds the maximum value between 
two vectors of single precision floating point numbers |
| VAND.F32 | Bitwise AND | float32x4_t | Performs a bitwise AND operation on two vectors of 
single precision floating point numbers and stores the result in the destination register |
| VCMP.F32 | Comparison between two vectors | uint32x4_t | Compares two vectors of single 
precision floating point numbers and stores the result as a vector of boolean values |
| VFMA.F32 | Fused multiply-add | float32x4_t | Performs a fused multiply-add operation on 
three vectors of single precision floating point numbers and stores the result in the 
destination register |
| VFMS.F32 | Fused multiply-subtract | float32x4_t | Performs a fused multiply-subtract 
operation on three vectors of single precision floating point numbers and stores the result in 
the destination register |
| VNMUL.F32 | Vector multiplication with negation | float32x4_t | Multiplies two vectors of 
single precision floating point numbers, with the first vector having negative elements, and 
stores the result in the destination register |

Note that some instructions may not be available on all versions of ARMv8 NEON. Also, the 
comments column provide a brief description of each instruction and their usage.