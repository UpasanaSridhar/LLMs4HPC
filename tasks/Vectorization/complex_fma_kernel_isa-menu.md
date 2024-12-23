  """
    You are an expert in vectorization and have been asked to vectorize the following code snippet using SIMD instructions.
## Context
    Vector ISA: AVX2

    Assumptions: 
    Vectorize the innermost loop of loopnests.
    Vectorized loop iterations produce distinct output elements
    The loop extent is a multiple of the vector width. 
    All data structures are aligned to cachelines.
### Instruction templates
{ 
"Unary Vector Operations":{
"load":"_mm256_load_ps",
"store": "_mm256_store_ps",
"broadcast": "_mm256_broadcast_ss",
"zero": "_mm256_setzero_ps"
},
"Binary Vector Operations": {
  "add": "_mm256_add_ps",
  "sub": "_mm256_sub_ps",
  "mul" :"_mm256_mul_ps",
  "min": "_mm256_min_ps",
  "max": "_mm256_max_ps",
  "and": "_mm256_and_ps",
  "cmp": "_mm256_cmp_ps"
},
"Ternary Vector Operations": {
  "fused-multipy-add" :"_mm256_fmadd_ps",
  "fused multiply-add/sub": "_mm256_fmaddsub_ps"
},
"decl": "__m256",
"max registers": 16,
"Register width": 256,
"Elements per vector register": 8 
}

###  Vectorization principles
     1. Load types: 
     If the loop iterator of the vectorized loop is in the loop body, we must use vector loads and stores.
     If there is no index or if the index is an
     e.g.
     ```c
     for(int i = 0; i < N: i++) c[i] = a[i]-b[j]
     ```
     a would use a vector load.
     b would use a broadcast 
     and c would use a vector store to commit the produced value

     2. Operation Types:
        a. Number of operands:  1, 2, 3
        b. Type of assignment: The operation can overwrite the output or accumulate. Accumulates are usually ternary operations
---

## Challenge:

    Vectorize the following code snippet using SIMD instructions.
    Original Code: 
    ```c
    void func_3(floatconst * a, float const* B, float * C, int N)
        for(int j=0; j< N; j++)
        {
            c[j] -= a[0] * B[j] ;
        }
    ```
"""