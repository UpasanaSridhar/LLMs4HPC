  """
    You are an expert in vectorization and have been asked to vectorize the following code snippet using SIMD instructions.
## Context
    Vector ISA: AVX2
    
## Challenge:

    Vectorize the following code snippet using SIMD instructions.
    Original Code: 
    ```c
    void func_3(floatconst * a, float const* B, float * C, int N)
        for(int j=0; j< N; j++)
        {
            c[j] -= a[0] * B[j];
        }
    ```
"""